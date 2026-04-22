"""Model factory for BirdCLEF+ 2026.

The baseline is ``eca_nfnet_l0`` (timm) with:
    * 1-channel input (stacked mel-spec)   — stem conv adapted from 3-channel
    * Linear(num_features -> n_classes) classifier head
    * Optional dropout before the classifier

Pretrained weights come from the Sydorskyy 2025 Kaggle dataset
``bird-clef-2025-all-pretrained-models``. Those checkpoints have a 206-class
head from the 2025 competition; we strip that and replace with our 234-class
head, loading the backbone weights strict=False.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ModelCfg:
    backbone: str = "eca_nfnet_l0"
    n_classes: int = 234
    in_chans: int = 1
    drop_rate: float = 0.2
    head_drop: float = 0.2
    pretrained_path: Optional[Path] = None  # local .pt/.pth checkpoint
    pretrained_strict: bool = False         # must be False for head size mismatch


class BirdCLEFModel(nn.Module):
    """Thin wrapper: timm backbone (features only) -> global pool -> dropout -> linear.

    timm's ``num_classes=0`` gives the pooled feature vector. We add our own
    dropout + linear so the classifier layer's state-dict key stays stable
    (``head.0.weight`` / ``head.1.weight``) for resume + export.

    ``pretrained_timm=True`` uses timm's ImageNet weights — adapted to 1-channel
    input automatically by averaging the RGB stem kernel. We switch this on when
    the caller couldn't match a domain-specific ckpt, so we never end up with a
    purely random init (which would be 5 epochs behind).
    """

    def __init__(self, cfg: ModelCfg, pretrained_timm: bool = False):
        super().__init__()
        import timm
        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=pretrained_timm,
            in_chans=cfg.in_chans,
            num_classes=0,
            drop_rate=cfg.drop_rate,
        )
        n_feat = int(self.backbone.num_features)
        self.head = nn.Sequential(
            nn.Dropout(cfg.head_drop) if cfg.head_drop > 0 else nn.Identity(),
            nn.Linear(n_feat, cfg.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, 1, n_mels, T) mel-spec → logits (B, n_classes)."""
        feats = self.backbone(x)
        return self.head(feats)


# --------------------------------------------------------------------------
# Pretrained weight loader
# --------------------------------------------------------------------------

def _try_load_pretrained(model: BirdCLEFModel, path: Path) -> tuple[int, int]:
    """Load weights from ``path``, skipping head / classifier mismatches.

    Returns ``(loaded, skipped)`` counts. Handles common ckpt quirks:
        * lightning wrappers with ``state_dict`` or ``model`` top-level key
        * common prefixes (``module.``, ``model.``, ``ema_model.``, ``backbone.``)
        * source key names using underscores where timm uses dots
          (e.g. Sydorskyy's NFNet: ``stem_conv1.weight`` -> ``stem.conv1.weight``)

    Never raises — a failed/empty load lets the caller fall back to timm
    ImageNet init instead of random init.
    """
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict) and all(
            hasattr(v, "shape") for v in list(state["model"].values())[:3]
        ):
            state = state["model"]

    # Strip common prefixes.
    stripped: dict[str, "torch.Tensor"] = {}
    for k, v in state.items():
        nk = k
        for pref in ("module.", "model.", "ema_model.", "backbone."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        stripped[nk] = v

    target_sd = model.state_dict()

    def candidate_names(k: str) -> list[str]:
        # Generate plausible target names by toggling underscore/dot.
        out = [k, f"backbone.{k}"]
        parts = k.split("_", 1)
        if len(parts) == 2:
            dotted = parts[0] + "." + parts[1]
            out += [dotted, f"backbone.{dotted}"]
        # Also try converting every underscore in the first token.
        if "_" in k:
            dotted_full = k.replace("_", ".", 1)
            out += [dotted_full, f"backbone.{dotted_full}"]
        return out

    ok: dict[str, "torch.Tensor"] = {}
    skipped = 0
    for k, v in stripped.items():
        hit = None
        for cand in candidate_names(k):
            if cand in target_sd and target_sd[cand].shape == v.shape:
                hit = cand
                break
        if hit is not None:
            ok[hit] = v
        else:
            skipped += 1
    model.load_state_dict(ok, strict=False)
    return len(ok), skipped


def build_model(cfg: ModelCfg, logger=None) -> BirdCLEFModel:
    """Build a model; try domain ckpt first, fall back to timm ImageNet init
    if that yields zero matched tensors (a bad match is worse than no match
    and we don't want random init)."""
    model = BirdCLEFModel(cfg, pretrained_timm=False)

    loaded = 0
    if cfg.pretrained_path is not None and Path(cfg.pretrained_path).exists():
        loaded, skipped = _try_load_pretrained(model, Path(cfg.pretrained_path))
        if logger:
            logger.info("pretrained: loaded %d tensors, skipped %d from %s",
                        loaded, skipped, cfg.pretrained_path)
    elif cfg.pretrained_path is not None and logger:
        logger.warning("pretrained_path %s missing", cfg.pretrained_path)

    if loaded == 0:
        if logger:
            logger.info("pretrained: domain ckpt unusable, falling back to timm ImageNet init")
        try:
            model = BirdCLEFModel(cfg, pretrained_timm=True)
            if logger:
                logger.info("pretrained: timm ImageNet init loaded (in_chans=%d auto-adapted)",
                            cfg.in_chans)
        except Exception as e:  # noqa: BLE001
            if logger:
                logger.warning("timm ImageNet init failed (%s) — training from scratch", e)
    return model


# --------------------------------------------------------------------------
# EMA utility (small; inlined here to avoid a one-file module)
# --------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters.

    Call ``update(model)`` every step. For validation / ckpt copy into a clone
    via ``copy_to(target)``.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                continue
            self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, target: nn.Module) -> None:
        sd = target.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, st: dict) -> None:
        self.decay = float(st["decay"])
        self.shadow = {k: v.clone() for k, v in st["shadow"].items()}
