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
    """

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        import timm
        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=False,          # we load Sydorskyy weights manually below
            in_chans=cfg.in_chans,
            num_classes=0,             # emit pooled features
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
    """Load weights from ``path``, skipping head shape mismatches.

    Returns (loaded, skipped) counts for logging. Never raises — we log and
    let training continue from random head init if the file is bad.
    """
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Strip common prefixes.
    stripped = {}
    for k, v in state.items():
        nk = k
        for pref in ("module.", "model.", "ema_model."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        stripped[nk] = v
    # We want to load anything that goes into ``self.backbone.*``. Head is
    # size-mismatched, drop it.
    target_sd = model.state_dict()
    ok = {}
    skipped = 0
    for k, v in stripped.items():
        # Head tensors in the source can be named e.g. "classifier.weight" or
        # "head.*"; we don't map those. If there's a matching backbone key in
        # our model with identical shape, include it.
        candidates = [k, f"backbone.{k}"]
        hit = None
        for cand in candidates:
            if cand in target_sd and target_sd[cand].shape == v.shape:
                hit = cand
                break
        if hit is not None:
            ok[hit] = v
        else:
            skipped += 1
    missing, unexpected = model.load_state_dict(ok, strict=False)
    loaded = len(ok)
    return loaded, skipped


def build_model(cfg: ModelCfg, logger=None) -> BirdCLEFModel:
    model = BirdCLEFModel(cfg)
    if cfg.pretrained_path is not None and Path(cfg.pretrained_path).exists():
        loaded, skipped = _try_load_pretrained(model, Path(cfg.pretrained_path))
        if logger:
            logger.info("pretrained: loaded %d tensors, skipped %d (head reset)",
                        loaded, skipped)
    elif cfg.pretrained_path is not None and logger:
        logger.warning("pretrained_path %s missing — training from scratch",
                       cfg.pretrained_path)
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
