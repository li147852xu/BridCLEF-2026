"""Joint multi-class MLP probe — Stage 2 drop-in replacement for per-class LogReg.

Design goals (vs. Stage 1 ``common/probes.py``):

* **Joint over classes.** One small MLP maps Perch embeddings (1536-D) → ``K``
  logits in a single forward pass, sharing a 512-D hidden trunk across species.
  This exploits class correlations that the per-class logreg cannot see and
  avoids the PCA information bottleneck (Stage 1 used 128-D PCA input).
* **Weighted sigmoid BCE + focal γ=2** with species-frequency class weights:
  rare classes receive a positive-class up-weight :math:`w_k = \sqrt{(N_{neg}+1)/(N_{pos}+1)}`
  and focal suppresses easy negatives.
* **Numpy-exportable** artifact — at inference time the Kaggle notebook only
  needs two matmuls + ReLU. No PyTorch required.
* **Drop-in under GroupKFold**: same artifact contract as ``LinearProbe``
  (``active_class_idx`` + a forward that returns ``(N, K)`` logits), plus a
  ``probe_type="mlp"`` tag so the packaging/inference stack can dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ----------------------------------------------------------------- artifact


@dataclass
class MLPProbeArtifact:
    """Plain numpy weights for a 2-layer MLP probe.

    Forward::

        h = relu(X @ W0 + b0)
        logits = h @ W1 + b1
    """

    active_class_idx: np.ndarray
    w0: np.ndarray
    b0: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    hidden_dim: int = 512

    def predict_logits_active(self, X: np.ndarray) -> np.ndarray:
        h = X.astype(np.float32) @ self.w0 + self.b0
        np.maximum(h, 0.0, out=h)
        return (h @ self.w1 + self.b1).astype(np.float32)

    def to_artifact(self) -> dict:
        return {
            "probe_type": "mlp",
            "hidden_dim": int(self.hidden_dim),
            "active_class_idx": self.active_class_idx.astype(np.int32),
            "w0": self.w0.astype(np.float32),
            "b0": self.b0.astype(np.float32),
            "w1": self.w1.astype(np.float32),
            "b1": self.b1.astype(np.float32),
        }


def apply_mlp_probe_artifact(X: np.ndarray, art: dict) -> np.ndarray:
    """Pure-numpy forward pass for the MLP probe artifact."""
    h = X.astype(np.float32) @ art["w0"] + art["b0"]
    np.maximum(h, 0.0, out=h)
    return (h @ art["w1"] + art["b1"]).astype(np.float32)


# ------------------------------------------------------------------ trainer


@dataclass
class MLPProbeConfig:
    hidden: int = 512
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 60
    focal_gamma: float = 2.0
    mixup_alpha: float = 0.2
    label_smoothing: float = 0.02
    pos_weight_clip: float = 20.0
    device: str = "mps"
    seed: int = 42
    verbose: bool = True


def _torch_device(req: str) -> str:
    import torch
    if req == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return req


def fit_mlp_probe(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    cfg: MLPProbeConfig | None = None,
    sample_weight: np.ndarray | None = None,
) -> MLPProbeArtifact:
    """Train a joint 2-layer MLP with focal+BCE on multi-label targets.

    Parameters
    ----------
    X : ``(N, D) float32`` raw Perch embedding (NOT PCA).
    Y : ``(N, C) uint8/float32`` multi-label targets in {0, 1}.
    sample_weight : optional ``(N,)`` per-row weight. Useful if mixing labeled
        (weight=1.0) and pseudo-labeled (weight<1.0) rows — caller decides.

    Returns
    -------
    :class:`MLPProbeArtifact` with weights for the ``active`` class subset
    (columns with any positive in ``Y``).
    """
    cfg = cfg or MLPProbeConfig()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _torch_device(cfg.device)

    N, D = X.shape
    C_full = Y.shape[1]
    active = np.where(Y.sum(axis=0) > 0)[0].astype(np.int32)
    K = len(active)
    if K == 0:
        return MLPProbeArtifact(
            active_class_idx=active,
            w0=np.zeros((D, cfg.hidden), dtype=np.float32),
            b0=np.zeros(cfg.hidden, dtype=np.float32),
            w1=np.zeros((cfg.hidden, 0), dtype=np.float32),
            b1=np.zeros(0, dtype=np.float32),
            hidden_dim=cfg.hidden,
        )

    Y_use = Y[:, active].astype(np.float32)

    # Label smoothing squeezes targets toward 0.5 slightly (helps focal)
    if cfg.label_smoothing > 0:
        Y_use = Y_use * (1 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing

    # Per-class pos_weight ≈ sqrt((Nneg+1)/(Npos+1)) capped to avoid explosions
    # on ultra-rare classes.
    n_pos = Y[:, active].sum(axis=0).astype(np.float32)
    n_neg = (Y.shape[0] - n_pos).astype(np.float32)
    pos_w = np.sqrt((n_neg + 1.0) / (n_pos + 1.0)).astype(np.float32)
    pos_w = np.clip(pos_w, 1.0, cfg.pos_weight_clip)

    if sample_weight is None:
        sample_weight = np.ones(N, dtype=np.float32)
    else:
        sample_weight = sample_weight.astype(np.float32).reshape(-1)
        assert len(sample_weight) == N

    x_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(Y_use)
    w_t = torch.from_numpy(sample_weight)
    ds = TensorDataset(x_t, y_t, w_t)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model = nn.Sequential(
        nn.Linear(D, cfg.hidden),
        nn.ReLU(),
        nn.Dropout(cfg.dropout),
        nn.Linear(cfg.hidden, K),
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    pos_w_t = torch.from_numpy(pos_w).to(device)                 # (K,)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w_t, reduction="none")

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            # embedding-space mixup (scalar lambda); mixes both x and y
            if cfg.mixup_alpha > 0 and xb.size(0) > 1:
                lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                perm = torch.randperm(xb.size(0), device=device)
                xb = lam * xb + (1 - lam) * xb[perm]
                yb = lam * yb + (1 - lam) * yb[perm]
                wb = 0.5 * (wb + wb[perm])

            logits = model(xb)                                     # (B, K)
            bce_pc = bce(logits, yb)                               # (B, K)

            if cfg.focal_gamma > 0:
                # focal modulation on top of pos_weighted BCE:
                #   (1 - p_t)^γ where p_t = p for y=1 else 1 - p
                with torch.no_grad():
                    p = torch.sigmoid(logits)
                    p_t = yb * p + (1 - yb) * (1 - p)
                    focal = (1.0 - p_t).clamp_min(1e-6).pow(cfg.focal_gamma)
                loss_pc = bce_pc * focal
            else:
                loss_pc = bce_pc

            # per-row weighting, then average over rows×classes
            loss = (loss_pc * wb.unsqueeze(1)).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)

        sched.step()
        if cfg.verbose and (epoch + 1) % max(1, cfg.epochs // 10) == 0:
            print(
                f"[mlp-probe] epoch {epoch+1:02d}/{cfg.epochs}  "
                f"lr={optim.param_groups[0]['lr']:.2e}  "
                f"loss={running/max(n_seen,1):.4f}"
            )

    model.eval()
    W0 = model[0].weight.detach().cpu().numpy().T.astype(np.float32)   # (D, H)
    b0 = model[0].bias.detach().cpu().numpy().astype(np.float32)
    W1 = model[3].weight.detach().cpu().numpy().T.astype(np.float32)   # (H, K)
    b1 = model[3].bias.detach().cpu().numpy().astype(np.float32)

    return MLPProbeArtifact(
        active_class_idx=active,
        w0=W0, b0=b0, w1=W1, b1=b1, hidden_dim=cfg.hidden,
    )
