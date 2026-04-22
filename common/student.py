"""Lightweight student head distilled on teacher soft probabilities.

Design goals:

- Cheap to train locally on a 16 GB M-series Mac (uses PyTorch MPS when available).
- Export weights as plain numpy so the Kaggle notebook can evaluate them without
  PyTorch (a tiny MLP is just a couple of matrix multiplications).
- Supports both soft-target (BCE on teacher prob) and hybrid hard/soft targets.
- Embedding mixup (interpolate in Perch embedding space) is optional; it matched
  the 2nd-place 2025 recipe without requiring waveform mixing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class MLPArtifact:
    """Plain numpy weights for a 2-layer MLP.

    Layer spec::

        X -> [W0 b0] -> relu -> dropout(train-only) -> [W1 b1] -> relu -> [W2 b2] -> logits

    Artifact dict keys are ``w0, b0, w1, b1, w2, b2`` (all float32 numpy arrays).
    ``active_class_idx`` marks which class columns are trained.
    """

    active_class_idx: np.ndarray
    weights: dict[str, np.ndarray]

    def predict_logits_active(self, X: np.ndarray) -> np.ndarray:
        h = X @ self.weights["w0"] + self.weights["b0"]
        h = np.maximum(h, 0.0)
        h = h @ self.weights["w1"] + self.weights["b1"]
        h = np.maximum(h, 0.0)
        return (h @ self.weights["w2"] + self.weights["b2"]).astype(np.float32)

    def to_artifact(self) -> dict:
        return {
            "active_class_idx": self.active_class_idx.astype(np.int32),
            **{k: v.astype(np.float32) for k, v in self.weights.items()},
        }


# ----------------------------------------------------------------------- Trainer


@dataclass
class MLPTrainConfig:
    hidden: tuple[int, int] = (512, 256)
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 40
    mixup_alpha: float = 0.4
    label_smoothing: float = 0.05
    device: str = "mps"
    seed: int = 42


def _torch_device(req: str) -> str:
    import torch
    if req == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return req


def fit_mlp_student(
    X: np.ndarray,
    Y_soft: np.ndarray,
    mask: np.ndarray,
    cfg: MLPTrainConfig = MLPTrainConfig(),
) -> MLPArtifact:
    """Train a small MLP distilling teacher soft probs under a supervision mask.

    Parameters
    ----------
    X : ``(N, D) float32`` — per-window input features (PCA-projected embedding).
    Y_soft : ``(N, C) float32`` — teacher probabilities in [0, 1].
    mask : ``(N, C) uint8/bool`` — 1 where the entry is supervised, else ignored.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _torch_device(cfg.device)

    N, D = X.shape
    C = Y_soft.shape[1]
    active_class_idx = np.where(mask.sum(axis=0) > 0)[0].astype(np.int32)
    K = len(active_class_idx)

    # Restrict training target to active classes
    Y_use = Y_soft[:, active_class_idx].astype(np.float32)
    M_use = mask[:, active_class_idx].astype(np.float32)

    # Label smoothing (soft): squeeze toward 0.5 slightly
    if cfg.label_smoothing > 0:
        Y_use = Y_use * (1 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing

    x_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(Y_use)
    m_t = torch.from_numpy(M_use)
    ds = TensorDataset(x_t, y_t, m_t)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
    )

    h0, h1 = cfg.hidden
    model = nn.Sequential(
        nn.Linear(D, h0),
        nn.ReLU(),
        nn.Dropout(cfg.dropout),
        nn.Linear(h0, h1),
        nn.ReLU(),
        nn.Linear(h1, K),
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb, mb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)

            if cfg.mixup_alpha > 0 and xb.size(0) > 1:
                lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
                perm = torch.randperm(xb.size(0), device=device)
                xb = lam * xb + (1 - lam) * xb[perm]
                yb = lam * yb + (1 - lam) * yb[perm]
                mb = torch.maximum(mb, mb[perm])  # masked where either is masked

            logits = model(xb)
            loss = (bce(logits, yb) * mb).sum() / mb.sum().clamp_min(1.0)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)

        sched.step()
        print(f"[student] epoch {epoch+1:02d}/{cfg.epochs} | lr={optim.param_groups[0]['lr']:.2e} | loss={running/max(n_seen,1):.4f}")

    model.eval()
    weights: dict[str, np.ndarray] = {}
    # model[0], model[3], model[5] are the linears; dropout has no params
    W0, b0 = model[0].weight.detach().cpu().numpy().T, model[0].bias.detach().cpu().numpy()
    W1, b1 = model[3].weight.detach().cpu().numpy().T, model[3].bias.detach().cpu().numpy()
    W2, b2 = model[5].weight.detach().cpu().numpy().T, model[5].bias.detach().cpu().numpy()
    weights.update({"w0": W0, "b0": b0, "w1": W1, "b1": b1, "w2": W2, "b2": b2})

    return MLPArtifact(active_class_idx=active_class_idx, weights=weights)


def apply_mlp_student_artifact(X: np.ndarray, artifact: dict) -> np.ndarray:
    h = X @ artifact["w0"] + artifact["b0"]
    h = np.maximum(h, 0.0)
    h = h @ artifact["w1"] + artifact["b1"]
    h = np.maximum(h, 0.0)
    return (h @ artifact["w2"] + artifact["b2"]).astype(np.float32)
