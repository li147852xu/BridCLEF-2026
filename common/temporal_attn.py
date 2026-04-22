"""Self-attention temporal head — Stage 2 drop-in replacement for temporal-lite.

Stage 1 temporal-lite summarizes each 60-s soundscape with four hand-picked
pooling statistics (``mean/max/std/delta``) of the Perch per-class score stream,
then runs per-class logistic regression on that. That's a strong but crude
file-level aggregator: it cannot learn *when* within the 60 s a call occurs
or how consecutive windows reinforce each other.

``common.temporal_attn`` replaces it with:

* **Input**: per-file ``(12, 1536)`` stack of Perch *embeddings* (not scores).
* A tiny transformer block — ``1 layer × 4 heads``, ``d_model = 128`` — learns
  contextual per-window representations and attention pooling over the 12
  windows.
* **Output**: per-file ``K``-dim logits (``K`` = number of active classes),
  broadcast over the 12 windows so downstream fusion stays window-level.

Numpy-exportable end-to-end so the Kaggle notebook runs on pure numpy
(torch is not needed at inference). The numpy forward pass is ~0.5 ms per
file on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- numpy ops


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


# ------------------------------------------------------------ artifact class


@dataclass
class TemporalAttnArtifact:
    """Plain-numpy weights for a single-block transformer temporal head.

    Layer stack (pre-LayerNorm)::

        x      = emb_seq @ W_in.T + b_in         # (N, T, d)
        x      = x + pos_emb                      # broadcast (T, d)
        h      = layernorm(x, g1, b1)
        qkv    = h @ W_qkv.T + b_qkv              # (N, T, 3d)
        q,k,v  = split(qkv, 3)
        attn   = softmax(q @ k.T / sqrt(d_k)) @ v
        attn   = attn @ W_out.T + b_out
        x      = x + attn
        h      = layernorm(x, g2, b2)
        ffn    = relu(h @ W_ff1.T + b_ff1) @ W_ff2.T + b_ff2
        x      = x + ffn
        pooled = x.mean(axis=1)                   # (N, d)
        logits = pooled @ W_cls.T + b_cls         # (N, K)
    """

    active_class_idx: np.ndarray
    d_model: int
    n_heads: int
    window_per_file: int

    w_in: np.ndarray
    b_in: np.ndarray
    pos_emb: np.ndarray

    g1: np.ndarray
    b1: np.ndarray

    w_qkv: np.ndarray
    b_qkv: np.ndarray
    w_out: np.ndarray
    b_out: np.ndarray

    g2: np.ndarray
    b2: np.ndarray

    w_ff1: np.ndarray
    b_ff1: np.ndarray
    w_ff2: np.ndarray
    b_ff2: np.ndarray

    w_cls: np.ndarray
    b_cls: np.ndarray

    def predict_file_logits(self, emb_seq: np.ndarray) -> np.ndarray:
        """Return ``(N, K)`` file-level logits for the active classes."""
        return _temporal_attn_forward(emb_seq.astype(np.float32), self.to_artifact())

    def to_artifact(self) -> dict:
        return {
            "temporal_type": "attn",
            "d_model": int(self.d_model),
            "n_heads": int(self.n_heads),
            "window_per_file": int(self.window_per_file),
            "active_class_idx": self.active_class_idx.astype(np.int32),
            "w_in": self.w_in.astype(np.float32),
            "b_in": self.b_in.astype(np.float32),
            "pos_emb": self.pos_emb.astype(np.float32),
            "g1": self.g1.astype(np.float32),
            "b1": self.b1.astype(np.float32),
            "w_qkv": self.w_qkv.astype(np.float32),
            "b_qkv": self.b_qkv.astype(np.float32),
            "w_out": self.w_out.astype(np.float32),
            "b_out": self.b_out.astype(np.float32),
            "g2": self.g2.astype(np.float32),
            "b2": self.b2.astype(np.float32),
            "w_ff1": self.w_ff1.astype(np.float32),
            "b_ff1": self.b_ff1.astype(np.float32),
            "w_ff2": self.w_ff2.astype(np.float32),
            "b_ff2": self.b_ff2.astype(np.float32),
            "w_cls": self.w_cls.astype(np.float32),
            "b_cls": self.b_cls.astype(np.float32),
        }


def _temporal_attn_forward(emb_seq: np.ndarray, art: dict) -> np.ndarray:
    """Pure-numpy forward, shapes ``(N, T, 1536) -> (N, K)``."""
    d_model = int(art["d_model"])
    n_heads = int(art["n_heads"])
    d_k = d_model // n_heads

    # Input projection + positional embedding
    x = emb_seq @ art["w_in"] + art["b_in"]                          # (N, T, d)
    x = x + art["pos_emb"][None, :, :]

    # Attention block (pre-LN)
    h = _layernorm(x, art["g1"], art["b1"])                          # (N, T, d)
    qkv = h @ art["w_qkv"] + art["b_qkv"]                            # (N, T, 3d)
    q, k, v = np.split(qkv, 3, axis=-1)
    N, T, _ = q.shape
    q = q.reshape(N, T, n_heads, d_k).transpose(0, 2, 1, 3)          # (N, H, T, d_k)
    k = k.reshape(N, T, n_heads, d_k).transpose(0, 2, 1, 3)
    v = v.reshape(N, T, n_heads, d_k).transpose(0, 2, 1, 3)
    att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_k)                  # (N, H, T, T)
    att = _softmax(att.astype(np.float32), axis=-1)
    attn = att @ v                                                    # (N, H, T, d_k)
    attn = attn.transpose(0, 2, 1, 3).reshape(N, T, d_model)
    attn = attn @ art["w_out"] + art["b_out"]
    x = x + attn

    # FFN block (pre-LN)
    h = _layernorm(x, art["g2"], art["b2"])
    ff = h @ art["w_ff1"] + art["b_ff1"]
    np.maximum(ff, 0.0, out=ff)
    ff = ff @ art["w_ff2"] + art["b_ff2"]
    x = x + ff

    pooled = x.mean(axis=1)                                           # (N, d)
    logits = pooled @ art["w_cls"] + art["b_cls"]
    return logits.astype(np.float32)


def apply_temporal_attn_artifact(emb_seq: np.ndarray, art: dict) -> np.ndarray:
    return _temporal_attn_forward(emb_seq, art)


# ------------------------------------------------------------------ trainer


@dataclass
class TemporalAttnConfig:
    d_model: int = 128
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.2
    attn_dropout: float = 0.1
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 80
    focal_gamma: float = 2.0
    label_smoothing: float = 0.02
    mixup_alpha: float = 0.2
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


def fit_temporal_attn(
    emb_seq: np.ndarray,
    Y_file: np.ndarray,
    *,
    cfg: TemporalAttnConfig | None = None,
    sample_weight: np.ndarray | None = None,
) -> TemporalAttnArtifact:
    """Train the temporal-attention file-level head.

    Parameters
    ----------
    emb_seq : ``(n_files, T, D)`` float32  Perch embeddings per window.
    Y_file  : ``(n_files, C)`` uint8/float32  multi-label (file = logical-OR of
              its 12 window labels).
    sample_weight : optional ``(n_files,)`` per-file weight.
    """
    cfg = cfg or TemporalAttnConfig()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _torch_device(cfg.device)

    n_files, T, D = emb_seq.shape
    C = Y_file.shape[1]
    active = np.where(Y_file.sum(axis=0) > 0)[0].astype(np.int32)
    K = len(active)
    Y_use = Y_file[:, active].astype(np.float32)

    if K == 0 or n_files < 4:
        # Degenerate fold; return zero-weights that forward to zeros.
        def zf(*shape): return np.zeros(shape, dtype=np.float32)
        return TemporalAttnArtifact(
            active_class_idx=active,
            d_model=cfg.d_model, n_heads=cfg.n_heads, window_per_file=T,
            w_in=zf(D, cfg.d_model), b_in=zf(cfg.d_model),
            pos_emb=zf(T, cfg.d_model),
            g1=np.ones(cfg.d_model, dtype=np.float32), b1=zf(cfg.d_model),
            w_qkv=zf(cfg.d_model, 3 * cfg.d_model), b_qkv=zf(3 * cfg.d_model),
            w_out=zf(cfg.d_model, cfg.d_model), b_out=zf(cfg.d_model),
            g2=np.ones(cfg.d_model, dtype=np.float32), b2=zf(cfg.d_model),
            w_ff1=zf(cfg.d_model, cfg.ffn_mult * cfg.d_model),
            b_ff1=zf(cfg.ffn_mult * cfg.d_model),
            w_ff2=zf(cfg.ffn_mult * cfg.d_model, cfg.d_model),
            b_ff2=zf(cfg.d_model),
            w_cls=zf(cfg.d_model, max(K, 1)),
            b_cls=zf(max(K, 1)),
        )

    if cfg.label_smoothing > 0:
        Y_use = Y_use * (1 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing

    n_pos = Y_file[:, active].sum(axis=0).astype(np.float32)
    n_neg = (n_files - n_pos).astype(np.float32)
    pos_w = np.sqrt((n_neg + 1.0) / (n_pos + 1.0))
    pos_w = np.clip(pos_w, 1.0, cfg.pos_weight_clip).astype(np.float32)

    if sample_weight is None:
        sample_weight = np.ones(n_files, dtype=np.float32)
    sample_weight = sample_weight.astype(np.float32).reshape(-1)

    x_t = torch.from_numpy(emb_seq.astype(np.float32))            # (F, T, D)
    y_t = torch.from_numpy(Y_use)
    w_t = torch.from_numpy(sample_weight)
    ds = TensorDataset(x_t, y_t, w_t)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    class Block(nn.Module):
        def __init__(self, d, h, ff, p_drop, p_attn):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(
                d, h, dropout=p_attn, batch_first=True, bias=True,
            )
            self.ln2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, ff),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(ff, d),
            )
            self.drop = nn.Dropout(p_drop)

        def forward(self, x):
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, need_weights=False)
            x = x + self.drop(a)
            h = self.ln2(x)
            x = x + self.drop(self.ffn(h))
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(D, cfg.d_model)
            self.pos = nn.Parameter(torch.zeros(T, cfg.d_model))
            nn.init.normal_(self.pos, std=0.02)
            self.block = Block(cfg.d_model, cfg.n_heads,
                               cfg.d_model * cfg.ffn_mult,
                               cfg.dropout, cfg.attn_dropout)
            self.head = nn.Linear(cfg.d_model, K)

        def forward(self, x):
            h = self.proj(x) + self.pos[None, :, :]
            h = self.block(h)
            return self.head(h.mean(dim=1))

    model = Model().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    pos_w_t = torch.from_numpy(pos_w).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w_t, reduction="none")

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            if cfg.mixup_alpha > 0 and xb.size(0) > 1:
                lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                perm = torch.randperm(xb.size(0), device=device)
                xb = lam * xb + (1 - lam) * xb[perm]
                yb = lam * yb + (1 - lam) * yb[perm]
                wb = 0.5 * (wb + wb[perm])

            logits = model(xb)
            bce_pc = bce(logits, yb)
            if cfg.focal_gamma > 0:
                with torch.no_grad():
                    p = torch.sigmoid(logits)
                    p_t = yb * p + (1 - yb) * (1 - p)
                    focal = (1.0 - p_t).clamp_min(1e-6).pow(cfg.focal_gamma)
                loss_pc = bce_pc * focal
            else:
                loss_pc = bce_pc
            loss = (loss_pc * wb.unsqueeze(1)).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)

        sched.step()
        if cfg.verbose and (epoch + 1) % max(1, cfg.epochs // 10) == 0:
            print(
                f"[tattn] epoch {epoch+1:02d}/{cfg.epochs}  "
                f"lr={optim.param_groups[0]['lr']:.2e}  "
                f"loss={running/max(n_seen,1):.4f}"
            )

    model.eval()
    # ---- extract numpy weights in the exact convention of the numpy forward
    proj = model.proj
    w_in = proj.weight.detach().cpu().numpy().T.astype(np.float32)   # (D, d_model)
    b_in = proj.bias.detach().cpu().numpy().astype(np.float32)
    pos_emb = model.pos.detach().cpu().numpy().astype(np.float32)     # (T, d_model)

    ln1 = model.block.ln1
    g1 = ln1.weight.detach().cpu().numpy().astype(np.float32)
    b1 = ln1.bias.detach().cpu().numpy().astype(np.float32)

    # PyTorch MHA in_proj_weight is (3*d, d) with order [Wq; Wk; Wv]. Our numpy
    # forward does `h @ w_qkv + b_qkv`, so we transpose to (d, 3*d).
    in_w = model.block.attn.in_proj_weight.detach().cpu().numpy().T.astype(np.float32)  # (d, 3d)
    in_b = model.block.attn.in_proj_bias.detach().cpu().numpy().astype(np.float32)

    out_w = model.block.attn.out_proj.weight.detach().cpu().numpy().T.astype(np.float32)
    out_b = model.block.attn.out_proj.bias.detach().cpu().numpy().astype(np.float32)

    ln2 = model.block.ln2
    g2 = ln2.weight.detach().cpu().numpy().astype(np.float32)
    b2 = ln2.bias.detach().cpu().numpy().astype(np.float32)

    ff = model.block.ffn
    w_ff1 = ff[0].weight.detach().cpu().numpy().T.astype(np.float32)
    b_ff1 = ff[0].bias.detach().cpu().numpy().astype(np.float32)
    w_ff2 = ff[3].weight.detach().cpu().numpy().T.astype(np.float32)
    b_ff2 = ff[3].bias.detach().cpu().numpy().astype(np.float32)

    head = model.head
    w_cls = head.weight.detach().cpu().numpy().T.astype(np.float32)
    b_cls = head.bias.detach().cpu().numpy().astype(np.float32)

    return TemporalAttnArtifact(
        active_class_idx=active,
        d_model=cfg.d_model, n_heads=cfg.n_heads, window_per_file=T,
        w_in=w_in, b_in=b_in, pos_emb=pos_emb,
        g1=g1, b1=b1,
        w_qkv=in_w, b_qkv=in_b,
        w_out=out_w, b_out=out_b,
        g2=g2, b2=b2,
        w_ff1=w_ff1, b_ff1=b_ff1,
        w_ff2=w_ff2, b_ff2=b_ff2,
        w_cls=w_cls, b_cls=b_cls,
    )
