"""Audio and spectrogram augmentations.

All functions operate on numpy arrays so we can apply them inside the DataLoader
worker (cheap, parallel). The only torch-aware helpers (``mixup_batch``,
``cutmix_batch``) run on already-stacked tensors inside the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------
# Waveform augments
# --------------------------------------------------------------------------

def time_shift(y: np.ndarray, sr: int, max_seconds: float,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Random circular time shift up to ``±max_seconds``."""
    if max_seconds <= 0:
        return y
    rng = rng or np.random.default_rng()
    shift = int(rng.integers(-int(max_seconds * sr), int(max_seconds * sr) + 1))
    if shift == 0:
        return y
    return np.roll(y, shift)


def gaussian_noise(y: np.ndarray, snr_db: float,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    signal_power = float(np.mean(y.astype(np.float64) ** 2)) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    n = rng.normal(0.0, float(np.sqrt(noise_power)), size=y.shape).astype(y.dtype)
    return y + n


# --------------------------------------------------------------------------
# Spectrogram augments
# --------------------------------------------------------------------------

@dataclass
class SpecAugParams:
    freq_masks: int = 2
    freq_mask_param: int = 16   # max mask width in mel bins
    time_masks: int = 2
    time_mask_param: int = 40   # max mask width in frames
    fill_value: float | None = None  # None → column mean


def spec_augment(mel: np.ndarray, params: SpecAugParams,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply SpecAugment to a single mel-spec ``(n_mels, T)`` in-place.

    Matches the canonical recipe (Park et al. 2019): F freq masks with width
    uniform in ``[0, F_param]``, T time masks with width uniform in
    ``[0, T_param]``. Filled with the mel mean (so relative stats unaffected).
    """
    rng = rng or np.random.default_rng()
    mel = mel.copy()
    n_mels, T = mel.shape
    fill = float(mel.mean()) if params.fill_value is None else float(params.fill_value)

    for _ in range(params.freq_masks):
        f = int(rng.integers(0, params.freq_mask_param + 1))
        if f == 0 or f >= n_mels:
            continue
        f0 = int(rng.integers(0, n_mels - f + 1))
        mel[f0:f0 + f, :] = fill

    for _ in range(params.time_masks):
        t = int(rng.integers(0, params.time_mask_param + 1))
        if t == 0 or t >= T:
            continue
        t0 = int(rng.integers(0, T - t + 1))
        mel[:, t0:t0 + t] = fill

    return mel


# --------------------------------------------------------------------------
# Batch-level augments (torch tensors)
# --------------------------------------------------------------------------

def mixup_batch(x, y, alpha: float = 0.4, prob: float = 0.5,
                rng: np.random.Generator | None = None):
    """Return ``(x_mix, y_mix)`` or the originals with prob ``1 - prob``.

    ``x``: (B, C, H, W) float tensor;  ``y``: (B, n_classes) float multi-label.
    Soft-mix both x and y by the same lambda.
    """
    import torch
    rng = rng or np.random.default_rng()
    if rng.random() >= prob:
        return x, y
    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_m = lam * x + (1.0 - lam) * x[perm]
    y_m = lam * y + (1.0 - lam) * y[perm]
    return x_m, y_m


def cutmix_batch(x, y, prob: float = 0.3,
                 rng: np.random.Generator | None = None):
    """Frequency CutMix (paste a vertical slice from a random other sample).

    For spectrograms we slice along the time axis (W): replace a random time
    segment of the target sample with the corresponding segment from a shuffled
    batch buddy; target is a convex combination by the fraction replaced.
    """
    import torch
    rng = rng or np.random.default_rng()
    if rng.random() >= prob:
        return x, y

    B, _, _, W = x.shape
    perm = torch.randperm(B, device=x.device)
    cut_w = int(W * float(rng.uniform(0.15, 0.5)))
    if cut_w == 0:
        return x, y
    start = int(rng.integers(0, W - cut_w + 1))
    x_m = x.clone()
    x_m[:, :, :, start:start + cut_w] = x[perm][:, :, :, start:start + cut_w]
    lam = 1.0 - cut_w / W
    y_m = lam * y + (1.0 - lam) * y[perm]
    return x_m, y_m


# --------------------------------------------------------------------------
# Utility: map uint8 mel back to dB float32
# --------------------------------------------------------------------------

def uint8_mel_to_float(u8: np.ndarray, db_lo: float, db_hi: float) -> np.ndarray:
    """Inverse of the quantisation done in S2. Returns dB float32."""
    return db_lo + (u8.astype(np.float32) / 255.0) * (db_hi - db_lo)
