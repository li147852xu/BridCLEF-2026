"""PyTorch datasets for BirdCLEF+ 2026 fine-tuning.

Three sources, all emitting ``(mel: (1, n_mels, T) float32,
target: (n_classes,) float32, weight: float)``:

    TrainAudioDataset        weak-label, XC/iNat recordings (decoded on the fly)
    SoundscapeWindowDataset  hard-label, 708 rows with cached mel from S2
    PseudoWindowDataset      S7 only; thresholded soft labels on train_soundscapes

The mel front-end matches S2's numpy recipe exactly so cache-hit and
cache-miss paths produce identical inputs.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset

from common.augment import SpecAugParams, spec_augment, time_shift, gaussian_noise, uint8_mel_to_float
from stages.s2_prepare_mel import _build_mel_fb, _mel_uint8  # reuse recipe


# --------------------------------------------------------------------------
# Shared mel config bundle
# --------------------------------------------------------------------------

@dataclass
class MelConfig:
    sr: int = 32000
    window_seconds: int = 5
    n_fft: int = 1024
    hop: int = 512
    win_length: int = 1024
    n_mels: int = 128
    f_min: float = 50.0
    f_max: float = 16000.0
    power: float = 2.0
    db_lo: float = -80.0
    db_hi: float = 0.0

    @classmethod
    def from_yaml(cls, raw: dict) -> "MelConfig":
        m = raw["mel"]; a = raw["audio"]
        lo, hi = m["db_clip"]
        return cls(
            sr=a["sample_rate"], window_seconds=a["window_seconds"],
            n_fft=m["n_fft"], hop=m["hop_length"], win_length=m["win_length"],
            n_mels=m["n_mels"], f_min=m["f_min"], f_max=m["f_max"],
            power=m["power"], db_lo=float(lo), db_hi=float(hi),
        )

    def build_fb(self) -> np.ndarray:
        return _build_mel_fb(self.sr, self.n_fft, self.n_mels,
                             self.f_min, self.f_max)


def compute_mel_from_wave(y: np.ndarray, cfg: MelConfig, fb: np.ndarray) -> np.ndarray:
    """(n_mels, T) float32 in dB; matches S2 exactly but skips the uint8 quantisation."""
    u8 = _mel_uint8(y, fb, n_fft=cfg.n_fft, hop=cfg.hop, win_length=cfg.win_length,
                   db_lo=cfg.db_lo, db_hi=cfg.db_hi, power=cfg.power)
    return uint8_mel_to_float(u8, cfg.db_lo, cfg.db_hi)


def normalize_mel(mel: np.ndarray, cfg: MelConfig) -> np.ndarray:
    """Scale dB [db_lo, db_hi] → [0, 1] for the CNN input."""
    return (mel - cfg.db_lo) / (cfg.db_hi - cfg.db_lo)


# --------------------------------------------------------------------------
# Label utilities
# --------------------------------------------------------------------------

def _parse_secondary(val) -> list[str]:
    """``secondary_labels`` column is a Python-repr list like ``"['a','b']"``."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    s = str(val).strip()
    if s in ("", "[]", "nan"):
        return []
    try:
        parsed = ast.literal_eval(s)
        return [str(x) for x in parsed] if isinstance(parsed, (list, tuple)) else []
    except Exception:  # noqa: BLE001
        return []


def _parse_hard_primary(val) -> list[str]:
    """train_soundscapes_labels.csv uses ``;``-separated strings for multi-label."""
    if val is None:
        return []
    s = str(val).strip()
    return [tok for tok in s.split(";") if tok]


# --------------------------------------------------------------------------
# Dataset 1: XC / iNat weakly labeled audio (decoded on the fly)
# --------------------------------------------------------------------------

class TrainAudioDataset(Dataset):
    """Variable-length ``train_audio/*.ogg`` with file-level primary + soft secondary."""

    def __init__(
        self,
        df: pd.DataFrame,
        audio_root: Path,
        label_to_idx: dict[str, int],
        mel_cfg: MelConfig,
        *,
        specaug: Optional[SpecAugParams] = None,
        time_shift_s: float = 1.0,
        noise_snr_db: Optional[float] = 20.0,   # None disables
        secondary_soft: float = 0.3,
        weight: float = 1.0,
        train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.root = Path(audio_root)
        self.lti = label_to_idx
        self.mel_cfg = mel_cfg
        self.fb = mel_cfg.build_fb()
        self.specaug = specaug
        self.time_shift_s = time_shift_s
        self.noise_snr = noise_snr_db
        self.secondary_soft = secondary_soft
        self.weight = weight
        self.train = train
        self.n_classes = len(label_to_idx)
        self.win_samples = mel_cfg.sr * mel_cfg.window_seconds

    def __len__(self) -> int:
        return len(self.df)

    def _decode(self, path: Path) -> np.ndarray:
        """Read, downmix, pad/crop to 5s. Randomness applied here.

        Invariant: return array has length exactly ``self.win_samples``.
        We always fall through to pad/crop at the bottom — OGG headers
        occasionally over-report frame counts, so even the partial-read
        path can return fewer samples than requested.
        """
        y: np.ndarray
        with sf.SoundFile(str(path)) as fh:
            sr = fh.samplerate
            n = fh.frames
            if sr != self.mel_cfg.sr:
                # should be rare (XC is usually already 32k); fall back to full read + resampy
                y = fh.read(dtype="float32", always_2d=False)
                if y.ndim == 2:
                    y = y.mean(axis=1)
                import resampy
                y = resampy.resample(y, sr, self.mel_cfg.sr).astype(np.float32)
            elif n > self.win_samples and self.train:
                # Partial random read for long files. soundfile.read(n) may
                # return fewer samples if the header over-reports, so we
                # don't return early — pad/crop below normalises length.
                start = int(np.random.randint(0, n - self.win_samples + 1))
                fh.seek(start)
                y = fh.read(self.win_samples, dtype="float32", always_2d=False)
                if y.ndim == 2:
                    y = y.mean(axis=1)
            else:
                y = fh.read(dtype="float32", always_2d=False)
                if y.ndim == 2:
                    y = y.mean(axis=1)

        # pad or center/random-crop to exactly win_samples
        if len(y) < self.win_samples:
            pad = self.win_samples - len(y)
            left = int(np.random.randint(0, pad + 1)) if self.train else 0
            y = np.pad(y, (left, pad - left))
        elif len(y) > self.win_samples:
            if self.train:
                start = int(np.random.randint(0, len(y) - self.win_samples + 1))
            else:
                start = (len(y) - self.win_samples) // 2
            y = y[start:start + self.win_samples]
        return y.astype(np.float32, copy=False)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.root / str(row["filename"])
        try:
            y = self._decode(path)
        except Exception:  # noqa: BLE001 — corrupt ogg → zero input, skip impact via weight
            y = np.zeros(self.win_samples, dtype=np.float32)

        if self.train:
            y = time_shift(y, self.mel_cfg.sr, self.time_shift_s)
            if self.noise_snr is not None and np.random.random() < 0.3:
                y = gaussian_noise(y, self.noise_snr)

        mel = compute_mel_from_wave(y, self.mel_cfg, self.fb)  # (n_mels, T)
        if self.train and self.specaug is not None:
            mel = spec_augment(mel, self.specaug)
        mel = normalize_mel(mel, self.mel_cfg).astype(np.float32)
        mel = mel[None, :, :]   # (1, n_mels, T)

        target = np.zeros(self.n_classes, dtype=np.float32)
        prim = str(row["primary_label"])
        if prim in self.lti:
            target[self.lti[prim]] = 1.0
        for sec in _parse_secondary(row.get("secondary_labels", None)):
            if sec in self.lti:
                target[self.lti[sec]] = max(target[self.lti[sec]], self.secondary_soft)

        return mel, target, np.float32(self.weight)


# --------------------------------------------------------------------------
# Dataset 2: 708 hard per-window labels (uses S2 mel cache)
# --------------------------------------------------------------------------

class SoundscapeWindowDataset(Dataset):
    """Each item is one 5s window of a train_soundscapes file with a hard multi-label."""

    def __init__(
        self,
        labels_df: pd.DataFrame,          # must have columns filename, window_idx, primary_label (optionally fold)
        mel_cache_dir: Path,              # ${mel_cache}/train_soundscapes
        label_to_idx: dict[str, int],
        mel_cfg: MelConfig,
        *,
        specaug: Optional[SpecAugParams] = None,
        weight: float = 3.0,
        train: bool = True,
    ):
        self.df = labels_df.reset_index(drop=True)
        self.cache = Path(mel_cache_dir)
        self.lti = label_to_idx
        self.mel_cfg = mel_cfg
        self.specaug = specaug
        self.weight = weight
        self.train = train
        self.n_classes = len(label_to_idx)

    def __len__(self) -> int:
        return len(self.df)

    def _load_mel(self, stem: str, widx: int) -> np.ndarray:
        path = self.cache / f"{stem}.npz"
        data = np.load(path, allow_pickle=True)
        u8 = data["mel_u8"][widx]  # (n_mels, T)
        return uint8_mel_to_float(u8, self.mel_cfg.db_lo, self.mel_cfg.db_hi)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        stem = Path(str(row["filename"])).stem
        widx = int(row["window_idx"])
        mel = self._load_mel(stem, widx)
        if self.train and self.specaug is not None:
            mel = spec_augment(mel, self.specaug)
        mel = normalize_mel(mel, self.mel_cfg).astype(np.float32)
        mel = mel[None, :, :]

        target = np.zeros(self.n_classes, dtype=np.float32)
        for tok in _parse_hard_primary(row["primary_label"]):
            if tok in self.lti:
                target[self.lti[tok]] = 1.0

        return mel, target, np.float32(self.weight)


# --------------------------------------------------------------------------
# Dataset 3: pseudo-labeled windows (S7 only)
# --------------------------------------------------------------------------

class PseudoWindowDataset(Dataset):
    """S6 writes a sparse (filename, window_idx, target_234) list.

    Stored fields:
        items : np.ndarray, dtype object, each element is (filename, window_idx)
        targets : (N, 234) float16 — clipped to [0,1]

    ``exclude_sites`` (Plan Y.1): drop every pseudo-labelled window whose
    filename encodes a site in this set. Used by S7 to remove pseudo
    samples from the *current fold's val sites* so the model never trains
    on targets derived for its own held-out distribution (the Plan Y
    post-mortem root cause).
    """

    def __init__(
        self,
        pseudo_npz: Path,
        mel_cache_dir: Path,
        mel_cfg: MelConfig,
        *,
        specaug: Optional[SpecAugParams] = None,
        weight: float = 1.0,
        train: bool = True,
        exclude_sites: Optional[set[str]] = None,
    ):
        d = np.load(pseudo_npz, allow_pickle=True)
        files = d["files"]
        window_idx = d["window_idx"].astype(np.int32)
        targets = d["targets"]  # (N, 234) float16

        if exclude_sites:
            from common.filenames import parse_soundscape_filename
            # Vectorise the site parse by iterating once; filenames are
            # short strings so pure-Python is fine for ~128k rows.
            keep = np.ones(len(files), dtype=bool)
            dropped = 0
            for i, fname in enumerate(files):
                site = parse_soundscape_filename(str(fname)).get("site")
                if site in exclude_sites:
                    keep[i] = False
                    dropped += 1
            files = files[keep]
            window_idx = window_idx[keep]
            targets = targets[keep]
            import logging as _log
            _log.getLogger("bridclef.pseudo").info(
                "PseudoWindowDataset: excluded %d / %d windows from sites %s -> %d remain",
                dropped, len(keep), sorted(exclude_sites), len(files),
            )

        self.files = files
        self.window_idx = window_idx
        self.targets = targets
        self.cache = Path(mel_cache_dir)
        self.mel_cfg = mel_cfg
        self.specaug = specaug
        self.weight = weight
        self.train = train

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        stem = Path(str(self.files[idx])).stem
        widx = int(self.window_idx[idx])
        data = np.load(self.cache / f"{stem}.npz", allow_pickle=True)
        u8 = data["mel_u8"][widx]
        mel = uint8_mel_to_float(u8, self.mel_cfg.db_lo, self.mel_cfg.db_hi)
        if self.train and self.specaug is not None:
            mel = spec_augment(mel, self.specaug)
        mel = normalize_mel(mel, self.mel_cfg).astype(np.float32)[None, :, :]
        target = self.targets[idx].astype(np.float32)
        return mel, target, np.float32(self.weight)


# --------------------------------------------------------------------------
# Validation set helper — per-window, no augment
# --------------------------------------------------------------------------

def build_val_dataset(labels_df: pd.DataFrame, mel_cache_dir: Path,
                      label_to_idx: dict[str, int], mel_cfg: MelConfig) -> SoundscapeWindowDataset:
    return SoundscapeWindowDataset(
        labels_df=labels_df,
        mel_cache_dir=mel_cache_dir,
        label_to_idx=label_to_idx,
        mel_cfg=mel_cfg,
        specaug=None,
        weight=1.0,
        train=False,
    )


# --------------------------------------------------------------------------
# Collate — default PyTorch collate works, but we want torch tensors here
# --------------------------------------------------------------------------

def collate(batch):
    """Collate that drops shape-outlier samples instead of crashing.

    Belt-and-suspenders safety net: _decode guarantees a uniform waveform
    length, but if some rare file slips through with a mel shape different
    from the batch majority we drop the outliers and move on. Beats taking
    down the whole fold over one bad file.
    """
    import torch
    if not batch:
        raise ValueError("empty batch")
    # Find the mode shape among b[0] entries.
    from collections import Counter
    shapes = Counter(tuple(b[0].shape) for b in batch)
    target_shape = shapes.most_common(1)[0][0]
    kept = [b for b in batch if tuple(b[0].shape) == target_shape]
    if len(kept) < len(batch):
        import logging as _log
        _log.getLogger("bridclef.collate").warning(
            "dropped %d/%d batch items with non-matching mel shape (target=%s, got=%s)",
            len(batch) - len(kept), len(batch), target_shape, dict(shapes),
        )
    mels = torch.from_numpy(np.stack([b[0] for b in kept]))
    targets = torch.from_numpy(np.stack([b[1] for b in kept]))
    weights = torch.from_numpy(np.array([b[2] for b in kept], dtype=np.float32))
    return mels, targets, weights
