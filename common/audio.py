"""Audio reading and 5-second windowing.

Soundscape files for BirdCLEF+ 2026 are nominally 60 seconds at 32 kHz, yielding
12 non-overlapping 5-second windows per file. Shorter files are zero-padded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def read_soundscape(
    path: str | Path,
    *,
    target_sr: int = 32000,
    file_seconds: int = 60,
) -> np.ndarray:
    """Read an ``.ogg`` soundscape file as a 1-D ``float32`` array of length ``target_sr * file_seconds``.

    - If the file is stereo, it's averaged to mono.
    - If the sample rate doesn't match, a ``ValueError`` is raised (soundscape set
      is always 32 kHz in BirdCLEF; avoid implicit resampling to keep Perch inputs clean).
    - If the file is shorter than expected, it's right-padded with zeros.
    - If longer, it's truncated.
    """
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != target_sr:
        raise ValueError(f"Unexpected sample rate {sr} in {path}; expected {target_sr}")

    expected = target_sr * file_seconds
    if len(y) < expected:
        y = np.pad(y, (0, expected - len(y)))
    elif len(y) > expected:
        y = y[:expected]
    return y.astype(np.float32, copy=False)


def windowize(
    y: np.ndarray,
    *,
    target_sr: int = 32000,
    window_seconds: int = 5,
    windows_per_file: int = 12,
) -> np.ndarray:
    """Reshape a 1-D waveform into ``(windows_per_file, window_samples)``."""
    window_samples = target_sr * window_seconds
    expected = window_samples * windows_per_file
    if len(y) != expected:
        raise ValueError(
            f"windowize: waveform length {len(y)} != expected {expected} "
            f"({windows_per_file}×{window_seconds}s at {target_sr} Hz)"
        )
    return y.reshape(windows_per_file, window_samples)
