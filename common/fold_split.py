"""GroupKFold by site for BirdCLEF+ 2026.

The big failure mode in the v2 pipeline was training folds split by *file*,
which hid site-level distribution shift. train_soundscapes has ~23 distinct
sites (S01, S08, S22, …); we split those into N folds so the validation fold
contains entire unseen sites.

For the 708 hard-labeled rows (``train_soundscapes_labels.csv``), the site
is derived from the ``filename`` via ``common.filenames.parse_soundscape_filename``.

train_audio rows are untagged by site (they're XC/iNat recordings from all
over South America), so by convention we always keep them in the training
set — they never land in any val fold.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class FoldAssignment:
    """One row per item; ``fold`` is in [0, n_folds) or -1 for items that
    should always stay in the train set (e.g. train_audio files)."""

    index: np.ndarray     # (N,) original row index
    site: np.ndarray      # (N,) site string ("S01" or "__train_audio__")
    fold: np.ndarray      # (N,) int fold id


def _site_hash_fold(site: str, n_folds: int) -> int:
    """Deterministic fold assignment per site via md5, independent of sklearn
    presence. Collisions are fine since we deduplicate unique sites first."""
    h = hashlib.md5(site.encode()).digest()
    return int.from_bytes(h[:4], "big") % n_folds


def assign_folds_by_site(
    sites: Iterable[str],
    n_folds: int = 5,
    *,
    train_audio_marker: str = "__train_audio__",
) -> np.ndarray:
    """Given a per-item site list (possibly containing the sentinel
    ``train_audio_marker``), return ``folds[i] in [-1, 0, …, n_folds-1]``.

    Algorithm:
      1. Collect unique sites (excluding the train_audio marker).
      2. Sort alphabetically for determinism.
      3. Round-robin assign to folds — so if we have 23 sites and 5 folds,
         folds get 5, 5, 5, 4, 4 sites each.
      4. train_audio marker always → -1 (keep in train).

    This is deterministic across runs / machines given the same site list.
    """
    sites = list(sites)
    uniq = sorted({s for s in sites if s != train_audio_marker})
    site_to_fold = {s: i % n_folds for i, s in enumerate(uniq)}
    folds = np.array(
        [-1 if s == train_audio_marker else site_to_fold[s] for s in sites],
        dtype=np.int32,
    )
    return folds


def build_soundscape_fold_table(
    labels_csv: Path,
    *,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Return a DataFrame of (filename, window_idx, site, fold) for every row
    in train_soundscapes_labels.csv.

    window_idx is derived from ``start`` (``00:00:05`` → 1).
    """
    from common.filenames import parse_soundscape_filename

    df = pd.read_csv(labels_csv)
    # parse site
    df["site"] = df["filename"].map(lambda n: parse_soundscape_filename(n)["site"] or "UNK")
    # window_idx from start string "HH:MM:SS"
    def to_w(s: str) -> int:
        hh, mm, ss = s.split(":")
        sec = int(hh) * 3600 + int(mm) * 60 + int(ss)
        return sec // 5
    df["window_idx"] = df["start"].map(to_w).astype(np.int32)
    df["fold"] = assign_folds_by_site(df["site"].tolist(), n_folds=n_folds)
    return df


def split_mask(folds: np.ndarray, val_fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for (train_rows, val_rows) given the val fold index."""
    val = (folds == val_fold)
    train = (folds != val_fold)            # includes -1 (train_audio) and other fold ids
    return train, val
