"""Post-processing stages applied to the final predicted probabilities.

Implemented:

1. ``topn_smoothing`` — for each file × class, multiply per-window probs by the
   mean of the TopN largest probs of that class in the same file. N=1 reduces
   to "multiply by file-level max prob". This is the 2nd-place 2025 trick
   (~+1–1.5% PB).
2. ``isotonic_calibration`` — per-class sklearn ``IsotonicRegression`` fit on
   OOF predictions. Applied monotonically to raw probs before submission.
3. ``rank_transform`` — replace each column by its empirical CDF rank. Useful
   before ensembling heterogeneous score sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


# ------------------------------------------------------------------ TopN

def topn_smoothing(
    probs: np.ndarray,
    file_ids: np.ndarray,
    n: int = 1,
) -> np.ndarray:
    """Multiply each window prob by the average of the top-N probs in the same file.

    Parameters
    ----------
    probs : ``(N, C) float32``
    file_ids : ``(N,)`` array-like — stable grouping key per window (e.g. filename).
    n : top-N to average over.

    Returns
    -------
    smoothed : ``(N, C) float32``
    """
    probs = probs.astype(np.float32, copy=False)
    out = np.empty_like(probs)
    df = pd.DataFrame({"_gid": file_ids})
    for _, idx in df.groupby("_gid").groups.items():
        idx = np.asarray(idx, dtype=np.int64)
        sub = probs[idx]
        if n <= 1:
            avg_top = sub.max(axis=0)
        else:
            k = min(n, sub.shape[0])
            partitioned = np.partition(sub, -k, axis=0)[-k:]
            avg_top = partitioned.mean(axis=0)
        out[idx] = sub * avg_top[None, :]
    return out


# ------------------------------------------------------------------ Calibration


@dataclass
class PerClassIsotonic:
    """Per-class isotonic calibrators.

    Calibrators are stored as a dict keyed by class index. Classes without enough
    positives to fit a calibrator (< ``min_positives``) are left uncalibrated and
    fall back to the identity function.
    """

    calibrators: dict[int, IsotonicRegression] = field(default_factory=dict)
    min_positives: int = 5

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "PerClassIsotonic":
        n, c = y_true.shape
        assert y_pred.shape == (n, c)
        for j in range(c):
            if int(y_true[:, j].sum()) < self.min_positives:
                continue
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            ir.fit(y_pred[:, j], y_true[:, j])
            self.calibrators[j] = ir
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        out = y_pred.astype(np.float32, copy=True)
        for j, ir in self.calibrators.items():
            out[:, j] = ir.transform(y_pred[:, j]).astype(np.float32)
        return out

    def to_artifact(self) -> dict:
        """Pack as knots (x_thresholds_, y_thresholds_) to avoid pickling sklearn."""
        items = {}
        for j, ir in self.calibrators.items():
            items[int(j)] = {
                "x": ir.X_thresholds_.astype(np.float32),
                "y": ir.y_thresholds_.astype(np.float32),
            }
        return {"min_positives": self.min_positives, "per_class": items}


def apply_isotonic_artifact(y_pred: np.ndarray, artifact: dict) -> np.ndarray:
    """Runtime-only apply (no sklearn dependency needed for inference)."""
    out = y_pred.astype(np.float32, copy=True)
    for j, knots in artifact["per_class"].items():
        j = int(j)
        out[:, j] = np.interp(y_pred[:, j], knots["x"], knots["y"]).astype(np.float32)
    return out


# ------------------------------------------------------------------ Rank transform

def rank_transform(probs: np.ndarray) -> np.ndarray:
    """Column-wise empirical CDF rank in ``[0, 1]``.

    ``out[i, c] = rank(probs[i, c]) / (N - 1)`` within column ``c``.
    Handles ties by 'average' ranking (same as ``scipy.stats.rankdata(method='average')``).
    """
    n = probs.shape[0]
    if n <= 1:
        return probs.astype(np.float32, copy=True)
    ranks = np.empty_like(probs, dtype=np.float64)
    for c in range(probs.shape[1]):
        ranks[:, c] = _rank_average(probs[:, c])
    return (ranks / (n - 1)).astype(np.float32)


def _rank_average(a: np.ndarray) -> np.ndarray:
    # Local impl to avoid pulling scipy just for rankdata.
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sa = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) * 0.5
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks
