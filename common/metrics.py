"""Macro ROC-AUC that skips classes without any positive labels.

Mirrors the official Kaggle metric for BirdCLEF.
Reference: https://www.kaggle.com/code/metric/birdclef-roc-auc
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_roc_auc_skip_empty(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute macro ROC-AUC, skipping classes that have 0 positives or all positives.

    Parameters
    ----------
    y_true : (N, C) uint8 / bool / int
    y_pred : (N, C) float

    Returns
    -------
    (macro_score, per_class_scores)
        ``per_class_scores[j] = nan`` for skipped classes.
    """
    assert y_true.shape == y_pred.shape, (y_true.shape, y_pred.shape)
    n_classes = y_true.shape[1]
    per_class = np.full(n_classes, np.nan, dtype=np.float64)

    pos_counts = y_true.sum(axis=0)
    eligible = np.where((pos_counts > 0) & (pos_counts < len(y_true)))[0]

    for j in eligible:
        try:
            per_class[j] = roc_auc_score(y_true[:, j], y_pred[:, j])
        except ValueError:
            per_class[j] = np.nan

    macro = float(np.nanmean(per_class))
    return macro, per_class


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    top_k: int = 10,
) -> dict:
    """Summary for dashboards / logs."""
    macro, per_class = macro_roc_auc_skip_empty(y_true, y_pred)
    pos_counts = y_true.sum(axis=0)
    eligible = ~np.isnan(per_class)

    order = np.argsort(per_class)
    eligible_order = [int(i) for i in order if eligible[i]]

    worst = [
        {"class": class_names[i], "auc": float(per_class[i]), "pos": int(pos_counts[i])}
        for i in eligible_order[:top_k]
    ]
    best = [
        {"class": class_names[i], "auc": float(per_class[i]), "pos": int(pos_counts[i])}
        for i in eligible_order[::-1][:top_k]
    ]
    return {
        "macro_auc": macro,
        "n_eligible_classes": int(eligible.sum()),
        "worst": worst,
        "best": best,
    }
