"""Fold-safe prior tables.

For each grouping (global / site / hour / month / site×hour), compute the
per-class empirical frequency using **only training-fold rows**. These tables
become (cheap) priors added to the fused teacher/inference logits.

At inference time we look up the prior vector for each soundscape's
``(site, hour, month, site×hour)`` combination, average the available ones with
the global prior, and convert to a logit via ``logit(clip(p, ε, 1-ε))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


EPS = 1e-4


def logit_clip(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


@dataclass
class PriorTables:
    primary_labels: list[str]
    global_prob: np.ndarray
    site: pd.DataFrame
    hour: pd.DataFrame
    month: pd.DataFrame
    site_hour: pd.DataFrame

    def summary(self) -> dict:
        return {
            "n_classes": len(self.primary_labels),
            "n_site_groups": int(len(self.site)),
            "n_hour_groups": int(len(self.hour)),
            "n_month_groups": int(len(self.month)),
            "n_site_hour_groups": int(len(self.site_hour)),
        }


def fit_prior_tables(
    full_truth_df: pd.DataFrame,
    y_true: np.ndarray,
    primary_labels: Sequence[str],
) -> PriorTables:
    """Fit prior tables from the ``full_truth`` window-level DataFrame.

    Parameters
    ----------
    full_truth_df : DataFrame with columns ``site``, ``hour_utc``, ``month``
        aligned 1:1 with ``y_true``.
    y_true : ``(N, C) uint8`` — multi-label ground truth per window.
    primary_labels : canonical label order (C,).
    """
    df = full_truth_df.reset_index(drop=True).copy()
    primary_labels = list(primary_labels)
    tmp = pd.DataFrame(y_true, columns=primary_labels)
    merged = pd.concat([df.reset_index(drop=True), tmp], axis=1)

    global_prob = y_true.mean(axis=0).astype(np.float32)

    def group_prob(keys: list[str]) -> pd.DataFrame:
        return merged.groupby(keys)[primary_labels].mean().astype(np.float32)

    return PriorTables(
        primary_labels=primary_labels,
        global_prob=global_prob,
        site=group_prob(["site"]),
        hour=group_prob(["hour_utc"]),
        month=group_prob(["month"]),
        site_hour=group_prob(["site", "hour_utc"]),
    )


def serialize_priors(priors: PriorTables) -> dict:
    """Flatten to a plain dict suitable for pickling into the Kaggle bundle."""

    def table_keys(df: pd.DataFrame):
        if isinstance(df.index, pd.MultiIndex):
            return list(df.index)
        return list(df.index.tolist())

    return {
        "primary_labels": priors.primary_labels,
        "global_prob": priors.global_prob.astype(np.float32),
        "site_keys": table_keys(priors.site),
        "site_vals": priors.site.to_numpy(dtype=np.float32),
        "hour_keys": table_keys(priors.hour),
        "hour_vals": priors.hour.to_numpy(dtype=np.float32),
        "month_keys": table_keys(priors.month),
        "month_vals": priors.month.to_numpy(dtype=np.float32),
        "site_hour_keys": table_keys(priors.site_hour),
        "site_hour_vals": priors.site_hour.to_numpy(dtype=np.float32),
    }


def lookups_from_serialized(artifact: Mapping) -> dict:
    """Convert a serialized priors dict into efficient runtime lookups."""
    return {
        "global_prob": artifact["global_prob"],
        "site": dict(zip(artifact["site_keys"], artifact["site_vals"])),
        "hour": {int(k): v for k, v in zip(artifact["hour_keys"], artifact["hour_vals"])},
        "month": {int(k): v for k, v in zip(artifact["month_keys"], artifact["month_vals"])},
        "site_hour": {tuple(k): v for k, v in zip(artifact["site_hour_keys"], artifact["site_hour_vals"])},
    }


def build_prior_logits(
    meta_df: pd.DataFrame,
    lookups: Mapping,
    n_classes: int,
) -> np.ndarray:
    """Legacy per-row implementation (kept for compatibility)."""
    return build_prior_logits_vec(meta_df, lookups, n_classes)


def build_prior_logits_vec(
    meta_df: pd.DataFrame,
    lookups: Mapping,
    n_classes: int,
) -> np.ndarray:
    """Vectorized prior-logit builder; O(N) dict lookups + numpy arithmetic.

    For each row, averages the global prior with whichever of
    ``site / hour / month / (site, hour)`` priors are available. The result is
    clipped and converted to logit.
    """
    n = len(meta_df)
    global_prob = lookups["global_prob"].astype(np.float32)
    site_map: Mapping = lookups["site"]
    hour_map: Mapping = lookups["hour"]
    month_map: Mapping = lookups["month"]
    site_hour_map: Mapping = lookups["site_hour"]

    sites = meta_df["site"].to_numpy()
    hours = meta_df["hour_utc"].to_numpy().astype(np.int64, copy=False)
    months = meta_df["month"].to_numpy().astype(np.int64, copy=False)

    acc = np.tile(global_prob, (n, 1))
    cnt = np.ones(n, dtype=np.float32)

    for i in range(n):
        s = sites[i]
        h = int(hours[i])
        m = int(months[i])
        k = 0.0
        v = site_map.get(s)
        if v is not None:
            acc[i] += v; k += 1.0
        v = hour_map.get(h)
        if v is not None:
            acc[i] += v; k += 1.0
        v = month_map.get(m)
        if v is not None:
            acc[i] += v; k += 1.0
        v = site_hour_map.get((s, h))
        if v is not None:
            acc[i] += v; k += 1.0
        if k:
            cnt[i] = 1.0 + k

    acc /= cnt[:, None]
    return logit_clip(acc).astype(np.float32)
