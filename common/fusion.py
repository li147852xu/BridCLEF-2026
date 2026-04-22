"""Weighted fusion of per-source logits.

The teacher and the online pipeline both build a fused score vector per
5-second window:

    final_logits = αp · perch_logits
                 + αP · prior_logits
                 + αb · probe_logits_active  (only on active classes)
                 + αt · temporal_logits_active (only on active classes)

Weights are kept in ``configs/base.yaml`` under ``teacher_fusion`` and
refitted later via a small OOF weight sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FusionWeights:
    alpha_perch: float = 0.60
    alpha_prior: float = 0.15
    alpha_probe: float = 0.10
    alpha_temp: float = 0.15


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def fuse_logits(
    perch_logits: np.ndarray,
    prior_logits: np.ndarray,
    probe_logits_active: np.ndarray | None,
    temp_logits_active: np.ndarray | None,
    *,
    active_class_idx: np.ndarray | None,
    temp_active_class_idx: np.ndarray | None,
    weights: FusionWeights,
) -> np.ndarray:
    """Compute fused final logits.

    ``probe_logits_active`` and ``temp_logits_active`` may be None (e.g., when
    those components are disabled). When present, they're only added on the
    corresponding ``*_active_class_idx`` columns.
    """
    n, n_classes = perch_logits.shape
    fused = (
        weights.alpha_perch * perch_logits.astype(np.float32)
        + weights.alpha_prior * prior_logits.astype(np.float32)
    )

    if probe_logits_active is not None and active_class_idx is not None:
        fused[:, active_class_idx] += weights.alpha_probe * probe_logits_active.astype(np.float32)

    if temp_logits_active is not None and temp_active_class_idx is not None:
        fused[:, temp_active_class_idx] += weights.alpha_temp * temp_logits_active.astype(np.float32)

    return fused
