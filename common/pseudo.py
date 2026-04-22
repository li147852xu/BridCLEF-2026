"""Pseudo-label selection.

Given a teacher's window-level probabilities and file-level pooled probabilities,
produce hard ``{0,1,-1}`` labels where ``-1`` means "ignore". Two gating rules:

- **Positive**: window prob ≥ ``pos_thr`` **AND** file max prob ≥ ``file_thr``.
- **Negative**: window prob ≤ ``neg_thr``.
- Everything else: ignored (``-1``).

We also support **soft-target** mode: for each supervised entry we keep the raw
teacher probability as well, so students can distil instead of fitting
binary labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PseudoLabelConfig:
    name: str = "strict"
    pos_thr: float = 0.98
    neg_thr: float = 0.02
    file_thr: float = 0.98


def select_pseudo(
    teacher_probs: np.ndarray,
    file_probs_max: np.ndarray,
    windows_per_file: int,
    cfg: PseudoLabelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(hard_labels, mask)``.

    Shapes:
      - ``teacher_probs``: (N_rows, C) with N_rows = n_files * windows_per_file
      - ``file_probs_max``: (n_files, C)
      - ``hard_labels``: int8, values in {-1, 0, 1}
      - ``mask``: uint8, 1 where ``hard_labels != -1``
    """
    file_pos_mask = file_probs_max >= cfg.file_thr
    file_pos_mask_expanded = np.repeat(file_pos_mask, windows_per_file, axis=0)

    pseudo_pos = (teacher_probs >= cfg.pos_thr) & file_pos_mask_expanded
    pseudo_neg = teacher_probs <= cfg.neg_thr

    hard = np.full_like(teacher_probs, fill_value=-1, dtype=np.int8)
    hard[pseudo_neg] = 0
    hard[pseudo_pos] = 1

    mask = (pseudo_pos | pseudo_neg).astype(np.uint8)
    return hard, mask


def summary(hard: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "pos_count": int((hard == 1).sum()),
        "neg_count": int((hard == 0).sum()),
        "ignore_count": int((hard == -1).sum()),
        "class_pos_coverage": int(((hard == 1).sum(axis=0) > 0).sum()),
        "class_neg_coverage": int(((hard == 0).sum(axis=0) > 0).sum()),
        "supervised_entries": int(mask.sum()),
    }
