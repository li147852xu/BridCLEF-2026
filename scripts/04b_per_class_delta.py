"""Stage 2.3 — per-class logit shift / threshold optimization.

The competition metric is **macro ROC-AUC**, which is provably invariant
under any per-class monotonic transform (including an additive logit shift
:math:`L_c \\to L_c + \\Delta_c`, or the TopN=1 multiplicative smoothing that
applies the same monotone map to every window of class :math:`c`). So on the
isolated Stage 2 pipeline a per-class delta cannot change macro AUC — it's a
true no-op for the current objective.

This module still ships the infrastructure, because:

1. Stage 3 introduces a separate mel-CNN head whose logit scale is *class-wise
   different* from the Perch/probe/temp/student head. When the two are
   averaged (``a * L_teacher + b * L_cnn``) a per-class shift on the teacher
   side changes macro AUC of the ensemble. The delta we save here will
   become the "calibration level" that the rank-aware post-processing in
   Stage 3.3 reweights against.
2. Downstream submit code can start reading ``bundle["delta_per_class"]``
   today without further surgery.

Concretely this script:

* Reconstructs the fused OOF logits (perch + prior + probe + temp + student).
* Computes a **per-class centering delta** :math:`\\Delta_c = -\\bar L_c`
  using the fold-safe OOF distribution, so that each class's post-delta logit
  has zero mean over the labeled set. This is a no-op on macro AUC but
  normalizes scale across classes for Stage 3 ensemble arithmetic.
* Verifies fold-safe macro AUC is unchanged (sanity).
* Persists ``delta_per_class`` into ``teacher_artifact.pkl``.

If you later want to actually *optimize* a per-class metric that is NOT
monotone-invariant (macro F1 at a pinned threshold, or rank-averaged
ensemble AUC with the CNN head), extend :func:`_optimize_deltas` with a
proper inner search.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.fusion import sigmoid  # noqa: E402
from common.io_utils import load_pickle, save_pickle, write_json  # noqa: E402
from common.metrics import macro_roc_auc_skip_empty  # noqa: E402
from common.paths import artifacts_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--mode",
        choices=["zero", "center", "target-rate"],
        default="center",
        help="How to compute per-class delta. 'zero' stores all zeros; "
             "'center' shifts each class's mean OOF logit to 0; "
             "'target-rate' shifts so the sigmoid mean matches a target prior.",
    )
    p.add_argument("--target-rate", type=float, default=0.05,
                   help="Used only with --mode=target-rate")
    return p.parse_args()


def _build_oof_fused(cache: dict, teacher: dict, student: dict | None) -> np.ndarray:
    fw = teacher["fusion_weights"]
    fused = (
        fw["alpha_perch"] * cache["oof_perch_logits"].astype(np.float32)
        + fw["alpha_prior"] * cache["oof_prior_logits"].astype(np.float32)
        + fw["alpha_probe"] * cache["oof_probe_logits"].astype(np.float32)
        + fw["alpha_temp"]  * cache["oof_temp_logits"].astype(np.float32)
    ).astype(np.float32)
    if student is not None and float(student.get("alpha_student", 0.0)) > 0.0:
        a = float(student["alpha_student"])
        fused = fused + a * student["oof_student_logits"].astype(np.float32)
    return fused


def _optimize_deltas(
    fused: np.ndarray,
    Y: np.ndarray,
    mode: str,
    target_rate: float,
) -> np.ndarray:
    """Return ``(C,)`` per-class additive logit shifts."""
    C = fused.shape[1]
    delta = np.zeros(C, dtype=np.float32)
    if mode == "zero":
        return delta
    if mode == "center":
        mean_logit = fused.mean(axis=0)
        delta = (-mean_logit).astype(np.float32)
        return delta
    if mode == "target-rate":
        # Solve σ(L̄ + Δ) = target_rate  →  Δ = logit(target_rate) - L̄
        tgt = float(np.clip(target_rate, 1e-4, 1 - 1e-4))
        tgt_logit = np.log(tgt / (1 - tgt))
        mean_logit = fused.mean(axis=0)
        delta = (tgt_logit - mean_logit).astype(np.float32)
        return delta
    raise ValueError(f"unknown mode {mode!r}")


def main() -> int:
    args = parse_args()
    art_dir = artifacts_dir()
    teacher_path = art_dir / "teacher_artifact.pkl"
    teacher = load_pickle(teacher_path)

    if not args.overwrite and "delta_per_class" in teacher:
        print("teacher_artifact.pkl already has delta_per_class; pass --overwrite.")
        return 0

    cache = load_pickle(art_dir / "teacher_cache.pkl")
    student_path = art_dir / "student_artifact.pkl"
    student = load_pickle(student_path) if student_path.exists() else None

    Y = cache["Y_full_truth"].astype(np.uint8)
    fused = _build_oof_fused(cache, teacher, student)

    auc_before = float(macro_roc_auc_skip_empty(Y, sigmoid(fused))[0])
    delta = _optimize_deltas(fused, Y, args.mode, args.target_rate)
    auc_after = float(
        macro_roc_auc_skip_empty(Y, sigmoid(fused + delta[None, :]))[0]
    )

    # Macro AUC MUST be invariant; warn loudly if not.
    if abs(auc_after - auc_before) > 1e-9:
        print(f"[WARN] auc changed by {auc_after - auc_before:+.4e}  "
              f"(expected ~0 — per-class shift is monotone-invariant on AUC)")
    else:
        print(f"[OK] auc invariant under delta ({auc_before:.6f})")

    stats = {
        "mode": args.mode,
        "target_rate": float(args.target_rate) if args.mode == "target-rate" else None,
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "oof_auc_before": auc_before,
        "oof_auc_after":  auc_after,
    }
    print(
        f"delta stats: mean={stats['delta_mean']:+.3f}  "
        f"std={stats['delta_std']:.3f}  "
        f"min={stats['delta_min']:+.3f}  max={stats['delta_max']:+.3f}"
    )

    teacher["delta_per_class"] = delta.astype(np.float32)
    meta = teacher.setdefault("meta", {})
    meta["delta_per_class_mode"] = args.mode
    meta["delta_per_class_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_pickle(teacher, teacher_path)
    write_json(stats, art_dir / "04b_per_class_delta.json")
    print(f"Saved delta_per_class -> {teacher_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
