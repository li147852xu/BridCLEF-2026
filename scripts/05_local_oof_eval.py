"""Local evaluation — macro ROC-AUC of the fused teacher on full_truth.

This is **in-sample** on the teacher (it was fit and scored on the same rows).
Still useful to:

1. Confirm the rebuild matches the legacy best (~0.88–0.89 on full_truth).
2. Inspect per-class AUCs to find the weakest classes.
3. Compare across stages (pre-/post-postproc/calibration).

Stage 4 of the roadmap replaces this with a true OOF setup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.io_utils import load_pickle, write_json  # noqa: E402
from common.metrics import per_class_report  # noqa: E402
from common.paths import artifacts_dir  # noqa: E402
from common.postproc import apply_isotonic_artifact, topn_smoothing  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--with-topn", type=int, default=1, help="apply TopN=N post-proc (0 disables)")
    p.add_argument("--with-calibration", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    art_dir = artifacts_dir()
    cache = load_pickle(art_dir / "teacher_cache.pkl")
    y_true = cache["Y_full_truth"].astype(np.uint8)
    probs = cache["teacher_probs"].astype(np.float32)
    primary_labels = cache["primary_labels"]
    meta_full = cache["meta_full"]

    print("Before postproc:")
    rep = per_class_report(y_true, probs, primary_labels)
    _print_report(rep)

    if args.with_topn and args.with_topn >= 1:
        filenames = np.asarray(meta_full["filename"])
        probs_pp = topn_smoothing(probs, filenames, n=int(args.with_topn))
        print(f"After TopN={args.with_topn}:")
        rep_pp = per_class_report(y_true, probs_pp, primary_labels)
        _print_report(rep_pp)
        probs = probs_pp

    if args.with_calibration:
        cal_path = art_dir / "calibration_artifact.pkl"
        if not cal_path.exists():
            print("no calibration_artifact.pkl — run scripts/04_calibrate.py first")
        else:
            art = load_pickle(cal_path)
            probs_cal = apply_isotonic_artifact(probs, art)
            print("After isotonic calibration:")
            rep_cal = per_class_report(y_true, probs_cal, primary_labels)
            _print_report(rep_cal)
            probs = probs_cal

    write_json(rep, art_dir / "05_local_oof_eval.json")
    return 0


def _print_report(rep: dict) -> None:
    print(f"  macro AUC: {rep['macro_auc']:.4f}   eligible classes: {rep['n_eligible_classes']}")
    print("  worst 5:")
    for r in rep["worst"][:5]:
        print(f"    {r['class']:<20}  auc={r['auc']:.3f}  pos={r['pos']}")


if __name__ == "__main__":
    raise SystemExit(main())
