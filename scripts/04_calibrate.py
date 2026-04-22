"""Per-class isotonic calibration + TopN post-processing A/B test.

Reads the OOF predictions produced by ``02_fit_teacher.py`` +
``02c_pseudo_iter.py`` + ``03_train_student.py``, builds the final
fused **OOF** probs (perch + prior + probe + temp + α_student * student),
and runs a fold-safe A/B on four post-processing variants:

- ``raw``           — just the fused probs.
- ``topn1``         — multiply per-window probs by the file-level max prob
                      of the same class (2nd-place BirdCLEF 2025 recipe).
- ``isotonic``      — per-class ``IsotonicRegression`` fit fold-safely on
                      OOF probs and applied to held-out fold probs.
- ``both_iso_topn`` — isotonic first, then topn1.
- ``both_topn_iso`` — topn1 first, then isotonic.

The winning configuration is recorded in ``teacher_artifact.pkl`` as a
``postprocess`` flag dict; the fitted calibrators are persisted to
``calibration_artifact.pkl`` (full-data isotonic for inference).

Outputs
-------
- ``artifacts/calibration_artifact.pkl``  — knots for the chosen per-class
  isotonic calibrators (empty dict if ``isotonic`` is disabled).
- ``artifacts/teacher_artifact.pkl``      — updated ``postprocess`` block
  with ``apply_topn1`` / ``apply_isotonic`` / ``topn_first`` booleans.
- ``artifacts/04_calibrate.json``         — per-variant OOF AUC report.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.fusion import sigmoid  # noqa: E402
from common.io_utils import load_pickle, save_pickle, write_json  # noqa: E402
from common.metrics import macro_roc_auc_skip_empty  # noqa: E402
from common.paths import artifacts_dir  # noqa: E402
from common.postproc import PerClassIsotonic, apply_isotonic_artifact, topn_smoothing  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--min-positives", type=int, default=5)
    p.add_argument("--n-folds", type=int, default=5)
    return p.parse_args()


def _build_oof_fused(cache: dict, teacher: dict, student: dict | None) -> np.ndarray:
    fw = teacher["fusion_weights"]
    perch = cache["oof_perch_logits"].astype(np.float32)
    prior = cache["oof_prior_logits"].astype(np.float32)
    probe = cache["oof_probe_logits"].astype(np.float32)
    temp = cache["oof_temp_logits"].astype(np.float32)
    fused = (
        fw["alpha_perch"] * perch
        + fw["alpha_prior"] * prior
        + fw["alpha_probe"] * probe
        + fw["alpha_temp"] * temp
    ).astype(np.float32)
    if student is not None and float(student.get("alpha_student", 0.0)) > 0:
        oof_logits = student.get("oof_student_logits")
        if oof_logits is None:
            raise RuntimeError(
                "student artifact missing 'oof_student_logits'. Re-run 03_train_student.py "
                "without --skip-oof so fold-safe student logits are persisted."
            )
        fused = fused + float(student["alpha_student"]) * oof_logits.astype(np.float32)
    return fused


def _eval_auc(probs: np.ndarray, Y: np.ndarray) -> float:
    return float(macro_roc_auc_skip_empty(Y, probs)[0])


def _fold_safe_isotonic(
    probs: np.ndarray,
    Y: np.ndarray,
    group_ids: np.ndarray,
    n_folds: int,
    min_positives: int,
) -> np.ndarray:
    """Return fold-safe isotonic-calibrated probs of shape (N, C)."""
    out = np.zeros_like(probs, dtype=np.float32)
    gkf = GroupKFold(n_splits=n_folds)
    for fold_i, (tr, va) in enumerate(
        gkf.split(np.zeros(len(probs)), groups=group_ids), start=1
    ):
        cal = PerClassIsotonic(min_positives=min_positives).fit(Y[tr], probs[tr])
        art = cal.to_artifact()
        out[va] = apply_isotonic_artifact(probs[va], art)
    return out


def main() -> int:
    args = parse_args()
    art_dir = artifacts_dir()
    out_path = art_dir / "calibration_artifact.pkl"
    if out_path.exists() and not args.overwrite:
        print(f"{out_path} exists. Use --overwrite to refit.")
        return 0

    cache = load_pickle(art_dir / "teacher_cache.pkl")
    teacher = load_pickle(art_dir / "teacher_artifact.pkl")

    student_path = art_dir / "student_artifact.pkl"
    student = load_pickle(student_path) if student_path.exists() else None
    if student is not None:
        print(f"student: α={student.get('alpha_student', 0.0)}  "
              f"active={len(student['active_class_idx'])}")

    Y = cache["Y_full_truth"].astype(np.uint8)
    group_ids = cache["oof_group_ids"].astype(np.int64)
    windows_per_file = int(cache.get("windows_per_file", 12))

    # Build OOF fused probs (raw)
    fused = _build_oof_fused(cache, teacher, student)
    probs_raw = sigmoid(fused).astype(np.float32)

    # Build file_ids for TopN (unique id per file aligned to 12-window blocks)
    n_rows = probs_raw.shape[0]
    assert n_rows % windows_per_file == 0
    n_files = n_rows // windows_per_file
    file_ids = np.repeat(np.arange(n_files, dtype=np.int64), windows_per_file)

    # ---- Variant evaluation --------------------------------------------------
    auc_raw = _eval_auc(probs_raw, Y)
    print(f"[A/B] raw                    OOF AUC = {auc_raw:.4f}")

    probs_topn1 = topn_smoothing(probs_raw, file_ids, n=1).astype(np.float32)
    auc_topn1 = _eval_auc(probs_topn1, Y)
    print(f"[A/B] topn1                  OOF AUC = {auc_topn1:.4f}  "
          f"(Δ vs raw = {auc_topn1 - auc_raw:+.4f})")

    probs_iso = _fold_safe_isotonic(
        probs_raw, Y, group_ids, args.n_folds, args.min_positives,
    ).astype(np.float32)
    auc_iso = _eval_auc(probs_iso, Y)
    print(f"[A/B] isotonic (fold-safe)   OOF AUC = {auc_iso:.4f}  "
          f"(Δ vs raw = {auc_iso - auc_raw:+.4f})")

    probs_both_iso_topn = topn_smoothing(probs_iso, file_ids, n=1).astype(np.float32)
    auc_both_iso_topn = _eval_auc(probs_both_iso_topn, Y)
    print(f"[A/B] iso + topn1            OOF AUC = {auc_both_iso_topn:.4f}  "
          f"(Δ vs raw = {auc_both_iso_topn - auc_raw:+.4f})")

    probs_both_topn_iso = _fold_safe_isotonic(
        probs_topn1, Y, group_ids, args.n_folds, args.min_positives,
    ).astype(np.float32)
    auc_both_topn_iso = _eval_auc(probs_both_topn_iso, Y)
    print(f"[A/B] topn1 + iso            OOF AUC = {auc_both_topn_iso:.4f}  "
          f"(Δ vs raw = {auc_both_topn_iso - auc_raw:+.4f})")

    # ---- Pick winner --------------------------------------------------------
    variants = {
        "raw":            (auc_raw,            False, False, False),
        "topn1":          (auc_topn1,          True,  False, False),
        "isotonic":       (auc_iso,            False, True,  False),
        "iso_topn":       (auc_both_iso_topn,  True,  True,  False),  # iso first
        "topn_iso":       (auc_both_topn_iso,  True,  True,  True),   # topn first
    }
    winner = max(variants.items(), key=lambda kv: kv[1][0])
    name, (auc_win, apply_topn1, apply_iso, topn_first) = winner
    print(f"[A/B] winner = {name}  OOF AUC = {auc_win:.4f}")

    # ---- Fit FULL-DATA isotonic for inference (if enabled) -----------------
    if apply_iso:
        # For inference we fit isotonic on the full OOF set (not fold-safe any more,
        # because inference has never seen these rows; the fold-safe A/B above is
        # what's used to *decide* whether to turn isotonic on.
        probs_for_full_fit = probs_topn1 if topn_first else probs_raw
        full_iso = PerClassIsotonic(min_positives=args.min_positives).fit(Y, probs_for_full_fit)
        calib_art = full_iso.to_artifact()
        print(
            f"[iso-full] fit on {probs_for_full_fit.shape[0]} rows  "
            f"classes calibrated = {len(calib_art['per_class'])}"
        )
    else:
        calib_art = {"min_positives": args.min_positives, "per_class": {}}

    # ---- Persist ------------------------------------------------------------
    save_pickle(calib_art, out_path)

    teacher.setdefault("postprocess", {})
    teacher["postprocess"] = {
        "apply_topn1":   bool(apply_topn1),
        "apply_isotonic": bool(apply_iso),
        "topn_first":    bool(topn_first),
        "oof_auc_raw":   float(auc_raw),
        "oof_auc_best":  float(auc_win),
        "variant":       name,
        "updated_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_pickle(teacher, art_dir / "teacher_artifact.pkl")

    write_json(
        {
            "variants": {k: float(v[0]) for k, v in variants.items()},
            "winner": name,
            "winner_auc": float(auc_win),
            "apply_topn1": bool(apply_topn1),
            "apply_isotonic": bool(apply_iso),
            "topn_first": bool(topn_first),
            "n_classes_calibrated": len(calib_art["per_class"]),
        },
        art_dir / "04_calibrate.json",
    )
    print(
        f"Saved calibration_artifact.pkl  flags=apply_topn1={apply_topn1} "
        f"apply_isotonic={apply_iso} topn_first={topn_first}  "
        f"best OOF AUC={auc_win:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
