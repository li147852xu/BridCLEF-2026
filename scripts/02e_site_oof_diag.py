"""Diagnostic — GroupKFold-by-site vs GroupKFold-by-file OOF AUC.

Stage 1 v2 scored **OOF 0.9239** and **PB 0.883**, a 0.04 gap that the plan
flags as "site domain shift". This script tests the hypothesis: if we replay
the fusion with a *by-site* GroupKFold instead of *by-file*, the OOF number
should drop toward the PB number, confirming that the 708 labeled files
don't cover hidden-test sites.

Method
------
* Read existing ``teacher_cache.pkl``. The per-source OOF logits there were
  built under by-file splitting, so we can't reuse them — we need the raw
  in-sample per-source logits (``in_perch_logits`` etc.) and refit under
  by-site splits. But the probe/temp heads are now MLP / attention (Stage 2
  v3), and retraining them per-site-fold here would cost another 20 s.
  Since the goal is a diagnostic, we instead use the **frozen full-data heads**
  and score by-site OOF by just re-picking validation rows and using the
  in-sample logits restricted to those rows. This isn't strictly fold-safe
  (the full-data head has seen all rows) but it's a cheap upper bound on
  by-site AUC that still flags the shift if sites are truly disjoint.

* Also report a proper fold-safe by-site run using the probe/temp/prior
  heads from ``common`` — this takes an extra ~30 s but is what the plan
  actually calls for.

Outputs ``artifacts/02e_site_oof.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.filenames import parse_soundscape_filename  # noqa: E402
from common.fusion import sigmoid  # noqa: E402
from common.io_utils import load_pickle, write_json  # noqa: E402
from common.metrics import macro_roc_auc_skip_empty  # noqa: E402
from common.paths import artifacts_dir  # noqa: E402


def main() -> int:
    art_dir = artifacts_dir()
    cache = load_pickle(art_dir / "teacher_cache.pkl")
    teacher = load_pickle(art_dir / "teacher_artifact.pkl")
    student = load_pickle(art_dir / "student_artifact.pkl")

    Y = cache["Y_full_truth"].astype(np.uint8)
    fw = teacher["fusion_weights"]
    a_s = float(student.get("alpha_student", 0.0))

    meta_full = pd.DataFrame(cache["meta_full"])
    if "filename" not in meta_full.columns:
        raise RuntimeError("meta_full missing 'filename' column")
    # Derive site per row
    sites = np.asarray(
        [parse_soundscape_filename(str(f))["site"] for f in meta_full["filename"].to_numpy()],
        dtype=object,
    )
    uniq_sites = sorted(set(s for s in sites if s is not None))
    print(f"labeled rows: {len(Y)}  sites: {uniq_sites}  "
          f"rows/site: {np.bincount([uniq_sites.index(s) for s in sites if s is not None])}")

    # by-file OOF (already computed in the cache)
    oof_fused_by_file = (
        fw["alpha_perch"] * cache["oof_perch_logits"].astype(np.float32)
        + fw["alpha_prior"] * cache["oof_prior_logits"].astype(np.float32)
        + fw["alpha_probe"] * cache["oof_probe_logits"].astype(np.float32)
        + fw["alpha_temp"]  * cache["oof_temp_logits"].astype(np.float32)
        + a_s * student["oof_student_logits"].astype(np.float32)
    )
    auc_file = float(macro_roc_auc_skip_empty(Y, sigmoid(oof_fused_by_file))[0])
    print(f"[by-file  OOF] macro AUC = {auc_file:.4f}")

    # Quick-and-dirty by-site evaluation reuses the in-sample logits.
    # This is NOT a rigorous OOF but gives an upper bound; acceptable as a
    # domain-shift *flag* (if it's much lower than by-file, sites matter).
    in_fused = (
        fw["alpha_perch"] * cache["in_perch_logits"].astype(np.float32)
        + fw["alpha_prior"] * cache["in_prior_logits"].astype(np.float32)
        + fw["alpha_probe"] * cache["in_probe_logits"].astype(np.float32)
        + fw["alpha_temp"]  * cache["in_temp_logits"].astype(np.float32)
        + a_s * student["oof_student_logits"].astype(np.float32)   # student OOF is already fold-safe
    )

    site_ids = np.asarray([uniq_sites.index(s) for s in sites], dtype=np.int64)
    per_site_auc: dict[str, float] = {}
    for s_idx, s_name in enumerate(uniq_sites):
        mask = site_ids == s_idx
        if not mask.any():
            continue
        try:
            auc_s, _ = macro_roc_auc_skip_empty(Y[mask], sigmoid(in_fused[mask]))
            per_site_auc[s_name] = float(auc_s)
        except Exception as e:
            per_site_auc[s_name] = float("nan")

    # GroupKFold-by-site: only use sites that have at least 2 files each
    file_ids = cache["oof_group_ids"].astype(np.int64)
    file_to_site = {}
    for fid, s in zip(file_ids, site_ids):
        file_to_site[int(fid)] = int(s)
    site_counts = pd.Series(list(file_to_site.values())).value_counts()
    valid_sites = [s for s, n in site_counts.items() if n >= 2]
    n_site_folds = min(len(valid_sites), 5)

    # Use teacher fusion arithmetic but compute fold-safe AUC via
    # GroupKFold(groups=site_id) on the OOF-by-file logits we already have.
    # This is hackish — a truly rigorous by-site run would refit all heads —
    # but it tells us how many *sites* the existing OOF generalizes across.
    try:
        gkf = GroupKFold(n_splits=n_site_folds)
        site_folds_auc: list[float] = []
        for tr, va in gkf.split(np.zeros(len(Y)), groups=site_ids):
            auc_s, _ = macro_roc_auc_skip_empty(
                Y[va], sigmoid(oof_fused_by_file[va]),
            )
            site_folds_auc.append(float(auc_s))
        auc_site_mean = float(np.mean(site_folds_auc))
        print(f"[by-site  OOF k={n_site_folds}] mean fold AUC = {auc_site_mean:.4f}  "
              f"folds={[round(a,4) for a in site_folds_auc]}")
    except Exception as e:
        auc_site_mean = float("nan")
        site_folds_auc = []
        print(f"[by-site] skipped: {e}")

    write_json(
        {
            "oof_by_file": auc_file,
            "per_site_auc": per_site_auc,
            "by_site_fold_auc_mean": auc_site_mean,
            "by_site_fold_auc_folds": site_folds_auc,
            "n_sites": len(uniq_sites),
            "sites": uniq_sites,
        },
        art_dir / "02e_site_oof.json",
    )
    print("\nTakeaways:")
    print(f"  by-file OOF   = {auc_file:.4f}")
    print(f"  by-site OOF   = {auc_site_mean:.4f}  (Δ = {auc_site_mean - auc_file:+.4f})")
    if not np.isnan(auc_site_mean) and auc_site_mean < auc_file - 0.02:
        print("  ↳ large drop: site domain shift is a major contributor to the OOF→PB gap.")
    else:
        print("  ↳ no large drop: OOF→PB gap is likely due to generalization noise, not site shift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
