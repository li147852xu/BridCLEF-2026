#!/usr/bin/env python
"""Stage 3.3: sweep alpha_cnn on the 708 labeled rows.

INPUTS
------
* ``artifacts/teacher_cache.pkl`` — carries the v3 per-source OOF logits and
  ``Y_full_truth`` (708, 234).
* ``artifacts/teacher_artifact.pkl`` — carries the current v3 fusion weights
  and ``postprocess`` choice.
* ``--cnn-probs <PATH>`` — NumPy array of shape ``(708, 234)`` with the
  3-seed CNN ensemble sigmoid probabilities on the labeled rows, as exported
  by the Stage 3.1 GPU notebook (``cnn_probs_labeled.npy``).

OUTPUTS
-------
* Prints an OOF-AUC sweep table for ``alpha_cnn``.
* Optionally writes the chosen ``alpha_cnn`` into ``teacher_artifact.pkl``
  under the ``cnn`` key so ``scripts/06_package_submission.py`` picks it up.

CAVEATS
-------
The CNN was trained on *all* 127 k cached windows including the 708 labeled
ones — so ``cnn_probs_labeled`` is **in-sample** for the CNN. By-file OOF
inflates the CNN's apparent benefit. We therefore also compute a
**by-site GroupKFold** evaluation (which mirrors the true domain shift seen
between our OOF and the public leaderboard) and recommend the lower of the
two winning alphas as a conservative choice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.io_utils import load_pickle, save_pickle  # noqa: E402
from common.paths import artifacts_dir  # noqa: E402


EPS = 1e-4


def _logit_clip(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def _macro_auc(Y: np.ndarray, S: np.ndarray) -> float:
    cols = np.where(Y.sum(axis=0) > 0)[0]
    if len(cols) == 0: return float("nan")
    return float(roc_auc_score(Y[:, cols], S[:, cols], average="macro"))


def _per_site_auc(Y: np.ndarray, S: np.ndarray, sites: np.ndarray) -> dict[str, float]:
    out = {}
    for site in sorted(np.unique(sites)):
        mask = sites == site
        if mask.sum() < 3: continue
        Ym, Sm = Y[mask], S[mask]
        cols = np.where(Ym.sum(axis=0) > 0)[0]
        if len(cols) == 0: continue
        try:
            out[site] = float(roc_auc_score(Ym[:, cols], Sm[:, cols], average="macro"))
        except Exception:
            pass
    return out


def _fused_oof(tc: dict, art: dict, alpha_cnn: float,
               cnn_logits: np.ndarray, student_oof: np.ndarray | None) -> np.ndarray:
    K = tc["Y_full_truth"].shape[1]
    N = tc["Y_full_truth"].shape[0]
    fw = art["fusion_weights"]
    final = (
        float(fw["alpha_perch"]) * tc["oof_perch_logits"]
        + float(fw["alpha_prior"]) * tc["oof_prior_logits"]
    )
    probe_active = tc["oof_active_class_idx"]
    final[:, probe_active] += float(fw["alpha_probe"]) * tc["oof_probe_logits"][:, probe_active]
    temp_active = tc.get("oof_temp_active_class_idx", probe_active)
    final[:, temp_active] += float(fw["alpha_temp"]) * tc["oof_temp_logits"][:, temp_active]
    if student_oof is not None:
        alpha_student = float(art.get("student_meta", {}).get("alpha_student",
                              art.get("student", {}).get("alpha_student", 0.0)))
        student_active = art.get("student_meta", {}).get("active_class_idx", None)
        if student_active is None:
            # fall back to all classes
            final += alpha_student * student_oof
        else:
            final[:, student_active] += alpha_student * student_oof
    if alpha_cnn != 0.0:
        final = final + alpha_cnn * cnn_logits
    # delta_per_class is AUC-invariant at macro level; include for parity
    delta = art.get("delta_per_class")
    if delta is not None:
        final = final + delta[None, :]
    return final


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cnn-probs", required=True, type=Path,
                   help="Path to cnn_probs_labeled.npy from the GPU notebook.")
    p.add_argument("--sites-meta", type=Path, default=None,
                   help="Optional: parquet/csv with columns row_id,site matching "
                        "labeled rows. If omitted, we derive sites from "
                        "artifacts/perch_cache/meta.parquet via labeled_cache_idx.")
    p.add_argument("--grid", nargs="+", type=float,
                   default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                            0.50, 0.60, 0.75, 1.00, 1.25, 1.50],
                   help="alpha_cnn values to try.")
    p.add_argument("--write", action="store_true",
                   help="Write the chosen alpha_cnn back to teacher_artifact.pkl.")
    p.add_argument("--pick", choices=["by_file", "by_site", "conservative"],
                   default="conservative",
                   help="Which optimum to commit (conservative = min of by_file and by_site).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    art_dir = artifacts_dir()

    print("[cnn-sweep] loading artifacts")
    tc = load_pickle(art_dir / "teacher_cache.pkl")
    art = load_pickle(art_dir / "teacher_artifact.pkl")

    Y = tc["Y_full_truth"].astype(np.int32)           # (708, 234)
    N, K = Y.shape
    print(f"[cnn-sweep]   Y: {Y.shape}  pos/row: mean={Y.sum(axis=1).mean():.2f}")

    # CNN probs on the 708 labeled rows, converted to logits
    if not args.cnn_probs.exists():
        raise FileNotFoundError(args.cnn_probs)
    cnn_probs = np.load(args.cnn_probs).astype(np.float32)
    assert cnn_probs.shape == (N, K), \
        f"cnn_probs shape {cnn_probs.shape} != expected {(N, K)}"
    cnn_logits = _logit_clip(cnn_probs)
    print(f"[cnn-sweep]   cnn_probs: {cnn_probs.shape}  mean={cnn_probs.mean():.4f}")

    # Student OOF logits (if available)
    student_oof = None
    sp = art_dir / "student_artifact.pkl"
    if sp.exists():
        stu = load_pickle(sp)
        if "oof_student_logits" in stu:
            student_oof = stu["oof_student_logits"].astype(np.float32)
            # remember its active_class_idx to restrict application
            art.setdefault("student_meta", {})
            art["student_meta"]["alpha_student"] = float(stu.get("alpha_student", 0.0))
            art["student_meta"]["active_class_idx"] = stu.get("active_class_idx", None)
            print(f"[cnn-sweep]   student_oof: {student_oof.shape}  "
                  f"alpha={art['student_meta']['alpha_student']:.3f}")

    # ---- sites for the labeled rows -----------------------------------------
    if args.sites_meta is not None:
        meta = (pd.read_parquet(args.sites_meta)
                if args.sites_meta.suffix == ".parquet"
                else pd.read_csv(args.sites_meta))
        assert "site" in meta.columns, "sites-meta must have a 'site' column"
        sites = meta["site"].to_numpy()[:N]
    else:
        cache_meta = pd.read_parquet(art_dir / "perch_cache" / "meta.parquet")
        labeled_idx = np.asarray(tc["labeled_cache_idx"], dtype=np.int64)
        sites = cache_meta["site"].to_numpy()[labeled_idx]
    assert sites.shape == (N,), f"sites shape {sites.shape} != {(N,)}"
    unique_sites, site_counts = np.unique(sites, return_counts=True)
    print(f"[cnn-sweep]   sites: {dict(zip(unique_sites.tolist(), site_counts.tolist()))}")

    # ---- sweep ---------------------------------------------------------------
    print(f"\n{'alpha_cnn':>10} | {'by_file':>8} | {'by_site':>8} | per-site snapshot")
    print("-" * 80)
    results = []
    for a in args.grid:
        fused = _fused_oof(tc, art, a, cnn_logits, student_oof)
        auc_file = _macro_auc(Y, fused)
        # by-site GKF: hold out each site once, score the held-out rows
        try:
            gkf = GroupKFold(n_splits=min(len(unique_sites),
                                           max(2, int(min(8, (sites != None).sum())))))
            site_scores = []
            for _, val_idx in gkf.split(np.arange(N), Y.max(axis=1), groups=sites):
                site_scores.append(_macro_auc(Y[val_idx], fused[val_idx]))
            auc_site = float(np.nanmean(site_scores))
        except Exception:
            auc_site = float("nan")
        per_site = _per_site_auc(Y, fused, sites)
        snap = " ".join(f"{s}:{v:.3f}" for s, v in sorted(per_site.items())[:4])
        results.append({"alpha_cnn": a, "auc_by_file": auc_file, "auc_by_site": auc_site})
        print(f"{a:>10.3f} | {auc_file:>8.4f} | {auc_site:>8.4f} | {snap}")

    df = pd.DataFrame(results)
    best_file = df.loc[df["auc_by_file"].idxmax()]
    best_site = df.loc[df["auc_by_site"].idxmax()]
    print("\n[cnn-sweep] BEST by_file:  alpha={:.3f}  auc={:.4f}".format(
        best_file["alpha_cnn"], best_file["auc_by_file"]))
    print("[cnn-sweep] BEST by_site:  alpha={:.3f}  auc={:.4f}".format(
        best_site["alpha_cnn"], best_site["auc_by_site"]))

    if args.pick == "by_file":
        pick = float(best_file["alpha_cnn"])
    elif args.pick == "by_site":
        pick = float(best_site["alpha_cnn"])
    else:
        pick = min(float(best_file["alpha_cnn"]), float(best_site["alpha_cnn"]))
    print(f"[cnn-sweep] picked ({args.pick}):  alpha_cnn = {pick:.3f}")

    df.to_csv(art_dir / "cnn_alpha_sweep.csv", index=False)
    print(f"[cnn-sweep] sweep table -> {art_dir / 'cnn_alpha_sweep.csv'}")

    if args.write:
        art.setdefault("cnn", {})
        art["cnn"]["alpha_cnn"] = float(pick)
        art["cnn"]["sweep_pick_mode"] = args.pick
        art["cnn"]["sweep_best_by_file"] = {
            "alpha": float(best_file["alpha_cnn"]),
            "auc"  : float(best_file["auc_by_file"]),
        }
        art["cnn"]["sweep_best_by_site"] = {
            "alpha": float(best_site["alpha_cnn"]),
            "auc"  : float(best_site["auc_by_site"]),
        }
        # also compute and store by-site AUC at the chosen alpha
        chosen_fused = _fused_oof(tc, art, pick, cnn_logits, student_oof)
        art["cnn"]["sweep_chosen"] = {
            "alpha_cnn": float(pick),
            "auc_by_file": float(_macro_auc(Y, chosen_fused)),
            "auc_by_site_mean": float(df.loc[df["alpha_cnn"] == pick, "auc_by_site"].mean()),
        }
        save_pickle(art, art_dir / "teacher_artifact.pkl")
        print(f"[cnn-sweep] wrote alpha_cnn={pick:.3f} to teacher_artifact.pkl['cnn']")
        print("[cnn-sweep] next: run scripts/06_package_submission.py --tag v4")
    else:
        print("[cnn-sweep] dry run (no --write). Teacher artifact unchanged.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
