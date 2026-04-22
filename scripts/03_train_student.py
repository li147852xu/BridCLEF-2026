"""Distill the student head on teacher soft targets + labeled truth.

Reads ``artifacts/pseudo_export.pkl`` (produced by ``02c_pseudo_iter.py``) and
trains a small MLP on the 60k+ supervised rows (708 labeled ground truth +
~59k pseudo-positive rows from the strict round). Scoring design:

- **Hybrid targets**: labeled rows use their hard 0/1 Y, pseudo rows use the
  teacher's soft probability (``full_probs`` at the fused-teacher level).
- **Label smoothing** compresses toward 0.5 slightly to counter over-confident
  teacher targets. Applied to both labeled and pseudo rows.
- **Embedding-mixup** inside the MLP training pipeline (no waveform needed).
- **OOF evaluation**: 5-fold GroupKFold-by-file on the 708 labeled rows.
  Each fold trains on (pseudos ∪ labeled_train) and scores labeled_val, so
  the per-fold student logits are fold-safe for the fusion sweep.
- **Full-data student**: one additional pass fit on all supervised rows.
  This is the artifact shipped in the Kaggle bundle at inference time.

Fusion integration:

    fused = α_perch * perch + α_prior * prior + α_probe * probe + α_temp * temp
          + α_student * student

α_student is swept on the OOF grid and the best value is stored alongside
the student weights in ``student_artifact.pkl``. If no positive gain is
found, α_student is set to 0 (safety fallback).

Output: ``artifacts/student_artifact.pkl`` with keys
    type, active_class_idx, w0/b0/w1/b1/w2/b2, alpha_student,
    oof_gain, train_cfg.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.features import apply_embedding_artifact  # noqa: E402
from common.fusion import sigmoid  # noqa: E402
from common.io_utils import load_pickle, save_pickle  # noqa: E402
from common.metrics import macro_roc_auc_skip_empty  # noqa: E402
from common.paths import artifacts_dir, load_config, perch_cache_dir  # noqa: E402
from common.student import MLPArtifact, MLPTrainConfig, fit_mlp_student  # noqa: E402


ALPHA_GRID = [0.0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25,
              0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.0]

JOINT_GRID = {
    "alpha_perch":   [0.30, 0.40, 0.50, 0.60],
    "alpha_prior":   [0.05, 0.10, 0.15],
    "alpha_probe":   [0.05, 0.10, 0.15, 0.20],
    "alpha_temp":    [0.05, 0.10, 0.15],
    "alpha_student": [0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None, help="mps | cpu | cuda")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--skip-oof", action="store_true",
                   help="Skip per-fold OOF training and use in-sample student logits "
                        "for the alpha sweep (faster but leaks).")
    p.add_argument("--sweep-only", action="store_true",
                   help="Reuse existing student MLP weights + cached OOF logits; "
                        "only re-run alpha / joint fusion sweeps against the current "
                        "teacher OOF logits (use after 02d head refit).")
    return p.parse_args()


def _build_train_cfg(cfg: dict, args) -> MLPTrainConfig:
    mlp_cfg = dict(cfg["student"]["mlp"])
    return MLPTrainConfig(
        hidden=tuple(mlp_cfg["hidden"]),
        dropout=float(mlp_cfg["dropout"]),
        lr=float(mlp_cfg["lr"]),
        weight_decay=float(mlp_cfg["weight_decay"]),
        batch_size=int(args.batch_size or mlp_cfg["batch_size"]),
        epochs=int(args.epochs or mlp_cfg["epochs"]),
        mixup_alpha=float(mlp_cfg["mixup_alpha"]),
        label_smoothing=float(mlp_cfg["label_smoothing"]),
        device=str(args.device or mlp_cfg["device"]),
        seed=int(cfg["misc"]["seed"]),
    )


def _predict_logits_from_artifact(X: np.ndarray, artifact: MLPArtifact, n_classes: int) -> np.ndarray:
    """Return window-level student logits mapped to all n_classes (zeros on inactive)."""
    out = np.zeros((X.shape[0], n_classes), dtype=np.float32)
    active = artifact.active_class_idx
    if len(active) == 0:
        return out
    logits = artifact.predict_logits_active(X)
    out[:, active] = logits.astype(np.float32)
    return out


def main() -> int:
    args = parse_args()
    cfg = load_config()
    art_dir = artifacts_dir()
    cache_dir = perch_cache_dir()

    out_path = art_dir / "student_artifact.pkl"
    if out_path.exists() and not args.overwrite and not args.sweep_only:
        print(f"{out_path} exists. Use --overwrite to retrain or --sweep-only to re-sweep alphas.")
        return 0
    if args.sweep_only and not out_path.exists():
        raise FileNotFoundError(f"--sweep-only requires an existing {out_path}")

    # ---- load pseudo export + teacher artifacts --------------------------
    pseudo = load_pickle(art_dir / "pseudo_export.pkl")
    teacher = load_pickle(art_dir / "teacher_artifact.pkl")
    cache = load_pickle(art_dir / "teacher_cache.pkl")

    X_sup = pseudo["X_full_pca"].astype(np.float32)              # (N_sup, D)
    soft = pseudo["soft"].astype(np.float32)                     # (N_sup, C)
    hard = pseudo["hard"].astype(np.int8)                        # (N_sup, C)
    mask = pseudo["mask"].astype(np.uint8)                       # (N_sup, C)
    cache_row_idx_sup = pseudo["cache_row_idx"].astype(np.int64)  # (N_sup,)
    print(
        f"pseudo_export: rows={X_sup.shape[0]}  dim={X_sup.shape[1]}  "
        f"classes={soft.shape[1]}  mask_positives={int(mask.sum())}"
    )

    n_classes = soft.shape[1]
    Y_full_truth = cache["Y_full_truth"].astype(np.uint8)
    labeled_cache_idx = cache["labeled_cache_idx"].astype(np.int64)
    n_labeled = len(labeled_cache_idx)

    # Locate labeled rows inside the pseudo_export row order
    row_to_export = {int(r): i for i, r in enumerate(cache_row_idx_sup)}
    labeled_sup_idx = np.asarray(
        [row_to_export[int(r)] for r in labeled_cache_idx], dtype=np.int64
    )
    is_labeled_in_sup = np.zeros(len(cache_row_idx_sup), dtype=bool)
    is_labeled_in_sup[labeled_sup_idx] = True

    # ---- hybrid targets: hard on labeled, soft on pseudo -----------------
    Y_hybrid = soft.copy()
    Y_hybrid[labeled_sup_idx] = hard[labeled_sup_idx].astype(np.float32)

    train_cfg = _build_train_cfg(cfg, args)
    print(
        f"train_cfg: hidden={train_cfg.hidden}  lr={train_cfg.lr}  "
        f"batch={train_cfg.batch_size}  epochs={train_cfg.epochs}  "
        f"device={train_cfg.device}  mixup_a={train_cfg.mixup_alpha}"
    )

    # ---- OOF student: per-fold training on (pseudos ∪ labeled_train) -----
    group_ids = cache["oof_group_ids"].astype(np.int64)
    oof_student_logits = np.zeros((n_labeled, n_classes), dtype=np.float32)
    oof_active_union = np.zeros(n_classes, dtype=bool)

    if args.sweep_only:
        print("[sweep-only] reusing existing student_artifact.pkl "
              "(MLP weights + cached oof_student_logits unchanged).")
        existing = load_pickle(out_path)
        mlp_full = MLPArtifact(
            active_class_idx=existing["active_class_idx"].astype(np.int32),
            weights={k: existing[k].astype(np.float32)
                     for k in ("w0", "b0", "w1", "b1", "w2", "b2")},
        )
        oof_student_logits = existing["oof_student_logits"].astype(np.float32)
        oof_active_union[mlp_full.active_class_idx] = True
    elif not args.skip_oof:
        gkf = GroupKFold(n_splits=args.n_folds)
        for fold_i, (tr708, va708) in enumerate(
            gkf.split(np.zeros(n_labeled), groups=group_ids), start=1
        ):
            # Mask out val rows in the pseudo_export
            val_sup_idx = labeled_sup_idx[va708]
            keep = np.ones(len(X_sup), dtype=bool)
            keep[val_sup_idx] = False
            X_fold = X_sup[keep]
            Y_fold = Y_hybrid[keep]
            M_fold = mask[keep]
            print(
                f"[OOF] fold {fold_i}/{args.n_folds}  train_rows={keep.sum()}  "
                f"val_rows={len(va708)}  val_files={len(np.unique(group_ids[va708]))}"
            )
            t0 = time.time()
            mlp_k = fit_mlp_student(
                X_fold.astype(np.float32), Y_fold.astype(np.float32), M_fold, train_cfg
            )
            print(f"[OOF] fold {fold_i} trained in {time.time() - t0:.1f}s  "
                  f"active={len(mlp_k.active_class_idx)}")
            X_val = X_sup[val_sup_idx]
            oof_student_logits[va708] = _predict_logits_from_artifact(X_val, mlp_k, n_classes)
            oof_active_union[mlp_k.active_class_idx] = True
    else:
        print("[OOF] skipped — will use in-sample student logits (leaky).")

    # ---- full-data student (for inference) -------------------------------
    if args.sweep_only:
        print("[full] sweep-only: skipping full-data student retrain.")
    else:
        print("[full] training full-data student on all supervised rows...")
        t0 = time.time()
        mlp_full = fit_mlp_student(X_sup, Y_hybrid.astype(np.float32), mask, train_cfg)
        print(f"[full] full-data student trained in {time.time() - t0:.1f}s  "
              f"active={len(mlp_full.active_class_idx)}")

        # If OOF was skipped, substitute in-sample logits on labeled rows
        if args.skip_oof:
            X_lab = X_sup[labeled_sup_idx]
            oof_student_logits = _predict_logits_from_artifact(X_lab, mlp_full, n_classes)

    # ---- alpha_student sweep on OOF --------------------------------------
    fw = teacher["fusion_weights"]
    oof_perch = cache["oof_perch_logits"].astype(np.float32)
    oof_prior = cache["oof_prior_logits"].astype(np.float32)
    oof_probe = cache["oof_probe_logits"].astype(np.float32)
    oof_temp = cache["oof_temp_logits"].astype(np.float32)
    base_fused = (
        fw["alpha_perch"] * oof_perch
        + fw["alpha_prior"] * oof_prior
        + fw["alpha_probe"] * oof_probe
        + fw["alpha_temp"] * oof_temp
    ).astype(np.float32)
    base_auc = macro_roc_auc_skip_empty(Y_full_truth, sigmoid(base_fused))[0]
    print(f"[sweep] baseline OOF macro AUC (no student) = {base_auc:.4f}")

    best = {"alpha": 0.0, "auc": float(base_auc)}
    for a in ALPHA_GRID:
        fused = base_fused + a * oof_student_logits
        auc = macro_roc_auc_skip_empty(Y_full_truth, sigmoid(fused))[0]
        gain = auc - base_auc
        marker = "*" if auc > best["auc"] else " "
        print(f"[sweep] {marker} alpha_student={a:4.2f}  OOF AUC={auc:.4f}  (Δ={gain:+.4f})")
        if auc > best["auc"]:
            best = {"alpha": float(a), "auc": float(auc)}
    print(f"[sweep] best alpha_student={best['alpha']:.2f}  OOF AUC={best['auc']:.4f} "
          f"(Δ vs baseline={best['auc'] - base_auc:+.4f})")

    # ---- joint fusion re-sweep (all 5 weights) with student active ------
    joint_best = {"auc": -1.0}
    n_comb = (
        len(JOINT_GRID["alpha_perch"])
        * len(JOINT_GRID["alpha_prior"])
        * len(JOINT_GRID["alpha_probe"])
        * len(JOINT_GRID["alpha_temp"])
        * len(JOINT_GRID["alpha_student"])
    )
    print(f"[joint] scanning {n_comb} weight combinations (perch/prior/probe/temp/student)...")
    t0 = time.time()
    for ap in JOINT_GRID["alpha_perch"]:
        for aP in JOINT_GRID["alpha_prior"]:
            for ab in JOINT_GRID["alpha_probe"]:
                for at in JOINT_GRID["alpha_temp"]:
                    for ast in JOINT_GRID["alpha_student"]:
                        fused = (
                            ap * oof_perch + aP * oof_prior + ab * oof_probe
                            + at * oof_temp + ast * oof_student_logits
                        )
                        auc = macro_roc_auc_skip_empty(Y_full_truth, sigmoid(fused))[0]
                        if auc > joint_best["auc"]:
                            joint_best = {
                                "auc": float(auc),
                                "alpha_perch": float(ap),
                                "alpha_prior": float(aP),
                                "alpha_probe": float(ab),
                                "alpha_temp":  float(at),
                                "alpha_student": float(ast),
                            }
    print(
        f"[joint] done in {time.time() - t0:.1f}s  best OOF AUC={joint_best['auc']:.4f}  "
        f"fw={{ap={joint_best['alpha_perch']}, aP={joint_best['alpha_prior']}, "
        f"ab={joint_best['alpha_probe']}, at={joint_best['alpha_temp']}, "
        f"ast={joint_best['alpha_student']}}}"
    )

    # If joint sweep beats the alpha-only sweep, adopt the joint fusion
    # weights (they replace the probe-only fusion in teacher_artifact.pkl).
    if joint_best["auc"] > best["auc"]:
        print(
            f"[joint] joint sweep improves OOF AUC "
            f"(+{joint_best['auc'] - best['auc']:.4f}); updating teacher fusion weights."
        )
        teacher["fusion_weights"] = {
            "alpha_perch":   joint_best["alpha_perch"],
            "alpha_prior":   joint_best["alpha_prior"],
            "alpha_probe":   joint_best["alpha_probe"],
            "alpha_temp":    joint_best["alpha_temp"],
        }
        teacher.setdefault("meta", {})
        teacher["meta"]["joint_fusion_auc"] = float(joint_best["auc"])
        teacher["meta"]["joint_fusion_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save_pickle(teacher, art_dir / "teacher_artifact.pkl")
        best_alpha_final = float(joint_best["alpha_student"])
        best_auc_final = float(joint_best["auc"])
    else:
        best_alpha_final = float(best["alpha"])
        best_auc_final = float(best["auc"])

    # ---- save artifact ---------------------------------------------------
    artifact = {
        "type": "mlp",
        **mlp_full.to_artifact(),
        "alpha_student": float(best_alpha_final),
        "oof_gain": float(best_auc_final - base_auc),
        "oof_auc_without_student": float(base_auc),
        "oof_auc_with_student": float(best_auc_final),
        "alpha_only_best": {"alpha": float(best["alpha"]), "auc": float(best["auc"])},
        "joint_best": joint_best,
        "oof_student_logits": oof_student_logits.astype(np.float32),
        "train_cfg": train_cfg.__dict__,
        "notes": {
            "n_sup_rows": int(X_sup.shape[0]),
            "labeled_rows": int(n_labeled),
            "pseudo_rows": int(X_sup.shape[0] - n_labeled),
            "pseudo_round": str(pseudo.get("round", "unknown")),
            "oof_student_used": bool(not args.skip_oof),
        },
    }
    save_pickle(artifact, out_path)
    print(f"Saved student artifact -> {out_path}  alpha_student={best_alpha_final}  "
          f"OOF AUC {base_auc:.4f} -> {best_auc_final:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
