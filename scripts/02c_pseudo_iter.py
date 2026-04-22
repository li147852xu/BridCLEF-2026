"""Iterative pseudo-labeling of the 127k Perch-cached windows (fold-safe OOF).

This is the ``s1_pseudo_iter`` stage. It consumes the labeled-only teacher
artifacts produced by ``02_fit_teacher.py`` + ``02b_fusion_sweep.py`` and runs
**1-3 rounds** of window-level pseudo-labeling, refitting the *probe head* on
``labeled ∪ pseudo_positives`` and re-sweeping the fusion weights.

Key correctness design
----------------------
The baseline OOF macro AUC (~0.8946) relies on **per-fold PCA** for the probe
head: each fold's PCA basis is fit on that fold's labeled-train embeddings
only. A naive global-PCA re-fit degrades OOF AUC by ~0.17 simply from the
basis change, even without pseudos. So this script:

1. Keeps the frozen per-fold ``pipeline_k`` + priors_k + temp_k from
   round-0 (labeled-only). Only the **probe head** is retrained per round.
2. For each fold ``k``:
     - transforms the full 127k cache with ``pipeline_k``,
     - fuses (perch, prior_k_full_cache, probe_k_{r-1}, temp_k_full_cache)
       to get ``full_probs_k``,
     - selects pseudos against ``full_probs_k``,
     - refits ``probe_k_r`` on ``X_cache_pca_k`` with a mask covering
       ``labeled_train_k ∪ pseudo_positive_rows`` (the fold's validation
       rows are **never** used for supervision),
     - scores ``labeled_val_k`` with ``probe_k_r`` → new OOF probe logits.
3. In parallel, the full-data teacher is rebuilt the same way (global PCA,
   global priors/temp, probe_full refit on labeled ∪ full-data pseudos) for
   eventual inference use.
4. Fusion weights are re-swept on the new OOF logits; a round is accepted
   only if the swept OOF macro AUC gains ≥ ``--min-gain``.

Pseudo-negative rows are **discarded**: for rare classes the teacher
predicts ~0 on ~99% of windows, and pushing those as ground-truth negatives
overwhelms the handful of positives. Labeled negatives are plenty.

Outputs under ``artifacts/``
----------------------------
- ``teacher_artifact.pkl`` — updated in place with the best-round full-data
  probe and fusion weights, plus an isotonic placeholder key.
- ``teacher_cache.pkl``    — updated OOF probe logits + final full-cache
  probs + ``final_round`` tag.
- ``pseudo_export.pkl``    — supervised rows (labeled + pseudo positives)
  with PCA features, soft probs and hard targets for the student stage.
- ``02c_pseudo_iter.json`` — per-round report.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.features import (  # noqa: E402
    EmbeddingPipeline,
    apply_embedding_artifact,
    fit_embedding_pipeline,
    temporal_features_from_seq,
)
from common.fusion import sigmoid  # noqa: E402
from common.io_utils import load_pickle, save_pickle, write_json  # noqa: E402
from common.metrics import macro_roc_auc_skip_empty  # noqa: E402
from common.paths import artifacts_dir, load_config, perch_cache_dir  # noqa: E402
from common.priors import (  # noqa: E402
    build_prior_logits_vec,
    fit_prior_tables,
    lookups_from_serialized,
    serialize_priors,
)
from common.probes import fit_linear_probe, fit_linear_probe_masked  # noqa: E402
from common.pseudo import PseudoLabelConfig, select_pseudo  # noqa: E402


ROUNDS_DEFAULT = [
    {"name": "r1_strict",   "pos_thr": 0.99, "neg_thr": 0.005, "file_thr": 0.99},
    {"name": "r2_balanced", "pos_thr": 0.97, "neg_thr": 0.01,  "file_thr": 0.97},
    {"name": "r3_loose",    "pos_thr": 0.95, "neg_thr": 0.02,  "file_thr": 0.95},
]


FUSION_GRID = {
    "alpha_perch": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    "alpha_prior": [0.05, 0.10, 0.15, 0.20],
    "alpha_probe": [0.05, 0.10, 0.15, 0.20, 0.25],
    "alpha_temp":  [0.05, 0.10, 0.15, 0.20, 0.25],
}


# ===================================================================== helpers


def _fit_temporal_head(
    scores_raw: np.ndarray,
    Y: np.ndarray,
    windows_per_file: int,
    n_classes: int,
    stats: tuple[str, ...],
    seed: int,
) -> dict:
    n_files = Y.shape[0] // windows_per_file
    seq_scores = scores_raw.reshape(n_files, windows_per_file, n_classes)
    stacked, _ = temporal_features_from_seq(seq_scores, stats=stats)
    Y_file = Y.reshape(n_files, windows_per_file, n_classes).max(axis=1).astype(np.uint8)
    active = np.where(Y_file.sum(axis=0) > 0)[0].astype(np.int32)
    n_stats = stacked.shape[1]
    coef = np.zeros((len(active), n_stats), dtype=np.float32)
    intercept = np.zeros(len(active), dtype=np.float32)
    for k, j in enumerate(active):
        clf = LogisticRegression(
            C=1.0, max_iter=300, solver="liblinear",
            class_weight="balanced", random_state=seed,
        )
        clf.fit(stacked[:, :, j], Y_file[:, j])
        coef[k] = clf.coef_[0].astype(np.float32)
        intercept[k] = np.float32(clf.intercept_[0])
    return {
        "stats": list(stats),
        "active_class_idx": active,
        "coef_mat": coef,
        "intercept_vec": intercept,
    }


def _score_temporal_head(
    scores_raw: np.ndarray,
    temp_art: dict,
    windows_per_file: int,
    n_classes: int,
) -> np.ndarray:
    n = scores_raw.shape[0]
    n_files = n // windows_per_file
    out = np.zeros((n, n_classes), dtype=np.float32)
    active = temp_art["active_class_idx"]
    if len(active) == 0:
        return out
    seq = scores_raw.reshape(n_files, windows_per_file, n_classes)
    stats = tuple(temp_art["stats"])
    stacked, _ = temporal_features_from_seq(seq, stats=stats)
    sub = stacked[:, :, active]
    file_logits = np.einsum("fsk,ks->fk", sub, temp_art["coef_mat"])
    file_logits += temp_art["intercept_vec"][None, :]
    out[:, active] = np.repeat(file_logits, windows_per_file, axis=0).astype(np.float32)
    return out


def _score_probe(X_pca: np.ndarray, probe_art: dict, n_classes: int) -> np.ndarray:
    out = np.zeros((X_pca.shape[0], n_classes), dtype=np.float32)
    active = probe_art["active_class_idx"]
    if len(active) == 0:
        return out
    out[:, active] = (X_pca @ probe_art["coef_mat"].T + probe_art["intercept_vec"][None, :]).astype(np.float32)
    return out


def _fuse(perch: np.ndarray, prior: np.ndarray, probe: np.ndarray, temp: np.ndarray, fw: dict) -> np.ndarray:
    return (
        fw["alpha_perch"] * perch
        + fw["alpha_prior"] * prior
        + fw["alpha_probe"] * probe
        + fw["alpha_temp"] * temp
    ).astype(np.float32)


def _sweep_fusion(
    Y: np.ndarray,
    perch: np.ndarray,
    prior: np.ndarray,
    probe: np.ndarray,
    temp: np.ndarray,
) -> tuple[dict, float]:
    best = {"auc": -np.inf}
    for ap in FUSION_GRID["alpha_perch"]:
        for aP in FUSION_GRID["alpha_prior"]:
            for ab in FUSION_GRID["alpha_probe"]:
                for at in FUSION_GRID["alpha_temp"]:
                    fused = ap * perch + aP * prior + ab * probe + at * temp
                    auc, _ = macro_roc_auc_skip_empty(Y, sigmoid(fused))
                    if auc > best["auc"]:
                        best = {
                            "auc": float(auc),
                            "alpha_perch": float(ap),
                            "alpha_prior": float(aP),
                            "alpha_probe": float(ab),
                            "alpha_temp":  float(at),
                        }
    fw = {k: best[k] for k in ("alpha_perch", "alpha_prior", "alpha_probe", "alpha_temp")}
    return fw, best["auc"]


def _select_pseudos_from_probs(
    full_probs: np.ndarray,
    n_files_cache: int,
    windows_per_file: int,
    n_classes: int,
    labeled_cache_idx: np.ndarray,
    unlabeled_cache_idx: np.ndarray,
    pl_cfg: PseudoLabelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (combined_mask, combined_Y) over cache rows, using pseudo POSITIVES
    from ``full_probs`` + labeled ground truth on labeled rows."""
    file_probs_max = (
        full_probs.reshape(n_files_cache, windows_per_file, n_classes)
        .max(axis=1)
        .astype(np.float32)
    )
    hard, mask = select_pseudo(full_probs, file_probs_max, windows_per_file, pl_cfg)
    # never let pseudos touch labeled rows
    hard[labeled_cache_idx] = -1
    mask[labeled_cache_idx] = 0
    pos_mask_unl = np.zeros(full_probs.shape, dtype=np.uint8)
    pos_mask_unl[unlabeled_cache_idx] = (hard[unlabeled_cache_idx] == 1).astype(np.uint8)
    return pos_mask_unl, hard  # caller merges with labeled (positive-only pseudos)


def _merge_pseudos_with_labeled(
    n_rows_cache: int,
    n_classes: int,
    labeled_cache_idx: np.ndarray,
    labeled_Y: np.ndarray,
    labeled_rows_subset: np.ndarray | None,   # 708-index subset of labeled rows to supervise (None = all)
    pseudo_pos_mask: np.ndarray,              # (N, C) uint8; 1 on pseudo-positive rows only
) -> tuple[np.ndarray, np.ndarray]:
    combined_Y = np.zeros((n_rows_cache, n_classes), dtype=np.int8)
    combined_mask = np.zeros((n_rows_cache, n_classes), dtype=np.uint8)
    # pseudo positives as Y=1, mask=1
    combined_Y[pseudo_pos_mask == 1] = 1
    combined_mask[:] = pseudo_pos_mask
    # labeled: mark per-row from subset
    lab_idx = labeled_cache_idx if labeled_rows_subset is None else labeled_cache_idx[labeled_rows_subset]
    lab_Y = labeled_Y if labeled_rows_subset is None else labeled_Y[labeled_rows_subset]
    combined_Y[lab_idx] = lab_Y.astype(np.int8)
    combined_mask[lab_idx] = 1
    return combined_Y, combined_mask


# ===================================================================== main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--min-gain", type=float, default=0.001)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    art_dir = artifacts_dir()
    cache_dir = perch_cache_dir()
    windows_per_file = cfg["audio"]["windows_per_file"]
    pca_dim = int(cfg["features"]["pca_dim"])
    stats = tuple(cfg["features"]["temporal_stats"])
    seed = int(cfg["misc"]["seed"])
    n_folds = int(args.n_folds)

    teacher_path = art_dir / "teacher_artifact.pkl"
    cache_path = art_dir / "teacher_cache.pkl"
    pseudo_path = art_dir / "pseudo_export.pkl"

    if pseudo_path.exists() and not args.overwrite:
        print(f"{pseudo_path} exists. Use --overwrite to re-iterate.")
        return 0

    teacher = load_pickle(teacher_path)
    cache = load_pickle(cache_path)
    primary_labels = teacher["primary_labels"]
    n_classes = len(primary_labels)

    Y_full_truth = cache["Y_full_truth"].astype(np.uint8)
    labeled_cache_idx = cache["labeled_cache_idx"].astype(np.int64)
    n_labeled = len(labeled_cache_idx)
    assert n_labeled == len(Y_full_truth)

    meta_cache = pd.read_parquet(cache_dir / "meta.parquet")
    embeddings = np.load(cache_dir / "embeddings.npy")
    scores_raw = np.load(cache_dir / "scores.npy")
    n_rows_cache = len(meta_cache)
    n_files_cache = n_rows_cache // windows_per_file
    unlabeled_cache_idx = np.setdiff1d(
        np.arange(n_rows_cache, dtype=np.int64), labeled_cache_idx, assume_unique=False
    )
    print(f"cache rows={n_rows_cache}  labeled={n_labeled}  unlabeled={len(unlabeled_cache_idx)}")

    # Labeled-only emb/meta/scores (reconstructed from labeled_cache_idx)
    emb_labeled = embeddings[labeled_cache_idx]
    meta_labeled = meta_cache.iloc[labeled_cache_idx].reset_index(drop=True)
    scores_labeled = scores_raw[labeled_cache_idx]

    # fold groups already in cache
    group_ids = cache["oof_group_ids"].astype(np.int64)
    assert len(group_ids) == n_labeled

    # ---- Rebuild per-fold round-0 components (PCA_k, priors_k, temp_k, probe_k_r0)
    print("[setup] rebuilding per-fold teachers (PCA + priors + temp + probe r0)...")
    t0 = time.time()
    gkf = GroupKFold(n_splits=n_folds)
    folds: list[dict] = []
    for fold_i, (tr708, va708) in enumerate(
        gkf.split(np.zeros(n_labeled), y=None, groups=group_ids), start=1
    ):
        emb_tr = emb_labeled[tr708]
        meta_tr = meta_labeled.iloc[tr708].reset_index(drop=True)
        scores_tr = scores_labeled[tr708]
        Y_tr = Y_full_truth[tr708]

        pipe_k = fit_embedding_pipeline(emb_tr, pca_dim=pca_dim, random_state=seed)
        X_tr_pca = pipe_k.transform(emb_tr)

        priors_k = serialize_priors(fit_prior_tables(meta_tr, Y_tr, primary_labels))
        temp_k = _fit_temporal_head(scores_tr, Y_tr, windows_per_file, n_classes, stats, seed)

        probe_k_r0 = fit_linear_probe(
            X_tr_pca, Y_tr,
            C=1.0, max_iter=300, solver="liblinear",
            class_weight="balanced", random_state=seed,
            tqdm_desc=f"r0/fold{fold_i}/probe",
        ).to_artifact()

        folds.append({
            "fold": fold_i,
            "tr": tr708.astype(np.int64),
            "va": va708.astype(np.int64),
            "pipeline": pipe_k.to_artifact(),
            "priors": priors_k,
            "temp": temp_k,
            "probe_r": probe_k_r0,    # current round's probe (starts at r0 = labeled-only)
        })
    print(f"[setup] folds ready in {time.time() - t0:.1f}s")

    # ---- Per-fold full-cache prior + temp logits (frozen across rounds) --------
    print("[setup] scoring per-fold prior/temp on full cache (frozen)...")
    t0 = time.time()
    for f in folds:
        lookups = lookups_from_serialized(f["priors"])
        f["prior_full_cache"] = build_prior_logits_vec(meta_cache, lookups, n_classes).astype(np.float32)
        f["temp_full_cache"] = _score_temporal_head(scores_raw, f["temp"], windows_per_file, n_classes)
    print(f"[setup] per-fold prior+temp full-cache in {time.time() - t0:.1f}s")

    # ---- Full-data teacher frozen components (round-0) -------------------------
    pipeline_full_art = teacher["embedding_pipeline"]     # PCA on all 708 labeled
    priors_full_art = teacher["priors"]
    temp_full_art = teacher["temporal"]
    probe_full_current = teacher["probe"]

    X_cache_pca_full = apply_embedding_artifact(embeddings, pipeline_full_art)
    prior_full_cache_logits = build_prior_logits_vec(
        meta_cache, lookups_from_serialized(priors_full_art), n_classes
    ).astype(np.float32)
    temp_full_cache_logits = _score_temporal_head(scores_raw, temp_full_art, windows_per_file, n_classes)

    # ---- Frozen OOF per-source (perch/prior/temp) and baseline OOF -------------
    oof_perch = cache["oof_perch_logits"].astype(np.float32)
    oof_prior = cache["oof_prior_logits"].astype(np.float32)
    oof_temp = cache["oof_temp_logits"].astype(np.float32)
    oof_probe_current = cache["oof_probe_logits"].astype(np.float32)
    fw = deepcopy(teacher["fusion_weights"])
    fused_oof = _fuse(oof_perch, oof_prior, oof_probe_current, oof_temp, fw)
    base_auc = macro_roc_auc_skip_empty(Y_full_truth, sigmoid(fused_oof))[0]
    print(f"[round 0] baseline OOF macro AUC = {base_auc:.4f}  fw={fw}")

    best_auc = float(base_auc)
    best_state = {
        "round": "baseline",
        "fw": fw,
        "probe_full": probe_full_current,
        "oof_probe": oof_probe_current.copy(),
        "probe_full_cache_logits": _score_probe(X_cache_pca_full, probe_full_current, n_classes),
        "pseudo_pos_mask_full": np.zeros(n_rows_cache, dtype=np.uint8),  # placeholder
        "fold_probes": [f["probe_r"] for f in folds],
    }

    reports: list[dict] = []
    latest_pseudo_export: dict | None = None
    rounds = ROUNDS_DEFAULT[: args.max_rounds]

    for r_i, round_cfg_dict in enumerate(rounds, start=1):
        round_name = round_cfg_dict["name"]
        pl_cfg = PseudoLabelConfig(
            name=round_name,
            pos_thr=float(round_cfg_dict["pos_thr"]),
            neg_thr=float(round_cfg_dict["neg_thr"]),
            file_thr=float(round_cfg_dict["file_thr"]),
        )
        print(
            f"[round {r_i}:{round_name}] pos_thr={pl_cfg.pos_thr} "
            f"neg_thr={pl_cfg.neg_thr} file_thr={pl_cfg.file_thr}"
        )

        # ------- Full-data teacher probe refit on labeled ∪ full pseudos --------
        probe_full_cache_current = _score_probe(X_cache_pca_full, probe_full_current, n_classes)
        full_fused = _fuse(
            scores_raw.astype(np.float32),
            prior_full_cache_logits,
            probe_full_cache_current,
            temp_full_cache_logits,
            fw,
        )
        full_probs_full = sigmoid(full_fused).astype(np.float32)

        pos_mask_full, _ = _select_pseudos_from_probs(
            full_probs_full, n_files_cache, windows_per_file, n_classes,
            labeled_cache_idx, unlabeled_cache_idx, pl_cfg,
        )
        combined_Y_full, combined_mask_full = _merge_pseudos_with_labeled(
            n_rows_cache, n_classes, labeled_cache_idx, Y_full_truth,
            labeled_rows_subset=None, pseudo_pos_mask=pos_mask_full,
        )
        pseudo_pos_rows_full = int((pos_mask_full.any(axis=1)).sum())
        pseudo_pos_total_full = int(pos_mask_full.sum())
        classes_with_pos_full = int((pos_mask_full.sum(axis=0) > 0).sum())
        print(
            f"  [full] pseudo rows={pseudo_pos_rows_full} "
            f"total labels={pseudo_pos_total_full} classes={classes_with_pos_full}"
        )

        t0 = time.time()
        probe_full_new = fit_linear_probe_masked(
            X_cache_pca_full, combined_Y_full, combined_mask_full,
            C=1.0, max_iter=300, solver="liblinear",
            class_weight="balanced", random_state=seed,
            neg_per_pos=None, max_rows_per_class=None,
            tqdm_desc=f"r{r_i}/full/probe",
        ).to_artifact()
        print(f"  [full] probe refit in {time.time() - t0:.1f}s")

        # ------- Per-fold probe refit (OOF) ------------------------------------
        oof_probe_new = np.zeros_like(oof_probe_current)
        fold_probes_new: list[dict] = []
        total_pseudo_per_fold: list[int] = []
        t0 = time.time()
        for f in folds:
            fold_i = f["fold"]
            X_cache_pca_k = apply_embedding_artifact(embeddings, f["pipeline"]).astype(np.float32)
            probe_k_cache_current = _score_probe(X_cache_pca_k, f["probe_r"], n_classes)

            full_fused_k = _fuse(
                scores_raw.astype(np.float32),
                f["prior_full_cache"],
                probe_k_cache_current,
                f["temp_full_cache"],
                fw,
            )
            full_probs_k = sigmoid(full_fused_k).astype(np.float32)
            pos_mask_k, _ = _select_pseudos_from_probs(
                full_probs_k, n_files_cache, windows_per_file, n_classes,
                labeled_cache_idx, unlabeled_cache_idx, pl_cfg,
            )
            combined_Y_k, combined_mask_k = _merge_pseudos_with_labeled(
                n_rows_cache, n_classes, labeled_cache_idx, Y_full_truth,
                labeled_rows_subset=f["tr"], pseudo_pos_mask=pos_mask_k,
            )
            total_pseudo_per_fold.append(int(pos_mask_k.sum()))

            probe_k_new = fit_linear_probe_masked(
                X_cache_pca_k, combined_Y_k, combined_mask_k,
                C=1.0, max_iter=300, solver="liblinear",
                class_weight="balanced", random_state=seed,
                neg_per_pos=None, max_rows_per_class=None,
                tqdm_desc=f"r{r_i}/fold{fold_i}/probe",
            ).to_artifact()
            fold_probes_new.append(probe_k_new)

            va = f["va"]
            X_val_k = X_cache_pca_k[labeled_cache_idx[va]]
            oof_probe_new[va] = _score_probe(X_val_k, probe_k_new, n_classes)

            del X_cache_pca_k, probe_k_cache_current, full_fused_k, full_probs_k
            gc.collect()
        print(
            f"  [folds] probe refits in {time.time() - t0:.1f}s  "
            f"pseudo labels/fold={total_pseudo_per_fold}"
        )

        # ------- Re-sweep fusion on new OOF probe --------------------------------
        fw_new, auc_new = _sweep_fusion(
            Y_full_truth, oof_perch, oof_prior, oof_probe_new, oof_temp,
        )
        gain = auc_new - best_auc
        print(
            f"[round {r_i}] swept fw={fw_new}  OOF macro AUC = {auc_new:.4f}  "
            f"(Δ vs best={gain:+.4f})"
        )

        reports.append({
            "round": round_name,
            "pseudo_cfg": pl_cfg.__dict__,
            "full_pseudo_rows": pseudo_pos_rows_full,
            "full_pseudo_labels": pseudo_pos_total_full,
            "full_pseudo_classes": classes_with_pos_full,
            "pseudo_labels_per_fold": total_pseudo_per_fold,
            "fw_swept": fw_new,
            "oof_auc": float(auc_new),
            "oof_delta_vs_best": float(gain),
            "accepted": bool(gain >= args.min_gain),
        })

        # Always keep the latest round's pseudo export for the student stage,
        # even if OOF AUC did not improve — the student (non-linear MLP) may
        # still benefit from an order-of-magnitude larger training set.
        latest_pseudo_export = {
            "combined_Y_full": combined_Y_full,
            "combined_mask_full": combined_mask_full,
            "full_probs_full": full_probs_full,
            "round": round_name,
        }

        if gain >= args.min_gain:
            fw = fw_new
            best_auc = float(auc_new)
            oof_probe_current = oof_probe_new.astype(np.float32)
            probe_full_current = probe_full_new
            for f, pk in zip(folds, fold_probes_new):
                f["probe_r"] = pk
            best_state = {
                "round": round_name,
                "fw": fw_new,
                "probe_full": probe_full_new,
                "oof_probe": oof_probe_new.astype(np.float32),
                "probe_full_cache_logits": _score_probe(X_cache_pca_full, probe_full_new, n_classes),
                "pseudo_pos_mask_full": pos_mask_full,
                "fold_probes": fold_probes_new,
                "combined_Y_full": combined_Y_full,
                "combined_mask_full": combined_mask_full,
                "full_probs_full": full_probs_full,
            }
        else:
            print(f"[round {r_i}] plateau — stopping iteration (probe does not benefit)")
            break

    # ===================================================================== write-outs
    teacher["probe"] = best_state["probe_full"]
    teacher["fusion_weights"] = best_state["fw"]
    teacher.setdefault("meta", {})
    teacher["meta"]["pseudo_round"] = best_state["round"]
    teacher["meta"]["oof_auc"] = best_state.get("auc", best_auc)
    teacher["meta"]["pseudo_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_pickle(teacher, teacher_path)
    print(
        f"Wrote teacher_artifact.pkl  round={best_state['round']}  "
        f"oof_auc={best_auc:.4f}  fw={best_state['fw']}"
    )

    # final full-cache probs using the best full-data teacher and swept fw
    final_full_fused = _fuse(
        scores_raw.astype(np.float32),
        prior_full_cache_logits,
        best_state["probe_full_cache_logits"],
        temp_full_cache_logits,
        best_state["fw"],
    )
    final_full_probs = sigmoid(final_full_fused).astype(np.float16)
    cache["oof_probe_logits"] = best_state["oof_probe"].astype(np.float32)
    cache["full_cache_probs"] = final_full_probs
    cache["final_round"] = best_state["round"]
    save_pickle(cache, cache_path)
    print(f"Updated teacher_cache.pkl  final_round={best_state['round']}")

    # pseudo export: labeled + full-data pseudo positives.
    # Prefer the latest round's pseudos (even if not accepted by AUC) so the
    # student can consume a larger training set.
    if latest_pseudo_export is not None:
        combined_Y = latest_pseudo_export["combined_Y_full"]
        combined_mask = latest_pseudo_export["combined_mask_full"]
        soft_source = latest_pseudo_export["full_probs_full"]
        pseudo_round_tag = latest_pseudo_export["round"]
    elif "combined_Y_full" in best_state:
        combined_Y = best_state["combined_Y_full"]
        combined_mask = best_state["combined_mask_full"]
        soft_source = best_state["full_probs_full"]
        pseudo_round_tag = best_state["round"]
    else:
        combined_Y = np.zeros((n_rows_cache, n_classes), dtype=np.int8)
        combined_mask = np.zeros((n_rows_cache, n_classes), dtype=np.uint8)
        combined_Y[labeled_cache_idx] = Y_full_truth.astype(np.int8)
        combined_mask[labeled_cache_idx] = 1
        soft_source = sigmoid(final_full_fused).astype(np.float32)
        pseudo_round_tag = "labeled_only"

    sup_rows = np.where(combined_mask.any(axis=1))[0]
    print(f"supervised rows in export = {len(sup_rows)}  "
          f"(labeled={n_labeled}  pseudo={len(sup_rows) - n_labeled})")
    save_pickle(
        {
            "round": pseudo_round_tag,
            "best_round": best_state["round"],
            "X_full_pca": X_cache_pca_full[sup_rows].astype(np.float32),
            "soft": soft_source[sup_rows].astype(np.float32),
            "mask": combined_mask[sup_rows].astype(np.uint8),
            "hard": combined_Y[sup_rows].astype(np.int8),
            "cache_row_idx": sup_rows.astype(np.int64),
        },
        pseudo_path,
    )
    print(f"Wrote pseudo_export.pkl  rows={len(sup_rows)}  round={pseudo_round_tag}")

    write_json(
        {
            "baseline_oof_auc": float(base_auc),
            "final_oof_auc": float(best_auc),
            "final_round": best_state["round"],
            "rounds": reports,
            "final_fusion_weights": best_state["fw"],
        },
        art_dir / "02c_pseudo_iter.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
