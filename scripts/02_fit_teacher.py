"""Build the teacher artifact — fold-safe OOF + full-data version.

Reads the Perch cache and the labeled slice of train_soundscapes, then:

1. Identifies the ``full_truth`` subset (files whose 12 windows are all labeled).
2. Fits the full-data teacher (priors + scaler/PCA + probe + temporal-lite) on
   **all** 708 labeled windows. This is the teacher that scores the ~127k
   unlabeled cached windows for pseudo-labeling and that ships in the bundle.
3. Runs **GroupKFold-by-file (k=5)** over the 59 labeled files. Each fold fits
   the same components on the training files and predicts on the validation
   files, yielding per-source OOF logits of shape ``(708, 234)`` so every
   downstream stage (fusion sweep, pseudo iter, calibration) can evaluate on
   honest OOF predictions.
4. Scores the entire Perch cache with the full-data teacher so later stages
   can filter pseudo-labels without repeating Perch.

Outputs under ``artifacts/``:

- ``teacher_artifact.pkl``  — full-data teacher ready for the Kaggle bundle.
- ``teacher_cache.pkl``     — OOF per-source logits on the 708 labeled rows,
                               full-cache per-row teacher probs, meta, Y_full.
- ``02_fit_teacher.json``   — summary metrics.

Idempotent: keeps existing outputs unless ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.features import fit_embedding_pipeline, temporal_features_from_seq  # noqa: E402
from common.filenames import parse_soundscape_filename  # noqa: E402
from common.fusion import FusionWeights, fuse_logits, sigmoid  # noqa: E402
from common.io_utils import save_pickle, write_json  # noqa: E402
from common.metrics import macro_roc_auc_skip_empty  # noqa: E402
from common.paths import artifacts_dir, comp_dir, load_config, perch_cache_dir  # noqa: E402
from common.priors import (  # noqa: E402
    build_prior_logits_vec,
    fit_prior_tables,
    lookups_from_serialized,
    serialize_priors,
)
from common.probes import fit_linear_probe  # noqa: E402


# --------------------------------------------------------------------------- args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--skip-fullcache", action="store_true",
                   help="skip scoring the 127k unlabeled cache (debug only)")
    return p.parse_args()


# -------------------------------------------------------------------- data loads


def _load_full_truth(
    comp: Path, windows_per_file: int, primary_labels: list[str]
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return (full_truth_df, Y_full_truth, ordered list of full_files)."""
    labels = pd.read_csv(comp / "train_soundscapes_labels.csv")
    labels["primary_label"] = labels["primary_label"].astype(str)

    def union_labels(series):
        out: set[str] = set()
        for x in series:
            if pd.isna(x):
                continue
            for t in str(x).split(";"):
                t = t.strip()
                if t:
                    out.add(t)
        return sorted(out)

    sc_clean = (
        labels.groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )

    sc_clean["start_sec"] = pd.to_timedelta(sc_clean["start"]).dt.total_seconds().astype(int)
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = (
        sc_clean["filename"].str.replace(".ogg", "", regex=False)
        + "_"
        + sc_clean["end_sec"].astype(str)
    )

    meta_rows = sc_clean["filename"].apply(parse_soundscape_filename).apply(pd.Series)
    sc_clean = pd.concat([sc_clean, meta_rows], axis=1)

    windows_per_filename = sc_clean.groupby("filename").size()
    full_files = sorted(windows_per_filename[windows_per_filename == windows_per_file].index.tolist())
    sc_clean["file_fully_labeled"] = sc_clean["filename"].isin(full_files)

    label_to_idx = {c: i for i, c in enumerate(primary_labels)}
    n_classes = len(primary_labels)
    Y_all = np.zeros((len(sc_clean), n_classes), dtype=np.uint8)
    for i, labels_list in enumerate(sc_clean["label_list"]):
        idxs = [label_to_idx[lbl] for lbl in labels_list if lbl in label_to_idx]
        if idxs:
            Y_all[i, idxs] = 1

    full_truth = (
        sc_clean[sc_clean["file_fully_labeled"]]
        .sort_values(["filename", "end_sec"])
        .reset_index(drop=False)
    )
    Y_full_truth = Y_all[full_truth["index"].to_numpy()]
    return full_truth, Y_full_truth, full_files


# ------------------------------------------------------------------- components


def _fit_temporal_head(
    scores_raw: np.ndarray,
    Y: np.ndarray,
    windows_per_file: int,
    n_classes: int,
    stats: tuple[str, ...],
    seed: int,
) -> dict:
    """Fit per-active-class LogReg over 4 pooled statistics of window-logits."""
    n_files = Y.shape[0] // windows_per_file
    assert Y.shape[0] == n_files * windows_per_file, (Y.shape, windows_per_file)

    seq_scores = scores_raw.reshape(n_files, windows_per_file, n_classes)
    stacked, _ = temporal_features_from_seq(seq_scores, stats=stats)
    Y_file = Y.reshape(n_files, windows_per_file, n_classes).max(axis=1).astype(np.uint8)
    active = np.where(Y_file.sum(axis=0) > 0)[0].astype(np.int32)

    n_stats = stacked.shape[1]
    coef = np.zeros((len(active), n_stats), dtype=np.float32)
    intercept = np.zeros(len(active), dtype=np.float32)

    for k, j in enumerate(active):
        yj = Y_file[:, j]
        Xj = stacked[:, :, j]
        clf = LogisticRegression(
            C=1.0, max_iter=300, solver="liblinear",
            class_weight="balanced", random_state=seed,
        )
        clf.fit(Xj, yj)
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
    temp_artifact: dict,
    windows_per_file: int,
    n_classes: int,
) -> np.ndarray:
    """Return (N, C) window-level temporal logits; 0 on non-active classes."""
    n = scores_raw.shape[0]
    n_files = n // windows_per_file
    assert n == n_files * windows_per_file
    out = np.zeros((n, n_classes), dtype=np.float32)
    active = temp_artifact["active_class_idx"]
    if len(active) == 0:
        return out
    seq = scores_raw.reshape(n_files, windows_per_file, n_classes)
    stats = tuple(temp_artifact["stats"])
    stacked, _ = temporal_features_from_seq(seq, stats=stats)
    sub = stacked[:, :, active]                              # (F, S, K)
    file_logits = np.einsum("fsk,ks->fk", sub, temp_artifact["coef_mat"])
    file_logits += temp_artifact["intercept_vec"][None, :]
    window_logits = np.repeat(file_logits, windows_per_file, axis=0)
    out[:, active] = window_logits.astype(np.float32)
    return out


def _fit_probe_head(X_pca: np.ndarray, Y: np.ndarray, seed: int) -> dict:
    """Per-class LogReg probe; returns artifact-shaped dict."""
    probe = fit_linear_probe(
        X_pca, Y, C=1.0, max_iter=300, solver="liblinear",
        class_weight="balanced", random_state=seed,
    )
    return probe.to_artifact()


def _score_probe_head(X_pca: np.ndarray, probe_artifact: dict, n_classes: int) -> np.ndarray:
    n = X_pca.shape[0]
    out = np.zeros((n, n_classes), dtype=np.float32)
    active = probe_artifact["active_class_idx"]
    if len(active) == 0:
        return out
    logits = (X_pca @ probe_artifact["coef_mat"].T + probe_artifact["intercept_vec"][None, :]).astype(np.float32)
    out[:, active] = logits
    return out


def _fit_priors_artifact(
    meta: pd.DataFrame, Y: np.ndarray, primary_labels: list[str]
) -> dict:
    priors = fit_prior_tables(meta, Y, primary_labels)
    return serialize_priors(priors)


def _score_priors_head(meta: pd.DataFrame, priors_artifact: dict, n_classes: int) -> np.ndarray:
    lookups = lookups_from_serialized(priors_artifact)
    return build_prior_logits_vec(meta, lookups, n_classes)


# -------------------------------------------------------------------- main flow


def main() -> int:
    args = parse_args()
    cfg = load_config()
    comp = comp_dir()
    cache_dir = perch_cache_dir()
    art_dir = artifacts_dir()
    windows_per_file = cfg["audio"]["windows_per_file"]
    pca_dim = cfg["features"]["pca_dim"]
    stats = tuple(cfg["features"]["temporal_stats"])
    seed = int(cfg["misc"]["seed"])

    teacher_path = art_dir / "teacher_artifact.pkl"
    teacher_cache_path = art_dir / "teacher_cache.pkl"

    if all(p.exists() for p in [teacher_path, teacher_cache_path]) and not args.overwrite:
        print("Teacher artifacts already exist. Use --overwrite to rebuild.")
        return 0

    # ---- load cache ----------------------------------------------------

    meta_cache = pd.read_parquet(cache_dir / "meta.parquet")
    embeddings = np.load(cache_dir / "embeddings.npy")
    scores_raw = np.load(cache_dir / "scores.npy")

    with (cache_dir / "mapping.json").open() as f:
        mapping_serialized = json.load(f)
    primary_labels: list[str] = mapping_serialized["primary_labels"]
    n_classes = len(primary_labels)
    print(f"Perch cache: rows={len(meta_cache)}  emb={embeddings.shape}  scores={scores_raw.shape}  C={n_classes}")

    # ---- align labeled full_truth onto cache indices -------------------

    full_truth, Y_full_truth, full_files = _load_full_truth(comp, windows_per_file, primary_labels)
    n_labeled = len(full_truth)
    assert n_labeled == len(full_files) * windows_per_file
    print(f"full_truth: files={len(full_files)}  windows={n_labeled}  active={int((Y_full_truth.sum(axis=0) > 0).sum())}")

    meta_cache_with_idx = meta_cache.reset_index().rename(columns={"index": "_cache_idx"})
    keys = full_truth["row_id"].to_numpy()
    sel = meta_cache_with_idx.set_index("row_id").loc[keys].reset_index()
    sel_idx = sel["_cache_idx"].to_numpy()

    emb_full = embeddings[sel_idx]
    scores_full_raw = scores_raw[sel_idx]
    meta_full = meta_cache.iloc[sel_idx].reset_index(drop=True)
    assert len(meta_full) == n_labeled

    # ---- full-data teacher ---------------------------------------------

    print("[full-data] fit embedding pipeline (scaler + PCA)...")
    pipeline_full = fit_embedding_pipeline(emb_full, pca_dim=pca_dim, random_state=seed)
    X_full_pca = pipeline_full.transform(emb_full)

    print("[full-data] fit priors / probe / temporal-lite on all 708 labeled rows...")
    priors_full_art = _fit_priors_artifact(meta_full, Y_full_truth, primary_labels)
    probe_full_art = _fit_probe_head(X_full_pca, Y_full_truth, seed)
    temp_full_art = _fit_temporal_head(scores_full_raw, Y_full_truth, windows_per_file, n_classes, stats, seed)

    # per-source logits on the labeled rows (in-sample, for sanity)
    in_perch_logits = scores_full_raw.astype(np.float32)
    in_prior_logits = _score_priors_head(meta_full, priors_full_art, n_classes)
    in_probe_logits = _score_probe_head(X_full_pca, probe_full_art, n_classes)
    in_temp_logits = _score_temporal_head(scores_full_raw, temp_full_art, windows_per_file, n_classes)

    # ---- fold-safe OOF on the 708 labeled rows -------------------------

    print(f"[OOF] GroupKFold(k={args.n_folds}) on files...")
    group_ids = np.arange(len(full_files)).repeat(windows_per_file)
    assert len(group_ids) == n_labeled

    oof_perch_logits = in_perch_logits.copy()                       # perch never retrained
    oof_prior_logits = np.zeros_like(oof_perch_logits)
    oof_probe_logits = np.zeros_like(oof_perch_logits)
    oof_temp_logits = np.zeros_like(oof_perch_logits)

    fold_sizes: list[int] = []
    gkf = GroupKFold(n_splits=args.n_folds)
    fold_idx = 0
    t0 = time.time()
    for train_mask, val_mask in [
        (np.isin(np.arange(n_labeled), tr), np.isin(np.arange(n_labeled), va))
        for tr, va in gkf.split(np.zeros(n_labeled), y=None, groups=group_ids)
    ]:
        fold_idx += 1
        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        fold_sizes.append(n_val)
        print(f"  fold {fold_idx}/{args.n_folds}  train={n_train}  val={n_val}")

        emb_tr = emb_full[train_mask]
        emb_va = emb_full[val_mask]
        meta_tr = meta_full.loc[train_mask].reset_index(drop=True)
        meta_va = meta_full.loc[val_mask].reset_index(drop=True)
        scores_tr = scores_full_raw[train_mask]
        scores_va = scores_full_raw[val_mask]
        Y_tr = Y_full_truth[train_mask]

        pipeline_k = fit_embedding_pipeline(emb_tr, pca_dim=pca_dim, random_state=seed)
        X_tr_pca = pipeline_k.transform(emb_tr)
        X_va_pca = pipeline_k.transform(emb_va)

        priors_k = _fit_priors_artifact(meta_tr, Y_tr, primary_labels)
        probe_k = _fit_probe_head(X_tr_pca, Y_tr, seed)
        temp_k = _fit_temporal_head(scores_tr, Y_tr, windows_per_file, n_classes, stats, seed)

        # Validation scoring; temporal head must be evaluated on file-aligned
        # slices (12-window contiguous blocks), which GroupKFold preserves.
        val_idx = np.where(val_mask)[0]
        oof_prior_logits[val_idx] = _score_priors_head(meta_va, priors_k, n_classes)
        oof_probe_logits[val_idx] = _score_probe_head(X_va_pca, probe_k, n_classes)
        oof_temp_logits[val_idx] = _score_temporal_head(scores_va, temp_k, windows_per_file, n_classes)

    print(f"[OOF] done in {time.time() - t0:.1f}s  val sizes={fold_sizes}")

    # ---- seed fusion (0.55/0.15/0.10/0.20) and macro-AUC ---------------

    seed_fw = FusionWeights(alpha_perch=0.55, alpha_prior=0.15, alpha_probe=0.10, alpha_temp=0.20)
    oof_active_mask = oof_probe_logits != 0.0   # columns ever active
    oof_active_cols = np.where(oof_active_mask.any(axis=0))[0].astype(np.int32)
    temp_active_cols = np.where((oof_temp_logits != 0.0).any(axis=0))[0].astype(np.int32)

    fused_oof = (
        seed_fw.alpha_perch * oof_perch_logits
        + seed_fw.alpha_prior * oof_prior_logits
        + seed_fw.alpha_probe * oof_probe_logits
        + seed_fw.alpha_temp * oof_temp_logits
    )
    probs_oof = sigmoid(fused_oof)
    auc, _ = macro_roc_auc_skip_empty(Y_full_truth, probs_oof)
    print(f"[OOF] seed fusion macro AUC = {auc:.4f}  (weights 0.55/0.15/0.10/0.20)")

    # ---- full-cache scoring with the full-data teacher -----------------

    if args.skip_fullcache:
        full_cache_probs = None
        full_cache_logits = None
        print("[full-cache] skipped (--skip-fullcache)")
    else:
        print("[full-cache] scoring all Perch cache rows with full-data teacher...")
        t0 = time.time()
        X_cache_pca = pipeline_full.transform(embeddings)
        prior_full = _score_priors_head(meta_cache, priors_full_art, n_classes)
        probe_full = _score_probe_head(X_cache_pca, probe_full_art, n_classes)
        temp_full = _score_temporal_head(scores_raw, temp_full_art, windows_per_file, n_classes)

        fused_full = (
            seed_fw.alpha_perch * scores_raw.astype(np.float32)
            + seed_fw.alpha_prior * prior_full
            + seed_fw.alpha_probe * probe_full
            + seed_fw.alpha_temp * temp_full
        ).astype(np.float32)
        full_cache_logits = fused_full
        full_cache_probs = sigmoid(fused_full).astype(np.float16)
        del X_cache_pca, prior_full, probe_full, temp_full, fused_full
        print(f"[full-cache] done in {time.time() - t0:.1f}s  probs dtype=float16  shape={full_cache_probs.shape}")

    # ---- write artifacts -----------------------------------------------

    teacher_artifact = {
        "primary_labels": primary_labels,
        "priors": priors_full_art,
        "embedding_pipeline": pipeline_full.to_artifact(),
        "probe": probe_full_art,
        "temporal": {
            "stats": list(stats),
            "active_class_idx": temp_full_art["active_class_idx"].astype(np.int32),
            "coef_mat": temp_full_art["coef_mat"].astype(np.float32),
            "intercept_vec": temp_full_art["intercept_vec"].astype(np.float32),
        },
        "fusion_weights": {
            "alpha_perch": float(seed_fw.alpha_perch),
            "alpha_prior": float(seed_fw.alpha_prior),
            "alpha_probe": float(seed_fw.alpha_probe),
            "alpha_temp":  float(seed_fw.alpha_temp),
        },
        "meta": {
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pseudo_round": None,
            "note": "seed fusion weights 0.55/0.15/0.10/0.20; overwritten by 02b_fusion_sweep.py",
        },
    }
    save_pickle(teacher_artifact, teacher_path)

    save_pickle(
        {
            "primary_labels": primary_labels,
            "full_files": full_files,
            "meta_full": meta_full.to_dict(orient="list"),
            "Y_full_truth": Y_full_truth,
            "labeled_cache_idx": sel_idx.astype(np.int32),
            "windows_per_file": int(windows_per_file),
            # in-sample (sanity / legacy downstream)
            "in_perch_logits": in_perch_logits.astype(np.float32),
            "in_prior_logits": in_prior_logits.astype(np.float32),
            "in_probe_logits": in_probe_logits.astype(np.float32),
            "in_temp_logits":  in_temp_logits.astype(np.float32),
            # OOF per-source logits on the 708 labeled rows
            "oof_perch_logits": oof_perch_logits.astype(np.float32),
            "oof_prior_logits": oof_prior_logits.astype(np.float32),
            "oof_probe_logits": oof_probe_logits.astype(np.float32),
            "oof_temp_logits":  oof_temp_logits.astype(np.float32),
            "oof_active_class_idx": oof_active_cols,
            "oof_temp_active_class_idx": temp_active_cols,
            "oof_group_ids": group_ids.astype(np.int32),
            # full cache scoring (for pseudo-label selection)
            "full_cache_probs": full_cache_probs,
            "full_cache_meta_ok": True,
        },
        teacher_cache_path,
    )

    write_json(
        {
            "n_labeled_windows": int(n_labeled),
            "n_labeled_files": int(len(full_files)),
            "n_classes": int(n_classes),
            "n_folds": int(args.n_folds),
            "oof_seed_macro_auc": float(auc),
            "full_cache_scored": bool(full_cache_probs is not None),
            "seed_fusion_weights": teacher_artifact["fusion_weights"],
        },
        art_dir / "02_fit_teacher.json",
    )
    print("Wrote teacher_artifact.pkl and teacher_cache.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
