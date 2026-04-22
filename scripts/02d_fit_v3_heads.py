"""Stage 2 head refresh — swap per-class LogReg + temporal-lite → MLP + attention.

Reads ``artifacts/teacher_cache.pkl`` (OOF perch/prior + labeled slice) and
retrains two heads with Stage 2 architectures from scratch:

* **Probe**   joint MLP (``1536 → 512 → K``) with focal+BCE class weights
              (see :mod:`common.probes_mlp`).
* **Temporal** 1-layer 4-head self-attention over the 12-window Perch
               embedding sequence (see :mod:`common.temporal_attn`).

Both heads are fit under the same ``GroupKFold(k=5)`` by-file split that
Stage 1 used (``cache['oof_group_ids']``), so the new OOF logit matrices
are drop-in replacements for ``oof_probe_logits`` / ``oof_temp_logits`` in
``teacher_cache.pkl``. The full-data heads overwrite the corresponding
sections of ``teacher_artifact.pkl`` and set ``probe_type="mlp"`` /
``temporal_type="attn"`` so downstream packaging can dispatch.

The cached per-source *in-sample* logit matrices are also refreshed; legacy
``oof_perch_logits``, ``oof_prior_logits`` are unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.fusion import sigmoid  # noqa: E402
from common.io_utils import load_pickle, save_pickle, write_json  # noqa: E402
from common.metrics import macro_roc_auc_skip_empty  # noqa: E402
from common.paths import artifacts_dir, load_config, perch_cache_dir  # noqa: E402
from common.probes_mlp import (  # noqa: E402
    MLPProbeConfig,
    apply_mlp_probe_artifact,
    fit_mlp_probe,
)
from common.temporal_attn import (  # noqa: E402
    TemporalAttnConfig,
    apply_temporal_attn_artifact,
    fit_temporal_attn,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--probe-epochs", type=int, default=60)
    p.add_argument("--probe-batch", type=int, default=256)
    p.add_argument("--probe-hidden", type=int, default=512)
    p.add_argument("--temp-epochs", type=int, default=80)
    p.add_argument("--temp-batch", type=int, default=16)
    p.add_argument("--temp-dmodel", type=int, default=128)
    p.add_argument("--temp-heads", type=int, default=4)
    p.add_argument("--device", default=None, help="mps | cpu | cuda (default: config)")
    p.add_argument("--skip-probe", action="store_true",
                   help="Keep the existing Stage 1 LogReg probe; only replace temporal.")
    p.add_argument("--skip-temp", action="store_true",
                   help="Keep Stage 1 temporal-lite head; only replace probe.")
    return p.parse_args()


# ------------------------------------------------------------------- helpers


def _score_probe_to_full(
    X: np.ndarray, artifact: dict, n_classes: int
) -> np.ndarray:
    """Scatter MLP probe logits back to the full (N, C) class matrix."""
    out = np.zeros((X.shape[0], n_classes), dtype=np.float32)
    active = artifact["active_class_idx"]
    if len(active) == 0:
        return out
    out[:, active] = apply_mlp_probe_artifact(X, artifact)
    return out


def _score_temporal_to_window(
    emb_seq: np.ndarray,
    artifact: dict,
    n_classes: int,
    windows_per_file: int,
) -> np.ndarray:
    """File-level attn logits broadcast back to all windows."""
    n_files = emb_seq.shape[0]
    n_rows = n_files * windows_per_file
    out = np.zeros((n_rows, n_classes), dtype=np.float32)
    active = artifact["active_class_idx"]
    if len(active) == 0:
        return out
    file_logits = apply_temporal_attn_artifact(emb_seq, artifact)           # (F, K)
    window_logits = np.repeat(file_logits, windows_per_file, axis=0)         # (F*T, K)
    out[:, active] = window_logits.astype(np.float32)
    return out


def _file_level_Y(Y: np.ndarray, windows_per_file: int) -> np.ndarray:
    n_rows, C = Y.shape
    n_files = n_rows // windows_per_file
    return Y.reshape(n_files, windows_per_file, C).max(axis=1).astype(np.uint8)


# --------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    cfg = load_config()
    art_dir = artifacts_dir()
    cache_dir = perch_cache_dir()
    seed = int(cfg["misc"]["seed"])

    teacher_path = art_dir / "teacher_artifact.pkl"
    cache_path = art_dir / "teacher_cache.pkl"
    teacher = load_pickle(teacher_path)
    cache = load_pickle(cache_path)

    if not args.overwrite and teacher.get("probe", {}).get("probe_type") == "mlp" \
            and teacher.get("temporal", {}).get("temporal_type") == "attn":
        print("teacher_artifact.pkl is already v3 (mlp probe + attn temporal); "
              "pass --overwrite to refit.")
        return 0

    # ---- load raw Perch cache (we need 1536-D embeddings, not PCA) ------
    meta_cache = pd.read_parquet(cache_dir / "meta.parquet")
    embeddings = np.load(cache_dir / "embeddings.npy")                       # (N_cache, 1536)
    scores_raw = np.load(cache_dir / "scores.npy").astype(np.float32)        # (N_cache, C)
    with (cache_dir / "mapping.json").open() as f:
        mapping = json.load(f)
    primary_labels = mapping["primary_labels"]
    n_classes = len(primary_labels)

    labeled_idx = cache["labeled_cache_idx"].astype(np.int64)
    emb_lab = embeddings[labeled_idx].astype(np.float32)                     # (708, 1536)
    Y_lab = cache["Y_full_truth"].astype(np.uint8)                           # (708, C)
    n_labeled = Y_lab.shape[0]
    windows_per_file = int(cache["windows_per_file"])
    n_files = n_labeled // windows_per_file
    assert n_files * windows_per_file == n_labeled
    group_ids = cache["oof_group_ids"].astype(np.int64)

    # Per-file view of embeddings and labels (for the attention head)
    emb_seq_lab = emb_lab.reshape(n_files, windows_per_file, emb_lab.shape[1])
    Y_file_lab = _file_level_Y(Y_lab, windows_per_file)                      # (n_files, C)

    # ---- device + configs ---------------------------------------------------
    device = args.device or cfg.get("student", {}).get("mlp", {}).get("device", "mps")
    probe_cfg = MLPProbeConfig(
        hidden=args.probe_hidden,
        epochs=args.probe_epochs,
        batch_size=args.probe_batch,
        device=device,
        seed=seed,
        verbose=True,
    )
    temp_cfg = TemporalAttnConfig(
        d_model=args.temp_dmodel,
        n_heads=args.temp_heads,
        epochs=args.temp_epochs,
        batch_size=args.temp_batch,
        device=device,
        seed=seed,
        verbose=True,
    )
    print(f"device={device}  probe_cfg={probe_cfg}\n  temp_cfg={temp_cfg}")

    # ---- full-data fits ----------------------------------------------------
    if not args.skip_probe:
        t0 = time.time()
        print("\n[full] fitting MLP probe on all 708 labeled windows...")
        probe_full = fit_mlp_probe(emb_lab, Y_lab, cfg=probe_cfg)
        probe_art = probe_full.to_artifact()
        print(f"[full] probe trained in {time.time()-t0:.1f}s  "
              f"active={len(probe_art['active_class_idx'])}")
    else:
        probe_art = teacher["probe"]
        print("[full] skipping probe refit — keeping Stage 1 LogReg probe.")

    if not args.skip_temp:
        t0 = time.time()
        print("\n[full] fitting temporal-attn on all 59 labeled files...")
        temp_full = fit_temporal_attn(emb_seq_lab, Y_file_lab, cfg=temp_cfg)
        temp_art = temp_full.to_artifact()
        print(f"[full] temporal trained in {time.time()-t0:.1f}s  "
              f"active={len(temp_art['active_class_idx'])}")
    else:
        temp_art = teacher["temporal"]
        print("[full] skipping temporal refit — keeping Stage 1 temporal-lite head.")

    # ---- fold-safe OOF -----------------------------------------------------
    oof_probe_logits = np.zeros((n_labeled, n_classes), dtype=np.float32)
    oof_temp_logits = np.zeros((n_labeled, n_classes), dtype=np.float32)

    gkf = GroupKFold(n_splits=args.n_folds)
    t_oof = time.time()
    for fold_i, (tr, va) in enumerate(
        gkf.split(np.zeros(n_labeled), groups=group_ids), start=1
    ):
        emb_tr = emb_lab[tr]
        emb_va = emb_lab[va]
        Y_tr = Y_lab[tr]

        # File-level slices: fold is group-by-file so files never split.
        # Re-derive by unique group_ids within tr/va.
        tr_files = np.unique(group_ids[tr])
        va_files = np.unique(group_ids[va])
        tr_seq = emb_seq_lab[tr_files]
        va_seq = emb_seq_lab[va_files]
        Y_tr_file = Y_file_lab[tr_files]
        # Remap va back onto full 708 ordering
        va_row_idx_full = va

        print(
            f"\n[OOF] fold {fold_i}/{args.n_folds}  "
            f"train_rows={len(tr)} train_files={len(tr_files)}  "
            f"val_rows={len(va)} val_files={len(va_files)}"
        )

        if not args.skip_probe:
            t0 = time.time()
            probe_k = fit_mlp_probe(emb_tr, Y_tr, cfg=probe_cfg)
            probe_k_art = probe_k.to_artifact()
            print(f"[OOF] fold {fold_i} probe in {time.time()-t0:.1f}s")
            oof_probe_logits[va_row_idx_full] = _score_probe_to_full(
                emb_va, probe_k_art, n_classes,
            )

        if not args.skip_temp:
            t0 = time.time()
            temp_k = fit_temporal_attn(tr_seq, Y_tr_file, cfg=temp_cfg)
            temp_k_art = temp_k.to_artifact()
            print(f"[OOF] fold {fold_i} temp in {time.time()-t0:.1f}s")
            va_scores = _score_temporal_to_window(
                va_seq, temp_k_art, n_classes, windows_per_file,
            )  # (len(va_files) * T, C) in va_files order
            # Rows are sorted by file in the teacher cache; group_id == file_idx,
            # so each va_file occupies rows 12*f .. 12*f+12 in the 708 matrix.
            for local_i, f in enumerate(va_files):
                dst = np.arange(int(f) * windows_per_file,
                                (int(f) + 1) * windows_per_file)
                src = np.arange(local_i * windows_per_file,
                                (local_i + 1) * windows_per_file)
                oof_temp_logits[dst] = va_scores[src]

    # ---- Stage 1 fallback: if a head is skipped, reuse the existing OOF
    if args.skip_probe:
        oof_probe_logits = cache["oof_probe_logits"].astype(np.float32).copy()
    if args.skip_temp:
        oof_temp_logits = cache["oof_temp_logits"].astype(np.float32).copy()

    print(f"[OOF] done in {time.time()-t_oof:.1f}s")

    # ---- evaluate OOF vs. existing Stage 1 baseline ------------------------
    Y = Y_lab
    oof_perch = cache["oof_perch_logits"].astype(np.float32)
    oof_prior = cache["oof_prior_logits"].astype(np.float32)

    fw = teacher["fusion_weights"]
    base_fused = (
        fw["alpha_perch"] * oof_perch + fw["alpha_prior"] * oof_prior
        + fw["alpha_probe"] * cache["oof_probe_logits"].astype(np.float32)
        + fw["alpha_temp"]  * cache["oof_temp_logits"].astype(np.float32)
    )
    v3_fused = (
        fw["alpha_perch"] * oof_perch + fw["alpha_prior"] * oof_prior
        + fw["alpha_probe"] * oof_probe_logits
        + fw["alpha_temp"]  * oof_temp_logits
    )
    base_auc = float(macro_roc_auc_skip_empty(Y, sigmoid(base_fused))[0])
    v3_auc = float(macro_roc_auc_skip_empty(Y, sigmoid(v3_fused))[0])
    print(f"\n[OOF] baseline Stage-1 fused AUC = {base_auc:.4f}")
    print(f"[OOF] Stage-2 (v3 heads)     AUC = {v3_auc:.4f}  Δ={v3_auc - base_auc:+.4f}")

    # ---- persist -----------------------------------------------------------
    # Mirror the new heads' in-sample logits for downstream debugging
    if not args.skip_probe:
        in_probe_v3 = _score_probe_to_full(emb_lab, probe_art, n_classes)
    else:
        in_probe_v3 = cache["in_probe_logits"].astype(np.float32)
    if not args.skip_temp:
        in_temp_v3 = _score_temporal_to_window(
            emb_seq_lab, temp_art, n_classes, windows_per_file,
        )
    else:
        in_temp_v3 = cache["in_temp_logits"].astype(np.float32)

    # Update active-class summary for the fusion sweep
    oof_probe_active = np.where((oof_probe_logits != 0.0).any(axis=0))[0].astype(np.int32)
    oof_temp_active = np.where((oof_temp_logits != 0.0).any(axis=0))[0].astype(np.int32)

    cache.update(
        {
            "in_probe_logits": in_probe_v3.astype(np.float32),
            "in_temp_logits":  in_temp_v3.astype(np.float32),
            "oof_probe_logits": oof_probe_logits.astype(np.float32),
            "oof_temp_logits":  oof_temp_logits.astype(np.float32),
            "oof_active_class_idx": oof_probe_active,
            "oof_temp_active_class_idx": oof_temp_active,
            "heads_version": "v3",
        }
    )
    save_pickle(cache, cache_path)

    teacher["probe"] = probe_art
    teacher["temporal"] = temp_art
    meta = teacher.setdefault("meta", {})
    meta["heads_version"] = "v3"
    meta["heads_refit_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["heads_oof_auc_before"] = base_auc
    meta["heads_oof_auc_after_same_fw"] = v3_auc
    save_pickle(teacher, teacher_path)

    write_json(
        {
            "heads_version": "v3",
            "baseline_auc": base_auc,
            "v3_same_fw_auc": v3_auc,
            "delta": v3_auc - base_auc,
            "probe_active": int(len(teacher["probe"]["active_class_idx"])),
            "temp_active":  int(len(teacher["temporal"]["active_class_idx"])),
            "skip_probe": bool(args.skip_probe),
            "skip_temp":  bool(args.skip_temp),
        },
        art_dir / "02d_fit_v3_heads.json",
    )
    print("Saved v3 heads into teacher_artifact.pkl and teacher_cache.pkl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
