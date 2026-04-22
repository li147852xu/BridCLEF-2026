"""S3 — Perch v2 teacher pseudo-labels for all ``train_soundscapes``.

Purpose:
    Produce a dense (n_windows, 234) soft-label tensor over every 5-second
    window of every file in ``train_soundscapes/``. Downstream stages use this
    as:
      * FT positive/negative mining signal,
      * pseudo-label threshold base,
      * domain-shift robustness (Perch sees all 23 sites, unlike the 708 hard
        rows which only cover 8).

Sharding / resume:
    * Files are sorted alphabetically, then split into ``s3.output_shards``
      contiguous chunks. Each shard writes one ``.npz`` when it finishes.
    * Within a shard we also write a ``.progress.json`` every
      ``s3.ckpt_every_files`` files. On restart we reload the partial arrays
      from that file and continue from the next file index.

Output layout (under ``${perch_cache}/``):
    meta.parquet                       row_id, filename, site, hour_utc, month
    shard_{i:02d}/probs_fp16.npy       (N_i, 234) sigmoid-scaled probabilities
    shard_{i:02d}/embeddings_fp16.npy  (N_i, 1536) Perch embeddings  [optional]
    shard_{i:02d}/meta.parquet         per-shard meta
    shard_{i:02d}/.progress.json       resume state
    shard_{i:02d}/.done                set on complete
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from common.cloud_paths import CloudCfg
from stages._common import (
    JsonlLogger,
    atomic_write_bytes,
    is_done,
    mark_done,
)

log = logging.getLogger("bridclef.s3")

SAVE_EMBEDDINGS = False  # Plan Y doesn't need them at inference; skip to save disk


def _prep_tf_env(device: str) -> None:
    """Must run BEFORE importing ``common.perch``.

    ``device`` is ``"cuda"`` or ``"cpu"``. TF 2.20 + Blackwell (sm_120, RTX
    5090) has no pre-baked kernels as of early 2026 — it JIT-compiles at
    startup and sometimes falls back noisily. Default is CPU for reliability;
    flip to cuda only if you've tested on the specific host.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    if device == "cuda":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    else:
        # Empty string forces TF to skip GPU enumeration entirely.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _file_list(comp_dir: Path, source: str) -> list[Path]:
    src = comp_dir / source
    files = sorted(src.glob("*.ogg"))
    if not files:
        raise RuntimeError(f"S3: no files in {src}")
    return files


def _shard_slices(n: int, k: int) -> list[tuple[int, int]]:
    sz = (n + k - 1) // k
    return [(i, min(i + sz, n)) for i in range(0, n, sz)]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid; fp32 in, fp32 out.
    x = x.astype(np.float32, copy=False)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _shard_dir(cfg: CloudCfg, shard_idx: int) -> Path:
    p = cfg.perch_cache / f"shard_{shard_idx:02d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_progress(shard_dir: Path) -> dict:
    pg = shard_dir / ".progress.json"
    if not pg.exists():
        return {"done_files": 0}
    try:
        return json.loads(pg.read_text())
    except Exception:  # noqa: BLE001
        return {"done_files": 0}


def _save_progress(shard_dir: Path, payload: dict) -> None:
    pg = shard_dir / ".progress.json"
    atomic_write_bytes(pg, json.dumps(payload, indent=2).encode())


def _run_one_shard(
    shard_idx: int,
    files: Sequence[Path],
    cfg: CloudCfg,
    mapping,           # LabelMapping
    engine,            # PerchEngine
    jl: JsonlLogger,
) -> None:
    sdir = _shard_dir(cfg, shard_idx)
    done_flag = sdir / ".done"
    if is_done(done_flag):
        log.info("shard %02d: already done, skipping", shard_idx)
        return

    n_classes = len(mapping.primary_labels)
    wpf = engine.windows_per_file
    batch_files = int(cfg.raw["s3"]["batch_files"])
    ckpt_every = int(cfg.raw["s3"]["ckpt_every_files"])

    progress = _load_progress(sdir)
    done_files = int(progress.get("done_files", 0))
    if done_files >= len(files):
        mark_done(done_flag, {"n_files": len(files)})
        log.info("shard %02d: progress says complete, marking done", shard_idx)
        return

    # Load partial arrays if resuming.
    probs_path = sdir / "probs_fp16.npy"
    meta_path  = sdir / "meta.parquet"
    if done_files > 0 and probs_path.exists() and meta_path.exists():
        probs = np.load(probs_path, mmap_mode=None).astype(np.float16)
        meta_df = pd.read_parquet(meta_path)
        assert probs.shape[0] == len(meta_df) == done_files * wpf, \
            f"resume size mismatch: probs={probs.shape} meta={len(meta_df)} " \
            f"done={done_files}*{wpf}={done_files * wpf}"
        log.info("shard %02d: resuming at file %d/%d (rows=%d)",
                 shard_idx, done_files, len(files), len(meta_df))
    else:
        probs = np.empty((0, n_classes), dtype=np.float16)
        meta_df = pd.DataFrame()
        done_files = 0

    remaining = list(files[done_files:])
    t0 = time.time()
    new_files_done = 0

    # Delegate all the heavy lifting to PerchEngine.infer (it already batches
    # files internally and applies the label projection + genus proxy).
    for sub_meta, sub_scores, _emb in engine.infer(
        paths=remaining,
        mapping=mapping,
        batch_files=batch_files,
        proxy_reduce="max",
        verbose=False,
    ):
        # Perch returns raw logits already remapped to the 234-class space.
        # Convert to probabilities for easier downstream pseudo-label thresholding.
        sub_probs = _sigmoid(sub_scores).astype(np.float16)

        # Append.
        probs = np.concatenate([probs, sub_probs], axis=0) if probs.size else sub_probs
        meta_df = pd.concat([meta_df, sub_meta], ignore_index=True) if len(meta_df) else sub_meta

        batch_file_count = len(sub_meta) // wpf
        new_files_done += batch_file_count
        done_files += batch_file_count

        # Periodic ckpt so a crash costs < ckpt_every_files.
        if new_files_done and (new_files_done % ckpt_every == 0 or done_files == len(files)):
            np.save(probs_path, probs)
            meta_df.to_parquet(meta_path, index=False)
            _save_progress(sdir, {
                "done_files": done_files,
                "total_files": len(files),
                "rows": int(probs.shape[0]),
                "ts": time.time(),
            })
            dt = time.time() - t0
            spd = new_files_done / max(dt, 1e-3)
            eta_s = (len(files) - done_files) / max(spd, 1e-3)
            log.info("shard %02d: %d/%d files  (%.2f file/s  eta %.1f min)",
                     shard_idx, done_files, len(files), spd, eta_s / 60)
            jl.log(shard=shard_idx, done=done_files, total=len(files),
                   file_per_s=spd, eta_min=eta_s / 60, rows=int(probs.shape[0]))

        # Be nice to RAM — TF holds references a while.
        del sub_meta, sub_scores, sub_probs
        gc.collect()

    # Final flush + mark done.
    np.save(probs_path, probs)
    meta_df.to_parquet(meta_path, index=False)
    _save_progress(sdir, {
        "done_files": done_files,
        "total_files": len(files),
        "rows": int(probs.shape[0]),
        "ts": time.time(),
        "final": True,
    })
    mark_done(done_flag, {"n_files": len(files), "rows": int(probs.shape[0])})
    log.info("shard %02d: complete (files=%d, rows=%d)",
             shard_idx, done_files, int(probs.shape[0]))


# --------------------------------------------------------------------------
# Stage entry
# --------------------------------------------------------------------------

def run(cfg: CloudCfg, args: argparse.Namespace) -> int:  # noqa: ARG001
    s3 = cfg.raw["s3"]
    device = str(s3.get("device", "cpu")).lower()
    log.info("S3 starting on device=%s", device)
    _prep_tf_env(device)

    # Lazy imports so S2 users don't pay the TF load cost.
    from common.perch import PerchEngine
    from common.taxonomy import build_label_mapping, load_primary_labels, load_taxonomy, load_perch_labels

    # Build label mapping (234 classes with genus proxy for Amphibia/Insecta).
    taxonomy = load_taxonomy(cfg.comp_dir)
    primary_labels = load_primary_labels(cfg.comp_dir)
    bc_labels = load_perch_labels(cfg.perch_model)
    mapping = build_label_mapping(
        primary_labels=primary_labels,
        taxonomy=taxonomy,
        bc_labels=bc_labels,
        proxy_taxa=("Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"),
    )
    log.info("S3: label mapping  n_classes=%d  direct=%d  proxy=%d",
             len(mapping.primary_labels), len(mapping.mapped_pos),
             len(mapping.proxy_pos_to_bc))

    # Load Perch v2.
    log.info("S3: loading Perch v2 from %s", cfg.perch_model)
    engine = PerchEngine(
        model_dir=cfg.perch_model,
        sample_rate=cfg.raw["audio"]["sample_rate"],
        window_seconds=cfg.raw["audio"]["window_seconds"],
        file_seconds=cfg.raw["audio"]["file_seconds"],
        windows_per_file=cfg.raw["audio"]["windows_per_file"],
    )

    # Shard.
    files = _file_list(cfg.comp_dir, s3["input_source"])
    slices = _shard_slices(len(files), int(s3["output_shards"]))
    log.info("S3: %d files -> %d shards", len(files), len(slices))
    jl = JsonlLogger(cfg.stage_log("S3"))

    for i, (lo, hi) in enumerate(slices):
        _run_one_shard(i, files[lo:hi], cfg, mapping, engine, jl)

    # Write a global manifest so later stages (S5, S6) can mmap shards.
    manifest = {
        "n_shards": len(slices),
        "source": s3["input_source"],
        "primary_labels": mapping.primary_labels,
        "proxy_taxa": ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"],
        "generated_at": time.time(),
    }
    atomic_write_bytes(cfg.perch_cache / "manifest.json",
                       json.dumps(manifest, indent=2).encode())
    log.info("S3: all shards done, manifest written to %s",
             cfg.perch_cache / "manifest.json")
    return 0


if __name__ == "__main__":
    from common.cloud_paths import load_cloud_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_cloud_config(a.config)
    cfg.mkdirs()
    sys.exit(run(cfg, argparse.Namespace()))
