"""Run Perch v2 over all train soundscapes and cache outputs.

Outputs under ``artifacts/perch_cache/``:

- ``meta.parquet``          — per-window metadata (row_id, filename, site, hour_utc, month)
- ``embeddings.npy``        — (N, 1536) float32 Perch embeddings
- ``scores.npy``             — (N, C) float32 projected-to-primary-labels logits
- ``filenames.parquet``     — 1 row per file (for later grouping)
- ``mapping.json``          — serialized LabelMapping (primary_labels + proxy info)

Idempotent: if all outputs already exist and ``--overwrite`` is not set, exit early.

Memory: a full cache for ~10k files × 12 windows = 120k rows
- embeddings: 120k × 1536 × 4B ≈ 720 MB
- scores:     120k × 234 × 4B ≈ 112 MB
OK on 16 GB RAM.
"""

from __future__ import annotations

# IMPORTANT: Import TensorFlow BEFORE pandas/pyarrow on macOS. Pandas 2.x eagerly
# imports pyarrow, which statically links its own abseil. When TF loads later it
# ends up resolving abseil symbols against libarrow's copy, which deadlocks
# Perch v2's StableHLO executor (SingleThreadedExecutorImpl -> ThreadPool dtor
# waiting on workers that never signaled exit). Import order here fixes that.
import os as _os
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import tensorflow as _tf  # noqa: F401  (side-effect: resolve absl before arrow)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make common/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.io_utils import write_json  # noqa: E402
from common.paths import artifacts_dir, comp_dir, load_config, perch_cache_dir, perch_model_dir  # noqa: E402
from common.perch import PerchEngine  # noqa: E402
from common.taxonomy import (  # noqa: E402
    build_label_mapping,
    describe_mapping,
    load_perch_labels,
    load_primary_labels,
    load_taxonomy,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overwrite", action="store_true", help="re-run even if cache exists")
    p.add_argument("--limit", type=int, default=None, help="cap #files (debug)")
    p.add_argument("--batch-files", type=int, default=None, help="override perch_batch_files")
    p.add_argument("--subset", choices=["train_soundscapes", "test_soundscapes"], default="train_soundscapes")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    comp = comp_dir()
    cache_dir = perch_cache_dir()
    out_meta = cache_dir / "meta.parquet"
    out_emb = cache_dir / "embeddings.npy"
    out_scores = cache_dir / "scores.npy"
    out_files = cache_dir / "filenames.parquet"
    out_mapping = cache_dir / "mapping.json"

    paths_to_check = [out_meta, out_emb, out_scores, out_files, out_mapping]
    if all(p.exists() for p in paths_to_check) and not args.overwrite:
        print("Cache exists. Use --overwrite to rebuild.")
        for p in paths_to_check:
            print(" -", p, f"{p.stat().st_size / (1<<20):.1f} MB")
        return 0

    # ---- assets ---------------------------------------------------------

    assert comp.exists(), (
        f"competition dir not found: {comp}\n"
        "Download via `uv run --with kagglehub python -c \"import kagglehub; "
        "kagglehub.competition_download('birdclef-2026')\"` and move it into data/birdclef-2026/."
    )
    primary_labels = load_primary_labels(comp)
    taxonomy = load_taxonomy(comp)
    bc_labels = load_perch_labels(perch_model_dir(), labels_name_col=cfg["perch"]["labels_name_col"])
    mapping = build_label_mapping(
        primary_labels=primary_labels,
        taxonomy=taxonomy,
        bc_labels=bc_labels,
        proxy_taxa=cfg["proxy"]["taxa_allowed"],
    )
    print("Label mapping:", describe_mapping(mapping))

    # ---- collect files --------------------------------------------------

    subdir = comp / args.subset
    paths = sorted(subdir.glob("*.ogg"))
    if args.limit:
        paths = paths[: args.limit]
    print(f"{args.subset}: {len(paths)} files")

    # ---- engine ---------------------------------------------------------

    engine = PerchEngine(
        model_dir=perch_model_dir(),
        sample_rate=cfg["audio"]["sample_rate"],
        window_seconds=cfg["audio"]["window_seconds"],
        file_seconds=cfg["audio"]["file_seconds"],
        windows_per_file=cfg["audio"]["windows_per_file"],
        embedding_dim=cfg["perch"]["embedding_dim"],
    )

    batch_files = args.batch_files or cfg["misc"]["perch_batch_files"]
    proxy_reduce = cfg["proxy"]["reduce"]

    n_classes = len(primary_labels)
    n_rows_total = len(paths) * cfg["audio"]["windows_per_file"]
    embeddings = np.empty((n_rows_total, cfg["perch"]["embedding_dim"]), dtype=np.float32)
    scores = np.empty((n_rows_total, n_classes), dtype=np.float32)
    meta_parts: list[pd.DataFrame] = []

    write_row = 0
    t0 = time.time()
    for batch_meta, batch_scores, batch_emb in engine.infer(
        paths,
        mapping,
        batch_files=batch_files,
        proxy_reduce=proxy_reduce,
        verbose=True,
    ):
        n = len(batch_meta)
        embeddings[write_row : write_row + n] = batch_emb
        scores[write_row : write_row + n] = batch_scores
        meta_parts.append(batch_meta)
        write_row += n

    assert write_row == n_rows_total, (write_row, n_rows_total)
    meta_df = pd.concat(meta_parts, ignore_index=True)
    files_df = pd.DataFrame({"filename": [p.name for p in paths]})
    print(f"Perch cache built in {time.time() - t0:.1f}s. Writing...")

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_emb, embeddings)
    np.save(out_scores, scores)
    meta_df.to_parquet(out_meta, index=False)
    files_df.to_parquet(out_files, index=False)
    write_json(
        {
            "subset": args.subset,
            "n_files": len(paths),
            "n_classes": n_classes,
            "primary_labels": mapping.primary_labels,
            "bc_indices": mapping.bc_indices.tolist(),
            "mapped_pos": mapping.mapped_pos.tolist(),
            "mapped_bc_indices": mapping.mapped_bc_indices.tolist(),
            "proxy_pos_to_bc": {int(k): v.tolist() for k, v in mapping.proxy_pos_to_bc.items()},
            "no_label_index": int(mapping.no_label_index),
            "summary": describe_mapping(mapping),
        },
        out_mapping,
    )

    print("Wrote:")
    for p in paths_to_check:
        print(" -", p, f"{p.stat().st_size / (1<<20):.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
