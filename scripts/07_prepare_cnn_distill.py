#!/usr/bin/env python
"""
Stage 3.1 prep: bake a small pickle that the Kaggle GPU notebook can ingest for
CNN distillation. Pulls from artifacts/teacher_cache.pkl, keeps:

  - meta_row_id        : list[str], len 127896
  - full_cache_probs   : uint8 quantized (0-255) to keep size down (~30 MB)
  - primary_labels     : list[str]
  - labeled_cache_idx  : int32[708]
  - Y_full_truth       : uint8[708, K]

Writes to artifacts/cnn_distill/teacher_cache_distill.pkl and prepares a
Kaggle dataset-metadata.json for one-shot upload:

  kaggle datasets version -p artifacts/cnn_distill -m "distill v1" -r skip
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULTS = {
    "teacher_cache": "artifacts/teacher_cache.pkl",
    "perch_meta"   : "artifacts/perch_cache/meta.parquet",
    "out_dir"      : "artifacts/cnn_distill",
    "dataset_slug" : "tiantanghuaxiao/birdclef-2026-distill",
    "dataset_title": "BirdCLEF 2026 CNN distillation cache",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-cache", default=DEFAULTS["teacher_cache"])
    ap.add_argument("--perch-meta",    default=DEFAULTS["perch_meta"])
    ap.add_argument("--out-dir",       default=DEFAULTS["out_dir"])
    ap.add_argument("--dataset-slug",  default=DEFAULTS["dataset_slug"])
    ap.add_argument("--dataset-title", default=DEFAULTS["dataset_title"])
    ap.add_argument("--keep-float16",  action="store_true",
                    help="Ship full_cache_probs as float16 instead of uint8 (~60 MB)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    tc_path   = (root / args.teacher_cache).resolve()
    meta_path = (root / args.perch_meta   ).resolve()
    out_dir   = (root / args.out_dir      ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[distill] loading {tc_path}")
    with open(tc_path, "rb") as f:
        tc = pickle.load(f)

    print(f"[distill] loading {meta_path}")
    meta = pd.read_parquet(meta_path)
    meta_row_id = meta["row_id"].tolist()
    n = len(meta_row_id)

    probs = np.asarray(tc["full_cache_probs"])
    assert probs.shape[0] == n, f"probs rows {probs.shape[0]} != meta rows {n}"
    K = probs.shape[1]

    if args.keep_float16:
        probs_out = probs.astype(np.float16)
        print(f"[distill] keeping float16  ({probs_out.nbytes / 1e6:.1f} MB)")
    else:
        p32 = probs.astype(np.float32, copy=False)
        p32 = np.clip(p32, 0.0, 1.0)
        probs_out = np.rint(p32 * 255.0).astype(np.uint8)
        print(f"[distill] quantized to uint8  ({probs_out.nbytes / 1e6:.1f} MB)")

    payload = {
        "meta_row_id"      : meta_row_id,
        "full_cache_probs" : probs_out,
        "primary_labels"   : list(tc["primary_labels"]),
        "labeled_cache_idx": np.asarray(tc["labeled_cache_idx"], dtype=np.int32),
        "Y_full_truth"     : np.asarray(tc["Y_full_truth"], dtype=np.uint8),
        "K"                : int(K),
        "N"                : int(n),
    }

    out_pkl = out_dir / "teacher_cache_distill.pkl"
    print(f"[distill] writing {out_pkl}")
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[distill]   size = {out_pkl.stat().st_size / 1e6:.2f} MB")

    # Only emit dataset-metadata.json if missing (don't clobber user config)
    ds_meta_path = out_dir / "dataset-metadata.json"
    if not ds_meta_path.exists():
        ds_meta = {
            "title"   : args.dataset_title,
            "id"      : args.dataset_slug,
            "licenses": [{"name": "CC0-1.0"}],
        }
        ds_meta_path.write_text(json.dumps(ds_meta, indent=2))
        print(f"[distill] wrote {ds_meta_path}")
    else:
        print(f"[distill] keeping existing {ds_meta_path}")

    print("\nNext steps:")
    print(f"  # First time only (creates the dataset):")
    print(f"  kaggle datasets create -p {out_dir.relative_to(root)} -r skip")
    print(f"  # Subsequent updates:")
    print(f"  kaggle datasets version -p {out_dir.relative_to(root)} "
          f"-m 'distill v1' -r skip")
    print(f"\nThen in the Kaggle GPU notebook, add '{args.dataset_slug}' as an input.")


if __name__ == "__main__":
    main()
