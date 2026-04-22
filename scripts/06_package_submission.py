"""Bundle all artifacts for Kaggle submission.

Produces ``artifacts/submission_bundle/submission_bundle.pkl`` which contains
everything the Kaggle notebook needs to reproduce the teacher pipeline at
inference time:

- primary_labels (ordered)
- Perch mapping (bc_indices / mapped_pos / proxy dict)
- prior tables
- scaler + PCA weights
- probe weights (linear)
- temporal-lite weights
- fusion weights
- (optional) student MLP weights
- (optional) isotonic calibration knots

The notebook loads this single file, applies it to Perch outputs, and writes
``submission.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.io_utils import load_pickle, save_pickle, write_json  # noqa: E402
from common.paths import artifacts_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="v0", help="version tag, embedded in the output")
    p.add_argument("--include-student", action="store_true")
    p.add_argument("--include-calibration", action="store_true")
    p.add_argument("--alpha-cnn", type=float, default=None,
                   help="fusion weight for the mel-CNN ensemble at inference. "
                        "If omitted, falls back to teacher_artifact.pkl['cnn']['alpha_cnn'] "
                        "or 0.0.")
    p.add_argument("--alpha-perch", type=float, default=None,
                   help="override fusion_weights['alpha_perch'] (default: keep teacher_artifact).")
    p.add_argument("--alpha-prior", type=float, default=None,
                   help="override fusion_weights['alpha_prior'].")
    p.add_argument("--alpha-probe", type=float, default=None,
                   help="override fusion_weights['alpha_probe']. Set to 0 to drop MLP probe.")
    p.add_argument("--alpha-temp", type=float, default=None,
                   help="override fusion_weights['alpha_temp']. Set to 0 to drop self-attn temporal head.")
    p.add_argument("--alpha-student", type=float, default=None,
                   help="override student_artifact['alpha_student'] (only if student bundled).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    art_dir = artifacts_dir()
    bundle_dir = art_dir / "submission_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    out_path = bundle_dir / "submission_bundle.pkl"

    teacher = load_pickle(art_dir / "teacher_artifact.pkl")
    cache_dir = art_dir / "perch_cache"
    with (cache_dir / "mapping.json").open() as f:
        mapping = json.load(f)

    # Build fusion_weights from teacher's current values, then apply overrides
    # from CLI. This keeps teacher_artifact.pkl pristine for other experiments.
    fusion_weights = dict(teacher["fusion_weights"])
    for key, val in [("alpha_perch", args.alpha_perch),
                     ("alpha_prior", args.alpha_prior),
                     ("alpha_probe", args.alpha_probe),
                     ("alpha_temp",  args.alpha_temp)]:
        if val is not None:
            print(f"  fusion override: {key} {fusion_weights.get(key)} -> {val}")
            fusion_weights[key] = float(val)

    n_classes = len(teacher["primary_labels"])
    bundle = {
        "tag": args.tag,
        "primary_labels": teacher["primary_labels"],
        "mapping": {
            "bc_indices": np.asarray(mapping["bc_indices"], dtype=np.int32),
            "mapped_pos": np.asarray(mapping["mapped_pos"], dtype=np.int32),
            "mapped_bc_indices": np.asarray(mapping["mapped_bc_indices"], dtype=np.int32),
            "proxy_pos_to_bc": {
                int(k): np.asarray(v, dtype=np.int32) for k, v in mapping["proxy_pos_to_bc"].items()
            },
            "no_label_index": int(mapping["no_label_index"]),
        },
        "priors": teacher["priors"],
        "embedding_pipeline": teacher["embedding_pipeline"],
        "probe": teacher["probe"],
        "temporal": teacher["temporal"],
        "fusion_weights": fusion_weights,
        "postprocess": teacher.get("postprocess", {
            "apply_topn1": True,           # safe default
            "apply_isotonic": False,
            "topn_first": False,
        }),
        # Stage 2.3 per-class delta (monotone-invariant on macro AUC; reserved
        # for Stage 3 ensemble / rank-aware fusion). Default zeros.
        "delta_per_class": teacher.get(
            "delta_per_class",
            np.zeros(n_classes, dtype=np.float32),
        ).astype(np.float32),
        "meta": teacher.get("meta", {}),
    }

    # Expose head types at top level for quick dispatch in the submit notebook.
    probe_type = bundle["probe"].get("probe_type", "linear")
    temporal_type = bundle["temporal"].get("temporal_type", "lite")
    bundle["probe_type"] = str(probe_type)
    bundle["temporal_type"] = str(temporal_type)

    # Always include the student if the artifact exists (alpha_student=0 is a no-op
    # and we'd rather keep the weights for downstream debugging / Stage 2 fusion).
    sp = art_dir / "student_artifact.pkl"
    if args.include_student or sp.exists():
        if not sp.exists():
            raise FileNotFoundError(sp)
        student_blob = load_pickle(sp)
        if args.alpha_student is not None:
            orig = float(student_blob.get("alpha_student", 0.0))
            print(f"  student override: alpha_student {orig} -> {args.alpha_student}")
            student_blob = dict(student_blob)
            student_blob["alpha_student"] = float(args.alpha_student)
        bundle["student"] = student_blob

    cp = art_dir / "calibration_artifact.pkl"
    if args.include_calibration or cp.exists():
        if not cp.exists():
            raise FileNotFoundError(cp)
        bundle["calibration"] = load_pickle(cp)

    # Stage 3.2 / 3.3: mel-CNN ensemble fusion weight. ONNX files themselves
    # live in a separate Kaggle dataset (tiantanghuaxiao/birdclef-2026-cnn);
    # the submit notebook auto-discovers them. Here we only record alpha_cnn.
    cnn_meta = teacher.get("cnn", {})
    if args.alpha_cnn is not None:
        cnn_meta["alpha_cnn"] = float(args.alpha_cnn)
    else:
        cnn_meta["alpha_cnn"] = float(cnn_meta.get("alpha_cnn", 0.0))
    bundle["cnn"] = cnn_meta

    save_pickle(bundle, out_path)
    size_mb = out_path.stat().st_size / (1 << 20)
    print(f"Wrote bundle -> {out_path}  ({size_mb:.2f} MB)")
    info = {
        "tag": args.tag,
        "size_mb": size_mb,
        "keys": sorted(bundle.keys()),
        "n_classes": n_classes,
        "n_mapped_direct": int(len(mapping["mapped_pos"])),
        "n_proxy": int(len(mapping["proxy_pos_to_bc"])),
        "probe_type": bundle["probe_type"],
        "temporal_type": bundle["temporal_type"],
        "fusion_weights": bundle["fusion_weights"],
        "postprocess": bundle["postprocess"],
        "delta_per_class_stats": {
            "min": float(bundle["delta_per_class"].min()),
            "max": float(bundle["delta_per_class"].max()),
            "mean": float(bundle["delta_per_class"].mean()),
        },
    }
    if "student" in bundle:
        info["student_alpha"] = float(bundle["student"].get("alpha_student", 0.0))
        info["student_oof_auc"] = float(bundle["student"].get("oof_auc_with_student", 0.0))
    if "calibration" in bundle:
        info["n_classes_calibrated"] = len(bundle["calibration"].get("per_class", {}))
    info["cnn"] = {
        "alpha_cnn": float(bundle["cnn"]["alpha_cnn"]),
        "note": "ONNX models are not in this bundle; loaded by submit notebook from "
                "'tiantanghuaxiao/birdclef-2026-cnn' dataset if alpha_cnn > 0.",
    }
    write_json(info, bundle_dir / "bundle_info.json")

    # Emit dataset-metadata.json ONLY if one doesn't already exist; never clobber
    # an edited version (the user has a real `tiantanghuaxiao/birdclef-2026`
    # dataset that we don't want to replace with a placeholder id).
    ds_meta_path = bundle_dir / "dataset-metadata.json"
    if not ds_meta_path.exists():
        ds_meta = {
            "title": "BridCLEF+ 2026 submission bundle",
            "id": "REPLACE_WITH_YOUR_USERNAME/birdclef-2026-bundle",
            "licenses": [{"name": "CC0-1.0"}],
        }
        ds_meta_path.write_text(json.dumps(ds_meta, indent=2))
    print(
        "Edit artifacts/submission_bundle/dataset-metadata.json to replace the username,\n"
        "then run: `kaggle datasets create -p artifacts/submission_bundle/` (first time)\n"
        "or     : `kaggle datasets version -p artifacts/submission_bundle/ -m 'v{tag}'`"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
