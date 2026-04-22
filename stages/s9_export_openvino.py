"""S9 — export S7 fold checkpoints to OpenVINO fp16 IR.

Why OpenVINO (not ONNXRuntime):
    The 2025 2nd place team's submission notebook runs 5-fold inference in
    ~40 min on the Kaggle CPU using OpenVINO fp16 — about 2× faster than
    ONNXRuntime at the same precision. For our 90-min budget that's the
    difference between "fits with room" and "timeout".

Inputs:
    checkpoints/S7/fold_{k}/best.pt   (or swa.pt if present)

Outputs under ``${export_dir}/``:
    fold_{k}.xml      — OpenVINO IR topology
    fold_{k}.bin      — quantized weights (fp16)
    manifest.json     — backbone name, n_classes, mel spec params, fold list
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from common.cloud_paths import CloudCfg
from common.datasets import MelConfig
from common.models import ModelCfg, build_model
from common.taxonomy import load_primary_labels

log = logging.getLogger("bridclef.s9")


def _pick_fold_ckpt(cfg: CloudCfg, stage: str, fold: int) -> Optional[Path]:
    d = cfg.stage_ckpt_dir(stage, fold=fold)
    for name in ("swa.pt", "best.pt", "last.pt"):
        p = d / name
        if p.exists():
            return p
    return None


def _export_one_fold(
    cfg: CloudCfg,
    fold: int,
    n_classes: int,
    mel_cfg: MelConfig,
    source_stage: str = "S7",
) -> dict:
    ck = _pick_fold_ckpt(cfg, source_stage, fold)
    if ck is None:
        log.warning("S9: no ckpt for fold %d under %s, skipping", fold, source_stage)
        return {"fold": fold, "skipped": True}

    backbone_cfg = cfg.raw["backbones"][0]
    m = build_model(ModelCfg(backbone=backbone_cfg["name"], n_classes=n_classes, in_chans=1))
    state = torch.load(str(ck), map_location="cpu", weights_only=False)
    m.load_state_dict(state["model"])
    m.eval()

    # Export input: (1, 1, n_mels, T). T is frames-per-5s-window.
    n_frames = int(np.ceil((mel_cfg.sr * mel_cfg.window_seconds) / mel_cfg.hop)) + 1
    dummy = torch.zeros((1, 1, mel_cfg.n_mels, n_frames), dtype=torch.float32)

    # Step 1: torch -> ONNX.
    onnx_path = cfg.export_dir / f"fold_{fold}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # Use the dynamo exporter (available from torch 2.2+) since it handles
    # nfnet's agc without complaint.
    try:
        torch.onnx.export(
            m, dummy, str(onnx_path),
            input_names=["mel"], output_names=["logits"],
            opset_version=17,
            dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}},
        )
    except Exception as e:  # noqa: BLE001 — fall back to dynamo
        log.warning("torch.onnx.export (legacy) failed (%s); retrying with dynamo", e)
        ep = torch.onnx.export(m, (dummy,), dynamo=True)
        ep.save(str(onnx_path))

    log.info("S9 fold %d: ONNX -> %s", fold, onnx_path)

    # Step 2: ONNX -> OpenVINO IR (fp16).
    ir_xml = cfg.export_dir / f"fold_{fold}.xml"
    try:
        import openvino as ov
        core = ov.Core()
        ov_model = core.read_model(str(onnx_path))
        # Compress to fp16 — roughly halves disk and speeds up Kaggle CPU.
        ov.save_model(ov_model, str(ir_xml), compress_to_fp16=True)
    except Exception as e:  # noqa: BLE001
        log.error("S9 fold %d: OpenVINO convert failed: %s", fold, e)
        raise

    # Best-effort sanity test: random input -> OV forward shape check.
    try:
        import openvino as ov
        compiled = ov.Core().compile_model(str(ir_xml), "CPU")
        result = compiled(dummy.numpy())[compiled.outputs[0]]
        assert result.shape[-1] == n_classes, f"OV output shape {result.shape}"
        log.info("S9 fold %d: OV sanity ok (output %s)", fold, result.shape)
    except Exception as e:  # noqa: BLE001
        log.warning("S9 fold %d: OV sanity check skipped/failed: %s", fold, e)

    # Clean up the intermediate ONNX if configured (default: keep, small).
    return {"fold": fold, "ir_xml": str(ir_xml),
            "size_xml_kb": ir_xml.stat().st_size / 1024,
            "size_bin_kb": (ir_xml.with_suffix(".bin")).stat().st_size / 1024,
            "source_ckpt": str(ck)}


def run(cfg: CloudCfg, args: argparse.Namespace) -> int:  # noqa: ARG001
    primary_labels = load_primary_labels(cfg.comp_dir)
    n_classes = len(primary_labels)
    mel_cfg = MelConfig.from_yaml(cfg.raw)
    n_folds = int(cfg.raw["s5"]["n_folds"])

    results = []
    # Prefer S7 (pseudo-refined), fall back to S5 per-fold if S7 missing.
    for f in range(n_folds):
        src = "S7" if _pick_fold_ckpt(cfg, "S7", f) else "S5"
        out = _export_one_fold(cfg, f, n_classes, mel_cfg, source_stage=src)
        out["source_stage"] = src
        results.append(out)

    manifest = {
        "backbone": cfg.raw["backbones"][0]["name"],
        "n_classes": n_classes,
        "n_folds": n_folds,
        "primary_labels": primary_labels,
        "mel": {
            "sr": mel_cfg.sr, "n_mels": mel_cfg.n_mels,
            "n_fft": mel_cfg.n_fft, "hop": mel_cfg.hop,
            "win_length": mel_cfg.win_length,
            "f_min": mel_cfg.f_min, "f_max": mel_cfg.f_max,
            "power": mel_cfg.power,
            "db_lo": mel_cfg.db_lo, "db_hi": mel_cfg.db_hi,
            "window_seconds": mel_cfg.window_seconds,
            "windows_per_file": cfg.raw["audio"]["windows_per_file"],
        },
        "folds": results,
    }
    (cfg.export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("S9: manifest written -> %s", cfg.export_dir / "manifest.json")
    return 0
