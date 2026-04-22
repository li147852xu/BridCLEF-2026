"""S6 — iter-1 pseudo-label inference on all train_soundscapes windows.

Pipeline:
  1. Load the 5 fold ``best.pt`` (EMA weights already folded in at val).
  2. For each train_soundscapes window (cached in mel_cache/train_soundscapes):
       - forward under TTA shifts listed in cfg.s6.tta
       - average sigmoid across folds × shifts
  3. Threshold:
       target_vec[c] = 1.0 if prob >= pos_thr
                       0.0 if prob <= neg_thr
                       (skip -> drop from target; treated as "don't know")
  4. Save rows whose target has at least one non-neutral entry into a sparse
     npz (``pseudo_iter1.npz``) for S7 to ingest.

TTA note:
  We only have cached mel per 5s non-overlapping window; "shift" at this level
  means *fetching a different cached window*. For proper ±1.5s shifts we'd
  need raw audio and re-mel. For Plan Y we settle for the simpler
  "neighbor-window averaging" — already covered by the post-processing
  smoothing in the submission notebook, so S6 runs with only the identity
  shift (shift_s: 0). The YAML keeps the other entries for future S7 tuning
  but this file ignores non-zero shifts with a warning.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.cloud_paths import CloudCfg
from common.datasets import (
    MelConfig,
    SoundscapeWindowDataset,
    collate,
)
from common.models import ModelCfg, build_model
from common.taxonomy import load_primary_labels
from stages._common import JsonlLogger

log = logging.getLogger("bridclef.s6")


def _load_fold_models(cfg: CloudCfg, n_folds: int, n_classes: int, device: str):
    backbone_cfg = cfg.raw["backbones"][0]
    models = []
    for f in range(n_folds):
        ck = cfg.stage_ckpt_dir("S5", fold=f) / "best.pt"
        if not ck.exists():
            ck = cfg.stage_ckpt_dir("S5", fold=f) / "last.pt"
        if not ck.exists():
            log.warning("S6: fold %d has no checkpoint, skipping", f)
            continue
        m = build_model(ModelCfg(backbone=backbone_cfg["name"], n_classes=n_classes, in_chans=1))
        state = torch.load(str(ck), map_location=device, weights_only=False)
        m.load_state_dict(state["model"])
        m.to(device).eval()
        models.append(m)
        log.info("S6: loaded fold %d from %s", f, ck)
    if not models:
        raise RuntimeError("S6: no fold checkpoints found under S5/")
    return models


def _enumerate_all_windows(cfg: CloudCfg) -> list[tuple[str, int]]:
    """Every 5s window of every soundscape mel cache file."""
    mel_dir = cfg.mel_cache / "train_soundscapes"
    items: list[tuple[str, int]] = []
    for f in sorted(mel_dir.glob("*.npz")):
        try:
            with np.load(f, allow_pickle=True) as d:
                n = int(d["mel_u8"].shape[0])
        except Exception as e:  # noqa: BLE001
            log.warning("S6: bad cache %s (%s), skipping", f, e)
            continue
        items.extend((f.stem, i) for i in range(n))
    return items


def run(cfg: CloudCfg, args: argparse.Namespace) -> int:  # noqa: ARG001
    import pandas as pd
    primary_labels = load_primary_labels(cfg.comp_dir)
    n_classes = len(primary_labels)
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}
    mel_cfg = MelConfig.from_yaml(cfg.raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_folds = int(cfg.raw["s5"]["n_folds"])
    models = _load_fold_models(cfg, n_folds, n_classes, device)

    # Build a fresh "inference" dataset over *every* cached window, with
    # dummy labels — SoundscapeWindowDataset ignores label columns when we
    # don't need them, but it still expects a DataFrame; fake one.
    items = _enumerate_all_windows(cfg)
    fake_df = pd.DataFrame({
        "filename": [f"{stem}.ogg" for stem, _ in items],
        "window_idx": [w for _, w in items],
        "primary_label": [""] * len(items),
    })
    ds = SoundscapeWindowDataset(
        labels_df=fake_df,
        mel_cache_dir=cfg.mel_cache / "train_soundscapes",
        label_to_idx=label_to_idx,
        mel_cfg=mel_cfg,
        specaug=None,
        weight=1.0,
        train=False,
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4,
                        collate_fn=collate)

    tta = cfg.raw["s6"].get("tta", [{"shift_s": 0.0}])
    if any(abs(float(t.get("shift_s", 0))) > 1e-6 for t in tta):
        log.warning("S6: non-zero TTA shifts are ignored here (mel cache is "
                    "windowed). Submission notebook handles waveform-level TTA.")

    pos_thr = float(cfg.raw["s6"]["pos_threshold"])
    neg_thr = float(cfg.raw["s6"]["neg_threshold"])
    jsonl = JsonlLogger(cfg.stage_log("S6"))

    all_probs = np.zeros((len(items), n_classes), dtype=np.float32)
    with torch.no_grad():
        offset = 0
        for b, (x, _, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            probs = None
            for m in models:
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits = m(x)
                p = torch.sigmoid(logits).float().cpu().numpy()
                probs = p if probs is None else probs + p
            probs /= float(len(models))
            all_probs[offset:offset + probs.shape[0]] = probs
            offset += probs.shape[0]
            if (b + 1) % 50 == 0:
                log.info("S6: batch %d / %d  rows=%d", b + 1, len(loader), offset)
                jsonl.log(batch=b + 1, total=len(loader), rows=offset)

    # Threshold.
    targets = np.full_like(all_probs, fill_value=np.nan, dtype=np.float16)
    targets[all_probs >= pos_thr] = 1.0
    targets[all_probs <= neg_thr] = 0.0
    keep_mask = np.isfinite(targets).any(axis=1)
    log.info("S6: thresholded rows kept=%d / %d", int(keep_mask.sum()), len(items))

    out = cfg.artifacts_dir() if hasattr(cfg, "artifacts_dir") else cfg.work_root / "artifacts"
    out_dir = cfg.work_root / "artifacts" / "pseudo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fill NaN with a neutral 0.5 for storage, but S7 filters via a "mask"
    # companion array.
    tgt_store = np.where(np.isfinite(targets), targets, np.float16(0.5))
    mask = np.isfinite(targets).astype(np.uint8)
    files = np.array([f"{stem}.ogg" for stem, _ in items], dtype=object)
    wins = np.array([w for _, w in items], dtype=np.int32)

    files_kept = files[keep_mask]
    wins_kept = wins[keep_mask]
    targets_kept = tgt_store[keep_mask].astype(np.float16)
    mask_kept = mask[keep_mask].astype(np.uint8)

    out_path = out_dir / "pseudo_iter1.npz"
    np.savez_compressed(
        out_path,
        files=files_kept,
        window_idx=wins_kept,
        targets=targets_kept,
        mask=mask_kept,
        primary_labels=np.array(primary_labels, dtype=object),
        pos_threshold=np.float32(pos_thr),
        neg_threshold=np.float32(neg_thr),
    )
    log.info("S6: wrote %s  rows=%d  size_mb=%.1f",
             out_path, len(files_kept), out_path.stat().st_size / 1e6)
    return 0
