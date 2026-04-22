"""S5 — 5-fold fine-tune of eca_nfnet_l0 on competition data.

Per-fold flow:
  1. Build label index, mel config, augment params.
  2. Build datasets:
       - train_audio (weak labels, always in train set)
       - train_soundscapes 708 hard labels split by site via GroupKFold
  3. ConcatDataset with hard labels up-sampled to roughly match train_audio size.
  4. Run common.training.Trainer for ``epochs`` epochs with resume.
  5. Save best.pt + swa.pt + last.pt into ``${ckpt_root}/S5/fold_{k}/``.

The outer loop always tries to resume every fold; when ``--fold`` is passed on
the CLI, only that fold runs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from common.augment import SpecAugParams
from common.cloud_paths import CloudCfg
from common.datasets import (
    MelConfig,
    SoundscapeWindowDataset,
    TrainAudioDataset,
    build_val_dataset,
    collate,
)
from common.fold_split import build_soundscape_fold_table
from common.models import ModelCfg, build_model
from common.taxonomy import load_primary_labels
from common.training import Trainer, TrainerConfig
from stages._common import JsonlLogger, seed_everything

log = logging.getLogger("bridclef.s5")


def _build_label_index(comp_dir: Path) -> tuple[list[str], dict[str, int]]:
    labels = load_primary_labels(comp_dir)
    return labels, {c: i for i, c in enumerate(labels)}


def _pick_pretrained(pretrain_dir: Path, backbone: str) -> Optional[Path]:
    """Find a pretrained checkpoint in the Sydorskyy 2025 dataset matching ``backbone``.

    Strict: returns ``None`` (not a random fallback) if no filename contains
    the backbone token — otherwise we'd load an unrelated architecture's
    weights and wreck init. Prefers files under ``models_2025/`` over
    ``models_2024/`` since 2025 was trained with a similar task setup.
    """
    if not pretrain_dir.exists():
        return None
    cand: list[Path] = []
    for ext in ("*.ckpt", "*.pt", "*.pth"):
        cand.extend(pretrain_dir.rglob(ext))
    key = backbone.replace("_", "").lower()
    matches = [p for p in cand if key in p.stem.replace("_", "").lower()]
    if not matches:
        return None
    # Prefer 2025 models.
    matches.sort(key=lambda p: (0 if "2025" in str(p) else 1, len(p.stem)))
    return matches[0]


def _build_datasets(
    cfg: CloudCfg,
    primary_labels: list[str],
    label_to_idx: dict[str, int],
    mel_cfg: MelConfig,
    val_fold: int,
    n_folds: int,
) -> tuple[ConcatDataset, SoundscapeWindowDataset]:
    """Return (train_ds, val_ds) for a given val_fold."""
    # Hard-label soundscape rows (708)
    fold_tbl = build_soundscape_fold_table(cfg.comp_dir / "train_soundscapes_labels.csv",
                                           n_folds=n_folds)
    train_mask = fold_tbl["fold"] != val_fold
    val_mask = fold_tbl["fold"] == val_fold
    hard_train_df = fold_tbl[train_mask].reset_index(drop=True)
    hard_val_df = fold_tbl[val_mask].reset_index(drop=True)
    log.info("fold %d: 708-row split train=%d val=%d", val_fold, len(hard_train_df), len(hard_val_df))

    # train_audio weak-label df — always fully in train
    train_audio_csv = cfg.comp_dir / "train.csv"
    train_audio_df = pd.read_csv(train_audio_csv)
    log.info("train_audio rows: %d", len(train_audio_df))

    specaug = SpecAugParams(
        freq_masks=cfg.raw["s5"]["augment"]["spec_augment"]["freq_masks"],
        freq_mask_param=cfg.raw["s5"]["augment"]["spec_augment"]["freq_mask_param"],
        time_masks=cfg.raw["s5"]["augment"]["spec_augment"]["time_masks"],
        time_mask_param=cfg.raw["s5"]["augment"]["spec_augment"]["time_mask_param"],
    )

    audio_ds = TrainAudioDataset(
        df=train_audio_df,
        audio_root=cfg.comp_dir / "train_audio",
        label_to_idx=label_to_idx,
        mel_cfg=mel_cfg,
        specaug=specaug,
        time_shift_s=cfg.raw["s5"]["augment"]["time_shift_max_s"],
        noise_snr_db=20.0,
        secondary_soft=cfg.raw["s5"]["data_mix"]["train_audio"]["secondary_soft"],
        weight=cfg.raw["s5"]["data_mix"]["train_audio"]["weight"],
        train=True,
    )

    hard_ds = SoundscapeWindowDataset(
        labels_df=hard_train_df,
        mel_cache_dir=cfg.mel_cache / "train_soundscapes",
        label_to_idx=label_to_idx,
        mel_cfg=mel_cfg,
        specaug=specaug,
        weight=cfg.raw["s5"]["data_mix"]["train_soundscapes_labels"]["weight"],
        train=True,
    )

    # Up-sample hard-label set so it accounts for ~15% of the epoch regardless
    # of weight (sample weight is a loss multiplier, not a sampling probability).
    replicas = max(1, int(0.15 * len(audio_ds) / max(1, len(hard_ds))))
    if replicas > 1:
        from torch.utils.data import ConcatDataset as _CD
        hard_ds = _CD([hard_ds] * replicas)
        log.info("fold %d: up-sampled hard set x%d -> %d items",
                 val_fold, replicas, len(hard_ds))

    train_ds = ConcatDataset([audio_ds, hard_ds])
    val_ds = build_val_dataset(
        labels_df=hard_val_df,
        mel_cache_dir=cfg.mel_cache / "train_soundscapes",
        label_to_idx=label_to_idx,
        mel_cfg=mel_cfg,
    )
    return train_ds, val_ds


def _run_one_fold(
    cfg: CloudCfg,
    fold: int,
    primary_labels: list[str],
    label_to_idx: dict[str, int],
    mel_cfg: MelConfig,
    stage_name: str = "S5",
) -> dict:
    s5 = cfg.raw["s5"]
    backbone_cfg = cfg.raw["backbones"][0]  # Plan Y: single backbone

    seed_everything(int(cfg.raw["misc"]["seed"]) + fold)

    train_ds, val_ds = _build_datasets(
        cfg, primary_labels, label_to_idx, mel_cfg,
        val_fold=fold, n_folds=s5["n_folds"],
    )

    pretrained = _pick_pretrained(cfg.pretrain_dir, backbone_cfg["name"])
    if pretrained:
        log.info("fold %d: using pretrained %s", fold, pretrained)
    else:
        log.info("fold %d: no pretrained found, training from timm default", fold)
    model_cfg = ModelCfg(
        backbone=backbone_cfg["name"],
        n_classes=len(primary_labels),
        in_chans=1,
        drop_rate=0.2,
        head_drop=backbone_cfg.get("head_dropout", 0.2),
        pretrained_path=pretrained,
    )
    model = build_model(model_cfg, logger=log)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin = (device == "cuda")

    def _worker_init(worker_id: int) -> None:
        # Keep every worker to a single math thread — prevents OMP/MKL/torch
        # thread fan-out (AutoDL sets OMP_NUM_THREADS=25 globally, which with
        # persistent_workers + soundfile starves I/O and stalls the loader).
        import os
        import torch as _t
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        _t.set_num_threads(1)

    n_workers = int(cfg.raw.get("s5", {}).get("num_workers", 4))
    train_loader = DataLoader(
        train_ds, batch_size=backbone_cfg["batch_size"],
        shuffle=True, num_workers=n_workers,
        pin_memory=pin, drop_last=True, collate_fn=collate,
        persistent_workers=False,           # fresh workers each epoch -> avoids leaks
        prefetch_factor=2,
        worker_init_fn=_worker_init,
    )
    val_loader = DataLoader(
        val_ds, batch_size=backbone_cfg["batch_size"],
        shuffle=False, num_workers=2, pin_memory=pin,
        collate_fn=collate, persistent_workers=False,
        worker_init_fn=_worker_init,
    )

    trainer_cfg = TrainerConfig(
        stage=stage_name,
        fold=fold,
        epochs=int(backbone_cfg["epochs_ft"]),
        lr=float(backbone_cfg["lr"]),
        weight_decay=float(s5["optimizer"]["weight_decay"]),
        warmup_epochs=int(s5["scheduler"]["warmup_epochs"]),
        focal_gamma=float(s5["loss"]["gamma"]),
        label_smoothing=float(s5["loss"]["label_smoothing"]),
        mixup_alpha=float(s5["augment"]["mixup_alpha"]),
        mixup_prob=float(s5["augment"]["mixup_prob"]),
        cutmix_prob=float(s5["augment"]["cutmix_prob"]),
        ema_decay=float(s5["ema_decay"]),
        swa_last_epochs=int(s5["swa"]["last_epochs"]),
        swa_enable=bool(s5["swa"]["enable"]),
        amp=bool(s5["amp"]),
        save_every_steps=int(s5["save_every_steps"]),
        eval_every_epochs=int(s5["eval_every_epochs"]),
        seed=int(cfg.raw["misc"]["seed"]),
        hf_repo_id=str(cfg.raw["checkpoint"]["hf_backup"]["repo_id"]) or "",
        hf_backup_interval_min=float(cfg.raw["checkpoint"]["hf_backup"]["interval_min"]),
    )

    jsonl = JsonlLogger(cfg.stage_log(stage_name))
    ckpt_dir = cfg.stage_ckpt_dir(stage_name, fold=fold)

    trainer = Trainer(
        cfg=trainer_cfg, model=model,
        train_loader=train_loader, val_loader=val_loader,
        device=device, ckpt_dir=ckpt_dir, jsonl=jsonl,
    )
    out = trainer.fit()
    log.info("fold %d finished: best_val_auc=%.4f  swa_count=%d",
             fold, out["best_val_auc"], out["swa_count"])
    return out


# --------------------------------------------------------------------------
# Stage entry
# --------------------------------------------------------------------------

def run(cfg: CloudCfg, args: argparse.Namespace) -> int:
    s5 = cfg.raw["s5"]
    primary_labels, label_to_idx = _build_label_index(cfg.comp_dir)
    mel_cfg = MelConfig.from_yaml(cfg.raw)

    folds_to_run = [int(args.fold)] if getattr(args, "fold", None) is not None \
        else list(range(int(s5["n_folds"])))
    log.info("S5: running folds %s", folds_to_run)

    results: list[dict] = []
    for f in folds_to_run:
        out = _run_one_fold(cfg, f, primary_labels, label_to_idx, mel_cfg)
        results.append({"fold": f, **out})

    # Write a tiny summary so S6/S9 know which fold to load.
    import json
    summ = {"folds": results, "backbone": cfg.raw["backbones"][0]["name"]}
    (cfg.stage_ckpt_dir("S5").parent / "summary.json").write_text(
        json.dumps(summ, indent=2, default=str)
    )
    return 0
