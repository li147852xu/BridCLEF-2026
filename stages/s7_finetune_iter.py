"""S7 — second fine-tune pass with iter-1 pseudo labels mixed in.

Identical to S5 except:
  * Warm-start from S5 ``best.pt`` per fold.
  * Add PseudoWindowDataset to the training mix (S6 output).
  * Fewer epochs, smaller LR (via ``lr_multiplier``).
  * Writes into ``${ckpt_root}/S7/fold_{k}/``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from common.augment import SpecAugParams
from common.cloud_paths import CloudCfg
from common.datasets import (
    MelConfig,
    PseudoWindowDataset,
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
from stages.s5_finetune import _pick_pretrained

log = logging.getLogger("bridclef.s7")


def _build_datasets(
    cfg: CloudCfg,
    primary_labels: list[str],
    label_to_idx: dict[str, int],
    mel_cfg: MelConfig,
    val_fold: int,
    n_folds: int,
):
    s7 = cfg.raw["s7"]
    # Inherit S5's augment / mix config defaults.
    s5 = cfg.raw["s5"]

    fold_tbl = build_soundscape_fold_table(
        cfg.comp_dir / "train_soundscapes_labels.csv", n_folds=n_folds
    )
    hard_train_df = fold_tbl[fold_tbl["fold"] != val_fold].reset_index(drop=True)
    hard_val_df = fold_tbl[fold_tbl["fold"] == val_fold].reset_index(drop=True)

    specaug = SpecAugParams(
        freq_masks=s5["augment"]["spec_augment"]["freq_masks"],
        freq_mask_param=s5["augment"]["spec_augment"]["freq_mask_param"],
        time_masks=s5["augment"]["spec_augment"]["time_masks"],
        time_mask_param=s5["augment"]["spec_augment"]["time_mask_param"],
    )

    train_audio_df = pd.read_csv(cfg.comp_dir / "train.csv")
    audio_ds = TrainAudioDataset(
        df=train_audio_df,
        audio_root=cfg.comp_dir / "train_audio",
        label_to_idx=label_to_idx,
        mel_cfg=mel_cfg,
        specaug=specaug,
        time_shift_s=s5["augment"]["time_shift_max_s"],
        noise_snr_db=20.0,
        secondary_soft=s7["data_mix"]["train_audio"]["secondary_soft"],
        weight=s7["data_mix"]["train_audio"]["weight"],
        train=True,
    )

    hard_ds = SoundscapeWindowDataset(
        labels_df=hard_train_df,
        mel_cache_dir=cfg.mel_cache / "train_soundscapes",
        label_to_idx=label_to_idx,
        mel_cfg=mel_cfg,
        specaug=specaug,
        weight=s7["data_mix"]["train_soundscapes_labels"]["weight"],
        train=True,
    )

    pseudo_npz = cfg.work_root / "artifacts" / "pseudo" / "pseudo_iter1.npz"
    if not pseudo_npz.exists():
        raise FileNotFoundError(
            f"S7 requires S6 output: {pseudo_npz}. Run S6 first."
        )
    pseudo_ds = PseudoWindowDataset(
        pseudo_npz=pseudo_npz,
        mel_cache_dir=cfg.mel_cache / "train_soundscapes",
        mel_cfg=mel_cfg,
        specaug=specaug,
        weight=s7["data_mix"]["pseudo_iter1"]["weight"],
        train=True,
    )
    log.info("S7 fold %d sizes: audio=%d hard=%d pseudo=%d",
             val_fold, len(audio_ds), len(hard_ds), len(pseudo_ds))

    # Up-sample hard to stay meaningful against pseudo mass.
    replicas = max(1, int(0.15 * (len(audio_ds) + len(pseudo_ds)) / max(1, len(hard_ds))))
    if replicas > 1:
        hard_ds = ConcatDataset([hard_ds] * replicas)

    train_ds = ConcatDataset([audio_ds, hard_ds, pseudo_ds])
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
) -> dict:
    s5 = cfg.raw["s5"]; s7 = cfg.raw["s7"]
    backbone_cfg = cfg.raw["backbones"][0]
    seed_everything(int(cfg.raw["misc"]["seed"]) + fold + 101)

    train_ds, val_ds = _build_datasets(
        cfg, primary_labels, label_to_idx, mel_cfg,
        val_fold=fold, n_folds=s5["n_folds"],
    )

    # Warm start: prefer S5 best.pt, fall back to Sydorskyy pretrained.
    warm_from = cfg.stage_ckpt_dir("S5", fold=fold) / "best.pt"
    if not warm_from.exists():
        warm_from = cfg.stage_ckpt_dir("S5", fold=fold) / "last.pt"
    if not warm_from.exists():
        warm_from = _pick_pretrained(cfg.pretrain_dir, backbone_cfg["name"])
        log.warning("S7 fold %d: S5 ckpt missing, falling back to %s", fold, warm_from)

    model_cfg = ModelCfg(
        backbone=backbone_cfg["name"],
        n_classes=len(primary_labels),
        in_chans=1,
        drop_rate=0.2,
        head_drop=backbone_cfg.get("head_dropout", 0.2),
        pretrained_path=warm_from,
    )
    model = build_model(model_cfg, logger=log)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin = (device == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=backbone_cfg["batch_size"],
        shuffle=True, num_workers=max(2, int(cfg.raw["s2"]["num_workers"])),
        pin_memory=pin, drop_last=True, collate_fn=collate,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=backbone_cfg["batch_size"],
        shuffle=False, num_workers=2, pin_memory=pin,
        collate_fn=collate,
    )

    trainer_cfg = TrainerConfig(
        stage="S7",
        fold=fold,
        epochs=int(backbone_cfg.get("epochs_ft_iter", 12)),
        lr=float(backbone_cfg["lr"]),
        weight_decay=float(s5["optimizer"]["weight_decay"]),
        warmup_epochs=max(1, int(s5["scheduler"]["warmup_epochs"]) - 1),
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
        lr_multiplier=float(s7.get("lr_multiplier", 0.5)),
        hf_repo_id=str(cfg.raw["checkpoint"]["hf_backup"]["repo_id"]) or "",
        hf_backup_interval_min=float(cfg.raw["checkpoint"]["hf_backup"]["interval_min"]),
    )

    jsonl = JsonlLogger(cfg.stage_log("S7"))
    ckpt_dir = cfg.stage_ckpt_dir("S7", fold=fold)
    trainer = Trainer(
        cfg=trainer_cfg, model=model,
        train_loader=train_loader, val_loader=val_loader,
        device=device, ckpt_dir=ckpt_dir, jsonl=jsonl,
    )
    out = trainer.fit()
    return out


def run(cfg: CloudCfg, args: argparse.Namespace) -> int:
    primary_labels = load_primary_labels(cfg.comp_dir)
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}
    mel_cfg = MelConfig.from_yaml(cfg.raw)
    n_folds = int(cfg.raw["s5"]["n_folds"])
    folds_to_run = [int(args.fold)] if getattr(args, "fold", None) is not None \
        else list(range(n_folds))
    log.info("S7: running folds %s", folds_to_run)
    for f in folds_to_run:
        _run_one_fold(cfg, f, primary_labels, label_to_idx, mel_cfg)
    return 0
