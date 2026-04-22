"""Shared training loop + full resume for S5 and S7.

Checkpoint payload (single ``.pt`` file, torch.save with pickle):

    {
      "model":         state_dict,
      "ema":           EMA.state_dict (optional),
      "optimizer":     optim.state_dict,
      "scheduler":     sched.state_dict,
      "scaler":        amp.GradScaler.state_dict (optional),
      "epoch":         int,                     # epoch about to start
      "global_step":   int,
      "rng":           {"torch":, "numpy":, "python":},
      "best_val_auc":  float,
      "cfg":           dict (sanity check on resume),
      "swa_state":     optional dict (swa buffers),
    }

Atomic write: save to ``last.pt.tmp`` then rename to ``last.pt``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.augment import mixup_batch, cutmix_batch
from common.models import EMA
from stages._common import (
    CheckpointPaths,
    JsonlLogger,
    rotate_epoch_ckpts,
)

log = logging.getLogger("bridclef.train")


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

def focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor,
                          gamma: float = 2.0, label_smoothing: float = 0.0,
                          weight: Optional[torch.Tensor] = None,
                          reduction: str = "mean") -> torch.Tensor:
    """Multi-label focal BCE. ``logits`` and ``target`` are (B, C)."""
    if label_smoothing > 0.0:
        target = target.clamp(min=label_smoothing,
                              max=1.0 - label_smoothing)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits).detach()
    # focal term: (1-p)^gamma for y=1, p^gamma for y=0  →  combined via target
    focal = target * (1 - p).pow(gamma) + (1 - target) * p.pow(gamma)
    loss = focal * bce
    if weight is not None:
        loss = loss * weight[:, None]
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# --------------------------------------------------------------------------
# Trainer config
# --------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    stage: str
    fold: int

    epochs: int
    lr: float
    weight_decay: float
    warmup_epochs: int

    focal_gamma: float = 2.0
    label_smoothing: float = 0.02

    mixup_alpha: float = 0.4
    mixup_prob: float = 0.5
    cutmix_prob: float = 0.3

    ema_decay: float = 0.9995
    swa_last_epochs: int = 5
    swa_enable: bool = True

    amp: bool = True
    clip_grad_norm: float = 5.0

    save_every_steps: int = 500
    eval_every_epochs: int = 1
    keep_last_n_epochs: int = 3

    seed: int = 0
    lr_multiplier: float = 1.0

    # HF backup
    hf_repo_id: str = ""
    hf_backup_interval_min: float = 30.0


# --------------------------------------------------------------------------
# RNG snapshot / restore
# --------------------------------------------------------------------------

def snapshot_rng() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


# --------------------------------------------------------------------------
# Macro-AUC (multi-label, skip classes without positives)
# --------------------------------------------------------------------------

def macro_auc(probs: np.ndarray, targets: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(targets.shape[1]):
        t = targets[:, c]
        if t.sum() <= 0 or t.sum() >= len(t):
            continue
        try:
            aucs.append(roc_auc_score(t, probs[:, c]))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------

class Trainer:
    """Drives one fold's training loop with full resume."""

    def __init__(
        self,
        cfg: TrainerConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        ckpt_dir: Path,
        jsonl: JsonlLogger,
    ):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cp = CheckpointPaths.at(ckpt_dir)
        self.jsonl = jsonl

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr * cfg.lr_multiplier,
            weight_decay=cfg.weight_decay,
        )
        steps_per_epoch = max(1, len(train_loader))
        total_steps = steps_per_epoch * cfg.epochs
        warmup_steps = steps_per_epoch * cfg.warmup_epochs

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")
        self.ema = EMA(self.model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None

        self.epoch = 0
        self.global_step = 0
        self.best_val_auc = -1.0
        self.rng_seed = cfg.seed + cfg.fold * 1000

        # SWA bookkeeping (accumulated at end of each last-k epoch)
        self.swa_count = 0
        self.swa_state: dict[str, torch.Tensor] = {}

        # HF backup (optional)
        from stages._common import HFBackupDaemon
        self.hf = HFBackupDaemon(
            repo_id=cfg.hf_repo_id,
            local_dir=ckpt_dir,
            remote_subdir=f"{cfg.stage}/fold_{cfg.fold}",
            interval_min=cfg.hf_backup_interval_min,
            logger=log,
        )

    # ---- checkpoint IO -------------------------------------------------

    def _payload(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "rng": snapshot_rng(),
            "best_val_auc": self.best_val_auc,
            "swa_state": self.swa_state,
            "swa_count": self.swa_count,
            "cfg": {
                "stage": self.cfg.stage, "fold": self.cfg.fold,
                "epochs": self.cfg.epochs, "lr": self.cfg.lr,
                "seed": self.cfg.seed,
            },
        }

    def save_last(self) -> None:
        tmp = self.cp.last.with_suffix(".pt.tmp")
        torch.save(self._payload(), tmp)
        tmp.replace(self.cp.last)

    def save_epoch(self, e: int) -> None:
        torch.save(self._payload(), self.cp.epoch(e))
        rotate_epoch_ckpts(self.cp.dir, self.cfg.keep_last_n_epochs)

    def save_best(self) -> None:
        torch.save(self._payload(), self.cp.best)

    def try_resume(self) -> bool:
        if not self.cp.last.exists():
            return False
        st = torch.load(str(self.cp.last), map_location=self.device, weights_only=False)
        self.model.load_state_dict(st["model"])
        if self.ema is not None and st.get("ema") is not None:
            self.ema.load_state_dict(st["ema"])
        self.optimizer.load_state_dict(st["optimizer"])
        self.scheduler.load_state_dict(st["scheduler"])
        self.scaler.load_state_dict(st["scaler"])
        self.epoch = int(st["epoch"])
        self.global_step = int(st["global_step"])
        self.best_val_auc = float(st.get("best_val_auc", -1.0))
        self.swa_state = st.get("swa_state") or {}
        self.swa_count = int(st.get("swa_count", 0))
        try:
            restore_rng(st["rng"])
        except Exception as e:  # noqa: BLE001
            log.warning("could not restore RNG state: %s", e)
        log.info("resumed fold %d from epoch %d (step %d, best_val %.4f)",
                 self.cfg.fold, self.epoch, self.global_step, self.best_val_auc)
        return True

    # ---- SWA -----------------------------------------------------------

    @torch.no_grad()
    def swa_update(self) -> None:
        sd = self.model.state_dict()
        if self.swa_count == 0:
            self.swa_state = {k: v.detach().clone().float() for k, v in sd.items()
                              if v.dtype.is_floating_point}
        else:
            n = self.swa_count + 1
            for k, avg in self.swa_state.items():
                avg.mul_(self.swa_count / n).add_(sd[k].detach().float() / n)
        self.swa_count += 1

    def swa_load_into_model(self) -> None:
        if not self.swa_state:
            return
        sd = self.model.state_dict()
        for k, v in self.swa_state.items():
            sd[k].copy_(v.to(sd[k].dtype))

    # ---- train / eval --------------------------------------------------

    def train_one_epoch(self) -> float:
        self.model.train()
        rng = np.random.default_rng(self.rng_seed + self.epoch)
        losses = []
        t0 = time.time()
        for step_in_epoch, (x, y, w) in enumerate(self.train_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            w = w.to(self.device, non_blocking=True)

            # Batch augments
            if self.cfg.mixup_alpha > 0 and self.cfg.mixup_prob > 0:
                x, y = mixup_batch(x, y, self.cfg.mixup_alpha, self.cfg.mixup_prob, rng)
            if self.cfg.cutmix_prob > 0:
                x, y = cutmix_batch(x, y, self.cfg.cutmix_prob, rng)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.cfg.amp and self.device == "cuda"):
                logits = self.model(x)
                loss = focal_bce_with_logits(
                    logits, y, gamma=self.cfg.focal_gamma,
                    label_smoothing=self.cfg.label_smoothing,
                    weight=w,
                )

            self.scaler.scale(loss).backward()
            if self.cfg.clip_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)

            self.global_step += 1
            losses.append(float(loss.detach().cpu()))
            if self.global_step % self.cfg.save_every_steps == 0:
                self.save_last()
            if step_in_epoch % 50 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.jsonl.log(stage=self.cfg.stage, fold=self.cfg.fold,
                               epoch=self.epoch, step=self.global_step,
                               loss=float(np.mean(losses[-50:])), lr=lr)
        dt = time.time() - t0
        log.info("fold %d epoch %d: train_loss=%.4f  (%.1f min)",
                 self.cfg.fold, self.epoch, float(np.mean(losses)), dt / 60)
        return float(np.mean(losses))

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        # Use EMA weights for validation if enabled.
        backup = None
        if self.ema is not None:
            backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.ema.copy_to(self.model)

        all_p, all_y = [], []
        for x, y, _ in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.cfg.amp and self.device == "cuda"):
                logits = self.model(x)
            all_p.append(torch.sigmoid(logits).float().cpu().numpy())
            all_y.append(y.numpy())

        if backup is not None:
            self.model.load_state_dict(backup)

        probs = np.concatenate(all_p, axis=0)
        targets = np.concatenate(all_y, axis=0)
        auc = macro_auc(probs, targets)
        log.info("fold %d epoch %d: val_macro_auc=%.4f", self.cfg.fold, self.epoch, auc)
        return auc

    # ---- public entry --------------------------------------------------

    def fit(self) -> dict:
        self.try_resume()
        self.hf.start()
        try:
            while self.epoch < self.cfg.epochs:
                self.train_one_epoch()
                if (self.epoch + 1) % self.cfg.eval_every_epochs == 0:
                    val = self.validate()
                    if val > self.best_val_auc:
                        self.best_val_auc = val
                        self.save_best()
                        log.info("fold %d: new best val=%.4f, saved best.pt",
                                 self.cfg.fold, val)
                    self.jsonl.log(stage=self.cfg.stage, fold=self.cfg.fold,
                                   epoch=self.epoch, val_auc=val,
                                   best_val_auc=self.best_val_auc)
                if self.cfg.swa_enable and (self.epoch + 1) > (self.cfg.epochs - self.cfg.swa_last_epochs):
                    self.swa_update()
                self.epoch += 1
                self.save_last()
                self.save_epoch(self.epoch)
        finally:
            # Flush SWA into last.pt one last time so S9 can pick it up.
            if self.cfg.swa_enable and self.swa_count > 0:
                self.swa_load_into_model()
                torch.save(self._payload(), self.cp.dir / "swa.pt")
                log.info("fold %d: SWA (%d epochs averaged) saved to swa.pt",
                         self.cfg.fold, self.swa_count)
            self.hf.stop(flush=True)
        return {"best_val_auc": self.best_val_auc, "swa_count": self.swa_count}
