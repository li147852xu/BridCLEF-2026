"""Shared stage helpers: done flags, atomic checkpoint IO, JSONL logging, HF backup.

Design rule: each helper is small, easy to audit, and does exactly one thing.
Callers import only what they need; nothing here imports torch so importing
this module is cheap even in CPU-only stages (S2, S3, S9).
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# --------------------------------------------------------------------------
# Done flags
# --------------------------------------------------------------------------

def is_done(flag_path: Path) -> bool:
    return flag_path.exists()


def mark_done(flag_path: Path, payload: dict[str, Any] | None = None) -> None:
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = flag_path.with_suffix(flag_path.suffix + ".tmp")
    body = {"ts": time.time(), **(payload or {})}
    tmp.write_text(json.dumps(body, indent=2, default=str))
    tmp.replace(flag_path)


def clear_done(flag_path: Path) -> None:
    if flag_path.exists():
        flag_path.unlink()


# --------------------------------------------------------------------------
# JSONL + console logging
# --------------------------------------------------------------------------

class JsonlLogger:
    """Append one JSON object per line to a metrics file.

    Thread-safe (protected by a mutex) so the HF backup daemon can read mid-flight.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, **kwargs: Any) -> None:
        row = {"ts": time.time(), **kwargs}
        line = json.dumps(row, default=str) + "\n"
        with self._lock:
            with self.path.open("a") as f:
                f.write(line)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname).1s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("bridclef")


# --------------------------------------------------------------------------
# Atomic file write
# --------------------------------------------------------------------------

def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


# --------------------------------------------------------------------------
# Seed
# --------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Checkpoint IO (torch-agnostic skeleton; stage scripts extend)
# --------------------------------------------------------------------------

@dataclass
class CheckpointPaths:
    dir: Path
    last: Path
    best: Path

    @classmethod
    def at(cls, ckpt_dir: Path) -> "CheckpointPaths":
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return cls(ckpt_dir, ckpt_dir / "last.pt", ckpt_dir / "best.pt")

    def epoch(self, idx: int) -> Path:
        return self.dir / f"epoch_{idx:03d}.pt"


def rotate_epoch_ckpts(ckpt_dir: Path, keep_last_n: int) -> None:
    """Delete old epoch_*.pt except the newest N. 'last.pt' and 'best.pt' are
    never touched (they live outside this pattern)."""
    files = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    for old in files[:-keep_last_n] if len(files) > keep_last_n else []:
        try:
            old.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------
# HF backup daemon (optional; no-op if repo_id missing or hf not installed)
# --------------------------------------------------------------------------

class HFBackupDaemon:
    """Spawn a background thread that periodically uploads a folder to HF Hub.

    The upload is delegated to a ``huggingface-cli upload`` subprocess so the
    training process never touches the hub lib directly (avoids dep weirdness).
    Failures are logged but never raised — training is always more important.
    """

    def __init__(self, repo_id: str, local_dir: Path, remote_subdir: str,
                 interval_min: float = 30.0, logger: logging.Logger | None = None):
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.remote_subdir = remote_subdir.strip("/")
        self.interval_s = max(60.0, interval_min * 60.0)
        self.log = logger or logging.getLogger("bridclef.hf_backup")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.repo_id or not os.environ.get("HF_TOKEN"):
            self.log.info("HF backup disabled (no repo_id or HF_TOKEN).")
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="hf-backup")
        self._thread.start()
        self.log.info("HF backup daemon started -> %s/%s every %.1f min",
                      self.repo_id, self.remote_subdir, self.interval_s / 60)

    def stop(self, flush: bool = True) -> None:
        self._stop.set()
        if flush:
            self._push_once()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._push_once()

    def _push_once(self) -> None:
        if not self.local_dir.exists():
            return
        # Newer huggingface_hub ships `hf` (>= 0.26); older installs have
        # `huggingface-cli`. Try the new one first so we are not surprised
        # by deprecation warnings that exit non-zero.
        import shutil
        for exe in ("hf", "huggingface-cli"):
            if shutil.which(exe) is None:
                continue
            try:
                if exe == "hf":
                    cmd = [
                        exe, "upload", self.repo_id,
                        str(self.local_dir), self.remote_subdir,
                        "--repo-type", "model",
                        "--commit-message", f"auto-backup {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    ]
                else:
                    cmd = [
                        exe, "upload", self.repo_id,
                        str(self.local_dir), self.remote_subdir,
                        "--repo-type", "model",
                        "--commit-message", f"auto-backup {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    ]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                if r.returncode == 0:
                    return
                self.log.warning("HF backup (%s) failed rc=%d: %s",
                                 exe, r.returncode, r.stderr.strip()[:400])
            except Exception as e:  # noqa: BLE001
                self.log.warning("HF backup (%s) exception: %s", exe, e)
        self.log.warning("HF backup: no working CLI found on PATH.")


# --------------------------------------------------------------------------
# Graceful ctrl-C: run finally blocks before exit
# --------------------------------------------------------------------------

@contextmanager
def graceful_sigint() -> Iterator[None]:
    """Convert the first SIGINT to a KeyboardInterrupt at a safe point.

    Standard behavior already does this, but we also install SIGTERM handler so
    AutoDL's ``docker stop`` sends a clean shutdown to the trainer.
    """
    original = signal.getsignal(signal.SIGTERM)
    def handler(signum, frame):  # noqa: ANN001
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, original)


# --------------------------------------------------------------------------
# Disk sanity
# --------------------------------------------------------------------------

def free_disk_gb(path: Path) -> float:
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)


def require_free_disk(path: Path, need_gb: float) -> None:
    got = free_disk_gb(path)
    if got < need_gb:
        raise RuntimeError(
            f"insufficient disk on {path}: need {need_gb:.1f} GB, got {got:.1f} GB"
        )
