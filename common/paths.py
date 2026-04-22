"""Path resolution.

Unifies access from both the local training environment and the Kaggle
inference runtime. Reads ``configs/base.yaml`` relative to the repo root
and auto-switches to ``/kaggle/input/*`` when run inside a Kaggle notebook.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Return the project root.

    Strategy:
      1. If ``$BIRDCLEF_ROOT`` is set, use it.
      2. Else walk up from this file's parent until ``configs/base.yaml`` is found.
      3. Fallback to the CWD.
    """
    env = os.environ.get("BIRDCLEF_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "configs" / "base.yaml").exists():
            return candidate

    return Path.cwd().resolve()


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    cfg_path = repo_root() / "configs" / "base.yaml"
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def is_kaggle_runtime() -> bool:
    """Detect whether we're running inside a Kaggle kernel."""
    return Path("/kaggle/input").exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def _resolve(base: Path, rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else base / p


def comp_dir() -> Path:
    """Competition data directory (contains taxonomy.csv, train_soundscapes/, etc.)."""
    cfg = load_config()
    if is_kaggle_runtime():
        return Path(cfg["paths"]["kaggle"]["comp_dir"])
    return _resolve(repo_root(), cfg["paths"]["comp_dir"])


def perch_model_dir() -> Path:
    cfg = load_config()
    if is_kaggle_runtime():
        return Path(cfg["paths"]["kaggle"]["perch_model"])
    return _resolve(repo_root(), cfg["paths"]["perch_model"])


def artifacts_dir() -> Path:
    cfg = load_config()
    p = _resolve(repo_root(), cfg["paths"]["artifacts_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def perch_cache_dir() -> Path:
    cfg = load_config()
    p = _resolve(repo_root(), cfg["paths"]["perch_cache_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def kaggle_bundle_path() -> Path:
    cfg = load_config()
    return Path(cfg["paths"]["kaggle"]["bundle"])


def ensure_repo_on_syspath() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
