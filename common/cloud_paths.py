"""Cloud-aware path / config resolver.

The existing ``common/paths.py`` is keyed to ``configs/base.yaml`` (the local
Mac/Kaggle pipeline). For the Plan Y cloud pipeline we use a different config
file (``configs/cloud_Y.yaml``) with absolute paths on the AutoDL box.

Design goals:
    * Single function ``load_cloud_config(cfg_path)`` returns a ``CloudCfg``
      dataclass with every path resolved to an absolute ``Path``.
    * ``${paths.x}`` interpolation in the YAML is expanded.
    * Env var ``BIRDCLEF_WORK_ROOT`` overrides ``paths.work_root``.
    * Works on both the cloud box (absolute paths) and locally for dry-run
      (caller can pass ``--config`` to a YAML pointing at a laptop scratch dir).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_INTERP = re.compile(r"\$\{([^}]+)\}")


def _dotted_get(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"config: missing key {dotted!r}")
        cur = cur[part]
    return cur


def _expand_interpolations(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``${a.b.c}`` in every string value, recursively.

    Simple two-pass: first pass expands leaf strings that have all their refs
    available (e.g. ``paths.work_root``); second pass picks up chains that only
    resolved after pass 1. Two passes are enough for our depth-2 config.
    """
    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        if isinstance(obj, str):
            def sub(match: re.Match[str]) -> str:
                key = match.group(1)
                try:
                    val = _dotted_get(cfg, key)
                except KeyError:
                    return match.group(0)
                return str(val)
            return _INTERP.sub(sub, obj)
        return obj

    for _ in range(4):
        new = walk(cfg)
        if new == cfg:
            return cfg
        cfg = new
    return cfg


@dataclass
class CloudCfg:
    """Resolved config bundle. Only path objects are normalized here; the rest
    of the YAML stays as a raw dict for each stage to parse itself."""

    raw: dict[str, Any]
    cfg_path: Path

    # --- resolved paths -----------------------------------------------------
    work_root: Path
    repo_root: Path
    comp_dir: Path
    perch_model: Path
    pretrain_dir: Path
    mel_cache: Path
    perch_cache: Path
    ckpt_root: Path
    export_dir: Path
    logs_dir: Path
    flags_dir: Path

    # -----------------------------------------------------------------------
    def stage_log(self, stage: str) -> Path:
        p = self.logs_dir / f"{stage}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def stage_flag(self, stage: str) -> Path:
        return self.flags_dir / f"{stage}.done"

    def stage_ckpt_dir(self, stage: str, fold: int | None = None) -> Path:
        sub = f"fold_{fold}" if fold is not None else "_single"
        p = self.ckpt_root / stage / sub
        p.mkdir(parents=True, exist_ok=True)
        return p

    def mkdirs(self) -> None:
        for p in (self.mel_cache, self.perch_cache, self.ckpt_root,
                  self.export_dir, self.logs_dir, self.flags_dir):
            p.mkdir(parents=True, exist_ok=True)


def load_cloud_config(cfg_path: str | Path) -> CloudCfg:
    cfg_path = Path(cfg_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"cloud config not found: {cfg_path}")
    with cfg_path.open("r") as f:
        raw = yaml.safe_load(f)

    env_work_root = os.environ.get("BIRDCLEF_WORK_ROOT")
    if env_work_root:
        raw.setdefault("paths", {})["work_root"] = env_work_root

    raw = _expand_interpolations(raw)
    p = raw["paths"]

    def P(key: str) -> Path:
        v = p[key]
        return Path(v).expanduser().resolve()

    return CloudCfg(
        raw=raw,
        cfg_path=cfg_path,
        work_root=P("work_root"),
        repo_root=P("repo_root"),
        comp_dir=P("comp_dir"),
        perch_model=P("perch_model"),
        pretrain_dir=P("pretrain_dir"),
        mel_cache=P("mel_cache"),
        perch_cache=P("perch_cache"),
        ckpt_root=P("ckpt_root"),
        export_dir=P("export_dir"),
        logs_dir=P("logs_dir"),
        flags_dir=P("flags_dir"),
    )
