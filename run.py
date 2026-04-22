#!/usr/bin/env python3
"""BirdCLEF+ 2026 cloud pipeline driver.

Usage:
    python run.py status   --config configs/cloud_Y.yaml
    python run.py stage    --config configs/cloud_Y.yaml S2
    python run.py stage    --config configs/cloud_Y.yaml S5 --fold 0
    python run.py pipeline --config configs/cloud_Y.yaml --from S2 --to S9
    python run.py pipeline --config configs/cloud_Y.yaml          # default: all
    python run.py stage    --config configs/cloud_Y.yaml S5 --fresh   # re-run

Design:
    * Each stage has a flag file ``${flags_dir}/<stage>.done``.
    * ``pipeline`` skips stages whose flag is present unless ``--fresh``.
    * ``stage`` always re-enters (but the stage itself should be resumable).
    * Ctrl-C / SIGTERM triggers a clean exit so checkpoints get flushed.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

# Make ``common`` + ``stages`` importable regardless of cwd.
_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.cloud_paths import CloudCfg, load_cloud_config  # noqa: E402
from stages import STAGE_ORDER, load_stage                   # noqa: E402
from stages._common import (                                 # noqa: E402
    graceful_sigint,
    is_done,
    mark_done,
    setup_logging,
)


def _mk_parser() -> argparse.ArgumentParser:
    # --config is shared by every subcommand, attached via parents=.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True, help="Path to cloud YAML config")

    p = argparse.ArgumentParser(description="BirdCLEF+ 2026 cloud driver")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", parents=[common],
                   help="Show per-stage done/pending state")

    pst = sub.add_parser("stage", parents=[common], help="Run a single stage")
    pst.add_argument("stage", choices=STAGE_ORDER)
    pst.add_argument("--fold", type=int, default=None,
                     help="Fold index (for S5/S7). If omitted, run all folds.")
    pst.add_argument("--fresh", action="store_true",
                     help="Ignore any previous checkpoint and restart")
    pst.add_argument("--resume", action="store_true",
                     help="Explicitly resume from last ckpt (default when ckpt exists)")
    pst.add_argument("--extra", nargs="*", default=[],
                     help="Extra key=value overrides (stage-specific)")

    pp = sub.add_parser("pipeline", parents=[common],
                        help="Run multiple stages in order")
    pp.add_argument("--from", dest="from_stage", choices=STAGE_ORDER, default=STAGE_ORDER[0])
    pp.add_argument("--to",   dest="to_stage",   choices=STAGE_ORDER, default=STAGE_ORDER[-1])
    pp.add_argument("--fresh", action="store_true",
                    help="Ignore all existing done flags; re-run everything")

    return p


def _stages_in_window(frm: str, to: str) -> list[str]:
    i, j = STAGE_ORDER.index(frm), STAGE_ORDER.index(to)
    if i > j:
        raise SystemExit(f"--from {frm} is after --to {to}")
    return STAGE_ORDER[i:j + 1]


def _cmd_status(cfg: CloudCfg) -> int:
    print(f"config:    {cfg.cfg_path}")
    print(f"work_root: {cfg.work_root}")
    print(f"comp_dir:  {cfg.comp_dir}")
    print("stages:")
    for s in STAGE_ORDER:
        flag = cfg.stage_flag(s)
        state = "DONE" if is_done(flag) else "pending"
        print(f"  {s}  [{state}]  {flag}")
    return 0


def _run_one_stage(cfg: CloudCfg, stage: str, args: argparse.Namespace,
                   logger) -> int:
    logger.info("=== stage %s start ===", stage)
    t0 = time.time()
    try:
        runner = load_stage(stage)
        rc = runner(cfg, args)
    except KeyboardInterrupt:
        logger.warning("stage %s interrupted; checkpoint should be on disk", stage)
        return 130
    except Exception as e:  # noqa: BLE001
        logger.error("stage %s FAILED: %s", stage, e)
        logger.error(traceback.format_exc())
        return 1

    dt = time.time() - t0
    if rc == 0:
        mark_done(cfg.stage_flag(stage),
                  payload={"stage": stage, "elapsed_s": dt})
        logger.info("=== stage %s DONE in %.1f min ===", stage, dt / 60)
    else:
        logger.error("=== stage %s returned rc=%s ===", stage, rc)
    return rc


def _cmd_stage(cfg: CloudCfg, args: argparse.Namespace, logger) -> int:
    # 'stage' always runs; --fresh just clears the flag so mark_done is honest.
    if args.fresh and is_done(cfg.stage_flag(args.stage)):
        cfg.stage_flag(args.stage).unlink()
    return _run_one_stage(cfg, args.stage, args, logger)


def _cmd_pipeline(cfg: CloudCfg, args: argparse.Namespace, logger) -> int:
    todo = _stages_in_window(args.from_stage, args.to_stage)
    logger.info("pipeline: %s", " -> ".join(todo))
    for s in todo:
        flag = cfg.stage_flag(s)
        if is_done(flag) and not args.fresh:
            logger.info("stage %s: already done, skipping", s)
            continue
        # For per-fold stages, 'pipeline' runs all folds in one go.
        sub_args = argparse.Namespace(fold=None, fresh=args.fresh,
                                      resume=True, extra=[])
        rc = _run_one_stage(cfg, s, sub_args, logger)
        if rc != 0:
            logger.error("pipeline aborted at %s (rc=%s)", s, rc)
            return rc
    logger.info("pipeline complete.")
    return 0


def main() -> int:
    parser = _mk_parser()
    args = parser.parse_args()
    cfg = load_cloud_config(args.config)
    cfg.mkdirs()

    log_level = cfg.raw.get("log", {}).get("level", "INFO")
    logger = setup_logging(log_level)
    logger.info("config: %s  plan=%s  work_root=%s",
                cfg.cfg_path, cfg.raw.get("plan"), cfg.work_root)

    with graceful_sigint():
        if args.cmd == "status":
            return _cmd_status(cfg)
        if args.cmd == "stage":
            return _cmd_stage(cfg, args, logger)
        if args.cmd == "pipeline":
            return _cmd_pipeline(cfg, args, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
