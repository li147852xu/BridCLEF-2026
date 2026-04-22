"""OOF fusion-weight sweep on the teacher's per-source logits.

Reads ``teacher_cache.pkl`` (produced by ``02_fit_teacher.py``) and sweeps a
grid around the seed ``0.55 / 0.15 / 0.10 / 0.20`` fusion weights to maximize
macro ROC-AUC on the 708 labeled rows' OOF predictions. The winning weights
are written back into ``teacher_artifact.pkl`` without retraining any
component — only the linear combination changes.

Usage::

    python scripts/02b_fusion_sweep.py --grid coarse   # default
    python scripts/02b_fusion_sweep.py --grid fine     # ±0.05 step, slower
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.fusion import sigmoid  # noqa: E402
from common.io_utils import load_pickle, save_pickle, write_json  # noqa: E402
from common.metrics import macro_roc_auc_skip_empty  # noqa: E402
from common.paths import artifacts_dir  # noqa: E402


SEED = (0.55, 0.15, 0.10, 0.20)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", choices=["coarse", "medium", "fine"], default="medium")
    p.add_argument("--dry-run", action="store_true", help="don't update teacher_artifact.pkl")
    return p.parse_args()


def _grid_for(mode: str) -> tuple[list[float], list[float], list[float], list[float]]:
    # Each list is the values for alpha_perch, alpha_prior, alpha_probe, alpha_temp.
    if mode == "coarse":
        ap = [0.40, 0.50, 0.55, 0.60, 0.70]
        aP = [0.10, 0.15, 0.20]
        ab = [0.05, 0.10, 0.15]
        at = [0.10, 0.15, 0.20, 0.25]
    elif mode == "fine":
        ap = [round(v, 2) for v in np.arange(0.40, 0.76, 0.05)]
        aP = [round(v, 2) for v in np.arange(0.05, 0.26, 0.05)]
        ab = [round(v, 2) for v in np.arange(0.00, 0.21, 0.05)]
        at = [round(v, 2) for v in np.arange(0.05, 0.31, 0.05)]
    else:  # medium
        ap = [0.45, 0.50, 0.55, 0.60, 0.65]
        aP = [0.10, 0.15, 0.20]
        ab = [0.05, 0.10, 0.15]
        at = [0.10, 0.15, 0.20, 0.25, 0.30]
    return ap, aP, ab, at


def main() -> int:
    args = parse_args()
    art_dir = artifacts_dir()
    teacher_path = art_dir / "teacher_artifact.pkl"
    cache_path = art_dir / "teacher_cache.pkl"

    teacher = load_pickle(teacher_path)
    cache = load_pickle(cache_path)

    Y = cache["Y_full_truth"].astype(np.uint8)
    perch = cache["oof_perch_logits"].astype(np.float32)
    prior = cache["oof_prior_logits"].astype(np.float32)
    probe = cache["oof_probe_logits"].astype(np.float32)
    temp = cache["oof_temp_logits"].astype(np.float32)
    assert perch.shape == prior.shape == probe.shape == temp.shape == Y.shape

    ap, aP, ab, at = _grid_for(args.grid)
    print(f"Grid '{args.grid}': perch={ap}  prior={aP}  probe={ab}  temp={at}"
          f"  total combos={len(ap) * len(aP) * len(ab) * len(at)}")

    best = {"auc": -np.inf}
    seed_auc = None
    t0 = time.time()
    for a_perch, a_prior, a_probe, a_temp in itertools.product(ap, aP, ab, at):
        fused = (
            a_perch * perch + a_prior * prior
            + a_probe * probe + a_temp * temp
        )
        probs = sigmoid(fused)
        auc, _ = macro_roc_auc_skip_empty(Y, probs)
        if (a_perch, a_prior, a_probe, a_temp) == SEED:
            seed_auc = float(auc)
        if auc > best["auc"]:
            best = {
                "auc": float(auc),
                "alpha_perch": float(a_perch),
                "alpha_prior": float(a_prior),
                "alpha_probe": float(a_probe),
                "alpha_temp":  float(a_temp),
            }
    print(f"Sweep done in {time.time() - t0:.1f}s")
    print(f"Seed   AUC = {seed_auc}")
    print(f"Best   AUC = {best['auc']:.4f}  "
          f"weights=({best['alpha_perch']:.2f}/{best['alpha_prior']:.2f}/"
          f"{best['alpha_probe']:.2f}/{best['alpha_temp']:.2f})")

    if args.dry_run:
        print("(dry-run) not updating teacher_artifact.pkl")
    else:
        teacher["fusion_weights"] = {
            "alpha_perch": best["alpha_perch"],
            "alpha_prior": best["alpha_prior"],
            "alpha_probe": best["alpha_probe"],
            "alpha_temp":  best["alpha_temp"],
        }
        teacher.setdefault("meta", {})
        teacher["meta"]["fusion_sweep"] = {
            "grid": args.grid,
            "oof_seed_auc": seed_auc,
            "oof_best_auc": best["auc"],
        }
        save_pickle(teacher, teacher_path)
        print(f"Wrote new fusion weights -> {teacher_path}")

    write_json(
        {
            "grid": args.grid,
            "seed_weights": dict(zip(
                ["alpha_perch", "alpha_prior", "alpha_probe", "alpha_temp"], SEED
            )),
            "seed_oof_auc": seed_auc,
            "best": best,
        },
        art_dir / "02b_fusion_sweep.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
