"""Cloud training pipeline stages (Plan Y).

Each stage is a module with a ``run(cfg, args)`` entrypoint. The driver in
``run.py`` imports them lazily and calls ``run``. Stages are:

    S2  prepare_mel       comp audio -> uint8 mel cache
    S3  perch_pseudo      Perch v2 -> soft labels on all train_soundscapes
    S5  finetune          eca_nfnet_l0 5-fold FT on comp data
    S6  infer_pseudo      iter-1 pseudo from S5 ensemble
    S7  finetune_iter     FT again with S6 pseudo
    S9  export_openvino   .pt -> ONNX -> OpenVINO fp16
"""

from __future__ import annotations

from typing import Callable

# Core pipeline order. S3 (Perch v2 CPU teacher pseudo) was originally
# intended as an auxiliary supervision signal on all train_soundscapes, but
# downstream stages (S5/S7) don't consume it — they rely on the 708
# per-window hard labels + iterative pseudo via S6. S3 is also prohibitively
# slow on CPU (TF 2.20 has no sm_120 kernels so it can't use the 5090 GPU),
# so we exclude it from the default pipeline. Users who want the Perch
# pseudo cache can still run `python run.py stage S3 --config ...` manually.
STAGE_ORDER = ["S2", "S5", "S6", "S7", "S9"]
ALL_STAGES  = ["S2", "S3", "S5", "S6", "S7", "S9"]


def load_stage(name: str) -> Callable:
    """Import a stage module on demand and return its ``run`` function."""
    mod_map = {
        "S2": "stages.s2_prepare_mel",
        "S3": "stages.s3_perch_pseudo",   # optional; not in STAGE_ORDER
        "S5": "stages.s5_finetune",
        "S6": "stages.s6_infer_pseudo",
        "S7": "stages.s7_finetune_iter",
        "S9": "stages.s9_export_openvino",
    }
    if name not in mod_map:
        raise KeyError(f"unknown stage {name!r}; valid: {list(mod_map)}")
    import importlib
    mod = importlib.import_module(mod_map[name])
    if not hasattr(mod, "run"):
        raise AttributeError(f"{mod_map[name]} has no run(cfg, args)")
    return mod.run
