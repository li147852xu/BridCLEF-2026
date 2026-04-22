# BirdCLEF+ 2026 — Cloud pipeline (Plan Y)

One-page ops guide for running the whole training + export pipeline on a
rented **RTX 5090** box (AutoDL / Vast.ai). Target: **PB 0.90 – 0.92** in one
pass, ~30 GPU-hours, fits into **30 GB system + 50 GB data** AutoDL defaults
(expand only if you hit a ceiling — see disk budget below).

If you want the deep background (why this architecture, how it compares to
2025 winning solutions, etc.) read [README.md](./README.md) — it's the old
plan. The *cloud* plan supersedes it.

---

## Target env (pin these on AutoDL)

| Component | Version |
|---|---|
| GPU | NVIDIA RTX 5090 (Blackwell, sm_120) × 1 |
| Host | Ubuntu 22.04 |
| CUDA | 12.8 |
| Python | 3.12 |
| PyTorch | 2.8.0 (cu128 wheel) |
| TensorFlow | 2.20 (for Perch; we run it on CPU — see note below) |
| OpenVINO | 2024.4 (for final submission export) |

### 5090-specific notes

- The stock PyTorch 2.8.0 wheel on `download.pytorch.org/whl/cu128` has
  Blackwell kernels baked in, so there's no 5-min JIT compile on startup.
- **Perch v2 runs on CPU by default** (`s3.device: cpu` in the YAML). TF 2.20
  does not ship sm_120 kernels yet; forcing GPU Perch risks a silent fallback
  or JIT crash. AutoDL 5090 hosts come with 16–32 core Xeons / EPYCs, so CPU
  Perch finishes S3 in ~2–3 hours — acceptable.

## TL;DR

```bash
# 1. rent a 5090 box on AutoDL (CUDA 12.8 / PyTorch 2.8 image), SSH in
# 2. clone this repo
git clone https://github.com/li147852xu/BridCLEF-2026.git
cd BridCLEF-2026

# 3. export credentials (do NOT commit these)
export KAGGLE_USERNAME=li147852xu
export KAGGLE_KEY=<your-kaggle-api-key>
export HF_TOKEN=<your-hf-token>          # optional, enables ckpt backup

# 4. bootstrap (installs deps, downloads data + weights, ~52 GB)
bash scripts/cloud_bootstrap.sh --config configs/cloud_Y.yaml

# 5. run everything (resumable; skips stages already marked done)
python run.py pipeline --config configs/cloud_Y.yaml

# 6. before shutting the instance down
bash scripts/cloud_shutdown.sh --config configs/cloud_Y.yaml
```

All intermediate artifacts land under ``/root/autodl-tmp/bridclef/`` (override
with ``BIRDCLEF_WORK_ROOT``). Stage progress tracked via ``.flags/*.done``
files; see ``python run.py status`` for a one-glance view.

---

## SSH to AutoDL — what to do

AutoDL gives you an SSH command on the instance detail page, something like
``ssh -p 35427 root@connect.bjb1.seetacloud.com``. Three ways to connect:

1. **Plain terminal** — copy the command from the web console, paste in your
   Mac Terminal. Password is in the web panel; right-click → copy.
2. **Key-based (no password after setup)**
   ```bash
   # on your Mac, once
   ssh-keygen -t ed25519 -C "autodl"                 # if you don't have one
   ssh-copy-id -p 35427 root@connect.bjb1.seetacloud.com   # sends public key
   # next time: no password prompt
   ```
3. **VS Code Remote-SSH** (most comfortable)
   - Install the "Remote - SSH" extension.
   - Cmd+Shift+P → "Remote-SSH: Add New SSH Host..." → paste the AutoDL command.
   - Connect; VS Code opens the remote filesystem like a local project.

Tips:
- AutoDL ``autodl-tmp`` is the 50 GB data disk mount point. Put all heavy data
  there; the 30 GB system disk just holds the repo + venv.
- Transfers: ``rsync -avz -e 'ssh -p 35427' ./ckpts/ root@...:/root/autodl-tmp/``
  or VS Code's file explorer drag-drop.
- If you reboot the instance the ``autodl-tmp`` disk survives; the system
  disk also persists. Only ``docker stop + docker rm`` wipes state — and
  ``cloud_shutdown.sh`` protects against that with HF Hub backup.

---

## Disk budget (30 + 50 GB)

Fits if we only cache soundscapes mels (not all of train_audio) and rotate
checkpoints aggressively:

| Item | Size | Where |
|---|---|---|
| birdclef-2026 data (unzipped) | ~30 GB | `/root/autodl-tmp/bridclef/data/` |
| Perch v2 SavedModel | 0.2 GB | `weights/perch_v2_cpu/1/` |
| Sydorskyy 2025 pretrained | 2 GB | `weights/birdclef_2025_pretrained/` |
| mel cache (soundscapes only) | ~1 GB | `cache/mel/train_soundscapes/` |
| Perch probs cache | ~0.2 GB | `cache/perch/shard_*/` |
| Checkpoints (5 fold × top-3 rotate) | ~8 GB | `checkpoints/` |
| OpenVINO export | 0.1 GB | `export/` |
| Logs / tmp | ~0.5 GB | `logs/` |
| **Total** | **~42 GB** | fits 50 GB data disk |

If you expand to a second backbone (Plan X) add ~10 GB; still fits in 60–70 GB.
AutoDL's data disk expansion is live — add 50 GB more without rebooting.

---

## Stage map

| ID | Module | Input | Output | ~time on 5090 |
|----|---|---|---|---|
| S2 | ``stages.s2_prepare_mel``   | ``train_soundscapes`` OGG                    | uint8 mel-spec NPZ per file   | 0.5 h (CPU-bound) |
| S3 | ``stages.s3_perch_pseudo``  | ``train_soundscapes`` OGG + Perch v2         | Perch sigmoid probs per shard | 2–3 h (CPU Perch) |
| S5 | ``stages.s5_finetune``      | mel cache + train.csv + 708 hard labels      | 5 fold ``eca_nfnet_l0`` ckpts | ~10 h (5090 is ~1.5× 4090) |
| S6 | ``stages.s6_infer_pseudo``  | S5 ensemble + ``train_soundscapes`` mels     | sparse pseudo npz             | 0.5 h |
| S7 | ``stages.s7_finetune_iter`` | S5 ckpt + S6 pseudo + comp data              | 5 fold FT-iter ckpts          | ~8 h |
| S9 | ``stages.s9_export_openvino`` | S7 ckpts                                   | ``export/fold_*.xml`` fp16    | 0.5 h |

Each stage is **idempotent**: re-running either skips (done flag set) or
resumes from the last checkpoint. Lose the box mid-run? Spin a new one,
``git clone``, ``huggingface-cli download`` the backup, re-enter, continue.

---

## File & disk layout on the box

```
/root/BridCLEF-2026/                     # this repo (code only)
/root/autodl-tmp/bridclef/               # all heavy data (configurable)
├── data/birdclef-2026/                  # ~50 GB — kaggle comp dump
├── weights/
│   ├── perch_v2_cpu/1/                  # ~200 MB — Perch SavedModel
│   └── birdclef_2025_pretrained/        # ~2 GB  — Sydorskyy backbones
├── cache/
│   ├── mel/train_audio/*.npz            # ~8 GB  — S2
│   ├── mel/train_soundscapes/*.npz      # ~2 GB  — S2
│   └── perch/shard_*/probs_fp16.npy     # ~200 MB — S3
├── checkpoints/{stage}/fold_{k}/        # 5–15 GB — S5, S7
├── export/                              # ~80 MB — S9 (OpenVINO fp16)
├── logs/*.jsonl                         # streaming metrics
└── .flags/{stage}.done                  # pipeline state
```

---

## Checkpoint & resume invariants (important)

Every training stage saves `{model, optimizer, lr_scheduler, epoch, global_step,
rng_state_torch/numpy/python, best_val_auc}`. On restart the runner detects
``checkpoints/{stage}/fold_{k}/last.pt`` and resumes exactly where it stopped
— no lost epochs. Atomic writes (``.tmp`` → ``rename``) guarantee you never
end up with half-written files.

Default cadence:
- Every 500 optimizer steps  → update ``last.pt``
- Every epoch                → write ``epoch_N.pt`` + update ``best.pt`` on val improvement
- Every 30 min               → background ``huggingface-cli upload`` to the HF repo named in config

If HF backup is enabled, you can always rebuild locally by:
```bash
huggingface-cli download <your-repo> --local-dir ./backup --repo-type model
```

---

## Credentials & secrets

Never commit tokens. The workflow is:

1. Tokens live in your shell as env vars (``KAGGLE_USERNAME``, ``KAGGLE_KEY``, ``HF_TOKEN``).
2. ``cloud_bootstrap.sh`` writes ``~/.kaggle/kaggle.json`` + ``~/.cache/huggingface/token`` from those vars.
3. Training scripts never see the raw tokens — they just call ``kaggle`` / ``huggingface-cli``.

If you need to change the HF backup repo, edit ``checkpoint.hf_backup.repo_id``
in ``configs/cloud_Y.yaml`` — keep the token out of git.

---

## Sanity probes (run these after bootstrap to catch issues early)

```bash
# does kaggle know who I am?
kaggle competitions files -c birdclef-2026 | head

# does TF see the GPU?
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# does torch see CUDA 12.8 and the 5090?
python -c "
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'avail:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0), 'sm:', torch.cuda.get_device_capability(0))
"

# does Perch load?
python -c "
from common.cloud_paths import load_cloud_config
from common.perch import PerchEngine
cfg = load_cloud_config('configs/cloud_Y.yaml')
e = PerchEngine(model_dir=cfg.perch_model)
print('Perch loaded, window_samples =', e.window_samples)
"

# pipeline status board
python run.py status --config configs/cloud_Y.yaml
```

---

## Which stages still need to be written

This README + `run.py` + the S2 / S3 scaffold are **batch 1**. Still coming:

- [ ] ``stages/s5_finetune.py`` — eca_nfnet_l0 5-fold FT with resume
- [ ] ``stages/s6_infer_pseudo.py`` — iter-1 pseudo from S5 ensemble
- [ ] ``stages/s7_finetune_iter.py`` — FT again with pseudo
- [ ] ``stages/s9_export_openvino.py`` — ONNX → OpenVINO fp16
- [ ] ``kaggle_submit/submit_v5.ipynb`` — multi-fold OpenVINO + TTA + neighbour smoothing

Batch 2 lands in the same repo shortly. S2 / S3 are already enough to
exercise the bootstrap + resume machinery end-to-end.
