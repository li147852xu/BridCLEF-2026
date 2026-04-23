# BirdCLEF+ 2026 — Cloud pipeline (Plan Y)

One-page ops guide for running the whole training + export pipeline on a
rented **RTX 5090** box (AutoDL / Vast.ai). ~30 GPU-hours, fits into
**30 GB system + 50 GB data** AutoDL defaults.

## Status snapshot (2026-04-23)

- Plan Y fully trained, exported, and submitted to Kaggle.
- **Public LB = 0.866** (below the 0.90 target; under the `0.883` v2 baseline).
- Root cause of the gap is identified (**S7 pseudo-label leakage**, see
  [Known issues](#known-issues--why-plan-y-underperformed) below). Fix
  scoped for Plan Y.1 / Plan X.

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
#    Kaggle: prefer the new-style KGAT_ token from
#    https://www.kaggle.com/settings → API → Create New Token.
export KAGGLE_API_TOKEN=KGAT_<your-kaggle-access-token>
# legacy fallback (if you only have the old-style kaggle.json):
#   export KAGGLE_USERNAME=<your-username>; export KAGGLE_KEY=<32-char-hex>
export HF_TOKEN=hf_<your-hf-token>       # optional, enables ckpt backup

# 4. bootstrap (installs deps, downloads data + weights, ~52 GB)
bash scripts/cloud_bootstrap.sh --config configs/cloud_Y.yaml

# 5. run everything (resumable; skips stages already marked done)
python run.py pipeline --config configs/cloud_Y.yaml

# 6. bundle for Kaggle (see "Submission workflow" below)
#    — uses the HF bridge because AutoDL can't reach Kaggle's upload endpoint
python scripts/build_submit_notebook.py   # regen submit_v5.ipynb
# ... then run kaggle_submit/hf_to_kaggle_dataset.ipynb from inside Kaggle

# 7. before shutting the instance down
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
| S2 | ``stages.s2_prepare_mel``     | ``train_soundscapes`` OGG                   | uint8 mel-spec NPZ per file   | **2.7 min** (measured) |
| S5 | ``stages.s5_finetune``        | mel cache + train.csv + 1478 hard labels    | 5 fold ``eca_nfnet_l0`` ckpts | **2.75 h** (measured, incl. recovery) |
| S6 | ``stages.s6_infer_pseudo``    | S5 ensemble + all soundscape mels           | sparse pseudo npz (127k rows) | **4 min** (measured) |
| S7 | ``stages.s7_finetune_iter``   | S5 ckpt + S6 pseudo + comp data             | 5 fold FT-iter ckpts          | **6.3 h** (measured) |
| S9 | ``stages.s9_export_openvino`` | S7 ckpts                                    | ``export/fold_*.xml`` fp16    | **~1 min** (measured) |

**S3 (Perch v2 teacher pseudo) is intentionally excluded** from the default
pipeline order — TF 2.20 has no Blackwell (sm_120) kernels so it can only
run on CPU, and none of the downstream stages consume its output anyway. It
stays importable via ``python run.py stage S3`` if you ever want the Perch
cache for other experiments.

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

## Full pipeline — what each file does

| Module | Role | Resumable? |
|---|---|---|
| ``common/cloud_paths.py`` | YAML config loader with ``${a.b}`` interpolation | n/a |
| ``common/fold_split.py`` | Deterministic GroupKFold by ``site`` | n/a |
| ``common/augment.py`` | SpecAugment, mixup, cutmix, time-shift, noise | n/a |
| ``common/models.py`` | timm backbone + 234-class head + Sydorskyy ckpt loader + EMA | n/a |
| ``common/datasets.py`` | train_audio / soundscape / pseudo datasets (mel cache aware) | n/a |
| ``common/training.py`` | Trainer w/ AMP, focal BCE, EMA, SWA, atomic ckpt, HF backup | ✓ |
| ``stages/s2_prepare_mel.py`` | Competition audio → uint8 mel NPZ | per-file ✓ |
| ``stages/s3_perch_pseudo.py`` | Perch v2 teacher → soft probs per shard | per-shard ✓ |
| ``stages/s5_finetune.py`` | 5 fold FT on comp data (warm-start from 2025 weights) | per-fold ✓ |
| ``stages/s6_infer_pseudo.py`` | 5 fold ensemble → sparse pseudo NPZ (pos ≥ 0.9 / neg ≤ 0.05) | n/a (fast) |
| ``stages/s7_finetune_iter.py`` | Second FT pass with pseudo mixed in (warm-start from S5) | per-fold ✓ |
| ``stages/s9_export_openvino.py`` | S7 ``.pt`` → ONNX → OpenVINO fp16 IR | per-fold ✓ |
| ``kaggle_submit/submit_v5.ipynb``       | 5-fold OV + adaptive TTA + neighbour smoothing + per-taxon temp + placeholder CSV | n/a |
| ``kaggle_submit/hf_to_kaggle_dataset.ipynb`` | HF → Kaggle dataset bridge (run from inside Kaggle) | n/a |
| ``scripts/build_submit_notebook.py``    | Regenerate ``submit_v5.ipynb`` cells (in-place rewrite) | n/a |

## Checkpoint resume guarantees

Every training stage writes ``last.pt`` atomically (``.pt.tmp`` → rename) every
``save_every_steps`` steps, plus ``epoch_N.pt`` per epoch (top-3 rotated) and
``best.pt`` on val-AUC improvement. Restart the same ``python run.py stage ...``
command and the Trainer:

1. Loads model, optimizer, scheduler, AMP scaler, EMA shadow, and SWA state.
2. Restores Python / numpy / CUDA RNG.
3. Continues at the same epoch + global step.

Every 30 min a background thread pushes the whole fold directory to the HF
repo in ``checkpoint.hf_backup.repo_id`` so a nuked VM isn't fatal.

---

## Submission workflow

Post-S9 the 5 OpenVINO IR files plus ``manifest.json`` and the offline
``wheels/`` directory go to Kaggle as a dataset. AutoDL CN POPs **cannot
reach Kaggle's GCS resumable-upload endpoint** (the IP range hosting
``storage.googleapis.com``'s blob-upload path is unreachable even though
``www.kaggle.com`` and plain ``storage.googleapis.com`` are both fine), so we
go through HuggingFace as a bridge:

```
AutoDL → (network_turbo HF proxy) → HF private repo bundle_y/
                                         │
                                         ↓
                       kaggle_submit/hf_to_kaggle_dataset.ipynb
                      (runs inside Kaggle, uses Kaggle Secrets HF_TOKEN)
                                         │
                                         ↓
                         Kaggle dataset: <user>/birdclef-2026-bundle-y
                                         │
                                         ↓
                         attach to kaggle_submit/submit_v5.ipynb
```

**Step by step:**

1. On the 5090 box, bundle artefacts into one folder (e.g.
   ``/root/autodl-tmp/bridclef/bundle``): ``fold_{0..4}.{xml,bin}``,
   ``manifest.json``, ``wheels/*.whl`` (openvino + openvino_telemetry + numpy
   matching the Kaggle runtime Python), plus ``dataset-metadata.json``.
2. ``hf upload-folder`` that dir into the ckpt HF repo under ``bundle_y/``
   (the existing backup repo — commit quota is independent of other repos).
3. On Kaggle, open a fresh notebook and paste the cells of
   ``kaggle_submit/hf_to_kaggle_dataset.ipynb``. Add ``HF_TOKEN`` as a
   notebook Secret, set Internet: On, Save & Run All. It
   ``snapshot_download`` s the bundle, rewrites ``dataset-metadata.json``,
   then ``kaggle datasets create`` — all from within Kaggle's network so
   the upload endpoint is reachable.
4. Open ``kaggle_submit/submit_v5.ipynb`` on Kaggle, attach the competition
   + the new dataset, Save & Run All, then Submit to competition.

### Submit notebook design (submit_v5.ipynb)

Generated / regenerable via ``python scripts/build_submit_notebook.py``.
Key defensive patterns (learned the hard way — see git log for the saga):

- **Cell 0** installs **every** wheel in the bundle (not just openvino-\*)
  so transitive deps like ``openvino_telemetry`` don't silently break the
  import in Cell 4.
- **Cell 1** writes a **placeholder all-zero submission.csv immediately**.
  Any later crash leaves a valid CSV, so the scored result is PB 0.500
  (random) rather than "Notebook Threw Exception". Also enumerates test
  files with a 3-tier fallback (``BASE/test_soundscapes``, rglob,
  stem-scan) and drops to a **dry-run on train_soundscapes[:20]** when
  editor Save Version commit sees no test files.
- **Cell 2** builds ``LABEL_PERM`` from ``sample_sub.columns[1:]`` →
  ``manifest.primary_labels`` indices — never trust column order, always
  derive a permutation.
- **Cell 7** runs an **adaptive ETA loop**: probes the first 8 files with
  1-TTA to learn real per-file cost, then picks the largest TTA set
  (3/2/1 shifts) that fits in an 86-min budget. Writes ``submission.csv``
  atomically every 50 files; emergency brake at 88 min.

---

## Known issues — why Plan Y underperformed

### S7 val AUC looked amazing but PB landed at 0.866

S7's in-training val AUC numbers:

| Fold | val rows | S5 best val | S7 best val | jump |
|---|---|---|---|---|
| 0 | 78  | 0.7162 | 0.8519 | +0.14 |
| 1 | 192 | 0.5856 | 0.9459 | **+0.36** |
| 2 | 992 | 0.7227 | 0.9731 | **+0.25** |
| 3 | 120 | 0.8673 | 0.9783 | +0.11 |
| 4 | 96  | 0.7708 | 0.9162 | +0.15 |
| weighted mean | — | 0.7194 | **0.9599** | +0.24 |

A +0.24 weighted-mean jump purely from one round of self-pseudo is
**physically implausible**; the real generalisation step should be on the
order of +0.02 – +0.05. Observed Public LB 0.866 lines up much better
with the S5-level 0.72 signal than the S7-level 0.96.

**Root cause:** ``stages/s6_infer_pseudo.py`` generates pseudo labels for
**every** 5 s window of every soundscape file, then
``stages/s7_finetune_iter.py`` loads **all** of them as training data —
including windows that are in each fold's val split. Each fold's model
then trains on S5-ensemble predictions for its own val windows, so at
val time it sees targets that are almost-identical to its own training
pseudo targets. Val AUC becomes a memorisation metric.

### Fix (Plan Y.1 — do this before Plan Z)

Patch S6/S7 so that when generating pseudo labels for **fold k**, we
exclude the windows whose (filename, window_idx) pair falls into fold k's
val set. Mechanically:

```python
# s6_infer_pseudo.py: produce a per-fold pseudo file.
for fold in range(n_folds):
    val_set = set(build_soundscape_fold_table(...).query("fold == @fold")
                  .apply(lambda r: (r.filename_stem, r.window_idx), axis=1))
    # Use ensemble of *other* folds for prediction, and exclude val windows:
    save(f"pseudo_iter1_excl_fold{fold}.npz", ...)

# s7_finetune_iter.py: fold k loads pseudo_iter1_excl_fold{k}.npz
```

Expected effect: val AUC numbers drop to honest levels (~0.80 weighted
mean), **Public LB goes up** by roughly the gap we currently see between
the fold-2 S5 signal (0.72, val 992 rows) and the current PB (0.866) —
probably +0.02 – +0.03.

This is a code change of ~50 lines + one S6 + S7 re-run (≈ 7 h GPU).
Strictly better than adding a second backbone (Plan Z) before fixing
this, because Plan Z would inherit the same leakage bug.

### Secondary lever (Plan Z)

After Plan Y.1 is in, adding a second backbone
(``tf_efficientnetv2_s_in21k``, pretrained on the Sydorskyy 2025 bundle
same as the nfnet) is the next expected +0.02 – +0.03. See
[README.md](./README.md) historical context section for the comparison
with BirdCLEF 2025 Nth-place solutions that used a two-backbone stack.
