#!/usr/bin/env bash
# BirdCLEF+ 2026 — Plan Z full pipeline driver (AutoDL 5090).
#
# Runs the second-backbone training + bundle + HF push in one shot.
# Idempotent: re-running picks up where the last invocation stopped, since
# every stage writes a done-flag at ${flags_dir_z} and ``run.py pipeline``
# skips completed stages by default.
#
# Stages (with rough wall-clock budget on a single RTX 5090):
#
#   S2  prepare_mel       30 min   (idempotent; usually skip-confirms because
#                                  Y already populated mel_cache/)
#   S5  finetune          ~3-4h   (5 folds × 18 epochs of EfficientNetV2-S)
#   S6  infer_pseudo      ~30 min  (TTA inference on all train_soundscapes)
#   S7  finetune_iter     ~2.5-3h  (5 folds × 12 epochs with iter-1 pseudo)
#   S9  export_openvino   ~5-10 min (5x .pt → ONNX → OV fp16 IR)
#   bundle + HF push      ~5-10 min
#
# Total: ~7-9 hours. Add ~30 min for fresh-box bootstrap on first run.
#
# Usage (intended SSH-time invocation, in a tmux session):
#   tmux new -s planz
#   cd /root/BridCLEF-2026
#   bash scripts/cloud_planz_run.sh
#   # detach: Ctrl+b d
#   # re-attach later: tmux a -t planz
#
# Env vars (must be set BEFORE running, same as Plan Y):
#   KAGGLE_USERNAME, KAGGLE_KEY      (or KAGGLE_API_TOKEN)
#   HF_TOKEN                         (mandatory — backup + bundle push)
#
# Args:
#   --skip-bootstrap     Don't re-run cloud_bootstrap.sh (faster repeat runs)
#   --skip-hf            Build the bundle locally but don't upload to HF
#   --to <stage>         Stop after this stage (e.g. --to S5 to just train)

set -euo pipefail

CONFIG="configs/cloud_Z.yaml"
SKIP_BOOTSTRAP=0
SKIP_HF=0
STOP_AFTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-bootstrap) SKIP_BOOTSTRAP=1; shift;;
    --skip-hf)        SKIP_HF=1; shift;;
    --to)             STOP_AFTER="$2"; shift 2;;
    -h|--help)        sed -n '2,33p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "==[ planz ]===================================================="
echo "repo:      $REPO_ROOT"
echo "config:    $CONFIG"
echo "stop_at:   ${STOP_AFTER:-(end)}"
echo "skip_hf:   $SKIP_HF"
echo "==============================================================="

# ---- 0. AutoDL turbo (CN POPs need this for HF/github reach) --------------
if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
  echo "[net] AutoDL network_turbo active"
fi

# ---- 1. Bootstrap (idempotent) --------------------------------------------
# cloud_bootstrap.sh is safe to re-run; it skips already-downloaded data,
# already-installed wheels, etc. But on a warm box it still takes ~1-2 min,
# so allow --skip-bootstrap for repeat runs in the same session.
if (( ! SKIP_BOOTSTRAP )); then
  echo ""
  echo "==[ stage 0: bootstrap ]======================================="
  bash scripts/cloud_bootstrap.sh --config "$CONFIG"
fi

# ---- 2. Sanity: GPU + config -------------------------------------------
echo ""
echo "==[ stage 1: sanity ]=========================================="
python3 - <<PY
import torch, yaml
print(f"[torch] {torch.__version__}  cuda_avail={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[gpu]   {torch.cuda.get_device_name(0)}  "
          f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
cfg = yaml.safe_load(open("$CONFIG"))
print(f"[cfg]   plan={cfg['plan']}  backbone={cfg['backbones'][0]['name']}")
print(f"[cfg]   epochs_ft={cfg['backbones'][0]['epochs_ft']}  bs={cfg['backbones'][0]['batch_size']}")
PY

python3 run.py status --config "$CONFIG" || true

# ---- 3. Pipeline run (skips completed stages thanks to done-flags) --------
# We let run.py handle stage ordering + flag bookkeeping. Two reasons NOT to
# call individual stages here:
#   (a) `run.py pipeline` is the one path that respects --to/--from + flag
#       cleanup. Bypassing it risks leaving inconsistent flag state.
#   (b) Single entry point = single place to interpret Ctrl-C. Each stage's
#       graceful_sigint handler flushes ckpts at the right granularity.
echo ""
echo "==[ stage 2: pipeline ]========================================"
PIPELINE_ARGS=()
if [[ -n "$STOP_AFTER" ]]; then
  PIPELINE_ARGS+=("--to" "$STOP_AFTER")
fi
python3 run.py pipeline --config "$CONFIG" "${PIPELINE_ARGS[@]}"

# Stop here if user asked to stop before S9 (e.g. --to S5 for training only)
if [[ -n "$STOP_AFTER" && "$STOP_AFTER" != "S9" ]]; then
  echo ""
  echo "[stopped] requested --to $STOP_AFTER — bundle/push skipped."
  echo "         re-run without --to (or with --to S9) when ready to ship."
  exit 0
fi

# ---- 4. Bundle + HF push --------------------------------------------------
echo ""
echo "==[ stage 3: bundle ]=========================================="
BUNDLE_ARGS=()
if (( SKIP_HF )); then
  BUNDLE_ARGS+=("--skip-hf")
fi
python3 scripts/cloud_planz_bundle.py --config "$CONFIG" "${BUNDLE_ARGS[@]}"

# ---- 5. Done summary ------------------------------------------------------
echo ""
echo "==[ done ]====================================================="
echo "Plan Z bundle is built and (if not --skip-hf) pushed to HF as"
echo "  Tiantanghuaxiao/birdclef-2026-ckpts:bundle_y2/"
echo ""
echo "Next steps (do these on Kaggle, NOT here):"
echo ""
echo "  1. Open kaggle_submit/00_setup/hf_to_kaggle_dataset.ipynb on Kaggle."
echo "     Copy & Edit it to a new notebook named 'hf_to_kaggle_dataset_y2'."
echo "     In the 'snapshot_download' call, change 'bundle_y1/*' → 'bundle_y2/*',"
echo "     and in 'dataset-metadata.json' override change the slug to"
echo "     'tiantanghuaxiao/birdclef-2026-bundle-y2'. Save & Run All."
echo ""
echo "  2. After the new dataset exists, attach BOTH bundles to your hybrid"
echo "     submit notebook (e.g. exp_A1_W030_rank_arith). The OV cell needs"
echo "     a small patch to load both — see scripts/build_submit_hybrid.py"
echo "     (a follow-up commit will wire dual-bundle ensembling)."
echo "==============================================================="
