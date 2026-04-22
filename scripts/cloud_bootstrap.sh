#!/usr/bin/env bash
# BirdCLEF+ 2026 — one-shot cloud bootstrap for AutoDL / Vast.ai.
#
# What this does (idempotent, safe to re-run after reboot):
#   1. Resolve work root, repo root from args.
#   2. pip install -r requirements-cloud.txt (skips already-installed pins).
#   3. Write ~/.kaggle/kaggle.json if KAGGLE_USERNAME / KAGGLE_KEY are in env.
#   4. Download competition data (BirdCLEF+ 2026) via kaggle CLI, unzip.
#   5. Download Perch v2 SavedModel via kaggle models API.
#   6. Download Sydorskyy 2025 pretrained backbones (Kaggle dataset).
#   7. Print a GO/NO-GO summary.
#
# Usage:
#   bash scripts/cloud_bootstrap.sh [--config configs/cloud_Y.yaml]
#
# Required env vars (set before running):
#   KAGGLE_USERNAME, KAGGLE_KEY         — Kaggle API credentials
#   HF_TOKEN                            — optional, enables ckpt backup to HF Hub
#
# Exit 0 only if every mandatory asset is present. Training scripts can rely
# on that invariant rather than probing paths themselves.

set -euo pipefail

# -------- 0. parse args -----------------------------------------------------
CONFIG="configs/cloud_Y.yaml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    -h|--help) sed -n '2,24p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# Go to repo root (dir of this script's parent).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Read work_root from config using Python (no yaml parser in pure bash).
WORK_ROOT="$(python3 - <<PY
import sys, yaml, os
from pathlib import Path
cfg = yaml.safe_load(open("$CONFIG"))
p = os.environ.get("BIRDCLEF_WORK_ROOT") or cfg["paths"]["work_root"]
print(Path(p).expanduser())
PY
)"

echo "==[ bootstrap ]================================================"
echo "repo:       $REPO_ROOT"
echo "config:     $CONFIG"
echo "work_root:  $WORK_ROOT"
echo "==============================================================="

mkdir -p "$WORK_ROOT"/{data,weights,cache,checkpoints,export,logs,.flags}

# -------- 1. Python deps ----------------------------------------------------
echo "[1/6] installing Python deps (this may take a few minutes)..."
python3 -m pip install --quiet --upgrade pip
# Install torch first from cu128 index so Blackwell (sm_120, RTX 5090) kernels
# are in the wheel — avoids a 5-minute JIT compile at every process start.
python3 -m pip install --quiet torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128 || {
    echo "[warn] cu128 wheel install failed; falling back to default PyPI index"
    python3 -m pip install --quiet torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
}
python3 -m pip install --quiet -r requirements-cloud.txt

# 5090-specific sanity: report GPU + print any sm-compat warnings early.
python3 - <<'PY' || true
import torch
print(f"[torch] version={torch.__version__}  cuda={torch.version.cuda}  "
      f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[gpu]   {name}  sm_{cap[0]}{cap[1]}")
    if cap[0] >= 12:
        print("[gpu]   Blackwell detected (sm_120+); cu128 required — ok.")
PY

# -------- 2. Kaggle credentials --------------------------------------------
echo "[2/6] setting up Kaggle credentials..."
mkdir -p "$HOME/.kaggle"
if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
  cat > "$HOME/.kaggle/kaggle.json" <<EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_KEY}"}
EOF
  chmod 600 "$HOME/.kaggle/kaggle.json"
  echo "       wrote ~/.kaggle/kaggle.json from env vars"
elif [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
  chmod 600 "$HOME/.kaggle/kaggle.json"
  echo "       found existing ~/.kaggle/kaggle.json"
else
  echo "ERROR: no Kaggle credentials. Set KAGGLE_USERNAME + KAGGLE_KEY, or upload kaggle.json to ~/.kaggle/" >&2
  exit 3
fi

# -------- 3. HF token -------------------------------------------------------
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[3/6] HF_TOKEN present — checkpoint backup will be enabled"
  # huggingface-cli login is non-interactive via env var; we also write the
  # token file so subprocesses pick it up.
  mkdir -p "$HOME/.cache/huggingface"
  echo -n "$HF_TOKEN" > "$HOME/.cache/huggingface/token"
else
  echo "[3/6] no HF_TOKEN — ckpt backup disabled (training still works)"
fi

# -------- 4. competition data ----------------------------------------------
COMP_DIR="$WORK_ROOT/data/birdclef-2026"
echo "[4/6] competition data -> $COMP_DIR"
if [[ -f "$COMP_DIR/taxonomy.csv" && -d "$COMP_DIR/train_soundscapes" ]]; then
  echo "       already present, skipping download"
else
  mkdir -p "$COMP_DIR"
  cd "$COMP_DIR"
  kaggle competitions download -c birdclef-2026 -p "$COMP_DIR" --force
  # unzip (may be single zip or multiple)
  for z in "$COMP_DIR"/*.zip; do
    [[ -f "$z" ]] || continue
    echo "       unzip $(basename "$z")..."
    unzip -n -q "$z" -d "$COMP_DIR"
    rm -f "$z"
  done
  cd "$REPO_ROOT"
fi

# -------- 5. Perch v2 ------------------------------------------------------
PERCH_DIR="$WORK_ROOT/weights/perch_v2_cpu/1"
echo "[5/6] Perch v2 -> $PERCH_DIR"
if [[ -f "$PERCH_DIR/saved_model.pb" ]]; then
  echo "       already present"
else
  mkdir -p "$(dirname "$PERCH_DIR")"
  # Perch lives as a Kaggle "model". The CLI writes it under the target path.
  kaggle models instances versions download \
      google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1 \
      -p "$PERCH_DIR" --force || {
    echo "ERROR: Perch v2 download failed." >&2; exit 4;
  }
  # Unzip if needed (kaggle model downloads arrive as a .tar.gz sometimes)
  for ar in "$PERCH_DIR"/*.tar.gz "$PERCH_DIR"/*.zip; do
    [[ -f "$ar" ]] || continue
    case "$ar" in
      *.tar.gz) tar -xzf "$ar" -C "$PERCH_DIR" && rm -f "$ar";;
      *.zip)    unzip -n -q "$ar" -d "$PERCH_DIR" && rm -f "$ar";;
    esac
  done
fi

# -------- 6. 2025 pretrained backbones ------------------------------------
PRETRAIN_DIR="$WORK_ROOT/weights/birdclef_2025_pretrained"
echo "[6/6] 2025 pretrained backbones -> $PRETRAIN_DIR"
if [[ -d "$PRETRAIN_DIR" && -n "$(ls -A "$PRETRAIN_DIR" 2>/dev/null)" ]]; then
  echo "       already present"
else
  mkdir -p "$PRETRAIN_DIR"
  # Sydorskyy team dataset slug (public). Adjust here if slug changes.
  kaggle datasets download -d vladimirsydor/bird-clef-2025-all-pretrained-models \
      -p "$PRETRAIN_DIR" --unzip || {
    echo "WARN: failed to pull 2025 pretrained dataset — will need manual download." >&2
  }
fi

# -------- summary ----------------------------------------------------------
echo ""
echo "==[ summary ]=================================================="
ok=true
check () {
  local name=$1 path=$2 need=$3
  if [[ -e "$path" ]]; then
    echo "  [ok]   $name  ($path)"
  else
    echo "  [MISS] $name  ($path)  — $need"
    ok=false
  fi
}
check "comp taxonomy"   "$COMP_DIR/taxonomy.csv"             "re-run kaggle download"
check "comp audio"      "$COMP_DIR/train_soundscapes"        "re-run kaggle download"
check "perch savedmodel" "$PERCH_DIR/saved_model.pb"         "check kaggle models perms"
check "2025 pretrained" "$PRETRAIN_DIR"                      "optional but recommended"
echo "==============================================================="
$ok && echo "bootstrap OK — ready to run:  python run.py status --config $CONFIG" \
    || { echo "bootstrap INCOMPLETE — fix issues above then re-run" >&2; exit 5; }
