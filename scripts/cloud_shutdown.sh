#!/usr/bin/env bash
# BirdCLEF+ 2026 — pre-shutdown sync.
#
# Run this before you stop the AutoDL instance, so the latest checkpoints +
# logs survive the box being wiped. Uploads to the Hugging Face repo
# configured in the YAML under `checkpoint.hf_backup.repo_id`.
#
# Usage:
#   bash scripts/cloud_shutdown.sh [--config configs/cloud_Y.yaml]
#
# Requires HF_TOKEN in env.

set -euo pipefail

CONFIG="configs/cloud_Y.yaml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set — cannot back up. Set it and re-run." >&2
  exit 3
fi

# Extract paths + hf repo from config.
read -r WORK_ROOT HF_REPO <<<"$(python3 - <<PY
import yaml, os
cfg = yaml.safe_load(open("$CONFIG"))
work_root = os.environ.get("BIRDCLEF_WORK_ROOT") or cfg["paths"]["work_root"]
repo = cfg.get("checkpoint", {}).get("hf_backup", {}).get("repo_id", "")
print(work_root, repo)
PY
)"

if [[ -z "$HF_REPO" ]]; then
  echo "ERROR: checkpoint.hf_backup.repo_id empty in $CONFIG — edit the YAML." >&2
  exit 4
fi

echo "==[ shutdown sync ]==========================================="
echo "work_root: $WORK_ROOT"
echo "hf_repo:   $HF_REPO"
echo "=============================================================="

CKPT="$WORK_ROOT/checkpoints"
LOGS="$WORK_ROOT/logs"
FLAGS="$WORK_ROOT/.flags"
EXPORT="$WORK_ROOT/export"

# Use huggingface-cli; these are additive per-folder uploads (not destructive).
upload () {
  local src=$1 sub=$2
  if [[ -d "$src" && -n "$(ls -A "$src" 2>/dev/null)" ]]; then
    echo "[uploading] $src -> $HF_REPO:$sub"
    huggingface-cli upload "$HF_REPO" "$src" "$sub" \
      --repo-type model \
      --commit-message "final sync $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --token "$HF_TOKEN"
  else
    echo "[skip] $src (empty or missing)"
  fi
}

upload "$CKPT"   "checkpoints"
upload "$LOGS"   "logs"
upload "$FLAGS"  ".flags"
upload "$EXPORT" "export"

echo "shutdown sync complete. You can now stop the instance."
