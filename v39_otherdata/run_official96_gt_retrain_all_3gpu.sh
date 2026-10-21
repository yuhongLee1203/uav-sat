#!/usr/bin/env bash
# Fresh Official96 c4m temporal training, followed by every v39 ablation.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SOURCE="${GT_SOURCE_SUITE:-$ROOT/v39_otherdata/output/abcd_table6_official96_temporal_20260923}"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
SUITE="${GT_RETRAIN_OUTPUT:-$ROOT/v39_otherdata/output/official96_gt_retrained_${STAMP}}"

if [[ -e "$SUITE" ]]; then
  echo "Refusing to reuse an existing run directory: $SUITE" >&2
  echo "Set GT_RETRAIN_OUTPUT to a new, empty path for a fresh training run." >&2
  exit 2
fi
for city in citya cityb cityc cityd; do
  [[ -s "$SOURCE/$city/prepared/official96_split_audit.json" ]] || {
    echo "Missing Official96 prepared data: $SOURCE/$city/prepared" >&2
    exit 2
  }
done
mkdir -p "$SUITE"
export BEARING_FREEZE_VISUAL=0
printf '%s\n' \
  'Official96 c4m temporal GT assisted experiment' \
  "prepared_source=$SOURCE" \
  'weights=initialized_from_scratch; no historical checkpoint imported' \
  'training=ABCD city episodes, frames 3/2/1 on GPUs 0/5/6' \
  'evaluation=Full and nine ablations on GPUs 0/5/6' \
  "started=$(date -Is)" > "$SUITE/FRESH_TRAINING_PROTOCOL.txt"
echo "[FRESH TRAIN] $SUITE"
GT_SOURCE_SUITE="$SOURCE" GT_ABLATION_SUITE="$SUITE" \
  bash v39_otherdata/run_official96_gt_ablation_3gpu.sh
if [[ "${GT_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[PREFLIGHT ONLY] Training and evaluation were not started."
  exit 0
fi
test -s "$SUITE/official96_gt_all_metrics.md"
printf '%s\n' "completed=$(date -Is)" >> "$SUITE/FRESH_TRAINING_PROTOCOL.txt"
echo "[ALL RESULTS] $SUITE/official96_gt_all_metrics.md"
