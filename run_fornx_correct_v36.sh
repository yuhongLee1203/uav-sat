#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
bash "$ROOT/prepare_fornx_gvsk.sh"

SRC="$ROOT/v36-GvsK/original_forNX"
OUT="$ROOT/v36-GvsK/output/original_full"
GPU="${GPU:-0}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
JITTER_M="${JITTER_M:-8}"

rm -rf "$OUT"
mkdir -p "$OUT"
cp "$ROOT/v36-GvsK/output/source_audit.txt" "$OUT/source_audit.txt" 2>/dev/null || true
cp "$ROOT/v36-GvsK/output/forNX_copy_diff.txt" "$OUT/forNX_copy_diff.txt" 2>/dev/null || true

cd "$SRC"
echo "================================================================================================"
echo "CORRECT forNX / ORIGINAL FULL V36"
echo "source       : $SRC"
echo "output       : $OUT"
echo "GPU          : $GPU"
echo "jitter       : $JITTER_M m"
echo "Kalman mode  : learned (original full V36)"
echo "================================================================================================"

# Explicitly pin every V36 ablation switch so shell leftovers from another
# experiment cannot silently change the architecture.
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS="${CPU_THREADS:-4}" \
MKL_NUM_THREADS="${CPU_THREADS:-4}" \
OPENBLAS_NUM_THREADS="${CPU_THREADS:-4}" \
NUMEXPR_NUM_THREADS="${CPU_THREADS:-4}" \
UAVSAT_OUTPUT_DIR="$OUT" \
UAVSAT_REFERENCE_PROTOCOL="controlled_gt_jitter" \
UAVSAT_EXPERIMENT_VARIANT="full_v36" \
UAVSAT_EXPERIMENT_ANCHOR="softms" \
UAVSAT_EXPERIMENT_FRAME_COUNT="3" \
UAVSAT_EXPERIMENT_MOTION="quadratic" \
UAVSAT_EXPERIMENT_KALMAN="learned" \
UAVSAT_EXPERIMENT_DISABLE_GRU="0" \
UAVSAT_EXPERIMENT_FORWARD_ONLY="1" \
UAVSAT_FORWARD_ORIGIN_BACKSHIFT_M="0.0" \
python3 -u robust_tracker.py \
  --mode train_eval \
  --visual-epochs "$VISUAL_EPOCHS" \
  --temporal-epochs "$TEMPORAL_EPOCHS" \
  --patience "$PATIENCE" \
  --jitter-m "$JITTER_M" \
  2>&1 | tee "$OUT/run.log"

echo "[forNX full] summary should be under: $OUT"
