#!/usr/bin/env bash
# Train/evaluate the GvsK variant: final localization = SoftMS anchor + GRU correction.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${ROOT}/meanshift_gru"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${ROOT}/v36_training_data}"
OUT="${UAVSAT_OUTPUT_DIR:-${ROOT}/output/meanshift_gru}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
BACKBONE="${UAVSAT_BACKBONE:-mobileclip2_s2}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
JITTER_M="${JITTER_M:-8}"

for route in route_A route_B route_C; do
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || {
    echo "Missing ${DATA_ROOT}/routes/${route}/frames.csv" >&2
    echo "Training requires a complete Route A/B/C data root; packaged forNX has only B/C." >&2
    exit 2
  }
done
[[ -f "${DATA_ROOT}/satellite/sim_map_competition_roi_crop.png" ]] || {
  echo "Missing satellite map under ${DATA_ROOT}/satellite" >&2; exit 2; }

mkdir -p "${OUT}"
export TORCH_HOME="${ROOT}/pretrained_cache/torch"
export HF_HOME="${ROOT}/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "=== V36 GvsK: SoftMS + GRU visual final localization ==="
echo "data=${DATA_ROOT} output=${OUT} backbone=${BACKBONE} device=${DEVICE}"
(
  cd "${SRC}"
  UAVSAT_DEVICE="${DEVICE}" UAVSAT_OUTPUT_DIR="${OUT}" UAVSAT_DATA_ROOT="${DATA_ROOT}" \
  UAVSAT_BACKBONE="${BACKBONE}" UAVSAT_EXPERIMENT_KALMAN=none \
  UAVSAT_EXPERIMENT_ANCHOR=softms UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
  UAVSAT_EXPERIMENT_MOTION=quadratic UAVSAT_EXPERIMENT_DISABLE_GRU=0 \
  UAVSAT_EXPERIMENT_FORWARD_ONLY=1 UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
  python3 -u robust_tracker.py --mode train_eval --visual-epochs "${VISUAL_EPOCHS}" \
    --temporal-epochs "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}" --jitter-m "${JITTER_M}"
) 2>&1 | tee "${OUT}/train_eval.log"

echo "[DONE] GvsK outputs: ${OUT}"
