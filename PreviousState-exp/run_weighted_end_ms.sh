#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEVICE="${UAVSAT_DEVICE:-cuda:0}"
JITTER_M="${JITTER_M:-8}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"

echo "[1/2] MobileNetV3: forward 3x6 Weighted Centroid -> GRU -> Polynomial -> Kalman"
UAVSAT_DEVICE="${DEVICE}" JITTER_M="${JITTER_M}" \
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS}" PATIENCE="${PATIENCE}" \
  bash "${ROOT}/run_mobilenetv3_weighted_train_eval.sh"

echo "[2/2] MobileNetV3 + END_MS: reference-centered full 6x6 -> Soft MeanShift"
UAVSAT_DEVICE="${DEVICE}" JITTER_M="${JITTER_M}" \
  bash "${ROOT}/run_mobilenetv3_end_ms_eval.sh"

echo "[DONE]"
echo "baseline: ${ROOT}/output/mobilenetv3_weighted/robust_tracker_summary.json"
echo "END_MS:   ${ROOT}/output/mobilenetv3_END_MS/robust_tracker_summary.json"
