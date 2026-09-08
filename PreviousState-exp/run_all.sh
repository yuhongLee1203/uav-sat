#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEVICE="${UAVSAT_DEVICE:-cuda:0}"
JITTER_M="${JITTER_M:-8}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"

echo "[1/3] Current MobileCLIP2-S2 Previous-State baseline"
UAVSAT_DEVICE="${DEVICE}" JITTER_M="${JITTER_M}" \
  bash "${ROOT}/run_mobileclip_current_eval.sh"

echo "[2/3] MobileNetV3 Previous-State train + eval"
UAVSAT_DEVICE="${DEVICE}" JITTER_M="${JITTER_M}" \
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS}" PATIENCE="${PATIENCE}" \
  bash "${ROOT}/run_mobilenetv3_train_eval.sh"

echo "[3/3] MobileNetV3 + post-Kalman full 6x6 SoftMS eval"
UAVSAT_DEVICE="${DEVICE}" JITTER_M="${JITTER_M}" \
  bash "${ROOT}/run_mobilenetv3_postkalman6x6_eval.sh"

echo "[DONE] outputs under ${ROOT}/output/"
