#!/usr/bin/env bash
set -euo pipefail

cd /yh/study/uav-sat/v37
output=/yh/study/uav-sat/v37/outputs/trainA_valC_testB_4x6_mobilenetv3small
mkdir -p "$output/logs"

python3 scripts/build_scheduled_references.py

CUDA_VISIBLE_DEVICES=6 \
UAVSAT_BACKBONE=mobilenet_v3_small \
UAVSAT_REFERENCE_PROTOCOL=scheduled_route_reference \
UAVSAT_TRAIN_ROUTES=route_A \
UAVSAT_VALIDATION_ROUTES=route_C \
UAVSAT_EVAL_ROUTES=route_B \
UAVSAT_OUTPUT_DIR="$output" \
python3 -u robust_tracker.py \
  --mode train_eval \
  --visual-epochs 30 \
  --temporal-epochs 110 \
  --jitter-m 0 \
  2>&1 | tee "$output/logs/trainA_valC_testB_gpu6.log"

CUDA_VISIBLE_DEVICES=6 \
UAVSAT_BACKBONE=mobilenet_v3_small \
UAVSAT_REFERENCE_PROTOCOL=scheduled_route_reference \
UAVSAT_TRAIN_ROUTES=route_A \
UAVSAT_VALIDATION_ROUTES=route_C \
UAVSAT_EVAL_ROUTES=route_B \
UAVSAT_OUTPUT_DIR="$output" \
python3 -u render_results_video.py --route route_B \
  2>&1 | tee "$output/logs/render_routeB_gpu6.log"
