#!/usr/bin/env bash
set -euo pipefail

cd /yh/study/uav-sat/v37
mkdir -p outputs/bc_train_a_validation_4x6/logs

python3 scripts/build_scheduled_references.py

CUDA_VISIBLE_DEVICES=0 \
UAVSAT_BACKBONE=mobilenet_v3_small \
UAVSAT_REFERENCE_PROTOCOL=scheduled_route_reference \
UAVSAT_EVAL_ROUTES=route_A \
UAVSAT_OUTPUT_DIR=/yh/study/uav-sat/v37/outputs/bc_train_a_validation_4x6 \
python3 -u robust_tracker.py \
  --mode train_eval \
  --visual-epochs 30 \
  --temporal-epochs 220 \
  --jitter-m 0 \
  2>&1 | tee outputs/bc_train_a_validation_4x6/logs/train_eval_gpu0.log
