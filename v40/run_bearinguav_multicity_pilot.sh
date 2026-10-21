#!/usr/bin/env bash
set -euo pipefail
cd /yh/study/uav-sat
export UAVSAT_BACKBONE=mobilenet_v3_small
export UAVSAT_REFERENCE_PROTOCOL=route_reference
export UAVSAT_EXPERIMENT_FRAME_COUNT=3
export UAVSAT_EXPERIMENT_FORWARD_ONLY=1
export UAVSAT_ARCHITECTURE_NAME=ICLR_Bearing4City_Forward18SoftMS_ResidualTemporalGRU_Kalman_FinalMS
export MS_REFERENCE_PRIOR_WEIGHT=0
export MS_ENABLED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

read -r -a train_cities <<< "${V40_TRAIN_CITIES:-citya cityb}"
read -r -a eval_cities <<< "${V40_EVAL_CITIES:-citya cityb}"
supervision=${V40_SUPERVISION:-image_only}
if [[ "$supervision" != image_only && "$supervision" != gt_train ]]; then
  echo "V40_SUPERVISION must be image_only or gt_train" >&2
  exit 2
fi
out=${1:-/yh/study/uav-sat/v40/output/multicity_${supervision}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$out"
echo "[V40] train=${train_cities[*]} eval=${eval_cities[*]} supervision=$supervision output=$out"
CUDA_VISIBLE_DEVICES=${V40_GPU:-0} python3 -u v40/train_multicity_routes.py \
  --train-cities "${train_cities[@]}" \
  --eval-cities "${eval_cities[@]}" \
  --supervision "$supervision" \
  --init "${V40_INIT:-shared}" \
  --epochs "${V40_EPOCHS:-2}" \
  --max-train-frames "${V40_MAX_TRAIN_FRAMES:-136}" \
  --lr "${V40_LR:-0.00002}" \
  --teacher-acquisition-ratio "${V40_TEACHER_ACQ:-0}" \
  --teacher-center-ratio "${V40_TEACHER_CENTER:-0}" \
  --teacher-state-ratio "${V40_TEACHER_STATE:-0}" \
  --acquisition-loss-weight "${V40_ACQ_LOSS_WEIGHT:-1}" \
  --measurement-loss-weight "${V40_MEAS_LOSS_WEIGHT:-2}" \
  --motion-loss-weight "${V40_MOTION_LOSS_WEIGHT:-2}" \
  --output "$out" \
  2>&1 | tee "$out/run.log"
