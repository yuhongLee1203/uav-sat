#!/usr/bin/env bash
set -euo pipefail
cd /yh/study/uav-sat

# v39 model, with a MobileNetV3-Small backbone. GT is used only by losses;
# neither stage selects a forward-pass satellite patch from frame GT.
export UAVSAT_BACKBONE=mobilenet_v3_small
export UAVSAT_REFERENCE_PROTOCOL=route_reference
export UAVSAT_EXPERIMENT_FRAME_COUNT=3
export UAVSAT_EXPERIMENT_FORWARD_ONLY=1
export UAVSAT_ARCHITECTURE_NAME=ICLR_Bearing4City_Forward18SoftMS_ResidualTemporalGRU_Kalman_FinalMS
export MS_REFERENCE_PRIOR_WEIGHT=0
export MS_ENABLED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export V40_ACQ_HYPOTHESES=${V40_ACQ_HYPOTHESES:-13}
export V40_ACQ_BANK_RADIUS_M=${V40_ACQ_BANK_RADIUS_M:-60}
export V40_ACQ_RAW_VISUAL_WEIGHT=${V40_ACQ_RAW_VISUAL_WEIGHT:-2}
export V40_ACQ_MOTION_PRIOR_WEIGHT=${V40_ACQ_MOTION_PRIOR_WEIGHT:-0.2}
export V40_NO_GT_MAX_BACKWARD_CORRECTION_M=${V40_NO_GT_MAX_BACKWARD_CORRECTION_M:-5}
export UAVSAT_KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M=${UAVSAT_KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M:-12}
export UAVSAT_KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M=${UAVSAT_KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M:-8}

out=${1:-/yh/study/uav-sat/v40/output/strict_gtloss_citya_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$out"
CUDA_VISIBLE_DEVICES=${V40_VISUAL_GPU:-0} python3 -u v40/train_visual_global_gtloss.py \
  --city citya \
  --samples "${V40_VISUAL_SAMPLES:-20000}" \
  --epochs "${V40_VISUAL_EPOCHS:-15}" \
  --batch-size "${V40_VISUAL_BATCH_SIZE:-128}" \
  --cache-root /yh/study/uav-sat/v40/cache/visual90k \
  --output "$out/visual" 2>&1 | tee "$out/visual.log"

export V40_VISUAL_CHECKPOINT_ROOT="$out/visual/checkpoints"
dense_loss_args=()
if [[ ${V40_DENSE_OFFBANK_LOSS:-0} == 1 ]]; then
  dense_loss_args+=(--dense-offbank-loss)
fi
CUDA_VISIBLE_DEVICES=${V40_TEMPORAL_GPU:-5} python3 -u v40/train_multicity_routes.py \
  --train-cities citya --eval-cities citya \
  --supervision gt_train --init random_temporal \
  --epochs "${V40_TEMPORAL_EPOCHS:-2}" \
  --max-train-frames "${V40_MAX_TRAIN_FRAMES:-136}" \
  --lr "${V40_TEMPORAL_LR:-0.00002}" \
  --teacher-acquisition-ratio 0 \
  --teacher-center-ratio 0 \
  --teacher-state-ratio 0 \
  "${dense_loss_args[@]}" \
  --output "$out/temporal" 2>&1 | tee "$out/temporal.log"
echo "[V40 STRICT REPORT] $out/temporal/report.md"
