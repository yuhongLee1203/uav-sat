#!/usr/bin/env bash
set -euo pipefail

cd /yh/study/uav-sat
export V40_BASE_CITYA=/yh/study/uav-sat/v39_otherdata/output/abcd_table6_official96_temporal_20260923/citya
export V40_PREPARED_ROOT=/yh/study/uav-sat/v40/data/official96_fixed_routes
export UAVSAT_BACKBONE=mobilenet_v3_small
export UAVSAT_ARCHITECTURE_NAME=ICLR_Bearing4City_Forward18SoftMS_ResidualTemporalGRU_Kalman_FinalMS
export CUDA_VISIBLE_DEVICES="${CITYB_GPU:-0}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

if [[ -n "${WAIT_FOR_PID:-}" ]]; then
    while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do sleep 15; done
fi

out="${CITYB_OUTPUT_ROOT:-/yh/study/uav-sat/v40/output/official96_fixed_cityb_global18k_20260925}"
mkdir -p "$out"
python3 -u v40/train_visual_global_gtloss.py \
  --city cityb --samples 18000 --epochs 6 --batch-size 128 \
  --lr 0.0003 --target-sigma-m 4 --distance-loss-weight 5 \
  --official96-train-only --rebuild-satellite-gallery \
  --output "$out" 2>&1 | tee "$out/run.log"
