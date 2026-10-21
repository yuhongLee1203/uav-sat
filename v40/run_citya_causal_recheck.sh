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
OUT=${1:-/yh/study/uav-sat/v40/output/causal_recheck_$(date +%Y%m%d_%H%M%S)}
CUDA_VISIBLE_DEVICES=${V40_GPU:-0} python3 -u v40/evaluate_causal_citya.py --output "$OUT"
echo "[V40 REPORT] $OUT/report.md"
