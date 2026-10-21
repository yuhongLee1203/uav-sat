#!/usr/bin/env bash
set -euo pipefail

cd /yh/study/uav-sat
ROOT=/yh/study/uav-sat/v40
BASE=${V40_BASE_CITYA:-/yh/study/uav-sat/v39_otherdata/output/abcd_shared_20260922_121010/citya}
export V40_BASE_CITYA="$BASE"
export UAVSAT_BACKBONE=mobilenet_v3_small
export UAVSAT_REFERENCE_PROTOCOL=route_reference
export UAVSAT_EXPERIMENT_FRAME_COUNT=3
export UAVSAT_EXPERIMENT_FORWARD_ONLY=1
export UAVSAT_ARCHITECTURE_NAME=ICLR_Bearing4City_Forward18SoftMS_ResidualTemporalGRU_Kalman_FinalMS
export MS_REFERENCE_PRIOR_WEIGHT=0
export MS_ENABLED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

RUN=${1:-$ROOT/output/citya_image_only_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RUN"
if [[ ! -f "$BASE/train_frames3/checkpoints/visual_retrieval_A_only.pt" ]]; then
  echo "Missing City A visual checkpoint under $BASE" >&2
  exit 1
fi
if [[ ! -f "$BASE/train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt" ]]; then
  echo "Missing City A temporal checkpoint under $BASE" >&2
  exit 1
fi
echo "[V40] output=$RUN"
declare -a pids=()
for spec in visual_pseudo:0 temporal_cycle:5 confidence_gated:6; do
  design=${spec%%:*}
  gpu=${spec##*:}
  mkdir -p "$RUN/$design"
  CUDA_VISIBLE_DEVICES="$gpu" python3 -u "$ROOT/image_only_citya.py" \
    --design "$design" --output "$RUN/$design" \
    --epochs "${V40_EPOCHS:-2}" --max-train-frames "${V40_MAX_TRAIN_FRAMES:-136}" \
    >"$RUN/$design/train.log" 2>&1 &
  pids+=("$!")
  echo "[V40] $design on GPU $gpu pid=${pids[-1]}"
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" != 0 ]]; then
  echo "[V40] one or more designs failed; inspect $RUN/*/train.log" >&2
  exit 1
fi
python3 "$ROOT/summarize.py" "$RUN"
echo "[V40 DONE] $RUN"
