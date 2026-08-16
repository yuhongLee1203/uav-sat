#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/scheduled_route_single3x6_gpu0"
LOG_DIR="${SCRIPT_DIR}/logs"
BASE_OUTPUT="${ROOT_DIR}/outputs/v36_v34protocol_compact_gru_softms_mode_variance_forward3x6_polynomial_kalman"
SHARED_CACHE="${ROOT_DIR}/v36-exp/cache/mobileclip2_s2"
REFERENCE_DIR="${SCRIPT_DIR}/references"

mkdir -p "${OUTPUT_DIR}/checkpoints" "${LOG_DIR}" "${SHARED_CACHE}" "${REFERENCE_DIR}"
bash "${ROOT_DIR}/v36-exp/prepare_shared_cache.sh"
cp -p "${BASE_OUTPUT}/checkpoints/visual_retrieval_A_only.pt" \
  "${OUTPUT_DIR}/checkpoints/visual_retrieval_A_only.pt"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS="${FRAME_REF_CPU_THREADS:-2}"
export MKL_NUM_THREADS="${FRAME_REF_CPU_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${FRAME_REF_CPU_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${FRAME_REF_CPU_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
export UAVSAT_OUTPUT_DIR="${OUTPUT_DIR}"
export UAVSAT_ARCHITECTURE_NAME="ScheduledRouteReferenceSingle3x6GRUNextPositionSoftMSKalman_v5"
export UAVSAT_FEATURE_CACHE_DIR="${SHARED_CACHE}"
export UAVSAT_DENSE_ROUTE_REFERENCE_DIR="${REFERENCE_DIR}"
export UAVSAT_EXPERIMENT_VARIANT="scheduled_route_reference_next_position"
export UAVSAT_REFERENCE_PROTOCOL="scheduled_route_reference"
export UAVSAT_EXPERIMENT_ANCHOR="softms"
export UAVSAT_EXPERIMENT_FRAME_COUNT=3
export UAVSAT_EXPERIMENT_MOTION="quadratic"
export UAVSAT_EXPERIMENT_KALMAN="learned"
export UAVSAT_EXPERIMENT_DISABLE_GRU=0
export UAVSAT_EXPERIMENT_FORWARD_ONLY=1
# Restore the trainable frame-reference setup: exactly one forward 3x6 window.
# Only its centre changes, from online current-frame GT to the fixed dense
# frame-route reference manifest. No window bank or raw selector.
export UAVSAT_FORWARD_ORIGIN_BACKSHIFT_M=0.0
export UAVSAT_ROUTE_REFERENCE_HYPOTHESES=1
export UAVSAT_ACQ_RAW_VISUAL_EVIDENCE_WEIGHT=0.0
export UAVSAT_MEASURE_LATENCY=1
export UAVSAT_LATENCY_WARMUP=30

cd "${ROOT_DIR}"
python3 -u frame-reference-exp/build_dense_route_references.py \
  --cache-dir "${SHARED_CACHE}" \
  --visual-checkpoint "${OUTPUT_DIR}/checkpoints/visual_retrieval_A_only.pt" \
  --output-dir "${REFERENCE_DIR}"

python3 -u robust_tracker.py \
  --mode train_eval \
  --reuse-visual \
  --temporal-epochs "${FRAME_REF_EPOCHS:-60}" \
  --patience "${FRAME_REF_PATIENCE:-10}" \
  --jitter-m 0 \
  2>&1 | tee "${LOG_DIR}/scheduled_route_single3x6_gpu0.log"

echo "Result: ${OUTPUT_DIR}/robust_tracker_summary.json"
