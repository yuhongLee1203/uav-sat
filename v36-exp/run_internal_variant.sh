#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXP_DIR="${ROOT_DIR}/v36-exp"
RUNNER_ROOT="${V36_EXP_RUNNER_ROOT:-${ROOT_DIR}}"
VARIANT="${1:?usage: run_internal_variant.sh VARIANT GPU}"
GPU_ID="${2:?usage: run_internal_variant.sh VARIANT GPU}"
BASE_OUTPUT="${ROOT_DIR}/outputs/v36_v34protocol_compact_gru_softms_mode_variance_forward3x6_polynomial_kalman"
OUTPUT_ROOT="${V36_EXP_OUTPUT_ROOT:-${EXP_DIR}/outputs/internal/corrected_v2}"
OUTPUT_DIR="${OUTPUT_ROOT}/${VARIANT}"
SHARED_CACHE="${EXP_DIR}/cache/mobileclip2_s2"
EPOCHS="${V36_EXP_EPOCHS:-60}"
PATIENCE="${V36_EXP_PATIENCE:-10}"

mkdir -p "${OUTPUT_DIR}/checkpoints" "${SHARED_CACHE}" "${EXP_DIR}/logs"
cp -p "${BASE_OUTPUT}/checkpoints/visual_retrieval_A_only.pt" "${OUTPUT_DIR}/checkpoints/visual_retrieval_A_only.pt"

anchor=softms
frames=3
motion=quadratic
kalman=learned
disable_gru=0
forward_only=1
forward_backshift_m=0.0

case "${VARIANT}" in
  full_v36) ;;
  weighted_centroid) anchor=weighted_centroid ;;
  full_6x6) forward_only=0 ;;
  forward_3x6_aligned) forward_backshift_m=4.75 ;;
  frame1) frames=1 ;;
  frame2) frames=2 ;;
  softms_only) disable_gru=1; motion=none; kalman=none ;;
  softms_gru) motion=none; kalman=none ;;
  softms_gru_poly|no_kalman) kalman=none ;;
  motion_kalman_cv) motion=none ;;
  motion_velocity) motion=velocity ;;
  *) echo "Unknown internal variant: ${VARIANT}" >&2; exit 2 ;;
esac

if [[ -s "${OUTPUT_DIR}/robust_tracker_summary.json" && "${V36_EXP_FORCE:-0}" != "1" ]]; then
  echo "SKIP completed ${VARIANT}"
  exit 0
fi

mode=train_eval
architecture="V36Exp_${VARIANT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${V36_EXP_CPU_THREADS:-2}"
export MKL_NUM_THREADS="${V36_EXP_CPU_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${V36_EXP_CPU_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${V36_EXP_CPU_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
export UAVSAT_OUTPUT_DIR="${OUTPUT_DIR}"
export UAVSAT_ARCHITECTURE_NAME="${architecture}"
export UAVSAT_FEATURE_CACHE_DIR="${SHARED_CACHE}"
export UAVSAT_EXPERIMENT_VARIANT="${VARIANT}"
export UAVSAT_EXPERIMENT_ANCHOR="${anchor}"
export UAVSAT_EXPERIMENT_FRAME_COUNT="${frames}"
export UAVSAT_EXPERIMENT_MOTION="${motion}"
export UAVSAT_EXPERIMENT_KALMAN="${kalman}"
export UAVSAT_EXPERIMENT_DISABLE_GRU="${disable_gru}"
export UAVSAT_EXPERIMENT_FORWARD_ONLY="${forward_only}"
export UAVSAT_FORWARD_ORIGIN_BACKSHIFT_M="${forward_backshift_m}"
export UAVSAT_MEASURE_LATENCY=1
export UAVSAT_LATENCY_WARMUP=30

cd "${RUNNER_ROOT}"
python3 -u "${RUNNER_ROOT}/robust_tracker.py" \
  --mode "${mode}" \
  --reuse-visual \
  --temporal-epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --jitter-m 8
