#!/usr/bin/env bash
# Compare backbones under the unchanged full V36 protocol.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULT_ROOT="${SCRIPT_DIR}/outputs"
BASE_OUTPUT="${ROOT_DIR}/outputs/v36_v34protocol_compact_gru_softms_mode_variance_forward3x6_polynomial_kalman"

GPUS="${GPUS:-0 3 6}"
# First three run concurrently; remaining models start on the GPU that owns their queue.
BACKBONES="${BACKBONES:-mobileclip2_s2 resnet18 mobilenet_v3_small resnet50 vgg16}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
JITTER_M="${JITTER_M:-8}"
WARMUP="${WARMUP:-30}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-64}"
RESUME="${RESUME:-0}"
FORCE="${FORCE:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash backbone-exp/run_v36_backbone_experiments_gpu036.sh

The default runs MobileCLIP2-S2 (the existing full-V36 checkpoint), ResNet-18,
MobileNetV3-Small, ResNet-50 and VGG-16. GPU 0, 3 and 6 each run one queue.

Useful options:
  RESUME=1  bash backbone-exp/run_v36_backbone_experiments_gpu036.sh
  BACKBONES="resnet18 resnet50" GPUS="0 3" bash backbone-exp/run_v36_backbone_experiments_gpu036.sh
  VISUAL_EPOCHS=30 TEMPORAL_EPOCHS=60 PATIENCE=10 bash backbone-exp/run_v36_backbone_experiments_gpu036.sh

Outputs: backbone-exp/outputs/v36_<backbone>/
Table:   backbone-exp/outputs/v36_backbone_comparison.md
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
[[ -z "${1:-}" ]] || { usage >&2; exit 2; }

read -r -a GPU_ARRAY <<< "${GPUS}"
read -r -a BACKBONE_ARRAY <<< "${BACKBONES}"
(( ${#GPU_ARRAY[@]} > 0 && ${#BACKBONE_ARRAY[@]} > 0 )) || { echo "ERROR: empty GPUS/BACKBONES" >&2; exit 2; }

BASE_VISUAL="${BASE_OUTPUT}/checkpoints/visual_retrieval_A_only.pt"
BASE_TEMPORAL="${BASE_OUTPUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
[[ -s "${BASE_VISUAL}" && -s "${BASE_TEMPORAL}" ]] || {
  echo "ERROR: Missing current V36 checkpoints in ${BASE_OUTPUT}/checkpoints" >&2; exit 2;
}
mkdir -p "${RESULT_ROOT}" "${SCRIPT_DIR}/logs"

run_one() {
  local backbone="$1" gpu="$2"
  local output="${RESULT_ROOT}/v36_${backbone}"
  local ckpt="${output}/checkpoints"
  local log="${SCRIPT_DIR}/logs/v36_${backbone}.log"
  mkdir -p "${ckpt}" "${output}/feature_cache"
  if [[ -s "${output}/robust_tracker_summary.json" && "${FORCE}" != "1" ]]; then
    echo "[$(date -Is)] SKIP completed V36 ${backbone}" | tee -a "${log}"
    return
  fi
  (
    cd "${ROOT_DIR}"
    echo "[$(date -Is)] V36 backbone=${backbone}, physical GPU=${gpu}"
    echo "fixed: SoftMS + 3-frame GRU + quadratic motion + learned external Kalman + forward 3x6"
    common_env=(
      "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONUNBUFFERED=1" "TOKENIZERS_PARALLELISM=false"
      "OMP_NUM_THREADS=2" "MKL_NUM_THREADS=2" "OPENBLAS_NUM_THREADS=2" "NUMEXPR_NUM_THREADS=2"
      # The temporal checkpoint records the architecture name.  Keep the exact
      # V36 name for every backbone so MobileCLIP can load the supplied V36
      # baseline checkpoint; output directory/backbone key identify each run.
      "UAVSAT_OUTPUT_DIR=${output}" "UAVSAT_ARCHITECTURE_NAME=V34ProtocolCompactGRUSoftMSModeVarianceForward3x6PolynomialKalman_v36"
      "UAVSAT_BACKBONE=${backbone}" "UAVSAT_FEATURE_CACHE_DIR=${output}/feature_cache"
      "UAVSAT_EXPERIMENT_VARIANT=full_v36_backbone_${backbone}"
      "UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter"
      "UAVSAT_EXPERIMENT_ANCHOR=softms" "UAVSAT_EXPERIMENT_FRAME_COUNT=3"
      "UAVSAT_EXPERIMENT_MOTION=quadratic" "UAVSAT_EXPERIMENT_KALMAN=learned"
      "UAVSAT_EXPERIMENT_DISABLE_GRU=0" "UAVSAT_EXPERIMENT_FORWARD_ONLY=1"
      "UAVSAT_MEASURE_LATENCY=1" "UAVSAT_LATENCY_WARMUP=${WARMUP}"
      "UAVSAT_VISUAL_CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE}"
    )
    if [[ "${backbone}" == "mobileclip2_s2" ]]; then
      cp -p "${BASE_VISUAL}" "${ckpt}/visual_retrieval_A_only.pt"
      cp -p "${BASE_TEMPORAL}" "${ckpt}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
      env "${common_env[@]}" python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m "${JITTER_M}"
    else
      resume_args=()
      if [[ "${RESUME}" == "1" ]]; then
        [[ -s "${ckpt}/visual_retrieval_A_only.pt" ]] && resume_args+=(--resume-visual)
        [[ -s "${ckpt}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only_latest.pt" ]] && resume_args+=(--resume-temporal)
      fi
      env "${common_env[@]}" python3 -u robust_tracker.py --mode train_eval \
        --visual-epochs "${VISUAL_EPOCHS}" --temporal-epochs "${TEMPORAL_EPOCHS}" \
        --patience "${PATIENCE}" --jitter-m "${JITTER_M}" "${resume_args[@]}"
    fi
    test -s "${output}/robust_tracker_summary.json"
  ) 2>&1 | tee "${log}"
}

worker() {
  local gpu="$1"; shift
  local backbone
  for backbone in "$@"; do run_one "${backbone}" "${gpu}"; done
}

declare -a QUEUES
for ((i=0; i<${#GPU_ARRAY[@]}; i++)); do QUEUES[i]=""; done
for ((i=0; i<${#BACKBONE_ARRAY[@]}; i++)); do
  slot=$((i % ${#GPU_ARRAY[@]}))
  QUEUES[slot]="${QUEUES[slot]} ${BACKBONE_ARRAY[i]}"
done

echo "V36 backbone comparison: GPUs=${GPUS}; backbones=${BACKBONES}"
pids=()
for ((i=0; i<${#GPU_ARRAY[@]}; i++)); do
  read -r -a queue <<< "${QUEUES[i]}"
  (( ${#queue[@]} )) || continue
  worker "${GPU_ARRAY[i]}" "${queue[@]}" & pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  # Always reap every worker.  This prevents a failed GPU-0 baseline from
  # leaving training processes on the other GPUs detached in the background.
  if ! wait "${pid}"; then failed=1; fi
done

python3 "${SCRIPT_DIR}/collect_v36_backbone_results.py" --output-root "${RESULT_ROOT}" --backbones "${BACKBONE_ARRAY[@]}"
(( failed == 0 )) || {
  echo "ERROR: one or more backbone jobs failed; completed rows were still collected." >&2
  exit 1
}
