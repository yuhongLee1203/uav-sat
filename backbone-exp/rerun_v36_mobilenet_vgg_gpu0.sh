#!/usr/bin/env bash
# Re-measure two already trained V36 backbones on the SAME physical GPU.
# This never trains or replaces a checkpoint.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULT_ROOT="${SCRIPT_DIR}/outputs"
GPU_ID="${GPU_ID:-0}"
BACKBONES="${BACKBONES:-mobilenet_v3_small vgg16}"
WARMUP="${WARMUP:-100}"
JITTER_M="${JITTER_M:-8}"

read -r -a BACKBONE_ARRAY <<< "${BACKBONES}"
(( ${#BACKBONE_ARRAY[@]} > 0 )) || { echo "ERROR: BACKBONES is empty" >&2; exit 2; }

for backbone in "${BACKBONE_ARRAY[@]}"; do
  output="${RESULT_ROOT}/v36_${backbone}"
  temporal="${output}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  visual="${output}/checkpoints/visual_retrieval_A_only.pt"
  [[ -s "${temporal}" && -s "${visual}" ]] || {
    echo "ERROR: missing trained V36 checkpoint for ${backbone} under ${output}" >&2
    exit 2
  }

  # Earlier benchmark jobs recorded a per-backbone architecture name. Read it
  # from the actual checkpoint, rather than guessing, so this is evaluation-only.
  architecture="$(python3 - "${temporal}" <<'PY'
import sys
import torch
payload = torch.load(sys.argv[1], map_location="cpu")
name = payload.get("architecture")
if not name:
    raise SystemExit("temporal checkpoint has no architecture field")
print(name)
PY
)"

  echo "[$(date -Is)] Re-measure V36 ${backbone} on physical GPU ${GPU_ID}"
  echo "checkpoint architecture: ${architecture}"
  (
    cd "${ROOT_DIR}"
    env \
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
      UAVSAT_OUTPUT_DIR="${output}" \
      UAVSAT_ARCHITECTURE_NAME="${architecture}" \
      UAVSAT_BACKBONE="${backbone}" \
      UAVSAT_FEATURE_CACHE_DIR="${output}/feature_cache" \
      UAVSAT_EXPERIMENT_VARIANT="full_v36_backbone_${backbone}_gpu${GPU_ID}_remeasure" \
      UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
      UAVSAT_EXPERIMENT_ANCHOR=softms UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
      UAVSAT_EXPERIMENT_MOTION=quadratic UAVSAT_EXPERIMENT_KALMAN=learned \
      UAVSAT_EXPERIMENT_DISABLE_GRU=0 UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
      UAVSAT_MEASURE_LATENCY=1 UAVSAT_LATENCY_WARMUP="${WARMUP}" \
      UAVSAT_VISUAL_CACHE_BATCH_SIZE=64 \
      python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m "${JITTER_M}"
  ) 2>&1 | tee "${SCRIPT_DIR}/logs/v36_${backbone}_gpu${GPU_ID}_remeasure.log"
done

python3 "${SCRIPT_DIR}/collect_v36_backbone_results.py" \
  --output-root "${RESULT_ROOT}" \
  --backbones mobileclip2_s2 resnet18 mobilenet_v3_small vgg16
