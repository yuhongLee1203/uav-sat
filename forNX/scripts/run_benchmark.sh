#!/usr/bin/env bash
# Run the exact pretrained V36 backbone variants on packaged Route B+C data.
set -Eeuo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKBONES="${BACKBONES:-mobileclip2_s2 resnet18 mobilenet_v3_small vgg16}"
WARMUP="${WARMUP:-100}"
DEVICE="${UAVSAT_DEVICE:-cuda}"

export TORCH_HOME="${PACKAGE_ROOT}/pretrained_cache/torch"
export HF_HOME="${PACKAGE_ROOT}/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

for backbone in ${BACKBONES}; do
  case "${backbone}" in
    mobileclip2_s2|resnet18|mobilenet_v3_small|vgg16) ;;
    *) echo "Unsupported or untrained backbone: ${backbone}" >&2; exit 2 ;;
  esac
  checkpoint_dir="${PACKAGE_ROOT}/weights/v36_${backbone}/checkpoints"
  temporal="${checkpoint_dir}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  visual="${checkpoint_dir}/visual_retrieval_A_only.pt"
  [[ -s "${temporal}" && -s "${visual}" ]] || { echo "Missing checkpoint for ${backbone}" >&2; exit 2; }
  architecture="$(python3 - "${temporal}" <<'PY'
import sys
import torch
print(torch.load(sys.argv[1], map_location="cpu")["architecture"])
PY
)"
  output_dir="${PACKAGE_ROOT}/runs/v36_${backbone}"
  mkdir -p "${output_dir}"
  echo "=== V36 ${backbone} (${architecture}) ==="
  (
    cd "${PACKAGE_ROOT}/src"
    env \
      UAVSAT_DEVICE="${DEVICE}" \
      UAVSAT_OUTPUT_DIR="${output_dir}" \
      UAVSAT_CHECKPOINT_DIR="${checkpoint_dir}" \
      UAVSAT_DATA_ROOT="${PACKAGE_ROOT}/data" \
      UAVSAT_ARCHITECTURE_NAME="${architecture}" \
      UAVSAT_BACKBONE="${backbone}" \
      UAVSAT_EXPERIMENT_VARIANT="v36_nx_${backbone}" \
      UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
      UAVSAT_EXPERIMENT_ANCHOR=softms \
      UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
      UAVSAT_EXPERIMENT_MOTION=quadratic \
      UAVSAT_EXPERIMENT_KALMAN=learned \
      UAVSAT_EXPERIMENT_DISABLE_GRU=0 \
      UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
      UAVSAT_MEASURE_LATENCY=1 \
      UAVSAT_LATENCY_WARMUP="${WARMUP}" \
      python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m 8
  ) 2>&1 | tee "${output_dir}/benchmark.log"
done

python3 "${PACKAGE_ROOT}/scripts/collect_v36_backbone_results.py" \
  --output-root "${PACKAGE_ROOT}/runs" --backbones ${BACKBONES}
