#!/usr/bin/env bash
# Re-evaluate the supplied, unmodified forNX V36 checkpoints on Route B+C.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORNX="${ROOT}/../forNX"
BACKBONES="${BACKBONES:-mobileclip2_s2 resnet18 mobilenet_v3_small vgg16}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
WARMUP="${WARMUP:-100}"
OUT_ROOT="${UAVSAT_OUTPUT_ROOT:-${ROOT}/output/original_fornx_benchmark}"

[[ -f "${FORNX}/src/robust_tracker.py" ]] || { echo "Missing forNX source: ${FORNX}" >&2; exit 2; }
mkdir -p "${OUT_ROOT}"
export TORCH_HOME="${FORNX}/pretrained_cache/torch"
export HF_HOME="${FORNX}/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

for backbone in ${BACKBONES}; do
  case "${backbone}" in mobileclip2_s2|resnet18|mobilenet_v3_small|vgg16) ;; *)
    echo "Unsupported packaged backbone: ${backbone}" >&2; exit 2;; esac
  ckpt="${FORNX}/weights/v36_${backbone}/checkpoints"
  [[ -s "${ckpt}/visual_retrieval_A_only.pt" && -s "${ckpt}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt" ]] || {
    echo "Missing packaged checkpoint for ${backbone}" >&2; exit 2; }
  architecture="$(python3 - "${ckpt}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt" <<'PY'
import sys
import torch
print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["architecture"])
PY
)"
  out="${OUT_ROOT}/v36_${backbone}"
  mkdir -p "${out}"
  echo "=== original forNX V36 ${backbone}: Route B+C eval only ==="
  (
    cd "${FORNX}/src"
    UAVSAT_DEVICE="${DEVICE}" UAVSAT_OUTPUT_DIR="${out}" UAVSAT_CHECKPOINT_DIR="${ckpt}" \
    UAVSAT_DATA_ROOT="${FORNX}/data" UAVSAT_BACKBONE="${backbone}" UAVSAT_ARCHITECTURE_NAME="${architecture}" \
    UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter UAVSAT_EXPERIMENT_ANCHOR=softms \
    UAVSAT_EXPERIMENT_FRAME_COUNT=3 UAVSAT_EXPERIMENT_MOTION=quadratic \
    UAVSAT_EXPERIMENT_KALMAN=learned UAVSAT_EXPERIMENT_DISABLE_GRU=0 \
    UAVSAT_EXPERIMENT_FORWARD_ONLY=1 UAVSAT_MEASURE_LATENCY=1 \
    UAVSAT_LATENCY_WARMUP="${WARMUP}" \
    python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m 8
  ) 2>&1 | tee "${out}/benchmark.log"
done

echo "[DONE] original forNX outputs: ${OUT_ROOT}"
