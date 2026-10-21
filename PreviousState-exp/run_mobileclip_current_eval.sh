#!/usr/bin/env bash
set -Eeuo pipefail

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"
BASE_SRC="${REPO_ROOT}/v36_GvsK/previous_state_only"
SRC="${EXP_ROOT}/mobileclip_current/src"
OLD_OUT="${REPO_ROOT}/v36_GvsK/output/previous_state_only"
OUT="${UAVSAT_OUTPUT_DIR:-${EXP_ROOT}/output/mobileclip_current}"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR:-${REPO_ROOT}/v36_GvsK/output/meanshift_gru/feature_cache}"
ARCH="V36_NoSatContext_PreviousStateOnly_Forward3x6_PolynomialKalman"
JITTER_M="${JITTER_M:-8}"

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -f "${BASE_SRC}/${f}" ]] || { echo "ERROR: missing ${BASE_SRC}/${f}" >&2; exit 2; }
done
VISUAL_CKPT="${OLD_OUT}/checkpoints/visual_retrieval_A_only.pt"
TEMPORAL_CKPT="${OLD_OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
[[ -s "${VISUAL_CKPT}" ]] || { echo "ERROR: missing current visual checkpoint ${VISUAL_CKPT}" >&2; exit 2; }
[[ -s "${TEMPORAL_CKPT}" ]] || { echo "ERROR: missing current 3.9m temporal checkpoint ${TEMPORAL_CKPT}" >&2; exit 2; }

rm -rf "${SRC}"
mkdir -p "${SRC}" "${OUT}/checkpoints"
cp -a "${BASE_SRC}/." "${SRC}/"
ln -sfn "${VISUAL_CKPT}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"
ln -sfn "${TEMPORAL_CKPT}" "${OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

export TORCH_HOME="${REPO_ROOT}/v36_GvsK/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/v36_GvsK/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

(
  cd "${SRC}"
  UAVSAT_DEVICE="${DEVICE}" \
  UAVSAT_OUTPUT_DIR="${OUT}" \
  UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
  UAVSAT_FEATURE_CACHE_DIR="${CACHE_DIR}" \
  UAVSAT_DATA_ROOT="${DATA_ROOT}" \
  UAVSAT_BACKBONE=mobileclip2_s2 \
  UAVSAT_ARCHITECTURE_NAME="${ARCH}" \
  UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
  UAVSAT_EXPERIMENT_ANCHOR=softms \
  UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
  UAVSAT_EXPERIMENT_MOTION=quadratic \
  UAVSAT_EXPERIMENT_KALMAN=learned \
  UAVSAT_EXPERIMENT_DISABLE_GRU=0 \
  UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
  python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m "${JITTER_M}"
) 2>&1 | tee "${OUT}/eval.log"

echo "[DONE] current MobileCLIP Previous-State result: ${OUT}/robust_tracker_summary.json"
