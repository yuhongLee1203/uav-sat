#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OUTPUT_DIR="outputs/rnn_kalman_train_A_test_BC"
TARGET_VISUAL="${OUTPUT_DIR}/checkpoints/visual_retrieval_A_only.pt"

# Search old/archived Route-A-only visual checkpoints.
VISUAL_CANDIDATES=(
  "archive/rtl_crf_20260809/outputs/strict_train_A_test_BC_t2only_w5/checkpoints/visual_retrieval_A_only.pt"
  "archive/rtl_crf_20260809/outputs/strict_train_A_test_BC_no_position_scale/checkpoints/visual_retrieval_A_only.pt"
  "outputs/strict_train_A_test_BC_t2only_w5/checkpoints/visual_retrieval_A_only.pt"
  "outputs/strict_train_A_test_BC_no_position_scale/checkpoints/visual_retrieval_A_only.pt"
)

mkdir -p "${OUTPUT_DIR}/checkpoints"

# By default, reuse the already-trained Route-A-only visual retrieval checkpoint.
# Set RETRAIN_VISUAL=1 only if you really want to retrain the visual heads.
RETRAIN_VISUAL="${RETRAIN_VISUAL:-0}"

VISUAL_EPOCHS=0

if [[ "${RETRAIN_VISUAL}" == "1" ]]; then
  VISUAL_EPOCHS="${VISUAL_EPOCHS_OVERRIDE:-30}"
  echo "Visual retrieval will be retrained on Route A."
else
  if [[ ! -f "${TARGET_VISUAL}" ]]; then
    FOUND=""
    for candidate in "${VISUAL_CANDIDATES[@]}"; do
      if [[ -f "${candidate}" ]]; then
        FOUND="${candidate}"
        break
      fi
    done

    if [[ -z "${FOUND}" ]]; then
      echo "ERROR: cannot find an existing Route-A-only visual checkpoint." >&2
      echo "Either restore one of these:" >&2
      printf '  %s\n' "${VISUAL_CANDIDATES[@]}" >&2
      echo "or run:" >&2
      echo "  RETRAIN_VISUAL=1 CUDA_VISIBLE_DEVICES=0 bash run_robust_tracker.sh ..." >&2
      exit 2
    fi

    cp -p "${FOUND}" "${TARGET_VISUAL}"
    echo "Copied visual checkpoint:"
    echo "  ${FOUND}"
    echo "  -> ${TARGET_VISUAL}"
  fi
fi

echo "================================================================================"
echo "RECURRENT MOTION STATE + KALMAN FILTER"
echo "================================================================================"
echo "GRU hidden state: persistent"
echo "Physical state:   [x,y,vx,vy,ax,ay]"
echo "Prediction:       constant-acceleration polynomial"
echo "Visual output:    coordinate + learned measurement variance"
echo "Final output:     Kalman-filtered coordinate"
echo "Training:         Route A only"
echo "Evaluation:       Route B + Route C"
echo "Sequence length:  ${RNN_SEQUENCE_LENGTH:-16} during truncated-BPTT training"
echo "Inference:        state persists continuously through the whole route"
echo "================================================================================"

python3 robust_tracker.py \
  --visual-epochs "${VISUAL_EPOCHS}" \
  "$@"
