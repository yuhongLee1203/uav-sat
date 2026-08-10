#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

MODE="${MODE:-train_eval}"
GPU="${GPU:-0}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-50}"
REUSE_VISUAL="${REUSE_VISUAL:-1}"

OUTPUT_DIR="outputs/route_conditioned_inertial_lstm_v4"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
VISUAL_CKPT="${CHECKPOINT_DIR}/visual_retrieval_A_only.pt"
TEMPORAL_CKPT="${CHECKPOINT_DIR}/route_inertial_lstm_A_only.pt"

mkdir -p "${CHECKPOINT_DIR}"

echo "================================================================================================"
echo "ROUTE-CONDITIONED INERTIAL LSTM v4"
echo "================================================================================================"
echo "Mode            : ${MODE}"
echo "GPU             : ${GPU}"
echo "Visual epochs   : ${VISUAL_EPOCHS}"
echo "Temporal epochs : ${TEMPORAL_EPOCHS}"
echo "Reuse visual    : ${REUSE_VISUAL}"
echo
echo "Core design:"
echo "  current image -> current continuous visual localization"
echo "  previous LSTM state -> recurrent visual memory"
echo "  previous [v_parallel,v_cross,a_parallel,a_cross] -> second-order polynomial soft prior"
echo "  waypoint start/end -> route-relative context"
echo "  polynomial / waypoint / Kalman NEVER push the search center forward by themselves"
echo "  next SAT grid center = previous IMAGE-derived visual localization"
echo "  FilterPy Kalman = final-output smoother only"
echo "================================================================================================"

case "${MODE}" in
  train|eval|train_eval)
    ;;
  *)
    echo "ERROR: MODE must be train, eval, or train_eval" >&2
    exit 2
    ;;
esac

python3 - <<'PY'
required = [
    "torch",
    "filterpy",
    "matplotlib",
    "cv2",
]
for name in required:
    __import__(name)
    print("import OK:", name)
PY

# -----------------------------------------------------------------------------
# Reuse an existing Route-A single-frame visual retrieval checkpoint.
# This does NOT reuse any old recurrent/CRF/GRU/LSTM temporal checkpoint.
# -----------------------------------------------------------------------------
if [[ "${MODE}" != "eval" && "${REUSE_VISUAL}" == "1" ]]; then
  if [[ ! -f "${VISUAL_CKPT}" ]]; then
    CANDIDATES=(
      "outputs/pure_visual_lstm_waypoint_inference_v2_aligned/checkpoints/visual_retrieval_A_only.pt"
      "outputs/pure_visual_lstm_waypoint_inference/checkpoints/visual_retrieval_A_only.pt"
      "outputs/route_coordinate_gru_kalman_v3/checkpoints/visual_retrieval_A_only.pt"
      "outputs/route_coordinate_gru_kalman_v2/checkpoints/visual_retrieval_A_only.pt"
      "outputs/route_rnn_filterpy_full_retrain/checkpoints/visual_retrieval_A_only.pt"
      "outputs/strict_train_A_test_BC_t2only_w5/checkpoints/visual_retrieval_A_only.pt"
    )

    FOUND=""

    for candidate in "${CANDIDATES[@]}"; do
      if [[ -f "${candidate}" ]]; then
        FOUND="${candidate}"
        break
      fi
    done

    if [[ -z "${FOUND}" ]]; then
      echo "ERROR: REUSE_VISUAL=1 but no existing Route-A visual checkpoint was found." >&2
      echo "Run REUSE_VISUAL=0 once to retrain the single-frame visual model." >&2
      exit 20
    fi

    cp -p "${FOUND}" "${VISUAL_CKPT}"

    echo
    echo "Copied visual checkpoint:"
    echo "  ${FOUND}"
    echo "  -> ${VISUAL_CKPT}"
  fi
fi

if [[ "${MODE}" == "eval" ]]; then
  if [[ ! -f "${VISUAL_CKPT}" ]]; then
    echo "ERROR: eval mode requires ${VISUAL_CKPT}" >&2
    exit 21
  fi

  if [[ ! -f "${TEMPORAL_CKPT}" ]]; then
    echo "ERROR: eval mode requires ${TEMPORAL_CKPT}" >&2
    exit 22
  fi
fi

EXTRA_ARGS=()

if [[ "${REUSE_VISUAL}" == "1" ]]; then
  EXTRA_ARGS+=(--reuse-visual)
fi

echo
echo "[1/2] ${MODE}: model pipeline"

PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES="${GPU}" \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
NUMEXPR_NUM_THREADS=4 \
python3 -u robust_tracker.py \
  --mode "${MODE}" \
  --visual-epochs "${VISUAL_EPOCHS}" \
  --temporal-epochs "${TEMPORAL_EPOCHS}" \
  "${EXTRA_ARGS[@]}"

if [[ "${MODE}" != "train" ]]; then
  echo
  echo "[2/2] render synchronized B/C inference videos and frame-labelled figures"

  PYTHONUNBUFFERED=1 \
  python3 -u render_results_video.py \
    --route all
else
  echo
  echo "[2/2] skipped visualization because MODE=train"
fi

echo
echo "================================================================================================"
echo "DONE"
echo "Summary:"
echo "  ${OUTPUT_DIR}/robust_tracker_summary.json"
echo
echo "Checkpoints:"
echo "  ${VISUAL_CKPT}"
echo "  ${TEMPORAL_CKPT}"
echo
echo "Inference CSV:"
echo "  ${OUTPUT_DIR}/route_B_route_inertial_lstm_frames.csv"
echo "  ${OUTPUT_DIR}/route_C_route_inertial_lstm_frames.csv"
echo
echo "Visualization:"
echo "  ${OUTPUT_DIR}/visualizations/"
echo "================================================================================================"
