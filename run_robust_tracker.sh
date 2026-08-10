#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

GPU="${GPU:-0}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-50}"
REUSE_VISUAL="${REUSE_VISUAL:-0}"

OUTPUT_DIR="outputs/pure_visual_lstm_waypoint_inference_v2_aligned"
VISUAL_CKPT="${OUTPUT_DIR}/checkpoints/visual_retrieval_A_only.pt"

mkdir -p "${OUTPUT_DIR}/checkpoints"

echo "================================================================================"
echo "VISUAL FIXED-HARDMS + WAYPOINT-INERTIAL KALMAN LOCALIZATION"
echo "================================================================================"
echo "GPU             : ${GPU}"
echo "Visual epochs   : ${VISUAL_EPOCHS}"
echo "Temporal epochs : not used (the legacy LSTM decoder is disabled)"
echo "Reuse visual    : ${REUSE_VISUAL}"
echo
echo "TRAINING:"
echo "  Network input = UAV/SAT image embeddings only"
echo "  NO waypoint / XY / GPS / velocity / Kalman state / timestamp enters retrieval"
echo
echo "INFERENCE:"
echo "  A fixed nominal speed initializes the filter; B/C GT is never read by inference"
echo "  waypoint coordinates/order initialize and sequence the inertial motion only"
echo "  waypoint frame_index/timestamp are ignored"
echo "  test GT/GPS is metrics/visualization only"
echo "================================================================================"

python3 - <<'PY'
for name in ("torch", "filterpy", "matplotlib", "cv2"):
    __import__(name)
    print("import OK:", name)
PY

EXTRA_ARGS=()

if [[ "${REUSE_VISUAL}" == "1" ]]; then
  if [[ ! -f "${VISUAL_CKPT}" ]]; then
    CANDIDATES=(
      "outputs/route_coordinate_gru_kalman_v3/checkpoints/visual_retrieval_A_only.pt"
      "outputs/route_coordinate_gru_kalman_v2/checkpoints/visual_retrieval_A_only.pt"
      "outputs/route_rnn_filterpy_full_retrain/checkpoints/visual_retrieval_A_only.pt"
      "outputs/strict_train_A_test_BC_t2only_w5/checkpoints/visual_retrieval_A_only.pt"
      "outputs/pure_visual_lstm_waypoint_inference/checkpoints/visual_retrieval_A_only.pt"
    )

    FOUND=""

    for candidate in "${CANDIDATES[@]}"; do
      if [[ -f "${candidate}" ]]; then
        FOUND="${candidate}"
        break
      fi
    done

    if [[ -z "${FOUND}" ]]; then
      echo "No existing Route-A visual checkpoint found." >&2
      exit 20
    fi

    cp -p "${FOUND}" "${VISUAL_CKPT}"

    echo "Copied visual checkpoint:"
    echo "  ${FOUND}"
    echo "  -> ${VISUAL_CKPT}"
  fi

  EXTRA_ARGS+=(--reuse-visual)
fi

echo
echo "[1/2] TRAIN VISUAL RETRIEVAL + B/C WAYPOINT-INERTIAL INFERENCE"

PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES="${GPU}" \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
NUMEXPR_NUM_THREADS=4 \
python3 -u robust_tracker.py \
  --mode train_eval \
  --visual-epochs "${VISUAL_EPOCHS}" \
  --temporal-epochs "${TEMPORAL_EPOCHS}" \
  "${EXTRA_ARGS[@]}"

echo
echo "[2/2] RENDER SYNCHRONIZED FRAME-BY-FRAME RESULTS"

PYTHONUNBUFFERED=1 \
python3 -u render_results_video.py \
  --route all

echo
echo "================================================================================"
echo "DONE"
echo "Summary:"
echo "  ${OUTPUT_DIR}/robust_tracker_summary.json"
echo "Videos / figures:"
echo "  ${OUTPUT_DIR}/visualizations/"
echo "================================================================================"
