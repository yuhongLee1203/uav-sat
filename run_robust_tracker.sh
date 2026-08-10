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
echo "RECURRENT VISUAL FIXED-HARDMS LOCALIZATION"
echo "================================================================================"
echo "GPU             : ${GPU}"
echo "Visual epochs   : ${VISUAL_EPOCHS}"
echo "Recurrent epochs: ${TEMPORAL_EPOCHS}"
echo "Reuse visual    : ${REUSE_VISUAL}"
echo
echo "TRAINING:"
echo "  RNN input = visual embeddings/logits + relative 6x6 offsets + prior model motion/state"
echo "  NO waypoint / absolute XY / GPS / velocity / Kalman state / timestamp enters the RNN"
echo
echo "INFERENCE:"
echo "  B/C GT is never read by inference"
echo "  waypoint W0 initializes only the first local lattice; it does not drive motion"
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
echo "[1/2] TRAIN VISUAL RETRIEVAL + RECURRENT VISUAL DECODER"

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
