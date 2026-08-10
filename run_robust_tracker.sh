#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

GPU="${GPU:-0}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-50}"
JITTER_M="${JITTER_M:-12}"
REUSE_VISUAL="${REUSE_VISUAL:-0}"

OUTPUT_DIR="outputs/route_coordinate_gru_kalman_v3"
VISUAL_CKPT="${OUTPUT_DIR}/checkpoints/visual_retrieval_A_only.pt"
mkdir -p "${OUTPUT_DIR}/checkpoints"

trap 'code=$?; echo "ERROR: stopped at line ${LINENO}, exit=${code}" >&2; exit ${code}' ERR

echo "================================================================================"
echo "TIMESTAMP-AWARE ROUTE-COORDINATE GRU + FILTERPY KALMAN v3"
echo "================================================================================"
echo "GPU             : ${GPU}"
echo "Visual epochs   : ${VISUAL_EPOCHS}"
echo "Temporal epochs : ${TEMPORAL_EPOCHS}"
echo "Reuse visual    : ${REUSE_VISUAL}"
echo "Every JSON waypoint is loaded and adjacent pairs are rebuilt automatically."
echo "Time unit is seconds; velocity unit is m/s."
echo "Inference automatically renders synchronized UAV+map videos and frame figures."
echo "================================================================================"

for f in config.py data.py visual_model.py visual_localizer.py robust_tracker.py render_results_video.py; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 10; }
done

python3 -m py_compile config.py visual_model.py robust_tracker.py render_results_video.py

python3 - <<'PY'
for name in ("torch", "filterpy", "matplotlib", "cv2"):
    __import__(name)
    print("import OK:", name)
PY

EXTRA_ARGS=()
if [[ "${REUSE_VISUAL}" == "1" ]]; then
  if [[ ! -f "${VISUAL_CKPT}" ]]; then
    OLD_CANDIDATES=(
      "outputs/route_coordinate_gru_kalman_v2/checkpoints/visual_retrieval_A_only.pt"
      "outputs/route_rnn_filterpy_full_retrain/checkpoints/visual_retrieval_A_only.pt"
      "outputs/strict_train_A_test_BC_t2only_w5/checkpoints/visual_retrieval_A_only.pt"
    )
    FOUND=""
    for candidate in "${OLD_CANDIDATES[@]}"; do
      if [[ -f "${candidate}" ]]; then
        FOUND="${candidate}"
        break
      fi
    done
    [[ -n "${FOUND}" ]] || {
      echo "REUSE_VISUAL=1 but no Route-A visual checkpoint was found." >&2
      exit 20
    }
    cp -p "${FOUND}" "${VISUAL_CKPT}"
    echo "Copied visual checkpoint:"
    echo "  ${FOUND}"
    echo "  -> ${VISUAL_CKPT}"
  fi
  EXTRA_ARGS+=(--reuse-visual)
fi

echo
echo "[1/2] TRAIN + B/C INFERENCE"
PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES="${GPU}" \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
NUMEXPR_NUM_THREADS=4 \
python3 -u robust_tracker.py \
  --mode train_eval \
  --visual-epochs "${VISUAL_EPOCHS}" \
  --epochs "${TEMPORAL_EPOCHS}" \
  --jitter-m "${JITTER_M}" \
  "${EXTRA_ARGS[@]}"

echo
echo "[2/2] SYNCHRONIZED VIDEO + FRAME-LABELLED FIGURES"
PYTHONUNBUFFERED=1 python3 -u render_results_video.py --route all

echo "================================================================================"
echo "DONE"
echo "Summary: ${OUTPUT_DIR}/robust_tracker_summary.json"
echo "Visualizations: ${OUTPUT_DIR}/visualizations/"
echo "================================================================================"
