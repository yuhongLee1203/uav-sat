#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

MODE="${MODE:-train_eval}"
GPU="${GPU:-0}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-40}"
REUSE_VISUAL="${REUSE_VISUAL:-1}"

OUTPUT_DIR="outputs/route_bounded_hypothesis_lstm_v6"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
VISUAL_CKPT="${CHECKPOINT_DIR}/visual_retrieval_A_only.pt"
TEMPORAL_CKPT="${CHECKPOINT_DIR}/route_hypothesis_lstm_A_only.pt"

ROUTE_B_CSV="${OUTPUT_DIR}/route_B_route_hypothesis_lstm_frames.csv"
ROUTE_C_CSV="${OUTPUT_DIR}/route_C_route_hypothesis_lstm_frames.csv"
SUMMARY_JSON="${OUTPUT_DIR}/robust_tracker_summary.json"

mkdir -p "${CHECKPOINT_DIR}"

echo "===================================================================================================="
echo "ROUTE-BOUNDED HYPOTHESIS LSTM v6 - CSV SAFE RUNNER"
echo "===================================================================================================="
echo "MODE            : ${MODE}"
echo "GPU             : ${GPU}"
echo "Visual epochs   : ${VISUAL_EPOCHS}"
echo "Temporal epochs : ${TEMPORAL_EPOCHS}"
echo "Reuse visual    : ${REUSE_VISUAL}"
echo
echo "Important:"
echo "  train_eval is explicitly split into TRAIN -> EVAL -> VERIFY CSV -> RENDER"
echo "  renderer is never started unless both Route-B and Route-C CSV files exist"
echo "===================================================================================================="

case "${MODE}" in
  train|eval|train_eval)
    ;;
  *)
    echo "ERROR: MODE must be train, eval, or train_eval" >&2
    exit 2
    ;;
esac

python3 - <<'PY'
for name in ("torch", "filterpy", "cv2", "matplotlib", "pandas"):
    __import__(name)
    print("import OK:", name)
PY

# --------------------------------------------------------------------------------------
# Visual checkpoint reuse.
# --------------------------------------------------------------------------------------
if [[ "${MODE}" != "eval" && "${REUSE_VISUAL}" == "1" ]]; then
  if [[ ! -f "${VISUAL_CKPT}" ]]; then
    CANDIDATES=(
      "outputs/visual_motion_gated_route_lstm_v5/checkpoints/visual_retrieval_A_only.pt"
      "outputs/route_conditioned_inertial_lstm_v4/checkpoints/visual_retrieval_A_only.pt"
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
      echo "ERROR: no reusable Route-A visual checkpoint found." >&2
      echo "Run once with REUSE_VISUAL=0." >&2
      exit 20
    fi

    cp -p "${FOUND}" "${VISUAL_CKPT}"

    echo
    echo "Copied visual checkpoint:"
    echo "  ${FOUND}"
    echo "  -> ${VISUAL_CKPT}"
  fi
fi

run_tracker () {
  local tracker_mode="$1"
  shift || true

  echo
  echo "----------------------------------------------------------------------------------------------------"
  echo "robust_tracker.py --mode ${tracker_mode}"
  echo "----------------------------------------------------------------------------------------------------"

  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 \
  python3 -u robust_tracker.py \
    --mode "${tracker_mode}" \
    --visual-epochs "${VISUAL_EPOCHS}" \
    --temporal-epochs "${TEMPORAL_EPOCHS}" \
    "$@"
}

verify_eval_outputs () {
  local failed=0

  echo
  echo "----------------------------------------------------------------------------------------------------"
  echo "VERIFY INFERENCE OUTPUTS"
  echo "----------------------------------------------------------------------------------------------------"

  for path in "${ROUTE_B_CSV}" "${ROUTE_C_CSV}" "${SUMMARY_JSON}"; do
    if [[ -s "${path}" ]]; then
      echo "OK: ${path}"
    else
      echo "MISSING: ${path}" >&2
      failed=1
    fi
  done

  if [[ "${failed}" != "0" ]]; then
    echo >&2
    echo "Inference finished without the required v6 CSV files." >&2
    echo "Do NOT run render_results_video.py yet." >&2
    echo >&2
    echo "Current files under ${OUTPUT_DIR}:" >&2
    find "${OUTPUT_DIR}" -maxdepth 2 -type f -printf '  %p\n' 2>/dev/null | sort >&2 || true
    exit 30
  fi
}

# --------------------------------------------------------------------------------------
# Explicit execution plan.
# --------------------------------------------------------------------------------------
case "${MODE}" in
  train)
    if [[ "${REUSE_VISUAL}" == "1" ]]; then
      run_tracker train --reuse-visual
    else
      run_tracker train
    fi
    ;;

  eval)
    if [[ ! -s "${VISUAL_CKPT}" ]]; then
      echo "ERROR: eval requires ${VISUAL_CKPT}" >&2
      exit 21
    fi

    if [[ ! -s "${TEMPORAL_CKPT}" ]]; then
      echo "ERROR: eval requires ${TEMPORAL_CKPT}" >&2
      exit 22
    fi

    # Remove stale/incomplete inference files so success cannot be confused
    # with outputs from an earlier run.
    rm -f "${ROUTE_B_CSV}" "${ROUTE_C_CSV}" "${SUMMARY_JSON}"

    run_tracker eval
    verify_eval_outputs

    echo
    echo "[RENDER] B/C synchronized inference videos + figures"

    PYTHONUNBUFFERED=1 \
    python3 -u render_results_video.py \
      --route all
    ;;

  train_eval)
    # TRAIN is intentionally a separate process.
    # This guarantees the checkpoint is flushed to disk before evaluation.
    if [[ "${REUSE_VISUAL}" == "1" ]]; then
      run_tracker train --reuse-visual
    else
      run_tracker train
    fi

    if [[ ! -s "${TEMPORAL_CKPT}" ]]; then
      echo "ERROR: training returned but temporal checkpoint is missing:" >&2
      echo "  ${TEMPORAL_CKPT}" >&2
      exit 23
    fi

    # EVAL is intentionally a fresh Python process.
    rm -f "${ROUTE_B_CSV}" "${ROUTE_C_CSV}" "${SUMMARY_JSON}"

    run_tracker eval
    verify_eval_outputs

    echo
    echo "[RENDER] B/C synchronized inference videos + figures"

    PYTHONUNBUFFERED=1 \
    python3 -u render_results_video.py \
      --route all
    ;;
esac

echo
echo "===================================================================================================="
echo "DONE"
echo "===================================================================================================="
echo "Temporal checkpoint:"
echo "  ${TEMPORAL_CKPT}"
echo
if [[ "${MODE}" != "train" ]]; then
  echo "Inference CSV:"
  echo "  ${ROUTE_B_CSV}"
  echo "  ${ROUTE_C_CSV}"
  echo
  echo "Summary:"
  echo "  ${SUMMARY_JSON}"
  echo
  echo "Videos / figures:"
  echo "  ${OUTPUT_DIR}/visualizations/"
fi
echo "===================================================================================================="
