#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

MODE="${MODE:-train_eval}"
GPU="${GPU:-0}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-50}"
REUSE_VISUAL="${REUSE_VISUAL:-1}"
FORCE_FULL_RETRAIN="${FORCE_FULL_RETRAIN:-0}"

OUTPUT_DIR="outputs/crf_inertial_rnn_kalman_v20"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
VISUAL_CKPT="${CHECKPOINT_DIR}/visual_retrieval_A_only.pt"
WARMUP_CKPT="${CHECKPOINT_DIR}/candidate_nextstep_warmup_A_only.pt"
TEMPORAL_CKPT="${CHECKPOINT_DIR}/crf_inertial_rnn_A_only.pt"

ROUTE_B_CSV="${OUTPUT_DIR}/route_B_crf_inertial_rnn_kalman_frames.csv"
ROUTE_C_CSV="${OUTPUT_DIR}/route_C_crf_inertial_rnn_kalman_frames.csv"
SUMMARY_JSON="${OUTPUT_DIR}/robust_tracker_summary.json"

mkdir -p "${CHECKPOINT_DIR}"

case "${MODE}" in
  train|eval|train_eval)
    ;;
  *)
    echo "ERROR: MODE must be train, eval, or train_eval" >&2
    exit 2
    ;;
esac

echo "======================================================================================================================"
echo "CRF-Inertial RNN + External Kalman v20"
echo "======================================================================================================================"
echo "MODE            : ${MODE}"
echo "GPU             : ${GPU}"
echo "Reuse visual    : ${REUSE_VISUAL}"
echo "Force full      : ${FORCE_FULL_RETRAIN}"
echo "Visual epochs   : ${VISUAL_EPOCHS}"
echo "Temporal epochs : ${TEMPORAL_EPOCHS}"
echo "Output          : ${OUTPUT_DIR}"
echo "======================================================================================================================"

python3 - <<'PY'
for name in ("torch", "filterpy", "cv2", "pandas"):
    __import__(name)
    print("import OK:", name)

import config
import robust_tracker
import visual_model

assert robust_tracker.ARCHITECTURE_NAME == "CRFInertialRNNKalman_v20"
assert hasattr(visual_model, "CRFCandidateRefiner")
assert hasattr(visual_model, "CRFInertialRNN")
assert int(config.GRID_SIZE) == 6
assert int(config.CANDIDATE_COUNT) == 36
assert bool(config.USE_HARD_FORWARD_MASK) is False

print("candidate layer : CRF-style 36-candidate emission + inertial transition")
print("HardMS          : applied AFTER candidate refinement")
print("RNN             : plain nn.RNNCell")
print("polynomial      : p_next = p_final + v_rnn + 0.5*a_rnn")
print("forward search  : full 6x6 moved to polynomial prediction")
print("hard 3x6 mask   : disabled")
print("final output    : external Kalman")
print("early stop      : Route-A episode valMLE")
PY

find_visual_checkpoint () {
  local candidates=(
    "${VISUAL_CKPT}"
    "outputs/twostage_autoregressive_hardms_rnn_kalman_v19/checkpoints/visual_retrieval_A_only.pt"
    "outputs/rnn_state_polynomial_hardms_kalman_v18/checkpoints/visual_retrieval_A_only.pt"
    "outputs/polynomial_hardms_state_rnn_kalman_v17/checkpoints/visual_retrieval_A_only.pt"
    "outputs/hardms_state_rnn_kalman_v16/checkpoints/visual_retrieval_A_only.pt"
    "outputs/recurrent_visual_measurement_kalman_v15/checkpoints/visual_retrieval_A_only.pt"
    "outputs/stable_visual_inertial_rnn_v14/checkpoints/visual_retrieval_A_only.pt"
  )

  local candidate

  for candidate in "${candidates[@]}"; do
    if [[ -s "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

prepare_visual () {
  if [[ "${FORCE_FULL_RETRAIN}" == "1" || "${REUSE_VISUAL}" != "1" ]]; then
    rm -f "${VISUAL_CKPT}"
    return 1
  fi

  local found=""

  if found="$(find_visual_checkpoint)"; then
    if [[ "${found}" != "${VISUAL_CKPT}" ]]; then
      cp -p "${found}" "${VISUAL_CKPT}"
      echo "Reuse visual checkpoint:"
      echo "  ${found}"
      echo "  -> ${VISUAL_CKPT}"
    else
      echo "Reuse existing v20 visual checkpoint:"
      echo "  ${VISUAL_CKPT}"
    fi
    return 0
  fi

  return 1
}

run_tracker () {
  local tracker_mode="$1"
  shift || true

  CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTHONUNBUFFERED=1 \
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

verify_outputs () {
  local failed=0

  for path in \
    "${WARMUP_CKPT}" \
    "${TEMPORAL_CKPT}" \
    "${ROUTE_B_CSV}" \
    "${ROUTE_C_CSV}" \
    "${SUMMARY_JSON}"
  do
    if [[ -s "${path}" ]]; then
      echo "OK: ${path}"
    else
      echo "MISSING: ${path}" >&2
      failed=1
    fi
  done

  if [[ "${failed}" != "0" ]]; then
    exit 20
  fi
}

case "${MODE}" in
  train)
    rm -f "${WARMUP_CKPT}" "${TEMPORAL_CKPT}"

    if prepare_visual; then
      run_tracker train --reuse-visual
    else
      run_tracker train
    fi
    ;;

  train_eval)
    rm -f \
      "${WARMUP_CKPT}" \
      "${TEMPORAL_CKPT}" \
      "${ROUTE_B_CSV}" \
      "${ROUTE_C_CSV}" \
      "${SUMMARY_JSON}"

    if prepare_visual; then
      run_tracker train_eval --reuse-visual
    else
      run_tracker train_eval
    fi

    verify_outputs
    python3 -u render_results_video.py --route all
    ;;

  eval)
    if [[ ! -s "${VISUAL_CKPT}" ]]; then
      if ! prepare_visual; then
        echo "ERROR: eval requires a visual checkpoint." >&2
        exit 22
      fi
    fi

    if [[ ! -s "${TEMPORAL_CKPT}" ]]; then
      echo "ERROR: eval requires v20 temporal checkpoint:" >&2
      echo "  ${TEMPORAL_CKPT}" >&2
      exit 23
    fi

    rm -f \
      "${ROUTE_B_CSV}" \
      "${ROUTE_C_CSV}" \
      "${SUMMARY_JSON}"

    run_tracker eval
    verify_outputs
    python3 -u render_results_video.py --route all
    ;;
esac

echo
echo "======================================================================================================================"
echo "DONE"
echo "======================================================================================================================"
echo "Warmup checkpoint : ${WARMUP_CKPT}"
echo "Temporal checkpoint: ${TEMPORAL_CKPT}"
echo "Route-B CSV       : ${ROUTE_B_CSV}"
echo "Route-C CSV       : ${ROUTE_C_CSV}"
echo "Summary           : ${SUMMARY_JSON}"
