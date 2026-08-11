#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

MODE="${MODE:-train_eval}"
GPU="${GPU:-0}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-50}"
REUSE_VISUAL="${REUSE_VISUAL:-1}"
FORCE_FULL_RETRAIN="${FORCE_FULL_RETRAIN:-0}"

OUTPUT_DIR="outputs/stable_visual_inertial_rnn_v14"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
VISUAL_CKPT="${CHECKPOINT_DIR}/visual_retrieval_A_only.pt"
TEMPORAL_CKPT="${CHECKPOINT_DIR}/stable_visual_inertial_rnn_A_only.pt"

ROUTE_B_CSV="${OUTPUT_DIR}/route_B_stable_visual_inertial_rnn_frames.csv"
ROUTE_C_CSV="${OUTPUT_DIR}/route_C_stable_visual_inertial_rnn_frames.csv"
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

echo "===================================================================================================="
echo "STABLE VISUAL-INERTIAL RNN v14"
echo "===================================================================================================="
echo "MODE            : ${MODE}"
echo "GPU             : ${GPU}"
echo "Reuse visual    : ${REUSE_VISUAL}"
echo "Force full      : ${FORCE_FULL_RETRAIN}"
echo "Visual epochs   : ${VISUAL_EPOCHS}"
echo "Temporal epochs : ${TEMPORAL_EPOCHS}"
echo "Output          : ${OUTPUT_DIR}"
echo "===================================================================================================="

python3 - <<'PY'
for name in ("torch", "filterpy", "cv2", "pandas"):
    __import__(name)
    print("import OK:", name)

import config
import robust_tracker
import visual_model

assert robust_tracker.ARCHITECTURE_NAME == "StableVisualInertialRNN_v14"
assert hasattr(visual_model, "StableVisualInertialRNN")
assert int(config.CANDIDATE_COUNT) == 36
assert int(config.GRID_SIZE) == 6
assert float(config.MAX_STEP_M_PER_FRAME) == 10.0

print("architecture OK:", robust_tracker.ARCHITECTURE_NAME)
print("RNN unit          : nn.RNNCell")
print("search            : full 6x6 / 36")
print("max motion/output : 10 m/frame")
print("hard forward mask : DISABLED")
PY

find_visual_checkpoint () {
  local candidates=(
    "${VISUAL_CKPT}"
    "outputs/timestamp_velocity_visual_rnn_v13/checkpoints/visual_retrieval_A_only.pt"
    "outputs/direct_displacement_visual_rnn_v12/checkpoints/visual_retrieval_A_only.pt"
    "outputs/continuous_progress_visual_rnn_v11/checkpoints/visual_retrieval_A_only.pt"
    "outputs/reversible_topology_recovery_lstm_v10/checkpoints/visual_retrieval_A_only.pt"
    "outputs/image_causal_forward_lstm_v9/checkpoints/visual_retrieval_A_only.pt"
    "outputs/route_tangent_forward_lstm_v8/checkpoints/visual_retrieval_A_only.pt"
    "outputs/heading_forward_lstm_v7/checkpoints/visual_retrieval_A_only.pt"
    "outputs/route_bounded_hypothesis_lstm_v6/checkpoints/visual_retrieval_A_only.pt"
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
      echo "Reuse Route-A visual checkpoint:"
      echo "  ${found}"
      echo "  -> ${VISUAL_CKPT}"
    else
      echo "Reuse existing v14 visual checkpoint:"
      echo "  ${VISUAL_CKPT}"
    fi
    return 0
  fi

  return 1
}

run_tracker () {
  local tracker_mode="$1"
  shift || true

  echo
  echo "----------------------------------------------------------------------------------------------------"
  echo "robust_tracker.py --mode ${tracker_mode}"
  echo "----------------------------------------------------------------------------------------------------"

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
    rm -f "${TEMPORAL_CKPT}"

    if prepare_visual; then
      run_tracker train --reuse-visual
    else
      run_tracker train
    fi
    ;;

  train_eval)
    rm -f \
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
        echo "ERROR: eval requires a Route-A visual checkpoint." >&2
        exit 22
      fi
    fi

    if [[ ! -s "${TEMPORAL_CKPT}" ]]; then
      echo "ERROR: eval requires v14 temporal checkpoint:" >&2
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
echo "===================================================================================================="
echo "DONE"
echo "===================================================================================================="
echo "Temporal checkpoint: ${TEMPORAL_CKPT}"
echo "Route-B CSV       : ${ROUTE_B_CSV}"
echo "Route-C CSV       : ${ROUTE_C_CSV}"
echo "Summary           : ${SUMMARY_JSON}"
