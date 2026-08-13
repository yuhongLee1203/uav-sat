#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

MODE="${MODE:-train_eval}"
GPU="${GPU:-0}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
JITTER_M="${JITTER_M:-8}"
REUSE_VISUAL="${REUSE_VISUAL:-1}"
FORCE_FULL_RETRAIN="${FORCE_FULL_RETRAIN:-0}"
RESUME_TEMPORAL="${RESUME_TEMPORAL:-0}"
RENDER="${RENDER:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --visual-epochs) VISUAL_EPOCHS="$2"; shift 2 ;;
    --temporal-epochs|--epochs) TEMPORAL_EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --jitter-m) JITTER_M="$2"; shift 2 ;;
    --reuse-visual) REUSE_VISUAL="$2"; shift 2 ;;
    --force-full-retrain) FORCE_FULL_RETRAIN=1; shift ;;
    --resume-temporal) RESUME_TEMPORAL=1; shift ;;
    --no-render) RENDER=0; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: bash run_robust_tracker.sh [options]
  --mode train|eval|train_eval
  --gpu N
  --visual-epochs N
  --temporal-epochs N   (alias: --epochs N)
  --patience N          controlled-validation early-stop patience
  --jitter-m M          GT-prior jitter radius in metres
  --reuse-visual 0|1
  --force-full-retrain
  --resume-temporal
  --no-render
EOF
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "${MODE}" in
  train|eval|train_eval) ;;
  *) echo "ERROR: --mode must be train, eval, or train_eval" >&2; exit 2 ;;
esac

OUTPUT_DIR="outputs/controlled_gtprior_forward3x6_continuous_waypoint_rnn_polynomial_kalman_v33"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
VISUAL_CKPT="${CHECKPOINT_DIR}/visual_retrieval_A_only.pt"
TEMPORAL_CKPT="${CHECKPOINT_DIR}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
LATEST_TEMPORAL_CKPT="${CHECKPOINT_DIR}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only_latest.pt"
ROUTE_B_CSV="${OUTPUT_DIR}/route_B_controlled_gtprior_forward3x6_continuous_waypoint_rnn_polynomial_kalman_frames.csv"
ROUTE_C_CSV="${OUTPUT_DIR}/route_C_controlled_gtprior_forward3x6_continuous_waypoint_rnn_polynomial_kalman_frames.csv"
SUMMARY_JSON="${OUTPUT_DIR}/robust_tracker_summary.json"
mkdir -p "${CHECKPOINT_DIR}"

find_visual_checkpoint() {
  local candidates=(
    "${VISUAL_CKPT}"
    "outputs/controlled_gtprior_continuous_waypoint_rnn_polynomial_kalman_v32/checkpoints/visual_retrieval_A_only.pt"
    "outputs/controlled_gtprior_causal_heading_rnn_polynomial_kalman_v31/checkpoints/visual_retrieval_A_only.pt"
    "outputs/controlled_gtprior_heading_rnn_polynomial_kalman_v30/checkpoints/visual_retrieval_A_only.pt"
    "outputs/controlled_gtprior_nojump_rnn_polynomial_kalman_v29/checkpoints/visual_retrieval_A_only.pt"
    "outputs/controlled_gtprior_rnn_polynomial_kalman_v28/checkpoints/visual_retrieval_A_only.pt"
    "outputs/three_frame_multihyp_acquisition_gru_kalman_v27/checkpoints/visual_retrieval_A_only.pt"
    "outputs/three_frame_state_gru_polynomial_kalman_v26/checkpoints/visual_retrieval_A_only.pt"
    "outputs/route_progress_gru_polynomial_kalman_v25/checkpoints/visual_retrieval_A_only.pt"
    "outputs/waypoint_local_primary_recovery_gru_kalman_v24/checkpoints/visual_retrieval_A_only.pt"
    "outputs/waypoint_routeglobal_recovery_gru_kalman_v23/checkpoints/visual_retrieval_A_only.pt"
    "outputs/waypoint_temporal_motion_gru_kalman_v22/checkpoints/visual_retrieval_A_only.pt"
    "outputs/waypoint_routeframe_gru_kalman_v21/checkpoints/visual_retrieval_A_only.pt"
    "outputs/crf_inertial_rnn_kalman_v20/checkpoints/visual_retrieval_A_only.pt"
    "outputs/recurrent_visual_measurement_kalman_v15/checkpoints/visual_retrieval_A_only.pt"
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

prepare_visual() {
  if [[ "${FORCE_FULL_RETRAIN}" == "1" || "${REUSE_VISUAL}" != "1" ]]; then
    rm -f "${VISUAL_CKPT}"
    return 1
  fi
  local found=""
  if found="$(find_visual_checkpoint)"; then
    if [[ "${found}" != "${VISUAL_CKPT}" ]]; then
      cp -p "${found}" "${VISUAL_CKPT}"
      echo "Reuse visual checkpoint: ${found} -> ${VISUAL_CKPT}"
    else
      echo "Reuse visual checkpoint: ${VISUAL_CKPT}"
    fi
    return 0
  fi
  return 1
}

verify_python() {
  python3 - <<'PY'
import config
import robust_tracker
import visual_model
expected = "ControlledGTPriorThreeFrameForward3x6CausalHeadingContinuousWaypointGRUPolynomialKalman_v33"
checks = [
    (config.ARCHITECTURE_NAME == expected, "config architecture"),
    (robust_tracker.ARCHITECTURE_NAME == expected, "tracker architecture"),
    (hasattr(visual_model, "ThreeFrameRouteStateGRU"), "ThreeFrameRouteStateGRU"),
    (bool(config.CONTROLLED_GT_PRIOR), "controlled GT prior enabled"),
    (bool(config.CONTROLLED_FINAL_PROGRESS_CAP_TO_GT), "final progress cap enabled"),
    (int(config.GRID_SIZE) == 6, "6x6 base geometry"),
    (int(config.ACQ_HYPOTHESIS_COUNT) == 1, "single GT+jitter local window"),
    (bool(config.FORWARD_ONLY_LOCAL_SEARCH), "forward-only local search enabled"),
    (int(config.FORWARD_SEARCH_CANDIDATE_COUNT) == 18, "forward 3x6=18 candidates"),
]
failed = [name for ok, name in checks if not ok]
if failed:
    raise RuntimeError("v33 preflight failed: " + ", ".join(failed))
print("architecture :", expected)
print("protocol     : controlled GT+smooth-jitter local prior on every frame")
print("temporal     : 3 UAV frames + recurrent GRU state")
print("motion       : v/a + causal heading -> second-order inertial polynomial; turn-rate is recurrent state only")
print("visual       : causal-heading forward 3x6 (18 of 36 geometry) UAV-SAT measurement")
print("final filter : constrained route-coordinate Kalman (final output)")
print("direction    : causal RNN heading; route frame rotates only AFTER waypoint crossing")
print("pace         : controlled causal GT-speed envelope + <=0.75m/frame smooth catch-up")
print("display/eval : final progress cannot pass current GT; abnormal jump means excess over GT motion")
PY
}

run_tracker() {
  local tracker_mode="$1"
  shift || true
  local extra=()
  if [[ "${RESUME_TEMPORAL}" == "1" ]]; then
    extra+=(--resume-temporal)
  fi
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
    --patience "${PATIENCE}" \
    --jitter-m "${JITTER_M}" \
    "${extra[@]}" \
    "$@"
}

verify_eval_outputs() {
  local failed=0
  for path in "${TEMPORAL_CKPT}" "${ROUTE_B_CSV}" "${ROUTE_C_CSV}" "${SUMMARY_JSON}"; do
    if [[ -s "${path}" ]]; then
      echo "OK: ${path}"
    else
      echo "MISSING: ${path}" >&2
      failed=1
    fi
  done
  [[ "${failed}" == "0" ]] || exit 20
}

printf '%*s\n' 108 '' | tr ' ' '='
echo "Controlled GT-Prior Forward-3x6 Continuous-Waypoint Three-Frame GRU + Polynomial + Kalman v33"
printf '%*s\n' 108 '' | tr ' ' '='
echo "MODE              : ${MODE}"
echo "GPU               : ${GPU}"
echo "Visual epochs     : ${VISUAL_EPOCHS}"
echo "Temporal epochs   : ${TEMPORAL_EPOCHS}"
echo "EarlyStop patience: ${PATIENCE}"
echo "GT-prior jitter   : ${JITTER_M} m"
echo "Reuse visual      : ${REUSE_VISUAL}"
echo "Resume temporal   : ${RESUME_TEMPORAL}"
echo "Output            : ${OUTPUT_DIR}"
printf '%*s\n' 108 '' | tr ' ' '='

verify_python

case "${MODE}" in
  train)
    if [[ "${RESUME_TEMPORAL}" != "1" ]]; then
      rm -f "${TEMPORAL_CKPT}" "${LATEST_TEMPORAL_CKPT}"
    fi
    if prepare_visual; then
      run_tracker train --reuse-visual
    else
      run_tracker train
    fi
    ;;
  train_eval)
    if [[ "${RESUME_TEMPORAL}" != "1" ]]; then
      rm -f "${TEMPORAL_CKPT}" "${LATEST_TEMPORAL_CKPT}"
    fi
    rm -f "${ROUTE_B_CSV}" "${ROUTE_C_CSV}" "${SUMMARY_JSON}"
    if prepare_visual; then
      run_tracker train_eval --reuse-visual
    else
      run_tracker train_eval
    fi
    verify_eval_outputs
    if [[ "${RENDER}" == "1" ]]; then
      python3 -u render_results_video.py --route all
    fi
    ;;
  eval)
    if [[ ! -s "${VISUAL_CKPT}" ]]; then
      if ! prepare_visual; then
        echo "ERROR: eval requires visual checkpoint" >&2
        exit 22
      fi
    fi
    if [[ ! -s "${TEMPORAL_CKPT}" ]]; then
      echo "ERROR: eval requires temporal checkpoint: ${TEMPORAL_CKPT}" >&2
      exit 23
    fi
    rm -f "${ROUTE_B_CSV}" "${ROUTE_C_CSV}" "${SUMMARY_JSON}"
    run_tracker eval --reuse-visual
    verify_eval_outputs
    if [[ "${RENDER}" == "1" ]]; then
      python3 -u render_results_video.py --route all
    fi
    ;;
esac

echo
echo "DONE"
echo "Temporal checkpoint : ${TEMPORAL_CKPT}"
echo "Route-B CSV         : ${ROUTE_B_CSV}"
echo "Route-C CSV         : ${ROUTE_C_CSV}"
echo "Summary             : ${SUMMARY_JSON}"
