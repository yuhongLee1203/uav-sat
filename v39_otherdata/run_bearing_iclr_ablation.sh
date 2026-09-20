#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CITY="${CITY:-citya}"
case "${CITY}" in citya|cityb|cityc|cityd) ;; *) echo "ERROR: CITY must be citya/cityb/cityc/cityd" >&2; exit 2;; esac

SUITE_ROOT="${ICLR_SUITE_ROOT:-${REPO_ROOT}/v39_otherdata/iclr_bearing_${CITY}_ablation_v2_${TS}}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-100}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
PATIENCE="${PATIENCE:-14}"
SEED="${SEED:-2033}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"
RESUME_EVAL="${RESUME_EVAL:-0}"
VARIANTS=(full no_gru no_kalman no_ms frames1 frames2 grid4 grid5 grid7 grid8)

case "${RESUME_EVAL}" in 0|1) ;; *) echo "ERROR: RESUME_EVAL must be 0 or 1" >&2; exit 2;; esac
mkdir -p "${SUITE_ROOT}/logs"

# Same supervision for 1/2/3-frame checkpoints.
export UAVSAT_LOSS_MEASUREMENT="${UAVSAT_LOSS_MEASUREMENT:-2.0}"
export UAVSAT_LOSS_NEXT_STEP="${UAVSAT_LOSS_NEXT_STEP:-2.5}"
export UAVSAT_LOSS_VELOCITY="${UAVSAT_LOSS_VELOCITY:-0.40}"
export UAVSAT_LOSS_ACCELERATION="${UAVSAT_LOSS_ACCELERATION:-0.25}"
export UAVSAT_TEMPORAL_LR="${UAVSAT_TEMPORAL_LR:-8e-5}"
export UAVSAT_RNN_DROPOUT="${UAVSAT_RNN_DROPOUT:-0.05}"

# Residual temporal motion.
export UAVSAT_MOTION_VEL_ALPHA="${UAVSAT_MOTION_VEL_ALPHA:-0.65}"
export UAVSAT_MOTION_STEP_ALPHA="${UAVSAT_MOTION_STEP_ALPHA:-0.70}"
export UAVSAT_MOTION_RESIDUAL_FORWARD_M="${UAVSAT_MOTION_RESIDUAL_FORWARD_M:-2.5}"
export UAVSAT_MOTION_RESIDUAL_CROSS_M="${UAVSAT_MOTION_RESIDUAL_CROSS_M:-1.25}"
export UAVSAT_MOTION_RESIDUAL_ACCEL_FORWARD_M="${UAVSAT_MOTION_RESIDUAL_ACCEL_FORWARD_M:-1.25}"
export UAVSAT_MOTION_RESIDUAL_ACCEL_CROSS_M="${UAVSAT_MOTION_RESIDUAL_ACCEL_CROSS_M:-0.75}"
export UAVSAT_TEMPORAL_ADAPTER_2FRAME_SCALE="${UAVSAT_TEMPORAL_ADAPTER_2FRAME_SCALE:-0.45}"
export UAVSAT_TEMPORAL_ADAPTER_3FRAME_SCALE="${UAVSAT_TEMPORAL_ADAPTER_3FRAME_SCALE:-1.00}"

# Measurement-preserving residual Kalman.
export UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2="${UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2:-4.0}"
export UAVSAT_KALMAN_Q_PROGRESS="${UAVSAT_KALMAN_Q_PROGRESS:-1.50}"
export UAVSAT_KALMAN_Q_CROSS="${UAVSAT_KALMAN_Q_CROSS:-0.40}"
export UAVSAT_KALMAN_Q_VELOCITY="${UAVSAT_KALMAN_Q_VELOCITY:-1.00}"
export UAVSAT_KALMAN_CONFIDENCE_POWER="${UAVSAT_KALMAN_CONFIDENCE_POWER:-0.35}"
export UAVSAT_KALMAN_PRIOR_BLEND_BASE="${UAVSAT_KALMAN_PRIOR_BLEND_BASE:-0.08}"
export UAVSAT_KALMAN_PRIOR_BLEND_LOWCONF_GAIN="${UAVSAT_KALMAN_PRIOR_BLEND_LOWCONF_GAIN:-0.18}"
export UAVSAT_KALMAN_PRIOR_BLEND_MAX="${UAVSAT_KALMAN_PRIOR_BLEND_MAX:-0.30}"
export UAVSAT_KALMAN_STEP_RELAX_CONFIDENCE="${UAVSAT_KALMAN_STEP_RELAX_CONFIDENCE:-0.52}"
export UAVSAT_KALMAN_STEP_VISUAL_SLACK_M="${UAVSAT_KALMAN_STEP_VISUAL_SLACK_M:-1.5}"

python3 -m py_compile \
  v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/build_iclr_ablation_tables.py \
  v39_otherdata/bearing_prepare_multicity.py \
  v39_DirectFinalMS/patch_direct_finalms.py \
  v39_DirectFinalMS/patch_simple_figure_gru.py

PREPARED="${SUITE_ROOT}/${CITY}/prepared"

if [[ "${RESUME_EVAL}" == "1" ]]; then
  [[ -s "${PREPARED}/experiment.json" ]] || {
    echo "ERROR: RESUME_EVAL=1 but prepared data missing: ${PREPARED}" >&2
    exit 2
  }
  echo "[PREP] reuse current-suite ${CITY}: ${PREPARED}"
else
  rm -rf "${PREPARED}" "${PREPARED}__building"
  mkdir -p "${SUITE_ROOT}/${CITY}"
  echo "================================================================================"
  echo "[FRESH BEARING PREP] ONLY ${CITY}"
  echo "[FRESH BEARING PREP] source=${DATASET_ROOT}"
  echo "[FRESH BEARING PREP] cityb/cityc/cityd WILL NOT RUN unless CITY is changed"
  echo "================================================================================"
  python3 -u v39_otherdata/bearing_prepare_multicity.py \
    --dataset-root "${DATASET_ROOT}" \
    --city "${CITY}" \
    --output-root "${PREPARED}" \
    2>&1 | tee "${SUITE_ROOT}/logs/${CITY}_prepare.log"
fi

common_args(){
  local gpu="$1"
  COMMON=(
    --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}"
    --city "${CITY}" --gpu "${gpu}" --backbone mobilenet_v3_small
    --visual-epochs "${VISUAL_EPOCHS}" --temporal-epochs "${TEMPORAL_EPOCHS}"
    --epochs-per-route "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}"
    --jitter-m 8 --max-sample-distance-m 15
    --heading-weight-px-per-deg 0 --ms-bandwidth-m 7 --seed "${SEED}"
  )
}

run_train(){
  local frames="$1" gpu="$2"
  common_args "${gpu}"
  echo "[TRAIN START] ${CITY} frames=${frames} GPU${gpu}"
  python3 -u v39_otherdata/bearing_iclr_ablation.py train \
    "${COMMON[@]}" --train-frames "${frames}" \
    2>&1 | tee "${SUITE_ROOT}/logs/${CITY}_train_f${frames}.log"
  echo "[TRAIN DONE] ${CITY} frames=${frames}"
}

share_visual_checkpoint(){
  local src="${SUITE_ROOT}/${CITY}/train_frames1/checkpoints/visual_retrieval_A_only.pt"
  [[ -s "${src}" ]] || { echo "ERROR: visual checkpoint missing: ${src}" >&2; exit 19; }
  local f dst
  for f in 2 3; do
    dst="${SUITE_ROOT}/${CITY}/train_frames${f}/checkpoints/visual_retrieval_A_only.pt"
    mkdir -p "$(dirname "${dst}")"
    rm -f "${dst}"
    ln -s "${src}" "${dst}"
  done
  echo "[TEMPORAL FAIRNESS] frame1/frame2/frame3 share one visual checkpoint"
}

run_eval_group(){
  local gpu="$1"; shift
  common_args "${gpu}"
  local variant
  for variant in "$@"; do
    echo "[EVAL START] ${CITY} ${variant} GPU${gpu}"
    python3 -u v39_otherdata/bearing_iclr_ablation.py eval \
      "${COMMON[@]}" --variant "${variant}" \
      2>&1 | tee "${SUITE_ROOT}/logs/${CITY}_${variant}.log"
  done
}

echo "================================================================================"
echo "Bearing-UAV SINGLE-CITY ablation"
echo "CITY            : ${CITY}"
echo "Held-out tracks : nav50 / nav51 inside ${CITY}"
echo "Architecture    : Forward18 SoftMS -> temporal residual GRU"
echo "                  -> measurement-preserving residual Kalman"
echo "                  -> final MeanShift -> XY"
echo "Other cities    : NOT RUN"
echo "================================================================================"

common_args 0
python3 -u v39_otherdata/bearing_iclr_ablation.py check \
  "${COMMON[@]}" --train-frames 3 \
  2>&1 | tee "${SUITE_ROOT}/logs/${CITY}_preflight.log"

if [[ "${RESUME_EVAL}" == "0" ]]; then
  run_train 1 0
  share_visual_checkpoint

  ( run_train 2 5 ) & p2=$!
  ( run_train 3 6 ) & p3=$!
  status=0
  wait "${p2}" || status=1
  wait "${p3}" || status=1
  [[ "${status}" == "0" ]] || { echo "ERROR: temporal training failed" >&2; exit 20; }
else
  for f in 1 2 3; do
    ck="${SUITE_ROOT}/${CITY}/train_frames${f}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
    [[ -s "${ck}" ]] || { echo "ERROR: missing ${ck}" >&2; exit 2; }
  done
fi

# Full first warms shared held-out caches.
run_eval_group 0 full

( run_eval_group 0 no_gru grid4 grid7 ) & p0=$!
( run_eval_group 5 no_kalman frames1 grid5 ) & p5=$!
( run_eval_group 6 no_ms frames2 grid8 ) & p6=$!
status=0
wait "${p0}" || status=1
wait "${p5}" || status=1
wait "${p6}" || status=1
[[ "${status}" == "0" ]] || { echo "ERROR: evaluation failed; inspect logs" >&2; exit 21; }

python3 v39_otherdata/build_iclr_ablation_tables.py \
  --suite-root "${SUITE_ROOT}" \
  --cities "${CITY}"

printf '%s\n' "${SUITE_ROOT}" > v39_otherdata/LATEST_ICLR_BEARING_ABLATION.txt

if [[ "${UPLOAD_RESULTS}" == "1" ]]; then
  upload_wt="$(mktemp -d "${REPO_ROOT%/*}/uav-sat-iclr-upload-XXXXXX")"
  upload_branch="iclr-${CITY}-ablation-v2-${TS}-$$"
  cleanup(){
    git -C "${REPO_ROOT}" worktree remove --force "${upload_wt}" >/dev/null 2>&1 || true
    git -C "${REPO_ROOT}" branch -D "${upload_branch}" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT

  git fetch origin v39_otherdata
  git worktree add -b "${upload_branch}" "${upload_wt}" origin/v39_otherdata
  dest="paper_results/iclr_bearing_${CITY}_ablation_v2_${TS}"
  mkdir -p "${upload_wt}/${dest}"

  cp "${SUITE_ROOT}"/paper_ablation_results.{json,csv} "${upload_wt}/${dest}/"
  cp "${SUITE_ROOT}"/paper_ablation_tables.{md,tex} "${upload_wt}/${dest}/"
  cp "${SUITE_ROOT}/paper_trend_audit.json" "${upload_wt}/${dest}/"
  cp "${PREPARED}/experiment.json" "${upload_wt}/${dest}/${CITY}_prepared_experiment.json"

  for f in 1 2 3; do
    src="${SUITE_ROOT}/${CITY}/train_frames${f}"
    dst="${upload_wt}/${dest}/${CITY}/train_frames${f}"
    mkdir -p "${dst}"
    cp "${src}/experiment_manifest.json" "${dst}/" 2>/dev/null || true
    if [[ "${f}" == "3" ]]; then
      cp "${src}/kalman_calibration.json" "${dst}/" 2>/dev/null || true
    fi
  done

  for variant in "${VARIANTS[@]}"; do
    src="${SUITE_ROOT}/${CITY}/variants/${variant}"
    dst="${upload_wt}/${dest}/${CITY}/${variant}"
    mkdir -p "${dst}"
    cp "${src}/bearing_v39_summary.json" "${dst}/"
    cp "${src}/experiment_manifest.json" "${dst}/"
    cp "${src}"/*_frames.csv "${dst}/" 2>/dev/null || true
  done

  (
    cd "${upload_wt}"
    git add "${dest}"
    git commit -m "Add measured ${CITY} Bearing single-city ablation v2 ${TS}"
    git fetch origin v39_otherdata
    git rebase origin/v39_otherdata
    git push origin HEAD:v39_otherdata
  )
  echo "GitHub results: ${dest}"
fi

echo "================================================================================"
echo "DONE: Bearing-UAV SINGLE-CITY ablation"
echo "CITY  : ${CITY}"
echo "Suite : ${SUITE_ROOT}"
echo "Table : ${SUITE_ROOT}/paper_ablation_tables.md"
echo "Audit : ${SUITE_ROOT}/paper_trend_audit.json"
echo "================================================================================"
