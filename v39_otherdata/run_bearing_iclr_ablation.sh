#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CITY="${CITY:-citya}"
case "${CITY}" in citya|cityb|cityc|cityd) ;; *) echo "ERROR: invalid CITY=${CITY}" >&2; exit 2;; esac
SUITE_ROOT="${ICLR_SUITE_ROOT:-${REPO_ROOT}/v39_otherdata/iclr_bearing_softms_fair_${TS}}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-80}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
PATIENCE="${PATIENCE:-6}"
SEED="${SEED:-2033}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"
RESUME_EVAL="${RESUME_EVAL:-0}"
case "${RESUME_EVAL}" in 0|1) ;; *) echo "ERROR: RESUME_EVAL must be 0 or 1" >&2; exit 2;; esac
VARIANTS=(full no_gru no_kalman no_ms frames1 frames2 grid4 grid5 grid7 grid8)
mkdir -p "${SUITE_ROOT}/logs"

# Same Route-A-only temporal objective for all frame-count variants.
export UAVSAT_LOSS_MEASUREMENT="${UAVSAT_LOSS_MEASUREMENT:-1.0}"
export UAVSAT_LOSS_NEXT_STEP="${UAVSAT_LOSS_NEXT_STEP:-3.0}"
export UAVSAT_LOSS_VELOCITY="${UAVSAT_LOSS_VELOCITY:-0.25}"
export UAVSAT_TEMPORAL_LR="${UAVSAT_TEMPORAL_LR:-2e-4}"
export UAVSAT_RNN_DROPOUT="${UAVSAT_RNN_DROPOUT:-0.10}"

python3 -m py_compile \
  v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/build_iclr_ablation_tables.py \
  v39_otherdata/bearing_prepare_multicity.py \
  v39_DirectFinalMS/patch_direct_finalms.py \
  v39_DirectFinalMS/patch_simple_figure_gru.py

prepare_city(){
  local city="$1"
  local target="${SUITE_ROOT}/${city}/prepared"
  if [[ -s "${target}/experiment.json" ]]; then
    echo "[PREP] reuse ${target}"
    return
  fi
  local existing="${REPO_ROOT}/v39_otherdata/generated/${city}"
  if [[ -s "${existing}/experiment.json" ]]; then
    mkdir -p "${SUITE_ROOT}/${city}"
    ln -s "${existing}" "${target}"
    echo "[PREP] ${city}: reuse existing preparation; validate before training"
    return
  fi
  echo "ERROR: missing prepared data for ${city}: ${target} or ${existing}. No routes or GT were regenerated." >&2
  return 1
}

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
  echo "[TRAIN DONE] ${CITY} frames=${frames} GPU${gpu}"
}

run_eval_group(){
  local gpu="$1"
  shift
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
echo "Bearing-UAV ICLR SoftMS-only fair ablation"
echo "Architecture: 6x6 geometry -> forward 3x6 = 18 scored patches -> front SoftMS"
echo "              -> temporal GRU -> fixed-R Kalman -> final MeanShift -> XY"
echo "Temporal fairness: 1/2/3-frame models are trained separately on Route A."
echo "Losses: measurement=${UAVSAT_LOSS_MEASUREMENT}, next-step=${UAVSAT_LOSS_NEXT_STEP}, velocity=${UAVSAT_LOSS_VELOCITY}"
echo "Held-out B/C outputs are measured and never edited to force a ranking."
echo "================================================================================"

prepare_city "${CITY}"
common_args 0
python3 -u v39_otherdata/bearing_iclr_ablation.py check \
  "${COMMON[@]}" --train-frames 3 \
  2>&1 | tee "${SUITE_ROOT}/logs/${CITY}_preflight.log"

if [[ "${RESUME_EVAL}" == "1" ]]; then
  for f in 1 2 3; do
    ck="${SUITE_ROOT}/${CITY}/train_frames${f}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
    [[ -s "${ck}" ]] || { echo "ERROR: RESUME_EVAL=1 missing ${ck}" >&2; exit 2; }
  done
  echo "[RESUME] all 1/2/3-frame Route-A checkpoints found"
else
  # Train 3-frame first to create the shared Route-A backbone cache once.
  run_train 3 0
  ( run_train 1 5 ) & p1=$!
  ( run_train 2 6 ) & p2=$!
  status=0
  wait "${p1}" || status=1
  wait "${p2}" || status=1
  [[ "${status}" == "0" ]] || { echo "ERROR: 1/2-frame training failed" >&2; exit 20; }
fi

# Warm B/C shared caches with the true Full model before parallel evaluation.
run_eval_group 0 full

( run_eval_group 0 no_gru grid4 grid7 ) & p0=$!
( run_eval_group 5 no_kalman frames1 grid5 ) & p5=$!
( run_eval_group 6 no_ms frames2 grid8 ) & p6=$!
status=0
wait "${p0}" || status=1
wait "${p5}" || status=1
wait "${p6}" || status=1
[[ "${status}" == "0" ]] || { echo "ERROR: evaluation failed; inspect ${SUITE_ROOT}/logs" >&2; exit 21; }

python3 v39_otherdata/build_iclr_ablation_tables.py --suite-root "${SUITE_ROOT}" --cities "${CITY}"
printf '%s\n' "${SUITE_ROOT}" > v39_otherdata/LATEST_ICLR_BEARING_ABLATION.txt

if [[ "${UPLOAD_RESULTS}" == "1" ]]; then
  upload_wt="$(mktemp -d "${REPO_ROOT%/*}/uav-sat-iclr-upload-XXXXXX")"
  upload_branch="iclr-softms-results-${TS}-$$"
  cleanup(){
    git -C "${REPO_ROOT}" worktree remove --force "${upload_wt}" >/dev/null 2>&1 || true
    git -C "${REPO_ROOT}" branch -D "${upload_branch}" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT
  git fetch origin v39_otherdata
  git worktree add -b "${upload_branch}" "${upload_wt}" origin/v39_otherdata
  dest="paper_results/iclr_bearing_softms_fair_${TS}"
  mkdir -p "${upload_wt}/${dest}"
  cp "${SUITE_ROOT}"/paper_ablation_results.{json,csv} "${upload_wt}/${dest}/"
  cp "${SUITE_ROOT}"/paper_ablation_tables.{md,tex} "${upload_wt}/${dest}/"
  cp "${SUITE_ROOT}/paper_trend_audit.json" "${upload_wt}/${dest}/"
  for f in 1 2 3; do
    src="${SUITE_ROOT}/${CITY}/train_frames${f}"
    dst="${upload_wt}/${dest}/${CITY}/train_frames${f}"
    mkdir -p "${dst}"
    cp "${src}/experiment_manifest.json" "${dst}/" 2>/dev/null || true
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
    git commit -m "Add measured SoftMS-only fair Bearing ICLR ablation ${TS}"
    git fetch origin v39_otherdata
    git rebase origin/v39_otherdata
    git push origin HEAD:v39_otherdata
  )
  echo "GitHub results: ${dest}"
fi

echo "================================================================================"
echo "DONE"
echo "Suite : ${SUITE_ROOT}"
echo "Tables: ${SUITE_ROOT}/paper_ablation_tables.md"
echo "Audit : ${SUITE_ROOT}/paper_trend_audit.json"
echo "================================================================================"
