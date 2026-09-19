#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
SUITE_ROOT="${ICLR_SUITE_ROOT:-${REPO_ROOT}/v39_otherdata/iclr_bearing_4city_residual_${TS}}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-80}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
PATIENCE="${PATIENCE:-10}"
SEED="${SEED:-2033}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"
RESUME_EVAL="${RESUME_EVAL:-0}"
CITIES=(citya cityb cityc cityd)
VARIANTS=(full no_gru no_kalman no_ms frames1 frames2 grid4 grid5 grid7 grid8)

case "${RESUME_EVAL}" in 0|1) ;; *) echo "ERROR: RESUME_EVAL must be 0 or 1" >&2; exit 2;; esac
mkdir -p "${SUITE_ROOT}/logs"

# Same objective for 1/2/3-frame models.  The 3-frame model alone has a real
# second temporal difference available, and acceleration supervision gives that
# information a direct purpose without changing the main architecture.
export UAVSAT_LOSS_MEASUREMENT="${UAVSAT_LOSS_MEASUREMENT:-2.0}"
export UAVSAT_LOSS_NEXT_STEP="${UAVSAT_LOSS_NEXT_STEP:-2.0}"
export UAVSAT_LOSS_VELOCITY="${UAVSAT_LOSS_VELOCITY:-0.50}"
export UAVSAT_LOSS_ACCELERATION="${UAVSAT_LOSS_ACCELERATION:-0.20}"
export UAVSAT_TEMPORAL_LR="${UAVSAT_TEMPORAL_LR:-1e-4}"
export UAVSAT_RNN_DROPOUT="${UAVSAT_RNN_DROPOUT:-0.08}"
export UAVSAT_MOTION_VEL_ALPHA="${UAVSAT_MOTION_VEL_ALPHA:-0.60}"
export UAVSAT_MOTION_STEP_ALPHA="${UAVSAT_MOTION_STEP_ALPHA:-0.65}"
export UAVSAT_MOTION_RESIDUAL_FORWARD_M="${UAVSAT_MOTION_RESIDUAL_FORWARD_M:-3.0}"
export UAVSAT_MOTION_RESIDUAL_CROSS_M="${UAVSAT_MOTION_RESIDUAL_CROSS_M:-1.5}"
export UAVSAT_MOTION_RESIDUAL_ACCEL_FORWARD_M="${UAVSAT_MOTION_RESIDUAL_ACCEL_FORWARD_M:-1.5}"
export UAVSAT_MOTION_RESIDUAL_ACCEL_CROSS_M="${UAVSAT_MOTION_RESIDUAL_ACCEL_CROSS_M:-1.0}"

# Initial Kalman profile.  After the 3-frame model is trained, a small grid is
# selected using only that city's training-sequence validation split and saved
# as kalman_calibration.json.  Held-out nav50/nav51 are never used to select it.
export UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2="${UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2:-4.0}"
export UAVSAT_KALMAN_Q_PROGRESS="${UAVSAT_KALMAN_Q_PROGRESS:-1.50}"
export UAVSAT_KALMAN_Q_CROSS="${UAVSAT_KALMAN_Q_CROSS:-0.40}"
export UAVSAT_KALMAN_Q_VELOCITY="${UAVSAT_KALMAN_Q_VELOCITY:-1.00}"
export UAVSAT_KALMAN_CONFIDENCE_POWER="${UAVSAT_KALMAN_CONFIDENCE_POWER:-0.5}"

python3 -m py_compile \
  v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/build_iclr_ablation_tables.py \
  v39_otherdata/bearing_prepare_multicity.py \
  v39_DirectFinalMS/patch_direct_finalms.py \
  v39_DirectFinalMS/patch_simple_figure_gru.py

prepare_city_fresh(){
  local city="$1"
  local target="${SUITE_ROOT}/${city}/prepared"
  if [[ "${RESUME_EVAL}" == "1" ]]; then
    [[ -s "${target}/experiment.json" ]] || {
      echo "ERROR: RESUME_EVAL=1 but prepared city is missing: ${target}" >&2
      exit 2
    }
    echo "[PREP] reuse current-suite ${city}: ${target}"
    return
  fi
  rm -rf "${target}" "${target}__building"
  mkdir -p "${SUITE_ROOT}/${city}"
  echo "================================================================================"
  echo "[FRESH BEARING PREP] ${city} from ${DATASET_ROOT}"
  echo "[FRESH BEARING PREP] old v39_otherdata/generated data is NOT reused"
  echo "================================================================================"
  python3 -u v39_otherdata/bearing_prepare_multicity.py \
    --dataset-root "${DATASET_ROOT}" \
    --city "${city}" \
    --output-root "${target}" \
    2>&1 | tee "${SUITE_ROOT}/logs/${city}_prepare.log"
  [[ -s "${target}/experiment.json" ]] || {
    echo "ERROR: fresh preparation did not create ${target}/experiment.json" >&2
    exit 3
  }
}

common_args(){
  local city="$1" gpu="$2"
  COMMON=(
    --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}"
    --city "${city}" --gpu "${gpu}" --backbone mobilenet_v3_small
    --visual-epochs "${VISUAL_EPOCHS}" --temporal-epochs "${TEMPORAL_EPOCHS}"
    --epochs-per-route "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}"
    --jitter-m 8 --max-sample-distance-m 15
    --heading-weight-px-per-deg 0 --ms-bandwidth-m 7 --seed "${SEED}"
  )
}

run_train(){
  local city="$1" frames="$2" gpu="$3"
  common_args "${city}" "${gpu}"
  echo "[TRAIN START] ${city} frames=${frames} GPU${gpu}"
  python3 -u v39_otherdata/bearing_iclr_ablation.py train \
    "${COMMON[@]}" --train-frames "${frames}" \
    2>&1 | tee "${SUITE_ROOT}/logs/${city}_train_f${frames}.log"
  echo "[TRAIN DONE] ${city} frames=${frames} GPU${gpu}"
}

run_eval_group(){
  local city="$1" gpu="$2"
  shift 2
  common_args "${city}" "${gpu}"
  local variant
  for variant in "$@"; do
    echo "[EVAL START] ${city} ${variant} GPU${gpu}"
    python3 -u v39_otherdata/bearing_iclr_ablation.py eval \
      "${COMMON[@]}" --variant "${variant}" \
      2>&1 | tee "${SUITE_ROOT}/logs/${city}_${variant}.log"
  done
}

echo "================================================================================"
echo "Bearing-UAV four-city residual-temporal ablation"
echo "Dataset domains : citya cityb cityc cityd"
echo "Held-out labels : official per-city waypoint trajectories nav50 / nav51"
echo "Architecture    : 6x6 geometry -> forward 18 -> front SoftMS"
echo "                  -> residual temporal GRU -> constrained Kalman"
echo "                  -> final MeanShift -> XY"
echo "IMPORTANT       : every city is freshly prepared from Bearing_UAV_90K"
echo "                  old generated/citya/cityb/cityc/cityd data is not reused"
echo "================================================================================"

for city in "${CITIES[@]}"; do
  prepare_city_fresh "${city}"
  common_args "${city}" 0
  python3 -u v39_otherdata/bearing_iclr_ablation.py check \
    "${COMMON[@]}" --train-frames 3 \
    2>&1 | tee "${SUITE_ROOT}/logs/${city}_preflight.log"

  if [[ "${RESUME_EVAL}" == "1" ]]; then
    for f in 1 2 3; do
      ck="${SUITE_ROOT}/${city}/train_frames${f}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
      [[ -s "${ck}" ]] || { echo "ERROR: missing ${ck}" >&2; exit 2; }
    done
    [[ -s "${SUITE_ROOT}/${city}/train_frames3/kalman_calibration.json" ]] || {
      echo "ERROR: RESUME_EVAL=1 missing train-only Kalman calibration for ${city}" >&2
      exit 2
    }
    echo "[RESUME] ${city}: 1/2/3-frame checkpoints + calibration found"
  else
    # Train frame-1 first so it creates the shared feature cache before any
    # calibration exists.  Then frame-2 and frame-3 train concurrently from the
    # same uncalibrated filter defaults.  Frame-3 writes the calibration only
    # after its training finishes, so temporal training remains fair.
    run_train "${city}" 1 0
    ( run_train "${city}" 2 5 ) & p2=$!
    ( run_train "${city}" 3 6 ) & p3=$!
    status=0
    wait "${p2}" || status=1
    wait "${p3}" || status=1
    [[ "${status}" == "0" ]] || { echo "ERROR: ${city} temporal training failed" >&2; exit 20; }
  fi

  # Warm shared held-out feature caches with Full before parallel ablations.
  run_eval_group "${city}" 0 full

  ( run_eval_group "${city}" 0 no_gru grid4 grid7 ) & p0=$!
  ( run_eval_group "${city}" 5 no_kalman frames1 grid5 ) & p5=$!
  ( run_eval_group "${city}" 6 no_ms frames2 grid8 ) & p6=$!
  status=0
  wait "${p0}" || status=1
  wait "${p5}" || status=1
  wait "${p6}" || status=1
  [[ "${status}" == "0" ]] || { echo "ERROR: ${city} evaluation failed; inspect logs" >&2; exit 21; }

done

python3 v39_otherdata/build_iclr_ablation_tables.py \
  --suite-root "${SUITE_ROOT}" \
  --cities citya cityb cityc cityd
printf '%s\n' "${SUITE_ROOT}" > v39_otherdata/LATEST_ICLR_BEARING_ABLATION.txt

if [[ "${UPLOAD_RESULTS}" == "1" ]]; then
  upload_wt="$(mktemp -d "${REPO_ROOT%/*}/uav-sat-iclr-upload-XXXXXX")"
  upload_branch="iclr-bearing4city-results-${TS}-$$"
  cleanup(){
    git -C "${REPO_ROOT}" worktree remove --force "${upload_wt}" >/dev/null 2>&1 || true
    git -C "${REPO_ROOT}" branch -D "${upload_branch}" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT
  git fetch origin v39_otherdata
  git worktree add -b "${upload_branch}" "${upload_wt}" origin/v39_otherdata
  dest="paper_results/iclr_bearing_4city_residual_${TS}"
  mkdir -p "${upload_wt}/${dest}"
  cp "${SUITE_ROOT}"/paper_ablation_results.{json,csv} "${upload_wt}/${dest}/"
  cp "${SUITE_ROOT}"/paper_ablation_tables.{md,tex} "${upload_wt}/${dest}/"
  cp "${SUITE_ROOT}/paper_trend_audit.json" "${upload_wt}/${dest}/"

  for city in "${CITIES[@]}"; do
    cp "${SUITE_ROOT}/${city}/prepared/experiment.json" \
      "${upload_wt}/${dest}/${city}_prepared_experiment.json"
    mkdir -p "${upload_wt}/${dest}/${city}/train_frames3"
    cp "${SUITE_ROOT}/${city}/train_frames3/kalman_calibration.json" \
      "${upload_wt}/${dest}/${city}/train_frames3/" 2>/dev/null || true
    for f in 1 2 3; do
      src="${SUITE_ROOT}/${city}/train_frames${f}"
      dst="${upload_wt}/${dest}/${city}/train_frames${f}"
      mkdir -p "${dst}"
      cp "${src}/experiment_manifest.json" "${dst}/" 2>/dev/null || true
    done
    for variant in "${VARIANTS[@]}"; do
      src="${SUITE_ROOT}/${city}/variants/${variant}"
      dst="${upload_wt}/${dest}/${city}/${variant}"
      mkdir -p "${dst}"
      cp "${src}/bearing_v39_summary.json" "${dst}/"
      cp "${src}/experiment_manifest.json" "${dst}/"
      cp "${src}"/*_frames.csv "${dst}/" 2>/dev/null || true
    done
  done

  (
    cd "${upload_wt}"
    git add "${dest}"
    git commit -m "Add measured four-city residual Bearing ICLR ablation ${TS}"
    git fetch origin v39_otherdata
    git rebase origin/v39_otherdata
    git push origin HEAD:v39_otherdata
  )
  echo "GitHub results: ${dest}"
fi

echo "================================================================================"
echo "DONE: four-city Bearing-UAV experiment"
echo "Suite : ${SUITE_ROOT}"
echo "Cities: citya cityb cityc cityd"
echo "Tables: ${SUITE_ROOT}/paper_ablation_tables.md"
echo "Audit : ${SUITE_ROOT}/paper_trend_audit.json"
echo "================================================================================"
