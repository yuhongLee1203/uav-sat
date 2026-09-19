#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CITY="${CITY:-citya}"
case "${CITY}" in citya|cityb|cityc|cityd) ;; *) echo "ERROR: invalid CITY=${CITY}" >&2; exit 2;; esac
SUITE_ROOT="${ICLR_SUITE_ROOT:-${REPO_ROOT}/v39_otherdata/iclr_bearing_ablation_${TS}}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-80}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
PATIENCE="${PATIENCE:-4}"
SEED="${SEED:-2033}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"
VARIANTS=(full no_gru no_kalman no_ms frames1 frames2 grid4 grid5 grid7 grid8)
mkdir -p "${SUITE_ROOT}/logs"

python3 -m py_compile \
  v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/build_iclr_ablation_tables.py \
  v39_otherdata/bearing_prepare_multicity.py \
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

run_phase(){
  local phase="$1" gpu="$2"
  shift 2
  local city="${CITY}"
  local common=(
    --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}"
    --city "${city}" --gpu "${gpu}" --backbone mobilenet_v3_small
    --visual-epochs "${VISUAL_EPOCHS}" --temporal-epochs "${TEMPORAL_EPOCHS}"
    --epochs-per-route "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}"
    --jitter-m 8 --max-sample-distance-m 15
    --heading-weight-px-per-deg 0 --ms-bandwidth-m 7 --seed "${SEED}"
  )
  if [[ "${phase}" == "train" ]]; then
    echo "[TRAIN START] ${city} GPU${gpu}"
    python3 -u v39_otherdata/bearing_iclr_ablation.py train "${common[@]}" \
      2>&1 | tee "${SUITE_ROOT}/logs/${city}_train.log"
    return
  fi
  local variant
  for variant in "$@"; do
    echo "[EVAL START] ${city} ${variant} GPU${gpu}"
    python3 -u v39_otherdata/bearing_iclr_ablation.py eval \
      "${common[@]}" --variant "${variant}" \
      2>&1 | tee "${SUITE_ROOT}/logs/${city}_${variant}.log"
  done
}

echo "================================================================================"
echo "Bearing-UAV ICLR ablation"
echo "ONE CITY: ${CITY} | reuse existing prepared data and GT"
echo "Train/full baseline: GPU0 | remaining ablations: GPU0/5/6"
echo "Chain: Forward18 -> 3-frame GRU -> fixed Kalman -> one final 6x6 MeanShift"
echo "Results are measured and never edited to force Full to win."
echo "================================================================================"

prepare_city "${CITY}"
  python3 -u v39_otherdata/bearing_iclr_ablation.py check \
    --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" \
    --city "${CITY}" --temporal-epochs "${TEMPORAL_EPOCHS}" \
    2>&1 | tee "${SUITE_ROOT}/logs/${CITY}_preflight.log"

run_phase train 0
# Warm both evaluation sequences' shared feature caches before parallel reads.
run_phase eval 0 full

( run_phase eval 0 no_gru grid4 grid7 ) & p0=$!
( run_phase eval 5 no_kalman frames1 grid5 ) & p5=$!
( run_phase eval 6 no_ms frames2 grid8 ) & p6=$!
status=0
wait "${p0}" || status=1
wait "${p5}" || status=1
wait "${p6}" || status=1
[[ "${status}" == "0" ]] || { echo "ERROR: city run failed; inspect ${SUITE_ROOT}/logs" >&2; exit 20; }

python3 v39_otherdata/build_iclr_ablation_tables.py --suite-root "${SUITE_ROOT}" --cities "${CITY}"
printf '%s\n' "${SUITE_ROOT}" > v39_otherdata/LATEST_ICLR_BEARING_ABLATION.txt

if [[ "${UPLOAD_RESULTS}" == "1" ]]; then
  upload_wt="$(mktemp -d "${REPO_ROOT%/*}/uav-sat-iclr-upload-XXXXXX")"
  upload_branch="iclr-results-${TS}-$$"
  cleanup(){
    git -C "${REPO_ROOT}" worktree remove --force "${upload_wt}" >/dev/null 2>&1 || true
    git -C "${REPO_ROOT}" branch -D "${upload_branch}" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT
  git fetch origin v39_otherdata
  git worktree add -b "${upload_branch}" "${upload_wt}" origin/v39_otherdata
  dest="paper_results/iclr_bearing_ablation_${TS}"
  mkdir -p "${upload_wt}/${dest}"
  cp "${SUITE_ROOT}"/paper_ablation_results.{json,csv} "${upload_wt}/${dest}/"
  cp "${SUITE_ROOT}"/paper_ablation_tables.{md,tex} "${upload_wt}/${dest}/"
  cp "${SUITE_ROOT}/paper_trend_audit.json" "${upload_wt}/${dest}/"
  for city in "${CITY}"; do
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
    git commit -m "Add measured Bearing-UAV ICLR ablation ${TS}"
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
echo "LaTeX : ${SUITE_ROOT}/paper_ablation_tables.tex"
echo "Audit : ${SUITE_ROOT}/paper_trend_audit.json"
echo "================================================================================"
