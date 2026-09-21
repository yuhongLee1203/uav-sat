#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFICIAL_ROOT="${BEARING_OFFICIAL_ROOT:-/yh/study/bearinguav_updateByMe}"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${PAPER_OUTPUT_ROOT:-${ROOT}/v39_otherdata/output/run_${TS}}"
CUSTOM_ROOT="${RUN_ROOT}/frozen_v5"
OFFICIAL_OUT="${RUN_ROOT}/official_bearinguav"
RUN_NAVIGATION="${RUN_NAVIGATION:-0}"

export OMP_NUM_THREADS="${CPU_THREADS_PER_TASK:-2}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_TASK:-2}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_TASK:-2}"
export BEARING_NUM_WORKERS="${BEARING_NUM_WORKERS:-2}"
export TOKENIZERS_PARALLELISM=false

[[ -d "${DATASET_ROOT}" ]] || { echo "ERROR: dataset missing: ${DATASET_ROOT}" >&2; exit 2; }
[[ -f "${OFFICIAL_ROOT}/Bearing_UAV/cross_view/best_model.pth" ]] || { echo "ERROR: official cross-view checkpoint missing" >&2; exit 2; }
[[ -f "${OFFICIAL_ROOT}/Bearing_UAV/satellite_view/best_model.pth" ]] || { echo "ERROR: official satellite checkpoint missing" >&2; exit 2; }
mkdir -p "${RUN_ROOT}/logs" "${OFFICIAL_OUT}/models/cross_view" "${OFFICIAL_OUT}/models/satellite_view"
ln -sfn "${OFFICIAL_ROOT}/Bearing_UAV/cross_view/best_model.pth" "${OFFICIAL_OUT}/models/cross_view/best_model.pth"
ln -sfn "${OFFICIAL_ROOT}/Bearing_UAV/satellite_view/best_model.pth" "${OFFICIAL_OUT}/models/satellite_view/best_model.pth"
ln -sfn "${OFFICIAL_ROOT}/Bearing_UAV/cross_view/training_configure.json" "${OFFICIAL_OUT}/models/cross_view/training_configure.json"
ln -sfn "${OFFICIAL_ROOT}/Bearing_UAV/satellite_view/training_configure.json" "${OFFICIAL_OUT}/models/satellite_view/training_configure.json"

printf '%s\n' "${RUN_ROOT}" > "${ROOT}/v39_otherdata/output/LATEST_RUN.txt"
echo "[1/4] Frozen V5 City A-D + core/temporal/grid/search/decoder accuracy"
echo "Training schedule: temporal_epochs=${TEMPORAL_EPOCHS:-100}, patience=${PATIENCE:-4}, lr=${UAVSAT_TEMPORAL_LR:-8e-5}"
BEARING_DATASET_ROOT="${DATASET_ROOT}" PAPER_SUITE_ROOT="${CUSTOM_ROOT}" \
UPLOAD_RESULTS=0 FROZEN_SEQUENTIAL=0 CPU_THREADS_PER_TASK="${CPU_THREADS_PER_TASK:-2}" \
RESUME_EXISTING="${RESUME_EXISTING:-0}" \
bash "${ROOT}/v39_otherdata/run_frozen_v5_all_paper_v2.sh" \
  2>&1 | tee "${RUN_ROOT}/logs/frozen_v5_all.log"

echo "[2/4] Official Bearing-UAV geo-localization: Sat GPU5, UAV GPU6"
(
  cd "${OFFICIAL_ROOT}"
  CUDA_VISIBLE_DEVICES=5 python3 -m cvphr.test.cvphr_test --rsi_id 96 --n_sample 100 --is_3d 0 \
    --device_id 0 --factor_bslr "${OFFICIAL_BATCH_FACTOR:-0.5}" \
    --bestpth_dir "${OFFICIAL_OUT}/models/satellite_view"
) 2>&1 | tee "${RUN_ROOT}/logs/official_sat_geo.log" & p_sat=$!
(
  cd "${OFFICIAL_ROOT}"
  CUDA_VISIBLE_DEVICES=6 python3 -m cvphr.test.cvphr_test --rsi_id 96 --n_sample 100 --is_3d 1 \
    --device_id 0 --factor_bslr "${OFFICIAL_BATCH_FACTOR:-0.5}" \
    --bestpth_dir "${OFFICIAL_OUT}/models/cross_view"
) 2>&1 | tee "${RUN_ROOT}/logs/official_uav_geo.log" & p_uav=$!
CUDA_VISIBLE_DEVICES=0 python3 "${ROOT}/v39_otherdata/benchmark_decoder_aggregation.py" \
  --device cuda:0 --output-dir "${CUSTOM_ROOT}/paper_bundle/decoder_microbenchmark" \
  2>&1 | tee "${RUN_ROOT}/logs/decoder_microbenchmark.log" & p_bench=$!
status=0; wait "${p_sat}" || status=1; wait "${p_uav}" || status=1; wait "${p_bench}" || status=1
[[ "${status}" == 0 ]] || { echo "ERROR: official geo-localization or decoder benchmark failed" >&2; exit 30; }
python3 "${ROOT}/v39_otherdata/collect_official_bearinguav_metrics.py" \
  --sat-root "${OFFICIAL_OUT}/models/satellite_view" --uav-root "${OFFICIAL_OUT}/models/cross_view" \
  --output-dir "${OFFICIAL_OUT}/paper_bundle"

if [[ "${RUN_NAVIGATION}" == 1 ]]; then
  echo "[3/4] Official Bearing-Naver: 4 cities x nav50/nav51, Sat GPU5, UAV GPU6"
  nav_common=(--th_arrive 20 --uav_step 25 --rsi_id 34bc 36bc 37bc 38bc --traj_id 50 51
    --project_dir "${OFFICIAL_ROOT}" --cvphr_3d_best_model_dir "${OFFICIAL_OUT}/models/cross_view"
    --cvphr_2d_best_model_dir "${OFFICIAL_OUT}/models/satellite_view")
  (
    cd "${OFFICIAL_ROOT}"
    BEARING_NAV_OUTPUT_ROOT="${OFFICIAL_OUT}/navigation" CUDA_VISIBLE_DEVICES=5 python3 -m naver.runners.nav "${nav_common[@]}" --uav_2d3d 2d
  ) 2>&1 | tee "${RUN_ROOT}/logs/official_sat_navigation.log" & p_nav_sat=$!
  (
    cd "${OFFICIAL_ROOT}"
    BEARING_NAV_OUTPUT_ROOT="${OFFICIAL_OUT}/navigation" CUDA_VISIBLE_DEVICES=6 python3 -m naver.runners.nav "${nav_common[@]}" --uav_2d3d 3d
  ) 2>&1 | tee "${RUN_ROOT}/logs/official_uav_navigation.log" & p_nav_uav=$!
  status=0; wait "${p_nav_sat}" || status=1; wait "${p_nav_uav}" || status=1
  [[ "${status}" == 0 ]] || { echo "ERROR: official navigation failed" >&2; exit 31; }
  python3 "${ROOT}/v39_otherdata/collect_official_navigation_metrics.py" \
    --repo-root "${OFFICIAL_OUT}/navigation" --output-dir "${OFFICIAL_OUT}/paper_bundle"
else
  echo "[3/4] Navigation skipped (RUN_NAVIGATION=0)"
fi

echo "[4/4] Final output inventory"
find "${RUN_ROOT}" -type f | sort > "${RUN_ROOT}/OUTPUT_MANIFEST.txt"
echo "DONE: ${RUN_ROOT}"
echo "Custom tables/figures : ${CUSTOM_ROOT}/paper_bundle"
echo "Official tables/plots : ${OFFICIAL_OUT}/paper_bundle"
