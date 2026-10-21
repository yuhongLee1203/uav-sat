#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BASE_SUITE="${ABCD_BASE_SUITE_ROOT:-${ROOT}/v39_otherdata/output/abcd_shared_20260922_121010}"
OUTPUT_SUITE="${ABCD_EPISODE_SUITE_ROOT:-${ROOT}/v39_otherdata/output/abcd_episode_component_$(date +%Y%m%d_%H%M%S)}"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"

for city in citya cityb cityc cityd; do
  route="${BASE_SUITE}/${city}/prepared/routes/train_01/manifest.csv"
  [[ -s "${route}" ]] || { echo "ERROR: missing training episode: ${route}" >&2; exit 2; }
done

# One shared model/optimizer is updated on a deterministic shuffled city order
# per epoch. The runtime resets recurrent/filter state at every route call.
BEARING_DATASET_ROOT="${DATASET_ROOT}" \
ABCD_BASE_SUITE_ROOT="${BASE_SUITE}" \
ABCD_ROUTE_REF_SUITE_ROOT="${OUTPUT_SUITE}" \
BEARING_FEATURE_CACHE_ROOT="${BEARING_FEATURE_CACHE_ROOT:-${BASE_SUITE}}" \
ROUTE_REF_ADAPT_ROUNDS="${ABCD_EPISODE_EPOCHS:-4}" \
ROUTE_REF_VISUAL_EPOCHS_PER_CITY=0 \
ROUTE_REF_TEMPORAL_EPOCHS_PER_CITY=1 \
ROUTE_REF_MAX_TRAINVAL_MLE_M="${ROUTE_REF_MAX_TRAINVAL_MLE_M:-0}" \
REQUIRE_MLE_VALIDATION="${REQUIRE_MLE_VALIDATION:-0}" \
REQUIRE_STRICT_VALIDATION="${REQUIRE_STRICT_VALIDATION:-0}" \
CALIBRATION_GPUS="${CALIBRATION_GPUS:-0 5 6}" \
TRAIN_GPU="${TRAIN_GPU:-0}" \
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}" \
KALMAN_CALIBRATION_MODE="${KALMAN_CALIBRATION_MODE:-fast}" \
bash v39_otherdata/run_bearing_table6_route_reference.sh

echo "[ABCD EPISODE COMPONENT DONE] ${OUTPUT_SUITE}/table6_global_component.md"
