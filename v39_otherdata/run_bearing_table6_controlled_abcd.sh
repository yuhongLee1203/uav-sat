#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BASE_SUITE="${ABCD_BASE_SUITE_ROOT:-${ROOT}/v39_otherdata/output/abcd_shared_20260922_121010}"
SUITE_ROOT="${ABCD_CONTROLLED_SUITE_ROOT:-${ROOT}/v39_otherdata/output/abcd_table6_controlled_valid_20260923}"
CITIES=(citya cityb cityc cityd)
FILES=(
  visual_retrieval_A_only.pt
  controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt
  controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only_latest.pt
)

[[ -d "${BASE_SUITE}" ]] || { echo "ERROR: base suite missing: ${BASE_SUITE}" >&2; exit 2; }
mkdir -p "${SUITE_ROOT}"
for city in "${CITIES[@]}"; do
  base_city="${BASE_SUITE}/${city}"
  city_root="${SUITE_ROOT}/${city}"
  checkpoint_root="${city_root}/train_frames3/checkpoints"
  mkdir -p "${city_root}" "${checkpoint_root}"
  ln -sfn "${base_city}/prepared" "${city_root}/prepared"
  ln -sfn "${base_city}/feature_cache" "${city_root}/feature_cache"
  for filename in "${FILES[@]}"; do
    source="${base_city}/train_frames3/checkpoints/${filename}"
    [[ -s "${source}" ]] || { echo "ERROR: checkpoint missing: ${source}" >&2; exit 3; }
    ln -sfn "${source}" "${checkpoint_root}/${filename}"
  done
done

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  python3 -u v39_otherdata/audit_bearing_temporal_routes.py \
    --suite-root "${SUITE_ROOT}" --cities "${CITIES[@]}"
  PYTHONPATH="${ROOT}/v39_otherdata${PYTHONPATH:+:${PYTHONPATH}}" \
  KALMAN_CALIBRATION_MODE="${KALMAN_CALIBRATION_MODE:-fast}" \
  python3 - <<'PY'
import calibrate_final_output_kalman_global_city as entry
mode, profiles = entry.cal._active_profiles()
if not profiles:
    raise RuntimeError("controlled calibration selected zero profiles")
print("[CONTROLLED PREFLIGHT PASS]", mode, len(profiles), [p["name"] for p in profiles])
PY
  exit 0
fi

ABCD_SUITE_ROOT="${SUITE_ROOT}" \
BEARING_DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}" \
CALIBRATION_GPUS="${CALIBRATION_GPUS:-0 5 6}" \
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}" \
KALMAN_CALIBRATION_MODE="${KALMAN_CALIBRATION_MODE:-fast}" \
REQUIRE_STRICT_VALIDATION="${REQUIRE_STRICT_VALIDATION:-0}" \
bash v39_otherdata/run_bearing_table6_global_ablation.sh

echo "[CONTROLLED ABCD TABLE6 DONE] ${SUITE_ROOT}/table6_global_component.md"
