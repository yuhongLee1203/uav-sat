#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
SUITE_ROOT="${ABCD_SUITE_ROOT:-${ROOT}/v39_otherdata/output/abcd_shared_20260922_121010}"
SEED="${SEED:-2033}"
CITIES=(citya cityb cityc cityd)
GPUS=(${CALIBRATION_GPUS:-0 5 6})

if [[ ! -d "${SUITE_ROOT}" ]]; then
  echo "ERROR: suite not found: ${SUITE_ROOT}" >&2
  exit 2
fi
if (( ${#GPUS[@]} == 0 )); then
  echo "ERROR: CALIBRATION_GPUS resolved to an empty list" >&2
  exit 2
fi

python3 -m py_compile \
  v39_otherdata/calibrate_finalms_prior_trainval.py \
  v39_otherdata/select_global_finalms_profile.py \
  v39_otherdata/build_kalman_pair_table.py \
  v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/build_iclr_ablation_tables.py

for city in "${CITIES[@]}"; do
  test -s "${SUITE_ROOT}/${city}/train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt" || {
    echo "ERROR: missing frames3 checkpoint for ${city}" >&2
    exit 3
  }
done

mkdir -p "${SUITE_ROOT}/logs/finalms_trainval"

# -----------------------------------------------------------------------------
# 1) Calibrate final-MS prior weights using Route-A validation ONLY.
#    Held-out test_01/test_02 are not read by the calibration code.
# -----------------------------------------------------------------------------
jobs=()
job_index=0
for city in "${CITIES[@]}"; do
  gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
  echo "[CALIBRATE] city=${city} gpu=${gpu}"
  python3 -u v39_otherdata/calibrate_finalms_prior_trainval.py \
    --suite-root "${SUITE_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --city "${city}" \
    --gpu "${gpu}" \
    --seed "${SEED}" \
    >"${SUITE_ROOT}/logs/finalms_trainval/${city}.log" 2>&1 &
  jobs+=("$!")
  job_index=$((job_index + 1))
  if (( ${#jobs[@]} == ${#GPUS[@]} )); then
    for pid in "${jobs[@]}"; do wait "${pid}"; done
    jobs=()
  fi
done
for pid in "${jobs[@]}"; do wait "${pid}"; done

python3 v39_otherdata/select_global_finalms_profile.py \
  --suite-root "${SUITE_ROOT}" \
  --cities citya cityb cityc cityd \
  | tee "${SUITE_ROOT}/logs/finalms_trainval/global.log"

readarray -t FINALMS_ENV < <(python3 - "${SUITE_ROOT}/finalms_global_calibration.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding='utf-8'))['best']
print(x['ms_kf_prior_weight'])
print(x['ms_reference_prior_weight'])
print(x['ms_kf_sigma_m'])
print(x['ms_reference_sigma_m'])
print(x['name'])
PY
)
export MS_KF_PRIOR_WEIGHT="${FINALMS_ENV[0]}"
export MS_REFERENCE_PRIOR_WEIGHT="${FINALMS_ENV[1]}"
export MS_KF_SIGMA_M="${FINALMS_ENV[2]}"
export MS_REFERENCE_SIGMA_M="${FINALMS_ENV[3]}"
PROFILE_NAME="${FINALMS_ENV[4]}"

echo "[GLOBAL PROFILE] ${PROFILE_NAME} | KF weight=${MS_KF_PRIOR_WEIGHT} | REF weight=${MS_REFERENCE_PRIOR_WEIGHT} | KF sigma=${MS_KF_SIGMA_M} | REF sigma=${MS_REFERENCE_SIGMA_M}"

# -----------------------------------------------------------------------------
# 2) Fair paired held-out evaluation.
#    Full and w/o-Kalman use the SAME checkpoint, seed, final-MS profile,
#    prepared routes, candidate geometry and MeanShift settings.
# -----------------------------------------------------------------------------
VARIANTS=(full no_kalman)
jobs=()
job_index=0
for city in "${CITIES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
    out="${SUITE_ROOT}/${city}/variants/${variant}"
    rm -rf "${out}"
    echo "[EVAL] city=${city} variant=${variant} gpu=${gpu} profile=${PROFILE_NAME}"
    python3 -u v39_otherdata/bearing_iclr_ablation.py eval \
      --suite-root "${SUITE_ROOT}" \
      --dataset-root "${DATASET_ROOT}" \
      --city "${city}" \
      --variant "${variant}" \
      --train-frames 3 \
      --gpu "${gpu}" \
      --seed "${SEED}" \
      >"${SUITE_ROOT}/logs/finalms_trainval/eval_${city}_${variant}.log" 2>&1 &
    jobs+=("$!")
    job_index=$((job_index + 1))
    if (( ${#jobs[@]} == ${#GPUS[@]} )); then
      for pid in "${jobs[@]}"; do wait "${pid}"; done
      jobs=()
    fi
  done
done
for pid in "${jobs[@]}"; do wait "${pid}"; done

for city in "${CITIES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    test -s "${SUITE_ROOT}/${city}/variants/${variant}/bearing_v39_summary.json" || {
      echo "ERROR: missing result ${city}/${variant}" >&2
      exit 31
    }
  done
done

# Do NOT rebuild the complete ablation table here: the other variants still
# belong to the previous decoder profile.  First prove the Kalman contribution
# with this isolated, paired comparison.  If it is useful, freeze this profile
# and rerun every paper variant under the same settings.
python3 v39_otherdata/build_kalman_pair_table.py \
  --suite-root "${SUITE_ROOT}" \
  --cities citya cityb cityc cityd \
  | tee "${SUITE_ROOT}/logs/finalms_trainval/kalman_pair_table.log"

echo "[DONE] train-validation-calibrated Full vs w/o Kalman"
echo "[PROFILE] ${SUITE_ROOT}/finalms_global_calibration.json"
echo "[PAIR TABLE] ${SUITE_ROOT}/kalman_pair_table.md"
echo "[PAIR JSON] ${SUITE_ROOT}/kalman_pair_results.json"
echo "[NOTE] Full paper ablation table was intentionally NOT rebuilt yet."
