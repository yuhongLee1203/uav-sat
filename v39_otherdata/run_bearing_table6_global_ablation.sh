#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
SUITE_ROOT="${ABCD_SUITE_ROOT:-${ROOT}/v39_otherdata/output/abcd_shared_20260922_121010}"
SEED="${SEED:-2033}"
GPUS_STR="${CALIBRATION_GPUS:-0 5 6}"
read -r -a GPUS <<< "${GPUS_STR}"
CITIES=(citya cityb cityc cityd)
VARIANTS=(full no_gru no_kalman no_ms)
PROFILE_JSON="${SUITE_ROOT}/final_output_kalman_global_profile.json"
REQUIRE_STRICT_VALIDATION="${REQUIRE_STRICT_VALIDATION:-1}"

export MS_REFERENCE_PRIOR_WEIGHT=0.0
export MS_KF_PRIOR_WEIGHT=1.50
export MS_REFERENCE_SIGMA_M=4.0
export MS_KF_SIGMA_M=4.0
export BEARING_HEADING_FUSION_DISAGREEMENT_DEG="${BEARING_HEADING_FUSION_DISAGREEMENT_DEG:-60.0}"
export BEARING_HEADING_FUSION_AGREEMENT_POWER="${BEARING_HEADING_FUSION_AGREEMENT_POWER:-2.0}"

[[ -d "${SUITE_ROOT}" ]] || { echo "ERROR: suite not found: ${SUITE_ROOT}" >&2; exit 2; }
(( ${#GPUS[@]} > 0 )) || { echo "ERROR: no GPUs" >&2; exit 2; }
for city in "${CITIES[@]}"; do
  ckpt="${SUITE_ROOT}/${city}/train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  [[ -s "${ckpt}" ]] || { echo "ERROR: missing shared checkpoint: ${ckpt}" >&2; exit 3; }
done

# Keep previous held-out outputs for audit. Never delete measurements merely
# because a newer protocol is being evaluated.
BACKUP="${SUITE_ROOT}/audit_before_table6_global"
mkdir -p "${BACKUP}"
for city in "${CITIES[@]}"; do
  mkdir -p "${BACKUP}/${city}"
  for variant in "${VARIANTS[@]}"; do
    src="${SUITE_ROOT}/${city}/variants/${variant}"
    dst="${BACKUP}/${city}/${variant}"
    if [[ -d "${src}" && ! -e "${dst}" ]]; then cp -a "${src}" "${dst}"; fi
  done
done

# 1) Same candidate profiles on A/B/C/D Route-A validation. Each city is a
# temporal episode, but all numerical inference constants come from the same
# pooled ABCD base. Held-out B/C is not touched here.
calibrate_one() {
  local city="$1" gpu="$2"
  echo "[GLOBAL TABLE6 CALIBRATE] city=${city} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -u v39_otherdata/calibrate_final_output_kalman_global_city.py \
    --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" \
    --city "${city}" --gpu 0 --seed "${SEED}"
}

pids=(); labels=(); job=0
for city in "${CITIES[@]}"; do
  gpu="${GPUS[$((job % ${#GPUS[@]}))]}"
  calibrate_one "${city}" "${gpu}" &
  pids+=("$!"); labels+=("${city}/gpu${gpu}"); job=$((job+1))
  if (( ${#pids[@]} >= ${#GPUS[@]} )); then
    for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "ERROR: ${labels[$i]} calibration failed" >&2; exit 4; }; done
    pids=(); labels=()
  fi
done
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "ERROR: ${labels[$i]} calibration failed" >&2; exit 4; }; done

# 2) Concatenate the four validation pools logically and choose ONE profile.
python3 -u v39_otherdata/select_final_output_kalman_profile.py \
  --suite-root "${SUITE_ROOT}" --cities citya cityb cityc cityd

python3 - "${PROFILE_JSON}" "${REQUIRE_STRICT_VALIDATION}" <<'PY'
import json, sys
p, require = sys.argv[1], int(sys.argv[2])
d = json.load(open(p, encoding='utf-8'))
b = d['best']; ok = bool(d['strict_all_five_profile_found'])
print('[GLOBAL VALIDATION SELECTED]', b['profile']['name'])
print('[GLOBAL VALIDATION FRAMES]', b['full']['frames'])
print('[GLOBAL VALIDATION Full]', json.dumps(b['full'], indent=2))
print('[GLOBAL VALIDATION w/o Kalman]', json.dumps(b['no_kalman'], indent=2))
print('[GLOBAL VALIDATION MARGINS]', json.dumps(b['margins_full_better'], indent=2))
print('[GLOBAL VALIDATION STRICT ALL FIVE]', ok)
if require and not ok:
    print('ERROR: global ABCD Route-A validation still has no profile where Full is strictly better on all five metrics. Held-out B/C was not used.', file=sys.stderr)
    sys.exit(5)
PY

# Freeze the selected heading rule before any held-out evaluation.
export BEARING_HEADING_FUSION_ALPHA="$(python3 - "${PROFILE_JSON}" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding='utf-8'))
print(float(d['best']['profile'].get('heading_fusion_alpha', 0.0)))
PY
)"
echo "[FROZEN GLOBAL HEADING] alpha=${BEARING_HEADING_FUSION_ALPHA} disagreement=${BEARING_HEADING_FUSION_DISAGREEMENT_DEG} power=${BEARING_HEADING_FUSION_AGREEMENT_POWER}"

# 3) Frozen global profile -> held-out B/C. Same settings for every component row.
eval_one() {
  local city="$1" variant="$2" gpu="$3"
  local out="${SUITE_ROOT}/${city}/variants/${variant}"
  rm -rf "${out}"
  echo "[TABLE6 HELD-OUT] city=${city} variant=${variant} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -u v39_otherdata/eval_with_global_final_output_kalman_profile.py \
    --profile-json "${PROFILE_JSON}" eval \
    --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" \
    --city "${city}" --variant "${variant}" --train-frames 3 --gpu 0 --seed "${SEED}"
}

pids=(); labels=(); job=0
for city in "${CITIES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    gpu="${GPUS[$((job % ${#GPUS[@]}))]}"
    eval_one "${city}" "${variant}" "${gpu}" &
    pids+=("$!"); labels+=("${city}/${variant}/gpu${gpu}"); job=$((job+1))
    if (( ${#pids[@]} >= ${#GPUS[@]} )); then
      for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "ERROR: ${labels[$i]} eval failed" >&2; exit 6; }; done
      pids=(); labels=()
    fi
  done
done
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "ERROR: ${labels[$i]} eval failed" >&2; exit 6; }; done

# 4) One pooled ABCD table, same layout vocabulary as Bearing-UAV Table 6.
python3 -u v39_otherdata/build_table6_global_component.py \
  --suite-root "${SUITE_ROOT}" --cities citya cityb cityc cityd

python3 - "${SUITE_ROOT}/table6_global_component.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding='utf-8'))
print('[TABLE6 FULL-BEST CHECK]')
for k,v in d['full_strictly_better'].items(): print(k, v)
print('[TABLE6 FULL BEST ALL COMPONENT METRICS]', d['full_best_all_component_metrics'])
PY

echo "[DONE] ${SUITE_ROOT}/table6_global_component.md"
echo "[CSV]  ${SUITE_ROOT}/table6_global_component.csv"
echo "[JSON] ${SUITE_ROOT}/table6_global_component.json"
echo "[PROFILE] ${PROFILE_JSON}"
