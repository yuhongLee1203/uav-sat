#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
BASE_SUITE="${ABCD_BASE_SUITE_ROOT:-${ROOT}/v39_otherdata/output/abcd_shared_20260922_121010}"
SUITE_ROOT="${ABCD_ROUTE_REF_SUITE_ROOT:-${ROOT}/v39_otherdata/output/abcd_table6_route_reference_20260923}"
TRAIN_GPU="${TRAIN_GPU:-0}"
GPUS_STR="${CALIBRATION_GPUS:-0 5 6}"
read -r -a GPUS <<< "${GPUS_STR}"
SEED="${SEED:-2033}"
ADAPT_ROUNDS="${ROUTE_REF_ADAPT_ROUNDS:-2}"
VISUAL_PER_CITY="${ROUTE_REF_VISUAL_EPOCHS_PER_CITY:-1}"
TEMPORAL_PER_CITY="${ROUTE_REF_TEMPORAL_EPOCHS_PER_CITY:-12}"
PATIENCE="${PATIENCE:-20}"
REQUIRE_STRICT_VALIDATION="${REQUIRE_STRICT_VALIDATION:-1}"
CITIES=(citya cityb cityc cityd)
VARIANTS=(full no_gru no_kalman no_ms)
PROFILE_JSON="${SUITE_ROOT}/final_output_kalman_global_profile.json"

export BEARING_ROUTE_REFERENCE_HYPOTHESES="${BEARING_ROUTE_REFERENCE_HYPOTHESES:-13}"
export BEARING_ROUTE_REFERENCE_BANK_RADIUS_M="${BEARING_ROUTE_REFERENCE_BANK_RADIUS_M:-60.0}"
export MS_REFERENCE_PRIOR_WEIGHT=0.0
export MS_KF_PRIOR_WEIGHT="${MS_KF_PRIOR_WEIGHT:-1.50}"
export MS_REFERENCE_SIGMA_M=4.0
export MS_KF_SIGMA_M="${MS_KF_SIGMA_M:-4.0}"
export BEARING_HEADING_FUSION_DISAGREEMENT_DEG="${BEARING_HEADING_FUSION_DISAGREEMENT_DEG:-60.0}"
export BEARING_HEADING_FUSION_AGREEMENT_POWER="${BEARING_HEADING_FUSION_AGREEMENT_POWER:-2.0}"

[[ -d "${BASE_SUITE}" ]] || { echo "ERROR: base suite missing: ${BASE_SUITE}" >&2; exit 2; }
(( ${#GPUS[@]} > 0 )) || { echo "ERROR: CALIBRATION_GPUS is empty" >&2; exit 2; }
mkdir -p "${SUITE_ROOT}/logs" "${SUITE_ROOT}/shared/frames3"

# Reuse the exact prepared routes. A temporal city is an episode; state resets
# at city boundaries. Feature cache is safe to share because visual weights and
# physical satellite geometry are unchanged.
for city in "${CITIES[@]}"; do
  mkdir -p "${SUITE_ROOT}/${city}"
  [[ -d "${BASE_SUITE}/${city}/prepared" ]] || { echo "ERROR: missing prepared ${city}" >&2; exit 3; }
  ln -sfn "${BASE_SUITE}/${city}/prepared" "${SUITE_ROOT}/${city}/prepared"
  if [[ -d "${BASE_SUITE}/${city}/feature_cache" ]]; then
    ln -sfn "${BASE_SUITE}/${city}/feature_cache" "${SUITE_ROOT}/${city}/feature_cache"
  fi
done

SHARED="${SUITE_ROOT}/shared/frames3"
VIS="visual_retrieval_A_only.pt"
LATEST="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only_latest.pt"
FINAL="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

# Initialize once from the already-trained ABCD shared model; route-reference
# adaptation below uses Route-A only and therefore does not inspect held-out B/C.
if [[ ! -s "${SHARED}/${VIS}" || ! -s "${SHARED}/${LATEST}" ]]; then
  if [[ -s "${BASE_SUITE}/shared/frames3/${VIS}" && -s "${BASE_SUITE}/shared/frames3/${LATEST}" ]]; then
    cp -f "${BASE_SUITE}/shared/frames3/${VIS}" "${SHARED}/${VIS}"
    cp -f "${BASE_SUITE}/shared/frames3/${LATEST}" "${SHARED}/${LATEST}"
    [[ -s "${BASE_SUITE}/shared/frames3/${FINAL}" ]] && cp -f "${BASE_SUITE}/shared/frames3/${FINAL}" "${SHARED}/${FINAL}"
  else
    src="${BASE_SUITE}/citya/train_frames3/checkpoints"
    cp -f "${src}/${VIS}" "${SHARED}/${VIS}"
    if [[ -s "${src}/${LATEST}" ]]; then cp -f "${src}/${LATEST}" "${SHARED}/${LATEST}"; else cp -f "${src}/${FINAL}" "${SHARED}/${LATEST}"; fi
    [[ -s "${src}/${FINAL}" ]] && cp -f "${src}/${FINAL}" "${SHARED}/${FINAL}"
  fi
fi

read -r BASE_VIS_EPOCH BASE_TEMP_EPOCH < <(python3 - "${SHARED}/${VIS}" "${SHARED}/${LATEST}" <<'PY'
import sys, torch
v=torch.load(sys.argv[1], map_location='cpu')
t=torch.load(sys.argv[2], map_location='cpu')
print(int(v.get('epoch',0)), int(t.get('epoch',0)))
PY
)
echo "[ROUTE-REF INIT] visual_epoch=${BASE_VIS_EPOCH} temporal_epoch=${BASE_TEMP_EPOCH}"

# One shared model is adapted through A->B->C->D episodes. Only Route-A is used.
episode=0
for ((round=1; round<=ADAPT_ROUNDS; round++)); do
  for city in "${CITIES[@]}"; do
    episode=$((episode+1))
    dest="${SUITE_ROOT}/${city}/train_frames3/checkpoints"
    mkdir -p "${dest}"
    cp -f "${SHARED}/${VIS}" "${dest}/${VIS}"
    cp -f "${SHARED}/${LATEST}" "${dest}/${LATEST}"
    rm -f "${dest}/${FINAL}"
    target_visual=$((BASE_VIS_EPOCH + episode * VISUAL_PER_CITY))
    target_temporal=$((BASE_TEMP_EPOCH + episode * TEMPORAL_PER_CITY))
    echo "[ROUTE-REF ADAPT] round=${round} city=${city} visual=${target_visual} temporal=${target_temporal}"
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python3 -u v39_otherdata/bearing_iclr_ablation_route_reference.py train \
      --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" --city "${city}" \
      --gpu 0 --backbone mobilenet_v3_small --train-frames 3 \
      --visual-epochs "${target_visual}" --temporal-epochs "${target_temporal}" \
      --epochs-per-route "${target_temporal}" --patience "${PATIENCE}" \
      --jitter-m 8 --max-sample-distance-m 15 --heading-weight-px-per-deg 0 \
      --ms-bandwidth-m 7 --seed "${SEED}" --no-reuse-visual --continue-train \
      2>&1 | tee "${SUITE_ROOT}/logs/route_ref_adapt_r${round}_${city}.log"
    cp -f "${dest}/${VIS}" "${SHARED}/${VIS}"
    cp -f "${dest}/${LATEST}" "${SHARED}/${LATEST}"
    cp -f "${dest}/${FINAL}" "${SHARED}/${FINAL}"
  done
done

# Bind identical learned weights to each city's own satellite gallery.
for city in "${CITIES[@]}"; do
  dest="${SUITE_ROOT}/${city}/train_frames3/checkpoints"
  mkdir -p "${dest}"
  cp -f "${SHARED}/${VIS}" "${dest}/${VIS}"
  cp -f "${SHARED}/${LATEST}" "${dest}/${LATEST}"
  cp -f "${SHARED}/${FINAL}" "${dest}/${FINAL}"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python3 -u v39_otherdata/bearing_iclr_ablation_route_reference.py rebind \
    --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" --city "${city}" \
    --train-frames 3 --gpu 0 --seed "${SEED}" \
    2>&1 | tee "${SUITE_ROOT}/logs/route_ref_rebind_${city}.log"
done

# Route-A validation only: same global candidate set on four independent city episodes.
calibrate_one() {
  local city="$1" gpu="$2"
  echo "[ROUTE-REF GLOBAL CALIBRATE] city=${city} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -u v39_otherdata/calibrate_final_output_kalman_route_reference_city.py \
    --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" \
    --city "${city}" --gpu 0 --seed "${SEED}"
}

pids=(); labels=(); job=0
for city in "${CITIES[@]}"; do
  gpu="${GPUS[$((job % ${#GPUS[@]}))]}"
  calibrate_one "${city}" "${gpu}" &
  pids+=("$!"); labels+=("${city}/gpu${gpu}"); job=$((job+1))
  if (( ${#pids[@]} >= ${#GPUS[@]} )); then
    for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "ERROR: calibration failed ${labels[$i]}" >&2; exit 4; }; done
    pids=(); labels=()
  fi
done
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "ERROR: calibration failed ${labels[$i]}" >&2; exit 4; }; done

python3 -u v39_otherdata/select_final_output_kalman_profile.py \
  --suite-root "${SUITE_ROOT}" --cities citya cityb cityc cityd

python3 - "${PROFILE_JSON}" "${REQUIRE_STRICT_VALIDATION}" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding='utf-8')); require=int(sys.argv[2])
b=d['best']; ok=bool(d['strict_all_five_profile_found'])
print('[ROUTE-REF VALIDATION SELECTED]', b['profile']['name'])
print('[ROUTE-REF VALIDATION FRAMES]', b['full']['frames'])
print('[ROUTE-REF VALIDATION Full]', json.dumps(b['full'], indent=2))
print('[ROUTE-REF VALIDATION w/o Kalman]', json.dumps(b['no_kalman'], indent=2))
print('[ROUTE-REF VALIDATION MARGINS]', json.dumps(b['margins_full_better'], indent=2))
print('[ROUTE-REF VALIDATION STRICT ALL FIVE]', ok)
if require and not ok:
    print('ERROR: route-reference Route-A validation has no all-five Full win; held-out B/C was NOT used.', file=sys.stderr)
    raise SystemExit(5)
PY

export BEARING_HEADING_FUSION_ALPHA="$(python3 - "${PROFILE_JSON}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
print(float(x['best']['profile'].get('heading_fusion_alpha',0.0)))
PY
)"
echo "[FROZEN ROUTE-REF HEADING] alpha=${BEARING_HEADING_FUSION_ALPHA}"

# Only after validation selection: evaluate B/C held-out component rows.
eval_one() {
  local city="$1" variant="$2" gpu="$3"
  local out="${SUITE_ROOT}/${city}/variants/${variant}"
  rm -rf "${out}"
  echo "[ROUTE-REF TABLE6 TEST] city=${city} variant=${variant} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -u v39_otherdata/eval_with_global_route_reference_profile.py \
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
      for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "ERROR: eval failed ${labels[$i]}" >&2; exit 6; }; done
      pids=(); labels=()
    fi
  done
done
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "ERROR: eval failed ${labels[$i]}" >&2; exit 6; }; done

python3 -u v39_otherdata/build_table6_global_component.py \
  --suite-root "${SUITE_ROOT}" --cities citya cityb cityc cityd

echo "[DONE] ${SUITE_ROOT}/table6_global_component.md"
echo "[PROFILE] ${PROFILE_JSON}"
cat "${SUITE_ROOT}/table6_global_component.md"
