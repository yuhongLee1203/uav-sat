#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
BASE_SUITE="${ABCD_BASE_SUITE_ROOT:-${ROOT}/v39_otherdata/output/abcd_shared_20260922_121010}"
SUITE_ROOT="${ABCD_ROUTE_REF_SUITE_ROOT:-${ROOT}/v39_otherdata/output/abcd_table6_route_reference_kalman_tuned_20260923}"
CACHE_ROOT="${BEARING_FEATURE_CACHE_ROOT:-${BASE_SUITE}}"
TRAIN_GPU="${TRAIN_GPU:-0}"
GPUS_STR="${CALIBRATION_GPUS:-0 5 6}"
read -r -a GPUS <<< "${GPUS_STR}"
SEED="${SEED:-2033}"
ADAPT_ROUNDS="${ROUTE_REF_ADAPT_ROUNDS:-0}"
VISUAL_PER_CITY="${ROUTE_REF_VISUAL_EPOCHS_PER_CITY:-1}"
TEMPORAL_PER_CITY="${ROUTE_REF_TEMPORAL_EPOCHS_PER_CITY:-12}"
PATIENCE="${PATIENCE:-20}"
REQUIRE_STRICT_VALIDATION="${REQUIRE_STRICT_VALIDATION:-0}"
REQUIRE_MLE_VALIDATION="${REQUIRE_MLE_VALIDATION:-1}"
KALMAN_CALIBRATION_MODE="${KALMAN_CALIBRATION_MODE:-fast}"
CITIES=(citya cityb cityc cityd)
VARIANTS=(full no_gru no_kalman no_ms)
PROFILE_JSON="${SUITE_ROOT}/final_output_kalman_global_profile.json"
USE_EXISTING_PREPARED="${USE_EXISTING_PREPARED:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

export OMP_NUM_THREADS="${CPU_THREADS_PER_JOB:-2}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_JOB:-2}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_JOB:-2}"

export BEARING_ROUTE_REFERENCE_HYPOTHESES="${BEARING_ROUTE_REFERENCE_HYPOTHESES:-7}"
export BEARING_ROUTE_REFERENCE_BANK_RADIUS_M="${BEARING_ROUTE_REFERENCE_BANK_RADIUS_M:-30.0}"
export UAVSAT_ACQ_RAW_VISUAL_EVIDENCE_WEIGHT="${UAVSAT_ACQ_RAW_VISUAL_EVIDENCE_WEIGHT:-0.25}"
export UAVSAT_ACQ_HYPOTHESIS_MOTION_PRIOR_WEIGHT="${UAVSAT_ACQ_HYPOTHESIS_MOTION_PRIOR_WEIGHT:-8.0}"
export MS_REFERENCE_PRIOR_WEIGHT=0.0
export MS_KF_PRIOR_WEIGHT="${MS_KF_PRIOR_WEIGHT:-1.50}"
export MS_REFERENCE_SIGMA_M=4.0
export MS_KF_SIGMA_M="${MS_KF_SIGMA_M:-4.0}"
export BEARING_HEADING_FUSION_DISAGREEMENT_DEG="${BEARING_HEADING_FUSION_DISAGREEMENT_DEG:-60.0}"
export BEARING_HEADING_FUSION_AGREEMENT_POWER="${BEARING_HEADING_FUSION_AGREEMENT_POWER:-2.0}"
export KALMAN_CALIBRATION_MODE
export BEARING_FREEZE_VISUAL=1
export BEARING_SKIP_LOCAL_KALMAN_CALIBRATION=1

# Exercise the exact two-layer calibration entry point before any GPU work.
# This catches profile-list overrides in the global wrapper immediately.
PYTHONPATH="${ROOT}/v39_otherdata${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<'PY'
import calibrate_final_output_kalman_route_reference_city as entry
mode, profiles = entry.global_cal.cal._active_profiles()
if not profiles:
    raise RuntimeError("calibration preflight selected zero profiles")
print(
    "[CALIBRATION PREFLIGHT PASS]",
    "mode=%s" % mode,
    "profiles=%d" % len(profiles),
    [p["name"] for p in profiles],
    flush=True,
)
PY

[[ -d "${BASE_SUITE}" ]] || { echo "ERROR: base suite missing: ${BASE_SUITE}" >&2; exit 2; }
(( ${#GPUS[@]} > 0 )) || { echo "ERROR: CALIBRATION_GPUS is empty" >&2; exit 2; }
mkdir -p "${SUITE_ROOT}/logs" "${SUITE_ROOT}/shared/frames3"

# Reuse the exact prepared routes. A temporal city is an episode; state resets
# at city boundaries. Keep one persistent cache per city across result runs.
for city in "${CITIES[@]}"; do
  mkdir -p "${SUITE_ROOT}/${city}"
  if (( USE_EXISTING_PREPARED )); then
    [[ -d "${SUITE_ROOT}/${city}/prepared" ]] || { echo "ERROR: missing existing prepared ${city}" >&2; exit 3; }
  else
    [[ -d "${BASE_SUITE}/${city}/prepared" ]] || { echo "ERROR: missing prepared ${city}" >&2; exit 3; }
    ln -sfn "${BASE_SUITE}/${city}/prepared" "${SUITE_ROOT}/${city}/prepared"
  fi

  # The real training/evaluation process creates missing cache files. Keeping
  # this directory stable means later runs reuse them instead of recomputing.
  cache_src="${CACHE_ROOT}/${city}/feature_cache"
  cache_dst="${SUITE_ROOT}/${city}/feature_cache"
  mkdir -p "${cache_src}"
  if [[ -L "${cache_dst}" ]]; then
    ln -sfn "${cache_src}" "${cache_dst}"
  elif [[ ! -e "${cache_dst}" ]]; then
    ln -s "${cache_src}" "${cache_dst}"
  fi
  echo "[PERSISTENT CACHE] ${city}: $(readlink -f "${cache_dst}")"
done

# A temporal tracker is invalid when independently sampled images are merely
# chained into a pseudo-route whose cadence/geometry differs from training.
# Refuse such inputs before checkpoint binding or GPU inference.
python3 -u v39_otherdata/audit_bearing_temporal_routes.py \
  --suite-root "${SUITE_ROOT}" --cities "${CITIES[@]}"

if (( PREFLIGHT_ONLY )); then
  echo "[PREFLIGHT ONLY PASS] ${SUITE_ROOT}"
  exit 0
fi

SHARED="${SUITE_ROOT}/shared/frames3"
VIS="visual_retrieval_A_only.pt"
LATEST="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only_latest.pt"
FINAL="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

# Initialize once from the already-trained ABCD shared model; route-reference
# adaptation below uses Route-A only and therefore does not inspect held-out B/C.
if [[ ! -s "${SHARED}/${VIS}" || ! -s "${SHARED}/${LATEST}" ]]; then
  if [[ -s "${BASE_SUITE}/shared/frames3/${VIS}" && -s "${BASE_SUITE}/shared/frames3/${LATEST}" ]]; then
    cp -pf "${BASE_SUITE}/shared/frames3/${VIS}" "${SHARED}/${VIS}"
    cp -pf "${BASE_SUITE}/shared/frames3/${LATEST}" "${SHARED}/${LATEST}"
    [[ -s "${BASE_SUITE}/shared/frames3/${FINAL}" ]] && cp -pf "${BASE_SUITE}/shared/frames3/${FINAL}" "${SHARED}/${FINAL}"
  else
    src="${BASE_SUITE}/citya/train_frames3/checkpoints"
    cp -pf "${src}/${VIS}" "${SHARED}/${VIS}"
    if [[ -s "${src}/${LATEST}" ]]; then cp -pf "${src}/${LATEST}" "${SHARED}/${LATEST}"; else cp -pf "${src}/${FINAL}" "${SHARED}/${LATEST}"; fi
    [[ -s "${src}/${FINAL}" ]] && cp -pf "${src}/${FINAL}" "${SHARED}/${FINAL}"
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

if (( ADAPT_ROUNDS > 0 )); then
  (( TEMPORAL_PER_CITY == 1 )) || {
    echo "ERROR: set ROUTE_REF_TEMPORAL_EPOCHS_PER_CITY=1 for epoch-shuffled episodes" >&2
    exit 3
  }
  SCHEDULE_JSON="${SHARED}/episode_schedule.json"
  python3 - "${SCHEDULE_JSON}" "${SEED}" "${ADAPT_ROUNDS}" "${BASE_TEMP_EPOCH}" <<'PY'
import json, random, sys
from pathlib import Path
path = Path(sys.argv[1])
seed, rounds, base_epoch = map(int, sys.argv[2:])
if path.exists():
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan["seed"] != seed or plan["rounds"] != rounds:
        raise SystemExit("ERROR: existing episode schedule has different seed/rounds")
else:
    episodes = []
    for epoch in range(1, rounds + 1):
        cities = ["citya", "cityb", "cityc", "cityd"]
        random.Random(seed + epoch).shuffle(cities)
        episodes.extend({"epoch": epoch, "city": city, "route": "train_01"} for city in cities)
    plan = {"seed": seed, "rounds": rounds, "base_temporal_epoch": base_epoch,
            "episodes": episodes, "state_reset": "each route", "shared_optimizer": True}
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
print("[ABCD EPISODE SCHEDULE]", path)
for epoch in range(1, rounds + 1):
    print("[ABCD EPOCH]", epoch,
          [item["city"] for item in plan["episodes"] if item["epoch"] == epoch])
PY
  read -r SCHEDULE_BASE_EPOCH COMPLETED_EPISODES < <(python3 - "${SCHEDULE_JSON}" "${SHARED}/${LATEST}" <<'PY'
import json, sys, torch
plan = json.load(open(sys.argv[1], encoding="utf-8"))
latest = torch.load(sys.argv[2], map_location="cpu")
base = int(plan["base_temporal_epoch"])
completed = int(latest.get("epoch", 0)) - base
if not 0 <= completed <= len(plan["episodes"]):
    raise SystemExit("ERROR: shared checkpoint epoch does not match episode schedule")
print(base, completed)
PY
  )
  mapfile -t EPISODE_LINES < <(python3 - "${SCHEDULE_JSON}" <<'PY'
import json, sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["episodes"]:
    print(item["epoch"], item["city"], item["route"])
PY
  )
fi

# The shared model and optimizer cross episodes. The runtime starts each call
# with fresh hidden, Kalman, previous-frame and motion state.
episode=0
if (( ADAPT_ROUNDS > 0 )); then
  for episode_line in "${EPISODE_LINES[@]}"; do
    read -r round city train_route <<< "${episode_line}"
    episode=$((episode+1))
    if (( episode <= COMPLETED_EPISODES )); then
      echo "[ABCD EPISODE REUSE] ${episode} epoch=${round} ${city}/${train_route}"
      continue
    fi
    export BEARING_EPISODE_ACQ_TEACHER_RATIO="$(python3 - "${episode}" "${#EPISODE_LINES[@]}" <<'PY'
import sys
position, total = map(int, sys.argv[1:])
print(max(0.0, 1.0 - (position - 1) / max(total - 1, 1)))
PY
    )"
    dest="${SUITE_ROOT}/${city}/train_frames3/checkpoints"
    mkdir -p "${dest}"
    cp -pf "${SHARED}/${LATEST}" "${dest}/${LATEST}"
    [[ -s "${SHARED}/${FINAL}" ]] && cp -pf "${SHARED}/${FINAL}" "${dest}/${FINAL}"
    # The carried visual weights are global, but gallery coordinates/origin are
    # city-specific. Bind them before building any route cache or training.
    if ! python3 v39_otherdata/check_bearing_visual_binding.py \
      --city-checkpoint "${dest}/${VIS}" --shared-checkpoint "${SHARED}/${VIS}" --city "${city}"; then
      cp -pf "${SHARED}/${VIS}" "${dest}/${VIS}"
      CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python3 -u v39_otherdata/bearing_iclr_ablation_route_reference.py rebind \
        --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" --city "${city}" \
        --train-frames 3 --gpu 0 --seed "${SEED}" \
        2>&1 | tee "${SUITE_ROOT}/logs/route_ref_pretrain_rebind_r${round}_${city}.log"
    fi
    rm -f "${dest}/${FINAL}"
    target_visual=$((BASE_VIS_EPOCH + episode * VISUAL_PER_CITY))
    target_temporal=$((SCHEDULE_BASE_EPOCH + episode))
    echo "[ABCD EPISODE] epoch=${round} episode=${episode}/${#EPISODE_LINES[@]} city=${city} route=${train_route} teacher=${BEARING_EPISODE_ACQ_TEACHER_RATIO}"
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python3 -u v39_otherdata/bearing_iclr_ablation_route_reference.py train \
      --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" --city "${city}" \
      --gpu 0 --backbone mobilenet_v3_small --train-frames 3 \
      --visual-epochs "${target_visual}" --temporal-epochs "${target_temporal}" \
      --epochs-per-route "${target_temporal}" --patience "${PATIENCE}" \
      --jitter-m 8 --max-sample-distance-m 15 --heading-weight-px-per-deg 0 \
      --ms-bandwidth-m 7 --seed "${SEED}" --no-reuse-visual --continue-train \
      2>&1 | tee "${SUITE_ROOT}/logs/route_ref_adapt_r${round}_${city}.log"
    python3 - "${SUITE_ROOT}/logs/route_ref_adapt_r${round}_${city}.log" "${ROUTE_REF_MAX_TRAINVAL_MLE_M:-30}" <<'PY'
import re, sys
log = open(sys.argv[1], encoding="utf-8").read()
values = [float(x) for x in re.findall(r"val_mle=([0-9.]+)m", log)]
limit = float(sys.argv[2])
if not values or (limit > 0 and min(values) > limit):
    raise SystemExit(
        f"ERROR: Route-A validation MLE best={min(values) if values else 'missing'}m "
        f"exceeds {limit}m; stopping before four-city calibration/test."
    )
print(f"[ROUTE-REF TRAINVAL GATE] best MLE={min(values):.3f}m <= {limit:.3f}m")
PY
    python3 - "${dest}/${LATEST}" "${round}" "${city}" "${train_route}" <<'PY'
import sys, torch
path, epoch, city, route = sys.argv[1:]
payload = torch.load(path, map_location="cpu")
payload["episode_optimizer_compatible"] = True
payload["episode_boundary"] = {"epoch": int(epoch), "city": city, "route": route,
                               "recurrent_state_reset": True, "kalman_state_reset": True}
torch.save(payload, path)
PY
    cp -pf "${dest}/${VIS}" "${SHARED}/${VIS}"
    cp -pf "${dest}/${LATEST}" "${SHARED}/${LATEST}"
    cp -pf "${dest}/${FINAL}" "${SHARED}/${FINAL}"
  done
fi
unset BEARING_EPISODE_ACQ_TEACHER_RATIO

# Bind identical learned weights to each city's own satellite gallery.
for city in "${CITIES[@]}"; do
  dest="${SUITE_ROOT}/${city}/train_frames3/checkpoints"
  mkdir -p "${dest}"
  cp -pf "${SHARED}/${LATEST}" "${dest}/${LATEST}"
  cp -pf "${SHARED}/${FINAL}" "${dest}/${FINAL}"
  if ! python3 v39_otherdata/check_bearing_visual_binding.py \
    --city-checkpoint "${dest}/${VIS}" --shared-checkpoint "${SHARED}/${VIS}" --city "${city}"; then
    cp -pf "${SHARED}/${VIS}" "${dest}/${VIS}"
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python3 -u v39_otherdata/bearing_iclr_ablation_route_reference.py rebind \
      --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" --city "${city}" \
      --train-frames 3 --gpu 0 --seed "${SEED}" \
      2>&1 | tee "${SUITE_ROOT}/logs/route_ref_rebind_${city}.log"
  fi
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

python3 - "${PROFILE_JSON}" "${REQUIRE_STRICT_VALIDATION}" "${REQUIRE_MLE_VALIDATION}" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding='utf-8'))
require_strict=int(sys.argv[2]); require_mle=int(sys.argv[3])
b=d['best']; ok=bool(d['strict_all_five_profile_found'])
print('[ROUTE-REF VALIDATION SELECTED]', b['profile']['name'])
print('[ROUTE-REF VALIDATION FRAMES]', b['full']['frames'])
print('[ROUTE-REF VALIDATION Full]', json.dumps(b['full'], indent=2))
print('[ROUTE-REF VALIDATION w/o Kalman]', json.dumps(b['no_kalman'], indent=2))
print('[ROUTE-REF VALIDATION MARGINS]', json.dumps(b['margins_full_better'], indent=2))
print('[ROUTE-REF VALIDATION STRICT ALL FIVE]', ok)
print('[ROUTE-REF VALIDATION MLE PROFILE FOUND]', bool(d.get('primary_mle_profile_found')))
if require_strict and not ok:
    print('ERROR: route-reference Route-A validation has no all-five Full win; held-out B/C was NOT used.', file=sys.stderr)
    raise SystemExit(5)
if require_mle and not bool(d.get('primary_mle_profile_found')):
    print('ERROR: no Route-A validation profile improves Full MLE over w/o Kalman. Held-out B/C was NOT used.', file=sys.stderr)
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
  if [[ -d "${out}" ]]; then
    local archive="${SUITE_ROOT}/audit_before_route_ref_eval/${RUN_STAMP}/${city}"
    mkdir -p "${archive}"
    mv "${out}" "${archive}/${variant}"
  fi
  echo "[ROUTE-REF TABLE6 TEST] city=${city} variant=${variant} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -u v39_otherdata/eval_with_global_route_reference_profile.py \
    --profile-json "${PROFILE_JSON}" eval \
    --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" \
    --city "${city}" --variant "${variant}" --train-frames 3 --gpu 0 --seed "${SEED}"
}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
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
