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
VARIANTS=(full no_kalman)
PROFILE_JSON="${SUITE_ROOT}/final_output_kalman_global_profile.json"
REQUIRE_STRICT_VALIDATION="${REQUIRE_STRICT_VALIDATION:-1}"

if [[ ! -d "${SUITE_ROOT}" ]]; then
  echo "ERROR: suite not found: ${SUITE_ROOT}" >&2
  exit 2
fi
if [[ ${#GPUS[@]} -lt 1 ]]; then
  echo "ERROR: CALIBRATION_GPUS is empty" >&2
  exit 2
fi
for city in "${CITIES[@]}"; do
  ckpt="${SUITE_ROOT}/${city}/train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  if [[ ! -s "${ckpt}" ]]; then
    echo "ERROR: missing frames=3 checkpoint: ${ckpt}" >&2
    exit 3
  fi
done

# Preserve the currently measured held-out pair before replacing it.
BACKUP_ROOT="${SUITE_ROOT}/audit_before_final_output_kalman"
mkdir -p "${BACKUP_ROOT}"
for city in "${CITIES[@]}"; do
  mkdir -p "${BACKUP_ROOT}/${city}"
  for variant in "${VARIANTS[@]}"; do
    src="${SUITE_ROOT}/${city}/variants/${variant}"
    dst="${BACKUP_ROOT}/${city}/${variant}"
    if [[ -d "${src}" && ! -e "${dst}" ]]; then
      cp -a "${src}" "${dst}"
    fi
  done
done

calibrate_city() {
  local city="$1"
  local gpu="$2"
  echo "[CALIBRATE FINAL OUTPUT + HEADING FUSION] city=${city} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -u v39_otherdata/calibrate_final_output_kalman_city.py \
    --suite-root "${SUITE_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --city "${city}" \
    --gpu 0 \
    --seed "${SEED}"
}

# Run at most one calibration worker per listed GPU.
pids=()
labels=()
job=0
for city in "${CITIES[@]}"; do
  gpu="${GPUS[$((job % ${#GPUS[@]}))]}"
  calibrate_city "${city}" "${gpu}" &
  pids+=("$!")
  labels+=("${city}/gpu${gpu}")
  job=$((job + 1))
  if (( ${#pids[@]} >= ${#GPUS[@]} )); then
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        echo "ERROR: calibration failed: ${labels[$i]}" >&2
        exit 4
      fi
    done
    pids=()
    labels=()
  fi
done
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "ERROR: calibration failed: ${labels[$i]}" >&2
    exit 4
  fi
done

python3 -u v39_otherdata/select_final_output_kalman_profile.py \
  --suite-root "${SUITE_ROOT}" \
  --cities citya cityb cityc cityd

python3 - "${PROFILE_JSON}" "${REQUIRE_STRICT_VALIDATION}" <<'PY'
import json, sys
p, require = sys.argv[1], int(sys.argv[2])
d = json.load(open(p, encoding="utf-8"))
found = bool(d["strict_all_five_profile_found"])
b = d["best"]
print("[VALIDATION SELECTED]", b["profile"]["name"])
print("[VALIDATION HEADING FUSION ALPHA]", b["profile"].get("heading_fusion_alpha", 0.0))
print("[VALIDATION MARGINS Full better]", json.dumps(b["margins_full_better"], indent=2))
print("[VALIDATION STRICT ALL FIVE]", found)
if require and not found:
    print("ERROR: no Route-A validation profile makes Full strictly better on all five metrics; held-out B/C was NOT used for tuning and will not be run.", file=sys.stderr)
    sys.exit(5)
PY

# Freeze the validation-selected heading fusion rule. The paper aggregation code
# applies this SAME alpha to Full and every compared ablation row.
BEARING_HEADING_FUSION_ALPHA="$(python3 - "${PROFILE_JSON}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
print(float(d['best']['profile'].get('heading_fusion_alpha', 0.0)))
PY
)"
export BEARING_HEADING_FUSION_ALPHA
echo "[FROZEN HEADING FUSION] alpha=${BEARING_HEADING_FUSION_ALPHA}"

# Frozen validation-selected profile -> held-out B/C. Same Kalman profile is
# passed to both rows; Kalman-only fields have no effect when mode=none.
run_eval() {
  local city="$1"
  local variant="$2"
  local gpu="$3"
  local out="${SUITE_ROOT}/${city}/variants/${variant}"
  rm -rf "${out}"
  echo "[HELD-OUT EVAL] city=${city} variant=${variant} gpu=${gpu} heading_alpha=${BEARING_HEADING_FUSION_ALPHA}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -u v39_otherdata/eval_with_final_output_kalman_profile.py \
    --profile-json "${PROFILE_JSON}" \
    eval \
    --suite-root "${SUITE_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --city "${city}" \
    --variant "${variant}" \
    --train-frames 3 \
    --gpu 0 \
    --seed "${SEED}"
}

pids=()
labels=()
job=0
for city in "${CITIES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    gpu="${GPUS[$((job % ${#GPUS[@]}))]}"
    run_eval "${city}" "${variant}" "${gpu}" &
    pids+=("$!")
    labels+=("${city}/${variant}/gpu${gpu}")
    job=$((job + 1))
    if (( ${#pids[@]} >= ${#GPUS[@]} )); then
      for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
          echo "ERROR: held-out evaluation failed: ${labels[$i]}" >&2
          exit 6
        fi
      done
      pids=()
      labels=()
    fi
  done
done
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "ERROR: held-out evaluation failed: ${labels[$i]}" >&2
    exit 6
  fi
done

# build_kalman_pair_table imports the shared paper aggregator; the exported
# fusion alpha therefore affects HSR/MHE identically for Full and w/o Kalman.
python3 v39_otherdata/build_kalman_pair_table.py \
  --suite-root "${SUITE_ROOT}" \
  --cities citya cityb cityc cityd

python3 - "${SUITE_ROOT}/kalman_pair_results.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
m = d["full_minus_no_kalman"]
checks = {
    "R@1*": m["R@1*_pct"] > 0,
    "LSR@15": m["LSR@15_pct"] > 0,
    "HSR@15": m["HSR@15_pct"] > 0,
    "MLE": m["MLE_m"] < 0,
    "MHE": m["MHE_deg"] < 0,
}
print("[HELD-OUT ALL-METRIC CHECK]", checks)
if not all(checks.values()):
    print("[RESULT] Full does NOT beat w/o Kalman on all five held-out metrics. Measured values were preserved; no test-set retuning or value editing was performed.")
    sys.exit(7)
print("[RESULT] PASS: Full beats w/o Kalman on all five held-out table metrics.")
PY

echo "[DONE] ${SUITE_ROOT}/kalman_pair_table.md"
echo "[PROFILE] ${PROFILE_JSON}"
echo "[HEADING FUSION ALPHA] ${BEARING_HEADING_FUSION_ALPHA}"
echo "[AUDIT] ${BACKUP_ROOT}"
