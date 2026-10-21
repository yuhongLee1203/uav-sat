#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
SUITE_ROOT="${ABCD_SUITE_ROOT:?set ABCD_SUITE_ROOT to the completed shared suite}"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
export OMP_NUM_THREADS="${CPU_THREADS:-2}"
export MKL_NUM_THREADS="${CPU_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS:-2}"
CITIES=(citya cityb cityc cityd)
VARIANTS=(full no_gru no_kalman no_ms frames1 frames2 grid4 grid5 grid7 grid8)
GPUS=(0 5 6)
mkdir -p "${SUITE_ROOT}/logs/path_v6" "${SUITE_ROOT}/pre_path_v6_results"

python3 -m py_compile v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/build_iclr_ablation_tables.py

# Preserve every previous measured output before replacing evaluation files.
for city in "${CITIES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    src="${SUITE_ROOT}/${city}/variants/${variant}"
    dst="${SUITE_ROOT}/pre_path_v6_results/${city}/${variant}"
    mkdir -p "${dst}"
    if [[ -d "${src}" ]]; then
      find "${src}" -maxdepth 1 -type f \( -name '*.json' -o -name '*.csv' \) \
        -exec cp -n {} "${dst}/" \;
    fi
  done
done

run_calibration() {
  local city="$1" gpu="$2"
  local calibration="${SUITE_ROOT}/${city}/train_frames3/kalman_calibration.json"
  if [[ -s "${calibration}" ]] && python3 - "${calibration}" <<'PY'
import json,sys
try: x=json.load(open(sys.argv[1]))
except Exception: raise SystemExit(1)
raise SystemExit(0 if x.get('search') == 'three_frame_full_path_kalman_v6' else 1)
PY
  then
    echo "[CALIBRATION SKIP] ${city} path-v6 already complete"
    return
  fi
  python3 -u v39_otherdata/bearing_iclr_ablation.py calibrate \
    --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" \
    --city "${city}" --gpu "${gpu}" --train-frames 3 --seed "${SEED:-2033}" \
    >"${SUITE_ROOT}/logs/path_v6/calibrate_${city}.log" 2>&1
}

# Three GPUs, then the fourth city on the first available GPU.
run_calibration citya 0 & p0=$!
run_calibration cityb 5 & p1=$!
run_calibration cityc 6 & p2=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"
run_calibration cityd 0

for city in "${CITIES[@]}"; do
  calibration="${SUITE_ROOT}/${city}/train_frames3/kalman_calibration.json"
  [[ -s "${calibration}" ]] || { echo "ERROR: missing ${calibration}" >&2; exit 41; }
  python3 - "${calibration}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['search']=='three_frame_full_path_kalman_v6'
assert x['held_out_navigation_read'] is False
assert x['best']['temporal_3frame_scale'] == 1.0
assert x['best']['delta2_scale'] == 1.0
print('[CALIBRATION AUDIT] PASS',sys.argv[1],x['best']['val_mle_m'])
PY
done

jobs=()
for city in "${CITIES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    case "${variant}" in frames1) frames=1;; frames2) frames=2;; *) frames=3;; esac
    gpu="${GPUS[${#jobs[@]}]}"
    python3 -u v39_otherdata/bearing_iclr_ablation.py eval \
      --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" \
      --city "${city}" --variant "${variant}" --train-frames "${frames}" \
      --gpu "${gpu}" --seed "${SEED:-2033}" \
      >"${SUITE_ROOT}/logs/path_v6/eval_${city}_${variant}.log" 2>&1 &
    jobs+=("$!")
    if (( ${#jobs[@]} == 3 )); then
      for pid in "${jobs[@]}"; do wait "${pid}"; done
      jobs=()
    fi
  done
done
for pid in "${jobs[@]}"; do wait "${pid}"; done

for city in "${CITIES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    test -s "${SUITE_ROOT}/${city}/variants/${variant}/bearing_v39_summary.json"
  done
done

python3 v39_otherdata/build_iclr_ablation_tables.py \
  --suite-root "${SUITE_ROOT}" --cities citya cityb cityc cityd \
  | tee "${SUITE_ROOT}/logs/path_v6/table.log"
echo "[DONE] ${SUITE_ROOT}/paper_ablation_tables.md"
echo "[BACKUP] ${SUITE_ROOT}/pre_path_v6_results"
