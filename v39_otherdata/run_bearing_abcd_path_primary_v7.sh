#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
SUITE_ROOT="${ABCD_SUITE_ROOT:?set ABCD_SUITE_ROOT}"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
export OMP_NUM_THREADS="${CPU_THREADS:-2}" MKL_NUM_THREADS="${CPU_THREADS:-2}" OPENBLAS_NUM_THREADS="${CPU_THREADS:-2}"
CITIES=(citya cityb cityc cityd)
VARIANTS=(path_full path_no_gru path_no_ms path_frames1 path_frames2 full)
GPUS=(0 5 6)
mkdir -p "${SUITE_ROOT}/logs/path_primary_v7"

python3 -m py_compile v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/build_iclr_ablation_tables.py v39_otherdata/build_path_primary_tables.py

calibrate() {
  local city="$1" gpu="$2"
  python3 -u v39_otherdata/bearing_iclr_ablation.py calibrate \
    --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" \
    --city "${city}" --gpu "${gpu}" --train-frames 3 --seed "${SEED:-2033}" \
    >"${SUITE_ROOT}/logs/path_primary_v7/calibrate_${city}.log" 2>&1
}
calibrate citya 0 & p0=$!
calibrate cityb 5 & p1=$!
calibrate cityc 6 & p2=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"
calibrate cityd 0

for city in "${CITIES[@]}"; do
  p="${SUITE_ROOT}/${city}/train_frames3/kalman_calibration.json"
  python3 - "${p}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); b=x['best']
assert x['search']=='positive_three_frame_path_primary_v7'
assert x['held_out_navigation_read'] is False
assert b['temporal_3frame_scale'] > 0 and b['delta2_scale'] > 0
print('[PATH-V7 CALIBRATION PASS]',x['city'],'scale=',b['temporal_3frame_scale'],'delta2=',b['delta2_scale'],'val_mle=',b['val_mle_m'])
PY
done

jobs=()
for city in "${CITIES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    case "${variant}" in path_frames1) frames=1;; path_frames2) frames=2;; *) frames=3;; esac
    gpu="${GPUS[${#jobs[@]}]}"
    python3 -u v39_otherdata/bearing_iclr_ablation.py eval \
      --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" \
      --city "${city}" --variant "${variant}" --train-frames "${frames}" \
      --gpu "${gpu}" --seed "${SEED:-2033}" \
      >"${SUITE_ROOT}/logs/path_primary_v7/eval_${city}_${variant}.log" 2>&1 &
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
python3 v39_otherdata/build_path_primary_tables.py \
  --suite-root "${SUITE_ROOT}" --cities citya cityb cityc cityd \
  | tee "${SUITE_ROOT}/logs/path_primary_v7/table.log"
echo "[DONE] ${SUITE_ROOT}/paper_path_primary_tables.md"
