#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
SUITE_ROOT="${ABCD_SUITE_ROOT:?set ABCD_SUITE_ROOT}"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
export OMP_NUM_THREADS="${CPU_THREADS:-2}" MKL_NUM_THREADS="${CPU_THREADS:-2}" OPENBLAS_NUM_THREADS="${CPU_THREADS:-2}"
CITIES=(citya cityb cityc cityd)
# Original complete ablation layout. Kalman remains part of Full.
VARIANTS=(full no_gru no_kalman no_ms frames1 frames2 grid4 grid5 grid7 grid8)
GPUS=(0 5 6)
mkdir -p "${SUITE_ROOT}/logs/full_table_v7"

python3 -m py_compile v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/build_iclr_ablation_tables.py
for city in "${CITIES[@]}"; do
  calibration="${SUITE_ROOT}/${city}/train_frames3/kalman_calibration.json"
  python3 - "${calibration}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); b=x['best']
assert x['search']=='positive_three_frame_path_primary_v7'
assert x['held_out_navigation_read'] is False
assert b['temporal_3frame_scale'] > 0 and b['delta2_scale'] > 0
print('[FULL-V7 PROFILE PASS]',x['city'],b['temporal_3frame_scale'],b['delta2_scale'])
PY
done

jobs=()
for city in "${CITIES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    # Full was already evaluated by path-primary-v7, but running it again keeps
    # this final table self-contained and identically timestamped/provenanced.
    case "${variant}" in frames1) frames=1;; frames2) frames=2;; *) frames=3;; esac
    gpu="${GPUS[${#jobs[@]}]}"
    python3 -u v39_otherdata/bearing_iclr_ablation.py eval \
      --suite-root "${SUITE_ROOT}" --dataset-root "${DATASET_ROOT}" \
      --city "${city}" --variant "${variant}" --train-frames "${frames}" \
      --gpu "${gpu}" --seed "${SEED:-2033}" \
      >"${SUITE_ROOT}/logs/full_table_v7/eval_${city}_${variant}.log" 2>&1 &
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
  | tee "${SUITE_ROOT}/logs/full_table_v7/table.log"
echo "[DONE] ${SUITE_ROOT}/paper_ablation_tables.md"
