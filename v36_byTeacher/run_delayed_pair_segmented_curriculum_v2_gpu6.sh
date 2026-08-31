#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p logs/delayed_pair_segmented_curriculum_v2_center6x6

EPOCHS="${EPOCHS:-60}"
CPU_THREADS="${CPU_THREADS:-2}"
SEG_START="${SEG_START:-4}"
SEG_END="${SEG_END:-48}"
SEG_STEP="${SEG_STEP:-2}"
ADV_SEARCH="${ADV_SEARCH:-12.0}"
ADV_WITHIN20="${ADV_WITHIN20:-85.0}"
ADV_PATIENCE="${ADV_PATIENCE:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m py_compile \
  six_architecture_model.py \
  six_architecture_autoref_experiment.py \
  delayed_pair_ms_kg_gk_experiment.py \
  delayed_pair_segmented_closed_loop_experiment.py \
  delayed_pair_segmented_curriculum_v2_experiment.py \
  gpu_grid_runner.py

"$PYTHON_BIN" - <<'PY'
import config
import robust_tracker as rt
print('[preflight] device:', rt.resolve_device('cuda:0'))
print('[preflight] visual checkpoint:', config.VISUAL_CHECKPOINT)
print('[preflight] v2: adaptive segmented closed-loop curriculum')
print('[preflight] B/C evaluation: one full-route closed-loop rollout, no segment resets')
PY

echo "[preflight] centered 6x6 MeanShift = 36 patches"
echo "[preflight] segment start: reference XY + reference velocity ONCE"
echo "[preflight] inside segment: previous provisional XY only"
echo "[preflight] curriculum: ${SEG_START} -> ${SEG_END}, step=${SEG_STEP}"
echo "[preflight] advance gate: closed_search_mle<=${ADV_SEARCH}m AND within20>=${ADV_WITHIN20}% for ${ADV_PATIENCE} epochs"
echo "[preflight] GPU6 only; KG then GK sequentially"

run_one () {
  local arch="$1"
  echo "[GPU 6] starting seg-curriculum-v2 ${arch}"
  OMP_NUM_THREADS="${CPU_THREADS}" \
  MKL_NUM_THREADS="${CPU_THREADS}" \
  OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
  NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
  UAVSAT_CPU_THREADS="${CPU_THREADS}" \
  CUDA_VISIBLE_DEVICES="6" \
  "$PYTHON_BIN" gpu_grid_runner.py delayed_pair_segmented_curriculum_v2_experiment.py \
    --mode train-eval \
    --arch "${arch}" \
    --device cuda:0 \
    --epochs "${EPOCHS}" \
    --segment-frames-start "${SEG_START}" \
    --segment-frames-end "${SEG_END}" \
    --segment-step "${SEG_STEP}" \
    --advance-search-mle "${ADV_SEARCH}" \
    --advance-within20 "${ADV_WITHIN20}" \
    --advance-patience "${ADV_PATIENCE}" \
    2>&1 | tee "logs/delayed_pair_segmented_curriculum_v2_center6x6/${arch}_gpu6.log"
}

run_one KG
run_one GK

echo "Adaptive segmented curriculum v2 KG/GK experiments finished."
