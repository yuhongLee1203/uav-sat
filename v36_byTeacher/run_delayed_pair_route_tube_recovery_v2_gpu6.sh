#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p logs/delayed_pair_route_tube_recovery_v2_center6x6

EPOCHS="${EPOCHS:-60}"
CPU_THREADS="${CPU_THREADS:-2}"
SEG_START="${SEG_START:-4}"
SEG_END="${SEG_END:-64}"
SEG_STEP="${SEG_STEP:-2}"
SOFT_CORRIDOR="${SOFT_CORRIDOR:-12}"
HARD_CORRIDOR="${HARD_CORRIDOR:-20}"
CORRIDOR_WEIGHT="${CORRIDOR_WEIGHT:-2.0}"
MAX_PROGRESS="${MAX_PROGRESS:-10}"
MAX_SPEED="${MAX_SPEED:-8}"
RECOVERY_AUG_PROB="${RECOVERY_AUG_PROB:-0.30}"
RECOVERY_AUG_MAX="${RECOVERY_AUG_MAX:-10}"
ADV_SEARCH="${ADV_SEARCH:-12}"
ADV_WITHIN20="${ADV_WITHIN20:-85}"
ADV_PATIENCE="${ADV_PATIENCE:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m py_compile \
  six_architecture_model.py \
  six_architecture_autoref_experiment.py \
  delayed_pair_ms_kg_gk_experiment.py \
  delayed_pair_segmented_closed_loop_experiment.py \
  delayed_pair_route_tube_recovery_experiment.py \
  delayed_pair_route_tube_recovery_v2_experiment.py \
  gpu_grid_runner.py

"$PYTHON_BIN" - <<'PY'
import config
import robust_tracker as rt
print('[preflight] device:', rt.resolve_device('cuda:0'))
print('[preflight] visual checkpoint:', config.VISUAL_CHECKPOINT)
print('[preflight] corrected route tube: lateral limit + causal progress limit')
PY

echo "[preflight] centered 6x6 MeanShift = 36 patches"
echo "[preflight] soft/hard corridor = ${SOFT_CORRIDOR}m / ${HARD_CORRIDOR}m"
echo "[preflight] max actual causal progress per frame = ${MAX_PROGRESS}m"
echo "[preflight] recovery augmentation p=${RECOVERY_AUG_PROB}, max=${RECOVERY_AUG_MAX}m"
echo "[preflight] GPU6 only; KG then GK sequentially"

run_one () {
  local arch="$1"
  echo "[GPU 6] starting corrected route-tube ${arch}"
  OMP_NUM_THREADS="${CPU_THREADS}" \
  MKL_NUM_THREADS="${CPU_THREADS}" \
  OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
  NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
  UAVSAT_CPU_THREADS="${CPU_THREADS}" \
  CUDA_VISIBLE_DEVICES="6" \
  "$PYTHON_BIN" gpu_grid_runner.py delayed_pair_route_tube_recovery_v2_experiment.py \
    --mode train-eval \
    --arch "${arch}" \
    --device cuda:0 \
    --epochs "${EPOCHS}" \
    --segment-frames-start "${SEG_START}" \
    --segment-frames-end "${SEG_END}" \
    --segment-step "${SEG_STEP}" \
    --soft-corridor-m "${SOFT_CORRIDOR}" \
    --hard-corridor-m "${HARD_CORRIDOR}" \
    --corridor-loss-weight "${CORRIDOR_WEIGHT}" \
    --max-progress-m "${MAX_PROGRESS}" \
    --max-speed-mpf "${MAX_SPEED}" \
    --recovery-aug-probability "${RECOVERY_AUG_PROB}" \
    --recovery-aug-max-m "${RECOVERY_AUG_MAX}" \
    --advance-search-mle "${ADV_SEARCH}" \
    --advance-within20 "${ADV_WITHIN20}" \
    --advance-patience "${ADV_PATIENCE}" \
    2>&1 | tee "logs/delayed_pair_route_tube_recovery_v2_center6x6/${arch}_gpu6.log"
}

run_one KG
run_one GK

echo "Corrected route-tube KG/GK experiments finished."
