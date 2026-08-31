#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p logs/delayed_pair_one_frame_center6x6
EPOCHS="${EPOCHS:-60}"
CPU_THREADS="${CPU_THREADS:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m py_compile \
  six_architecture_model.py \
  six_architecture_autoref_experiment.py \
  delayed_pair_ms_kg_gk_experiment.py \
  gpu_grid_runner.py

"$PYTHON_BIN" - <<'PY'
import config
import robust_tracker as rt
print('[preflight] device:', rt.resolve_device('cuda:0'))
print('[preflight] visual checkpoint:', config.VISUAL_CHECKPOINT)
print('[preflight] protocol: [I(t-1), I(t)] -> MS(t-1) -> KG/GK -> provisional t; next pair MS(t) -> final t')
PY

echo "[preflight] full centered 6x6 MeanShift = 36 patches"
echo "[preflight] Route A: reference-centered MS training"
echo "[preflight] Route B/C: start-point initialization only; then previous provisional center"
echo "[preflight] GPU grid lookup enabled; CPU threads/process=${CPU_THREADS}"

run_one () {
  local gpu="$1"
  local arch="$2"
  echo "[GPU ${gpu}] starting delayed-pair ${arch}"
  OMP_NUM_THREADS="${CPU_THREADS}" \
  MKL_NUM_THREADS="${CPU_THREADS}" \
  OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
  NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
  UAVSAT_CPU_THREADS="${CPU_THREADS}" \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  "$PYTHON_BIN" gpu_grid_runner.py delayed_pair_ms_kg_gk_experiment.py \
    --mode train-eval \
    --arch "${arch}" \
    --device cuda:0 \
    --epochs "${EPOCHS}" \
    2>&1 | tee "logs/delayed_pair_one_frame_center6x6/${arch}_gpu${gpu}.log"
}

run_one 0 KG &
P0=$!
run_one 5 GK &
P5=$!

wait "$P0" "$P5"
echo "Delayed-pair KG/GK experiments finished."
