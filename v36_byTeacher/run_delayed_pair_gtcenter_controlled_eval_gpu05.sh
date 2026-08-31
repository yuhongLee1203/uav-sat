#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p logs/delayed_pair_gtcenter_controlled_center6x6
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS="${CPU_THREADS:-2}"

"$PYTHON_BIN" -m py_compile \
  delayed_pair_ms_kg_gk_experiment.py \
  delayed_pair_gtcenter_controlled_eval.py \
  gpu_grid_runner.py

run_one () {
  local gpu="$1"
  local arch="$2"
  echo "[GPU ${gpu}] controlled delayed eval ${arch}"
  OMP_NUM_THREADS="${CPU_THREADS}" \
  MKL_NUM_THREADS="${CPU_THREADS}" \
  OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
  NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
  UAVSAT_CPU_THREADS="${CPU_THREADS}" \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  "$PYTHON_BIN" gpu_grid_runner.py delayed_pair_gtcenter_controlled_eval.py \
    --arch "${arch}" \
    --device cuda:0 \
    2>&1 | tee "logs/delayed_pair_gtcenter_controlled_center6x6/${arch}_gpu${gpu}.log"
}

run_one 0 KG &
P0=$!
run_one 5 GK &
P5=$!
wait "$P0" "$P5"

echo "Controlled delayed KG/GK evaluation finished."
