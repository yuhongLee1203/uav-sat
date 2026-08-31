#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p logs/delayed_pair_segmented_closed_loop_center6x6

EPOCHS="${EPOCHS:-60}"
CPU_THREADS="${CPU_THREADS:-2}"
SEG_START="${SEG_START:-16}"
SEG_END="${SEG_END:-96}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m py_compile \
  six_architecture_model.py \
  six_architecture_autoref_experiment.py \
  delayed_pair_ms_kg_gk_experiment.py \
  delayed_pair_segmented_closed_loop_experiment.py \
  gpu_grid_runner.py

"$PYTHON_BIN" - <<'PY'
import config
import robust_tracker as rt
print('[preflight] device:', rt.resolve_device('cuda:0'))
print('[preflight] visual checkpoint:', config.VISUAL_CHECKPOINT)
print('[preflight] protocol: segmented closed-loop training; full-route closed-loop B/C evaluation')
print('[preflight] segment start: reference XY + reference velocity ONCE; no per-frame reference search center')
PY

echo "[preflight] full centered 6x6 MeanShift = 36 patches"
echo "[preflight] curriculum segment frames: ${SEG_START} -> ${SEG_END}"
echo "[preflight] GPU6 only, KG then GK sequentially"
echo "[preflight] CPU threads/process=${CPU_THREADS}"

run_one () {
  local arch="$1"
  echo "[GPU 6] starting segmented closed-loop ${arch}"
  OMP_NUM_THREADS="${CPU_THREADS}" \
  MKL_NUM_THREADS="${CPU_THREADS}" \
  OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
  NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
  UAVSAT_CPU_THREADS="${CPU_THREADS}" \
  CUDA_VISIBLE_DEVICES="6" \
  "$PYTHON_BIN" gpu_grid_runner.py delayed_pair_segmented_closed_loop_experiment.py \
    --mode train-eval \
    --arch "${arch}" \
    --device cuda:0 \
    --epochs "${EPOCHS}" \
    --segment-frames-start "${SEG_START}" \
    --segment-frames-end "${SEG_END}" \
    2>&1 | tee "logs/delayed_pair_segmented_closed_loop_center6x6/${arch}_gpu6.log"
}

run_one KG
run_one GK

echo "Segmented closed-loop KG/GK experiments finished."
