#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p logs/six_architecture_autonomous_reference_delay1_center6x6
EPOCHS="${EPOCHS:-60}"
REF_SPACING_M="${REF_SPACING_M:-5.0}"
CPU_THREADS="${CPU_THREADS:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m py_compile \
  six_architecture_model.py \
  six_architecture_autoref_experiment.py \
  gpu_grid_runner.py
"$PYTHON_BIN" - <<'PY'
import config
import robust_tracker as rt
print('[preflight] device:', rt.resolve_device('cuda:0'))
print('[preflight] visual checkpoint:', config.VISUAL_CHECKPOINT)
PY

echo "[preflight] autonomous one-frame-delay + full centered 6x6 ready"
echo "[preflight] Final(t) is emitted only after M(t+1); GRU pair=[z_t,z_t+1]"
echo "[preflight] next autonomous reference query comes from previous base MS only"
echo "[preflight] GPU grid lookup enabled; CPU threads/process=${CPU_THREADS}"

run_pair () {
  local gpu="$1"; shift
  for arch in "$@"; do
    echo "[GPU ${gpu}] starting ${arch} autonomous delay1 centered-6x6"
    OMP_NUM_THREADS="${CPU_THREADS}" \
    MKL_NUM_THREADS="${CPU_THREADS}" \
    OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
    NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
    UAVSAT_CPU_THREADS="${CPU_THREADS}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "$PYTHON_BIN" gpu_grid_runner.py six_architecture_autoref_experiment.py \
      --mode train-eval \
      --arch "${arch}" \
      --device cuda:0 \
      --epochs "${EPOCHS}" \
      --reference-spacing-m "${REF_SPACING_M}" \
      2>&1 | tee "logs/six_architecture_autonomous_reference_delay1_center6x6/${arch}_gpu${gpu}.log"
  done
}

run_pair 0 MKG MGK &
P0=$!
run_pair 5 GMK GKM &
P5=$!
run_pair 6 KGM KMG &
P6=$!

wait "$P0" "$P5" "$P6"
echo "All six autonomous one-frame-delayed centered-6x6 architectures finished."
