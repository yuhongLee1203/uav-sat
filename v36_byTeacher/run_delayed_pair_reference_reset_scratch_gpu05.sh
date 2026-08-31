#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

EPOCHS="${EPOCHS:-60}"
CPU_THREADS="${CPU_THREADS:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Remove ONLY the previous misleading controlled-eval outputs where the final
# next-frame MS was centered directly at the reference, plus this experiment's
# own old outputs/checkpoints. Do not touch route-tube recovery files/processes.
rm -rf output/mobilenet_v3_small/delayed_pair_gtcenter_controlled_center6x6
rm -rf output/mobilenet_v3_small/delayed_pair_reference_reset_scratch_center6x6
rm -f output/mobilenet_v3_small/checkpoints/delayed_pair_reference_reset_scratch_kg_center6x6_mobilenet_v3_small.pt
rm -f output/mobilenet_v3_small/checkpoints/delayed_pair_reference_reset_scratch_gk_center6x6_mobilenet_v3_small.pt
rm -rf logs/delayed_pair_reference_reset_scratch_center6x6
mkdir -p logs/delayed_pair_reference_reset_scratch_center6x6

"$PYTHON_BIN" -m py_compile \
  delayed_pair_ms_kg_gk_experiment.py \
  delayed_pair_reference_reset_train_from_scratch_experiment.py \
  six_architecture_model.py \
  gpu_grid_runner.py

echo "[preflight] FROM SCRATCH: KG and GK use new independent checkpoints"
echo "[preflight] sample: ref(t) -> MS(t) -> KG/GK -> x'(t+1) -> MS(t+1 centered at x') -> FINAL(t+1)"
echo "[preflight] next sample resets at ref(t+1); no long-horizon drift"
echo "[preflight] final next-frame MS never uses reference as its center"
echo "[preflight] GPU0=KG, GPU5=GK; GPU6 is untouched"
echo "[preflight] epochs=${EPOCHS}, CPU threads/process=${CPU_THREADS}"

run_one () {
  local gpu="$1"
  local arch="$2"
  echo "[GPU ${gpu}] starting ${arch} from scratch"
  OMP_NUM_THREADS="${CPU_THREADS}" \
  MKL_NUM_THREADS="${CPU_THREADS}" \
  OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
  NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
  UAVSAT_CPU_THREADS="${CPU_THREADS}" \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  "$PYTHON_BIN" gpu_grid_runner.py delayed_pair_reference_reset_train_from_scratch_experiment.py \
    --mode train-eval \
    --arch "${arch}" \
    --device cuda:0 \
    --epochs "${EPOCHS}" \
    2>&1 | tee "logs/delayed_pair_reference_reset_scratch_center6x6/${arch}_gpu${gpu}.log"
}

run_one 0 KG &
pid_kg=$!
run_one 5 GK &
pid_gk=$!

wait "$pid_kg"
wait "$pid_gk"

echo "From-scratch reference-reset delayed KG/GK comparison finished."
