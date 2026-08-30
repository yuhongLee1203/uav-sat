#!/usr/bin/env bash
set -euo pipefail

# Six-order experiment launcher using physical GPUs 0, 5, and 6.
# Two architectures are trained sequentially on each GPU, while the three GPUs
# run in parallel. This avoids six jobs competing for memory on the same device.
#
# Usage:
#   bash run_six_orders.sh all 60
#   bash run_six_orders.sh train 60
#   bash run_six_orders.sh eval

MODE="${1:-all}"
EPOCHS="${2:-${UAVSAT_TEMPORAL_EPOCHS:-60}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS="${UAVSAT_CPU_THREADS:-2}"
EVAL_ROUTES="${UAVSAT_EVAL_ROUTES:-route_C route_B}"

case "${MODE}" in
  all) PIPE_MODE="train_eval" ;;
  train) PIPE_MODE="train" ;;
  eval) PIPE_MODE="eval" ;;
  *) echo "usage: bash run_six_orders.sh {all|train|eval} [epochs]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs/six_order

export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"

"${PYTHON_BIN}" -m py_compile six_order_pipeline.py compare_six_orders.py

# Prepare the shared Route-A-only visual retrieval checkpoint before parallel jobs start.
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" six_order_pipeline.py \
  --order MKG --mode prepare_visual --train-visual-if-missing \
  > logs/six_order/prepare_visual.log 2>&1

LATENCY_FLAG=()
if [[ "${UAVSAT_MEASURE_LATENCY:-0}" == "1" ]]; then
  LATENCY_FLAG+=(--measure-latency)
fi

run_order() {
  local gpu="$1"
  local order="$2"
  echo "[$(date '+%F %T')] GPU=${gpu} order=${order} mode=${PIPE_MODE}" | tee -a "logs/six_order/${order}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" six_order_pipeline.py \
    --order "${order}" \
    --mode "${PIPE_MODE}" \
    --temporal-epochs "${EPOCHS}" \
    --eval-routes ${EVAL_ROUTES} \
    "${LATENCY_FLAG[@]}" \
    >> "logs/six_order/${order}.log" 2>&1
}

(
  run_order 0 MKG
  run_order 0 KGM
) & pid0=$!

(
  run_order 5 MGK
  run_order 5 GKM
) & pid5=$!

(
  run_order 6 GMK
  run_order 6 KMG
) & pid6=$!

wait "${pid0}"
wait "${pid5}"
wait "${pid6}"

if [[ "${PIPE_MODE}" != "train" ]]; then
  "${PYTHON_BIN}" compare_six_orders.py
fi

echo "Six-order experiment completed."
