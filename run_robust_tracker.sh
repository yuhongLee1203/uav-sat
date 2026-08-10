#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

GPU="${GPU:-0}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-50}"
JITTER_M="${JITTER_M:-12}"
REUSE_VISUAL="${REUSE_VISUAL:-0}"

trap 'code=$?; echo; echo "ERROR: run_robust_tracker.sh stopped at line ${LINENO}, exit=${code}" >&2; exit ${code}' ERR

echo "================================================================================"
echo "ROUTE-COORDINATE GRU + CONSTRAINED FILTERPY KALMAN"
echo "================================================================================"
echo "PWD             : $(pwd)"
echo "Python          : $(command -v python3)"
python3 --version
echo "GPU             : ${GPU}"
echo "Visual epochs   : ${VISUAL_EPOCHS}"
echo "Temporal epochs : ${TEMPORAL_EPOCHS}"
echo "Reuse visual    : ${REUSE_VISUAL}"
echo "================================================================================"

echo
echo "[PRECHECK 1/4] required project files"
for f in \
  config.py \
  data.py \
  visual_model.py \
  visual_localizer.py \
  robust_tracker.py \
  route_waypoints/route_A_waypoints.json \
  route_waypoints/route_B_waypoints.json \
  route_waypoints/route_C_waypoints.json
do
  if [[ ! -f "${f}" ]]; then
    echo "MISSING: ${f}" >&2
    exit 10
  fi
  echo "OK: ${f}"
done

echo
echo "[PRECHECK 2/4] Python syntax"
python3 -m py_compile \
  config.py \
  data.py \
  visual_model.py \
  visual_localizer.py \
  robust_tracker.py
echo "Python syntax OK"

echo
echo "[PRECHECK 3/4] imports"
PYTHONUNBUFFERED=1 python3 -u - <<'PY'
import importlib
import sys
import traceback

modules = [
    "torch",
    "filterpy",
    "matplotlib",
    "config",
    "data",
    "visual_model",
    "visual_localizer",
    "robust_tracker",
]

for name in modules:
    print("[IMPORT]", name, "...", flush=True)
    try:
        module = importlib.import_module(name)
    except BaseException:
        print("[IMPORT FAILED]", name, flush=True)
        traceback.print_exc()
        raise
    else:
        print("[IMPORT OK]", name, flush=True)

import torch
print(
    "[CUDA] available=",
    torch.cuda.is_available(),
    "device_count=",
    torch.cuda.device_count(),
    flush=True,
)
if torch.cuda.is_available():
    print(
        "[CUDA] device0=",
        torch.cuda.get_device_name(0),
        flush=True,
    )
PY

echo
echo "[PRECHECK 4/4] checkpoints / output"
mkdir -p outputs/route_coordinate_gru_kalman_v2/checkpoints

VISUAL_CKPT="outputs/route_coordinate_gru_kalman_v2/checkpoints/visual_retrieval_A_only.pt"

if [[ "${REUSE_VISUAL}" == "1" ]]; then
  if [[ ! -f "${VISUAL_CKPT}" ]]; then
    echo "REUSE_VISUAL=1 but ${VISUAL_CKPT} does not exist." >&2
    echo "Either copy the visual checkpoint there or run with REUSE_VISUAL=0." >&2
    exit 11
  fi
  echo "Visual checkpoint found: ${VISUAL_CKPT}"
else
  echo "Visual retrieval will train from scratch."
fi

CMD=(
  python3
  -u
  robust_tracker.py
  --mode
  train_eval
  --visual-epochs
  "${VISUAL_EPOCHS}"
  --epochs
  "${TEMPORAL_EPOCHS}"
  --jitter-m
  "${JITTER_M}"
)

if [[ "${REUSE_VISUAL}" == "1" ]]; then
  CMD+=(--reuse-visual)
fi

echo
echo "================================================================================"
echo "[LAUNCH] training process starts NOW"
printf 'COMMAND:'
printf ' %q' "${CMD[@]}"
echo
echo "================================================================================"

# exec is deliberate:
# the shell process is replaced by Python. If training is alive, `ps` must show
# robust_tracker.py. PYTHONUNBUFFERED=1 prevents silent buffered output.
exec env \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 \
  "${CMD[@]}"
