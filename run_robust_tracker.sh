#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU="${GPU:-0}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-50}"
JITTER_M="${JITTER_M:-12}"

mkdir -p outputs/route_rnn_filterpy_full_retrain

python3 - <<'PY'
try:
    import filterpy
    print("FilterPy:", filterpy.__version__)
except Exception as exc:
    raise SystemExit(
        "FilterPy is required. Install once with:\n"
        "  pip install filterpy\n"
        f"original error: {exc}"
    )
PY

echo "================================================================================"
echo "FULL RETRAIN: SINGLE-FRAME GRU + FORWARD ROUTE SEARCH + FILTERPY KALMAN"
echo "================================================================================"
echo "GPU              : ${GPU}"
echo "Visual epochs    : ${VISUAL_EPOCHS}"
echo "Temporal epochs  : ${TEMPORAL_EPOCHS}"
echo "Jitter           : ${JITTER_M} m"
echo
echo "Visual retrieval will be trained FROM SCRATCH on Route A."
echo "GRU temporal model will be trained FROM SCRATCH on Route A."
echo "Final evaluation will run on Route B and Route C."
echo "================================================================================"

CUDA_VISIBLE_DEVICES="${GPU}" \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
NUMEXPR_NUM_THREADS=4 \
python3 robust_tracker.py \
  --mode train_eval \
  --visual-epochs "${VISUAL_EPOCHS}" \
  --epochs "${TEMPORAL_EPOCHS}" \
  --jitter-m "${JITTER_M}"
