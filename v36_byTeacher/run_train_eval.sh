#!/usr/bin/env bash
set -euo pipefail

# Autonomous v36_byTeacher runner.
# Usage:
#   bash run_train_eval.sh train
#   bash run_train_eval.sh eval
#   bash run_train_eval.sh all
#
# Optional environment overrides:
#   PYTHON_BIN=python3
#   UAVSAT_BACKBONE=mobilenet_v3_small
#   UAVSAT_TEMPORAL_EPOCHS=60
#   UAVSAT_TEMPORAL_EXTRA_A_STRIDE=2
#   UAVSAT_RUN_TAG=my_run

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN to a Python 3 interpreter." >&2
  exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 8):
    raise SystemExit(
        "ERROR: v36_byTeacher requires Python >= 3.8; current=%s" % sys.version.split()[0]
    )
print("Python:", sys.executable, sys.version.split()[0], flush=True)
PY

# Fail immediately on syntax errors before spending time loading models/data.
"${PYTHON_BIN}" -m py_compile \
  config.py \
  data.py \
  visual_model.py \
  visual_localizer.py \
  robust_tracker_base.py \
  robust_tracker.py \
  train_multirate_a.py

echo "Python syntax preflight: OK" >&2

case "${MODE}" in
  train)
    "${PYTHON_BIN}" train_multirate_a.py \
      --temporal-epochs "${UAVSAT_TEMPORAL_EPOCHS:-60}" \
      --extra-stride "${UAVSAT_TEMPORAL_EXTRA_A_STRIDE:-2}" \
      --train-visual-if-missing
    ;;
  eval)
    "${PYTHON_BIN}" robust_tracker.py \
      --mode eval \
      --extra-stride "${UAVSAT_TEMPORAL_EXTRA_A_STRIDE:-2}" \
      --eval-routes route_C route_B
    ;;
  all)
    "${PYTHON_BIN}" train_multirate_a.py \
      --temporal-epochs "${UAVSAT_TEMPORAL_EPOCHS:-60}" \
      --extra-stride "${UAVSAT_TEMPORAL_EXTRA_A_STRIDE:-2}" \
      --train-visual-if-missing
    "${PYTHON_BIN}" robust_tracker.py \
      --mode eval \
      --extra-stride "${UAVSAT_TEMPORAL_EXTRA_A_STRIDE:-2}" \
      --eval-routes route_C route_B
    ;;
  *)
    echo "usage: bash run_train_eval.sh {train|eval|all}" >&2
    exit 2
    ;;
esac
