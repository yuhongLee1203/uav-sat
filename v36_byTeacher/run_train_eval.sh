#!/usr/bin/env bash
set -euo pipefail

# Autonomous v36_byTeacher runner.
# Temporal training uses ONLY Route-A at its original recorded frame rate/speed.
# Route-C = validation/checkpoint selection; Route-B = final test.
#
# Usage:
#   bash run_train_eval.sh train
#   bash run_train_eval.sh eval
#   bash run_train_eval.sh all
#
# Optional environment overrides:
#   PYTHON_BIN=python3
#   UAVSAT_TEMPORAL_EPOCHS=60

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

# Fail immediately on syntax errors before loading models/data.
"${PYTHON_BIN}" -m py_compile \
  config.py \
  data.py \
  visual_model.py \
  visual_localizer.py \
  robust_tracker_base.py \
  robust_tracker.py \
  train_multirate_a.py

echo "Python syntax preflight: OK" >&2

echo "Temporal data policy: Route-A native speed ONLY (no stride-2 / 2x-speed sequence)" >&2

case "${MODE}" in
  train)
    "${PYTHON_BIN}" train_multirate_a.py \
      --mode train \
      --temporal-epochs "${UAVSAT_TEMPORAL_EPOCHS:-60}" \
      --train-visual-if-missing
    ;;
  eval)
    "${PYTHON_BIN}" train_multirate_a.py \
      --mode eval
    ;;
  all)
    "${PYTHON_BIN}" train_multirate_a.py \
      --mode all \
      --temporal-epochs "${UAVSAT_TEMPORAL_EPOCHS:-60}" \
      --train-visual-if-missing
    ;;
  *)
    echo "usage: bash run_train_eval.sh {train|eval|all}" >&2
    exit 2
    ;;
esac
