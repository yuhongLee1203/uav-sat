#!/usr/bin/env bash
set -euo pipefail

# v36_byTeacher v2 runner.
# Training = Route-A native + Route-A stride-N (default stride=2).
# Validation = Route-C. Final test = Route-B.
#
# Usage:
#   bash run_train_eval.sh train
#   bash run_train_eval.sh eval
#   bash run_train_eval.sh all
#
# Optional:
#   PYTHON_BIN=python3
#   UAVSAT_TEMPORAL_EPOCHS=60
#   UAVSAT_TEMPORAL_EXTRA_A_STRIDE=2

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EPOCHS="${UAVSAT_TEMPORAL_EPOCHS:-60}"
STRIDE="${UAVSAT_TEMPORAL_EXTRA_A_STRIDE:-2}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found." >&2
  exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 8):
    raise SystemExit(
        "ERROR: v36_byTeacher requires Python >= 3.8; current=%s"
        % sys.version.split()[0]
    )
print("Python:", sys.executable, sys.version.split()[0], flush=True)
PY

"${PYTHON_BIN}" -m py_compile \
  config.py \
  data.py \
  visual_model.py \
  visual_localizer.py \
  robust_tracker_base.py \
  robust_tracker.py \
  train_multirate_a.py \
  plot_final_trajectory.py \
  plot_final_map_results.py \
  render_results_video.py

echo "Python syntax preflight: OK" >&2
echo "Temporal data policy: Route-A native + stride-${STRIDE}; C=val; B=test" >&2

case "${MODE}" in
  train)
    "${PYTHON_BIN}" train_multirate_a.py \
      --mode train \
      --temporal-epochs "${EPOCHS}" \
      --extra-stride "${STRIDE}" \
      --train-visual-if-missing
    ;;
  eval)
    "${PYTHON_BIN}" train_multirate_a.py \
      --mode eval \
      --extra-stride "${STRIDE}"
    ;;
  all)
    "${PYTHON_BIN}" train_multirate_a.py \
      --mode all \
      --temporal-epochs "${EPOCHS}" \
      --extra-stride "${STRIDE}" \
      --train-visual-if-missing
    ;;
  *)
    echo "usage: bash run_train_eval.sh {train|eval|all}" >&2
    exit 2
    ;;
esac
