#!/usr/bin/env bash
set -euo pipefail

# v36_byTeacher controlled reference-assisted v8r1 runner.
# Training = Route-A native/original speed ONLY.
# Validation = Route-C. Final test = Route-B.
#
# Usage:
#   bash run_train_eval.sh train [ablation]
#   bash run_train_eval.sh eval  [ablation]
#   bash run_train_eval.sh all   [ablation]

MODE="${1:-all}"
ABLATION="${2:-${UAVSAT_GRU_ABLATION:-full}}"
ABLATION="$(printf '%s' "${ABLATION}" | tr '[:upper:]' '[:lower:]')"
CPU_THREADS="${UAVSAT_CPU_THREADS:-2}"

case "${ABLATION}" in
  full|no_ms_xy|no_temporal_mean|no_first_difference|no_previous_motion) ;;
  *)
    echo "ERROR: invalid GRU ablation '${ABLATION}'." >&2
    exit 2
    ;;
esac

# Keep concurrent GPU experiments from saturating the host CPU.
export UAVSAT_GRU_ABLATION="${ABLATION}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EPOCHS="${UAVSAT_TEMPORAL_EPOCHS:-60}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found." >&2
  exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import os
import sys
import torch
import config

threads = max(1, int(os.environ.get("UAVSAT_CPU_THREADS", os.environ.get("OMP_NUM_THREADS", "2"))))
torch.set_num_threads(threads)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

if sys.version_info < (3, 8):
    raise SystemExit("ERROR: Python >= 3.8 required")
requested = os.environ.get("UAVSAT_GRU_ABLATION", "").strip().lower()
if config.GRU_ABLATION != requested:
    raise SystemExit("ERROR: ablation propagation failed: env=%r config=%r" % (requested, config.GRU_ABLATION))
expected_input_dim = 128 * len(config.GRU_ACTIVE_GROUPS)
if int(config.RNN_COMBINED_INPUT_DIM) != expected_input_dim:
    raise SystemExit("ERROR: recurrent input dimension mismatch")

print("Python:", sys.executable, sys.version.split()[0], flush=True)
print("RUN REVISION: v8r1", flush=True)
print("GRU ABLATION:", config.GRU_ABLATION, flush=True)
print("GRU ACTIVE GROUPS:", ",".join(config.GRU_ACTIVE_GROUPS), flush=True)
print("GRU INPUT DIM:", int(config.RNN_COMBINED_INPUT_DIM), flush=True)
print("CPU THREAD LIMIT:", threads, flush=True)
PY

"${PYTHON_BIN}" -m py_compile config.py data.py visual_model.py visual_localizer.py robust_tracker_base.py robust_tracker.py train_multirate_a.py

echo "Python syntax preflight: OK" >&2
echo "CONTROLLED v8r1 ablation=${ABLATION}; CPU threads=${CPU_THREADS}" >&2

case "${MODE}" in
  train)
    "${PYTHON_BIN}" train_multirate_a.py --mode train --temporal-epochs "${EPOCHS}" --train-visual-if-missing
    ;;
  eval)
    "${PYTHON_BIN}" train_multirate_a.py --mode eval
    ;;
  all)
    "${PYTHON_BIN}" train_multirate_a.py --mode all --temporal-epochs "${EPOCHS}" --train-visual-if-missing
    ;;
  *)
    echo "usage: bash run_train_eval.sh {train|eval|all} [ablation]" >&2
    exit 2
    ;;
esac
