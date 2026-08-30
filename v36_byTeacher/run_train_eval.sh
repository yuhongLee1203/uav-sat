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
#
# Ablation choices:
#   full
#   no_ms_xy
#   no_temporal_mean
#   no_first_difference
#   no_previous_motion
#
# The second positional argument is preferred because it makes the selected
# ablation explicit and prevents accidental reuse of the default "full" model.
# UAVSAT_GRU_ABLATION remains supported as a fallback for compatibility.

MODE="${1:-all}"
ABLATION="${2:-${UAVSAT_GRU_ABLATION:-full}}"
ABLATION="$(printf '%s' "${ABLATION}" | tr '[:upper:]' '[:lower:]')"

case "${ABLATION}" in
  full|no_ms_xy|no_temporal_mean|no_first_difference|no_previous_motion)
    ;;
  *)
    echo "ERROR: invalid GRU ablation '${ABLATION}'." >&2
    echo "choices: full no_ms_xy no_temporal_mean no_first_difference no_previous_motion" >&2
    exit 2
    ;;
esac

# Export exactly once here so every Python subprocess sees the same setting.
export UAVSAT_GRU_ABLATION="${ABLATION}"

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
import config

if sys.version_info < (3, 8):
    raise SystemExit(
        "ERROR: v36_byTeacher requires Python >= 3.8; current=%s"
        % sys.version.split()[0]
    )

requested = os.environ.get("UAVSAT_GRU_ABLATION", "").strip().lower()
if config.GRU_ABLATION != requested:
    raise SystemExit(
        "ERROR: ablation propagation failed: env=%r config=%r"
        % (requested, config.GRU_ABLATION)
    )

expected_input_dim = 128 * len(config.GRU_ACTIVE_GROUPS)
if int(config.RNN_COMBINED_INPUT_DIM) != expected_input_dim:
    raise SystemExit(
        "ERROR: recurrent input dimension mismatch: groups=%r configured=%d expected=%d"
        % (
            config.GRU_ACTIVE_GROUPS,
            int(config.RNN_COMBINED_INPUT_DIM),
            expected_input_dim,
        )
    )

print("Python:", sys.executable, sys.version.split()[0], flush=True)
print("RUN REVISION: v8r1", flush=True)
print("GRU ABLATION:", config.GRU_ABLATION, flush=True)
print("GRU ACTIVE GROUPS:", ",".join(config.GRU_ACTIVE_GROUPS), flush=True)
print("GRU INPUT DIM:", int(config.RNN_COMBINED_INPUT_DIM), flush=True)
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
echo "Temporal data policy: Route-A native/original speed ONLY; C=val; B=test" >&2
echo "CONTROLLED v8r1 ablation=${ABLATION}" >&2
echo "UAVSAT_GRU_ABLATION=${UAVSAT_GRU_ABLATION}" >&2

case "${MODE}" in
  train)
    "${PYTHON_BIN}" train_multirate_a.py \
      --mode train \
      --temporal-epochs "${EPOCHS}" \
      --train-visual-if-missing
    ;;
  eval)
    "${PYTHON_BIN}" train_multirate_a.py --mode eval
    ;;
  all)
    "${PYTHON_BIN}" train_multirate_a.py \
      --mode all \
      --temporal-epochs "${EPOCHS}" \
      --train-visual-if-missing
    ;;
  *)
    echo "usage: bash run_train_eval.sh {train|eval|all} [ablation]" >&2
    exit 2
    ;;
esac
