#!/usr/bin/env bash
set -euo pipefail

# Fast inference-only estimator/decoder sensitivity sweep for v8r1.
# IMPORTANT: these parameters do not require retraining the GRU.
# Every case reuses the already-trained FULL v8r1 temporal checkpoint and only
# evaluates Route-C / Route-B with different inference-time parameters.
# Outputs stay isolated under output/<backbone>/parameter_ablation/<case>/.

GROUP="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS="${UAVSAT_CPU_THREADS:-2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

case "${GROUP}" in
  all|ms2|kf|meanshift) ;;
  *)
    echo "usage: bash run_parameter_ablation.sh {all|ms2|kf|meanshift}" >&2
    exit 2
    ;;
esac

export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found." >&2
  exit 2
fi

"${PYTHON_BIN}" -m py_compile config.py data.py visual_model.py visual_localizer.py robust_tracker_base.py robust_tracker.py train_multirate_a.py

echo "FAST parameter ablation: eval-only, CPU threads=${CPU_THREADS}" >&2

run_case() {
  local tag="$1"
  local sigma="$2"
  local weight="$3"
  local kf_r="$4"
  local bandwidth="$5"

  echo
  echo "================================================================================================"
  echo "FAST PARAMETER CASE: ${tag}"
  echo "  mode                = EVAL ONLY (reuse full v8r1 checkpoint)"
  echo "  MS2 KF prior sigma  = ${sigma} m"
  echo "  MS2 KF prior weight = ${weight}"
  echo "  Kalman R position   = ${kf_r} m^2"
  echo "  MeanShift bandwidth = ${bandwidth} m"
  echo "================================================================================================"

  UAVSAT_GRU_ABLATION=full \
  UAVSAT_PARAMETER_TAG="${tag}" \
  UAVSAT_MS2_KF_PRIOR_SIGMA_M="${sigma}" \
  UAVSAT_MS2_KF_PRIOR_WEIGHT="${weight}" \
  UAVSAT_KF_R_POS="${kf_r}" \
  UAVSAT_MS_BANDWIDTH_M="${bandwidth}" \
  UAVSAT_CPU_THREADS="${CPU_THREADS}" \
  "${PYTHON_BIN}" - <<'PY'
import os
import runpy
import shutil
import sys
from pathlib import Path

import torch
import config

threads = max(1, int(os.environ.get("UAVSAT_CPU_THREADS", "2")))
torch.set_num_threads(threads)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

tag = os.environ["UAVSAT_PARAMETER_TAG"].strip()
if not tag or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in tag):
    raise SystemExit("ERROR: invalid UAVSAT_PARAMETER_TAG=%r" % tag)
if config.GRU_ABLATION != "full":
    raise SystemExit("ERROR: parameter sweep must use full GRU")

# Shared trained model/cache from the normal FULL v8r1 experiment.
shared_backbone_root = Path(config.BACKBONE_OUTPUT_DIR)
shared_visual_checkpoint = Path(config.VISUAL_CHECKPOINT)
shared_feature_cache = Path(config.FEATURE_CACHE_DIR)
source_full_checkpoint = (
    Path(config.CHECKPOINT_DIR)
    / f"reference_prior_compact_gru_A_native_v8r1_full_{config.BACKBONE_KEY}.pt"
)
if not source_full_checkpoint.exists():
    raise SystemExit(
        "ERROR: FULL v8r1 checkpoint not found: %s\n"
        "Finish the GPU0/full run first, then rerun GPU6." % source_full_checkpoint
    )

# Isolate outputs per parameter case, while keeping the frozen visual/cache shared.
parameter_root = shared_backbone_root / "parameter_ablation" / tag
parameter_checkpoint_dir = parameter_root / "checkpoints"
parameter_checkpoint_dir.mkdir(parents=True, exist_ok=True)

config.BACKBONE_OUTPUT_DIR = parameter_root
config.CHECKPOINT_DIR = parameter_checkpoint_dir
config.VISUAL_CHECKPOINT = shared_visual_checkpoint
config.FEATURE_CACHE_DIR = shared_feature_cache

destination_checkpoint = (
    parameter_checkpoint_dir
    / f"reference_prior_compact_gru_A_native_v8r1_full_{config.BACKBONE_KEY}.pt"
)
if (not destination_checkpoint.exists()
        or destination_checkpoint.stat().st_size != source_full_checkpoint.stat().st_size
        or destination_checkpoint.stat().st_mtime_ns < source_full_checkpoint.stat().st_mtime_ns):
    shutil.copy2(source_full_checkpoint, destination_checkpoint)

print("CPU THREAD LIMIT:", threads, flush=True)
print("SOURCE FULL CHECKPOINT:", source_full_checkpoint, flush=True)
print("ISOLATED OUTPUT ROOT:", parameter_root, flush=True)
print("MODE: eval only", flush=True)
print("MS2 SIGMA:", config.MS2_KALMAN_PRIOR_SIGMA_M, flush=True)
print("MS2 WEIGHT:", config.MS2_KALMAN_PRIOR_WEIGHT, flush=True)
print("KF R POSITION:", config.KALMAN_R_POSITION, flush=True)
print("MEANSHIFT BANDWIDTH:", config.MEANSHIFT_BANDWIDTH_M, flush=True)

sys.argv = ["train_multirate_a.py", "--mode", "eval"]
runpy.run_path("train_multirate_a.py", run_name="__main__")
PY
}

run_ms2_group() {
  run_case "ms2_no_kf_prior" 12.0 0.0 9.0 8.0
  run_case "ms2_sigma_6m" 6.0 1.0 9.0 8.0
  run_case "ms2_sigma_24m" 24.0 1.0 9.0 8.0
  run_case "ms2_weight_0p5" 12.0 0.5 9.0 8.0
  run_case "ms2_weight_2p0" 12.0 2.0 9.0 8.0
}

run_kf_group() {
  run_case "kf_r_4" 12.0 1.0 4.0 8.0
  run_case "kf_r_25" 12.0 1.0 25.0 8.0
}

run_meanshift_group() {
  run_case "meanshift_bw_4m" 12.0 1.0 9.0 4.0
  run_case "meanshift_bw_12m" 12.0 1.0 9.0 12.0
}

case "${GROUP}" in
  ms2) run_ms2_group ;;
  kf) run_kf_group ;;
  meanshift) run_meanshift_group ;;
  all)
    run_ms2_group
    run_kf_group
    run_meanshift_group
    ;;
esac

echo
echo "All requested FAST parameter ablations completed."
