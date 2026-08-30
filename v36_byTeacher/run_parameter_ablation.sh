#!/usr/bin/env bash
set -euo pipefail

# Isolated estimator / decoder parameter ablations for v36_byTeacher v8r1.
# This runner intentionally does NOT reuse the normal full/GRU-ablation output
# or temporal checkpoint directories. It keeps the shared frozen visual
# checkpoint/cache, but every parameter case gets its own temporal checkpoints,
# CSVs, and outputs under:
#   output/<backbone>/parameter_ablation/<case>/...
#
# Baseline is the normal full v8r1 run:
#   MS2 sigma=12m, MS2 prior weight=1.0, KF R=9m^2, MeanShift bandwidth=8m.
# The baseline is NOT rerun here; compare these cases against the GPU0/full run.
#
# Usage:
#   bash run_parameter_ablation.sh all
#   bash run_parameter_ablation.sh ms2
#   bash run_parameter_ablation.sh kf
#   bash run_parameter_ablation.sh meanshift

GROUP="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EPOCHS="${UAVSAT_TEMPORAL_EPOCHS:-60}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

case "${GROUP}" in
  all|ms2|kf|meanshift) ;;
  *)
    echo "usage: bash run_parameter_ablation.sh {all|ms2|kf|meanshift}" >&2
    exit 2
    ;;
esac

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found." >&2
  exit 2
fi

"${PYTHON_BIN}" -m py_compile \
  config.py data.py visual_model.py visual_localizer.py \
  robust_tracker_base.py robust_tracker.py train_multirate_a.py

echo "Parameter-ablation syntax preflight: OK" >&2

run_case() {
  local tag="$1"
  local sigma="$2"
  local weight="$3"
  local kf_r="$4"
  local bandwidth="$5"

  echo
  echo "================================================================================================"
  echo "PARAMETER CASE: ${tag}"
  echo "  GRU ablation       = full"
  echo "  MS2 KF prior sigma = ${sigma} m"
  echo "  MS2 KF prior weight= ${weight}"
  echo "  Kalman R position  = ${kf_r} m^2"
  echo "  MeanShift bandwidth= ${bandwidth} m"
  echo "================================================================================================"

  UAVSAT_GRU_ABLATION=full \
  UAVSAT_PARAMETER_TAG="${tag}" \
  UAVSAT_MS2_KF_PRIOR_SIGMA_M="${sigma}" \
  UAVSAT_MS2_KF_PRIOR_WEIGHT="${weight}" \
  UAVSAT_KF_R_POS="${kf_r}" \
  UAVSAT_MS_BANDWIDTH_M="${bandwidth}" \
  UAVSAT_TEMPORAL_EPOCHS="${EPOCHS}" \
  "${PYTHON_BIN}" - <<'PY'
import os
import runpy
import sys
from pathlib import Path

import config

tag = os.environ["UAVSAT_PARAMETER_TAG"].strip()
if not tag or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in tag):
    raise SystemExit("ERROR: invalid UAVSAT_PARAMETER_TAG=%r" % tag)

# Preserve the shared visual-retrieval checkpoint and feature cache. Only the
# temporal experiment output/checkpoints are redirected, so GPU5's GRU-input
# ablations and GPU0's full baseline cannot be overwritten by this sweep.
shared_visual_checkpoint = Path(config.VISUAL_CHECKPOINT)
shared_feature_cache = Path(config.FEATURE_CACHE_DIR)
parameter_root = Path(config.BACKBONE_OUTPUT_DIR) / "parameter_ablation" / tag

config.BACKBONE_OUTPUT_DIR = parameter_root
config.CHECKPOINT_DIR = parameter_root / "checkpoints"
config.VISUAL_CHECKPOINT = shared_visual_checkpoint
config.FEATURE_CACHE_DIR = shared_feature_cache

print("ISOLATED PARAMETER OUTPUT ROOT:", parameter_root, flush=True)
print("SHARED VISUAL CHECKPOINT:", config.VISUAL_CHECKPOINT, flush=True)
print("GRU ABLATION:", config.GRU_ABLATION, flush=True)
print("MS2 SIGMA:", config.MS2_KALMAN_PRIOR_SIGMA_M, flush=True)
print("MS2 WEIGHT:", config.MS2_KALMAN_PRIOR_WEIGHT, flush=True)
print("KF R POSITION:", config.KALMAN_R_POSITION, flush=True)
print("MEANSHIFT BANDWIDTH:", config.MEANSHIFT_BANDWIDTH_M, flush=True)

sys.argv = [
    "train_multirate_a.py",
    "--mode", "all",
    "--temporal-epochs", os.environ.get("UAVSAT_TEMPORAL_EPOCHS", "60"),
    "--train-visual-if-missing",
]
runpy.run_path("train_multirate_a.py", run_name="__main__")
PY
}

run_ms2_group() {
  # Structural ablation: remove the new Kalman-centered MS2 posterior prior.
  run_case "ms2_no_kf_prior" 12.0 0.0 9.0 8.0

  # Prior spatial scale sensitivity around the 12m baseline.
  run_case "ms2_sigma_6m" 6.0 1.0 9.0 8.0
  run_case "ms2_sigma_24m" 24.0 1.0 9.0 8.0

  # Prior strength sensitivity around the weight=1 baseline.
  run_case "ms2_weight_0p5" 12.0 0.5 9.0 8.0
  run_case "ms2_weight_2p0" 12.0 2.0 9.0 8.0
}

run_kf_group() {
  # Measurement-noise sensitivity. Lower R trusts MS1 more; higher R trusts
  # the previous-state prediction more. Baseline R is 9 m^2.
  run_case "kf_r_4" 12.0 1.0 4.0 8.0
  run_case "kf_r_25" 12.0 1.0 25.0 8.0
}

run_meanshift_group() {
  # Decoder-scale sensitivity around the 8m baseline.
  run_case "meanshift_bw_4m" 12.0 1.0 9.0 4.0
  run_case "meanshift_bw_12m" 12.0 1.0 9.0 12.0
}

case "${GROUP}" in
  ms2)
    run_ms2_group
    ;;
  kf)
    run_kf_group
    ;;
  meanshift)
    run_meanshift_group
    ;;
  all)
    run_ms2_group
    run_kf_group
    run_meanshift_group
    ;;
esac

echo
echo "All requested parameter ablations completed."