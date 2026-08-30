#!/usr/bin/env bash
set -euo pipefail

# v36_byTeacher v8r1 MODEL hyperparameter sensitivity study.
# This is intentionally different from the GPU5 component ablation:
#   GPU5 removes one GRU input branch at a time.
#   GPU6 keeps the full four-branch GRU and changes model-capacity/
#   regularization hyperparameters that are NOT learned automatically.
#
# Normal full baseline (do not rerun here):
#   hidden_dim=256, feature_dim=128, dropout=0.0
#
# GPU6 cases:
#   hidden: 128 / [256 baseline] / 512
#   feature projection: 64 / [128 baseline] / 256
#   dropout: [0.0 baseline] / 0.1 / 0.2
#
# Every case MUST retrain because changing hidden/feature/dropout changes the
# trainable model or its optimization behavior. Frozen visual checkpoint and
# UAV feature cache are shared; temporal outputs/checkpoints are isolated under:
#   output/<backbone>/model_hparam_ablation/<case>/...

GROUP="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS="${UAVSAT_CPU_THREADS:-2}"
EPOCHS="${UAVSAT_TEMPORAL_EPOCHS:-60}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

case "${GROUP}" in
  all|hidden|feature|dropout) ;;
  *)
    echo "usage: bash run_parameter_ablation.sh {all|hidden|feature|dropout}" >&2
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

"${PYTHON_BIN}" -m py_compile \
  config.py data.py visual_model.py visual_localizer.py \
  robust_tracker_base.py robust_tracker.py train_multirate_a.py

echo "MODEL hyperparameter ablation: CPU threads=${CPU_THREADS}; epochs=${EPOCHS}" >&2

run_case() {
  local tag="$1"
  local hidden="$2"
  local feature="$3"
  local dropout="$4"

  echo
  echo "================================================================================================"
  echo "MODEL HYPERPARAM CASE: ${tag}"
  echo "  GRU input branches   = FULL (all four branches)"
  echo "  GRU hidden dim       = ${hidden}"
  echo "  branch feature dim   = ${feature}"
  echo "  GRU/head dropout     = ${dropout}"
  echo "  TBPTT                = 32 (fixed)"
  echo "  temporal LR          = 3e-4 (fixed)"
  echo "================================================================================================"

  UAVSAT_GRU_ABLATION=full \
  UAVSAT_MODEL_HPARAM_TAG="${tag}" \
  UAVSAT_RNN_HIDDEN="${hidden}" \
  UAVSAT_RNN_FEATURE="${feature}" \
  UAVSAT_RNN_DROPOUT="${dropout}" \
  UAVSAT_TBPTT_STEPS=32 \
  UAVSAT_TEMPORAL_LR=3e-4 \
  UAVSAT_TEMPORAL_EPOCHS="${EPOCHS}" \
  UAVSAT_CPU_THREADS="${CPU_THREADS}" \
  "${PYTHON_BIN}" - <<'PY'
import os
import runpy
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

tag = os.environ["UAVSAT_MODEL_HPARAM_TAG"].strip()
if not tag or any(
    ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    for ch in tag
):
    raise SystemExit("ERROR: invalid UAVSAT_MODEL_HPARAM_TAG=%r" % tag)
if config.GRU_ABLATION != "full":
    raise SystemExit("ERROR: GPU6 model-hparam study must keep the full GRU")

# Keep the expensive frozen visual model and UAV feature cache shared.
shared_backbone_root = Path(config.BACKBONE_OUTPUT_DIR)
shared_visual_checkpoint = Path(config.VISUAL_CHECKPOINT)
shared_feature_cache = Path(config.FEATURE_CACHE_DIR)

# Isolate ALL temporal outputs/checkpoints from GPU0/GPU5 and other GPU6 cases.
case_root = shared_backbone_root / "model_hparam_ablation" / tag
config.BACKBONE_OUTPUT_DIR = case_root
config.CHECKPOINT_DIR = case_root / "checkpoints"
config.VISUAL_CHECKPOINT = shared_visual_checkpoint
config.FEATURE_CACHE_DIR = shared_feature_cache

expected_input = int(config.RNN_FEATURE_DIM) * len(config.GRU_ACTIVE_GROUPS)
print("CPU THREAD LIMIT:", threads, flush=True)
print("MODEL HPARAM TAG:", tag, flush=True)
print("FULL GRU ACTIVE GROUPS:", ",".join(config.GRU_ACTIVE_GROUPS), flush=True)
print("RNN HIDDEN DIM:", int(config.RNN_HIDDEN_DIM), flush=True)
print("RNN FEATURE DIM:", int(config.RNN_FEATURE_DIM), flush=True)
print("RNN COMBINED INPUT DIM:", expected_input, flush=True)
print("RNN DROPOUT:", float(config.RNN_DROPOUT), flush=True)
print("ISOLATED CASE ROOT:", case_root, flush=True)
print("SHARED VISUAL CHECKPOINT:", shared_visual_checkpoint, flush=True)
print("SHARED FEATURE CACHE:", shared_feature_cache, flush=True)

sys.argv = [
    "train_multirate_a.py",
    "--mode", "all",
    "--temporal-epochs", os.environ.get("UAVSAT_TEMPORAL_EPOCHS", "60"),
    "--train-visual-if-missing",
]
runpy.run_path("train_multirate_a.py", run_name="__main__")
PY
}

run_hidden_group() {
  # Baseline hidden=256 is provided by the normal full v8r1 experiment.
  run_case "hidden_128" 128 128 0.0
  run_case "hidden_512" 512 128 0.0
}

run_feature_group() {
  # Baseline projected branch feature=128 is provided by the normal full run.
  run_case "feature_64" 256 64 0.0
  run_case "feature_256" 256 256 0.0
}

run_dropout_group() {
  # Baseline dropout=0.0 is provided by the normal full run.
  run_case "dropout_0p1" 256 128 0.1
  run_case "dropout_0p2" 256 128 0.2
}

case "${GROUP}" in
  hidden) run_hidden_group ;;
  feature) run_feature_group ;;
  dropout) run_dropout_group ;;
  all)
    run_hidden_group
    run_feature_group
    run_dropout_group
    ;;
esac

echo
echo "All requested MODEL hyperparameter ablations completed."
