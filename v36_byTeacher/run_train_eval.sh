#!/usr/bin/env bash
set -euo pipefail

# Autonomous v36_byTeacher runner.
# Usage:
#   bash run_train_eval.sh train
#   bash run_train_eval.sh eval
#   bash run_train_eval.sh all
#
# Optional environment overrides:
#   UAVSAT_BACKBONE=mobilenet_v3_small
#   UAVSAT_TEMPORAL_EPOCHS=60
#   UAVSAT_TEMPORAL_EXTRA_A_STRIDE=2
#   UAVSAT_RUN_TAG=my_run

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

case "${MODE}" in
  train)
    python train_multirate_a.py \
      --temporal-epochs "${UAVSAT_TEMPORAL_EPOCHS:-60}" \
      --extra-stride "${UAVSAT_TEMPORAL_EXTRA_A_STRIDE:-2}" \
      --train-visual-if-missing
    ;;
  eval)
    python robust_tracker.py \
      --mode eval \
      --extra-stride "${UAVSAT_TEMPORAL_EXTRA_A_STRIDE:-2}" \
      --eval-routes route_C route_B
    ;;
  all)
    python train_multirate_a.py \
      --temporal-epochs "${UAVSAT_TEMPORAL_EPOCHS:-60}" \
      --extra-stride "${UAVSAT_TEMPORAL_EXTRA_A_STRIDE:-2}" \
      --train-visual-if-missing
    python robust_tracker.py \
      --mode eval \
      --extra-stride "${UAVSAT_TEMPORAL_EXTRA_A_STRIDE:-2}" \
      --eval-routes route_C route_B
    ;;
  *)
    echo "usage: bash run_train_eval.sh {train|eval|all}" >&2
    exit 2
    ;;
esac
