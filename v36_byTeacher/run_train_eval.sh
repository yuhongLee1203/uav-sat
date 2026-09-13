#!/usr/bin/env bash
set -euo pipefail

# v36_byTeacher v8
# Route A: visual + temporal training
# Route B: whole-route validation / model selection
# Route C: final test only
# No current-frame reference coordinate is used by the estimator.

cd "$(dirname "$0")"

VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-15}"

python robust_tracker.py \
  --mode train_eval \
  --visual-epochs "${VISUAL_EPOCHS}" \
  --temporal-epochs "${TEMPORAL_EPOCHS}" \
  --patience "${PATIENCE}" \
  --measure-latency
