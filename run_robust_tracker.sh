#!/usr/bin/env bash
set -euo pipefail

# Fixed default protocol:
#   Train/validation: Route A only
#   Evaluation: unseen Route B and Route C test splits
#
# Extra arguments are appended last, so commands such as
# --resume, --epochs 40, --jitter-m 12 or --mode eval can
# still be supplied when running this script.

OUTPUT_DIR="outputs/temporal_prior_hardms_cross_route/train_A_test_BC"
mkdir -p "${OUTPUT_DIR}"

python3 robust_tracker.py \
  --mode train_eval \
  --train-routes route_A \
  --eval-routes route_B route_C \
  --eval-split test \
  --experiment-name train_A_test_BC \
  "$@" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"