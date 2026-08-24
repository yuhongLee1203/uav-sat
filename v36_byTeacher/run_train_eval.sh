#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p output/checkpoints
python3 -u robust_tracker.py \
  --mode train_eval \
  --reuse-visual \
  --visual-epochs 30 \
  --temporal-epochs 60 \
  --patience 15 \
  --jitter-m 8 \
  --forward-rows 3 \
  "$@"
