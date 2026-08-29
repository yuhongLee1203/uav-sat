#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$HERE/.." && pwd)"
mkdir -p "$HERE/output"

# Same-scene BearingUAV protocol:
#   train_1 = 1000 frames, slow, multi-L route
#   train_2 = 1000 frames, fast, multi-L route
#   val_1   = 1000 frames, intermediate speed, multi-L route
# All three use the SAME city-A satellite image. Every selected image keeps its
# own source position label; no synthetic route coordinate is substituted.
python3 "$HERE/prepare_bearinguav_routes.py" \
  --corridor-m "${BEARING_CORRIDOR_M:-14.0}" \
  --min-step-m "${BEARING_MIN_STEP_M:-0.8}" \
  --max-step-m "${BEARING_PREFERRED_MAX_STEP_M:-13.0}" \
  --hard-max-step-m "${BEARING_HARD_MAX_STEP_M:-22.0}" \
  --lookahead-m "${BEARING_LOOKAHEAD_M:-32.0}" \
  --target-train-frames "${BEARING_TRAIN_FRAMES:-1000}" \
  --target-eval-frames "${BEARING_VAL_FRAMES:-1000}" \
  --rebuild

export PYTHONPATH="$HERE:$PARENT${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$HERE/train_multiroute.py" "$@"
