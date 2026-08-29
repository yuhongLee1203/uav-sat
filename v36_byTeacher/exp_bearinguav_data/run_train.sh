#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$HERE/.." && pwd)"
mkdir -p "$HERE/output"

# Same-scene BearingUAV protocol:
#   train_1 = up to 600 frames, slower irregular route
#   train_2 = up to 600 frames, faster irregular route
#   val_1   = up to 600 frames, intermediate irregular route
# All three use the SAME city-A satellite image. Routes contain long straight
# segments at arbitrary angles; different routes may cross. Every selected
# image keeps its own source position label.
python3 "$HERE/prepare_bearinguav_routes.py" \
  --corridor-m "${BEARING_CORRIDOR_M:-18.0}" \
  --min-step-m "${BEARING_MIN_STEP_M:-0.8}" \
  --max-step-m "${BEARING_PREFERRED_MAX_STEP_M:-13.0}" \
  --hard-max-step-m "${BEARING_HARD_MAX_STEP_M:-22.0}" \
  --lookahead-m "${BEARING_LOOKAHEAD_M:-34.0}" \
  --target-train-frames "${BEARING_TRAIN_FRAMES:-600}" \
  --target-eval-frames "${BEARING_VAL_FRAMES:-600}" \
  --min-accepted-frames "${BEARING_MIN_ACCEPTED_FRAMES:-400}" \
  --rebuild

export PYTHONPATH="$HERE:$PARENT${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$HERE/train_multiroute.py" "$@"
