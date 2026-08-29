#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$HERE/.." && pwd)"
mkdir -p "$HERE/output"

# Build v36-compatible BearingUAV pseudo-sequences from actual source poses.
# train_1/2/3 each contain exactly 1000 real samples selected along long
# multi-segment routes with many 90-degree turns.  Route-specific target
# step-per-frame profiles provide slow/medium/fast training motion; validation
# and test use intermediate effective rates.  No source position label is
# synthesized or moved.
python3 "$HERE/prepare_bearinguav_routes.py" \
  --corridor-m "${BEARING_CORRIDOR_M:-14.0}" \
  --min-step-m "${BEARING_MIN_STEP_M:-0.8}" \
  --max-step-m "${BEARING_PREFERRED_MAX_STEP_M:-13.0}" \
  --hard-max-step-m "${BEARING_HARD_MAX_STEP_M:-22.0}" \
  --lookahead-m "${BEARING_LOOKAHEAD_M:-32.0}" \
  --target-train-frames "${BEARING_TRAIN_FRAMES:-1000}" \
  --target-eval-frames "${BEARING_EVAL_FRAMES:-1000}" \
  --rebuild

export PYTHONPATH="$HERE:$PARENT${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$HERE/train_multiroute.py" "$@"
