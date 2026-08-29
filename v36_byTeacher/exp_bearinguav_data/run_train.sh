#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$HERE/.." && pwd)"
mkdir -p "$HERE/output"

# Build v36-compatible BearingUAV pseudo-sequences from actual source poses.
# The adapter now collects all real samples inside a route corridor, orders them
# by route progress, and forms a bounded variable-step chain.  It does not chase
# fixed query points and it never writes synthetic fixed-spacing position labels.
python3 "$HERE/prepare_bearinguav_routes.py" \
  --corridor-m "${BEARING_CORRIDOR_M:-14.0}" \
  --min-step-m "${BEARING_MIN_STEP_M:-1.0}" \
  --preferred-step-m "${BEARING_PREFERRED_STEP_M:-5.5}" \
  --max-step-m "${BEARING_PREFERRED_MAX_STEP_M:-10.0}" \
  --hard-max-step-m "${BEARING_HARD_MAX_STEP_M:-18.0}" \
  --lookahead-m "${BEARING_LOOKAHEAD_M:-24.0}" \
  --rebuild

export PYTHONPATH="$HERE:$PARENT${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$HERE/train_multiroute.py" "$@"
