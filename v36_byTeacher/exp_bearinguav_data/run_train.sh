#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$HERE/.." && pwd)"
mkdir -p "$HERE/output"

# Build v36-compatible temporal routes from the selected BearingUAV samples'
# actual positions.  Query spacing only controls how densely the planned route
# is searched; output frame spacing is variable and constrained by the selected
# samples' real displacement.
python3 "$HERE/prepare_bearinguav_routes.py" \
  --query-spacing-m "${BEARING_QUERY_SPACING_M:-1.5}" \
  --min-step-m "${BEARING_MIN_STEP_M:-1.5}" \
  --max-step-m "${BEARING_MAX_STEP_M:-5.0}" \
  --max-query-error-m "${BEARING_MAX_QUERY_ERROR_M:-8.0}" \
  --rebuild

export PYTHONPATH="$HERE:$PARENT${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$HERE/train_multiroute.py" "$@"
