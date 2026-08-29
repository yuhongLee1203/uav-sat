#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$HERE/.." && pwd)"
mkdir -p "$HERE/output"

# Build v36-compatible pseudo-sequences from each selected BearingUAV image's
# actual source position.  1.5--5.0 m is the preferred natural step range;
# sparse regions may use a bounded fallback up to 12 m rather than aborting the
# whole route.  No synthetic fixed-spacing position label is written.
python3 "$HERE/prepare_bearinguav_routes.py" \
  --query-spacing-m "${BEARING_QUERY_SPACING_M:-1.5}" \
  --min-step-m "${BEARING_MIN_STEP_M:-1.5}" \
  --max-step-m "${BEARING_MAX_STEP_M:-5.0}" \
  --hard-max-step-m "${BEARING_HARD_MAX_STEP_M:-12.0}" \
  --max-query-error-m "${BEARING_MAX_QUERY_ERROR_M:-8.0}" \
  --rebuild

export PYTHONPATH="$HERE:$PARENT${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$HERE/train_multiroute.py" "$@"
