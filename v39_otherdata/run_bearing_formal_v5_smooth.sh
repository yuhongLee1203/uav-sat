#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Apply the model-side smoothing profile before the standard Formal V5 patcher
# copies base_src into each city runtime.
python3 -u v39_otherdata/patch_formal_v5_smooth.py

# Make the inherited formal result uploader target the smooth branch, not the
# original formal branch. This changes only where artifacts are committed.
python3 - <<'PY'
from pathlib import Path
p = Path('v39_otherdata/run_bearing_iclr_ablation.sh')
s = p.read_text(encoding='utf-8')
s = s.replace('bearing-v5-formal-allcities', 'bearing-v5-formal-smooth-v1')
s = s.replace('formal_bearing_v5_allcities_', 'formal_bearing_v5_smooth_')
p.write_text(s, encoding='utf-8')
print('[SMOOTH RUNNER] result upload branch: bearing-v5-formal-smooth-v1')
PY

# Resource-safe defaults inherited from the current formal pipeline.
export CPU_THREADS_PER_CITY="${CPU_THREADS_PER_CITY:-2}"
export CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-128}"
export CPU_NICE="${CPU_NICE:-5}"

# Explicit smooth-v1 profile. These are estimator/training constraints, not
# display post-processing. The plotter still draws raw final_x/final_y.
export UAVSAT_MAX_FORWARD_SPEED_M_PER_FRAME="${UAVSAT_MAX_FORWARD_SPEED_M_PER_FRAME:-12.0}"
export UAVSAT_MAX_CROSS_SPEED_M_PER_FRAME="${UAVSAT_MAX_CROSS_SPEED_M_PER_FRAME:-3.0}"
export UAVSAT_MAX_CROSS_ACCEL_M_PER_FRAME2="${UAVSAT_MAX_CROSS_ACCEL_M_PER_FRAME2:-2.0}"
export UAVSAT_MAX_POLYNOMIAL_STEP_M_PER_FRAME="${UAVSAT_MAX_POLYNOMIAL_STEP_M_PER_FRAME:-12.0}"

# These two are owned by the canonical V5 runtime patch. Keep them here so the
# tighter visual correction is applied without rewriting the base config first.
export UAVSAT_CORR_PARALLEL_M="${UAVSAT_CORR_PARALLEL_M:-0.70}"
export UAVSAT_CORR_CROSS_M="${UAVSAT_CORR_CROSS_M:-0.45}"

export UAVSAT_HEADING_STATE_EMA_ALPHA="${UAVSAT_HEADING_STATE_EMA_ALPHA:-0.22}"
export UAVSAT_TURN_RATE_EMA_ALPHA="${UAVSAT_TURN_RATE_EMA_ALPHA:-0.20}"
export UAVSAT_MAX_HEADING_DELTA_DEG_PER_FRAME="${UAVSAT_MAX_HEADING_DELTA_DEG_PER_FRAME:-3.0}"
export UAVSAT_MAX_TURN_RATE_DELTA_DEG_PER_FRAME2="${UAVSAT_MAX_TURN_RATE_DELTA_DEG_PER_FRAME2:-3.0}"
export UAVSAT_LOSS_CROSS_MOTION_REG="${UAVSAT_LOSS_CROSS_MOTION_REG:-0.03}"
export UAVSAT_KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M="${UAVSAT_KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M:-1.25}"
export UAVSAT_KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME="${UAVSAT_KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME:-0.90}"
export UAVSAT_KALMAN_FINAL_STEP_MAX_M="${UAVSAT_KALMAN_FINAL_STEP_MAX_M:-6.0}"
export UAVSAT_ROUTE_FRAME_SMOOTH_RADIUS_M="${UAVSAT_ROUTE_FRAME_SMOOTH_RADIUS_M:-36.0}"

printf '%s\n' \
  "================================================================================" \
  "FORMAL V5 SMOOTH-v1" \
  "Model output smoothing : estimator-side only" \
  "Plot smoothing         : DISABLED" \
  "Cross speed max        : ${UAVSAT_MAX_CROSS_SPEED_M_PER_FRAME} m/frame" \
  "Cross accel max        : ${UAVSAT_MAX_CROSS_ACCEL_M_PER_FRAME2} m/frame^2" \
  "Visual corr parallel   : ${UAVSAT_CORR_PARALLEL_M} m" \
  "Visual corr cross      : ${UAVSAT_CORR_CROSS_M} m" \
  "Heading EMA alpha      : ${UAVSAT_HEADING_STATE_EMA_ALPHA}" \
  "Heading delta max      : ${UAVSAT_MAX_HEADING_DELTA_DEG_PER_FRAME} deg/frame" \
  "Kalman cross correction: ${UAVSAT_KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M} m" \
  "Kalman final step max  : ${UAVSAT_KALMAN_FINAL_STEP_MAX_M} m" \
  "Route-frame smooth     : ${UAVSAT_ROUTE_FRAME_SMOOTH_RADIUS_M} m" \
  "CPU threads/city       : ${CPU_THREADS_PER_CITY}" \
  "Results branch         : bearing-v5-formal-smooth-v1" \
  "================================================================================"

export UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"

exec nice -n "${CPU_NICE}" bash v39_otherdata/run_bearing_iclr_ablation_fixed.sh
