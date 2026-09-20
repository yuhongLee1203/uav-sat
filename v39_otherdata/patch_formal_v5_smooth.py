#!/usr/bin/env python3
from pathlib import Path

p = Path('v39_DirectFinalMS/base_src/config.py')
s = p.read_text(encoding='utf-8')

# Smooth-v1 modifies only knobs that are NOT subsequently rewritten by
# patch_simple_figure_gru.py. In particular, MAX_MEASUREMENT_CORRECTION_CROSS_M
# is intentionally left to the canonical V5 runtime patch, which owns that
# setting via UAVSAT_CORR_CROSS_M. This avoids a double-patch conflict.
replacements = {
    'MAX_FORWARD_SPEED_M_PER_FRAME = 14.0': 'MAX_FORWARD_SPEED_M_PER_FRAME = float(os.environ.get("UAVSAT_MAX_FORWARD_SPEED_M_PER_FRAME", "12.0"))',
    'MAX_CROSS_SPEED_M_PER_FRAME = 5.0': 'MAX_CROSS_SPEED_M_PER_FRAME = float(os.environ.get("UAVSAT_MAX_CROSS_SPEED_M_PER_FRAME", "3.0"))',
    'MAX_CROSS_ACCEL_M_PER_FRAME2 = 4.0': 'MAX_CROSS_ACCEL_M_PER_FRAME2 = float(os.environ.get("UAVSAT_MAX_CROSS_ACCEL_M_PER_FRAME2", "2.0"))',
    'MAX_POLYNOMIAL_STEP_M_PER_FRAME = 14.0': 'MAX_POLYNOMIAL_STEP_M_PER_FRAME = float(os.environ.get("UAVSAT_MAX_POLYNOMIAL_STEP_M_PER_FRAME", "12.0"))',
    'HEADING_STATE_EMA_ALPHA = 0.35': 'HEADING_STATE_EMA_ALPHA = float(os.environ.get("UAVSAT_HEADING_STATE_EMA_ALPHA", "0.22"))',
    'TURN_RATE_EMA_ALPHA = 0.30': 'TURN_RATE_EMA_ALPHA = float(os.environ.get("UAVSAT_TURN_RATE_EMA_ALPHA", "0.20"))',
    'MAX_HEADING_DELTA_DEG_PER_FRAME = 5.0': 'MAX_HEADING_DELTA_DEG_PER_FRAME = float(os.environ.get("UAVSAT_MAX_HEADING_DELTA_DEG_PER_FRAME", "3.0"))',
    'MAX_TURN_RATE_DELTA_DEG_PER_FRAME2 = 5.0': 'MAX_TURN_RATE_DELTA_DEG_PER_FRAME2 = float(os.environ.get("UAVSAT_MAX_TURN_RATE_DELTA_DEG_PER_FRAME2", "3.0"))',
    'LOSS_CROSS_MOTION_REG = 0.0': 'LOSS_CROSS_MOTION_REG = float(os.environ.get("UAVSAT_LOSS_CROSS_MOTION_REG", "0.03"))',
    'KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M = 1.75': 'KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M = float(os.environ.get("UAVSAT_KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M", "1.25"))',
    'KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME = 1.25': 'KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME = float(os.environ.get("UAVSAT_KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME", "0.90"))',
    'KALMAN_FINAL_STEP_MAX_M = 7.00': 'KALMAN_FINAL_STEP_MAX_M = float(os.environ.get("UAVSAT_KALMAN_FINAL_STEP_MAX_M", "6.0"))',
    'ROUTE_FRAME_SMOOTH_RADIUS_M = 24.0': 'ROUTE_FRAME_SMOOTH_RADIUS_M = float(os.environ.get("UAVSAT_ROUTE_FRAME_SMOOTH_RADIUS_M", "36.0"))',
}

for old, new in replacements.items():
    if new in s:
        continue
    if s.count(old) != 1:
        raise SystemExit(f'[SMOOTH PATCH] expected exactly one match for: {old!r}; got {s.count(old)}')
    s = s.replace(old, new, 1)

# The canonical V5 patch expects this exact numeric assignment before it changes
# the bound to UAVSAT_CORR_CROSS_M. Refuse a dirty/double-patched base config.
canonical_cross = 'MAX_MEASUREMENT_CORRECTION_CROSS_M = 4.0'
if canonical_cross not in s:
    raise SystemExit(
        '[SMOOTH PATCH] base config is already modified at '
        'MAX_MEASUREMENT_CORRECTION_CROSS_M; refetch base_src/config.py '
        'from bearing-v5-formal-smooth-v1 before rerunning.'
    )

compile(s, str(p), 'exec')
p.write_text(s, encoding='utf-8')

checks = {
    'cross_speed_3m': 'UAVSAT_MAX_CROSS_SPEED_M_PER_FRAME' in s,
    'cross_accel_2m': 'UAVSAT_MAX_CROSS_ACCEL_M_PER_FRAME2' in s,
    'heading_ema_0p22': 'UAVSAT_HEADING_STATE_EMA_ALPHA' in s,
    'heading_delta_3deg': 'UAVSAT_MAX_HEADING_DELTA_DEG_PER_FRAME' in s,
    'cross_regularizer': 'UAVSAT_LOSS_CROSS_MOTION_REG' in s,
    'kalman_cross_correction': 'UAVSAT_KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M' in s,
    'kalman_velocity_correction': 'UAVSAT_KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME' in s,
    'kalman_final_step': 'UAVSAT_KALMAN_FINAL_STEP_MAX_M' in s,
    'route_frame_smooth': 'UAVSAT_ROUTE_FRAME_SMOOTH_RADIUS_M' in s,
    'canonical_v5_cross_bound_preserved': canonical_cross in s,
}
for k, ok in checks.items():
    print(f'[SMOOTH PATCH] {k}: {"PASS" if ok else "FAIL"}')
if not all(checks.values()):
    raise SystemExit('Smooth patch audit failed')

print('[SMOOTH PATCH] model-side trajectory smoothing enabled; plot smoothing remains disabled')
print('[SMOOTH PATCH] V5 measurement-correction bound ownership preserved')
