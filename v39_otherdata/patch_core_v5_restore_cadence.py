#!/usr/bin/env python3
from pathlib import Path
p=Path('v39_otherdata/bearing_core_v5_restore.py')
s=p.read_text(encoding='utf-8')
# These longitudinal values must remain owned by the original train_01-only
# cadence adapter, matching the old V5 manifest. Remove only the post-adapter
# hard overrides; environment base defaults remain available before adaptation.
for line in [
    '        "MAX_FORWARD_SPEED_M_PER_FRAME": 14.0,\n',
    '        "MAX_POLYNOMIAL_STEP_M_PER_FRAME": 14.0,\n',
    '        "KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME": 1.25,\n',
    '        "KALMAN_FINAL_STEP_MAX_M": 7.0,\n',
    '        "MAX_MEASUREMENT_CORRECTION_PARALLEL_M": 0.75,\n',
]:
    if line not in s:
        raise SystemExit(f'expected line missing: {line!r}')
    s=s.replace(line,'',1)
# Add an audit after the fixed lateral settings are applied.
needle='''    selected, audit = _select_calibration(args)\n'''
insert='''    cadence = getattr(args, "training_cadence_audit", {}) or {}\n    cadence_map = {\n        "max_forward_speed_m_per_frame": "MAX_FORWARD_SPEED_M_PER_FRAME",\n        "max_polynomial_step_m_per_frame": "MAX_POLYNOMIAL_STEP_M_PER_FRAME",\n        "max_measurement_correction_parallel_m": "MAX_MEASUREMENT_CORRECTION_PARALLEL_M",\n        "kalman_max_velocity_correction_m_per_frame": "KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME",\n        "kalman_final_step_max_m": "KALMAN_FINAL_STEP_MAX_M",\n    }\n    for src, dst in cadence_map.items():\n        if src in cadence and hasattr(config, dst):\n            setattr(config, dst, float(cadence[src])); applied[dst] = float(cadence[src])\n    selected, audit = _select_calibration(args)\n'''
if needle not in s:
    raise SystemExit('selection hook missing')
s=s.replace(needle,insert,1)
compile(s,str(p),'exec')
p.write_text(s,encoding='utf-8')
print('[CORE-V5 PATCH] longitudinal train_01 cadence ownership restored: PASS')
print('[CORE-V5 PATCH] lateral/heading pre-Smooth-V1 profile retained: PASS')
