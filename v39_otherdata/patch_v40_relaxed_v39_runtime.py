#!/usr/bin/env python3
"""Remove v40-identified manual GT motion clamps from an isolated v39 runtime.

This leaves GT-centered visual reference search intact. It changes only the
runtime copy belonging to an opt-in fresh experiment, never canonical source.
"""
from pathlib import Path
import sys

path = Path(sys.argv[1])
s = path.read_text()
changes = (
    ('max_progress_s=(None if bool(getattr(config, "NO_GT_INFERENCE", False)) else float(gt_state["se"][index, 0]))',
     'max_progress_s=None', 6),
    ('max_step_m=(None if bool(getattr(config, "NO_GT_INFERENCE", False)) else float(gt_state["gt_step_norm"][index]))',
     'max_step_m=None', 3),
    ('max_final_step_m=(None if bool(getattr(config, "NO_GT_INFERENCE", False)) else float(gt_state["gt_step_norm"][index]))',
     'max_final_step_m=None', 3),
)
for old, new, count in changes:
    actual = s.count(old)
    if actual != count:
        raise SystemExit(f'v40 runtime patch mismatch: expected {count}, found {actual}: {old[:72]}')
    s = s.replace(old, new)
# Motion labels also follow raw frame-to-frame metadata XY, expressed in the
# model's current route frame. A projected progress delta can otherwise teach
# a wrong speed at corners even when the original XY displacement is correct.
old = '''    if n > 1:
        ds[1:] = np.maximum(0.0, se[1:, 0] - se[:-1, 0])
        de[1:] = se[1:, 1] - se[:-1, 1]
        gt_step_xy[1:] = gt_xy[1:] - gt_xy[:-1]
        gt_step_norm[1:] = np.linalg.norm(gt_step_xy[1:], axis=1)
'''
new = '''    if n > 1:
        gt_step_xy[1:] = gt_xy[1:] - gt_xy[:-1]
        gt_step_norm[1:] = np.linalg.norm(gt_step_xy[1:], axis=1)
        for frame_index in range(1, n):
            unit = route.smooth_route_unit(float(se[frame_index, 0]))
            cross = np.asarray([-unit[1], unit[0]], dtype=np.float64)
            ds[frame_index] = max(0.0, float(np.dot(gt_step_xy[frame_index], unit)))
            de[frame_index] = float(np.dot(gt_step_xy[frame_index], cross))
'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
# The legacy target used the full XY norm as forward speed while also keeping
# its cross component, double-counting lateral motion in the motion label.
old = '''    forward_motion = np.clip(
        gt_step_norm, 0.0, float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
    )
'''
new = '    forward_motion = ds.copy()\n'
assert s.count(old) == 1
s = s.replace(old, new, 1)

# v40 found that monotonic route projection can move an off-route target away
# from the metadata XY. Supervise the measurement head in the original XY.
old = 'def temporal_loss(\n'
assert s.count(old) == 1
helper = '''def v40_route_xy_from_se_torch(route, se):
    points = torch.as_tensor(route.points, dtype=se.dtype, device=se.device)
    units = torch.as_tensor(route.units, dtype=se.dtype, device=se.device)
    cumulative = torch.as_tensor(route.cumulative_s, dtype=se.dtype, device=se.device)
    progress = se[:, 0].clamp(0.0, float(route.total_length_m))
    leg = (torch.bucketize(progress.contiguous(), cumulative, right=True) - 1).clamp(
        0, len(route.units) - 1)
    unit = units[leg]
    center = points[leg] + (progress - cumulative[leg])[:, None] * unit
    cross = torch.stack((-unit[:, 1], unit[:, 0]), dim=1)
    return center + se[:, 1:2] * cross


def temporal_loss(
'''
s = s.replace(old, helper, 1)
old = '    next_reference_se=None,\n):\n    device = output.measurement_se.device'
new = '    next_reference_se=None,\n    target_xy=None,\n    route=None,\n):\n    device = output.measurement_se.device'
assert s.count(old) == 1
s = s.replace(old, new, 1)
old = '''    if captured:
        measurement_loss = F.smooth_l1_loss(output.measurement_se, target_se_t)
        residual = output.measurement_se - target_se_t
        variance = output.measurement_variance_se.clamp_min(
            float(config.KALMAN_R_MIN_VAR)
        )
        variance_nll = 0.5 * (
            residual.square() / variance + variance.log()
        ).mean()
'''
new = '''    if captured:
        if target_xy is None or route is None:
            raise ValueError("v40 direct XY supervision requires metadata XY and route")
        metadata_xy = torch.as_tensor(
            target_xy, dtype=output.measurement_se.dtype,
            device=device).reshape(1, 2)
        measurement_xy = v40_route_xy_from_se_torch(route, output.measurement_se)
        measurement_loss = F.smooth_l1_loss(measurement_xy, metadata_xy)
        residual = measurement_xy - metadata_xy
        variance = output.measurement_variance_se.mean(dim=1, keepdim=True).clamp_min(
            float(config.KALMAN_R_MIN_VAR)
        )
        variance_nll = 0.5 * (
            residual.square() / variance + variance.log()
        ).mean()
'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
old = '                target_se=gt_state["se"][index],\n'
new = ('                target_se=gt_state["se"][index],\n'
       '                target_xy=cache.gt_xy[index].cpu().numpy(),\n'
       '                route=route,\n')
assert s.count(old) == 1
s = s.replace(old, new, 1)
compile(s, str(path), 'exec')
path.write_text(s)
print('[V40 RELAXED V39 RUNTIME] train-only motion bounds; current GT motion caps removed')
