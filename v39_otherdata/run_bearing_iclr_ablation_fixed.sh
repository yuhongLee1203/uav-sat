#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python3 v39_otherdata/patch_bearing_iclr_main_alignment.py \
  v39_otherdata/bearing_iclr_ablation.py

# V5: directly supervise the third-frame second difference through acceleration
# and next-step outputs. No new project file is introduced.
python3 - <<'PY'
from pathlib import Path
p = Path('v39_DirectFinalMS/patch_simple_figure_gru.py')
s = p.read_text(encoding='utf-8')

old_delta2 = '''        # Explicit second-order residual exists only for the 3-frame model.
        if frame_count >= 3:
            delta2_raw = self.delta2_motion_head(accel_h)
            raw_motion = raw_motion + float(config.TEMPORAL_DELTA2_SCALE) * delta2_raw
'''
new_delta2 = '''        # Explicit second-order state exists only for the 3-frame model. V5
        # does not mix it back into generic raw_motion. The four outputs are
        # directly supervised later: [accel_s, accel_e, step_s, step_e].
        if frame_count >= 3:
            delta2_direct = torch.tanh(self.delta2_motion_head(accel_h))
        else:
            delta2_direct = torch.zeros(
                accel_h.shape[0], 4, device=accel_h.device, dtype=accel_h.dtype
            )
'''
if new_delta2 not in s:
    if s.count(old_delta2) != 1:
        raise SystemExit(f'V5 patch failed: delta2 generic block matches={s.count(old_delta2)}')
    s = s.replace(old_delta2, new_delta2, 1)

# The velocity/acceleration tail exists in both the canonical old block and the
# replacement block. Locate the occurrence specifically inside new_motion.
old_accel = '''        velocity = torch.cat([v_parallel, v_cross], dim=1)
        acceleration = torch.cat([a_parallel, a_cross], dim=1)
'''
new_accel = '''        # Frame-3-only direct acceleration correction. Acceleration itself is
        # supervised, so delta2 now receives an explicit training signal.
        if frame_count >= 3:
            d2_scale = float(config.TEMPORAL_DELTA2_SCALE)
            a_parallel = (
                a_parallel
                + d2_scale * delta2_direct[:, 0:1]
                * float(config.TEMPORAL_DIRECT_ACCEL_FORWARD_M)
            ).clamp(
                min=-float(config.MAX_FORWARD_ACCEL_M_PER_FRAME2),
                max=float(config.MAX_FORWARD_ACCEL_M_PER_FRAME2),
            )
            a_cross = (
                a_cross
                + d2_scale * delta2_direct[:, 1:2]
                * float(config.TEMPORAL_DIRECT_ACCEL_CROSS_M)
            ).clamp(
                min=-float(config.MAX_CROSS_ACCEL_M_PER_FRAME2),
                max=float(config.MAX_CROSS_ACCEL_M_PER_FRAME2),
            )
        velocity = torch.cat([v_parallel, v_cross], dim=1)
        acceleration = torch.cat([a_parallel, a_cross], dim=1)
'''
if new_accel not in s:
    new_motion_pos = s.find("new_motion = '''")
    accel_pos = s.find(old_accel, new_motion_pos)
    if new_motion_pos < 0 or accel_pos < 0:
        raise SystemExit('V5 patch failed: could not locate acceleration tail inside new_motion')
    s = s[:accel_pos] + new_accel + s[accel_pos + len(old_accel):]

boundary = "    s = s.replace(old_motion, new_motion, 1)\n\nold_init ="
insertion = r"""    s = s.replace(old_motion, new_motion, 1)

# V5 short gradient path: delta2 directly adjusts next_step, so LOSS_NEXT_STEP
# trains a signal that only the 3-frame model can use.
old_next_step = '''        base_forward = (v_parallel + 0.5 * a_parallel).clamp(
            min=0.0, max=float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
        )
        base_cross = v_cross + 0.5 * a_cross
        effective_heading = heading_residual
        cos_h = torch.cos(effective_heading)
        sin_h = torch.sin(effective_heading)
        next_parallel = base_forward * cos_h - base_cross * sin_h
        next_cross = base_forward * sin_h + base_cross * cos_h
        next_parallel = next_parallel.clamp(min=0.0)
        next_step = torch.cat([next_parallel, next_cross], dim=1)
        norm = torch.linalg.norm(next_step, dim=1, keepdim=True).clamp_min(1e-6)
        scale = torch.clamp(
            float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME) / norm, max=1.0
        )
        next_step = next_step * scale
'''
new_next_step = '''        base_forward = (v_parallel + 0.5 * a_parallel).clamp(
            min=0.0, max=float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
        )
        base_cross = v_cross + 0.5 * a_cross
        effective_heading = heading_residual
        cos_h = torch.cos(effective_heading)
        sin_h = torch.sin(effective_heading)
        next_parallel = base_forward * cos_h - base_cross * sin_h
        next_cross = base_forward * sin_h + base_cross * cos_h

        # Direct second-order next-step residual. This branch is unavailable to
        # 1-frame/2-frame and is directly trained by next_loss.
        if frame_count >= 3:
            d2_scale = float(config.TEMPORAL_DELTA2_SCALE)
            next_parallel = next_parallel + (
                d2_scale * delta2_direct[:, 2:3]
                * float(config.TEMPORAL_DIRECT_STEP_FORWARD_M)
            )
            next_cross = next_cross + (
                d2_scale * delta2_direct[:, 3:4]
                * float(config.TEMPORAL_DIRECT_STEP_CROSS_M)
            )

        next_parallel = next_parallel.clamp(min=0.0)
        next_step = torch.cat([next_parallel, next_cross], dim=1)
        norm = torch.linalg.norm(next_step, dim=1, keepdim=True).clamp_min(1e-6)
        scale = torch.clamp(
            float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME) / norm, max=1.0
        )
        next_step = next_step * scale
'''
if new_next_step not in s:
    if s.count(old_next_step) != 1:
        raise SystemExit(f"V5 patch failed: canonical next-step block matches={s.count(old_next_step)}")
    s = s.replace(old_next_step, new_next_step, 1)

old_init ="""
if 'V5 short gradient path' not in s:
    if boundary not in s:
        raise SystemExit('V5 patch failed: could not locate old_motion -> old_init boundary')
    s = s.replace(boundary, insertion, 1)

old_cfg = 'TEMPORAL_DELTA2_SCALE = float(os.environ.get("UAVSAT_TEMPORAL_DELTA2_SCALE", "1.00"))\n'
new_cfg = old_cfg + '''TEMPORAL_DIRECT_ACCEL_FORWARD_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_ACCEL_FORWARD_M", "1.25"))
TEMPORAL_DIRECT_ACCEL_CROSS_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_ACCEL_CROSS_M", "0.75"))
TEMPORAL_DIRECT_STEP_FORWARD_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_STEP_FORWARD_M", "2.00"))
TEMPORAL_DIRECT_STEP_CROSS_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_STEP_CROSS_M", "1.00"))
'''
if 'TEMPORAL_DIRECT_STEP_FORWARD_M' not in s:
    if s.count(old_cfg) != 1:
        raise SystemExit(f'V5 patch failed: delta2 config matches={s.count(old_cfg)}')
    s = s.replace(old_cfg, new_cfg, 1)

s = s.replace(
    'print("[PATCH OK] 3-frame has dedicated delta2-only residual head")',
    'print("[PATCH OK] 3-frame delta2 directly corrects acceleration + next_step")',
)

compile(s, str(p), 'exec')
p.write_text(s, encoding='utf-8')
print('[TEMPORAL V5] direct delta2 acceleration + next-step supervision: PASS')
PY

# Upgrade the generated experiment runner. Selection is train/validation only.
python3 - <<'PY'
from pathlib import Path
p = Path('v39_otherdata/bearing_iclr_ablation.py')
s = p.read_text(encoding='utf-8')

old = '    base._patch_context_gru(runtime_root)\n'
new = '''    # Legacy 5-block Context-GRU prepatch intentionally disabled here.\n    # patch_simple_figure_gru.py owns the complete temporal architecture.\n'''
if old in s:
    if s.count(old) != 1:
        raise SystemExit(f'expected one legacy Context-GRU call, got {s.count(old)}')
    s = s.replace(old, new, 1)
elif 'Legacy 5-block Context-GRU prepatch intentionally disabled here.' not in s:
    raise SystemExit('could not locate legacy Context-GRU prepatch call')

old_map = '''            ("confidence_power", "KALMAN_CONFIDENCE_POWER"),
        ):
'''
new_map = '''            ("confidence_power", "KALMAN_CONFIDENCE_POWER"),
            ("temporal_3frame_scale", "TEMPORAL_ADAPTER_3FRAME_SCALE"),
            ("delta2_scale", "TEMPORAL_DELTA2_SCALE"),
            ("prior_blend_base", "KALMAN_PRIOR_BLEND_BASE"),
            ("prior_blend_lowconf_gain", "KALMAN_PRIOR_BLEND_LOWCONF_GAIN"),
            ("prior_blend_max", "KALMAN_PRIOR_BLEND_MAX"),
            ("prior_blend_cutoff", "KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF"),
            ("step_relax_confidence", "KALMAN_STEP_RELAX_CONFIDENCE"),
            ("step_relax_width", "KALMAN_STEP_RELAX_WIDTH"),
            ("step_visual_slack_m", "KALMAN_STEP_VISUAL_SLACK_M"),
        ):
'''
if new_map not in s:
    if s.count(old_map) < 1:
        raise SystemExit('could not extend calibration profile loader')
    s = s.replace(old_map, new_map, 1)

start = s.find('def _calibrate_kalman_on_training_validation(')
end = s.find('\ndef train_full(args: argparse.Namespace) -> None:', start)
if start < 0 or end < 0:
    raise SystemExit('could not locate train-only calibration helper')

helper = r'''def _calibrate_kalman_on_training_validation(args, config, tracker, visual, model, cache, route):
    """V5 train-validation selection for direct second-order motion + Kalman."""
    if int(args.train_frames) != 3:
        return None

    gt_state = tracker.build_gt_route_state(cache, route)
    split = tracker.split_ranges(len(cache))
    val_range = split["val"]
    device = tracker.resolve_device()
    profiles = []

    fields = (
        ("fixed_variance_m2", "EXPERIMENT_FIXED_VARIANCE_M2"),
        ("q_progress", "KALMAN_Q_PROGRESS"),
        ("q_cross", "KALMAN_Q_CROSS"),
        ("q_velocity", "KALMAN_Q_VELOCITY"),
        ("confidence_power", "KALMAN_CONFIDENCE_POWER"),
        ("temporal_3frame_scale", "TEMPORAL_ADAPTER_3FRAME_SCALE"),
        ("delta2_scale", "TEMPORAL_DELTA2_SCALE"),
        ("prior_blend_base", "KALMAN_PRIOR_BLEND_BASE"),
        ("prior_blend_lowconf_gain", "KALMAN_PRIOR_BLEND_LOWCONF_GAIN"),
        ("prior_blend_max", "KALMAN_PRIOR_BLEND_MAX"),
        ("prior_blend_cutoff", "KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF"),
        ("step_relax_confidence", "KALMAN_STEP_RELAX_CONFIDENCE"),
        ("step_relax_width", "KALMAN_STEP_RELAX_WIDTH"),
        ("step_visual_slack_m", "KALMAN_STEP_VISUAL_SLACK_M"),
    )

    def snapshot():
        return {key: float(getattr(config, attr)) for key, attr in fields}

    def apply_values(values):
        for key, attr in fields:
            if key in values:
                setattr(config, attr, float(values[key]))

    def evaluate(stage, updates):
        apply_values(updates)
        result = tracker.evaluate_closed_loop(
            model, visual, cache, route, gt_state, val_range, device
        )
        row = snapshot()
        row.update({
            "stage": stage,
            "val_mle_m": float(result["mle"]),
            "val_p90_m": float(result["p90"]),
            "val_speed_mae": float(result["speed_mae"]),
            "val_progress_mae": float(result["progress_mae"]),
        })
        row["objective"] = float(
            row["val_mle_m"]
            + 0.08 * row["val_p90_m"]
            + 0.02 * row["val_speed_mae"]
            + 0.01 * row["val_progress_mae"]
        )
        profiles.append(row)
        return row

    def choose(rows):
        best = min(rows, key=lambda r: (
            r["objective"], r["val_mle_m"], r["val_p90_m"]
        ))
        apply_values(best)
        return best

    baseline = evaluate("baseline", {})

    temporal_rows = []
    for temporal_scale in (0.90, 1.00, 1.10):
        for delta2_scale in (0.00, 0.50, 1.00, 1.50, 2.00):
            temporal_rows.append(evaluate("direct_delta2", {
                "temporal_3frame_scale": temporal_scale,
                "delta2_scale": delta2_scale,
            }))
    temporal_best = choose(temporal_rows + [baseline])

    blend_rows = []
    for base in (0.00, 0.02, 0.05):
        for gain in (0.10, 0.20, 0.30):
            for cutoff in (0.60, 0.70):
                blend_rows.append(evaluate("blend", {
                    "prior_blend_base": base,
                    "prior_blend_lowconf_gain": gain,
                    "prior_blend_max": 0.30,
                    "prior_blend_cutoff": cutoff,
                }))
    blend_best = choose(blend_rows + [temporal_best])

    step_rows = []
    for center, width, slack in (
        (0.00, 0.05, 10.0),
        (0.35, 0.08, 8.0),
        (0.50, 0.08, 6.0),
    ):
        step_rows.append(evaluate("step", {
            "step_relax_confidence": center,
            "step_relax_width": width,
            "step_visual_slack_m": slack,
        }))
    step_best = choose(step_rows + [blend_best])

    filter_rows = []
    for fixed_r in (6.0, 9.0):
        for q_scale in (0.75, 1.00):
            for conf_power in (0.75, 1.00):
                filter_rows.append(evaluate("filter", {
                    "fixed_variance_m2": fixed_r,
                    "q_progress": 1.50 * q_scale,
                    "q_cross": 0.40 * q_scale,
                    "q_velocity": 1.00 * q_scale,
                    "confidence_power": conf_power,
                }))
    best = choose(filter_rows + [step_best])

    selected = snapshot()
    kalman_mode = str(config.EXPERIMENT_KALMAN)
    config.EXPERIMENT_KALMAN = "none"
    no_k = tracker.evaluate_closed_loop(
        model, visual, cache, route, gt_state, val_range, device
    )
    config.EXPERIMENT_KALMAN = kalman_mode
    apply_values(selected)

    payload = {
        "selection_source": "current_city_training_validation_only",
        "city": args.city,
        "validation_range": [int(val_range[0]), int(val_range[1])],
        "criterion": "mle + .08*p90 + .02*speed_mae + .01*progress_mae",
        "search": "direct_delta2_second_order_v5",
        "best": best,
        "validation_no_kalman_diagnostic": {
            "mle_m": float(no_k["mle"]),
            "p90_m": float(no_k["p90"]),
        },
        "profiles": profiles,
        "held_out_navigation_read": False,
    }
    out = _train_root(args, 3) / "kalman_calibration.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[TRAIN-ONLY V5 CALIBRATION]", json.dumps(best, sort_keys=True), flush=True)
    print("[TRAIN-ONLY V5 NO-KALMAN DIAGNOSTIC]", json.dumps(payload["validation_no_kalman_diagnostic"], sort_keys=True), flush=True)
    return payload

'''
s = s[:start] + helper + s[end + 1:]

if 'direct_delta2_second_order_v5' not in s:
    raise SystemExit('V5 calibration marker missing after patch')

compile(s, str(p), 'exec')
p.write_text(s, encoding='utf-8')
print('[PATCH ORDER] legacy 5-block Context-GRU prepatch disabled: PASS')
print('[CALIBRATION V5] direct delta2 + residual Kalman train-validation search: PASS')
PY

# Keep patience=4, promote output naming to V5, and strengthen the losses that
# directly supervise the new second-order branch.
python3 - <<'PY'
from pathlib import Path
p = Path('v39_otherdata/run_bearing_iclr_ablation.sh')
s = p.read_text(encoding='utf-8')
s = s.replace('ablation_v4_', 'ablation_v5_')
s = s.replace('PATIENCE="${PATIENCE:-14}"', 'PATIENCE="${PATIENCE:-4}"')
s = s.replace('UAVSAT_LOSS_NEXT_STEP:-2.5', 'UAVSAT_LOSS_NEXT_STEP:-3.0')
s = s.replace('UAVSAT_LOSS_ACCELERATION:-0.25', 'UAVSAT_LOSS_ACCELERATION:-0.50')
p.write_text(s, encoding='utf-8')
print('[RUNNER V5] patience=4, next-step=3.0, acceleration=0.50: PASS')
PY

python3 -m py_compile \
  v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/patch_bearing_iclr_main_alignment.py \
  v39_otherdata/build_iclr_ablation_tables.py \
  v39_DirectFinalMS/patch_direct_finalms.py \
  v39_DirectFinalMS/patch_simple_figure_gru.py

python3 - <<'PY'
from pathlib import Path
runner = Path('v39_otherdata/bearing_iclr_ablation.py').read_text(encoding='utf-8')
shell = Path('v39_otherdata/run_bearing_iclr_ablation.sh').read_text(encoding='utf-8')
patch = Path('v39_DirectFinalMS/patch_simple_figure_gru.py').read_text(encoding='utf-8')
legacy='weighted'+'_'+'centroid'
checks={
    'single_city_runner': 'CITY="${CITY:-citya}"' in shell and 'CITIES=(' not in shell,
    'other_cities_not_looped': 'for city in' not in shell,
    'patience_is_4': 'PATIENCE="${PATIENCE:-4}"' in shell,
    'fresh_prepare_current_city_only': '--city "${CITY}"' in shell,
    'legacy_context_gru_disabled': 'base._patch_context_gru(runtime_root)' not in runner,
    'active_runner_has_no_legacy_centroid_decoder': legacy not in runner.lower(),
    'runner_requests_front_softms': 'UAVSAT_EXPERIMENT_ANCHOR": "softms"' in runner,
    'quadratic_next_step': 'UAVSAT_EXPERIMENT_MOTION": "quadratic"' in runner,
    'separate_1_2_3_checkpoints': 'checkpoint_frames = int(variant["frames"])' in runner,
    'seven_block_current_delta_delta2_gru': 'feature_dim * 7' in patch and 'current_h = self.uav_projection(z_uav)' in patch,
    'direct_delta2_acceleration': 'TEMPORAL_DIRECT_ACCEL_FORWARD_M' in patch and 'delta2_direct[:, 0:1]' in patch,
    'direct_delta2_next_step': 'TEMPORAL_DIRECT_STEP_FORWARD_M' in patch and 'delta2_direct[:, 2:3]' in patch,
    'smooth_measurement_preserving_kalman': 'KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF' in patch,
    'continuous_step_relaxation': 'KALMAN_STEP_RELAX_WIDTH' in patch,
    'raw_visual_previous_measurement': 'self.last_used_measurement = raw_z.copy()' in patch,
    'v5_train_only_calibration': 'direct_delta2_second_order_v5' in runner,
    'calibration_loads_delta2': '("delta2_scale", "TEMPORAL_DELTA2_SCALE")' in runner,
    'forward_backshift_enabled': 'FORWARD_SEARCH_ORIGIN_BACKSHIFT_M' in runner,
}
for name,ok in checks.items():
    print(f'[PRE-RUN AUDIT] {name}: {"PASS" if ok else "FAIL"}')
if not all(checks.values()):
    raise SystemExit('PRE-RUN AUDIT FAILED')
PY

bash -n v39_otherdata/run_bearing_iclr_ablation.sh
exec bash v39_otherdata/run_bearing_iclr_ablation.sh
