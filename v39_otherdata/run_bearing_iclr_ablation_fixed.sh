#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# 1) Start from the audited city-native alignment patch.
python3 v39_otherdata/patch_bearing_iclr_main_alignment.py \
  v39_otherdata/bearing_iclr_ablation.py

# 2) Freeze the V5 third-frame architecture directly into the patch source that
#    will be applied to every city runtime.
python3 - <<'PY'
from pathlib import Path

p = Path('v39_DirectFinalMS/patch_simple_figure_gru.py')
s = p.read_text(encoding='utf-8')

old_delta2 = '''        # Explicit second-order residual exists only for the 3-frame model.
        if frame_count >= 3:
            delta2_raw = self.delta2_motion_head(accel_h)
            raw_motion = raw_motion + float(config.TEMPORAL_DELTA2_SCALE) * delta2_raw
'''
new_delta2 = '''        # V5: third-frame-only second-order state. It is directly supervised
        # through acceleration and next-step outputs instead of being mixed back
        # into the generic motion residual.
        if frame_count >= 3:
            delta2_direct = torch.tanh(self.delta2_motion_head(accel_h))
        else:
            delta2_direct = torch.zeros(
                accel_h.shape[0], 4, device=accel_h.device, dtype=accel_h.dtype
            )
'''
if 'delta2_direct = torch.tanh(self.delta2_motion_head(accel_h))' not in s:
    if s.count(old_delta2) != 1:
        raise SystemExit(f'V5 patch failed: generic delta2 block matches={s.count(old_delta2)}')
    s = s.replace(old_delta2, new_delta2, 1)

old_accel = '''        velocity = torch.cat([v_parallel, v_cross], dim=1)
        acceleration = torch.cat([a_parallel, a_cross], dim=1)
'''
new_accel = '''        # V5 direct acceleration correction: available only to 3-frame.
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
if 'TEMPORAL_DIRECT_ACCEL_FORWARD_M' not in s.split('required_model =', 1)[0]:
    new_motion_pos = s.find("new_motion = '''")
    accel_pos = s.find(old_accel, new_motion_pos)
    if new_motion_pos < 0 or accel_pos < 0:
        raise SystemExit('V5 patch failed: acceleration tail not found inside new_motion')
    s = s[:accel_pos] + new_accel + s[accel_pos + len(old_accel):]

# Insert V5 next-step replacement logic into the generated patch, after the
# generic motion block is installed but before initialization replacement.
if 'V5_DIRECT_NEXT_STEP_PATCH' not in s:
    boundary = "    s = s.replace(old_motion, new_motion, 1)\n\nold_init ="
    if boundary not in s:
        raise SystemExit('V5 patch failed: old_motion/old_init boundary not found')
    insertion = r'''    s = s.replace(old_motion, new_motion, 1)

# V5_DIRECT_NEXT_STEP_PATCH
old_next_step = ''' + "'''" + r'''        base_forward = (v_parallel + 0.5 * a_parallel).clamp(
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
''' + "'''" + r'''
new_next_step = ''' + "'''" + r'''        base_forward = (v_parallel + 0.5 * a_parallel).clamp(
            min=0.0, max=float(config.MAX_POLYNOMIAL_STEP_M_PER_FRAME)
        )
        base_cross = v_cross + 0.5 * a_cross
        effective_heading = heading_residual
        cos_h = torch.cos(effective_heading)
        sin_h = torch.sin(effective_heading)
        next_parallel = base_forward * cos_h - base_cross * sin_h
        next_cross = base_forward * sin_h + base_cross * cos_h
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
''' + "'''" + r'''
if new_next_step not in s:
    if s.count(old_next_step) != 1:
        raise SystemExit(f"V5 patch failed: canonical next-step block matches={s.count(old_next_step)}")
    s = s.replace(old_next_step, new_next_step, 1)

old_init ='''
    s = s.replace(boundary, insertion, 1)

# Critical V5 config definitions. Check real assignment lines, not token mentions
# inside generated model source.
config_needle = 'TEMPORAL_DELTA2_SCALE = float(os.environ.get("UAVSAT_TEMPORAL_DELTA2_SCALE", "1.00"))\\n'
config_add = config_needle + '''TEMPORAL_DIRECT_ACCEL_FORWARD_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_ACCEL_FORWARD_M", "1.25"))\nTEMPORAL_DIRECT_ACCEL_CROSS_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_ACCEL_CROSS_M", "0.75"))\nTEMPORAL_DIRECT_STEP_FORWARD_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_STEP_FORWARD_M", "2.00"))\nTEMPORAL_DIRECT_STEP_CROSS_M = float(os.environ.get("UAVSAT_TEMPORAL_DIRECT_STEP_CROSS_M", "1.00"))\n'''
config_marker = 'TEMPORAL_DIRECT_ACCEL_FORWARD_M = float(os.environ.get('
if config_marker not in s:
    if s.count(config_needle) != 1:
        raise SystemExit(f'V5 config patch failed: TEMPORAL_DELTA2_SCALE definition matches={s.count(config_needle)}')
    s = s.replace(config_needle, config_add, 1)

s = s.replace(
    'print("[PATCH OK] 3-frame has dedicated delta2-only residual head")',
    'print("[PATCH OK] V5 3-frame direct acceleration + next-step correction")',
)

compile(s, str(p), 'exec')
p.write_text(s, encoding='utf-8')
print('[FORMAL V5] temporal architecture patch: PASS')
PY

# 3) Make the generated experiment runner use the 7-block V5 patch directly,
#    and load all validation-selected V5 temporal/Kalman fields.
python3 - <<'PY'
from pathlib import Path
p = Path('v39_otherdata/bearing_iclr_ablation.py')
s = p.read_text(encoding='utf-8')

old = '    base._patch_context_gru(runtime_root)\n'
new = '''    # Legacy 5-block Context-GRU disabled; V5 owns the complete temporal model.\n'''
if old in s:
    if s.count(old) != 1:
        raise SystemExit(f'expected one legacy Context-GRU call, got {s.count(old)}')
    s = s.replace(old, new, 1)
elif 'Legacy 5-block Context-GRU disabled' not in s:
    raise SystemExit('legacy Context-GRU call not found')

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
if '("delta2_scale", "TEMPORAL_DELTA2_SCALE")' not in s:
    if old_map not in s:
        raise SystemExit('calibration profile loader insertion point not found')
    s = s.replace(old_map, new_map, 1)

start = s.find('def _calibrate_kalman_on_training_validation(')
end = s.find('\ndef train_full(args: argparse.Namespace) -> None:', start)
if start < 0 or end < 0:
    raise SystemExit('train-only calibration helper not found')

helper = r'''def _calibrate_kalman_on_training_validation(args, config, tracker, visual, model, cache, route):
    """Formal V5 calibration using only the current city's training validation."""
    if int(args.train_frames) != 3:
        return None

    gt_state = tracker.build_gt_route_state(cache, route)
    val_range = tracker.split_ranges(len(cache))["val"]
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
        return {k: float(getattr(config, a)) for k, a in fields}

    def apply(values):
        for k, a in fields:
            if k in values:
                setattr(config, a, float(values[k]))

    def evaluate(stage, updates):
        apply(updates)
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
        best = min(rows, key=lambda r: (r["objective"], r["val_mle_m"], r["val_p90_m"]))
        apply(best)
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
    apply(selected)

    payload = {
        "selection_source": "current_city_training_validation_only",
        "city": args.city,
        "validation_range": [int(val_range[0]), int(val_range[1])],
        "criterion": "mle + .08*p90 + .02*speed_mae + .01*progress_mae",
        "search": "formal_v5_direct_delta2_residual_kalman",
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
    print("[FORMAL TRAIN-ONLY V5 CALIBRATION]", json.dumps(best, sort_keys=True), flush=True)
    return payload

'''
s = s[:start] + helper + s[end + 1:]

compile(s, str(p), 'exec')
p.write_text(s, encoding='utf-8')
print('[FORMAL V5] runner alignment + train-only calibration: PASS')
PY

# Match the successful CityA V5 baseline before each city's validation selects
# its own final profile.
export UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2="${UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2:-25.0}"

python3 -m py_compile \
  v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/patch_bearing_iclr_main_alignment.py \
  v39_otherdata/bearing_prepare_multicity.py \
  v39_DirectFinalMS/patch_direct_finalms.py \
  v39_DirectFinalMS/patch_simple_figure_gru.py

python3 - <<'PY'
from pathlib import Path
runner = Path('v39_otherdata/bearing_iclr_ablation.py').read_text(encoding='utf-8')
shell = Path('v39_otherdata/run_bearing_iclr_ablation.sh').read_text(encoding='utf-8')
patch = Path('v39_DirectFinalMS/patch_simple_figure_gru.py').read_text(encoding='utf-8')
checks = {
    'formal_four_cities': 'CITIES=(citya cityb cityc cityd)' in shell,
    'formal_full_only': '--variant full' in shell and 'no_kalman' not in shell,
    'gpu_0_5_6': 'GPUS=(0 5 6)' in shell,
    'dynamic_gpu_scheduler': 'wait -n -p done_pid' in shell,
    'patience_4': 'PATIENCE="${PATIENCE:-4}"' in shell,
    'legacy_context_gru_disabled': 'base._patch_context_gru(runtime_root)' not in runner,
    'seven_block_gru_patch': 'feature_dim * 7' in patch,
    'direct_delta2_acceleration': 'TEMPORAL_DIRECT_ACCEL_FORWARD_M' in patch,
    'direct_delta2_next_step': 'TEMPORAL_DIRECT_STEP_FORWARD_M' in patch,
    'formal_train_only_calibration': 'formal_v5_direct_delta2_residual_kalman' in runner,
    'no_heldout_calibration': 'held_out_navigation_read": False' in runner,
}
for name, ok in checks.items():
    print(f'[FORMAL PRE-RUN AUDIT] {name}: {"PASS" if ok else "FAIL"}')
if not all(checks.values()):
    raise SystemExit('FORMAL PRE-RUN AUDIT FAILED')
PY

bash -n v39_otherdata/run_bearing_iclr_ablation.sh
exec bash v39_otherdata/run_bearing_iclr_ablation.sh
