#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python3 v39_otherdata/patch_bearing_iclr_main_alignment.py \
  v39_otherdata/bearing_iclr_ablation.py

# The single-city 7-block temporal patch replaces the legacy 5-block Context-GRU
# patch. Also upgrade the generated runner to v4 train/validation calibration.
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

# Extend the profile loader so the validation-selected temporal and adaptive
# Kalman parameters are frozen before nav50/nav51 evaluation.
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
    """Select 3-frame temporal + residual-Kalman settings on train validation only.

    nav50/nav51 are never read. Calibration is staged to keep the search small:
      1) 3-frame temporal/delta2 residual scale,
      2) confidence-adaptive prior blend,
      3) continuous step-corridor relaxation,
      4) a small final Q/R refinement.
    """
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
        })
        # MLE is the primary paper metric; P90 is a light tail tie-breaker.
        row["objective"] = float(row["val_mle_m"] + 0.10 * row["val_p90_m"])
        profiles.append(row)
        return row

    def choose(rows):
        best = min(rows, key=lambda r: (r["objective"], r["val_mle_m"], r["val_p90_m"]))
        apply_values(best)
        return best

    baseline = evaluate("baseline", {})

    temporal_rows = []
    for temporal_scale, delta2_scale in (
        (0.75, 0.00),
        (0.90, 0.50),
        (1.00, 0.75),
        (1.00, 1.00),
        (1.15, 1.00),
        (1.25, 1.25),
        (1.40, 1.50),
    ):
        temporal_rows.append(evaluate("temporal", {
            "temporal_3frame_scale": temporal_scale,
            "delta2_scale": delta2_scale,
        }))
    temporal_best = choose(temporal_rows + [baseline])

    blend_rows = []
    for gain in (0.00, 0.10, 0.20, 0.30):
        for cutoff in (0.50, 0.60, 0.70):
            blend_rows.append(evaluate("blend", {
                "prior_blend_base": 0.00,
                "prior_blend_lowconf_gain": gain,
                "prior_blend_max": 0.30,
                "prior_blend_cutoff": cutoff,
            }))
    blend_best = choose(blend_rows + [temporal_best])

    step_rows = []
    for center in (0.40, 0.55, 0.70):
        for width in (0.05, 0.10):
            for slack in (2.0, 4.0, 8.0):
                step_rows.append(evaluate("step", {
                    "step_relax_confidence": center,
                    "step_relax_width": width,
                    "step_visual_slack_m": slack,
                }))
    # Near-visual fallback is part of the declared validation search and makes
    # the residual filter capable of approaching the no-Kalman measurement path
    # when the validation data says the visual observation should dominate.
    step_rows.append(evaluate("step", {
        "step_relax_confidence": 0.00,
        "step_relax_width": 0.05,
        "step_visual_slack_m": 10.0,
    }))
    step_best = choose(step_rows + [blend_best])

    filter_rows = []
    for fixed_r in (4.0, 9.0):
        for q_scale in (0.75, 1.25):
            for conf_power in (0.50, 1.00):
                filter_rows.append(evaluate("filter", {
                    "fixed_variance_m2": fixed_r,
                    "q_progress": 1.50 * q_scale,
                    "q_cross": 0.40 * q_scale,
                    "q_velocity": 1.00 * q_scale,
                    "confidence_power": conf_power,
                }))
    best = choose(filter_rows + [step_best])

    # Diagnostic only: no-Kalman validation is recorded but never used to alter
    # held-out results or discard the selected Full profile.
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
        "criterion": "val_mle + 0.10 * val_p90",
        "search": "staged_temporal_blend_step_filter_v4",
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
    print("[TRAIN-ONLY V4 CALIBRATION]", json.dumps(best, sort_keys=True), flush=True)
    print("[TRAIN-ONLY V4 NO-KALMAN DIAGNOSTIC]", json.dumps(payload["validation_no_kalman_diagnostic"], sort_keys=True), flush=True)
    return payload

'''
s = s[:start] + helper + s[end + 1:]

marker = 'staged_temporal_blend_step_filter_v4'
if marker not in s:
    raise SystemExit('v4 calibration marker missing after patch')

compile(s, str(p), 'exec')
p.write_text(s, encoding='utf-8')
print('[PATCH ORDER] legacy 5-block Context-GRU prepatch disabled: PASS')
print('[CALIBRATION V4] temporal + delta2 + blend + step + Q/R train-validation search: PASS')
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
    'early_min_epoch_reduced': 'UAVSAT_EARLY_MIN_EPOCH' in shell,
    'fresh_prepare_current_city_only': '--city "${CITY}"' in shell,
    'legacy_context_gru_disabled': 'base._patch_context_gru(runtime_root)' not in runner,
    'active_runner_has_no_legacy_centroid_decoder': legacy not in runner.lower(),
    'runner_requests_front_softms': 'UAVSAT_EXPERIMENT_ANCHOR": "softms"' in runner,
    'quadratic_next_step': 'UAVSAT_EXPERIMENT_MOTION": "quadratic"' in runner,
    'separate_1_2_3_checkpoints': 'checkpoint_frames = int(variant["frames"])' in runner,
    'seven_block_current_delta_delta2_gru': 'feature_dim * 7' in patch and 'current_h = self.uav_projection(z_uav)' in patch,
    'dedicated_delta2_only_head': 'self.delta2_motion_head' in patch and 'TEMPORAL_DELTA2_SCALE' in patch,
    'smooth_measurement_preserving_kalman': 'KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF' in patch,
    'continuous_step_relaxation': 'KALMAN_STEP_RELAX_WIDTH' in patch,
    'raw_visual_previous_measurement': 'self.last_used_measurement = raw_z.copy()' in patch,
    'staged_train_only_calibration': 'staged_temporal_blend_step_filter_v4' in runner,
    'calibration_loads_delta2': '("delta2_scale", "TEMPORAL_DELTA2_SCALE")' in runner,
    'calibration_loads_blend': '("prior_blend_cutoff", "KALMAN_PRIOR_BLEND_CONFIDENCE_CUTOFF")' in runner,
    'forward_backshift_enabled': 'FORWARD_SEARCH_ORIGIN_BACKSHIFT_M' in runner,
}
for name,ok in checks.items():
    print(f'[PRE-RUN AUDIT] {name}: {"PASS" if ok else "FAIL"}')
if not all(checks.values()):
    raise SystemExit('PRE-RUN AUDIT FAILED')
PY

bash -n v39_otherdata/run_bearing_iclr_ablation.sh
exec bash v39_otherdata/run_bearing_iclr_ablation.sh
