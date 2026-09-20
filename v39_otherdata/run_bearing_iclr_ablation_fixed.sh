#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python3 v39_otherdata/patch_bearing_iclr_main_alignment.py \
  v39_otherdata/bearing_iclr_ablation.py

# The new single-city 7-block temporal patch replaces the legacy 5-block
# Context-GRU patch. Do not apply both to the same runtime.
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
compile(s, str(p), 'exec')
p.write_text(s, encoding='utf-8')
print('[PATCH ORDER] legacy 5-block Context-GRU prepatch disabled: PASS')
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
    'fresh_prepare_current_city_only': '--city "${CITY}"' in shell,
    'legacy_context_gru_disabled': 'base._patch_context_gru(runtime_root)' not in runner,
    'active_runner_has_no_legacy_centroid_decoder': legacy not in runner.lower(),
    'runner_requests_front_softms': 'UAVSAT_EXPERIMENT_ANCHOR": "softms"' in runner,
    'quadratic_next_step': 'UAVSAT_EXPERIMENT_MOTION": "quadratic"' in runner,
    'separate_1_2_3_checkpoints': 'checkpoint_frames = int(variant["frames"])' in runner,
    'seven_block_current_delta_delta2_gru': 'feature_dim * 7' in patch and 'current_h = self.uav_projection(z_uav)' in patch,
    'dedicated_temporal_residual_adapter': 'self.temporal_motion_head' in patch and 'TEMPORAL_ADAPTER_3FRAME_SCALE' in patch,
    'measurement_preserving_kalman': 'KALMAN_PRIOR_BLEND_BASE' in patch,
    'confidence_relaxed_step_corridor': 'KALMAN_STEP_RELAX_CONFIDENCE' in patch,
    'raw_visual_previous_measurement': 'self.last_used_measurement = raw_z.copy()' in patch,
    'train_only_kalman_calibration': '_calibrate_kalman_on_training_validation' in runner,
    'forward_backshift_enabled': 'FORWARD_SEARCH_ORIGIN_BACKSHIFT_M' in runner,
}
for name,ok in checks.items():
    print(f'[PRE-RUN AUDIT] {name}: {"PASS" if ok else "FAIL"}')
if not all(checks.values()):
    raise SystemExit('PRE-RUN AUDIT FAILED')
PY

bash -n v39_otherdata/run_bearing_iclr_ablation.sh
exec bash v39_otherdata/run_bearing_iclr_ablation.sh
