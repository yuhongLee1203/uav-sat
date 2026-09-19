#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python3 v39_otherdata/patch_bearing_iclr_main_alignment.py \
  v39_otherdata/bearing_iclr_ablation.py

python3 -m py_compile \
  v39_otherdata/bearing_iclr_ablation.py \
  v39_otherdata/patch_bearing_iclr_main_alignment.py \
  v39_otherdata/build_iclr_ablation_tables.py \
  v39_DirectFinalMS/patch_direct_finalms.py \
  v39_DirectFinalMS/patch_simple_figure_gru.py

python3 - <<'PY'
from pathlib import Path
p=Path('v39_otherdata/bearing_iclr_ablation.py')
s=p.read_text(encoding='utf-8')
legacy='weighted'+'_'+'centroid'
checks={
    'active_runner_has_no_legacy_centroid_decoder': legacy not in s.lower(),
    'runner_requests_front_softms': 'UAVSAT_EXPERIMENT_ANCHOR": "softms"' in s,
    'temporal_motion_uses_quadratic_next_step': 'UAVSAT_EXPERIMENT_MOTION": "quadratic"' in s,
    'training_city_motion_scale_initialization': 'INIT_FORWARD_SPEED_M_PER_FRAME' in s,
    'residual_temporal_architecture': 'ResidualTemporalGRU' in s,
    'train_only_kalman_calibration': '_calibrate_kalman_on_training_validation' in s,
    'city_native_nav50': '("nav50", "route_B"' in s,
    'city_native_nav51': '("nav51", "route_C"' in s,
    'fair_train_frames_argument': '--train-frames' in s,
    'separate_temporal_checkpoint_selection': 'checkpoint_frames = int(variant["frames"])' in s,
    'forward_backshift_enabled': 'FORWARD_SEARCH_ORIGIN_BACKSHIFT_M' in s,
}
for name,ok in checks.items():
    print(f'[PRE-RUN AUDIT] {name}: {"PASS" if ok else "FAIL"}')
if not all(checks.values()):
    raise SystemExit('PRE-RUN AUDIT FAILED')
PY

python3 - <<'PY'
from pathlib import Path
s=Path('v39_otherdata/run_bearing_iclr_ablation.sh').read_text(encoding='utf-8')
checks={
    'four_city_loop': 'CITIES=(citya cityb cityc cityd)' in s,
    'fresh_prepare_from_raw_dataset': 'bearing_prepare_multicity.py' in s,
    'no_old_generated_reuse': 'ln -s "${existing}"' not in s and 'reuse existing preparation' not in s,
    'acceleration_supervision_enabled': 'UAVSAT_LOSS_ACCELERATION' in s,
    'residual_motion_knobs_enabled': 'UAVSAT_MOTION_RESIDUAL_FORWARD_M' in s,
}
for name,ok in checks.items():
    print(f'[DATA-PROTOCOL AUDIT] {name}: {"PASS" if ok else "FAIL"}')
if not all(checks.values()):
    raise SystemExit('DATA-PROTOCOL AUDIT FAILED')
PY

bash -n v39_otherdata/run_bearing_iclr_ablation.sh
exec bash v39_otherdata/run_bearing_iclr_ablation.sh
