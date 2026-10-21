#!/usr/bin/env bash
# Fresh four-city v13 (4 m) GT-assisted v39 training with v40 limit audit fixes.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SOURCE="${V40_V39_V13_SOURCE:-$ROOT/v39_otherdata/output/abcd_shared_20260922_121010}"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
SUITE="${V40_V39_V13_OUTPUT:-$ROOT/v40/output/v39_v40_relaxed_v13_${STAMP}}"
DATASET="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
if [[ -e "$SUITE" ]]; then
  echo "Refusing existing directory (fresh training required): $SUITE" >&2
  exit 2
fi
python3 - "$SOURCE" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); total=0
for city in ('citya','cityb','cityc','cityd'):
    p=root/city/'prepared'/'experiment.json'
    x=json.loads(p.read_text())
    assert x['city']==city and x['sequence_selection_version']=='soft_sequence_v13_multicity_auto_train_fullroute'
    assert {x['route_stats'][r]['sample_step_m'] for r in ('train_01','test_01','test_02')}=={4.0}
    for route in ('train_01','test_01','test_02'):
        assert (p.parent/'routes'/route/'manifest.csv').is_file()
    total+=sum(x['route_stats'][r]['frames'] for r in ('test_01','test_02'))
assert total==1482, total
print('[V13 FOUR-CITY 4M SOURCE AUDIT PASS] test frames:',total,flush=True)
PY
mkdir -p "$SUITE/logs"
for city in citya cityb cityc cityd; do
  mkdir -p "$SUITE/$city/prepared"
  cp -a --reflink=auto "$SOURCE/$city/prepared/." "$SUITE/$city/prepared/"
done
export BEARING_V40_RELAX_LIMITS=1
export BEARING_FRAME_ISOLATED_CACHE=1
export BEARING_FREEZE_VISUAL=0
export OMP_NUM_THREADS="${CPU_THREADS_PER_JOB:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
printf '%s\n' \
  'Original four-city v13 4 m prepared routes; fresh GT-assisted v39 training' \
  "prepared_source=$SOURCE" \
  'no historical checkpoint imported' \
  'v40 fixes=train-only dynamic motion bounds, no GT progress/step clipping, direct metadata-XY measurement and motion labels, no route-frame smoothing, isolated feature caches' \
  'GT remains an inference input for the controlled reference patch search' \
  "started=$(date -Is)" > "$SUITE/PROTOCOL.txt"
for city in citya cityb cityc cityd; do
  python3 v39_otherdata/bearing_iclr_ablation.py check \
    --suite-root "$SUITE" --dataset-root "$DATASET" --city "$city"
done
PYTHONPATH=v39_otherdata python3 - "$SUITE" <<'PY'
import json,sys
from pathlib import Path
from bearing_iclr_ablation import _make_runtime
from v40_train_route_limit_profile import build_profile
root=Path(sys.argv[1])
for city in ('citya','cityb','cityc','cityd'):
    prepared=root/city/'prepared'
    profile=build_profile(prepared)
    assert profile['source']=='train_01_manifest_only'
    assert profile['kalman_final_step_max_m'] > profile['train_step_max_m']
    print('[V40 RELAX PROFILE]',city,'max train step',round(profile['train_step_max_m'],3),'m',flush=True)
_make_runtime(root/'citya'/'prepared',root/'logs'/'runtime_preflight')
print('[V40 RELAX RUNTIME PATCH PASS]',flush=True)
PY
if [[ "${V40_V39_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[PREFLIGHT ONLY] $SUITE"
  exit 0
fi
pids=()
for spec in '3 0' '2 5' '1 6'; do
  read -r frames gpu <<< "$spec"
  (
    ABCD_SUITE_ROOT="$SUITE" BEARING_DATASET_ROOT="$DATASET" \
    ABCD_SKIP_ALIGNMENT_PATCH=1 ABCD_FRAMES="$frames" ABCD_TRAIN_ONLY=1 TRAIN_GPU="$gpu" \
    bash v39_otherdata/run_bearing_abcd_shared_ablation.sh
  ) > "$SUITE/logs/train_frames${frames}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "[TRAIN START] frames=$frames gpu=$gpu pid=${pids[-1]}"
done
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "Training failed; see $SUITE/logs/train_frames*_gpu*.log" >&2; exit 3; }
done
ABCD_SUITE_ROOT="$SUITE" BEARING_DATASET_ROOT="$DATASET" \
ABCD_SKIP_ALIGNMENT_PATCH=1 ABCD_EVAL_ONLY=1 ABCD_EVAL_GPUS='0 5 6' \
bash v39_otherdata/run_bearing_abcd_shared_ablation.sh \
  2>&1 | tee "$SUITE/logs/evaluate_all_ablation.log"
python3 v40/build_v39_v40_relaxed_v13_report.py --suite-root "$SUITE"
printf '%s\n' "completed=$(date -Is)" >> "$SUITE/PROTOCOL.txt"
echo "[RESULTS] $SUITE/v39_v40_relaxed_ablation.md"
