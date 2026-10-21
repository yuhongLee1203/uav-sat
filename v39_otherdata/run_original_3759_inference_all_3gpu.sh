#!/usr/bin/env bash
# Re-evaluate the exact four-city v13 4 m paper checkpoints; never train.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SOURCE="${ORIGINAL_3759_SOURCE:-$ROOT/v39_otherdata/output/abcd_shared_20260922_121010}"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUTPUT="${ORIGINAL_3759_OUTPUT:-$ROOT/v39_otherdata/output/original3759_inference_${STAMP}}"
DATASET="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing existing inference directory: $OUTPUT" >&2; exit 2; }
python3 - "$SOURCE" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); total=0
for city in ('citya','cityb','cityc','cityd'):
    prepared=root/city/'prepared'
    exp=json.loads((prepared/'experiment.json').read_text())
    assert exp['city']==city and exp['sequence_selection_version']=='soft_sequence_v13_multicity_auto_train_fullroute'
    assert {exp['route_stats'][r]['sample_step_m'] for r in ('train_01','test_01','test_02')}=={4.0}
    total+=sum(exp['route_stats'][r]['frames'] for r in ('test_01','test_02'))
    for frames in (1,2,3):
        ck=root/city/f'train_frames{frames}'/'checkpoints'
        for name in ('visual_retrieval_A_only.pt','controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt'):
            assert (ck/name).is_file(),ck/name
    profile=json.loads((root/city/'train_frames3'/'kalman_calibration.json').read_text())
    assert profile['search']=='positive_three_frame_path_primary_v7'
    assert profile['held_out_navigation_read'] is False
assert total==1482,total
print('[ORIGINAL 3759 SOURCE PASS] four cities, 4 m routes, 1482 test frames, V7 profiles',flush=True)
PY
mkdir -p "$OUTPUT/logs"
for city in citya cityb cityc cityd; do
  mkdir -p "$OUTPUT/$city/prepared"
  cp -a --reflink=auto "$SOURCE/$city/prepared/." "$OUTPUT/$city/prepared/"
  for frames in 1 2 3; do
    target="$OUTPUT/$city/train_frames$frames/checkpoints"
    mkdir -p "$target"
    cp -a --reflink=auto "$SOURCE/$city/train_frames$frames/checkpoints/." "$target/"
  done
  cp -a "$SOURCE/$city/train_frames3/kalman_calibration.json" \
    "$OUTPUT/$city/train_frames3/kalman_calibration.json"
done
export BEARING_V40_RELAX_LIMITS=0
export BEARING_FRAME_ISOLATED_CACHE=1
export BEARING_HEADING_FUSION_ALPHA=0
printf '%s\n' \
  'Inference only: historical ABCD shared v13 GT-assisted checkpoints' \
  "source=$SOURCE" "output=$OUTPUT" \
  'dataset=original v13 4 m prepared routes; City A/B/C/D; 1482 test frames' \
  'weights=historical; no retraining; city-specific gallery binding retained' \
  'kalman=historical positive_three_frame_path_primary_v7 profiles' \
  'GPUs=0,5,6' "started=$(date -Is)" > "$OUTPUT/PROTOCOL.txt"
if [[ "${ORIGINAL_3759_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[ORIGINAL 3759 PREFLIGHT PASS] $OUTPUT"
  exit 0
fi
ABCD_SUITE_ROOT="$OUTPUT" BEARING_DATASET_ROOT="$DATASET" \
  bash v39_otherdata/run_bearing_abcd_full_table_v7.sh \
  2>&1 | tee "$OUTPUT/logs/replay_all_variants.log"
python3 v39_otherdata/build_original_3759_inference_metrics.py --suite-root "$OUTPUT"
python3 v39_otherdata/sanitize_original_3759_metrics.py --suite-root "$OUTPUT"
python3 v39_otherdata/audit_original_3759_publication.py --suite-root "$OUTPUT"
printf '%s\n' "completed=$(date -Is)" >> "$OUTPUT/PROTOCOL.txt"
echo "[RESULTS] $OUTPUT/original_3759_all_metrics.md"
