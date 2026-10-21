#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BASE="${GT_3759_SOURCE:-$ROOT/v39_otherdata/output/abcd_shared_20260922_121010}"
OFFICIAL="${GT_3759_OFFICIAL_SOURCE:-$ROOT/v39_otherdata/output/abcd_table6_official96_temporal_20260923}"
OUTPUT="${GT_3759_OUTPUT:-$ROOT/v39_otherdata/output/official96_gt_3759_transfer_20260925}"
REPRO="$OUTPUT/reproduction_v13_4m"
TRANSFER="$OUTPUT/official96_temporal_8m"
DATASET="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
export OMP_NUM_THREADS="${CPU_THREADS_PER_JOB:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$OUTPUT/logs" "$REPRO/logs" "$TRANSFER/logs"

# The exact 3.759 m artifact contains one shared checkpoint per frame count,
# rebound to each city's satellite gallery, and a city-specific V7 profile.
for city in citya cityb cityc cityd; do
  for frames in 1 2 3; do
    src="$BASE/$city/train_frames$frames/checkpoints"
    test -s "$src/visual_retrieval_A_only.pt"
    test -s "$src/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  done
  python3 - "$BASE/$city/train_frames3/kalman_calibration.json" <<'PY'
import json,sys
profile=json.load(open(sys.argv[1]))
assert profile['search']=='positive_three_frame_path_primary_v7'
assert profile['held_out_navigation_read'] is False
PY
  test -s "$BASE/$city/prepared/experiment.json"
  test -s "$OFFICIAL/$city/prepared/official96_split_audit.json"
done

python3 - "$OFFICIAL" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); hashes=None
for city in ('citya','cityb','cityc','cityd'):
    audit=json.loads((root/city/'prepared'/'official96_split_audit.json').read_text())
    assert audit['official_dataset_id']==96 and audit['split_seed']==42
    assert audit['split_counts_global']=={'train':76500,'val':4500,'test':9000}
    assert audit['no_cross_split_samples'] is True
    if hashes is not None: assert hashes==audit['split_sha256_global']
    hashes=audit['split_sha256_global']
print('[3759 SOURCE + OFFICIAL96 AUDIT PASS]', flush=True)
PY

for city in citya cityb cityc cityd; do
  for spec in "reproduction_v13_4m:$BASE" "official96_temporal_8m:$OFFICIAL"; do
    label="${spec%%:*}"; source="${spec#*:}"
    target="$OUTPUT/$label/$city/prepared"
    if [[ ! -s "$target/experiment.json" ]]; then
      mkdir -p "$target"
      cp -a --reflink=auto "$source/$city/prepared/." "$target/"
    fi
  done
  for frames in 1 2 3; do
    for suite in "$REPRO" "$TRANSFER"; do
      target="$suite/$city/train_frames$frames/checkpoints"
      if [[ ! -s "$target/visual_retrieval_A_only.pt" ]]; then
        mkdir -p "$target"
        cp -a --reflink=auto "$BASE/$city/train_frames$frames/checkpoints/." "$target/"
      fi
    done
  done
  for suite in "$REPRO" "$TRANSFER"; do
    target="$suite/$city/train_frames3/kalman_calibration.json"
    if [[ ! -s "$target" ]]; then
      cp -a "$BASE/$city/train_frames3/kalman_calibration.json" "$target"
    fi
  done
done

python3 v39_otherdata/audit_3759_transfer_overlap.py --source "$BASE" --official "$OFFICIAL" --output "$TRANSFER/source_train_overlap_audit.json"

if [[ "${GT_3759_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[3759 PREFLIGHT PASS] $OUTPUT"
  exit 0
fi

# First establish that the copied checkpoint reproduces the published v13
# result on its original 4 m prepared routes. Never overwrite the source run.
if [[ ! -s "$REPRO/paper_ablation_results.json" ]]; then
  ABCD_SUITE_ROOT="$REPRO" BEARING_DATASET_ROOT="$DATASET" \
    bash v39_otherdata/run_bearing_abcd_full_table_v7.sh \
    2>&1 | tee "$OUTPUT/logs/reproduction.log"
fi
python3 v39_otherdata/check_3759_reproduction.py --suite-root "$REPRO"


# The learned task heads remain unchanged. Rebuild only the city satellite
# gallery because the Official96 train episode has a different XY origin.
export BEARING_FRAME_ISOLATED_CACHE=1
pids=(); labels=()
for city in citya cityb cityc cityd; do
  for frames in 3 2 1; do
    marker="$TRANSFER/$city/train_frames$frames/official96_gallery_bound.done"
    [[ -s "$marker" ]] && continue
    gpu=0
    case "$frames" in 2) gpu=5;; 1) gpu=6;; esac
    (
      python3 -u v39_otherdata/bearing_iclr_ablation.py rebind \
        --suite-root "$TRANSFER" --dataset-root "$DATASET" --city "$city" \
        --train-frames "$frames" --gpu "$gpu" --seed 2033
      printf '%s\n' "source=$BASE/$city/train_frames$frames/checkpoints" > "$marker"
    ) > "$OUTPUT/logs/rebind_${city}_frames${frames}_gpu${gpu}.log" 2>&1 &
    pids+=("$!"); labels+=("$city/$frames/gpu$gpu")
    if (( ${#pids[@]} == 3 )); then
      for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "Rebind failed: ${labels[$i]}" >&2; exit 4; }
      done
      pids=(); labels=()
    fi
  done
done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "Rebind failed: ${labels[$i]}" >&2; exit 4; }
done

ABCD_SUITE_ROOT="$TRANSFER" BEARING_DATASET_ROOT="$DATASET" \
  ABCD_SKIP_ALIGNMENT_PATCH=1 ABCD_SKIP_PREPARED_AUDIT=1 \
  ABCD_EVAL_ONLY=1 ABCD_EVAL_GPUS='0 5 6' \
  bash v39_otherdata/run_bearing_abcd_shared_ablation.sh \
  2>&1 | tee "$OUTPUT/logs/official96_evaluation.log"
python3 v39_otherdata/build_official96_gt_all_metrics.py \
  --suite-root "$TRANSFER" --cities citya cityb cityc cityd
echo "[DONE] Original result: $REPRO/paper_ablation_tables.md"
echo "[DONE] Official96 transfer: $TRANSFER/official96_gt_all_metrics.md"
