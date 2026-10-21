#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE="${GT_SOURCE_SUITE:-$ROOT/v39_otherdata/output/abcd_table6_official96_temporal_20260923}"
SUITE="${GT_ABLATION_SUITE:-$ROOT/v39_otherdata/output/official96_gt_ablation_3gpu_20260925}"
DATASET="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
mkdir -p "$SUITE/logs"

for city in citya cityb cityc cityd; do
  src="$SOURCE/$city/prepared"
  dst="$SUITE/$city/prepared"
  [[ -s "$src/official96_split_audit.json" ]] || { echo "Missing prepared $src" >&2; exit 2; }
  if [[ ! -e "$dst/experiment.json" ]]; then
    mkdir -p "$dst"
    cp -a "$src/." "$dst/"
  fi
done

python3 - "$SUITE" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = {'train': 76500, 'val': 4500, 'test': 9000}
hashes = None
for city in ('citya', 'cityb', 'cityc', 'cityd'):
    data = json.loads((root / city / 'prepared' / 'official96_split_audit.json').read_text())
    assert data['official_dataset_id'] == 96 and data['split_seed'] == 42
    assert data['split_counts_global'] == expected and data['no_cross_split_samples']
    if hashes is not None and data['split_sha256_global'] != hashes:
        raise RuntimeError('Official96 split hashes differ across cities')
    hashes = data['split_sha256_global']
print('[GT OFFICIAL96 AUDIT PASS]', expected, flush=True)
PY

python3 v39_otherdata/patch_bearing_iclr_main_alignment.py v39_otherdata/bearing_iclr_ablation.py
for city in citya cityb cityc cityd; do
  python3 v39_otherdata/audit_bearing_training_contract.py --dataset-root "$DATASET" --prepared-root "$SUITE/$city/prepared" --city "$city"
  python3 v39_otherdata/bearing_iclr_ablation.py check --suite-root "$SUITE" --dataset-root "$DATASET" --city "$city"
done

export BEARING_FRAME_ISOLATED_CACHE=1
export ABCD_SKIP_ALIGNMENT_PATCH=1
export ABCD_SKIP_PREPARED_AUDIT=1
if [[ "${GT_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[GT PREFLIGHT PASS] $SUITE"
  exit 0
fi
export OMP_NUM_THREADS="${CPU_THREADS_PER_JOB:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

# Frame groups own distinct checkpoints and caches; all three train at once.
pids=()
for spec in '3 0' '2 5' '1 6'; do
  read -r frames gpu <<< "$spec"
  (
    ABCD_SUITE_ROOT="$SUITE" BEARING_DATASET_ROOT="$DATASET" \
    ABCD_FRAMES="$frames" ABCD_TRAIN_ONLY=1 TRAIN_GPU="$gpu" \
    bash v39_otherdata/run_bearing_abcd_shared_ablation.sh
  ) >"$SUITE/logs/train_frames${frames}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "[TRAIN START] frames=$frames gpu=$gpu pid=${pids[-1]}"
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "Training failed; inspect $SUITE/logs/train_frames*.log" >&2; exit 3; }

ABCD_SUITE_ROOT="$SUITE" BEARING_DATASET_ROOT="$DATASET" \
ABCD_EVAL_ONLY=1 ABCD_EVAL_GPUS='0 5 6' \
bash v39_otherdata/run_bearing_abcd_shared_ablation.sh \
  2>&1 | tee "$SUITE/logs/evaluation_and_tables.log"

python3 v39_otherdata/build_official96_gt_all_metrics.py --suite-root "$SUITE" \
  --cities citya cityb cityc cityd
echo "[GT ABLATION DONE] $SUITE/official96_gt_all_metrics.md"
