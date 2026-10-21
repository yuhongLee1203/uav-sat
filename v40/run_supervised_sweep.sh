#!/usr/bin/env bash
set -euo pipefail
cd /yh/study/uav-sat
ROOT=${1:-/yh/study/uav-sat/v40/output/gt_sweep_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$ROOT"
declare -a pids=()
for spec in 'causal:0:0:0:0.00002' 'mixed:5:0.5:0.25:0.00001' 'teacher:6:1:0.5:0.00001'; do
  IFS=: read -r name gpu acq center lr <<< "$spec"
  mkdir -p "$ROOT/$name"
  V40_GPU="$gpu" V40_TRAIN_CITIES="${V40_TRAIN_CITIES:-citya cityb}" \
    V40_EVAL_CITIES="${V40_EVAL_CITIES:-citya cityb}" \
    V40_SUPERVISION=gt_train V40_INIT=shared \
    V40_EPOCHS="${V40_EPOCHS:-2}" V40_MAX_TRAIN_FRAMES="${V40_MAX_TRAIN_FRAMES:-136}" \
    V40_TEACHER_ACQ="$acq" V40_TEACHER_CENTER="$center" V40_LR="$lr" \
    bash v40/run_bearinguav_multicity_pilot.sh "$ROOT/$name" \
    >"$ROOT/$name/console.log" 2>&1 &
  pids+=("$!")
  echo "[SWEEP] $name GPU=$gpu pid=${pids[-1]}"
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" != 0 ]]; then
  echo "One or more trials failed; inspect $ROOT/*/console.log" >&2
  exit 1
fi
python3 v40/summarize_multicity_sweep.py "$ROOT"
echo "[SWEEP DONE] $ROOT/summary.md"
