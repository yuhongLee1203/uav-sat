#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p logs/six_architecture_gt_reference_center6x6
EPOCHS="${EPOCHS:-60}"
REF_SPACING_M="${REF_SPACING_M:-5.0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m py_compile \
  six_architecture_model.py \
  six_architecture_autoref_experiment.py \
  six_architecture_gtref_experiment.py

"$PYTHON_BIN" - <<'PY'
import config
import robust_tracker as rt
print('[preflight] device:', rt.resolve_device('cuda:0'))
print('[preflight] visual checkpoint:', config.VISUAL_CHECKPOINT)
PY

echo "[preflight] GT/reference-assisted + full centered 6x6 MeanShift ready"

run_pair () {
  local gpu="$1"; shift
  for arch in "$@"; do
    echo "[GPU ${gpu}] starting ${arch} GT-reference centered-6x6"
    CUDA_VISIBLE_DEVICES="${gpu}" "$PYTHON_BIN" six_architecture_gtref_experiment.py \
      --mode train-eval \
      --arch "${arch}" \
      --device cuda:0 \
      --epochs "${EPOCHS}" \
      --reference-spacing-m "${REF_SPACING_M}" \
      2>&1 | tee "logs/six_architecture_gt_reference_center6x6/${arch}_gpu${gpu}.log"
  done
}

run_pair 0 MKG MGK &
P0=$!
run_pair 5 GMK GKM &
P5=$!
run_pair 6 KGM KMG &
P6=$!

wait "$P0" "$P5" "$P6"
echo "All six GT-reference centered-6x6 architectures finished."
