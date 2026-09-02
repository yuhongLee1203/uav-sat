#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

bash "$ROOT/prepare_fornx_gvsk.sh"

SRC="$ROOT/v36_GvsK/original_forNX"
OUT="$ROOT/v36_GvsK/output/original_full"
GPU="${GPU:-0}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"

rm -rf "$OUT"
mkdir -p "$OUT"
cp "$ROOT/v36_GvsK/output/source_audit.txt" "$OUT/source_audit.txt" 2>/dev/null || true

cd "$SRC"
export CUDA_VISIBLE_DEVICES="$GPU"
export UAVSAT_OUTPUT_DIR="$OUT"
unset UAVSAT_EXPERIMENT_KALMAN || true

ARGS=(--mode train_eval --gpu 0 --visual-epochs "$VISUAL_EPOCHS" --epochs "$TEMPORAL_EPOCHS" --patience "$PATIENCE" --no-render)

echo "=== ORIGINAL forNX V36 ==="
echo "source : $SRC"
echo "output : $OUT"
echo "GPU    : physical $GPU -> cuda:0"
echo "Kalman : original default (learned)"

if [[ -f ./run_robust_tracker.sh ]]; then
  chmod +x ./run_robust_tracker.sh
  bash ./run_robust_tracker.sh "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
else
  python -u ./robust_tracker.py "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
fi

echo "[DONE] original result: $OUT"
