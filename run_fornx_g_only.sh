#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if [[ ! -d "$ROOT/v36_GvsK/G_only" ]]; then
  bash "$ROOT/prepare_fornx_gvsk.sh"
fi

SRC="$ROOT/v36_GvsK/G_only"
OUT="$ROOT/v36_GvsK/output/G_only"
GPU="${GPU:-5}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"

for f in config.py robust_tracker.py visual_model.py visual_localizer.py data.py; do
  [[ -f "$SRC/$f" ]] || { echo "ERROR: missing $SRC/$f" >&2; exit 20; }
done
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found" >&2; exit 21; }

rm -rf "$OUT"
mkdir -p "$OUT"
cp "$ROOT/v36_GvsK/output/source_audit.txt" "$OUT/source_audit.txt" 2>/dev/null || true

cd "$SRC"
export CUDA_VISIBLE_DEVICES="$GPU"
export UAVSAT_OUTPUT_DIR="$OUT"
export UAVSAT_EXPERIMENT_KALMAN="none"

# Some old forNX shell scripts call `python`, while this container maps that to
# Python 2. Force every nested `python` invocation to the same Python 3 runtime.
PY3="$(command -v python3)"
PYSHIM="$OUT/python3_shim"
mkdir -p "$PYSHIM"
ln -sf "$PY3" "$PYSHIM/python"
ln -sf "$PY3" "$PYSHIM/python3"
export PATH="$PYSHIM:$PATH"
export PYTHON="$PY3"

ARGS=(--mode train_eval --gpu 0 --visual-epochs "$VISUAL_EPOCHS" --epochs "$TEMPORAL_EPOCHS" --patience "$PATIENCE" --no-render)

echo "=== forNX V36 / G ONLY ==="
echo "source : $SRC"
echo "output : $OUT"
echo "GPU    : physical $GPU -> cuda:0"
echo "Python : $PY3"
echo "ONLY architecture change: UAVSAT_EXPERIMENT_KALMAN=none"

echo "Preflight Kalman mode:"
"$PY3" - <<'PY'
import config
print("EXPERIMENT_KALMAN =", getattr(config, "EXPERIMENT_KALMAN", "<missing>"))
if getattr(config, "EXPERIMENT_KALMAN", None) != "none":
    raise SystemExit("ERROR: config did not resolve EXPERIMENT_KALMAN=none")
PY

if [[ -f ./run_robust_tracker.sh ]]; then
  chmod +x ./run_robust_tracker.sh
  bash ./run_robust_tracker.sh "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
else
  "$PY3" -u ./robust_tracker.py "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
fi

echo "[DONE] G-only result: $OUT"
