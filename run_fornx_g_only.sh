#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if [[ ! -d "$ROOT/v36_GvsK/G_only" ]]; then
  bash "$ROOT/prepare_fornx_gvsk.sh"
fi

SRC="$ROOT/v36_GvsK/G_only"
OUT="$ROOT/v36_GvsK/output/G_only"
SHARED_CACHE="$ROOT/v36_GvsK/shared_feature_cache"
GPU="${GPU:-5}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"

for f in config.py robust_tracker.py visual_model.py visual_localizer.py data.py; do
  [[ -f "$SRC/$f" ]] || { echo "ERROR: missing $SRC/$f" >&2; exit 20; }
done
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found" >&2; exit 21; }

# Use exactly the same persistent feature cache as the original-full run.
# If an older cache already exists elsewhere, migrate it once.
mkdir -p "$SHARED_CACHE"
if ! find "$SHARED_CACHE" -type f -print -quit 2>/dev/null | grep -q .; then
  while IFS= read -r OLD_CACHE; do
    [[ "$OLD_CACHE" == "$SHARED_CACHE" ]] && continue
    if find "$OLD_CACHE" -type f -print -quit 2>/dev/null | grep -q .; then
      echo "[cache] reusing existing cache from: $OLD_CACHE"
      cp -a "$OLD_CACHE/." "$SHARED_CACHE/"
      break
    fi
  done < <(find "$ROOT/v36_GvsK" -type d -name feature_cache 2>/dev/null | sort)
fi

if find "$SHARED_CACHE" -type f -print -quit 2>/dev/null | grep -q .; then
  echo "[cache] shared cache already exists: $SHARED_CACHE"
else
  echo "[cache] no existing cache found; it will be created ONCE at: $SHARED_CACHE"
fi

rm -rf "$OUT"
mkdir -p "$OUT"
cp "$ROOT/v36_GvsK/output/source_audit.txt" "$OUT/source_audit.txt" 2>/dev/null || true

cd "$SRC"
export CUDA_VISIBLE_DEVICES="$GPU"
export UAVSAT_OUTPUT_DIR="$OUT"
export UAVSAT_FEATURE_CACHE_DIR="$SHARED_CACHE"
export UAVSAT_EXPERIMENT_KALMAN="none"

# Some old forNX shell scripts call `python`; force them to Python 3.
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
echo "cache  : $SHARED_CACHE"
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
echo "[CACHE] persistent shared cache: $SHARED_CACHE"
