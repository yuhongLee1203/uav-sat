#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

bash "$ROOT/prepare_fornx_gvsk.sh"

SRC="$ROOT/v36_GvsK/original_forNX"
OUT="$ROOT/v36_GvsK/output/original_full"
SHARED_CACHE="$ROOT/v36_GvsK/shared_feature_cache"
GPU="${GPU:-0}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"

for f in config.py robust_tracker.py visual_model.py visual_localizer.py data.py; do
  [[ -f "$SRC/$f" ]] || { echo "ERROR: missing $SRC/$f" >&2; exit 20; }
done
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found" >&2; exit 21; }

# One persistent cache for BOTH original-full and G-only runs.
# If an older cache already exists, migrate it once instead of recomputing it.
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
unset UAVSAT_EXPERIMENT_KALMAN || true

# Old forNX scripts may call `python`; force nested calls to Python 3.
PY3="$(command -v python3)"
PYSHIM="$OUT/python3_shim"
mkdir -p "$PYSHIM"
ln -sf "$PY3" "$PYSHIM/python"
ln -sf "$PY3" "$PYSHIM/python3"
export PATH="$PYSHIM:$PATH"
export PYTHON="$PY3"

ARGS=(--mode train_eval --gpu 0 --visual-epochs "$VISUAL_EPOCHS" --epochs "$TEMPORAL_EPOCHS" --patience "$PATIENCE" --no-render)

echo "=== ORIGINAL forNX V36 ==="
echo "source : $SRC"
echo "output : $OUT"
echo "cache  : $SHARED_CACHE"
echo "GPU    : physical $GPU -> cuda:0"
echo "Python : $PY3"
echo "Kalman : original default (learned)"

if [[ -f ./run_robust_tracker.sh ]]; then
  chmod +x ./run_robust_tracker.sh
  bash ./run_robust_tracker.sh "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
else
  "$PY3" -u ./robust_tracker.py "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
fi

echo "[DONE] original result: $OUT"
echo "[CACHE] persistent shared cache: $SHARED_CACHE"
