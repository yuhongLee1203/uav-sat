#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
bash "$ROOT/prepare_fornx_gvsk.sh"

SRC="$ROOT/v36_GvsK/original_forNX"
CENTRAL="$ROOT/v36_GvsK/output/original_full"
GPU="${GPU:-0}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"

for f in config.py robust_tracker.py visual_model.py visual_localizer.py data.py run_robust_tracker.sh; do
  [[ -f "$SRC/$f" ]] || { echo "ERROR: missing $SRC/$f" >&2; exit 20; }
done

# Keep the ORIGINAL forNX run_robust_tracker.sh and its relative `outputs/...`
# semantics. We only redirect that directory with a symlink so every generated
# artifact ends under v36_GvsK/output/original_full as requested.
rm -rf "$CENTRAL"
mkdir -p "$CENTRAL"
if [[ -L "$SRC/outputs" ]]; then
  rm "$SRC/outputs"
elif [[ -d "$SRC/outputs" ]]; then
  cp -a "$SRC/outputs/." "$CENTRAL/"
  rm -rf "$SRC/outputs"
fi
ln -s "$CENTRAL" "$SRC/outputs"
cp "$ROOT/v36_GvsK/output/source_audit.txt" "$CENTRAL/source_audit.txt" 2>/dev/null || true
cp "$ROOT/v36_GvsK/output/forNX_copy_diff.txt" "$CENTRAL/forNX_copy_diff.txt" 2>/dev/null || true

cd "$SRC"
chmod +x run_robust_tracker.sh

echo "================================================================================================"
echo "CORRECT forNX / ORIGINAL FULL V36"
echo "project root : $SRC"
echo "config.py    : $SRC/config.py"
echo "runner       : ORIGINAL forNX/run_robust_tracker.sh"
echo "output       : $CENTRAL"
echo "GPU          : $GPU"
echo "IMPORTANT    : no architecture env override is injected"
echo "================================================================================================"

# Critical reproducibility rule: execute the original forNX runner itself.
# Do NOT inject UAVSAT_REFERENCE_PROTOCOL / EXPERIMENT_* / OUTPUT_DIR here.
# The only supplied arguments are ordinary run controls; architecture defaults
# come from the user's correct forNX source exactly as they did originally.
bash ./run_robust_tracker.sh \
  --mode train_eval \
  --gpu "$GPU" \
  --visual-epochs "$VISUAL_EPOCHS" \
  --epochs "$TEMPORAL_EPOCHS" \
  --patience "$PATIENCE" \
  --reuse-visual 1 \
  --no-render \
  2>&1 | tee "$CENTRAL/wrapper_run.log"

echo "[correct forNX] all generated outputs are under: $CENTRAL"
