#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
bash "$ROOT/prepare_fornx_gvsk.sh"

SRC="$ROOT/v36_GvsK/G_only"
CENTRAL="$ROOT/v36_GvsK/output/G_only"
GPU="${GPU:-5}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"

for f in config.py robust_tracker.py visual_model.py visual_localizer.py data.py run_robust_tracker.sh; do
  [[ -f "$SRC/$f" ]] || { echo "ERROR: missing $SRC/$f" >&2; exit 20; }
done

python3 - "$SRC/config.py" <<'PY'
from pathlib import Path
import re, sys
s = Path(sys.argv[1]).read_text(encoding="utf-8")
m = re.search(r'EXPERIMENT_KALMAN\s*=\s*os\.environ\.get\(\s*["\']UAVSAT_EXPERIMENT_KALMAN["\']\s*,\s*["\']([^"\']+)', s, re.S)
if not m or m.group(1) != "none":
    raise SystemExit("ERROR: G_only config is not pinned to original V36 Kalman mode 'none'")
print("G-only preflight: EXPERIMENT_KALMAN default = none")
PY

# Preserve any pretrained/checkpoint content that already exists in the correct
# forNX copy, while making the original relative `outputs/...` path land inside
# v36_GvsK/output/G_only.
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
cp "$ROOT/v36_GvsK/output/original_vs_Gonly_code_diff.txt" "$CENTRAL/original_vs_Gonly_code_diff.txt" 2>/dev/null || true

cd "$SRC"
chmod +x run_robust_tracker.sh

echo "================================================================================================"
echo "forNX COPY / G-ONLY"
echo "project root : $SRC"
echo "config.py    : $SRC/config.py"
echo "runner       : copied ORIGINAL forNX/run_robust_tracker.sh"
echo "output       : $CENTRAL"
echo "GPU          : $GPU"
echo "ONLY CHANGE  : config.py EXPERIMENT_KALMAN default learned -> none"
echo "================================================================================================"

# No architecture-related environment variable is injected. This executes the
# exact copied forNX runner; the sole architecture difference is the one config
# default changed above.
bash ./run_robust_tracker.sh \
  --mode train_eval \
  --gpu "$GPU" \
  --visual-epochs "$VISUAL_EPOCHS" \
  --epochs "$TEMPORAL_EPOCHS" \
  --patience "$PATIENCE" \
  --reuse-visual 1 \
  --no-render \
  2>&1 | tee "$CENTRAL/wrapper_run.log"

echo "[G-only] all generated outputs are under: $CENTRAL"
