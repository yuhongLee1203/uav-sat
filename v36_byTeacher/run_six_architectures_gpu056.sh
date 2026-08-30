#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p logs/six_architecture_ablation
EPOCHS="${EPOCHS:-60}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# The six-architecture implementation uses Python-3-only syntax/features
# (matrix @ operator, dataclasses, annotations, f-strings). Do not silently
# fall back to a system `python` that may still point to Python 2.x.
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON_BIN' was not found. Set PYTHON_BIN to the Python 3 executable used by this project." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 7):
    raise SystemExit(
        "ERROR: six-architecture experiments require Python >= 3.7; got %s" %
        (sys.version.replace("\n", " "),)
    )

try:
    import torch
except Exception as exc:
    raise SystemExit("ERROR: failed to import PyTorch with this Python: %r" % (exc,))

print("[preflight] Python:", sys.executable, sys.version.replace("\n", " "))
print("[preflight] PyTorch:", torch.__version__)
print("[preflight] CUDA available:", torch.cuda.is_available())
print("[preflight] visible CUDA device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available to the selected Python/PyTorch environment.")
PY

"$PYTHON_BIN" -m py_compile six_architecture_model.py six_architecture_experiment.py

echo "[preflight] six-architecture Python files compile successfully."

run_pair () {
  local gpu="$1"; shift
  for arch in "$@"; do
    echo "[GPU ${gpu}] starting ${arch}"
    CUDA_VISIBLE_DEVICES="${gpu}" "$PYTHON_BIN" six_architecture_experiment.py \
      --mode train-eval --arch "${arch}" --device cuda:0 --epochs "${EPOCHS}" \
      2>&1 | tee "logs/six_architecture_ablation/${arch}_gpu${gpu}.log"
  done
}

run_pair 0 MKG MGK &
P0=$!
run_pair 5 GMK GKM &
P5=$!
run_pair 6 KGM KMG &
P6=$!

wait "$P0" "$P5" "$P6"
echo "All six architectures finished."
