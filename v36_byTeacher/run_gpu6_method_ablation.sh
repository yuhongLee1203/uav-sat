#!/usr/bin/env bash
set -euo pipefail

# GPU6 wrapper for the one-variable-at-a-time spatial sensitivity study.
#
# Groups:
#   ms1       : forward 3x6 baseline / 4x6 / 5x6 / 6x6 / 7x6
#   ms2       : centered 5x5 / 6x6 baseline / 7x7
#   meanshift : bandwidth 4 / 8 baseline / 12 / 16 m
#   search    : ms1 + ms2
#   all       : ms1 + ms2 + meanshift
#
# IMPORTANT:
#   This wrapper is STRICTLY EVAL ONLY. Spatial support, MS2 grid size, and
#   MeanShift bandwidth are inference-time method parameters and do NOT require
#   retraining the GRU. The same FULL v8r1 checkpoint is reused for every case.
#
# If the normal FULL checkpoint path is empty, this wrapper attempts to recover
# an architecture-compatible FULL v8r1 checkpoint already present elsewhere
# under v36_byTeacher. It NEVER starts training automatically.

GROUP="${1:-all}"
CPU_THREADS="${UAVSAT_CPU_THREADS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

case "${GROUP}" in
  all|ms1|ms2|meanshift|search) ;;
  *)
    echo "usage: bash run_gpu6_method_ablation.sh {all|ms1|ms2|meanshift|search}" >&2
    exit 2
    ;;
esac

export UAVSAT_GRU_ABLATION=full
export UAVSAT_CPU_THREADS="${CPU_THREADS}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"

BACKBONE="${UAVSAT_BACKBONE:-mobilenet_v3_small}"
CHECKPOINT_DIR="${SCRIPT_DIR}/output/${BACKBONE}/checkpoints"
BEST="${CHECKPOINT_DIR}/reference_prior_compact_gru_A_native_v8r1_full_${BACKBONE}.pt"
LATEST="${CHECKPOINT_DIR}/reference_prior_compact_gru_A_native_v8r1_full_${BACKBONE}_latest.pt"

mkdir -p "${CHECKPOINT_DIR}"

if [[ -f "${BEST}" ]]; then
  echo "FULL v8r1 checkpoint: BEST -> ${BEST}"
elif [[ -f "${LATEST}" ]]; then
  echo "FULL v8r1 checkpoint: LATEST -> ${LATEST}"
  cp -f "${LATEST}" "${BEST}"
  echo "Recovered eval checkpoint from normal latest -> ${BEST}"
else
  echo "Normal FULL v8r1 checkpoint path is empty; searching existing v36_byTeacher outputs..."

  RECOVERED="$(${PYTHON_BIN} - <<'PY'
from pathlib import Path
import os
import torch

root = Path('.').resolve()
backbone = os.environ.get('UAVSAT_BACKBONE', 'mobilenet_v3_small').strip().lower()
expected_arch = (
    'V36_byTeacher_ReferencePrior_MS1StrictForwardHalf3x6_KalmanPrevFinal_'
    'CompactGRUChange_MSXY_TemporalMean_FirstDiff_PrevMotion_'
    'MS2KalmanPosterior6x6_v8r1_nativeA_full'
)
patterns = [
    f'reference_prior_compact_gru_A_native_v8r1_full_{backbone}.pt',
    f'reference_prior_compact_gru_A_native_v8r1_full_{backbone}_latest.pt',
]
seen = set()
candidates = []
for pattern in patterns:
    for path in root.rglob(pattern):
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        candidates.append(path)

for path in sorted(candidates, key=lambda p: p.stat().st_mtime_ns, reverse=True):
    try:
        payload = torch.load(path, map_location='cpu')
    except Exception:
        continue
    if payload.get('architecture') != expected_arch:
        continue
    state = payload.get('model')
    if not isinstance(state, dict):
        continue
    weight = state.get('gru.weight_ih')
    if weight is None:
        matches = [v for k, v in state.items() if k.endswith('gru.weight_ih')]
        weight = matches[0] if len(matches) == 1 else None
    if weight is not None and tuple(weight.shape) != (768, 512):
        continue
    print(path)
    break
PY
)"

  if [[ -n "${RECOVERED}" && -f "${RECOVERED}" ]]; then
    cp -f "${RECOVERED}" "${BEST}"
    echo "Recovered compatible FULL v8r1 checkpoint:"
    echo "  source: ${RECOVERED}"
    echo "  target: ${BEST}"
  else
    echo "ERROR: no compatible FULL v8r1 checkpoint exists." >&2
    echo "GPU6 sensitivity is eval-only and will not train automatically." >&2
    echo "Provide/train one FULL v8r1 checkpoint first, then rerun GPU6." >&2
    exit 3
  fi
fi

echo
echo "FULL checkpoint ready. Starting STRICTLY EVAL-ONLY sensitivity sweep..."
echo "  MS1       : 3x6 baseline, 4x6, 5x6, 6x6, 7x6"
echo "  MS2       : 5x5, 6x6 baseline, 7x7"
echo "  MeanShift : 4m, 8m baseline, 12m, 16m"
echo "  fixed     : same FULL weights, same KF, merge radius=2m, iterations=3"
echo

exec bash run_parameter_ablation.sh "${GROUP}"
