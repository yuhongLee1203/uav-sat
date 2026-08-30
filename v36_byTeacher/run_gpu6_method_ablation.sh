#!/usr/bin/env bash
set -euo pipefail

# GPU6 wrapper for the method-level spatial / MeanShift ablation study.
#
# Why this wrapper exists:
#   run_parameter_ablation.sh is intentionally EVAL ONLY and therefore needs
#   one trained FULL v8r1 temporal checkpoint (512-d GRU input). GPU5's four
#   branch-removal models are 384-d and are not compatible substitutes.
#
# Behaviour:
#   1) use the normal FULL v8r1 best checkpoint when it exists;
#   2) otherwise use the normal FULL v8r1 latest checkpoint;
#   3) otherwise search the whole v36_byTeacher tree for a checkpoint whose
#      saved architecture exactly matches FULL v8r1 and recover it;
#   4) only if no compatible checkpoint exists anywhere, train ONE FULL v8r1
#      baseline once; then all method-ablation cases are evaluation-only.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=6 UAVSAT_CPU_THREADS=1 bash run_gpu6_method_ablation.sh all
#   CUDA_VISIBLE_DEVICES=6 UAVSAT_CPU_THREADS=1 bash run_gpu6_method_ablation.sh search
#   CUDA_VISIBLE_DEVICES=6 UAVSAT_CPU_THREADS=1 bash run_gpu6_method_ablation.sh meanshift

GROUP="${1:-all}"
CPU_THREADS="${UAVSAT_CPU_THREADS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

case "${GROUP}" in
  all|search|meanshift|iterations) ;;
  *)
    echo "usage: bash run_gpu6_method_ablation.sh {all|search|meanshift|iterations}" >&2
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
  # The eval loader reads the best-name path. A latest checkpoint has a normal
  # 'model' field too, so it is usable for preliminary sensitivity evaluation.
  cp -f "${LATEST}" "${BEST}"
  echo "Recovered eval checkpoint from latest -> ${BEST}"
else
  echo "Normal FULL v8r1 checkpoint path is empty; searching v36_byTeacher..."

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

# Prefer newer files, but accept only an exact architecture match and a state
# dict that can be loaded by the current FULL model. This prevents accidentally
# using a hidden-size experiment whose filename happens to look similar.
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
    # Full v8r1 GRUCell input is 512 and hidden state is 256. The recurrent
    # input weight therefore has shape [3*256, 512] = [768, 512].
    weight = state.get('gru.weight_ih')
    if weight is None:
        # Some PyTorch module naming may use a nested prefix; locate by suffix.
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
    echo "No compatible FULL v8r1 checkpoint exists anywhere."
    echo "Training ONE FULL v8r1 baseline now; this is the only training GPU6 needs."
    echo "After it finishes, every spatial/MeanShift case below is EVAL ONLY."
    UAVSAT_GRU_ABLATION=full UAVSAT_CPU_THREADS="${CPU_THREADS}" \
      bash run_train_eval.sh train full 2>&1 | tee full_v8r1_baseline_gpu6.log

    if [[ ! -f "${BEST}" && -f "${LATEST}" ]]; then
      cp -f "${LATEST}" "${BEST}"
    fi
    if [[ ! -f "${BEST}" ]]; then
      echo "ERROR: FULL v8r1 training finished without a usable checkpoint: ${BEST}" >&2
      exit 3
    fi
  fi
fi

echo
echo "FULL checkpoint ready. Starting GPU6 method-level evaluation sweep..."
echo "  search geometry : MS1 forward depth/full support + MS2 grid size"
echo "  MeanShift       : bandwidth + mode-merge distance"
echo "  network weights : fixed FULL v8r1"
echo

exec bash run_parameter_ablation.sh "${GROUP}"
