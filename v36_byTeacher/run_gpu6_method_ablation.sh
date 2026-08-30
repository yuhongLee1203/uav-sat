#!/usr/bin/env bash
set -euo pipefail

# GPU6 wrapper for the one-variable-at-a-time spatial sensitivity study.
#
# Groups:
#   ms1       : forward 3x6 baseline / 4x6 / 5x6 / 7x6
#   ms2       : centered 5x5 / 6x6 baseline / 7x7
#   meanshift : bandwidth 4 / 8 baseline / 12 / 16 m
#   search    : ms1 + ms2
#   all       : ms1 + ms2 + meanshift
#
# GPU6 is evaluation-only once one clean FULL v8r1 checkpoint exists.
# It never substitutes a 384-d GPU5 branch-removal checkpoint.
#
# Optional final-paper safeguard:
#   UAVSAT_FORCE_FULL_RETRAIN=1
# removes the normal FULL best/latest checkpoint and trains one fresh FULL
# v8r1 baseline before running the requested sensitivity group.

GROUP="${1:-all}"
CPU_THREADS="${UAVSAT_CPU_THREADS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_FULL_RETRAIN="${UAVSAT_FORCE_FULL_RETRAIN:-0}"
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

if [[ "${FORCE_FULL_RETRAIN}" == "1" ]]; then
  echo "FORCE FULL RETRAIN enabled: removing normal FULL best/latest checkpoints."
  rm -f "${BEST}" "${LATEST}"
fi

if [[ -f "${BEST}" ]]; then
  echo "FULL v8r1 checkpoint: BEST -> ${BEST}"
elif [[ -f "${LATEST}" ]]; then
  echo "FULL v8r1 checkpoint: LATEST -> ${LATEST}"
  echo "WARNING: latest/in-progress checkpoint is being used for sensitivity evaluation."
  cp -f "${LATEST}" "${BEST}"
else
  echo "No normal FULL v8r1 checkpoint exists."
  echo "Training ONE fresh FULL v8r1 baseline; sensitivity cases remain eval-only."
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

echo
echo "FULL checkpoint ready. Starting one-variable-at-a-time GPU6 sweep..."
echo "  MS1       : 3x6 baseline, 4x6, 5x6, 7x6"
echo "  MS2       : 5x5, 6x6 baseline, 7x7"
echo "  MeanShift : 4m, 8m baseline, 12m, 16m"
echo "  fixed     : same FULL weights, same KF, merge radius=2m, iterations=3"
echo

exec bash run_parameter_ablation.sh "${GROUP}"
