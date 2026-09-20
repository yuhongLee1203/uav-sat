#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Resource-safe defaults for the formal 4-city run.
# Three GPU jobs still run concurrently, but each Python process is prevented
# from spawning an unbounded number of CPU math threads.
CPU_THREADS_PER_CITY="${CPU_THREADS_PER_CITY:-2}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-128}"
CPU_NICE="${CPU_NICE:-5}"

export OMP_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS_PER_CITY}"
export BLIS_NUM_THREADS="${CPU_THREADS_PER_CITY}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# This only changes cache batching / peak host load. It does not change the
# model architecture, training labels, validation protocol, or held-out metrics.
export UAVSAT_VISUAL_CACHE_BATCH_SIZE="${UAVSAT_VISUAL_CACHE_BATCH_SIZE:-${CACHE_BATCH_SIZE}}"

# Avoid CPU contention from tokenizers/libraries that may create helper pools.
export TOKENIZERS_PARALLELISM=false

printf '%s\n' \
  "================================================================================" \
  "FORMAL V5 RESOURCE-SAFE RUN" \
  "GPU jobs             : 0 / 5 / 6 (unchanged)" \
  "CPU threads per city : ${CPU_THREADS_PER_CITY}" \
  "SAT cache batch      : ${UAVSAT_VISUAL_CACHE_BATCH_SIZE}" \
  "CPU nice             : ${CPU_NICE}" \
  "================================================================================"

# nice lowers CPU scheduling priority so the host stays responsive while GPU
# training remains concurrent. All children inherit the thread caps above.
exec nice -n "${CPU_NICE}" bash v39_otherdata/run_bearing_iclr_ablation_fixed.sh
