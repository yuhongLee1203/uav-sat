#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS="${CPU_THREADS:-2}"
GPU="${GPU:-6}"

if [[ -n "${SUITE_TAG:-}" ]]; then
  TAG="$SUITE_TAG"
elif [[ -f logs/mkg_final_ablation_suite/LATEST_SUITE_TAG.txt ]]; then
  TAG="$(tr -d '[:space:]' < logs/mkg_final_ablation_suite/LATEST_SUITE_TAG.txt)"
else
  echo "Cannot determine suite tag. Set SUITE_TAG=..." >&2
  exit 2
fi

"$PYTHON_BIN" -m py_compile \
  ms_vs_weighted_aggregation_timing.py \
  collect_mkg_final_ablation.py \
  gpu_grid_runner.py

echo "[decoder timing] suite_tag=${TAG}"
echo "[decoder timing] GPU=${GPU}"
echo "[decoder timing] measure ONLY final XY aggregation"
echo "[decoder timing] Weighted: all 36 patch points -> XY"
echo "[decoder timing] SoftMS: already-converged K modes -> XY"
echo "[decoder timing] no feature/scoring/MS-convergence/K/G/full-FPS timing"

OMP_NUM_THREADS="$CPU_THREADS" \
MKL_NUM_THREADS="$CPU_THREADS" \
OPENBLAS_NUM_THREADS="$CPU_THREADS" \
NUMEXPR_NUM_THREADS="$CPU_THREADS" \
UAVSAT_CPU_THREADS="$CPU_THREADS" \
CUDA_VISIBLE_DEVICES="$GPU" \
"$PYTHON_BIN" gpu_grid_runner.py ms_vs_weighted_aggregation_timing.py \
  --suite-tag "$TAG" \
  --device cuda:0 \
  --grid-size 6 \
  --frames-per-route 256 \
  --repeats 200 \
  --warmup 20

"$PYTHON_BIN" collect_mkg_final_ablation.py --suite-tag "$TAG"

echo "============================================================"
echo "Corrected tables rebuilt for suite: ${TAG}"
echo "Only decoder aggregation time is valid for Weighted-vs-SoftMS timing."
echo "Main architecture table is table_09_* and intentionally last."
echo "============================================================"
