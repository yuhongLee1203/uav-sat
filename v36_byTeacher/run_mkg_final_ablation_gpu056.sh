#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS="${CPU_THREADS:-2}"
EPOCHS="${EPOCHS:-60}"
SUITE_TAG="${SUITE_TAG:-mkg_final_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="logs/mkg_final_ablation_suite/${SUITE_TAG}"

if [[ -e "$LOG_ROOT" ]]; then
  echo "Refusing to overwrite existing suite log directory: $LOG_ROOT" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT"
printf '%s\n' "$SUITE_TAG" > logs/mkg_final_ablation_suite/LATEST_SUITE_TAG.txt

"$PYTHON_BIN" -m py_compile \
  mkg_final_ablation_experiment.py \
  collect_mkg_final_ablation.py \
  ms_vs_weighted_aggregation_timing.py \
  mkg_ablation_gpu_runner.py \
  six_architecture_model.py \
  six_architecture_autoref_experiment.py

echo "[MKG suite] suite_tag=${SUITE_TAG}"
echo "[MKG suite] GPU0/GPU5/GPU6 run independent jobs in parallel"
echo "[MKG suite] Existing checkpoints/results are NOT removed or overwritten"
echo "[MKG suite] baseline = MKG, 6x6, SoftMS, bandwidth=8m, tau=0.30"
echo "[MKG suite] Decoder accuracy compares Weighted Centroid vs SoftMS only"
echo "[MKG suite] Decoder timing later measures ONLY final XY aggregation"

cat > "${LOG_ROOT}/manifest.txt" <<EOF
MKG final-method thesis ablation suite
suite_tag=${SUITE_TAG}
epochs=${EPOCHS}

01 Window-size sensitivity:
  4x4, 5x5, 6x6(baseline), 7x7, 8x8

02 Component ablation:
  M, M+K, M+G, M+K+G

03 GRU-input ablation:
  full, no_xy, no_variance, no_temporal_mean, no_first_difference

04 MeanShift bandwidth:
  4m, 8m(baseline), 12m

05 Score temperature:
  tau 0.20, 0.30(baseline), 0.50

06 Decoder accuracy:
  Weighted Centroid vs SoftMS; Top-1 excluded

07 Reference-center robustness:
  jitter 0,5,10,15,20m

08 Seed stability:
  2026, 123, 456

09 Main six-order architecture table:
  generated last from existing controlled six-architecture results

Decoder aggregation timing (separate from accuracy):
  Weighted = all N patch centers/weights -> XY
  SoftMS = already-converged K mode centers/weights -> XY
  Excludes feature extraction, scoring, MeanShift convergence, K/G, full FPS.
EOF

run_job () {
  local gpu="$1"; shift
  local run_id="$1"; shift
  local pipeline="$1"; shift
  local grid="$1"; shift
  local decoder="$1"; shift
  local gru="$1"; shift
  local bandwidth="$1"; shift
  local tau="$1"; shift
  local seed="$1"; shift
  local jitters="$1"; shift

  echo "[GPU ${gpu}] START ${run_id}: pipeline=${pipeline} grid=${grid} decoder=${decoder} gru=${gru} bw=${bandwidth} tau=${tau} seed=${seed}"
  OMP_NUM_THREADS="$CPU_THREADS" \
  MKL_NUM_THREADS="$CPU_THREADS" \
  OPENBLAS_NUM_THREADS="$CPU_THREADS" \
  NUMEXPR_NUM_THREADS="$CPU_THREADS" \
  UAVSAT_CPU_THREADS="$CPU_THREADS" \
  CUDA_VISIBLE_DEVICES="$gpu" \
  "$PYTHON_BIN" mkg_ablation_gpu_runner.py mkg_final_ablation_experiment.py \
    --run-id "$run_id" --suite-tag "$SUITE_TAG" \
    --pipeline "$pipeline" --grid-size "$grid" --decoder "$decoder" \
    --gru-ablation "$gru" --bandwidth "$bandwidth" --tau "$tau" \
    --seed "$seed" --eval-jitters "$jitters" --epochs "$EPOCHS" --device cuda:0 \
    2>&1 | tee "${LOG_ROOT}/${run_id}_gpu${gpu}.log"
}

worker_gpu0 () {
  run_job 0 baseline_mkg_6x6  MKG 6 softms full 8 0.30 2026 "0,5,10,15,20"
  run_job 0 component_m       M   6 softms full 8 0.30 2026 "0"
  run_job 0 window_4x4        MKG 4 softms full 8 0.30 2026 "0"
  run_job 0 window_7x7        MKG 7 softms full 8 0.30 2026 "0"
  run_job 0 gru_no_xy         MKG 6 softms no_xy 8 0.30 2026 "0"
  run_job 0 bandwidth_4m      MKG 6 softms full 4 0.30 2026 "0"
}

worker_gpu5 () {
  run_job 5 component_mk      MK  6 softms full 8 0.30 2026 "0"
  run_job 5 component_mg      MG  6 softms full 8 0.30 2026 "0"
  run_job 5 window_5x5        MKG 5 softms full 8 0.30 2026 "0"
  run_job 5 window_8x8        MKG 8 softms full 8 0.30 2026 "0"
  run_job 5 gru_no_variance   MKG 6 softms no_variance 8 0.30 2026 "0"
  run_job 5 tau_0p20          MKG 6 softms full 8 0.20 2026 "0"
  run_job 5 seed_123          MKG 6 softms full 8 0.30 123 "0"
}

worker_gpu6 () {
  run_job 6 gru_no_temporal_mean     MKG 6 softms full 8 0.30 2026 "0"
  run_job 6 gru_no_first_difference MKG 6 softms no_first_difference 8 0.30 2026 "0"
  run_job 6 bandwidth_12m            MKG 6 softms full 12 0.30 2026 "0"
  run_job 6 tau_0p50                 MKG 6 softms full 8 0.50 2026 "0"
  run_job 6 decoder_weighted         MKG 6 weighted full 8 0.30 2026 "0"
  run_job 6 seed_456                 MKG 6 softms full 8 0.30 456 "0"
}

worker_gpu0 & p0=$!
worker_gpu5 & p5=$!
worker_gpu6 & p6=$!
status=0
wait "$p0" || status=1
wait "$p5" || status=1
wait "$p6" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "At least one GPU worker failed; completed outputs are preserved." >&2
  exit 1
fi

# Correct decoder TIME comparison: only final coordinate aggregation.
OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" \
OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS" \
UAVSAT_CPU_THREADS="$CPU_THREADS" CUDA_VISIBLE_DEVICES=0 \
"$PYTHON_BIN" mkg_ablation_gpu_runner.py ms_vs_weighted_aggregation_timing.py \
  --suite-tag "$SUITE_TAG" --device cuda:0 --grid-size 6 \
  --frames-per-route 256 --repeats 200 --warmup 20 \
  2>&1 | tee "${LOG_ROOT}/decoder_aggregation_timing_gpu0.log"

"$PYTHON_BIN" collect_mkg_final_ablation.py --suite-tag "$SUITE_TAG" \
  2>&1 | tee "${LOG_ROOT}/collect.log"

echo "MKG final ablation suite finished: ${SUITE_TAG}"
echo "Purpose-specific tables are table_01 ... table_09; architecture table is last."
