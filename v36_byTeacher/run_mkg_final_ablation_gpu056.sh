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
  mkg_ablation_gpu_runner.py \
  six_architecture_model.py \
  six_architecture_autoref_experiment.py

echo "[MKG suite] suite_tag=${SUITE_TAG}"
echo "[MKG suite] GPU0/GPU5/GPU6 run independent jobs in parallel"
echo "[MKG suite] Route A train; Route B/C eval; frame-aligned reference center"
echo "[MKG suite] Existing checkpoints/results are NOT removed or overwritten"
echo "[MKG suite] baseline = MKG, 6x6, SoftMS, bandwidth=8m, tau=0.30"
echo "[MKG suite] baseline also evaluates reference-center jitter 0/5/10/15/20m"
echo "[MKG suite] Top-1/Weighted latency excludes unused SoftMS computation"

cat > "${LOG_ROOT}/manifest.txt" <<EOF
MKG final-method thesis ablation suite
suite_tag=${SUITE_TAG}
epochs=${EPOCHS}

Component ablation:
  component_m        : M
  component_mk       : M -> K
  component_mg       : M -> G
  baseline_mkg_6x6   : M -> K -> G

Window-size sensitivity:
  4x4, 5x5, 6x6(baseline), 7x7, 8x8

GRU-input ablation:
  full(baseline), no_xy, no_variance, no_temporal_mean, no_first_difference

MeanShift sensitivity:
  bandwidth 4m, 8m(baseline), 12m
  tau 0.20, 0.30(baseline), 0.50

Decoder ablation:
  top1, weighted centroid, SoftMS(baseline)

Seed stability:
  seed 2026(baseline), 123, 456

Robustness:
  baseline evaluation at deterministic center jitter 0,5,10,15,20m

Efficiency automatically reported for every run:
  latency ms/frame, FPS, peak CUDA allocated MB, candidate count
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

  echo "[GPU ${gpu}] START ${run_id}: pipeline=${pipeline} grid=${grid} decoder=${decoder} gru=${gru} bw=${bandwidth} tau=${tau} seed=${seed} jitters=${jitters}"
  OMP_NUM_THREADS="$CPU_THREADS" \
  MKL_NUM_THREADS="$CPU_THREADS" \
  OPENBLAS_NUM_THREADS="$CPU_THREADS" \
  NUMEXPR_NUM_THREADS="$CPU_THREADS" \
  UAVSAT_CPU_THREADS="$CPU_THREADS" \
  CUDA_VISIBLE_DEVICES="$gpu" \
  "$PYTHON_BIN" mkg_ablation_gpu_runner.py mkg_final_ablation_experiment.py \
    --run-id "$run_id" \
    --suite-tag "$SUITE_TAG" \
    --pipeline "$pipeline" \
    --grid-size "$grid" \
    --decoder "$decoder" \
    --gru-ablation "$gru" \
    --bandwidth "$bandwidth" \
    --tau "$tau" \
    --seed "$seed" \
    --eval-jitters "$jitters" \
    --epochs "$EPOCHS" \
    --device cuda:0 \
    2>&1 | tee "${LOG_ROOT}/${run_id}_gpu${gpu}.log"
  echo "[GPU ${gpu}] DONE ${run_id}"
}

worker_gpu0 () {
  # Baseline is reused conceptually by grid=6, full-GRU, bw=8, tau=.30, SoftMS.
  # It also performs the only multi-jitter robustness evaluation.
  run_job 0 baseline_mkg_6x6  MKG 6 softms   full               8 0.30 2026 "0,5,10,15,20"
  run_job 0 component_m       M   6 softms   full               8 0.30 2026 "0"
  run_job 0 window_4x4       MKG 4 softms   full               8 0.30 2026 "0"
  run_job 0 window_7x7       MKG 7 softms   full               8 0.30 2026 "0"
  run_job 0 gru_no_xy        MKG 6 softms   no_xy              8 0.30 2026 "0"
  run_job 0 bandwidth_4m     MKG 6 softms   full               4 0.30 2026 "0"
  run_job 0 decoder_top1     MKG 6 top1     full               8 0.30 2026 "0"
}

worker_gpu5 () {
  run_job 5 component_mk      MK  6 softms   full               8 0.30 2026 "0"
  run_job 5 component_mg      MG  6 softms   full               8 0.30 2026 "0"
  run_job 5 window_5x5       MKG 5 softms   full               8 0.30 2026 "0"
  run_job 5 window_8x8       MKG 8 softms   full               8 0.30 2026 "0"
  run_job 5 gru_no_variance  MKG 6 softms   no_variance        8 0.30 2026 "0"
  run_job 5 tau_0p20         MKG 6 softms   full               8 0.20 2026 "0"
  run_job 5 seed_123         MKG 6 softms   full               8 0.30 123  "0"
}

worker_gpu6 () {
  run_job 6 gru_no_temporal_mean     MKG 6 softms   no_temporal_mean     8  0.30 2026 "0"
  run_job 6 gru_no_first_difference MKG 6 softms   no_first_difference 8  0.30 2026 "0"
  run_job 6 bandwidth_12m            MKG 6 softms   full                12 0.30 2026 "0"
  run_job 6 tau_0p50                 MKG 6 softms   full                8  0.50 2026 "0"
  run_job 6 decoder_weighted        MKG 6 weighted full                8  0.30 2026 "0"
  run_job 6 seed_456                MKG 6 softms   full                8  0.30 456  "0"
}

worker_gpu0 & p0=$!
worker_gpu5 & p5=$!
worker_gpu6 & p6=$!

status=0
wait "$p0" || status=1
wait "$p5" || status=1
wait "$p6" || status=1

if [[ "$status" -ne 0 ]]; then
  echo "At least one GPU worker failed. Completed run outputs are preserved in the unique suite directory." >&2
  echo "suite_tag=${SUITE_TAG}" >&2
  exit 1
fi

"$PYTHON_BIN" collect_mkg_final_ablation.py --suite-tag "$SUITE_TAG" \
  2>&1 | tee "${LOG_ROOT}/collect.log"

echo "============================================================"
echo "MKG final ablation suite finished"
echo "suite_tag=${SUITE_TAG}"
echo "logs=${LOG_ROOT}"
echo "results=output/<backbone>/mkg_final_ablation_suite/${SUITE_TAG}/"
echo "key tables: bc_average_results.csv, jitter0_ranking.csv, seed_stability.json"
echo "============================================================"
