#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EPOCHS="${EPOCHS:-60}"
CPU_THREADS="${CPU_THREADS:-2}"
BASE_SEED="${BASE_SEED:-42}"
RESET="${RESET:-1}"
ROOT="output/mobilenet_v3_small/mkg_final_thesis_ablation"
LOGROOT="logs/mkg_final_thesis_ablation"

# Only this new ablation suite is reset. Existing six-order, delayed,
# autonomous, route-tube, visual checkpoints and their outputs are untouched.
if [[ "$RESET" == "1" ]]; then
  rm -rf "$ROOT" "$LOGROOT"
  rm -f output/mobilenet_v3_small/checkpoints/mkg_final_ablation_*_mobilenet_v3_small.pt
fi
mkdir -p "$LOGROOT"

"$PYTHON_BIN" -m py_compile \
  mkg_final_thesis_ablation.py \
  six_architecture_autoref_experiment.py \
  six_architecture_gtref_experiment.py \
  six_architecture_model.py \
  visual_localizer.py \
  gpu_grid_runner.py

COMMON=(--device cuda:0 --epochs "$EPOCHS" --seed "$BASE_SEED")

run_task () {
  local gpu="$1"
  shift
  local tag="$1"
  shift
  echo "[GPU ${gpu}] START ${tag}"
  OMP_NUM_THREADS="$CPU_THREADS" \
  MKL_NUM_THREADS="$CPU_THREADS" \
  OPENBLAS_NUM_THREADS="$CPU_THREADS" \
  NUMEXPR_NUM_THREADS="$CPU_THREADS" \
  UAVSAT_CPU_THREADS="$CPU_THREADS" \
  CUDA_VISIBLE_DEVICES="$gpu" \
  "$PYTHON_BIN" gpu_grid_runner.py mkg_final_thesis_ablation.py \
    --tag "$tag" "${COMMON[@]}" "$@" \
    2>&1 | tee "$LOGROOT/${tag}_gpu${gpu}.log"
  echo "[GPU ${gpu}] DONE ${tag}"
}

# ---------------------------------------------------------------------------
# TRAINING PHASE
# 17 independent training jobs are split 6/6/5 over GPU 0/5/6.
# Baseline 6x6 is also the window=6, decoder=SoftMS, bandwidth=8,
# component=MKG, GRU-full and seed=42 reference point, so it is not retrained.
# ---------------------------------------------------------------------------
worker0 () {
  run_task 0 baseline_mkg_g6_softms_bw8 \
    --mode train-eval --group baseline --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 8

  run_task 0 window_g3 \
    --mode train-eval --group window --component MKG \
    --grid-size 3 --decoder softms --ms-bandwidth 8
  run_task 0 window_g5 \
    --mode train-eval --group window --component MKG \
    --grid-size 5 --decoder softms --ms-bandwidth 8

  run_task 0 decoder_top1 \
    --mode train-eval --group decoder --component MKG \
    --grid-size 6 --decoder top1 --ms-bandwidth 8

  run_task 0 gru_no_xy \
    --mode train-eval --group gru --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 8 \
    --gru-ablation no_xy

  run_task 0 seed_123 \
    --mode train-eval --group seed --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 8 --seed 123
}

worker5 () {
  run_task 5 window_g7 \
    --mode train-eval --group window --component MKG \
    --grid-size 7 --decoder softms --ms-bandwidth 8
  run_task 5 window_g9 \
    --mode train-eval --group window --component MKG \
    --grid-size 9 --decoder softms --ms-bandwidth 8

  run_task 5 decoder_weighted \
    --mode train-eval --group decoder --component MKG \
    --grid-size 6 --decoder weighted --ms-bandwidth 8

  run_task 5 gru_no_variance \
    --mode train-eval --group gru --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 8 \
    --gru-ablation no_variance
  run_task 5 gru_no_hidden \
    --mode train-eval --group gru --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 8 \
    --gru-ablation no_hidden

  run_task 5 seed_2026 \
    --mode train-eval --group seed --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 8 --seed 2026
}

worker6 () {
  # M and MK need no learned G and are evaluated later. Only MG needs training.
  run_task 6 component_mg \
    --mode train-eval --group component --component MG \
    --grid-size 6 --decoder softms --ms-bandwidth 8

  run_task 6 gru_no_temporal_mean \
    --mode train-eval --group gru --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 8 \
    --gru-ablation no_temporal_mean
  run_task 6 gru_no_first_difference \
    --mode train-eval --group gru --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 8 \
    --gru-ablation no_first_difference

  run_task 6 bandwidth_4m \
    --mode train-eval --group bandwidth --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 4
  run_task 6 bandwidth_12m \
    --mode train-eval --group bandwidth --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 12
}

worker0 & p0=$!
worker5 & p5=$!
worker6 & p6=$!
wait "$p0"
wait "$p5"
wait "$p6"

# ---------------------------------------------------------------------------
# CHEAP EVALUATION PHASE
# M/MK require no temporal training. Robustness reuses the one baseline
# checkpoint so the only changed factor is the deterministic center error.
# ---------------------------------------------------------------------------
eval0 () {
  run_task 0 component_m \
    --mode eval --group component --component M \
    --grid-size 6 --decoder softms --ms-bandwidth 8
  run_task 0 robustness_prior_5m \
    --mode eval --group robustness --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 8 \
    --prior-error-m 5 --load-tag baseline_mkg_g6_softms_bw8
  run_task 0 robustness_prior_20m \
    --mode eval --group robustness --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 8 \
    --prior-error-m 20 --load-tag baseline_mkg_g6_softms_bw8
}

eval5 () {
  run_task 5 component_mk \
    --mode eval --group component --component MK \
    --grid-size 6 --decoder softms --ms-bandwidth 8
  run_task 5 robustness_prior_10m \
    --mode eval --group robustness --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 8 \
    --prior-error-m 10 --load-tag baseline_mkg_g6_softms_bw8
  run_task 5 robustness_prior_25m \
    --mode eval --group robustness --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 8 \
    --prior-error-m 25 --load-tag baseline_mkg_g6_softms_bw8
}

eval6 () {
  run_task 6 robustness_prior_15m \
    --mode eval --group robustness --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 8 \
    --prior-error-m 15 --load-tag baseline_mkg_g6_softms_bw8
  run_task 6 robustness_prior_30m \
    --mode eval --group robustness --component MKG \
    --grid-size 6 --decoder softms --ms-bandwidth 8 \
    --prior-error-m 30 --load-tag baseline_mkg_g6_softms_bw8
}

eval0 & e0=$!
eval5 & e5=$!
eval6 & e6=$!
wait "$e0"
wait "$e5"
wait "$e6"

# Baseline itself is the 0 m robustness point. Collect every result and also
# ingest the already-completed six-order controlled summaries without rerunning.
OMP_NUM_THREADS="$CPU_THREADS" \
MKL_NUM_THREADS="$CPU_THREADS" \
OPENBLAS_NUM_THREADS="$CPU_THREADS" \
NUMEXPR_NUM_THREADS="$CPU_THREADS" \
UAVSAT_CPU_THREADS="$CPU_THREADS" \
"$PYTHON_BIN" mkg_final_thesis_ablation.py --mode collect

echo "MKG thesis ablation suite finished."
echo "Tables: $ROOT/tables/"
echo "Manifest: $ROOT/manifest.json"
