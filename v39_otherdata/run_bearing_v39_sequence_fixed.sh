#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CITY="${CITY:-cityb}"
GPU="${GPU:-0}"
PREPARED_ROOT="${REPO_ROOT}/v39_otherdata/generated/${CITY}"
OUTPUT_DIR="${PREPARED_ROOT}/v39_output_exact"

cd "${REPO_ROOT}"

prepare_sequence() {
  local safety_cap="$1"
  rm -rf "${PREPARED_ROOT}"
  echo "[PREP] Bearing data adapter only; trying sequence safety cap ${safety_cap} m"
  python3 v39_otherdata/bearing_prepare_sequence_v3.py \
    --dataset-root "${DATASET_ROOT}" \
    --city "${CITY}" \
    --step-m 8 \
    --max-sample-distance-m 15 \
    --preferred-step-m 8 \
    --safety-max-step-m "${safety_cap}" \
    --candidate-limit 64 \
    --beam-width 128 \
    --skip-penalty 30 \
    --continuity-weight 1.5 \
    --large-step-weight 0.35 \
    --cross-weight 1.0 \
    --backward-weight 8.0 \
    --min-selected-ratio 0.70
}

# This modifies only the Bearing pseudo-flight data adapter. It never changes
# GRU/Kalman/MeanShift parameters. Start with the successful 22 m absolute data
# safety cap and relax only if the independent Bearing observations are too sparse.
if ! prepare_sequence 22; then
  echo "[PREP] 22 m data safety cap too sparse; retrying 26 m"
  if ! prepare_sequence 26; then
    echo "[PREP] 26 m data safety cap too sparse; final retry 30 m"
    prepare_sequence 30
  fi
fi

# IMPORTANT: exact canonical-v39 estimator. No Bearing cadence adaptation.
# Model/inference settings are audited against the saved v39 weighted-centroid
# main experiment before training starts.
python3 v39_otherdata/bearing_runner_exact_v39.py \
  --dataset-root "${DATASET_ROOT}" \
  --city "${CITY}" \
  --gpu "${GPU}" \
  --backbone mobilenet_v3_small \
  --visual-epochs 30 \
  --epochs-per-route 20 \
  --patience 10 \
  --jitter-m 8 \
  --step-m 8 \
  --max-sample-distance-m 15 \
  --heading-weight-px-per-deg 0

# Requested visualization: GT/reference + final prediction only.
python3 v39_otherdata/bearing_plot_final_vs_gt.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --routes test_01 test_02

echo ""
echo "================================================================================================="
echo "Bearing exact-v39 experiment finished"
echo "Model: Weighted Centroid -> 3-frame Context-GRU -> velocity -> fixed Kalman -> final 5x5 MS"
echo "Bearing cadence adaptation: DISABLED"
echo "Canonical Kalman final-step cap: 7.0 m"
echo "Summary: ${OUTPUT_DIR}/bearing_v39_summary.json"
echo "Audit  : ${OUTPUT_DIR}/v39_bearing_training_audit.json"
echo "Plot 1 : ${OUTPUT_DIR}/test_01_final_vs_gt_zoom.jpg"
echo "Plot 2 : ${OUTPUT_DIR}/test_02_final_vs_gt_zoom.jpg"
echo "Route diagnostics: ${PREPARED_ROOT}/experiment.json"
echo "================================================================================================="
