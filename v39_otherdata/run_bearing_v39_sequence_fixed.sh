#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CITY="${CITY:-cityb}"
GPU="${GPU:-0}"
PREPARED_ROOT="${REPO_ROOT}/v39_otherdata/generated/${CITY}"

cd "${REPO_ROOT}"

prepare_sequence() {
  local safety_cap="$1"
  rm -rf "${PREPARED_ROOT}"
  echo "[PREP] trying soft temporal sequence with safety max step ${safety_cap} m"
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

# Bearing-UAV is not video. Start tight, then relax only the absolute safety cap
# if the independent observations are too sparse to form a usable sequence.
if ! prepare_sequence 22; then
  echo "[PREP] 22 m safety cap was too sparse; retrying with 26 m"
  if ! prepare_sequence 26; then
    echo "[PREP] 26 m safety cap was too sparse; final retry with 30 m"
    prepare_sequence 30
  fi
fi

# Same v39 architecture, but temporal/Kalman cadence limits are derived from
# TRAIN route frame-step statistics only. Test-route cadence is never used to
# set the adaptation.
python3 v39_otherdata/bearing_runner_cadence_adapted.py \
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

# Simplified visualization: only GT/reference + final prediction.
python3 v39_otherdata/bearing_plot_final_vs_gt.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${PREPARED_ROOT}/v39_output_corrected" \
  --routes test_01 test_02

echo ""
echo "================================================================================================="
echo "Bearing v39 soft-sequence + TRAIN-cadence experiment finished"
echo "Summary: ${PREPARED_ROOT}/v39_output_corrected/bearing_v39_summary.json"
echo "Audit  : ${PREPARED_ROOT}/v39_output_corrected/v39_bearing_training_audit.json"
echo "Plot 1 : ${PREPARED_ROOT}/v39_output_corrected/test_01_final_vs_gt_zoom.jpg"
echo "Plot 2 : ${PREPARED_ROOT}/v39_output_corrected/test_02_final_vs_gt_zoom.jpg"
echo "Route diagnostics: ${PREPARED_ROOT}/experiment.json"
echo "================================================================================================="
