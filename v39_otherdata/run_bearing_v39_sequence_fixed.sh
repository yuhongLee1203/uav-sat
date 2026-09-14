#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CITY="${CITY:-cityb}"
GPU="${GPU:-0}"
PREPARED_ROOT="${REPO_ROOT}/v39_otherdata/generated/${CITY}"

cd "${REPO_ROOT}"

# Fresh external-dataset experiment: remove the old 8 m nearest-per-target
# pseudo-flight routes and their checkpoints/results.
rm -rf "${PREPARED_ROOT}"

python3 v39_otherdata/bearing_prepare_continuous.py \
  --dataset-root "${DATASET_ROOT}" \
  --city "${CITY}" \
  --step-m 4 \
  --max-sample-distance-m 10 \
  --max-frame-step-m 8 \
  --max-cross-step-m 5 \
  --max-backward-step-m 1 \
  --heading-weight-px-per-deg 0 \
  --continuity-weight 2.0 \
  --cross-weight 1.25 \
  --backward-weight 6.0 \
  --skip-penalty 18 \
  --candidate-limit 48 \
  --beam-width 64 \
  --min-selected-ratio 0.55

# IMPORTANT: do not pass --reprepare here. bearing_runner.py's legacy preparer is
# intentionally bypassed; it reuses the sequence-aware experiment prepared above.
python3 v39_otherdata/bearing_runner.py \
  --dataset-root "${DATASET_ROOT}" \
  --city "${CITY}" \
  --gpu "${GPU}" \
  --backbone mobilenet_v3_small \
  --visual-epochs 30 \
  --epochs-per-route 20 \
  --patience 10 \
  --jitter-m 8 \
  --step-m 4 \
  --max-sample-distance-m 10 \
  --heading-weight-px-per-deg 0

python3 v39_otherdata/bearing_plot_final_vs_gt.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${PREPARED_ROOT}/v39_output_corrected" \
  --routes test_01 test_02

echo ""
echo "================================================================================================="
echo "Bearing v39 sequence-fixed experiment finished"
echo "Summary: ${PREPARED_ROOT}/v39_output_corrected/bearing_v39_summary.json"
echo "Plot 1 : ${PREPARED_ROOT}/v39_output_corrected/test_01_final_vs_gt_zoom.jpg"
echo "Plot 2 : ${PREPARED_ROOT}/v39_output_corrected/test_02_final_vs_gt_zoom.jpg"
echo "Route diagnostics: ${PREPARED_ROOT}/experiment.json"
echo "================================================================================================="
