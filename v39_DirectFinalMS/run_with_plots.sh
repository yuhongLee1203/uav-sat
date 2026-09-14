#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"

DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
OUT="${UAVSAT_OUTPUT_DIR:-${ROOT}/output_wc_single}"
VISUAL_CKPT="${UAVSAT_VISUAL_CKPT_FOR_PLOT:-${REPO_ROOT}/forNX/weights/v36_mobilenet_v3_small/checkpoints/visual_retrieval_A_only.pt}"

# Keep the original v39 inference path unchanged. run.sh writes Route B/C CSVs
# and robust_tracker_summary.json into OUT.
UAVSAT_DATA_ROOT="${DATA_ROOT}" \
UAVSAT_OUTPUT_DIR="${OUT}" \
bash "${ROOT}/run.sh"

# Diagnostic rendering only; it does not modify the inference outputs.
python3 "${ROOT}/plot_inference_trajectory.py" \
  --output-dir "${OUT}" \
  --data-root "${DATA_ROOT}" \
  --visual-checkpoint "${VISUAL_CKPT}" \
  --routes route_B route_C

echo "[DONE] trajectory figures:"
echo "  ${OUT}/route_B_inference_trajectory_full.jpg"
echo "  ${OUT}/route_B_inference_trajectory_zoom.jpg"
echo "  ${OUT}/route_C_inference_trajectory_full.jpg"
echo "  ${OUT}/route_C_inference_trajectory_zoom.jpg"
