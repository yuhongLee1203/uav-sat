#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CITY="cityb"
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--city" ]] && (( i + 1 < ${#ARGS[@]} )); then
    CITY="${ARGS[$((i+1))]}"
  elif [[ "${ARGS[$i]}" == --city=* ]]; then
    CITY="${ARGS[$i]#--city=}"
  fi
done

cd "${REPO_ROOT}"

python3 v39_otherdata/bearing_runner.py "$@"

PREPARED_ROOT="${SCRIPT_DIR}/generated/${CITY}"
OUTPUT_DIR="${PREPARED_ROOT}/v39_output_corrected"

python3 v39_otherdata/bearing_plot_trajectory.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --routes test_01 test_02

echo ""
echo "[DONE] Bearing v39 inference trajectory images:"
echo "  ${OUTPUT_DIR}/test_01_inference_trajectory_full.jpg"
echo "  ${OUTPUT_DIR}/test_01_inference_trajectory_zoom.jpg"
echo "  ${OUTPUT_DIR}/test_02_inference_trajectory_full.jpg"
echo "  ${OUTPUT_DIR}/test_02_inference_trajectory_zoom.jpg"
