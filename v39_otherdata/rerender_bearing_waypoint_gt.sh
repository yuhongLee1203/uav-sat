#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CITY="${CITY:-cityb}"
PREPARED_ROOT="${REPO_ROOT}/v39_otherdata/generated/${CITY}"
OUTPUT_DIR="${PREPARED_ROOT}/v39_output_bearing_adapted"
PAPER_DIR="${OUTPUT_DIR}/paper_figures_waypoint_gt"

case "${CITY}" in
  citya|cityb|cityc|cityd) ;;
  *) echo "ERROR: unsupported CITY=${CITY}" >&2; exit 2 ;;
esac

# Plot-only safety contract: never prepare, train, delete or overwrite model outputs.
[[ -f "${PREPARED_ROOT}/routes/test_01/waypoints.json" ]] || {
  echo "ERROR: missing test_01 waypoints.json" >&2; exit 3;
}
[[ -f "${PREPARED_ROOT}/routes/test_02/waypoints.json" ]] || {
  echo "ERROR: missing test_02 waypoints.json" >&2; exit 3;
}
[[ -f "${OUTPUT_DIR}/bearing_v39_summary.json" ]] || {
  echo "ERROR: missing existing bearing_v39_summary.json; run inference first" >&2; exit 4;
}

python3 -m py_compile v39_otherdata/bearing_plot_final_vs_gt.py

python3 v39_otherdata/bearing_plot_final_vs_gt.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --routes test_01 test_02

for f in \
  "${PAPER_DIR}/test_01_waypoint_gt_green.jpg" \
  "${PAPER_DIR}/test_02_waypoint_gt_green.jpg" \
  "${PAPER_DIR}/plot_source_audit.json"; do
  [[ -s "${f}" ]] || { echo "ERROR: expected output missing: ${f}" >&2; exit 5; }
done

echo "================================================================================"
echo "PLOT-ONLY DONE -- NO training/inference was rerun"
echo "GT      : GREEN SOLID sparse official waypoints only"
echo "Predict : RED SOLID raw final_x/final_y, no smoothing"
echo "Open ONLY these paper figures:"
echo "  ${PAPER_DIR}/test_01_waypoint_gt_green.jpg"
echo "  ${PAPER_DIR}/test_02_waypoint_gt_green.jpg"
echo "Audit:"
echo "  ${PAPER_DIR}/plot_source_audit.json"
echo "================================================================================"
