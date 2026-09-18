#!/usr/bin/env bash
set -Eeuo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PKG_ROOT}/.." && pwd)"
DATA_SOURCE="${V39_DATA_SOURCE:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
VISUAL_SOURCE="${V39_VISUAL_CKPT_SOURCE:-${PKG_ROOT}/weights/v36_mobilenet_v3_small/checkpoints/visual_retrieval_A_only.pt}"
TEMPORAL_SOURCE="${V39_TEMPORAL_CKPT_SOURCE:-${REPO_ROOT}/PreviousState-exp/output/mobilenetv3_prevstate/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt}"
WEIGHT_DIR="${PKG_ROOT}/weights/v39_directfinalms/checkpoints"
VISUAL_DST="${WEIGHT_DIR}/visual_retrieval_A_only.pt"
TEMPORAL_DST="${WEIGHT_DIR}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

fail() { echo "ERROR: $*" >&2; exit 2; }

[[ -d "${DATA_SOURCE}" ]] || fail "missing V39 data source: ${DATA_SOURCE}"
[[ -s "${VISUAL_SOURCE}" ]] || fail "missing visual checkpoint: ${VISUAL_SOURCE}"
[[ -s "${TEMPORAL_SOURCE}" ]] || fail "missing temporal checkpoint: ${TEMPORAL_SOURCE}"
for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -s "${PKG_ROOT}/src/${f}" ]] || fail "missing packaged V39 source: ${PKG_ROOT}/src/${f}"
done
[[ -s "${PKG_ROOT}/patch_direct_finalms.py" ]] || fail "missing DirectFinalMS patch"

mkdir -p "${WEIGHT_DIR}" "${PKG_ROOT}/data" "${PKG_ROOT}/pretrained_cache/torch" "${PKG_ROOT}/pretrained_cache/huggingface"
cp -a "${VISUAL_SOURCE}" "${VISUAL_DST}"
cp -a "${TEMPORAL_SOURCE}" "${TEMPORAL_DST}"
cp -a "${DATA_SOURCE}/." "${PKG_ROOT}/data/"

for route in route_A route_B route_C; do
  [[ -s "${PKG_ROOT}/data/routes/${route}/frames.csv" ]] || fail "missing packaged route: ${route}/frames.csv"
done
[[ -s "${PKG_ROOT}/data/satellite/sim_map_competition_roi_crop.png" ]] || fail "missing packaged satellite image"
[[ -s "${PKG_ROOT}/data/satellite/sim_map_competition_roi_crop_worldfile_epsg3826.json" ]] || fail "missing packaged satellite metadata"

if ! find "${PKG_ROOT}/pretrained_cache/torch" -type f -name 'mobilenet_v3_small-*.pth' -print -quit 2>/dev/null | grep -q .; then
  if [[ -d "${HOME}/.cache/torch/hub/checkpoints" ]]; then
    mkdir -p "${PKG_ROOT}/pretrained_cache/torch/hub/checkpoints"
    find "${HOME}/.cache/torch/hub/checkpoints" -maxdepth 1 -type f -name 'mobilenet_v3_small-*.pth' -exec cp -a {} "${PKG_ROOT}/pretrained_cache/torch/hub/checkpoints/" \;
  fi
fi

python3 -m py_compile \
  "${PKG_ROOT}/src/config.py" \
  "${PKG_ROOT}/src/data.py" \
  "${PKG_ROOT}/src/robust_tracker.py" \
  "${PKG_ROOT}/src/visual_localizer.py" \
  "${PKG_ROOT}/src/visual_model.py" \
  "${PKG_ROOT}/patch_direct_finalms.py"

echo "================================================================================"
echo "V39 DirectFinalMS NX PACKAGE READY"
echo "Package : ${PKG_ROOT}"
echo "Backbone: mobilenet_v3_small"
echo "Pipeline: Weighted Centroid -> 3-frame GRU -> fixed-R Kalman -> final MS 6x6/BW7"
echo "Weights : ${WEIGHT_DIR}"
echo "Data    : ${PKG_ROOT}/data"
if find "${PKG_ROOT}/pretrained_cache/torch" -type f -name 'mobilenet_v3_small-*.pth' -print -quit 2>/dev/null | grep -q .; then
  echo "Torch pretrained cache: FOUND"
else
  echo "WARNING: MobileNetV3 torchvision pretrained cache was not found in the package."
  echo "         Prepare it on a machine with internet before using an offline NX."
fi
echo "Next: copy the entire forNX/ directory to the Jetson NX."
echo "On NX: bash run_v39_nx_latency.sh"
echo "================================================================================"
