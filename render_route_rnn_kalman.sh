#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

INPUT_DIR="${INPUT_DIR:-outputs/route_rnn_filterpy_full_retrain}"
FPS="${FPS:-20}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-500}"

python3 render_route_rnn_kalman.py \
  --input-dir "${INPUT_DIR}" \
  --route both \
  --fps "${FPS}" \
  --max-video-frames "${MAX_VIDEO_FRAMES}"
