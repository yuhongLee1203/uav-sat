#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="${ROOT_DIR}/outputs/v36_v34protocol_compact_gru_softms_mode_variance_forward3x6_polynomial_kalman/feature_cache"
TARGET="${ROOT_DIR}/v36-exp/cache/mobileclip2_s2"
mkdir -p "${TARGET}"
for route in route_A route_B route_C; do
  src="${SOURCE}/${route}_uav_clip.pt"
  dst="${TARGET}/${route}_uav_clip.pt"
  if [[ ! -s "${dst}" ]]; then
    cp -p "${src}" "${dst}"
  fi
done
echo "Shared frozen-backbone cache ready: ${TARGET}"
