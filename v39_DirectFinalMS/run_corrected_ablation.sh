#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[CLEAN v39] Only change: front MS1 -> posterior Weighted Centroid + posterior spatial variance"
echo "[CLEAN v39] Original temporal checkpoint is reused; no temporal retraining or training-hyperparameter changes"
RUN_ALL_EXPERIMENTS=1 bash "${ROOT}/run.sh"
