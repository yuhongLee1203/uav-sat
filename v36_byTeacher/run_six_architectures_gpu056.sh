#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p logs/six_architecture_ablation
EPOCHS="${EPOCHS:-60}"
run_pair () {
  local gpu="$1"; shift
  for arch in "$@"; do
    echo "[GPU ${gpu}] starting ${arch}"
    CUDA_VISIBLE_DEVICES="${gpu}" python six_architecture_experiment.py \
      --mode train-eval --arch "${arch}" --device cuda:0 --epochs "${EPOCHS}" \
      2>&1 | tee "logs/six_architecture_ablation/${arch}_gpu${gpu}.log"
  done
}
run_pair 0 MKG MGK &
P0=$!
run_pair 5 GMK GKM &
P5=$!
run_pair 6 KGM KMG &
P6=$!
wait "$P0" "$P5" "$P6"
echo "All six architectures finished."
