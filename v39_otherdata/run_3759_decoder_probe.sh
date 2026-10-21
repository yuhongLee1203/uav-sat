#!/usr/bin/env bash
set -Eeuo pipefail
cd /yh/study/uav-sat
suite="${ORIGINAL_3759_SUITE:-/yh/study/uav-sat/v39_otherdata/output/original3759_inference_20260925_224113_1642368}"
dataset="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
export BEARING_V40_RELAX_LIMITS=0 BEARING_FRAME_ISOLATED_CACHE=1 BEARING_HEADING_FUSION_ALPHA=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$suite/logs/decoder_probe"
pids=()
for entry in citya:0 cityb:5 cityc:6 cityd:0; do
  city="${entry%:*}"; gpu="${entry#*:}"
  python3 -u v39_otherdata/bearing_iclr_ablation.py eval \
    --suite-root "$suite" --dataset-root "$dataset" \
    --city "$city" --variant decoder_probe --train-frames 3 \
    --gpu "$gpu" --seed 2033 \
    >"$suite/logs/decoder_probe/$city.log" 2>&1 &
  pids+=("$!")
  if (( ${#pids[@]} == 3 )); then
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done
for city in citya cityb cityc cityd; do
  test -s "$suite/$city/variants/decoder_probe/bearing_v39_summary.json"
done
printf '[DECODER PROBE DONE] %s\n' "$suite"
