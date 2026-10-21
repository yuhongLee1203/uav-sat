#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
SUITE_ROOT="${ABCD_SUITE_ROOT:-${ROOT}/v39_otherdata/output/abcd_shared_20260922_121010}"
GPU="${GPU:-0}"
SEED="${SEED:-2033}"
CITIES=(citya cityb cityc cityd)
VARIANTS=(full no_kalman)

if [[ ! -d "${SUITE_ROOT}" ]]; then
  echo "ERROR: suite not found: ${SUITE_ROOT}" >&2
  echo "Set ABCD_SUITE_ROOT to the existing ABCD shared suite." >&2
  exit 2
fi

for city in "${CITIES[@]}"; do
  ckpt="${SUITE_ROOT}/${city}/train_frames3/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  if [[ ! -s "${ckpt}" ]]; then
    echo "ERROR: missing shared frames=3 checkpoint: ${ckpt}" >&2
    exit 3
  fi

done

# Re-evaluate the two rows from the exact same shared 3-frame model, seed,
# prepared data, SoftMS settings and runtime. The only intended component
# difference is EXPERIMENT_KALMAN=fixed vs none.
for city in "${CITIES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    out="${SUITE_ROOT}/${city}/variants/${variant}"
    rm -rf "${out}"
    echo "[EVAL] city=${city} variant=${variant} gpu=${GPU}"
    python3 -u v39_otherdata/bearing_iclr_ablation.py eval \
      --suite-root "${SUITE_ROOT}" \
      --dataset-root "${DATASET_ROOT}" \
      --city "${city}" \
      --variant "${variant}" \
      --train-frames 3 \
      --gpu "${GPU}" \
      --seed "${SEED}"
  done
done

python3 v39_otherdata/build_iclr_ablation_tables.py \
  --suite-root "${SUITE_ROOT}" \
  --cities citya cityb cityc cityd

echo "[DONE] Full vs w/o Kalman regenerated from the same shared checkpoints."
echo "[TABLE] ${SUITE_ROOT}/paper_ablation_tables.md"
echo "[AUDIT] ${SUITE_ROOT}/paper_trend_audit.json"
