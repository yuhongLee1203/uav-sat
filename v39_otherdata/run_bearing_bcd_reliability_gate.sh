#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT}/v39_otherdata/output}"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"

for city_name in cityb cityc cityd; do
  suite="${OUTPUT_BASE}/${city_name}_full3f_gate_${RUN_TAG}"
  log="${OUTPUT_BASE}/${city_name}_full3f_gate_${RUN_TAG}.log"
  echo "[BCD GATED START] city=${city_name} suite=${suite}"
  CITY="${city_name}" \
  BEARING_DATASET_ROOT="${DATASET_ROOT}" \
  ICLR_SUITE_ROOT="${suite}" \
  TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-160}" \
  VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}" \
  PATIENCE="${PATIENCE:-14}" \
  SEED="${SEED:-2033}" \
  RESUME_EVAL=0 \
  UPLOAD_RESULTS=0 \
  OMP_NUM_THREADS="${CPU_THREADS_PER_TASK:-2}" \
  MKL_NUM_THREADS="${CPU_THREADS_PER_TASK:-2}" \
  OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_TASK:-2}" \
  NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_TASK:-2}" \
  UAVSAT_TEMPORAL_LR="${UAVSAT_TEMPORAL_LR:-4e-5}" \
  UAVSAT_EARLY_MIN_EPOCH="${UAVSAT_EARLY_MIN_EPOCH:-18}" \
  UAVSAT_TEMPORAL_ADAPTER_3FRAME_SCALE="${UAVSAT_TEMPORAL_ADAPTER_3FRAME_SCALE:-0.50}" \
  UAVSAT_TEMPORAL_DELTA2_SCALE="${UAVSAT_TEMPORAL_DELTA2_SCALE:-0.25}" \
  UAVSAT_KALMAN_PRIOR_BLEND_MAX="${UAVSAT_KALMAN_PRIOR_BLEND_MAX:-0.15}" \
  UAVSAT_LOSS_MEASUREMENT=2.0 \
  UAVSAT_LOSS_NEXT_STEP=3.0 \
  UAVSAT_LOSS_VARIANCE_NLL=0.05 \
  UAVSAT_LOSS_VELOCITY=0 \
  UAVSAT_LOSS_ACCELERATION=0 \
  UAVSAT_LOSS_HEADING=0 \
  bash v39_otherdata/run_bearing_iclr_ablation_fixed.sh \
    2>&1 | tee "${log}"
  echo "[BCD GATED DONE] city=${city_name} suite=${suite}"
done

echo "[BCD GATED COMPLETE] tag=${RUN_TAG}"
