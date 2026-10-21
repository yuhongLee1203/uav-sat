#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFICIAL_ROOT="${BEARING_OFFICIAL_ROOT:-/yh/study/bearinguav_updateByMe}"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
OUTPUT_ROOT="${BEARING_OFFICIAL_OUTPUT_ROOT:-${ROOT}/v39_otherdata/output/official96_merged}"
METADATA="${DATASET_ROOT}/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv"
EPOCHS="${BEARING_OFFICIAL_EPOCHS:-100}"
GPU="${BEARING_OFFICIAL_GPU:-0}"

[[ -s "${METADATA}" ]] || { echo "ERROR: missing merged metadata: ${METADATA}" >&2; exit 2; }
[[ -d "${OFFICIAL_ROOT}/cvphr" && -d "${OFFICIAL_ROOT}/config" ]] || {
  echo "ERROR: official Bearing-UAV checkout missing: ${OFFICIAL_ROOT}" >&2; exit 2;
}
[[ "$(readlink -f "${OFFICIAL_ROOT}/Bearing_UAV_90K")" == "$(readlink -f "${DATASET_ROOT}")" ]] || {
  echo "ERROR: official checkout dataset link does not point to ${DATASET_ROOT}" >&2; exit 2;
}
[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid epochs: ${EPOCHS}" >&2; exit 2; }

mkdir -p "${OUTPUT_ROOT}"
export OMP_NUM_THREADS="${CPU_THREADS_PER_JOB:-2}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_JOB:-2}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_JOB:-2}"
export BEARING_NUM_WORKERS="${BEARING_NUM_WORKERS:-2}"
export PYTHONPATH="${OFFICIAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Official 96 is one merged 90K metadata index. The official dataloader makes
# its deterministic 85/5/10 split and shuffles all cities together in training.
# This is the Bearing-UAV static localization model, not the v39 temporal model.
cd "${OFFICIAL_ROOT}"
echo "[OFFICIAL96 MERGED] metadata=${METADATA} epochs=${EPOCHS} gpu=${GPU}"
echo "[OFFICIAL96 MERGED] full results=${OFFICIAL_ROOT}/results/c4ma"
CUDA_VISIBLE_DEVICES="${GPU}" python3 -u -m cvphr.train.cvphr_train \
  --rsi_id 96 --n_sample 100 --n_block 15 --is_3d 1 \
  --num_epochs "${EPOCHS}" --device_id 0 --flag_test 1 \
  --local_dataset 0 --flag_ckpt 1 \
  2>&1 | tee "${OUTPUT_ROOT}/train_and_test.log"

if [[ -d "${OFFICIAL_ROOT}/results/c4ma" ]]; then
  ln -sfn "${OFFICIAL_ROOT}/results/c4ma" "${OUTPUT_ROOT}/results"
fi
echo "[OFFICIAL96 DONE] ${OUTPUT_ROOT}/train_and_test.log"
