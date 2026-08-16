#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/native-paper-data/U1652"
LOG_DIR="${SCRIPT_DIR}/logs/native-papers"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/native-papers"
MANIFEST="${SCRIPT_DIR}/native-paper-data/manifest.json"
BATCH="${NATIVE_PAPER_BATCH:-16}"
DENSE_EPOCHS="${DENSEUAV_EPOCHS:-120}"
GAME_EPOCHS="${GAME4LOC_EPOCHS:-5}"
SAMPLE_EPOCHS="${SAMPLE4GEO_EPOCHS:-1}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${NATIVE_PAPER_CPU_THREADS:-2}"
export MKL_NUM_THREADS="${NATIVE_PAPER_CPU_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${NATIVE_PAPER_CPU_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NATIVE_PAPER_CPU_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${SCRIPT_DIR}/cache/huggingface"
export TORCH_HOME="${SCRIPT_DIR}/cache/torch"

cd "${ROOT_DIR}"
python3 -u v36-exp/prepare_native_paper_dataset.py \
  2>&1 | tee "${LOG_DIR}/prepare_dataset.log"

worker_gpu5() {
  export CUDA_VISIBLE_DEVICES=5

  if [[ ! -f "${OUTPUT_DIR}/DenseUAV/summary.json" ]]; then
    cd "${SCRIPT_DIR}/others_paper/DenseUAV"
    python3 -u train.py \
    --gpu_ids 0 \
    --name uavsat_native \
    --data_dir "${DATA_DIR}/train" \
    --num_worker 2 \
    --batchsize "${BATCH}" \
    --num_epochs "${DENSE_EPOCHS}" \
    --backbone resnet50 \
    --head FSRA_CNN \
    --h 224 --w 224 \
      2>&1 | tee "${LOG_DIR}/gpu5_DenseUAV_native.log"

    cd "${SCRIPT_DIR}/others_paper/DenseUAV/checkpoints/uavsat_native"
    dense_checkpoint="$(printf 'net_%03d.pth' "$((DENSE_EPOCHS - 1))")"
    python3 -u test.py \
    --gpu_ids 0 \
    --name uavsat_native \
    --checkpoint "${dense_checkpoint}" \
    --test_dir "${DATA_DIR}/test" \
    --batchsize 128 \
    --num_worker 2 \
      2>&1 | tee "${LOG_DIR}/gpu5_DenseUAV_test.log"
    python3 -u evaluate_gpu.py \
      2>&1 | tee "${LOG_DIR}/gpu5_DenseUAV_evaluate.log"
    cd "${ROOT_DIR}"
    python3 -u v36-exp/collect_native_retrieval.py \
    --method DenseUAV \
    --format denseuav-mat \
    --features "${SCRIPT_DIR}/others_paper/DenseUAV/checkpoints/uavsat_native/pytorch_result_1.mat" \
    --manifest "${MANIFEST}" \
    --output "${OUTPUT_DIR}/DenseUAV/summary.json" \
      2>&1 | tee "${LOG_DIR}/gpu5_DenseUAV_metrics.log"
  else
    echo "reuse completed native DenseUAV"
  fi

  if [[ ! -f "${OUTPUT_DIR}/Game4Loc/summary.json" ]]; then
    cd "${SCRIPT_DIR}/others_paper/GTA-UAV/Game4Loc"
    export UAVSAT_NATIVE_OUTPUT="${OUTPUT_DIR}/Game4Loc"
    export UAVSAT_NATIVE_FEATURE_DUMP="${OUTPUT_DIR}/Game4Loc/features.pt"
    python3 -u train_university.py \
    --data_dir "${DATA_DIR}" \
    --gpu_ids 0 \
    --epochs "${GAME_EPOCHS}" \
    --batch_size "${BATCH}" \
    --model convnext_base.fb_in22k_ft_in1k_384 \
    --log_path "${LOG_DIR}/gpu5_Game4Loc_internal.log" \
      2>&1 | tee "${LOG_DIR}/gpu5_Game4Loc_native.log"
    cd "${ROOT_DIR}"
    python3 -u v36-exp/collect_native_retrieval.py \
    --method Game4Loc \
    --features "${OUTPUT_DIR}/Game4Loc/features.pt" \
    --manifest "${MANIFEST}" \
    --output "${OUTPUT_DIR}/Game4Loc/summary.json" \
      2>&1 | tee "${LOG_DIR}/gpu5_Game4Loc_metrics.log"
  else
    echo "reuse completed native Game4Loc"
  fi
}

worker_gpu6() {
  export CUDA_VISIBLE_DEVICES=6
  export UAVSAT_NATIVE_DATA="${DATA_DIR}"
  export UAVSAT_NATIVE_OUTPUT="${OUTPUT_DIR}/Sample4Geo"
  export UAVSAT_NATIVE_EPOCHS="${SAMPLE_EPOCHS}"
  export UAVSAT_NATIVE_BATCH="${BATCH}"
  export UAVSAT_NATIVE_FEATURE_DUMP="${OUTPUT_DIR}/Sample4Geo/features.pt"

  if [[ ! -f "${OUTPUT_DIR}/Sample4Geo/summary.json" ]]; then
    cd "${SCRIPT_DIR}/others_paper/Sample4Geo"
    python3 -u train_university.py \
      2>&1 | tee "${LOG_DIR}/gpu6_Sample4Geo_native.log"
    cd "${ROOT_DIR}"
    python3 -u v36-exp/collect_native_retrieval.py \
    --method Sample4Geo \
    --features "${OUTPUT_DIR}/Sample4Geo/features.pt" \
    --manifest "${MANIFEST}" \
    --output "${OUTPUT_DIR}/Sample4Geo/summary.json" \
      2>&1 | tee "${LOG_DIR}/gpu6_Sample4Geo_metrics.log"
  else
    echo "reuse completed native Sample4Geo"
  fi
}

worker_gpu5 & pid5=$!
worker_gpu6 & pid6=$!
echo "native paper workers: GPU5=${pid5} GPU6=${pid6}"

status=0
wait "${pid5}" || status=1
wait "${pid6}" || status=1

if [[ "${status}" != "0" ]]; then
  echo "At least one native paper failed; inspect ${LOG_DIR}." >&2
  exit 1
fi

cd "${ROOT_DIR}"
python3 v36-exp/collect_results.py
echo "Native DenseUAV, Game4Loc, and Sample4Geo runs completed."
echo "Logs: ${LOG_DIR}"
