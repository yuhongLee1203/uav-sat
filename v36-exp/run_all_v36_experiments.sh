#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${SCRIPT_DIR}/outputs/internal" "${SCRIPT_DIR}/outputs/papers" "${SCRIPT_DIR}/cache"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HOME="${SCRIPT_DIR}/cache/huggingface"
export TORCH_HOME="${SCRIPT_DIR}/cache/torch"
export TIMM_HOME="${SCRIPT_DIR}/cache/timm"
export OMP_NUM_THREADS="${V36_EXP_CPU_THREADS:-2}"
export MKL_NUM_THREADS="${V36_EXP_CPU_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${V36_EXP_CPU_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${V36_EXP_CPU_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TIMM_HOME}"

for spec in \
  "DenseUAV|https://github.com/Dmmm1997/DenseUAV.git" \
  "Sample4Geo|https://github.com/Skyy93/Sample4Geo.git" \
  "GTA-UAV|https://github.com/Yux1angJi/GTA-UAV.git" \
  "InfoGeo|https://github.com/HRT00/Official_InfoGeo.git" \
  "Bearing-UAV|https://github.com/liukejia121/bearinguav.git"; do
  name="${spec%%|*}"; url="${spec#*|}"; path="${SCRIPT_DIR}/others_paper/${name}"
  if [[ ! -d "${path}/.git" ]]; then
    git clone --depth 1 "${url}" "${path}"
  else
    echo "reuse official repository: ${path}"
  fi
done

# Reuse the already encoded MobileCLIP UAV features for every internal
# ablation.  Do not run the obsolete "official backbone + our local-18 prior"
# paper adapter: that is not the papers' native protocol and its numbers are
# deliberately excluded from Table 8.
bash "${SCRIPT_DIR}/prepare_shared_cache.sh"

run_internal() {
  local gpu="$1" variant="$2"
  echo "[GPU ${gpu}] internal ${variant}"
  bash "${SCRIPT_DIR}/run_internal_variant.sh" "${variant}" "${gpu}" \
    2>&1 | tee "${LOG_DIR}/gpu${gpu}_internal_${variant}.log"
}

worker_gpu0() {
  run_internal 0 full_v36
  run_internal 0 weighted_centroid
  run_internal 0 frame1
  run_internal 0 motion_kalman_cv
}

worker_gpu5() {
  run_internal 5 full_6x6
  run_internal 5 forward_3x6_aligned
  run_internal 5 frame2
  run_internal 5 motion_velocity
}

worker_gpu6() {
  run_internal 6 softms_only
  run_internal 6 softms_gru
  run_internal 6 softms_gru_poly
}

worker_gpu0 >"${LOG_DIR}/worker_gpu0.log" 2>&1 & pid0=$!
worker_gpu5 >"${LOG_DIR}/worker_gpu5.log" 2>&1 & pid5=$!
worker_gpu6 >"${LOG_DIR}/worker_gpu6.log" 2>&1 & pid6=$!
echo "workers: GPU0=${pid0} GPU5=${pid5} GPU6=${pid6}"

status=0
wait "${pid0}" || status=1
wait "${pid5}" || status=1
wait "${pid6}" || status=1
CUDA_VISIBLE_DEVICES=0 python3 -u "${SCRIPT_DIR}/benchmark_anchor_aggregation.py" \
  2>&1 | tee "${LOG_DIR}/aggregation_benchmark.log" || status=1
python3 "${SCRIPT_DIR}/collect_results.py"

if [[ "${status}" != "0" ]]; then
  echo "At least one GPU queue failed. Completed rows were preserved and collected; inspect ${LOG_DIR}." >&2
  exit 1
fi
echo "Corrected V36/internal experiments complete: ${SCRIPT_DIR}/results.md"
echo "Table 8 native-paper rows remain PENDING; invalid local-18 adapters were not run."
