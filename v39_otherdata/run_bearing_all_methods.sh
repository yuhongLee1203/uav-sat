#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
GPU_IDS=(0 5 6)
CPU_THREADS="${CPU_THREADS:-2}"
FORCE_OURS="${FORCE_OURS:-0}"
FORCE_CACHE="${FORCE_CACHE:-0}"
FORCE_BASELINES="${FORCE_BASELINES:-0}"
FORCE_OFFICIAL="${FORCE_OFFICIAL:-0}"

GEN_ROOT="${REPO_ROOT}/v39_otherdata/generated"
EXT_ROOT="${REPO_ROOT}/v39_otherdata/external/all_methods"
CACHE_ROOT="${GEN_ROOT}/route_baseline_cache"
BASE_ROOT="${GEN_ROOT}/route_baselines"
OFFICIAL_RESULT_ROOT="${GEN_ROOT}/official_bearinguav_same_routes"
OUT_ROOT="${GEN_ROOT}/all_methods_comparison"
LOG_ROOT="${GEN_ROOT}/all_methods_logs"
VENV="${EXT_ROOT}/env"

UNI_ROOT="${EXT_ROOT}/University1652-Baseline"
SUES_ROOT="${EXT_ROOT}/SUES-200-Benchmark"
DENSE_ROOT="${EXT_ROOT}/DenseUAV"
GTA_ROOT="${EXT_ROOT}/GTA-UAV"
BEARING_ROOT="${EXT_ROOT}/bearinguav"

UNI_COMMIT="c81e4b5c76ba29882a7b1e06b91f3c05ff41dc88"
SUES_COMMIT="85b7488d06db1e6c4c1c52d4a5dc55553f768fb0"
DENSE_COMMIT="b8de18751675fcc0a7a40938b4cd66549647d690"
GTA_COMMIT="1fbc3dbe452db75e82539474f6e2c79d54ad52ac"
BEARING_COMMIT="d16e558a14bd0a9142c6901793fe6a6813e4e954"

cd "${REPO_ROOT}"
mkdir -p "${EXT_ROOT}" "${GEN_ROOT}" "${CACHE_ROOT}" "${BASE_ROOT}" "${LOG_ROOT}"

export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME="${EXT_ROOT}/model_cache/torch"
export HF_HOME="${EXT_ROOT}/model_cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TIMM_FAST_DOWNLOAD=1
mkdir -p "${TORCH_HOME}" "${HF_HOME}"

echo "================================================================================"
echo "Bearing-UAV ALL PUBLIC METHODS / SAME ROUTES"
echo "GPUs            : ${GPU_IDS[*]}"
echo "CPU threads/job : ${CPU_THREADS}"
echo "Image policy    : decode once -> shared mmap .npy cache; model jobs use 0 workers"
echo "Methods retrain : Ours + University-1652 + SUES-200 + DenseUAV + GTA-UAV + Bearing-UAV"
echo "Extra reference : Bearing-UAV authors' official pretrained checkpoint on same frames"
echo "================================================================================"

for gpu in "${GPU_IDS[@]}"; do
  nvidia-smi -i "${gpu}" >/dev/null 2>&1 || { echo "GPU ${gpu} unavailable" >&2; exit 2; }
done

python3 -m py_compile \
  v39_otherdata/bearing_route_baseline_cache.py \
  v39_otherdata/bearing_route_baseline_runner.py \
  v39_otherdata/bearinguav_route_adapted.py \
  v39_otherdata/bearinguav_official_route_eval.py \
  v39_otherdata/bearing_all_methods_compare.py \
  v39_otherdata/bearing_plot_final_vs_gt.py \
  v39_otherdata/bearing_published_reference.py

echo "[CODE-AUDIT] PASS"

ensure_repo() {
  local dir="$1" url="$2" commit="$3" name="$4"
  if [[ ! -d "${dir}/.git" ]]; then
    echo "[SETUP] clone ${name}"
    git clone --filter=blob:none "${url}" "${dir}"
  fi
  local head
  head="$(git -C "${dir}" rev-parse HEAD 2>/dev/null || true)"
  if [[ "${head}" != "${commit}" ]]; then
    git -C "${dir}" fetch --depth 1 origin "${commit}"
    git -C "${dir}" checkout --detach "${commit}"
  fi
  echo "[SETUP] ${name} @ $(git -C "${dir}" rev-parse --short HEAD)"
}

ensure_repo "${UNI_ROOT}" "https://github.com/layumi/University1652-Baseline.git" "${UNI_COMMIT}" "University-1652"
ensure_repo "${SUES_ROOT}" "https://github.com/Reza-Zhu/SUES-200-Benchmark.git" "${SUES_COMMIT}" "SUES-200"
ensure_repo "${DENSE_ROOT}" "https://github.com/Dmmm1997/DenseUAV.git" "${DENSE_COMMIT}" "DenseUAV"
ensure_repo "${GTA_ROOT}" "https://github.com/Yux1angJi/GTA-UAV.git" "${GTA_COMMIT}" "GTA-UAV"
ensure_repo "${BEARING_ROOT}" "https://github.com/liukejia121/bearinguav.git" "${BEARING_COMMIT}" "Bearing-UAV"

# The official Bearing code resolves its default dataset relative to repo root.
if [[ -e "${BEARING_ROOT}/Bearing_UAV_90K" && ! -L "${BEARING_ROOT}/Bearing_UAV_90K" ]]; then
  echo "${BEARING_ROOT}/Bearing_UAV_90K exists and is not a symlink; refusing to overwrite" >&2
  exit 3
fi
ln -sfn "${DATASET_ROOT}" "${BEARING_ROOT}/Bearing_UAV_90K"

# One environment and one pretrained-model cache for every baseline.  The user's
# installed CUDA/PyTorch is reused; only light Python dependencies are added.
if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${VENV}"
fi
PY="${VENV}/bin/python"
if ! "${PY}" - <<'PY' >/dev/null 2>&1
import timm, pandas, PIL, scipy, yaml, cv2, imgaug, albumentations, einops, thop
import pytorch_metric_learning, sklearn, torch, torchvision
PY
then
  echo "[SETUP] install shared non-PyTorch dependencies once"
  PIP_DISABLE_PIP_VERSION_CHECK=1 "${PY}" -m pip install -q \
    timm pandas pillow scipy pyyaml opencv-python-headless imgaug albumentations \
    einops thop pytorch-metric-learning scikit-learn
fi

# -----------------------------------------------------------------------------
# OUR METHOD: reuse valid current runs; if any are missing, run only those cities
# across GPU 0/5/6.  The strong paper figures are always regenerated afterwards.
# -----------------------------------------------------------------------------
ours_ok() {
  local city="$1" out="${GEN_ROOT}/${city}/v39_output_bearing_adapted"
  [[ -s "${out}/bearing_v39_summary.json" && -d "${GEN_ROOT}/${city}/routes/test_01" && -d "${GEN_ROOT}/${city}/routes/test_02" ]]
}
run_ours() {
  local city="$1" gpu="$2"
  echo "[OURS] START ${city} GPU${gpu}"
  env OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" OPENBLAS_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
      DATASET_ROOT="${DATASET_ROOT}" CITY="${city}" GPU="${gpu}" \
      bash v39_otherdata/run_bearing_v39_sequence_fixed.sh > >(tee "${LOG_ROOT}/ours_${city}.log") 2>&1
  echo "[OURS] DONE ${city}"
}
pending=()
for city in citya cityb cityc cityd; do
  if [[ "${FORCE_OURS}" == "1" ]] || ! ours_ok "${city}"; then pending+=("${city}"); else echo "[OURS] cache hit ${city}"; fi
done
for ((base=0;base<${#pending[@]};base+=3)); do
  pids=(); names=()
  for slot in 0 1 2; do
    idx=$((base+slot)); ((idx<${#pending[@]})) || break
    run_ours "${pending[$idx]}" "${GPU_IDS[$slot]}" & pids+=("$!"); names+=("${pending[$idx]}")
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "[OURS] FAILED ${names[$i]}" >&2; exit 4; }; done
done
for city in citya cityb cityc cityd; do ours_ok "${city}" || { echo "[OURS] incomplete ${city}" >&2; exit 5; }; done

# Re-render OUR eight figures with true per-frame GT (purple dashed) and final
# prediction (thick red + white halo). No retraining if results are cached.
RUN_MODEL=0 DATASET_ROOT="${DATASET_ROOT}" GPU=0 bash v39_otherdata/run_bearing_paper_bundle.sh \
  > >(tee "${LOG_ROOT}/ours_paper_bundle.log") 2>&1

# -----------------------------------------------------------------------------
# SHARED ROUTE CACHE: JPEG/RSI decode once. Later 20 baseline training jobs mmap
# these arrays and do resize/normalization on GPU with zero DataLoader workers.
# -----------------------------------------------------------------------------
cache_args=()
[[ "${FORCE_CACHE}" == "1" ]] && cache_args+=(--force)
"${PY}" v39_otherdata/bearing_route_baseline_cache.py \
  --generated-root "${GEN_ROOT}" --cache-root "${CACHE_ROOT}" \
  --cities citya cityb cityc cityd "${cache_args[@]}" \
  > >(tee "${LOG_ROOT}/route_cache.log") 2>&1

# -----------------------------------------------------------------------------
# ROUTE-ADAPTED PUBLIC METHODS.  Jobs are deliberately interleaved by method so
# the first three GPUs do not simultaneously download the same pretrained model.
# Every city/method checkpoint is cached.
# -----------------------------------------------------------------------------
repo_for() {
  case "$1" in
    university1652) echo "${UNI_ROOT}";; sues200) echo "${SUES_ROOT}";;
    denseuav) echo "${DENSE_ROOT}";; gtauav) echo "${GTA_ROOT}";;
    *) return 1;; esac
}
commit_for() {
  case "$1" in
    university1652) echo "${UNI_COMMIT}";; sues200) echo "${SUES_COMMIT}";;
    denseuav) echo "${DENSE_COMMIT}";; gtauav) echo "${GTA_COMMIT}";;
    *) return 1;; esac
}
baseline_ok() {
  local method="$1" city="$2"
  if [[ "${method}" == "bearinguav_route_adapted" ]]; then
    [[ -s "${BASE_ROOT}/${method}/${city}/result.json" && -s "${BASE_ROOT}/${method}/${city}/test_01_final_result.jpg" && -s "${BASE_ROOT}/${method}/${city}/test_02_final_result.jpg" ]]
  else
    [[ -s "${BASE_ROOT}/${method}/${city}/result.json" && -s "${BASE_ROOT}/${method}/${city}/test_01_final_result.jpg" && -s "${BASE_ROOT}/${method}/${city}/test_02_final_result.jpg" ]]
  fi
}
run_route_job() {
  local method="$1" city="$2" gpu="$3"
  echo "[ROUTE-BASELINE] START ${method}/${city} GPU${gpu}"
  local force=(); [[ "${FORCE_BASELINES}" == "1" ]] && force+=(--force)
  if [[ "${method}" == "bearinguav_route_adapted" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" v39_otherdata/bearinguav_route_adapted.py \
      --official-root "${BEARING_ROOT}" --dataset-root "${DATASET_ROOT}" \
      --prepared-root "${GEN_ROOT}/${city}" --cache-root "${CACHE_ROOT}/${city}" \
      --output-root "${BASE_ROOT}" --city "${city}" --repo-commit "${BEARING_COMMIT}" \
      --cpu-threads "${CPU_THREADS}" "${force[@]}" \
      > >(tee "${LOG_ROOT}/${method}_${city}.log") 2>&1
  else
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" v39_otherdata/bearing_route_baseline_runner.py \
      --method "${method}" --repo-dir "$(repo_for "${method}")" \
      --repo-commit "$(commit_for "${method}")" --cache-dir "${CACHE_ROOT}/${city}" \
      --prepared-root "${GEN_ROOT}/${city}" --output-root "${BASE_ROOT}" --city "${city}" \
      --cpu-threads "${CPU_THREADS}" "${force[@]}" \
      > >(tee "${LOG_ROOT}/${method}_${city}.log") 2>&1
  fi
  echo "[ROUTE-BASELINE] DONE ${method}/${city}"
}

jobs=()
for city in citya cityb cityc cityd; do
  for method in university1652 sues200 denseuav gtauav bearinguav_route_adapted; do
    if [[ "${FORCE_BASELINES}" == "1" ]] || ! baseline_ok "${method}" "${city}"; then
      jobs+=("${method}|${city}")
    else
      echo "[ROUTE-BASELINE] cache hit ${method}/${city}"
    fi
  done
done

for ((base=0;base<${#jobs[@]};base+=3)); do
  pids=(); names=()
  for slot in 0 1 2; do
    idx=$((base+slot)); ((idx<${#jobs[@]})) || break
    IFS='|' read -r method city <<<"${jobs[$idx]}"
    run_route_job "${method}" "${city}" "${GPU_IDS[$slot]}" & pids+=("$!"); names+=("${method}/${city}")
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "[ROUTE-BASELINE] FAILED ${names[$i]} -- see log" >&2; exit 6; }
  done
done
for city in citya cityb cityc cityd; do
  for method in university1652 sues200 denseuav gtauav bearinguav_route_adapted; do
    baseline_ok "${method}" "${city}" || { echo "[ROUTE-BASELINE] missing ${method}/${city}" >&2; exit 7; }
  done
done

# -----------------------------------------------------------------------------
# BEARING-UAV AUTHORS' PRETRAINED CHECKPOINT: additional same-frame reference.
# It is intentionally a separate row from route-adapted Bearing-UAV.
# -----------------------------------------------------------------------------
WEIGHT_ZIP="${EXT_ROOT}/Bearing_UAV.zip"
WEIGHT_DIR="${BEARING_ROOT}/Bearing_UAV/cross_view"
if [[ ! -s "${WEIGHT_DIR}/best_model.pth" || ! -s "${WEIGHT_DIR}/training_configure.json" ]]; then
  echo "[OFFICIAL] download authors' Bearing_UAV.zip once"
  tmp="${WEIGHT_ZIP}.part"; rm -f "${tmp}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 4 --retry-delay 3 -o "${tmp}" "https://huggingface.co/HaoyZhou/bearinguav/resolve/main/Bearing_UAV.zip?download=true"
  else
    wget -O "${tmp}" "https://huggingface.co/HaoyZhou/bearinguav/resolve/main/Bearing_UAV.zip?download=true"
  fi
  mv "${tmp}" "${WEIGHT_ZIP}"
  "${PY}" - "${WEIGHT_ZIP}" "${BEARING_ROOT}" <<'PY'
import sys,zipfile
z,out=sys.argv[1:]
with zipfile.ZipFile(z) as f:f.extractall(out)
PY
fi

official_ok(){ [[ -s "${OFFICIAL_RESULT_ROOT}/$1/official_bearinguav_same_route.json" ]]; }
run_official(){
  local city="$1" gpu="$2"
  echo "[OFFICIAL-PRETRAINED] START ${city} GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" v39_otherdata/bearinguav_official_route_eval.py \
    --official-root "${BEARING_ROOT}" --weights-dir "${WEIGHT_DIR}" --dataset-root "${DATASET_ROOT}" \
    --generated-root "${GEN_ROOT}" --city "${city}" --output-root "${OFFICIAL_RESULT_ROOT}" \
    --workers 0 --batch-size 16 --cpu-threads "${CPU_THREADS}" \
    > >(tee "${LOG_ROOT}/bearinguav_official_${city}.log") 2>&1
}
pending=()
for city in citya cityb cityc cityd; do
  if [[ "${FORCE_OFFICIAL}" == "1" ]] || ! official_ok "${city}"; then pending+=("${city}"); else echo "[OFFICIAL-PRETRAINED] cache hit ${city}"; fi
done
for ((base=0;base<${#pending[@]};base+=3)); do
  pids=();names=()
  for slot in 0 1 2;do idx=$((base+slot));((idx<${#pending[@]}))||break;run_official "${pending[$idx]}" "${GPU_IDS[$slot]}" & pids+=("$!");names+=("${pending[$idx]}");done
  for i in "${!pids[@]}";do wait "${pids[$i]}"||{ echo "[OFFICIAL-PRETRAINED] FAILED ${names[$i]}" >&2;exit 8;};done
done

# -----------------------------------------------------------------------------
# FINAL TABLES + FIGURES.
# -----------------------------------------------------------------------------
rm -rf "${OUT_ROOT}";mkdir -p "${OUT_ROOT}/figures/ours" "${OUT_ROOT}/figures/route_adapted"
"${PY}" v39_otherdata/bearing_all_methods_compare.py \
  --generated-root "${GEN_ROOT}" --baseline-root "${BASE_ROOT}" \
  --official-result-root "${OFFICIAL_RESULT_ROOT}" --output-dir "${OUT_ROOT}"
"${PY}" v39_otherdata/bearing_published_reference.py --generated-root "${GEN_ROOT}" --output-dir "${OUT_ROOT}"

for city in citya cityb cityc cityd;do
  for route in test_01 test_02;do
    cp "${GEN_ROOT}/paper_bundle/${city}_${route}_final_result.jpg" "${OUT_ROOT}/figures/ours/"
    for method in university1652 sues200 denseuav gtauav bearinguav_route_adapted;do
      mkdir -p "${OUT_ROOT}/figures/route_adapted/${method}"
      cp "${BASE_ROOT}/${method}/${city}/${route}_final_result.jpg" "${OUT_ROOT}/figures/route_adapted/${method}/${city}_${route}_final_result.jpg"
    done
  done
done

cat > "${OUT_ROOT}/README.txt" <<EOF
PRIMARY SAME-ROUTE TABLE:
  all_methods_same_routes_pooled.csv
  all_methods_same_routes_route_level.csv

Actually retrained on selected Route-A and tested on the same 8 routes:
  Ours v39 Bearing-adapted
  University-1652 route-adapted
  SUES-200 route-adapted
  DenseUAV route-adapted
  GTA-UAV route-adapted
  Bearing-UAV route-adapted

Additional same-frame reference:
  Bearing-UAV official pretrained VGG-16 (authors' released full-dataset checkpoint)

Published-only reference (NOT rerun):
  bearinguav_published_uav_reference.csv

Figure semantics:
  purple dashed = true per-frame GT
  red solid     = prediction
  gray dotted   = planned waypoint/reference route

Important: route adaptation puts methods on the same selected data/GT, but algorithmic priors still differ.
Do not replace published benchmark claims with route-adapted reproduction values.
EOF

echo ""
echo "================================================================================"
echo "ALL METHODS COMPLETE"
echo "Primary pooled table : ${OUT_ROOT}/all_methods_same_routes_pooled.csv"
echo "Per-route table       : ${OUT_ROOT}/all_methods_same_routes_route_level.csv"
echo "Published references  : ${OUT_ROOT}/bearinguav_published_uav_reference.csv"
echo "OUR clear figures     : ${OUT_ROOT}/figures/ours/"
echo "Baseline figures      : ${OUT_ROOT}/figures/route_adapted/"
echo "Logs                  : ${LOG_ROOT}/"
echo "================================================================================"
