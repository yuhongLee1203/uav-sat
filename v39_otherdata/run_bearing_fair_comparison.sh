#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
GPU_IDS=(0 5 6)
CPU_THREADS="${CPU_THREADS:-2}"
DL_WORKERS="${DL_WORKERS:-2}"
OFFICIAL_BATCH="${OFFICIAL_BATCH:-4}"
FORCE_OURS="${FORCE_OURS:-0}"
FORCE_OFFICIAL="${FORCE_OFFICIAL:-0}"

GEN_ROOT="${REPO_ROOT}/v39_otherdata/generated"
EXT_ROOT="${REPO_ROOT}/v39_otherdata/external"
OFFICIAL_ROOT="${EXT_ROOT}/bearinguav"
OFFICIAL_COMMIT="d16e558a14bd0a9142c6901793fe6a6813e4e954"
WEIGHT_ZIP="${EXT_ROOT}/Bearing_UAV.zip"
WEIGHT_DIR="${OFFICIAL_ROOT}/Bearing_UAV/cross_view"
VENV="${EXT_ROOT}/bearinguav_env"
OFFICIAL_RESULT_ROOT="${GEN_ROOT}/official_bearinguav_same_routes"
FAIR_DIR="${GEN_ROOT}/fair_comparison"
PAPER_BUNDLE="${GEN_ROOT}/paper_bundle"
LOG_DIR="${GEN_ROOT}/fair_comparison_logs"

cd "${REPO_ROOT}"
mkdir -p "${GEN_ROOT}" "${EXT_ROOT}" "${LOG_DIR}"

export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"
export TOKENIZERS_PARALLELISM=false

# Keep CPU/I/O pressure bounded: at most one model process per GPU and only two
# DataLoader workers per official-Bearing process.  The three jobs read distinct
# city images, while metadata/weights are cached and reused.
echo "================================================================================"
echo "Bearing fair-comparison runner"
echo "GPUs            : ${GPU_IDS[*]}"
echo "CPU threads/job : ${CPU_THREADS}"
echo "DL workers/job  : ${DL_WORKERS}"
echo "Policy          : cache first; only missing/stale stages rerun"
echo "================================================================================"

for gpu in "${GPU_IDS[@]}"; do
  if ! nvidia-smi -i "${gpu}" >/dev/null 2>&1; then
    echo "Required GPU ${gpu} is not available." >&2
    exit 2
  fi
done

python3 -m py_compile \
  v39_otherdata/bearinguav_official_route_eval.py \
  v39_otherdata/bearing_same_route_comparison.py \
  v39_otherdata/bearing_published_reference.py \
  v39_otherdata/bearing_plot_final_vs_gt.py

echo "[CODE-AUDIT] PASS"

ours_ok() {
  local city="$1"
  local out="${GEN_ROOT}/${city}/v39_output_bearing_adapted"
  [[ -s "${out}/bearing_v39_summary.json" \
     && -s "${out}/bearing_paper_metrics.json" \
     && -s "${out}/test_01_final_result.jpg" \
     && -s "${out}/test_02_final_result.jpg" ]]
}

run_ours_city() {
  local city="$1" gpu="$2"
  echo "[OURS] START ${city} on GPU ${gpu}"
  env \
    OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" \
    OPENBLAS_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
    DATASET_ROOT="${DATASET_ROOT}" CITY="${city}" GPU="${gpu}" \
    bash v39_otherdata/run_bearing_v39_sequence_fixed.sh \
    > >(tee "${LOG_DIR}/ours_${city}.log") 2>&1
  echo "[OURS] DONE ${city}"
}

# ---------------------------------------------------------------------------
# 1) OUR MODEL. Reuse the already valid four-city results; only rerun missing
#    cities unless FORCE_OURS=1. Missing cities are distributed across 0/5/6.
# ---------------------------------------------------------------------------
pending=()
for city in citya cityb cityc cityd; do
  if [[ "${FORCE_OURS}" == "1" ]] || ! ours_ok "${city}"; then
    pending+=("${city}")
  else
    echo "[OURS] cache hit: ${city}"
  fi
done

for ((base=0; base<${#pending[@]}; base+=3)); do
  pids=()
  names=()
  for slot in 0 1 2; do
    idx=$((base + slot))
    if (( idx >= ${#pending[@]} )); then
      break
    fi
    city="${pending[$idx]}"
    gpu="${GPU_IDS[$slot]}"
    run_ours_city "${city}" "${gpu}" &
    pids+=("$!")
    names+=("${city}")
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[OURS] FAILED ${names[$i]}" >&2
      exit 3
    fi
  done
done

for city in citya cityb cityc cityd; do
  ours_ok "${city}" || { echo "[OURS] incomplete output for ${city}" >&2; exit 4; }
done

# Re-render paper-style figures and refresh published-reference tables without
# retraining; this is mostly image/CSV work and is intentionally serial.
RUN_MODEL=0 DATASET_ROOT="${DATASET_ROOT}" GPU=0 \
  bash v39_otherdata/run_bearing_paper_bundle.sh \
  > >(tee "${LOG_DIR}/paper_bundle.log") 2>&1

# ---------------------------------------------------------------------------
# 2) OFFICIAL BEARING-UAV CODE + OFFICIAL CHECKPOINT.
#    Cache clone, dataset symlink, Python env, weights. No dataset copy.
# ---------------------------------------------------------------------------
if [[ ! -d "${OFFICIAL_ROOT}/.git" ]]; then
  echo "[OFFICIAL] cloning Bearing-UAV once"
  git clone --filter=blob:none https://github.com/liukejia121/bearinguav.git "${OFFICIAL_ROOT}"
fi
if [[ "$(git -C "${OFFICIAL_ROOT}" rev-parse HEAD)" != "${OFFICIAL_COMMIT}" ]]; then
  git -C "${OFFICIAL_ROOT}" fetch --depth 1 origin "${OFFICIAL_COMMIT}"
  git -C "${OFFICIAL_ROOT}" checkout --detach "${OFFICIAL_COMMIT}"
fi

if [[ -e "${OFFICIAL_ROOT}/Bearing_UAV_90K" && ! -L "${OFFICIAL_ROOT}/Bearing_UAV_90K" ]]; then
  echo "${OFFICIAL_ROOT}/Bearing_UAV_90K exists and is not a symlink; refusing to overwrite." >&2
  exit 5
fi
ln -sfn "${DATASET_ROOT}" "${OFFICIAL_ROOT}/Bearing_UAV_90K"

if [[ ! -s "${WEIGHT_DIR}/best_model.pth" || ! -s "${WEIGHT_DIR}/training_configure.json" ]]; then
  echo "[OFFICIAL] downloading official Bearing_UAV.zip once (~131 MB)"
  tmp="${WEIGHT_ZIP}.part"
  rm -f "${tmp}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 4 --retry-delay 3 \
      -o "${tmp}" \
      "https://huggingface.co/HaoyZhou/bearinguav/resolve/main/Bearing_UAV.zip?download=true"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${tmp}" \
      "https://huggingface.co/HaoyZhou/bearinguav/resolve/main/Bearing_UAV.zip?download=true"
  else
    echo "curl or wget is required for the official pretrained checkpoint." >&2
    exit 6
  fi
  mv "${tmp}" "${WEIGHT_ZIP}"
  python3 - "${WEIGHT_ZIP}" "${OFFICIAL_ROOT}" <<'PY'
import sys, zipfile
from pathlib import Path
z, out = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(z) as f:
    f.extractall(out)
print("[OFFICIAL] checkpoint archive extracted")
PY
fi

test -s "${WEIGHT_DIR}/best_model.pth"
test -s "${WEIGHT_DIR}/training_configure.json"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "[OFFICIAL] creating cached lightweight venv (reuses system CUDA PyTorch)"
  python3 -m venv --system-site-packages "${VENV}"
fi
OFF_PY="${VENV}/bin/python"

if ! "${OFF_PY}" - <<'PY' >/dev/null 2>&1
import cv2, numpy, pandas, PIL, scipy, skimage, imageio, albumentations, imgaug, timm, einops, torch, torchvision
PY
then
  echo "[OFFICIAL] installing non-PyTorch Bearing-UAV dependencies once"
  req="${EXT_ROOT}/bearinguav_requirements_no_torch.txt"
  grep -Ev '^[[:space:]]*(torch|torchvision|torchaudio)==|^[[:space:]]*$|^[[:space:]]*#' \
    "${OFFICIAL_ROOT}/requirements.txt" > "${req}"
  PIP_DISABLE_PIP_VERSION_CHECK=1 "${OFF_PY}" -m pip install -q -r "${req}"
fi

# ---------------------------------------------------------------------------
# 3) SAME-ROUTE official Bearing-UAV rerun. Three cities in parallel on 0/5/6,
#    then the remaining city. Each process reads only the selected route frames.
# ---------------------------------------------------------------------------
official_ok() {
  local city="$1"
  [[ -s "${OFFICIAL_RESULT_ROOT}/${city}/official_bearinguav_same_route.json" ]]
}

run_official_city() {
  local city="$1" gpu="$2"
  echo "[OFFICIAL] START ${city} on physical GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" \
  OPENBLAS_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
  "${OFF_PY}" v39_otherdata/bearinguav_official_route_eval.py \
    --official-root "${OFFICIAL_ROOT}" \
    --weights-dir "${WEIGHT_DIR}" \
    --dataset-root "${DATASET_ROOT}" \
    --generated-root "${GEN_ROOT}" \
    --city "${city}" \
    --output-root "${OFFICIAL_RESULT_ROOT}" \
    --workers "${DL_WORKERS}" \
    --batch-size "${OFFICIAL_BATCH}" \
    --cpu-threads "${CPU_THREADS}" \
    > >(tee "${LOG_DIR}/official_${city}.log") 2>&1
  echo "[OFFICIAL] DONE ${city}"
}

pending=()
for city in citya cityb cityc cityd; do
  if [[ "${FORCE_OFFICIAL}" == "1" ]] || ! official_ok "${city}"; then
    pending+=("${city}")
  else
    echo "[OFFICIAL] cache hit: ${city}"
  fi
done

for ((base=0; base<${#pending[@]}; base+=3)); do
  pids=()
  names=()
  for slot in 0 1 2; do
    idx=$((base + slot))
    if (( idx >= ${#pending[@]} )); then
      break
    fi
    city="${pending[$idx]}"
    gpu="${GPU_IDS[$slot]}"
    run_official_city "${city}" "${gpu}" &
    pids+=("$!")
    names+=("${city}")
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[OFFICIAL] FAILED ${names[$i]} -- see ${LOG_DIR}/official_${names[$i]}.log" >&2
      exit 7
    fi
  done
done

for city in citya cityb cityc cityd; do
  official_ok "${city}" || { echo "[OFFICIAL] missing ${city} result" >&2; exit 8; }
done

# ---------------------------------------------------------------------------
# 4) Aggregate: same-route rerun table + published external-baseline references.
# ---------------------------------------------------------------------------
rm -rf "${FAIR_DIR}"
mkdir -p "${FAIR_DIR}"

python3 v39_otherdata/bearing_same_route_comparison.py \
  --generated-root "${GEN_ROOT}" \
  --official-result-root "${OFFICIAL_RESULT_ROOT}" \
  --output-dir "${FAIR_DIR}"

# Refresh published rows. These are reference values, NOT re-runs.
python3 v39_otherdata/bearing_published_reference.py \
  --generated-root "${GEN_ROOT}" \
  --output-dir "${FAIR_DIR}"

# Copy the eight current paper-style figures next to the comparison tables.
for city in citya cityb cityc cityd; do
  for route in test_01 test_02; do
    cp "${PAPER_BUNDLE}/${city}_${route}_final_result.jpg" "${FAIR_DIR}/"
  done
done

python3 - "${FAIR_DIR}" "${OFFICIAL_COMMIT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
payload = {
  "rerun_same_route_methods": [
    "Ours v39 Bearing-adapted",
    "Bearing-UAV official VGG-16"
  ],
  "published_reference_only_methods": [
    "University-1652", "SUES-200", "DenseUAV", "GTA-UAV",
    "Bearing-UAV VGG-16 published full-benchmark row",
    "Bearing-UAV VGG-16 + weather augmentation published row"
  ],
  "official_bearinguav_commit": sys.argv[2],
  "gpu_schedule": [0, 5, 6],
  "cpu_policy": "2 BLAS/OpenMP threads per process; 2 DataLoader workers per official process; max 3 concurrent GPU jobs",
  "comparison_rule": (
    "Use same_route_pooled.csv for the strongest direct experimental comparison. "
    "Use bearinguav_published_uav_reference.csv only as cited published reference; "
    "do not describe University-1652/SUES-200/DenseUAV/GTA-UAV rows as rerun results."
  )
}
(root / "comparison_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

echo ""
echo "================================================================================"
echo "FAIR COMPARISON COMPLETE"
echo "Same-route direct table : ${FAIR_DIR}/same_route_pooled.csv"
echo "Per-route direct table  : ${FAIR_DIR}/same_route_route_level.csv"
echo "Published references    : ${FAIR_DIR}/bearinguav_published_uav_reference.csv"
echo "Bearing backbone refs   : ${FAIR_DIR}/bearinguav_backbone_supplement.csv"
echo "Paper figures           : ${FAIR_DIR}/city*_test_0*_final_result.jpg"
echo "Manifest / caveats      : ${FAIR_DIR}/comparison_manifest.json"
echo "Logs                    : ${LOG_DIR}/"
echo "================================================================================"
