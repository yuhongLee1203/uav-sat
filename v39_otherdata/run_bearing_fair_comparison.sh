#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
GPU_IDS=(0 5 6)
CPU_THREADS="${CPU_THREADS:-2}"
FORCE_OURS="${FORCE_OURS:-0}"
FORCE_OFFICIAL="${FORCE_OFFICIAL:-0}"
FORCE_BASELINES="${FORCE_BASELINES:-0}"

GEN_ROOT="${REPO_ROOT}/v39_otherdata/generated"
EXT_ROOT="${REPO_ROOT}/v39_otherdata/external"
CACHE_ROOT="${GEN_ROOT}/native_m2t_cache"
BASE_ROOT="${GEN_ROOT}/native_m2t_baselines"
OFF_RESULT="${GEN_ROOT}/official_bearinguav_same_routes"
OUT="${GEN_ROOT}/native_comparison"
LOG="${GEN_ROOT}/native_comparison_logs"
VENV="${EXT_ROOT}/native_baseline_env"

BEARING_REPO="${EXT_ROOT}/bearinguav"
UNI_REPO="${EXT_ROOT}/University1652-Baseline"
SUES_REPO="${EXT_ROOT}/SUES-200-Benchmark"
DENSE_REPO="${EXT_ROOT}/DenseUAV"
GTA_REPO="${EXT_ROOT}/GTA-UAV"

BEARING_COMMIT="d16e558a14bd0a9142c6901793fe6a6813e4e954"
UNI_COMMIT="c81e4b5c76ba29882a7b1e06b91f3c05ff41dc88"
SUES_COMMIT="85b7488d06db1e6c4c1c52d4a5dc55553f768fb0"
DENSE_COMMIT="b8de18751675fcc0a7a40938b4cd66549647d690"
GTA_COMMIT="1fbc3dbe452db75e82539474f6e2c79d54ad52ac"

cd "${REPO_ROOT}"
mkdir -p "${GEN_ROOT}" "${EXT_ROOT}" "${LOG}"
export OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" OPENBLAS_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME="${EXT_ROOT}/torch_cache"
export HF_HOME="${EXT_ROOT}/hf_cache"
mkdir -p "${TORCH_HOME}" "${HF_HOME}"

for g in "${GPU_IDS[@]}"; do nvidia-smi -i "$g" >/dev/null 2>&1 || { echo "GPU $g unavailable" >&2; exit 2; }; done
python3 -m py_compile \
  v39_otherdata/bearing_route_baseline_cache.py \
  v39_otherdata/bearing_route_baseline_runner.py \
  v39_otherdata/bearing_native_comparison.py \
  v39_otherdata/bearing_plot_final_vs_gt.py \
  v39_otherdata/bearinguav_official_route_eval.py

echo "[PROTOCOL] Public M2T baselines receive NO waypoint/local/temporal prior."
echo "[PROTOCOL] Test search space = all 16x16 = 256 fixed RST tiles of the city."
echo "[PROTOCOL] A baseline may drift/jump/miss endpoint; no correction is applied."
echo "[RESOURCE] GPUs=${GPU_IDS[*]} CPU_THREADS/job=${CPU_THREADS}; JPEG cache built once then mmap."

clone_pin(){
  local url="$1" dir="$2" sha="$3"
  if [[ ! -d "$dir/.git" ]]; then git clone --filter=blob:none "$url" "$dir"; fi
  if [[ "$(git -C "$dir" rev-parse HEAD)" != "$sha" ]]; then
    git -C "$dir" fetch --depth 1 origin "$sha"
    git -C "$dir" checkout --detach "$sha"
  fi
}
clone_pin https://github.com/liukejia121/bearinguav.git "$BEARING_REPO" "$BEARING_COMMIT"
clone_pin https://github.com/layumi/University1652-Baseline.git "$UNI_REPO" "$UNI_COMMIT"
clone_pin https://github.com/Reza-Zhu/SUES-200-Benchmark.git "$SUES_REPO" "$SUES_COMMIT"
clone_pin https://github.com/Dmmm1997/DenseUAV.git "$DENSE_REPO" "$DENSE_COMMIT"
clone_pin https://github.com/Yux1angJi/GTA-UAV.git "$GTA_REPO" "$GTA_COMMIT"

if [[ ! -x "${VENV}/bin/python" ]]; then python3 -m venv --system-site-packages "$VENV"; fi
PY="${VENV}/bin/python"
if ! "$PY" - <<'PY' >/dev/null 2>&1
from packaging.version import Version
import torch,torchvision,timm,pytorch_metric_learning,transformers,einops,yaml,scipy
assert Version(timm.__version__) >= Version("1.0.7")
PY
then
  PIP_DISABLE_PIP_VERSION_CHECK=1 "$PY" -m pip install -q --upgrade \
    'timm>=1.0.7,<2' pytorch-metric-learning transformers einops pyyaml scipy tqdm thop yacs omegaconf packaging
fi

# 1) OUR METHOD ---------------------------------------------------------------
ours_ok(){ local c="$1" o="${GEN_ROOT}/${c}/v39_output_bearing_adapted"; [[ -s "$o/bearing_v39_summary.json" && -s "$o/bearing_paper_metrics.json" ]]; }
run_ours(){
  local c="$1" g="$2"
  echo "[OURS] START $c GPU$g"
  CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" \
    DATASET_ROOT="$DATASET_ROOT" CITY="$c" GPU=0 \
    bash v39_otherdata/run_bearing_v39_sequence_fixed.sh > >(tee "$LOG/ours_${c}.log") 2>&1
}
pending=(); for c in citya cityb cityc cityd; do if [[ "$FORCE_OURS" == 1 ]] || ! ours_ok "$c"; then pending+=("$c"); else echo "[OURS] cache hit $c"; fi; done
for ((b=0;b<${#pending[@]};b+=3)); do
  p=(); n=()
  for s in 0 1 2; do i=$((b+s)); ((i<${#pending[@]})) || break; c="${pending[$i]}"; run_ours "$c" "${GPU_IDS[$s]}" & p+=("$!"); n+=("$c"); done
  for i in "${!p[@]}"; do wait "${p[$i]}" || { echo "OURS failed ${n[$i]}" >&2; exit 3; }; done
done
for c in citya cityb cityc cityd; do ours_ok "$c" || exit 4; done
RUN_MODEL=0 DATASET_ROOT="$DATASET_ROOT" GPU=0 bash v39_otherdata/run_bearing_paper_bundle.sh > >(tee "$LOG/paper_bundle.log") 2>&1

# 2) NATIVE FULL-CITY MATCHING-TO-TILE CACHE ----------------------------------
CACHE_ARGS=(); [[ "$FORCE_BASELINES" == 1 ]] && CACHE_ARGS+=(--force)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" v39_otherdata/bearing_route_baseline_cache.py \
  --generated-root "$GEN_ROOT" --cache-root "$CACHE_ROOT" \
  --cities citya cityb cityc cityd "${CACHE_ARGS[@]}" > >(tee "$LOG/m2t_cache.log") 2>&1

# 3) FOUR PUBLIC RETRIEVAL METHODS --------------------------------------------
declare -A RDIR RCOM
RDIR[university1652]="$UNI_REPO"; RCOM[university1652]="$UNI_COMMIT"
RDIR[sues200]="$SUES_REPO";       RCOM[sues200]="$SUES_COMMIT"
RDIR[denseuav]="$DENSE_REPO";    RCOM[denseuav]="$DENSE_COMMIT"
RDIR[gtauav]="$GTA_REPO";        RCOM[gtauav]="$GTA_COMMIT"
base_ok(){
  local m="$1" c="$2" d="$BASE_ROOT/$m/$c"
  [[ -s "$d/result.json" && -s "$d/test_01_final_result.jpg" && -s "$d/test_02_final_result.jpg" ]] \
    && grep -q '"uses_waypoint_or_route_prior": false' "$d/result.json" \
    && grep -q '16x16=256' "$d/result.json"
}
run_base(){
  local m="$1" c="$2" g="$3"; local f=(); [[ "$FORCE_BASELINES" == 1 ]] && f+=(--force)
  echo "[M2T] START $m/$c GPU$g"
  CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" \
  "$PY" v39_otherdata/bearing_route_baseline_runner.py \
    --method "$m" --repo-dir "${RDIR[$m]}" --repo-commit "${RCOM[$m]}" \
    --cache-dir "$CACHE_ROOT/$c" --prepared-root "$GEN_ROOT/$c" \
    --output-root "$BASE_ROOT" --city "$c" --cpu-threads "$CPU_THREADS" "${f[@]}" \
    > >(tee "$LOG/${m}_${c}.log") 2>&1
  echo "[M2T] DONE $m/$c"
}
tasks=(); for m in university1652 sues200 denseuav gtauav; do for c in citya cityb cityc cityd; do if [[ "$FORCE_BASELINES" == 1 ]] || ! base_ok "$m" "$c"; then tasks+=("$m:$c"); else echo "[M2T] cache hit $m/$c"; fi; done; done
for ((b=0;b<${#tasks[@]};b+=3)); do
  p=(); n=()
  for s in 0 1 2; do i=$((b+s)); ((i<${#tasks[@]})) || break; IFS=: read -r m c <<<"${tasks[$i]}"; run_base "$m" "$c" "${GPU_IDS[$s]}" & p+=("$!"); n+=("$m/$c"); done
  for i in "${!p[@]}"; do wait "${p[$i]}" || { echo "M2T failed ${n[$i]} -- see $LOG" >&2; exit 5; }; done
done
for m in university1652 sues200 denseuav gtauav; do for c in citya cityb cityc cityd; do base_ok "$m" "$c" || { echo "missing/native-protocol mismatch $m/$c" >&2; exit 6; }; done; done

# 4) BEARING-UAV AUTHORS' OWN METHOD ------------------------------------------
WEIGHT_ZIP="${EXT_ROOT}/Bearing_UAV.zip"; WEIGHT_DIR="${BEARING_REPO}/Bearing_UAV/cross_view"
if [[ ! -s "$WEIGHT_DIR/best_model.pth" || ! -s "$WEIGHT_DIR/training_configure.json" ]]; then
  tmp="${WEIGHT_ZIP}.part"; rm -f "$tmp"; URL='https://huggingface.co/HaoyZhou/bearinguav/resolve/main/Bearing_UAV.zip?download=true'
  if command -v curl >/dev/null 2>&1; then curl -L --fail --retry 4 --retry-delay 3 -o "$tmp" "$URL"; elif command -v wget >/dev/null 2>&1; then wget -O "$tmp" "$URL"; else echo "curl or wget required" >&2; exit 7; fi
  mv "$tmp" "$WEIGHT_ZIP"
  "$PY" - "$WEIGHT_ZIP" "$BEARING_REPO" <<'PY'
import sys,zipfile
with zipfile.ZipFile(sys.argv[1]) as z:z.extractall(sys.argv[2])
PY
fi
if ! "$PY" - <<'PY' >/dev/null 2>&1
import cv2,pandas,albumentations,imgaug,skimage,imageio
PY
then
  PIP_DISABLE_PIP_VERSION_CHECK=1 "$PY" -m pip install -q opencv-python-headless pandas albumentations imgaug scikit-image imageio
fi
off_ok(){ local d="$OFF_RESULT/$1"; [[ -s "$d/official_bearinguav_same_route.json" && -s "$d/test_01_final_result.jpg" && -s "$d/test_02_final_result.jpg" ]]; }
run_off(){
  local c="$1" g="$2"; echo "[BEARING] START $c GPU$g"
  CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" \
  "$PY" v39_otherdata/bearinguav_official_route_eval.py \
    --official-root "$BEARING_REPO" --weights-dir "$WEIGHT_DIR" \
    --dataset-root "$DATASET_ROOT" --generated-root "$GEN_ROOT" --city "$c" \
    --output-root "$OFF_RESULT" --workers 1 --batch-size 8 --cpu-threads "$CPU_THREADS" \
    > >(tee "$LOG/bearinguav_${c}.log") 2>&1
}
pending=(); for c in citya cityb cityc cityd; do if [[ "$FORCE_OFFICIAL" == 1 ]] || ! off_ok "$c"; then pending+=("$c"); else echo "[BEARING] cache hit $c"; fi; done
for ((b=0;b<${#pending[@]};b+=3)); do
  p=(); n=()
  for s in 0 1 2; do i=$((b+s)); ((i<${#pending[@]})) || break; c="${pending[$i]}"; run_off "$c" "${GPU_IDS[$s]}" & p+=("$!"); n+=("$c"); done
  for i in "${!p[@]}"; do wait "${p[$i]}" || { echo "Bearing failed ${n[$i]}" >&2; exit 8; }; done
done
for c in citya cityb cityc cityd; do off_ok "$c" || exit 9; done

# 5) PAPER TABLES + ALL RESULT FIGURES ----------------------------------------
rm -rf "$OUT"
mkdir -p "$OUT/figures/ours" "$OUT/figures/bearinguav_official" \
  "$OUT/figures/university1652" "$OUT/figures/sues200" "$OUT/figures/denseuav" "$OUT/figures/gtauav"
"$PY" v39_otherdata/bearing_native_comparison.py \
  --generated-root "$GEN_ROOT" --baseline-root "$BASE_ROOT" \
  --official-root "$OFF_RESULT" --output-dir "$OUT"
"$PY" v39_otherdata/bearing_published_reference.py --generated-root "$GEN_ROOT" --output-dir "$OUT"
for c in citya cityb cityc cityd; do
  for r in test_01 test_02; do
    cp "${GEN_ROOT}/paper_bundle/${c}_${r}_final_result.jpg" "$OUT/figures/ours/${c}_${r}_final_result.jpg"
    cp "$OFF_RESULT/$c/${r}_final_result.jpg" "$OUT/figures/bearinguav_official/${c}_${r}_final_result.jpg"
    for m in university1652 sues200 denseuav gtauav; do
      cp "$BASE_ROOT/$m/$c/${r}_final_result.jpg" "$OUT/figures/$m/${c}_${r}_final_result.jpg"
    done
  done
done
for method in ours bearinguav_official university1652 sues200 denseuav gtauav; do
  count=$(find "$OUT/figures/$method" -maxdepth 1 -type f -name '*_final_result.jpg' | wc -l)
  [[ "$count" -eq 8 ]] || { echo "figure audit failed $method: $count/8" >&2; exit 10; }
done
cat > "$OUT/README.txt" <<'EOF'
RERUN TABLES:
  same_frames_pooled.csv
  same_frames_route_level.csv

University-1652 / SUES-200 / DenseUAV / GTA-UAV:
  training UAV = selected train_01 observations
  positive satellite = fixed 256px RST tile containing the observation
  test gallery = ALL 16x16=256 fixed RST tiles of the city
  NO waypoint / route centerline / temporal history / previous position / local prior
  drift, jumps and endpoint miss are retained as raw method output

Bearing-UAV official:
  authors' official VGG-16 checkpoint and native four-neighbour RST pose-regression input
  same selected test UAV frames; no v39 prior

Ours:
  own temporal controlled-local-prior protocol

Published paper rows are stored separately in bearinguav_published_uav_reference.csv.
They are cited references, not rerun results.
EOF

echo "================================================================================"
echo "ALL METHODS COMPLETE"
echo "Direct rerun table : $OUT/same_frames_pooled.csv"
echo "Per-route table    : $OUT/same_frames_route_level.csv"
echo "Published refs     : $OUT/bearinguav_published_uav_reference.csv"
echo "Figures            : $OUT/figures/"
echo "Logs               : $LOG/"
echo "================================================================================"
