#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CPU_THREADS="${CPU_THREADS:-2}"
GPU_IDS=(0 5 6)
FORCE_OURS="${FORCE_OURS:-0}"
FORCE_BASELINES="${FORCE_BASELINES:-1}"
FORCE_OFFICIAL="${FORCE_OFFICIAL:-0}"
cd "$ROOT"

GEN="$ROOT/v39_otherdata/generated"
EXT="$ROOT/v39_otherdata/external"
CACHE="$GEN/native_four_rst_cache"
BASE="$GEN/native_four_rst_baselines"
OFF="$GEN/official_bearinguav_same_routes"
OUT="$GEN/native_four_rst_comparison"
LOG="$GEN/native_four_rst_logs"
VENV="$EXT/native_baseline_env"
BEARING="$EXT/bearinguav"
UNI="$EXT/University1652-Baseline"
SUES="$EXT/SUES-200-Benchmark"
DENSE="$EXT/DenseUAV"
GTA="$EXT/GTA-UAV"

BEARING_COMMIT="d16e558a14bd0a9142c6901793fe6a6813e4e954"
UNI_COMMIT="c81e4b5c76ba29882a7b1e06b91f3c05ff41dc88"
SUES_COMMIT="85b7488d06db1e6c4c1c52d4a5dc55553f768fb0"
DENSE_COMMIT="b8de18751675fcc0a7a40938b4cd66549647d690"
GTA_COMMIT="1fbc3dbe452db75e82539474f6e2c79d54ad52ac"

mkdir -p "$GEN" "$EXT" "$LOG" "$OUT"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS"
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EXT/torch_cache" HF_HOME="$EXT/hf_cache"
mkdir -p "$TORCH_HOME" "$HF_HOME"

for g in "${GPU_IDS[@]}"; do nvidia-smi -i "$g" >/dev/null 2>&1 || { echo "GPU $g unavailable" >&2; exit 2; }; done

clone_pin(){
  local url="$1" dir="$2" sha="$3"
  if [[ ! -d "$dir/.git" ]]; then git clone --filter=blob:none "$url" "$dir"; fi
  if [[ "$(git -C "$dir" rev-parse HEAD)" != "$sha" ]]; then
    git -C "$dir" fetch --depth 1 origin "$sha"
    git -C "$dir" checkout --detach "$sha"
  fi
}
clone_pin https://github.com/liukejia121/bearinguav.git "$BEARING" "$BEARING_COMMIT"
clone_pin https://github.com/layumi/University1652-Baseline.git "$UNI" "$UNI_COMMIT"
clone_pin https://github.com/Reza-Zhu/SUES-200-Benchmark.git "$SUES" "$SUES_COMMIT"
clone_pin https://github.com/Dmmm1997/DenseUAV.git "$DENSE" "$DENSE_COMMIT"
clone_pin https://github.com/Yux1angJi/GTA-UAV.git "$GTA" "$GTA_COMMIT"

if [[ ! -x "$VENV/bin/python" ]]; then python3 -m venv --system-site-packages "$VENV"; fi
PY="$VENV/bin/python"
if ! "$PY" - <<'PY' >/dev/null 2>&1
from packaging.version import Version
import torch,torchvision,timm,pytorch_metric_learning,transformers,einops,yaml,scipy
assert Version(timm.__version__) >= Version('1.0.7')
PY
then
  PIP_DISABLE_PIP_VERSION_CHECK=1 "$PY" -m pip install -q --upgrade \
    'timm>=1.0.7,<2' pytorch-metric-learning transformers einops pyyaml scipy tqdm thop yacs omegaconf packaging
fi
if ! "$PY" - <<'PY' >/dev/null 2>&1
import cv2,pandas,albumentations,imgaug,skimage,imageio
PY
then
  PIP_DISABLE_PIP_VERSION_CHECK=1 "$PY" -m pip install -q opencv-python-headless pandas albumentations imgaug scikit-image imageio
fi

"$PY" -m py_compile \
  v39_otherdata/bearing_four_rst_route_cache.py \
  v39_otherdata/bearing_four_rst_baseline_runner.py \
  v39_otherdata/bearinguav_official_paper_eval.py \
  v39_otherdata/bearing_route_protocol_audit.py \
  v39_otherdata/bearing_native_comparison_v2.py \
  v39_otherdata/bearinguav_official_route_eval.py

echo "================================================================================"
echo "[PROTOCOL] Bearing-UAV paper/native comparison"
echo "[PROTOCOL] M2T Recall@1 candidate set = FOUR adjacent RSTs (p1,p2,p3,p4), NOT whole-city 256 tiles."
echo "[PROTOCOL] M2T prediction = retrieved RST center. No waypoint/temporal/local prior."
echo "[PROTOCOL] Navigation routes = official 8 Bearing-UAV routes (524..1119m)."
echo "[PROTOCOL] Paper navigation step=25m; waypoint-arrival threshold=20m."
echo "[RESOURCE] GPUs=${GPU_IDS[*]} CPU_THREADS/job=$CPU_THREADS; route JPEGs decoded once into mmap cache."
echo "================================================================================"

# ---------------------------------------------------------------------------
# 1) OUR METHOD: keep current four-city results unless explicitly forced/missing.
# ---------------------------------------------------------------------------
ours_ok(){ local d="$GEN/$1/v39_output_bearing_adapted"; [[ -s "$d/bearing_v39_summary.json" && -s "$d/bearing_paper_metrics.json" ]]; }
need_ours=0
for c in citya cityb cityc cityd; do ours_ok "$c" || need_ours=1; done
if [[ "$FORCE_OURS" == 1 || "$need_ours" == 1 ]]; then
  echo "[OURS] rebuilding four cities"
  DATASET_ROOT="$DATASET_ROOT" GPU=0 bash v39_otherdata/run_bearing_v39_all_cities.sh
else
  echo "[OURS] cache hit: all four cities"
fi
for c in citya cityb cityc cityd; do ours_ok "$c" || { echo "missing ours/$c" >&2; exit 3; }; done
RUN_MODEL=0 DATASET_ROOT="$DATASET_ROOT" GPU=0 bash v39_otherdata/run_bearing_paper_bundle.sh > >(tee "$LOG/ours_paper_bundle.log") 2>&1

# ---------------------------------------------------------------------------
# 2) NATIVE FOUR-RST CACHE. This is the critical correction.
# ---------------------------------------------------------------------------
CACHE_FORCE=(); [[ "$FORCE_BASELINES" == 1 ]] && CACHE_FORCE+=(--force)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" v39_otherdata/bearing_four_rst_route_cache.py \
  --dataset-root "$DATASET_ROOT" --generated-root "$GEN" --cache-root "$CACHE" \
  --cities citya cityb cityc cityd "${CACHE_FORCE[@]}" > >(tee "$LOG/four_rst_cache.log") 2>&1

"$PY" v39_otherdata/bearing_route_protocol_audit.py \
  --generated-root "$GEN" --cache-root "$CACHE" --output "$OUT/route_protocol_audit.json" \
  > >(tee "$LOG/route_protocol_audit.log") 2>&1

# ---------------------------------------------------------------------------
# 3) UNIVERSITY-1652 / SUES-200 / DenseUAV / GTA-UAV.
#    Each test UAV sees ONLY its official p1..p4 RSTs.
# ---------------------------------------------------------------------------
declare -A RDIR RCOM
RDIR[university1652]="$UNI";  RCOM[university1652]="$UNI_COMMIT"
RDIR[sues200]="$SUES";        RCOM[sues200]="$SUES_COMMIT"
RDIR[denseuav]="$DENSE";      RCOM[denseuav]="$DENSE_COMMIT"
RDIR[gtauav]="$GTA";          RCOM[gtauav]="$GTA_COMMIT"

base_ok(){
  local d="$BASE/$1/$2"
  [[ -s "$d/result.json" && -s "$d/test_01_final_result.jpg" && -s "$d/test_02_final_result.jpg" ]] \
    && grep -q '"candidate_scope": "four adjacent p1/p2/p3/p4 RSTs from official metadata"' "$d/result.json"
}
run_base(){
  local m="$1" c="$2" g="$3"; local ff=(); [[ "$FORCE_BASELINES" == 1 ]] && ff+=(--force)
  echo "[4RST] START $m/$c GPU$g"
  CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" \
  "$PY" v39_otherdata/bearing_four_rst_baseline_runner.py \
    --method "$m" --repo-dir "${RDIR[$m]}" --repo-commit "${RCOM[$m]}" \
    --cache-dir "$CACHE/$c" --prepared-root "$GEN/$c" --output-root "$BASE" \
    --city "$c" --cpu-threads "$CPU_THREADS" "${ff[@]}" \
    > >(tee "$LOG/${m}_${c}.log") 2>&1
  echo "[4RST] DONE $m/$c"
}

tasks=()
for m in university1652 sues200 denseuav gtauav; do
  for c in citya cityb cityc cityd; do
    if [[ "$FORCE_BASELINES" == 1 ]] || ! base_ok "$m" "$c"; then tasks+=("$m:$c"); else echo "[4RST] cache hit $m/$c"; fi
  done
done
for ((b=0;b<${#tasks[@]};b+=3)); do
  pids=(); names=()
  for slot in 0 1 2; do
    idx=$((b+slot)); ((idx<${#tasks[@]})) || break
    IFS=: read -r m c <<<"${tasks[$idx]}"
    run_base "$m" "$c" "${GPU_IDS[$slot]}" & pids+=("$!"); names+=("$m/$c")
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "[4RST] FAILED ${names[$i]} -- see $LOG" >&2; exit 4; }; done
done
for m in university1652 sues200 denseuav gtauav; do for c in citya cityb cityc cityd; do base_ok "$m" "$c" || exit 5; done; done

# ---------------------------------------------------------------------------
# 4) BEARING-UAV AUTHORS' OFFICIAL VGG-16 CHECKPOINT.
# ---------------------------------------------------------------------------
WEIGHT_ZIP="$EXT/Bearing_UAV.zip"; WEIGHT_DIR="$BEARING/Bearing_UAV/cross_view"
if [[ ! -s "$WEIGHT_DIR/best_model.pth" || ! -s "$WEIGHT_DIR/training_configure.json" ]]; then
  tmp="$WEIGHT_ZIP.part"; rm -f "$tmp"; URL='https://huggingface.co/HaoyZhou/bearinguav/resolve/main/Bearing_UAV.zip?download=true'
  if command -v curl >/dev/null 2>&1; then curl -L --fail --retry 4 --retry-delay 3 -o "$tmp" "$URL"; else wget -O "$tmp" "$URL"; fi
  mv "$tmp" "$WEIGHT_ZIP"
  "$PY" - "$WEIGHT_ZIP" "$BEARING" <<'PY'
import sys,zipfile
with zipfile.ZipFile(sys.argv[1]) as z:z.extractall(sys.argv[2])
PY
fi

off_ok(){ local d="$OFF/$1"; [[ -s "$d/official_bearinguav_same_route.json" && -s "$d/test_01_final_result.jpg" && -s "$d/test_02_final_result.jpg" ]]; }
run_off(){
  local c="$1" g="$2"
  echo "[BEARING-OFFICIAL] START $c GPU$g"
  CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" \
  "$PY" v39_otherdata/bearinguav_official_route_eval.py \
    --official-root "$BEARING" --weights-dir "$WEIGHT_DIR" --dataset-root "$DATASET_ROOT" \
    --generated-root "$GEN" --city "$c" --output-root "$OFF" --workers 1 --batch-size 8 --cpu-threads "$CPU_THREADS" \
    > >(tee "$LOG/bearinguav_route_${c}.log") 2>&1
}
pending=(); for c in citya cityb cityc cityd; do if [[ "$FORCE_OFFICIAL" == 1 ]] || ! off_ok "$c"; then pending+=("$c"); fi; done
for ((b=0;b<${#pending[@]};b+=3)); do
  pids=(); names=()
  for slot in 0 1 2; do idx=$((b+slot)); ((idx<${#pending[@]})) || break; c="${pending[$idx]}"; run_off "$c" "${GPU_IDS[$slot]}" & pids+=("$!"); names+=("$c"); done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "official Bearing failed ${names[$i]}" >&2; exit 6; }; done
done
for c in citya cityb cityc cityd; do off_ok "$c" || exit 7; done

# Paper-protocol verification is deliberately separate from route evaluation.
# It uses the full metadata 85/5/10 split with seed=42, as in the released code.
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" \
"$PY" v39_otherdata/bearinguav_official_paper_eval.py \
  --official-root "$BEARING" --weights-dir "$WEIGHT_DIR" --dataset-root "$DATASET_ROOT" \
  --output "$OUT/bearinguav_official_paper_protocol.json" --workers 1 --batch-size 16 --cpu-threads "$CPU_THREADS" \
  > >(tee "$LOG/bearinguav_paper_verify.log") 2>&1

# ---------------------------------------------------------------------------
# 5) TABLES + HIGH-CONTRAST FIGURES.
# ---------------------------------------------------------------------------
rm -rf "$OUT/figures"
mkdir -p "$OUT/figures/ours" "$OUT/figures/bearinguav_official" \
  "$OUT/figures/university1652" "$OUT/figures/sues200" "$OUT/figures/denseuav" "$OUT/figures/gtauav"

"$PY" v39_otherdata/bearing_native_comparison_v2.py \
  --generated-root "$GEN" --baseline-root "$BASE" --official-root "$OFF" --output-dir "$OUT"
"$PY" v39_otherdata/bearing_published_reference.py --generated-root "$GEN" --output-dir "$OUT"

for c in citya cityb cityc cityd; do
  for r in test_01 test_02; do
    cp "$GEN/paper_bundle/${c}_${r}_final_result.jpg" "$OUT/figures/ours/${c}_${r}_final_result.jpg"
    cp "$OFF/$c/${r}_final_result.jpg" "$OUT/figures/bearinguav_official/${c}_${r}_final_result.jpg"
    for m in university1652 sues200 denseuav gtauav; do
      cp "$BASE/$m/$c/${r}_final_result.jpg" "$OUT/figures/$m/${c}_${r}_final_result.jpg"
    done
  done
done

for method in ours bearinguav_official university1652 sues200 denseuav gtauav; do
  count=$(find "$OUT/figures/$method" -maxdepth 1 -type f -name '*_final_result.jpg' | wc -l)
  [[ "$count" -eq 8 ]] || { echo "figure audit failed $method: $count/8" >&2; exit 8; }
done
for f in same_routes_pooled.csv same_routes_route_level.csv bearinguav_published_uav_reference.csv comparison_manifest.json route_protocol_audit.json bearinguav_official_paper_protocol.json; do
  test -s "$OUT/$f" || { echo "missing artifact $f" >&2; exit 9; }
done

cat > "$OUT/README_FINAL.txt" <<'EOF'
CORRECTED PROTOCOL
==================
1. The 8 test routes are exactly the official Bearing-UAV navigation waypoint routes.
   Official lengths: 524..1119 m; navigation step=25 m; waypoint arrival threshold=20 m.
2. University-1652 / SUES-200 / DenseUAV / GTA-UAV route evaluation uses ONLY
   the four adjacent p1/p2/p3/p4 RSTs for each UAV frame. Prediction is the
   retrieved tile centre. They receive no waypoint, route, previous-frame,
   temporal, Kalman, or v39 local prior.
3. Bearing-UAV official uses the authors' released VGG-16 checkpoint and native
   four-RST pose regression.
4. Ours retains its own temporal controlled-local-prior protocol.
5. Published Table values are kept separately. Route-selected results are not
   expected to exactly equal the full static localization benchmark.
6. bearinguav_official_paper_protocol.json verifies the official checkpoint on
   the released code's full-metadata 85/5/10 seed-42 test protocol.

KEY FILES
=========
same_routes_pooled.csv
same_routes_route_level.csv
bearinguav_published_uav_reference.csv
bearinguav_official_paper_protocol.json
route_protocol_audit.json
figures/
EOF

echo "================================================================================"
echo "CORRECTED BEARING ALL-METHOD BUNDLE: PASS"
echo "Route comparison : $OUT/same_routes_pooled.csv"
echo "Paper references : $OUT/bearinguav_published_uav_reference.csv"
echo "Paper verification: $OUT/bearinguav_official_paper_protocol.json"
echo "Route audit      : $OUT/route_protocol_audit.json"
echo "Figures          : $OUT/figures/"
echo "Logs             : $LOG/"
echo "================================================================================"
