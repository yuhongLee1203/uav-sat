#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CPU_THREADS="${CPU_THREADS:-2}"
GPU_IDS=(0 5 6)
GEN="$ROOT/v39_otherdata/generated"
EXT="$ROOT/v39_otherdata/external"
VENV="$EXT/native_baseline_env"
FULL_CACHE="$GEN/full_benchmark_cache"
ROUTE_CACHE="$GEN/native_four_rst_cache"
BASE="$GEN/full_benchmark_baselines"
OFF="$GEN/official_bearinguav_corrected_routes"
OUT="$GEN/corrected_full_comparison"
LOG="$GEN/corrected_full_logs"
mkdir -p "$GEN" "$EXT" "$BASE" "$OFF" "$OUT" "$LOG"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS"
export TOKENIZERS_PARALLELISM=false TORCH_HOME="$EXT/torch_cache" HF_HOME="$EXT/hf_cache"
mkdir -p "$TORCH_HOME" "$HF_HOME"
for g in "${GPU_IDS[@]}"; do nvidia-smi -i "$g" >/dev/null 2>&1 || { echo "GPU $g unavailable" >&2; exit 2; }; done

BEARING="$EXT/bearinguav"; UNI="$EXT/University1652-Baseline"; SUES="$EXT/SUES-200-Benchmark"; DENSE="$EXT/DenseUAV"; GTA="$EXT/GTA-UAV"
BEARING_COMMIT="d16e558a14bd0a9142c6901793fe6a6813e4e954"
UNI_COMMIT="c81e4b5c76ba29882a7b1e06b91f3c05ff41dc88"
SUES_COMMIT="85b7488d06db1e6c4c1c52d4a5dc55553f768fb0"
DENSE_COMMIT="b8de18751675fcc0a7a40938b4cd66549647d690"
GTA_COMMIT="1fbc3dbe452db75e82539474f6e2c79d54ad52ac"
clone_pin(){ local url="$1" dir="$2" sha="$3"; if [[ ! -d "$dir/.git" ]]; then git clone --filter=blob:none "$url" "$dir"; fi; if [[ "$(git -C "$dir" rev-parse HEAD)" != "$sha" ]]; then git -C "$dir" fetch --depth 1 origin "$sha"; git -C "$dir" checkout --detach "$sha"; fi; }
clone_pin https://github.com/liukejia121/bearinguav.git "$BEARING" "$BEARING_COMMIT"
clone_pin https://github.com/layumi/University1652-Baseline.git "$UNI" "$UNI_COMMIT"
clone_pin https://github.com/Reza-Zhu/SUES-200-Benchmark.git "$SUES" "$SUES_COMMIT"
clone_pin https://github.com/Dmmm1997/DenseUAV.git "$DENSE" "$DENSE_COMMIT"
clone_pin https://github.com/Yux1angJi/GTA-UAV.git "$GTA" "$GTA_COMMIT"

if [[ ! -x "$VENV/bin/python" ]]; then python3 -m venv --system-site-packages "$VENV"; fi
PY="$VENV/bin/python"
if ! "$PY" - <<'PY' >/dev/null 2>&1
from packaging.version import Version
import torch,torchvision,timm,pytorch_metric_learning,transformers,einops,yaml,scipy,pandas,PIL
assert Version(timm.__version__)>=Version('1.0.7')
PY
then
 PIP_DISABLE_PIP_VERSION_CHECK=1 "$PY" -m pip install -q --upgrade 'timm>=1.0.7,<2' pytorch-metric-learning transformers einops pyyaml scipy tqdm thop yacs omegaconf packaging pandas pillow
fi

"$PY" -m py_compile v39_otherdata/bearing_full_benchmark_cache.py v39_otherdata/bearing_full_benchmark_baseline_runner.py v39_otherdata/bearinguav_official_paper_eval.py v39_otherdata/bearinguav_official_route_eval.py v39_otherdata/bearing_full_benchmark_comparison.py

# Keep the user's v39 four-city result; rebuild only when missing or explicitly requested.
need_ours=0
for c in citya cityb cityc cityd; do [[ -s "$GEN/$c/v39_output_bearing_adapted/bearing_paper_metrics.json" ]] || need_ours=1; done
if [[ "${FORCE_OURS:-0}" == 1 || "$need_ours" == 1 ]]; then DATASET_ROOT="$DATASET_ROOT" GPU=0 bash v39_otherdata/run_bearing_v39_all_cities.sh; fi
RUN_MODEL=0 DATASET_ROOT="$DATASET_ROOT" GPU=0 bash v39_otherdata/run_bearing_paper_bundle.sh > >(tee "$LOG/ours_bundle.log") 2>&1

# Route cache is only for evaluating the full-trained baselines on the same 8 official routes.
"$PY" v39_otherdata/bearing_four_rst_route_cache.py --dataset-root "$DATASET_ROOT" --generated-root "$GEN" --cache-root "$ROUTE_CACHE" --cities citya cityb cityc cityd > >(tee "$LOG/route_cache.log") 2>&1

# One expensive decode pass for the full 90k benchmark, then all methods mmap it.
CACHE_FORCE=(); [[ "${FORCE_FULL_CACHE:-0}" == 1 ]] && CACHE_FORCE+=(--force)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" v39_otherdata/bearing_full_benchmark_cache.py --dataset-root "$DATASET_ROOT" --output-dir "$FULL_CACHE" "${CACHE_FORCE[@]}" > >(tee "$LOG/full_cache.log") 2>&1

# Public baselines trained on the SAME official 85% Bearing-UAV split.
declare -A RDIR RCOM
RDIR[university1652]="$UNI"; RCOM[university1652]="$UNI_COMMIT"
RDIR[sues200]="$SUES"; RCOM[sues200]="$SUES_COMMIT"
RDIR[denseuav]="$DENSE"; RCOM[denseuav]="$DENSE_COMMIT"
RDIR[gtauav]="$GTA"; RCOM[gtauav]="$GTA_COMMIT"
run_method(){ local m="$1" g="$2"; local force=(); [[ "${FORCE_BASELINES:-0}" == 1 ]] && force+=(--force); echo "[FULL-BASELINE] START $m GPU$g"; CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" "$PY" v39_otherdata/bearing_full_benchmark_baseline_runner.py --method "$m" --repo-dir "${RDIR[$m]}" --repo-commit "${RCOM[$m]}" --full-cache "$FULL_CACHE" --route-cache-root "$ROUTE_CACHE" --generated-root "$GEN" --output-root "$BASE" --cpu-threads "$CPU_THREADS" "${force[@]}" > >(tee "$LOG/${m}.log") 2>&1; echo "[FULL-BASELINE] DONE $m"; }
# GPU0/5/6 saturated with three jobs; fourth begins as soon as the first wave ends.
run_method university1652 0 & p0=$!
run_method sues200 5 & p5=$!
run_method denseuav 6 & p6=$!
wait "$p0" || { echo 'University-1652 failed' >&2; exit 20; }
wait "$p5" || { echo 'SUES-200 failed' >&2; exit 21; }
wait "$p6" || { echo 'DenseUAV failed' >&2; exit 22; }
run_method gtauav 0
for m in university1652 sues200 denseuav gtauav; do [[ -s "$BASE/$m/full_benchmark_and_routes.json" ]] || { echo "missing $m result" >&2; exit 23; }; done

# Official Bearing-UAV checkpoint and corrected UNI_PIXEL=128 metric.
WEIGHT_ZIP="$EXT/Bearing_UAV.zip"; WEIGHT_DIR="$BEARING/Bearing_UAV/cross_view"
if [[ ! -s "$WEIGHT_DIR/best_model.pth" || ! -s "$WEIGHT_DIR/training_configure.json" ]]; then
 tmp="$WEIGHT_ZIP.part"; rm -f "$tmp"; URL='https://huggingface.co/HaoyZhou/bearinguav/resolve/main/Bearing_UAV.zip?download=true'; if command -v curl >/dev/null 2>&1; then curl -L --fail --retry 4 --retry-delay 3 -o "$tmp" "$URL"; else wget -O "$tmp" "$URL"; fi; mv "$tmp" "$WEIGHT_ZIP"; "$PY" - "$WEIGHT_ZIP" "$BEARING" <<'PY'
import sys,zipfile
with zipfile.ZipFile(sys.argv[1]) as z:z.extractall(sys.argv[2])
PY
fi
run_off(){ local c="$1" g="$2"; CUDA_VISIBLE_DEVICES="$g" "$PY" v39_otherdata/bearinguav_official_route_eval.py --official-root "$BEARING" --weights-dir "$WEIGHT_DIR" --dataset-root "$DATASET_ROOT" --generated-root "$GEN" --city "$c" --output-root "$OFF" --workers 1 --batch-size 16 --cpu-threads "$CPU_THREADS" > >(tee "$LOG/bearinguav_${c}.log") 2>&1; }
run_off citya 0 & o0=$!; run_off cityb 5 & o5=$!; run_off cityc 6 & o6=$!; wait "$o0"; wait "$o5"; wait "$o6"; run_off cityd 0
CUDA_VISIBLE_DEVICES=0 "$PY" v39_otherdata/bearinguav_official_paper_eval.py --official-root "$BEARING" --weights-dir "$WEIGHT_DIR" --dataset-root "$DATASET_ROOT" --output "$OUT/bearinguav_official_paper_protocol.json" --workers 1 --batch-size 32 --cpu-threads "$CPU_THREADS" > >(tee "$LOG/bearinguav_paper.log") 2>&1

# Export two separate tables: true paper-protocol reproduction and same-route experiment.
"$PY" v39_otherdata/bearing_full_benchmark_comparison.py --generated-root "$GEN" --baseline-root "$BASE" --official-route-root "$OFF" --official-paper-json "$OUT/bearinguav_official_paper_protocol.json" --output-dir "$OUT"
"$PY" v39_otherdata/bearing_published_reference.py --generated-root "$GEN" --output-dir "$OUT"

# Collect high-contrast route figures.  No trajectory smoothing is applied.
rm -rf "$OUT/figures"; mkdir -p "$OUT/figures/ours" "$OUT/figures/bearinguav_official"
for m in university1652 sues200 denseuav gtauav; do mkdir -p "$OUT/figures/$m"; done
for c in citya cityb cityc cityd; do for r in test_01 test_02; do cp "$GEN/paper_bundle/${c}_${r}_final_result.jpg" "$OUT/figures/ours/${c}_${r}_final_result.jpg"; cp "$OFF/$c/${r}_final_result.jpg" "$OUT/figures/bearinguav_official/${c}_${r}_final_result.jpg"; for m in university1652 sues200 denseuav gtauav; do cp "$BASE/$m/$c/${r}_final_result.jpg" "$OUT/figures/$m/${c}_${r}_final_result.jpg"; done; done; done

cat > "$OUT/README_FINAL.txt" <<'EOF'
CORRECTED EXPERIMENT
1) paper_benchmark_reproduction.csv = official 85/5/10 Bearing-UAV localization benchmark.
   This is the table that should be compared with the paper's University-1652,
   SUES-200, DenseUAV, GTA-UAV and Bearing-UAV localization numbers.
2) same_routes_pooled.csv = the same eight official navigation-route frame sets.
   These route results are separate from the static paper benchmark.
3) External M2T methods receive only their native four adjacent RST candidates.
   No v39 waypoint, previous position, temporal state or local prior is injected.
4) Bearing-UAV regression metrics use official UNI_PIXEL=128, not PATCH_SIZE=256.
5) figures/ contains GT/prediction route plots; predictions are not smoothed.
EOF

echo '================================================================================'
echo 'CORRECTED FULL BEARING COMPARISON: PASS'
echo "Paper reproduction : $OUT/paper_benchmark_reproduction.csv"
echo "Same-route results  : $OUT/same_routes_pooled.csv"
echo "Figures             : $OUT/figures/"
echo "Logs                : $LOG/"
echo '================================================================================'
