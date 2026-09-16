#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CPU_THREADS="${CPU_THREADS:-2}"
GPU_IDS=(0 5 6)
cd "$ROOT"

# First run ours + the four native full-city matching-to-tile baselines + the
# authors' official Bearing-UAV checkpoint.  This stage also creates the shared
# mmap image cache and isolated helper environment.
DATASET_ROOT="$DATASET_ROOT" CPU_THREADS="$CPU_THREADS" \
  FORCE_OURS="${FORCE_OURS:-0}" \
  FORCE_BASELINES="${FORCE_BASELINES:-0}" \
  FORCE_OFFICIAL="${FORCE_OFFICIAL:-0}" \
  bash v39_otherdata/run_bearing_fair_comparison.sh

GEN="$ROOT/v39_otherdata/generated"
EXT="$ROOT/v39_otherdata/external"
PY="$EXT/native_baseline_env/bin/python"
BEARING="$EXT/bearinguav"
CACHE="$GEN/native_m2t_cache"
BASE="$GEN/native_m2t_baselines"
OFF="$GEN/official_bearinguav_same_routes"
OUT="$GEN/native_comparison"
LOG="$GEN/native_comparison_logs"
BEARING_COMMIT="d16e558a14bd0a9142c6901793fe6a6813e4e954"

export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS"
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EXT/torch_cache"
export HF_HOME="$EXT/hf_cache"
mkdir -p "$LOG" "$TORCH_HOME" "$HF_HOME"

"$PY" -m py_compile v39_otherdata/bearinguav_route_adapted.py v39_otherdata/bearing_native_comparison.py

# Same-training-scope Bearing-UAV comparison:
# train only on selected train_01, but use Bearing-UAV's OWN four-RST input,
# position+heading regression and loss.  No v39 waypoint/temporal/local prior.
br_ok(){
  local d="$BASE/bearinguav_route_adapted/$1"
  [[ -s "$d/result.json" && -s "$d/test_01_final_result.jpg" && -s "$d/test_02_final_result.jpg" ]] \
    && grep -q '"training_scope": "selected train_01 only"' "$d/result.json"
}
run_br(){
  local city="$1" gpu="$2"; local force=()
  [[ "${FORCE_BEARING_ROUTE:-0}" == 1 ]] && force+=(--force)
  echo "[BEARING-ROUTE] START $city GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" \
  "$PY" v39_otherdata/bearinguav_route_adapted.py \
    --official-root "$BEARING" \
    --dataset-root "$DATASET_ROOT" \
    --prepared-root "$GEN/$city" \
    --cache-root "$CACHE/$city" \
    --output-root "$BASE" \
    --city "$city" \
    --repo-commit "$BEARING_COMMIT" \
    --epochs 100 --batch-size 16 --cpu-threads "$CPU_THREADS" "${force[@]}" \
    > >(tee "$LOG/bearinguav_route_${city}.log") 2>&1
  echo "[BEARING-ROUTE] DONE $city"
}

pending=()
for city in citya cityb cityc cityd; do
  if [[ "${FORCE_BEARING_ROUTE:-0}" == 1 ]] || ! br_ok "$city"; then pending+=("$city"); else echo "[BEARING-ROUTE] cache hit $city"; fi
done
for ((base=0;base<${#pending[@]};base+=3)); do
  pids=(); names=()
  for slot in 0 1 2; do
    idx=$((base+slot)); ((idx<${#pending[@]})) || break
    city="${pending[$idx]}"; run_br "$city" "${GPU_IDS[$slot]}" & pids+=("$!"); names+=("$city")
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "[BEARING-ROUTE] FAILED ${names[$i]} -- see $LOG" >&2; exit 30; }
  done
done
for city in citya cityb cityc cityd; do br_ok "$city" || { echo "missing Bearing route-adapted result: $city" >&2; exit 31; }; done

# Rebuild final table now that the same-training-scope Bearing-UAV row exists.
"$PY" v39_otherdata/bearing_native_comparison.py \
  --generated-root "$GEN" --baseline-root "$BASE" --official-root "$OFF" --output-dir "$OUT"

mkdir -p "$OUT/figures/bearinguav_route_adapted"
for city in citya cityb cityc cityd; do
  for route in test_01 test_02; do
    cp "$BASE/bearinguav_route_adapted/$city/${route}_final_result.jpg" \
       "$OUT/figures/bearinguav_route_adapted/${city}_${route}_final_result.jpg"
  done
done

# Final paper-artifact audit: 7 experimental rows/families, 8 route figures each
# where a trajectory figure is meaningful.
for method in ours bearinguav_route_adapted bearinguav_official university1652 sues200 denseuav gtauav; do
  dir="$OUT/figures/$method"
  count=$(find "$dir" -maxdepth 1 -type f -name '*_final_result.jpg' | wc -l)
  [[ "$count" -eq 8 ]] || { echo "figure audit failed $method: $count/8" >&2; exit 32; }
done
for f in same_frames_pooled.csv same_frames_route_level.csv bearinguav_published_uav_reference.csv comparison_manifest.json; do
  test -s "$OUT/$f" || { echo "missing final artifact: $OUT/$f" >&2; exit 33; }
done
grep -q 'Bearing-UAV-route-adapted' "$OUT/same_frames_pooled.csv" || { echo "final table missing Bearing-UAV-route-adapted" >&2; exit 34; }

cat > "$OUT/README_FINAL.txt" <<'EOF'
Primary rerun table: same_frames_pooled.csv
Per-route table:      same_frames_route_level.csv
Published references: bearinguav_published_uav_reference.csv

Native-input rules:
- University-1652 / SUES-200 / DenseUAV / GTA-UAV:
  independent UAV->satellite matching against ALL 256 city RST tiles.
  They receive NO waypoint, planned route, previous position, temporal state,
  local prior, or endpoint correction. Their failures/drift/jumps are retained.
- Bearing-UAV-route-adapted:
  official four-neighbour RST pose-regression architecture/objective, trained
  on selected train_01 only; no v39 prior.
- Bearing-UAV-official-pretrained:
  authors' released full-data checkpoint, shown separately because training
  scope differs.
- Ours-v39:
  retains its own temporal controlled-local-prior protocol.
EOF

echo "================================================================================"
echo "ALL-METHOD PAPER BUNDLE: PASS"
echo "Main table : $OUT/same_frames_pooled.csv"
echo "Route table: $OUT/same_frames_route_level.csv"
echo "Figures    : $OUT/figures/"
echo "Published  : $OUT/bearinguav_published_uav_reference.csv"
echo "Logs       : $LOG/"
echo "================================================================================"
