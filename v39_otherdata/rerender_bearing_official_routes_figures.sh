#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GEN="$ROOT/v39_otherdata/generated"
OUT="$GEN/official_routes_only"
mkdir -p "$OUT"

python3 -m py_compile v39_otherdata/bearing_plot_final_vs_gt.py

for city in citya cityb cityc cityd; do
  prepared="$GEN/$city"
  model_out="$prepared/v39_output_bearing_adapted"
  test -s "$model_out/bearing_v39_summary.json" || { echo "missing existing inference: $model_out" >&2; exit 2; }
  python3 v39_otherdata/bearing_plot_final_vs_gt.py \
    --prepared-root "$prepared" \
    --output-dir "$model_out" \
    --routes test_01 test_02
  cp "$model_out/test_01_final_result.jpg" "$OUT/${city}_test_01_final_result.jpg"
  cp "$model_out/test_02_final_result.jpg" "$OUT/${city}_test_02_final_result.jpg"
done

count=$(find "$OUT" -maxdepth 1 -type f -name '*_final_result.jpg' | wc -l)
[[ "$count" -eq 8 ]] || { echo "figure audit failed: $count/8" >&2; exit 3; }
echo "[RERENDER] PASS: 8 clearer official-route figures written to $OUT"
