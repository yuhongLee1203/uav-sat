#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
GPU="${GPU:-0}"
RUN_MODEL="${RUN_MODEL:-0}"
GEN_ROOT="${REPO_ROOT}/v39_otherdata/generated"
FINAL_DIR="${GEN_ROOT}/final_results"
BUNDLE_DIR="${GEN_ROOT}/paper_bundle"

cd "${REPO_ROOT}"

python3 -m py_compile \
  v39_otherdata/bearing_plot_final_vs_gt.py \
  v39_otherdata/bearing_published_reference.py

echo "[PAPER-BUNDLE] code compile: PASS"

need_model=0
for city in citya cityb cityc cityd; do
  out="${GEN_ROOT}/${city}/v39_output_bearing_adapted"
  if [[ ! -s "${out}/bearing_v39_summary.json" ]]; then
    need_model=1
  fi
done

if [[ "${RUN_MODEL}" == "1" || "${need_model}" == "1" ]]; then
  echo "[PAPER-BUNDLE] running/rebuilding the four-city v39 experiment first"
  DATASET_ROOT="${DATASET_ROOT}" GPU="${GPU}" \
    bash v39_otherdata/run_bearing_v39_all_cities.sh
else
  echo "[PAPER-BUNDLE] reusing existing four-city inference outputs (RUN_MODEL=0)"
fi

rm -rf "${BUNDLE_DIR}"
mkdir -p "${BUNDLE_DIR}" "${FINAL_DIR}"

for city in citya cityb cityc cityd; do
  prepared="${GEN_ROOT}/${city}"
  out="${prepared}/v39_output_bearing_adapted"
  test -s "${out}/bearing_v39_summary.json"
  echo "[PAPER-BUNDLE] render ${city}"
  python3 v39_otherdata/bearing_plot_final_vs_gt.py \
    --prepared-root "${prepared}" --output-dir "${out}" --routes test_01 test_02
  for route in test_01 test_02; do
    src="${out}/${route}_final_result.jpg"
    dst="${FINAL_DIR}/${city}_${route}_final_result.jpg"
    test -s "${src}"
    cp "${src}" "${dst}"
    cp "${src}" "${BUNDLE_DIR}/${city}_${route}_final_result.jpg"
  done
done

# Published values remain a separate reference table; they are not described as reruns.
python3 v39_otherdata/bearing_published_reference.py \
  --generated-root "${GEN_ROOT}" --output-dir "${BUNDLE_DIR}"

for f in \
  bearing_multicity_status.json \
  bearing_multicity_summary.csv \
  bearing_multicity_summary.json \
  bearing_paper_comparison_multicity.csv \
  bearing_paper_comparison_multicity.json; do
  if [[ -s "${GEN_ROOT}/${f}" ]]; then cp "${GEN_ROOT}/${f}" "${BUNDLE_DIR}/${f}"; fi
done

python3 - "${BUNDLE_DIR}" <<'PY'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]);rows=[]
for city in ("citya","cityb","cityc","cityd"):
    for route in ("test_01","test_02"):
        image=root/f"{city}_{route}_final_result.jpg"
        if not image.exists() or image.stat().st_size==0:raise SystemExit(f"[PAPER-BUNDLE] missing image: {image}")
        rows.append({
            "city":city,"route":route,"figure":image.name,
            "prediction_style":"red solid + white halo",
            "ground_truth_style":"purple dashed + white halo (true per-frame GT)",
            "planned_route_style":"gray dotted (context only)",
        })
with (root/"figure_manifest.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
(root/"figure_manifest.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
print(f"[PAPER-BUNDLE] figures={len(rows)}: PASS")
PY

echo ""
echo "================================================================================"
echo "BEARING PAPER BUNDLE COMPLETE"
echo "High-contrast final figures (8): ${BUNDLE_DIR}/city*_test_0*_final_result.jpg"
echo "Published main-paper rows       : ${BUNDLE_DIR}/bearinguav_published_uav_reference.csv"
echo "Bearing-UAV backbone supplement: ${BUNDLE_DIR}/bearinguav_backbone_supplement.csv"
echo "Our results + published rows    : ${BUNDLE_DIR}/ours_vs_bearinguav_published.csv"
echo "Our route-level results         : ${BUNDLE_DIR}/bearing_multicity_summary.csv"
echo "Figure manifest                 : ${BUNDLE_DIR}/figure_manifest.csv"
echo "================================================================================"
