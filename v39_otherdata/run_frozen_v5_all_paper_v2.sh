#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

FROZEN_SHA="6911ac1dbccbfc162b3bf77a253c235063fbd60c"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-100}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
SEED="${SEED:-2033}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${PAPER_SUITE_ROOT:-${ROOT}/v39_otherdata/generated/frozen_v5_all_paper_${TS}}"
WT="${ROOT%/*}/uav-sat-frozen-v5-all-${TS}"
UPLOAD_WT="${ROOT%/*}/uav-sat-frozen-v5-upload-${TS}"
UPLOAD_BRANCH="paper-repro-upload-${TS}-$$"
DEST="paper_results/frozen_v5_all_paper_${TS}"
PATCHER="${ROOT}/v39_otherdata/patch_frozen_v5_runner_sequential.py"
BUILDER="${ROOT}/v39_otherdata/build_frozen_v5_all_paper.py"

cleanup(){
  git -C "${ROOT}" worktree remove --force "${WT}" >/dev/null 2>&1 || true
  git -C "${ROOT}" worktree remove --force "${UPLOAD_WT}" >/dev/null 2>&1 || true
  git -C "${ROOT}" branch -D "${UPLOAD_BRANCH}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Fail before any training if the launcher/helper syntax is broken.
bash -n "$0"
python3 -m py_compile "${PATCHER}" "${BUILDER}"
[[ -d "${DATASET_ROOT}" ]] || { echo "ERROR: dataset missing: ${DATASET_ROOT}" >&2; exit 2; }
mkdir -p "${OUT}/logs" "${OUT}/paper_ablation_by_city"

echo "============================================================"
echo "FROZEN V5 ALL-PAPER SUITE V2 (SYNTAX-CHECKED / SEQUENTIAL)"
echo "Frozen SHA : ${FROZEN_SHA}"
echo "Dataset    : ${DATASET_ROOT}"
echo "Output     : ${OUT}"
echo "Cities     : A B C D"
echo "Ablations  : Full / -GRU / -Kalman / -MS / 1f / 2f / 3f / MS grids"
echo "GPU policy : temporal/eval sequential on GPU0"
echo "============================================================"

git fetch origin bearing-v5-citya-pass-20260920 bearing-v5-paper-repro-20260921
actual="$(git rev-parse origin/bearing-v5-citya-pass-20260920)"
[[ "${actual}" == "${FROZEN_SHA}" ]] || { echo "ERROR: frozen branch moved: ${actual}" >&2; exit 2; }

git worktree add --detach "${WT}" "${FROZEN_SHA}"

for city in citya cityb cityc cityd; do
  echo "============================================================"
  echo "[CITY START] ${city}"
  echo "============================================================"

  git -C "${WT}" reset --hard "${FROZEN_SHA}" >/dev/null
  git -C "${WT}" clean -fdx >/dev/null

  # Patch only launcher parallelism. Model/algorithm source remains frozen.
  python3 "${PATCHER}" "${WT}/v39_otherdata/run_bearing_iclr_ablation.sh"
  bash -n "${WT}/v39_otherdata/run_bearing_iclr_ablation.sh"
  bash -n "${WT}/v39_otherdata/run_bearing_iclr_ablation_fixed.sh"
  echo "[CITY PRECHECK] ${city} frozen runner syntax + sequential patch: PASS"

  (
    cd "${WT}"
    CITY="${city}" \
    ICLR_SUITE_ROOT="${OUT}" \
    BEARING_DATASET_ROOT="${DATASET_ROOT}" \
    TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS}" \
    VISUAL_EPOCHS="${VISUAL_EPOCHS}" \
    PATIENCE=4 \
    SEED="${SEED}" \
    UPLOAD_RESULTS=0 \
    bash v39_otherdata/run_bearing_iclr_ablation_fixed.sh
  ) 2>&1 | tee "${OUT}/logs/${city}_frozen_v5.log"

  adst="${OUT}/paper_ablation_by_city/${city}"
  mkdir -p "${adst}"
  cp "${OUT}"/paper_ablation_results.{json,csv} "${adst}/"
  cp "${OUT}"/paper_ablation_tables.{md,tex} "${adst}/"
  cp "${OUT}/paper_trend_audit.json" "${adst}/"

  python3 "${WT}/v39_otherdata/bearing_plot_final_vs_gt.py" \
    --prepared-root "${OUT}/${city}/prepared" \
    --output-dir "${OUT}/${city}/variants/full" \
    --routes test_01 test_02

  for req in \
    "${OUT}/${city}/variants/full/test_01_final_result.jpg" \
    "${OUT}/${city}/variants/full/test_02_final_result.jpg" \
    "${OUT}/${city}/variants/full/bearing_v39_summary.json" \
    "${adst}/paper_ablation_results.json" \
    "${adst}/paper_trend_audit.json"; do
    [[ -s "${req}" ]] || { echo "ERROR: missing ${req}" >&2; exit 3; }
  done

  echo "[CITY DONE] ${city}"
done

python3 "${BUILDER}" --suite-root "${OUT}"

python3 - "${OUT}" <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys

root = Path(sys.argv[1])
cities = ("citya", "cityb", "cityc", "cityd")
routes = ("test_01", "test_02")
items = []
for city in cities:
    for route in routes:
        p = root / city / "variants" / "full" / f"{route}_final_result.jpg"
        items.append((city, route, Image.open(p).convert("RGB")))

w, h, label_h = 720, 500, 42
canvas = Image.new("RGB", (2 * w, 4 * (h + label_h)), "white")
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except Exception:
    font = ImageFont.load_default()

for i, (city, route, image) in enumerate(items):
    image.thumbnail((w, h))
    col, row = i % 2, i // 2
    x = col * w + (w - image.width) // 2
    y = row * (h + label_h) + label_h + (h - image.height) // 2
    canvas.paste(image, (x, y))
    draw.text((col * w + 14, row * (h + label_h) + 8), f"{city.upper()} {route}", fill="black", font=font)

out = root / "paper_bundle" / "all4_full_results_contact_sheet.jpg"
canvas.save(out, quality=95)
print("[FIGURE SHEET]", out)
PY

python3 - "${OUT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
citya = json.loads((root / "paper_ablation_by_city/citya/paper_trend_audit.json").read_text())
print("[CITYA FROZEN AUDIT]", json.dumps({
    k: citya.get(k) for k in ("FULL_TREND_CHECK", "component_full_best", "three_frame_full_best")
}, sort_keys=True))

with (root / "paper_bundle/table_ablation_core_pooled.csv").open() as f:
    core = list(csv.DictReader(f))
with (root / "paper_bundle/table_temporal_context_pooled.csv").open() as f:
    temporal = list(csv.DictReader(f))
print("[ALL4 CORE POOLED]")
for row in core:
    print(row)
print("[ALL4 TEMPORAL POOLED]")
for row in temporal:
    print(row)
PY

printf '%s\n' "${OUT}" > "${ROOT}/v39_otherdata/generated/LATEST_FROZEN_V5_ALL_PAPER.txt"

if [[ "${UPLOAD_RESULTS}" == "1" ]]; then
  git worktree add -b "${UPLOAD_BRANCH}" "${UPLOAD_WT}" origin/bearing-v5-paper-repro-20260921
  mkdir -p "${UPLOAD_WT}/${DEST}"
  cp -a "${OUT}/paper_bundle" "${UPLOAD_WT}/${DEST}/"
  cp -a "${OUT}/paper_ablation_by_city" "${UPLOAD_WT}/${DEST}/"

  for city in citya cityb cityc cityd; do
    dst="${UPLOAD_WT}/${DEST}/${city}"
    mkdir -p "${dst}/full" "${dst}/ablation_summaries"
    cp "${OUT}/${city}/prepared/experiment.json" "${dst}/prepared_experiment.json"
    cp "${OUT}/${city}/variants/full/bearing_v39_summary.json" "${dst}/full/"
    cp "${OUT}/${city}/variants/full"/*_frames.csv "${dst}/full/" 2>/dev/null || true
    cp "${OUT}/${city}/variants/full/test_01_final_result.jpg" "${dst}/full/"
    cp "${OUT}/${city}/variants/full/test_02_final_result.jpg" "${dst}/full/"
    cp -a "${OUT}/${city}/variants/full/paper_figures_waypoint_gt" "${dst}/full/" 2>/dev/null || true

    for variant in full no_gru no_kalman no_ms frames1 frames2 grid4 grid5 grid7 grid8; do
      mkdir -p "${dst}/ablation_summaries/${variant}"
      cp "${OUT}/${city}/variants/${variant}/bearing_v39_summary.json" "${dst}/ablation_summaries/${variant}/"
      cp "${OUT}/${city}/variants/${variant}/experiment_manifest.json" "${dst}/ablation_summaries/${variant}/" 2>/dev/null || true
    done
  done

  cat > "${UPLOAD_WT}/${DEST}/README.txt" <<EOF
Frozen V5 all-city paper suite V2.
Exact algorithm source: ${FROZEN_SHA} (bearing-v5-citya-pass-20260920).
The only runtime modification is sequential launcher scheduling to avoid background GPU worker failures; model/algorithm source remains frozen.
Every city runs the same component, temporal, and MeanShift-grid ablations.
Full figures use raw final_x/final_y with official sparse waypoint GT; no display smoothing.
Protocol caveat: controlled_gt_jitter local-prior sequential refinement. Camera-heading and Bearing-Naver closed-loop metrics are not fabricated; motion-heading/offline replay diagnostics are separate.
EOF

  (
    cd "${UPLOAD_WT}"
    git add "${DEST}"
    git commit -m "Add frozen V5 all-city paper suite ${TS}"
    git push origin HEAD:bearing-v5-paper-repro-20260921
  )
  echo "[UPLOAD DONE] ${DEST}"
fi

echo "============================================================"
echo "FROZEN V5 ALL-PAPER V2 COMPLETE"
echo "Suite   : ${OUT}"
echo "Tables  : ${OUT}/paper_bundle/PAPER_TABLES.md"
echo "Figures : ${OUT}/paper_bundle/all4_full_results_contact_sheet.jpg"
echo "============================================================"
