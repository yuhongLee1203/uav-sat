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

cleanup(){
  git -C "${ROOT}" worktree remove --force "${WT}" >/dev/null 2>&1 || true
  git -C "${ROOT}" worktree remove --force "${UPLOAD_WT}" >/dev/null 2>&1 || true
  git -C "${ROOT}" branch -D "${UPLOAD_BRANCH}" >/dev/null 2>&1 || true
}
trap cleanup EXIT
mkdir -p "${OUT}/logs" "${OUT}/paper_ablation_by_city"

echo "============================================================"
echo "FROZEN V5 ALL-PAPER SUITE"
echo "Frozen SHA : ${FROZEN_SHA}"
echo "Dataset    : ${DATASET_ROOT}"
echo "Output     : ${OUT}"
echo "Cities     : A B C D"
echo "Ablations  : Full / -GRU / -Kalman / -MS / 1f / 2f / 3f / MS grids"
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

  # Preserve each city's own ablation tables before the next city overwrites
  # suite-root paper_ablation_* files.
  adst="${OUT}/paper_ablation_by_city/${city}"
  mkdir -p "${adst}"
  cp "${OUT}"/paper_ablation_results.{json,csv} "${adst}/"
  cp "${OUT}"/paper_ablation_tables.{md,tex} "${adst}/"
  cp "${OUT}/paper_trend_audit.json" "${adst}/"

  # Paper figures for Full only: raw final_x/final_y, official sparse-waypoint GT.
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

python3 v39_otherdata/build_frozen_v5_all_paper.py --suite-root "${OUT}"

# Build an 8-panel figure sheet from the exact raw-result plots.
python3 - "${OUT}" <<'PY'
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import sys
r=Path(sys.argv[1]); cities=('citya','cityb','cityc','cityd'); routes=('test_01','test_02')
items=[]
for c in cities:
  for q in routes:
    p=r/c/'variants/full'/f'{q}_final_result.jpg'; items.append((c,q,Image.open(p).convert('RGB')))
w,h,lh=720,500,42
can=Image.new('RGB',(2*w,4*(h+lh)),'white');d=ImageDraw.Draw(can)
try:f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22)
except Exception:f=ImageFont.load_default()
for i,(c,q,im) in enumerate(items):
  im.thumbnail((w,h)); col=i%2; row=i//2; x=col*w+(w-im.width)//2; y=row*(h+lh)+lh+(h-im.height)//2
  can.paste(im,(x,y));d.text((col*w+14,row*(h+lh)+8),f'{c.upper()} {q}',fill='black',font=f)
out=r/'paper_bundle'/'all4_full_results_contact_sheet.jpg';can.save(out,quality=95);print('[FIGURE SHEET]',out)
PY

# Audit frozen CityA known-good trend and also report all-city pooled trend.
python3 - "${OUT}" <<'PY'
import csv,json,sys
from pathlib import Path
r=Path(sys.argv[1]); citya=json.loads((r/'paper_ablation_by_city/citya/paper_trend_audit.json').read_text())
print('[CITYA FROZEN AUDIT]',json.dumps({k:citya.get(k) for k in ('FULL_TREND_CHECK','component_full_best','three_frame_full_best')},sort_keys=True))
with (r/'paper_bundle/table_ablation_core_pooled.csv').open() as f: core=list(csv.DictReader(f))
with (r/'paper_bundle/table_temporal_context_pooled.csv').open() as f: temp=list(csv.DictReader(f))
print('[ALL4 CORE POOLED]');[print(x) for x in core]
print('[ALL4 TEMPORAL POOLED]');[print(x) for x in temp]
PY

printf '%s\n' "${OUT}" > v39_otherdata/generated/LATEST_FROZEN_V5_ALL_PAPER.txt

if [[ "${UPLOAD_RESULTS}" == "1" ]]; then
  git worktree add -b "${UPLOAD_BRANCH}" "${UPLOAD_WT}" origin/bearing-v5-paper-repro-20260921
  mkdir -p "${UPLOAD_WT}/${DEST}"
  cp -a "${OUT}/paper_bundle" "${UPLOAD_WT}/${DEST}/"
  cp -a "${OUT}/paper_ablation_by_city" "${UPLOAD_WT}/${DEST}/"
  for city in citya cityb cityc cityd; do
    dst="${UPLOAD_WT}/${DEST}/${city}"; mkdir -p "${dst}/full" "${dst}/ablation_summaries"
    cp "${OUT}/${city}/prepared/experiment.json" "${dst}/prepared_experiment.json"
    cp "${OUT}/${city}/variants/full/bearing_v39_summary.json" "${dst}/full/"
    cp "${OUT}/${city}/variants/full"/*_frames.csv "${dst}/full/" 2>/dev/null || true
    cp "${OUT}/${city}/variants/full/test_01_final_result.jpg" "${dst}/full/"
    cp "${OUT}/${city}/variants/full/test_02_final_result.jpg" "${dst}/full/"
    cp -a "${OUT}/${city}/variants/full/paper_figures_waypoint_gt" "${dst}/full/" 2>/dev/null || true
    for v in full no_gru no_kalman no_ms frames1 frames2 grid4 grid5 grid7 grid8; do
      mkdir -p "${dst}/ablation_summaries/${v}"
      cp "${OUT}/${city}/variants/${v}/bearing_v39_summary.json" "${dst}/ablation_summaries/${v}/"
      cp "${OUT}/${city}/variants/${v}/experiment_manifest.json" "${dst}/ablation_summaries/${v}/" 2>/dev/null || true
    done
  done
  cat > "${UPLOAD_WT}/${DEST}/README.txt" <<EOF
Frozen V5 all-city paper suite.
Exact algorithm source: ${FROZEN_SHA} (bearing-v5-citya-pass-20260920).
Every city uses the same Frozen-V5 experiment definition and runs full component/temporal/grid ablations.
Main Full figures use raw final_x/final_y with official sparse waypoint GT; no display smoothing.
Comparison caveat: current method is controlled_gt_jitter local-prior sequential refinement. Camera-heading metrics and Bearing-Naver closed-loop navigation metrics are not fabricated; motion-heading/offline replay are exported separately.
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
echo "FROZEN V5 ALL-PAPER COMPLETE"
echo "Suite : ${OUT}"
echo "Tables: ${OUT}/paper_bundle/PAPER_TABLES.md"
echo "Figures: ${OUT}/paper_bundle/all4_full_results_contact_sheet.jpg"
echo "============================================================"
