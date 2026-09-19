#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
HIST_COMMIT="9bb0ae28400d783535430b54f9c3417feba53103"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
HIST_WT="${REPO_ROOT%/*}/uav-sat-bearing-all4-${TS}"
LOCAL_ROOT="${REPO_ROOT}/v39_otherdata/generated/all4_softms_${TS}"
UPLOAD_WT="${REPO_ROOT%/*}/uav-sat-upload-bearing-all4-${TS}"
UPLOAD_BRANCH="upload-bearing-all4-${TS}"
UPLOAD_DEST="paper_results/bearing_all4_softms_${TS}"
LATEST_PLOTTER="${REPO_ROOT}/v39_otherdata/bearing_plot_final_vs_gt.py"

cleanup(){
  if [[ -d "${HIST_WT}" ]]; then
    git -C "${REPO_ROOT}" worktree remove --force "${HIST_WT}" >/dev/null 2>&1 || true
  fi
  if [[ -d "${UPLOAD_WT}" ]]; then
    git -C "${REPO_ROOT}" worktree remove --force "${UPLOAD_WT}" >/dev/null 2>&1 || true
  fi
  git -C "${REPO_ROOT}" branch -D "${UPLOAD_BRANCH}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

[[ -s "${LATEST_PLOTTER}" ]] || { echo "ERROR: missing latest plotter ${LATEST_PLOTTER}" >&2; exit 2; }
python3 -m py_compile "${LATEST_PLOTTER}"
mkdir -p "${LOCAL_ROOT}/logs"

echo "================================================================================"
echo "Bearing-UAV ALL FOUR CITIES / Forward18 SoftMS"
echo "Historical verified physical adapter: ${HIST_COMMIT}"
echo "GPUs: citya+cityd -> GPU0, cityb -> GPU5, cityc -> GPU6"
echo "Persistent local output: ${LOCAL_ROOT}"
echo "Figures: green waypoint GT + raw red prediction, NO display smoothing"
echo "================================================================================"

git fetch origin v39_otherdata
git worktree add --detach "${HIST_WT}" "${HIST_COMMIT}"

# Use the newest audited plotter while keeping the verified low-error Bearing
# data/model adapter from the historical commit.
cp "${LATEST_PLOTTER}" "${HIST_WT}/v39_otherdata/bearing_plot_final_vs_gt.py"
python3 -m py_compile "${HIST_WT}/v39_otherdata/bearing_plot_final_vs_gt.py"

# Change ONLY the front decoder from weighted centroid to SoftMS and patience to 4.
# Keep the verified Bearing-specific physical adaptation unchanged:
# step=4m, metre-matched SAT stride/crop, Route-A-only cadence adaptation,
# fixed-R constrained Kalman, final 6x6 MS/BW7 and route-centerline final-MS reference.
python3 - \
  "${HIST_WT}/v39_otherdata/bearing_runner.py" \
  "${HIST_WT}/v39_otherdata/bearing_runner_exact_v39.py" \
  "${HIST_WT}/v39_otherdata/run_bearing_v39_sequence_fixed.sh" <<'PY'
from pathlib import Path
import sys
base, exact, sh = map(Path, sys.argv[1:])

s=base.read_text(encoding='utf-8')
s=s.replace('"UAVSAT_EXPERIMENT_ANCHOR": "weighted_centroid"','"UAVSAT_EXPERIMENT_ANCHOR": "softms"')
s=s.replace('"EXPERIMENT_ANCHOR": "weighted_centroid"','"EXPERIMENT_ANCHOR": "softms"')
s=s.replace('"visual_decoder": "weighted_centroid"','"visual_decoder": "softms"')
base.write_text(s,encoding='utf-8')
compile(s,str(base),'exec')

s=exact.read_text(encoding='utf-8')
s=s.replace('"UAVSAT_EXPERIMENT_ANCHOR": "weighted_centroid"','"UAVSAT_EXPERIMENT_ANCHOR": "softms"')
s=s.replace('"visual_decoder": "weighted_centroid"','"visual_decoder": "softms"')
s=s.replace('Weighted Centroid -> 3-frame Context-GRU','Forward18 SoftMS -> 3-frame Context-GRU')
exact.write_text(s,encoding='utf-8')
compile(s,str(exact),'exec')

s=sh.read_text(encoding='utf-8')
s=s.replace('--patience 10 \\', '--patience 4 \\')
sh.write_text(s,encoding='utf-8')
PY

run_city(){
  local city="$1" gpu="$2"
  echo "[START] ${city} GPU${gpu}"
  (
    cd "${HIST_WT}"
    CITY="${city}" GPU="${gpu}" DATASET_ROOT="${DATASET_ROOT}" \
      bash v39_otherdata/run_bearing_v39_directfinalms_official_routes.sh
  ) 2>&1 | tee "${LOCAL_ROOT}/logs/${city}.log"
  echo "[DONE] ${city}"
}

# Maximise GPU0/5/6 without putting every job on GPU0.
( run_city citya 0; run_city cityd 0 ) & p0=$!
( run_city cityb 5 ) & p5=$!
( run_city cityc 6 ) & p6=$!
status=0
wait "${p0}" || status=1
wait "${p5}" || status=1
wait "${p6}" || status=1
[[ "${status}" == "0" ]] || { echo "ERROR: one or more city runs failed; inspect ${LOCAL_ROOT}/logs" >&2; exit 20; }

# Persist ALL paper-facing outputs in the current working repository BEFORE the
# detached historical worktree is cleaned up.
for city in citya cityb cityc cityd; do
  src_root="${HIST_WT}/v39_otherdata/generated/${city}"
  src_out="${src_root}/v39_output_bearing_adapted"
  dst="${LOCAL_ROOT}/${city}"
  mkdir -p "${dst}"

  for req in \
    bearing_v39_summary.json \
    bearing_paper_metrics.json \
    bearing_paper_metrics.csv \
    final_quality_audit.json \
    v39_bearing_training_audit.json \
    test_01_final_result.jpg \
    test_02_final_result.jpg \
    paper_figures_waypoint_gt/test_01_waypoint_gt_green.jpg \
    paper_figures_waypoint_gt/test_02_waypoint_gt_green.jpg \
    paper_figures_waypoint_gt/plot_source_audit.json; do
      [[ -s "${src_out}/${req}" ]] || { echo "ERROR: missing ${city}/${req}" >&2; exit 21; }
  done

  # Keep summaries, raw per-frame CSVs and every final/paper figure.
  cp "${src_out}/bearing_v39_summary.json" "${dst}/"
  cp "${src_out}/bearing_paper_metrics.json" "${dst}/"
  cp "${src_out}/bearing_paper_metrics.csv" "${dst}/"
  cp "${src_out}/final_quality_audit.json" "${dst}/"
  cp "${src_out}/v39_bearing_training_audit.json" "${dst}/"
  cp "${src_out}/test_01_final_result.jpg" "${dst}/"
  cp "${src_out}/test_02_final_result.jpg" "${dst}/"
  cp "${src_out}"/*_frames.csv "${dst}/" 2>/dev/null || true
  mkdir -p "${dst}/paper_figures_waypoint_gt"
  cp -a "${src_out}/paper_figures_waypoint_gt/." "${dst}/paper_figures_waypoint_gt/"
  cp "${src_root}/experiment.json" "${dst}/experiment.json"
done

# Aggregate all eight routes from the copied raw frame CSVs, so the summary is
# independent of the soon-to-be-deleted historical worktree paths.
python3 - "${LOCAL_ROOT}" <<'PY'
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root=Path(sys.argv[1])
cities=['citya','cityb','cityc','cityd']
rows=[]; all_errors=[]
for city in cities:
    out=root/city
    summary=json.loads((out/'bearing_v39_summary.json').read_text(encoding='utf-8'))
    for route in ('test_01','test_02'):
        s=summary[route]
        old=Path(str(s['CSV']))
        candidates=list(out.glob(f'{route}_*_frames.csv'))
        match=[p for p in candidates if p.name==old.name]
        p=(match[0] if match else candidates[-1])
        with p.open(newline='',encoding='utf-8') as f:
            rr=list(csv.DictReader(f))
        e=np.asarray([float(x['error_final_m']) for x in rr],dtype=float)
        all_errors.extend(e.tolist())
        row={
            'city':city,'route':route,'frames':len(e),
            'MLE_m':float(e.mean()),'MedLE_m':float(np.median(e)),
            'P90_m':float(np.percentile(e,90)),'P95_m':float(np.percentile(e,95)),
            'P99_m':float(np.percentile(e,99)),
            'LSR@5_pct':100*float(np.mean(e<=5)),'LSR@10_pct':100*float(np.mean(e<=10)),
            'LSR@15_pct':100*float(np.mean(e<=15)),'LSR@20_pct':100*float(np.mean(e<=20)),
        }
        rows.append(row)
arr=np.asarray(all_errors,dtype=float)
overall={
    'city':'ALL','route':'ALL_8_ROUTES','frames':int(arr.size),
    'MLE_m':float(arr.mean()),'MedLE_m':float(np.median(arr)),
    'P90_m':float(np.percentile(arr,90)),'P95_m':float(np.percentile(arr,95)),
    'P99_m':float(np.percentile(arr,99)),
    'LSR@5_pct':100*float(np.mean(arr<=5)),'LSR@10_pct':100*float(np.mean(arr<=10)),
    'LSR@15_pct':100*float(np.mean(arr<=15)),'LSR@20_pct':100*float(np.mean(arr<=20)),
}
payload={
    'method':'Forward18 SoftMS -> 3-frame Context-GRU -> fixed-R Kalman -> final 6x6 Soft MeanShift',
    'bearing_adapter_source':'9bb0ae28400d783535430b54f9c3417feba53103',
    'metric_note':'Measured raw outputs only; no metric values are edited.',
    'routes':rows,'overall_8_routes':overall,
}
(root/'paper_all4_summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
fields=['city','route','frames','MLE_m','MedLE_m','P90_m','P95_m','P99_m','LSR@5_pct','LSR@10_pct','LSR@15_pct','LSR@20_pct']
with (root/'paper_all4_summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows); w.writerow(overall)

# Easy-to-open 4x2 contact sheet using the exact already-rendered paper figures.
items=[]
for city in cities:
    for route in ('test_01','test_02'):
        p=root/city/'paper_figures_waypoint_gt'/f'{route}_waypoint_gt_green.jpg'
        items.append((city,route,Image.open(p).convert('RGB')))
thumb_w,thumb_h=720,500
label_h=42
canvas=Image.new('RGB',(thumb_w*2,(thumb_h+label_h)*4),'white')
draw=ImageDraw.Draw(canvas)
try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',24)
except OSError: font=ImageFont.load_default()
for i,(city,route,img) in enumerate(items):
    img.thumbnail((thumb_w,thumb_h))
    x=(i%2)*thumb_w+(thumb_w-img.width)//2
    y=(i//2)*(thumb_h+label_h)+label_h+(thumb_h-img.height)//2
    canvas.paste(img,(x,y))
    draw.text(((i%2)*thumb_w+18,(i//2)*(thumb_h+label_h)+8),f'{city.upper()}  {route}',fill='black',font=font)
canvas.save(root/'all4_results_contact_sheet.jpg',quality=95)

print('================ ALL-4 MEASURED SUMMARY ================')
for r in rows:
    print(f"{r['city']} {r['route']}: MLE={r['MLE_m']:.3f}m P90={r['P90_m']:.3f}m LSR@15={r['LSR@15_pct']:.2f}%")
print(f"ALL 8: MLE={overall['MLE_m']:.3f}m P90={overall['P90_m']:.3f}m LSR@15={overall['LSR@15_pct']:.2f}%")
print('Contact sheet:',root/'all4_results_contact_sheet.jpg')
PY

# Write an easy local pointer.
printf '%s\n' "${LOCAL_ROOT}" > "${REPO_ROOT}/v39_otherdata/generated/LATEST_ALL4_SOFTMS.txt"

# Compact paper bundle for GitHub: figures + summaries + audits + frame CSVs.
BUNDLE="${LOCAL_ROOT}/github_bundle"
mkdir -p "${BUNDLE}"
cp "${LOCAL_ROOT}/paper_all4_summary.json" "${BUNDLE}/"
cp "${LOCAL_ROOT}/paper_all4_summary.csv" "${BUNDLE}/"
cp "${LOCAL_ROOT}/all4_results_contact_sheet.jpg" "${BUNDLE}/"
for city in citya cityb cityc cityd; do
  mkdir -p "${BUNDLE}/${city}/paper_figures_waypoint_gt"
  cp "${LOCAL_ROOT}/${city}/bearing_v39_summary.json" "${BUNDLE}/${city}/"
  cp "${LOCAL_ROOT}/${city}/bearing_paper_metrics.json" "${BUNDLE}/${city}/"
  cp "${LOCAL_ROOT}/${city}/bearing_paper_metrics.csv" "${BUNDLE}/${city}/"
  cp "${LOCAL_ROOT}/${city}/final_quality_audit.json" "${BUNDLE}/${city}/"
  cp "${LOCAL_ROOT}/${city}/v39_bearing_training_audit.json" "${BUNDLE}/${city}/"
  cp "${LOCAL_ROOT}/${city}/experiment.json" "${BUNDLE}/${city}/"
  cp "${LOCAL_ROOT}/${city}/test_01_final_result.jpg" "${BUNDLE}/${city}/"
  cp "${LOCAL_ROOT}/${city}/test_02_final_result.jpg" "${BUNDLE}/${city}/"
  cp "${LOCAL_ROOT}/${city}"/*_frames.csv "${BUNDLE}/${city}/" 2>/dev/null || true
  cp -a "${LOCAL_ROOT}/${city}/paper_figures_waypoint_gt/." "${BUNDLE}/${city}/paper_figures_waypoint_gt/"
done
cat > "${BUNDLE}/README.txt" <<EOF
Bearing-UAV all-four-city measured rerun.
Architecture: Forward18 SoftMS -> 3-frame Context-GRU -> fixed-R Kalman -> final 6x6 Soft MeanShift.
Physical adapter source: ${HIST_COMMIT}.
Preserved adapter: step=4m; metre-matched SAT geometry; Route-A-only cadence adaptation; final 6x6/BW7; route-centerline final-MS reference.
Plotter: current v39_otherdata/bearing_plot_final_vs_gt.py.
Green solid = sparse waypoint GT. Red solid = raw final_x/final_y prediction. No prediction display smoothing.
All metric values are measured outputs and are not edited to force a ranking.
EOF

# Safe automatic upload from a clean worktree. Current user's dirty working tree is untouched.
git worktree add -b "${UPLOAD_BRANCH}" "${UPLOAD_WT}" origin/v39_otherdata
mkdir -p "${UPLOAD_WT}/${UPLOAD_DEST}"
cp -a "${BUNDLE}/." "${UPLOAD_WT}/${UPLOAD_DEST}/"
(
  cd "${UPLOAD_WT}"
  git add "${UPLOAD_DEST}"
  git commit -m "Add all-city Bearing SoftMS figures and measured results"
  git fetch origin v39_otherdata
  git rebase origin/v39_otherdata
  git push origin HEAD:v39_otherdata
)

echo "================================================================================"
echo "ALL DONE"
echo "LOCAL RESULTS : ${LOCAL_ROOT}"
echo "CONTACT SHEET : ${LOCAL_ROOT}/all4_results_contact_sheet.jpg"
echo "SUMMARY JSON  : ${LOCAL_ROOT}/paper_all4_summary.json"
echo "SUMMARY CSV   : ${LOCAL_ROOT}/paper_all4_summary.csv"
echo "CITY FIGURES  : ${LOCAL_ROOT}/{citya,cityb,cityc,cityd}/paper_figures_waypoint_gt/"
echo "GITHUB PATH   : ${UPLOAD_DEST}"
echo "Pointer file  : v39_otherdata/generated/LATEST_ALL4_SOFTMS.txt"
echo "================================================================================"
