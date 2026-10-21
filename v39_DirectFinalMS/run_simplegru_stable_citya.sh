#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
MAIN_RUNNER="${ROOT}/run_paper_ready_simplegru_everything.sh"
PATCH="${ROOT}/patch_simple_figure_gru.py"
MAIN_SUITE="${ROOT}/stable_simplegru_${TS}"
TMP_MAIN="${ROOT}/.stable_main_${TS}_$$.sh"
HIST_COMMIT="9bb0ae28400d783535430b54f9c3417feba53103"
HIST_WT="${REPO_ROOT%/*}/uav-sat-bearing-hist-${TS}"
UPLOAD_WT="${REPO_ROOT%/*}/uav-sat-upload-stable-${TS}"
UPLOAD_BRANCH="upload-stable-${TS}"
RESULT_DEST="paper_results/stable_simplegru_citya_${TS}"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"

cleanup(){
  rm -f "${TMP_MAIN}" || true
  if [[ -d "${HIST_WT}" ]]; then git -C "${REPO_ROOT}" worktree remove --force "${HIST_WT}" >/dev/null 2>&1 || true; fi
  if [[ -d "${UPLOAD_WT}" ]]; then git -C "${REPO_ROOT}" worktree remove --force "${UPLOAD_WT}" >/dev/null 2>&1 || true; fi
  git -C "${REPO_ROOT}" branch -D "${UPLOAD_BRANCH}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

[[ -s "${MAIN_RUNNER}" ]] || { echo "ERROR missing ${MAIN_RUNNER}" >&2; exit 2; }
[[ -s "${PATCH}" ]] || { echo "ERROR missing ${PATCH}" >&2; exit 2; }
python3 -m py_compile "${PATCH}"

# -----------------------------------------------------------------------------
# MAIN EXPERIMENT
# Use the current one-command paper runner, but stop before its broken generic
# Bearing phase.  The updated simple-GRU patch is applied inside every runtime.
# -----------------------------------------------------------------------------
python3 - "${MAIN_RUNNER}" "${TMP_MAIN}" <<'PY'
from pathlib import Path
import sys
src=Path(sys.argv[1]).read_text(encoding='utf-8')
marker="printf '\\n===== PHASE 4: Bearing-UAV city A/B/C/D on GPU0/5/6 =====\\n'"
pos=src.find(marker)
if pos < 0:
    # tolerate literal-newline variant
    pos=src.find('===== PHASE 4: Bearing-UAV city A/B/C/D on GPU0/5/6 =====')
    if pos < 0: raise SystemExit('cannot locate Bearing phase marker')
    pos=src.rfind('\n',0,pos)
Path(sys.argv[2]).write_text(src[:pos]+'\necho "[MAIN] measured main suite complete"\n',encoding='utf-8')
PY
chmod +x "${TMP_MAIN}"
bash -n "${TMP_MAIN}"

echo "================================================================================"
echo "PHASE A: STABLE SIMPLE-GRU MAIN EXPERIMENT"
echo "No split/dual gate. Forward18 SoftMS -> simple 3-frame GRU -> fixed Kalman -> final 6x6 MS."
echo "Patience=4; Route A training only; B/C evaluation only."
echo "================================================================================"
UPLOAD_RESULTS=0 \
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-80}" \
PATIENCE="${PATIENCE:-4}" \
MAIN_MS_GRID=6 \
EXPERIMENT_SUITE_DIR="${MAIN_SUITE}" \
bash "${TMP_MAIN}"

[[ -s "${MAIN_SUITE}/paper_tables.md" ]] || { echo "ERROR main paper table missing" >&2; exit 20; }
[[ -s "${MAIN_SUITE}/paper_trend_audit.json" ]] || { echo "ERROR main audit missing" >&2; exit 20; }

# -----------------------------------------------------------------------------
# BEARING cityA
# The previous 50-60m run used the generic Bearing runner and ignored the
# historically validated physical adapter. Re-run the verified low-error path
# from commit 9bb0ae2 in an isolated detached worktree, changing ONLY the front
# decoder from weighted centroid to SoftMS and patience 10 -> 4.
# It retains: step=4m, metre-matched SAT geometry, Route-A-only cadence scaling,
# fixed-R Kalman, final 6x6 MS and route-centerline final-MS reference.
# -----------------------------------------------------------------------------
echo "================================================================================"
echo "PHASE B: CORRECTED BEARING-UAV cityA"
echo "Historical physical adapter: ${HIST_COMMIT}"
echo "Only method change: front weighted centroid -> Forward18 SoftMS"
echo "GPU6; cityA only"
echo "================================================================================"

git fetch origin v39_otherdata
git worktree add --detach "${HIST_WT}" "${HIST_COMMIT}"

python3 - "${HIST_WT}/v39_otherdata/bearing_runner.py" "${HIST_WT}/v39_otherdata/bearing_runner_exact_v39.py" "${HIST_WT}/v39_otherdata/run_bearing_v39_sequence_fixed.sh" <<'PY'
from pathlib import Path
import sys
base=Path(sys.argv[1]); exact=Path(sys.argv[2]); sh=Path(sys.argv[3])

s=base.read_text(encoding='utf-8')
s=s.replace('"UAVSAT_EXPERIMENT_ANCHOR": "weighted_centroid"','"UAVSAT_EXPERIMENT_ANCHOR": "softms"')
s=s.replace('"EXPERIMENT_ANCHOR": "weighted_centroid"','"EXPERIMENT_ANCHOR": "softms"')
s=s.replace('"visual_decoder": "weighted_centroid"','"visual_decoder": "softms"')
base.write_text(s,encoding='utf-8'); compile(s,str(base),'exec')

s=exact.read_text(encoding='utf-8')
s=s.replace('"UAVSAT_EXPERIMENT_ANCHOR": "weighted_centroid"','"UAVSAT_EXPERIMENT_ANCHOR": "softms"')
s=s.replace('"visual_decoder": "weighted_centroid"','"visual_decoder": "softms"')
s=s.replace('Weighted Centroid -> 3-frame Context-GRU','Forward18 SoftMS -> 3-frame Context-GRU')
exact.write_text(s,encoding='utf-8'); compile(s,str(exact),'exec')

s=sh.read_text(encoding='utf-8')
s=s.replace('--patience 10 \\', '--patience 4 \\')
sh.write_text(s,encoding='utf-8')
PY

(
  cd "${HIST_WT}"
  CITY=citya GPU=6 DATASET_ROOT="${DATASET_ROOT}" \
    bash v39_otherdata/run_bearing_v39_directfinalms_official_routes.sh
)

CITY_OUT="${HIST_WT}/v39_otherdata/generated/citya/v39_output_bearing_adapted"
[[ -s "${CITY_OUT}/bearing_v39_summary.json" ]] || { echo "ERROR Bearing cityA summary missing" >&2; exit 30; }

# Make a transparent numeric audit from the measured cityA result.
python3 - "${CITY_OUT}/bearing_v39_summary.json" "${MAIN_SUITE}/bearing_citya_check.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
r={k:{m:s[k][m] for m in ('MLE_m','MedLE_m','P90_m','LSR@5_pct','LSR@10_pct','LSR@15_pct','LSR@20_pct')} for k in ('test_01','test_02')}
r['sanity_check_under_10m_both_routes']=all(r[k]['MLE_m']<10.0 for k in ('test_01','test_02'))
r['note']='Measured values only. This check detects the 50-60m runner/protocol failure; it does not edit metrics.'
json.dump(r,open(sys.argv[2],'w',encoding='utf-8'),indent=2)
print(json.dumps(r,indent=2))
PY

# -----------------------------------------------------------------------------
# PACKAGE + SAFE AUTO-UPLOAD
# -----------------------------------------------------------------------------
BUNDLE="${MAIN_SUITE}/github_bundle"
mkdir -p "${BUNDLE}/main" "${BUNDLE}/bearing/citya"
cp "${MAIN_SUITE}/paper_tables.md" "${BUNDLE}/"
cp "${MAIN_SUITE}/paper_trend_audit.json" "${BUNDLE}/"
cp "${MAIN_SUITE}/experiment_definition.txt" "${BUNDLE}/" 2>/dev/null || true
cp "${MAIN_SUITE}/bearing_citya_check.json" "${BUNDLE}/bearing/"

for d in context_1frame context_2frame context_3frame removal_no_gru removal_no_kalman removal_no_final sens_ms_grid4x4 sens_ms_grid5x5 sens_ms_grid6x6 sens_ms_grid7x7 sens_ms_grid8x8 runtime_e2e; do
  if [[ -s "${MAIN_SUITE}/main_ablation/${d}/robust_tracker_summary.json" ]]; then
    mkdir -p "${BUNDLE}/main/${d}"
    cp "${MAIN_SUITE}/main_ablation/${d}/robust_tracker_summary.json" "${BUNDLE}/main/${d}/"
  fi
done

for f in bearing_v39_summary.json bearing_paper_metrics.json bearing_paper_metrics.csv final_quality_audit.json v39_bearing_training_audit.json; do
  [[ -s "${CITY_OUT}/${f}" ]] && cp "${CITY_OUT}/${f}" "${BUNDLE}/bearing/citya/${f}"
done
[[ -s "${HIST_WT}/v39_otherdata/generated/citya/experiment.json" ]] && cp "${HIST_WT}/v39_otherdata/generated/citya/experiment.json" "${BUNDLE}/bearing/citya/experiment.json"

# Add a precise provenance note.
cat > "${BUNDLE}/README.txt" <<EOF
Main model: current v39 simple figure GRU with conservative fixed residual bounds; no split/dual/inference gates.
Main protocol: Route A training only, Route B/C evaluation only, Forward18 SoftMS, final 6x6 MS.
Bearing cityA: historical verified physical adapter from ${HIST_COMMIT}, with only front decoder changed to SoftMS and patience changed to 4.
Bearing keeps step=4m, metre-matched SAT geometry, Route-A-only cadence adaptation, fixed Kalman, final 6x6 MS, centerline final-MS reference.
All uploaded metrics are measured outputs; no value is edited to force a ranking.
EOF

git worktree add -b "${UPLOAD_BRANCH}" "${UPLOAD_WT}" origin/v39_otherdata
mkdir -p "${UPLOAD_WT}/${RESULT_DEST}"
cp -a "${BUNDLE}/." "${UPLOAD_WT}/${RESULT_DEST}/"
(
  cd "${UPLOAD_WT}"
  git add "${RESULT_DEST}"
  git commit -m "Add stable simple-GRU and corrected Bearing cityA results"
  git fetch origin v39_otherdata
  git rebase origin/v39_otherdata
  git push origin HEAD:v39_otherdata
)

echo "================================================================================"
echo "ALL DONE"
echo "Main results : ${MAIN_SUITE}/paper_tables.md"
echo "Main audit   : ${MAIN_SUITE}/paper_trend_audit.json"
echo "Bearing A    : ${CITY_OUT}/bearing_v39_summary.json"
echo "GitHub path  : ${RESULT_DEST}"
echo "Reply only: 跑完了"
echo "================================================================================"
