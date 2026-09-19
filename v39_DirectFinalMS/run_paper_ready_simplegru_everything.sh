#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
SOURCE_RUN="${ROOT}/run.sh"
FIGURE_PATCH="${ROOT}/patch_simple_figure_gru.py"
BEARING_RUNNER="${REPO_ROOT}/v39_otherdata/bearing_runner.py"
TS="$(date +%Y%m%d_%H%M%S)"
SUITE_ROOT="${EXPERIMENT_SUITE_DIR:-${ROOT}/paper_ready_simplegru_${TS}}"
MAIN_ROOT="${SUITE_ROOT}/main_ablation"
LOG_ROOT="${SUITE_ROOT}/launcher_logs"
BUNDLE="${SUITE_ROOT}/paper_bundle"
ARCH="V39_SimpleFigureGRU_SoftMS18_Kalman_FinalMS"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
FEATURE_CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache}"
VISUAL_CKPT="${REPO_ROOT}/forNX/weights/v36_mobilenet_v3_small/checkpoints/visual_retrieval_A_only.pt"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-80}"
PATIENCE="${PATIENCE:-15}"
JITTER_M="${JITTER_M:-8}"
MAIN_MS_GRID="${MAIN_MS_GRID:-6}"
MAIN_MS_BW="${MAIN_MS_BW:-7.0}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"

mkdir -p "${SUITE_ROOT}" "${LOG_ROOT}" "${BUNDLE}"
[[ -f "${SOURCE_RUN}" ]] || { echo "ERROR: missing ${SOURCE_RUN}" >&2; exit 2; }
[[ -f "${FIGURE_PATCH}" ]] || { echo "ERROR: missing ${FIGURE_PATCH}" >&2; exit 2; }
[[ -f "${BEARING_RUNNER}" ]] || { echo "ERROR: missing ${BEARING_RUNNER}" >&2; exit 2; }
[[ -s "${VISUAL_CKPT}" ]] || { echo "ERROR: missing ${VISUAL_CKPT}" >&2; exit 2; }

PATCHED_RUN="${ROOT}/.run_paper_ready_${TS}_$$.sh"
PATCHED_BEARING="${REPO_ROOT}/v39_otherdata/.bearing_paper_ready_${TS}_$$.py"
UPLOAD_WORKTREE=""
UPLOAD_BRANCH=""
cleanup() {
  rm -f "${PATCHED_RUN}" "${PATCHED_BEARING}"
  if [[ -n "${UPLOAD_WORKTREE}" && -d "${UPLOAD_WORKTREE}" ]]; then
    git -C "${REPO_ROOT}" worktree remove --force "${UPLOAD_WORKTREE}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${UPLOAD_BRANCH}" ]]; then
    git -C "${REPO_ROOT}" branch -D "${UPLOAD_BRANCH}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

python3 - "${SOURCE_RUN}" "${PATCHED_RUN}" "${FIGURE_PATCH}" "${ARCH}" "${TEMPORAL_EPOCHS}" "${PATIENCE}" <<'PY'
from pathlib import Path
import sys
src=Path(sys.argv[1]).read_text(encoding='utf-8')
patch=Path(sys.argv[3]).resolve()
arch=sys.argv[4]
epochs=sys.argv[5]
patience=sys.argv[6]
s=src
s=s.replace('weighted_centroid','softms')
s=s.replace('V39_WeightedCentroid_GRU_Kalman_MS',arch)
s=s.replace('BASE_ARCH="V36_PreviousStateOnly_MobileNetV3_Forward3x6_PolynomialKalman"',f'BASE_ARCH="{arch}"')
s=s.replace('TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"',f'TEMPORAL_EPOCHS="${{TEMPORAL_EPOCHS:-{epochs}}}"')
s=s.replace('PATIENCE="${PATIENCE:-10}"',f'PATIENCE="${{PATIENCE:-{patience}}}"')
s=s.replace('Weighted Centroid','Front SoftMS(18)')
s=s.replace('wc_final_ablation_','paper_simplegru_ablation_')
s=s.replace('abl_wc_','abl_ms_')
needle='  python3 "${ROOT}/patch_direct_finalms.py" "${runtime}/robust_tracker.py"\n'
insert=needle+f'  python3 "{patch}" "${{runtime}}/visual_model.py"\n'
if s.count(needle) != 1:
    raise SystemExit('ERROR: make_runtime patch point did not match exactly once')
s=s.replace(needle,insert,1)
s=s.replace("d['final_chain']='Weighted Centroid -> GRU -> fixed-R Kalman -> one final MS -> Final Position'",
            "d['final_chain']='Forward18 SoftMS -> simple figure GRU -> fixed-R Kalman -> one final MS -> Final Position'")
s=s.replace("d['training_definition']='original v39 training code/hyperparameters; only front decoder is Weighted Centroid'",
            "d['training_definition']='V39 temporal training with figure-aligned direct GRU inputs; no split/residual gates'")
for bad in ('patch_gru_gate_eval.py','patch_gru_dualgate_eval.py','patch_split_gru_gate_calibrated.py','patch_temporal_fusion_v2.py'):
    if bad in s:
        raise SystemExit('ERROR: forbidden gate/fusion patch referenced: '+bad)
if 'weighted_centroid' in s:
    raise SystemExit('ERROR: weighted_centroid remains in generated main runner')
Path(sys.argv[2]).write_text(s,encoding='utf-8')
PY
chmod +x "${PATCHED_RUN}"
bash -n "${PATCHED_RUN}"

python3 - "${BEARING_RUNNER}" "${PATCHED_BEARING}" "${FIGURE_PATCH}" "${ARCH}" "${MAIN_MS_GRID}" <<'PY'
from pathlib import Path
import sys
src=Path(sys.argv[1]).read_text(encoding='utf-8')
patch=Path(sys.argv[3]).resolve()
arch=sys.argv[4]
grid=sys.argv[5]
s=src.replace('weighted_centroid','softms')
s=s.replace('V39_Forward3x6_ContextGRU_FixedKalman_FinalMS5x5_BearingUAV',arch+'_BearingUAV')
s=s.replace('CANONICAL_FINALMS_PATCH = CANONICAL_ROOT / "patch_direct_finalms.py"',
            'CANONICAL_FINALMS_PATCH = CANONICAL_ROOT / "patch_direct_finalms.py"\nSIMPLE_FIGURE_GRU_PATCH = Path('+repr(str(patch))+')')
needle='    _patch_context_gru(runtime)\n'
insert=needle+'    subprocess.run([sys.executable, str(SIMPLE_FIGURE_GRU_PATCH), str(runtime / "visual_model.py")], check=True)\n'
if s.count(needle) != 1:
    raise SystemExit('ERROR: Bearing GRU patch point did not match exactly once')
s=s.replace(needle,insert,1)
s=s.replace('"MS_GRID_SIZE": "5"',f'"MS_GRID_SIZE": "{grid}"')
s=s.replace('"self.gru = nn.GRUCell(feature_dim * 5, hidden_dim)" not in model_text',
            '"self.gru = nn.GRUCell(feature_dim * 6, hidden_dim)" not in model_text')
s=s.replace('"final_ms_grid": 5',f'"final_ms_grid": {grid}')
for bad in ('split_gate','dualgate','dual_gate','patch_gru_gate_eval','patch_gru_dualgate_eval'):
    if bad in s.lower():
        raise SystemExit('ERROR: forbidden gate logic found in generated Bearing runner: '+bad)
if 'weighted_centroid' in s:
    raise SystemExit('ERROR: weighted_centroid remains in generated Bearing runner')
Path(sys.argv[2]).write_text(s,encoding='utf-8')
compile(s,str(sys.argv[2]),'exec')
PY

cat > "${SUITE_ROOT}/experiment_definition.txt" <<EOF2
Paper-ready measured experiment; results are never edited or reordered.
Architecture: Forward 3x6 = 18 scored patches -> SoftMS visual position -> one simple GRU -> fixed-R Kalman -> final ${MAIN_MS_GRID}x${MAIN_MS_GRID} MeanShift -> XY
GRU direct inputs: temporal mean + first difference + second difference + SAT context + current Front-SoftMS visual position + previous recurrent state
Forbidden: split gate / dual gate / residual gate / position innovation
Training: Route A only; B/C are evaluation only
Temporal ablation: one trained 3-frame checkpoint; older context is masked at inference to isolate temporal-context contribution
GPU pool: 0,5,6
EOF2

printf '\n===== PHASE 1: train/evaluate base suite =====\n'
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS}" \
PATIENCE="${PATIENCE}" \
RUN_ALL_EXPERIMENTS=1 \
EXPERIMENT_SUITE_DIR="${MAIN_ROOT}" \
bash "${PATCHED_RUN}" 2>&1 | tee "${LOG_ROOT}/main_base_suite.log"

CKPT3="${MAIN_ROOT}/temporal_3frame/checkpoints/${CKPT_NAME}"
[[ -s "${CKPT3}" ]] || { echo "ERROR: missing 3-frame checkpoint ${CKPT3}" >&2; exit 20; }
RUNTIME3="${MAIN_ROOT}/runtime_temporal_3frame"
[[ -f "${RUNTIME3}/robust_tracker.py" ]] || { echo "ERROR: missing ${RUNTIME3}" >&2; exit 20; }

eval_cfg() {
  local gpu="$1" name="$2" frames="$3" kalman="$4" disable_gru="$5" ms_enabled="$6" grid="$7"
  local out="${MAIN_ROOT}/${name}"
  mkdir -p "${out}/checkpoints"
  ln -sfn "${VISUAL_CKPT}" "${out}/checkpoints/visual_retrieval_A_only.pt"
  if [[ "${disable_gru}" == "0" ]]; then
    ln -sfn "${CKPT3}" "${out}/checkpoints/${CKPT_NAME}"
  fi
  echo "[EVAL START] ${name} GPU${gpu} frames=${frames} kalman=${kalman} disable_gru=${disable_gru} ms=${ms_enabled} grid=${grid}"
  (
    cd "${RUNTIME3}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    UAVSAT_DEVICE=cuda:0 \
    UAVSAT_OUTPUT_DIR="${out}" \
    UAVSAT_CHECKPOINT_DIR="${out}/checkpoints" \
    UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR}" \
    UAVSAT_DATA_ROOT="${DATA_ROOT}" \
    UAVSAT_BACKBONE=mobilenet_v3_small \
    UAVSAT_ARCHITECTURE_NAME="${ARCH}" \
    UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
    UAVSAT_EXPERIMENT_ANCHOR=softms \
    UAVSAT_EXPERIMENT_FRAME_COUNT="${frames}" \
    UAVSAT_EXPERIMENT_MOTION=velocity \
    UAVSAT_EXPERIMENT_KALMAN="${kalman}" \
    UAVSAT_EXPERIMENT_DISABLE_GRU="${disable_gru}" \
    UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
    MS_ENABLED="${ms_enabled}" \
    MS_GRID_SIZE="${grid}" \
    MS_BANDWIDTH_M="${MAIN_MS_BW}" \
    python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m "${JITTER_M}" \
      2>&1 | tee "${out}/eval.log"
  )
  echo "[EVAL DONE] ${name}"
}

printf '\n===== PHASE 2: fair removal + temporal-context ablations =====\n'
( eval_cfg 0 context_1frame 1 fixed 0 1 "${MAIN_MS_GRID}"; \
  eval_cfg 0 context_2frame 2 fixed 0 1 "${MAIN_MS_GRID}"; \
  eval_cfg 0 context_3frame 3 fixed 0 1 "${MAIN_MS_GRID}" ) & q0=$!
( eval_cfg 5 removal_no_gru    3 fixed 1 1 "${MAIN_MS_GRID}"; \
  eval_cfg 5 removal_no_final  3 fixed 0 0 "${MAIN_MS_GRID}" ) & q5=$!
( eval_cfg 6 removal_no_kalman 3 none  0 1 "${MAIN_MS_GRID}" ) & q6=$!
status=0
wait "${q0}" || status=1
wait "${q5}" || status=1
wait "${q6}" || status=1
[[ "${status}" == "0" ]] || { echo "ERROR: paper ablation evaluation failed" >&2; exit 21; }

printf '\n===== PHASE 3: build paper tables from measured results =====\n'
python3 - "${MAIN_ROOT}" "${SUITE_ROOT}" <<'PY'
import json,sys
from pathlib import Path
main=Path(sys.argv[1]); suite=Path(sys.argv[2]); NB,NC=2276,1258

def load(name):
    p=main/name/'robust_tracker_summary.json'
    if not p.exists(): raise SystemExit(f'missing {p}')
    return json.loads(p.read_text(encoding='utf-8'))

def w(b,c): return (float(b)*NB+float(c)*NC)/(NB+NC)
def bc(d,key): return w(d['route_B'][key],d['route_C'][key])
def row(d): return dict(mle=bc(d,'MLE_m'),p90=bc(d,'P90_m'),lsr5=bc(d,'LSR@5_pct'))

full=load('context_3frame')
comp={'Full':full,'w/o GRU':load('removal_no_gru'),'w/o Kalman':load('removal_no_kalman'),'w/o Final MS':load('removal_no_final')}
temporal={str(i):load(f'context_{i}frame') for i in (1,2,3)}
windows={str(i):load(f'sens_ms_grid{i}x{i}') for i in range(4,9)}
runtime=load('runtime_e2e')
lines=['## Table 1. Component removal ablation','','| Setting | B+C MLE | B+C P90 | B+C LSR@5 |','|---|---:|---:|---:|']
for name,d in comp.items():
    r=row(d); lines.append(f'| {name} | {r["mle"]:.3f} | {r["p90"]:.3f} | {r["lsr5"]:.2f}% |')
lines += ['','## Table 2. Temporal-context input ablation','','All rows reuse the same 3-frame-trained checkpoint; older temporal context is masked at inference.','','| Frames | B+C MLE | B+C P90 | B+C LSR@5 |','|---:|---:|---:|---:|']
for name,d in temporal.items():
    r=row(d); lines.append(f'| {name} | {r["mle"]:.3f} | {r["p90"]:.3f} | {r["lsr5"]:.2f}% |')
lines += ['','## Table 3. Final MeanShift window sensitivity','','| Window | Candidates | B+C MLE | B+C P90 | B+C LSR@5 |','|---|---:|---:|---:|---:|']
for i in range(4,9):
    r=row(windows[str(i)]); lines.append(f'| {i}x{i} | {i*i} | {r["mle"]:.3f} | {r["p90"]:.3f} | {r["lsr5"]:.2f}% |')
rb=runtime['route_B'].get('EndToEndTiming',{}); rc=runtime['route_C'].get('EndToEndTiming',{})
lines += ['','## Table 4. End-to-end runtime','',f'- Route B: {float(rb.get("mean_ms",float("nan"))):.3f} ms/frame',f'- Route C: {float(rc.get("mean_ms",float("nan"))):.3f} ms/frame']
text='\n'.join(lines)+'\n'; (suite/'paper_tables.md').write_text(text,encoding='utf-8')
fr=row(full); crows={k:row(v) for k,v in comp.items() if k!='Full'}; trows={k:row(v) for k,v in temporal.items()}
audit={'full':fr,'component_full_best_mle':all(fr['mle']<=r['mle']+1e-12 for r in crows.values()),'component_full_best_p90':all(fr['p90']<=r['p90']+1e-12 for r in crows.values()),'component_full_best_lsr5':all(fr['lsr5']>=r['lsr5']-1e-12 for r in crows.values()),'three_frame_best_mle':all(trows['3']['mle']<=r['mle']+1e-12 for k,r in trows.items() if k!='3'),'three_frame_best_p90':all(trows['3']['p90']<=r['p90']+1e-12 for k,r in trows.items() if k!='3'),'three_frame_best_lsr5':all(trows['3']['lsr5']>=r['lsr5']-1e-12 for k,r in trows.items() if k!='3'),'integrity':'Measured results only; no metric value is edited to force a ranking.'}
audit['PAPER_TREND_CHECK']='PASS' if all(v for k,v in audit.items() if k.startswith(('component_','three_frame_'))) else 'FAIL'
(suite/'paper_trend_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(text); print(json.dumps(audit,indent=2))
PY

printf '\n===== PHASE 4: Bearing-UAV city A/B/C/D on GPU0/5/6 =====\n'
BEARING_DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
BEARING_VISUAL_EPOCHS="${BEARING_VISUAL_EPOCHS:-30}"
BEARING_EPOCHS_PER_ROUTE="${BEARING_EPOCHS_PER_ROUTE:-20}"
BEARING_PATIENCE="${BEARING_PATIENCE:-10}"
BEARING_JITTER_M="${BEARING_JITTER_M:-8}"
run_city() {
  local city="$1" gpu="$2"
  echo "[BEARING START] ${city} GPU${gpu}"
  python3 -u "${PATCHED_BEARING}" --dataset-root "${BEARING_DATASET_ROOT}" --city "${city}" --gpu "${gpu}" --visual-epochs "${BEARING_VISUAL_EPOCHS}" --epochs-per-route "${BEARING_EPOCHS_PER_ROUTE}" --patience "${BEARING_PATIENCE}" --jitter-m "${BEARING_JITTER_M}" 2>&1 | tee "${LOG_ROOT}/bearing_${city}.log"
  echo "[BEARING DONE] ${city}"
}
( run_city citya 0; run_city cityd 0 ) & b0=$!
( run_city cityb 5 ) & b5=$!
( run_city cityc 6 ) & b6=$!
status=0
wait "${b0}" || status=1
wait "${b5}" || status=1
wait "${b6}" || status=1
[[ "${status}" == "0" ]] || { echo "ERROR: Bearing run failed" >&2; exit 30; }

printf '\n===== PHASE 5: collect compact paper bundle =====\n'
cp "${SUITE_ROOT}/experiment_definition.txt" "${BUNDLE}/"
cp "${SUITE_ROOT}/paper_tables.md" "${BUNDLE}/"
cp "${SUITE_ROOT}/paper_trend_audit.json" "${BUNDLE}/"
for name in context_3frame removal_no_gru removal_no_kalman removal_no_final context_1frame context_2frame; do
  mkdir -p "${BUNDLE}/main/${name}"
  cp "${MAIN_ROOT}/${name}/robust_tracker_summary.json" "${BUNDLE}/main/${name}/"
done
for i in 4 5 6 7 8; do
  name="sens_ms_grid${i}x${i}"
  mkdir -p "${BUNDLE}/main/${name}"
  cp "${MAIN_ROOT}/${name}/robust_tracker_summary.json" "${BUNDLE}/main/${name}/"
done
mkdir -p "${BUNDLE}/bearing"
for city in citya cityb cityc cityd; do
  f="${REPO_ROOT}/v39_otherdata/generated/${city}/v39_output_corrected/bearing_v39_summary.json"
  a="${REPO_ROOT}/v39_otherdata/generated/${city}/v39_output_corrected/v39_bearing_training_audit.json"
  [[ -f "${f}" ]] && cp "${f}" "${BUNDLE}/bearing/${city}_summary.json"
  [[ -f "${a}" ]] && cp "${a}" "${BUNDLE}/bearing/${city}_audit.json"
done

if [[ "${UPLOAD_RESULTS}" == "1" ]]; then
  printf '\n===== PHASE 6: upload compact results to GitHub =====\n'
  cd "${REPO_ROOT}"
  git fetch origin v39_otherdata
  UPLOAD_WORKTREE="/tmp/uav-sat-paper-upload-${TS}-$$"
  UPLOAD_BRANCH="paper-upload-${TS}-$$"
  git worktree add -b "${UPLOAD_BRANCH}" "${UPLOAD_WORKTREE}" origin/v39_otherdata
  DEST_REL="paper_results/simple_figure_softms18_${TS}"
  mkdir -p "${UPLOAD_WORKTREE}/${DEST_REL}"
  cp -a "${BUNDLE}/." "${UPLOAD_WORKTREE}/${DEST_REL}/"
  (
    cd "${UPLOAD_WORKTREE}"
    git add "${DEST_REL}"
    git commit -m "Add simple-GRU SoftMS18 paper experiment ${TS}"
    if ! git push origin HEAD:v39_otherdata; then
      echo "[UPLOAD] remote advanced; rebasing clean upload commit and retrying"
      git fetch origin v39_otherdata
      git rebase origin/v39_otherdata
      git push origin HEAD:v39_otherdata
    fi
  )
  echo "[UPLOAD OK] ${DEST_REL}"
fi

printf '\n====================================================================================================\n'
echo "DONE"
echo "Suite       : ${SUITE_ROOT}"
echo "Paper table : ${SUITE_ROOT}/paper_tables.md"
echo "Trend audit : ${SUITE_ROOT}/paper_trend_audit.json"
echo "Bundle      : ${BUNDLE}"
if [[ "${UPLOAD_RESULTS}" == "1" ]]; then echo "GitHub      : paper_results/simple_figure_softms18_${TS}"; fi
echo "===================================================================================================="
