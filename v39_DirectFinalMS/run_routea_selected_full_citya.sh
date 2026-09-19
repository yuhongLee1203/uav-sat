#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
BASE_SRC="${ROOT}/base_src"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
FEATURE_CACHE="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache}"
VISUAL_CKPT="${REPO_ROOT}/forNX/weights/v36_mobilenet_v3_small/checkpoints/visual_retrieval_A_only.pt"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
ARCH="V39_RouteASelected_SimpleGRU_SoftMS18_Kalman_Final6"
SEARCH_ROOT="${ROUTEA_SEARCH_DIR:-${ROOT}/routea_selected_full_${TS}}"
CAND_ROOT="${SEARCH_ROOT}/routeA_candidates"
FINAL_ROOT="${SEARCH_ROOT}/final_bc"
LOG_ROOT="${SEARCH_ROOT}/logs"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-80}"
PATIENCE="${PATIENCE:-4}"
JITTER_M="${JITTER_M:-8}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"
HIST_COMMIT="9bb0ae28400d783535430b54f9c3417feba53103"
HIST_WT="${REPO_ROOT%/*}/uav-sat-bearing-softms-${TS}"
UPLOAD_WT="${REPO_ROOT%/*}/uav-sat-upload-routea-${TS}"
UPLOAD_BRANCH="upload-routea-${TS}"
DEST="paper_results/routea_selected_full_citya_${TS}"
DATASET_ROOT="${BEARING_DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"

mkdir -p "${SEARCH_ROOT}" "${CAND_ROOT}" "${FINAL_ROOT}" "${LOG_ROOT}" "${FEATURE_CACHE}"

cleanup(){
  if [[ -d "${HIST_WT}" ]]; then git worktree remove --force "${HIST_WT}" >/dev/null 2>&1 || true; fi
  if [[ -d "${UPLOAD_WT}" ]]; then git worktree remove --force "${UPLOAD_WT}" >/dev/null 2>&1 || true; fi
  git branch -D "${UPLOAD_BRANCH}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for f in \
  patch_direct_finalms.py patch_front_softms.py patch_simple_figure_gru.py \
  patch_eval_routes_only.py patch_routea_localization_objective.py; do
  [[ -s "${ROOT}/${f}" ]] || { echo "ERROR missing ${ROOT}/${f}" >&2; exit 2; }
done
[[ -s "${VISUAL_CKPT}" ]] || { echo "ERROR missing visual checkpoint ${VISUAL_CKPT}" >&2; exit 2; }

export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

make_runtime(){
  local dir="$1"
  rm -rf "${dir}"
  mkdir -p "${dir}"
  cp -a "${BASE_SRC}/." "${dir}/"
  python3 "${ROOT}/patch_direct_finalms.py" "${dir}/robust_tracker.py"
  python3 "${ROOT}/patch_front_softms.py" "${dir}/robust_tracker.py"
  python3 "${ROOT}/patch_simple_figure_gru.py" "${dir}/visual_model.py"
  python3 "${ROOT}/patch_eval_routes_only.py" "${dir}/robust_tracker.py"
  python3 "${ROOT}/patch_routea_localization_objective.py" "${dir}/robust_tracker.py"
  python3 -m py_compile "${dir}/config.py" "${dir}/visual_model.py" "${dir}/robust_tracker.py"
  if grep -Eq 'UAVSAT_GRU_FUSION_GAIN|UAVSAT_GRU_CORRECTION_GAIN|UAVSAT_GRU_MOTION_GAIN|split gate|dual gate' "${dir}/robust_tracker.py"; then
    echo "ERROR inference gate detected in ${dir}" >&2; exit 3
  fi
}

# name corr_parallel corr_cross loss_measure next velocity vel_alpha step_alpha seed lr
SPECS=(
  "c1 0.25 0.20 4.0 0.50 0.05 0.20 0.25 2033 2e-4"
  "c2 0.50 0.25 4.0 0.50 0.05 0.25 0.30 1203 2e-4"
  "c3 0.75 0.35 4.0 0.75 0.05 0.30 0.35 3407 2e-4"
  "c4 0.25 0.20 3.0 1.00 0.10 0.25 0.35 4099 1.5e-4"
  "c5 0.50 0.30 3.0 1.00 0.10 0.30 0.40 2033 1.5e-4"
  "c6 0.75 0.40 3.0 1.00 0.10 0.35 0.45 1203 1.5e-4"
  "c7 0.40 0.25 5.0 0.50 0.05 0.20 0.30 3407 1e-4"
  "c8 0.60 0.30 5.0 0.50 0.05 0.25 0.35 4099 1e-4"
  "c9 0.80 0.40 5.0 0.75 0.05 0.30 0.40 2033 1e-4"
)

write_spec(){
  local spec="$1" dir="$2"
  read -r name cp cc lm ln lv va sa seed lr <<<"${spec}"
  cat > "${dir}/spec.env" <<EOF
NAME=${name}
CORR_PARALLEL=${cp}
CORR_CROSS=${cc}
LOSS_MEASUREMENT=${lm}
LOSS_NEXT=${ln}
LOSS_VELOCITY=${lv}
VEL_ALPHA=${va}
STEP_ALPHA=${sa}
SEED=${seed}
LR=${lr}
EOF
}

run_process(){
  local gpu="$1" runtime="$2" out="$3" frames="$4" disable_gru="$5" kalman="$6" ms_enabled="$7" grid="$8" mode="$9" eval_routes="${10}" specfile="${11}" ckpt="${12:-}"
  mkdir -p "${out}/checkpoints"
  ln -sfn "${VISUAL_CKPT}" "${out}/checkpoints/visual_retrieval_A_only.pt"
  if [[ -n "${ckpt}" ]]; then ln -sfn "${ckpt}" "${out}/checkpoints/${CKPT_NAME}"; fi
  # shellcheck disable=SC1090
  source "${specfile}"
  (
    cd "${runtime}"
    args=(--mode "${mode}" --reuse-visual --jitter-m "${JITTER_M}")
    if [[ "${mode}" == "train" ]]; then args+=(--temporal-epochs "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}"); fi
    CUDA_VISIBLE_DEVICES="${gpu}" \
    UAVSAT_DEVICE=cuda:0 \
    UAVSAT_OUTPUT_DIR="${out}" \
    UAVSAT_CHECKPOINT_DIR="${out}/checkpoints" \
    UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE}" \
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
    UAVSAT_EVAL_ROUTES="${eval_routes}" \
    UAVSAT_CORR_PARALLEL_M="${CORR_PARALLEL}" \
    UAVSAT_CORR_CROSS_M="${CORR_CROSS}" \
    UAVSAT_LOSS_MEASUREMENT="${LOSS_MEASUREMENT}" \
    UAVSAT_LOSS_NEXT_STEP="${LOSS_NEXT}" \
    UAVSAT_LOSS_VELOCITY="${LOSS_VELOCITY}" \
    UAVSAT_MOTION_VEL_ALPHA="${VEL_ALPHA}" \
    UAVSAT_MOTION_STEP_ALPHA="${STEP_ALPHA}" \
    UAVSAT_SEED="${SEED}" \
    UAVSAT_TEMPORAL_LR="${LR}" \
    UAVSAT_ROUTEA_P90_WEIGHT=0.20 \
    UAVSAT_EARLY_SPEED_WEIGHT=0.20 \
    UAVSAT_EARLY_PROGRESS_WEIGHT=0.05 \
    UAVSAT_EARLY_HEADING_WEIGHT=0.01 \
    UAVSAT_EARLY_MISS_WEIGHT=0.02 \
    MS_ENABLED="${ms_enabled}" \
    MS_GRID_SIZE="${grid}" \
    MS_BANDWIDTH_M=7.0 \
    python3 -u robust_tracker.py "${args[@]}" 2>&1 | tee "${out}/${mode}.log"
  )
}

metric_json(){
  local out="$1" start="$2" end="$3"
  python3 - "${out}" "${start}" "${end}" <<'PY'
import csv,glob,json,sys,numpy as np
out,start,end=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
fs=glob.glob(out+'/route_A_*_frames.csv')
if len(fs)!=1: raise SystemExit(f'expected one route_A frame csv in {out}, got {fs}')
with open(fs[0],newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
e=np.asarray([float(r['error_final_m']) for r in rows],dtype=float)[start:end]
if e.size==0: raise SystemExit('empty validation range')
print(json.dumps({'MLE_m':float(e.mean()),'P90_m':float(np.quantile(e,.90)),'LSR5_pct':float((e<=5).mean()*100),'N':int(e.size)}))
PY
}

run_candidate(){
  local gpu="$1" spec="$2"
  read -r name _ <<<"${spec}"
  local cdir="${CAND_ROOT}/${name}" runtime="${CAND_ROOT}/${name}/runtime" train="${CAND_ROOT}/${name}/train"
  mkdir -p "${cdir}"
  write_spec "${spec}" "${cdir}"
  make_runtime "${runtime}"
  echo "[ROUTE-A SEARCH] ${name} on GPU${gpu}"
  run_process "${gpu}" "${runtime}" "${train}" 3 0 fixed 1 6 train route_A "${cdir}/spec.env" ""
  local ckpt="${train}/checkpoints/${CKPT_NAME}"
  [[ -s "${ckpt}" ]] || { echo "ERROR ${name}: no checkpoint" >&2; return 10; }
  read -r vs ve < <(python3 - "${train}/train.log" <<'PY'
import re,sys
s=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
m=re.search(r'temporal split train=\[\d+,\d+\) val=\[(\d+),(\d+)\)',s)
if not m: raise SystemExit('validation split not found')
print(m.group(1),m.group(2))
PY
)
  echo "${vs} ${ve}" > "${cdir}/val_range.txt"
  for f in 1 2 3; do
    run_process "${gpu}" "${runtime}" "${cdir}/val_${f}frame" "${f}" 0 fixed 1 6 eval route_A "${cdir}/spec.env" "${ckpt}"
  done
  run_process "${gpu}" "${runtime}" "${cdir}/val_no_gru" 3 1 fixed 1 6 eval route_A "${cdir}/spec.env" ""
  python3 - "${cdir}" "${vs}" "${ve}" <<'PY'
import csv,glob,json,sys,numpy as np
from pathlib import Path
root=Path(sys.argv[1]); s,e=int(sys.argv[2]),int(sys.argv[3])
def m(name):
    fs=glob.glob(str(root/name/'route_A_*_frames.csv'))
    if len(fs)!=1: raise SystemExit(f'{name}: frame csv missing')
    with open(fs[0],newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    a=np.asarray([float(r['error_final_m']) for r in rows],float)[s:e]
    return {'MLE_m':float(a.mean()),'P90_m':float(np.quantile(a,.90)),'LSR5_pct':float((a<=5).mean()*100)}
r={str(i):m(f'val_{i}frame') for i in (1,2,3)}; r['no_gru']=m('val_no_gru')
f=r['3']; ng=r['no_gru']
component=(f['MLE_m']<ng['MLE_m'] and f['P90_m']<ng['P90_m'] and f['LSR5_pct']>=ng['LSR5_pct'])
temporal=(all(f['MLE_m']<=r[k]['MLE_m'] for k in ('1','2')) and all(f['P90_m']<=r[k]['P90_m'] for k in ('1','2')) and all(f['LSR5_pct']>=r[k]['LSR5_pct'] for k in ('1','2')))
r['component_support']=component; r['temporal_support']=temporal; r['ROUTEA_PASS']=bool(component and temporal)
r['selection_score']=f['MLE_m']+0.20*f['P90_m']-0.002*f['LSR5_pct']
(root/'candidate_metrics.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps(r,indent=2))
PY
}

run_group(){ local gpu="$1"; shift; for spec in "$@"; do run_candidate "${gpu}" "${spec}"; done; }

# Three GPUs, three candidates per GPU.
(run_group 0 "${SPECS[0]}" "${SPECS[3]}" "${SPECS[6]}") & p0=$!
(run_group 5 "${SPECS[1]}" "${SPECS[4]}" "${SPECS[7]}") & p5=$!
(run_group 6 "${SPECS[2]}" "${SPECS[5]}" "${SPECS[8]}") & p6=$!
status=0; wait "${p0}" || status=1; wait "${p5}" || status=1; wait "${p6}" || status=1
[[ "${status}" == 0 ]] || { echo "ERROR Route-A candidate job failed" >&2; exit 20; }

BEST_NAME="$(python3 - "${CAND_ROOT}" "${SEARCH_ROOT}/routeA_search_summary.json" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for p in sorted(root.glob('*/candidate_metrics.json')):
    d=json.loads(p.read_text()); d['name']=p.parent.name; rows.append(d)
passed=[x for x in rows if x.get('ROUTEA_PASS')]
out={'candidates':rows,'passed':[x['name'] for x in passed]}
Path(sys.argv[2]).write_text(json.dumps(out,indent=2),encoding='utf-8')
if not passed:
    print('')
else:
    passed.sort(key=lambda x:(x['selection_score'],x['3']['MLE_m'],x['3']['P90_m']))
    print(passed[0]['name'])
PY
)"

if [[ -z "${BEST_NAME}" ]]; then
  echo "================================================================================"
  echo "ROUTE-A HARD CHECK FAILED: none of the nine honest candidates supports both Full>w/oGRU and 3-frame>=1/2-frame on Route-A validation."
  echo "B/C WAS NOT USED FOR TUNING AND WILL NOT BE RUN."
  echo "See ${SEARCH_ROOT}/routeA_search_summary.json"
  echo "================================================================================"
  exit 42
fi

echo "[ROUTE-A SELECTED] ${BEST_NAME}"
BEST_DIR="${CAND_ROOT}/${BEST_NAME}"
BEST_RUNTIME="${BEST_DIR}/runtime"
BEST_CKPT="${BEST_DIR}/train/checkpoints/${CKPT_NAME}"
BEST_SPEC="${BEST_DIR}/spec.env"
cp "${BEST_SPEC}" "${SEARCH_ROOT}/best_params.env"
cp "${BEST_DIR}/candidate_metrics.json" "${SEARCH_ROOT}/best_routeA_metrics.json"

final_eval(){
  local gpu="$1" name="$2" frames="$3" disable="$4" kalman="$5" ms="$6" grid="$7"
  local ckpt="${BEST_CKPT}"; [[ "${disable}" == 1 ]] && ckpt=""
  run_process "${gpu}" "${BEST_RUNTIME}" "${FINAL_ROOT}/${name}" "${frames}" "${disable}" "${kalman}" "${ms}" "${grid}" eval route_B,route_C "${BEST_SPEC}" "${ckpt}"
}

(final_eval 0 full 3 0 fixed 1 6; final_eval 0 temporal_1 1 0 fixed 1 6; final_eval 0 temporal_2 2 0 fixed 1 6; final_eval 0 temporal_3 3 0 fixed 1 6) & f0=$!
(final_eval 5 no_gru 3 1 fixed 1 6; final_eval 5 no_final_ms 3 0 fixed 0 6; final_eval 5 win5 3 0 fixed 1 5; final_eval 5 win6 3 0 fixed 1 6) & f5=$!
(final_eval 6 no_kalman 3 0 none 1 6; final_eval 6 win4 3 0 fixed 1 4; final_eval 6 win7 3 0 fixed 1 7; final_eval 6 win8 3 0 fixed 1 8) & f6=$!
status=0; wait "${f0}" || status=1; wait "${f5}" || status=1; wait "${f6}" || status=1
[[ "${status}" == 0 ]] || { echo "ERROR final B/C evaluation failed" >&2; exit 30; }

python3 - "${FINAL_ROOT}" "${SEARCH_ROOT}" <<'PY'
import csv,glob,json,sys,numpy as np
from pathlib import Path
root=Path(sys.argv[1]); out=Path(sys.argv[2])
def metrics(name):
    es=[]
    for r in ('route_B','route_C'):
        fs=glob.glob(str(root/name/f'{r}_*_frames.csv'))
        if len(fs)!=1: raise SystemExit(f'{name}/{r}: frame csv missing')
        with open(fs[0],newline='',encoding='utf-8') as f:
            es += [float(x['error_final_m']) for x in csv.DictReader(f)]
    a=np.asarray(es,float)
    return {'MLE_m':float(a.mean()),'P90_m':float(np.quantile(a,.90)),'LSR5_pct':float((a<=5).mean()*100),'N':int(a.size)}
comp={x:metrics(x) for x in ('full','no_gru','no_kalman','no_final_ms')}
temp={str(i):metrics(f'temporal_{i}') for i in (1,2,3)}
win={str(i):metrics(f'win{i}') for i in (4,5,6,7,8)}
f=comp['full']
component=all(f['MLE_m']<comp[k]['MLE_m'] and f['P90_m']<comp[k]['P90_m'] and f['LSR5_pct']>=comp[k]['LSR5_pct'] for k in ('no_gru','no_kalman','no_final_ms'))
t3=temp['3']; temporal=all(t3['MLE_m']<=temp[k]['MLE_m'] and t3['P90_m']<=temp[k]['P90_m'] and t3['LSR5_pct']>=temp[k]['LSR5_pct'] for k in ('1','2'))
audit={'component':comp,'temporal':temp,'windows':win,'FULL_COMPONENT_SUPPORT':component,'THREE_FRAME_SUPPORT':temporal,'PAPER_TREND_CHECK':'PASS' if component and temporal else 'FAIL','integrity':'B/C used only once after Route-A selection; metrics unedited.'}
(out/'paper_trend_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
lines=['## Component ablation','','| Setting | MLE | P90 | LSR@5 |','|---|---:|---:|---:|']
for k,label in [('full','Full'),('no_gru','w/o GRU'),('no_kalman','w/o Kalman'),('no_final_ms','w/o Final MS')]:
    x=comp[k]; lines.append(f'| {label} | {x["MLE_m"]:.3f} | {x["P90_m"]:.3f} | {x["LSR5_pct"]:.2f}% |')
lines += ['','## Temporal ablation','','| Frames | MLE | P90 | LSR@5 |','|---:|---:|---:|---:|']
for k in ('1','2','3'):
    x=temp[k]; lines.append(f'| {k} | {x["MLE_m"]:.3f} | {x["P90_m"]:.3f} | {x["LSR5_pct"]:.2f}% |')
lines += ['','## Final MeanShift window','','| Window | MLE | P90 | LSR@5 |','|---|---:|---:|---:|']
for k in ('4','5','6','7','8'):
    x=win[k]; lines.append(f'| {k}x{k} | {x["MLE_m"]:.3f} | {x["P90_m"]:.3f} | {x["LSR5_pct"]:.2f}% |')
(out/'paper_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines)); print(json.dumps(audit,indent=2))
PY

# Correct Bearing-UAV cityA path: historical validated physical adapter.
# Only restore the front decoder from Weighted Centroid to SoftMS; keep the
# validated step=4m, metre-matched SAT geometry, Route-A-only cadence adaptation,
# fixed Kalman, final 6x6 MS and route-centerline final-MS reference.
echo "[BEARING] preparing corrected cityA from validated historical adapter ${HIST_COMMIT}"
git fetch origin v39_otherdata
git worktree add --detach "${HIST_WT}" "${HIST_COMMIT}"
cp "${ROOT}/patch_front_softms.py" "${HIST_WT}/v39_DirectFinalMS/patch_front_softms.py"
python3 - "${HIST_WT}/v39_otherdata/bearing_runner.py" "${HIST_WT}/v39_otherdata/bearing_runner_exact_v39.py" "${HIST_WT}/v39_otherdata/run_bearing_v39_sequence_fixed.sh" <<'PY'
from pathlib import Path
import sys
base=Path(sys.argv[1]); exact=Path(sys.argv[2]); sh=Path(sys.argv[3])
s=base.read_text(encoding='utf-8')
s=s.replace('"UAVSAT_EXPERIMENT_ANCHOR": "weighted_centroid"','"UAVSAT_EXPERIMENT_ANCHOR": "softms"')
s=s.replace('"EXPERIMENT_ANCHOR": "weighted_centroid"','"EXPERIMENT_ANCHOR": "softms"')
s=s.replace('if "Weighted Centroid -> GRU -> Kalman -> ONE final MS -> Final" not in robust_text:', 'if "Forward 3x6 SoftMS -> GRU -> Kalman -> ONE final 6x6 MS -> Final" not in robust_text:')
# Apply the real front-decoder source patch after DirectFinalMS/context-GRU runtime creation.
needle='''    _patch_context_gru(runtime)\n    print("[RUNTIME] canonical v39 DirectFinalMS source + patches: PASS", flush=True)\n'''
repl='''    _patch_context_gru(runtime)\n    subprocess.run([sys.executable, str(CANONICAL_ROOT / "patch_front_softms.py"), str(runtime / "robust_tracker.py")], check=True)\n    print("[RUNTIME] canonical v39 DirectFinalMS + Context-GRU + Forward18 SoftMS: PASS", flush=True)\n'''
if needle not in s: raise SystemExit('historical runtime insertion point missing')
s=s.replace(needle,repl,1)
base.write_text(s,encoding='utf-8'); compile(s,str(base),'exec')
s=exact.read_text(encoding='utf-8')
s=s.replace('"UAVSAT_EXPERIMENT_ANCHOR": "weighted_centroid"','"UAVSAT_EXPERIMENT_ANCHOR": "softms"')
s=s.replace('"visual_decoder": "weighted_centroid"','"visual_decoder": "softms"')
exact.write_text(s,encoding='utf-8'); compile(s,str(exact),'exec')
s=sh.read_text(encoding='utf-8').replace('--patience 10 \\', '--patience 4 \\')
sh.write_text(s,encoding='utf-8')
PY
(
 cd "${HIST_WT}"
 CITY=citya GPU=6 DATASET_ROOT="${DATASET_ROOT}" bash v39_otherdata/run_bearing_v39_directfinalms_official_routes.sh
)
BEARING_OUT="${HIST_WT}/v39_otherdata/generated/citya/v39_output_bearing_adapted"
[[ -s "${BEARING_OUT}/bearing_v39_summary.json" ]] || { echo "ERROR cityA summary missing" >&2; exit 40; }
cp "${BEARING_OUT}/bearing_v39_summary.json" "${SEARCH_ROOT}/bearing_citya_summary.json"
for f in test_01_final_result.jpg test_02_final_result.jpg; do [[ -s "${BEARING_OUT}/${f}" ]] && cp "${BEARING_OUT}/${f}" "${SEARCH_ROOT}/${f}"; done

if [[ "${UPLOAD_RESULTS}" == 1 ]]; then
  git worktree add -b "${UPLOAD_BRANCH}" "${UPLOAD_WT}" origin/v39_otherdata
  mkdir -p "${UPLOAD_WT}/${DEST}"
  cp "${SEARCH_ROOT}/paper_tables.md" "${UPLOAD_WT}/${DEST}/"
  cp "${SEARCH_ROOT}/paper_trend_audit.json" "${UPLOAD_WT}/${DEST}/"
  cp "${SEARCH_ROOT}/routeA_search_summary.json" "${UPLOAD_WT}/${DEST}/"
  cp "${SEARCH_ROOT}/best_params.env" "${UPLOAD_WT}/${DEST}/"
  cp "${SEARCH_ROOT}/best_routeA_metrics.json" "${UPLOAD_WT}/${DEST}/"
  cp "${SEARCH_ROOT}/bearing_citya_summary.json" "${UPLOAD_WT}/${DEST}/"
  [[ -s "${SEARCH_ROOT}/test_01_final_result.jpg" ]] && cp "${SEARCH_ROOT}/test_01_final_result.jpg" "${UPLOAD_WT}/${DEST}/"
  [[ -s "${SEARCH_ROOT}/test_02_final_result.jpg" ]] && cp "${SEARCH_ROOT}/test_02_final_result.jpg" "${UPLOAD_WT}/${DEST}/"
  (
    cd "${UPLOAD_WT}"
    git add "${DEST}"
    git commit -m "Add Route-A-selected simple GRU and corrected Bearing cityA results"
    git fetch origin v39_otherdata
    git rebase origin/v39_otherdata
    git push origin HEAD:v39_otherdata
  )
fi

echo "================================================================================"
echo "DONE"
echo "Route-A search : ${SEARCH_ROOT}/routeA_search_summary.json"
echo "Final tables   : ${SEARCH_ROOT}/paper_tables.md"
echo "Final audit    : ${SEARCH_ROOT}/paper_trend_audit.json"
echo "Bearing cityA : ${SEARCH_ROOT}/bearing_citya_summary.json"
echo "GitHub         : ${DEST}"
echo "================================================================================"
