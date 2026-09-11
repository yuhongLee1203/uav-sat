#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
SCRIPT_PATH="${ROOT}/run.sh"
BASE_SRC="${ROOT}/base_src"
OUT="${UAVSAT_OUTPUT_DIR:-${ROOT}/output_wc_final}"
SRC="${UAVSAT_RUNTIME_DIR:-${ROOT}/runtime_wc_final}"
FEATURE_CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache}"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
DENSE_REF_DIR="${UAVSAT_DENSE_ROUTE_REFERENCE_DIR:-${REPO_ROOT}/frame-reference-exp/references}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
BACKBONE="mobilenet_v3_small"
BASE_ARCH="V39_WeightedCentroid_PreviousState_MobileNetV3_Forward3x6"
FINAL_ARCH="V39_WeightedCentroid_GRU_Kalman_MS"
DEFAULT_GRID="${DEFAULT_MS_GRID:-5}"
DEFAULT_BW="${DEFAULT_MS_BANDWIDTH:-7.0}"
DEFAULT_MOTION="velocity"
DEFAULT_KALMAN="fixed"

if [[ "${RUN_ALL_EXPERIMENTS:-0}" == "1" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  SUITE_ROOT="${EXPERIMENT_SUITE_DIR:-${ROOT}/wc_experiments_${TS}}"
  SHARED_CACHE="${UAVSAT_SHARED_FEATURE_CACHE_DIR:-${ROOT}/output/feature_cache}"
  mkdir -p "${SUITE_ROOT}" "${SHARED_CACHE}"

  run_cfg() {
    local gpu="$1" name="$2" frames="$3" kalman="$4" disable_gru="$5"
    local ms_enabled="$6" grid="$7" category="$8" train_mode="$9"
    local ckpt_source="${10:-}" measure_ms="${11:-0}" measure_e2e="${12:-0}"
    local out="${SUITE_ROOT}/${name}"
    local runtime="${SUITE_ROOT}/runtime_${name}"

    echo "[START][${name}][GPU ${gpu}] frames=${frames} kalman=${kalman} gru=$((1-disable_gru)) ms=${ms_enabled} grid=${grid} mode=${train_mode}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    UAVSAT_DEVICE=cuda:0 \
    UAVSAT_OUTPUT_DIR="${out}" \
    UAVSAT_RUNTIME_DIR="${runtime}" \
    UAVSAT_FEATURE_CACHE_DIR_OVERRIDE="${SHARED_CACHE}" \
    UAVSAT_DENSE_ROUTE_REFERENCE_DIR="${DENSE_REF_DIR}" \
    UAVSAT_REFERENCE_PROTOCOL=scheduled_route_reference \
    UAVSAT_EXPERIMENT_ANCHOR=weighted_centroid \
    UAVSAT_EXPERIMENT_FRAME_COUNT="${frames}" \
    UAVSAT_EXPERIMENT_MOTION="${DEFAULT_MOTION}" \
    UAVSAT_EXPERIMENT_KALMAN="${kalman}" \
    UAVSAT_EXPERIMENT_DISABLE_GRU="${disable_gru}" \
    UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
    MS_ENABLED="${ms_enabled}" \
    MS_GRID_SIZE="${grid}" \
    MS_BANDWIDTH_M="${DEFAULT_BW}" \
    MS_MEASURE_LATENCY="${measure_ms}" \
    MS_LATENCY_WARMUP=30 \
    UAVSAT_MEASURE_LATENCY="${measure_e2e}" \
    UAVSAT_LATENCY_WARMUP=30 \
    FORCE_RETRAIN_TEMPORAL="$([[ "${train_mode}" == "train" ]] && echo 1 || echo 0)" \
    UAVSAT_TEMPORAL_CKPT_SOURCE="${ckpt_source}" \
    EXPERIMENT_TAG="${name}" \
    EXPERIMENT_CATEGORY="${category}" \
    RUN_ALL_EXPERIMENTS=0 \
    bash "${SCRIPT_PATH}" 2>&1 | sed -u "s/^/[${name}] /"
    echo "[DONE ][${name}][GPU ${gpu}]"
  }

  echo "============================================================================================================"
  echo "FORMAL SUITE: Weighted Centroid -> GRU -> Kalman -> ONE final MS"
  echo "Route A: temporal training/validation; Route B/C: evaluation only"
  echo "Reference protocol: scheduled predefined route references"
  echo "Table 2: 1-frame / 2-frame / 3-frame GRU (each trained separately)"
  echo "MS operating point fixed before suite: ${DEFAULT_GRID}x${DEFAULT_GRID}, BW=${DEFAULT_BW} m"
  echo "============================================================================================================"

  run_cfg 0 full_model 3 fixed 0 1 "${DEFAULT_GRID}" module_ablation train
  CANON_CKPT="${SUITE_ROOT}/full_model/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  [[ -f "${CANON_CKPT}" && ! -L "${CANON_CKPT}" ]] || { echo "ERROR: fresh canonical checkpoint missing" >&2; exit 20; }

  ( run_cfg 0 temporal_1frame 1 fixed 0 1 "${DEFAULT_GRID}" temporal_frames train ) & p0=$!
  ( run_cfg 5 temporal_2frame 2 fixed 0 1 "${DEFAULT_GRID}" temporal_frames train ) & p5=$!
  (
    run_cfg 6 kalman_none 3 none 0 1 "${DEFAULT_GRID}" kalman_design train
    run_cfg 6 kalman_learned 3 learned 0 1 "${DEFAULT_GRID}" kalman_design train
  ) & p6=$!
  status=0
  wait "${p0}" || status=1
  wait "${p5}" || status=1
  wait "${p6}" || status=1
  [[ "${status}" == "0" ]] || { echo "ERROR: a training queue failed" >&2; exit 21; }

  ONE_CKPT="${SUITE_ROOT}/temporal_1frame/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  TWO_CKPT="${SUITE_ROOT}/temporal_2frame/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  NONE_CKPT="${SUITE_ROOT}/kalman_none/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  LEARNED_CKPT="${SUITE_ROOT}/kalman_learned/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
  for ck in "${ONE_CKPT}" "${TWO_CKPT}" "${NONE_CKPT}" "${LEARNED_CKPT}"; do
    [[ -f "${ck}" && ! -L "${ck}" ]] || { echo "ERROR: expected freshly trained checkpoint missing: ${ck}" >&2; exit 22; }
  done

  run_cfg 0 abl_wc_only 3 none 1 0 "${DEFAULT_GRID}" module_ablation eval
  run_cfg 0 abl_wc_gru 3 none 0 0 "${DEFAULT_GRID}" module_ablation eval "${NONE_CKPT}"
  run_cfg 0 abl_wc_gru_kalman 3 fixed 0 0 "${DEFAULT_GRID}" module_ablation eval "${CANON_CKPT}"

  for g in 4 5 6 7 8; do
    run_cfg 5 "sens_ms_grid${g}x${g}" 3 fixed 0 1 "${g}" ms_window eval "${CANON_CKPT}" 1 0
  done

  run_cfg 0 runtime_e2e 3 fixed 0 1 "${DEFAULT_GRID}" runtime eval "${CANON_CKPT}" 0 1

  python3 - "${SUITE_ROOT}" "${DEFAULT_GRID}" "${DEFAULT_BW}" <<'PY'
import csv, json, sys
from pathlib import Path
suite=Path(sys.argv[1]); selected_grid=int(sys.argv[2]); fixed_bw=float(sys.argv[3])
def read(name):
    p=suite/name/'robust_tracker_summary.json'
    if not p.exists(): raise SystemExit(f'AUDIT FAILED: missing {p}')
    return json.loads(p.read_text(encoding='utf-8'))
def nrows(d,route):
    p=Path(d[route].get('CSV',''))
    if p.exists():
        with p.open('r',encoding='utf-8') as f: return max(sum(1 for _ in f)-1,1)
    return 2276 if route=='route_B' else 1258
def w(d,key):
    nb,nc=nrows(d,'route_B'),nrows(d,'route_C')
    return (float(d['route_B'][key])*nb+float(d['route_C'][key])*nc)/(nb+nc)
def audit(name,d,gru,ms,frames,kalman):
    e=[]
    if d.get('reference_protocol')!='scheduled_route_reference': e.append('protocol')
    if d.get('uses_gt_center_at_inference') is not False: e.append('inference center')
    if d.get('experiment_anchor')!='weighted_centroid': e.append('front decoder')
    if int(d.get('experiment_frame_count',-1))!=frames: e.append('frame count')
    if d.get('experiment_kalman')!=kalman: e.append('Kalman mode')
    if (not bool(d.get('experiment_disable_gru',False)))!=gru: e.append('GRU switch')
    if bool(d.get('ms_enabled'))!=ms: e.append('MS switch')
    for route in ('route_B','route_C'):
        r=d.get(route,{})
        if r.get('VisualObservationDecoder')!='posterior weighted centroid': e.append(route+' decoder')
        if r.get('OnlineMeanShiftCount')!=(1 if ms else 0): e.append(route+' MeanShift count')
        if r.get('StaticSatelliteProjectionCache') is not True: e.append(route+' SAT cache')
        if ms and r.get('FinalMSReusesFrontUAVEmbedding') is not True: e.append(route+' UAV reuse')
        if r.get('CaptureDiagnosticOnly') is not True: e.append(route+' capture diagnostic flag')
    if e: raise SystemExit(f"AUDIT FAILED [{name}]: {', '.join(e)}")
names=['full_model','temporal_1frame','temporal_2frame','kalman_none','kalman_learned','abl_wc_only','abl_wc_gru','abl_wc_gru_kalman']+[f'sens_ms_grid{i}x{i}' for i in range(4,9)]+['runtime_e2e']
d={n:read(n) for n in names}
audit('full_model',d['full_model'],True,True,3,'fixed')
audit('temporal_1frame',d['temporal_1frame'],True,True,1,'fixed')
audit('temporal_2frame',d['temporal_2frame'],True,True,2,'fixed')
audit('kalman_none',d['kalman_none'],True,True,3,'none')
audit('kalman_learned',d['kalman_learned'],True,True,3,'learned')
audit('abl_wc_only',d['abl_wc_only'],False,False,3,'none')
audit('abl_wc_gru',d['abl_wc_gru'],True,False,3,'none')
audit('abl_wc_gru_kalman',d['abl_wc_gru_kalman'],True,False,3,'fixed')
for i in range(4,9):
    name=f'sens_ms_grid{i}x{i}'; audit(name,d[name],True,True,3,'fixed')
audit('runtime_e2e',d['runtime_e2e'],True,True,3,'fixed')
for n in ['full_model','temporal_1frame','temporal_2frame','kalman_none','kalman_learned']:
    ck=suite/n/'checkpoints'/'controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt'
    if not ck.exists() or ck.is_symlink(): raise SystemExit(f'AUDIT FAILED [{n}]: checkpoint was not freshly trained')
rows=[]
for n,x in d.items():
    rb,rc=x['route_B'],x['route_C']; eb,ec=rb.get('EndToEndTiming',{}),rc.get('EndToEndTiming',{})
    ms_lat=w(x,'MS_LatencyMean_ms') if float(rb.get('MS_LatencyMean_ms',0))>0 and float(rc.get('MS_LatencyMean_ms',0))>0 else 0.0
    e2e=0.0
    if eb and ec:
        nb,nc=nrows(x,'route_B'),nrows(x,'route_C'); e2e=(float(eb['mean_ms'])*nb+float(ec['mean_ms'])*nc)/(nb+nc)
    rows.append({'Experiment':n,'Frames':x.get('experiment_frame_count'),'GRU':'no' if x.get('experiment_disable_gru') else 'yes','Kalman':x.get('experiment_kalman'),'MS':'yes' if x.get('ms_enabled') else 'no','MS_grid':x.get('ms_grid_size','-'),'B_MLE_m':rb['MLE_m'],'C_MLE_m':rc['MLE_m'],'BC_MLE_m':w(x,'MLE_m'),'BC_P90_m':w(x,'P90_m'),'BC_LSR5_pct':w(x,'LSR@5_pct'),'B_Jump_pct':rb['JumpRate_pct'],'C_Jump_pct':rc['JumpRate_pct'],'B_Capture_pct':rb.get('SelectedCandidateCapture_pct'),'C_Capture_pct':rc.get('SelectedCandidateCapture_pct'),'MS_latency_ms':ms_lat,'E2E_latency_ms':e2e})
with (suite/'experiment_summary.csv').open('w',newline='',encoding='utf-8') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
fmt=lambda x,n=3:f'{float(x):.{n}f}'
md=['# Weighted-Centroid Final Ablation Tables','','Inference chain: Weighted Centroid -> GRU -> Kalman -> one final MeanShift.','Candidate capture is an evaluation-only diagnostic and never affects B/C inference.','']
md += ['## Table 1. Progressive architecture ablation','','| Setting | B MLE | C MLE | B+C MLE | B+C LSR@5 | B/C Jump |','|---|---:|---:|---:|---:|---:|']
for n,label in [('abl_wc_only','WC'),('abl_wc_gru','WC + GRU'),('abl_wc_gru_kalman','WC + GRU + Kalman'),('full_model','WC + GRU + Kalman + MS')]:
    x=d[n]; md.append(f"| {label} | {fmt(x['route_B']['MLE_m'])} | {fmt(x['route_C']['MLE_m'])} | {fmt(w(x,'MLE_m'))} | {fmt(w(x,'LSR@5_pct'),2)}% | {fmt(x['route_B']['JumpRate_pct'],3)}/{fmt(x['route_C']['JumpRate_pct'],3)}% |")
md += ['','## Table 2. Temporal input-frame ablation','','| UAV frames | B MLE | C MLE | B+C MLE | B+C P90 |','|---:|---:|---:|---:|---:|']
for n,k in [('temporal_1frame',1),('temporal_2frame',2),('full_model',3)]:
    x=d[n]; md.append(f"| {k} | {fmt(x['route_B']['MLE_m'])} | {fmt(x['route_C']['MLE_m'])} | {fmt(w(x,'MLE_m'))} | {fmt(w(x,'P90_m'))} |")
md += ['','## Table 3. Kalman design','','| Kalman | B MLE | C MLE | B+C MLE | B/C Jump |','|---|---:|---:|---:|---:|']
for n,label in [('kalman_none','No Kalman'),('kalman_learned','Learned variance'),('full_model','Fixed variance')]:
    x=d[n]; md.append(f"| {label} | {fmt(x['route_B']['MLE_m'])} | {fmt(x['route_C']['MLE_m'])} | {fmt(w(x,'MLE_m'))} | {fmt(x['route_B']['JumpRate_pct'],3)}/{fmt(x['route_C']['JumpRate_pct'],3)}% |")
md += ['','## Table 4. Final-MS window accuracy / stage latency','','| Window | Candidates | B+C MLE | MS latency (ms) | MS FPS |','|---|---:|---:|---:|---:|']
for g in range(4,9):
    x=d[f'sens_ms_grid{g}x{g}']; lat=w(x,'MS_LatencyMean_ms'); label=f'{g}x{g}'+(' (operating point)' if g==selected_grid else '')
    md.append(f"| {label} | {g*g} | {fmt(w(x,'MLE_m'))} | {fmt(lat)} | {fmt(1000.0/lat,1) if lat>0 else '-'} |")
rt=d['runtime_e2e']; eb,ec=rt['route_B']['EndToEndTiming'],rt['route_C']['EndToEndTiming']; nb,nc=nrows(rt,'route_B'),nrows(rt,'route_C'); e2e=(float(eb['mean_ms'])*nb+float(ec['mean_ms'])*nc)/(nb+nc)
md += ['','## Table 5. End-to-end online runtime','','Prepared UAV tensor -> backbone -> Weighted Centroid -> GRU -> Kalman -> one final MS -> XY.','',f"- Route B: {fmt(eb['mean_ms'])} ms / {fmt(eb['fps'],1)} FPS",f"- Route C: {fmt(ec['mean_ms'])} ms / {fmt(ec['fps'],1)} FPS",f"- Weighted B+C: **{fmt(e2e)} ms / {fmt(1000.0/e2e,1)} FPS**",'',f'Fixed MeanShift bandwidth: **{fixed_bw:g} m**.']
(suite/'paper_tables.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
(suite/'audit_report.json').write_text(json.dumps({'status':'PASS','chain':'Weighted Centroid -> GRU -> Kalman -> one final MS','reference_protocol':'scheduled_route_reference','capture':'evaluation-only diagnostic','fixed_ms_grid':selected_grid,'fixed_bandwidth_m':fixed_bw,'fresh_training':['full_model','temporal_1frame','temporal_2frame','kalman_none','kalman_learned']},indent=2),encoding='utf-8')
print('AUDIT PASS')
print(suite/'paper_tables.md')
PY

  echo "============================================================================================================"
  echo "ALL FORMAL ABLATIONS COMPLETED + AUDIT PASS"
  echo "Results: ${SUITE_ROOT}"
  echo "Tables : ${SUITE_ROOT}/paper_tables.md"
  echo "Audit  : ${SUITE_ROOT}/audit_report.json"
  echo "============================================================================================================"
  exit 0
fi

VISUAL_CKPT="${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt"
LOCAL_TEMPORAL_CKPT="${OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
LATEST_TEMPORAL_CKPT="${OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only_latest.pt"
for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -f "${BASE_SRC}/${f}" ]] || { echo "ERROR: missing ${BASE_SRC}/${f}" >&2; exit 2; }
done
[[ -s "${VISUAL_CKPT}" ]] || { echo "ERROR: missing visual checkpoint ${VISUAL_CKPT}" >&2; exit 2; }
[[ -f "${ROOT}/patch_direct_finalms.py" ]] || { echo "ERROR: missing patch_direct_finalms.py" >&2; exit 2; }
for route in route_A route_B route_C; do
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || { echo "ERROR: missing ${route}" >&2; exit 2; }
  [[ -s "${DENSE_REF_DIR}/${route}.npz" ]] || { echo "ERROR: missing scheduled reference ${DENSE_REF_DIR}/${route}.npz" >&2; exit 2; }
done

rm -rf "${SRC}"
mkdir -p "${SRC}" "${OUT}/checkpoints" "${FEATURE_CACHE_DIR}"
cp -a "${BASE_SRC}/." "${SRC}/"
python3 "${ROOT}/patch_direct_finalms.py" "${SRC}/robust_tracker.py" "${SRC}/visual_localizer.py" "${SRC}/config.py"

python3 - "${SRC}/robust_tracker.py" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text(encoding='utf-8')
old='''            gt_xy=(None if bool(getattr(config, "NO_GT_INFERENCE", False)) else gt),
            gt_se=(None if bool(getattr(config, "NO_GT_INFERENCE", False)) else gt_state["se"][index]),
            teacher_select=not bool(getattr(config, "NO_GT_INFERENCE", False)),
'''
new='''            # Evaluation-only target coordinate for capture diagnostics. It is
            # not used to select a hypothesis because gt_se=None and teacher_select=False
            # whenever NO_GT_INFERENCE is active.
            gt_xy=gt,
            gt_se=(None if bool(getattr(config, "NO_GT_INFERENCE", False)) else gt_state["se"][index]),
            teacher_select=not bool(getattr(config, "NO_GT_INFERENCE", False)),
'''
if s.count(old)!=1: raise SystemExit(f'diagnostic validation patch count={s.count(old)}')
s=s.replace(old,new,1)
old='''            gt_xy=(None if bool(getattr(config, "NO_GT_INFERENCE", False)) else gt_xy_t),
            gt_se=(None if bool(getattr(config, "NO_GT_INFERENCE", False)) else gt_state["se"][index]),
            teacher_select=not bool(getattr(config, "NO_GT_INFERENCE", False)),
'''
new='''            # Evaluation-only capture diagnostic; current evaluation coordinate
            # does not affect search-center selection or estimator state.
            gt_xy=gt_xy_t,
            gt_se=(None if bool(getattr(config, "NO_GT_INFERENCE", False)) else gt_state["se"][index]),
            teacher_select=not bool(getattr(config, "NO_GT_INFERENCE", False)),
'''
if s.count(old)!=1: raise SystemExit(f'diagnostic inference patch count={s.count(old)}')
s=s.replace(old,new,1)
needle='''    summary["SelectedCandidateCapture_pct"] = float(
        np.mean(captures) * 100.0
    )
'''
repl=needle+'''    summary["CaptureDiagnosticOnly"] = bool(getattr(config, "NO_GT_INFERENCE", False))
    summary["CaptureDiagnosticDefinition"] = "candidate containment computed after candidate selection using evaluation coordinates; never used by inference"
'''
if s.count(needle)!=1: raise SystemExit('capture summary patch failed')
s=s.replace(needle,repl,1)
old_msg='''    print(
        "Evaluation intentionally does not solve global acquisition/re-localization: "
        "every frame receives a bounded GT-centered local prior.",
        flush=True,
    )
'''
new_msg='''    print(
        "Evaluation uses the frozen scheduled route reference for local search. "
        "Current evaluation coordinates are used only for metrics/diagnostics.",
        flush=True,
    )
'''
if old_msg in s: s=s.replace(old_msg,new_msg,1)
compile(s,str(p),'exec'); p.write_text(s,encoding='utf-8')
print('[OK] evaluation-only capture diagnostics enabled without inference leakage')
PY

ln -sfn "${VISUAL_CKPT}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"
MODE="eval"
if [[ "${FORCE_RETRAIN_TEMPORAL:-0}" == "1" ]]; then
  rm -f "${LOCAL_TEMPORAL_CKPT}" "${LATEST_TEMPORAL_CKPT}"
  MODE="train_eval"
elif [[ -n "${UAVSAT_TEMPORAL_CKPT_SOURCE:-}" ]]; then
  [[ -s "${UAVSAT_TEMPORAL_CKPT_SOURCE}" ]] || { echo "ERROR: checkpoint source missing ${UAVSAT_TEMPORAL_CKPT_SOURCE}" >&2; exit 3; }
  rm -f "${LOCAL_TEMPORAL_CKPT}" "${LATEST_TEMPORAL_CKPT}"
  ln -s "${UAVSAT_TEMPORAL_CKPT_SOURCE}" "${LOCAL_TEMPORAL_CKPT}"
elif [[ "${UAVSAT_EXPERIMENT_DISABLE_GRU:-0}" == "1" ]]; then
  MODE="eval"
elif [[ ! -s "${LOCAL_TEMPORAL_CKPT}" ]]; then
  MODE="train_eval"
fi

export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MS_KF_SIGMA_M="${MS_KF_SIGMA_M:-4.0}" MS_REFERENCE_SIGMA_M="${MS_REFERENCE_SIGMA_M:-4.0}"
export MS_KF_PRIOR_WEIGHT="${MS_KF_PRIOR_WEIGHT:-1.50}" MS_REFERENCE_PRIOR_WEIGHT="${MS_REFERENCE_PRIOR_WEIGHT:-2.50}"
export MS_BANDWIDTH_M="${MS_BANDWIDTH_M:-${DEFAULT_BW}}" MS_ENABLED="${MS_ENABLED:-1}" MS_GRID_SIZE="${MS_GRID_SIZE:-${DEFAULT_GRID}}"
export MS_LATENCY_WARMUP="${MS_LATENCY_WARMUP:-30}" MS_MEASURE_LATENCY="${MS_MEASURE_LATENCY:-0}"

cd "${SRC}"
ARGS=(--mode "${MODE}" --reuse-visual --jitter-m 0)
if [[ "${MODE}" == "train_eval" ]]; then ARGS+=(--temporal-epochs "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}"); fi
UAVSAT_DEVICE="${DEVICE}" \
UAVSAT_OUTPUT_DIR="${OUT}" \
UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR}" \
UAVSAT_DATA_ROOT="${DATA_ROOT}" \
UAVSAT_DENSE_ROUTE_REFERENCE_DIR="${DENSE_REF_DIR}" \
UAVSAT_BACKBONE="${BACKBONE}" \
UAVSAT_ARCHITECTURE_NAME="${BASE_ARCH}" \
UAVSAT_REFERENCE_PROTOCOL="${UAVSAT_REFERENCE_PROTOCOL:-scheduled_route_reference}" \
UAVSAT_EXPERIMENT_ANCHOR="${UAVSAT_EXPERIMENT_ANCHOR:-weighted_centroid}" \
UAVSAT_EXPERIMENT_FRAME_COUNT="${UAVSAT_EXPERIMENT_FRAME_COUNT:-3}" \
UAVSAT_EXPERIMENT_MOTION="${UAVSAT_EXPERIMENT_MOTION:-velocity}" \
UAVSAT_EXPERIMENT_KALMAN="${UAVSAT_EXPERIMENT_KALMAN:-fixed}" \
UAVSAT_EXPERIMENT_DISABLE_GRU="${UAVSAT_EXPERIMENT_DISABLE_GRU:-0}" \
UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
UAVSAT_MEASURE_LATENCY="${UAVSAT_MEASURE_LATENCY:-0}" \
UAVSAT_LATENCY_WARMUP="${UAVSAT_LATENCY_WARMUP:-30}" \
python3 -u robust_tracker.py "${ARGS[@]}" 2>&1 | tee "${OUT}/${MODE}.log"

python3 - "${OUT}/robust_tracker_summary.json" "${FINAL_ARCH}" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8'))
d['architecture']=sys.argv[2]; d['experiment_tag']=os.environ.get('EXPERIMENT_TAG','single'); d['experiment_category']=os.environ.get('EXPERIMENT_CATEGORY','single')
d['experiment_motion']=os.environ.get('UAVSAT_EXPERIMENT_MOTION','velocity'); d['experiment_kalman']=os.environ.get('UAVSAT_EXPERIMENT_KALMAN','fixed')
d['experiment_disable_gru']=os.environ.get('UAVSAT_EXPERIMENT_DISABLE_GRU','0')=='1'; d['experiment_frame_count']=int(os.environ.get('UAVSAT_EXPERIMENT_FRAME_COUNT','3'))
d['ms_enabled']=os.environ.get('MS_ENABLED','1').lower() not in {'0','false','no','off'}; d['ms_grid_size']=int(os.environ.get('MS_GRID_SIZE','5'))
d['final_chain']='Weighted Centroid -> GRU -> Kalman Filter -> one MS -> Final Position'; d['ms_hyperparameters']={'bandwidth_m':float(os.environ.get('MS_BANDWIDTH_M','7'))}
p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
PY

echo "[DONE] ${OUT}/robust_tracker_summary.json"
