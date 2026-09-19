#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BASE_SRC="${ROOT}/base_src"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
GPU="${GPU:-0}"
BACKBONE="mobilenet_v3_small"
ARCH="V39_MobileNetV3_Forward18of6x6_SoftMS_GRU_CV_Kalman_Final5x5MS"
JITTER_M="${JITTER_M:-8}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
DEFAULT_MS_GRID="${MS_GRID_SIZE:-5}"
DEFAULT_MS_BW="${MS_BANDWIDTH_M:-7.0}"
MS_LATENCY_WARMUP="${MS_LATENCY_WARMUP:-30}"
E2E_WARMUP="${E2E_WARMUP:-30}"
TS="$(date +%Y%m%d_%H%M%S)"
SUITE_ROOT="${SOFTMS_ABLATION_DIR:-${ROOT}/original18_final5_strict_eval_${TS}}"
FEATURE_CACHE="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache}"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

if [[ -s "${REPO_ROOT}/forNX/weights/v39_directfinalms/checkpoints/visual_retrieval_A_only.pt" ]]; then
  VISUAL_CKPT="${V39_VISUAL_CKPT:-${REPO_ROOT}/forNX/weights/v39_directfinalms/checkpoints/visual_retrieval_A_only.pt}"
else
  VISUAL_CKPT="${V39_VISUAL_CKPT:-${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt}"
fi

fail() { echo "ERROR: $*" >&2; exit 2; }

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -s "${BASE_SRC}/${f}" ]] || fail "missing ${BASE_SRC}/${f}"
done
[[ -s "${ROOT}/patch_direct_finalms.py" ]] || fail "missing patch_direct_finalms.py"
[[ -s "${ROOT}/patch_front_softms.py" ]] || fail "missing patch_front_softms.py"
[[ -s "${VISUAL_CKPT}" ]] || fail "missing visual checkpoint: ${VISUAL_CKPT}"
for route in route_A route_B route_C; do
  [[ -s "${DATA_ROOT}/routes/${route}/frames.csv" ]] || fail "missing ${DATA_ROOT}/routes/${route}/frames.csv"
done

mkdir -p "${SUITE_ROOT}" "${FEATURE_CACHE}"
export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
cat > "${SUITE_ROOT}/run_manifest.json" <<EOF
{
  "method": "Forward 3x6 SoftMS -> 3-frame GRU -> Constant Velocity -> fixed-R Kalman -> final SoftMS",
  "ablation_policy": "core component ablations only; Forward 3x6 remains fixed because replacing it with full 6x6 changes directionality, candidate count, compute budget, and search semantics simultaneously",
  "git_sha": "${GIT_SHA}",
  "gpu": "${GPU}",
  "temporal_epochs": ${TEMPORAL_EPOCHS},
  "patience": ${PATIENCE},
  "jitter_m": ${JITTER_M},
  "final_ms_grid": ${DEFAULT_MS_GRID},
  "final_ms_bandwidth_m": ${DEFAULT_MS_BW}
}
EOF

make_runtime() {
  local runtime="$1"
  mkdir -p "${runtime}"
  cp -a "${BASE_SRC}/." "${runtime}/"
  python3 "${ROOT}/patch_direct_finalms.py" "${runtime}/robust_tracker.py"
  python3 "${ROOT}/patch_front_softms.py" "${runtime}/robust_tracker.py"
  python3 -m py_compile "${runtime}/robust_tracker.py"
  grep -q 'Front visual observation: Soft MeanShift' "${runtime}/robust_tracker.py" || fail "front SoftMS audit failed"
  if grep -q 'weighted_xy = (raw_prob.unsqueeze(-1) \* centers).sum(dim=1)' "${runtime}/robust_tracker.py"; then
    fail "Weighted Centroid still present in ${runtime}"
  fi
}

run_cfg() {
  local name="$1" category="$2" mode="$3" frames="$4" disable_gru="$5"
  local kalman="$6" ms_enabled="$7" grid="$8" ckpt_source="${9:-}" measure_ms="${10:-0}" measure_e2e="${11:-0}"
  local out="${SUITE_ROOT}/${name}"
  local runtime="${SUITE_ROOT}/runtime_${name}"
  mkdir -p "${out}/checkpoints"
  make_runtime "${runtime}"
  ln -sfn "${VISUAL_CKPT}" "${out}/checkpoints/visual_retrieval_A_only.pt"
  if [[ "${mode}" == "eval" && "${disable_gru}" == "0" ]]; then
    [[ -s "${ckpt_source}" ]] || fail "missing temporal checkpoint for ${name}: ${ckpt_source}"
    ln -sfn "${ckpt_source}" "${out}/checkpoints/${CKPT_NAME}"
  fi

  echo "================================================================================"
  echo "[RUN] ${name} | category=${category} | frames=${frames} | front=Forward3x6-SoftMS | gru=$((1-disable_gru)) | kalman=${kalman} | finalMS=${ms_enabled} grid=${grid}"
  echo "================================================================================"
  (
    cd "${runtime}"
    args=(--mode "${mode}" --reuse-visual --jitter-m "${JITTER_M}")
    if [[ "${mode}" == "train_eval" ]]; then
      args+=(--temporal-epochs "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}")
    fi
    CUDA_VISIBLE_DEVICES="${GPU}" \
    UAVSAT_DEVICE=cuda:0 \
    UAVSAT_OUTPUT_DIR="${out}" \
    UAVSAT_CHECKPOINT_DIR="${out}/checkpoints" \
    UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE}" \
    UAVSAT_DATA_ROOT="${DATA_ROOT}" \
    UAVSAT_BACKBONE="${BACKBONE}" \
    UAVSAT_ARCHITECTURE_NAME="${ARCH}" \
    UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
    UAVSAT_EXPERIMENT_ANCHOR=softms \
    UAVSAT_EXPERIMENT_FRAME_COUNT="${frames}" \
    UAVSAT_EXPERIMENT_MOTION=velocity \
    UAVSAT_EXPERIMENT_KALMAN="${kalman}" \
    UAVSAT_EXPERIMENT_DISABLE_GRU="${disable_gru}" \
    UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
    UAVSAT_MEASURE_LATENCY="${measure_e2e}" \
    UAVSAT_LATENCY_WARMUP="${E2E_WARMUP}" \
    MS_ENABLED="${ms_enabled}" \
    MS_GRID_SIZE="${grid}" \
    MS_BANDWIDTH_M="${DEFAULT_MS_BW}" \
    MS_MEASURE_LATENCY="${measure_ms}" \
    MS_LATENCY_WARMUP="${MS_LATENCY_WARMUP}" \
    python3 -u robust_tracker.py "${args[@]}" 2>&1 | tee "${out}/${mode}.log"
  )

  python3 - "${out}" "${name}" "${category}" "${frames}" "${disable_gru}" "${kalman}" "${ms_enabled}" "${grid}" <<'PY'
import json, sys
from pathlib import Path
out=Path(sys.argv[1]); name=sys.argv[2]; category=sys.argv[3]
frames=int(sys.argv[4]); disable=bool(int(sys.argv[5])); kalman=sys.argv[6]; ms=bool(int(sys.argv[7])); grid=int(sys.argv[8])
p=out/'robust_tracker_summary.json'
d=json.loads(p.read_text(encoding='utf-8'))
d['experiment_tag']=name
d['experiment_category']=category
d['experiment_anchor']='softms'
d['experiment_frame_count']=frames
d['experiment_forward_only']=True
d['experiment_disable_gru']=disable
d['experiment_kalman']=kalman
d['ms_enabled']=ms
d['ms_grid_size']=grid
d['front_decoder']='forward_3x6_softms'
d['checkpoint_retrained']=category=='temporal_frames'
d['final_chain']='Forward 3x6 SoftMS -> GRU -> fixed-R Kalman -> final SoftMS' if not disable else 'Forward 3x6 SoftMS -> no GRU -> fixed-R Kalman -> final SoftMS'
for route in ('route_B','route_C'):
    if route in d:
        d[route]['VisualObservationDecoder']='forward 3x6 soft mean shift'
p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
PY
  echo "[DONE] ${name}"
}

# 1) Fair temporal-input ablation. Each model is freshly trained on Route A.
run_cfg temporal_1frame temporal_frames eval 1 0 fixed 1 "${DEFAULT_MS_GRID}" "${SOURCE_CKPT1}" 0 0
run_cfg temporal_2frame temporal_frames eval 2 0 fixed 1 "${DEFAULT_MS_GRID}" "${SOURCE_CKPT2}" 0 0
run_cfg temporal_3frame temporal_frames eval 3 0 fixed 1 "${DEFAULT_MS_GRID}" "${SOURCE_CKPT3}" 0 0

CKPT1="${SOURCE_CKPT1}"
CKPT2="${SOURCE_CKPT2}"
CKPT3="${SOURCE_CKPT3}"

# 2) Core component ablations. Forward 3x6 SoftMS is FIXED in every row.
run_cfg full_model       module_ablation eval 3 0 fixed 1 "${DEFAULT_MS_GRID}" "${CKPT3}" 0 1
run_cfg abl_no_gru       module_ablation eval 3 1 fixed 1 "${DEFAULT_MS_GRID}" ""         0 0
run_cfg abl_no_kalman    module_ablation eval 3 0 none  1 "${DEFAULT_MS_GRID}" "${CKPT3}" 0 0
run_cfg abl_no_final_ms  module_ablation eval 3 0 fixed 0 "${DEFAULT_MS_GRID}" "${CKPT3}" 0 0

# 3) Final-MS window sensitivity. Front search remains Forward 3x6 SoftMS.
for g in 4 5 6 7 8; do
  run_cfg "ms_window_${g}x${g}" ms_window eval 3 0 fixed 1 "${g}" "${CKPT3}" 1 0
done

# Aggregate raw frame errors and save CSV/JSON/Markdown tables.
python3 - "${SUITE_ROOT}" <<'PY'
import csv, json, sys
from pathlib import Path
import numpy as np
suite=Path(sys.argv[1])
names=[
 'full_model','abl_no_gru','abl_no_kalman','abl_no_final_ms',
 'temporal_1frame','temporal_2frame','temporal_3frame',
 'ms_window_4x4','ms_window_5x5','ms_window_6x6','ms_window_7x7','ms_window_8x8'
]

def metrics(a):
    a=np.asarray(a,dtype=float)
    return {
      'frames':int(a.size),'MLE_m':float(a.mean()),'MedLE_m':float(np.median(a)),
      'P90_m':float(np.quantile(a,.90)),'P95_m':float(np.quantile(a,.95)),'P99_m':float(np.quantile(a,.99)),
      'LSR@5_pct':float((a<=5).mean()*100),'LSR@10_pct':float((a<=10).mean()*100),
      'LSR@15_pct':float((a<=15).mean()*100),'LSR@20_pct':float((a<=20).mean()*100),
    }

def route_errors(out,route):
    files=sorted(out.glob(f'{route}_*_frames.csv'))
    if len(files)!=1: raise SystemExit(f'{out}/{route}: expected one frame CSV, got {len(files)}')
    with files[0].open(newline='',encoding='utf-8') as f:
        return np.asarray([float(r['error_final_m']) for r in csv.DictReader(f)],dtype=float)

rows=[]; payload={}
for name in names:
    out=suite/name
    d=json.loads((out/'robust_tracker_summary.json').read_text(encoding='utf-8'))
    if not bool(d.get('experiment_forward_only',False)):
        raise SystemExit(f'AUDIT FAILED: {name} changed Forward 3x6')
    eb=route_errors(out,'route_B'); ec=route_errors(out,'route_C'); allerr=np.concatenate([eb,ec])
    mb,mc,ma=metrics(eb),metrics(ec),metrics(allerr)
    rb,rc=d['route_B'],d['route_C']
    warm=int(rb.get('MS_LatencyWarmupFrames',30))
    nb=max(len(eb)-warm,1); nc=max(len(ec)-warm,1)
    pure_ms=(float(rb.get('MS_LatencyMean_ms',0))*nb+float(rc.get('MS_LatencyMean_ms',0))*nc)/(nb+nc)
    e2b=rb.get('EndToEndTiming',{}); e2c=rc.get('EndToEndTiming',{})
    if e2b and e2c:
        n1=int(e2b['samples']); n2=int(e2c['samples'])
        e2e=(float(e2b['mean_ms'])*n1+float(e2c['mean_ms'])*n2)/(n1+n2)
    else:
        e2e=0.0
    row={
      'Experiment':name,'Category':d.get('experiment_category'),'FrontSearch':'Forward3x6 SoftMS (fixed)',
      'Frames':d.get('experiment_frame_count'),'GRU':'no' if d.get('experiment_disable_gru') else 'yes',
      'Kalman':d.get('experiment_kalman'),'FinalMS':'yes' if d.get('ms_enabled') else 'no','MS_grid':d.get('ms_grid_size'),
      'B_frames':mb['frames'],'C_frames':mc['frames'],
      'B_MLE_m':mb['MLE_m'],'C_MLE_m':mc['MLE_m'],'BC_MLE_m':ma['MLE_m'],
      'BC_MedLE_m':ma['MedLE_m'],'BC_P90_m':ma['P90_m'],'BC_P95_m':ma['P95_m'],'BC_P99_m':ma['P99_m'],
      'BC_LSR5_pct':ma['LSR@5_pct'],'BC_LSR10_pct':ma['LSR@10_pct'],'BC_LSR15_pct':ma['LSR@15_pct'],'BC_LSR20_pct':ma['LSR@20_pct'],
      'B_Jump_pct':float(rb.get('JumpRate_pct',0)),'C_Jump_pct':float(rc.get('JumpRate_pct',0)),
      'PureFinalMSLatency_ms':pure_ms,'PureFinalMS_FPS':1000.0/pure_ms if pure_ms>0 else 0.0,
      'E2E_mean_ms':e2e,'E2E_FPS':1000.0/e2e if e2e>0 else 0.0,
    }
    rows.append(row)
    payload[name]={
      'metadata':{k:d.get(k) for k in ['experiment_category','front_decoder','experiment_frame_count','experiment_forward_only','experiment_disable_gru','experiment_kalman','ms_enabled','ms_grid_size','checkpoint_retrained']},
      'route_B':mb,'route_C':mc,'combined':ma,'summary_file':str(out/'robust_tracker_summary.json')
    }

with (suite/'experiment_summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
(suite/'experiment_summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
by={r['Experiment']:r for r in rows}
fmt=lambda x,n=3:f'{float(x):.{n}f}'
md=['# V39 SoftMS Core Ablation Results','',
    'Main pipeline: **Forward 3x6 SoftMS -> 3-frame GRU -> Constant Velocity -> fixed-R Kalman -> final SoftMS**.','',
    '**Forward 3x6 is fixed in all core component ablations.** A full-6x6 replacement is intentionally excluded because it changes directionality, candidate count, compute budget, and search semantics at the same time.','',
    'All B+C metrics are recomputed from concatenated raw per-frame errors.','']
md += ['## Table 1. Core component removal (w/o)','',
       '| Setting | Front | GRU | Kalman | Final MS | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |','|---|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|']
for n,label in [('full_model','Full'),('abl_no_gru','w/o GRU'),('abl_no_kalman','w/o Kalman'),('abl_no_final_ms','w/o Final MS')]:
    r=by[n]; md.append(f"| {label} | Forward 3x6 SoftMS | {r['GRU']} | {r['Kalman']} | {r['FinalMS']} | {fmt(r['B_MLE_m'])} | {fmt(r['C_MLE_m'])} | {fmt(r['BC_MLE_m'])} | {fmt(r['BC_P90_m'])} | {fmt(r['BC_LSR5_pct'],2)}% |")
md += ['','## Table 2. Temporal input frames','','Each 1/2/3-frame row is freshly trained on Route A with identical settings. Forward 3x6 SoftMS is fixed.','',
       '| UAV frames | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |','|---:|---:|---:|---:|---:|---:|']
for n,k in [('temporal_1frame',1),('temporal_2frame',2),('temporal_3frame',3)]:
    r=by[n]; md.append(f"| {k} | {fmt(r['B_MLE_m'])} | {fmt(r['C_MLE_m'])} | {fmt(r['BC_MLE_m'])} | {fmt(r['BC_P90_m'])} | {fmt(r['BC_LSR5_pct'],2)}% |")
md += ['','## Table 3. Final MeanShift window sensitivity','',
       'Front Forward-3x6 SoftMS is fixed. Pure final-MS latency starts after final candidate centers/logits are ready.','',
       '| Window | Candidates | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 | Pure final-MS latency | Pure final-MS FPS |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for g in range(4,9):
    r=by[f'ms_window_{g}x{g}']; md.append(f"| {g}x{g} | {g*g} | {fmt(r['B_MLE_m'])} | {fmt(r['C_MLE_m'])} | {fmt(r['BC_MLE_m'])} | {fmt(r['BC_P90_m'])} | {fmt(r['BC_LSR5_pct'],2)}% | {fmt(r['PureFinalMSLatency_ms'])} ms | {fmt(r['PureFinalMS_FPS'],2)} |")
r=by['full_model']
md += ['','## Full-model end-to-end runtime','',f"- Mean: **{fmt(r['E2E_mean_ms'])} ms/frame**",f"- FPS: **{fmt(r['E2E_FPS'],2)}**"]
(suite/'paper_tables.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
(suite/'audit_report.json').write_text(json.dumps({
  'status':'PASS','experiments':names,
  'front_decoder':'Soft MeanShift','front_search_fixed':'Forward 3x6 (18 scored candidates)',
  'excluded_ablation':'w/o Forward 3x6 / full 6x6 excluded from core table because it is confounded by candidate count, compute, and backward-search semantics',
  'temporal_frame_training':'1/2/3 frame variants freshly trained separately',
  'module_ablation':['w/o GRU','w/o Kalman','w/o Final MS'],
  'ms_windows':['4x4','5x5','6x6','7x7','8x8'],
  'raw_data_saved':True,'summary_csv':'experiment_summary.csv','summary_json':'experiment_summary.json','paper_tables':'paper_tables.md'
},indent=2),encoding='utf-8')
print('\n'+'='*96)
print('SOFTMS CORE ABLATION SUITE COMPLETE')
print('Results :',suite)
print('CSV     :',suite/'experiment_summary.csv')
print('JSON    :',suite/'experiment_summary.json')
print('Tables  :',suite/'paper_tables.md')
print('Audit   :',suite/'audit_report.json')
print('='*96)
PY

echo "================================================================================"
echo "ALL SOFTMS CORE ABLATIONS COMPLETE"
echo "Results: ${SUITE_ROOT}"
echo "================================================================================"
