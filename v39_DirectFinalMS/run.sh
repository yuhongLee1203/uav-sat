#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BASE_SRC="${ROOT}/base_src"
FEATURE_CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache}"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
BACKBONE="mobilenet_v3_small"
BASE_ARCH="V36_PreviousStateOnly_MobileNetV3_Forward3x6_PolynomialKalman"
FINAL_ARCH="V39_WeightedCentroid_GRU_Kalman_MS"
JITTER_M="${JITTER_M:-8}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
DEFAULT_MOTION="velocity"
DEFAULT_KALMAN="fixed"
DEFAULT_MS_GRID="6"
DEFAULT_MS_BANDWIDTH="7.0"

VISUAL_CKPT="${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt"
ORIGINAL_V39_TEMPORAL_CKPT="${REPO_ROOT}/PreviousState-exp/output/mobilenetv3_prevstate/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -f "${BASE_SRC}/${f}" ]] || { echo "ERROR: missing ${BASE_SRC}/${f}" >&2; exit 2; }
done
[[ -f "${ROOT}/patch_direct_finalms.py" ]] || { echo "ERROR: missing patch_direct_finalms.py" >&2; exit 2; }
[[ -s "${VISUAL_CKPT}" ]] || { echo "ERROR: missing visual checkpoint ${VISUAL_CKPT}" >&2; exit 2; }
for route in route_A route_B route_C; do
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || { echo "ERROR: missing ${DATA_ROOT}/routes/${route}/frames.csv" >&2; exit 2; }
done

# Do not allow later experimental training edits to leak into this suite.
# 1/2/3-frame variants are retrained with the ORIGINAL v39 optimizer/loss code.
python3 - "${BASE_SRC}/config.py" "${BASE_SRC}/robust_tracker.py" "${BASE_SRC}/visual_model.py" <<'PY'
from pathlib import Path
import sys
cfg=Path(sys.argv[1]).read_text(encoding='utf-8')
trk=Path(sys.argv[2]).read_text(encoding='utf-8')
mdl=Path(sys.argv[3]).read_text(encoding='utf-8')
checks={
    'original temporal LR':'TEMPORAL_LR = 2e-4' in cfg,
    'original next-step loss':'LOSS_NEXT_STEP = 3.00' in cfg,
    'original velocity auxiliary weight':'LOSS_VELOCITY = 0.25' in cfg,
    'no cumulative-progress loss':'LOSS_CUMULATIVE_PROGRESS' not in cfg+trk,
    'original v39 next-step formula':'v_parallel + 0.5 * a_parallel' in mdl,
    'controlled_gt_jitter protocol':'controlled_gt_jitter' in cfg,
    'frame-count switch':'EXPERIMENT_FRAME_COUNT' in cfg and 'frame_count == 1' in mdl and 'frame_count == 2' in mdl,
}
for k,v in checks.items(): print(f"[STATIC] {k}: {'PASS' if v else 'FAIL'}")
bad=[k for k,v in checks.items() if not v]
if bad: raise SystemExit('STATIC AUDIT FAILED: '+', '.join(bad))
PY

make_runtime() {
  local out="$1" runtime="$2"
  rm -rf "${runtime}"
  mkdir -p "${runtime}" "${out}/checkpoints" "${FEATURE_CACHE_DIR}"
  cp -a "${BASE_SRC}/." "${runtime}/"
  python3 "${ROOT}/patch_direct_finalms.py" "${runtime}/robust_tracker.py"
  ln -sfn "${VISUAL_CKPT}" "${out}/checkpoints/visual_retrieval_A_only.pt"
}

# Build UAV backbone caches once before parallel training so the three GPU jobs
# only read the shared cache and never race while writing it.
prepare_feature_cache() {
  local out="${SUITE_ROOT}/cache_prep"
  local runtime="${SUITE_ROOT}/runtime_cache_prep"
  make_runtime "${out}" "${runtime}"
  echo "[CACHE] preparing/reusing Route A/B/C UAV backbone caches on GPU0"
  (
    cd "${runtime}"
    CUDA_VISIBLE_DEVICES=0 \
    UAVSAT_DEVICE=cuda:0 \
    UAVSAT_OUTPUT_DIR="${out}" \
    UAVSAT_CHECKPOINT_DIR="${out}/checkpoints" \
    UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR}" \
    UAVSAT_DATA_ROOT="${DATA_ROOT}" \
    UAVSAT_BACKBONE="${BACKBONE}" \
    UAVSAT_ARCHITECTURE_NAME="${BASE_ARCH}" \
    UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
    UAVSAT_EXPERIMENT_ANCHOR=weighted_centroid \
    UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
    UAVSAT_EXPERIMENT_MOTION="${DEFAULT_MOTION}" \
    UAVSAT_EXPERIMENT_KALMAN="${DEFAULT_KALMAN}" \
    python3 - <<'PY'
import config
from robust_tracker import build_route_cache, resolve_device
from visual_localizer import FrozenVisualLocalizer

device=resolve_device()
visual=FrozenVisualLocalizer(device)
for i,name in enumerate(config.ROUTE_NAMES):
    cache=build_route_cache(name, config.ROUTE_ROOTS[i], visual, device)
    print(f"[CACHE] {name}: {len(cache)} frames", flush=True)
PY
  )
}

run_cfg() {
  local gpu="$1" name="$2" frames="$3" kalman="$4" disable_gru="$5"
  local ms_enabled="$6" grid="$7" category="$8" mode="$9"
  local ckpt_source="${10:-}" measure_ms="${11:-0}" measure_e2e="${12:-0}"
  local out="${SUITE_ROOT}/${name}"
  local runtime="${SUITE_ROOT}/runtime_${name}"

  make_runtime "${out}" "${runtime}"
  if [[ "${mode}" == "eval" && "${disable_gru}" == "0" ]]; then
    [[ -s "${ckpt_source}" ]] || { echo "ERROR: missing checkpoint ${ckpt_source}" >&2; return 3; }
    ln -sfn "${ckpt_source}" "${out}/checkpoints/${CKPT_NAME}"
  fi

  echo "[START][${name}][GPU${gpu}] frames=${frames} kalman=${kalman} gru=$((1-disable_gru)) ms=${ms_enabled} grid=${grid} mode=${mode}"
  (
    cd "${runtime}"
    args=(--mode "${mode}" --reuse-visual --jitter-m "${JITTER_M}")
    if [[ "${mode}" == "train_eval" ]]; then
      args+=(--temporal-epochs "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}")
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" \
    UAVSAT_DEVICE=cuda:0 \
    UAVSAT_OUTPUT_DIR="${out}" \
    UAVSAT_CHECKPOINT_DIR="${out}/checkpoints" \
    UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR}" \
    UAVSAT_DATA_ROOT="${DATA_ROOT}" \
    UAVSAT_BACKBONE="${BACKBONE}" \
    UAVSAT_ARCHITECTURE_NAME="${BASE_ARCH}" \
    UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
    UAVSAT_EXPERIMENT_ANCHOR=weighted_centroid \
    UAVSAT_EXPERIMENT_FRAME_COUNT="${frames}" \
    UAVSAT_EXPERIMENT_MOTION="${DEFAULT_MOTION}" \
    UAVSAT_EXPERIMENT_KALMAN="${kalman}" \
    UAVSAT_EXPERIMENT_DISABLE_GRU="${disable_gru}" \
    UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
    MS_ENABLED="${ms_enabled}" \
    MS_GRID_SIZE="${grid}" \
    MS_BANDWIDTH_M="${DEFAULT_MS_BANDWIDTH}" \
    MS_MEASURE_LATENCY="${measure_ms}" \
    MS_LATENCY_WARMUP=30 \
    UAVSAT_MEASURE_LATENCY="${measure_e2e}" \
    UAVSAT_LATENCY_WARMUP=30 \
    python3 -u robust_tracker.py "${args[@]}" 2>&1 | sed -u "s/^/[${name}] /" | tee "${out}/${mode}.log"
  )

  # Write metadata from explicit run_cfg arguments, never from ambient shell
  # defaults. This prevents 1/2-frame rows from being mislabeled as 3-frame.
  python3 - "${out}/robust_tracker_summary.json" "${FINAL_ARCH}" "${name}" "${category}" "${mode}" "${frames}" "${kalman}" "${disable_gru}" "${ms_enabled}" "${grid}" "${DEFAULT_MS_BANDWIDTH}" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8'))
d['architecture']=sys.argv[2]
d['experiment_tag']=sys.argv[3]
d['experiment_category']=sys.argv[4]
d['experiment_run_mode']=sys.argv[5]
d['experiment_frame_count']=int(sys.argv[6])
d['experiment_kalman']=sys.argv[7]
d['experiment_disable_gru']=bool(int(sys.argv[8]))
d['ms_enabled']=bool(int(sys.argv[9]))
d['ms_grid_size']=int(sys.argv[10])
d['experiment_anchor']='weighted_centroid'
d['experiment_motion']='velocity'
d['final_chain']='Weighted Centroid -> GRU -> fixed-R Kalman -> one final MS -> Final Position'
d['training_definition']='original v39 training code/hyperparameters; only front decoder is Weighted Centroid'
d['ms_hyperparameters']={'bandwidth_m':float(sys.argv[11])}
p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
PY
  echo "[DONE][${name}]"
}

if [[ "${RUN_ALL_EXPERIMENTS:-0}" == "1" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  SUITE_ROOT="${EXPERIMENT_SUITE_DIR:-${ROOT}/wc_final_ablation_${TS}}"
  mkdir -p "${SUITE_ROOT}" "${FEATURE_CACHE_DIR}"

  echo "============================================================================================================"
  echo "FINAL NECESSARY ABLATION SUITE"
  echo "Method: front MS1 -> Weighted Centroid; original v39 training code otherwise unchanged"
  echo "KEEP: Constant Velocity and fixed-R Kalman as fixed main settings (not separate design tables)"
  echo "RUN: architecture ablation + fair 1/2/3-frame ablation + final-MS window/pure decoder latency + E2E runtime"
  echo "GPU plan: 0/5/6 train 1/2/3-frame variants in parallel"
  echo "MS latency definition: candidates + regularized logits READY -> one MeanShift -> metric XY"
  echo "============================================================================================================"

  prepare_feature_cache

  # Fair temporal-input ablation: each frame-count variant is trained from
  # scratch with exactly the same original v39 training code/hyperparameters.
  ( run_cfg 0 temporal_1frame 1 "${DEFAULT_KALMAN}" 0 1 "${DEFAULT_MS_GRID}" temporal_frames train_eval "" 0 0 ) & p0=$!
  ( run_cfg 5 temporal_2frame 2 "${DEFAULT_KALMAN}" 0 1 "${DEFAULT_MS_GRID}" temporal_frames train_eval "" 0 0 ) & p5=$!
  ( run_cfg 6 temporal_3frame 3 "${DEFAULT_KALMAN}" 0 1 "${DEFAULT_MS_GRID}" temporal_frames train_eval "" 0 0 ) & p6=$!
  status=0
  wait "${p0}" || status=1
  wait "${p5}" || status=1
  wait "${p6}" || status=1
  [[ "${status}" == "0" ]] || { echo "ERROR: temporal frame-count training failed" >&2; exit 20; }

  CKPT1="${SUITE_ROOT}/temporal_1frame/checkpoints/${CKPT_NAME}"
  CKPT2="${SUITE_ROOT}/temporal_2frame/checkpoints/${CKPT_NAME}"
  CKPT3="${SUITE_ROOT}/temporal_3frame/checkpoints/${CKPT_NAME}"
  for ck in "${CKPT1}" "${CKPT2}" "${CKPT3}"; do
    [[ -f "${ck}" && ! -L "${ck}" ]] || { echo "ERROR: expected freshly trained checkpoint missing: ${ck}" >&2; exit 21; }
  done

  # Progressive architecture ablation. No extra Kalman-design or motion-model
  # experiments: Kalman contribution is already measured by +Kalman here.
  run_cfg 0 abl_wc_only       3 none                1 0 "${DEFAULT_MS_GRID}" module_ablation eval ""        0 0
  run_cfg 0 abl_wc_gru        3 none                0 0 "${DEFAULT_MS_GRID}" module_ablation eval "${CKPT3}" 0 0
  run_cfg 0 abl_wc_gru_kalman 3 "${DEFAULT_KALMAN}" 0 0 "${DEFAULT_MS_GRID}" module_ablation eval "${CKPT3}" 0 0

  # Pure final-MeanShift decoder timing. Run sequentially on GPU5 after all
  # other jobs are finished. Search/indexing/SAT scoring/prior-logit creation are
  # outside this timer by construction.
  for g in 4 5 6 7 8; do
    run_cfg 5 "sens_ms_grid${g}x${g}" 3 "${DEFAULT_KALMAN}" 0 1 "${g}" ms_window eval "${CKPT3}" 1 0
  done

  # Isolated online end-to-end runtime; no concurrent jobs at this point.
  run_cfg 6 runtime_e2e 3 "${DEFAULT_KALMAN}" 0 1 "${DEFAULT_MS_GRID}" runtime eval "${CKPT3}" 0 1

  python3 - "${SUITE_ROOT}" <<'PY'
import csv,json,sys
from pathlib import Path
suite=Path(sys.argv[1]); NB,NC=2276,1258

def read(n):
    p=suite/n/'robust_tracker_summary.json'
    if not p.exists(): raise SystemExit(f'AUDIT FAILED: missing {p}')
    return json.loads(p.read_text(encoding='utf-8'))

def w(b,c): return (float(b)*NB+float(c)*NC)/(NB+NC)
def bc(d,key): return w(d['route_B'][key],d['route_C'][key])

names=['temporal_1frame','temporal_2frame','temporal_3frame','abl_wc_only','abl_wc_gru','abl_wc_gru_kalman']+[f'sens_ms_grid{i}x{i}' for i in range(4,9)]+['runtime_e2e']
d={n:read(n) for n in names}

expected={
 'temporal_1frame':(1,'fixed',False,True),
 'temporal_2frame':(2,'fixed',False,True),
 'temporal_3frame':(3,'fixed',False,True),
 'abl_wc_only':(3,'none',True,False),
 'abl_wc_gru':(3,'none',False,False),
 'abl_wc_gru_kalman':(3,'fixed',False,False),
 'runtime_e2e':(3,'fixed',False,True),
}
for i in range(4,9): expected[f'sens_ms_grid{i}x{i}']=(3,'fixed',False,True)
for name,x in d.items():
    frames,kalman,disable_gru,ms=expected[name]
    if x.get('reference_protocol')!='controlled_gt_jitter': raise SystemExit(f'AUDIT FAILED [{name}]: protocol')
    if x.get('experiment_anchor')!='weighted_centroid': raise SystemExit(f'AUDIT FAILED [{name}]: decoder')
    if int(x.get('experiment_frame_count',-1))!=frames: raise SystemExit(f'AUDIT FAILED [{name}]: frames')
    if str(x.get('experiment_kalman'))!=kalman: raise SystemExit(f'AUDIT FAILED [{name}]: kalman')
    if bool(x.get('experiment_disable_gru'))!=disable_gru: raise SystemExit(f'AUDIT FAILED [{name}]: GRU')
    if bool(x.get('ms_enabled'))!=ms: raise SystemExit(f'AUDIT FAILED [{name}]: MS')
    if str(x.get('experiment_motion'))!='velocity': raise SystemExit(f'AUDIT FAILED [{name}]: motion must stay fixed velocity')
    for route in ('route_B','route_C'):
        r=x[route]
        if r.get('VisualObservationDecoder')!='posterior weighted centroid': raise SystemExit(f'AUDIT FAILED [{name}/{route}]: visual decoder')
        if int(r.get('OnlineMeanShiftCount',-1))!=(1 if ms else 0): raise SystemExit(f'AUDIT FAILED [{name}/{route}]: MeanShift count')

for i in range(4,9):
    x=d[f'sens_ms_grid{i}x{i}']
    for route in ('route_B','route_C'):
        definition=x[route].get('MS_LatencyDefinition','')
        if 'regularized logits already prepared' not in definition:
            raise SystemExit(f'AUDIT FAILED [grid{i}/{route}]: MS timer definition')

for n in ('temporal_1frame','temporal_2frame','temporal_3frame'):
    ck=suite/n/'checkpoints'/'controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt'
    if not ck.exists() or ck.is_symlink(): raise SystemExit(f'AUDIT FAILED [{n}]: checkpoint is not freshly trained')

rows=[]
for name,x in d.items():
    b,c=x['route_B'],x['route_C']
    lat=w(b.get('MS_LatencyMean_ms',0),c.get('MS_LatencyMean_ms',0))
    rows.append({'Experiment':name,'Frames':x.get('experiment_frame_count'),'GRU':'no' if x.get('experiment_disable_gru') else 'yes','Kalman':x.get('experiment_kalman'),'MS':'yes' if x.get('ms_enabled') else 'no','MS_grid':x.get('ms_grid_size','-'),'B_MLE_m':b['MLE_m'],'C_MLE_m':c['MLE_m'],'BC_MLE_m':bc(x,'MLE_m'),'BC_P90_m':bc(x,'P90_m'),'BC_LSR5_pct':bc(x,'LSR@5_pct'),'B_Jump_pct':b['JumpRate_pct'],'C_Jump_pct':c['JumpRate_pct'],'PureMS_latency_ms':lat,'PureMS_FPS':1000.0/lat if lat>0 else 0.0})
with (suite/'experiment_summary.csv').open('w',newline='',encoding='utf-8') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

fmt=lambda v,n=3:f'{float(v):.{n}f}'
md=['# Final Weighted-Centroid Ablation Tables','',
    'Main method: **Weighted Centroid -> 3-frame GRU -> fixed-R external Kalman -> one final MeanShift**.','',
    'Constant Velocity is fixed for every experiment; there is no separate motion-model table. The Kalman is not trained; its contribution is tested only by the progressive architecture ablation.','']
md += ['## Table 1. Progressive architecture ablation','','| Setting | B MLE | C MLE | B+C MLE | B+C LSR@5 | B/C Jump |','|---|---:|---:|---:|---:|---:|']
for n,label in [('abl_wc_only','Weighted Centroid'),('abl_wc_gru','+ GRU'),('abl_wc_gru_kalman','+ Kalman'),('temporal_3frame','+ final MS')]:
    x=d[n]; md.append(f"| {label} | {fmt(x['route_B']['MLE_m'])} | {fmt(x['route_C']['MLE_m'])} | {fmt(bc(x,'MLE_m'))} | {fmt(bc(x,'LSR@5_pct'),2)}% | {fmt(x['route_B']['JumpRate_pct'],3)}/{fmt(x['route_C']['JumpRate_pct'],3)}% |")

md += ['','## Table 2. Temporal input-frame ablation','','Each row is trained separately on Route A with the same original v39 training settings.','',
       '| UAV frames | Temporal features | B MLE | C MLE | B+C MLE | B+C P90 |','|---:|---|---:|---:|---:|---:|']
for n,k,meaning in [('temporal_1frame',1,'current frame only'),('temporal_2frame',2,'current + previous; first difference'),('temporal_3frame',3,'current + previous two; first + second difference')]:
    x=d[n]; md.append(f"| {k} | {meaning} | {fmt(x['route_B']['MLE_m'])} | {fmt(x['route_C']['MLE_m'])} | {fmt(bc(x,'MLE_m'))} | {fmt(bc(x,'P90_m'))} |")

md += ['','## Table 3. Final MeanShift window sensitivity and PURE decoder latency','',
       'Latency starts only after the final candidate centers and regularized logits are ready. It measures **one soft_mean_shift call through metric XY output only**; candidate search/indexing, SAT projection, similarity scoring and prior-logit construction are excluded.','',
       '| Window | Candidates | B+C MLE | Pure MS latency | Pure MS FPS |','|---|---:|---:|---:|---:|']
for g in range(4,9):
    x=d[f'sens_ms_grid{g}x{g}']; lat=w(x['route_B']['MS_LatencyMean_ms'],x['route_C']['MS_LatencyMean_ms']); label=f'{g}x{g}'+(' (main)' if g==6 else '')
    md.append(f"| {label} | {g*g} | {fmt(bc(x,'MLE_m'))} | {fmt(lat)} ms | {fmt(1000.0/lat,1) if lat>0 else '-'} |")

rt=d['runtime_e2e']; eb=rt['route_B'].get('EndToEndTiming',{}); ec=rt['route_C'].get('EndToEndTiming',{})
if not eb or not ec: raise SystemExit('AUDIT FAILED: missing E2E timing')
e2e=w(eb['mean_ms'],ec['mean_ms'])
md += ['','## Table 4. Online end-to-end runtime','',
       'Prepared UAV tensor -> backbone -> Weighted Centroid -> 3-frame GRU -> fixed-R Kalman -> one final MS -> XY.','',
       f"- Route B: {fmt(eb['mean_ms'])} ms / {fmt(eb['fps'],1)} FPS",
       f"- Route C: {fmt(ec['mean_ms'])} ms / {fmt(ec['fps'],1)} FPS",
       f"- Weighted B+C: **{fmt(e2e)} ms / {fmt(1000.0/e2e,1)} FPS**"]
(suite/'paper_tables.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
(suite/'audit_report.json').write_text(json.dumps({
  'status':'PASS',
  'method':'front SoftMS/MS1 replaced by posterior Weighted Centroid + posterior spatial variance',
  'training':'1/2/3-frame variants freshly trained using unchanged original v39 training code/hyperparameters',
  'fixed_motion':'Constant Velocity; no motion-model ablation',
  'kalman':'fixed-R external Kalman; no learned-variance ablation; presence/absence tested in architecture table',
  'ms_latency':'candidate centers + regularized logits ready -> one soft_mean_shift -> metric XY',
  'gpus':'frame training uses GPU0/5/6 in parallel; timing phases run without concurrent experiment jobs'
},indent=2),encoding='utf-8')
print('AUDIT PASS')
print(suite/'paper_tables.md')
PY

  echo "============================================================================================================"
  echo "ALL NECESSARY ABLATIONS COMPLETED + AUDIT PASS"
  echo "Results: ${SUITE_ROOT}"
  echo "Tables : ${SUITE_ROOT}/paper_tables.md"
  echo "Audit  : ${SUITE_ROOT}/audit_report.json"
  echo "============================================================================================================"
  exit 0
fi

# Simple single inference path. No retraining unless explicitly requested through
# the full suite above. By default it reuses the original v39 3-frame checkpoint.
OUT="${UAVSAT_OUTPUT_DIR:-${ROOT}/output_wc_single}"
RUNTIME="${UAVSAT_RUNTIME_DIR:-${ROOT}/runtime_wc_single}"
make_runtime "${OUT}" "${RUNTIME}"
CKPT_SOURCE="${UAVSAT_TEMPORAL_CKPT_SOURCE:-${ORIGINAL_V39_TEMPORAL_CKPT}}"
[[ -s "${CKPT_SOURCE}" ]] || { echo "ERROR: missing temporal checkpoint ${CKPT_SOURCE}" >&2; exit 3; }
ln -sfn "${CKPT_SOURCE}" "${OUT}/checkpoints/${CKPT_NAME}"
(
  cd "${RUNTIME}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  UAVSAT_DEVICE="${UAVSAT_DEVICE:-cuda:0}" \
  UAVSAT_OUTPUT_DIR="${OUT}" \
  UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
  UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR}" \
  UAVSAT_DATA_ROOT="${DATA_ROOT}" \
  UAVSAT_BACKBONE="${BACKBONE}" \
  UAVSAT_ARCHITECTURE_NAME="${BASE_ARCH}" \
  UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
  UAVSAT_EXPERIMENT_ANCHOR=weighted_centroid \
  UAVSAT_EXPERIMENT_FRAME_COUNT="${UAVSAT_EXPERIMENT_FRAME_COUNT:-3}" \
  UAVSAT_EXPERIMENT_MOTION="${DEFAULT_MOTION}" \
  UAVSAT_EXPERIMENT_KALMAN="${DEFAULT_KALMAN}" \
  UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
  MS_ENABLED="${MS_ENABLED:-1}" \
  MS_GRID_SIZE="${MS_GRID_SIZE:-6}" \
  MS_BANDWIDTH_M="${MS_BANDWIDTH_M:-7.0}" \
  python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m "${JITTER_M}" 2>&1 | tee "${OUT}/eval.log"
)
