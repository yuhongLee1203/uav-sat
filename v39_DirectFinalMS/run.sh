#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
SCRIPT_PATH="${ROOT}/run.sh"
BASE_SRC="${ROOT}/base_src"
OUT="${UAVSAT_OUTPUT_DIR:-${ROOT}/output_wc_clean}"
SRC="${UAVSAT_RUNTIME_DIR:-${ROOT}/runtime_wc_clean}"
FEATURE_CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache}"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
JITTER_M="${JITTER_M:-8}"
BACKBONE="mobilenet_v3_small"
BASE_ARCH="V36_PreviousStateOnly_MobileNetV3_Forward3x6_PolynomialKalman"
FINAL_ARCH="V39_WeightedCentroid_GRU_Kalman_MS"

# IMPORTANT: these are the original v39 inference settings. The only method
# change in this runner is front SoftMS -> posterior Weighted Centroid.
DEFAULT_MOTION="${DEFAULT_MOTION:-velocity}"
DEFAULT_KALMAN="${DEFAULT_KALMAN:-fixed}"
DEFAULT_MS_GRID="${DEFAULT_MS_GRID:-6}"
DEFAULT_MS_BANDWIDTH="${DEFAULT_MS_BANDWIDTH:-7.0}"

VISUAL_CKPT="${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt"
TEMPORAL_CKPT="${REPO_ROOT}/PreviousState-exp/output/mobilenetv3_prevstate/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -f "${BASE_SRC}/${f}" ]] || { echo "ERROR: missing ${BASE_SRC}/${f}" >&2; exit 2; }
done
[[ -f "${ROOT}/patch_direct_finalms.py" ]] || { echo "ERROR: missing patch_direct_finalms.py" >&2; exit 2; }
[[ -s "${VISUAL_CKPT}" ]] || { echo "ERROR: missing visual checkpoint ${VISUAL_CKPT}" >&2; exit 2; }
[[ -s "${TEMPORAL_CKPT}" ]] || {
  echo "ERROR: original v39 temporal checkpoint is missing: ${TEMPORAL_CKPT}" >&2
  echo "This clean comparison intentionally does NOT retrain or alter the temporal model." >&2
  exit 2
}
for route in route_A route_B route_C; do
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || { echo "ERROR: missing ${DATA_ROOT}/routes/${route}/frames.csv" >&2; exit 2; }
done

# Verify that none of the later experimental training modifications leaked into
# base_src. This runner must be the original v39 training/model definition.
python3 - "${BASE_SRC}/config.py" "${BASE_SRC}/robust_tracker.py" "${BASE_SRC}/visual_model.py" <<'PY'
from pathlib import Path
import sys
cfg=Path(sys.argv[1]).read_text(encoding='utf-8')
trk=Path(sys.argv[2]).read_text(encoding='utf-8')
mdl=Path(sys.argv[3]).read_text(encoding='utf-8')
checks={
    'original temporal LR': 'TEMPORAL_LR = 2e-4' in cfg,
    'no cumulative-progress loss': 'LOSS_CUMULATIVE_PROGRESS' not in cfg+trk,
    'original next-step formula': 'v_parallel + 0.5 * a_parallel' in mdl,
    'original controlled protocol exists': 'controlled_gt_jitter' in cfg,
}
for k,v in checks.items(): print(f"[STATIC] {k}: {'PASS' if v else 'FAIL'}")
bad=[k for k,v in checks.items() if not v]
if bad: raise SystemExit('STATIC AUDIT FAILED: '+', '.join(bad))
PY

run_one() {
  local gpu="$1"
  local name="$2"
  local motion="$3"
  local kalman="$4"
  local disable_gru="$5"
  local ms_enabled="$6"
  local ms_grid="$7"
  local bandwidth="$8"
  local category="$9"
  local measure_ms="${10:-0}"
  local measure_e2e="${11:-0}"
  local out="${SUITE_ROOT:-${OUT}}/${name}"
  local runtime="${SUITE_ROOT:-${OUT}}/runtime_${name}"

  echo "============================================================================================================"
  echo "[START][${name}][GPU ${gpu}] gru=$((1-disable_gru)) kalman=${kalman} motion=${motion} ms=${ms_enabled} grid=${ms_grid} bw=${bandwidth}"
  echo "front decoder=weighted centroid | temporal checkpoint=fixed original v39"
  echo "============================================================================================================"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  UAVSAT_DEVICE=cuda:0 \
  UAVSAT_OUTPUT_DIR="${out}" \
  UAVSAT_RUNTIME_DIR="${runtime}" \
  UAVSAT_FEATURE_CACHE_DIR_OVERRIDE="${FEATURE_CACHE_DIR}" \
  UAVSAT_EXPERIMENT_ANCHOR=weighted_centroid \
  UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
  UAVSAT_EXPERIMENT_MOTION="${motion}" \
  UAVSAT_EXPERIMENT_KALMAN="${kalman}" \
  UAVSAT_EXPERIMENT_DISABLE_GRU="${disable_gru}" \
  UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
  MS_ENABLED="${ms_enabled}" \
  MS_GRID_SIZE="${ms_grid}" \
  MS_BANDWIDTH_M="${bandwidth}" \
  MS_MEASURE_LATENCY="${measure_ms}" \
  MS_LATENCY_WARMUP=30 \
  UAVSAT_MEASURE_LATENCY="${measure_e2e}" \
  UAVSAT_LATENCY_WARMUP=30 \
  EXPERIMENT_TAG="${name}" \
  EXPERIMENT_CATEGORY="${category}" \
  RUN_ALL_EXPERIMENTS=0 \
  bash "${SCRIPT_PATH}"

  echo "[DONE][${name}]"
}

if [[ "${RUN_ALL_EXPERIMENTS:-0}" == "1" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  SUITE_ROOT="${EXPERIMENT_SUITE_DIR:-${ROOT}/wc_clean_experiments_${TS}}"
  mkdir -p "${SUITE_ROOT}" "${FEATURE_CACHE_DIR}"

  echo "============================================================================================================"
  echo "CLEAN v39 ABLATION SUITE"
  echo "ONLY method change: front MS1 -> Weighted Centroid + posterior spatial variance"
  echo "UNCHANGED: original checkpoint, LR/loss definition, motion model code, Kalman, teacher forcing, jitter/protocol"
  echo "Selected v39 operating point remains: motion=${DEFAULT_MOTION}, Kalman=${DEFAULT_KALMAN}, final MS=${DEFAULT_MS_GRID}x${DEFAULT_MS_GRID}, BW=${DEFAULT_MS_BANDWIDTH}m"
  echo "All runs are sequential so runtime measurements are not contaminated by concurrent jobs."
  echo "Results: ${SUITE_ROOT}"
  echo "============================================================================================================"

  # Canonical model and architecture ablation. Same original checkpoint for all.
  run_one 0 full_model "${DEFAULT_MOTION}" "${DEFAULT_KALMAN}" 0 1 "${DEFAULT_MS_GRID}" "${DEFAULT_MS_BANDWIDTH}" module_ablation 0 0
  run_one 0 abl_gru_only "${DEFAULT_MOTION}" none 0 0 "${DEFAULT_MS_GRID}" "${DEFAULT_MS_BANDWIDTH}" module_ablation 0 0
  run_one 0 abl_gru_kalman "${DEFAULT_MOTION}" "${DEFAULT_KALMAN}" 0 0 "${DEFAULT_MS_GRID}" "${DEFAULT_MS_BANDWIDTH}" module_ablation 0 0

  # Motion-model ablation. GRU temporal input remains 3 frames in all rows.
  run_one 0 design_motion_none none "${DEFAULT_KALMAN}" 0 1 "${DEFAULT_MS_GRID}" "${DEFAULT_MS_BANDWIDTH}" motion_model 0 0
  run_one 0 design_motion_acceleration quadratic "${DEFAULT_KALMAN}" 0 1 "${DEFAULT_MS_GRID}" "${DEFAULT_MS_BANDWIDTH}" motion_model 0 0

  # Kalman design ablation.
  run_one 0 design_kalman_none "${DEFAULT_MOTION}" none 0 1 "${DEFAULT_MS_GRID}" "${DEFAULT_MS_BANDWIDTH}" kalman_design 0 0
  run_one 0 design_kalman_learned "${DEFAULT_MOTION}" learned 0 1 "${DEFAULT_MS_GRID}" "${DEFAULT_MS_BANDWIDTH}" kalman_design 0 0

  # Final-MS grid sensitivity and accurate stage latency. All points run alone,
  # sequentially, on the same GPU. Timer contains candidate indexing/scoring +
  # exactly ONE final MeanShift; no hidden candidate_batch SoftMS is executed.
  for g in 4 5 6 7 8; do
    run_one 5 "sens_ms_grid${g}x${g}" "${DEFAULT_MOTION}" "${DEFAULT_KALMAN}" 0 1 "${g}" "${DEFAULT_MS_BANDWIDTH}" ms_window 1 0
  done

  # MeanShift bandwidth sensitivity. Fixed original 6x6 window; no latency claim.
  for bw in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    run_one 6 "sens_ms_bandwidth${bw}" "${DEFAULT_MOTION}" "${DEFAULT_KALMAN}" 0 1 "${DEFAULT_MS_GRID}" "${bw}.0" meanshift_bandwidth 0 0
  done

  # One isolated end-to-end runtime run on the canonical operating point.
  run_one 0 runtime_e2e "${DEFAULT_MOTION}" "${DEFAULT_KALMAN}" 0 1 "${DEFAULT_MS_GRID}" "${DEFAULT_MS_BANDWIDTH}" runtime 0 1

  python3 - "${SUITE_ROOT}" "${DEFAULT_MS_GRID}" "${DEFAULT_MS_BANDWIDTH}" <<'PY'
import csv, json, math, sys
from pathlib import Path
suite=Path(sys.argv[1]); fixed_grid=int(sys.argv[2]); fixed_bw=float(sys.argv[3])
NB,NC=2276,1258

def read(name):
    p=suite/name/'robust_tracker_summary.json'
    if not p.exists(): raise SystemExit(f'AUDIT FAILED: missing {p}')
    return json.loads(p.read_text(encoding='utf-8'))

def w(b,c): return (float(b)*NB+float(c)*NC)/(NB+NC)

def bc(d,key): return w(d['route_B'][key],d['route_C'][key])

names=['full_model','abl_gru_only','abl_gru_kalman','design_motion_none','design_motion_acceleration','design_kalman_none','design_kalman_learned']+[f'sens_ms_grid{i}x{i}' for i in range(4,9)]+[f'sens_ms_bandwidth{i}' for i in range(1,15)]+['runtime_e2e']
d={n:read(n) for n in names}

# Method audit: all runs must differ from original v39 only at the front decoder.
for name,x in d.items():
    if x.get('reference_protocol')!='controlled_gt_jitter':
        raise SystemExit(f'AUDIT FAILED [{name}]: reference protocol changed')
    if x.get('experiment_anchor')!='weighted_centroid':
        raise SystemExit(f'AUDIT FAILED [{name}]: front decoder is not weighted centroid')
    for route in ('route_B','route_C'):
        r=x[route]
        if r.get('VisualObservationDecoder')!='posterior weighted centroid':
            raise SystemExit(f'AUDIT FAILED [{name}/{route}]: decoder metadata')
        expected=1 if bool(x.get('ms_enabled',True)) else 0
        if int(r.get('OnlineMeanShiftCount',-1))!=expected:
            raise SystemExit(f'AUDIT FAILED [{name}/{route}]: online MeanShift count')

rows=[]
for name,x in d.items():
    b,c=x['route_B'],x['route_C']
    lat=0.0
    if float(b.get('MS_LatencyMean_ms',0))>0 and float(c.get('MS_LatencyMean_ms',0))>0:
        lat=w(b['MS_LatencyMean_ms'],c['MS_LatencyMean_ms'])
    rows.append({
        'Experiment':name,
        'Motion':x.get('experiment_motion'),
        'Kalman':x.get('experiment_kalman'),
        'GRU':'no' if x.get('experiment_disable_gru') else 'yes',
        'MS':'yes' if x.get('ms_enabled',True) else 'no',
        'MS_grid':x.get('ms_grid_size','-'),
        'B_MLE_m':b['MLE_m'],'C_MLE_m':c['MLE_m'],'BC_MLE_m':bc(x,'MLE_m'),
        'BC_P90_m':bc(x,'P90_m'),'BC_LSR5_pct':bc(x,'LSR@5_pct'),
        'B_Jump_pct':b['JumpRate_pct'],'C_Jump_pct':c['JumpRate_pct'],
        'MS_latency_ms':lat,'MS_FPS':1000.0/lat if lat>0 else 0.0,
    })
with (suite/'experiment_summary.csv').open('w',newline='',encoding='utf-8') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

fmt=lambda v,n=3:f'{float(v):.{n}f}'
md=['# Clean v39 Weighted-Centroid Ablation Tables','',
    '**Only methodological change:** front SoftMS/MS1 is replaced by posterior Weighted Centroid with posterior spatial variance. The original v39 temporal checkpoint and all remaining inference settings are unchanged.','']
md += ['## Table 1. Progressive architecture ablation','','| Setting | B MLE | C MLE | B+C MLE | B+C LSR@5 | B/C Jump |','|---|---:|---:|---:|---:|---:|']
for n,label in [('abl_gru_only','WC + GRU'),('abl_gru_kalman','WC + GRU + Kalman'),('full_model','WC + GRU + Kalman + MS')]:
    x=d[n]; md.append(f"| {label} | {fmt(x['route_B']['MLE_m'])} | {fmt(x['route_C']['MLE_m'])} | {fmt(bc(x,'MLE_m'))} | {fmt(bc(x,'LSR@5_pct'),2)}% | {fmt(x['route_B']['JumpRate_pct'],3)}/{fmt(x['route_C']['JumpRate_pct'],3)}% |")

md += ['','## Table 2. Motion prediction model (all use the original 3-frame GRU)','','| Motion | B MLE | C MLE | B+C MLE |','|---|---:|---:|---:|']
for n,label in [('design_motion_none','No learned motion'),('full_model','Constant Velocity'),('design_motion_acceleration','Velocity + Acceleration')]:
    x=d[n]; md.append(f"| {label} | {fmt(x['route_B']['MLE_m'])} | {fmt(x['route_C']['MLE_m'])} | {fmt(bc(x,'MLE_m'))} |")

md += ['','## Table 3. Kalman design','','| Kalman | B MLE | C MLE | B+C MLE | B/C Jump |','|---|---:|---:|---:|---:|']
for n,label in [('design_kalman_none','No Kalman'),('design_kalman_learned','Learned variance'),('full_model','Fixed variance')]:
    x=d[n]; md.append(f"| {label} | {fmt(x['route_B']['MLE_m'])} | {fmt(x['route_C']['MLE_m'])} | {fmt(bc(x,'MLE_m'))} | {fmt(x['route_B']['JumpRate_pct'],3)}/{fmt(x['route_C']['JumpRate_pct'],3)}% |")

md += ['','## Table 4. Final-MS window sensitivity and correctly isolated stage latency','',
       'All rows were run sequentially on GPU 5. Latency is Kalman output → candidate indexing/scoring → exactly one final MeanShift → XY.','',
       '| Window | Candidates | B+C MLE | MS latency | MS FPS |','|---|---:|---:|---:|---:|']
for g in range(4,9):
    x=d[f'sens_ms_grid{g}x{g}']; lat=w(x['route_B']['MS_LatencyMean_ms'],x['route_C']['MS_LatencyMean_ms'])
    label=f'{g}x{g}' + (' (fixed v39 operating point)' if g==fixed_grid else '')
    md.append(f"| {label} | {g*g} | {fmt(bc(x,'MLE_m'))} | {fmt(lat)} ms | {fmt(1000.0/lat,1) if lat>0 else '-'} |")

md += ['','## Table 5. MeanShift bandwidth sensitivity','',f'Final-MS window remains fixed at {fixed_grid}x{fixed_grid}. B/C results are reported as sensitivity only; they are not used to retune the operating point.','',
       '| Bandwidth | B+C MLE | B+C P90 | B+C LSR@5 |','|---:|---:|---:|---:|']
for bw in range(1,15):
    x=d[f'sens_ms_bandwidth{bw}']; label=f'{bw} m'+(' (fixed v39 operating point)' if abs(bw-fixed_bw)<1e-9 else '')
    md.append(f"| {label} | {fmt(bc(x,'MLE_m'))} | {fmt(bc(x,'P90_m'))} | {fmt(bc(x,'LSR@5_pct'),2)}% |")

rt=d['runtime_e2e']; eb=rt['route_B'].get('EndToEndTiming',{}); ec=rt['route_C'].get('EndToEndTiming',{})
if not eb or not ec: raise SystemExit('AUDIT FAILED: missing EndToEndTiming')
e2e=w(eb['mean_ms'],ec['mean_ms'])
md += ['','## Table 6. End-to-end runtime','',
       'Prepared UAV tensor → backbone → Weighted Centroid → original GRU → original Kalman → exactly one final MS → XY.','',
       f"- Route B: {fmt(eb['mean_ms'])} ms / {fmt(eb['fps'],1)} FPS",
       f"- Route C: {fmt(ec['mean_ms'])} ms / {fmt(ec['fps'],1)} FPS",
       f"- Weighted B+C: **{fmt(e2e)} ms / {fmt(1000.0/e2e,1)} FPS**"]
(suite/'paper_tables.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
(suite/'audit_report.json').write_text(json.dumps({
    'status':'PASS',
    'only_method_change':'front SoftMS/MS1 -> posterior Weighted Centroid + posterior spatial variance',
    'temporal_checkpoint':'fixed original v39 Previous-State checkpoint; no retraining',
    'reference_protocol':'controlled_gt_jitter',
    'fixed_motion':'velocity','fixed_kalman':'fixed','fixed_ms_grid':fixed_grid,'fixed_bandwidth_m':fixed_bw,
    'runtime_rule':'no concurrent experiments during MS or E2E timing; final path executes exactly one MeanShift'
},indent=2),encoding='utf-8')
print('AUDIT PASS')
print(suite/'paper_tables.md')
PY

  echo "============================================================================================================"
  echo "ALL CLEAN v39 ABLATIONS COMPLETED + AUDIT PASS"
  echo "Results: ${SUITE_ROOT}"
  echo "Tables : ${SUITE_ROOT}/paper_tables.md"
  echo "Audit  : ${SUITE_ROOT}/audit_report.json"
  echo "============================================================================================================"
  exit 0
fi

# Single-run path. Always use the original temporal checkpoint; never retrain.
rm -rf "${SRC}"
mkdir -p "${SRC}" "${OUT}/checkpoints" "${FEATURE_CACHE_DIR}"
cp -a "${BASE_SRC}/." "${SRC}/"
python3 "${ROOT}/patch_direct_finalms.py" "${SRC}/robust_tracker.py"
ln -sfn "${VISUAL_CKPT}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"
ln -sfn "${TEMPORAL_CKPT}" "${OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MS_KF_SIGMA_M="${MS_KF_SIGMA_M:-4.0}"
export MS_REFERENCE_SIGMA_M="${MS_REFERENCE_SIGMA_M:-4.0}"
export MS_KF_PRIOR_WEIGHT="${MS_KF_PRIOR_WEIGHT:-1.50}"
export MS_REFERENCE_PRIOR_WEIGHT="${MS_REFERENCE_PRIOR_WEIGHT:-2.50}"
export MS_BANDWIDTH_M="${MS_BANDWIDTH_M:-${DEFAULT_MS_BANDWIDTH}}"
export MS_ENABLED="${MS_ENABLED:-1}"
export MS_GRID_SIZE="${MS_GRID_SIZE:-${DEFAULT_MS_GRID}}"
export MS_LATENCY_WARMUP="${MS_LATENCY_WARMUP:-30}"
export MS_MEASURE_LATENCY="${MS_MEASURE_LATENCY:-0}"

cd "${SRC}"
UAVSAT_DEVICE="${DEVICE}" \
UAVSAT_OUTPUT_DIR="${OUT}" \
UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR}" \
UAVSAT_DATA_ROOT="${DATA_ROOT}" \
UAVSAT_BACKBONE="${BACKBONE}" \
UAVSAT_ARCHITECTURE_NAME="${BASE_ARCH}" \
UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
UAVSAT_EXPERIMENT_ANCHOR=weighted_centroid \
UAVSAT_EXPERIMENT_FRAME_COUNT="${UAVSAT_EXPERIMENT_FRAME_COUNT:-3}" \
UAVSAT_EXPERIMENT_MOTION="${UAVSAT_EXPERIMENT_MOTION:-${DEFAULT_MOTION}}" \
UAVSAT_EXPERIMENT_KALMAN="${UAVSAT_EXPERIMENT_KALMAN:-${DEFAULT_KALMAN}}" \
UAVSAT_EXPERIMENT_DISABLE_GRU="${UAVSAT_EXPERIMENT_DISABLE_GRU:-0}" \
UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
UAVSAT_MEASURE_LATENCY="${UAVSAT_MEASURE_LATENCY:-0}" \
UAVSAT_LATENCY_WARMUP="${UAVSAT_LATENCY_WARMUP:-30}" \
python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m "${JITTER_M}" 2>&1 | tee "${OUT}/eval.log"

python3 - "${OUT}/robust_tracker_summary.json" "${FINAL_ARCH}" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8'))
d['architecture']=sys.argv[2]
d['experiment_tag']=os.environ.get('EXPERIMENT_TAG','single_default')
d['experiment_category']=os.environ.get('EXPERIMENT_CATEGORY','single')
d['experiment_anchor']='weighted_centroid'
d['experiment_motion']=os.environ.get('UAVSAT_EXPERIMENT_MOTION','velocity')
d['experiment_kalman']=os.environ.get('UAVSAT_EXPERIMENT_KALMAN','fixed')
d['experiment_disable_gru']=os.environ.get('UAVSAT_EXPERIMENT_DISABLE_GRU','0')=='1'
d['experiment_frame_count']=int(os.environ.get('UAVSAT_EXPERIMENT_FRAME_COUNT','3'))
d['ms_enabled']=os.environ.get('MS_ENABLED','1').lower() not in {'0','false','no','off'}
d['ms_grid_size']=int(os.environ.get('MS_GRID_SIZE','6'))
d['final_chain']='Weighted Centroid -> GRU -> Kalman Filter -> one final MS -> Final Position'
d['temporal_training']='unchanged original v39 checkpoint; no retraining in this experiment suite'
d['only_method_change']='front SoftMS/MS1 -> posterior Weighted Centroid + posterior spatial variance'
d['ms_hyperparameters']={'bandwidth_m':float(os.environ.get('MS_BANDWIDTH_M','7.0'))}
p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
PY

echo "[DONE] result: ${OUT}/robust_tracker_summary.json"
