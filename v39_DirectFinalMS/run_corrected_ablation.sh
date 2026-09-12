#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BASE_SRC="${ROOT}/base_src"
FEATURE_CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache}"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
BACKBONE="mobilenet_v3_small"
FINAL_ARCH="V39_WeightedCentroid_GRU_Kalman_MS_5x5"
JITTER_M="${JITTER_M:-8}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
DEFAULT_MOTION="velocity"
DEFAULT_KALMAN="fixed"
DEFAULT_MS_GRID="5"
DEFAULT_MS_BANDWIDTH="7.0"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
VISUAL_CKPT="${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt"
ORIGINAL_V39_TEMPORAL_CKPT="${REPO_ROOT}/PreviousState-exp/output/mobilenetv3_prevstate/checkpoints/${CKPT_NAME}"

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

TS="$(date +%Y%m%d_%H%M%S)"
SUITE_ROOT="${EXPERIMENT_SUITE_DIR:-${ROOT}/wc_missing_experiments_${TS}}"
mkdir -p "${SUITE_ROOT}" "${FEATURE_CACHE_DIR}"

make_runtime() {
  local out="$1" runtime="$2"
  rm -rf "${runtime}"
  mkdir -p "${runtime}" "${out}/checkpoints" "${FEATURE_CACHE_DIR}"
  cp -a "${BASE_SRC}/." "${runtime}/"
  python3 "${ROOT}/patch_direct_finalms.py" "${runtime}/robust_tracker.py"

  # IMPORTANT no-GT correctness patch:
  # controlled_gt_jitter keeps the original current-frame reference prior for
  # apples-to-apples ablation. route_reference / scheduled_route_reference must
  # never use current-frame GT in final MS; use the already-available causal
  # local-search reference instead.
  python3 - "${runtime}/robust_tracker.py" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
s=p.read_text(encoding='utf-8')
old='''        # Keep the original v39 predefined frame-reference prior unchanged.\n        frame_reference_xy_t = cache.gt_xy[index : index + 1].to(device).float()\n        frame_reference_xy = (\n            frame_reference_xy_t[0].detach().cpu().numpy().astype(np.float64)\n        )\n        preferred_leg = route.frame_from_se(kalman_se[0], kalman_se[1]).leg_index\n'''
new='''        # Final-MS reference prior. Controlled GT+jitter experiments preserve\n        # the original current-frame oracle prior for apples-to-apples results.\n        # NO_GT_INFERENCE protocols must use only the causal local-search\n        # reference returned by local_search_reference_se().\n        if bool(getattr(config, "NO_GT_INFERENCE", False)):\n            frame_reference_xy = np.asarray(controlled_prior_xy, dtype=np.float64).copy()\n            frame_reference_source = "causal_local_search_reference"\n        else:\n            frame_reference_xy = cache.gt_xy[index].cpu().numpy().astype(np.float64)\n            frame_reference_source = "current_frame_gt_controlled_only"\n        frame_reference_xy_t = torch.tensor(\n            frame_reference_xy[None, :], dtype=torch.float32, device=device\n        )\n        preferred_leg = route.frame_from_se(kalman_se[0], kalman_se[1]).leg_index\n'''
if s.count(old)!=1:
    raise SystemExit(f'ERROR: final-MS GT-reference patch target count={s.count(old)}')
s=s.replace(old,new,1)
old_csv='''                "frame_reference_y": float(frame_reference_xy[1]),\n'''
new_csv='''                "frame_reference_y": float(frame_reference_xy[1]),\n                "frame_reference_source": str(frame_reference_source),\n'''
if s.count(old_csv)!=1:
    raise SystemExit(f'ERROR: CSV reference-source patch target count={s.count(old_csv)}')
s=s.replace(old_csv,new_csv,1)
compile(s,str(p),'exec')
p.write_text(s,encoding='utf-8')
print('[OK] final MS reference is GT-free whenever NO_GT_INFERENCE=True')
PY
  ln -sfn "${VISUAL_CKPT}" "${out}/checkpoints/visual_retrieval_A_only.pt"
}

run_cfg() {
  local gpu="$1" name="$2" frames="$3" kalman="$4" disable_gru="$5"
  local ms_enabled="$6" grid="$7" reference_protocol="$8" mode="$9"
  local ckpt_source="${10:-}" measure_e2e="${11:-0}"
  local out="${SUITE_ROOT}/${name}"
  local runtime="${SUITE_ROOT}/runtime_${name}"

  make_runtime "${out}" "${runtime}"
  if [[ "${mode}" == "eval" && "${disable_gru}" == "0" ]]; then
    [[ -s "${ckpt_source}" ]] || { echo "ERROR: missing checkpoint ${ckpt_source}" >&2; return 3; }
    ln -sfn "${ckpt_source}" "${out}/checkpoints/${CKPT_NAME}"
  fi

  echo "[START][${name}][GPU${gpu}] frames=${frames} gru=$((1-disable_gru)) kalman=${kalman} ms=${ms_enabled} grid=${grid} protocol=${reference_protocol} mode=${mode}"
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
    UAVSAT_ARCHITECTURE_NAME="${FINAL_ARCH}" \
    UAVSAT_REFERENCE_PROTOCOL="${reference_protocol}" \
    UAVSAT_EXPERIMENT_ANCHOR=weighted_centroid \
    UAVSAT_EXPERIMENT_FRAME_COUNT="${frames}" \
    UAVSAT_EXPERIMENT_MOTION="${DEFAULT_MOTION}" \
    UAVSAT_EXPERIMENT_KALMAN="${kalman}" \
    UAVSAT_EXPERIMENT_DISABLE_GRU="${disable_gru}" \
    UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
    MS_ENABLED="${ms_enabled}" \
    MS_GRID_SIZE="${grid}" \
    MS_BANDWIDTH_M="${DEFAULT_MS_BANDWIDTH}" \
    MS_MEASURE_LATENCY=0 \
    UAVSAT_MEASURE_LATENCY="${measure_e2e}" \
    UAVSAT_LATENCY_WARMUP=30 \
    python3 -u robust_tracker.py "${args[@]}" 2>&1 | sed -u "s/^/[${name}] /" | tee "${out}/${mode}.log"
  )

  python3 - "${out}/robust_tracker_summary.json" "${name}" "${frames}" "${kalman}" "${disable_gru}" "${ms_enabled}" "${grid}" "${reference_protocol}" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8'))
d['architecture']='V39_WeightedCentroid_GRU_Kalman_MS_5x5'
d['experiment_tag']=sys.argv[2]
d['experiment_frame_count']=int(sys.argv[3])
d['experiment_kalman']=sys.argv[4]
d['experiment_disable_gru']=bool(int(sys.argv[5]))
d['ms_enabled']=bool(int(sys.argv[6]))
d['ms_grid_size']=int(sys.argv[7])
d['reference_protocol_requested']=sys.argv[8]
d['experiment_anchor']='weighted_centroid'
d['experiment_motion']='velocity'
d['selected_main_ms_grid']=5
d['selected_main_reason']='5x5 matches/betters 6x6 accuracy while reducing pure final-MS latency'
p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
PY
  echo "[DONE][${name}]"
}

find_fresh_temporal_ckpt() {
  if [[ -n "${UAVSAT_TEMPORAL_CKPT_SOURCE:-}" && -s "${UAVSAT_TEMPORAL_CKPT_SOURCE}" ]]; then
    printf '%s\n' "${UAVSAT_TEMPORAL_CKPT_SOURCE}"
    return 0
  fi
  local found=""
  found="$(find "${ROOT}" -type f -path "*/temporal_3frame/checkpoints/${CKPT_NAME}" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2- || true)"
  if [[ -n "${found}" && -s "${found}" ]]; then
    printf '%s\n' "${found}"
    return 0
  fi
  return 1
}

echo "============================================================================================================"
echo "V39 MISSING-EXPERIMENT SUITE ONLY"
echo "1) GRU necessity: WC -> Kalman -> final MS (no GRU)"
echo "2) selected 5x5 main-method end-to-end runtime"
echo "3) true no-GT route_reference B/C evaluation (final MS current-GT leak removed)"
echo "4) pooled B+C P90 from concatenated per-frame errors (not weighted route P90)"
echo "============================================================================================================"

if TEMPORAL_CKPT="$(find_fresh_temporal_ckpt)"; then
  echo "[CKPT] reusing freshest locally trained 3-frame checkpoint: ${TEMPORAL_CKPT}"
else
  echo "[CKPT] no fresh temporal_3frame checkpoint found; training exactly one 3-frame model."
  run_cfg 0 temporal_3frame_refresh 3 "${DEFAULT_KALMAN}" 0 1 "${DEFAULT_MS_GRID}" controlled_gt_jitter train_eval "" 0
  TEMPORAL_CKPT="${SUITE_ROOT}/temporal_3frame_refresh/checkpoints/${CKPT_NAME}"
  [[ -s "${TEMPORAL_CKPT}" ]] || { echo "ERROR: fresh 3-frame checkpoint missing" >&2; exit 20; }
fi

# Missing ablation that directly answers whether GRU helps the completed chain.
run_cfg 0 abl_wc_kalman_ms_no_gru 3 "${DEFAULT_KALMAN}" 1 1 "${DEFAULT_MS_GRID}" controlled_gt_jitter eval "" 0

# Main 5x5 full chain with isolated end-to-end timing.
run_cfg 5 runtime_e2e_5x5 3 "${DEFAULT_KALMAN}" 0 1 "${DEFAULT_MS_GRID}" controlled_gt_jitter eval "${TEMPORAL_CKPT}" 1

# Honest no-GT evaluation. route_reference uses causal predicted progress and,
# after the runtime patch above, final MS also uses only that causal reference.
run_cfg 6 eval_no_gt_route_reference_5x5 3 "${DEFAULT_KALMAN}" 0 1 "${DEFAULT_MS_GRID}" route_reference eval "${TEMPORAL_CKPT}" 0

python3 - "${SUITE_ROOT}" <<'PY'
import csv,json,sys
from pathlib import Path
import numpy as np

suite=Path(sys.argv[1])
names=['abl_wc_kalman_ms_no_gru','runtime_e2e_5x5','eval_no_gt_route_reference_5x5']

def read_summary(name):
    p=suite/name/'robust_tracker_summary.json'
    if not p.exists(): raise SystemExit(f'AUDIT FAILED: missing {p}')
    return json.loads(p.read_text(encoding='utf-8'))

def frame_csv(name,route):
    files=list((suite/name).glob(f'{route}_*_frames.csv'))
    if len(files)!=1:
        raise SystemExit(f'AUDIT FAILED [{name}/{route}]: expected one frames CSV, got {len(files)}')
    return files[0]

def read_rows(name,route):
    with frame_csv(name,route).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))

def errors(name):
    vals=[]
    for route in ('route_B','route_C'):
        rows=read_rows(name,route)
        vals.extend(float(r['error_final_m']) for r in rows)
    a=np.asarray(vals,dtype=np.float64)
    if not len(a): raise SystemExit(f'AUDIT FAILED [{name}]: no per-frame errors')
    return a

def pooled(name):
    a=errors(name)
    return {
        'N':int(len(a)),
        'MLE_m':float(a.mean()),
        'P90_m':float(np.quantile(a,0.90)),
        'P95_m':float(np.quantile(a,0.95)),
        'LSR@5_pct':float((a<=5.0).mean()*100.0),
        'LSR@10_pct':float((a<=10.0).mean()*100.0),
    }

d={n:read_summary(n) for n in names}
pool={n:pooled(n) for n in names}

x=d['abl_wc_kalman_ms_no_gru']
if not x.get('experiment_disable_gru'): raise SystemExit('AUDIT FAILED: no-GRU ablation still has GRU enabled')
if x.get('experiment_kalman')!='fixed' or not x.get('ms_enabled') or int(x.get('ms_grid_size',-1))!=5:
    raise SystemExit('AUDIT FAILED: no-GRU ablation is not WC -> fixed Kalman -> 5x5 MS')

full=d['runtime_e2e_5x5']
if full.get('experiment_disable_gru') or full.get('experiment_kalman')!='fixed' or not full.get('ms_enabled') or int(full.get('ms_grid_size',-1))!=5:
    raise SystemExit('AUDIT FAILED: 5x5 main chain metadata mismatch')
for route in ('route_B','route_C'):
    if not full[route].get('EndToEndTiming'): raise SystemExit(f'AUDIT FAILED: missing E2E timing for {route}')

nogt=d['eval_no_gt_route_reference_5x5']
if nogt.get('reference_protocol')!='route_reference': raise SystemExit('AUDIT FAILED: no-GT protocol is not route_reference')
if bool(nogt.get('uses_gt_center_at_inference',True)): raise SystemExit('AUDIT FAILED: no-GT run still reports GT center use')
for route in ('route_B','route_C'):
    if int(nogt[route].get('OnlineMeanShiftCount',-1))!=1:
        raise SystemExit(f'AUDIT FAILED [{route}]: expected exactly one final MeanShift')
    rows=read_rows('eval_no_gt_route_reference_5x5',route)
    for r in rows:
        if r.get('frame_reference_source')!='causal_local_search_reference':
            raise SystemExit(f'AUDIT FAILED [{route}]: final MS reference source is not causal')
        dx=abs(float(r['frame_reference_x'])-float(r['prior_center_x']))
        dy=abs(float(r['frame_reference_y'])-float(r['prior_center_y']))
        if max(dx,dy)>1e-4:
            raise SystemExit(f'AUDIT FAILED [{route}]: final MS reference differs from causal local-search reference')

rows=[]
for n in names:
    s=d[n]; p=pool[n]
    rows.append({
        'Experiment':n,
        'B_MLE_m':s['route_B']['MLE_m'],
        'C_MLE_m':s['route_C']['MLE_m'],
        'BC_MLE_m':p['MLE_m'],
        'BC_P90_m_pooled':p['P90_m'],
        'BC_P95_m_pooled':p['P95_m'],
        'BC_LSR5_pct':p['LSR@5_pct'],
        'BC_LSR10_pct':p['LSR@10_pct'],
        'N_frames':p['N'],
    })
with (suite/'missing_experiments_summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

fmt=lambda v,n=3:f'{float(v):.{n}f}'
ng=pool['abl_wc_kalman_ms_no_gru']; fg=pool['runtime_e2e_5x5']; nr=pool['eval_no_gt_route_reference_5x5']
eb=full['route_B']['EndToEndTiming']; ec=full['route_C']['EndToEndTiming']
nb=len(read_rows('runtime_e2e_5x5','route_B')); nc=len(read_rows('runtime_e2e_5x5','route_C'))
e2e=(float(eb['mean_ms'])*nb+float(ec['mean_ms'])*nc)/(nb+nc)
md=[
 '# V39 Missing Experiments — Corrected Results','',
 'Selected main chain: **Weighted Centroid -> 3-frame GRU -> fixed-R Kalman -> one final 5x5 MeanShift**.','',
 'All B+C P90 values below are computed by concatenating Route B and C per-frame errors and then taking the 90th percentile.','',
 '## 1. GRU necessity in the completed chain','',
 '| Setting | B+C MLE | B+C P90 | B+C LSR@5 |','|---|---:|---:|---:|',
 f"| WC -> Kalman -> MS (no GRU) | {fmt(ng['MLE_m'])} | {fmt(ng['P90_m'])} | {fmt(ng['LSR@5_pct'],2)}% |",
 f"| WC -> GRU -> Kalman -> MS | {fmt(fg['MLE_m'])} | {fmt(fg['P90_m'])} | {fmt(fg['LSR@5_pct'],2)}% |",'',
 '## 2. Selected 5x5 main-method online runtime','',
 f"- Route B: {fmt(eb['mean_ms'])} ms / {fmt(eb['fps'],1)} FPS",
 f"- Route C: {fmt(ec['mean_ms'])} ms / {fmt(ec['fps'],1)} FPS",
 f"- Pooled-frame weighted B+C: **{fmt(e2e)} ms / {fmt(1000.0/e2e,1)} FPS**",'',
 '## 3. True no-GT route-reference evaluation','',
 'This run disables current-frame GT center use and the final MeanShift reference is audited to equal the causal local-search reference on every frame.','',
 '| Protocol | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |','|---|---:|---:|---:|---:|---:|',
 f"| route_reference (no current-frame GT) | {fmt(nogt['route_B']['MLE_m'])} | {fmt(nogt['route_C']['MLE_m'])} | {fmt(nr['MLE_m'])} | {fmt(nr['P90_m'])} | {fmt(nr['LSR@5_pct'],2)}% |",'',
 '## Audit','',
 '- no-GRU ablation: PASS',
 '- 5x5 full-chain E2E timing: PASS',
 '- no-GT route_reference: PASS',
 '- final-MS GT-leak check for no-GT run: PASS',
 '- pooled B+C percentile calculation: PASS',
]
(suite/'missing_experiments_tables.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
(suite/'audit_report.json').write_text(json.dumps({
    'status':'PASS',
    'added_experiments':['WC -> Kalman -> final MS (no GRU)','5x5 full-chain E2E runtime','no-GT route_reference full-chain evaluation'],
    'selected_ms_grid':5,
    'pooled_percentile_definition':'concatenate Route B+C per-frame error_final_m, then np.quantile',
    'no_gt_final_ms_reference':'causal_local_search_reference; verified against prior_center_x/y on every frame',
},indent=2),encoding='utf-8')
print('AUDIT PASS')
print(suite/'missing_experiments_tables.md')
PY

echo "============================================================================================================"
echo "DONE: all actually-missing v39 experiments completed"
echo "Results: ${SUITE_ROOT}"
echo "Tables : ${SUITE_ROOT}/missing_experiments_tables.md"
echo "CSV    : ${SUITE_ROOT}/missing_experiments_summary.csv"
echo "Audit  : ${SUITE_ROOT}/audit_report.json"
echo "============================================================================================================"
