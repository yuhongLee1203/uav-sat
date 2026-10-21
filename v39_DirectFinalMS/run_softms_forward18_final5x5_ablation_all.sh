#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BASE_SRC="${ROOT}/base_src"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
GPU="${GPU:-0}"
BACKBONE="mobilenet_v3_small"
JITTER_M="${JITTER_M:-8}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
DEFAULT_MS_GRID="5"
DEFAULT_MS_BW="7.0"
MS_LATENCY_WARMUP="${MS_LATENCY_WARMUP:-30}"
E2E_WARMUP="${E2E_WARMUP:-30}"
TS="$(date +%Y%m%d_%H%M%S)"
SUITE_ROOT="${SOFTMS_ABLATION_DIR:-${ROOT}/softms_forward18_final5x5_ablation_${TS}}"
FEATURE_CACHE="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache}"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
ARCH_NAME="V39_MobileNetV3_Forward18of6x6_SoftMS_GRU_CV_Kalman_Final5x5MS"

if [[ -s "${REPO_ROOT}/forNX/weights/v39_directfinalms/checkpoints/visual_retrieval_A_only.pt" ]]; then
  VISUAL_CKPT="${V39_VISUAL_CKPT:-${REPO_ROOT}/forNX/weights/v39_directfinalms/checkpoints/visual_retrieval_A_only.pt}"
else
  VISUAL_CKPT="${V39_VISUAL_CKPT:-${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt}"
fi

fail() { echo "ERROR: $*" >&2; exit 2; }
for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -s "${BASE_SRC}/${f}" ]] || fail "missing ${BASE_SRC}/${f}"
done
for p in patch_direct_finalms.py patch_front_softms.py; do
  [[ -s "${ROOT}/${p}" ]] || fail "missing ${ROOT}/${p}"
done
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
cat > "${SUITE_ROOT}/run_manifest.json" <<EOF_MANIFEST
{
  "method": "6x6 local geometry -> heading-forward 18 -> front SoftMS -> temporal GRU -> constant velocity -> fixed-R Kalman -> final 5x5 SoftMS",
  "git_sha": "${GIT_SHA}",
  "front_grid": 6,
  "front_candidates": 18,
  "front_rows": 3,
  "front_cols": 6,
  "final_ms_grid": 5,
  "final_ms_bandwidth_m": 7.0,
  "temporal_epochs": ${TEMPORAL_EPOCHS},
  "patience": ${PATIENCE},
  "jitter_m": ${JITTER_M}
}
EOF_MANIFEST

make_runtime() {
  local runtime="$1"
  mkdir -p "${runtime}"
  cp -a "${BASE_SRC}/." "${runtime}/"
  python3 "${ROOT}/patch_direct_finalms.py" "${runtime}/robust_tracker.py"
  python3 "${ROOT}/patch_front_softms.py" "${runtime}/robust_tracker.py"
  python3 -m py_compile "${runtime}/robust_tracker.py" "${runtime}/config.py"

  grep -q 'forward 3x6 soft mean shift' "${runtime}/robust_tracker.py" || fail "Forward-18 SoftMS decoder audit failed"
  grep -q '^ACQ_LOCAL_GRID_SIZE = 6' "${runtime}/config.py" || fail "expected ACQ_LOCAL_GRID_SIZE=6"
  grep -q '^FORWARD_SEARCH_ROWS = 3' "${runtime}/config.py" || fail "expected FORWARD_SEARCH_ROWS=3"
  grep -q '^FORWARD_SEARCH_COLS = 6' "${runtime}/config.py" || fail "expected FORWARD_SEARCH_COLS=6"
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

  echo "============================================================================================================"
  echo "[RUN] ${name} | frames=${frames} | front=6x6->forward18 | GRU=$((1-disable_gru)) | Kalman=${kalman} | FinalMS=${ms_enabled} grid=${grid}"
  echo "============================================================================================================"

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
    UAVSAT_ARCHITECTURE_NAME="${ARCH_NAME}" \
    UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
    UAVSAT_EXPERIMENT_ANCHOR=softms \
    UAVSAT_EXPERIMENT_FRAME_COUNT="${frames}" \
    UAVSAT_EXPERIMENT_MOTION=velocity \
    UAVSAT_EXPERIMENT_KALMAN="${kalman}" \
    UAVSAT_EXPERIMENT_DISABLE_GRU="${disable_gru}" \
    UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
    UAVSAT_ACQ_LOCAL_GRID_SIZE=6 \
    UAVSAT_FORWARD_SEARCH_ROWS=3 \
    UAVSAT_FORWARD_SEARCH_COLS=6 \
    UAVSAT_MEASURE_LATENCY="${measure_e2e}" \
    UAVSAT_LATENCY_WARMUP="${E2E_WARMUP}" \
    MS_ENABLED="${ms_enabled}" \
    MS_GRID_SIZE="${grid}" \
    MS_BANDWIDTH_M="${DEFAULT_MS_BW}" \
    MS_MEASURE_LATENCY="${measure_ms}" \
    MS_LATENCY_WARMUP="${MS_LATENCY_WARMUP}" \
    python3 -u robust_tracker.py "${args[@]}" 2>&1 | tee "${out}/${mode}.log"
  )

  [[ -s "${out}/robust_tracker_summary.json" ]] || fail "missing summary for ${name}"
  python3 - "${out}" "${name}" "${category}" "${frames}" "${disable_gru}" "${kalman}" "${ms_enabled}" "${grid}" <<'PY_META'
import json, sys
from pathlib import Path
out=Path(sys.argv[1]); name=sys.argv[2]; category=sys.argv[3]
frames=int(sys.argv[4]); disable=bool(int(sys.argv[5])); kalman=sys.argv[6]
ms=bool(int(sys.argv[7])); grid=int(sys.argv[8])
p=out/'robust_tracker_summary.json'
d=json.loads(p.read_text(encoding='utf-8'))
d.update({
    'experiment_tag':name,
    'experiment_category':category,
    'experiment_anchor':'softms',
    'experiment_frame_count':frames,
    'experiment_forward_only':True,
    'front_grid_size':6,
    'front_candidate_count':18,
    'experiment_disable_gru':disable,
    'experiment_kalman':kalman,
    'ms_enabled':ms,
    'ms_grid_size':grid,
    'checkpoint_retrained':category=='temporal_frames',
})
for route in ('route_B','route_C'):
    if route not in d:
        raise SystemExit(f'AUDIT FAILED: missing {route}')
    dec=str(d[route].get('VisualObservationDecoder',''))
    if dec != 'forward 3x6 soft mean shift':
        raise SystemExit(f'AUDIT FAILED: {route} decoder={dec!r}')
p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
PY_META
}

# Temporal-context study. Each frame-count variant is freshly trained on Route A.
run_cfg temporal_1frame temporal_frames train_eval 1 0 fixed 1 5 "" 0 0
run_cfg temporal_2frame temporal_frames train_eval 2 0 fixed 1 5 "" 0 0
run_cfg temporal_3frame temporal_frames train_eval 3 0 fixed 1 5 "" 0 0

CKPT1="${SUITE_ROOT}/temporal_1frame/checkpoints/${CKPT_NAME}"
CKPT2="${SUITE_ROOT}/temporal_2frame/checkpoints/${CKPT_NAME}"
CKPT3="${SUITE_ROOT}/temporal_3frame/checkpoints/${CKPT_NAME}"
for ck in "${CKPT1}" "${CKPT2}" "${CKPT3}"; do
  [[ -f "${ck}" && ! -L "${ck}" ]] || fail "fresh checkpoint missing: ${ck}"
done

# Component ablation: Forward-18 and Final 5x5 are fixed except the component being removed.
run_cfg full_model       module_ablation eval 3 0 fixed 1 5 "${CKPT3}" 0 1
run_cfg abl_no_gru       module_ablation eval 3 1 fixed 1 5 ""         0 0
run_cfg abl_no_kalman    module_ablation eval 3 0 none  1 5 "${CKPT3}" 0 0
run_cfg abl_no_final_ms  module_ablation eval 3 0 fixed 0 5 "${CKPT3}" 0 0

# Final MeanShift-window sensitivity. Front remains 6x6 -> forward 18 in every row.
for g in 4 5 6 7 8; do
  run_cfg "ms_window_${g}x${g}" ms_window eval 3 0 fixed 1 "${g}" "${CKPT3}" 1 0
done

python3 - "${SUITE_ROOT}" <<'PY_TABLES'
import csv, json, sys
from pathlib import Path
import numpy as np

suite=Path(sys.argv[1])
names=[
    'full_model','abl_no_gru','abl_no_kalman','abl_no_final_ms',
    'temporal_1frame','temporal_2frame','temporal_3frame',
    'ms_window_4x4','ms_window_5x5','ms_window_6x6','ms_window_7x7','ms_window_8x8'
]

def metric(a):
    a=np.asarray(a,float)
    return {
        'frames':int(len(a)), 'MLE_m':float(a.mean()), 'MedLE_m':float(np.median(a)),
        'P90_m':float(np.quantile(a,.90)), 'P95_m':float(np.quantile(a,.95)),
        'P99_m':float(np.quantile(a,.99)),
        'LSR5_pct':float((a<=5).mean()*100), 'LSR10_pct':float((a<=10).mean()*100),
        'LSR15_pct':float((a<=15).mean()*100), 'LSR20_pct':float((a<=20).mean()*100),
    }

def errors(out,route):
    files=sorted(out.glob(f'{route}_*_frames.csv'))
    if len(files)!=1:
        raise SystemExit(f'{out}/{route}: expected one frame CSV, got {len(files)}')
    with files[0].open(newline='',encoding='utf-8') as f:
        return np.asarray([float(r['error_final_m']) for r in csv.DictReader(f)])

rows=[]; payload={}
for name in names:
    out=suite/name
    d=json.loads((out/'robust_tracker_summary.json').read_text(encoding='utf-8'))
    if int(d.get('front_grid_size',0)) != 6 or int(d.get('front_candidate_count',0)) != 18:
        raise SystemExit(f'AUDIT FAILED: {name} is not front 6x6/18')
    eb,ec=errors(out,'route_B'),errors(out,'route_C')
    mb,mc,ma=metric(eb),metric(ec),metric(np.concatenate([eb,ec]))
    rb,rc=d['route_B'],d['route_C']

    warm=int(rb.get('MS_LatencyWarmupFrames',30))
    nb=max(len(eb)-warm,1); nc=max(len(ec)-warm,1)
    ms=(float(rb.get('MS_LatencyMean_ms',0))*nb + float(rc.get('MS_LatencyMean_ms',0))*nc)/(nb+nc)

    e2b,e2c=rb.get('EndToEndTiming',{}),rc.get('EndToEndTiming',{})
    if e2b and e2c:
        n1,n2=int(e2b['samples']),int(e2c['samples'])
        e2=(float(e2b['mean_ms'])*n1 + float(e2c['mean_ms'])*n2)/(n1+n2)
    else:
        e2=0.0

    row={
        'Experiment':name,
        'Category':d.get('experiment_category'),
        'Front':'6x6->18 SoftMS',
        'Frames':d.get('experiment_frame_count'),
        'GRU':'no' if d.get('experiment_disable_gru') else 'yes',
        'Kalman':d.get('experiment_kalman'),
        'FinalMS':'yes' if d.get('ms_enabled') else 'no',
        'MS_grid':d.get('ms_grid_size'),
        'B_MLE_m':mb['MLE_m'], 'C_MLE_m':mc['MLE_m'], 'BC_MLE_m':ma['MLE_m'],
        'BC_MedLE_m':ma['MedLE_m'], 'BC_P90_m':ma['P90_m'], 'BC_P95_m':ma['P95_m'],
        'BC_P99_m':ma['P99_m'], 'BC_LSR5_pct':ma['LSR5_pct'], 'BC_LSR10_pct':ma['LSR10_pct'],
        'BC_LSR15_pct':ma['LSR15_pct'], 'BC_LSR20_pct':ma['LSR20_pct'],
        'PureFinalMSLatency_ms':ms, 'PureFinalMS_FPS':1000/ms if ms>0 else 0.0,
        'E2E_mean_ms':e2, 'E2E_FPS':1000/e2 if e2>0 else 0.0,
    }
    rows.append(row)
    payload[name]={
        'route_B':mb,'route_C':mc,'combined':ma,
        'metadata':{k:d.get(k) for k in [
            'experiment_category','experiment_frame_count','experiment_disable_gru',
            'experiment_kalman','ms_enabled','ms_grid_size','front_grid_size','front_candidate_count'
        ]}
    }

with (suite/'experiment_summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
(suite/'experiment_summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')

by={r['Experiment']:r for r in rows}
def f3(x): return f'{float(x):.3f}'
def f2(x): return f'{float(x):.2f}'

md=[
    '# V39 Forward-18 + Final-5x5 SoftMS Ablation Results','',
    'Candidate pipeline: **6x6 local geometry -> heading-forward 18 -> front SoftMS -> 3-frame GRU -> Constant Velocity -> fixed-R Kalman -> final 5x5 SoftMS (BW=7m)**.','',
    'Forward 6x6->18 is fixed in all component and final-MS-window rows.','',
    '## Table 1. Component ablation','',
    '| Setting | GRU | Kalman | Final MS | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |',
    '|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|',
]
for n,label in [('full_model','Full'),('abl_no_gru','w/o GRU'),('abl_no_kalman','w/o Kalman'),('abl_no_final_ms','w/o Final MS')]:
    r=by[n]
    md.append(f"| {label} | {r['GRU']} | {r['Kalman']} | {r['FinalMS']} | {f3(r['B_MLE_m'])} | {f3(r['C_MLE_m'])} | {f3(r['BC_MLE_m'])} | {f3(r['BC_P90_m'])} | {f2(r['BC_LSR5_pct'])}% |")

md += ['', '## Table 2. Temporal frames','',
       '| Frames | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |',
       '|---:|---:|---:|---:|---:|---:|']
for n in ['temporal_1frame','temporal_2frame','temporal_3frame']:
    r=by[n]
    md.append(f"| {r['Frames']} | {f3(r['B_MLE_m'])} | {f3(r['C_MLE_m'])} | {f3(r['BC_MLE_m'])} | {f3(r['BC_P90_m'])} | {f2(r['BC_LSR5_pct'])}% |")

md += ['', '## Table 3. Final MeanShift window','',
       '| Window | Candidates | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 | Pure final-MS latency | FPS |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for g in [4,5,6,7,8]:
    r=by[f'ms_window_{g}x{g}']
    md.append(f"| {g}x{g} | {g*g} | {f3(r['B_MLE_m'])} | {f3(r['C_MLE_m'])} | {f3(r['BC_MLE_m'])} | {f3(r['BC_P90_m'])} | {f2(r['BC_LSR5_pct'])}% | {f3(r['PureFinalMSLatency_ms'])} ms | {f2(r['PureFinalMS_FPS'])} |")

full=by['full_model']
md += ['', '## Full-model E2E runtime','',
       f"- Mean: **{f3(full['E2E_mean_ms'])} ms/frame**",
       f"- FPS: **{f2(full['E2E_FPS'])}**", '']
(suite/'paper_tables.md').write_text('\n'.join(md),encoding='utf-8')

report={
    'status':'PASS',
    'front_grid':6,
    'front_candidates':18,
    'default_final_ms_grid':5,
    'experiments':names,
    'raw_data_saved':True,
}
(suite/'audit_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

print('\n'+'='*104)
print('FORWARD-18 + FINAL-5x5 ABLATION COMPLETE')
print('='*104)
print((suite/'paper_tables.md').read_text(encoding='utf-8'))
print('Results:',suite)
print('='*104)
PY_TABLES
