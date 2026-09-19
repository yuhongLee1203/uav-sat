#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BASE_SRC="${ROOT}/base_src"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
GPU="${GPU:-0}"
BACKBONE="mobilenet_v3_small"
JITTER_M="${JITTER_M:-8}"
DEFAULT_MS_BW="7.0"
TS="$(date +%Y%m%d_%H%M%S)"
SUITE_ROOT="${NOFINALGT_EVAL_DIR:-${ROOT}/forward18_no_finalgt_eval_${TS}}"
FEATURE_CACHE="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache}"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
ARCH_NAME="V39_MobileNetV3_Forward18of6x6_SoftMS_GRU_CV_Kalman_Final5x5MS"

fail(){ echo "ERROR: $*" >&2; exit 2; }

SOURCE_SUITE="${SOURCE_FORWARD18_SUITE:-}"
if [[ -z "${SOURCE_SUITE}" ]]; then
  SOURCE_SUITE="$(ls -dt "${ROOT}"/softms_forward18_final5x5_ablation_* 2>/dev/null | head -n 1 || true)"
fi
[[ -n "${SOURCE_SUITE}" && -d "${SOURCE_SUITE}" ]] || fail "cannot find prior Forward-18 ablation suite; set SOURCE_FORWARD18_SUITE=/path/to/softms_forward18_final5x5_ablation_*"

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -s "${BASE_SRC}/${f}" ]] || fail "missing ${BASE_SRC}/${f}"
done
for p in patch_direct_finalms.py patch_front_softms.py; do
  [[ -s "${ROOT}/${p}" ]] || fail "missing ${ROOT}/${p}"
done

if [[ -s "${REPO_ROOT}/forNX/weights/v39_directfinalms/checkpoints/visual_retrieval_A_only.pt" ]]; then
  VISUAL_CKPT="${V39_VISUAL_CKPT:-${REPO_ROOT}/forNX/weights/v39_directfinalms/checkpoints/visual_retrieval_A_only.pt}"
else
  VISUAL_CKPT="${V39_VISUAL_CKPT:-${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt}"
fi
[[ -s "${VISUAL_CKPT}" ]] || fail "missing visual checkpoint: ${VISUAL_CKPT}"

CKPT1="${SOURCE_SUITE}/temporal_1frame/checkpoints/${CKPT_NAME}"
CKPT2="${SOURCE_SUITE}/temporal_2frame/checkpoints/${CKPT_NAME}"
CKPT3="${SOURCE_SUITE}/temporal_3frame/checkpoints/${CKPT_NAME}"
for ck in "${CKPT1}" "${CKPT2}" "${CKPT3}"; do [[ -s "${ck}" ]] || fail "missing checkpoint ${ck}"; done

mkdir -p "${SUITE_ROOT}" "${FEATURE_CACHE}"
export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cat > "${SUITE_ROOT}/run_manifest.json" <<EOF
{
  "mode": "eval_only",
  "source_suite": "${SOURCE_SUITE}",
  "front": "6x6 -> heading-forward 18 -> Front SoftMS",
  "checkpoint_training": "reused existing 1/2/3-frame checkpoints; no retraining",
  "final_ms_default_grid": 5,
  "final_ms_bandwidth_m": 7.0,
  "final_ms_reference_prior_weight": 0.0,
  "important_scope": "only the direct current-frame GT/reference prior inside Final MS is disabled; the front controlled GT+jitter local prior remains active"
}
EOF

make_runtime(){
  local runtime="$1"
  mkdir -p "${runtime}"
  cp -a "${BASE_SRC}/." "${runtime}/"
  python3 "${ROOT}/patch_direct_finalms.py" "${runtime}/robust_tracker.py"
  python3 "${ROOT}/patch_front_softms.py" "${runtime}/robust_tracker.py"
  python3 -m py_compile "${runtime}/robust_tracker.py" "${runtime}/config.py"
}

run_eval(){
  local name="$1" frames="$2" disable_gru="$3" kalman="$4" ms_enabled="$5" grid="$6" ckpt="$7" measure_e2e="${8:-0}" measure_ms="${9:-0}"
  local out="${SUITE_ROOT}/${name}"
  local runtime="${SUITE_ROOT}/runtime_${name}"
  mkdir -p "${out}/checkpoints"
  make_runtime "${runtime}"
  ln -sfn "${VISUAL_CKPT}" "${out}/checkpoints/visual_retrieval_A_only.pt"
  if [[ "${disable_gru}" == "0" ]]; then
    [[ -s "${ckpt}" ]] || fail "missing temporal checkpoint for ${name}: ${ckpt}"
    ln -sfn "${ckpt}" "${out}/checkpoints/${CKPT_NAME}"
  fi

  echo "============================================================================================================"
  echo "[EVAL-ONLY] ${name} | frames=${frames} | front=6x6->18 | GRU=$((1-disable_gru)) | Kalman=${kalman} | FinalMS=${ms_enabled} grid=${grid} | final-GT-prior=OFF"
  echo "============================================================================================================"
  (
    cd "${runtime}"
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
    UAVSAT_LATENCY_WARMUP=30 \
    MS_ENABLED="${ms_enabled}" \
    MS_GRID_SIZE="${grid}" \
    MS_BANDWIDTH_M="${DEFAULT_MS_BW}" \
    MS_REFERENCE_PRIOR_WEIGHT=0.0 \
    MS_MEASURE_LATENCY="${measure_ms}" \
    MS_LATENCY_WARMUP=30 \
    python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m "${JITTER_M}" 2>&1 | tee "${out}/eval.log"
  )
}

# Reuse the already-trained temporal checkpoints: NO TRAINING in this script.
run_eval temporal_1frame 1 0 fixed 1 5 "${CKPT1}" 0 0
run_eval temporal_2frame 2 0 fixed 1 5 "${CKPT2}" 0 0
run_eval temporal_3frame 3 0 fixed 1 5 "${CKPT3}" 0 0

run_eval full_model      3 0 fixed 1 5 "${CKPT3}" 1 0
run_eval abl_no_gru      3 1 fixed 1 5 ""         0 0
run_eval abl_no_kalman   3 0 none  1 5 "${CKPT3}" 0 0
run_eval abl_no_final_ms 3 0 fixed 0 5 "${CKPT3}" 0 0

for g in 4 5 6 7 8; do
  run_eval "ms_window_${g}x${g}" 3 0 fixed 1 "${g}" "${CKPT3}" 0 1
done

python3 - "${SUITE_ROOT}" <<'PY'
import csv,json,sys
from pathlib import Path
import numpy as np
suite=Path(sys.argv[1])
names=['full_model','abl_no_gru','abl_no_kalman','abl_no_final_ms','temporal_1frame','temporal_2frame','temporal_3frame','ms_window_4x4','ms_window_5x5','ms_window_6x6','ms_window_7x7','ms_window_8x8']

def errs(out,route):
    fs=sorted(out.glob(f'{route}_*_frames.csv'))
    if len(fs)!=1: raise SystemExit(f'{out}: expected one {route} CSV, got {len(fs)}')
    with fs[0].open(newline='',encoding='utf-8') as f:
        return np.array([float(r['error_final_m']) for r in csv.DictReader(f)])

def m(a):
    return dict(MLE=float(a.mean()),P90=float(np.quantile(a,.9)),LSR5=float((a<=5).mean()*100))
rows=[]
for n in names:
    out=suite/n; d=json.loads((out/'robust_tracker_summary.json').read_text())
    eb,ec=errs(out,'route_B'),errs(out,'route_C'); ea=np.concatenate([eb,ec]); mm=m(ea)
    rb,rc=d['route_B'],d['route_C']
    e2b,e2c=rb.get('EndToEndTiming',{}),rc.get('EndToEndTiming',{})
    if e2b and e2c:
        n1,n2=int(e2b['samples']),int(e2c['samples']); e2=(float(e2b['mean_ms'])*n1+float(e2c['mean_ms'])*n2)/(n1+n2)
    else: e2=0.0
    warm=int(rb.get('MS_LatencyWarmupFrames',30)); n1=max(len(eb)-warm,1); n2=max(len(ec)-warm,1)
    ms=(float(rb.get('MS_LatencyMean_ms',0))*n1+float(rc.get('MS_LatencyMean_ms',0))*n2)/(n1+n2)
    rows.append(dict(Experiment=n,B_MLE=float(eb.mean()),C_MLE=float(ec.mean()),BC_MLE=mm['MLE'],BC_P90=mm['P90'],BC_LSR5=mm['LSR5'],E2E_ms=e2,E2E_FPS=(1000/e2 if e2 else 0),PureMS_ms=ms))

by={r['Experiment']:r for r in rows}
md=['# Forward18 eval-only, Final-MS direct GT/reference prior disabled','',
    '> Reused existing trained checkpoints. No temporal retraining was performed.','',
    '> Important: the front local search still uses the controlled GT+jitter prior; only the direct GT/reference term inside Final MS is disabled.','',
    '## Component ablation','','| Setting | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |','|---|---:|---:|---:|---:|---:|']
for n,l in [('full_model','Full'),('abl_no_gru','w/o GRU'),('abl_no_kalman','w/o Kalman'),('abl_no_final_ms','w/o Final MS')]:
    r=by[n]; md.append(f"| {l} | {r['B_MLE']:.3f} | {r['C_MLE']:.3f} | {r['BC_MLE']:.3f} | {r['BC_P90']:.3f} | {r['BC_LSR5']:.2f}% |")
md += ['','## Temporal frames','','| Frames | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |','|---:|---:|---:|---:|---:|---:|']
for f in (1,2,3):
    r=by[f'temporal_{f}frame']; md.append(f"| {f} | {r['B_MLE']:.3f} | {r['C_MLE']:.3f} | {r['BC_MLE']:.3f} | {r['BC_P90']:.3f} | {r['BC_LSR5']:.2f}% |")
md += ['','## Final MeanShift window','','| Window | B+C MLE | B+C P90 | B+C LSR@5 | Pure final-MS latency |','|---|---:|---:|---:|---:|']
for g in (4,5,6,7,8):
    r=by[f'ms_window_{g}x{g}']; md.append(f"| {g}x{g} | {r['BC_MLE']:.3f} | {r['BC_P90']:.3f} | {r['BC_LSR5']:.2f}% | {r['PureMS_ms']:.3f} ms |")
r=by['full_model']; md += ['','## Full-model E2E runtime','',f"- Mean: **{r['E2E_ms']:.3f} ms/frame**",f"- FPS: **{r['E2E_FPS']:.2f}**"]
(suite/'paper_tables.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
with (suite/'experiment_summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('\n'.join(md))
PY

echo "RESULT_DIR=${SUITE_ROOT}"
