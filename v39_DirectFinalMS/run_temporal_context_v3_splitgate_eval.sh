#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BASE_SRC="${ROOT}/base_src"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
GPU="${GPU:-0}"
BACKBONE="mobilenet_v3_small"
JITTER_M="${JITTER_M:-8}"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
TS="$(date +%Y%m%d_%H%M%S)"
SUITE_ROOT="${SPLITGATE_DIR:-${ROOT}/temporal_context_v3_splitgate_eval_${TS}}"
FEATURE_CACHE="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache}"

fail(){ echo "ERROR: $*" >&2; exit 2; }

# Reuse the newest successfully trained visual-position/no-innovation V3 model.
if [[ -z "${SOURCE_SUITE:-}" ]]; then
  SOURCE_SUITE="$({ ls -dt "${ROOT}"/temporal_context_v3_noinnovation_* 2>/dev/null || true; } | while read -r d; do
    [[ -s "$d/train_3frame/checkpoints/$CKPT_NAME" ]] || continue
    [[ -s "$d/train_3frame/train.log" ]] || continue
    [[ -s "$d/best_params.json" ]] || continue
    echo "$d"; break
  done)"
fi
[[ -n "${SOURCE_SUITE:-}" && -d "${SOURCE_SUITE}" ]] || fail "set SOURCE_SUITE to the completed temporal_context_v3_noinnovation_* suite"
SOURCE_SUITE="$(cd "${SOURCE_SUITE}" && pwd)"

CKPT3="${SOURCE_SUITE}/train_3frame/checkpoints/${CKPT_NAME}"
TRAIN_LOG="${SOURCE_SUITE}/train_3frame/train.log"
BEST_PARAMS_SOURCE="${SOURCE_SUITE}/best_params.json"
VISUAL_LINK="${SOURCE_SUITE}/train_3frame/checkpoints/visual_retrieval_A_only.pt"
[[ -s "${CKPT3}" ]] || fail "missing temporal checkpoint: ${CKPT3}"
[[ -s "${TRAIN_LOG}" ]] || fail "missing training log: ${TRAIN_LOG}"
[[ -s "${BEST_PARAMS_SOURCE}" ]] || fail "missing source best_params.json"
[[ -e "${VISUAL_LINK}" ]] || fail "missing visual checkpoint link: ${VISUAL_LINK}"
VISUAL_CKPT="$(readlink -f "${VISUAL_LINK}")"
[[ -s "${VISUAL_CKPT}" ]] || fail "resolved visual checkpoint missing: ${VISUAL_CKPT}"

SOURCE_ARCH="$(python3 - "${CKPT3}" <<'PY'
import sys, torch
p=torch.load(sys.argv[1], map_location='cpu')
print(p.get('architecture',''))
PY
)"
[[ -n "${SOURCE_ARCH}" ]] || fail "checkpoint architecture metadata empty"

read VAL_START VAL_END < <(python3 - "${TRAIN_LOG}" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
m=re.search(r'temporal split train=\[\d+,\d+\) val=\[(\d+),(\d+)\)', text)
if not m: raise SystemExit('cannot recover Route-A validation split')
print(m.group(1),m.group(2))
PY
)

read MS_BW MS_KFW MS_REFW < <(python3 - "${BEST_PARAMS_SOURCE}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
print(float(d.get('ms_bandwidth_m',7.0)), float(d.get('ms_kf_prior_weight',1.5)), float(d.get('ms_reference_prior_weight',2.5)))
PY
)

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -s "${BASE_SRC}/${f}" ]] || fail "missing ${BASE_SRC}/${f}"
done
for p in patch_direct_finalms.py patch_front_softms.py patch_temporal_context_v3.py patch_gru_gate_eval.py patch_gru_dualgate_eval.py; do
  [[ -s "${ROOT}/${p}" ]] || fail "missing ${ROOT}/${p}"
done
for route in route_A route_B route_C; do
  [[ -s "${DATA_ROOT}/routes/${route}/frames.csv" ]] || fail "missing ${DATA_ROOT}/routes/${route}/frames.csv"
done

mkdir -p "${SUITE_ROOT}" "${FEATURE_CACHE}"
RUNTIME="${SUITE_ROOT}/runtime"
mkdir -p "${RUNTIME}"
cp -a "${BASE_SRC}/." "${RUNTIME}/"
python3 "${ROOT}/patch_direct_finalms.py" "${RUNTIME}/robust_tracker.py"
python3 "${ROOT}/patch_front_softms.py" "${RUNTIME}/robust_tracker.py"
python3 "${ROOT}/patch_temporal_context_v3.py" "${RUNTIME}"
python3 "${ROOT}/patch_gru_gate_eval.py" "${RUNTIME}/robust_tracker.py"
python3 "${ROOT}/patch_gru_dualgate_eval.py" "${RUNTIME}/robust_tracker.py"
python3 -m py_compile "${RUNTIME}/config.py" "${RUNTIME}/visual_model.py" "${RUNTIME}/robust_tracker.py"

# Strong architecture audit.
grep -q 'GRUCell(feature_dim \* 6' "${RUNTIME}/visual_model.py" || fail "GRU 6-branch audit failed"
grep -q 'self.sat_projection(sat_context)' "${RUNTIME}/visual_model.py" || fail "SAT context missing"
grep -q 'self.visual_position_projection(visual_position)' "${RUNTIME}/visual_model.py" || fail "Forward18 SoftMS visual position missing"
grep -q 'UAVSAT_GRU_CORRECTION_GAIN' "${RUNTIME}/robust_tracker.py" || fail "split correction gate missing"
grep -q 'UAVSAT_GRU_MOTION_GAIN' "${RUNTIME}/robust_tracker.py" || fail "split motion gate missing"
if grep -q 'innovation_projection\|visual_anchor_se - predicted_se' "${RUNTIME}/visual_model.py"; then
  fail "forbidden position innovation present"
fi

echo "============================================================================================================"
echo "SPLIT-GATE GRU EVAL / NO RETRAINING"
echo "Source       : ${SOURCE_SUITE}"
echo "Architecture : ${SOURCE_ARCH}"
echo "GRU inputs   : mean + delta + delta2 + SAT context + Forward18 SoftMS visual position + previous state"
echo "Forbidden    : visual_anchor_se - predicted_se"
echo "Tune data    : Route-A validation [${VAL_START},${VAL_END}) only"
echo "Final MS     : 6x6 main, bw=${MS_BW}, KF weight=${MS_KFW}, reference weight=${MS_REFW}"
echo "What changes : ONLY GRU correction/motion/variance residual gains"
echo "============================================================================================================"

export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

run_eval(){
  local name="$1" routes="$2" frames="$3" disable_gru="$4" kalman="$5" ms_enabled="$6" grid="$7"
  local cgain="$8" mgain="$9" vgain="${10}" measure="${11:-0}"
  local out="${SUITE_ROOT}/${name}"
  mkdir -p "${out}/checkpoints"
  ln -sfn "${VISUAL_CKPT}" "${out}/checkpoints/visual_retrieval_A_only.pt"
  if [[ "${disable_gru}" == "0" ]]; then
    ln -sfn "${CKPT3}" "${out}/checkpoints/${CKPT_NAME}"
  fi
  echo "[RUN] ${name} routes=${routes} frames=${frames} GRU=$((1-disable_gru)) K=${kalman} MS=${ms_enabled}/${grid} c=${cgain} m=${mgain} v=${vgain}"
  (
    cd "${RUNTIME}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    UAVSAT_DEVICE=cuda:0 \
    UAVSAT_OUTPUT_DIR="${out}" \
    UAVSAT_CHECKPOINT_DIR="${out}/checkpoints" \
    UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE}" \
    UAVSAT_DATA_ROOT="${DATA_ROOT}" \
    UAVSAT_BACKBONE="${BACKBONE}" \
    UAVSAT_ARCHITECTURE_NAME="${SOURCE_ARCH}" \
    UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
    UAVSAT_EXPERIMENT_ANCHOR=softms \
    UAVSAT_EXPERIMENT_FRAME_COUNT="${frames}" \
    UAVSAT_EXPERIMENT_MOTION=velocity \
    UAVSAT_EXPERIMENT_KALMAN="${kalman}" \
    UAVSAT_EXPERIMENT_DISABLE_GRU="${disable_gru}" \
    UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
    UAVSAT_GRU_CORRECTION_GAIN="${cgain}" \
    UAVSAT_GRU_MOTION_GAIN="${mgain}" \
    UAVSAT_GRU_VARIANCE_GAIN="${vgain}" \
    UAVSAT_EVAL_ROUTES="${routes}" \
    UAVSAT_MEASURE_LATENCY="${measure}" \
    UAVSAT_LATENCY_WARMUP=30 \
    MS_ENABLED="${ms_enabled}" \
    MS_GRID_SIZE="${grid}" \
    MS_BANDWIDTH_M="${MS_BW}" \
    MS_KF_PRIOR_WEIGHT="${MS_KFW}" \
    MS_REFERENCE_PRIOR_WEIGHT="${MS_REFW}" \
    MS_MEASURE_LATENCY=0 \
    python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m "${JITTER_M}" \
      2>&1 | tee "${out}/eval.log"
  )
}

# -----------------------------------------------------------------------------
# Phase 1: tune ONLY GRU residual strengths on Route-A validation.
# All candidates keep a positive correction and/or motion contribution, so the
# selected Full model is not a disguised no-GRU model.
# -----------------------------------------------------------------------------
CAL="${SUITE_ROOT}/routeA_splitgate_calibration.csv"
echo "name,correction_gain,motion_gain,variance_gain,val_mle,val_p90,val_lsr5" > "${CAL}"
CANDIDATES=(
  "c05_m15_v00 0.05 0.15 0.00"
  "c05_m25_v00 0.05 0.25 0.00"
  "c05_m40_v00 0.05 0.40 0.00"
  "c10_m15_v00 0.10 0.15 0.00"
  "c10_m25_v00 0.10 0.25 0.00"
  "c10_m40_v00 0.10 0.40 0.00"
  "c15_m15_v00 0.15 0.15 0.00"
  "c15_m25_v00 0.15 0.25 0.00"
  "c15_m40_v00 0.15 0.40 0.00"
  "c20_m25_v00 0.20 0.25 0.00"
  "c20_m40_v00 0.20 0.40 0.00"
  "c10_m25_v25 0.10 0.25 0.25"
  "c15_m25_v25 0.15 0.25 0.25"
  "c15_m40_v25 0.15 0.40 0.25"
  "c20_m40_v25 0.20 0.40 0.25"
)

for spec in "${CANDIDATES[@]}"; do
  read -r name cg mg vg <<<"${spec}"
  run_eval "cal_${name}" route_A 3 0 fixed 1 6 "${cg}" "${mg}" "${vg}" 0
  python3 - "${SUITE_ROOT}/cal_${name}" "${VAL_START}" "${VAL_END}" "${name}" "${cg}" "${mg}" "${vg}" >> "${CAL}" <<'PY'
import csv,glob,sys,numpy as np
out,start,end,name,cg,mg,vg=sys.argv[1:]
start,end=int(start),int(end)
files=glob.glob(out+'/route_A_*_frames.csv')
if len(files)!=1: raise SystemExit(f'expected one Route-A csv in {out}; got {len(files)}')
with open(files[0],newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
e=np.asarray([float(r['error_final_m']) for r in rows],float)[start:end]
print(','.join([name,cg,mg,vg,f'{e.mean():.9f}',f'{np.quantile(e,.90):.9f}',f'{(e<=5).mean()*100:.9f}']))
PY
done

BEST="$(python3 - "${CAL}" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1],encoding='utf-8')))
# Primary target MLE, then P90, then higher LSR@5.
r.sort(key=lambda x:(float(x['val_mle']),float(x['val_p90']),-float(x['val_lsr5'])))
x=r[0]
print(x['name'],x['correction_gain'],x['motion_gain'],x['variance_gain'],x['val_mle'],x['val_p90'],x['val_lsr5'])
PY
)"
read BEST_NAME BEST_CG BEST_MG BEST_VG BEST_MLE BEST_P90 BEST_LSR5 <<<"${BEST}"

cat > "${SUITE_ROOT}/best_params.json" <<EOF
{
  "selection_data": "Route-A validation only",
  "source_suite": "${SOURCE_SUITE}",
  "source_architecture": "${SOURCE_ARCH}",
  "gru_inputs": ["temporal_mean", "delta", "delta2", "satellite_context", "forward18_softms_visual_position", "previous_state"],
  "position_innovation_input": false,
  "gru_correction_gain": ${BEST_CG},
  "gru_motion_gain": ${BEST_MG},
  "gru_variance_gain": ${BEST_VG},
  "final_ms_grid": 6,
  "ms_bandwidth_m": ${MS_BW},
  "ms_kf_prior_weight": ${MS_KFW},
  "ms_reference_prior_weight": ${MS_REFW},
  "routeA_val_mle_m": ${BEST_MLE},
  "routeA_val_p90_m": ${BEST_P90},
  "routeA_val_lsr5_pct": ${BEST_LSR5}
}
EOF

echo "BEST Route-A split gate: ${BEST_NAME} correction=${BEST_CG} motion=${BEST_MG} variance=${BEST_VG} MLE=${BEST_MLE} P90=${BEST_P90}"

# -----------------------------------------------------------------------------
# Phase 2: lock Route-A-selected gains and evaluate B/C.
# -----------------------------------------------------------------------------
run_eval full_model      route_B,route_C 3 0 fixed 1 6 "${BEST_CG}" "${BEST_MG}" "${BEST_VG}" 1
run_eval abl_no_gru      route_B,route_C 3 1 fixed 1 6 "${BEST_CG}" "${BEST_MG}" "${BEST_VG}" 0
run_eval abl_no_kalman   route_B,route_C 3 0 none  1 6 "${BEST_CG}" "${BEST_MG}" "${BEST_VG}" 0
run_eval abl_no_final_ms route_B,route_C 3 0 fixed 0 6 "${BEST_CG}" "${BEST_MG}" "${BEST_VG}" 0

for f in 1 2 3; do
  run_eval "temporal_${f}frame" route_B,route_C "${f}" 0 fixed 1 6 "${BEST_CG}" "${BEST_MG}" "${BEST_VG}" 0
done

for g in 4 5 6 7 8; do
  run_eval "ms_window_${g}x${g}" route_B,route_C 3 0 fixed 1 "${g}" "${BEST_CG}" "${BEST_MG}" "${BEST_VG}" 0
done

python3 - "${SUITE_ROOT}" <<'PY'
import csv,glob,json,sys
from pathlib import Path
import numpy as np
suite=Path(sys.argv[1])

def errors(name,route):
    fs=glob.glob(str(suite/name/f'{route}_*_frames.csv'))
    if len(fs)!=1: raise SystemExit(f'{name}/{route}: expected one CSV, got {len(fs)}')
    with open(fs[0],newline='',encoding='utf-8') as f:
        return np.asarray([float(r['error_final_m']) for r in csv.DictReader(f)],float)

def met(a):
    return {'MLE':float(a.mean()),'P90':float(np.quantile(a,.90)),'LSR5':float((a<=5).mean()*100)}

component=['full_model','abl_no_gru','abl_no_kalman','abl_no_final_ms']
temporal=['temporal_1frame','temporal_2frame','temporal_3frame']
windows=[f'ms_window_{g}x{g}' for g in range(4,9)]
rows={}
for n in component+temporal+windows:
    b,c=errors(n,'route_B'),errors(n,'route_C')
    rows[n]={'B':met(b),'C':met(c),'BC':met(np.concatenate([b,c]))}
(suite/'experiment_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

lines=[
'# Temporal-context V3 split-gate evaluation','',
'GRU input: temporal mean + delta + delta2 + SAT context + direct Forward18 SoftMS visual position + previous state.','No position-innovation feature is used. Split GRU gains are selected on Route-A validation only and locked before B/C evaluation.','',
'## Component ablation','',
'| Setting | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |',
'|---|---:|---:|---:|---:|---:|']
labels={'full_model':'Full','abl_no_gru':'w/o GRU','abl_no_kalman':'w/o Kalman','abl_no_final_ms':'w/o Final MS'}
for n in component:
    r=rows[n]; lines.append(f"| {labels[n]} | {r['B']['MLE']:.3f} | {r['C']['MLE']:.3f} | {r['BC']['MLE']:.3f} | {r['BC']['P90']:.3f} | {r['BC']['LSR5']:.2f}% |")
lines += ['', '## Temporal context ablation','',
'All rows reuse the same 3-frame-trained checkpoint; 1/2-frame rows mask older context at inference.','',
'| Frames | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |','|---:|---:|---:|---:|---:|---:|']
for f,n in zip([1,2,3],temporal):
    r=rows[n]; lines.append(f"| {f} | {r['B']['MLE']:.3f} | {r['C']['MLE']:.3f} | {r['BC']['MLE']:.3f} | {r['BC']['P90']:.3f} | {r['BC']['LSR5']:.2f}% |")
lines += ['', '## Final MeanShift window sensitivity','',
'The main method uses 6x6; all Route-A-selected GRU/MS parameters are held fixed across windows.','',
'| Window | Candidates | B+C MLE | B+C P90 | B+C LSR@5 |','|---|---:|---:|---:|---:|']
for g,n in zip(range(4,9),windows):
    r=rows[n]['BC']; lines.append(f"| {g}x{g} | {g*g} | {r['MLE']:.3f} | {r['P90']:.3f} | {r['LSR5']:.2f}% |")
(suite/'paper_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
PY

echo "============================================================================================================"
echo "DONE: ${SUITE_ROOT}"
echo "Selected split gains: ${SUITE_ROOT}/best_params.json"
echo "Paper tables        : ${SUITE_ROOT}/paper_tables.md"
echo "============================================================================================================"
