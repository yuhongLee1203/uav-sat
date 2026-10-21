#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BASE_SRC="${ROOT}/base_src"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
GPU="${GPU:-0}"
BACKBONE="mobilenet_v3_small"
ARCH="V39_Forward18_TemporalContextV3_VisualPosition_NoInnovation_GRU_Kalman_Final6"
JITTER_M="${JITTER_M:-8}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-80}"
PATIENCE="${PATIENCE:-15}"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
TS="$(date +%Y%m%d_%H%M%S)"
SUITE_ROOT="${TCV3_DIR:-${ROOT}/temporal_context_v3_noinnovation_${TS}}"
FEATURE_CACHE="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache}"

fail(){ echo "ERROR: $*" >&2; exit 2; }

if [[ -z "${SOURCE_SUITE:-}" ]]; then
  SOURCE_SUITE="$(ls -dt "${ROOT}"/softms_forward18_final5x5_ablation_* 2>/dev/null | head -n 1 || true)"
fi
[[ -n "${SOURCE_SUITE:-}" && -d "${SOURCE_SUITE}" ]] || fail "set SOURCE_SUITE to previous Forward18 suite"
SOURCE_SUITE="$(cd "${SOURCE_SUITE}" && pwd)"
VISUAL_LINK="${SOURCE_SUITE}/temporal_3frame/checkpoints/visual_retrieval_A_only.pt"
[[ -e "${VISUAL_LINK}" ]] || fail "missing visual checkpoint link: ${VISUAL_LINK}"
VISUAL_CKPT="$(readlink -f "${VISUAL_LINK}")"
[[ -s "${VISUAL_CKPT}" ]] || fail "resolved visual checkpoint missing: ${VISUAL_CKPT}"

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -s "${BASE_SRC}/${f}" ]] || fail "missing ${BASE_SRC}/${f}"
done
for p in patch_direct_finalms.py patch_front_softms.py patch_temporal_context_v3.py patch_gru_gate_eval.py; do
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
python3 -m py_compile "${RUNTIME}/config.py" "${RUNTIME}/visual_model.py" "${RUNTIME}/robust_tracker.py"

# Architecture audit: six direct inputs and NO position innovation.
grep -Fq 'GRUCell(feature_dim * 6' "${RUNTIME}/visual_model.py" || fail "GRU 6-branch audit failed"
grep -Fq 'self.sat_projection(sat_context)' "${RUNTIME}/visual_model.py" || fail "SAT context missing from main GRU"
grep -Fq 'self.visual_position_projection(visual_position)' "${RUNTIME}/visual_model.py" || fail "Forward18 SoftMS visual position missing from main GRU"
grep -Fq 'visual_anchor_se[:, 0:1]' "${RUNTIME}/visual_model.py" || fail "direct visual position is not sourced from visual_anchor_se"
if grep -Fq 'visual_anchor_se - predicted_se' "${RUNTIME}/visual_model.py"; then
  fail "position innovation unexpectedly present in main GRU"
fi
if grep -Fq 'innovation_projection' "${RUNTIME}/visual_model.py"; then
  fail "innovation projection unexpectedly present in main GRU"
fi

echo "[AUDIT OK] GRU = mean + delta + delta2 + SAT context + Forward18 SoftMS visual position + previous_state; NO position innovation"

export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

TRAIN_OUT="${SUITE_ROOT}/train_3frame"
mkdir -p "${TRAIN_OUT}/checkpoints"
ln -sfn "${VISUAL_CKPT}" "${TRAIN_OUT}/checkpoints/visual_retrieval_A_only.pt"

cat > "${SUITE_ROOT}/run_manifest.json" <<EOF
{
  "architecture": "${ARCH}",
  "main_chain": "Forward18 SoftMS -> 3-frame GRU -> fixed-R Kalman -> Final 6x6 MeanShift",
  "gru_inputs": ["temporal_mean", "delta", "delta2", "satellite_context", "forward18_softms_visual_position", "previous_state"],
  "position_innovation_input": false,
  "visual_checkpoint_reused": true,
  "temporal_checkpoint_retrained_route_A": true,
  "hyperparameter_selection": "Route-A validation only",
  "B_C_used_for_selection": false
}
EOF

echo "============================================================================================================"
echo "TEMPORAL-CONTEXT V3 / NO POSITION INNOVATION"
echo "GRU input : temporal mean + delta + delta2 + satellite context + Forward18 SoftMS visual position + previous state"
echo "Feedback  : previous state only; NO visual-anchor minus predicted-position input"
echo "Train     : temporal GRU on Route A only; visual model reused"
echo "Main      : Forward18 -> GRU -> Kalman -> Final 6x6 MeanShift"
echo "Tune      : Route-A validation only; B/C locked evaluation"
echo "============================================================================================================"

# 1) Fresh 3-frame temporal training on Route A.
(
  cd "${RUNTIME}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  UAVSAT_DEVICE=cuda:0 \
  UAVSAT_OUTPUT_DIR="${TRAIN_OUT}" \
  UAVSAT_CHECKPOINT_DIR="${TRAIN_OUT}/checkpoints" \
  UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE}" \
  UAVSAT_DATA_ROOT="${DATA_ROOT}" \
  UAVSAT_BACKBONE="${BACKBONE}" \
  UAVSAT_ARCHITECTURE_NAME="${ARCH}" \
  UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
  UAVSAT_EXPERIMENT_ANCHOR=softms \
  UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
  UAVSAT_EXPERIMENT_MOTION=velocity \
  UAVSAT_EXPERIMENT_KALMAN=fixed \
  UAVSAT_EXPERIMENT_DISABLE_GRU=0 \
  UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
  MS_ENABLED=1 MS_GRID_SIZE=6 MS_BANDWIDTH_M=7.0 \
  python3 -u robust_tracker.py --mode train --reuse-visual --jitter-m "${JITTER_M}" \
    --temporal-epochs "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}" \
    2>&1 | tee "${TRAIN_OUT}/train.log"
)

CKPT3="${TRAIN_OUT}/checkpoints/${CKPT_NAME}"
[[ -s "${CKPT3}" && ! -L "${CKPT3}" ]] || fail "fresh temporal checkpoint missing: ${CKPT3}"

read VAL_START VAL_END < <(python3 - "${TRAIN_OUT}/train.log" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
m=re.search(r'temporal split train=\[\d+,\d+\) val=\[(\d+),(\d+)\)', text)
if not m: raise SystemExit('cannot recover Route-A validation split')
print(m.group(1),m.group(2))
PY
)
echo "Route-A validation range=[${VAL_START},${VAL_END})"

run_eval(){
  local name="$1" routes="$2" frames="$3" disable_gru="$4" kalman="$5" ms_enabled="$6" grid="$7"
  local gain="$8" bw="$9" kfw="${10}" refw="${11}" measure="${12:-0}"
  local out="${SUITE_ROOT}/${name}"
  mkdir -p "${out}/checkpoints"
  ln -sfn "${VISUAL_CKPT}" "${out}/checkpoints/visual_retrieval_A_only.pt"
  if [[ "${disable_gru}" == "0" ]]; then
    ln -sfn "${CKPT3}" "${out}/checkpoints/${CKPT_NAME}"
  fi
  echo "[RUN] ${name} routes=${routes} frames=${frames} GRU=$((1-disable_gru)) K=${kalman} MS=${ms_enabled}/${grid} gain=${gain} bw=${bw}"
  (
    cd "${RUNTIME}"
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
    UAVSAT_GRU_FUSION_GAIN="${gain}" \
    UAVSAT_EVAL_ROUTES="${routes}" \
    UAVSAT_MEASURE_LATENCY="${measure}" \
    UAVSAT_LATENCY_WARMUP=30 \
    MS_ENABLED="${ms_enabled}" \
    MS_GRID_SIZE="${grid}" \
    MS_BANDWIDTH_M="${bw}" \
    MS_KF_PRIOR_WEIGHT="${kfw}" \
    MS_REFERENCE_PRIOR_WEIGHT="${refw}" \
    MS_MEASURE_LATENCY=0 \
    python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m "${JITTER_M}" \
      2>&1 | tee "${out}/eval.log"
  )
}

# 2) Route-A-only calibration. Final grid stays FIXED at 6x6.
CALIBRATION_CSV="${SUITE_ROOT}/routeA_calibration.csv"
echo "name,gain,bw,kf_weight,ref_weight,val_mle,val_p90" > "${CALIBRATION_CSV}"
CANDIDATES=(
  "g100_b7 1.00 7.0 1.50 2.50"
  "g075_b7 0.75 7.0 1.50 2.50"
  "g050_b7 0.50 7.0 1.50 2.50"
  "g035_b7 0.35 7.0 1.50 2.50"
  "g025_b7 0.25 7.0 1.50 2.50"
  "g050_b6 0.50 6.0 1.50 2.50"
  "g035_b6 0.35 6.0 1.50 2.50"
  "g025_b6 0.25 6.0 1.50 2.50"
  "g050_b6_r3 0.50 6.0 1.50 3.00"
  "g035_b6_r3 0.35 6.0 1.50 3.00"
)

for spec in "${CANDIDATES[@]}"; do
  read -r name gain bw kfw refw <<<"${spec}"
  run_eval "cal_${name}" route_A 3 0 fixed 1 6 "${gain}" "${bw}" "${kfw}" "${refw}" 0
  python3 - "${SUITE_ROOT}/cal_${name}" "${VAL_START}" "${VAL_END}" "${name}" "${gain}" "${bw}" "${kfw}" "${refw}" >> "${CALIBRATION_CSV}" <<'PY'
import csv,glob,sys,numpy as np
out,start,end,name,gain,bw,kfw,refw=sys.argv[1:]
start,end=int(start),int(end)
files=glob.glob(out+'/route_A_*_frames.csv')
if len(files)!=1: raise SystemExit(f'expected one Route-A csv in {out}, got {files}')
with open(files[0],newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
err=np.asarray([float(r['error_final_m']) for r in rows],float)[start:end]
print(','.join([name,gain,bw,kfw,refw,f'{err.mean():.9f}',f'{np.quantile(err,.90):.9f}']))
PY
done

BEST_LINE="$(python3 - "${CALIBRATION_CSV}" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1],encoding='utf-8')))
rows.sort(key=lambda r:(float(r['val_mle']),float(r['val_p90'])))
r=rows[0]
print(r['name'],r['gain'],r['bw'],r['kf_weight'],r['ref_weight'],r['val_mle'],r['val_p90'])
PY
)"
read BEST_NAME BEST_GAIN BEST_BW BEST_KFW BEST_REFW BEST_VAL_MLE BEST_VAL_P90 <<<"${BEST_LINE}"
cat > "${SUITE_ROOT}/best_params.json" <<EOF
{
  "selection_data": "Route-A validation only",
  "validation_range": [${VAL_START}, ${VAL_END}],
  "final_ms_main_grid": 6,
  "gru_fusion_gain": ${BEST_GAIN},
  "ms_bandwidth_m": ${BEST_BW},
  "ms_kf_prior_weight": ${BEST_KFW},
  "ms_reference_prior_weight": ${BEST_REFW},
  "routeA_val_mle_m": ${BEST_VAL_MLE},
  "routeA_val_p90_m": ${BEST_VAL_P90},
  "position_innovation_input": false
}
EOF

echo "BEST Route-A validation: ${BEST_NAME} gain=${BEST_GAIN} bw=${BEST_BW} MLE=${BEST_VAL_MLE} P90=${BEST_VAL_P90}"

# 3) Freeze Route-A-selected parameters; evaluate B/C once.
run_eval full_model      route_B,route_C 3 0 fixed 1 6 "${BEST_GAIN}" "${BEST_BW}" "${BEST_KFW}" "${BEST_REFW}" 1
run_eval abl_no_gru      route_B,route_C 3 1 fixed 1 6 "${BEST_GAIN}" "${BEST_BW}" "${BEST_KFW}" "${BEST_REFW}" 0
run_eval abl_no_kalman   route_B,route_C 3 0 none  1 6 "${BEST_GAIN}" "${BEST_BW}" "${BEST_KFW}" "${BEST_REFW}" 0
run_eval abl_no_final_ms route_B,route_C 3 0 fixed 0 6 "${BEST_GAIN}" "${BEST_BW}" "${BEST_KFW}" "${BEST_REFW}" 0

# Temporal context ablation: SAME 3-frame-trained checkpoint, older context masked.
for f in 1 2 3; do
  run_eval "temporal_${f}frame" route_B,route_C "${f}" 0 fixed 1 6 "${BEST_GAIN}" "${BEST_BW}" "${BEST_KFW}" "${BEST_REFW}" 0
done

# Window sensitivity: same locked parameters for every window.
for g in 4 5 6 7 8; do
  run_eval "ms_window_${g}x${g}" route_B,route_C 3 0 fixed 1 "${g}" "${BEST_GAIN}" "${BEST_BW}" "${BEST_KFW}" "${BEST_REFW}" 0
done

python3 - "${SUITE_ROOT}" <<'PY'
import csv,glob,json,sys
from pathlib import Path
import numpy as np
suite=Path(sys.argv[1])

def errs(name,route):
    files=glob.glob(str(suite/name/f'{route}_*_frames.csv'))
    if len(files)!=1: raise SystemExit(f'{name}/{route}: expected one CSV, got {len(files)}')
    with open(files[0],newline='',encoding='utf-8') as f:
        return np.asarray([float(r['error_final_m']) for r in csv.DictReader(f)],float)

def met(a):
    return {'MLE':float(a.mean()),'P90':float(np.quantile(a,.90)),'LSR5':float((a<=5).mean()*100)}

component=['full_model','abl_no_gru','abl_no_kalman','abl_no_final_ms']
temporal=['temporal_1frame','temporal_2frame','temporal_3frame']
windows=[f'ms_window_{g}x{g}' for g in range(4,9)]
rows={}
for name in component+temporal+windows:
    b,c=errs(name,'route_B'),errs(name,'route_C')
    rows[name]={'B':met(b),'C':met(c),'BC':met(np.concatenate([b,c]))}
(suite/'experiment_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

lines=[
'# Temporal-context v3 (no position innovation)', '',
'GRU input: temporal mean + delta + delta2 + satellite context + Forward18 SoftMS visual position + previous state.',
'No visual-anchor minus predicted-position feature is used as GRU input.',
'Hyperparameters are selected on Route-A validation only, then locked for Route-B/C.', '',
'## Component ablation','',
'| Setting | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |',
'|---|---:|---:|---:|---:|---:|']
labels={'full_model':'Full','abl_no_gru':'w/o GRU','abl_no_kalman':'w/o Kalman','abl_no_final_ms':'w/o Final MS'}
for n in component:
    r=rows[n]; lines.append(f"| {labels[n]} | {r['B']['MLE']:.3f} | {r['C']['MLE']:.3f} | {r['BC']['MLE']:.3f} | {r['BC']['P90']:.3f} | {r['BC']['LSR5']:.2f}% |")
lines += ['', '## Temporal context ablation','',
'All rows reuse the same 3-frame-trained checkpoint; 1/2-frame rows mask older temporal context at inference.','',
'| Frames | B MLE | C MLE | B+C MLE | B+C P90 | B+C LSR@5 |',
'|---:|---:|---:|---:|---:|---:|']
for f,n in zip([1,2,3],temporal):
    r=rows[n]; lines.append(f"| {f} | {r['B']['MLE']:.3f} | {r['C']['MLE']:.3f} | {r['BC']['MLE']:.3f} | {r['BC']['P90']:.3f} | {r['BC']['LSR5']:.2f}% |")
lines += ['', '## Final MeanShift window sensitivity','',
'The main method uses 6x6. Route-A-selected parameters are held fixed for every window.','',
'| Window | Candidates | B+C MLE | B+C P90 | B+C LSR@5 |',
'|---|---:|---:|---:|---:|']
for g,n in zip(range(4,9),windows):
    r=rows[n]['BC']; lines.append(f"| {g}x{g} | {g*g} | {r['MLE']:.3f} | {r['P90']:.3f} | {r['LSR5']:.2f}% |")
(suite/'paper_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
PY

echo "============================================================================================================"
echo "DONE: ${SUITE_ROOT}"
echo "Selected params: ${SUITE_ROOT}/best_params.json"
echo "Paper tables   : ${SUITE_ROOT}/paper_tables.md"
echo "============================================================================================================"
