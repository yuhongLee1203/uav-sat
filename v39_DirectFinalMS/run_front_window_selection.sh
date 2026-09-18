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
WARMUP="${V39_LATENCY_WARMUP:-30}"
TOLERANCE_PCT="${FRONT_SELECTION_TOLERANCE_PCT:-0.5}"
TS="$(date +%Y%m%d_%H%M%S)"
ROOT_OUT="${FRONT_SELECTION_DIR:-${ROOT}/front_window_selection_${TS}}"
FEATURE_CACHE="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache}"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

if [[ -s "${REPO_ROOT}/forNX/weights/v39_directfinalms/checkpoints/visual_retrieval_A_only.pt" ]]; then
  VISUAL_CKPT="${V39_VISUAL_CKPT:-${REPO_ROOT}/forNX/weights/v39_directfinalms/checkpoints/visual_retrieval_A_only.pt}"
else
  VISUAL_CKPT="${V39_VISUAL_CKPT:-${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt}"
fi

fail() { echo "ERROR: $*" >&2; exit 2; }
for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do [[ -s "${BASE_SRC}/${f}" ]] || fail "missing ${BASE_SRC}/${f}"; done
for p in patch_direct_finalms.py patch_front_softms.py patch_forward5x5_15.py; do [[ -s "${ROOT}/${p}" ]] || fail "missing ${ROOT}/${p}"; done
[[ -s "${VISUAL_CKPT}" ]] || fail "missing visual checkpoint: ${VISUAL_CKPT}"
for route in route_A route_B route_C; do [[ -s "${DATA_ROOT}/routes/${route}/frames.csv" ]] || fail "missing ${DATA_ROOT}/routes/${route}/frames.csv"; done

mkdir -p "${ROOT_OUT}" "${FEATURE_CACHE}"
export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

run_variant() {
  local tag="$1" use5="$2"
  local run="${ROOT_OUT}/${tag}" runtime="${ROOT_OUT}/runtime_${tag}" out="${run}/output"
  mkdir -p "${runtime}" "${out}/checkpoints"
  cp -a "${BASE_SRC}/." "${runtime}/"
  python3 "${ROOT}/patch_direct_finalms.py" "${runtime}/robust_tracker.py"
  python3 "${ROOT}/patch_front_softms.py" "${runtime}/robust_tracker.py"
  if [[ "${use5}" == "1" ]]; then
    python3 "${ROOT}/patch_forward5x5_15.py" "${runtime}/robust_tracker.py" "${runtime}/config.py"
  fi
  python3 -m py_compile "${runtime}/robust_tracker.py" "${runtime}/config.py"
  ln -sfn "${VISUAL_CKPT}" "${out}/checkpoints/visual_retrieval_A_only.pt"

  local grid=6 rows=3 cols=6 arch="V39_Front6x6_Forward18_SoftMS_Final5x5"
  if [[ "${use5}" == "1" ]]; then grid=5; rows=3; cols=5; arch="V39_Front5x5_Forward15_SoftMS_Final5x5"; fi

  echo "================================================================================"
  echo "TRAIN/EVAL ${tag}: front ${grid}x${grid} -> forward $((rows*cols)); final MS fixed 5x5"
  echo "================================================================================"
  (
    cd "${runtime}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    UAVSAT_DEVICE=cuda:0 \
    UAVSAT_OUTPUT_DIR="${out}" \
    UAVSAT_CHECKPOINT_DIR="${out}/checkpoints" \
    UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE}" \
    UAVSAT_DATA_ROOT="${DATA_ROOT}" \
    UAVSAT_BACKBONE="${BACKBONE}" \
    UAVSAT_ARCHITECTURE_NAME="${arch}" \
    UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
    UAVSAT_EXPERIMENT_ANCHOR=softms \
    UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
    UAVSAT_EXPERIMENT_MOTION=velocity \
    UAVSAT_EXPERIMENT_KALMAN=fixed \
    UAVSAT_EXPERIMENT_DISABLE_GRU=0 \
    UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
    UAVSAT_ACQ_LOCAL_GRID_SIZE="${grid}" \
    UAVSAT_FORWARD_SEARCH_ROWS="${rows}" \
    UAVSAT_FORWARD_SEARCH_COLS="${cols}" \
    UAVSAT_MEASURE_LATENCY=1 \
    UAVSAT_LATENCY_WARMUP="${WARMUP}" \
    MS_ENABLED=1 \
    MS_GRID_SIZE=5 \
    MS_BANDWIDTH_M=7.0 \
    python3 -u robust_tracker.py --mode train_eval --reuse-visual --jitter-m "${JITTER_M}" --temporal-epochs "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}" 2>&1 | tee "${out}/train_eval.log"
  )
  [[ -s "${out}/checkpoints/${CKPT_NAME}" ]] || fail "${tag}: missing best temporal checkpoint"
  [[ -s "${out}/robust_tracker_summary.json" ]] || fail "${tag}: missing summary"
}

# Identical training settings. Only front geometry/scored-candidate budget differs.
# Final MS is fixed to 5x5 for both variants.
run_variant front6x6_forward18 0
run_variant front5x5_forward15 1

python3 - "${ROOT_OUT}" "${CKPT_NAME}" "${TOLERANCE_PCT}" <<'PY'
import json, sys
from pathlib import Path
import torch
root=Path(sys.argv[1]); ckpt_name=sys.argv[2]; tol_pct=float(sys.argv[3])
variants=['front6x6_forward18','front5x5_forward15']
rows=[]
for name in variants:
    ck=root/name/'output'/'checkpoints'/ckpt_name
    try:
        p=torch.load(ck,map_location='cpu',weights_only=False)
    except TypeError:
        p=torch.load(ck,map_location='cpu')
    val=p.get('validation',{})
    if 'mle' not in val: raise SystemExit(f'{name}: checkpoint lacks validation.mle')
    s=json.loads((root/name/'output'/'robust_tracker_summary.json').read_text())
    lat_num=lat_den=0.0
    test={}
    for route in ('route_B','route_C'):
        r=s[route]; e=r.get('EndToEndTiming',{})
        test[route]={'MLE_m':r.get('MLE_m'),'P90_m':r.get('P90_m'),'LSR@15_pct':r.get('LSR@15_pct')}
        if e:
            n=int(e['samples']); lat_num += float(e['mean_ms'])*n; lat_den += n
    mean_ms=lat_num/lat_den if lat_den else float('inf')
    rows.append({'name':name,'validation_mle_m':float(val['mle']),'validation_p90_m':float(val.get('p90',0.0)),'validation_score':float(p.get('best_score',float('nan'))),'e2e_mean_ms':mean_ms,'e2e_fps':1000.0/mean_ms if mean_ms>0 and mean_ms<float('inf') else 0.0,'test_report_only':test})

best_val=min(r['validation_mle_m'] for r in rows)
eligible=[r for r in rows if r['validation_mle_m'] <= best_val*(1.0+tol_pct/100.0)]
selected=min(eligible,key=lambda r:r['e2e_mean_ms']) if len(eligible)>1 else eligible[0]
reason=(f"validation MLE within {tol_pct:.3f}% of best; choose lower E2E latency" if len(eligible)>1 else f"only variant within {tol_pct:.3f}% of best validation MLE")
result={'selection_metric':'Route-A validation MLE','accuracy_tolerance_pct':tol_pct,'tie_breaker':'lower full online inference latency','selected':selected['name'],'reason':reason,'variants':rows,'B_C_metrics_used_for_selection':False}
(root/'front_window_selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
md=['# Front-window architecture selection','',f'Selection uses **Route-A validation MLE**. B/C metrics are report-only and are not used for selection.','',f'Rule: variants within **{tol_pct:.3f}%** of the best validation MLE are considered accuracy-equivalent; among them choose lower E2E latency.','', '| Variant | Val MLE | Val P90 | E2E ms | FPS |','|---|---:|---:|---:|---:|']
for r in rows: md.append(f"| {r['name']} | {r['validation_mle_m']:.3f} | {r['validation_p90_m']:.3f} | {r['e2e_mean_ms']:.3f} | {r['e2e_fps']:.2f} |")
md += ['',f"**Selected: {selected['name']}**",'',reason]
(root/'front_window_selection.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print('\n'+'='*96)
print('FRONT WINDOW SELECTION')
for r in rows: print(f"{r['name']}: val_MLE={r['validation_mle_m']:.3f} m | val_P90={r['validation_p90_m']:.3f} m | E2E={r['e2e_mean_ms']:.3f} ms | FPS={r['e2e_fps']:.2f}")
print('-'*96)
print('SELECTED =',selected['name'])
print('REASON   =',reason)
print('JSON     =',root/'front_window_selection.json')
print('TABLE    =',root/'front_window_selection.md')
print('='*96)
PY
