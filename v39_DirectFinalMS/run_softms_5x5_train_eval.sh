#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BASE_SRC="${ROOT}/base_src"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
BACKBONE="mobilenet_v3_small"
GPU="${GPU:-0}"
JITTER_M="${JITTER_M:-8}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
WARMUP="${V39_LATENCY_WARMUP:-30}"
TS="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="${UAVSAT_OUTPUT_DIR:-${ROOT}/softms_5x5_forward15_train_eval_${TS}}"
RUNTIME="${RUN_ROOT}/runtime"
OUT="${RUN_ROOT}/output"
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
for p in patch_direct_finalms.py patch_front_softms.py patch_forward5x5_15.py; do
  [[ -s "${ROOT}/${p}" ]] || fail "missing ${ROOT}/${p}"
done
[[ -s "${VISUAL_CKPT}" ]] || fail "missing visual checkpoint: ${VISUAL_CKPT}"
for route in route_A route_B route_C; do
  [[ -s "${DATA_ROOT}/routes/${route}/frames.csv" ]] || fail "missing ${DATA_ROOT}/routes/${route}/frames.csv"
done

mkdir -p "${RUNTIME}" "${OUT}/checkpoints" "${FEATURE_CACHE}"
cp -a "${BASE_SRC}/." "${RUNTIME}/"
python3 "${ROOT}/patch_direct_finalms.py" "${RUNTIME}/robust_tracker.py"
python3 "${ROOT}/patch_front_softms.py" "${RUNTIME}/robust_tracker.py"
python3 "${ROOT}/patch_forward5x5_15.py" "${RUNTIME}/robust_tracker.py" "${RUNTIME}/config.py"
python3 -m py_compile "${RUNTIME}/robust_tracker.py" "${RUNTIME}/config.py"

grep -q 'forward-15 local posterior' "${RUNTIME}/robust_tracker.py" || fail "5x5/15 runtime audit failed"
grep -q 'UAVSAT_ACQ_LOCAL_GRID_SIZE.*, "5"' "${RUNTIME}/config.py" || fail "5x5 config audit failed"
grep -q 'UAVSAT_FORWARD_SEARCH_COLS.*, "5"' "${RUNTIME}/config.py" || fail "forward 15 config audit failed"

ln -sfn "${VISUAL_CKPT}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"

export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cat > "${RUN_ROOT}/run_manifest.json" <<EOF
{
  "method": "5x5 local geometry -> heading-forward 15 -> front SoftMS -> 3-frame GRU -> constant velocity -> fixed-R Kalman -> final 5x5 SoftMS",
  "visual_checkpoint_reused": true,
  "temporal_checkpoint_retrained": true,
  "front_base_grid": 5,
  "front_scored_candidates": 15,
  "front_rows": 3,
  "front_cols": 5,
  "final_ms_grid": 5,
  "final_ms_bandwidth_m": 7.0,
  "temporal_epochs": ${TEMPORAL_EPOCHS},
  "patience": ${PATIENCE},
  "jitter_m": ${JITTER_M}
}
EOF

echo "================================================================================"
echo "V39 5x5 / FORWARD-15 SOFTMS -- FRESH TEMPORAL TRAINING + EVAL"
echo "Pipeline: 5x5 -> forward 15 -> SoftMS -> 3-frame GRU -> CV -> fixed-R Kalman -> final 5x5 SoftMS BW7"
echo "Visual checkpoint : reused"
echo "Temporal checkpoint: freshly trained on Route A"
echo "Output            : ${RUN_ROOT}"
echo "================================================================================"

(
  cd "${RUNTIME}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  UAVSAT_DEVICE=cuda:0 \
  UAVSAT_OUTPUT_DIR="${OUT}" \
  UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
  UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE}" \
  UAVSAT_DATA_ROOT="${DATA_ROOT}" \
  UAVSAT_BACKBONE="${BACKBONE}" \
  UAVSAT_ARCHITECTURE_NAME=V39_MobileNetV3_Forward15of5x5_SoftMS_GRU_CV_Kalman_Final5x5MS \
  UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
  UAVSAT_EXPERIMENT_ANCHOR=softms \
  UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
  UAVSAT_EXPERIMENT_MOTION=velocity \
  UAVSAT_EXPERIMENT_KALMAN=fixed \
  UAVSAT_EXPERIMENT_DISABLE_GRU=0 \
  UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
  UAVSAT_ACQ_LOCAL_GRID_SIZE=5 \
  UAVSAT_FORWARD_SEARCH_ROWS=3 \
  UAVSAT_FORWARD_SEARCH_COLS=5 \
  UAVSAT_MEASURE_LATENCY=1 \
  UAVSAT_LATENCY_WARMUP="${WARMUP}" \
  MS_ENABLED=1 \
  MS_GRID_SIZE=5 \
  MS_BANDWIDTH_M=7.0 \
  python3 -u robust_tracker.py \
    --mode train_eval \
    --reuse-visual \
    --jitter-m "${JITTER_M}" \
    --temporal-epochs "${TEMPORAL_EPOCHS}" \
    --patience "${PATIENCE}" \
    2>&1 | tee "${OUT}/train_eval.log"
)

SUMMARY="${OUT}/robust_tracker_summary.json"
[[ -s "${SUMMARY}" ]] || fail "missing ${SUMMARY}"
[[ -s "${OUT}/checkpoints/${CKPT_NAME}" ]] || fail "fresh temporal checkpoint missing"

python3 - "${SUMMARY}" "${RUN_ROOT}/selected_result.json" <<'PY'
import json, sys
from pathlib import Path
summary_path=Path(sys.argv[1]); result_path=Path(sys.argv[2])
d=json.loads(summary_path.read_text(encoding='utf-8'))
rows=[]
for route in ('route_B','route_C'):
    r=d[route]
    dec=str(r.get('VisualObservationDecoder',''))
    if dec != 'forward 15-of-5x5 soft mean shift':
        raise SystemExit(f'ERROR: {route} decoder={dec!r}, expected forward 15-of-5x5 soft mean shift')
    e=r.get('EndToEndTiming',{})
    rows.append((route,r,e))
lat_num=0.0; lat_den=0
for _,_,e in rows:
    if e:
        n=int(e.get('samples',0)); lat_num += float(e['mean_ms'])*n; lat_den += n
payload={
  'pipeline':'5x5 -> forward 15 -> front SoftMS -> 3-frame GRU -> constant velocity -> fixed-R Kalman -> final 5x5 SoftMS BW7',
  'front_grid':5,
  'front_candidates':15,
  'final_ms_grid':5,
  'final_ms_bandwidth_m':7.0,
  'temporal_retrained':True,
  'routes':{},
}
print('\n'+'='*96)
print('V39 5x5 FORWARD-15 + FINAL-5x5 RESULT')
for route,r,e in rows:
    payload['routes'][route]={
      'MLE_m':r.get('MLE_m'),'P90_m':r.get('P90_m'),'LSR@5_pct':r.get('LSR@5_pct'),
      'LSR@15_pct':r.get('LSR@15_pct'),'EndToEndTiming':e,
    }
    print(f"{route}: MLE={float(r['MLE_m']):.3f}m P90={float(r['P90_m']):.3f}m LSR@15={float(r['LSR@15_pct']):.2f}%")
    if e:
        print(f"       latency mean={float(e['mean_ms']):.3f} ms median={float(e['median_ms']):.3f} ms P90={float(e['p90_ms']):.3f} ms FPS={float(e['fps']):.2f}")
if lat_den:
    mean=lat_num/lat_den
    payload['overall_e2e_mean_ms']=mean
    payload['overall_e2e_fps']=1000.0/mean
    print(f'OVERALL_E2E_MEAN_MS = {mean:.3f} ms')
    print(f'OVERALL_E2E_FPS     = {1000.0/mean:.2f} FPS')
result_path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
print('Result JSON:', result_path)
print('='*96)
PY

echo "All raw frame CSVs, checkpoint, log and summaries are under: ${RUN_ROOT}"
