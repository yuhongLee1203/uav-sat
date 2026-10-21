#!/usr/bin/env bash
set -Eeuo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="${PKG_ROOT}/src"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${PKG_ROOT}/data}"
WEIGHT_ROOT="${V39_WEIGHT_ROOT:-${PKG_ROOT}/weights/v39_directfinalms/checkpoints}"
VISUAL_CKPT="${V39_VISUAL_CKPT:-${WEIGHT_ROOT}/visual_retrieval_A_only.pt}"
TEMPORAL_CKPT="${V39_TEMPORAL_CKPT:-${WEIGHT_ROOT}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt}"
WARMUP="${V39_LATENCY_WARMUP:-30}"
JITTER_M="${JITTER_M:-8}"
TS="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="${PKG_ROOT}/runs/v39_nx_5x5_forward15_${TS}"
RUNTIME="${RUN_ROOT}/runtime"
OUT="${RUN_ROOT}/output"
FEATURE_CACHE="${PKG_ROOT}/pretrained_cache/v39_directfinalms_mobilenet_v3_small"

fail() { echo "ERROR: $*" >&2; exit 2; }
for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -s "${SRC_ROOT}/${f}" ]] || fail "missing ${SRC_ROOT}/${f}"
done
for p in patch_direct_finalms.py patch_front_softms.py patch_forward5x5_15.py; do
  [[ -s "${PKG_ROOT}/${p}" ]] || fail "missing ${PKG_ROOT}/${p}"
done
[[ -s "${VISUAL_CKPT}" ]] || fail "missing visual checkpoint: ${VISUAL_CKPT}"
[[ -s "${TEMPORAL_CKPT}" ]] || fail "missing temporal checkpoint: ${TEMPORAL_CKPT}"
for route in route_B route_C; do
  [[ -s "${DATA_ROOT}/routes/${route}/frames.csv" ]] || fail "missing ${DATA_ROOT}/routes/${route}/frames.csv"
done

export TORCH_HOME="${TORCH_HOME:-${PKG_ROOT}/pretrained_cache/torch}"
export HF_HOME="${HF_HOME:-${PKG_ROOT}/pretrained_cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${RUNTIME}" "${OUT}/checkpoints" "${FEATURE_CACHE}"
cp -a "${SRC_ROOT}/." "${RUNTIME}/"
python3 "${PKG_ROOT}/patch_direct_finalms.py" "${RUNTIME}/robust_tracker.py"
python3 "${PKG_ROOT}/patch_front_softms.py" "${RUNTIME}/robust_tracker.py"
python3 "${PKG_ROOT}/patch_forward5x5_15.py" "${RUNTIME}/robust_tracker.py" "${RUNTIME}/config.py"
python3 -m py_compile "${RUNTIME}/robust_tracker.py" "${RUNTIME}/config.py"
grep -q 'forward-15 local posterior' "${RUNTIME}/robust_tracker.py" || fail "5x5/15 patch audit failed"

ln -s "${VISUAL_CKPT}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"
ln -s "${TEMPORAL_CKPT}" "${OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

BOARD_MODEL="unknown"
if [[ -r /sys/firmware/devicetree/base/model ]]; then BOARD_MODEL="$(tr -d '\000' </sys/firmware/devicetree/base/model || true)"; fi

echo "================================================================================"
echo "V39 NX 5x5 -> FORWARD15 -> SOFTMS -> GRU -> KALMAN -> FINAL 5x5 SOFTMS"
echo "Board   : ${BOARD_MODEL}"
echo "Warmup  : ${WARMUP} frames per route"
echo "Timing  : prepared UAV tensor -> full online inference -> final XY"
echo "Run dir : ${RUN_ROOT}"
echo "IMPORTANT: TEMPORAL_CKPT must be the freshly trained 5x5/15 checkpoint."
echo "================================================================================"

(
  cd "${RUNTIME}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  UAVSAT_DEVICE=cuda:0 \
  UAVSAT_OUTPUT_DIR="${OUT}" \
  UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
  UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE}" \
  UAVSAT_DATA_ROOT="${DATA_ROOT}" \
  UAVSAT_BACKBONE=mobilenet_v3_small \
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
  MS_MEASURE_LATENCY=1 \
  MS_LATENCY_WARMUP="${WARMUP}" \
  python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m "${JITTER_M}" 2>&1 | tee "${OUT}/eval.log"
)

SUMMARY="${OUT}/robust_tracker_summary.json"
[[ -s "${SUMMARY}" ]] || fail "missing ${SUMMARY}"
RESULT="${RUN_ROOT}/v39_nx_5x5_forward15_result.json"
python3 - "${SUMMARY}" "${RESULT}" "${BOARD_MODEL}" <<'PY'
import json, sys
from pathlib import Path
import torch
s,r,b=sys.argv[1:4]
d=json.loads(Path(s).read_text())
rows=[]
for route in ('route_B','route_C'):
    q=d[route]
    if q.get('VisualObservationDecoder')!='forward 15-of-5x5 soft mean shift':
        raise SystemExit(f"ERROR: {route} decoder={q.get('VisualObservationDecoder')}")
    e=q.get('EndToEndTiming',{})
    if not e: raise SystemExit(f'ERROR: missing timing for {route}')
    rows.append((route,q,e))
total=sum(int(e['samples']) for _,_,e in rows)
mean=sum(float(e['mean_ms'])*int(e['samples']) for _,_,e in rows)/total
payload={'method':'V39 5x5 forward15 SoftMS','pipeline':'MobileNetV3-Small -> 5x5 geometry -> forward 15 SoftMS -> 3-frame GRU -> fixed-R Kalman -> final 5x5 SoftMS BW7 -> XY','board_model':b,'cuda_device':torch.cuda.get_device_name(0),'mean_ms':mean,'fps':1000/mean,'routes':{n:{'MLE_m':q['MLE_m'],'P90_m':q['P90_m'],'LSR@15_pct':q['LSR@15_pct'],'timing':e} for n,q,e in rows}}
Path(r).write_text(json.dumps(payload,indent=2))
print('\n'+'='*88)
print('V39 NX 5x5 FORWARD15 RESULT')
print('Device:',payload['cuda_device'])
for n,q,e in rows:
    print(f"{n}: MLE={float(q['MLE_m']):.3f} m | mean={float(e['mean_ms']):.3f} ms | median={float(e['median_ms']):.3f} ms | P90={float(e['p90_ms']):.3f} ms | FPS={float(e['fps']):.2f}")
print('-'*88)
print(f'V39_FULL_PIPELINE_MEAN_MS = {mean:.3f} ms')
print(f'V39_FULL_PIPELINE_FPS     = {1000/mean:.2f} FPS')
print('Result JSON               =',r)
print('='*88)
PY
