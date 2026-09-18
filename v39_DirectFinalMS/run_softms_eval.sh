#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BASE_SRC="${ROOT}/base_src"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
BACKBONE="mobilenet_v3_small"
JITTER_M="${JITTER_M:-8}"
WARMUP="${V39_LATENCY_WARMUP:-30}"
TS="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="${UAVSAT_OUTPUT_DIR:-${ROOT}/softms_eval_${TS}}"
RUNTIME="${RUN_ROOT}/runtime"
OUT="${RUN_ROOT}/output"
FEATURE_CACHE="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache}"

VISUAL_CKPT="${V39_VISUAL_CKPT:-${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt}"
TEMPORAL_CKPT="${V39_TEMPORAL_CKPT:-${REPO_ROOT}/PreviousState-exp/output/mobilenetv3_prevstate/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt}"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

fail() { echo "ERROR: $*" >&2; exit 2; }
for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -s "${BASE_SRC}/${f}" ]] || fail "missing ${BASE_SRC}/${f}"
done
[[ -s "${ROOT}/patch_direct_finalms.py" ]] || fail "missing patch_direct_finalms.py"
[[ -s "${ROOT}/patch_front_softms.py" ]] || fail "missing patch_front_softms.py"
[[ -s "${VISUAL_CKPT}" ]] || fail "missing visual checkpoint ${VISUAL_CKPT}"
[[ -s "${TEMPORAL_CKPT}" ]] || fail "missing temporal checkpoint ${TEMPORAL_CKPT}"
for route in route_A route_B route_C; do
  [[ -s "${DATA_ROOT}/routes/${route}/frames.csv" ]] || fail "missing ${DATA_ROOT}/routes/${route}/frames.csv"
done

mkdir -p "${RUNTIME}" "${OUT}/checkpoints" "${FEATURE_CACHE}"
cp -a "${BASE_SRC}/." "${RUNTIME}/"
python3 "${ROOT}/patch_direct_finalms.py" "${RUNTIME}/robust_tracker.py"
python3 "${ROOT}/patch_front_softms.py" "${RUNTIME}/robust_tracker.py"
python3 -m py_compile "${RUNTIME}/robust_tracker.py"
ln -sfn "${VISUAL_CKPT}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"
ln -sfn "${TEMPORAL_CKPT}" "${OUT}/checkpoints/${CKPT_NAME}"

export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "================================================================================"
echo "V39 EVAL ONLY -- NO RETRAINING"
echo "Forward 3x6 -> SoftMS -> 3-frame GRU -> Constant Velocity -> fixed-R Kalman -> final 6x6 SoftMS (BW7)"
echo "Output: ${RUN_ROOT}"
echo "================================================================================"

(
  cd "${RUNTIME}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  UAVSAT_DEVICE="${UAVSAT_DEVICE:-cuda:0}" \
  UAVSAT_OUTPUT_DIR="${OUT}" \
  UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
  UAVSAT_FEATURE_CACHE_DIR="${FEATURE_CACHE}" \
  UAVSAT_DATA_ROOT="${DATA_ROOT}" \
  UAVSAT_BACKBONE="${BACKBONE}" \
  UAVSAT_ARCHITECTURE_NAME=V36_PreviousStateOnly_MobileNetV3_Forward3x6_PolynomialKalman \
  UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
  UAVSAT_EXPERIMENT_ANCHOR=softms \
  UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
  UAVSAT_EXPERIMENT_MOTION=velocity \
  UAVSAT_EXPERIMENT_KALMAN=fixed \
  UAVSAT_EXPERIMENT_DISABLE_GRU=0 \
  UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
  UAVSAT_MEASURE_LATENCY=1 \
  UAVSAT_LATENCY_WARMUP="${WARMUP}" \
  MS_ENABLED=1 \
  MS_GRID_SIZE=6 \
  MS_BANDWIDTH_M=7.0 \
  python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m "${JITTER_M}" 2>&1 | tee "${OUT}/eval.log"
)

SUMMARY="${OUT}/robust_tracker_summary.json"
[[ -s "${SUMMARY}" ]] || fail "missing ${SUMMARY}"
python3 - "${SUMMARY}" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8'))
print('\n================================================================================')
print('V39 FORWARD3x6 SOFTMS RESULT')
weighted_num=0.0; weighted_den=0
lat_num=0.0; lat_den=0
for route in ('route_B','route_C'):
    r=d[route]
    print(f"{route}: MLE={r['MLE_m']:.3f}m P90={r['P90_m']:.3f}m LSR@15={r['LSR@15_pct']:.2f}% decoder={r.get('VisualObservationDecoder')}")
    e=r.get('EndToEndTiming',{})
    if e:
        print(f"       latency mean={e['mean_ms']:.3f} ms median={e['median_ms']:.3f} ms P90={e['p90_ms']:.3f} ms FPS={e['fps']:.2f}")
        n=int(e['samples']); lat_num += float(e['mean_ms'])*n; lat_den += n
if lat_den:
    mean=lat_num/lat_den
    print(f"OVERALL_E2E_MEAN_MS = {mean:.3f} ms")
    print(f"OVERALL_E2E_FPS     = {1000.0/mean:.2f} FPS")
print('================================================================================')
PY
