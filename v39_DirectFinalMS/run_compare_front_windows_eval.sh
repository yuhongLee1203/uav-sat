#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BASE_SRC="${ROOT}/base_src"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
GPU="${GPU:-0}"
BACKBONE="mobilenet_v3_small"
JITTER_M="${JITTER_M:-8}"
WARMUP="${V39_LATENCY_WARMUP:-30}"
TS="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="${COMPARE_OUTPUT_DIR:-${ROOT}/front_window_compare_${TS}}"
FEATURE_CACHE="${UAVSAT_FEATURE_CACHE_DIR_OVERRIDE:-${ROOT}/output/feature_cache}"

VISUAL_CKPT="${V39_VISUAL_CKPT:-${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt}"
TEMPORAL_CKPT="${V39_TEMPORAL_CKPT:-${REPO_ROOT}/PreviousState-exp/output/mobilenetv3_prevstate/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt}"
CKPT_NAME="controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

fail() { echo "ERROR: $*" >&2; exit 2; }
for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -s "${BASE_SRC}/${f}" ]] || fail "missing ${BASE_SRC}/${f}"
done
for f in patch_direct_finalms.py patch_front_softms.py patch_forward5x5_15.py; do
  [[ -s "${ROOT}/${f}" ]] || fail "missing ${ROOT}/${f}"
done
[[ -s "${VISUAL_CKPT}" ]] || fail "missing visual checkpoint: ${VISUAL_CKPT}"
[[ -s "${TEMPORAL_CKPT}" ]] || fail "missing temporal checkpoint: ${TEMPORAL_CKPT}"
for route in route_A route_B route_C; do
  [[ -s "${DATA_ROOT}/routes/${route}/frames.csv" ]] || fail "missing ${DATA_ROOT}/routes/${route}/frames.csv"
done

mkdir -p "${RUN_ROOT}" "${FEATURE_CACHE}"
export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

run_variant() {
  local name="$1"
  local front_grid="$2"
  local front_keep="$3"
  local runtime="${RUN_ROOT}/runtime_${name}"
  local out="${RUN_ROOT}/${name}"

  mkdir -p "${runtime}" "${out}/checkpoints"
  cp -a "${BASE_SRC}/." "${runtime}/"
  python3 "${ROOT}/patch_direct_finalms.py" "${runtime}/robust_tracker.py"
  python3 "${ROOT}/patch_front_softms.py" "${runtime}/robust_tracker.py"
  if [[ "${front_grid}" == "5" ]]; then
    python3 "${ROOT}/patch_forward5x5_15.py" "${runtime}/robust_tracker.py" "${runtime}/config.py"
  fi
  python3 -m py_compile "${runtime}/robust_tracker.py" "${runtime}/config.py"

  ln -sfn "${VISUAL_CKPT}" "${out}/checkpoints/visual_retrieval_A_only.pt"
  ln -sfn "${TEMPORAL_CKPT}" "${out}/checkpoints/${CKPT_NAME}"

  echo "================================================================================"
  echo "[COMPARE] ${name}"
  echo "Front geometry : ${front_grid}x${front_grid}"
  echo "Forward scored : ${front_keep}"
  echo "Front decoder  : SoftMS"
  echo "Final MS       : 5x5 / BW=7m"
  echo "Checkpoint     : SAME existing temporal checkpoint (NO RETRAINING)"
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
    UAVSAT_ARCHITECTURE_NAME="V39_SoftMS_FrontWindowCompare" \
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
    MS_GRID_SIZE=5 \
    MS_BANDWIDTH_M=7.0 \
    python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m "${JITTER_M}" 2>&1 | tee "${out}/eval.log"
  )

  [[ -s "${out}/robust_tracker_summary.json" ]] || fail "missing summary for ${name}"
}

# A: current forward 3x6 / 18, but final MS fixed to 5x5.
run_variant front3x6_final5x5 6 18

# B: 5x5 geometry -> heading-forward 3x5 / 15, final MS fixed to 5x5.
run_variant front3x5_final5x5 5 15

python3 - "${RUN_ROOT}" <<'PY'
import csv, json, sys
from pathlib import Path
root=Path(sys.argv[1])
variants=[
    ("front3x6_final5x5", "Front 3x6 (18) + Final 5x5"),
    ("front3x5_final5x5", "Front 3x5 (15) + Final 5x5"),
]
rows=[]
payload={}
for key,label in variants:
    p=root/key/'robust_tracker_summary.json'
    d=json.loads(p.read_text(encoding='utf-8'))
    row={'variant':key,'label':label}
    lat_num=lat_den=0.0
    for route in ('route_B','route_C'):
        r=d[route]
        row[f'{route}_MLE_m']=float(r['MLE_m'])
        row[f'{route}_P90_m']=float(r['P90_m'])
        row[f'{route}_LSR15_pct']=float(r['LSR@15_pct'])
        e=r.get('EndToEndTiming',{})
        if e:
            n=int(e['samples'])
            lat_num += float(e['mean_ms'])*n
            lat_den += n
    row['BC_MLE_mean_m']=(row['route_B_MLE_m']+row['route_C_MLE_m'])/2.0
    row['E2E_mean_ms']=lat_num/lat_den if lat_den else 0.0
    row['E2E_FPS']=1000.0/row['E2E_mean_ms'] if row['E2E_mean_ms']>0 else 0.0
    rows.append(row)
    payload[key]={'label':label,'summary':d,'comparison_row':row}

with (root/'front_window_comparison.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
(root/'front_window_comparison.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')

print('\n'+'='*100)
print('V39 FRONT-WINDOW EVAL-ONLY COMPARISON (same checkpoint, Final MS fixed at 5x5)')
print('='*100)
for r in rows:
    print(r['label'])
    print(f"  route_B MLE={r['route_B_MLE_m']:.3f} m  P90={r['route_B_P90_m']:.3f} m  LSR15={r['route_B_LSR15_pct']:.2f}%")
    print(f"  route_C MLE={r['route_C_MLE_m']:.3f} m  P90={r['route_C_P90_m']:.3f} m  LSR15={r['route_C_LSR15_pct']:.2f}%")
    print(f"  E2E={r['E2E_mean_ms']:.3f} ms  FPS={r['E2E_FPS']:.2f}")
print('-'*100)
print('CSV :', root/'front_window_comparison.csv')
print('JSON:', root/'front_window_comparison.json')
print('Raw per-frame CSVs/logs/summaries are retained inside each variant directory.')
print('='*100)
PY
