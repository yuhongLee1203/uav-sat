#!/usr/bin/env bash
# Backbone comparison for the ROOT uav-sat v33 architecture only:
# image backbone -> v33 retrieval -> 3-frame GRU -> polynomial motion ->
# forward 3x6 visual measurement -> external RouteKalman -> final XY.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GPUS="${GPUS:-0 5 6}"
BACKBONES="${BACKBONES:-mobileclip2_s2 resnet18 mobilenet_v3_small vgg16}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
JITTER_M="${JITTER_M:-8}"
WARMUP="${WARMUP:-30}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-32}"
RESUME="${RESUME:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash backbone-exp/run_v33_backbone_experiments.sh

Optional environment variables:
  GPUS="0 5 6"
  BACKBONES="mobileclip2_s2 resnet18 mobilenet_v3_small vgg16"
  VISUAL_EPOCHS=30
  TEMPORAL_EPOCHS=60
  PATIENCE=10
  JITTER_M=8
  WARMUP=30
  CACHE_BATCH_SIZE=32         GPU-safe backbone/gallery encoding batch size
  RESUME=1                 resume existing non-baseline training checkpoints

All code, checkpoints, logs and results stay under:
  /yh/study/uav-sat/backbone-exp/

The MobileCLIP2-S2 row is the current v33 checkpoint. Other backbones receive
fresh Route-A-only visual and temporal training, then Route B/C evaluation.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
elif [[ -n "${1:-}" ]]; then
  usage >&2
  exit 2
fi

read -r -a gpu_array <<< "${GPUS}"
read -r -a backbone_array <<< "${BACKBONES}"
if (( ${#gpu_array[@]} == 0 || ${#backbone_array[@]} == 0 )); then
  echo "ERROR: GPUS and BACKBONES cannot be empty." >&2
  exit 2
fi

CURRENT_DIR="${ROOT}/outputs/controlled_gtprior_forward3x6_continuous_waypoint_rnn_polynomial_kalman_v33"
CURRENT_VISUAL="${CURRENT_DIR}/checkpoints/visual_retrieval_A_only.pt"
CURRENT_TEMPORAL="${CURRENT_DIR}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
if [[ ! -s "${CURRENT_VISUAL}" || ! -s "${CURRENT_TEMPORAL}" ]]; then
  echo "ERROR: current v33 checkpoints are missing under ${CURRENT_DIR}/checkpoints" >&2
  exit 2
fi

mkdir -p "${SCRIPT_DIR}/outputs" "${SCRIPT_DIR}/logs"

run_backbone() {
  local backbone="$1" gpu="$2"
  local output="${SCRIPT_DIR}/outputs/v33_${backbone}"
  local checkpoint="${output}/checkpoints"
  local log="${SCRIPT_DIR}/logs/v33_${backbone}.log"
  mkdir -p "${checkpoint}"

  (
    cd "${ROOT}"
    echo "[$(date -Is)] ROOT v33 backbone=${backbone} physical_gpu=${gpu}"
    echo "pipeline=image -> retrieval -> 3-frame GRU -> polynomial -> forward3x6 -> external Kalman"

    common_env=(
      "CUDA_VISIBLE_DEVICES=${gpu}"
      "UAVSAT_BACKBONE_BENCHMARK=1"
      "UAVSAT_BACKBONE=${backbone}"
      "UAVSAT_MEASURE_LATENCY=1"
      "UAVSAT_LATENCY_WARMUP=${WARMUP}"
      "UAVSAT_VISUAL_CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE}"
      "OMP_NUM_THREADS=2"
      "MKL_NUM_THREADS=2"
      "OPENBLAS_NUM_THREADS=2"
      "NUMEXPR_NUM_THREADS=2"
      "PYTHONUNBUFFERED=1"
    )

    if [[ "${backbone}" == "mobileclip2_s2" ]]; then
      # Measure the exact current model/checkpoints, while keeping benchmark
      # outputs isolated under backbone-exp/.
      cp -p "${CURRENT_VISUAL}" "${checkpoint}/visual_retrieval_A_only.pt"
      cp -p "${CURRENT_TEMPORAL}" \
        "${checkpoint}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
      env "${common_env[@]}" python3 -u robust_tracker.py \
        --mode eval --reuse-visual --jitter-m "${JITTER_M}"
    else
      resume_args=()
      if [[ "${RESUME}" == "1" ]]; then
        [[ -s "${checkpoint}/visual_retrieval_A_only.pt" ]] && resume_args+=(--resume-visual)
        [[ -s "${checkpoint}/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only_latest.pt" ]] && resume_args+=(--resume-temporal)
      fi
      env "${common_env[@]}" python3 -u robust_tracker.py \
        --mode train_eval \
        --visual-epochs "${VISUAL_EPOCHS}" \
        --temporal-epochs "${TEMPORAL_EPOCHS}" \
        --patience "${PATIENCE}" \
        --jitter-m "${JITTER_M}" \
        "${resume_args[@]}"
    fi

    test -s "${output}/robust_tracker_summary.json"
  ) 2>&1 | tee "${log}"
}

echo "ROOT architecture : ${ROOT}/config.py + data.py + visual_model.py + visual_localizer.py + robust_tracker.py"
echo "GPUs              : ${GPUS}"
echo "Backbones         : ${BACKBONES}"
echo "Results           : ${SCRIPT_DIR}/outputs"

# Run at most one job per GPU. Each batch is parallel; a remaining fourth
# backbone starts after the first three finish.
for ((start=0; start<${#backbone_array[@]}; start+=${#gpu_array[@]})); do
  pids=()
  names=()
  for ((slot=0; slot<${#gpu_array[@]} && start+slot<${#backbone_array[@]}; slot++)); do
    backbone="${backbone_array[$((start+slot))]}"
    gpu="${gpu_array[$slot]}"
    run_backbone "${backbone}" "${gpu}" &
    pids+=("$!")
    names+=("${backbone}")
  done
  for ((i=0; i<${#pids[@]}; i++)); do
    if ! wait "${pids[$i]}"; then
      echo "ERROR: ${names[$i]} failed; see ${SCRIPT_DIR}/logs/v33_${names[$i]}.log" >&2
      exit 1
    fi
  done
done

BACKBONES="${BACKBONES}" RESULT_ROOT="${SCRIPT_DIR}/outputs" python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["RESULT_ROOT"])
rows = []
for backbone in os.environ["BACKBONES"].split():
    path = root / f"v33_{backbone}" / "robust_tracker_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = {"backbone": backbone}
    for route, prefix in (("route_B", "B"), ("route_C", "C")):
        result = payload[route]
        timing = result["EndToEndTiming"]
        row[f"{prefix}_error_MLE_m"] = result["MLE_m"]
        row[f"{prefix}_jump_rate_pct"] = result["JumpRate_pct"]
        row[f"{prefix}_mean_ms"] = timing["mean_ms"]
        row[f"{prefix}_p95_ms"] = timing["p95_ms"]
        row[f"{prefix}_fps"] = timing["fps"]
    row["BC_mean_error_m"] = (row["B_error_MLE_m"] + row["C_error_MLE_m"]) / 2.0
    row["BC_mean_ms"] = (row["B_mean_ms"] + row["C_mean_ms"]) / 2.0
    row["BC_mean_jump_rate_pct"] = (row["B_jump_rate_pct"] + row["C_jump_rate_pct"]) / 2.0
    rows.append(row)

columns = list(rows[0])
csv_path = root / "v33_backbone_comparison.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

json_path = root / "v33_backbone_comparison.json"
json_path.write_text(json.dumps({
    "architecture": "root v33 ThreeFrameRouteStateGRU + polynomial + forward3x6 + external RouteKalman",
    "protocol": "Route A train; Route B/C controlled GT+smooth-jitter local-prior evaluation",
    "timing": "prepared UAV tensor through external Kalman final XY; disk/preprocessing/loading/gallery-build excluded",
    "rows": rows,
}, ensure_ascii=False, indent=2), encoding="utf-8")

print("\nbackbone                 B error    B ms  B jump%   C error    C ms  C jump%")
print("-" * 84)
for r in rows:
    print(f"{r['backbone']:<24} {r['B_error_MLE_m']:8.3f} {r['B_mean_ms']:7.2f} "
          f"{r['B_jump_rate_pct']:8.3f} {r['C_error_MLE_m']:9.3f} "
          f"{r['C_mean_ms']:7.2f} {r['C_jump_rate_pct']:8.3f}")
print(f"\nCSV : {csv_path}")
print(f"JSON: {json_path}")
PY
