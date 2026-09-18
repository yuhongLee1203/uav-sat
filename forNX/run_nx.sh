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
RUN_ROOT="${PKG_ROOT}/runs/v39_nx_softms_latency_${TS}"
RUNTIME="${RUN_ROOT}/runtime"
OUT="${RUN_ROOT}/output"
FEATURE_CACHE="${PKG_ROOT}/pretrained_cache/v39_directfinalms_mobilenet_v3_small"

fail() { echo "ERROR: $*" >&2; exit 2; }

command -v python3 >/dev/null 2>&1 || fail "python3 not found"
for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -s "${SRC_ROOT}/${f}" ]] || fail "missing V39 source: ${SRC_ROOT}/${f}"
done
[[ -s "${PKG_ROOT}/patch_direct_finalms.py" ]] || fail "missing patch_direct_finalms.py"
[[ -s "${PKG_ROOT}/patch_front_softms.py" ]] || fail "missing patch_front_softms.py"
[[ -s "${VISUAL_CKPT}" ]] || fail "missing V39 visual checkpoint: ${VISUAL_CKPT}"
[[ -s "${TEMPORAL_CKPT}" ]] || fail "missing V39 temporal checkpoint: ${TEMPORAL_CKPT}"
for route in route_B route_C; do
  [[ -s "${DATA_ROOT}/routes/${route}/frames.csv" ]] || fail "missing ${DATA_ROOT}/routes/${route}/frames.csv"
done
[[ -s "${DATA_ROOT}/satellite/sim_map_competition_roi_crop.png" ]] || fail "missing satellite image under ${DATA_ROOT}/satellite"
[[ -s "${DATA_ROOT}/satellite/sim_map_competition_roi_crop_worldfile_epsg3826.json" ]] || fail "missing satellite metadata under ${DATA_ROOT}/satellite"

export TORCH_HOME="${TORCH_HOME:-${PKG_ROOT}/pretrained_cache/torch}"
export HF_HOME="${HF_HOME:-${PKG_ROOT}/pretrained_cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

python3 - <<'PY'
import sys
try:
    import torch
    import torchvision
    import open_clip
except Exception as exc:
    raise SystemExit(f"DEPENDENCY CHECK FAILED: {exc}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA CHECK FAILED: torch.cuda.is_available() is False")
print("[NX-CHECK] Python       :", sys.version.split()[0])
print("[NX-CHECK] PyTorch      :", torch.__version__)
print("[NX-CHECK] Torchvision  :", torchvision.__version__)
print("[NX-CHECK] CUDA runtime :", torch.version.cuda)
print("[NX-CHECK] CUDA device  :", torch.cuda.get_device_name(0))
PY

mkdir -p "${RUNTIME}" "${OUT}/checkpoints" "${FEATURE_CACHE}"
cp -a "${SRC_ROOT}/." "${RUNTIME}/"
python3 "${PKG_ROOT}/patch_direct_finalms.py" "${RUNTIME}/robust_tracker.py"
python3 "${PKG_ROOT}/patch_front_softms.py" "${RUNTIME}/robust_tracker.py"
python3 -m py_compile "${RUNTIME}/robust_tracker.py"
grep -q 'Front visual observation: Soft MeanShift' "${RUNTIME}/robust_tracker.py" || fail "front SoftMS patch audit failed"
if grep -q 'weighted_xy = (raw_prob.unsqueeze(-1) \* centers).sum(dim=1)' "${RUNTIME}/robust_tracker.py"; then
  fail "Weighted Centroid front decoder still present"
fi
ln -s "${VISUAL_CKPT}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"
ln -s "${TEMPORAL_CKPT}" "${OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

BOARD_MODEL="unknown"
if [[ -r /sys/firmware/devicetree/base/model ]]; then
  BOARD_MODEL="$(tr -d '\000' </sys/firmware/devicetree/base/model || true)"
fi

echo "================================================================================"
echo "V39 FORWARD3x6 SOFTMS END-TO-END LATENCY BENCHMARK"
echo "Board   : ${BOARD_MODEL}"
echo "Runtime : mobilenet_v3_small + Forward3x6 SoftMS + 3-frame GRU + fixed-R Kalman + final MS 6x6/BW7"
echo "Warmup  : ${WARMUP} frames per route"
echo "Timing  : prepared UAV tensor -> full V39 inference -> final XY"
echo "Excludes: disk image I/O, image preprocessing, model loading, one-time SAT gallery/cache construction"
echo "Run dir : ${RUN_ROOT}"
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
  MS_MEASURE_LATENCY=1 \
  MS_LATENCY_WARMUP="${WARMUP}" \
  python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m "${JITTER_M}" 2>&1 | tee "${OUT}/eval.log"
)

SUMMARY="${OUT}/robust_tracker_summary.json"
[[ -s "${SUMMARY}" ]] || fail "benchmark finished without ${SUMMARY}"
RESULT="${RUN_ROOT}/v39_nx_softms_latency_result.json"
python3 - "${SUMMARY}" "${RESULT}" "${BOARD_MODEL}" <<'PY'
import json, sys
from pathlib import Path
import torch

summary_path, result_path, board_model = map(str, sys.argv[1:4])
d = json.loads(Path(summary_path).read_text(encoding="utf-8"))
rows = []
for route in ("route_B", "route_C"):
    r = d.get(route, {})
    if r.get("VisualObservationDecoder") != "forward 3x6 soft mean shift":
        raise SystemExit(f"ERROR: {route} did not use front SoftMS: {r.get('VisualObservationDecoder')}")
    e = r.get("EndToEndTiming", {})
    if not e or int(e.get("samples", 0)) <= 0:
        raise SystemExit(f"ERROR: missing EndToEndTiming for {route}")
    rows.append((route, r, e))

total = sum(int(e["samples"]) for _, _, e in rows)
mean_ms = sum(float(e["mean_ms"]) * int(e["samples"]) for _, _, e in rows) / total
fps = 1000.0 / mean_ms
payload = {
    "method": "V39 Forward3x6 SoftMS",
    "pipeline": "MobileNetV3-Small -> Forward 3x6 SoftMS -> 3-frame GRU -> fixed-R Kalman -> final MeanShift 6x6 BW7 -> XY",
    "checkpoint_retraining": False,
    "board_model": board_model,
    "cuda_device": torch.cuda.get_device_name(0),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "timing_scope": "prepared UAV tensor -> full patched V39 inference -> final XY",
    "excluded": ["image disk I/O", "image preprocessing", "model/checkpoint loading", "one-time satellite gallery/cache construction"],
    "warmup_frames_per_route": int(rows[0][2].get("warmup_frames", 0)),
    "samples": total,
    "mean_ms": mean_ms,
    "fps": fps,
    "routes": {name: {"MLE_m": r.get("MLE_m"), "P90_m": r.get("P90_m"), "LSR@15_pct": r.get("LSR@15_pct"), "timing": e} for name, r, e in rows},
}
Path(result_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("\n================================================================================")
print("V39 NX FORWARD3x6 SOFTMS RESULT")
print(f"Device                  : {payload['cuda_device']}")
for name, r, e in rows:
    print(f"{name:23s}: MLE={float(r['MLE_m']):.3f} m | mean={float(e['mean_ms']):.3f} ms | median={float(e['median_ms']):.3f} ms | P90={float(e['p90_ms']):.3f} ms | FPS={float(e['fps']):.2f}")
print("--------------------------------------------------------------------------------")
print(f"V39_FULL_PIPELINE_MEAN_MS = {mean_ms:.3f} ms")
print(f"V39_FULL_PIPELINE_FPS     = {fps:.2f} FPS")
print(f"Measured frames           = {total}")
print(f"Result JSON               = {result_path}")
print("================================================================================")
PY
