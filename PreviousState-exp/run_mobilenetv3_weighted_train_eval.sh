#!/usr/bin/env bash
set -Eeuo pipefail

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"
BASE_SRC="${REPO_ROOT}/v36_GvsK/previous_state_only"
SRC="${EXP_ROOT}/mobilenetv3_weighted/src"
FORNX="${REPO_ROOT}/forNX"
OUT="${UAVSAT_OUTPUT_DIR:-${EXP_ROOT}/output/mobilenetv3_weighted}"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR:-${EXP_ROOT}/output/mobilenetv3_weighted/feature_cache}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
JITTER_M="${JITTER_M:-8}"
ARCH="V36_PreviousStateOnly_MobileNetV3_Forward3x6WeightedCentroid_PolynomialKalman"
BACKBONE="mobilenet_v3_small"

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -f "${BASE_SRC}/${f}" ]] || { echo "ERROR: missing ${BASE_SRC}/${f}" >&2; exit 2; }
done
for route in route_A route_B route_C; do
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || { echo "ERROR: missing ${DATA_ROOT}/routes/${route}/frames.csv" >&2; exit 2; }
done

VISUAL_CKPT="${FORNX}/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt"
[[ -s "${VISUAL_CKPT}" ]] || { echo "ERROR: missing packaged MobileNetV3 visual checkpoint: ${VISUAL_CKPT}" >&2; exit 2; }

# This is a new isolated experiment. Old PreviousState-exp outputs are untouched.
rm -rf "${SRC}" "${OUT}"
mkdir -p "${SRC}" "${OUT}/checkpoints" "${CACHE_DIR}"
cp -a "${BASE_SRC}/." "${SRC}/"
ln -sfn "${VISUAL_CKPT}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"

export TORCH_HOME="${FORNX}/pretrained_cache/torch"
export HF_HOME="${FORNX}/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "============================================================================================================"
echo "MobileNetV3 Previous-State-Only: forward 3x6 Weighted Centroid -> GRU -> Polynomial -> Kalman"
echo "source: ${SRC}"
echo "output: ${OUT}"
echo "visual checkpoint: ${VISUAL_CKPT}"
echo "temporal GRU: retrained on Route A"
echo "front decoder: Weighted Centroid (NOT MeanShift)"
echo "============================================================================================================"

(
  cd "${SRC}"
  UAVSAT_DEVICE="${DEVICE}" \
  UAVSAT_OUTPUT_DIR="${OUT}" \
  UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
  UAVSAT_FEATURE_CACHE_DIR="${CACHE_DIR}" \
  UAVSAT_DATA_ROOT="${DATA_ROOT}" \
  UAVSAT_BACKBONE="${BACKBONE}" \
  UAVSAT_ARCHITECTURE_NAME="${ARCH}" \
  UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
  UAVSAT_EXPERIMENT_ANCHOR=weighted_centroid \
  UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
  UAVSAT_EXPERIMENT_MOTION=quadratic \
  UAVSAT_EXPERIMENT_KALMAN=learned \
  UAVSAT_EXPERIMENT_DISABLE_GRU=0 \
  UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
  python3 -u robust_tracker.py \
    --mode train_eval \
    --reuse-visual \
    --temporal-epochs "${TEMPORAL_EPOCHS}" \
    --patience "${PATIENCE}" \
    --jitter-m "${JITTER_M}"
) 2>&1 | tee "${OUT}/train_eval.log"

python3 - "${OUT}/robust_tracker_summary.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
d["front_visual_decoder"] = "forward 3x6 posterior-weighted centroid"
d["end_refinement"] = "none; external Kalman posterior is final output"
p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
PY

echo "[DONE] MobileNetV3 Weighted-Centroid result: ${OUT}/robust_tracker_summary.json"
