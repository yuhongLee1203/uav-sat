#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BASE_SRC="${REPO_ROOT}/v39_DirectFinalMS/base_src"
SRC="${ROOT}/runtime_src"
OUT="${UAVSAT_OUTPUT_DIR:-${ROOT}/output}"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
JITTER_M="${JITTER_M:-8}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
BACKBONE="mobilenet_v3_small"
BASE_ARCH="V36_PreviousStateOnly_MobileNetV3_Forward3x6_PolynomialKalman"
FINAL_ARCH="V41_KalmanAnchoredMS_MobileNetV3_MS1_GRU_Kalman_AdaptiveMS2"

VISUAL_CKPT="${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt"
LEGACY_TEMPORAL_CKPT="${REPO_ROOT}/PreviousState-exp/output/mobilenetv3_prevstate/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
LOCAL_TEMPORAL_CKPT="${OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -f "${BASE_SRC}/${f}" ]] || { echo "ERROR: missing ${BASE_SRC}/${f}" >&2; exit 2; }
done
[[ -f "${REPO_ROOT}/v39_DirectFinalMS/patch_direct_finalms.py" ]] || { echo "ERROR: missing v39 direct-MS patch" >&2; exit 2; }
[[ -f "${ROOT}/patch_kalman_anchored_ms.py" ]] || { echo "ERROR: missing v41 Kalman-anchor patch" >&2; exit 2; }
[[ -s "${VISUAL_CKPT}" ]] || { echo "ERROR: missing visual checkpoint ${VISUAL_CKPT}" >&2; exit 2; }

for route in route_A route_B route_C; do
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || {
    echo "ERROR: missing ${DATA_ROOT}/routes/${route}/frames.csv" >&2
    exit 2
  }
done

rm -rf "${SRC}"
mkdir -p "${SRC}" "${OUT}/checkpoints" "${OUT}/feature_cache"
cp -a "${BASE_SRC}/." "${SRC}/"
python3 "${REPO_ROOT}/v39_DirectFinalMS/patch_direct_finalms.py" "${SRC}/robust_tracker.py"
python3 "${ROOT}/patch_kalman_anchored_ms.py" "${SRC}/robust_tracker.py"

ln -sfn "${VISUAL_CKPT}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"

MODE="eval"
if [[ ! -s "${LOCAL_TEMPORAL_CKPT}" ]]; then
  if [[ -s "${LEGACY_TEMPORAL_CKPT}" ]]; then
    ln -sfn "${LEGACY_TEMPORAL_CKPT}" "${LOCAL_TEMPORAL_CKPT}"
    echo "[INFO] Reusing trained Previous-State temporal checkpoint."
  else
    MODE="train_eval"
    echo "[INFO] Temporal checkpoint not found; training Route A before evaluation."
  fi
fi

export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Final MS2 uses ONLY the Kalman posterior + current UAV-SAT visual response.
# No predefined frame-reference coordinate is used in MS2.
export MS2_KF_SIGMA_M="${MS2_KF_SIGMA_M:-5.0}"
export MS2_KF_PRIOR_WEIGHT="${MS2_KF_PRIOR_WEIGHT:-1.0}"
export MS2_ANCHOR_MIN_MASS="${MS2_ANCHOR_MIN_MASS:-0.35}"
export MS2_ANCHOR_MAX_MASS="${MS2_ANCHOR_MAX_MASS:-0.90}"
export MS2_ANCHOR_CONF_GAMMA="${MS2_ANCHOR_CONF_GAMMA:-0.75}"
export MS2_BANDWIDTH_M="${MS2_BANDWIDTH_M:-4.0}"

echo "============================================================================================================"
echo "v41 Kalman-Anchored Adaptive FinalMS"
echo "flow: MS1 -> GRU -> Kalman Predict/Update -> MS2 -> FINAL"
echo "MS2 search: full 6x6 around the Kalman posterior"
echo "MS2 KDE: 36 UAV-SAT visual candidates + one exact Kalman anchor mode"
echo "anchor mass: confidence-adaptive; ambiguous visual response stays closer to Kalman"
echo "NO reference prior in MS2; NO second Kalman; NO post-MS clipping/filter"
echo "FINAL is the literal MeanShift output"
echo "output: ${OUT}"
echo "============================================================================================================"

cd "${SRC}"
ARGS=(--mode "${MODE}" --reuse-visual --jitter-m "${JITTER_M}")
if [[ "${MODE}" == "train_eval" ]]; then
  ARGS+=(--temporal-epochs "${TEMPORAL_EPOCHS}" --patience "${PATIENCE}")
fi

UAVSAT_DEVICE="${DEVICE}" \
UAVSAT_OUTPUT_DIR="${OUT}" \
UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
UAVSAT_FEATURE_CACHE_DIR="${OUT}/feature_cache" \
UAVSAT_DATA_ROOT="${DATA_ROOT}" \
UAVSAT_BACKBONE="${BACKBONE}" \
UAVSAT_ARCHITECTURE_NAME="${BASE_ARCH}" \
UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
UAVSAT_EXPERIMENT_ANCHOR=softms \
UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
UAVSAT_EXPERIMENT_MOTION=quadratic \
UAVSAT_EXPERIMENT_KALMAN=learned \
UAVSAT_EXPERIMENT_DISABLE_GRU=0 \
UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
python3 -u robust_tracker.py "${ARGS[@]}" 2>&1 | tee "${OUT}/${MODE}.log"

python3 - "${OUT}/robust_tracker_summary.json" "${FINAL_ARCH}" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
d["architecture"] = sys.argv[2]
d["final_chain"] = "MS1 -> GRU -> Kalman Predict/Update -> MS2 -> Final"
d["second_kalman_update"] = "none"
d["persistent_navigation_state"] = "single Kalman posterior"
d["final_decoder"] = "confidence-adaptive Kalman-anchored Soft MeanShift; MeanShift output is final"
d["ms2_search_center"] = "nearest permanent SAT lattice point to the single Kalman posterior"
d["ms2_reference_prior"] = "none"
d["ms2_score"] = "UAV-SAT visual likelihood + Kalman spatial prior + confidence-adaptive Kalman KDE anchor"
d["ms2_hyperparameters"] = {
    "kalman_sigma_m": float(os.environ.get("MS2_KF_SIGMA_M", "5.0")),
    "kalman_prior_weight": float(os.environ.get("MS2_KF_PRIOR_WEIGHT", "1.0")),
    "anchor_min_mass": float(os.environ.get("MS2_ANCHOR_MIN_MASS", "0.35")),
    "anchor_max_mass": float(os.environ.get("MS2_ANCHOR_MAX_MASS", "0.90")),
    "anchor_conf_gamma": float(os.environ.get("MS2_ANCHOR_CONF_GAMMA", "0.75")),
    "bandwidth_m": float(os.environ.get("MS2_BANDWIDTH_M", "4.0")),
}
p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
PY

echo "[DONE] result: ${OUT}/robust_tracker_summary.json"
