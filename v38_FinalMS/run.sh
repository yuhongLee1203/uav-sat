#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
SRC="${ROOT}/src"
OUT="${UAVSAT_OUTPUT_DIR:-${ROOT}/output}"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
JITTER_M="${JITTER_M:-8}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"
BACKBONE="mobilenet_v3_small"
BASE_ARCH="V36_PreviousStateOnly_MobileNetV3_Forward3x6_PolynomialKalman"
FINAL_ARCH="V38_FinalMS_MobileNetV3_MS1_GRU_KF1_TemporaryKF2_RegularizedMS2"

VISUAL_CKPT="${REPO_ROOT}/forNX/weights/v36_${BACKBONE}/checkpoints/visual_retrieval_A_only.pt"
LEGACY_TEMPORAL_CKPT="${REPO_ROOT}/PreviousState-exp/output/mobilenetv3_prevstate/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
LOCAL_TEMPORAL_CKPT="${OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

for f in config.py data.py robust_tracker.py visual_localizer.py visual_model.py; do
  [[ -f "${SRC}/${f}" ]] || { echo "ERROR: missing ${SRC}/${f}" >&2; exit 2; }
done
[[ -s "${VISUAL_CKPT}" ]] || { echo "ERROR: missing visual checkpoint ${VISUAL_CKPT}" >&2; exit 2; }
for route in route_A route_B route_C; do
  [[ -f "${DATA_ROOT}/routes/${route}/frames.csv" ]] || { echo "ERROR: missing ${DATA_ROOT}/routes/${route}/frames.csv" >&2; exit 2; }
done

mkdir -p "${OUT}/checkpoints" "${OUT}/feature_cache"
ln -sfn "${VISUAL_CKPT}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"

MODE="eval"
if [[ ! -s "${LOCAL_TEMPORAL_CKPT}" ]]; then
  if [[ -s "${LEGACY_TEMPORAL_CKPT}" ]]; then
    ln -sfn "${LEGACY_TEMPORAL_CKPT}" "${LOCAL_TEMPORAL_CKPT}"
    echo "[INFO] Reusing trained Previous-State temporal checkpoint."
  else
    MODE="train_eval"
    echo "[INFO] Temporal checkpoint not found; training on Route A before evaluation."
  fi
fi

export TORCH_HOME="${REPO_ROOT}/forNX/pretrained_cache/torch"
export HF_HOME="${REPO_ROOT}/forNX/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

export KF2_PROGRESS_SIGMA_BASE="${KF2_PROGRESS_SIGMA_BASE:-1.25}"
export KF2_CROSS_SIGMA_BASE="${KF2_CROSS_SIGMA_BASE:-1.00}"
export KF2_DISTANCE_GAIN="${KF2_DISTANCE_GAIN:-0.20}"
export KF2_MAX_PROGRESS_CORRECTION="${KF2_MAX_PROGRESS_CORRECTION:-4.0}"
export KF2_MAX_CROSS_CORRECTION="${KF2_MAX_CROSS_CORRECTION:-3.0}"
export MS2_KF_SIGMA_M="${MS2_KF_SIGMA_M:-4.0}"
export MS2_REFERENCE_SIGMA_M="${MS2_REFERENCE_SIGMA_M:-4.0}"
export MS2_KF_PRIOR_WEIGHT="${MS2_KF_PRIOR_WEIGHT:-1.50}"
export MS2_REFERENCE_PRIOR_WEIGHT="${MS2_REFERENCE_PRIOR_WEIGHT:-2.00}"
export MS2_BANDWIDTH_M="${MS2_BANDWIDTH_M:-5.0}"

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
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
d["architecture"] = sys.argv[2]
d["front_visual_decoder"] = "causal forward 3x6 Soft MeanShift"
d["second_kalman_update"] = "temporary current-frame predefined reference update; persistent KF1 state is unchanged"
d["kf2_persistent_feedback"] = False
d["persistent_navigation_state"] = "KF1 posterior only"
d["required_final_chain"] = "MS1 -> GRU -> KF Predict/Update #1 -> temporary KF Update #2 -> MS2 -> Final"
d["final_decoder"] = "full 6x6 prior-regularized Soft MeanShift; MeanShift output is final"
d["ms2_hyperparameters"] = {
    "kf_sigma_m": 4.0,
    "reference_sigma_m": 4.0,
    "kf_prior_weight": 1.5,
    "reference_prior_weight": 2.0,
    "bandwidth_m": 5.0,
}
p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
PY

echo "[DONE] v38_FinalMS result: ${OUT}/robust_tracker_summary.json"
