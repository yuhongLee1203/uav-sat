#!/usr/bin/env bash
set -Eeuo pipefail

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"
BASE_SRC="${REPO_ROOT}/v36_GvsK/previous_state_only"
SRC="${EXP_ROOT}/mobilenetv3_postkalman6x6/src"
FORNX="${REPO_ROOT}/forNX"
BASE_OUT="${EXP_ROOT}/output/mobilenetv3_prevstate"
OUT="${UAVSAT_OUTPUT_DIR:-${EXP_ROOT}/output/mobilenetv3_postkalman6x6}"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR:-${BASE_OUT}/feature_cache}"
JITTER_M="${JITTER_M:-8}"
BACKBONE="mobilenet_v3_small"
ARCH="V36_PreviousStateOnly_MobileNetV3_PostKalmanFull6x6SoftMS"

VISUAL_CKPT="${BASE_OUT}/checkpoints/visual_retrieval_A_only.pt"
TEMPORAL_CKPT="${BASE_OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

# This variant must use the exact MobileNetV3 model from variant 2.
if [[ ! -s "${VISUAL_CKPT}" || ! -s "${TEMPORAL_CKPT}" ]]; then
  echo "[INFO] MobileNetV3 Previous-State checkpoint is missing; training variant 2 first."
  UAVSAT_DEVICE="${DEVICE}" JITTER_M="${JITTER_M}" \
    bash "${EXP_ROOT}/run_mobilenetv3_train_eval.sh"
fi
[[ -s "${VISUAL_CKPT}" ]] || { echo "ERROR: missing ${VISUAL_CKPT}" >&2; exit 2; }
[[ -s "${TEMPORAL_CKPT}" ]] || { echo "ERROR: missing ${TEMPORAL_CKPT}" >&2; exit 2; }

rm -rf "${SRC}" "${OUT}"
mkdir -p "${SRC}" "${OUT}/checkpoints"
cp -a "${BASE_SRC}/." "${SRC}/"
ln -sfn "${VISUAL_CKPT}" "${OUT}/checkpoints/visual_retrieval_A_only.pt"
ln -sfn "${TEMPORAL_CKPT}" "${OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"

# Patch only inference output: after the normal Kalman posterior, run a second
# FULL 6x6 (36-candidate) SoftMS around that posterior. No forward selector is
# used in this second stage. The refined result is output-only and is not written
# back into the Kalman state.
python3 - "${SRC}/robust_tracker.py" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

old_init = '''    visual_measurement_errors = []\n    heading_errors = []'''
new_init = '''    visual_measurement_errors = []\n    post_kalman_6x6_shifts = []\n    heading_errors = []'''
if s.count(old_init) != 1:
    raise SystemExit(f"ERROR: post6 init pattern count={s.count(old_init)}")
s = s.replace(old_init, new_init, 1)

old_block = '''        if bool(getattr(config, "NO_GT_INFERENCE", False)):\n            progress_capped_to_gt = False\n        else:\n            final_se, progress_capped_to_gt = cap_kalman_to_current_gt(\n                kf, final_se, gt_state["se"][index]\n            )\n        final_xy = route.xy_from_se(final_se[0], final_se[1])'''

new_block = '''        if bool(getattr(config, "NO_GT_INFERENCE", False)):\n            progress_capped_to_gt = False\n        else:\n            final_se, progress_capped_to_gt = cap_kalman_to_current_gt(\n                kf, final_se, gt_state["se"][index]\n            )\n\n        # --------------------------------------------------------------\n        # Output-only post-Kalman FULL 6x6 SoftMS refinement.\n        # Center the second local gallery on the current Kalman posterior,\n        # score all 36 patches (NO forward 3x6 restriction), and report the\n        # resulting SoftMS coordinate as this frame's final localization.\n        # The Kalman internal state is intentionally NOT overwritten.\n        # --------------------------------------------------------------\n        kalman_pre_refine_se = np.asarray(final_se, dtype=np.float64).copy()\n        kalman_pre_refine_xy = route.xy_from_se(\n            kalman_pre_refine_se[0], kalman_pre_refine_se[1]\n        )\n        post6_center_xy = torch.tensor(\n            kalman_pre_refine_xy[None, :], dtype=torch.float32, device=device\n        )\n        post6_candidate = visual.candidate_batch(\n            uav_clip, post6_center_xy, grid_size=6\n        )\n        post6_softms_xy = (\n            post6_candidate.softms_xy[0].detach().cpu().numpy().astype(np.float64)\n        )\n        preferred_leg = route.frame_from_se(\n            kalman_pre_refine_se[0], kalman_pre_refine_se[1]\n        ).leg_index\n        post6_s, post6_e, _ = route.project_xy_local(\n            post6_softms_xy, preferred_leg\n        )\n        post6_s = float(np.clip(post6_s, 0.0, route.total_length_m))\n        post6_e = float(np.clip(\n            post6_e,\n            -float(config.MAX_FINAL_CROSS_TRACK_M),\n            float(config.MAX_FINAL_CROSS_TRACK_M),\n        ))\n        if not bool(getattr(config, "NO_GT_INFERENCE", False)):\n            post6_s = min(post6_s, float(gt_state["se"][index, 0]))\n        final_se = np.asarray([post6_s, post6_e], dtype=np.float64)\n        final_xy = route.xy_from_se(final_se[0], final_se[1])\n        post6_shift_m = float(np.linalg.norm(final_xy - kalman_pre_refine_xy))\n        post_kalman_6x6_shifts.append(post6_shift_m)'''

if s.count(old_block) != 1:
    raise SystemExit(f"ERROR: Kalman final block pattern count={s.count(old_block)}")
s = s.replace(old_block, new_block, 1)

old_row = '''                "progress_capped_to_gt": int(progress_capped_to_gt),\n                "frame_id": int(cache.frame_ids[index].item()),'''
new_row = '''                "progress_capped_to_gt": int(progress_capped_to_gt),\n                "post_kalman_full6x6": 1,\n                "kalman_pre_refine_x": float(kalman_pre_refine_xy[0]),\n                "kalman_pre_refine_y": float(kalman_pre_refine_xy[1]),\n                "post6x6_softms_x": float(post6_softms_xy[0]),\n                "post6x6_softms_y": float(post6_softms_xy[1]),\n                "post6x6_shift_m": float(post6_shift_m),\n                "post6x6_support": float(post6_candidate.softms_support[0].item()),\n                "post6x6_mode_count": int(post6_candidate.softms_mode_count[0].item()),\n                "frame_id": int(cache.frame_ids[index].item()),'''
if s.count(old_row) != 1:
    raise SystemExit(f"ERROR: CSV insertion pattern count={s.count(old_row)}")
s = s.replace(old_row, new_row, 1)

old_summary = '''    summary["VisualMeasurement_P90_m"] = float(np.quantile(visual_measurement_errors, 0.90))\n    summary["FinalPredictedWaypointLeg"] = int(rows[-1]["waypoint_leg"])'''
new_summary = '''    summary["VisualMeasurement_P90_m"] = float(np.quantile(visual_measurement_errors, 0.90))\n    summary["PostKalmanFull6x6MeanShift_m"] = float(np.mean(post_kalman_6x6_shifts)) if post_kalman_6x6_shifts else 0.0\n    summary["PostKalmanFull6x6MaxShift_m"] = float(np.max(post_kalman_6x6_shifts)) if post_kalman_6x6_shifts else 0.0\n    summary["PostKalmanFull6x6Definition"] = "output-only full 6x6 SoftMS centered on Kalman posterior; no forward selector; not fed back into Kalman"\n    summary["FinalPredictedWaypointLeg"] = int(rows[-1]["waypoint_leg"])'''
if s.count(old_summary) != 1:
    raise SystemExit(f"ERROR: summary insertion pattern count={s.count(old_summary)}")
s = s.replace(old_summary, new_summary, 1)

p.write_text(s, encoding="utf-8")
PY

export TORCH_HOME="${FORNX}/pretrained_cache/torch"
export HF_HOME="${FORNX}/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

(
  cd "${SRC}"
  UAVSAT_DEVICE="${DEVICE}" \
  UAVSAT_OUTPUT_DIR="${OUT}" \
  UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
  UAVSAT_FEATURE_CACHE_DIR="${CACHE_DIR}" \
  UAVSAT_DATA_ROOT="${DATA_ROOT}" \
  UAVSAT_BACKBONE="${BACKBONE}" \
  UAVSAT_ARCHITECTURE_NAME="V36_PreviousStateOnly_MobileNetV3_Forward3x6_PolynomialKalman" \
  UAVSAT_REFERENCE_PROTOCOL=controlled_gt_jitter \
  UAVSAT_EXPERIMENT_ANCHOR=softms \
  UAVSAT_EXPERIMENT_FRAME_COUNT=3 \
  UAVSAT_EXPERIMENT_MOTION=quadratic \
  UAVSAT_EXPERIMENT_KALMAN=learned \
  UAVSAT_EXPERIMENT_DISABLE_GRU=0 \
  UAVSAT_EXPERIMENT_FORWARD_ONLY=1 \
  python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m "${JITTER_M}"
) 2>&1 | tee "${OUT}/eval.log"

# Rewrite only the summary architecture label so the result is self-describing;
# checkpoint compatibility still used the variant-2 architecture string above.
python3 - "${OUT}/robust_tracker_summary.json" "${ARCH}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
d["architecture"] = sys.argv[2]
d["post_kalman_refinement"] = "full 6x6 SoftMS output-only refinement centered on Kalman posterior"
p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
PY

echo "[DONE] MobileNetV3 + post-Kalman full-6x6 result: ${OUT}/robust_tracker_summary.json"
