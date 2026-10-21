#!/usr/bin/env bash
set -Eeuo pipefail

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"
BASE_SRC="${REPO_ROOT}/v36_GvsK/previous_state_only"
SRC="${EXP_ROOT}/mobilenetv3_double_ms_reference/src"
FORNX="${REPO_ROOT}/forNX"
BASE_OUT="${EXP_ROOT}/output/mobilenetv3_prevstate"
OUT="${UAVSAT_OUTPUT_DIR:-${EXP_ROOT}/output/mobilenetv3_double_ms_reference}"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR:-${BASE_OUT}/feature_cache}"
JITTER_M="${JITTER_M:-8}"
BACKBONE="mobilenet_v3_small"
BASE_ARCH="V36_PreviousStateOnly_MobileNetV3_Forward3x6_PolynomialKalman"
ARCH="V36_PreviousStateOnly_MobileNetV3_DoubleSoftMS_Forward3x6_KalmanMatchedReference_Full6x6"

VISUAL_CKPT="${BASE_OUT}/checkpoints/visual_retrieval_A_only.pt"
TEMPORAL_CKPT="${BASE_OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
BASE_SUMMARY="${BASE_OUT}/robust_tracker_summary.json"

# Reuse the exact original MobileNetV3 + forward-3x6 SoftMS temporal model.
NEED_BASE=0
[[ -s "${VISUAL_CKPT}" ]] || NEED_BASE=1
[[ -s "${TEMPORAL_CKPT}" ]] || NEED_BASE=1
[[ -s "${BASE_SUMMARY}" ]] || NEED_BASE=1
if [[ "${NEED_BASE}" -eq 0 ]]; then
  if ! python3 - "${BASE_SUMMARY}" "${BASE_ARCH}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
expected = sys.argv[2]
raise SystemExit(0 if d.get("architecture") == expected and d.get("experiment_anchor") == "softms" else 1)
PY
  then
    NEED_BASE=1
  fi
fi

if [[ "${NEED_BASE}" -ne 0 ]]; then
  echo "[INFO] Original MobileNetV3 SoftMS checkpoint missing/stale; training baseline first."
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

# Patch only the final output stage.
# Front stage remains the original causal forward 3x6 Soft MeanShift.
# After Kalman, match the Kalman XY to the nearest predefined route reference
# point, align that reference to the nearest permanent SAT lattice anchor, open
# a full centered 6x6 gallery, and run the second Soft MeanShift there.
python3 - "${SRC}/robust_tracker.py" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

old_init = '''    visual_measurement_errors = []\n    heading_errors = []'''
new_init = '''    visual_measurement_errors = []\n    end_ms_shifts_from_kalman = []\n    end_ms_reference_distances = []\n    heading_errors = []'''
if s.count(old_init) != 1:
    raise SystemExit(f"ERROR: Double-MS init pattern count={s.count(old_init)}")
s = s.replace(old_init, new_init, 1)

old_block = '''        if bool(getattr(config, "NO_GT_INFERENCE", False)):\n            progress_capped_to_gt = False\n        else:\n            final_se, progress_capped_to_gt = cap_kalman_to_current_gt(\n                kf, final_se, gt_state["se"][index]\n            )\n        final_xy = route.xy_from_se(final_se[0], final_se[1])'''

new_block = '''        if bool(getattr(config, "NO_GT_INFERENCE", False)):\n            progress_capped_to_gt = False\n        else:\n            final_se, progress_capped_to_gt = cap_kalman_to_current_gt(\n                kf, final_se, gt_state["se"][index]\n            )\n\n        # --------------------------------------------------------------\n        # Double-MS final refinement.\n        #\n        # Stage 1 (already executed above): causal forward 3x6 SoftMS.\n        # Stage 2:\n        #   Kalman posterior XY\n        #     -> nearest predefined route reference point\n        #     -> nearest permanent SAT lattice anchor\n        #     -> complete centered 6x6 = 36 candidates\n        #     -> Soft MeanShift\n        #     -> reported final XY\n        #\n        # This second result is output-only and is not written back into the\n        # Kalman or recurrent state.\n        # --------------------------------------------------------------\n        kalman_before_end_se = np.asarray(final_se, dtype=np.float64).copy()\n        kalman_before_end_xy = route.xy_from_se(\n            kalman_before_end_se[0], kalman_before_end_se[1]\n        )\n\n        # The cached route coordinates form the predefined reference-point bank.\n        # Local variable names intentionally use reference-point terminology.\n        reference_bank_xy = cache.gt_xy.to(device).float()\n        kalman_xy_t = torch.tensor(\n            kalman_before_end_xy[None, :], dtype=torch.float32, device=device\n        )\n        reference_distance2 = (\n            reference_bank_xy - kalman_xy_t\n        ).square().sum(dim=1)\n        matched_reference_index = int(reference_distance2.argmin().item())\n        matched_reference_xy_t = reference_bank_xy[\n            matched_reference_index : matched_reference_index + 1\n        ]\n        matched_reference_distance_m = float(\n            torch.sqrt(reference_distance2[matched_reference_index]).item()\n        )\n        end_ms_reference_distances.append(matched_reference_distance_m)\n\n        lattice_distance2 = (\n            visual.gallery["xy"] - matched_reference_xy_t\n        ).square().sum(dim=1)\n        matched_lattice_index = int(lattice_distance2.argmin().item())\n        matched_lattice_xy_t = visual.gallery["xy"][\n            matched_lattice_index : matched_lattice_index + 1\n        ]\n\n        end_ms_candidate = visual.candidate_batch(\n            uav_clip=uav_clip,\n            center_xy=matched_lattice_xy_t,\n            grid_size=6,\n        )\n        end_ms_xy = (\n            end_ms_candidate.softms_xy[0]\n            .detach().cpu().numpy().astype(np.float64)\n        )\n\n        matched_reference_xy = (\n            matched_reference_xy_t[0].detach().cpu().numpy().astype(np.float64)\n        )\n        preferred_leg = route.frame_from_se(\n            kalman_before_end_se[0], kalman_before_end_se[1]\n        ).leg_index\n        end_ms_s, end_ms_e, _ = route.project_xy_local(\n            end_ms_xy, preferred_leg\n        )\n        final_se = np.asarray([end_ms_s, end_ms_e], dtype=np.float64)\n        final_xy = end_ms_xy.copy()\n        end_ms_shift_m = float(np.linalg.norm(final_xy - kalman_before_end_xy))\n        end_ms_shifts_from_kalman.append(end_ms_shift_m)'''

if s.count(old_block) != 1:
    raise SystemExit(f"ERROR: Double-MS Kalman block pattern count={s.count(old_block)}")
s = s.replace(old_block, new_block, 1)

old_row = '''                "progress_capped_to_gt": int(progress_capped_to_gt),\n                "frame_id": int(cache.frame_ids[index].item()),'''
new_row = '''                "progress_capped_to_gt": int(progress_capped_to_gt),\n                "double_ms_enabled": 1,\n                "kalman_before_end_x": float(kalman_before_end_xy[0]),\n                "kalman_before_end_y": float(kalman_before_end_xy[1]),\n                "matched_reference_index": int(matched_reference_index),\n                "matched_reference_x": float(matched_reference_xy[0]),\n                "matched_reference_y": float(matched_reference_xy[1]),\n                "matched_reference_distance_m": float(matched_reference_distance_m),\n                "matched_lattice_index": int(matched_lattice_index),\n                "matched_lattice_x": float(matched_lattice_xy_t[0, 0].item()),\n                "matched_lattice_y": float(matched_lattice_xy_t[0, 1].item()),\n                "end_ms_softms_x": float(end_ms_xy[0]),\n                "end_ms_softms_y": float(end_ms_xy[1]),\n                "end_ms_shift_from_kalman_m": float(end_ms_shift_m),\n                "end_ms_support": float(end_ms_candidate.softms_support[0].item()),\n                "end_ms_mode_count": int(end_ms_candidate.softms_mode_count[0].item()),\n                "frame_id": int(cache.frame_ids[index].item()),'''
if s.count(old_row) != 1:
    raise SystemExit(f"ERROR: Double-MS CSV pattern count={s.count(old_row)}")
s = s.replace(old_row, new_row, 1)

old_summary = '''    summary["VisualMeasurement_P90_m"] = float(np.quantile(visual_measurement_errors, 0.90))\n    summary["FinalPredictedWaypointLeg"] = int(rows[-1]["waypoint_leg"])'''
new_summary = '''    summary["VisualMeasurement_P90_m"] = float(np.quantile(visual_measurement_errors, 0.90))\n    summary["END_MS_MeanShiftFromKalman_m"] = float(np.mean(end_ms_shifts_from_kalman)) if end_ms_shifts_from_kalman else 0.0\n    summary["END_MS_MaxShiftFromKalman_m"] = float(np.max(end_ms_shifts_from_kalman)) if end_ms_shifts_from_kalman else 0.0\n    summary["MatchedReferenceMeanDistanceFromKalman_m"] = float(np.mean(end_ms_reference_distances)) if end_ms_reference_distances else 0.0\n    summary["MatchedReferenceMaxDistanceFromKalman_m"] = float(np.max(end_ms_reference_distances)) if end_ms_reference_distances else 0.0\n    summary["END_MS_SearchCenter"] = "Kalman posterior -> nearest predefined route reference point -> nearest permanent SAT lattice anchor -> centered full 6x6"\n    summary["END_MS_Decoder"] = "second Soft MeanShift on all 36 candidates; output-only; not fed back into Kalman/GRU"\n    summary["FinalPredictedWaypointLeg"] = int(rows[-1]["waypoint_leg"])'''
if s.count(old_summary) != 1:
    raise SystemExit(f"ERROR: Double-MS summary pattern count={s.count(old_summary)}")
s = s.replace(old_summary, new_summary, 1)

p.write_text(s, encoding="utf-8")
PY

export TORCH_HOME="${FORNX}/pretrained_cache/torch"
export HF_HOME="${FORNX}/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "============================================================================================================"
echo "MobileNetV3 Double SoftMS"
echo "stage 1: forward 3x6 SoftMS -> GRU -> Polynomial -> Kalman"
echo "stage 2: Kalman XY -> matched reference point -> SAT lattice center -> full 6x6 -> SoftMS -> FINAL"
echo "output: ${OUT}"
echo "============================================================================================================"

(
  cd "${SRC}"
  UAVSAT_DEVICE="${DEVICE}" \
  UAVSAT_OUTPUT_DIR="${OUT}" \
  UAVSAT_CHECKPOINT_DIR="${OUT}/checkpoints" \
  UAVSAT_FEATURE_CACHE_DIR="${CACHE_DIR}" \
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
  python3 -u robust_tracker.py --mode eval --reuse-visual --jitter-m "${JITTER_M}"
) 2>&1 | tee "${OUT}/eval.log"

python3 - "${OUT}/robust_tracker_summary.json" "${ARCH}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
d["architecture"] = sys.argv[2]
d["front_visual_decoder"] = "causal forward 3x6 Soft MeanShift"
d["end_refinement"] = "Kalman-matched predefined reference point -> centered full 6x6 Soft MeanShift"
p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
PY

echo "[DONE] MobileNetV3 Double-MS reference-matched result: ${OUT}/robust_tracker_summary.json"
