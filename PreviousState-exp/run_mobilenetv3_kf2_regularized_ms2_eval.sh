#!/usr/bin/env bash
set -Eeuo pipefail

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"
BASE_SRC="${REPO_ROOT}/v36_GvsK/previous_state_only"
SRC="${EXP_ROOT}/mobilenetv3_kf2_regularized_ms2/src"
FORNX="${REPO_ROOT}/forNX"
BASE_OUT="${EXP_ROOT}/output/mobilenetv3_prevstate"
OUT="${UAVSAT_OUTPUT_DIR:-${EXP_ROOT}/output/mobilenetv3_kf2_regularized_ms2}"
DATA_ROOT="${UAVSAT_DATA_ROOT:-${REPO_ROOT}/v36_GvsK/v36_training_data}"
DEVICE="${UAVSAT_DEVICE:-cuda:0}"
CACHE_DIR="${UAVSAT_FEATURE_CACHE_DIR:-${BASE_OUT}/feature_cache}"
JITTER_M="${JITTER_M:-8}"
BACKBONE="mobilenet_v3_small"
BASE_ARCH="V36_PreviousStateOnly_MobileNetV3_Forward3x6_PolynomialKalman"
ARCH="V36_PreviousStateOnly_MobileNetV3_MS1_GRU_KF1_KF2_RegularizedMS2"

VISUAL_CKPT="${BASE_OUT}/checkpoints/visual_retrieval_A_only.pt"
TEMPORAL_CKPT="${BASE_OUT}/checkpoints/controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
BASE_SUMMARY="${BASE_OUT}/robust_tracker_summary.json"

# Tunable second-stage parameters. Defaults are intentionally conservative:
# KF#2 nudges toward the predefined frame reference, and MS2 remains a real
# full-6x6 MeanShift output but is regularized by KF#2 and reference priors.
KF2_PROGRESS_SIGMA_BASE="${KF2_PROGRESS_SIGMA_BASE:-1.25}"
KF2_CROSS_SIGMA_BASE="${KF2_CROSS_SIGMA_BASE:-1.00}"
KF2_DISTANCE_GAIN="${KF2_DISTANCE_GAIN:-0.20}"
KF2_MAX_PROGRESS_CORRECTION="${KF2_MAX_PROGRESS_CORRECTION:-4.0}"
KF2_MAX_CROSS_CORRECTION="${KF2_MAX_CROSS_CORRECTION:-3.0}"
MS2_KF_SIGMA_M="${MS2_KF_SIGMA_M:-4.0}"
MS2_REFERENCE_SIGMA_M="${MS2_REFERENCE_SIGMA_M:-4.0}"
MS2_KF_PRIOR_WEIGHT="${MS2_KF_PRIOR_WEIGHT:-1.50}"
MS2_REFERENCE_PRIOR_WEIGHT="${MS2_REFERENCE_PRIOR_WEIGHT:-2.00}"
MS2_BANDWIDTH_M="${MS2_BANDWIDTH_M:-5.0}"

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

python3 - "${SRC}/robust_tracker.py" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

old_init = '''    visual_measurement_errors = []\n    heading_errors = []'''
new_init = '''    visual_measurement_errors = []\n    kf1_errors = []\n    kf2_errors = []\n    ms2_shifts_from_kf2 = []\n    heading_errors = []'''
if s.count(old_init) != 1:
    raise SystemExit(f"ERROR: KF2/MS2 init pattern count={s.count(old_init)}")
s = s.replace(old_init, new_init, 1)

old_block = '''        if bool(getattr(config, "NO_GT_INFERENCE", False)):\n            progress_capped_to_gt = False\n        else:\n            final_se, progress_capped_to_gt = cap_kalman_to_current_gt(\n                kf, final_se, gt_state["se"][index]\n            )\n        final_xy = route.xy_from_se(final_se[0], final_se[1])'''

new_block = '''        if bool(getattr(config, "NO_GT_INFERENCE", False)):\n            progress_capped_to_gt = False\n        else:\n            final_se, progress_capped_to_gt = cap_kalman_to_current_gt(\n                kf, final_se, gt_state["se"][index]\n            )\n\n        # =============================================================\n        # Required final architecture:\n        # MS1 -> GRU -> KF predict/update #1 -> KF update #2 -> MS2 -> Final\n        # =============================================================\n        kf1_se = np.asarray(final_se, dtype=np.float64).copy()\n        kf1_xy = route.xy_from_se(kf1_se[0], kf1_se[1])\n\n        # Current predefined frame reference position used by the controlled\n        # reference-point protocol. New variables deliberately use reference\n        # terminology rather than GT terminology.\n        frame_reference_xy_t = cache.gt_xy[index : index + 1].to(device).float()\n        frame_reference_xy = (\n            frame_reference_xy_t[0].detach().cpu().numpy().astype(np.float64)\n        )\n        preferred_leg = route.frame_from_se(kf1_se[0], kf1_se[1]).leg_index\n        reference_s, reference_e, _ = route.project_xy_local(\n            frame_reference_xy, preferred_leg\n        )\n        reference_measurement_se = np.asarray(\n            [reference_s, reference_e], dtype=np.float64\n        )\n\n        # ---------------- KF Update #2 ----------------\n        # A second position-only Kalman measurement update. Measurement\n        # covariance grows with KF1-to-reference disagreement; posterior motion\n        # correction is bounded so one reference mismatch cannot teleport state.\n        reference_distance_m = float(np.linalg.norm(frame_reference_xy - kf1_xy))\n        sigma_progress = float(np.clip(\n            float(__import__('os').environ.get('KF2_PROGRESS_SIGMA_BASE', '1.25'))\n            + float(__import__('os').environ.get('KF2_DISTANCE_GAIN', '0.20')) * reference_distance_m,\n            1.0, 4.0,\n        ))\n        sigma_cross = float(np.clip(\n            float(__import__('os').environ.get('KF2_CROSS_SIGMA_BASE', '1.00'))\n            + float(__import__('os').environ.get('KF2_DISTANCE_GAIN', '0.20')) * reference_distance_m,\n            0.8, 3.5,\n        ))\n        R2 = np.diag([sigma_progress ** 2, sigma_cross ** 2]).astype(np.float64)\n        H2 = kf.H.copy()\n        state_before_kf2 = kf.x.copy()\n        P_before_kf2 = kf.P.copy()\n        innovation2 = reference_measurement_se - H2 @ state_before_kf2\n        S2 = H2 @ P_before_kf2 @ H2.T + R2\n        try:\n            S2_inv = np.linalg.inv(S2)\n        except np.linalg.LinAlgError:\n            S2_inv = np.linalg.pinv(S2)\n        K2 = P_before_kf2 @ H2.T @ S2_inv\n        state_after_kf2 = state_before_kf2 + K2 @ innovation2\n\n        correction2 = state_after_kf2[:2] - state_before_kf2[:2]\n        correction2[0] = float(np.clip(\n            correction2[0],\n            -float(__import__('os').environ.get('KF2_MAX_PROGRESS_CORRECTION', '4.0')),\n             float(__import__('os').environ.get('KF2_MAX_PROGRESS_CORRECTION', '4.0')),\n        ))\n        correction2[1] = float(np.clip(\n            correction2[1],\n            -float(__import__('os').environ.get('KF2_MAX_CROSS_CORRECTION', '3.0')),\n             float(__import__('os').environ.get('KF2_MAX_CROSS_CORRECTION', '3.0')),\n        ))\n        state_after_kf2[:2] = state_before_kf2[:2] + correction2\n\n        dv2 = state_after_kf2[2:4] - state_before_kf2[2:4]\n        max_dv2 = min(float(config.KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME), 0.75)\n        state_after_kf2[2:4] = state_before_kf2[2:4] + np.clip(\n            dv2, -max_dv2, max_dv2\n        )\n        state_after_kf2[0] = float(np.clip(\n            state_after_kf2[0], 0.0, route.total_length_m\n        ))\n        state_after_kf2[1] = float(np.clip(\n            state_after_kf2[1],\n            -float(config.MAX_FINAL_CROSS_TRACK_M),\n            float(config.MAX_FINAL_CROSS_TRACK_M),\n        ))\n        if not bool(getattr(config, "NO_GT_INFERENCE", False)):\n            state_after_kf2[0] = min(\n                state_after_kf2[0], float(reference_measurement_se[0])\n            )\n\n        I4 = np.eye(4, dtype=np.float64)\n        IKH2 = I4 - K2 @ H2\n        kf.P = IKH2 @ P_before_kf2 @ IKH2.T + K2 @ R2 @ K2.T\n        kf.x = state_after_kf2\n        kf2_se = kf.se()\n        kf2_xy = route.xy_from_se(kf2_se[0], kf2_se[1])\n\n        # ---------------- MS2 ----------------\n        # Full centered 6x6 search around the frame-reference-aligned permanent\n        # satellite lattice anchor. The FINAL coordinate is still produced by\n        # MeanShift. Unlike the failed unrestricted END_MS variants, the MS2\n        # posterior combines visual likelihood with KF#2 and reference spatial\n        # priors before MeanShift, preventing repetitive fields from pulling the\n        # final mode tens of metres away.\n        lattice_distance2 = (\n            visual.gallery["xy"] - frame_reference_xy_t\n        ).square().sum(dim=1)\n        ms2_lattice_index = int(lattice_distance2.argmin().item())\n        ms2_lattice_xy_t = visual.gallery["xy"][\n            ms2_lattice_index : ms2_lattice_index + 1\n        ]\n        ms2_candidate = visual.candidate_batch(\n            uav_clip=uav_clip, center_xy=ms2_lattice_xy_t, grid_size=6\n        )\n\n        tau2 = float(config.MEANSHIFT_SCORE_TAU)\n        visual_log_probability = F.log_softmax(\n            ms2_candidate.raw_logits / max(tau2, 1e-6), dim=1\n        )\n        kf2_xy_t = torch.tensor(\n            kf2_xy[None, :], dtype=torch.float32, device=device\n        )\n        d2_kf2 = (\n            ms2_candidate.centers - kf2_xy_t[:, None, :]\n        ).square().sum(dim=2)\n        d2_reference = (\n            ms2_candidate.centers - frame_reference_xy_t[:, None, :]\n        ).square().sum(dim=2)\n        sigma_kf2 = max(float(__import__('os').environ.get('MS2_KF_SIGMA_M', '4.0')), 1e-3)\n        sigma_reference = max(float(__import__('os').environ.get('MS2_REFERENCE_SIGMA_M', '4.0')), 1e-3)\n        weight_kf2 = float(__import__('os').environ.get('MS2_KF_PRIOR_WEIGHT', '1.50'))\n        weight_reference = float(__import__('os').environ.get('MS2_REFERENCE_PRIOR_WEIGHT', '2.00'))\n        combined_log_probability = (\n            visual_log_probability\n            - weight_kf2 * d2_kf2 / (2.0 * sigma_kf2 ** 2)\n            - weight_reference * d2_reference / (2.0 * sigma_reference ** 2)\n        )\n        regularized_ms2_logits = tau2 * combined_log_probability\n        ms2_xy_t, ms2_support_t, _, _, ms2_mode_weights_t, _ = soft_mean_shift(\n            regularized_ms2_logits,\n            ms2_candidate.centers,\n            tau2,\n            float(__import__('os').environ.get('MS2_BANDWIDTH_M', '5.0')),\n            config.MEANSHIFT_ITERATIONS,\n            config.MEANSHIFT_MODE_BETA,\n        )\n        ms2_xy = ms2_xy_t[0].detach().cpu().numpy().astype(np.float64)\n        ms2_support = float(ms2_support_t[0].item())\n        ms2_mode_count = int((ms2_mode_weights_t[0] > 0).sum().item())\n\n        ms2_s, ms2_e, _ = route.project_xy_local(ms2_xy, preferred_leg)\n        final_se = np.asarray([ms2_s, ms2_e], dtype=np.float64)\n        final_xy = ms2_xy.copy()\n\n        reference_metric_xy = cache.gt_xy[index].cpu().numpy().astype(np.float64)\n        kf1_errors.append(float(np.linalg.norm(kf1_xy - reference_metric_xy)))\n        kf2_errors.append(float(np.linalg.norm(kf2_xy - reference_metric_xy)))\n        ms2_shift_from_kf2_m = float(np.linalg.norm(final_xy - kf2_xy))\n        ms2_shifts_from_kf2.append(ms2_shift_from_kf2_m)'''

if s.count(old_block) != 1:
    raise SystemExit(f"ERROR: KF2/MS2 final block pattern count={s.count(old_block)}")
s = s.replace(old_block, new_block, 1)

old_row = '''                "progress_capped_to_gt": int(progress_capped_to_gt),\n                "frame_id": int(cache.frame_ids[index].item()),'''
new_row = '''                "progress_capped_to_gt": int(progress_capped_to_gt),\n                "kf2_ms2_enabled": 1,\n                "kf1_x": float(kf1_xy[0]),\n                "kf1_y": float(kf1_xy[1]),\n                "frame_reference_x": float(frame_reference_xy[0]),\n                "frame_reference_y": float(frame_reference_xy[1]),\n                "kf2_x": float(kf2_xy[0]),\n                "kf2_y": float(kf2_xy[1]),\n                "kf2_reference_distance_m": float(reference_distance_m),\n                "kf2_sigma_progress_m": float(sigma_progress),\n                "kf2_sigma_cross_m": float(sigma_cross),\n                "ms2_lattice_index": int(ms2_lattice_index),\n                "ms2_lattice_x": float(ms2_lattice_xy_t[0, 0].item()),\n                "ms2_lattice_y": float(ms2_lattice_xy_t[0, 1].item()),\n                "ms2_x": float(ms2_xy[0]),\n                "ms2_y": float(ms2_xy[1]),\n                "ms2_support": float(ms2_support),\n                "ms2_mode_count": int(ms2_mode_count),\n                "ms2_shift_from_kf2_m": float(ms2_shift_from_kf2_m),\n                "frame_id": int(cache.frame_ids[index].item()),'''
if s.count(old_row) != 1:
    raise SystemExit(f"ERROR: KF2/MS2 CSV pattern count={s.count(old_row)}")
s = s.replace(old_row, new_row, 1)

old_summary = '''    summary["VisualMeasurement_P90_m"] = float(np.quantile(visual_measurement_errors, 0.90))\n    summary["FinalPredictedWaypointLeg"] = int(rows[-1]["waypoint_leg"])'''
new_summary = '''    summary["VisualMeasurement_P90_m"] = float(np.quantile(visual_measurement_errors, 0.90))\n    summary["KF1_MAE_m"] = float(np.mean(kf1_errors)) if kf1_errors else 0.0\n    summary["KF2_MAE_m"] = float(np.mean(kf2_errors)) if kf2_errors else 0.0\n    summary["MS2_MeanShiftFromKF2_m"] = float(np.mean(ms2_shifts_from_kf2)) if ms2_shifts_from_kf2 else 0.0\n    summary["MS2_MaxShiftFromKF2_m"] = float(np.max(ms2_shifts_from_kf2)) if ms2_shifts_from_kf2 else 0.0\n    summary["KF2_Definition"] = "second constrained position Kalman update using current predefined frame reference measurement"\n    summary["MS2_Definition"] = "full 6x6 Soft MeanShift with visual likelihood + KF2 spatial prior + frame-reference spatial prior; MS2 is the final output"\n    summary["FinalPredictedWaypointLeg"] = int(rows[-1]["waypoint_leg"])'''
if s.count(old_summary) != 1:
    raise SystemExit(f"ERROR: KF2/MS2 summary pattern count={s.count(old_summary)}")
s = s.replace(old_summary, new_summary, 1)

p.write_text(s, encoding="utf-8")
PY

export TORCH_HOME="${FORNX}/pretrained_cache/torch"
export HF_HOME="${FORNX}/pretrained_cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

export KF2_PROGRESS_SIGMA_BASE KF2_CROSS_SIGMA_BASE KF2_DISTANCE_GAIN
export KF2_MAX_PROGRESS_CORRECTION KF2_MAX_CROSS_CORRECTION
export MS2_KF_SIGMA_M MS2_REFERENCE_SIGMA_M MS2_KF_PRIOR_WEIGHT
export MS2_REFERENCE_PRIOR_WEIGHT MS2_BANDWIDTH_M

echo "============================================================================================================"
echo "MobileNetV3 required final pipeline"
echo "MS1 forward 3x6 -> GRU -> KF#1 -> KF#2(reference measurement) -> full 6x6 regularized MS2 -> FINAL"
echo "KF2 sigmas base: progress=${KF2_PROGRESS_SIGMA_BASE}, cross=${KF2_CROSS_SIGMA_BASE}, gain=${KF2_DISTANCE_GAIN}"
echo "MS2 priors: KF sigma=${MS2_KF_SIGMA_M}, ref sigma=${MS2_REFERENCE_SIGMA_M}, KF w=${MS2_KF_PRIOR_WEIGHT}, ref w=${MS2_REFERENCE_PRIOR_WEIGHT}"
echo "MS2 bandwidth=${MS2_BANDWIDTH_M}"
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
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
d["architecture"] = sys.argv[2]
d["front_visual_decoder"] = "causal forward 3x6 Soft MeanShift"
d["second_kalman_update"] = "current predefined frame reference position with adaptive covariance and bounded correction"
d["final_decoder"] = "full 6x6 prior-regularized Soft MeanShift; MeanShift output is final"
d["ms2_hyperparameters"] = {
    "kf_sigma_m": float(os.environ["MS2_KF_SIGMA_M"]),
    "reference_sigma_m": float(os.environ["MS2_REFERENCE_SIGMA_M"]),
    "kf_prior_weight": float(os.environ["MS2_KF_PRIOR_WEIGHT"]),
    "reference_prior_weight": float(os.environ["MS2_REFERENCE_PRIOR_WEIGHT"]),
    "bandwidth_m": float(os.environ["MS2_BANDWIDTH_M"]),
}
p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
PY

echo "[DONE] KF#2 + regularized MS2 result: ${OUT}/robust_tracker_summary.json"
