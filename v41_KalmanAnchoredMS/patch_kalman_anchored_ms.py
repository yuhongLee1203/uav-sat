#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_kalman_anchored_ms.py <robust_tracker.py>")

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

# This patch is applied AFTER v39_DirectFinalMS/patch_direct_finalms.py.
# It removes the predefined-reference prior from MS2 and turns the Kalman
# posterior itself into an adaptive KDE/MeanShift anchor mode.

# 1) Add diagnostics.
old_metrics = '''    kalman_errors = []
    ms2_shifts_from_kalman = []
'''
new_metrics = '''    kalman_errors = []
    ms2_shifts_from_kalman = []
    ms2_anchor_masses = []
    ms2_visual_confidences = []
'''
if s.count(old_metrics) != 1:
    raise SystemExit(f"ERROR: direct-MS metric block count={s.count(old_metrics)}")
s = s.replace(old_metrics, new_metrics, 1)

# 2) The frame reference must not participate in the final MS2 inference.
old_reference_block = '''        # The current predefined frame reference is NOT passed through another
        # Kalman update. It is used only as a spatial prior inside MS2.
        frame_reference_xy_t = cache.gt_xy[index : index + 1].to(device).float()
        frame_reference_xy = (
            frame_reference_xy_t[0].detach().cpu().numpy().astype(np.float64)
        )
        preferred_leg = route.frame_from_se(
            kalman_se[0], kalman_se[1]
        ).leg_index
'''
new_reference_block = '''        # No predefined frame-reference coordinate is used by MS2.
        # The single persistent Kalman posterior is the only spatial anchor.
        preferred_leg = route.frame_from_se(
            kalman_se[0], kalman_se[1]
        ).leg_index
'''
if s.count(old_reference_block) != 1:
    raise SystemExit(f"ERROR: direct-MS reference block count={s.count(old_reference_block)}")
s = s.replace(old_reference_block, new_reference_block, 1)

# 3) Replace Gaussian-reference regularization with a confidence-adaptive
#    Kalman anchor mode. The final coordinate remains the literal MeanShift
#    output; there is no post-MS filtering or clipping.
old_score_block = '''        tau2 = float(config.MEANSHIFT_SCORE_TAU)
        visual_log_probability = F.log_softmax(
            ms2_candidate.raw_logits / max(tau2, 1e-6), dim=1
        )
        d2_kalman = (
            ms2_candidate.centers - kalman_xy_t[:, None, :]
        ).square().sum(dim=2)
        d2_reference = (
            ms2_candidate.centers - frame_reference_xy_t[:, None, :]
        ).square().sum(dim=2)

        sigma_kalman = max(
            float(__import__("os").environ.get("MS2_KF_SIGMA_M", "4.0")),
            1e-3,
        )
        sigma_reference = max(
            float(__import__("os").environ.get("MS2_REFERENCE_SIGMA_M", "4.0")),
            1e-3,
        )
        weight_kalman = float(
            __import__("os").environ.get("MS2_KF_PRIOR_WEIGHT", "1.50")
        )
        weight_reference = float(
            __import__("os").environ.get("MS2_REFERENCE_PRIOR_WEIGHT", "2.50")
        )

        combined_log_probability = (
            visual_log_probability
            - weight_kalman
            * d2_kalman
            / (2.0 * sigma_kalman ** 2)
            - weight_reference
            * d2_reference
            / (2.0 * sigma_reference ** 2)
        )
        regularized_ms2_logits = tau2 * combined_log_probability

        ms2_xy_t, ms2_support_t, _, _, ms2_mode_weights_t, _ = soft_mean_shift(
            regularized_ms2_logits,
            ms2_candidate.centers,
            tau2,
            float(__import__("os").environ.get("MS2_BANDWIDTH_M", "5.0")),
            config.MEANSHIFT_ITERATIONS,
            config.MEANSHIFT_MODE_BETA,
        )
'''
new_score_block = '''        tau2 = float(config.MEANSHIFT_SCORE_TAU)
        visual_log_probability = F.log_softmax(
            ms2_candidate.raw_logits / max(tau2, 1e-6), dim=1
        )
        d2_kalman = (
            ms2_candidate.centers - kalman_xy_t[:, None, :]
        ).square().sum(dim=2)

        # Kalman is the only spatial prior. It regularizes the 36 visual
        # candidates, but unlike v40 it is ALSO inserted as an explicit KDE
        # point/mode. This gives MeanShift a valid "stay at Kalman" mode when
        # the agricultural visual posterior is ambiguous.
        sigma_kalman = max(
            float(__import__("os").environ.get("MS2_KF_SIGMA_M", "5.0")),
            1e-3,
        )
        weight_kalman = float(
            __import__("os").environ.get("MS2_KF_PRIOR_WEIGHT", "1.0")
        )
        spatial_visual_logp = (
            visual_log_probability
            - weight_kalman * d2_kalman / (2.0 * sigma_kalman ** 2)
        )
        visual_probs = F.softmax(spatial_visual_logp, dim=1)

        # Confidence is computed only from the current UAV-SAT posterior.
        # Sharp, separated visual peaks are allowed to refine Kalman more;
        # diffuse/repetitive responses keep most KDE mass on Kalman.
        eps = 1e-12
        n_visual = max(int(visual_probs.shape[1]), 2)
        entropy = -(
            visual_probs * torch.log(visual_probs.clamp_min(eps))
        ).sum(dim=1)
        entropy_norm = entropy / float(np.log(n_visual))
        top2 = torch.topk(visual_probs, k=2, dim=1).values
        margin_ratio = (top2[:, 0] - top2[:, 1]) / top2[:, 0].clamp_min(eps)
        entropy_conf = (1.0 - entropy_norm).clamp(0.0, 1.0)
        visual_confidence_t = (
            0.5 * entropy_conf + 0.5 * margin_ratio.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)

        anchor_min = float(
            __import__("os").environ.get("MS2_ANCHOR_MIN_MASS", "0.35")
        )
        anchor_max = float(
            __import__("os").environ.get("MS2_ANCHOR_MAX_MASS", "0.90")
        )
        anchor_min = min(max(anchor_min, 0.05), 0.95)
        anchor_max = min(max(anchor_max, anchor_min), 0.98)
        anchor_gamma = max(
            float(__import__("os").environ.get("MS2_ANCHOR_CONF_GAMMA", "0.75")),
            1e-3,
        )
        anchor_mass_t = anchor_max - (
            anchor_max - anchor_min
        ) * visual_confidence_t.pow(anchor_gamma)
        anchor_mass_t = anchor_mass_t.clamp(anchor_min, anchor_max)

        # Build one joint KDE distribution over 36 SAT locations + the exact
        # Kalman posterior location. The following soft_mean_shift therefore
        # remains the genuine final decoder.
        visual_mass_t = (1.0 - anchor_mass_t).unsqueeze(1)
        joint_visual_probs = visual_probs * visual_mass_t
        joint_anchor_probs = anchor_mass_t.unsqueeze(1)
        joint_probs = torch.cat(
            [joint_visual_probs, joint_anchor_probs], dim=1
        ).clamp_min(eps)
        anchored_ms2_logits = tau2 * torch.log(joint_probs)
        anchored_centers = torch.cat(
            [ms2_candidate.centers, kalman_xy_t[:, None, :]], dim=1
        )

        ms2_xy_t, ms2_support_t, _, _, ms2_mode_weights_t, _ = soft_mean_shift(
            anchored_ms2_logits,
            anchored_centers,
            tau2,
            float(__import__("os").environ.get("MS2_BANDWIDTH_M", "4.0")),
            config.MEANSHIFT_ITERATIONS,
            config.MEANSHIFT_MODE_BETA,
        )
        ms2_anchor_mass = float(anchor_mass_t[0].item())
        ms2_visual_confidence = float(visual_confidence_t[0].item())
        ms2_anchor_masses.append(ms2_anchor_mass)
        ms2_visual_confidences.append(ms2_visual_confidence)
'''
if s.count(old_score_block) != 1:
    raise SystemExit(f"ERROR: direct-MS score block count={s.count(old_score_block)}")
s = s.replace(old_score_block, new_score_block, 1)

# 4) Remove stale frame-reference CSV fields and record anchor diagnostics.
old_csv_reference = '''                "frame_reference_x": float(frame_reference_xy[0]),
                "frame_reference_y": float(frame_reference_xy[1]),
'''
if s.count(old_csv_reference) != 1:
    raise SystemExit(f"ERROR: frame-reference CSV block count={s.count(old_csv_reference)}")
s = s.replace(old_csv_reference, "", 1)

old_csv_tail = '''                "ms2_mode_count": int(ms2_mode_count),
                "ms2_shift_from_kalman_m": float(ms2_shift_from_kalman_m),
'''
new_csv_tail = '''                "ms2_mode_count": int(ms2_mode_count),
                "ms2_anchor_mass": float(ms2_anchor_mass),
                "ms2_visual_confidence": float(ms2_visual_confidence),
                "ms2_shift_from_kalman_m": float(ms2_shift_from_kalman_m),
'''
if s.count(old_csv_tail) != 1:
    raise SystemExit(f"ERROR: direct-MS CSV tail count={s.count(old_csv_tail)}")
s = s.replace(old_csv_tail, new_csv_tail, 1)

# 5) Summary diagnostics / definition.
old_summary_tail = '''    summary["MS2_MeanShiftFromKalman_m"] = float(np.mean(ms2_shifts_from_kalman)) if ms2_shifts_from_kalman else 0.0
    summary["MS2_MaxShiftFromKalman_m"] = float(np.max(ms2_shifts_from_kalman)) if ms2_shifts_from_kalman else 0.0
    summary["MS2_Definition"] = "full 6x6 Soft MeanShift after the single Kalman update; score = visual likelihood + Kalman spatial prior + predefined-reference spatial prior; MS2 output is final"
'''
new_summary_tail = '''    summary["MS2_MeanShiftFromKalman_m"] = float(np.mean(ms2_shifts_from_kalman)) if ms2_shifts_from_kalman else 0.0
    summary["MS2_MaxShiftFromKalman_m"] = float(np.max(ms2_shifts_from_kalman)) if ms2_shifts_from_kalman else 0.0
    summary["MS2_MeanAnchorMass"] = float(np.mean(ms2_anchor_masses)) if ms2_anchor_masses else 0.0
    summary["MS2_MeanVisualConfidence"] = float(np.mean(ms2_visual_confidences)) if ms2_visual_confidences else 0.0
    summary["MS2_Definition"] = "Kalman-centered full 6x6 Soft MeanShift with 36 visual candidates plus an adaptive Kalman anchor mode; no predefined-reference prior; MS2 output is final"
'''
if s.count(old_summary_tail) != 1:
    raise SystemExit(f"ERROR: direct-MS summary tail count={s.count(old_summary_tail)}")
s = s.replace(old_summary_tail, new_summary_tail, 1)

# The reference coordinate is still used later only for evaluation metrics;
# it must not occur in the MS2 scoring/refinement section after this patch.
for forbidden in [
    "frame_reference_xy_t",
    "frame_reference_xy =",
    "d2_reference",
    "weight_reference",
    "sigma_reference",
]:
    if forbidden in s:
        raise SystemExit(f"ERROR: stale reference-prior token remains: {forbidden}")

compile(s, str(p), "exec")
p.write_text(s, encoding="utf-8")
print("[OK] patched to Kalman-anchored adaptive MS2 with no reference prior")
