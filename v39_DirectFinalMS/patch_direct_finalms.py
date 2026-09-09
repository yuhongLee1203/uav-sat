#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_direct_finalms.py <robust_tracker.py>")

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

# 1) Metrics: one persistent Kalman only.
old_metrics = '''    kf1_errors = []
    kf2_errors = []
    ms2_shifts_from_kf2 = []
'''
new_metrics = '''    kalman_errors = []
    ms2_shifts_from_kalman = []
'''
if s.count(old_metrics) != 1:
    raise SystemExit(f"ERROR: metric block count={s.count(old_metrics)}")
s = s.replace(old_metrics, new_metrics, 1)

# 2) Replace the complete temporary-KF2 final refinement with a direct
#    Kalman -> regularized MS2 -> Final stage.
start_marker = '''        # =============================================================
        # Required final architecture:
'''
end_marker = '''        if prepared_uav is not None:
'''
start = s.find(start_marker)
end = s.find(end_marker, start)
if start < 0 or end < 0 or end <= start:
    raise SystemExit("ERROR: could not locate v38 final-refinement block")

direct_block = '''        # =============================================================
        # Direct final architecture:
        # MS1 -> GRU -> KF predict/update -> MS2 -> Final
        # =============================================================
        kalman_se = np.asarray(final_se, dtype=np.float64).copy()
        kalman_xy = route.xy_from_se(kalman_se[0], kalman_se[1])

        # The current predefined frame reference is NOT passed through another
        # Kalman update. It is used only as a spatial prior inside MS2.
        frame_reference_xy_t = cache.gt_xy[index : index + 1].to(device).float()
        frame_reference_xy = (
            frame_reference_xy_t[0].detach().cpu().numpy().astype(np.float64)
        )
        preferred_leg = route.frame_from_se(
            kalman_se[0], kalman_se[1]
        ).leg_index

        # ---------------- MS2 ----------------
        # Open a complete 6x6 SAT window around the Kalman posterior.
        # MeanShift is still the final decoder. Its score combines:
        #   (1) UAV-SAT visual likelihood,
        #   (2) distance to the Kalman posterior,
        #   (3) distance to the predefined frame reference.
        # No second Kalman update is performed.
        kalman_xy_t = torch.tensor(
            kalman_xy[None, :], dtype=torch.float32, device=device
        )
        lattice_distance2 = (
            visual.gallery["xy"] - kalman_xy_t
        ).square().sum(dim=1)
        ms2_lattice_index = int(lattice_distance2.argmin().item())
        ms2_lattice_xy_t = visual.gallery["xy"][
            ms2_lattice_index : ms2_lattice_index + 1
        ]
        ms2_candidate = visual.candidate_batch(
            uav_clip=uav_clip,
            center_xy=ms2_lattice_xy_t,
            grid_size=6,
        )

        tau2 = float(config.MEANSHIFT_SCORE_TAU)
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
        ms2_xy = (
            ms2_xy_t[0].detach().cpu().numpy().astype(np.float64)
        )
        ms2_support = float(ms2_support_t[0].item())
        ms2_mode_count = int(
            (ms2_mode_weights_t[0] > 0).sum().item()
        )

        ms2_s, ms2_e, _ = route.project_xy_local(
            ms2_xy, preferred_leg
        )
        final_se = np.asarray([ms2_s, ms2_e], dtype=np.float64)
        final_xy = ms2_xy.copy()

        reference_metric_xy = (
            cache.gt_xy[index].cpu().numpy().astype(np.float64)
        )
        kalman_errors.append(
            float(np.linalg.norm(kalman_xy - reference_metric_xy))
        )
        ms2_shift_from_kalman_m = float(
            np.linalg.norm(final_xy - kalman_xy)
        )
        ms2_shifts_from_kalman.append(ms2_shift_from_kalman_m)
'''
s = s[:start] + direct_block + s[end:]

# 3) CSV fields: remove KF2-specific logging.
csv_start = '''                "kf2_ms2_enabled": 1,
'''
csv_end_line = '''                "ms2_shift_from_kf2_m": float(ms2_shift_from_kf2_m),
'''
csv_start_i = s.find(csv_start)
csv_end_i = s.find(csv_end_line, csv_start_i)
if csv_start_i < 0 or csv_end_i < 0:
    raise SystemExit("ERROR: could not locate v38 CSV KF2 block")
csv_end_i += len(csv_end_line)
csv_block = '''                "direct_kalman_ms2_enabled": 1,
                "kalman_x": float(kalman_xy[0]),
                "kalman_y": float(kalman_xy[1]),
                "frame_reference_x": float(frame_reference_xy[0]),
                "frame_reference_y": float(frame_reference_xy[1]),
                "ms2_lattice_index": int(ms2_lattice_index),
                "ms2_lattice_x": float(ms2_lattice_xy_t[0, 0].item()),
                "ms2_lattice_y": float(ms2_lattice_xy_t[0, 1].item()),
                "ms2_x": float(ms2_xy[0]),
                "ms2_y": float(ms2_xy[1]),
                "ms2_support": float(ms2_support),
                "ms2_mode_count": int(ms2_mode_count),
                "ms2_shift_from_kalman_m": float(ms2_shift_from_kalman_m),
'''
s = s[:csv_start_i] + csv_block + s[csv_end_i:]

# 4) Summary fields.
old_summary = '''    summary["KF1_MAE_m"] = float(np.mean(kf1_errors)) if kf1_errors else 0.0
    summary["KF2_MAE_m"] = float(np.mean(kf2_errors)) if kf2_errors else 0.0
    summary["MS2_MeanShiftFromKF2_m"] = float(np.mean(ms2_shifts_from_kf2)) if ms2_shifts_from_kf2 else 0.0
    summary["MS2_MaxShiftFromKF2_m"] = float(np.max(ms2_shifts_from_kf2)) if ms2_shifts_from_kf2 else 0.0
    summary["KF2_Definition"] = "temporary current-frame second constrained Kalman update using predefined frame reference measurement; KF1 remains persistent state"
    summary["MS2_Definition"] = "full 6x6 Soft MeanShift with visual likelihood + KF2 spatial prior + frame-reference spatial prior; MS2 is the final output"
'''
new_summary = '''    summary["Kalman_MAE_m"] = float(np.mean(kalman_errors)) if kalman_errors else 0.0
    summary["MS2_MeanShiftFromKalman_m"] = float(np.mean(ms2_shifts_from_kalman)) if ms2_shifts_from_kalman else 0.0
    summary["MS2_MaxShiftFromKalman_m"] = float(np.max(ms2_shifts_from_kalman)) if ms2_shifts_from_kalman else 0.0
    summary["MS2_Definition"] = "full 6x6 Soft MeanShift after the single Kalman update; score = visual likelihood + Kalman spatial prior + predefined-reference spatial prior; MS2 output is final"
'''
if s.count(old_summary) != 1:
    raise SystemExit(f"ERROR: summary block count={s.count(old_summary)}")
s = s.replace(old_summary, new_summary, 1)

# 5) Console architecture description.
old_console = (
    '"causal-heading forward 3x6 local visual measurement -> robust constrained route-coordinate Kalman -> final XY.",'
)
new_console = (
    '"causal-heading forward 3x6 local visual measurement -> robust constrained route-coordinate Kalman -> full 6x6 regularized MS2 -> final XY.",'
)
if old_console in s:
    s = s.replace(old_console, new_console, 1)

# Sanity checks.
for forbidden in [
    "kf2_errors",
    "ms2_shifts_from_kf2",
    "KF Update #2",
    "kf2_ms2_enabled",
    "ms2_shift_from_kf2_m",
]:
    if forbidden in s:
        raise SystemExit(f"ERROR: stale KF2 token remains: {forbidden}")

compile(s, str(p), "exec")
p.write_text(s, encoding="utf-8")
print("[OK] patched to direct Kalman -> MS2 -> Final")
