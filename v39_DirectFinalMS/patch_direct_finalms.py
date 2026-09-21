#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_direct_finalms.py <robust_tracker.py>")

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

# -----------------------------------------------------------------------------
# Front decoder is FIXED to Forward-18 Soft MeanShift.
# Remove the old decoder switch entirely so a weighted-centroid path cannot be
# selected by an environment variable or accidentally reintroduced at runtime.
# -----------------------------------------------------------------------------
old_anchor = '''    # Anchor ablation: the default is V36 SoftMS; weighted centroid uses the
    # exact same local posterior and candidates without mean-shift iterations.
    if str(getattr(config, "EXPERIMENT_ANCHOR", "softms")) == "weighted_centroid":
        anchor_xy_all = (posterior.unsqueeze(-1) * candidate.centers).sum(dim=1)
    else:
        anchor_xy_all = candidate.softms_xy
'''
new_anchor = '''    # Forward-18 decoder is always Soft MeanShift.
    # No alternate centroid decoder exists in this runtime.
    anchor_xy_all = candidate.softms_xy
'''
if s.count(old_anchor) != 1:
    raise SystemExit(f"ERROR: front decoder switch count={s.count(old_anchor)}")
s = s.replace(old_anchor, new_anchor, 1)

old_uncertainty = '''    if str(getattr(config, "EXPERIMENT_ANCHOR", "softms")) == "softms":
        _, _, softms_modes_all, _, softms_mode_weights_all, _ = soft_mean_shift(
            candidate.raw_logits,
            candidate.centers,
            config.MEANSHIFT_SCORE_TAU,
            config.MEANSHIFT_BANDWIDTH_M,
            config.MEANSHIFT_ITERATIONS,
            config.MEANSHIFT_MODE_BETA,
        )
'''
new_uncertainty = '''    _, _, softms_modes_all, _, softms_mode_weights_all, _ = soft_mean_shift(
        candidate.raw_logits,
        candidate.centers,
        config.MEANSHIFT_SCORE_TAU,
        config.MEANSHIFT_BANDWIDTH_M,
        config.MEANSHIFT_ITERATIONS,
        config.MEANSHIFT_MODE_BETA,
    )
'''
if s.count(old_uncertainty) != 1:
    raise SystemExit(f"ERROR: SoftMS uncertainty switch count={s.count(old_uncertainty)}")
s = s.replace(old_uncertainty, new_uncertainty, 1)

old_variance = '''        if str(getattr(config, "EXPERIMENT_ANCHOR", "softms")) == "softms":
            variance_points = softms_modes_all[h]
            variance_weights = softms_mode_weights_all[h]
        else:
            variance_points = candidate.centers[h]
            variance_weights = posterior[h]
'''
new_variance = '''        variance_points = softms_modes_all[h]
        variance_weights = softms_mode_weights_all[h]
'''
if s.count(old_variance) != 1:
    raise SystemExit(f"ERROR: SoftMS variance switch count={s.count(old_variance)}")
s = s.replace(old_variance, new_variance, 1)

# -----------------------------------------------------------------------------
# One persistent pre-final-MS Kalman estimator, followed by exactly one final
# local MeanShift.  The original Forward-18 SoftMS remains intact in front.
# -----------------------------------------------------------------------------
old_metrics = '''    kf1_errors = []
    kf2_errors = []
    ms2_shifts_from_kf2 = []
'''
new_metrics = '''    kalman_errors = []
    ms_shifts_from_kalman = []
    ms_latency_rows_ms = []
'''
if s.count(old_metrics) != 1:
    raise SystemExit(f"ERROR: metric block count={s.count(old_metrics)}")
s = s.replace(old_metrics, new_metrics, 1)

start_marker = '''        # =============================================================
        # Required final architecture:
'''
end_marker = '''        if prepared_uav is not None:
'''
start = s.find(start_marker)
end = s.find(end_marker, start)
if start < 0 or end < 0 or end <= start:
    raise SystemExit("ERROR: could not locate legacy final-refinement block")

direct_block = '''        # =============================================================
        # Selected chain:
        # Forward-18 SoftMS -> GRU -> Kalman -> ONE final MS -> Final
        # =============================================================
        kalman_se = np.asarray(final_se, dtype=np.float64).copy()
        kalman_xy = route.xy_from_se(kalman_se[0], kalman_se[1])

        # Keep the original v39 predefined frame-reference prior unchanged.
        frame_reference_xy_t = cache.gt_xy[index : index + 1].to(device).float()
        frame_reference_xy = (
            frame_reference_xy_t[0].detach().cpu().numpy().astype(np.float64)
        )
        preferred_leg = route.frame_from_se(kalman_se[0], kalman_se[1]).leg_index

        env = __import__("os").environ
        ms_enabled = env.get("MS_ENABLED", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }
        ms_grid_size = int(env.get("MS_GRID_SIZE", "6"))
        if ms_grid_size < 2:
            raise ValueError("MS_GRID_SIZE must be >= 2")
        measure_ms = env.get("MS_MEASURE_LATENCY", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }

        kalman_xy_t = torch.tensor(
            kalman_xy[None, :], dtype=torch.float32, device=device
        )

        if ms_enabled:
            # Candidate lookup/scoring is deliberately OUTSIDE the MS timer.
            # The paper's MS latency means: candidate centers and final logits
            # are already available -> run ONE MeanShift decoder -> XY.
            lattice_distance2 = (
                visual.gallery["xy"] - kalman_xy_t
            ).square().sum(dim=1)
            ms_lattice_index = int(lattice_distance2.argmin().item())
            ms_lattice_xy_t = visual.gallery["xy"][
                ms_lattice_index : ms_lattice_index + 1
            ]

            # Score the final candidate set directly.  Do not call
            # visual.candidate_batch(), because that helper also contains its
            # own visual SoftMS; the final stage must execute exactly one MS.
            ms_indices = regular_grid_indices(
                visual.gallery["xy"],
                visual.gallery["pixel"],
                visual.pixel_index,
                ms_lattice_xy_t,
                ms_grid_size,
                config.SAT_STRIDE,
                visual.device,
            )
            ms_centers = visual.gallery["xy"][ms_indices]
            ms_satellite_clip = visual.gallery["clip_feat"][ms_indices]
            ms_z_uav = obs.candidate.z_uav
            ms_z_sat = visual.model.encode_sat_from_clip(
                ms_satellite_clip.reshape(-1, ms_satellite_clip.shape[-1]),
                ms_centers.reshape(-1, 2),
            ).reshape(ms_centers.shape[0], ms_centers.shape[1], -1)
            ms_raw_logits = visual.model.logit_scale.exp().clamp(max=100.0) * (
                ms_z_uav[:, None] * ms_z_sat
            ).sum(dim=2)

            tau = float(config.MEANSHIFT_SCORE_TAU)
            visual_log_probability = F.log_softmax(
                ms_raw_logits / max(tau, 1e-6), dim=1
            )
            d2_kalman = (
                ms_centers - kalman_xy_t[:, None, :]
            ).square().sum(dim=2)
            d2_reference = (
                ms_centers - frame_reference_xy_t[:, None, :]
            ).square().sum(dim=2)

            sigma_kalman = max(float(env.get("MS_KF_SIGMA_M", "4.0")), 1e-3)
            sigma_reference = max(float(env.get("MS_REFERENCE_SIGMA_M", "4.0")), 1e-3)
            weight_kalman = float(env.get("MS_KF_PRIOR_WEIGHT", "1.50"))
            weight_reference = float(env.get("MS_REFERENCE_PRIOR_WEIGHT", "2.50"))

            combined_log_probability = (
                visual_log_probability
                - weight_kalman * d2_kalman / (2.0 * sigma_kalman ** 2)
                - weight_reference * d2_reference / (2.0 * sigma_reference ** 2)
            )
            regularized_ms_logits = tau * combined_log_probability

            if measure_ms and device.type == "cuda":
                torch.cuda.synchronize(device)
            ms_timer_start = time.perf_counter() if measure_ms else None

            ms_xy_t, ms_support_t, _, _, ms_mode_weights_t, _ = soft_mean_shift(
                regularized_ms_logits,
                ms_centers,
                tau,
                float(env.get("MS_BANDWIDTH_M", "7.0")),
                config.MEANSHIFT_ITERATIONS,
                config.MEANSHIFT_MODE_BETA,
            )
            ms_xy = ms_xy_t[0].detach().cpu().numpy().astype(np.float64)
            ms_support = float(ms_support_t[0].item())
            ms_mode_count = int((ms_mode_weights_t[0] > 0).sum().item())
            ms_s, ms_e, _ = route.project_xy_local(ms_xy, preferred_leg)
            final_se = np.asarray([ms_s, ms_e], dtype=np.float64)
            final_xy = ms_xy.copy()

            if measure_ms:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                ms_latency_ms = (time.perf_counter() - ms_timer_start) * 1000.0
                ms_latency_rows_ms.append(float(ms_latency_ms))
            else:
                ms_latency_ms = 0.0
        else:
            ms_lattice_index = -1
            ms_lattice_xy_t = kalman_xy_t
            ms_xy = kalman_xy.copy()
            ms_support = 0.0
            ms_mode_count = 0
            ms_latency_ms = 0.0
            final_se = kalman_se.copy()
            final_xy = kalman_xy.copy()

        reference_metric_xy = cache.gt_xy[index].cpu().numpy().astype(np.float64)
        kalman_errors.append(float(np.linalg.norm(kalman_xy - reference_metric_xy)))
        ms_shift_from_kalman_m = float(np.linalg.norm(final_xy - kalman_xy))
        ms_shifts_from_kalman.append(ms_shift_from_kalman_m)
'''
s = s[:start] + direct_block + s[end:]

# CSV fields: remove temporary KF2/MS2 logging and record the actual final MS.
csv_start = '''                "kf2_ms2_enabled": 1,
'''
csv_end_line = '''                "ms2_shift_from_kf2_m": float(ms2_shift_from_kf2_m),
'''
csv_start_i = s.find(csv_start)
csv_end_i = s.find(csv_end_line, csv_start_i)
if csv_start_i < 0 or csv_end_i < 0:
    raise SystemExit("ERROR: could not locate legacy CSV KF2 block")
csv_end_i += len(csv_end_line)
csv_block = '''                "direct_kalman_ms_enabled": int(ms_enabled),
                "ms_grid_size": int(ms_grid_size),
                "kalman_x": float(kalman_xy[0]),
                "kalman_y": float(kalman_xy[1]),
                "frame_reference_x": float(frame_reference_xy[0]),
                "frame_reference_y": float(frame_reference_xy[1]),
                "ms_lattice_index": int(ms_lattice_index),
                "ms_lattice_x": float(ms_lattice_xy_t[0, 0].item()),
                "ms_lattice_y": float(ms_lattice_xy_t[0, 1].item()),
                "ms_x": float(ms_xy[0]),
                "ms_y": float(ms_xy[1]),
                "ms_support": float(ms_support),
                "ms_mode_count": int(ms_mode_count),
                "ms_shift_from_kalman_m": float(ms_shift_from_kalman_m),
                "ms_latency_ms": float(ms_latency_ms),
'''
s = s[:csv_start_i] + csv_block + s[csv_end_i:]

old_summary = '''    summary["KF1_MAE_m"] = float(np.mean(kf1_errors)) if kf1_errors else 0.0
    summary["KF2_MAE_m"] = float(np.mean(kf2_errors)) if kf2_errors else 0.0
    summary["MS2_MeanShiftFromKF2_m"] = float(np.mean(ms2_shifts_from_kf2)) if ms2_shifts_from_kf2 else 0.0
    summary["MS2_MaxShiftFromKF2_m"] = float(np.max(ms2_shifts_from_kf2)) if ms2_shifts_from_kf2 else 0.0
    summary["KF2_Definition"] = "temporary current-frame second constrained Kalman update using predefined frame reference measurement; KF1 remains persistent state"
    summary["MS2_Definition"] = "full 6x6 Soft MeanShift with visual likelihood + KF2 spatial prior + frame-reference spatial prior; MS2 is the final output"
'''
new_summary = '''    summary["Kalman_MAE_m"] = float(np.mean(kalman_errors)) if kalman_errors else 0.0
    summary["MS_MeanShiftFromKalman_m"] = float(np.mean(ms_shifts_from_kalman)) if ms_shifts_from_kalman else 0.0
    summary["MS_MaxShiftFromKalman_m"] = float(np.max(ms_shifts_from_kalman)) if ms_shifts_from_kalman else 0.0
    summary["VisualObservationDecoder"] = "Forward-18 Soft MeanShift"
    summary["VisualObservationUncertainty"] = "SoftMS converged-mode variance projected to route parallel/cross coordinates"
    summary["MS_Enabled"] = bool(str(__import__("os").environ.get("MS_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"})
    summary["MS_GridSize"] = int(__import__("os").environ.get("MS_GRID_SIZE", "6"))
    summary["OnlineMeanShiftCount"] = 1 if summary["MS_Enabled"] else 0
    _ms_warmup = int(__import__("os").environ.get("MS_LATENCY_WARMUP", "30"))
    _ms_latency_eval = ms_latency_rows_ms[_ms_warmup:] if len(ms_latency_rows_ms) > _ms_warmup else ms_latency_rows_ms
    summary["MS_LatencyMean_ms"] = float(np.mean(_ms_latency_eval)) if _ms_latency_eval else 0.0
    summary["MS_LatencyP90_ms"] = float(np.quantile(_ms_latency_eval, 0.90)) if _ms_latency_eval else 0.0
    summary["MS_ThroughputFPS"] = (1000.0 / summary["MS_LatencyMean_ms"]) if summary["MS_LatencyMean_ms"] > 0 else 0.0
    summary["MS_LatencyWarmupFrames"] = int(_ms_warmup)
    summary["MS_Definition"] = "exactly one final local Soft MeanShift after the Kalman estimator"
    summary["MS_LatencyDefinition"] = "final candidate centers + regularized logits already prepared -> one soft_mean_shift decoder -> metric XY"
'''
if s.count(old_summary) != 1:
    raise SystemExit(f"ERROR: summary block count={s.count(old_summary)}")
s = s.replace(old_summary, new_summary, 1)

old_console = (
    '"causal-heading forward 3x6 local visual measurement -> robust constrained route-coordinate Kalman -> final XY.",'
)
new_console = (
    '"Forward-18 SoftMS visual observation -> GRU -> robust constrained route-coordinate Kalman -> one final MeanShift -> final XY.",'
)
if old_console in s:
    s = s.replace(old_console, new_console, 1)

for forbidden in [
    "kf2_errors",
    "ms2_shifts_from_kf2",
    "KF Update #2",
    "kf2_ms2_enabled",
    "ms2_shift_from_kf2_m",
    "MS2_",
    "weighted_centroid",
    "Weighted Centroid",
]:
    if forbidden in s:
        raise SystemExit(f"ERROR: forbidden stale token remains in runtime: {forbidden}")

# Runtime must contain exactly three SoftMS calls:
#   1) Forward-18 visual position,
#   2) Forward-18 converged-mode uncertainty,
#   3) final post-Kalman MeanShift.
if s.count("soft_mean_shift(") != 3:
    raise SystemExit(f"ERROR: unexpected soft_mean_shift call count={s.count('soft_mean_shift(')}")
if "anchor_xy_all = candidate.softms_xy" not in s:
    raise SystemExit("ERROR: Forward-18 SoftMS anchor missing")

compile(s, str(p), "exec")
p.write_text(s, encoding="utf-8")
print("[OK] clean v39 runtime: Forward-18 SoftMS -> GRU -> Kalman -> one final MeanShift")
print("[OK] weighted-centroid decoder removed from generated runtime source")
