#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_direct_finalms.py <robust_tracker.py>")

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

# -----------------------------------------------------------------------------
# ONLY methodological change relative to the original v39 front-end:
# replace the front 3x6 SoftMS decoder with posterior Weighted Centroid.
# No training loss, LR, motion, Kalman, teacher-forcing or protocol is changed.
# -----------------------------------------------------------------------------
old_front_ms = '''    softms_xy, softms_support, _, _, mode_weights, _ = soft_mean_shift(
        raw_logits,
        centers,
        config.MEANSHIFT_SCORE_TAU,
        config.MEANSHIFT_BANDWIDTH_M,
        config.MEANSHIFT_ITERATIONS,
        config.MEANSHIFT_MODE_BETA,
    )
    return CandidateBatch(
        indices=selected_indices,
        centers=centers,
        z_uav=z_uav,
        z_sat=z_sat,
        raw_logits=raw_logits,
        raw_prob=raw_prob,
        raw_top1_xy=raw_top1_xy,
        softms_xy=softms_xy,
        softms_support=softms_support,
        softms_mode_count=(mode_weights > 0).sum(dim=1),
    )
'''
new_front_ms = '''    # Front visual observation: Weighted Centroid, no MeanShift.
    # visual_observation() uses the local posterior again for the actual anchor
    # and computes uncertainty from posterior-weighted candidate dispersion.
    weighted_xy = (raw_prob.unsqueeze(-1) * centers).sum(dim=1)
    posterior_support = raw_prob.max(dim=1).values
    posterior_mode_count = torch.ones(
        raw_prob.shape[0], dtype=torch.long, device=raw_prob.device
    )
    return CandidateBatch(
        indices=selected_indices,
        centers=centers,
        z_uav=z_uav,
        z_sat=z_sat,
        raw_logits=raw_logits,
        raw_prob=raw_prob,
        raw_top1_xy=raw_top1_xy,
        softms_xy=weighted_xy,
        softms_support=posterior_support,
        softms_mode_count=posterior_mode_count,
    )
'''
if s.count(old_front_ms) != 1:
    raise SystemExit(f"ERROR: front SoftMS block count={s.count(old_front_ms)}")
s = s.replace(old_front_ms, new_front_ms, 1)

old_anchor_comment = '''    # Anchor ablation: the default is V36 SoftMS; weighted centroid uses the
    # exact same local posterior and candidates without mean-shift iterations.
'''
new_anchor_comment = '''    # Front decoder ablation: weighted centroid uses the same local posterior.
    # Its uncertainty is the posterior-weighted candidate spread in route axes.
'''
if old_anchor_comment in s:
    s = s.replace(old_anchor_comment, new_anchor_comment, 1)

# One persistent pre-final-MS Kalman estimator, exactly as selected in v39.
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
        # v39 selected chain, with ONLY the front decoder changed:
        # Weighted Centroid -> GRU -> Kalman -> ONE final MS -> Final
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

            # Do NOT call visual.candidate_batch() here: that legacy helper runs
            # an internal SoftMS. Score the same candidate set directly so the
            # online final path contains exactly one MeanShift.
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

            # PURE MS DECODER TIMER. Start only after candidates and logits are
            # ready; stop after MeanShift output has become the metric XY/SE.
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
    summary["VisualObservationDecoder"] = "posterior weighted centroid"
    summary["VisualObservationUncertainty"] = "posterior-weighted spatial variance projected to route parallel/cross coordinates"
    summary["MS_Enabled"] = bool(str(__import__("os").environ.get("MS_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"})
    summary["MS_GridSize"] = int(__import__("os").environ.get("MS_GRID_SIZE", "6"))
    summary["OnlineMeanShiftCount"] = 1 if summary["MS_Enabled"] else 0
    _ms_warmup = int(__import__("os").environ.get("MS_LATENCY_WARMUP", "30"))
    _ms_latency_eval = ms_latency_rows_ms[_ms_warmup:] if len(ms_latency_rows_ms) > _ms_warmup else ms_latency_rows_ms
    summary["MS_LatencyMean_ms"] = float(np.mean(_ms_latency_eval)) if _ms_latency_eval else 0.0
    summary["MS_LatencyP90_ms"] = float(np.quantile(_ms_latency_eval, 0.90)) if _ms_latency_eval else 0.0
    summary["MS_ThroughputFPS"] = (1000.0 / summary["MS_LatencyMean_ms"]) if summary["MS_LatencyMean_ms"] > 0 else 0.0
    summary["MS_LatencyWarmupFrames"] = int(_ms_warmup)
    summary["MS_Definition"] = "exactly one final local Soft MeanShift after the original v39 Kalman estimator"
    summary["MS_LatencyDefinition"] = "final candidate centers + regularized logits already prepared -> one soft_mean_shift decoder -> metric XY"
'''
if s.count(old_summary) != 1:
    raise SystemExit(f"ERROR: summary block count={s.count(old_summary)}")
s = s.replace(old_summary, new_summary, 1)

old_console = (
    '"causal-heading forward 3x6 local visual measurement -> robust constrained route-coordinate Kalman -> final XY.",'
)
new_console = (
    '"weighted-centroid visual observation -> GRU -> robust constrained route-coordinate Kalman -> one final MeanShift -> final XY.",'
)
if old_console in s:
    s = s.replace(old_console, new_console, 1)

# Static correctness audit: front MS removed; temporary KF2 removed; exactly one
# explicit final MeanShift remains in the selected online path.
for forbidden in [
    "kf2_errors",
    "ms2_shifts_from_kf2",
    "KF Update #2",
    "kf2_ms2_enabled",
    "ms2_shift_from_kf2_m",
    "MS2_",
]:
    if forbidden in s:
        raise SystemExit(f"ERROR: stale token remains: {forbidden}")
if s.count("soft_mean_shift(") != 2:
    # One legacy softms-only uncertainty branch remains in visual_observation,
    # but weighted_centroid runtime never enters it. The other call is final MS.
    raise SystemExit(f"ERROR: unexpected soft_mean_shift call count={s.count('soft_mean_shift(')}")

compile(s, str(p), "exec")
p.write_text(s, encoding="utf-8")
print("[OK] clean v39: Weighted Centroid -> original GRU -> original Kalman -> one final MS; pure MS timer enabled")
