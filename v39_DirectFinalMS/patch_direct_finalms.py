#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 4:
    raise SystemExit("usage: patch_direct_finalms.py <robust_tracker.py> <visual_localizer.py> <config.py>")
tracker_path, visual_path, config_path = map(Path, sys.argv[1:])

def rep(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"ERROR: {label} count={n}")
    return text.replace(old, new, 1)

# ---- visual_localizer.py -----------------------------------------------------
v = visual_path.read_text(encoding="utf-8")
v = rep(v,
'''        self.pixel_index = build_pixel_index(self.gallery["pixel"])
''',
'''        self.pixel_index = build_pixel_index(self.gallery["pixel"])
        self.gallery_xy_cpu = self.gallery["xy"].detach().cpu()
        self.gallery_pixel_cpu = self.gallery["pixel"].detach().cpu()
        projected = []
        for start in range(0, int(self.gallery["clip_feat"].shape[0]), 2048):
            end = min(start + 2048, int(self.gallery["clip_feat"].shape[0]))
            projected.append(self.model.encode_sat_from_clip(
                self.gallery["clip_feat"][start:end], self.gallery["xy"][start:end]
            ))
        self.gallery["z_sat"] = torch.cat(projected, dim=0)
''', "visual init")

v = rep(v,
'''    @torch.no_grad()
    def candidate_batch(self, uav_clip, center_xy, grid_size=None):
''',
'''    @torch.no_grad()
    def regular_grid_indices_cached(self, prior_xy, grid_size):
        return regular_grid_indices(
            self.gallery_xy_cpu, self.gallery_pixel_cpu, self.pixel_index,
            prior_xy, int(grid_size), config.SAT_STRIDE, self.device,
        )

    @torch.no_grad()
    def score_candidate_indices(self, uav_clip, indices, z_uav=None):
        centers = self.gallery["xy"][indices]
        if z_uav is None:
            z_uav = self.model.encode_uav_from_clip(uav_clip)
        z_sat = self.gallery["z_sat"][indices]
        raw_logits = self.model.logit_scale.exp().clamp(max=100.0) * (
            z_uav[:, None] * z_sat
        ).sum(dim=2)
        raw_prob = torch.softmax(raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1)
        raw_index = raw_logits.argmax(dim=1)
        raw_top1_xy = centers[torch.arange(centers.shape[0], device=self.device), raw_index]
        return centers, z_uav, z_sat, raw_logits, raw_prob, raw_top1_xy

    @torch.no_grad()
    def candidate_batch(self, uav_clip, center_xy, grid_size=None):
''', "visual helper methods")

v = rep(v,
'''        indices = regular_grid_indices(
            self.gallery["xy"],
            self.gallery["pixel"],
            self.pixel_index,
            center_xy,
            grid_size,
            config.SAT_STRIDE,
            self.device,
        )
        centers = self.gallery["xy"][indices]
        satellite_clip = self.gallery["clip_feat"][indices]

        z_uav = self.model.encode_uav_from_clip(uav_clip)
        z_sat = self.model.encode_sat_from_clip(
            satellite_clip.reshape(-1, satellite_clip.shape[-1]),
            centers.reshape(-1, 2),
        ).reshape(centers.shape[0], centers.shape[1], -1)

        raw_logits = self.model.logit_scale.exp().clamp(max=100.0) * (
            z_uav[:, None] * z_sat
        ).sum(dim=2)
        raw_prob = torch.softmax(
            raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1
        )
        raw_index = raw_logits.argmax(dim=1)
        raw_top1_xy = centers[
            torch.arange(centers.shape[0], device=self.device), raw_index
        ]
''',
'''        indices = self.regular_grid_indices_cached(center_xy, grid_size)
        centers, z_uav, z_sat, raw_logits, raw_prob, raw_top1_xy = self.score_candidate_indices(
            uav_clip, indices
        )
''', "candidate scoring")

# The selected architecture never performs MeanShift inside the visual front-end,
# including the generic full-grid candidate path. Keep CandidateBatch field names
# only for compatibility with the legacy tracker interfaces.
v = rep(v,
'''        softms_xy, softms_support, _, _, mode_weights, _ = soft_mean_shift(
            raw_logits,
            centers,
            config.MEANSHIFT_SCORE_TAU,
            config.MEANSHIFT_BANDWIDTH_M,
            config.MEANSHIFT_ITERATIONS,
            config.MEANSHIFT_MODE_BETA,
        )

        return CandidateBatch(
            indices=indices,
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
''',
'''        weighted_xy = (raw_prob.unsqueeze(-1) * centers).sum(dim=1)
        posterior_support = raw_prob.max(dim=1).values
        return CandidateBatch(
            indices=indices,
            centers=centers,
            z_uav=z_uav,
            z_sat=z_sat,
            raw_logits=raw_logits,
            raw_prob=raw_prob,
            raw_top1_xy=raw_top1_xy,
            softms_xy=weighted_xy,
            softms_support=posterior_support,
            softms_mode_count=torch.ones(raw_prob.shape[0], dtype=torch.long, device=raw_prob.device),
        )
''', "generic candidate decoder")
compile(v, str(visual_path), "exec")
visual_path.write_text(v, encoding="utf-8")

# ---- robust_tracker.py -------------------------------------------------------
s = tracker_path.read_text(encoding="utf-8")
s = rep(s,
'''    full_indices = regular_grid_indices(
        visual.gallery["xy"],
        visual.gallery["pixel"],
        visual.pixel_index,
        grid_center_xy,
        grid_size,
        config.SAT_STRIDE,
        visual.device,
    )
''',
'''    full_indices = visual.regular_grid_indices_cached(grid_center_xy, grid_size)
''', "forward grid")

s = rep(s,
'''    centers = visual.gallery["xy"][selected_indices]
    satellite_clip = visual.gallery["clip_feat"][selected_indices]
    z_uav = visual.model.encode_uav_from_clip(uav_clip)
    z_sat = visual.model.encode_sat_from_clip(
        satellite_clip.reshape(-1, satellite_clip.shape[-1]),
        centers.reshape(-1, 2),
    ).reshape(centers.shape[0], centers.shape[1], -1)
    raw_logits = visual.model.logit_scale.exp().clamp(max=100.0) * (
        z_uav[:, None] * z_sat
    ).sum(dim=2)
    raw_prob = torch.softmax(
        raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1
    )
    raw_index = raw_logits.argmax(dim=1)
    raw_top1_xy = centers[
        torch.arange(centers.shape[0], device=visual.device), raw_index
    ]
    softms_xy, softms_support, _, _, mode_weights, _ = soft_mean_shift(
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
''',
'''    centers, z_uav, z_sat, raw_logits, raw_prob, raw_top1_xy = visual.score_candidate_indices(
        uav_clip, selected_indices
    )
    weighted_xy = (raw_prob.unsqueeze(-1) * centers).sum(dim=1)
    support = raw_prob.max(dim=1).values
    return CandidateBatch(
        indices=selected_indices, centers=centers, z_uav=z_uav, z_sat=z_sat,
        raw_logits=raw_logits, raw_prob=raw_prob, raw_top1_xy=raw_top1_xy,
        softms_xy=weighted_xy, softms_support=support,
        softms_mode_count=torch.ones(raw_prob.shape[0], dtype=torch.long, device=raw_prob.device),
    )
''', "forward decoder")

s = rep(s,
'''    # Anchor ablation: the default is V36 SoftMS; weighted centroid uses the
    # exact same local posterior and candidates without mean-shift iterations.
    if str(getattr(config, "EXPERIMENT_ANCHOR", "softms")) == "weighted_centroid":
        anchor_xy_all = (posterior.unsqueeze(-1) * candidate.centers).sum(dim=1)
    else:
        anchor_xy_all = candidate.softms_xy
''',
'''    if str(getattr(config, "EXPERIMENT_ANCHOR", "weighted_centroid")) != "weighted_centroid":
        raise RuntimeError("selected architecture requires weighted_centroid front-end")
    anchor_xy_all = (posterior.unsqueeze(-1) * candidate.centers).sum(dim=1)
''', "weighted anchor")

s = rep(s,
'''    # The visual anchor is the density-weighted average of the locations after
    # Soft Mean Shift converges.  Measure uncertainty in that same mode space:
    # spread between converged modes, not spread between the original patch
    # centres.  Thus adjacent patches that converge to one visual mode do not
    # falsely inflate measurement uncertainty.
    if str(getattr(config, "EXPERIMENT_ANCHOR", "softms")) == "softms":
        _, _, softms_modes_all, _, softms_mode_weights_all, _ = soft_mean_shift(
            candidate.raw_logits,
            candidate.centers,
            config.MEANSHIFT_SCORE_TAU,
            config.MEANSHIFT_BANDWIDTH_M,
            config.MEANSHIFT_ITERATIONS,
            config.MEANSHIFT_MODE_BETA,
        )

''',
'''    # Uncertainty: posterior-weighted candidate spread around the same centroid.
''', "front uncertainty MS")

s = rep(s,
'''        # softms_modes_all[h] contains one converged mode for every initial
        # patch seed.  Seeds converging to the same mode have negligible
        # relative displacement, while separated modes retain their weighted
        # between-mode uncertainty.
        if str(getattr(config, "EXPERIMENT_ANCHOR", "softms")) == "softms":
            variance_points = softms_modes_all[h]
            variance_weights = softms_mode_weights_all[h]
        else:
            variance_points = candidate.centers[h]
            variance_weights = posterior[h]
''',
'''        variance_points = candidate.centers[h]
        variance_weights = posterior[h]
''', "uncertainty branch")

s = s.replace('softms_support=candidate.softms_support,\n        hidden=hidden,',
              'softms_support=posterior.max(dim=1).values,\n        hidden=hidden,', 1)

s = rep(s,
'''    kf1_errors = []
    kf2_errors = []
    ms2_shifts_from_kf2 = []
''',
'''    kalman_errors = []
    ms_shifts_from_kalman = []
    ms_latency_rows_ms = []
''', "metrics")

start = s.find('''        # =============================================================\n        # Required final architecture:\n''')
end = s.find('''        if prepared_uav is not None:\n''', start)
if start < 0 or end <= start:
    raise SystemExit("ERROR: final block markers not found")

final_block = '''        # =============================================================
        # Weighted Centroid -> GRU -> Kalman -> ONE final MeanShift -> XY
        # =============================================================
        kalman_se = np.asarray(final_se, dtype=np.float64).copy()
        kalman_xy = route.xy_from_se(kalman_se[0], kalman_se[1])
        _, reference_xy, _ = local_search_reference_se(
            cache, route, gt_state, index, predicted_se=kalman_se
        )
        reference_xy = np.asarray(reference_xy, dtype=np.float64)
        reference_xy_t = torch.tensor(reference_xy[None, :], dtype=torch.float32, device=device)
        preferred_leg = route.frame_from_se(kalman_se[0], kalman_se[1]).leg_index
        env = __import__("os").environ
        ms_enabled = env.get("MS_ENABLED", "1").strip().lower() not in {"0","false","no","off"}
        ms_grid_size = int(env.get("MS_GRID_SIZE", "5"))
        measure_ms = env.get("MS_MEASURE_LATENCY", "0").strip().lower() in {"1","true","yes","on"}
        kalman_xy_t = torch.tensor(kalman_xy[None, :], dtype=torch.float32, device=device)

        if ms_enabled:
            if measure_ms and device.type == "cuda": torch.cuda.synchronize(device)
            ms_t0 = time.perf_counter() if measure_ms else None
            lattice_d2 = (visual.gallery["xy"] - kalman_xy_t).square().sum(dim=1)
            ms_lattice_index = int(lattice_d2.argmin().item())
            ms_lattice_xy_t = visual.gallery["xy"][ms_lattice_index:ms_lattice_index+1]
            ms_indices = visual.regular_grid_indices_cached(ms_lattice_xy_t, ms_grid_size)
            ms_centers, _, _, ms_raw_logits, _, _ = visual.score_candidate_indices(
                uav_clip, ms_indices, z_uav=obs.candidate.z_uav
            )
            tau = float(config.MEANSHIFT_SCORE_TAU)
            logp = F.log_softmax(ms_raw_logits / max(tau, 1e-6), dim=1)
            d2_k = (ms_centers - kalman_xy_t[:, None, :]).square().sum(dim=2)
            d2_r = (ms_centers - reference_xy_t[:, None, :]).square().sum(dim=2)
            sk = max(float(env.get("MS_KF_SIGMA_M", "4.0")), 1e-3)
            sr = max(float(env.get("MS_REFERENCE_SIGMA_M", "4.0")), 1e-3)
            wk = float(env.get("MS_KF_PRIOR_WEIGHT", "1.50"))
            wr = float(env.get("MS_REFERENCE_PRIOR_WEIGHT", "2.50"))
            final_logits = tau * (logp - wk*d2_k/(2*sk**2) - wr*d2_r/(2*sr**2))
            ms_xy_t, ms_support_t, _, _, ms_weights_t, _ = soft_mean_shift(
                final_logits, ms_centers, tau, float(env.get("MS_BANDWIDTH_M", "7.0")),
                config.MEANSHIFT_ITERATIONS, config.MEANSHIFT_MODE_BETA,
            )
            ms_xy = ms_xy_t[0].detach().cpu().numpy().astype(np.float64)
            ms_support = float(ms_support_t[0].item())
            ms_mode_count = int((ms_weights_t[0] > 0).sum().item())
            ms_s, ms_e, _ = route.project_xy_local(ms_xy, preferred_leg)
            final_se = np.asarray([ms_s, ms_e], dtype=np.float64)
            final_xy = ms_xy.copy()
            if measure_ms:
                if device.type == "cuda": torch.cuda.synchronize(device)
                ms_latency_ms = (time.perf_counter() - ms_t0) * 1000.0
                ms_latency_rows_ms.append(float(ms_latency_ms))
            else:
                ms_latency_ms = 0.0
        else:
            ms_lattice_index = -1
            ms_lattice_xy_t = kalman_xy_t
            ms_xy = kalman_xy.copy(); ms_support = 0.0; ms_mode_count = 0; ms_latency_ms = 0.0
            final_se = kalman_se.copy(); final_xy = kalman_xy.copy()

        reference_metric_xy = cache.gt_xy[index].cpu().numpy().astype(np.float64)
        kalman_errors.append(float(np.linalg.norm(kalman_xy - reference_metric_xy)))
        ms_shift_from_kalman_m = float(np.linalg.norm(final_xy - kalman_xy))
        ms_shifts_from_kalman.append(ms_shift_from_kalman_m)
'''
s = s[:start] + final_block + s[end:]

csv_start = s.find('''                "kf2_ms2_enabled": 1,\n''')
csv_end_token = '''                "ms2_shift_from_kf2_m": float(ms2_shift_from_kf2_m),\n'''
csv_end = s.find(csv_end_token, csv_start)
if csv_start < 0 or csv_end < 0:
    raise SystemExit("ERROR: CSV legacy block not found")
csv_end += len(csv_end_token)
s = s[:csv_start] + '''                "direct_kalman_ms_enabled": int(ms_enabled),
                "ms_grid_size": int(ms_grid_size),
                "kalman_x": float(kalman_xy[0]), "kalman_y": float(kalman_xy[1]),
                "reference_point_x": float(reference_xy[0]), "reference_point_y": float(reference_xy[1]),
                "ms_lattice_index": int(ms_lattice_index),
                "ms_lattice_x": float(ms_lattice_xy_t[0,0].item()), "ms_lattice_y": float(ms_lattice_xy_t[0,1].item()),
                "ms_x": float(ms_xy[0]), "ms_y": float(ms_xy[1]),
                "ms_support": float(ms_support), "ms_mode_count": int(ms_mode_count),
                "ms_shift_from_kalman_m": float(ms_shift_from_kalman_m),
                "ms_latency_ms": float(ms_latency_ms),
''' + s[csv_end:]

s = rep(s,
'''    summary["KF1_MAE_m"] = float(np.mean(kf1_errors)) if kf1_errors else 0.0
    summary["KF2_MAE_m"] = float(np.mean(kf2_errors)) if kf2_errors else 0.0
    summary["MS2_MeanShiftFromKF2_m"] = float(np.mean(ms2_shifts_from_kf2)) if ms2_shifts_from_kf2 else 0.0
    summary["MS2_MaxShiftFromKF2_m"] = float(np.max(ms2_shifts_from_kf2)) if ms2_shifts_from_kf2 else 0.0
    summary["KF2_Definition"] = "temporary current-frame second constrained Kalman update using predefined frame reference measurement; KF1 remains persistent state"
    summary["MS2_Definition"] = "full 6x6 Soft MeanShift with visual likelihood + KF2 spatial prior + frame-reference spatial prior; MS2 is the final output"
''',
'''    summary["Kalman_MAE_m"] = float(np.mean(kalman_errors)) if kalman_errors else 0.0
    summary["MS_MeanShiftFromKalman_m"] = float(np.mean(ms_shifts_from_kalman)) if ms_shifts_from_kalman else 0.0
    summary["MS_MaxShiftFromKalman_m"] = float(np.max(ms_shifts_from_kalman)) if ms_shifts_from_kalman else 0.0
    summary["VisualObservationDecoder"] = "posterior weighted centroid"
    summary["VisualObservationUncertainty"] = "posterior-weighted spatial variance projected to route parallel/cross coordinates"
    _on = str(__import__("os").environ.get("MS_ENABLED", "1")).lower() not in {"0","false","no","off"}
    summary["OnlineMeanShiftCount"] = 1 if _on else 0
    summary["MS_Enabled"] = bool(_on)
    summary["MS_GridSize"] = int(__import__("os").environ.get("MS_GRID_SIZE", "5"))
    _warm = int(__import__("os").environ.get("MS_LATENCY_WARMUP", "30"))
    _lat = ms_latency_rows_ms[_warm:] if len(ms_latency_rows_ms) > _warm else ms_latency_rows_ms
    summary["MS_LatencyMean_ms"] = float(np.mean(_lat)) if _lat else 0.0
    summary["MS_LatencyP90_ms"] = float(np.quantile(_lat, 0.90)) if _lat else 0.0
    summary["MS_ThroughputFPS"] = 1000.0 / summary["MS_LatencyMean_ms"] if summary["MS_LatencyMean_ms"] > 0 else 0.0
    summary["MS_LatencyWarmupFrames"] = int(_warm)
    summary["MS_LatencyDefinition"] = "Kalman posterior available -> candidate indexing/scoring -> one final MeanShift -> final XY"
    summary["MS_Definition"] = "single final local Soft MeanShift after Kalman"
    summary["StaticSatelliteProjectionCache"] = True
    summary["FinalMSReusesFrontUAVEmbedding"] = True
''', "summary")

for token in ["kf2_errors", "ms2_shifts_from_kf2", "kf2_ms2_enabled", "ms2_shift_from_kf2_m", "MS2_", "softms_modes_all", "softms_mode_weights_all"]:
    if token in s:
        raise SystemExit(f"ERROR: stale token remains: {token}")
compile(s, str(tracker_path), "exec")
tracker_path.write_text(s, encoding="utf-8")

# ---- config.py naming only ---------------------------------------------------
c = config_path.read_text(encoding="utf-8")
c = c.replace("scheduled_route_reference_forward3x6_SoftMS_3frame_GRU_", "scheduled_route_reference_forward3x6_WeightedCentroid_3frame_GRU_")
c = c.replace("route_reference_motion_prior_forward3x6_SoftMS_3frame_GRU_", "route_reference_motion_prior_forward3x6_WeightedCentroid_3frame_GRU_")
compile(c, str(config_path), "exec")
config_path.write_text(c, encoding="utf-8")
print("[OK] weighted-centroid front-end + single final MS patch applied")
