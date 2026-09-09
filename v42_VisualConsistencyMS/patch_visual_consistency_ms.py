#!/usr/bin/env python3
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_visual_consistency_ms.py <robust_tracker.py>")

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

# Applied AFTER v39_DirectFinalMS + v41_KalmanAnchoredMS.
# v42 fixes two design issues in v41:
#   1) confidence must be computed from the RAW UAV-SAT posterior, before any KF prior;
#   2) a sharp response is not sufficient in repetitive fields, so the top-1 / posterior-centroid
#      agreement and posterior spatial spread are used as visual self-consistency terms.
# The Kalman spatial prior is also changed from a fixed isotropic sigma to an anisotropic
# route-coordinate prior derived from the current Kalman covariance.
# Final XY remains the literal Soft MeanShift output.

old_metrics = '''    ms2_anchor_masses = []
    ms2_visual_confidences = []
'''
new_metrics = '''    ms2_anchor_masses = []
    ms2_visual_confidences = []
    ms2_visual_agreements = []
    ms2_visual_spreads = []
    ms2_visual_reliabilities = []
    ms2_kf_sigma_s_values = []
    ms2_kf_sigma_e_values = []
'''
if s.count(old_metrics) != 1:
    raise SystemExit(f"ERROR: v41 metric block count={s.count(old_metrics)}")
s = s.replace(old_metrics, new_metrics, 1)

start_marker = '''        sigma_kalman = max(
'''
end_marker = '''        # Build one joint KDE distribution over 36 SAT locations + the exact
'''
start = s.find(start_marker)
end = s.find(end_marker, start)
if start < 0 or end < 0 or end <= start:
    raise SystemExit("ERROR: could not locate v41 confidence/prior block")

new_block = '''        # -------------------------------------------------------------
        # v42: RAW visual confidence / self-consistency.
        # IMPORTANT: no Kalman spatial prior is allowed to influence these
        # statistics. Otherwise the KF prior itself can make a weak visual
        # response look artificially sharp (the v41 failure mode).
        # -------------------------------------------------------------
        raw_visual_probs = F.softmax(visual_log_probability, dim=1)
        eps = 1e-12
        n_visual = max(int(raw_visual_probs.shape[1]), 2)

        raw_entropy = -(
            raw_visual_probs * torch.log(raw_visual_probs.clamp_min(eps))
        ).sum(dim=1)
        raw_entropy_norm = raw_entropy / float(np.log(n_visual))
        raw_top2 = torch.topk(raw_visual_probs, k=2, dim=1).values
        raw_margin_ratio = (
            (raw_top2[:, 0] - raw_top2[:, 1])
            / raw_top2[:, 0].clamp_min(eps)
        )
        raw_entropy_conf = (1.0 - raw_entropy_norm).clamp(0.0, 1.0)
        visual_confidence_t = (
            0.60 * raw_entropy_conf
            + 0.40 * raw_margin_ratio.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)

        # Spatial self-consistency of the raw visual response.
        # A single sharp but isolated wrong peak is not automatically trusted.
        # We require the top-1 location to agree with the posterior centroid,
        # while also preferring a spatially compact posterior.
        raw_centroid_t = (
            raw_visual_probs.unsqueeze(2) * ms2_candidate.centers
        ).sum(dim=1)
        raw_top1_index_t = raw_visual_probs.argmax(dim=1)
        raw_batch_index_t = torch.arange(
            raw_visual_probs.shape[0], device=raw_visual_probs.device
        )
        raw_top1_xy_t = ms2_candidate.centers[
            raw_batch_index_t, raw_top1_index_t
        ]
        visual_agreement_m_t = torch.linalg.norm(
            raw_top1_xy_t - raw_centroid_t, dim=1
        )
        raw_center_delta_t = ms2_candidate.centers - raw_centroid_t[:, None, :]
        visual_spread_m_t = torch.sqrt(
            (
                raw_visual_probs
                * raw_center_delta_t.square().sum(dim=2)
            ).sum(dim=1).clamp_min(0.0)
        )

        agreement_sigma_m = max(
            float(__import__("os").environ.get("MS2_AGREEMENT_SIGMA_M", "5.0")),
            1e-3,
        )
        spread_sigma_m = max(
            float(__import__("os").environ.get("MS2_SPREAD_SIGMA_M", "9.0")),
            1e-3,
        )
        agreement_score_t = torch.exp(
            -0.5 * (visual_agreement_m_t / agreement_sigma_m).square()
        )
        spread_score_t = torch.exp(
            -0.5 * (visual_spread_m_t / spread_sigma_m).square()
        )
        visual_reliability_t = (
            visual_confidence_t * agreement_score_t * spread_score_t
        ).clamp(0.0, 1.0)

        # -------------------------------------------------------------
        # Kalman covariance prior.
        # Unlike v40/v41's fixed isotropic radius, use the actual uncertainty
        # along route-progress s and cross-track e. This lets MS2 move more in
        # a direction where KF is uncertain and suppresses motion in a direction
        # where KF is already confident.
        # -------------------------------------------------------------
        kf_sigma_floor_m = max(
            float(__import__("os").environ.get("MS2_KF_SIGMA_FLOOR_M", "2.0")),
            1e-3,
        )
        kf_sigma_ceil_m = max(
            float(__import__("os").environ.get("MS2_KF_SIGMA_CEIL_M", "8.0")),
            kf_sigma_floor_m,
        )
        kf_sigma_s = float(np.clip(
            np.sqrt(max(float(kf.P[0, 0]), 0.0)),
            kf_sigma_floor_m,
            kf_sigma_ceil_m,
        ))
        kf_sigma_e = float(np.clip(
            np.sqrt(max(float(kf.P[1, 1]), 0.0)),
            kf_sigma_floor_m,
            kf_sigma_ceil_m,
        ))
        route_unit_t = torch.tensor(
            route.smooth_route_unit(kalman_se[0]),
            dtype=ms2_candidate.centers.dtype,
            device=device,
        )
        route_cross_t = torch.tensor(
            route.smooth_route_cross(kalman_se[0]),
            dtype=ms2_candidate.centers.dtype,
            device=device,
        )
        candidate_delta_xy_t = ms2_candidate.centers - kalman_xy_t[:, None, :]
        candidate_delta_s_t = (
            candidate_delta_xy_t * route_unit_t[None, None, :]
        ).sum(dim=2)
        candidate_delta_e_t = (
            candidate_delta_xy_t * route_cross_t[None, None, :]
        ).sum(dim=2)
        mahalanobis2_t = (
            candidate_delta_s_t.square() / (kf_sigma_s ** 2)
            + candidate_delta_e_t.square() / (kf_sigma_e ** 2)
        )
        weight_kalman = float(
            __import__("os").environ.get("MS2_KF_PRIOR_WEIGHT", "0.75")
        )
        spatial_visual_logp = (
            visual_log_probability - 0.5 * weight_kalman * mahalanobis2_t
        )
        visual_probs = F.softmax(spatial_visual_logp, dim=1)

        # The exact Kalman point remains an explicit KDE mode. Its mass is
        # reduced only when RAW visual evidence is both confident and spatially
        # self-consistent. No hard gate, post-MS clipping, or reference prior.
        anchor_min = float(
            __import__("os").environ.get("MS2_ANCHOR_MIN_MASS", "0.25")
        )
        anchor_max = float(
            __import__("os").environ.get("MS2_ANCHOR_MAX_MASS", "0.97")
        )
        anchor_min = min(max(anchor_min, 0.05), 0.95)
        anchor_max = min(max(anchor_max, anchor_min), 0.995)
        anchor_gamma = max(
            float(__import__("os").environ.get("MS2_ANCHOR_CONF_GAMMA", "0.65")),
            1e-3,
        )
        anchor_mass_t = anchor_max - (
            anchor_max - anchor_min
        ) * visual_reliability_t.pow(anchor_gamma)
        anchor_mass_t = anchor_mass_t.clamp(anchor_min, anchor_max)

'''
s = s[:start] + new_block + s[end:]

old_diag = '''        ms2_anchor_mass = float(anchor_mass_t[0].item())
        ms2_visual_confidence = float(visual_confidence_t[0].item())
        ms2_anchor_masses.append(ms2_anchor_mass)
        ms2_visual_confidences.append(ms2_visual_confidence)
'''
new_diag = '''        ms2_anchor_mass = float(anchor_mass_t[0].item())
        ms2_visual_confidence = float(visual_confidence_t[0].item())
        ms2_visual_agreement_m = float(visual_agreement_m_t[0].item())
        ms2_visual_spread_m = float(visual_spread_m_t[0].item())
        ms2_visual_reliability = float(visual_reliability_t[0].item())
        ms2_anchor_masses.append(ms2_anchor_mass)
        ms2_visual_confidences.append(ms2_visual_confidence)
        ms2_visual_agreements.append(ms2_visual_agreement_m)
        ms2_visual_spreads.append(ms2_visual_spread_m)
        ms2_visual_reliabilities.append(ms2_visual_reliability)
        ms2_kf_sigma_s_values.append(kf_sigma_s)
        ms2_kf_sigma_e_values.append(kf_sigma_e)
'''
if s.count(old_diag) != 1:
    raise SystemExit(f"ERROR: v41 diagnostic block count={s.count(old_diag)}")
s = s.replace(old_diag, new_diag, 1)

old_csv = '''                "ms2_anchor_mass": float(ms2_anchor_mass),
                "ms2_visual_confidence": float(ms2_visual_confidence),
                "ms2_shift_from_kalman_m": float(ms2_shift_from_kalman_m),
'''
new_csv = '''                "ms2_anchor_mass": float(ms2_anchor_mass),
                "ms2_visual_confidence": float(ms2_visual_confidence),
                "ms2_visual_agreement_m": float(ms2_visual_agreement_m),
                "ms2_visual_spread_m": float(ms2_visual_spread_m),
                "ms2_visual_reliability": float(ms2_visual_reliability),
                "ms2_kf_sigma_s_m": float(kf_sigma_s),
                "ms2_kf_sigma_e_m": float(kf_sigma_e),
                "ms2_shift_from_kalman_m": float(ms2_shift_from_kalman_m),
'''
if s.count(old_csv) != 1:
    raise SystemExit(f"ERROR: v41 CSV diagnostic block count={s.count(old_csv)}")
s = s.replace(old_csv, new_csv, 1)

old_summary = '''    summary["MS2_MeanAnchorMass"] = float(np.mean(ms2_anchor_masses)) if ms2_anchor_masses else 0.0
    summary["MS2_MeanVisualConfidence"] = float(np.mean(ms2_visual_confidences)) if ms2_visual_confidences else 0.0
    summary["MS2_Definition"] = "Kalman-centered full 6x6 Soft MeanShift with 36 visual candidates plus an adaptive Kalman anchor mode; no predefined-reference prior; MS2 output is final"
'''
new_summary = '''    summary["MS2_MeanAnchorMass"] = float(np.mean(ms2_anchor_masses)) if ms2_anchor_masses else 0.0
    summary["MS2_MeanVisualConfidence"] = float(np.mean(ms2_visual_confidences)) if ms2_visual_confidences else 0.0
    summary["MS2_MeanVisualAgreement_m"] = float(np.mean(ms2_visual_agreements)) if ms2_visual_agreements else 0.0
    summary["MS2_MeanVisualSpread_m"] = float(np.mean(ms2_visual_spreads)) if ms2_visual_spreads else 0.0
    summary["MS2_MeanVisualReliability"] = float(np.mean(ms2_visual_reliabilities)) if ms2_visual_reliabilities else 0.0
    summary["MS2_MeanKFSigmaS_m"] = float(np.mean(ms2_kf_sigma_s_values)) if ms2_kf_sigma_s_values else 0.0
    summary["MS2_MeanKFSigmaE_m"] = float(np.mean(ms2_kf_sigma_e_values)) if ms2_kf_sigma_e_values else 0.0
    summary["MS2_Definition"] = "Kalman-centered full 6x6 Soft MeanShift with RAW-visual confidence/self-consistency gating and covariance-aware Kalman prior; no predefined-reference prior; MS2 output is final"
'''
if s.count(old_summary) != 1:
    raise SystemExit(f"ERROR: v41 summary block count={s.count(old_summary)}")
s = s.replace(old_summary, new_summary, 1)

# Sanity checks: v42 must not reintroduce the predefined frame reference into MS2.
for forbidden in [
    "frame_reference_xy_t",
    "d2_reference",
    "weight_reference",
    "sigma_reference",
]:
    if forbidden in s:
        raise SystemExit(f"ERROR: stale reference-prior token remains: {forbidden}")

compile(s, str(p), "exec")
p.write_text(s, encoding="utf-8")
print("[OK] patched to v42 raw-visual consistency + covariance-aware Kalman anchored MS2")
