#!/usr/bin/env python3
"""Paired G-vs-K evaluation for the restored original V36 pipeline.

This file deliberately does not change training or model semantics.  It loads the
same full ThreeFrameRouteStateGRU checkpoint and, for every B/C frame, records
three outputs from the SAME forward pass:

G          : SoftMS visual anchor + GRU correction = output.measurement_se
K_raw      : external Kalman posterior immediately after kf.update()
K_original : the original V36 reported final after cap_kalman_to_current_gt()

The K_raw row is essential: the historical V36 result applies a final GT-progress
cap after the Kalman update, so G versus K_original alone would conflate Kalman
fusion with that historical cap.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

import config
import robust_tracker as r
import robust_tracker_base as b
from visual_localizer import FrozenVisualLocalizer


def metric_summary(errors):
    x = np.asarray(errors, dtype=np.float64)
    if x.size == 0:
        raise RuntimeError("No evaluation errors were collected")
    p90 = float(np.quantile(x, 0.90))
    tail = x[x >= p90]
    return {
        "N": int(x.size),
        "MLE_m": float(np.mean(x)),
        "MedLE_m": float(np.median(x)),
        "P90_m": p90,
        "P95_m": float(np.quantile(x, 0.95)),
        "P99_m": float(np.quantile(x, 0.99)),
        "CVaR90_m": float(np.mean(tail)) if tail.size else p90,
        "LSR@3_pct": float(np.mean(x <= 3.0) * 100.0),
        "LSR@5_pct": float(np.mean(x <= 5.0) * 100.0),
        "LSR@10_pct": float(np.mean(x <= 10.0) * 100.0),
        "LSR@15_pct": float(np.mean(x <= 15.0) * 100.0),
        "LSR@20_pct": float(np.mean(x <= 20.0) * 100.0),
    }


def paired_summary(g_errors, kraw_errors, korig_errors):
    g = np.asarray(g_errors, dtype=np.float64)
    kr = np.asarray(kraw_errors, dtype=np.float64)
    ko = np.asarray(korig_errors, dtype=np.float64)
    if not (g.size == kr.size == ko.size and g.size > 0):
        raise RuntimeError("Paired error arrays are inconsistent")

    g_mle = float(g.mean())
    kr_mle = float(kr.mean())
    ko_mle = float(ko.mean())
    raw_gain = g_mle - kr_mle
    orig_gain = g_mle - ko_mle
    return {
        "G_MS_plus_GRU_measurement": metric_summary(g),
        "K_raw_after_kf_update_before_final_cap": metric_summary(kr),
        "K_original_reported_after_final_cap": metric_summary(ko),
        "paired_effect": {
            "KalmanRaw_MLE_gain_over_G_m": raw_gain,
            "KalmanRaw_relative_MLE_improvement_pct": float(raw_gain / max(g_mle, 1e-12) * 100.0),
            "KalmanRaw_frame_improvement_rate_pct": float(np.mean(kr < g) * 100.0),
            "KalmanRaw_frame_equal_rate_pct": float(np.mean(np.isclose(kr, g, atol=1e-9)) * 100.0),
            "KalmanOriginal_MLE_gain_over_G_m": orig_gain,
            "KalmanOriginal_relative_MLE_improvement_pct": float(orig_gain / max(g_mle, 1e-12) * 100.0),
            "KalmanOriginal_frame_improvement_rate_pct": float(np.mean(ko < g) * 100.0),
            "FinalCap_incremental_MLE_gain_over_KalmanRaw_m": float(kr_mle - ko_mle),
        },
    }


@torch.no_grad()
def run_route(route_name, visual, model, cache, route, device):
    model.eval()
    gt_state = r.build_gt_route_state(cache, route)
    kf = r.RouteKalman(0.0, 0.0)

    hidden = None
    previous_z = None
    previous2_z = None
    previous_measurement_se = None
    previous_velocity = torch.zeros(1, 2, device=device)
    previous_acceleration = torch.zeros(1, 2, device=device)
    previous_heading_state = torch.zeros(1, 2, device=device)
    previous_poly_step = torch.zeros(1, 2, device=device)
    previous_acq_confidence = float(config.ACQ_INITIAL_CONFIDENCE)

    rows = []
    g_errors = []
    kraw_errors = []
    korig_errors = []

    for index in range(len(cache)):
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()

        previous_output_se = kf.se().copy()
        feedback_se = previous_output_se.copy()
        feedback_xy = route.xy_from_se(*feedback_se)
        feedback_support = 0.0

        if index > 0:
            previous_output_se, feedback_se, feedback_xy, feedback_support = r.teacher_meanshift_feedback(
                visual, uav_clip, route, kf, previous_heading_state, device
            )
            previous_measurement_se = b.tensor2(feedback_se, device).detach()
            predicted_se = kf.predict(
                previous_velocity[0].cpu().numpy(),
                previous_acceleration[0].cpu().numpy(),
                route.total_length_m,
                max_progress_s=float(gt_state["se"][index, 0]),
                polynomial_step_se=previous_poly_step[0].cpu().numpy(),
                max_step_m=float(gt_state["gt_step_norm"][index]),
            )
        else:
            predicted_se = kf.se()

        predicted_se = b.cap_prediction_to_current_gt(kf, predicted_se, gt_state["se"][index])
        obs, prior_xy, jitter_xy, search_heading = r._frame_visual(
            model,
            visual,
            cache,
            route,
            gt_state,
            index,
            predicted_se,
            previous_z,
            previous2_z,
            hidden,
            previous_acq_confidence,
            previous_velocity,
            previous_heading_state,
            device,
            uav_clip,
        )
        output = b.model_forward(
            model,
            obs,
            previous_z,
            previous2_z,
            predicted_se,
            previous_measurement_se,
            previous_velocity,
            previous_acceleration,
            previous_heading_state,
            previous_poly_step,
            route,
            hidden,
            device,
        )

        # G: the exact visual measurement fed into the Kalman update.
        measurement_se = output.measurement_se[0].detach().cpu().numpy().astype(np.float64)
        measurement_xy = route.xy_from_se(float(measurement_se[0]), float(measurement_se[1]))
        reference_xy = cache.gt_xy[index].cpu().numpy().astype(np.float64)
        g_error = float(np.linalg.norm(measurement_xy - reference_xy))

        # K_raw: isolate the external Kalman update itself, before the historical
        # current-GT progress cap is applied.
        conf = b.visual_confidence_from_observation(obs)
        kalman_raw_se = kf.update(
            measurement_se,
            output.measurement_variance_se[0].detach().cpu().numpy(),
            route.total_length_m,
            acquisition_confidence=conf,
            max_progress_s=float(gt_state["se"][index, 0]),
            max_final_step_m=float(gt_state["gt_step_norm"][index]),
        ).copy()
        kalman_raw_xy = route.xy_from_se(float(kalman_raw_se[0]), float(kalman_raw_se[1]))
        kraw_error = float(np.linalg.norm(kalman_raw_xy - reference_xy))

        # K_original: preserve the exact historical V36 reported final path and
        # state hand-off for the following frame.
        final_se, _ = b.cap_kalman_to_current_gt(kf, kalman_raw_se, gt_state["se"][index])
        final_se = np.asarray(final_se, dtype=np.float64).copy()
        final_xy = route.xy_from_se(float(final_se[0]), float(final_se[1]))
        korig_error = float(np.linalg.norm(final_xy - reference_xy))

        g_errors.append(g_error)
        kraw_errors.append(kraw_error)
        korig_errors.append(korig_error)

        rows.append(
            {
                "frame_id": int(cache.frame_ids[index]),
                "image_path": cache.image_paths[index],
                "reference_x": float(reference_xy[0]),
                "reference_y": float(reference_xy[1]),
                "softms_x": float(obs.candidate.softms_xy[0, 0]),
                "softms_y": float(obs.candidate.softms_xy[0, 1]),
                "G_measurement_s": float(measurement_se[0]),
                "G_measurement_e": float(measurement_se[1]),
                "G_measurement_x": float(measurement_xy[0]),
                "G_measurement_y": float(measurement_xy[1]),
                "G_error_m": g_error,
                "measurement_var_s": float(output.measurement_variance_se[0, 0]),
                "measurement_var_e": float(output.measurement_variance_se[0, 1]),
                "K_raw_s": float(kalman_raw_se[0]),
                "K_raw_e": float(kalman_raw_se[1]),
                "K_raw_x": float(kalman_raw_xy[0]),
                "K_raw_y": float(kalman_raw_xy[1]),
                "K_raw_error_m": kraw_error,
                "K_original_s": float(final_se[0]),
                "K_original_e": float(final_se[1]),
                "K_original_x": float(final_xy[0]),
                "K_original_y": float(final_xy[1]),
                "K_original_error_m": korig_error,
                "K_raw_gain_vs_G_m": float(g_error - kraw_error),
                "K_original_gain_vs_G_m": float(g_error - korig_error),
                "previous_kalman_output_s": float(previous_output_se[0]),
                "previous_kalman_output_e": float(previous_output_se[1]),
                "teacher_feedback_ms_s": float(feedback_se[0]),
                "teacher_feedback_ms_e": float(feedback_se[1]),
                "teacher_feedback_ms_x": float(feedback_xy[0]),
                "teacher_feedback_ms_y": float(feedback_xy[1]),
                "teacher_feedback_support": float(feedback_support),
                "predicted_progress_s": float(predicted_se[0]),
                "predicted_cross_e": float(predicted_se[1]),
                "prior_center_x": float(prior_xy[0]),
                "prior_center_y": float(prior_xy[1]),
                "prior_jitter_x": float(jitter_xy[0]),
                "prior_jitter_y": float(jitter_xy[1]),
                "search_heading_deg": float(math.degrees(search_heading)),
            }
        )

        previous2_z, previous_z = previous_z, obs.candidate.z_uav.detach()
        previous_velocity, previous_acceleration, previous_poly_step = b.stabilize_motion_state(
            previous_velocity,
            previous_acceleration,
            previous_poly_step,
            output.velocity_se,
            output.acceleration_se,
            output.next_step_se,
        )
        previous_heading_state = b.stabilize_heading_state(
            previous_heading_state,
            output.heading_residual_rad,
            output.turn_rate_rad,
        )
        hidden = output.hidden
        previous_acq_confidence = float(conf)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.OUTPUT_DIR / f"{route_name}_GvsK_frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = paired_summary(g_errors, kraw_errors, korig_errors)
    summary.update(
        {
            "Route": route_name,
            "CSV": str(csv_path),
            "ComparisonDefinition": {
                "G": "output.measurement_se = SoftMS visual anchor + GRU correction head",
                "K_raw": "kf.update() posterior before cap_kalman_to_current_gt()",
                "K_original": "historical V36 final after cap_kalman_to_current_gt()",
            },
        }
    )
    (config.OUTPUT_DIR / f"{route_name}_GvsK_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary, g_errors, kraw_errors, korig_errors


def write_compact_csv(all_summary):
    fields = [
        "Route",
        "G_MLE_m",
        "G_MedLE_m",
        "G_P90_m",
        "G_LSR@3_pct",
        "G_LSR@5_pct",
        "G_LSR@10_pct",
        "Kraw_MLE_m",
        "Kraw_MedLE_m",
        "Kraw_P90_m",
        "Kraw_LSR@3_pct",
        "Kraw_LSR@5_pct",
        "Kraw_LSR@10_pct",
        "Koriginal_MLE_m",
        "Koriginal_MedLE_m",
        "Koriginal_P90_m",
        "Koriginal_LSR@3_pct",
        "Koriginal_LSR@5_pct",
        "Koriginal_LSR@10_pct",
        "Kraw_MLE_gain_vs_G_m",
        "Kraw_relative_improvement_pct",
        "Koriginal_MLE_gain_vs_G_m",
        "Koriginal_relative_improvement_pct",
    ]
    path = config.OUTPUT_DIR / "GvsK_comparison_table.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for route_name in ("route_B", "route_C", "B+C"):
            s = all_summary[route_name]
            g = s["G_MS_plus_GRU_measurement"]
            kr = s["K_raw_after_kf_update_before_final_cap"]
            ko = s["K_original_reported_after_final_cap"]
            e = s["paired_effect"]
            w.writerow(
                {
                    "Route": route_name,
                    "G_MLE_m": g["MLE_m"],
                    "G_MedLE_m": g["MedLE_m"],
                    "G_P90_m": g["P90_m"],
                    "G_LSR@3_pct": g["LSR@3_pct"],
                    "G_LSR@5_pct": g["LSR@5_pct"],
                    "G_LSR@10_pct": g["LSR@10_pct"],
                    "Kraw_MLE_m": kr["MLE_m"],
                    "Kraw_MedLE_m": kr["MedLE_m"],
                    "Kraw_P90_m": kr["P90_m"],
                    "Kraw_LSR@3_pct": kr["LSR@3_pct"],
                    "Kraw_LSR@5_pct": kr["LSR@5_pct"],
                    "Kraw_LSR@10_pct": kr["LSR@10_pct"],
                    "Koriginal_MLE_m": ko["MLE_m"],
                    "Koriginal_MedLE_m": ko["MedLE_m"],
                    "Koriginal_P90_m": ko["P90_m"],
                    "Koriginal_LSR@3_pct": ko["LSR@3_pct"],
                    "Koriginal_LSR@5_pct": ko["LSR@5_pct"],
                    "Koriginal_LSR@10_pct": ko["LSR@10_pct"],
                    "Kraw_MLE_gain_vs_G_m": e["KalmanRaw_MLE_gain_over_G_m"],
                    "Kraw_relative_improvement_pct": e["KalmanRaw_relative_MLE_improvement_pct"],
                    "Koriginal_MLE_gain_vs_G_m": e["KalmanOriginal_MLE_gain_over_G_m"],
                    "Koriginal_relative_improvement_pct": e["KalmanOriginal_relative_MLE_improvement_pct"],
                }
            )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jitter-m", type=float, default=8.0)
    args = parser.parse_args()

    config.LOCAL_PRIOR_JITTER_M = float(args.jitter_m)
    config.CONTROLLED_GT_PRIOR_JITTER_M = float(args.jitter_m)
    r.set_seed(config.SEED)
    device = r.resolve_device()

    visual = FrozenVisualLocalizer(device)
    model = r.load_temporal_model(device)

    all_summary = {
        "Provenance": {
            "restored_from_commit": "e732045cacc6d2bff152663e8b5966ee1b49b98b",
            "original_architecture_parent": "fe3ff2329fad0d43bfbe0a8650bbb675fd19d339",
            "architecture": str(config.ARCHITECTURE_NAME),
            "jitter_m": float(args.jitter_m),
            "note": "All G/K outputs come from the same trained full V36 model and same frame forward pass.",
        }
    }
    combined_g = []
    combined_kr = []
    combined_ko = []

    for route_name in ("route_B", "route_C"):
        i = config.ROUTE_NAMES.index(route_name)
        cache = r.build_route_cache(route_name, config.ROUTE_ROOTS[i], visual, device)
        route = r.WaypointRoute(r.load_waypoint_xy(route_name, visual.origin_lat, visual.origin_lon))
        route_summary, g, kr, ko = run_route(route_name, visual, model, cache, route, device)
        all_summary[route_name] = route_summary
        combined_g.extend(g)
        combined_kr.extend(kr)
        combined_ko.extend(ko)

    all_summary["B+C"] = paired_summary(combined_g, combined_kr, combined_ko)
    compact_csv = write_compact_csv(all_summary)
    all_summary["compact_table_csv"] = str(compact_csv)

    summary_path = config.OUTPUT_DIR / "GvsK_summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    c = all_summary["B+C"]
    g = c["G_MS_plus_GRU_measurement"]
    kr = c["K_raw_after_kf_update_before_final_cap"]
    ko = c["K_original_reported_after_final_cap"]
    e = c["paired_effect"]
    print("=" * 96)
    print("Original V36 G vs K, Routes B+C")
    print(f"G  (MS+GRU measurement): MLE={g['MLE_m']:.3f}  Median={g['MedLE_m']:.3f}  P90={g['P90_m']:.3f}")
    print(f"Kraw (Kalman only)     : MLE={kr['MLE_m']:.3f}  Median={kr['MedLE_m']:.3f}  P90={kr['P90_m']:.3f}")
    print(f"Korig (historical final): MLE={ko['MLE_m']:.3f}  Median={ko['MedLE_m']:.3f}  P90={ko['P90_m']:.3f}")
    print(f"Kalman-only MLE gain vs G = {e['KalmanRaw_MLE_gain_over_G_m']:.3f} m ({e['KalmanRaw_relative_MLE_improvement_pct']:.2f}%)")
    print(f"Historical-final gain vs G = {e['KalmanOriginal_MLE_gain_over_G_m']:.3f} m ({e['KalmanOriginal_relative_MLE_improvement_pct']:.2f}%)")
    print(f"summary={summary_path}")
    print(f"table={compact_csv}")
    print("=" * 96)


if __name__ == "__main__":
    main()
