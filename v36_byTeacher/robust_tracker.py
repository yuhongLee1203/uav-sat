"""v36_byTeacher entry point.

Teacher feedback rule
---------------------
The Kalman posterior of frame t is still the reported output X_t. When image
I_{t+1} arrives, the previous output is localized again by Soft Mean-Shift
around X_t:

    X_t(output) = Kalman(GRU(MeanShift_t))
    X_t^MS      = MeanShift(I_{t+1}, center=X_t(output))
    X_{t+1}     = Kalman(GRU(previous_position=X_t^MS))

X_t^MS replaces only the GRU previous-position / previous-measurement input.
The external Kalman keeps its own posterior X_t and covariance for prediction.
The normal current-frame visual measurement is computed with the same
controlled reference-point local prior as v36.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

import config
import robust_tracker_base as b
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only
from visual_model import ThreeFrameRouteStateGRU

ARCHITECTURE_NAME = str(config.ARCHITECTURE_NAME)

# Re-export the original v36 public helpers/classes for compatibility.
WaypointRoute = b.WaypointRoute
RouteKalman = b.RouteKalman
RouteCache = b.RouteCache
RouteFrame = b.RouteFrame
VisualObservation = b.VisualObservation
build_route_cache = b.build_route_cache
build_gt_route_state = b.build_gt_route_state
load_waypoint_xy = b.load_waypoint_xy
resolve_device = b.resolve_device
set_seed = b.set_seed
metric_summary = b.metric_summary


def teacher_meanshift_feedback(visual, uav_clip, route, kf, previous_heading_state, device):
    """Re-localize the previous Kalman output with the newly arrived image.

    The returned SoftMS position is used as the GRU previous-position input only.
    It must not overwrite the external Kalman's posterior state.
    """
    previous_output_se = kf.se().copy()
    previous_output_xy = route.xy_from_se(previous_output_se[0], previous_output_se[1])
    center_xy = torch.tensor(previous_output_xy, dtype=torch.float32, device=device).reshape(1, 2)
    heading_rad = b.wrap_angle_rad(
        route.route_heading_rad(float(previous_output_se[0]))
        + float(previous_heading_state[0, 0].item())
    )
    candidate = b.forward_3x6_candidate_batch(
        visual=visual,
        uav_clip=uav_clip,
        center_xy=center_xy,
        heading_rad=heading_rad,
        grid_size=int(config.ACQ_LOCAL_GRID_SIZE),
    )
    feedback_xy = candidate.softms_xy[0].detach().cpu().numpy().astype(np.float64)
    preferred_leg = route.leg_for_s(float(previous_output_se[0]))
    feedback_s, feedback_e, _ = route.project_xy_local(feedback_xy, preferred_leg)
    feedback_se = np.asarray([feedback_s, feedback_e], dtype=np.float64)
    feedback_se[0] = float(np.clip(feedback_se[0], 0.0, route.total_length_m))
    feedback_se[1] = float(np.clip(
        feedback_se[1], -float(config.MAX_FINAL_CROSS_TRACK_M), float(config.MAX_FINAL_CROSS_TRACK_M)
    ))

    # IMPORTANT: preserve kf.x / kf.P. MeanShift feedback is a GRU input only.
    return previous_output_se, feedback_se, feedback_xy, float(candidate.softms_support[0].item())


def _frame_visual(model, visual, cache, route, gt_state, index, predicted_se,
                  previous_z, previous2_z, hidden, previous_acq_confidence,
                  previous_velocity, previous_heading_state, device, uav_clip):
    gt_xy_t = cache.gt_xy[index:index + 1].to(device).float()
    controlled_center_se, controlled_prior_xy, controlled_jitter_xy = b.controlled_gt_prior_se(
        cache, route, gt_state, index
    )
    search_heading_rad = b.wrap_angle_rad(
        route.route_heading_rad(float(predicted_se[0]))
        + float(previous_heading_state[0, 0].item())
    )
    obs = b.visual_observation(
        model=model, visual=visual, uav_clip=uav_clip,
        search_center_se=controlled_center_se, route=route,
        predicted_se=predicted_se, previous_z_uav=previous_z,
        previous2_z_uav=previous2_z, hidden=hidden,
        previous_acquisition_confidence=previous_acq_confidence,
        kalman_progress_std=0.0,
        previous_forward_speed=float(previous_velocity[0, 0].item()),
        search_heading_rad=search_heading_rad, device=device,
        gt_xy=gt_xy_t, gt_se=gt_state["se"][index], teacher_select=True,
    )
    return obs, controlled_prior_xy, controlled_jitter_xy, search_heading_rad


@torch.no_grad()
def evaluate_closed_loop(model, visual, cache, route, gt_state, metric_range, device):
    model.eval()
    start, end = map(int, metric_range)
    kf = RouteKalman(0.0, 0.0)
    hidden = previous_z = previous2_z = previous_measurement_se = None
    previous_velocity = torch.zeros(1, 2, device=device)
    previous_acceleration = torch.zeros(1, 2, device=device)
    previous_heading_state = torch.zeros(1, 2, device=device)
    previous_poly_step = torch.zeros(1, 2, device=device)
    previous_acq_confidence = float(config.ACQ_INITIAL_CONFIDENCE)
    errors, speed_errors, progress_errors, heading_errors, captures = [], [], [], [], []

    for index in range(end):
        uav_clip = cache.uav_clip[index:index + 1].to(device).float()
        if index > 0:
            _, feedback_se, _, _ = teacher_meanshift_feedback(
                visual, uav_clip, route, kf, previous_heading_state, device
            )
            previous_measurement_se = b.tensor2(feedback_se, device).detach()
            predicted_se = kf.predict(
                previous_velocity[0].cpu().numpy(), previous_acceleration[0].cpu().numpy(),
                route.total_length_m, max_progress_s=float(gt_state["se"][index, 0]),
                polynomial_step_se=previous_poly_step[0].cpu().numpy(),
                max_step_m=float(gt_state["gt_step_norm"][index]),
            )
        else:
            predicted_se = kf.se()
        predicted_se = b.cap_prediction_to_current_gt(kf, predicted_se, gt_state["se"][index])
        obs, _, _, _ = _frame_visual(
            model, visual, cache, route, gt_state, index, predicted_se,
            previous_z, previous2_z, hidden, previous_acq_confidence,
            previous_velocity, previous_heading_state, device, uav_clip
        )
        output = b.model_forward(
            model, obs, previous_z, previous2_z, predicted_se,
            previous_measurement_se, previous_velocity, previous_acceleration,
            previous_heading_state, previous_poly_step, route, hidden, device,
        )
        conf = b.visual_confidence_from_observation(obs)
        final_se = kf.update(
            output.measurement_se[0].cpu().numpy(), output.measurement_variance_se[0].cpu().numpy(),
            route.total_length_m, acquisition_confidence=conf,
            max_progress_s=float(gt_state["se"][index, 0]),
            max_final_step_m=float(gt_state["gt_step_norm"][index]),
        )
        final_se, _ = b.cap_kalman_to_current_gt(kf, final_se, gt_state["se"][index])
        if index >= start:
            final_xy = route.xy_from_se(final_se[0], final_se[1])
            gt_xy = cache.gt_xy[index].cpu().numpy()
            errors.append(float(np.linalg.norm(final_xy - gt_xy)))
            speed_errors.append(abs(float(output.velocity_se[0, 0]) - float(gt_state["velocity"][index, 0])))
            progress_errors.append(abs(float(final_se[0]) - float(gt_state["se"][index, 0])))
            heading_errors.append(abs(math.degrees(b.angle_error_rad(
                float(output.heading_residual_rad[0, 0]), float(gt_state["heading_residual"][index])
            ))))
            captures.append(float(obs.capture.float().item()))
        previous2_z, previous_z = previous_z, obs.candidate.z_uav.detach()
        previous_velocity, previous_acceleration, previous_poly_step = b.stabilize_motion_state(
            previous_velocity, previous_acceleration, previous_poly_step,
            output.velocity_se, output.acceleration_se, output.next_step_se,
        )
        previous_heading_state = b.stabilize_heading_state(
            previous_heading_state, output.heading_residual_rad, output.turn_rate_rad
        )
        hidden = output.hidden
        previous_acq_confidence = float(conf)

    if not errors:
        return {"mle": float("inf"), "p90": float("inf"), "speed_mae": float("inf"),
                "progress_mae": float("inf"), "heading_mae_deg": float("inf"),
                "capture_pct": 0.0, "score": float("inf")}
    mle = float(np.mean(errors)); speed = float(np.mean(speed_errors)); progress = float(np.mean(progress_errors))
    heading = float(np.mean(heading_errors)); capture = float(np.mean(captures) * 100.0)
    score = (mle + float(config.EARLY_SCORE_SPEED_WEIGHT) * speed
             + float(config.EARLY_SCORE_PROGRESS_WEIGHT) * progress
             + float(config.EARLY_SCORE_HEADING_WEIGHT) * heading
             + float(config.EARLY_SCORE_MISS_WEIGHT) * (100.0 - capture))
    return {"mle": mle, "p90": float(np.quantile(errors, .90)), "speed_mae": speed,
            "progress_mae": progress, "heading_mae_deg": heading,
            "capture_pct": capture, "score": score}


def train_temporal_model(visual, cache, route, device, epochs, patience_limit, resume=False):
    model = ThreeFrameRouteStateGRU().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(config.TEMPORAL_LR), weight_decay=float(config.TEMPORAL_WEIGHT_DECAY))
    gt_state = build_gt_route_state(cache, route)
    split = b.split_ranges(len(cache)); train_start, train_end = split["train"]; val_range = split["val"]
    start_epoch, best_score, best_state, patience = 1, float("inf"), None, 0
    if resume and config.LATEST_TEMPORAL_CHECKPOINT.exists():
        payload = torch.load(config.LATEST_TEMPORAL_CHECKPOINT, map_location="cpu")
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1; best_score = float(payload.get("best_score", best_score))
        best_state = payload.get("best_model"); patience = int(payload.get("patience", 0))
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, int(epochs) + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        kf = RouteKalman(0.0, 0.0)
        hidden = previous_z = previous2_z = previous_measurement_se = None
        previous_velocity = torch.zeros(1, 2, device=device)
        previous_acceleration = torch.zeros(1, 2, device=device)
        previous_heading_state = torch.zeros(1, 2, device=device)
        previous_poly_step = torch.zeros(1, 2, device=device)
        previous_acq_confidence = float(config.ACQ_INITIAL_CONFIDENCE)
        chunk_loss = None; chunk_count = 0; losses = []

        for index in range(train_start, train_end):
            uav_clip = cache.uav_clip[index:index + 1].to(device).float()
            if index > train_start:
                _, feedback_se, _, _ = teacher_meanshift_feedback(
                    visual, uav_clip, route, kf, previous_heading_state, device
                )
                previous_measurement_se = b.tensor2(feedback_se, device).detach()
                predicted_se = kf.predict(
                    previous_velocity[0].detach().cpu().numpy(), previous_acceleration[0].detach().cpu().numpy(),
                    route.total_length_m, max_progress_s=float(gt_state["se"][index, 0]),
                    polynomial_step_se=previous_poly_step[0].detach().cpu().numpy(),
                    max_step_m=float(gt_state["gt_step_norm"][index]),
                )
            else:
                predicted_se = kf.se()
            predicted_se = b.cap_prediction_to_current_gt(kf, predicted_se, gt_state["se"][index])
            obs, _, _, _ = _frame_visual(
                model, visual, cache, route, gt_state, index, predicted_se,
                previous_z, previous2_z, hidden, previous_acq_confidence,
                previous_velocity, previous_heading_state, device, uav_clip
            )
            output = b.model_forward(
                model, obs, previous_z, previous2_z, predicted_se,
                previous_measurement_se, previous_velocity, previous_acceleration,
                previous_heading_state, previous_poly_step, route, hidden, device,
            )
            loss, _ = b.temporal_loss(
                output, obs, gt_state["se"][index], gt_state["velocity"][index],
                gt_state["acceleration"][index], gt_state["step"][index],
                gt_state["heading_residual"][index], gt_state["turn_rate"][index],
            )
            chunk_loss = loss if chunk_loss is None else chunk_loss + loss; chunk_count += 1
            conf = b.visual_confidence_from_observation(obs)
            final_se = kf.update(
                output.measurement_se[0].detach().cpu().numpy(), output.measurement_variance_se[0].detach().cpu().numpy(),
                route.total_length_m, acquisition_confidence=conf,
                max_progress_s=float(gt_state["se"][index, 0]), max_final_step_m=float(gt_state["gt_step_norm"][index]),
            )
            b.cap_kalman_to_current_gt(kf, final_se, gt_state["se"][index])
            previous2_z, previous_z = previous_z, obs.candidate.z_uav.detach()
            previous_velocity, previous_acceleration, previous_poly_step = b.stabilize_motion_state(
                previous_velocity, previous_acceleration, previous_poly_step,
                output.velocity_se, output.acceleration_se, output.next_step_se,
            )
            previous_heading_state = b.stabilize_heading_state(
                previous_heading_state, output.heading_residual_rad, output.turn_rate_rad
            )
            hidden = output.hidden; previous_acq_confidence = float(conf)
            if chunk_count >= int(config.TBPTT_STEPS) or index + 1 >= train_end:
                normalized = chunk_loss / float(chunk_count); normalized.backward()
                torch.nn.utils.clip_grad_norm_(params, float(config.GRAD_CLIP_NORM)); optimizer.step(); optimizer.zero_grad(set_to_none=True)
                losses.append(float(normalized.detach().cpu()))
                hidden = hidden.detach(); previous_z = previous_z.detach() if previous_z is not None else None
                previous2_z = previous2_z.detach() if previous2_z is not None else None
                previous_velocity = previous_velocity.detach(); previous_acceleration = previous_acceleration.detach()
                previous_heading_state = previous_heading_state.detach(); previous_poly_step = previous_poly_step.detach()
                previous_measurement_se = previous_measurement_se.detach() if previous_measurement_se is not None else None
                chunk_loss = None; chunk_count = 0

        val = evaluate_closed_loop(model, visual, cache, route, gt_state, val_range, device)
        score = float(val["score"]); improved = score < best_score - float(config.EARLY_STOP_MIN_DELTA)
        if improved:
            best_score = score; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; patience = 0
            torch.save({"architecture": ARCHITECTURE_NAME, "model": best_state, "epoch": epoch,
                        "validation": val, "teacher_feedback": "next-frame SoftMS replaces GRU previous-position input; Kalman posterior preserved"},
                       config.TEMPORAL_CHECKPOINT)
        else:
            patience += 1
        torch.save({"architecture": ARCHITECTURE_NAME, "model": model.state_dict(), "best_model": best_state,
                    "optimizer": optimizer.state_dict(), "epoch": epoch, "best_score": best_score, "patience": patience},
                   config.LATEST_TEMPORAL_CHECKPOINT)
        print(f"teacher epoch={epoch:03d}/{epochs} loss={np.mean(losses):.5f} val_mle={val['mle']:.3f}m val_p90={val['p90']:.3f}m score={score:.3f} best={best_score:.3f} patience={patience}/{patience_limit}", flush=True)
        if epoch >= int(config.EARLY_STOP_MIN_EPOCH) and patience >= int(patience_limit):
            break
    if best_state is None:
        raise RuntimeError("Temporal training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_score


@torch.no_grad()
def run_route_inference(route_name, visual, model, cache, route, device):
    model.eval(); gt_state = build_gt_route_state(cache, route); kf = RouteKalman(0.0, 0.0)
    hidden = previous_z = previous2_z = previous_measurement_se = None
    previous_velocity = torch.zeros(1, 2, device=device); previous_acceleration = torch.zeros(1, 2, device=device)
    previous_heading_state = torch.zeros(1, 2, device=device); previous_poly_step = torch.zeros(1, 2, device=device)
    previous_acq_confidence = float(config.ACQ_INITIAL_CONFIDENCE)
    rows, errors = [], []
    for index in range(len(cache)):
        uav_clip = cache.uav_clip[index:index + 1].to(device).float()
        previous_output_se = kf.se().copy(); feedback_se = previous_output_se.copy(); feedback_xy = route.xy_from_se(*feedback_se); feedback_support = 0.0
        if index > 0:
            previous_output_se, feedback_se, feedback_xy, feedback_support = teacher_meanshift_feedback(
                visual, uav_clip, route, kf, previous_heading_state, device
            )
            previous_measurement_se = b.tensor2(feedback_se, device).detach()
            predicted_se = kf.predict(
                previous_velocity[0].cpu().numpy(), previous_acceleration[0].cpu().numpy(), route.total_length_m,
                max_progress_s=float(gt_state["se"][index, 0]), polynomial_step_se=previous_poly_step[0].cpu().numpy(),
                max_step_m=float(gt_state["gt_step_norm"][index]),
            )
        else:
            predicted_se = kf.se()
        predicted_se = b.cap_prediction_to_current_gt(kf, predicted_se, gt_state["se"][index])
        obs, prior_xy, jitter_xy, search_heading = _frame_visual(
            model, visual, cache, route, gt_state, index, predicted_se,
            previous_z, previous2_z, hidden, previous_acq_confidence,
            previous_velocity, previous_heading_state, device, uav_clip
        )
        output = b.model_forward(model, obs, previous_z, previous2_z, predicted_se, previous_measurement_se,
                                 previous_velocity, previous_acceleration, previous_heading_state,
                                 previous_poly_step, route, hidden, device)
        conf = b.visual_confidence_from_observation(obs)
        final_se = kf.update(output.measurement_se[0].cpu().numpy(), output.measurement_variance_se[0].cpu().numpy(),
                             route.total_length_m, acquisition_confidence=conf,
                             max_progress_s=float(gt_state["se"][index, 0]), max_final_step_m=float(gt_state["gt_step_norm"][index]))
        final_se, _ = b.cap_kalman_to_current_gt(kf, final_se, gt_state["se"][index])
        final_xy = route.xy_from_se(final_se[0], final_se[1]); gt_xy = cache.gt_xy[index].cpu().numpy(); error = float(np.linalg.norm(final_xy - gt_xy)); errors.append(error)
        rows.append({
            "frame_id": int(cache.frame_ids[index]), "image_path": cache.image_paths[index],
            "gt_x": float(gt_xy[0]), "gt_y": float(gt_xy[1]),
            "previous_kalman_output_s": float(previous_output_se[0]), "previous_kalman_output_e": float(previous_output_se[1]),
            "teacher_feedback_ms_s": float(feedback_se[0]), "teacher_feedback_ms_e": float(feedback_se[1]),
            "teacher_feedback_ms_x": float(feedback_xy[0]), "teacher_feedback_ms_y": float(feedback_xy[1]),
            "teacher_feedback_support": float(feedback_support),
            "predicted_progress_s": float(predicted_se[0]), "predicted_cross_e": float(predicted_se[1]),
            "softms_x": float(obs.candidate.softms_xy[0,0]), "softms_y": float(obs.candidate.softms_xy[0,1]),
            "measurement_s": float(output.measurement_se[0,0]), "measurement_e": float(output.measurement_se[0,1]),
            "final_progress_s": float(final_se[0]), "final_cross_e": float(final_se[1]),
            "final_x": float(final_xy[0]), "final_y": float(final_xy[1]), "error_final_m": error,
            "waypoint_leg": int(route.frame_from_se(final_se[0], final_se[1]).leg_index),
            "prior_center_x": float(prior_xy[0]), "prior_center_y": float(prior_xy[1]),
            "prior_jitter_x": float(jitter_xy[0]), "prior_jitter_y": float(jitter_xy[1]),
            "search_heading_deg": float(math.degrees(search_heading)),
        })
        previous2_z, previous_z = previous_z, obs.candidate.z_uav.detach()
        previous_velocity, previous_acceleration, previous_poly_step = b.stabilize_motion_state(
            previous_velocity, previous_acceleration, previous_poly_step, output.velocity_se, output.acceleration_se, output.next_step_se)
        previous_heading_state = b.stabilize_heading_state(previous_heading_state, output.heading_residual_rad, output.turn_rate_rad)
        hidden = output.hidden; previous_acq_confidence = float(conf)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.OUTPUT_DIR / f"{route_name}_v36_byTeacher_frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    summary = metric_summary(errors); summary["CSV"] = str(csv_path); summary["TeacherFeedback"] = "next-frame SoftMS replaces GRU previous-position input; Kalman posterior preserved"
    print(f"{route_name}: MLE={summary['MLE_m']:.3f}m P90={summary['P90_m']:.3f}m LSR@15={summary['LSR@15_pct']:.2f}%", flush=True)
    return summary


def load_temporal_model(device):
    if not config.TEMPORAL_CHECKPOINT.exists():
        raise FileNotFoundError(config.TEMPORAL_CHECKPOINT)
    payload = torch.load(config.TEMPORAL_CHECKPOINT, map_location="cpu")
    if payload.get("architecture") != ARCHITECTURE_NAME:
        raise RuntimeError(f"checkpoint architecture mismatch: {payload.get('architecture')}")
    model = ThreeFrameRouteStateGRU().to(device); model.load_state_dict(payload["model"]); model.eval(); return model


def train_pipeline(args, device):
    if not args.reuse_visual or not config.VISUAL_CHECKPOINT.exists():
        train_visual_retrieval_a_only(device=device, epochs=int(args.visual_epochs), jitter_m=float(args.jitter_m), resume=bool(args.resume_visual))
    visual = FrozenVisualLocalizer(device)
    cache = build_route_cache("route_A", config.ROUTE_ROOTS[0], visual, device)
    route = WaypointRoute(load_waypoint_xy("route_A", visual.origin_lat, visual.origin_lon))
    _, score = train_temporal_model(visual, cache, route, device, int(args.temporal_epochs), int(args.patience), bool(args.resume_temporal))
    print(f"best teacher-feedback validation score={score:.3f}", flush=True)


def eval_pipeline(device):
    visual = FrozenVisualLocalizer(device); model = load_temporal_model(device)
    all_summary = {"architecture": ARCHITECTURE_NAME, "teacher_feedback": "MeanShift(X_t) becomes GRU previous-position input for X_{t+1}; Kalman posterior X_t is preserved", "route_B": None, "route_C": None}
    for route_name in ["route_B", "route_C"]:
        i = config.ROUTE_NAMES.index(route_name); cache = build_route_cache(route_name, config.ROUTE_ROOTS[i], visual, device)
        route = WaypointRoute(load_waypoint_xy(route_name, visual.origin_lat, visual.origin_lon))
        all_summary[route_name] = run_route_inference(route_name, visual, model, cache, route, device)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "robust_tracker_summary.json").write_text(json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = b.parse_args(); config.LOCAL_PRIOR_JITTER_M = float(args.jitter_m); config.CONTROLLED_GT_PRIOR_JITTER_M = float(args.jitter_m)
    set_seed(config.SEED); device = resolve_device()
    print("="*100); print(ARCHITECTURE_NAME); print("Teacher flow: X_t(output)=Kalman(...); next image -> MS(X_t) becomes GRU previous position; Kalman preserves X_t -> predict/update -> X_t+1"); print(f"output={config.OUTPUT_DIR}"); print("="*100)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True); config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode in ("train", "train_eval"): train_pipeline(args, device)
    if args.mode in ("eval", "train_eval"): eval_pipeline(device)


if __name__ == "__main__":
    main()
