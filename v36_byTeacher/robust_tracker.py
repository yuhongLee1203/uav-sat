"""v36_byTeacher main entry point.

Current temporal interpretation
-------------------------------
For frame t, the newly arrived UAV image is first used to re-localize the
previous Kalman output X_{t-1} by Soft Mean-Shift.  That re-localized position
is only a previous-position cue for the GRU; it never overwrites the external
Kalman posterior.

The GRU then sees the current visual evidence and the re-localized previous
position and predicts two different things:
  1) motion state (velocity / acceleration / heading) for the inertial prior;
  2) the current visual measurement z_t and its learned variance R_t.

The external Kalman keeps the previous posterior and covariance, predicts the
current prior with the recurrent motion state, and updates that prior with the
GRU visual measurement.  MeanShift confidence is not used by the Kalman.

The thesis/default search remains forward 3x6.  For the candidate-count
ablation the same implementation can evaluate forward 4x6, 5x6 and 6x6.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import config
import robust_tracker_base as b
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only
from visual_model import ThreeFrameRouteStateGRU

ARCHITECTURE_NAME = str(config.ARCHITECTURE_NAME)

# Re-export commonly used base helpers/classes for compatibility.
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


def _set_forward_rows(rows):
    rows = int(rows)
    if rows not in (3, 4, 5, 6):
        raise ValueError("forward rows must be one of 3, 4, 5, 6")
    config.FORWARD_SEARCH_ROWS = rows
    config.FORWARD_SEARCH_COLS = 6
    config.FORWARD_SEARCH_CANDIDATE_COUNT = rows * 6
    return rows


def forward_rows_candidate_batch(visual, uav_clip, center_xy, heading_rad, grid_size=6):
    """Score the heading-forward R x 6 subset of the original 6x6 geometry.

    R is config.FORWARD_SEARCH_ROWS and can be 3, 4, 5 or 6.  The candidates
    with the largest heading projection are retained, so 3x6 is the thesis
    forward-only setting and 6x6 keeps the complete local lattice.
    """
    grid_size = int(grid_size)
    if grid_size != 6:
        raise ValueError("forward-row ablation requires base grid_size=6")
    rows = int(config.FORWARD_SEARCH_ROWS)
    cols = int(config.FORWARD_SEARCH_COLS)
    if cols != 6 or rows not in (3, 4, 5, 6):
        raise ValueError("supported search shapes are 3x6, 4x6, 5x6 and 6x6")
    keep_count = rows * cols

    headings = torch.as_tensor(
        heading_rad, dtype=center_xy.dtype, device=center_xy.device
    ).reshape(-1)
    batch = int(center_xy.shape[0])
    if headings.numel() == 1 and batch > 1:
        headings = headings.expand(batch)
    if headings.numel() != batch:
        raise ValueError("heading count must match center batch size")

    forward_unit = torch.stack([torch.cos(headings), torch.sin(headings)], dim=1)
    cross_unit = torch.stack([-torch.sin(headings), torch.cos(headings)], dim=1)
    backshift_m = float(getattr(config, "FORWARD_SEARCH_ORIGIN_BACKSHIFT_M", 0.0))
    grid_center_xy = center_xy - backshift_m * forward_unit

    full_indices = b.regular_grid_indices(
        visual.gallery["xy"],
        visual.gallery["pixel"],
        visual.pixel_index,
        grid_center_xy,
        grid_size,
        config.SAT_STRIDE,
        visual.device,
    )
    full_centers = visual.gallery["xy"][full_indices]
    relative = full_centers - grid_center_xy[:, None, :]
    forward_projection = (relative * forward_unit[:, None, :]).sum(dim=2)
    cross_projection = (relative * cross_unit[:, None, :]).sum(dim=2)

    if keep_count == grid_size * grid_size:
        selected_local = torch.arange(
            grid_size * grid_size, device=full_indices.device
        ).reshape(1, -1).expand(batch, -1)
    else:
        selected_local = torch.topk(
            forward_projection, k=keep_count, dim=1, largest=True, sorted=False
        ).indices

    selected_indices = torch.gather(full_indices, 1, selected_local)
    selected_forward = torch.gather(forward_projection, 1, selected_local)
    selected_cross = torch.gather(cross_projection, 1, selected_local)

    # Stable front-to-back, left-to-right order.
    ordering_key = -selected_forward * 1000.0 + selected_cross
    order = torch.argsort(ordering_key, dim=1)
    selected_indices = torch.gather(selected_indices, 1, order)

    centers = visual.gallery["xy"][selected_indices]
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
    softms_xy, softms_support, _, _, mode_weights, _ = b.soft_mean_shift(
        raw_logits,
        centers,
        config.MEANSHIFT_SCORE_TAU,
        config.MEANSHIFT_BANDWIDTH_M,
        config.MEANSHIFT_ITERATIONS,
        config.MEANSHIFT_MODE_BETA,
    )
    return b.CandidateBatch(
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


# robust_tracker_base.visual_observation resolves this module-global function at
# runtime.  Replacing it here keeps the base file/checkpoint compatibility while
# making the search size configurable.
b.forward_3x6_candidate_batch = forward_rows_candidate_batch


def _kalman_update(kf, output, route, gt_state, index):
    """Kalman update with no MeanShift/local-posterior confidence input."""
    return kf.update(
        output.measurement_se[0].detach().cpu().numpy(),
        output.measurement_variance_se[0].detach().cpu().numpy(),
        route.total_length_m,
        acquisition_confidence=1.0,
        max_progress_s=float(gt_state["se"][index, 0]),
        max_final_step_m=float(gt_state["gt_step_norm"][index]),
    )


def teacher_meanshift_feedback(visual, uav_clip, route, kf, previous_heading_state, device):
    """Use the newly arrived image to re-localize the previous Kalman output.

    The returned X_(t-1)^MS is a GRU previous-position cue only.  kf.x and kf.P
    remain the true previous Kalman posterior used by the next predict/update.
    """
    previous_output_se = kf.se().copy()
    previous_output_xy = route.xy_from_se(previous_output_se[0], previous_output_se[1])
    center_xy = torch.tensor(
        previous_output_xy, dtype=torch.float32, device=device
    ).reshape(1, 2)
    heading_rad = b.wrap_angle_rad(
        route.route_heading_rad(float(previous_output_se[0]))
        + float(previous_heading_state[0, 0].item())
    )
    candidate = forward_rows_candidate_batch(
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
    feedback_se[1] = float(
        np.clip(
            feedback_se[1],
            -float(config.MAX_FINAL_CROSS_TRACK_M),
            float(config.MAX_FINAL_CROSS_TRACK_M),
        )
    )
    return (
        previous_output_se,
        feedback_se,
        feedback_xy,
        float(candidate.softms_support[0].item()),
    )


def _frame_visual(
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
    previous_velocity,
    previous_heading_state,
    device,
    uav_clip,
):
    gt_xy_t = cache.gt_xy[index : index + 1].to(device).float()
    controlled_center_se, controlled_prior_xy, controlled_jitter_xy = (
        b.controlled_gt_prior_se(cache, route, gt_state, index)
    )
    search_heading_rad = b.wrap_angle_rad(
        route.route_heading_rad(float(predicted_se[0]))
        + float(previous_heading_state[0, 0].item())
    )
    obs = b.visual_observation(
        model=model,
        visual=visual,
        uav_clip=uav_clip,
        search_center_se=controlled_center_se,
        route=route,
        predicted_se=predicted_se,
        previous_z_uav=previous_z,
        previous2_z_uav=previous2_z,
        hidden=hidden,
        previous_acquisition_confidence=1.0,
        kalman_progress_std=0.0,
        previous_forward_speed=float(previous_velocity[0, 0].item()),
        search_heading_rad=search_heading_rad,
        device=device,
        gt_xy=gt_xy_t,
        gt_se=gt_state["se"][index],
        teacher_select=True,
    )
    return obs, controlled_prior_xy, controlled_jitter_xy, search_heading_rad


@torch.no_grad()
def evaluate_closed_loop(model, visual, cache, route, gt_state, metric_range, device):
    model.eval()
    start, end = map(int, metric_range)
    kf = RouteKalman(0.0, 0.0)
    hidden = previous_z = previous2_z = previous_ms_se = None
    previous_velocity = torch.zeros(1, 2, device=device)
    previous_acceleration = torch.zeros(1, 2, device=device)
    previous_heading_state = torch.zeros(1, 2, device=device)
    previous_poly_step = torch.zeros(1, 2, device=device)
    errors, speed_errors, progress_errors, heading_errors, captures = [], [], [], [], []

    for index in range(end):
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        if index > 0:
            _, feedback_se, _, _ = teacher_meanshift_feedback(
                visual, uav_clip, route, kf, previous_heading_state, device
            )
            previous_ms_se = b.tensor2(feedback_se, device).detach()
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
        predicted_se = b.cap_prediction_to_current_gt(
            kf, predicted_se, gt_state["se"][index]
        )

        obs, _, _, _ = _frame_visual(
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
            previous_ms_se,
            previous_velocity,
            previous_acceleration,
            previous_heading_state,
            previous_poly_step,
            route,
            hidden,
            device,
        )
        final_se = _kalman_update(kf, output, route, gt_state, index)
        final_se, _ = b.cap_kalman_to_current_gt(
            kf, final_se, gt_state["se"][index]
        )

        if index >= start:
            final_xy = route.xy_from_se(final_se[0], final_se[1])
            reference_xy = cache.gt_xy[index].cpu().numpy()
            errors.append(float(np.linalg.norm(final_xy - reference_xy)))
            speed_errors.append(
                abs(
                    float(output.velocity_se[0, 0])
                    - float(gt_state["velocity"][index, 0])
                )
            )
            progress_errors.append(
                abs(float(final_se[0]) - float(gt_state["se"][index, 0]))
            )
            heading_errors.append(
                abs(
                    math.degrees(
                        b.angle_error_rad(
                            float(output.heading_residual_rad[0, 0]),
                            float(gt_state["heading_residual"][index]),
                        )
                    )
                )
            )
            captures.append(float(obs.capture.float().item()))

        previous2_z, previous_z = previous_z, obs.candidate.z_uav.detach()
        previous_velocity, previous_acceleration, previous_poly_step = (
            b.stabilize_motion_state(
                previous_velocity,
                previous_acceleration,
                previous_poly_step,
                output.velocity_se,
                output.acceleration_se,
                output.next_step_se,
            )
        )
        previous_heading_state = b.stabilize_heading_state(
            previous_heading_state,
            output.heading_residual_rad,
            output.turn_rate_rad,
        )
        hidden = output.hidden

    if not errors:
        return {
            "mle": float("inf"),
            "p90": float("inf"),
            "speed_mae": float("inf"),
            "progress_mae": float("inf"),
            "heading_mae_deg": float("inf"),
            "capture_pct": 0.0,
            "score": float("inf"),
        }
    mle = float(np.mean(errors))
    speed = float(np.mean(speed_errors))
    progress = float(np.mean(progress_errors))
    heading = float(np.mean(heading_errors))
    capture = float(np.mean(captures) * 100.0)
    score = (
        mle
        + float(config.EARLY_SCORE_SPEED_WEIGHT) * speed
        + float(config.EARLY_SCORE_PROGRESS_WEIGHT) * progress
        + float(config.EARLY_SCORE_HEADING_WEIGHT) * heading
        + float(config.EARLY_SCORE_MISS_WEIGHT) * (100.0 - capture)
    )
    return {
        "mle": mle,
        "p90": float(np.quantile(errors, 0.90)),
        "speed_mae": speed,
        "progress_mae": progress,
        "heading_mae_deg": heading,
        "capture_pct": capture,
        "score": score,
    }


def train_temporal_model(visual, cache, route, device, epochs, patience_limit, resume=False):
    model = ThreeFrameRouteStateGRU().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config.TEMPORAL_LR),
        weight_decay=float(config.TEMPORAL_WEIGHT_DECAY),
    )
    gt_state = build_gt_route_state(cache, route)
    split = b.split_ranges(len(cache))
    train_start, train_end = split["train"]
    val_range = split["val"]
    start_epoch, best_score, best_state, patience = 1, float("inf"), None, 0

    if resume and config.LATEST_TEMPORAL_CHECKPOINT.exists():
        payload = torch.load(config.LATEST_TEMPORAL_CHECKPOINT, map_location="cpu")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload.get("best_score", best_score))
        best_state = payload.get("best_model")
        patience = int(payload.get("patience", 0))

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        kf = RouteKalman(0.0, 0.0)
        hidden = previous_z = previous2_z = previous_ms_se = None
        previous_velocity = torch.zeros(1, 2, device=device)
        previous_acceleration = torch.zeros(1, 2, device=device)
        previous_heading_state = torch.zeros(1, 2, device=device)
        previous_poly_step = torch.zeros(1, 2, device=device)
        chunk_loss = None
        chunk_count = 0
        losses = []

        for index in range(train_start, train_end):
            uav_clip = cache.uav_clip[index : index + 1].to(device).float()
            if index > train_start:
                _, feedback_se, _, _ = teacher_meanshift_feedback(
                    visual, uav_clip, route, kf, previous_heading_state, device
                )
                previous_ms_se = b.tensor2(feedback_se, device).detach()
                predicted_se = kf.predict(
                    previous_velocity[0].detach().cpu().numpy(),
                    previous_acceleration[0].detach().cpu().numpy(),
                    route.total_length_m,
                    max_progress_s=float(gt_state["se"][index, 0]),
                    polynomial_step_se=previous_poly_step[0].detach().cpu().numpy(),
                    max_step_m=float(gt_state["gt_step_norm"][index]),
                )
            else:
                predicted_se = kf.se()

            predicted_se = b.cap_prediction_to_current_gt(
                kf, predicted_se, gt_state["se"][index]
            )
            obs, _, _, _ = _frame_visual(
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
                previous_ms_se,
                previous_velocity,
                previous_acceleration,
                previous_heading_state,
                previous_poly_step,
                route,
                hidden,
                device,
            )
            loss, _ = b.temporal_loss(
                output,
                obs,
                gt_state["se"][index],
                gt_state["velocity"][index],
                gt_state["acceleration"][index],
                gt_state["step"][index],
                gt_state["heading_residual"][index],
                gt_state["turn_rate"][index],
            )
            chunk_loss = loss if chunk_loss is None else chunk_loss + loss
            chunk_count += 1

            final_se = _kalman_update(kf, output, route, gt_state, index)
            b.cap_kalman_to_current_gt(kf, final_se, gt_state["se"][index])

            previous2_z, previous_z = previous_z, obs.candidate.z_uav.detach()
            previous_velocity, previous_acceleration, previous_poly_step = (
                b.stabilize_motion_state(
                    previous_velocity,
                    previous_acceleration,
                    previous_poly_step,
                    output.velocity_se,
                    output.acceleration_se,
                    output.next_step_se,
                )
            )
            previous_heading_state = b.stabilize_heading_state(
                previous_heading_state,
                output.heading_residual_rad,
                output.turn_rate_rad,
            )
            hidden = output.hidden

            if chunk_count >= int(config.TBPTT_STEPS) or index + 1 >= train_end:
                normalized = chunk_loss / float(chunk_count)
                normalized.backward()
                torch.nn.utils.clip_grad_norm_(params, float(config.GRAD_CLIP_NORM))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(normalized.detach().cpu()))
                hidden = hidden.detach()
                previous_z = previous_z.detach() if previous_z is not None else None
                previous2_z = (
                    previous2_z.detach() if previous2_z is not None else None
                )
                previous_velocity = previous_velocity.detach()
                previous_acceleration = previous_acceleration.detach()
                previous_heading_state = previous_heading_state.detach()
                previous_poly_step = previous_poly_step.detach()
                previous_ms_se = (
                    previous_ms_se.detach() if previous_ms_se is not None else None
                )
                chunk_loss = None
                chunk_count = 0

        val = evaluate_closed_loop(
            model, visual, cache, route, gt_state, val_range, device
        )
        score = float(val["score"])
        improved = score < best_score - float(config.EARLY_STOP_MIN_DELTA)
        if improved:
            best_score = score
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience = 0
            torch.save(
                {
                    "architecture": ARCHITECTURE_NAME,
                    "model": best_state,
                    "epoch": epoch,
                    "validation": val,
                    "train_forward_rows": int(config.FORWARD_SEARCH_ROWS),
                    "teacher_feedback": (
                        "new image MeanShift re-localizes previous Kalman output; "
                        "GRU predicts current measurement/variance and motion; "
                        "Kalman fuses recurrent motion prior with GRU measurement"
                    ),
                },
                config.TEMPORAL_CHECKPOINT,
            )
        else:
            patience += 1

        torch.save(
            {
                "architecture": ARCHITECTURE_NAME,
                "model": model.state_dict(),
                "best_model": best_state,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "patience": patience,
                "train_forward_rows": int(config.FORWARD_SEARCH_ROWS),
            },
            config.LATEST_TEMPORAL_CHECKPOINT,
        )
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(
            f"teacher {config.FORWARD_SEARCH_ROWS}x6 epoch={epoch:03d}/{epochs} "
            f"loss={mean_loss:.5f} val_mle={val['mle']:.3f}m "
            f"val_p90={val['p90']:.3f}m score={score:.3f} "
            f"best={best_score:.3f} patience={patience}/{patience_limit}",
            flush=True,
        )
        if (
            epoch >= int(config.EARLY_STOP_MIN_EPOCH)
            and patience >= int(patience_limit)
        ):
            break

    if best_state is None:
        raise RuntimeError("Temporal training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_score


@torch.no_grad()
def run_route_inference(route_name, visual, model, cache, route, device, measure_latency=False):
    model.eval()
    gt_state = build_gt_route_state(cache, route)
    kf = RouteKalman(0.0, 0.0)
    hidden = previous_z = previous2_z = previous_ms_se = None
    previous_velocity = torch.zeros(1, 2, device=device)
    previous_acceleration = torch.zeros(1, 2, device=device)
    previous_heading_state = torch.zeros(1, 2, device=device)
    previous_poly_step = torch.zeros(1, 2, device=device)

    rows = []
    errors = []
    timing_ms = []
    warmup = int(getattr(config, "LATENCY_WARMUP_FRAMES", 30))

    for index in range(len(cache)):
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        if measure_latency and device.type == "cuda":
            torch.cuda.synchronize(device)
        frame_start = time.perf_counter()

        previous_output_se = kf.se().copy()
        feedback_se = previous_output_se.copy()
        feedback_xy = route.xy_from_se(*feedback_se)
        feedback_support = 0.0
        if index > 0:
            (
                previous_output_se,
                feedback_se,
                feedback_xy,
                feedback_support,
            ) = teacher_meanshift_feedback(
                visual, uav_clip, route, kf, previous_heading_state, device
            )
            previous_ms_se = b.tensor2(feedback_se, device).detach()
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

        predicted_se = b.cap_prediction_to_current_gt(
            kf, predicted_se, gt_state["se"][index]
        )
        obs, prior_xy, jitter_xy, search_heading = _frame_visual(
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
            previous_ms_se,
            previous_velocity,
            previous_acceleration,
            previous_heading_state,
            previous_poly_step,
            route,
            hidden,
            device,
        )
        final_se = _kalman_update(kf, output, route, gt_state, index)
        final_se, _ = b.cap_kalman_to_current_gt(
            kf, final_se, gt_state["se"][index]
        )

        if measure_latency and device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - frame_start) * 1000.0
        if measure_latency:
            timing_ms.append(float(elapsed_ms))

        final_xy = route.xy_from_se(final_se[0], final_se[1])
        reference_xy = cache.gt_xy[index].cpu().numpy()
        error = float(np.linalg.norm(final_xy - reference_xy))
        errors.append(error)

        rows.append(
            {
                "frame_id": int(cache.frame_ids[index]),
                "image_path": cache.image_paths[index],
                "forward_rows": int(config.FORWARD_SEARCH_ROWS),
                "candidate_count": int(config.FORWARD_SEARCH_CANDIDATE_COUNT),
                "reference_x": float(reference_xy[0]),
                "reference_y": float(reference_xy[1]),
                "previous_kalman_output_s": float(previous_output_se[0]),
                "previous_kalman_output_e": float(previous_output_se[1]),
                "previous_ms_s": float(feedback_se[0]),
                "previous_ms_e": float(feedback_se[1]),
                "previous_ms_x": float(feedback_xy[0]),
                "previous_ms_y": float(feedback_xy[1]),
                "previous_ms_support": float(feedback_support),
                "kalman_prior_s": float(predicted_se[0]),
                "kalman_prior_e": float(predicted_se[1]),
                "current_softms_x": float(obs.candidate.softms_xy[0, 0]),
                "current_softms_y": float(obs.candidate.softms_xy[0, 1]),
                "gru_measurement_s": float(output.measurement_se[0, 0]),
                "gru_measurement_e": float(output.measurement_se[0, 1]),
                "gru_measurement_var_s": float(output.measurement_variance_se[0, 0]),
                "gru_measurement_var_e": float(output.measurement_variance_se[0, 1]),
                "gru_velocity_s": float(output.velocity_se[0, 0]),
                "gru_velocity_e": float(output.velocity_se[0, 1]),
                "gru_acceleration_s": float(output.acceleration_se[0, 0]),
                "gru_acceleration_e": float(output.acceleration_se[0, 1]),
                "gru_heading_residual_deg": float(
                    math.degrees(float(output.heading_residual_rad[0, 0]))
                ),
                "final_s": float(final_se[0]),
                "final_e": float(final_se[1]),
                "final_x": float(final_xy[0]),
                "final_y": float(final_xy[1]),
                "error_final_m": float(error),
                "prior_center_x": float(prior_xy[0]),
                "prior_center_y": float(prior_xy[1]),
                "prior_jitter_x": float(jitter_xy[0]),
                "prior_jitter_y": float(jitter_xy[1]),
                "search_heading_deg": float(math.degrees(search_heading)),
                "tracking_core_latency_ms": float(elapsed_ms),
            }
        )

        previous2_z, previous_z = previous_z, obs.candidate.z_uav.detach()
        previous_velocity, previous_acceleration, previous_poly_step = (
            b.stabilize_motion_state(
                previous_velocity,
                previous_acceleration,
                previous_poly_step,
                output.velocity_se,
                output.acceleration_se,
                output.next_step_se,
            )
        )
        previous_heading_state = b.stabilize_heading_state(
            previous_heading_state,
            output.heading_residual_rad,
            output.turn_rate_rad,
        )
        hidden = output.hidden

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"forward{config.FORWARD_SEARCH_ROWS}x6"
    csv_path = config.OUTPUT_DIR / f"{route_name}_{tag}_v36_byTeacher_frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = metric_summary(errors)
    summary["ForwardRows"] = int(config.FORWARD_SEARCH_ROWS)
    summary["CandidateCount"] = int(config.FORWARD_SEARCH_CANDIDATE_COUNT)
    summary["CSV"] = str(csv_path)
    summary["KalmanMeasurementSource"] = "GRU measurement head"
    summary["KalmanVarianceSource"] = "GRU variance head"
    summary["KalmanMSConfidence"] = False
    summary["MeanShiftRole"] = "re-localized previous-position cue + current GRU visual evidence"
    if measure_latency and timing_ms:
        steady = np.asarray(timing_ms[warmup:], dtype=np.float64)
        if steady.size == 0:
            steady = np.asarray(timing_ms, dtype=np.float64)
        mean_ms = float(np.mean(steady))
        summary["TrackingCoreTiming"] = {
            "definition": (
                "cached UAV backbone feature -> previous-position MeanShift + current local MeanShift "
                "-> GRU -> Kalman -> final XY"
            ),
            "excluded": [
                "image disk I/O",
                "image preprocessing",
                "UAV backbone encoding",
                "model/checkpoint loading",
                "satellite gallery construction",
            ],
            "warmup_frames": int(min(warmup, len(timing_ms))),
            "samples": int(steady.size),
            "mean_ms": mean_ms,
            "median_ms": float(np.median(steady)),
            "p90_ms": float(np.quantile(steady, 0.90)),
            "fps": float(1000.0 / max(mean_ms, 1e-12)),
        }

    timing_text = ""
    if "TrackingCoreTiming" in summary:
        timing_text = (
            f" core={summary['TrackingCoreTiming']['mean_ms']:.2f}ms "
            f"FPS={summary['TrackingCoreTiming']['fps']:.2f}"
        )
    print(
        f"{route_name} {config.FORWARD_SEARCH_ROWS}x6: "
        f"MLE={summary['MLE_m']:.3f}m P90={summary['P90_m']:.3f}m "
        f"LSR@15={summary['LSR@15_pct']:.2f}%{timing_text}",
        flush=True,
    )
    return summary


def load_temporal_model(device):
    if not config.TEMPORAL_CHECKPOINT.exists():
        raise FileNotFoundError(config.TEMPORAL_CHECKPOINT)
    payload = torch.load(config.TEMPORAL_CHECKPOINT, map_location="cpu")
    if payload.get("architecture") != ARCHITECTURE_NAME:
        raise RuntimeError(
            f"checkpoint architecture mismatch: {payload.get('architecture')}"
        )
    model = ThreeFrameRouteStateGRU().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def train_pipeline(args, device):
    if not args.reuse_visual or not config.VISUAL_CHECKPOINT.exists():
        train_visual_retrieval_a_only(
            device=device,
            epochs=int(args.visual_epochs),
            jitter_m=float(args.jitter_m),
            resume=bool(args.resume_visual),
        )
    visual = FrozenVisualLocalizer(device)
    cache = build_route_cache("route_A", config.ROUTE_ROOTS[0], visual, device)
    route = WaypointRoute(
        load_waypoint_xy("route_A", visual.origin_lat, visual.origin_lon)
    )
    _, score = train_temporal_model(
        visual,
        cache,
        route,
        device,
        int(args.temporal_epochs),
        int(args.patience),
        bool(args.resume_temporal),
    )
    print(f"best validation score={score:.3f}", flush=True)


def eval_pipeline(args, device):
    visual = FrozenVisualLocalizer(device)
    model = load_temporal_model(device)
    rows_to_eval = [3, 4, 5, 6] if args.eval_all_forward_rows else [int(args.forward_rows)]

    all_summary = {
        "architecture": ARCHITECTURE_NAME,
        "train_routes": ["route_A"],
        "eval_routes": ["route_B", "route_C"],
        "trained_default_search": "3x6",
        "ablation_note": (
            "4x6/5x6/6x6 use the same trained temporal checkpoint so the comparison isolates "
            "inference candidate-count accuracy/time trade-off"
        ),
        "kalman": (
            "previous posterior/covariance -> recurrent motion prior; update with GRU measurement and GRU variance"
        ),
        "meanshift": (
            "new image re-localizes previous Kalman output for GRU previous-position cue; "
            "current SoftMS position is GRU evidence, not algebraically added to measurement head"
        ),
        "results": {},
    }
    comparison_rows = []

    for forward_rows in rows_to_eval:
        _set_forward_rows(forward_rows)
        row_key = f"{forward_rows}x6"
        all_summary["results"][row_key] = {}
        for route_name in ["route_B", "route_C"]:
            route_index = config.ROUTE_NAMES.index(route_name)
            cache = build_route_cache(
                route_name, config.ROUTE_ROOTS[route_index], visual, device
            )
            route = WaypointRoute(
                load_waypoint_xy(route_name, visual.origin_lat, visual.origin_lon)
            )
            result = run_route_inference(
                route_name,
                visual,
                model,
                cache,
                route,
                device,
                measure_latency=bool(args.measure_latency),
            )
            all_summary["results"][row_key][route_name] = result
            timing = result.get("TrackingCoreTiming", {})
            comparison_rows.append(
                {
                    "search": row_key,
                    "candidate_count": int(forward_rows * 6),
                    "route": route_name,
                    "MLE_m": float(result["MLE_m"]),
                    "P90_m": float(result["P90_m"]),
                    "LSR@15_pct": float(result["LSR@15_pct"]),
                    "mean_core_ms": float(timing.get("mean_ms", float("nan"))),
                    "core_fps": float(timing.get("fps", float("nan"))),
                }
            )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.eval_all_forward_rows:
        summary_path = config.OUTPUT_DIR / "forward_search_ablation_summary.json"
        comparison_path = config.OUTPUT_DIR / "forward_search_ablation.csv"
        with comparison_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)
        all_summary["comparison_csv"] = str(comparison_path)
    else:
        summary_path = config.OUTPUT_DIR / (
            f"robust_tracker_summary_forward{args.forward_rows}x6.json"
        )
    summary_path.write_text(
        json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"summary: {summary_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["train", "eval", "train_eval"], default="train_eval"
    )
    parser.add_argument(
        "--visual-epochs", type=int, default=int(config.VISUAL_EPOCHS)
    )
    parser.add_argument(
        "--temporal-epochs", type=int, default=int(config.TEMPORAL_EPOCHS)
    )
    parser.add_argument(
        "--jitter-m", type=float, default=float(config.LOCAL_PRIOR_JITTER_M)
    )
    parser.add_argument(
        "--patience", type=int, default=int(config.EARLY_STOP_PATIENCE)
    )
    parser.add_argument("--reuse-visual", action="store_true")
    parser.add_argument("--resume-visual", action="store_true")
    parser.add_argument("--resume-temporal", action="store_true")
    parser.add_argument(
        "--forward-rows",
        type=int,
        choices=[3, 4, 5, 6],
        default=int(getattr(config, "FORWARD_SEARCH_ROWS", 3)),
        help="forward local search rows; thesis/default is 3 (3x6=18)",
    )
    parser.add_argument(
        "--eval-all-forward-rows",
        action="store_true",
        help="evaluate 3x6, 4x6, 5x6 and 6x6 with the same temporal checkpoint",
    )
    parser.add_argument(
        "--measure-latency",
        action="store_true",
        help="measure tracking-core latency/FPS after cached UAV backbone features",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _set_forward_rows(args.forward_rows)
    config.LOCAL_PRIOR_JITTER_M = float(args.jitter_m)
    config.CONTROLLED_GT_PRIOR_JITTER_M = float(args.jitter_m)
    set_seed(config.SEED)
    device = resolve_device()

    print("=" * 100, flush=True)
    print(ARCHITECTURE_NAME, flush=True)
    print(
        "Flow: new image -> MeanShift(previous Kalman output) -> GRU previous-position cue; "
        "GRU -> current measurement/variance + motion; Kalman prior + GRU measurement -> final state",
        flush=True,
    )
    print(
        f"search={config.FORWARD_SEARCH_ROWS}x6 "
        f"({config.FORWARD_SEARCH_CANDIDATE_COUNT} candidates); "
        "Kalman MeanShift-confidence input=OFF",
        flush=True,
    )
    print(f"output={config.OUTPUT_DIR}", flush=True)
    print("=" * 100, flush=True)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode in ("train", "train_eval"):
        train_pipeline(args, device)
    if args.mode in ("eval", "train_eval"):
        eval_pipeline(args, device)


if __name__ == "__main__":
    main()
