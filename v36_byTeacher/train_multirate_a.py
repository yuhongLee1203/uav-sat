"""Multi-rate Route-A temporal training for the image-aligned v36_byTeacher.

Training data:
  1) the original Route-A TRAIN split at native frame spacing;
  2) an in-memory stride-N sequence made only from the same Route-A TRAIN split.

Inference architecture is unchanged:
  previous final KF state -> ONE local MeanShift -> GRU measurement/variance;
  previous GRU motion/heading -> polynomial -> KF predict;
  KF update is the only motion-prior / visual-measurement fusion.

Training uses scheduled search-center teacher forcing only to avoid the failure
mode where an untrained KF immediately moves the 3x6 local window away from the
correct region. Early epochs use the previous route reference position as the
search center; the ratio then decays toward the model's own previous KF output.

The planned-route start is known by the method, so every sequence initializes the
external KF at its first predefined route-reference state instead of an arbitrary
[s=0,e=0]. This keeps training, validation, and inference initialization causal
and consistent with the stated problem setup.
"""

import argparse

import numpy as np
import torch

import config
import robust_tracker as rt
import robust_tracker_base as b
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only
from visual_model import ThreeFrameRouteStateGRU


ARCHITECTURE_NAME = str(config.ARCHITECTURE_NAME)


def _slice_cache(cache, indices, name):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size < 2:
        raise RuntimeError("stride Route-A training sequence needs at least two frames")
    tensor_index = torch.as_tensor(indices, dtype=torch.long)
    return b.RouteCache(
        route_name=name,
        frame_ids=cache.frame_ids[tensor_index].clone(),
        gt_xy=cache.gt_xy[tensor_index].clone(),
        uav_clip=cache.uav_clip[tensor_index].clone(),
        image_paths=[cache.image_paths[int(i)] for i in indices.tolist()],
    )


def _step_stats(cache):
    xy = cache.gt_xy.detach().cpu().numpy().astype(np.float64)
    if len(xy) < 2:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    step = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    return {
        "mean": float(np.mean(step)),
        "median": float(np.median(step)),
        "p90": float(np.quantile(step, 0.90)),
    }


def _teacher_ratio(epoch):
    epoch = int(epoch)
    warmup = int(config.IMAGE_FLOW_TEACHER_WARMUP_EPOCHS)
    decay = max(1, int(config.IMAGE_FLOW_TEACHER_DECAY_EPOCHS))
    final_ratio = float(config.IMAGE_FLOW_TEACHER_FINAL_RATIO)
    if epoch <= warmup:
        return 1.0
    progress = min(1.0, float(epoch - warmup) / float(decay))
    return float(1.0 + progress * (final_ratio - 1.0))


def _training_search_center(
    gt_state,
    current_index,
    previous_index,
    model_previous_se,
    teacher_ratio,
):
    """Blend previous reference position with previous model KF state.

    This affects only the TRAINING local-search center. Validation/evaluation
    always use the previous final KF state exactly as shown in the architecture.
    """
    if previous_index is None:
        reference_previous = np.asarray(
            gt_state["se"][int(current_index)], dtype=np.float64
        ).reshape(2)
    else:
        reference_previous = np.asarray(
            gt_state["se"][int(previous_index)], dtype=np.float64
        ).reshape(2)
    model_previous = np.asarray(model_previous_se, dtype=np.float64).reshape(2)
    ratio = float(np.clip(teacher_ratio, 0.0, 1.0))
    center = ratio * reference_previous + (1.0 - ratio) * model_previous
    center[0] = max(0.0, float(center[0]))
    return center


def _detach_temporal_state(
    hidden,
    previous_z,
    previous2_z,
    previous_velocity,
    previous_acceleration,
    previous_heading_state,
    previous_poly_step,
):
    return (
        hidden.detach() if hidden is not None else None,
        previous_z.detach() if previous_z is not None else None,
        previous2_z.detach() if previous2_z is not None else None,
        previous_velocity.detach(),
        previous_acceleration.detach(),
        previous_heading_state.detach(),
        previous_poly_step.detach(),
    )


def _train_one_sequence(
    model,
    optimizer,
    params,
    visual,
    cache,
    route,
    gt_state,
    indices,
    device,
    epoch,
):
    indices = list(indices)
    if not indices:
        return [], 0.0

    first_index = int(indices[0])
    initial_se = np.asarray(gt_state["se"][first_index], dtype=np.float64).reshape(2)
    teacher_ratio = _teacher_ratio(epoch)
    kf = rt.RouteKalman(float(initial_se[0]), float(initial_se[1]))
    hidden = previous_z = previous2_z = None
    previous_velocity = torch.zeros(1, 2, device=device)
    previous_acceleration = torch.zeros(1, 2, device=device)
    previous_heading_state = torch.zeros(1, 2, device=device)
    previous_poly_step = torch.zeros(1, 2, device=device)

    optimizer.zero_grad(set_to_none=True)
    chunk_loss = None
    chunk_count = 0
    losses = []
    captures = []
    previous_index = None

    for sequence_pos, index in enumerate(indices):
        index = int(index)
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        model_previous_se = kf.se().copy()

        if index != first_index:
            predicted_se = kf.predict(
                previous_velocity[0].detach().cpu().numpy(),
                previous_acceleration[0].detach().cpu().numpy(),
                route.total_length_m,
                max_progress_s=float(gt_state["se"][index, 0]),
                polynomial_step_se=previous_poly_step[0].detach().cpu().numpy(),
                max_step_m=float(gt_state["gt_step_norm"][index]),
            )
        else:
            predicted_se = model_previous_se.copy()

        predicted_se = b.cap_prediction_to_current_gt(
            kf, predicted_se, gt_state["se"][index]
        )

        search_center_se = _training_search_center(
            gt_state=gt_state,
            current_index=index,
            previous_index=previous_index,
            model_previous_se=model_previous_se,
            teacher_ratio=teacher_ratio,
        )

        obs, _, _ = rt._frame_visual(
            model,
            visual,
            cache,
            route,
            gt_state,
            index,
            search_center_se,
            predicted_se,
            previous_z,
            previous2_z,
            hidden,
            previous_velocity,
            previous_heading_state,
            device,
            uav_clip,
        )

        output = rt._model_step(
            model,
            obs,
            previous_z,
            previous2_z,
            search_center_se,
            predicted_se,
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
        captures.append(float(obs.capture.float().item()))

        final_se = rt._kalman_update(kf, output, route, gt_state, index)
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
        previous_index = index

        is_last = sequence_pos + 1 >= len(indices)
        if chunk_count >= int(config.TBPTT_STEPS) or is_last:
            normalized = chunk_loss / float(chunk_count)
            normalized.backward()
            torch.nn.utils.clip_grad_norm_(params, float(config.GRAD_CLIP_NORM))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(normalized.detach().cpu()))

            (
                hidden,
                previous_z,
                previous2_z,
                previous_velocity,
                previous_acceleration,
                previous_heading_state,
                previous_poly_step,
            ) = _detach_temporal_state(
                hidden,
                previous_z,
                previous2_z,
                previous_velocity,
                previous_acceleration,
                previous_heading_state,
                previous_poly_step,
            )
            chunk_loss = None
            chunk_count = 0

    capture_pct = float(np.mean(captures) * 100.0) if captures else 0.0
    return losses, capture_pct


def train_multirate(
    visual,
    cache,
    route,
    device,
    epochs,
    patience_limit,
    resume=False,
):
    stride = int(config.TEMPORAL_EXTRA_A_STRIDE)
    full_gt_state = b.build_gt_route_state(cache, route)
    split = b.split_ranges(len(cache))
    train_start, train_end = map(int, split["train"])
    val_range = tuple(map(int, split["val"]))

    stride_indices = np.arange(train_start, train_end, stride, dtype=np.int64)
    stride_cache = _slice_cache(cache, stride_indices, f"route_A_stride{stride}")
    stride_gt_state = b.build_gt_route_state(stride_cache, route)

    native_train_cache = _slice_cache(
        cache,
        np.arange(train_start, train_end, dtype=np.int64),
        "route_A_native_train",
    )
    native_stats = _step_stats(native_train_cache)
    stride_stats = _step_stats(stride_cache)

    print("=" * 100, flush=True)
    print("Route-A multi-rate temporal training: image-aligned single-MS flow v9", flush=True)
    print(
        f"native A train: frames={train_end-train_start} "
        f"mean_step={native_stats['mean']:.3f}m/frame "
        f"median={native_stats['median']:.3f}m/frame",
        flush=True,
    )
    print(
        f"stride-{stride} A train: frames={len(stride_cache)} "
        f"mean_step={stride_stats['mean']:.3f}m/frame "
        f"median={stride_stats['median']:.3f}m/frame",
        flush=True,
    )
    start_state = np.asarray(full_gt_state["se"][0], dtype=np.float64)
    print(
        f"known route-start KF initialization: s={start_state[0]:.3f}m e={start_state[1]:.3f}m",
        flush=True,
    )
    print(
        f"validation=closed-loop original Route-A native-rate split {val_range}; "
        "Route-B/C remain evaluation only",
        flush=True,
    )
    print(
        "training-only search-center curriculum: "
        f"warmup={config.IMAGE_FLOW_TEACHER_WARMUP_EPOCHS} epochs, "
        f"decay={config.IMAGE_FLOW_TEACHER_DECAY_EPOCHS} epochs, "
        f"final_reference_ratio={config.IMAGE_FLOW_TEACHER_FINAL_RATIO:.2f}",
        flush=True,
    )
    print("=" * 100, flush=True)

    model = ThreeFrameRouteStateGRU().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config.TEMPORAL_LR),
        weight_decay=float(config.TEMPORAL_WEIGHT_DECAY),
    )

    start_epoch = 1
    best_score = float("inf")
    best_state = None
    patience = 0

    if resume and config.LATEST_TEMPORAL_CHECKPOINT.exists():
        payload = torch.load(config.LATEST_TEMPORAL_CHECKPOINT, map_location="cpu")
        if payload.get("architecture") != ARCHITECTURE_NAME:
            raise RuntimeError(
                f"resume architecture mismatch: {payload.get('architecture')} != {ARCHITECTURE_NAME}"
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload.get("best_score", best_score))
        best_state = payload.get("best_model")
        patience = int(payload.get("patience", 0))

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, int(epochs) + 1):
        model.train()
        teacher_ratio = _teacher_ratio(epoch)

        native_losses, native_capture = _train_one_sequence(
            model=model,
            optimizer=optimizer,
            params=params,
            visual=visual,
            cache=cache,
            route=route,
            gt_state=full_gt_state,
            indices=range(train_start, train_end),
            device=device,
            epoch=epoch,
        )

        stride_losses, stride_capture = _train_one_sequence(
            model=model,
            optimizer=optimizer,
            params=params,
            visual=visual,
            cache=stride_cache,
            route=route,
            gt_state=stride_gt_state,
            indices=range(len(stride_cache)),
            device=device,
            epoch=epoch,
        )

        val = rt.evaluate_closed_loop(
            model,
            visual,
            cache,
            route,
            full_gt_state,
            val_range,
            device,
        )
        score = float(val["score"])
        improved = score < best_score - float(config.EARLY_STOP_MIN_DELTA)

        training_metadata = {
            "protocol": str(config.TEMPORAL_TRAINING_PROTOCOL),
            "architecture_flow": (
                "inference: previous final KF state -> one local MS -> GRU measurement/R; "
                "previous recurrent motion/heading -> polynomial -> KF predict; "
                "KF update is the only prior/measurement fusion"
            ),
            "known_start_initialization": True,
            "training_search_center": (
                "scheduled previous-reference -> previous-KF search-center curriculum; "
                "validation/evaluation are fully previous-KF closed-loop"
            ),
            "teacher_ratio": float(teacher_ratio),
            "native_train_capture_pct": float(native_capture),
            "stride_train_capture_pct": float(stride_capture),
            "native_train_frames": int(train_end - train_start),
            "stride": stride,
            "stride_train_frames": int(len(stride_cache)),
            "native_mean_step_m": float(native_stats["mean"]),
            "stride_mean_step_m": float(stride_stats["mean"]),
            "validation": "Route-A native-rate closed-loop validation split",
            "evaluation_routes": ["route_B", "route_C"],
        }

        if improved:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
            torch.save(
                {
                    "architecture": ARCHITECTURE_NAME,
                    "model": best_state,
                    "epoch": epoch,
                    "validation": val,
                    "train_forward_rows": int(config.FORWARD_SEARCH_ROWS),
                    "training_protocol": training_metadata,
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
                "training_protocol": training_metadata,
            },
            config.LATEST_TEMPORAL_CHECKPOINT,
        )

        native_loss = float(np.mean(native_losses)) if native_losses else float("nan")
        stride_loss = float(np.mean(stride_losses)) if stride_losses else float("nan")
        print(
            f"multirate-v9 epoch={epoch:03d}/{epochs} "
            f"teacher={teacher_ratio:.3f} "
            f"native_loss={native_loss:.5f} native_capture={native_capture:.2f}% "
            f"stride{stride}_loss={stride_loss:.5f} stride_capture={stride_capture:.2f}% "
            f"val_mle={val['mle']:.3f}m val_p90={val['p90']:.3f}m "
            f"val_capture={val['capture_pct']:.2f}% "
            f"score={score:.3f} best={best_score:.3f} "
            f"patience={patience}/{patience_limit}",
            flush=True,
        )

        if (
            epoch >= int(config.EARLY_STOP_MIN_EPOCH)
            and patience >= int(patience_limit)
        ):
            break

    if best_state is None:
        raise RuntimeError("multi-rate temporal training did not produce a checkpoint")

    model.load_state_dict(best_state)
    return model, best_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-epochs", type=int, default=int(config.TEMPORAL_EPOCHS))
    parser.add_argument("--patience", type=int, default=int(config.EARLY_STOP_PATIENCE))
    parser.add_argument("--jitter-m", type=float, default=float(config.LOCAL_PRIOR_JITTER_M))
    parser.add_argument("--forward-rows", type=int, choices=[3, 4, 5, 6], default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--train-visual-if-missing",
        action="store_true",
        help="train Route-A visual retrieval only when the visual checkpoint is missing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rt._set_forward_rows(args.forward_rows)
    config.LOCAL_PRIOR_JITTER_M = float(args.jitter_m)
    config.CONTROLLED_GT_PRIOR_JITTER_M = float(args.jitter_m)
    rt.set_seed(config.SEED)
    device = rt.resolve_device()

    if not config.VISUAL_CHECKPOINT.exists():
        if not args.train_visual_if_missing:
            raise FileNotFoundError(
                f"visual checkpoint not found: {config.VISUAL_CHECKPOINT}; "
                "run visual training first or pass --train-visual-if-missing"
            )
        train_visual_retrieval_a_only(
            device=device,
            epochs=int(config.VISUAL_EPOCHS),
            jitter_m=float(args.jitter_m),
            resume=False,
        )

    visual = FrozenVisualLocalizer(device)
    cache = rt.build_route_cache("route_A", config.ROUTE_ROOTS[0], visual, device)
    route = rt.WaypointRoute(
        rt.load_waypoint_xy("route_A", visual.origin_lat, visual.origin_lon)
    )

    _, score = train_multirate(
        visual=visual,
        cache=cache,
        route=route,
        device=device,
        epochs=int(args.temporal_epochs),
        patience_limit=int(args.patience),
        resume=bool(args.resume),
    )

    print(f"best validation score={score:.3f}", flush=True)
    print(f"best checkpoint={config.TEMPORAL_CHECKPOINT}", flush=True)
    print("evaluation routes remain Route-B and Route-C", flush=True)


if __name__ == "__main__":
    main()
