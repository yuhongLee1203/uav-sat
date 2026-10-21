"""Causal route training/evaluation for the v39 MobileNetV3-Small pipeline.

The frame GT may supervise a training loss, but the route search, recurrent
state and Kalman filter always consume only images, map and prior predictions.
Legacy shared checkpoints are GT-prior-trained and not a strict initialization.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runtime"))
import config
import data
import robust_tracker as tr
from visual_localizer import FrozenVisualLocalizer
from visual_model import ThreeFrameRouteStateGRU

DESIGNS = ("visual_pseudo", "temporal_cycle", "confidence_gated")
BASE = Path(os.environ.get(
    "V40_BASE_CITYA", "/yh/study/uav-sat/v39_otherdata/output/abcd_shared_20260922_121010/citya"
)).resolve()
PREPARED = BASE / "prepared"
BASE_CHECKPOINTS = BASE / "train_frames3" / "checkpoints"
BASE_CACHE = BASE / "feature_cache"
VISUAL_NAME = "visual_retrieval_A_only.pt"
TEMPORAL_NAME = "controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"


def configure(city="citya"):
    global PREPARED, BASE_CHECKPOINTS, BASE_CACHE
    if config.BACKBONE_KEY != "mobilenet_v3_small":
        raise RuntimeError("v39 architecture requires UAVSAT_BACKBONE=mobilenet_v3_small")
    if city not in ("citya", "cityb", "cityc", "cityd"):
        raise ValueError("city must be citya, cityb, cityc or cityd")
    city_base = BASE.parent / city
    prepared_override = os.environ.get("V40_PREPARED_ROOT")
    PREPARED = (Path(prepared_override).resolve() / city / "prepared"
                if prepared_override else city_base / "prepared")
    BASE_CHECKPOINTS = city_base / "train_frames3" / "checkpoints"
    BASE_CACHE = city_base / "feature_cache"
    exp = json.loads((PREPARED / "experiment.json").read_text())
    config.SAT_IMAGE = Path(exp["satellite_image"])
    config.SAT_JSON = PREPARED / "bearing_satellite.json"
    config.ROUTE_ROOTS = [PREPARED / "routes" / x for x in ("train_01", "test_01", "test_02")]
    config.ROUTE_NAMES = ["route_A", "route_B", "route_C"]
    config.WAYPOINT_FILES = dict(zip(config.ROUTE_NAMES, [r / "waypoints.json" for r in config.ROUTE_ROOTS]))
    mpp = float(exp["mpp"])
    official96_temporal = str(exp.get("sequence_selection_version", "")).startswith(
        "official_rsi96_seed42_")
    config.SAT_STRIDE = round(4.48 / mpp)
    crop_m = float(os.environ.get("V40_SAT_CROP_SIZE_M", "44.8"))
    if crop_m <= 0:
        raise ValueError("V40_SAT_CROP_SIZE_M must be positive")
    config.SAT_CROP_SIZE = 2 * round((crop_m / mpp) / 2)
    config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M = float(os.environ.get(
        "V40_FORWARD_SEARCH_ORIGIN_BACKSHIFT_M",
        str(8.0 + 0.5 * config.SAT_STRIDE * mpp)))
    config.INIT_FORWARD_SPEED_M_PER_FRAME = float(os.environ.get(
        "V40_INIT_FORWARD_SPEED_M_PER_FRAME", "8.0" if official96_temporal else "4.0"))
    config.MAX_FORWARD_SPEED_M_PER_FRAME = float(os.environ.get(
        "V40_MAX_FORWARD_SPEED_M_PER_FRAME", "50.0" if official96_temporal else "14.0"))
    config.MAX_POLYNOMIAL_STEP_M_PER_FRAME = float(os.environ.get(
        "V40_MAX_POLYNOMIAL_STEP_M_PER_FRAME", "50.0" if official96_temporal else "14.0"))
    config.MAX_CROSS_SPEED_M_PER_FRAME = float(os.environ.get(
        "V40_MAX_CROSS_SPEED_M_PER_FRAME", "50.0" if official96_temporal else "3.0"))
    config.MAX_FINAL_CROSS_TRACK_M = float(os.environ.get(
        "V40_MAX_FINAL_CROSS_TRACK_M", "70.0" if official96_temporal else "10.0"))
    config.KALMAN_FINAL_STEP_MAX_M = float(os.environ.get(
        "V40_KALMAN_FINAL_STEP_MAX_M", "50.0" if official96_temporal else "14.0"))
    config.NO_GT_VISUAL_RELAX_EXTRA_M = float(os.environ.get(
        "V40_NO_GT_VISUAL_RELAX_EXTRA_M", "40.0" if official96_temporal else "10.0"))
    config.MAX_FORWARD_ACCEL_M_PER_FRAME2 = float(os.environ.get(
        "V40_MAX_FORWARD_ACCEL_M_PER_FRAME2", "20.0" if official96_temporal else "5.0"))
    config.MAX_MOTION_VELOCITY_DELTA_M_PER_FRAME = float(os.environ.get(
        "V40_MAX_MOTION_VELOCITY_DELTA_M_PER_FRAME", "8.0" if official96_temporal else "2.0"))
    config.MAX_POLYNOMIAL_STEP_DELTA_M_PER_FRAME = float(os.environ.get(
        "V40_MAX_POLYNOMIAL_STEP_DELTA_M_PER_FRAME", "12.0" if official96_temporal else "2.5"))
    config.ROUTE_PROJECTION_SWITCH_RADIUS_M = float(os.environ.get(
        "V40_ROUTE_PROJECTION_SWITCH_RADIUS_M", "40.0" if official96_temporal else "24.0"))
    config.ROUTE_PROJECTION_PROGRESS_STEP_FACTOR = float(os.environ.get(
        "V40_ROUTE_PROJECTION_PROGRESS_STEP_FACTOR", "3.0" if official96_temporal else "1.75"))
    config.ROUTE_REFERENCE_ONLY = True
    config.NO_GT_INFERENCE = True
    config.REFERENCE_PROTOCOL = "route_reference"
    # Explicit validation oracle only. Ordinary inference defaults to False.
    config.GT_FRONT18_ORACLE = os.environ.get("V40_GT_FRONT18_ORACLE", "0") == "1"
    config.GT_FRONT18_PATCH_ORACLE = os.environ.get("V40_GT_FRONT18_PATCH_ORACLE", "0") == "1"
    config.GT_FINAL_MS_ORACLE_CENTER = os.environ.get("V40_GT_FINAL_MS_ORACLE_CENTER", "0") == "1"
    config.GT_FINAL_MS_PATCH_ORACLE = os.environ.get("V40_GT_FINAL_MS_PATCH_ORACLE", "0") == "1"
    config.GT_JITTER_EVAL_M = float(os.environ.get("V40_GT_JITTER_EVAL_M", "0"))
    if config.GT_JITTER_EVAL_M < 0:
        raise ValueError("V40_GT_JITTER_EVAL_M must be nonnegative")
    config.GT_JITTER_PATCH_SELECT = os.environ.get("V40_GT_JITTER_PATCH_SELECT", "0") == "1"
    config.CONTROLLED_GT_PRIOR_JITTER_M = config.GT_JITTER_EVAL_M or 8.0
    config.IMAGE_GUIDED_FRONT18 = os.environ.get("V40_IMAGE_GUIDED_FRONT18", "0") == "1"
    config.IMAGE_GUIDED_EXTRA_HYPOTHESIS = os.environ.get("V40_IMAGE_GUIDED_EXTRA_HYPOTHESIS", "0") == "1"
    # A wide +/-60 m bank let visually repetitive but wrong locations win by
    # frame 1. Use several genuinely local hypotheses around the causal state;
    # this does not require current-frame GT or known episode duration.
    config.ACQ_HYPOTHESIS_COUNT = int(os.environ.get("V40_ACQ_HYPOTHESES", "7"))
    config.ROUTE_REFERENCE_BANK_RADIUS_M = float(os.environ.get(
        "V40_ACQ_BANK_RADIUS_M", "60.0" if official96_temporal else "15.0"))
    cross_default = "-40,-20,0,20,40" if official96_temporal else "0"
    config.ACQ_CROSS_OFFSETS_M = tuple(float(value) for value in os.environ.get(
        "V40_ACQ_CROSS_OFFSETS_M", cross_default).split(","))
    config.ROUTE_REFERENCE_PREDICTED_CROSS = os.environ.get(
        "V40_ROUTE_REFERENCE_PREDICTED_CROSS", "1" if official96_temporal else "0") == "1"
    config.ACQ_RAW_VISUAL_EVIDENCE_WEIGHT = float(os.environ.get(
        "V40_ACQ_RAW_VISUAL_WEIGHT", "2.0" if official96_temporal else "0.0"))
    config.ACQ_HYPOTHESIS_MOTION_PRIOR_WEIGHT = float(os.environ.get(
        "V40_ACQ_MOTION_PRIOR_WEIGHT", "0.2" if official96_temporal else "2.0"))
    if (config.ACQ_HYPOTHESIS_COUNT < 1 or config.ROUTE_REFERENCE_BANK_RADIUS_M <= 0
            or not config.ACQ_CROSS_OFFSETS_M):
        raise ValueError("invalid causal acquisition bank")
    config.NO_GT_MAX_BACKWARD_CORRECTION_M = float(os.environ.get(
        "V40_NO_GT_MAX_BACKWARD_CORRECTION_M", "10.0" if official96_temporal else "1.0"))
    # Expose Kalman *parameters* for validation-only sweeps. The v39 filter
    # equations and model modules remain unchanged.
    for setting, environment in (
        ("KALMAN_R_MIN_VAR", "V40_KALMAN_R_MIN_VAR"),
        ("KALMAN_R_MAX_VAR", "V40_KALMAN_R_MAX_VAR"),
        ("KALMAN_NIS_MAX_R_SCALE", "V40_KALMAN_NIS_MAX_R_SCALE"),
        ("KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME", "V40_KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME"),
    ):
        if environment in os.environ:
            setattr(config, setting, float(os.environ[environment]))
    config.EXPERIMENT_FRAME_COUNT = 3
    config.FORWARD_ONLY_LOCAL_SEARCH = True
    cache_override = os.environ.get("V40_FEATURE_CACHE_ROOT")
    config.FEATURE_CACHE_DIR = (Path(cache_override).resolve() / city
                                if cache_override else BASE_CACHE)
    config.CHECKPOINT_DIR = BASE_CHECKPOINTS
    config.VISUAL_CHECKPOINT = BASE_CHECKPOINTS / VISUAL_NAME
    visual_root = os.environ.get("V40_VISUAL_CHECKPOINT_ROOT")
    if visual_root:
        config.VISUAL_CHECKPOINT = Path(visual_root).resolve() / city / VISUAL_NAME
    visual_file = os.environ.get("V40_VISUAL_CHECKPOINT_FILE")
    if visual_file:
        config.VISUAL_CHECKPOINT = Path(visual_file).resolve()
    config.TEMPORAL_CHECKPOINT = BASE_CHECKPOINTS / TEMPORAL_NAME
    config.TEMPORAL_FUSION_MLP = os.environ.get("V40_TEMPORAL_FUSION_MLP", "0") == "1"
    config.ARCHITECTURE_NAME = "ICLR_Bearing4City_Forward18SoftMS_ResidualTemporalGRU_Kalman_FinalMS"
    if config.TEMPORAL_FUSION_MLP:
        config.ARCHITECTURE_NAME += "_MeanDiffConcatSingleMLP"
    if not all(p.exists() for p in (config.SAT_IMAGE, config.VISUAL_CHECKPOINT,
                                    config.TEMPORAL_CHECKPOINT, *config.ROUTE_ROOTS)):
        raise FileNotFoundError("Prepared routes or pretrained checkpoints are missing under " + str(city_base))


def verify_visual_checkpoint_split():
    """Reject a visual checkpoint trained against a different fixed-route test set."""
    experiment = json.loads((PREPARED / "experiment.json").read_text())
    if experiment.get("sequence_selection_version") != (
            "official_rsi96_seed42_fixed_waypoint_temporal_v1"):
        return
    paths = set()
    for route_root in config.ROUTE_ROOTS[1:3]:
        with (route_root / "manifest.csv").open(newline="") as file:
            paths.update(row["image_path"] for row in csv.DictReader(file))
    expected = hashlib.sha256("\n".join(sorted(paths)).encode()).hexdigest()
    checkpoint = torch.load(config.VISUAL_CHECKPOINT, map_location="cpu")
    if (checkpoint.get("training_protocol") not in (
            "global_gallery_gt_loss_only", "global_gallery_gt_selected_train_patches")
            or checkpoint.get("excluded_test_manifest_sha256") != expected
            or checkpoint.get("official96_train_only") is not True
            or checkpoint.get("previous_task_checkpoint_loaded") is not False):
        raise RuntimeError("visual checkpoint is not audited for this fixed-route test split: "
                           + str(config.VISUAL_CHECKPOINT))


def correction_bounds():
    return [float(config.MAX_MEASUREMENT_CORRECTION_PARALLEL_M),
            float(config.MAX_MEASUREMENT_CORRECTION_CROSS_M)]


def verify_temporal_correction_bounds(checkpoint):
    saved = checkpoint.get("runtime_correction_bounds_m")
    if saved is not None and saved != correction_bounds():
        raise RuntimeError("temporal correction bounds differ from training: saved=%s current=%s; "
                           "set UAVSAT_CORR_PARALLEL_M and UAVSAT_CORR_CROSS_M" % (
                               saved, correction_bounds()))


def read_unlabeled_manifest():
    with (config.ROUTE_ROOTS[0] / "manifest.csv").open(newline="") as file:
        rows = [{"frame_id": int(r["frame_id"]), "image_path": r["image_path"]}
                for r in csv.DictReader(file)]
    if len(rows) < 3 or len(set(r["frame_id"] for r in rows)) != len(rows):
        raise RuntimeError("City A train manifest has too few or duplicate frames")
    return rows


def prepare_cache(visual, device, output):
    rows = read_unlabeled_manifest()
    cache_path = output / "train_image_only_clip.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu")
        if (cache.get("image_paths") == [r["image_path"] for r in rows]
                and cache.get("visual_checkpoint") == str(config.VISUAL_CHECKPOINT)
                and cache.get("backbone_state_sha256", "imagenet")
                == visual.backbone_state_sha256):
            print("[V40 CACHE] reuse image-only train cache", flush=True)
            return cache
    source = BASE_CACHE / "route_A_uav_clip.pt"
    if source.exists():
        old = torch.load(source, map_location="cpu")
        if (visual.backbone_state_sha256 == "imagenet"
                and old.get("image_paths") == [r["image_path"] for r in rows]
                and old.get("uav_clip") is not None
                and old["uav_clip"].shape[-1] == config.CLIP_DIM
                and old.get("signature", {}).get("backbone") == config.BACKBONE_KEY):
            clips = old["uav_clip"]
            print("[V40 CACHE] reused prior UAV backbone features; GT discarded", flush=True)
        else:
            clips = None
    else:
        clips = None
    if clips is None:
        transform = data.image_transform(False, source="uav")
        features = []
        for start in range(0, len(rows), 32):
            images = torch.stack([transform(Image.open(r["image_path"]).convert("RGB"))
                                  for r in rows[start:start + 32]]).to(device)
            with torch.no_grad():
                features.append(visual.encode_uav_clip(images).cpu().half())
        clips = torch.cat(features)
        print("[V40 CACHE] built image-only train cache", flush=True)
    cache = {"uav_clip": clips, "image_paths": [r["image_path"] for r in rows],
             "frame_ids": [r["frame_id"] for r in rows],
             "visual_checkpoint": str(config.VISUAL_CHECKPOINT),
             "backbone_state_sha256": visual.backbone_state_sha256}
    output.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + "." + str(os.getpid()) + ".tmp")
    torch.save(cache, temporary)
    os.replace(temporary, cache_path)
    return cache


def new_model(device):
    checkpoint = torch.load(config.TEMPORAL_CHECKPOINT, map_location="cpu")
    if checkpoint.get("architecture") != config.ARCHITECTURE_NAME:
        raise RuntimeError("Temporal checkpoint architecture mismatch")
    model = ThreeFrameRouteStateGRU().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def unlabeled_loss(design, observation, output, previous_anchor, previous_step):
    # Satellite/UAV similarity is an image-derived pseudo-label, not a frame GT.
    peak = observation.hypothesis_raw_peak.detach()
    target = torch.softmax((peak - peak.mean()) / peak.std(unbiased=False).clamp_min(1e-3), dim=0)
    acquisition = F.kl_div(F.log_softmax(observation.acquisition_logits, dim=0), target,
                           reduction="sum")
    anchor = observation.anchor_se.detach()
    correction = F.smooth_l1_loss(output.measurement_se, anchor)
    loss = acquisition + 0.1 * correction
    if design != "visual_pseudo" and previous_anchor is not None:
        # Image-derived displacement is noisy: clip it and never call it GT.
        delta = anchor - previous_anchor
        delta[:, 0] = delta[:, 0].clamp(0.0, 14.0)
        delta[:, 1] = delta[:, 1].clamp(-5.0, 5.0)
        motion = F.smooth_l1_loss(output.next_step_se, delta)
        loss = loss + (0.25 if design == "temporal_cycle" else 0.10) * motion
        if previous_step is not None:
            loss = loss + 0.05 * F.smooth_l1_loss(previous_step, delta)
    if design == "confidence_gated":
        confidence = target.max().detach()
        loss = confidence * loss + 0.01 * output.acceleration_se.square().mean()
    return loss


def route_xy_from_se_torch(route, se):
    """Differentiable equivalent of the unchanged runtime route.xy_from_se."""
    kwargs = {"dtype": se.dtype, "device": se.device}
    points = torch.as_tensor(route.points, **kwargs)
    units = torch.as_tensor(route.units, **kwargs)
    cumulative = torch.as_tensor(route.cumulative_s, **kwargs)
    s = se[:, 0].clamp(0.0, route.total_length_m)
    leg = (torch.bucketize(s.contiguous(), cumulative, right=True) - 1).clamp(
        0, len(route.units) - 1)
    unit = units[leg]
    center = points[leg] + (s - cumulative[leg])[:, None] * unit
    radius = float(getattr(config, "ROUTE_FRAME_SMOOTH_RADIUS_M", 0.0))
    if radius > 1e-6 and bool(getattr(config, "ROUTE_FRAME_POSTTURN_ONLY", True)):
        alpha = ((s - cumulative[leg]) / radius).clamp(0.0, 1.0)
        alpha = alpha.square() * (3.0 - 2.0 * alpha)
        previous = units[(leg - 1).clamp_min(0)]
        blended = previous * (1.0 - alpha[:, None]) + unit * alpha[:, None]
        norm = torch.linalg.norm(blended, dim=1, keepdim=True)
        unit = torch.where(norm > 1e-9, blended / norm.clamp_min(1e-9), unit)
    cross = torch.stack((-unit[:, 1], unit[:, 0]), dim=1)
    return center + se[:, 1:2] * cross


def labeled_loss(observation, output, target_xy, target_step_xy, visual, route,
                 predicted_progress_s, device, loss_weights=(1.0, 2.0, 2.0), dense_offbank=False,
                 acquisition_target_sigma_m=0.0):
    """Training-only labels; the selector and Kalman receive no GT."""
    xy = torch.as_tensor(target_xy, dtype=torch.float32, device=device).reshape(1, 2)
    displacement = torch.as_tensor(target_step_xy, dtype=torch.float32, device=device).reshape(1, 2)
    # The basis comes from the model state and predeclared route. Raw GT is
    # used only for labels, without monotonic projection or waypoint switching.
    basis = torch.as_tensor(np.stack((route.smooth_route_unit(predicted_progress_s),
                                     route.smooth_route_cross(predicted_progress_s))),
                            dtype=torch.float32, device=device)
    step = displacement @ basis.T
    centers = visual.gallery["xy"][observation.candidate_indices_all]
    distance = torch.linalg.norm(centers - xy[:, None, :], dim=2).min(dim=1).values
    chosen = observation.acquisition_selected_index
    if dense_offbank or distance.min().item() <= float(config.CANDIDATE_CAPTURE_RADIUS_M):
        logits = observation.acquisition_logits.reshape(-1)
        if acquisition_target_sigma_m > 0:
            # Overlapping windows can contain the same correct SAT patch.
            # Give them equal credit instead of an arbitrary argmin tie-break.
            soft_target = torch.softmax(-distance.square() /
                                        (2.0 * acquisition_target_sigma_m ** 2), dim=0)
            acquisition = -(soft_target * F.log_softmax(logits, dim=0)).sum()
        else:
            acquisition = F.cross_entropy(logits.reshape(1, -1), distance.argmin().reshape(1))
    else:
        acquisition = output.next_step_se.sum() * 0.0
    measurement_xy = route_xy_from_se_torch(route, output.measurement_se)
    measurement = (F.smooth_l1_loss(measurement_xy, xy)
                   if dense_offbank or distance[chosen].item() <= float(config.CANDIDATE_CAPTURE_RADIUS_M)
                   else output.next_step_se.sum() * 0.0)
    motion = F.smooth_l1_loss(output.next_step_se, step)
    return (float(loss_weights[0]) * acquisition
            + float(loss_weights[1]) * measurement
            + float(loss_weights[2]) * motion)


def train_epoch(design, model, visual, cache, route, optimizer, device, max_frames,
                targets=None, loss_weights=(1.0, 2.0, 2.0), dense_offbank=False,
                acquisition_target_sigma_m=0.0):
    model.train()
    # Optional training labels are supervised targets only; each route is a new episode.
    kf = tr.RouteKalman(0.0, 0.0)
    hidden = previous_z = previous2_z = previous_measurement = previous_anchor = previous_step = None
    speed = torch.tensor([[config.INIT_FORWARD_SPEED_M_PER_FRAME, 0.0]], device=device)
    accel = torch.zeros(1, 2, device=device)
    heading = torch.zeros(1, 2, device=device)
    polynomial = speed.clone()
    confidence = float(config.ACQ_INITIAL_CONFIDENCE)
    transform = data.image_transform(False, source="uav")
    losses = []
    total_frames = min(max_frames, len(cache["image_paths"]))
    started = time.monotonic()
    for index in range(total_frames):
        # Current image always traverses the frozen backbone; t-1 and t-2 use cache.
        with Image.open(cache["image_paths"][index]) as image:
            current = transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            clip = visual.encode_uav_clip(current)
            z = visual.model.encode_uav_from_clip(clip)
            if index > 0:
                previous_z = visual.model.encode_uav_from_clip(
                    cache["uav_clip"][index - 1:index].to(device).float())
            if index > 1:
                previous2_z = visual.model.encode_uav_from_clip(
                    cache["uav_clip"][index - 2:index - 1].to(device).float())
        predicted = (kf.se() if index == 0 else kf.predict(
            speed[0].detach().cpu().numpy(), accel[0].detach().cpu().numpy(),
            route.total_length_m, polynomial_step_se=polynomial[0].detach().cpu().numpy()))
        center = np.asarray([predicted[0], predicted[1] if
            config.ROUTE_REFERENCE_PREDICTED_CROSS else 0.0], dtype=np.float64)
        search_heading = tr.wrap_angle_rad(route.route_heading_rad(float(predicted[0]))
                                           + float(heading[0, 0].item()))
        observation = tr.visual_observation(
            model, visual, clip, center, route, predicted, previous_z, previous2_z,
            hidden, confidence, kf.progress_std(), float(speed[0, 0].item()),
            search_heading, device,
            gt_xy=None, gt_se=None, teacher_select=False)
        output = tr.model_forward(
            model, observation, previous_z, previous2_z, predicted,
            previous_measurement, speed, accel, heading, polynomial,
            route, hidden, device)
        if targets is None:
            loss = unlabeled_loss(design, observation, output, previous_anchor, previous_step)
        else:
            loss = labeled_loss(
                observation, output, targets["xy"][index],
                targets["step_xy"][index], visual, route, float(predicted[0]), device,
                loss_weights=loss_weights, dense_offbank=dense_offbank,
                acquisition_target_sigma_m=acquisition_target_sigma_m)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite image-only loss at frame %d" % index)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        # Kalman is a non-differentiable causal decoder; keep its state within
        # this route but truncate the GRU graph after each optimization step.
        confidence = tr.visual_confidence_from_observation(observation)
        kf.update(output.measurement_se[0].detach().cpu().numpy(),
                  output.measurement_variance_se[0].detach().cpu().numpy(),
                  route.total_length_m, acquisition_confidence=confidence)
        speed, accel, polynomial = tr.stabilize_motion_state(
            speed, accel, polynomial, output.velocity_se, output.acceleration_se,
            output.next_step_se)
        heading = tr.stabilize_heading_state(heading, output.heading_residual_rad,
                                            output.turn_rate_rad)
        hidden = output.hidden.detach()
        previous_measurement = output.measurement_se.detach()
        previous_anchor = observation.anchor_se.detach()
        previous_step = output.next_step_se.detach()
        if (index + 1) % 10 == 0 or index + 1 == total_frames:
            done = index + 1
            elapsed = time.monotonic() - started
            eta = elapsed * (total_frames - done) / done
            print("[V40 FRAME] %d/%d (%.1f%%) loss=%.4f elapsed=%.1fs eta=%.1fs" % (
                done, total_frames, 100.0 * done / total_frames,
                np.mean(losses[-10:]), elapsed, eta), flush=True)
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", choices=DESIGNS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-frames", type=int, default=136)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configure()
    config.OUTPUT_DIR = args.output.resolve()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    visual = FrozenVisualLocalizer(device)
    cache = prepare_cache(visual, device, args.output.parent)
    model = new_model(device)
    route = tr.WaypointRoute(tr.load_waypoint_xy("route_A", visual.origin_lat, visual.origin_lon))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    epochs = 1 if args.smoke else args.epochs
    max_frames = 3 if args.smoke else args.max_train_frames
    history = []
    for epoch in range(epochs):
        value = train_epoch(args.design, model, visual, cache, route, optimizer, device, max_frames)
        history.append(value)
        print("[%s] epoch=%d image_only_loss=%.5f" % (args.design, epoch + 1, value), flush=True)
    checkpoint = args.output / "temporal_image_only.pt"
    torch.save({"model": model.state_dict(), "architecture": config.ARCHITECTURE_NAME,
                "training_labels": "none", "warm_start": str(config.TEMPORAL_CHECKPOINT),
                "design": args.design, "loss_history": history}, checkpoint)
    result = {"design": args.design, "training_labels": "none", "warm_start_supervised": True,
              "warm_start_scope": "ABCD_shared_city_episodic",
              "epochs": epochs, "train_frames": max_frames, "image_only_loss": history,
              "checkpoint": str(checkpoint)}
    if not args.smoke:
        model.eval()
        # Inference also sends the current UAV image through the backbone;
        # the old route cache supplies only past features and metric GT.
        config.MEASURE_END_TO_END_LATENCY = True
        for name in ("route_B", "route_C"):
            index = config.ROUTE_NAMES.index(name)
            eval_route = tr.WaypointRoute(tr.load_waypoint_xy(name, visual.origin_lat, visual.origin_lon))
            eval_cache = tr.build_route_cache(name, config.ROUTE_ROOTS[index], visual, device)
            summary = tr.run_route_inference(name, visual, model, eval_cache, eval_route, device)
            result[name] = summary
    (args.output / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print("[V40 RESULT]", args.output / "result.json", flush=True)


if __name__ == "__main__":
    main()
