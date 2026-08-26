import os
from pathlib import Path
from config_base import *

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Backbone / experiment identity
# ---------------------------------------------------------------------------
BACKBONE_KEY = os.environ.get("UAVSAT_BACKBONE", "mobilenet_v3_small").strip().lower()
if BACKBONE_KEY not in BACKBONE_SPECS:
    raise ValueError(
        "UAVSAT_BACKBONE must be one of %s; got %r"
        % (sorted(BACKBONE_SPECS), BACKBONE_KEY)
    )
BACKBONE_NAME, CLIP_DIM = BACKBONE_SPECS[BACKBONE_KEY]

EXPERIMENT_FRAME_COUNT = int(os.environ.get("UAVSAT_EXPERIMENT_FRAME_COUNT", "2"))
if EXPERIMENT_FRAME_COUNT not in (1, 2):
    raise ValueError("UAVSAT_EXPERIMENT_FRAME_COUNT must be 1 or 2 for v36_byTeacher")
TEMPORAL_WINDOW_FRAMES = EXPERIMENT_FRAME_COUNT

# Legacy multi-rate metadata is retained for reproducibility of the old A-only run.
TEMPORAL_EXTRA_A_STRIDE = int(
    os.environ.get("UAVSAT_TEMPORAL_EXTRA_A_STRIDE", "2")
)
if TEMPORAL_EXTRA_A_STRIDE < 2:
    raise ValueError("UAVSAT_TEMPORAL_EXTRA_A_STRIDE must be >= 2")
TEMPORAL_TRAINING_PROTOCOL = f"routeA_native_plus_stride{TEMPORAL_EXTRA_A_STRIDE}"

# Pure-model mode is the default. The old manual limits remain available behind
# UAVSAT_MANUAL_CONSTRAINTS=1 and are not deleted.
MANUAL_DYNAMICS_CONSTRAINTS = (
    os.environ.get("UAVSAT_MANUAL_CONSTRAINTS", "0").strip() == "1"
)
PURE_MODEL_DYNAMICS = not MANUAL_DYNAMICS_CONSTRAINTS
ZERO_KINEMATIC_INITIALIZATION = True
DYNAMICS_TAG = "manual" if MANUAL_DYNAMICS_CONSTRAINTS else "puremodel"

# ---------------------------------------------------------------------------
# Temporal route protocol
# ---------------------------------------------------------------------------
# Default experiment requested here: full Route B then full Route C per epoch,
# resetting recurrent/Kalman state at the route boundary, and validating on the
# full Route A trajectory. This mirrors the v37 B+C -> A split while keeping the
# v36_byTeacher architecture and pure-model dynamics.
BC_TO_A_TEMPORAL = os.environ.get("UAVSAT_BC_TO_A_TEMPORAL", "1").strip() == "1"
TEMPORAL_TRAIN_ROUTES = [
    row.strip()
    for row in os.environ.get(
        "UAVSAT_TEMPORAL_TRAIN_ROUTES", "route_B,route_C"
    ).split(",")
    if row.strip()
]
TEMPORAL_VALIDATION_ROUTE = os.environ.get(
    "UAVSAT_TEMPORAL_VALIDATION_ROUTE", "route_A"
).strip()
if BC_TO_A_TEMPORAL:
    if not TEMPORAL_TRAIN_ROUTES:
        raise ValueError("UAVSAT_TEMPORAL_TRAIN_ROUTES must not be empty")
    for _route_name in TEMPORAL_TRAIN_ROUTES + [TEMPORAL_VALIDATION_ROUTE]:
        if _route_name not in ROUTE_NAMES:
            raise ValueError(f"unknown temporal route: {_route_name}")
    TEMPORAL_TRAINING_PROTOCOL = (
        "train-"
        + "+".join(name.replace("route_", "") for name in TEMPORAL_TRAIN_ROUTES)
        + "_validate-"
        + TEMPORAL_VALIDATION_ROUTE.replace("route_", "")
    )

# ---------------------------------------------------------------------------
# Optional old B/C in-domain diagnostic
# ---------------------------------------------------------------------------
# Retained. Set UAVSAT_BC_TO_A_TEMPORAL=0 before using this old diagnostic.
DIAGNOSTIC_TEMPORAL_ROUTE = os.environ.get(
    "UAVSAT_DIAGNOSTIC_TEMPORAL_ROUTE", ""
).strip().upper()
if DIAGNOSTIC_TEMPORAL_ROUTE not in ("", "B", "C"):
    raise ValueError("UAVSAT_DIAGNOSTIC_TEMPORAL_ROUTE must be empty, B, or C")

if DIAGNOSTIC_TEMPORAL_ROUTE and not BC_TO_A_TEMPORAL:
    TEMPORAL_TRAIN_SOURCE_ROUTE = f"route_{DIAGNOSTIC_TEMPORAL_ROUTE}"
    _diagnostic_route_index = ROUTE_NAMES.index(TEMPORAL_TRAIN_SOURCE_ROUTE)
    _diagnostic_roots = list(ROUTE_ROOTS)
    _diagnostic_roots[0] = _diagnostic_roots[_diagnostic_route_index]
    ROUTE_ROOTS = tuple(_diagnostic_roots)
    TEMPORAL_TRAINING_PROTOCOL = (
        f"diagnostic_route{DIAGNOSTIC_TEMPORAL_ROUTE}_in_domain"
    )
else:
    TEMPORAL_TRAIN_SOURCE_ROUTE = "route_A"

# ---------------------------------------------------------------------------
# Causal predefined-route reference points
# ---------------------------------------------------------------------------
# Current-frame GT/reference position is NEVER used to choose the local search
# centre or candidate. The model/Kalman causal progress selects only the nearest
# fixed reference point that has already been passed.
REFERENCE_POINT_SPACING_M = float(
    os.environ.get("UAVSAT_REFERENCE_POINT_SPACING_M", "4.48")
)
if REFERENCE_POINT_SPACING_M <= 0.0:
    raise ValueError("UAVSAT_REFERENCE_POINT_SPACING_M must be > 0")
REFERENCE_SEARCH_GT_FREE = True
REFERENCE_SELECTION_POLICY = "nearest_already_passed_from_causal_prediction"

if BC_TO_A_TEMPORAL:
    ARCHITECTURE_NAME = (
        "V36_byTeacher_MSPreviousPosition_"
        f"{EXPERIMENT_FRAME_COUNT}Frame_{BACKBONE_KEY}_"
        "GRUVisualMeasurementVariance_NoSatContext_Polynomial_Kalman_"
        f"CausalReferenceOnly_{TEMPORAL_TRAINING_PROTOCOL}_{DYNAMICS_TAG}_v9"
    )
elif DIAGNOSTIC_TEMPORAL_ROUTE:
    ARCHITECTURE_NAME = (
        "V36_byTeacher_MSPreviousPosition_"
        f"{EXPERIMENT_FRAME_COUNT}Frame_{BACKBONE_KEY}_"
        "GRUVisualMeasurementVariance_NoSatContext_Polynomial_Kalman_"
        f"CausalReferenceOnly_DiagnosticTrainRoute{DIAGNOSTIC_TEMPORAL_ROUTE}_"
        f"{DYNAMICS_TAG}_v8diag"
    )
else:
    ARCHITECTURE_NAME = (
        "V36_byTeacher_MSPreviousPosition_"
        f"{EXPERIMENT_FRAME_COUNT}Frame_{BACKBONE_KEY}_"
        "GRUVisualMeasurementVariance_NoSatContext_Polynomial_Kalman_"
        f"CausalReferenceOnly_MultiRateAstride{TEMPORAL_EXTRA_A_STRIDE}_{DYNAMICS_TAG}_v8"
    )

BACKBONE_OUTPUT_DIR = PROJECT_ROOT / "output" / BACKBONE_KEY
CHECKPOINT_DIR = BACKBONE_OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / f"visual_retrieval_A_only_{BACKBONE_KEY}.pt"

if BC_TO_A_TEMPORAL:
    OUTPUT_DIR = BACKBONE_OUTPUT_DIR / f"{EXPERIMENT_FRAME_COUNT}frame_BCtoA"
    TEMPORAL_CHECKPOINT = (
        CHECKPOINT_DIR
        / (
            "causal_reference_only_forward3x6_ms_previous_position_"
            f"{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_"
            f"{TEMPORAL_TRAINING_PROTOCOL}_{DYNAMICS_TAG}_v9.pt"
        )
    )
    LATEST_TEMPORAL_CHECKPOINT = (
        CHECKPOINT_DIR
        / (
            "causal_reference_only_forward3x6_ms_previous_position_"
            f"{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_"
            f"{TEMPORAL_TRAINING_PROTOCOL}_{DYNAMICS_TAG}_v9_latest.pt"
        )
    )
    FEATURE_CACHE_DIR = BACKBONE_OUTPUT_DIR / "feature_cache_BCtoA"
elif DIAGNOSTIC_TEMPORAL_ROUTE:
    _diag_lower = DIAGNOSTIC_TEMPORAL_ROUTE.lower()
    OUTPUT_DIR = (
        BACKBONE_OUTPUT_DIR
        / f"{EXPERIMENT_FRAME_COUNT}frame_diag_route_{_diag_lower}"
    )
    TEMPORAL_CHECKPOINT = (
        CHECKPOINT_DIR
        / (
            "diagnostic_causal_reference_only_forward3x6_ms_previous_position_"
            f"{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_"
            f"train_route_{_diag_lower}_{DYNAMICS_TAG}_v8diag.pt"
        )
    )
    LATEST_TEMPORAL_CHECKPOINT = (
        CHECKPOINT_DIR
        / (
            "diagnostic_causal_reference_only_forward3x6_ms_previous_position_"
            f"{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_"
            f"train_route_{_diag_lower}_{DYNAMICS_TAG}_v8diag_latest.pt"
        )
    )
    FEATURE_CACHE_DIR = (
        BACKBONE_OUTPUT_DIR / f"feature_cache_diag_route_{_diag_lower}"
    )
else:
    OUTPUT_DIR = BACKBONE_OUTPUT_DIR / f"{EXPERIMENT_FRAME_COUNT}frame"
    TEMPORAL_CHECKPOINT = (
        CHECKPOINT_DIR
        / (
            "causal_reference_only_forward3x6_ms_previous_position_"
            f"{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_"
            "gru_visual_measurement_variance_nosat_"
            f"multirate_A_native_plus_stride{TEMPORAL_EXTRA_A_STRIDE}_{DYNAMICS_TAG}_v8.pt"
        )
    )
    LATEST_TEMPORAL_CHECKPOINT = (
        CHECKPOINT_DIR
        / (
            "causal_reference_only_forward3x6_ms_previous_position_"
            f"{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_"
            "gru_visual_measurement_variance_nosat_"
            f"multirate_A_native_plus_stride{TEMPORAL_EXTRA_A_STRIDE}_{DYNAMICS_TAG}_v8_latest.pt"
        )
    )
    FEATURE_CACHE_DIR = BACKBONE_OUTPUT_DIR / "feature_cache"

# ---------------------------------------------------------------------------
# MeanShift ablation overrides
# ---------------------------------------------------------------------------
MEANSHIFT_ITERATIONS = int(
    os.environ.get("UAVSAT_MS_ITERATIONS", str(MEANSHIFT_ITERATIONS))
)
MEANSHIFT_BANDWIDTH_M = float(
    os.environ.get("UAVSAT_MS_BANDWIDTH_M", str(MEANSHIFT_BANDWIDTH_M))
)
MEANSHIFT_SCORE_TAU = float(
    os.environ.get("UAVSAT_MS_SCORE_TAU", str(MEANSHIFT_SCORE_TAU))
)
MEANSHIFT_MODE_BETA = float(
    os.environ.get("UAVSAT_MS_MODE_BETA", str(MEANSHIFT_MODE_BETA))
)
if MEANSHIFT_ITERATIONS < 1:
    raise ValueError("UAVSAT_MS_ITERATIONS must be >= 1")
if MEANSHIFT_BANDWIDTH_M <= 0.0:
    raise ValueError("UAVSAT_MS_BANDWIDTH_M must be > 0")
if MEANSHIFT_SCORE_TAU <= 0.0:
    raise ValueError("UAVSAT_MS_SCORE_TAU must be > 0")
if MEANSHIFT_MODE_BETA <= 0.0:
    raise ValueError("UAVSAT_MS_MODE_BETA must be > 0")

# ---------------------------------------------------------------------------
# Teacher-requested inter-frame hand-off
# ---------------------------------------------------------------------------
TEACHER_MEANSHIFT_FEEDBACK = True
TEACHER_FEEDBACK_PRESERVE_KALMAN_VELOCITY = True
TEACHER_FEEDBACK_USE_FORWARD_3X6 = True
MEANSHIFT_POSITION_AS_GRU_INPUT = True

# ---------------------------------------------------------------------------
# GRU visual measurement
# ---------------------------------------------------------------------------
DIRECT_SOFTMS_MEASUREMENT = False
USE_GRU_VISUAL_MEASUREMENT_HEAD = True
GRU_VISUAL_MEASUREMENT_PROGRESS_RANGE_M = 6.0
GRU_VISUAL_MEASUREMENT_CROSS_RANGE_M = 4.0
GRU_VISUAL_MEASUREMENT_INIT_PROGRESS_M = 0.0

# ---------------------------------------------------------------------------
# Visual uncertainty
# ---------------------------------------------------------------------------
MEANSHIFT_VARIANCE_AS_GRU_INPUT = True
DIRECT_SOFTMS_VARIANCE = False
USE_LEARNED_VARIANCE_HEAD = True
GRU_VISUAL_VARIANCE_INIT_M2 = 4.0
KALMAN_R_MIN_VAR = 1.0
KALMAN_R_MAX_VAR = 400.0
KALMAN_USE_MS_CONFIDENCE = False
VISUAL_CONFIDENCE_FLOOR = 1.0
VISUAL_CONFIDENCE_CEIL = 1.0
KALMAN_NIS_CONFIDENCE_BOOST = 0.0
KALMAN_NIS_MAX_R_SCALE = 1.0
ACQ_LOW_CONF_VARIANCE_GAIN = 0.0

# ---------------------------------------------------------------------------
# Kalman trust balance (legacy values retained; pure-model overrides below)
# ---------------------------------------------------------------------------
KALMAN_Q_PROGRESS = 0.80
KALMAN_Q_CROSS = 0.25
KALMAN_Q_VELOCITY = 0.40
KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M = 16.0
KALMAN_MAX_MEASUREMENT_INNOVATION_CROSS_M = 10.0
KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M = 7.0
KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M = 4.0
KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME = 2.0
KALMAN_FINAL_STEP_SLACK_M = 1.0
KALMAN_FINAL_STEP_MAX_M = 8.0

# ---------------------------------------------------------------------------
# Compact temporal state
# ---------------------------------------------------------------------------
USE_SATELLITE_CONTEXT_IN_GRU = False
RNN_NUMERIC_DIM = 10
RNN_HEADING_INPUT_SCALE_DEG = 70.0
RNN_TURN_RATE_INPUT_SCALE_DEG_PER_FRAME = 12.0

# ---------------------------------------------------------------------------
# Training balance
# ---------------------------------------------------------------------------
# GT/reference trajectory values are supervision targets only. They never choose
# the current reference point, search centre, hypothesis or candidate.
LOSS_MEASUREMENT = 2.0
LOSS_NEXT_STEP = 2.0
LOSS_VARIANCE_NLL = 0.02
TEMPORAL_LR = 1e-4
RNN_DROPOUT = 0.05
TBPTT_STEPS = 32
GRAD_CLIP_NORM = 3.0
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.02
EARLY_STOP_MIN_EPOCH = 20

# ---------------------------------------------------------------------------
# Forward-search accuracy/speed ablation
# ---------------------------------------------------------------------------
FORWARD_SEARCH_ROWS = 3
FORWARD_SEARCH_COLS = 6
FORWARD_SEARCH_CANDIDATE_COUNT = FORWARD_SEARCH_ROWS * FORWARD_SEARCH_COLS
FORWARD_SEARCH_EXPERIMENT_ROWS = (3, 4, 5, 6)
LATENCY_WARMUP_FRAMES = 30

# ---------------------------------------------------------------------------
# Pure-model dynamics
# ---------------------------------------------------------------------------
if PURE_MODEL_DYNAMICS:
    _NO_LIMIT = 1.0e9
    MAX_FORWARD_SPEED_M_PER_FRAME = _NO_LIMIT
    MAX_CROSS_SPEED_M_PER_FRAME = _NO_LIMIT
    MAX_FORWARD_ACCEL_M_PER_FRAME2 = _NO_LIMIT
    MAX_CROSS_ACCEL_M_PER_FRAME2 = _NO_LIMIT
    MAX_POLYNOMIAL_STEP_M_PER_FRAME = _NO_LIMIT
    MAX_FINAL_CROSS_TRACK_M = _NO_LIMIT

    MAX_HEADING_RESIDUAL_DEG = 180.0
    MAX_TURN_RATE_DEG_PER_FRAME = 180.0
    MAX_HEADING_DELTA_DEG_PER_FRAME = 180.0
    MAX_TURN_RATE_DELTA_DEG_PER_FRAME2 = 180.0
    HEADING_STATE_EMA_ALPHA = 1.0
    TURN_RATE_EMA_ALPHA = 1.0

    MOTION_VELOCITY_EMA_ALPHA = 1.0
    MOTION_ACCELERATION_EMA_ALPHA = 1.0
    MOTION_POLYNOMIAL_STEP_EMA_ALPHA = 1.0
    MAX_MOTION_VELOCITY_DELTA_M_PER_FRAME = _NO_LIMIT
    MAX_MOTION_ACCEL_DELTA_M_PER_FRAME2 = _NO_LIMIT
    MAX_POLYNOMIAL_STEP_DELTA_M_PER_FRAME = _NO_LIMIT

    CONTROLLED_FINAL_PROGRESS_CAP_TO_GT = False
    CONTROLLED_GT_MOTION_ENVELOPE = False
    CONTROLLED_PACE_ASSIST = False

    KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M = _NO_LIMIT
    KALMAN_MAX_MEASUREMENT_INNOVATION_CROSS_M = _NO_LIMIT
    KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M = _NO_LIMIT
    KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M = _NO_LIMIT
    KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME = _NO_LIMIT
    KALMAN_FINAL_STEP_MIN_M = 0.0
    KALMAN_FINAL_STEP_SLACK_M = _NO_LIMIT
    KALMAN_FINAL_STEP_MAX_M = _NO_LIMIT

    KALMAN_R_MIN_VAR = 1.0e-6
    KALMAN_R_MAX_VAR = 1.0e12
    ACQ_MAX_RESPONSE_VARIANCE_M2 = 1.0e12
    KALMAN_NIS_CONFIDENCE_BOOST = 0.0
    KALMAN_NIS_MAX_R_SCALE = 1.0
    ACQ_LOW_CONF_VARIANCE_GAIN = 0.0
    VISUAL_CONFIDENCE_FLOOR = 1.0
    VISUAL_CONFIDENCE_CEIL = 1.0

CONTROLLED_PROTOCOL_NAME = (
    "causal-prediction_to-nearest-already-passed-route-reference_"
    "forward3x6_MS-previous-position-to-GRU_"
    f"{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_"
    "current-MS-anchored-GRU-visual-measurement-and-variance_no-sat-context_"
    f"polynomial_Kalman_{TEMPORAL_TRAINING_PROTOCOL}_{DYNAMICS_TAG}_v9"
)

WAYPOINT_DIR = PROJECT_ROOT.parent / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}
if DIAGNOSTIC_TEMPORAL_ROUTE and not BC_TO_A_TEMPORAL:
    WAYPOINT_FILES["route_A"] = WAYPOINT_FILES[TEMPORAL_TRAIN_SOURCE_ROUTE]


def _reference_se_from_progress(route, progress_s):
    import numpy as _np
    predicted_s = float(
        _np.clip(float(progress_s), 0.0, float(route.total_length_m))
    )
    spacing = max(float(REFERENCE_POINT_SPACING_M), 1.0e-6)
    reference_index = int(_np.floor((predicted_s + 1.0e-9) / spacing))
    reference_s = min(reference_index * spacing, float(route.total_length_m))
    return _np.asarray([reference_s, 0.0], dtype=_np.float64)


# ---------------------------------------------------------------------------
# Runtime compatibility patch for the legacy base implementation
# ---------------------------------------------------------------------------
def _install_v36_runtime_patches():
    import robust_tracker_base as _b

    if bool(getattr(_b, "_V36_CAUSAL_REFERENCE_PATCHED", False)):
        return

    _original_visual_observation = _b.visual_observation

    def _causal_reference_visual_observation(
        model,
        visual,
        uav_clip,
        search_center_se,
        route,
        predicted_se,
        previous_z_uav,
        previous2_z_uav,
        hidden,
        previous_acquisition_confidence,
        kalman_progress_std,
        previous_forward_speed,
        search_heading_rad,
        device,
        gt_xy=None,
        gt_se=None,
        teacher_select=False,
    ):
        # The legacy caller may still construct a GT-based search_center_se.
        # Ignore it completely and replace it with the causal fixed reference.
        reference_center_se = _reference_se_from_progress(route, predicted_se[0])
        return _original_visual_observation(
            model=model,
            visual=visual,
            uav_clip=uav_clip,
            search_center_se=reference_center_se,
            route=route,
            predicted_se=predicted_se,
            previous_z_uav=previous_z_uav,
            previous2_z_uav=previous2_z_uav,
            hidden=hidden,
            previous_acquisition_confidence=previous_acquisition_confidence,
            kalman_progress_std=kalman_progress_std,
            previous_forward_speed=previous_forward_speed,
            search_heading_rad=search_heading_rad,
            device=device,
            # GT is supervision/metric information only. It cannot force the
            # selected hypothesis or search centre.
            gt_xy=gt_xy,
            gt_se=gt_se,
            teacher_select=False,
        )

    _b.visual_observation = _causal_reference_visual_observation

    if PURE_MODEL_DYNAMICS:
        _original_predict = _b.RouteKalman.predict
        _original_update = _b.RouteKalman.update

        def _pure_predict(
            self,
            velocity_se,
            acceleration_se,
            total_length_m,
            max_progress_s=None,
            polynomial_step_se=None,
            max_step_m=None,
        ):
            return _original_predict(
                self,
                velocity_se,
                acceleration_se,
                total_length_m,
                max_progress_s=None,
                polynomial_step_se=polynomial_step_se,
                max_step_m=None,
            )

        def _pure_update(
            self,
            measurement_se,
            variance_se,
            total_length_m,
            acquisition_confidence=1.0,
            max_progress_s=None,
            max_final_step_m=None,
        ):
            return _original_update(
                self,
                measurement_se,
                variance_se,
                total_length_m,
                acquisition_confidence=1.0,
                max_progress_s=None,
                max_final_step_m=None,
            )

        def _raw_motion_state(
            previous_velocity,
            previous_acceleration,
            previous_polynomial_step,
            raw_velocity,
            raw_acceleration,
            raw_polynomial_step,
        ):
            return (
                raw_velocity.detach(),
                raw_acceleration.detach(),
                raw_polynomial_step.detach(),
            )

        def _raw_heading_state(previous_state, raw_heading_residual, raw_turn_rate):
            return _b.torch.cat(
                [raw_heading_residual.detach(), raw_turn_rate.detach()], dim=1
            )

        def _no_prediction_reference_cap(kf, predicted_se, reference_se):
            return _b.np.asarray(predicted_se, dtype=_b.np.float64).copy()

        def _no_kalman_reference_cap(kf, final_se, reference_se):
            return _b.np.asarray(final_se, dtype=_b.np.float64).copy(), False

        _b.RouteKalman.predict = _pure_predict
        _b.RouteKalman.update = _pure_update
        _b.stabilize_motion_state = _raw_motion_state
        _b.stabilize_heading_state = _raw_heading_state
        _b.cap_prediction_to_current_gt = _no_prediction_reference_cap
        _b.cap_kalman_to_current_gt = _no_kalman_reference_cap

    _b._V36_CAUSAL_REFERENCE_PATCHED = True


# ---------------------------------------------------------------------------
# Patch the v36 main entry point without deleting the original A-only trainer.
# The patch is installed immediately before robust_tracker.main executes.
# ---------------------------------------------------------------------------
def _install_main_module_patches():
    if not BC_TO_A_TEMPORAL:
        return

    import sys as _sys

    def _profile(frame, event, arg):
        if event != "call" or frame.f_code.co_name != "main":
            return _profile
        g = frame.f_globals
        if str(g.get("ARCHITECTURE_NAME", "")) != str(ARCHITECTURE_NAME):
            return _profile
        if not str(g.get("__file__", "")).endswith("robust_tracker.py"):
            return _profile

        import json as _json
        import numpy as _np
        import torch as _torch

        def _causal_frame_visual(
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
            gt_xy_t = cache.gt_xy[index:index + 1].to(device).float()
            reference_center_se = _reference_se_from_progress(route, predicted_se[0])
            reference_xy = route.xy_from_se(reference_center_se[0], 0.0)
            search_heading_rad = g["b"].wrap_angle_rad(
                route.route_heading_rad(float(reference_center_se[0]))
                + float(previous_heading_state[0, 0].item())
            )
            obs = g["b"].visual_observation(
                model=model,
                visual=visual,
                uav_clip=uav_clip,
                search_center_se=reference_center_se,
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
                teacher_select=False,
            )
            return (
                obs,
                _np.asarray(reference_xy, dtype=_np.float64),
                _np.zeros(2, dtype=_np.float64),
                search_heading_rad,
            )

        def _causal_feedback(
            visual, uav_clip, route, kf, previous_heading_state, device
        ):
            previous_output_se = kf.se().copy()
            reference_center_se = _reference_se_from_progress(
                route, previous_output_se[0]
            )
            reference_xy = route.xy_from_se(reference_center_se[0], 0.0)
            center_xy = _torch.tensor(
                reference_xy, dtype=_torch.float32, device=device
            ).reshape(1, 2)
            heading_rad = g["b"].wrap_angle_rad(
                route.route_heading_rad(float(reference_center_se[0]))
                + float(previous_heading_state[0, 0].item())
            )
            candidate = g["forward_rows_candidate_batch"](
                visual=visual,
                uav_clip=uav_clip,
                center_xy=center_xy,
                heading_rad=heading_rad,
                grid_size=int(ACQ_LOCAL_GRID_SIZE),
            )
            feedback_xy = (
                candidate.softms_xy[0].detach().cpu().numpy().astype(_np.float64)
            )
            preferred_leg = route.leg_for_s(float(reference_center_se[0]))
            feedback_s, feedback_e, _ = route.project_xy_local(
                feedback_xy, preferred_leg
            )
            feedback_se = _np.asarray(
                [feedback_s, feedback_e], dtype=_np.float64
            )
            feedback_se[0] = float(
                _np.clip(feedback_se[0], 0.0, route.total_length_m)
            )
            return (
                previous_output_se,
                feedback_se,
                feedback_xy,
                float(candidate.softms_support[0].item()),
            )

        def _load_pair(route_name, visual, device):
            route_index = ROUTE_NAMES.index(route_name)
            cache = g["build_route_cache"](
                route_name, ROUTE_ROOTS[route_index], visual, device
            )
            route = g["WaypointRoute"](
                g["load_waypoint_xy"](
                    route_name, visual.origin_lat, visual.origin_lon
                )
            )
            return cache, route

        def _train_bc_to_a(args, device):
            # Reuse the current frozen visual checkpoint so this experiment
            # isolates whether the temporal model can learn under the new split.
            if not bool(args.reuse_visual) or not VISUAL_CHECKPOINT.exists():
                raise RuntimeError(
                    "B+C -> A temporal experiment requires the existing visual "
                    "checkpoint. Run with --reuse-visual."
                )

            visual = g["FrozenVisualLocalizer"](device)
            training_pairs = [
                (name, *_load_pair(name, visual, device))
                for name in TEMPORAL_TRAIN_ROUTES
            ]
            validation_cache, validation_route = _load_pair(
                TEMPORAL_VALIDATION_ROUTE, visual, device
            )
            validation_gt_state = g["build_gt_route_state"](
                validation_cache, validation_route
            )

            model = g["ThreeFrameRouteStateGRU"]().to(device)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = _torch.optim.AdamW(
                params,
                lr=float(TEMPORAL_LR),
                weight_decay=float(TEMPORAL_WEIGHT_DECAY),
            )
            start_epoch = 1
            best_score = float("inf")
            best_state = None
            patience = 0

            if bool(args.resume_temporal) and LATEST_TEMPORAL_CHECKPOINT.exists():
                payload = _torch.load(
                    LATEST_TEMPORAL_CHECKPOINT, map_location="cpu"
                )
                if payload.get("architecture") != ARCHITECTURE_NAME:
                    raise RuntimeError(
                        "Latest temporal checkpoint architecture mismatch"
                    )
                model.load_state_dict(payload["model"])
                optimizer.load_state_dict(payload["optimizer"])
                start_epoch = int(payload["epoch"]) + 1
                best_score = float(payload.get("best_score", best_score))
                best_state = payload.get("best_model")
                patience = int(payload.get("patience", 0))

            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            print(
                "temporal protocol: train=%s validation=%s; full trajectories; "
                "zero-state reset at every route boundary"
                % (TEMPORAL_TRAIN_ROUTES, TEMPORAL_VALIDATION_ROUTE),
                flush=True,
            )

            for epoch in range(start_epoch, int(args.temporal_epochs) + 1):
                model.train()
                all_losses = []
                route_loss_text = []

                for route_name, cache, route in training_pairs:
                    gt_state = g["build_gt_route_state"](cache, route)
                    kf = g["RouteKalman"](0.0, 0.0)
                    hidden = None
                    previous_z = None
                    previous2_z = None
                    previous_ms_se = None
                    previous_velocity = _torch.zeros(1, 2, device=device)
                    previous_acceleration = _torch.zeros(1, 2, device=device)
                    previous_heading_state = _torch.zeros(1, 2, device=device)
                    previous_poly_step = _torch.zeros(1, 2, device=device)
                    chunk_loss = None
                    chunk_count = 0
                    route_losses = []
                    optimizer.zero_grad(set_to_none=True)

                    for index in range(len(cache)):
                        uav_clip = cache.uav_clip[index:index + 1].to(device).float()
                        if index > 0:
                            _, feedback_se, _, _ = g[
                                "teacher_meanshift_feedback"
                            ](
                                visual,
                                uav_clip,
                                route,
                                kf,
                                previous_heading_state,
                                device,
                            )
                            previous_ms_se = g["b"].tensor2(
                                feedback_se, device
                            ).detach()
                            predicted_se = kf.predict(
                                previous_velocity[0].detach().cpu().numpy(),
                                previous_acceleration[0].detach().cpu().numpy(),
                                route.total_length_m,
                                max_progress_s=float(gt_state["se"][index, 0]),
                                polynomial_step_se=(
                                    previous_poly_step[0].detach().cpu().numpy()
                                ),
                                max_step_m=float(
                                    gt_state["gt_step_norm"][index]
                                ),
                            )
                        else:
                            predicted_se = kf.se()

                        # In pure-model mode this function is patched to a no-op.
                        predicted_se = g["b"].cap_prediction_to_current_gt(
                            kf, predicted_se, gt_state["se"][index]
                        )
                        obs, _, _, _ = g["_frame_visual"](
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
                        output = g["b"].model_forward(
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
                        loss, _ = g["b"].temporal_loss(
                            output,
                            obs,
                            gt_state["se"][index],
                            gt_state["velocity"][index],
                            gt_state["acceleration"][index],
                            gt_state["step"][index],
                            gt_state["heading_residual"][index],
                            gt_state["turn_rate"][index],
                        )
                        chunk_loss = (
                            loss if chunk_loss is None else chunk_loss + loss
                        )
                        chunk_count += 1

                        final_se = g["_kalman_update"](
                            kf, output, route, gt_state, index
                        )
                        g["b"].cap_kalman_to_current_gt(
                            kf, final_se, gt_state["se"][index]
                        )

                        previous2_z, previous_z = (
                            previous_z,
                            obs.candidate.z_uav.detach(),
                        )
                        (
                            previous_velocity,
                            previous_acceleration,
                            previous_poly_step,
                        ) = g["b"].stabilize_motion_state(
                            previous_velocity,
                            previous_acceleration,
                            previous_poly_step,
                            output.velocity_se,
                            output.acceleration_se,
                            output.next_step_se,
                        )
                        previous_heading_state = g["b"].stabilize_heading_state(
                            previous_heading_state,
                            output.heading_residual_rad,
                            output.turn_rate_rad,
                        )
                        hidden = output.hidden

                        boundary = (
                            chunk_count >= int(TBPTT_STEPS)
                            or index + 1 >= len(cache)
                        )
                        if boundary:
                            normalized = chunk_loss / float(max(1, chunk_count))
                            if not _torch.isfinite(normalized):
                                raise FloatingPointError(
                                    f"non-finite temporal loss: {route_name} "
                                    f"epoch={epoch} frame={index}"
                                )
                            normalized.backward()
                            _torch.nn.utils.clip_grad_norm_(
                                params, float(GRAD_CLIP_NORM)
                            )
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            value = float(normalized.detach().cpu())
                            all_losses.append(value)
                            route_losses.append(value)

                            hidden = hidden.detach()
                            previous_z = (
                                previous_z.detach()
                                if previous_z is not None
                                else None
                            )
                            previous2_z = (
                                previous2_z.detach()
                                if previous2_z is not None
                                else None
                            )
                            previous_velocity = previous_velocity.detach()
                            previous_acceleration = previous_acceleration.detach()
                            previous_heading_state = previous_heading_state.detach()
                            previous_poly_step = previous_poly_step.detach()
                            previous_ms_se = (
                                previous_ms_se.detach()
                                if previous_ms_se is not None
                                else None
                            )
                            chunk_loss = None
                            chunk_count = 0

                    route_loss_text.append(
                        f"{route_name}="
                        f"{float(_np.mean(route_losses)) if route_losses else float('nan'):.5f}"
                    )

                val = g["evaluate_closed_loop"](
                    model,
                    visual,
                    validation_cache,
                    validation_route,
                    validation_gt_state,
                    (0, len(validation_cache)),
                    device,
                )
                score = float(val["score"])
                improved = score < best_score - float(EARLY_STOP_MIN_DELTA)
                if improved:
                    best_score = score
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    patience = 0
                    _torch.save(
                        {
                            "architecture": ARCHITECTURE_NAME,
                            "model": best_state,
                            "epoch": epoch,
                            "validation": val,
                            "train_routes": list(TEMPORAL_TRAIN_ROUTES),
                            "validation_route": TEMPORAL_VALIDATION_ROUTE,
                            "train_forward_rows": int(FORWARD_SEARCH_ROWS),
                            "reference_search": REFERENCE_SELECTION_POLICY,
                            "gt_search": False,
                        },
                        TEMPORAL_CHECKPOINT,
                    )
                else:
                    patience += 1

                _torch.save(
                    {
                        "architecture": ARCHITECTURE_NAME,
                        "model": model.state_dict(),
                        "best_model": best_state,
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_score": best_score,
                        "patience": patience,
                        "train_routes": list(TEMPORAL_TRAIN_ROUTES),
                        "validation_route": TEMPORAL_VALIDATION_ROUTE,
                    },
                    LATEST_TEMPORAL_CHECKPOINT,
                )

                mean_loss = (
                    float(_np.mean(all_losses)) if all_losses else float("nan")
                )
                print(
                    f"teacher {FORWARD_SEARCH_ROWS}x6 BC->A "
                    f"epoch={epoch:03d}/{args.temporal_epochs} "
                    f"loss={mean_loss:.5f} ({', '.join(route_loss_text)}) "
                    f"val_mle={val['mle']:.3f}m "
                    f"val_p90={val['p90']:.3f}m "
                    f"score={score:.3f} best={best_score:.3f} "
                    f"patience={patience}/{args.patience}",
                    flush=True,
                )

                if (
                    epoch >= int(EARLY_STOP_MIN_EPOCH)
                    and patience >= int(args.patience)
                ):
                    break

            if best_state is None:
                raise RuntimeError(
                    "Temporal B+C -> A training did not produce a checkpoint"
                )
            print(
                f"best Route-A validation score={best_score:.3f}",
                flush=True,
            )

        def _eval_a(args, device):
            visual = g["FrozenVisualLocalizer"](device)
            model = g["load_temporal_model"](device)
            rows_to_eval = (
                [3, 4, 5, 6]
                if bool(args.eval_all_forward_rows)
                else [int(args.forward_rows)]
            )
            results = {}
            for forward_rows in rows_to_eval:
                g["_set_forward_rows"](forward_rows)
                cache, route = _load_pair(
                    TEMPORAL_VALIDATION_ROUTE, visual, device
                )
                results[f"{forward_rows}x6"] = g["run_route_inference"](
                    TEMPORAL_VALIDATION_ROUTE,
                    visual,
                    model,
                    cache,
                    route,
                    device,
                    measure_latency=bool(args.measure_latency),
                )
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            path = OUTPUT_DIR / "BC_train_A_validation_eval_summary.json"
            path.write_text(
                _json.dumps(
                    {
                        "architecture": ARCHITECTURE_NAME,
                        "train_routes": list(TEMPORAL_TRAIN_ROUTES),
                        "eval_route": TEMPORAL_VALIDATION_ROUTE,
                        "gt_used_for_search": False,
                        "reference_policy": REFERENCE_SELECTION_POLICY,
                        "results": results,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"summary: {path}", flush=True)

        # Replace only runtime entry points. The original functions remain in
        # robust_tracker.py and can be restored with UAVSAT_BC_TO_A_TEMPORAL=0.
        g["_frame_visual"] = _causal_frame_visual
        g["teacher_meanshift_feedback"] = _causal_feedback
        g["train_pipeline"] = _train_bc_to_a
        g["eval_pipeline"] = _eval_a
        _sys.setprofile(None)
        return None

    _sys.setprofile(_profile)


_install_v36_runtime_patches()
_install_main_module_patches()
