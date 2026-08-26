import os
from pathlib import Path
from config_base import *

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Backbone / experiment identity
# ---------------------------------------------------------------------------
# v36_byTeacher now defaults to MobileNetV3-Small.  The environment override is
# kept so the previous MobileCLIP setup can still be reproduced explicitly.
BACKBONE_KEY = os.environ.get("UAVSAT_BACKBONE", "mobilenet_v3_small").strip().lower()
if BACKBONE_KEY not in BACKBONE_SPECS:
    raise ValueError(
        "UAVSAT_BACKBONE must be one of %s; got %r"
        % (sorted(BACKBONE_SPECS), BACKBONE_KEY)
    )
BACKBONE_NAME, CLIP_DIM = BACKBONE_SPECS[BACKBONE_KEY]

# Main thesis setting is 2-frame.  1-frame is an ablation that keeps the same
# recurrent state but removes the previous-frame visual feature/difference cue.
EXPERIMENT_FRAME_COUNT = int(os.environ.get("UAVSAT_EXPERIMENT_FRAME_COUNT", "2"))
if EXPERIMENT_FRAME_COUNT not in (1, 2):
    raise ValueError("UAVSAT_EXPERIMENT_FRAME_COUNT must be 1 or 2 for v36_byTeacher")
TEMPORAL_WINDOW_FRAMES = EXPERIMENT_FRAME_COUNT

# Multi-rate Route-A temporal training metadata.  The intended protocol is
# Route-A native plus a second Route-A sequence sampled every Nth frame.  B/C
# remain evaluation-only routes.  NOTE: robust_tracker.py currently executes
# one temporal pass over the Route-A training split; this field is retained so
# the intended protocol is explicit and can be restored without deleting code.
TEMPORAL_EXTRA_A_STRIDE = int(
    os.environ.get("UAVSAT_TEMPORAL_EXTRA_A_STRIDE", "2")
)
if TEMPORAL_EXTRA_A_STRIDE < 2:
    raise ValueError("UAVSAT_TEMPORAL_EXTRA_A_STRIDE must be >= 2")
TEMPORAL_TRAINING_PROTOCOL = f"routeA_native_plus_stride{TEMPORAL_EXTRA_A_STRIDE}"

# Pure-model ablation is now the default.  Set UAVSAT_MANUAL_CONSTRAINTS=1 to
# restore the legacy hand-tuned dynamic limits without deleting any old logic.
MANUAL_DYNAMICS_CONSTRAINTS = (
    os.environ.get("UAVSAT_MANUAL_CONSTRAINTS", "0").strip() == "1"
)
PURE_MODEL_DYNAMICS = not MANUAL_DYNAMICS_CONSTRAINTS
ZERO_KINEMATIC_INITIALIZATION = True
DYNAMICS_TAG = "manual" if MANUAL_DYNAMICS_CONSTRAINTS else "puremodel"

# ---------------------------------------------------------------------------
# Optional B/C in-domain temporal diagnostic
# ---------------------------------------------------------------------------
# This is ONLY a learnability/sanity-check switch.  The normal thesis setting
# remains Route A temporal training and Route B/C evaluation.  Setting B or C
# remaps only the temporal trainer's legacy route_A entry to that route while
# keeping the already-trained Route-A visual retrieval checkpoint unchanged.
# Run with --reuse-visual.  B and C use isolated temporal checkpoints/caches so
# this diagnostic can never overwrite the normal A-only temporal experiment.
DIAGNOSTIC_TEMPORAL_ROUTE = os.environ.get(
    "UAVSAT_DIAGNOSTIC_TEMPORAL_ROUTE", ""
).strip().upper()
if DIAGNOSTIC_TEMPORAL_ROUTE not in ("", "B", "C"):
    raise ValueError("UAVSAT_DIAGNOSTIC_TEMPORAL_ROUTE must be empty, B, or C")

if DIAGNOSTIC_TEMPORAL_ROUTE:
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
# IMPORTANT: current-frame GT/reference position is NEVER used to choose the
# local search centre.  The known planned route is sampled into fixed reference
# points.  The causal model/Kalman prediction is used only to determine how far
# along the planned route we have already reached; the selected reference point
# is the nearest fixed point whose route progress does not exceed that prediction.
# The default 4.48 m spacing corresponds to one 32-pixel satellite lattice stride
# at 0.14 m/px.  It can be changed without touching the model code.
REFERENCE_POINT_SPACING_M = float(
    os.environ.get("UAVSAT_REFERENCE_POINT_SPACING_M", "4.48")
)
if REFERENCE_POINT_SPACING_M <= 0.0:
    raise ValueError("UAVSAT_REFERENCE_POINT_SPACING_M must be > 0")
REFERENCE_SEARCH_GT_FREE = True
REFERENCE_SELECTION_POLICY = "nearest_already_passed_from_causal_prediction"

if DIAGNOSTIC_TEMPORAL_ROUTE:
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

# Keep every backbone isolated so changing the backbone never overwrites the
# currently-good MobileCLIP checkpoint.  The visual checkpoint/feature cache are
# shared by the 1-frame and 2-frame temporal experiments of the same backbone.
BACKBONE_OUTPUT_DIR = PROJECT_ROOT / "output" / BACKBONE_KEY
CHECKPOINT_DIR = BACKBONE_OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / f"visual_retrieval_A_only_{BACKBONE_KEY}.pt"

if DIAGNOSTIC_TEMPORAL_ROUTE:
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
# No-training MeanShift ablation overrides
# ---------------------------------------------------------------------------
# These values default to config_base.py, so normal training/evaluation is
# unchanged.  They can be overridden only at inference time to sweep decoder
# hyperparameters using the already-trained visual checkpoint and UAV cache.
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
# For frame t, the newly arrived UAV image re-localizes the previous Kalman
# output X_(t-1) by MeanShift. X_(t-1)^MS is only the GRU previous-position cue;
# it never overwrites kf.x/kf.P and is not used as the absolute base of z_t.
TEACHER_MEANSHIFT_FEEDBACK = True
TEACHER_FEEDBACK_PRESERVE_KALMAN_VELOCITY = True
TEACHER_FEEDBACK_USE_FORWARD_3X6 = True
MEANSHIFT_POSITION_AS_GRU_INPUT = True

# ---------------------------------------------------------------------------
# GRU visual measurement
# ---------------------------------------------------------------------------
# Current measurement is anchored to the current local MeanShift observation.
# The previous-position MeanShift remains an input to the GRU and therefore
# affects h_t, but an accumulated previous Kalman error can no longer shift the
# absolute current measurement hundreds of metres away from visual evidence.
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

# MeanShift/local-posterior confidence must not independently alter Kalman R.
KALMAN_USE_MS_CONFIDENCE = False
VISUAL_CONFIDENCE_FLOOR = 1.0
VISUAL_CONFIDENCE_CEIL = 1.0
KALMAN_NIS_CONFIDENCE_BOOST = 0.0
KALMAN_NIS_MAX_R_SCALE = 1.0
ACQ_LOW_CONF_VARIANCE_GAIN = 0.0

# ---------------------------------------------------------------------------
# Kalman trust balance
# ---------------------------------------------------------------------------
# Legacy constrained values are intentionally kept here.  The pure-model block
# below overrides them at runtime instead of deleting them.
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

# SoftMS spread(2) + current SoftMS-prior innovation(2)
# + current SoftMS-X_(t-1)^MS displacement(2) + previous velocity(2)
# + heading residual/turn-rate(2).
RNN_NUMERIC_DIM = 10

# Keep input normalization separate from any output bound.  These are only
# feature scales, not clipping/rate limits.
RNN_HEADING_INPUT_SCALE_DEG = 70.0
RNN_TURN_RATE_INPUT_SCALE_DEG_PER_FRAME = 12.0

# ---------------------------------------------------------------------------
# Training balance
# ---------------------------------------------------------------------------
# GT/reference trajectory values are training targets only.  They are not used
# to select the current reference point, search centre, hypothesis or candidate.
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
# Pure-model dynamics ablation
# ---------------------------------------------------------------------------
# Leave route geometry, reference-point local search, forward 3x6 selection,
# MeanShift, GRU, polynomial and Kalman enabled.  Only hand-tuned dynamic
# clipping/smoothing/GT-progress caps are disabled.  A very large finite guard
# is used inside legacy base functions so their code remains intact while being
# inactive in the physical operating range.
if PURE_MODEL_DYNAMICS:
    _NO_LIMIT = 1.0e9

    # No manually imposed speed / acceleration / polynomial-step ceiling.
    MAX_FORWARD_SPEED_M_PER_FRAME = _NO_LIMIT
    MAX_CROSS_SPEED_M_PER_FRAME = _NO_LIMIT
    MAX_FORWARD_ACCEL_M_PER_FRAME2 = _NO_LIMIT
    MAX_CROSS_ACCEL_M_PER_FRAME2 = _NO_LIMIT
    MAX_POLYNOMIAL_STEP_M_PER_FRAME = _NO_LIMIT
    MAX_FINAL_CROSS_TRACK_M = _NO_LIMIT

    # Heading residual and frame-to-frame turn targets are angle wrapped by
    # definition; 180 deg is therefore the natural non-restrictive range.
    MAX_HEADING_RESIDUAL_DEG = 180.0
    MAX_TURN_RATE_DEG_PER_FRAME = 180.0
    MAX_HEADING_DELTA_DEG_PER_FRAME = 180.0
    MAX_TURN_RATE_DELTA_DEG_PER_FRAME2 = 180.0
    HEADING_STATE_EMA_ALPHA = 1.0
    TURN_RATE_EMA_ALPHA = 1.0

    # Disable state EMA/rate limit.  Runtime functions are also patched below to
    # pass the raw learned states through directly.
    MOTION_VELOCITY_EMA_ALPHA = 1.0
    MOTION_ACCELERATION_EMA_ALPHA = 1.0
    MOTION_POLYNOMIAL_STEP_EMA_ALPHA = 1.0
    MAX_MOTION_VELOCITY_DELTA_M_PER_FRAME = _NO_LIMIT
    MAX_MOTION_ACCEL_DELTA_M_PER_FRAME2 = _NO_LIMIT
    MAX_POLYNOMIAL_STEP_DELTA_M_PER_FRAME = _NO_LIMIT

    # Disable GT/reference-progress output caps and controlled motion envelope.
    CONTROLLED_FINAL_PROGRESS_CAP_TO_GT = False
    CONTROLLED_GT_MOTION_ENVELOPE = False
    CONTROLLED_PACE_ASSIST = False

    # Disable hand-tuned Kalman innovation/posterior/velocity/step corridors.
    KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M = _NO_LIMIT
    KALMAN_MAX_MEASUREMENT_INNOVATION_CROSS_M = _NO_LIMIT
    KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M = _NO_LIMIT
    KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M = _NO_LIMIT
    KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME = _NO_LIMIT
    KALMAN_FINAL_STEP_MIN_M = 0.0
    KALMAN_FINAL_STEP_SLACK_M = _NO_LIMIT
    KALMAN_FINAL_STEP_MAX_M = _NO_LIMIT

    # Learned variance remains positive, but its old hand-set 1..400 m^2 range
    # is removed.  These are numerical guards only.
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
    f"polynomial_Kalman_multirate-{TEMPORAL_TRAINING_PROTOCOL}_{DYNAMICS_TAG}_v8"
)

WAYPOINT_DIR = PROJECT_ROOT.parent / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}
if DIAGNOSTIC_TEMPORAL_ROUTE:
    WAYPOINT_FILES["route_A"] = WAYPOINT_FILES[TEMPORAL_TRAIN_SOURCE_ROUTE]


# ---------------------------------------------------------------------------
# Runtime compatibility patch for the legacy constrained base implementation
# ---------------------------------------------------------------------------
# Do not delete the old constrained functions.  In pure-model mode only, wrap
# them so max_progress/max_step reference caps are not passed into Kalman and
# learned motion/heading states are not EMA/rate-limited after the network.
# Independently of manual/pure dynamics, visual_observation is wrapped so the
# current-frame GT cannot choose a search centre or selected hypothesis.
def _install_v36_runtime_patches():
    import robust_tracker_base as _b

    if bool(getattr(_b, "_V36_CAUSAL_REFERENCE_PATCHED", False)):
        return

    # ------------------------------------------------------------------
    # Search-centre policy: prediction -> nearest already-passed reference
    # ------------------------------------------------------------------
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
        # search_center_se from the legacy caller may contain current-frame GT.
        # It is intentionally ignored here.
        predicted = _b.np.asarray(predicted_se, dtype=_b.np.float64).reshape(2)
        predicted_s = float(
            _b.np.clip(predicted[0], 0.0, float(route.total_length_m))
        )
        spacing = max(float(REFERENCE_POINT_SPACING_M), 1.0e-6)
        reference_index = int(_b.math.floor((predicted_s + 1.0e-9) / spacing))
        reference_s = min(reference_index * spacing, float(route.total_length_m))
        reference_center_se = _b.np.asarray(
            [reference_s, 0.0], dtype=_b.np.float64
        )

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
            # GT is retained only for supervision/metrics such as capture and
            # loss targets.  It is forbidden from selecting the hypothesis.
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


_install_v36_runtime_patches()
