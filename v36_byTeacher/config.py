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

# Main thesis setting is 2-frame. 1-frame is the controlled ablation.
EXPERIMENT_FRAME_COUNT = int(os.environ.get("UAVSAT_EXPERIMENT_FRAME_COUNT", "2"))
if EXPERIMENT_FRAME_COUNT not in (1, 2):
    raise ValueError("UAVSAT_EXPERIMENT_FRAME_COUNT must be 1 or 2 for v36_byTeacher")
TEMPORAL_WINDOW_FRAMES = EXPERIMENT_FRAME_COUNT

# Route-A native + stride-N multi-rate temporal training.
TEMPORAL_EXTRA_A_STRIDE = int(
    os.environ.get("UAVSAT_TEMPORAL_EXTRA_A_STRIDE", "2")
)
if TEMPORAL_EXTRA_A_STRIDE < 2:
    raise ValueError("UAVSAT_TEMPORAL_EXTRA_A_STRIDE must be >= 2")
TEMPORAL_TRAINING_PROTOCOL = f"routeA_native_plus_stride{TEMPORAL_EXTRA_A_STRIDE}"

# v7 is the architecture drawn in the supplied diagram:
# previous KF posterior -> one forward local MS re-localization -> GRU;
# GRU measurement/variance -> KF update; GRU motion/heading -> next-frame
# heading-aware second-order polynomial -> KF predict.
IMAGE_ALIGNED_SINGLE_MS = True
ARCHITECTURE_NAME = (
    "V36_byTeacher_ImageAlignedSingleMS_"
    f"{EXPERIMENT_FRAME_COUNT}Frame_{BACKBONE_KEY}_"
    "GRUMeasurementVariance_MotionHeading_Polynomial_Kalman_"
    f"MultiRateAstride{TEMPORAL_EXTRA_A_STRIDE}_v7"
)

# Keep every backbone isolated. Visual checkpoint and UAV backbone caches remain
# shared because this change only affects the temporal architecture.
BACKBONE_OUTPUT_DIR = PROJECT_ROOT / "output" / BACKBONE_KEY
OUTPUT_DIR = BACKBONE_OUTPUT_DIR / f"{EXPERIMENT_FRAME_COUNT}frame"
CHECKPOINT_DIR = BACKBONE_OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / f"visual_retrieval_A_only_{BACKBONE_KEY}.pt"
TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / (
        "image_aligned_single_ms_previous_state_forward3x6_"
        f"{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_"
        "gru_measurement_variance_motion_heading_"
        f"multirate_A_native_plus_stride{TEMPORAL_EXTRA_A_STRIDE}_v7.pt"
    )
)
LATEST_TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / (
        "image_aligned_single_ms_previous_state_forward3x6_"
        f"{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_"
        "gru_measurement_variance_motion_heading_"
        f"multirate_A_native_plus_stride{TEMPORAL_EXTRA_A_STRIDE}_v7_latest.pt"
    )
)
FEATURE_CACHE_DIR = BACKBONE_OUTPUT_DIR / "feature_cache"

# ---------------------------------------------------------------------------
# No-training MeanShift ablation overrides
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
# Image-aligned inter-frame hand-off
# ---------------------------------------------------------------------------
# At frame t, X_(t-1) from the final KF posterior is the center of the single
# heading-forward local search. The newly arrived UAV frame is matched against
# that local satellite set and MeanShift returns X_(t-1)^MS. There is no second
# hidden/current-frame MeanShift branch after this point.
TEACHER_MEANSHIFT_FEEDBACK = True
TEACHER_FEEDBACK_PRESERVE_KALMAN_VELOCITY = True
TEACHER_FEEDBACK_USE_FORWARD_3X6 = True
MEANSHIFT_POSITION_AS_GRU_INPUT = True

# ---------------------------------------------------------------------------
# GRU measurement head
# ---------------------------------------------------------------------------
# The one MeanShift result in the diagram is the visual base. The Measurement
# Head predicts only a bounded residual around that visual base:
#       z_t = X_(t-1)^MS + residual_head(h_t)
# The KF prior itself is NOT an input to the measurement head. This keeps the
# learned visual measurement distinct from the external KF fusion step.
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

# MeanShift confidence does not separately scale R; the GRU variance head is the
# single learned measurement-uncertainty source entering the KF.
KALMAN_USE_MS_CONFIDENCE = False
VISUAL_CONFIDENCE_FLOOR = 1.0
VISUAL_CONFIDENCE_CEIL = 1.0
KALMAN_NIS_CONFIDENCE_BOOST = 0.0
KALMAN_NIS_MAX_R_SCALE = 1.0
ACQ_LOW_CONF_VARIANCE_GAIN = 0.0

# ---------------------------------------------------------------------------
# Kalman trust balance
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
# Compact GRU state
# ---------------------------------------------------------------------------
USE_SATELLITE_CONTEXT_IN_GRU = False

# Exactly the numeric information represented by the diagram/causal state:
#   SoftMS mode-space variance (2)
# + re-localized-position displacement from previous KF state (2)
# + previous recurrent velocity (2)
# + previous heading residual / turn-rate (2)
# The current Kalman prior is deliberately excluded from the GRU measurement
# input so the KF does not fuse its own prior twice.
RNN_NUMERIC_DIM = 8

# ---------------------------------------------------------------------------
# Training balance
# ---------------------------------------------------------------------------
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

CONTROLLED_PROTOCOL_NAME = (
    "reference-route_single-MS-from-previous-KF_forward3x6_"
    f"{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_"
    "GRU-measurement-variance-motion-heading_polynomial-Kalman_"
    f"multirate-{TEMPORAL_TRAINING_PROTOCOL}_v7"
)

WAYPOINT_DIR = PROJECT_ROOT.parent / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}
