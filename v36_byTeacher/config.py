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

ARCHITECTURE_NAME = (
    "V36_byTeacher_MSPreviousPosition_"
    f"{EXPERIMENT_FRAME_COUNT}Frame_{BACKBONE_KEY}_"
    "GRUVisualMeasurementVariance_NoSatContext_Polynomial_Kalman_v5"
)

# Keep every backbone isolated so changing the backbone never overwrites the
# currently-good MobileCLIP checkpoint.  The visual checkpoint/feature cache are
# shared by the 1-frame and 2-frame temporal experiments of the same backbone.
BACKBONE_OUTPUT_DIR = PROJECT_ROOT / "output" / BACKBONE_KEY
OUTPUT_DIR = BACKBONE_OUTPUT_DIR / f"{EXPERIMENT_FRAME_COUNT}frame"
CHECKPOINT_DIR = BACKBONE_OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / f"visual_retrieval_A_only_{BACKBONE_KEY}.pt"
TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / (
        "controlled_referenceprior_forward3x6_ms_previous_position_"
        f"{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_"
        "gru_visual_measurement_variance_nosat_v5_A_only.pt"
    )
)
LATEST_TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / (
        "controlled_referenceprior_forward3x6_ms_previous_position_"
        f"{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_"
        "gru_visual_measurement_variance_nosat_v5_A_only_latest.pt"
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
# Current measurement is anchored to the current controlled local MeanShift:
#       z_t = current_MS_t + residual_head(h_t)
# The teacher-requested X_(t-1)^MS remains an input to the GRU and therefore
# affects h_t, but an accumulated previous Kalman error can no longer shift the
# absolute current measurement hundreds of metres away from the visual evidence.
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
# Keep enough process uncertainty/correction room for the current visual
# measurement to pull the posterior back toward the reference-supported local
# observation, while retaining bounded non-teleporting updates.
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

# ---------------------------------------------------------------------------
# Training balance
# ---------------------------------------------------------------------------
# The current visual anchor is already near the supervised reference under the
# controlled local protocol, so the residual head should learn a small refinement
# rather than a hundreds-of-metres absolute displacement.
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
    "reference-point+smooth-jitter_forward3x6_MS-previous-position-to-GRU_"
    f"{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_"
    "current-MS-anchored-GRU-visual-measurement-and-variance_no-sat-context_"
    "polynomial_Kalman_v5"
)

WAYPOINT_DIR = PROJECT_ROOT.parent / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}
