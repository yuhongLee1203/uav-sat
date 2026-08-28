import os
from pathlib import Path
from config_base import *

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Backbone / experiment identity
# ---------------------------------------------------------------------------
# v36_byTeacher defaults to MobileNetV3-Small. The environment override is kept
# so the previous MobileCLIP setup can still be reproduced explicitly.
BACKBONE_KEY = os.environ.get("UAVSAT_BACKBONE", "mobilenet_v3_small").strip().lower()
if BACKBONE_KEY not in BACKBONE_SPECS:
    raise ValueError(
        "UAVSAT_BACKBONE must be one of %s; got %r"
        % (sorted(BACKBONE_SPECS), BACKBONE_KEY)
    )
BACKBONE_NAME, CLIP_DIM = BACKBONE_SPECS[BACKBONE_KEY]

# Main thesis setting is 2-frame. 1-frame is the controlled temporal ablation.
EXPERIMENT_FRAME_COUNT = int(os.environ.get("UAVSAT_EXPERIMENT_FRAME_COUNT", "2"))
if EXPERIMENT_FRAME_COUNT not in (1, 2):
    raise ValueError("UAVSAT_EXPERIMENT_FRAME_COUNT must be 1 or 2 for v36_byTeacher")
TEMPORAL_WINDOW_FRAMES = EXPERIMENT_FRAME_COUNT

# Multi-rate Route-A temporal training. The original Route-A training sequence
# is kept unchanged and a second sequence is made only from the same Route-A
# TRAIN split by taking every Nth frame. B/C remain evaluation routes.
TEMPORAL_EXTRA_A_STRIDE = int(
    os.environ.get("UAVSAT_TEMPORAL_EXTRA_A_STRIDE", "2")
)
if TEMPORAL_EXTRA_A_STRIDE < 2:
    raise ValueError("UAVSAT_TEMPORAL_EXTRA_A_STRIDE must be >= 2")
TEMPORAL_TRAINING_PROTOCOL = f"routeA_native_plus_stride{TEMPORAL_EXTRA_A_STRIDE}"

# Role-separated v7:
#   SoftMS        -> current visual measurement z_t
#   GRU           -> motion/heading state + learned measurement variance R_t
#   Polynomial    -> motion step used by the next Kalman predict
#   Kalman        -> the only block that fuses motion prior and visual z_t
ARCHITECTURE_NAME = (
    "V36_byTeacher_MSPreviousPosition_"
    f"{EXPERIMENT_FRAME_COUNT}Frame_{BACKBONE_KEY}_"
    "DirectSoftMSMeasurement_GRUMotionHeadingLearnedVariance_Polynomial_Kalman_"
    f"MultiRateAstride{TEMPORAL_EXTRA_A_STRIDE}_v7"
)

# Keep every backbone isolated. Visual checkpoint and UAV feature cache are
# shared by the 1-frame/2-frame temporal experiments of the same backbone.
BACKBONE_OUTPUT_DIR = PROJECT_ROOT / "output" / BACKBONE_KEY
DEFAULT_OUTPUT_DIR = BACKBONE_OUTPUT_DIR / f"{EXPERIMENT_FRAME_COUNT}frame"
RUN_TAG = os.environ.get("UAVSAT_RUN_TAG", "").strip()
if RUN_TAG:
    OUTPUT_DIR = BACKBONE_OUTPUT_DIR / "experiments" / RUN_TAG
else:
    OUTPUT_DIR = DEFAULT_OUTPUT_DIR
CHECKPOINT_DIR = BACKBONE_OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / f"visual_retrieval_A_only_{BACKBONE_KEY}.pt"
TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / (
        "controlled_referenceprior_forward3x6_ms_previous_position_"
        f"{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_"
        "direct_softms_gru_motion_heading_learned_variance_"
        f"multirate_A_native_plus_stride{TEMPORAL_EXTRA_A_STRIDE}_v7.pt"
    )
)
LATEST_TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / (
        "controlled_referenceprior_forward3x6_ms_previous_position_"
        f"{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_"
        "direct_softms_gru_motion_heading_learned_variance_"
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
# Original teacher-requested inter-frame hand-off
# ---------------------------------------------------------------------------
# For frame t, the newly arrived UAV image re-localizes the previous Kalman
# output X_(t-1) by MeanShift. X_(t-1)^MS is a temporal/previous-position cue to
# the GRU; it does not overwrite kf.x/kf.P.
TEACHER_MEANSHIFT_FEEDBACK = True
TEACHER_FEEDBACK_PRESERVE_KALMAN_VELOCITY = True
TEACHER_FEEDBACK_USE_FORWARD_3X6 = True
MEANSHIFT_POSITION_AS_GRU_INPUT = True

# ---------------------------------------------------------------------------
# Visual measurement / GRU role separation
# ---------------------------------------------------------------------------
# The current SoftMS position is z_t directly. The GRU does NOT output another
# current position and therefore does not duplicate the Kalman fusion role.
DIRECT_SOFTMS_MEASUREMENT = True
USE_GRU_VISUAL_MEASUREMENT_HEAD = False

# Retained only for backward-compatible imports; unused by the v7 measurement.
GRU_VISUAL_MEASUREMENT_PROGRESS_RANGE_M = 6.0
GRU_VISUAL_MEASUREMENT_CROSS_RANGE_M = 4.0
GRU_VISUAL_MEASUREMENT_INIT_PROGRESS_M = 0.0

# ---------------------------------------------------------------------------
# Learned visual uncertainty
# ---------------------------------------------------------------------------
# SoftMS mode-space spread is a GRU input cue. The GRU variance head predicts the
# final R_t supplied to the Kalman update. Thus the GRU learns reliability, not
# final position fusion.
MEANSHIFT_VARIANCE_AS_GRU_INPUT = True
DIRECT_SOFTMS_VARIANCE = False
USE_LEARNED_VARIANCE_HEAD = True
GRU_VISUAL_VARIANCE_INIT_M2 = 4.0

# ---------------------------------------------------------------------------
# Learned-KF recovery regime
# ---------------------------------------------------------------------------
# The uploaded B/C results show that the old constrained-KF tail is produced when
# the motion prior is already wrong: the old update clips the visual innovation,
# enlarges R for that same large innovation, caps the posterior correction, then
# applies another final-step corridor. That combination prevents an accurate
# SoftMS observation from pulling the state back and turns a local error into a
# multi-frame drift. The defaults below deliberately make visual recovery easy
# while preserving the GRU/polynomial prior and learned-R Kalman structure.
#
# No Route-B/C reference error is used to select the posterior at inference.
# These are estimator-side trust parameters only and can be overridden from the
# environment for an inference-only sweep without retraining the checkpoint.
KALMAN_R_MIN_VAR = float(os.environ.get("UAVSAT_KF_R_MIN", "0.35"))
KALMAN_R_MAX_VAR = float(os.environ.get("UAVSAT_KF_R_MAX", "16.0"))

# MeanShift/local-posterior confidence does not separately rescale R_t.
KALMAN_USE_MS_CONFIDENCE = False
VISUAL_CONFIDENCE_FLOOR = 1.0
VISUAL_CONFIDENCE_CEIL = 1.0
KALMAN_NIS_CONFIDENCE_BOOST = 0.0
KALMAN_NIS_MAX_R_SCALE = 1.0
ACQ_LOW_CONF_VARIANCE_GAIN = 0.0

# ---------------------------------------------------------------------------
# Kalman trust balance -- visual recovery + motion-state preservation
# ---------------------------------------------------------------------------
# Larger position Q prevents a bad polynomial prior from becoming sticky.  The
# innovation/posterior limits are intentionally much wider than the 3x6 local
# observation spread, which effectively disables the old "large disagreement ->
# trust visual less" failure mode.  Velocity correction is kept very small so a
# single visual residual cannot poison the recurrent motion state for subsequent
# frames.  The final-step corridor is also opened; the current-progress cap still
# enforces the controlled no-ahead protocol.
KALMAN_Q_PROGRESS = float(os.environ.get("UAVSAT_KF_Q_PROGRESS", "24.0"))
KALMAN_Q_CROSS = float(os.environ.get("UAVSAT_KF_Q_CROSS", "8.0"))
KALMAN_Q_VELOCITY = float(os.environ.get("UAVSAT_KF_Q_VELOCITY", "1.50"))
KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M = float(
    os.environ.get("UAVSAT_KF_INNOVATION_S", "1000.0")
)
KALMAN_MAX_MEASUREMENT_INNOVATION_CROSS_M = float(
    os.environ.get("UAVSAT_KF_INNOVATION_E", "1000.0")
)
KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M = float(
    os.environ.get("UAVSAT_KF_POSTERIOR_S", "1000.0")
)
KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M = float(
    os.environ.get("UAVSAT_KF_POSTERIOR_E", "1000.0")
)
KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME = float(
    os.environ.get("UAVSAT_KF_VELOCITY_CORRECTION", "0.25")
)
KALMAN_FINAL_STEP_SLACK_M = float(os.environ.get("UAVSAT_KF_STEP_SLACK", "1000.0"))
KALMAN_FINAL_STEP_MAX_M = float(os.environ.get("UAVSAT_KF_STEP_MAX", "1000.0"))

# The no-ahead max_progress_s clamp remains active.  Disable the extra reference-
# speed envelope here because it exists only for display pacing and was adding a
# second restriction after the Kalman update that the no-KF ablation does not
# need.  This makes the Kalman/no-Kalman comparison about fusion, not about an
# asymmetric step limiter.
CONTROLLED_GT_MOTION_ENVELOPE = os.environ.get(
    "UAVSAT_CONTROLLED_GT_MOTION_ENVELOPE", "0"
) == "1"
CONTROLLED_PACE_ASSIST = os.environ.get("UAVSAT_CONTROLLED_PACE_ASSIST", "0") == "1"
CONTROLLED_MAX_STEP_RATIO = float(os.environ.get("UAVSAT_CONTROLLED_MAX_STEP_RATIO", "2.0"))
CONTROLLED_PACE_MIN_RATIO = float(os.environ.get("UAVSAT_CONTROLLED_PACE_MIN_RATIO", "0.95"))
CONTROLLED_PACE_CATCHUP_GAIN = float(os.environ.get("UAVSAT_CONTROLLED_PACE_CATCHUP_GAIN", "0.25"))
CONTROLLED_PACE_MAX_EXTRA_M = float(os.environ.get("UAVSAT_CONTROLLED_PACE_MAX_EXTRA_M", "3.0"))

# ---------------------------------------------------------------------------
# Compact temporal state
# ---------------------------------------------------------------------------
USE_SATELLITE_CONTEXT_IN_GRU = False

# GRU numeric input contains only information needed for temporal motion and
# uncertainty estimation:
#   SoftMS mode-space spread (2)
# + current SoftMS - previous re-localized position (2)
# + previous recurrent velocity (2)
# + previous heading residual / turn-rate (2)
# The current Kalman prior is deliberately NOT a GRU input in v7.
RNN_NUMERIC_DIM = 8

# ---------------------------------------------------------------------------
# Training balance
# ---------------------------------------------------------------------------
# z_t is direct SoftMS, so there is no learned current-position head to supervise.
# Motion/heading objectives train the inertial branch; variance NLL trains R_t.
LOSS_MEASUREMENT = 0.0
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
    "direct-SoftMS-z_GRU-motion-heading-learned-R_"
    f"polynomial_Kalman_multirate-{TEMPORAL_TRAINING_PROTOCOL}_v7_visual-recovery"
)

WAYPOINT_DIR = PROJECT_ROOT.parent / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}
