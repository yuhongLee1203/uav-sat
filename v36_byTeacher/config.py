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

TEMPORAL_EXTRA_A_STRIDE = int(
    os.environ.get("UAVSAT_TEMPORAL_EXTRA_A_STRIDE", "2")
)
if TEMPORAL_EXTRA_A_STRIDE < 2:
    raise ValueError("UAVSAT_TEMPORAL_EXTRA_A_STRIDE must be >= 2")
TEMPORAL_TRAINING_PROTOCOL = f"routeA_native_plus_stride{TEMPORAL_EXTRA_A_STRIDE}"

# Role-separated v7:
#   SoftMS     -> current visual measurement z_t
#   GRU        -> motion / heading + learned measurement variance R_t
#   Polynomial -> motion step for Kalman predict
#   Kalman     -> the only motion/visual fusion block
ARCHITECTURE_NAME = (
    "V36_byTeacher_MSPreviousPosition_"
    f"{EXPERIMENT_FRAME_COUNT}Frame_{BACKBONE_KEY}_"
    "DirectSoftMSMeasurement_GRUMotionHeadingLearnedVariance_Polynomial_Kalman_"
    f"MultiRateAstride{TEMPORAL_EXTRA_A_STRIDE}_v7"
)

BACKBONE_OUTPUT_DIR = PROJECT_ROOT / "output" / BACKBONE_KEY
DEFAULT_OUTPUT_DIR = BACKBONE_OUTPUT_DIR / f"{EXPERIMENT_FRAME_COUNT}frame"
RUN_TAG = os.environ.get("UAVSAT_RUN_TAG", "").strip()
OUTPUT_DIR = (
    BACKBONE_OUTPUT_DIR / "experiments" / RUN_TAG
    if RUN_TAG
    else DEFAULT_OUTPUT_DIR
)
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
# Previous-position MeanShift cue to GRU
# ---------------------------------------------------------------------------
TEACHER_MEANSHIFT_FEEDBACK = True
TEACHER_FEEDBACK_PRESERVE_KALMAN_VELOCITY = True
TEACHER_FEEDBACK_USE_FORWARD_3X6 = True
MEANSHIFT_POSITION_AS_GRU_INPUT = True

# ---------------------------------------------------------------------------
# GRU / Kalman role separation
# ---------------------------------------------------------------------------
DIRECT_SOFTMS_MEASUREMENT = True
USE_GRU_VISUAL_MEASUREMENT_HEAD = False
GRU_VISUAL_MEASUREMENT_PROGRESS_RANGE_M = 6.0
GRU_VISUAL_MEASUREMENT_CROSS_RANGE_M = 4.0
GRU_VISUAL_MEASUREMENT_INIT_PROGRESS_M = 0.0

MEANSHIFT_VARIANCE_AS_GRU_INPUT = True
DIRECT_SOFTMS_VARIANCE = False
USE_LEARNED_VARIANCE_HEAD = True
GRU_VISUAL_VARIANCE_INIT_M2 = 4.0

# ---------------------------------------------------------------------------
# Learned-R Kalman: visual recovery + small useful temporal smoothing
# ---------------------------------------------------------------------------
# The previous visual-first run reduced the old Route-B long tail and reached
# almost the same error as no-KF.  That showed the catastrophic failure was the
# old sticky prior/correction limits, but Q was then so large that the Kalman
# contributed almost no useful smoothing.
#
# The new default is intentionally anisotropic:
#   - along-route progress s: moderate process variance, so the GRU/polynomial
#     prior can remove a small amount of frame-to-frame SoftMS noise;
#   - cross-track e: very large process variance, because the visual measurement
#     is more reliable than the motion prior laterally;
#   - visual updates cannot modify velocity, so one visual residual cannot poison
#     later motion predictions.
# Innovation/posterior/final-step clipping remains effectively disabled so a bad
# prior can always recover immediately to the current visual measurement.
# No Route-B/C reference error is used inside the estimator.
KALMAN_R_MIN_VAR = float(os.environ.get("UAVSAT_KF_R_MIN", "0.25"))
KALMAN_R_MAX_VAR = float(os.environ.get("UAVSAT_KF_R_MAX", "9.0"))

KALMAN_USE_MS_CONFIDENCE = False
VISUAL_CONFIDENCE_FLOOR = 1.0
VISUAL_CONFIDENCE_CEIL = 1.0
KALMAN_NIS_CONFIDENCE_BOOST = 0.0
KALMAN_NIS_MAX_R_SCALE = 1.0
ACQ_LOW_CONF_VARIANCE_GAIN = 0.0

# A little more prior weight only on route progress; cross-track stays visual-first.
KALMAN_Q_PROGRESS = float(os.environ.get("UAVSAT_KF_Q_PROGRESS", "36.0"))
KALMAN_Q_CROSS = float(os.environ.get("UAVSAT_KF_Q_CROSS", "144.0"))
KALMAN_Q_VELOCITY = float(os.environ.get("UAVSAT_KF_Q_VELOCITY", "1.0"))

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
    os.environ.get("UAVSAT_KF_VELOCITY_CORRECTION", "0.0")
)
KALMAN_FINAL_STEP_SLACK_M = float(
    os.environ.get("UAVSAT_KF_STEP_SLACK", "1000.0")
)
KALMAN_FINAL_STEP_MAX_M = float(
    os.environ.get("UAVSAT_KF_STEP_MAX", "1000.0")
)

# Keep the current-progress no-ahead cap, but do not add a second reference-speed
# envelope after the Kalman update.  This makes KF/no-KF differ by fusion only.
CONTROLLED_GT_MOTION_ENVELOPE = os.environ.get(
    "UAVSAT_CONTROLLED_GT_MOTION_ENVELOPE", "0"
) == "1"
CONTROLLED_PACE_ASSIST = os.environ.get(
    "UAVSAT_CONTROLLED_PACE_ASSIST", "0"
) == "1"
CONTROLLED_MAX_STEP_RATIO = float(
    os.environ.get("UAVSAT_CONTROLLED_MAX_STEP_RATIO", "2.0")
)
CONTROLLED_PACE_MIN_RATIO = float(
    os.environ.get("UAVSAT_CONTROLLED_PACE_MIN_RATIO", "0.95")
)
CONTROLLED_PACE_CATCHUP_GAIN = float(
    os.environ.get("UAVSAT_CONTROLLED_PACE_CATCHUP_GAIN", "0.25")
)
CONTROLLED_PACE_MAX_EXTRA_M = float(
    os.environ.get("UAVSAT_CONTROLLED_PACE_MAX_EXTRA_M", "3.0")
)

# ---------------------------------------------------------------------------
# Compact temporal state
# ---------------------------------------------------------------------------
USE_SATELLITE_CONTEXT_IN_GRU = False
RNN_NUMERIC_DIM = 8

# ---------------------------------------------------------------------------
# Training balance
# ---------------------------------------------------------------------------
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
    f"polynomial_Kalman_multirate-{TEMPORAL_TRAINING_PROTOCOL}_v7_anisotropic"
)

WAYPOINT_DIR = PROJECT_ROOT.parent / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}
