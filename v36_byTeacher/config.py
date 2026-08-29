import os
from pathlib import Path
from config_base import *

PROJECT_ROOT = Path(__file__).resolve().parent

# =============================================================================
# v36_byTeacher autonomous closed-loop architecture, stable v3
# =============================================================================
# Inference never uses per-frame reference coordinates to choose a search
# center, route leg, speed limit, turn limit, Kalman gate, or posterior cap.
# Reference coordinates are labels/metrics only.
ARCHITECTURE_NAME = (
    "V36_byTeacher_Autonomous_MS1Forward3x6_Kalman_GRU_"
    "Polynomial_MS2Center6x6_v3_nativeA"
)

BACKBONE_KEY = os.environ.get(
    "UAVSAT_BACKBONE", "mobilenet_v3_small"
).strip().lower()
if BACKBONE_KEY not in BACKBONE_SPECS:
    raise ValueError(
        "UAVSAT_BACKBONE must be one of %s" % sorted(BACKBONE_SPECS)
    )
BACKBONE_NAME, CLIP_DIM = BACKBONE_SPECS[BACKBONE_KEY]

BACKBONE_OUTPUT_DIR = PROJECT_ROOT / "output" / BACKBONE_KEY
RUN_TAG = os.environ.get("UAVSAT_RUN_TAG", "").strip()
OUTPUT_DIR = (
    BACKBONE_OUTPUT_DIR / "experiments" / RUN_TAG
    if RUN_TAG
    else BACKBONE_OUTPUT_DIR / "autonomous_ms1_kf_gru_ms2_v3_nativeA"
)
CHECKPOINT_DIR = BACKBONE_OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = (
    CHECKPOINT_DIR / f"visual_retrieval_A_only_{BACKBONE_KEY}.pt"
)
TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / f"autonomous_motion_gru_A_native_v3_{BACKBONE_KEY}.pt"
)
LATEST_TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / f"autonomous_motion_gru_A_native_v3_{BACKBONE_KEY}_latest.pt"
)
FEATURE_CACHE_DIR = BACKBONE_OUTPUT_DIR / "feature_cache"

# Waypoints are predefined planned-route information. They are used only to
# initialize the known start position and initial heading. They are never
# matched to a current frame using reference coordinates.
WAYPOINT_DIR = PROJECT_ROOT.parent / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}

# -----------------------------------------------------------------------------
# Visual retrieval
# -----------------------------------------------------------------------------
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

MS1_BASE_GRID_SIZE = 6
MS1_FORWARD_ROWS = 3
MS1_FORWARD_COLS = 6
MS1_CANDIDATE_COUNT = MS1_FORWARD_ROWS * MS1_FORWARD_COLS
MS2_GRID_SIZE = 6

# Visual-retrieval training may use reference positions as supervised labels.
# Runtime tracking never uses this jitter to construct a reference-centered
# search window.
LOCAL_PRIOR_JITTER_M = float(
    os.environ.get("UAVSAT_VISUAL_TRAIN_JITTER_M", "12.0")
)

# -----------------------------------------------------------------------------
# Autonomous GRU motion model
# -----------------------------------------------------------------------------
RNN_HIDDEN_DIM = int(os.environ.get("UAVSAT_RNN_HIDDEN", "256"))
RNN_FEATURE_DIM = int(os.environ.get("UAVSAT_RNN_FEATURE", "128"))
# One Route-A sequence is enough; deterministic recurrent training is more
# stable than injecting dropout into this single-route motion learner.
RNN_DROPOUT = float(os.environ.get("UAVSAT_RNN_DROPOUT", "0.0"))
RNN_NUMERIC_DIM = 10

POSITION_INPUT_SCALE_M = float(
    os.environ.get("UAVSAT_POSITION_INPUT_SCALE_M", "50.0")
)
STEP_INPUT_SCALE_M = float(
    os.environ.get("UAVSAT_STEP_INPUT_SCALE_M", "10.0")
)

TEMPORAL_EPOCHS = int(os.environ.get("UAVSAT_TEMPORAL_EPOCHS", "60"))
TEMPORAL_LR = float(os.environ.get("UAVSAT_TEMPORAL_LR", "1e-3"))
TEMPORAL_WEIGHT_DECAY = float(
    os.environ.get("UAVSAT_TEMPORAL_WEIGHT_DECAY", "1e-4")
)
TBPTT_STEPS = int(os.environ.get("UAVSAT_TBPTT_STEPS", "32"))
GRAD_CLIP_NORM = float(os.environ.get("UAVSAT_GRAD_CLIP_NORM", "5.0"))
EARLY_STOP_PATIENCE = int(
    os.environ.get("UAVSAT_EARLY_STOP_PATIENCE", "12")
)
EARLY_STOP_MIN_EPOCH = int(
    os.environ.get("UAVSAT_EARLY_STOP_MIN_EPOCH", "15")
)
EARLY_STOP_MIN_DELTA = float(
    os.environ.get("UAVSAT_EARLY_STOP_MIN_DELTA", "0.01")
)

# Position-derived kinematic supervision.
LOSS_DELTA = float(os.environ.get("UAVSAT_LOSS_DELTA", "5.0"))
LOSS_SPEED = float(os.environ.get("UAVSAT_LOSS_SPEED", "1.0"))
LOSS_ACCELERATION = float(
    os.environ.get("UAVSAT_LOSS_ACCELERATION", "0.25")
)
LOSS_HEADING = float(os.environ.get("UAVSAT_LOSS_HEADING", "2.0"))
LOSS_HEADING_DELTA = float(
    os.environ.get("UAVSAT_LOSS_HEADING_DELTA", "0.5")
)

# -----------------------------------------------------------------------------
# Standard XY Kalman filter
# -----------------------------------------------------------------------------
# No innovation clipping, NIS gate, speed cap, acceleration cap, turn cap,
# posterior correction cap, reference progress cap, or final-step corridor.
KALMAN_INIT_POSITION_VAR = float(
    os.environ.get("UAVSAT_KF_INIT_POS_VAR", "25.0")
)
KALMAN_INIT_VELOCITY_VAR = float(
    os.environ.get("UAVSAT_KF_INIT_VEL_VAR", "25.0")
)
KALMAN_Q_POSITION = float(os.environ.get("UAVSAT_KF_Q_POS", "1.0"))
KALMAN_Q_VELOCITY = float(os.environ.get("UAVSAT_KF_Q_VEL", "1.0"))
KALMAN_R_POSITION = float(os.environ.get("UAVSAT_KF_R_POS", "9.0"))

# Disable inherited legacy controlled-protocol switches.
CONTROLLED_GT_PRIOR = False
CONTROLLED_GT_PRIOR_JITTER_M = 0.0
CONTROLLED_FINAL_PROGRESS_CAP_TO_GT = False
CONTROLLED_GT_MOTION_ENVELOPE = False
CONTROLLED_PACE_ASSIST = False

TEMPORAL_TRAIN_ROUTES = ["route_A"]
TEMPORAL_VALIDATION_ROUTE = "route_C"
TEMPORAL_TEST_ROUTE = "route_B"

LATENCY_WARMUP_FRAMES = int(
    os.environ.get("UAVSAT_LATENCY_WARMUP_FRAMES", "30")
)
