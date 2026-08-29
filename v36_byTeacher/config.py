import os
from pathlib import Path
from config_base import *

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Experiment identity
# ---------------------------------------------------------------------------
# Keep the existing v36 default backbone.  The architecture rewrite changes the
# temporal/localization flow, not the visual backbone; an environment variable
# can still select one of the existing ablation backbones.
BACKBONE_KEY = os.environ.get("UAVSAT_BACKBONE", "mobileclip2_s2").strip().lower()
if BACKBONE_KEY not in BACKBONE_SPECS:
    raise ValueError(
        "UAVSAT_BACKBONE must be one of %s; got %r"
        % (sorted(BACKBONE_SPECS), BACKBONE_KEY)
    )
BACKBONE_NAME, CLIP_DIM = BACKBONE_SPECS[BACKBONE_KEY]

EXPERIMENT_FRAME_COUNT = int(os.environ.get("UAVSAT_EXPERIMENT_FRAME_COUNT", "2"))
if EXPERIMENT_FRAME_COUNT not in (1, 2):
    raise ValueError("UAVSAT_EXPERIMENT_FRAME_COUNT must be 1 or 2")
TEMPORAL_WINDOW_FRAMES = EXPERIMENT_FRAME_COUNT

ARCHITECTURE_NAME = (
    "V36_byTeacher_Causal_MS1Forward3x6_GRUMotionPolynomial_"
    "PositionKalman_MS2Full6x6_NoReferenceInput_v8"
)

BACKBONE_OUTPUT_DIR = PROJECT_ROOT / "output" / BACKBONE_KEY
DEFAULT_OUTPUT_DIR = BACKBONE_OUTPUT_DIR / f"{EXPERIMENT_FRAME_COUNT}frame_no_reference_input"
RUN_TAG = os.environ.get("UAVSAT_RUN_TAG", "").strip()
OUTPUT_DIR = (
    BACKBONE_OUTPUT_DIR / "experiments" / RUN_TAG
    if RUN_TAG
    else DEFAULT_OUTPUT_DIR
)
CHECKPOINT_DIR = BACKBONE_OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / f"visual_A_train_B_val_global_{BACKBONE_KEY}_v8.pt"
TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / f"temporal_A_train_B_val_ms1_kf_ms2_{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_v8.pt"
)
LATEST_TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / f"temporal_A_train_B_val_ms1_kf_ms2_{EXPERIMENT_FRAME_COUNT}frame_{BACKBONE_KEY}_v8_latest.pt"
)
FEATURE_CACHE_DIR = BACKBONE_OUTPUT_DIR / "feature_cache_v8"

TRAIN_ROUTE_NAME = "route_A"
VALIDATION_ROUTE_NAME = "route_B"
TEST_ROUTE_NAME = "route_C"

# ---------------------------------------------------------------------------
# Causal localization protocol
# ---------------------------------------------------------------------------
# Inference never reads the current frame reference/GT position.  The only
# position used to open MS1 is the previous MS2 output.  The known planned-route
# start is used once for initialization; after that, every position is predicted.
REFERENCE_POSITION_AS_INFERENCE_INPUT = False
KNOWN_START_FROM_PLANNED_ROUTE = True
HARD_MOTION_LIMITS_ENABLED = False

MS1_BASE_GRID_SIZE = 6
MS1_FORWARD_ROWS = 3
MS1_FORWARD_COLS = 6
MS1_CANDIDATE_COUNT = MS1_FORWARD_ROWS * MS1_FORWARD_COLS
MS2_GRID_SIZE = 6
MS2_CANDIDATE_COUNT = MS2_GRID_SIZE * MS2_GRID_SIZE
FORWARD_SEARCH_ORIGIN_BACKSHIFT_M = float(
    os.environ.get("UAVSAT_FORWARD_ORIGIN_BACKSHIFT_M", "0.0")
)

# Keep the thesis Mean-Shift controls configurable.  These are decoder
# hyperparameters, not motion/heading clamps.
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

# ---------------------------------------------------------------------------
# GRU inputs/outputs
# ---------------------------------------------------------------------------
# Numeric input = MS1 offset from previous final(2)
#               + MS1 temporal displacement(2)
#               + previous predicted velocity XY(2)
#               + previous predicted acceleration XY(2)
#               + cos/sin(previous predicted heading)(2)
# No Mean-Shift variance/reference position is fed to the GRU.
RNN_NUMERIC_DIM = 10
RNN_DROPOUT = float(os.environ.get("UAVSAT_RNN_DROPOUT", "0.05"))

# ---------------------------------------------------------------------------
# Position-only external Kalman
# ---------------------------------------------------------------------------
# The GRU already predicts motion.  Kalman therefore fuses exactly two current
# position hypotheses: inertial prior X_pre and MS1 visual measurement.
# No reference-dependent clipping, speed envelope, turn envelope or step cap.
KALMAN_POSITION_INIT_VAR = float(os.environ.get("UAVSAT_KF_INIT_VAR", "16.0"))
KALMAN_POSITION_PROCESS_VAR_X = float(os.environ.get("UAVSAT_KF_Q_X", "4.0"))
KALMAN_POSITION_PROCESS_VAR_Y = float(os.environ.get("UAVSAT_KF_Q_Y", "4.0"))
KALMAN_NUMERICAL_VARIANCE_EPS = float(os.environ.get("UAVSAT_KF_VAR_EPS", "1e-4"))

# ---------------------------------------------------------------------------
# Visual retrieval training
# ---------------------------------------------------------------------------
# Route A is trained against the complete satellite gallery.  GT/reference XY
# selects the supervised target only; it never determines a local search window.
VISUAL_GLOBAL_TRAINING = True
VISUAL_COORD_LOSS_WEIGHT = float(os.environ.get("UAVSAT_VISUAL_COORD_W", "0.05"))
VISUAL_EARLY_STOPPING_PATIENCE = int(
    os.environ.get("UAVSAT_VISUAL_PATIENCE", str(VISUAL_EARLY_STOPPING_PATIENCE))
)

# ---------------------------------------------------------------------------
# Temporal supervision
# ---------------------------------------------------------------------------
# v and a targets are derived directly from the Route-A reference position
# sequence by finite differences.  For frame t (dt=1 frame):
#   v_t = (p_{t+1} - p_{t-1}) / 2
#   a_t = p_{t+1} - 2 p_t + p_{t-1}
# so v_t + 0.5 a_t = p_{t+1} - p_t for interior frames.
LOSS_NEXT_STEP = float(os.environ.get("UAVSAT_LOSS_NEXT_STEP", "3.0"))
LOSS_VELOCITY = float(os.environ.get("UAVSAT_LOSS_VELOCITY", "1.0"))
LOSS_ACCELERATION = float(os.environ.get("UAVSAT_LOSS_ACCELERATION", "0.5"))
LOSS_HEADING = float(os.environ.get("UAVSAT_LOSS_HEADING", "1.0"))
LOSS_TURN_RATE = float(os.environ.get("UAVSAT_LOSS_TURN_RATE", "0.25"))

TEMPORAL_LR = float(os.environ.get("UAVSAT_TEMPORAL_LR", "1e-4"))
TEMPORAL_WEIGHT_DECAY = float(os.environ.get("UAVSAT_TEMPORAL_WEIGHT_DECAY", "1e-3"))
TBPTT_STEPS = int(os.environ.get("UAVSAT_TBPTT_STEPS", "32"))
GRAD_CLIP_NORM = float(os.environ.get("UAVSAT_GRAD_CLIP_NORM", "5.0"))
EARLY_STOP_PATIENCE = int(os.environ.get("UAVSAT_PATIENCE", "15"))
EARLY_STOP_MIN_DELTA = float(os.environ.get("UAVSAT_EARLY_DELTA", "0.02"))
EARLY_STOP_MIN_EPOCH = int(os.environ.get("UAVSAT_EARLY_MIN_EPOCH", "20"))

LATENCY_WARMUP_FRAMES = 30

PROTOCOL_NAME = (
    "A-train_B-validation_C-test; known planned start only; "
    "MS1 forward3x6 from previous MS2 final; GRU predicts v/a/heading; "
    "polynomial predicts next inertial position; position Kalman fuses MS1+prior; "
    "MS2 full6x6 refines final; reference positions are supervision/metrics only"
)

WAYPOINT_DIR = PROJECT_ROOT.parent / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}
