from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# =============================================================================
# Experiment
# =============================================================================
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visual_motion_gated_route_lstm_v5"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "visual_motion_route_lstm_A_only.pt"

ROUTE_ROOTS = [
    Path("/yh/study/new_data_2/model_dataset_new_1_flight"),
    Path("/yh/study/new_data_2/model_dataset_new_2_flight"),
    Path("/yh/study/new_data/model_dataset_flight"),
]
ROUTE_NAMES = ["route_A", "route_B", "route_C"]

WAYPOINT_DIR = PROJECT_ROOT / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}

SAT_IMAGE = Path(
    "/yh/study/sim_data/sim_competition_crop_check/"
    "sim_map_competition_roi_crop.png"
)
SAT_JSON = Path(
    "/yh/study/sim_data/sim_competition_crop_check/"
    "sim_map_competition_roi_crop_worldfile_epsg3826.json"
)

# =============================================================================
# Existing single-frame visual retrieval
# =============================================================================
BACKBONE_NAME = "hf-hub:timm/MobileCLIP2-S2-OpenCLIP"
CLIP_DIM = 512
EMBED_DIM = 512

IMAGE_SIZE = 256
UAV_CENTER_CROP_SIZE = 256
UAV_CENTER_MAX_SQUARE_CROP = False
UAV_RESIZE_AFTER_CROP = None
TRAIN_UAV_AUGMENT = False

SAT_CROP_SIZE = 320
SAT_STRIDE = 32

USE_COORD_ENCODER = False
USE_QAH_MS_RELATION = False
USE_BASIN_RANK_MS = False
MOTION_SPATIAL_SIZE = 4

VISUAL_EPOCHS = 30
VISUAL_LR = 3e-4
VISUAL_WEIGHT_DECAY = 1e-3
VISUAL_BATCH_SIZE = 64
VISUAL_CACHE_BATCH_SIZE = 256
VISUAL_EARLY_STOPPING_PATIENCE = 8
VISUAL_LABEL_SMOOTHING = 0.05
VISUAL_COORD_LOSS_WEIGHT = 0.25

MEANSHIFT_SCORE_TAU = 0.30
MEANSHIFT_BANDWIDTH_M = 8.0
MEANSHIFT_ITERATIONS = 3

GRID_SIZE = 6
CANDIDATE_COUNT = GRID_SIZE * GRID_SIZE
LOCAL_PRIOR_JITTER_M = 12.0
CANDIDATE_CAPTURE_RADIUS_M = 7.5
MIN_TRAIN_CAPTURE_RATE = 0.95

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_GUARD_FRAMES = 16

# =============================================================================
# v5 temporal model
# =============================================================================
# NO fixed/nominal speed exists in v5.
#
# Previous motion state is OBSERVED from previous IMAGE-derived localization:
#   [delta_parallel, delta_cross, acceleration_parallel, acceleration_cross]
#
# The neural network is not allowed to invent a free-running velocity.
LSTM_HIDDEN_DIM = 256
LSTM_FEATURE_DIM = 128
LSTM_DROPOUT = 0.10

OBSERVED_MOTION_STATE_DIM = 4
OBSERVED_MOTION_NORMALIZE_M = 8.0

# Explicit image-pair motion cue from consecutive UAV frames.
# cv2.findTransformECC(EUCLIDEAN) produces:
#   center_dx_norm, center_dy_norm, sin(rotation), 1-cos(rotation), correlation
IMAGE_MOTION_CUE_DIM = 5
ECC_IMAGE_SIZE = 160
ECC_ITERATIONS = 35
ECC_EPSILON = 1e-5

# Soft pseudo-target calibration used only to train the 3-state phase head.
# These are NOT flight speed limits and never move the localization state.
ECC_TRANSLATION_GAIN = 28.0
ECC_ROTATION_SCALE_DEG = 18.0
GT_TRANSLATION_SOFT_SCALE_M = 2.0

# Phase classes.
PHASE_STATIONARY = 0
PHASE_TRANSLATION = 1
PHASE_ROTATION = 2
PHASE_COUNT = 3

# Polynomial is built only from OBSERVED image-derived previous steps:
#   delta_poly = previous_delta + 0.5 * previous_acceleration
# It is only a soft score prior.
POLY_SIGMA_MIN_M = 2.0
POLY_SIGMA_MAX_M = 12.0

# Route-relative start/end context.
ROUTE_CROSS_TRACK_SCALE_M = 30.0
ROUTE_LENGTH_LOG_SCALE_M = 1000.0

# =============================================================================
# Temporal training
# =============================================================================
TEMPORAL_EPOCHS = 50
TEMPORAL_LR = 2e-4
TEMPORAL_WEIGHT_DECAY = 1e-3
TBPTT_STEPS = 32
GRAD_CLIP_NORM = 5.0
EARLY_STOPPING_PATIENCE = 10

TEMPORAL_TRAIN_LEG_FRACTION = 0.70
TEMPORAL_VAL_LEG_FRACTION = 0.15

# Early scheduled sampling only; after this epoch the sequence is fully closed-loop.
TEACHER_CENTER_END_EPOCH = 10

LOSS_RETRIEVAL_CE = 1.00
LOSS_CURRENT_RELATIVE_POSITION = 0.45
LOSS_PHASE_SOFT_CE = 0.35
LOSS_STEP_DISPLACEMENT = 0.30

# =============================================================================
# Waypoint inference
# =============================================================================
# Switching requires actual current image-derived localization to enter the
# endpoint radius. v4's early "progress >= length-radius" shortcut is removed.
INFER_WAYPOINT_REACHED_RADIUS_M = 8.0

# At the FINAL mission waypoint we freeze the terminal IMAGE-derived state.
# This is a mission-end constraint, not a speed model.
TERMINAL_LOCK_ENABLED = True

# =============================================================================
# Final FilterPy smoother
# =============================================================================
# Position-only random-walk Kalman.
# There is NO vx/vy and NO constant-velocity prediction in v5.
KALMAN_Q_POSITION = 0.30
KALMAN_INIT_POSITION_VAR = 9.0
KALMAN_R_MIN_VAR = 1.0
KALMAN_R_MAX_VAR = 25.0

# =============================================================================
# Evaluation / visualization
# =============================================================================
JUMP_TOLERANCE_M = 3.0

VIDEO_FPS = 12.0
VIDEO_WIDTH = 1800
VIDEO_HEIGHT = 900
FRAME_LABEL_INTERVAL = 100
PROCESS_SNAPSHOT_COUNT = 12

SEED = 2027
DEVICE = "cuda"
FEATURE_CACHE_DTYPE = "float16"
