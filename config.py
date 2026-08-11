from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stable_visual_inertial_rnn_v14"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "stable_visual_inertial_rnn_A_only.pt"

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
# Existing Route-A-only visual retrieval
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
CANDIDATE_COUNT = 36
CANDIDATE_CAPTURE_RADIUS_M = 7.5

LOCAL_PRIOR_JITTER_M = 12.0
MIN_TRAIN_CAPTURE_RATE = 0.95
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_GUARD_FRAMES = 16

# =============================================================================
# v14 Stable Visual-Inertial RNN
# =============================================================================
#
# Sensor/model input:
#   - current UAV image embedding
#   - current 6x6 SAT image embeddings/similarities
#   - previous image-derived RNN hidden state
#   - previous image-derived motion state
#
# NOT neural inputs:
#   - current/previous GT
#   - GPS
#   - IMU
#   - yaw
#   - timestamp
#   - waypoint index
#   - frame index
#
# Why FULL 6x6:
# Forward-half hard masking repeatedly made errors unrecoverable. v14 retains
# all 36 patches. Previous RNN motion is given as a SOFT learned directional
# prior when refining the 36 visual candidates. Rear/side patches remain legal.
# =============================================================================

RNN_HIDDEN_DIM = 256
RNN_FEATURE_DIM = 128
RNN_DROPOUT = 0.10

# User-requested maximum motion state and output displacement.
MAX_STEP_M_PER_FRAME = 10.0

# Residual only performs sub-anchor refinement. It cannot become a free
# position generator.
MAX_RESIDUAL_M = 2.75

# Relative search-grid geometry scale used only for deterministic positional
# encoding of SAT candidate centers.
CANDIDATE_OFFSET_SCALE_M = 14.0

# Learned temporal candidate refinement can only make a bounded change to the
# already-trained visual similarity.
CANDIDATE_REFINEMENT_SCALE = 0.45

# Stop target for motion supervision.
STOP_STEP_THRESHOLD_M = 0.30
STOP_POS_WEIGHT_MAX = 20.0

# =============================================================================
# Stable scheduled-sampling training
# =============================================================================
#
# GT may choose the SEARCH CENTER during early Route-A training, but it is
# NEVER passed into RNN forward_step().
#
# This is scheduled sampling / teacher centering, not GT-as-model-input.
# Closed-loop validation always uses 0% teacher.
# =============================================================================

TEACHER_CENTER_WARMUP_EPOCHS = 5
TEACHER_CENTER_END_EPOCH = 25
TEACHER_CENTER_JITTER_M = 3.0

TEMPORAL_EPOCHS = 50
TEMPORAL_LR = 2e-4
TEMPORAL_WEIGHT_DECAY = 1e-3
TBPTT_STEPS = 32
GRAD_CLIP_NORM = 5.0

TEMPORAL_TRAIN_FRACTION = 0.82
TEMPORAL_EARLY_STOPPING_PATIENCE = 8
TEMPORAL_MIN_EPOCHS_BEFORE_STOP = 15

# Losses. No gigantic quadratic "ahead" loss.
LOSS_POSITION = 1.00
LOSS_MOTION = 0.45
LOSS_STOP = 0.20
LOSS_ACCELERATION = 0.10
LOSS_CANDIDATE_CE = 0.35
LOSS_RESIDUAL = 0.05
LOSS_VARIANCE_NLL = 0.02

# =============================================================================
# Final position-only Kalman
# =============================================================================
#
# State [x,y], F=I. No vx/vy and therefore no Kalman fixed-speed behavior.
# =============================================================================
KALMAN_INIT_VAR = 3.0
KALMAN_Q_VAR = 0.60
KALMAN_R_MIN_VAR = 0.20
KALMAN_R_MAX_VAR = 16.0

JUMP_TOLERANCE_M = 3.0

# =============================================================================
# Heading
# =============================================================================
#
# Heading is derived from the RNN motion vector in ENU:
# atan2(delta_y, delta_x). It is undefined at true/estimated stop.
# Renderer projects the world-coordinate arrow endpoint through the map mapper,
# avoiding the old ~90-degree screen-coordinate error.
# =============================================================================
HEADING_MIN_MOTION_M = 0.25
HEADING_RENDER_ARROW_LENGTH_M = 12.0

# =============================================================================
# Rendering
# =============================================================================
VIDEO_FPS = 12.0
VIDEO_WIDTH = 1800
VIDEO_HEIGHT = 900

SEED = 2031
DEVICE = "cuda"
FEATURE_CACHE_DTYPE = "float16"
