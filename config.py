from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# =============================================================================
# Experiment
# =============================================================================
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "route_conditioned_inertial_lstm_v4"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "route_inertial_lstm_A_only.pt"

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
# Frozen public backbone + task-specific visual retrieval heads
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
# Route-conditioned recurrent model
# =============================================================================
# The temporal network NEVER receives absolute GPS/XY.
#
# It receives:
#   - current UAV/SAT visual embeddings and similarity
#   - relative candidate offsets around the previous VISUAL state
#   - route-relative start/end context (unit direction, remaining ratio,
#     normalized cross-track)
#   - previous model motion state [v_parallel, v_cross, a_parallel, a_cross]
#   - previous LSTM hidden/cell
#
# Start/end waypoint coordinates are converted to this translation-invariant
# route frame BEFORE entering the network.
# =============================================================================
LSTM_HIDDEN_DIM = 256
LSTM_FEATURE_DIM = 128
LSTM_DROPOUT = 0.10

# Previous motion state is in metres per image step, not global coordinates.
MOTION_STATE_DIM = 4
MOTION_MAX_V_M_PER_FRAME = 6.0
MOTION_MAX_A_M_PER_FRAME2 = 3.0

# Polynomial prior:
#   delta_poly = v_(t-1) + 0.5 * a_(t-1)
# It is only a SOFT SCORE PRIOR inside the current 6x6 candidate lattice.
# It NEVER moves the localization state by itself.
POLY_SIGMA_MIN_M = 2.0
POLY_SIGMA_MAX_M = 12.0

# Translation-invariant route context normalization.
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

# Split by complete Route-A waypoint legs, not arbitrary frames.
TEMPORAL_TRAIN_LEG_FRACTION = 0.70
TEMPORAL_VAL_LEG_FRACTION = 0.15

# Scheduled closed-loop candidate-center training:
# epoch 0 starts teacher-centered; by epoch 12 the center is fully model-driven.
# The FINAL majority of training therefore exactly matches inference.
TEACHER_CENTER_END_EPOCH = 12

# Standard losses.
LOSS_RETRIEVAL_CE = 1.00
# Relative offset loss: target is GT - current search center, never absolute XY.
LOSS_RELATIVE_OFFSET = 0.30
# Supervise the recurrent velocity state with next-frame relative displacement.
LOSS_VELOCITY = 0.35
# Supervise second-order acceleration state.
LOSS_ACCELERATION = 0.10

# =============================================================================
# Inference waypoint use
# =============================================================================
# Inference uses mission waypoint coordinates/order only.
# waypoint frame_index/timestamp are NOT used for switching.
#
# The current 6x6 lattice is ALWAYS centered on the previous visual measurement,
# never on the polynomial prediction or Kalman prediction. This is the key
# anti-runaway design.
INFER_WAYPOINT_REACHED_RADIUS_M = 14.0

# =============================================================================
# Final output smoother: FilterPy Kalman
# =============================================================================
# RNN visual state drives search. Kalman is only the FINAL smoother.
# Kalman prediction NEVER controls the next SAT candidate center.
KALMAN_INIT_POSITION_VAR = 9.0
KALMAN_INIT_VELOCITY_VAR = 16.0
KALMAN_Q_POSITION = 0.40
KALMAN_Q_VELOCITY = 0.80
KALMAN_R_MIN_VAR = 1.0
KALMAN_R_MAX_VAR = 36.0

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
