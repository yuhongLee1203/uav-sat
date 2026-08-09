from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================================
# Experiment protocol
# ============================================================================
# Visual retrieval: Route A only, trained FROM SCRATCH.
# Recurrent temporal model: Route A only, trained FROM SCRATCH.
# Final evaluation: Route B + Route C.
#
# Each temporal step consumes ONE current-frame retrieval result.
# No multi-frame image tensor is fed into the GRU.
# History is carried only by:
#   1) GRU hidden state h_t
#   2) physical Kalman state [x,y,vx,vy,ax,ay]
# ============================================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "route_rnn_filterpy_full_retrain"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "route_gru_A_only.pt"

DATASETS_ROOT = Path("/yh/study/new_data_2")
ROUTE_ROOTS = [
    DATASETS_ROOT / "model_dataset_new_1_flight",
    DATASETS_ROOT / "model_dataset_new_2_flight",
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

# ============================================================================
# Visual retrieval -- same single-frame retrieval stage as before
# ============================================================================
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
VISUAL_CACHE_BATCH_SIZE = 64
VISUAL_EARLY_STOPPING_PATIENCE = 8
VISUAL_LABEL_SMOOTHING = 0.05
VISUAL_COORD_LOSS_WEIGHT = 0.25

MEANSHIFT_SCORE_TAU = 0.30
MEANSHIFT_BANDWIDTH_M = 8.0
MEANSHIFT_ITERATIONS = 3

# Used only by the independent visual-retrieval training stage.
GRID_SIZE = 6
LOCAL_PRIOR_JITTER_M = 12.0
CANDIDATE_CAPTURE_RADIUS_M = 7.5
MIN_TRAIN_CAPTURE_RATE = 0.95

# Visual-localizer compatibility.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_GUARD_FRAMES = 16

# ============================================================================
# Mission-route forward candidate search
# ============================================================================
ROUTE_CANDIDATE_COUNT = 36

# Only candidates at or ahead of the already accepted mission progress are
# searchable.  The corridor is widened if fewer than 36 gallery patches exist,
# but the code never intentionally searches behind accepted progress.
ROUTE_FORWARD_HORIZON_M = 45.0
ROUTE_CORRIDOR_HALF_WIDTH_M = 24.0
ROUTE_ENDPOINT_PADDING_M = 10.0

# Candidate ranking inside the valid forward corridor.
ROUTE_CROSS_TRACK_COST = 0.35

# ============================================================================
# GRU measurement model
# ============================================================================
RNN_HIDDEN_DIM = 256
RNN_FEATURE_DIM = 128
RNN_DROPOUT = 0.10

# GRU produces:
#   measurement coordinate z_t = [x,y]
#   measurement variance R_t = [var_x,var_y]
KALMAN_MIN_VARIANCE = 1e-4
KALMAN_MAX_MEASUREMENT_VAR = 400.0

# ============================================================================
# Constant-acceleration Kalman state
# ============================================================================
# State order everywhere:
#   [x, y, vx, vy, ax, ay]
KALMAN_INIT_POSITION_VAR = 16.0
KALMAN_INIT_VELOCITY_VAR = 25.0
KALMAN_INIT_ACCELERATION_VAR = 9.0

# Process noise diagonal base values.  FilterPy inference and differentiable
# PyTorch training use the same state transition and the same Q definition.
KALMAN_Q_POSITION = 0.10
KALMAN_Q_VELOCITY = 0.35
KALMAN_Q_ACCELERATION = 0.50

# When the mission controller advances to a new waypoint leg, the old speed
# magnitude is aligned with the new known leg direction.  This is treated as a
# known mission control event, not as a visual heuristic.
LEG_CHANGE_VELOCITY_COVARIANCE_BOOST = 4.0
LEG_CHANGE_ACCELERATION_COVARIANCE_BOOST = 4.0

# ============================================================================
# Temporal training
# ============================================================================
EPOCHS = 50
LR = 2e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP_NORM = 5.0
SEED = 2027
DEVICE = "cuda"
FEATURE_CACHE_DTYPE = "float16"
EARLY_STOPPING_PATIENCE = 10

# Truncated BPTT.  Still one frame per recurrent step; this only determines
# how many recurrent steps share one backward pass.
TBPTT_STEPS = 32

# Curriculum only affects the SEARCH CENTER during Route-A training.
# It begins near GT so early untrained closed-loop states cannot immediately
# lose the route, then reaches fully model-predicted search.
TEACHER_SEARCH_EPOCHS = 20

# Standard losses only.  No invented loss names.
LOSS_FINAL_SMOOTH_L1 = 1.00
LOSS_MEASUREMENT_GAUSSIAN_NLL = 0.20
LOSS_PREDICTION_SMOOTH_L1 = 0.30
LOSS_VELOCITY_SMOOTH_L1 = 0.10
LOSS_ACCELERATION_SMOOTH_L1 = 0.03

# Temporal Route-A split is by complete waypoint legs, not arbitrary frames.
TEMPORAL_TRAIN_LEG_FRACTION = 0.70
TEMPORAL_VAL_LEG_FRACTION = 0.15

# Validation checkpoint objective.
JUMP_TOLERANCE_M = 3.0
VAL_RPE_WEIGHT = 0.25
VAL_JUMP_WEIGHT = 0.05
