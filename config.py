from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

ARCHITECTURE_NAME = "WaypointRouteFrameGRUKalman_v21"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "waypoint_routeframe_gru_kalman_v21"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "waypoint_routeframe_gru_A_only.pt"
LATEST_TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "waypoint_routeframe_gru_A_only_latest.pt"

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

# -----------------------------------------------------------------------------
# Frozen MobileCLIP retrieval model. Keep these names compatible with the
# existing Route-A-only visual checkpoint and visual_localizer.py.
# -----------------------------------------------------------------------------
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
LOCAL_PRIOR_JITTER_M = 12.0
CANDIDATE_CAPTURE_RADIUS_M = 7.5
MIN_TRAIN_CAPTURE_RATE = 0.95

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_GUARD_FRAMES = 16

# -----------------------------------------------------------------------------
# Waypoint-conditioned route frame.
# The network never receives waypoint frame_index/timestamps at inference.
# Only waypoint coordinates and the monotonically progressing leg are used.
# -----------------------------------------------------------------------------
ROUTE_DISTANCE_SCALE_M = 100.0
ROUTE_CROSS_TRACK_SCALE_M = 25.0
ROUTE_STEP_SCALE_M = 10.0
WAYPOINT_MIN_LEG_LENGTH_M = 1.0

# -----------------------------------------------------------------------------
# Recurrent state estimator.
# The GRU predicts motion in route coordinates:
# [v_parallel, v_cross, a_parallel, a_cross].
# -----------------------------------------------------------------------------
RNN_HIDDEN_DIM = 256
RNN_FEATURE_DIM = 128
RNN_NUMERIC_DIM = 15
RNN_DROPOUT = 0.10
MAX_FORWARD_SPEED_M_PER_FRAME = 10.0
MAX_CROSS_SPEED_M_PER_FRAME = 5.0
MAX_FORWARD_ACCEL_M_PER_FRAME2 = 5.0
MAX_CROSS_ACCEL_M_PER_FRAME2 = 4.0
MAX_POLYNOMIAL_STEP_M_PER_FRAME = 10.0
MAX_MEASUREMENT_CORRECTION_PARALLEL_M = 5.0
MAX_MEASUREMENT_CORRECTION_CROSS_M = 5.0

# -----------------------------------------------------------------------------
# Temporal training.
# Epochs 1..MOTION_WARMUP_EPOCHS use a GT-centered local search. Afterwards the
# search center becomes increasingly autoregressive. Early stopping is based on
# held-out Route-A CLOSED-LOOP episodes, not teacher-forced training loss.
# -----------------------------------------------------------------------------
TEMPORAL_EPOCHS = 60
TEMPORAL_LR = 2e-4
TEMPORAL_WEIGHT_DECAY = 1e-3
TBPTT_STEPS = 32
GRAD_CLIP_NORM = 5.0
MOTION_WARMUP_EPOCHS = 8
TEACHER_RATIO_FINAL = 0.10
TEACHER_DECAY_EPOCHS = 30
TRAIN_CENTER_JITTER_M = 6.0
VAL_EPISODE_LENGTH = 96
VAL_EPISODE_COUNT = 4
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA_M = 0.10
EARLY_STOP_MIN_EPOCH = 12

# Losses.
LOSS_MEASUREMENT = 1.00
LOSS_NEXT_STEP = 1.50
LOSS_VELOCITY = 0.45
LOSS_ACCELERATION = 0.25
LOSS_VARIANCE_NLL = 0.08
LOSS_CONFIDENCE = 0.10
LOSS_CROSS_MOTION_REG = 0.02
CONFIDENCE_TARGET_SIGMA_M = 8.0

# -----------------------------------------------------------------------------
# External Kalman filter [x, y, vx, vy]. The GRU/polynomial supplies the motion
# prediction; visual measurement + learned covariance supplies z/R.
# -----------------------------------------------------------------------------
KALMAN_INIT_POSITION_VAR = 4.0
KALMAN_INIT_VELOCITY_VAR = 9.0
KALMAN_Q_POSITION = 0.35
KALMAN_Q_VELOCITY = 0.60
KALMAN_R_MIN_VAR = 0.25
KALMAN_R_MAX_VAR = 64.0
MAX_FINAL_SPEED_M_PER_FRAME = 12.0

# Diagnostics.
JUMP_TOLERANCE_M = 3.0
VIDEO_FPS = 12.0
VIDEO_WIDTH = 1800
VIDEO_HEIGHT = 900

SEED = 2031
DEVICE = "cuda"
FEATURE_CACHE_DTYPE = "float16"
