from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

ARCHITECTURE_NAME = "RouteProgressGRUPolynomialKalman_v25"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "route_progress_gru_polynomial_kalman_v25"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "route_progress_gru_A_only.pt"
LATEST_TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "route_progress_gru_A_only_latest.pt"

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

# Visual retrieval checkpoint compatibility.
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

# v25 navigation observation: do not search the whole route.  Keep the good
# local behavior from v22, but widen the deployment window enough to tolerate
# moderate propagation error.  Motion prior remains soft, so an informative
# image can move the posterior away from the polynomial center.
NAV_GRID_SIZE = 12
NAV_VISUAL_TEMPERATURE = 0.42
NAV_MOTION_PRIOR_SIGMA_M = 20.0
NAV_MOTION_PRIOR_WEIGHT = 0.75
NAV_POSTERIOR_EPS = 1e-8
NAV_MAX_RESPONSE_VARIANCE_M2 = 625.0

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_GUARD_FRAMES = 16

# Route-progress coordinates: s is distance along the ordered waypoint
# polyline; e is signed cross-track displacement.  Waypoint frame_index is not
# used at inference.
WAYPOINT_MIN_LEG_LENGTH_M = 1.0
ROUTE_PROGRESS_SCALE_M = 100.0
ROUTE_CROSS_TRACK_SCALE_M = 25.0
ROUTE_REMAINING_SCALE_M = 100.0
ROUTE_STEP_SCALE_M = 10.0

# Temporal model.
RNN_HIDDEN_DIM = 256
RNN_FEATURE_DIM = 128
RNN_NUMERIC_DIM = 20
RNN_DROPOUT = 0.10
MAX_FORWARD_SPEED_M_PER_FRAME = 12.0
MAX_CROSS_SPEED_M_PER_FRAME = 5.0
MAX_FORWARD_ACCEL_M_PER_FRAME2 = 4.0
MAX_CROSS_ACCEL_M_PER_FRAME2 = 4.0
MAX_POLYNOMIAL_STEP_M_PER_FRAME = 12.0
MAX_MEASUREMENT_CORRECTION_PARALLEL_M = 4.0
MAX_MEASUREMENT_CORRECTION_CROSS_M = 4.0

# Training is sequential TBPTT.  State is carried across chunks and detached,
# never randomly reset to GT every 32 frames.  Early epochs use GT-centered
# visual windows, then decay to full closed loop.
TEMPORAL_EPOCHS = 60
TEMPORAL_LR = 2e-4
TEMPORAL_WEIGHT_DECAY = 1e-3
TBPTT_STEPS = 32
GRAD_CLIP_NORM = 5.0
MOTION_WARMUP_EPOCHS = 8
TEACHER_RATIO_FINAL = 0.0
TEACHER_DECAY_EPOCHS = 24
TRAIN_CENTER_JITTER_M = 6.0
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 0.05
EARLY_STOP_MIN_EPOCH = 18

# Motion is deliberately stronger than visual residual losses in v25.  The
# deployment failure to fix is speed/progress collapse, not single-frame fit.
LOSS_MEASUREMENT = 0.75
LOSS_NEXT_STEP = 2.50
LOSS_VELOCITY = 1.50
LOSS_ACCELERATION = 0.35
LOSS_SPEED = 1.25
LOSS_CROSS_MOTION_REG = 0.02
LOSS_VARIANCE_NLL = 0.05
LOSS_PROGRESS = 1.00

# Route-coordinate external Kalman [s, e, vs, ve].
KALMAN_INIT_PROGRESS_VAR = 4.0
KALMAN_INIT_CROSS_VAR = 4.0
KALMAN_INIT_VELOCITY_VAR = 9.0
KALMAN_Q_PROGRESS = 0.50
KALMAN_Q_CROSS = 0.35
KALMAN_Q_VELOCITY = 0.75
KALMAN_R_MIN_VAR = 0.25
KALMAN_R_MAX_VAR = 625.0
KALMAN_NIS_SOFT_THRESHOLD = 9.21
KALMAN_NIS_MAX_R_SCALE = 9.0

# Composite early-stop score = MLE + weights * temporal fidelity.  This avoids
# selecting a checkpoint that obtains tolerable Route-A error by moving too
# slowly.
EARLY_SCORE_SPEED_WEIGHT = 2.0
EARLY_SCORE_PROGRESS_WEIGHT = 0.15

JUMP_TOLERANCE_M = 3.0
VIDEO_FPS = 12.0
VIDEO_WIDTH = 1800
VIDEO_HEIGHT = 900

SEED = 2032
DEVICE = "cuda"
FEATURE_CACHE_DTYPE = "float16"
