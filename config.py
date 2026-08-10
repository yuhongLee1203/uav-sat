from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "route_coordinate_gru_kalman_v3"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "route_coordinate_gru_A_only.pt"

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

# Visual retrieval -- compatible with the existing visual_localizer.py.
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

GRID_SIZE = 6
LOCAL_PRIOR_JITTER_M = 12.0
CANDIDATE_CAPTURE_RADIUS_M = 7.5
MIN_TRAIN_CAPTURE_RATE = 0.95
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_GUARD_FRAMES = 16

# Route search. Every adjacent JSON waypoint pair is one mission leg.
ROUTE_CANDIDATE_COUNT = 36
ROUTE_CORRIDOR_HALF_WIDTH_M = 24.0
ROUTE_ENDPOINT_PADDING_M = 12.0
SEARCH_LOOKAHEAD_SECONDS = 3.0

# Recurrent measurement model.
RNN_HIDDEN_DIM = 256
RNN_FEATURE_DIM = 128
RNN_DROPOUT = 0.10

# Kalman route state [s, v, d, vd]. Units are m, m/s, m, m/s.
KALMAN_STATE_DIM = 4
KALMAN_INIT_PROGRESS_VAR = 9.0
KALMAN_INIT_FORWARD_SPEED_VAR = 9.0
KALMAN_INIT_CROSS_TRACK_VAR = 9.0
KALMAN_INIT_CROSS_SPEED_VAR = 4.0
KALMAN_Q_PROGRESS = 0.10
KALMAN_Q_FORWARD_SPEED = 0.25
KALMAN_Q_CROSS_TRACK = 0.10
KALMAN_Q_CROSS_SPEED = 0.15
KALMAN_MIN_VARIANCE = 1e-4
KALMAN_MAX_MEASUREMENT_VAR = 100.0

# The physical speed envelope is learned from Route-A TRAINING waypoint legs:
# speed = leg length / real waypoint timestamp duration.
MIN_FORWARD_SPEED_MPS = 0.5
MIN_CROSS_SPEED_MPS = 0.2
MIN_DT_SECONDS = 0.02
MAX_DT_SECONDS = 2.0

# Temporal training.
EPOCHS = 50
LR = 2e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP_NORM = 5.0
SEED = 2027
DEVICE = "cuda"
FEATURE_CACHE_DTYPE = "float16"
EARLY_STOPPING_PATIENCE = 10
TBPTT_STEPS = 32
TEMPORAL_TRAIN_LEG_FRACTION = 0.70
TEMPORAL_VAL_LEG_FRACTION = 0.15

LOSS_FINAL_SMOOTH_L1 = 1.00
LOSS_MEASUREMENT_GAUSSIAN_NLL = 0.20
LOSS_PREDICTION_SMOOTH_L1 = 0.20
LOSS_VELOCITY_SMOOTH_L1 = 0.10

JUMP_TOLERANCE_M = 3.0
VAL_RPE_WEIGHT = 0.25
VAL_JUMP_WEIGHT = 0.10

# Visualization.
VIDEO_FPS = 12.0
VIDEO_WIDTH = 1800
VIDEO_HEIGHT = 900
FRAME_LABEL_INTERVAL = 100
PROCESS_SNAPSHOT_COUNT = 12
