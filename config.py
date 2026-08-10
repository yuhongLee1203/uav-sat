from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pure_visual_lstm_waypoint_inference_v2_aligned"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "pure_visual_lstm_A_only.pt"

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

SAT_IMAGE = Path("/yh/study/sim_data/sim_competition_crop_check/sim_map_competition_roi_crop.png")
SAT_JSON = Path("/yh/study/sim_data/sim_competition_crop_check/sim_map_competition_roi_crop_worldfile_epsg3826.json")

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
LOCAL_PRIOR_JITTER_M = 12.0
CANDIDATE_CAPTURE_RADIUS_M = 7.5
MIN_TRAIN_CAPTURE_RATE = 0.95
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_GUARD_FRAMES = 16

CANDIDATE_COUNT = 36

LSTM_HIDDEN_DIM = 256
LSTM_FEATURE_DIM = 128
LSTM_DROPOUT = 0.10

TEMPORAL_EPOCHS = 50
TEMPORAL_LR = 3e-4
TEMPORAL_WEIGHT_DECAY = 1e-3
TBPTT_STEPS = 32
GRAD_CLIP_NORM = 5.0
EARLY_STOPPING_PATIENCE = 10
TEMPORAL_TRAIN_FRACTION = 0.70
TEMPORAL_VAL_FRACTION = 0.15

TRAIN_CANDIDATE_JITTER_M = 12.0

LOSS_CE = 1.00
# Final deployment selects a discrete Fixed-HardMS anchor.  Cross-entropy is
# therefore primary; the centroid term is only a weak training stabilizer and
# does not define the deployed coordinate.
LOSS_COORD_SMOOTH_L1 = 0.05
LOSS_GAUSSIAN_NLL = 0.00
# Learned next-frame visual displacement. It moves the next local lattice;
# the current image still selects the final Fixed-HardMS anchor.
LOSS_MOTION_SMOOTH_L1 = 1.00
RNN_MAX_NEXT_DISPLACEMENT_M = 6.0

MIN_MEASUREMENT_VARIANCE = 1e-3
MAX_MEASUREMENT_VARIANCE = 225.0

# Inference-only waypoint search. Each proposal remains a 6x6 Fixed-HardMS
# lattice. A small 3x3 bank of overlapping proposals is evaluated around the
# Kalman prediction so normal flight-speed mismatch cannot immediately push the
# true location outside one 6x6 lattice.
RECOVERY_BANK_RADIUS = 1
RECOVERY_BANK_CENTER_STEP_M = 10.0
RECOVERY_GRID_SELECTION_DISTANCE_WEIGHT = 0.02
RECOVERY_GRID_PROGRESS_WEIGHT = 0.0

# Keep waypoint changes conservative.  A visual response close to a future
# waypoint is not sufficient evidence to turn early in repetitive fields.
INFER_WAYPOINT_REACHED_RADIUS_M = 12.0
INFER_WAYPOINT_CONFIRMATION_FRAMES = 3

# Inference-only FilterPy Kalman.
KALMAN_INIT_POSITION_VAR = 16.0
KALMAN_INIT_VELOCITY_VAR = 25.0
KALMAN_Q_POSITION = 0.20
KALMAN_Q_VELOCITY = 0.35
KALMAN_MAX_SPEED_MPS = 15.0
# Local visual modes correct a predicted state but must not overwrite its
# waypoint-inertial progression from a single ambiguous 6x6 response.
KALMAN_MIN_MEASUREMENT_VARIANCE = 400.0
KALMAN_DEFAULT_NOMINAL_SPEED_MPS = 9.4
KALMAN_MAX_VISUAL_INNOVATION_M = 2.0

JUMP_TOLERANCE_M = 3.0

VIDEO_FPS = 12.0
VIDEO_WIDTH = 1800
VIDEO_HEIGHT = 900
FRAME_LABEL_INTERVAL = 100
PROCESS_SNAPSHOT_COUNT = 12

SEED = 2027
DEVICE = "cuda"
FEATURE_CACHE_DTYPE = "float16"
