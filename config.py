from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "continuous_progress_visual_rnn_v11"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "continuous_progress_visual_rnn_A_only.pt"

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

# v11: plain recurrent network.
RNN_HIDDEN_DIM = 256
RNN_FEATURE_DIM = 128
RNN_DROPOUT = 0.10
RNN_HEADING_RESIDUAL_MAX_DEG = 25.0

# Full 6x6 is constructed, but only the forward half is passed to the RNN.
FORWARD_CANDIDATE_COUNT = 18

# Hard displacement bound requested by the experiment.
# Zero motion is valid.
MAX_STEP_M_PER_FRAME = 3.0

# Candidate -> route progress projection window.
CANDIDATE_PROJECT_BACK_M = 1.0
CANDIDATE_PROJECT_FORWARD_M = 16.0

# Second-order polynomial inertia:
# delta_poly = 2*delta_(t-1) - delta_(t-2)
# It is only an upper cap; it can never create movement without image support.
INERTIA_ACCEL_MARGIN_M = 1.25

# Recurrent history is allowed to refine raw image similarity only slightly.
CANDIDATE_REFINEMENT_SCALE = 0.35

TEMPORAL_EPOCHS = 50
TEMPORAL_LR = 2e-4
TEMPORAL_WEIGHT_DECAY = 1e-3
TBPTT_STEPS = 32
GRAD_CLIP_NORM = 5.0

TEMPORAL_TRAIN_FRACTION = 0.82
TEMPORAL_EARLY_STOPPING_PATIENCE = 7
TEMPORAL_MIN_EPOCHS_BEFORE_STOP = 12

# GT is never a model input/search center.
TEACHER_FORCING_RATIO = 0.0

# Prevent the RNN from becoming a memorized frame counter.
TRAIN_HIDDEN_RESET_PROB = 0.03
TRAIN_REPEAT_FRAME_PROB = 0.08

LOSS_PROGRESS = 1.00
LOSS_POSITION = 0.35
LOSS_STEP = 0.70
LOSS_CANDIDATE_CE = 0.30
LOSS_HEADING = 0.08
LOSS_AHEAD = 0.55
LOSS_VARIANCE_NLL = 0.03
AHEAD_TOLERANCE_M = 3.0

# Label-only sequential GT projection bound.
GT_LABEL_MAX_FORWARD_M = 6.0

# 1D progress-only Kalman. There is no velocity state.
KALMAN_INIT_PROGRESS_VAR = 4.0
KALMAN_Q_PROGRESS = 0.20
KALMAN_R_MIN_VAR = 0.25
KALMAN_R_MAX_VAR = 16.0

JUMP_TOLERANCE_M = 3.0
VIDEO_FPS = 12.0
VIDEO_WIDTH = 1800
VIDEO_HEIGHT = 900
FRAME_LABEL_INTERVAL = 100
PROCESS_SNAPSHOT_COUNT = 12

SEED = 2031
DEVICE = "cuda"
FEATURE_CACHE_DTYPE = "float16"
