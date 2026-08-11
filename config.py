from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

ARCHITECTURE_NAME = "CRFInertialRNNKalman_v20"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "crf_inertial_rnn_kalman_v20"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
WARMUP_CHECKPOINT = CHECKPOINT_DIR / "candidate_nextstep_warmup_A_only.pt"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "crf_inertial_rnn_A_only.pt"

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
CANDIDATE_COUNT = 36
LOCAL_PRIOR_JITTER_M = 12.0
CANDIDATE_CAPTURE_RADIUS_M = 7.5
MIN_TRAIN_CAPTURE_RATE = 0.95

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_GUARD_FRAMES = 16

# ------------------------------------------------------------------
# v20: CRF-inspired candidate refinement
# ------------------------------------------------------------------
TOKEN_DIM = 128
TEMPORAL_DROPOUT = 0.10
CANDIDATE_NUMERIC_DIM = 9

# Soft target because neighboring 320px SAT crops overlap heavily.
CANDIDATE_TARGET_SIGMA_M = 4.5

# Keep raw retrieval ordering initially, then learn residual calibration.
INITIAL_RAW_LOGIT_WEIGHT = 1.0
INITIAL_MOTION_PRIOR_WEIGHT = 0.20
INITIAL_FORWARD_PRIOR_WEIGHT = 0.08
MOTION_PRIOR_SIGMA_M = 7.0

# ------------------------------------------------------------------
# Stateful RNN
# ------------------------------------------------------------------
RNN_HIDDEN_DIM = 256
RNN_FEATURE_DIM = 128
RNN_STATE_DIM = 16
RNN_LATENT_STATE_DIM = 7
RNN_DROPOUT = 0.10
RNN_NUMERIC_DIM = 22

MAX_RNN_CORRECTION_M = 4.5
CORRECTION_GATE_INITIAL_BIAS = -1.0

MAX_MODEL_VELOCITY_M_PER_FRAME = 10.0
MAX_MODEL_ACCELERATION_M_PER_FRAME2 = 6.0
MAX_POLYNOMIAL_STEP_M_PER_FRAME = 10.0

# Teacher requirement:
# p^-_(t+1) = p_final_t + v_t + 0.5*a_t
# The polynomial prediction itself is the forward search.
# There is NO hard 3x6 forward mask, so recovery remains possible.
USE_HARD_FORWARD_MASK = False

# ------------------------------------------------------------------
# Two-stage training
# ------------------------------------------------------------------
WARMUP_EPOCHS = 8
WARMUP_SEARCH_JITTER_M = 6.0

TEMPORAL_EPOCHS = 50
TEMPORAL_LR = 2e-4
TEMPORAL_WEIGHT_DECAY = 1e-3
TBPTT_STEPS = 32
GRAD_CLIP_NORM = 5.0

TEMPORAL_TRAIN_FRACTION = 0.78

# Adaptive autoregressive horizon.
ROLLOUT_HORIZONS = (4, 8, 16, 32, 64, 128, 0)  # 0 = full train split
HORIZON_GOOD_EPOCHS_TO_ADVANCE = 2
HORIZON_TRAIN_CAPTURE_MIN_PCT = 90.0
HORIZON_TRAIN_PRED_ERROR_MAX_M = 7.0
HORIZON_EPISODE_CAPTURE_MIN_PCT = 70.0

# Episode validation selects checkpoint.
VAL_EPISODE_LENGTH = 64
VAL_EPISODE_COUNT = 4

# Full Route-A from W0 is a stress test only.
FULL_ROUTE_STRESS_EVERY = 1

# Early stopping uses episode validation; no need to wait until catastrophic
# full-route drift saturates around hundreds of meters.
EARLY_STOP_PATIENCE = 8
EARLY_STOP_MIN_DELTA_M = 0.15
EARLY_STOP_MIN_STAGE2_EPOCH = 12

# ------------------------------------------------------------------
# Losses
# ------------------------------------------------------------------
LOSS_CANDIDATE = 1.00
LOSS_MEASUREMENT = 0.75
LOSS_NEXT_STEP = 1.50
LOSS_VELOCITY = 0.35
LOSS_ACCELERATION = 0.20
LOSS_VARIANCE_NLL = 0.04
LOSS_CORRECTION_REG = 0.02
LOSS_GATE_REG = 0.01

# ------------------------------------------------------------------
# External Kalman
# ------------------------------------------------------------------
KALMAN_INIT_POSITION_VAR = 4.0
KALMAN_INIT_VELOCITY_VAR = 9.0
KALMAN_Q_POSITION = 0.35
KALMAN_Q_VELOCITY = 0.60
KALMAN_R_MIN_VAR = 0.25
KALMAN_R_MAX_VAR = 25.0

MAX_FINAL_STEP_M_PER_FRAME = 10.0
JUMP_TOLERANCE_M = 3.0

VIDEO_FPS = 12.0
VIDEO_WIDTH = 1800
VIDEO_HEIGHT = 900

SEED = 2031
DEVICE = "cuda"
FEATURE_CACHE_DTYPE = "float16"
