import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================================
# Strict protocol
# ============================================================================
# Visual retrieval heads: Route A only.
# Recurrent + Kalman temporal model: Route A only.
# Final evaluation: Route B + Route C only.
# No IMU, yaw, optical flow, GPS measurement, or other sensor is used by the
# temporal model.  GT is used only for training supervision / evaluation.
# ============================================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rnn_kalman_train_A_test_BC"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "recurrent_kalman_A_only.pt"

DATASETS_ROOT = Path("/yh/study/new_data_2")
ROUTE_ROOTS = [
    DATASETS_ROOT / "model_dataset_new_1_flight",
    DATASETS_ROOT / "model_dataset_new_2_flight",
    Path("/yh/study/new_data/model_dataset_flight"),
]
ROUTE_NAMES = ["route_A", "route_B", "route_C"]
TRAIN_ROUTE_NAMES = ["route_A"]
EVAL_ROUTE_NAMES = ["route_B", "route_C"]

SAT_IMAGE = Path(
    "/yh/study/sim_data/sim_competition_crop_check/"
    "sim_map_competition_roi_crop.png"
)
SAT_JSON = Path(
    "/yh/study/sim_data/sim_competition_crop_check/"
    "sim_map_competition_roi_crop_worldfile_epsg3826.json"
)

# ============================================================================
# Visual retrieval (unchanged)
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

GRID_SIZE = 6
LOCAL_PRIOR_JITTER_M = 12.0
CANDIDATE_CAPTURE_RADIUS_M = 7.5
MIN_TRAIN_CAPTURE_RATE = 0.95

# ============================================================================
# Recurrent Kalman model
# ============================================================================
# Training uses truncated BPTT windows.  Inference does NOT reset every 16
# frames: hidden state + Kalman state are carried continuously through the route.
RNN_SEQUENCE_LENGTH = int(os.environ.get("RNN_SEQUENCE_LENGTH", "16"))
WINDOW_STRIDE = 1
RNN_HIDDEN_DIM = 256
RNN_FEATURE_DIM = 128
RNN_DROPOUT = 0.10

# Explicit physical state:
#   [x, y, vx, vy, ax, ay]
KALMAN_STATE_DIM = 6

# Initial covariance (variances, meter/frame units where applicable).
KALMAN_INIT_POSITION_VAR = 25.0
KALMAN_INIT_VELOCITY_VAR = 16.0
KALMAN_INIT_ACCELERATION_VAR = 4.0

# Positive numerical floor for learned variances.
KALMAN_MIN_VARIANCE = 1e-4

# The measurement-variance head predicts R directly through softplus.
# This upper bound only prevents numerical overflow; it is not a gating rule.
KALMAN_MAX_MEASUREMENT_VAR = 400.0

# ============================================================================
# Temporal training
# ============================================================================
EPOCHS = 50
LR = 2e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP_NORM = 5.0
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 64
SEED = 2027
DEVICE = "cuda"
FEATURE_CACHE_DTYPE = "float16"
EARLY_STOPPING_PATIENCE = 12

# Standard losses only:
# 1) Smooth L1 on final Kalman-filtered position.
# 2) Gaussian NLL on the GRU visual measurement + predicted variance.
# 3) Smooth L1 on the motion-model prediction before Kalman correction.
# 4) Smooth L1 on velocity.
# 5) Smooth L1 on acceleration.
LOSS_FINAL_POSITION = 1.00
LOSS_MEASUREMENT_GAUSSIAN_NLL = 0.20
LOSS_PREDICTION_POSITION = 0.30
LOSS_VELOCITY = 0.15
LOSS_ACCELERATION = 0.05

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_GUARD_FRAMES = 16

JUMP_TOLERANCE_M = 3.0
VAL_RPE_WEIGHT = 0.25
VAL_JUMP_WEIGHT = 0.05
VAL_STATIONARY_WEIGHT = 0.05
