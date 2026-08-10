from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# =============================================================================
# Experiment
# =============================================================================
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "route_bounded_hypothesis_lstm_v6"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "route_hypothesis_lstm_A_only.pt"

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
LOCAL_PRIOR_JITTER_M = 12.0
CANDIDATE_CAPTURE_RADIUS_M = 7.5
MIN_TRAIN_CAPTURE_RATE = 0.95

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_GUARD_FRAMES = 16

# =============================================================================
# v6: Route-bounded hypothesis recurrent localization
# =============================================================================
#
# There is NO:
#   fixed speed
#   nominal speed
#   translation probability gate
#   free-running velocity head
#   constant-velocity Kalman
#
# Each frame has three hypotheses:
#   0 HOLD      = previous image-derived position
#   1 LOCAL     = current UAV image matched in a local 6x6 SAT lattice
#   2 RECOVERY  = current UAV image globally retrieves inside the CURRENT
#                 Start->End route corridor, then refines with a 6x6 lattice
#   3 WAYPOINT   = current UAV image matched in a small 6x6 transition
#                 neighborhood centered on the active endpoint
#
# LSTM selects/fuses hypotheses from current images + previous recurrent state.
# WAYPOINT being the strongest branch is also the learned leg-transition event.
# =============================================================================
HYPOTHESIS_HOLD = 0
HYPOTHESIS_LOCAL = 1
HYPOTHESIS_RECOVERY = 2
HYPOTHESIS_WAYPOINT = 3
HYPOTHESIS_COUNT = 4

LSTM_HIDDEN_DIM = 256
LSTM_FEATURE_DIM = 128
LSTM_DROPOUT = 0.10

BRANCH_SOFTMAX_TEMPERATURE = 0.30
BRANCH_TARGET_TAU_M = 5.0

# Route corridor: candidates can never run arbitrarily past the active endpoint.
ROUTE_CORRIDOR_HALF_WIDTH_M = 36.0
ROUTE_ALONG_PADDING_M = 12.0

# Global route retrieval diagnostics / features.
GLOBAL_STATS_TOPK = 8

# Previous observed motion is measured only from previous image-derived outputs.
OBSERVED_MOTION_DIM = 4
OBSERVED_MOTION_SCALE_M = 8.0

# Advisor-requested second-order inertial polynomial:
# delta_poly = previous_observed_delta + 0.5 * observed_acceleration
# It is INPUT CONTEXT ONLY. It never moves XY and has no additive score prior.
POLYNOMIAL_SCALE_M = 10.0

# Route-relative numeric context.
ROUTE_CROSS_TRACK_SCALE_M = 36.0
ROUTE_LENGTH_LOG_SCALE_M = 1000.0
CANDIDATE_OFFSET_SCALE_M = 24.0

# =============================================================================
# Temporal training
# =============================================================================
# All Route-A frames are used for temporal training.
# B/C are never used for training, validation, or checkpoint selection.
TEMPORAL_EPOCHS = 40
TEMPORAL_LR = 2e-4
TEMPORAL_WEIGHT_DECAY = 1e-3
TBPTT_STEPS = 32
GRAD_CLIP_NORM = 5.0

# Causal scheduled local-center guidance only.
# Current GT is never an input.
# After this epoch local search is fully model-centered.
TEACHER_CENTER_END_EPOCH = 6

LOSS_POSITION = 1.00
LOSS_BRANCH_DISTRIBUTION = 0.40
LOSS_LOCAL_CANDIDATE_CE = 0.20
LOSS_RECOVERY_CANDIDATE_CE = 0.25
LOSS_WAYPOINT_CANDIDATE_CE = 0.20
LOSS_STEP = 0.20

# =============================================================================
# Inference waypoint switching
# =============================================================================
# NO test waypoint frame_index is used.
# There is no arrival-distance threshold and no timer.
# A leg changes when the recurrent model's strongest hypothesis is WAYPOINT.

# Final mission state is frozen after the learned arrival detector says the
# final endpoint has been reached.
TERMINAL_LOCK_ENABLED = True

# =============================================================================
# Position-only FilterPy smoother
# =============================================================================
# state = [x, y], F = identity. No vx/vy.
KALMAN_INIT_POSITION_VAR = 9.0
KALMAN_Q_POSITION = 0.25
KALMAN_R_MIN_VAR = 1.0
KALMAN_R_MAX_VAR = 25.0

# =============================================================================
# Evaluation / rendering
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
