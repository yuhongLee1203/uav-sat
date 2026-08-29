import os
from pathlib import Path
from config_base import *

PROJECT_ROOT = Path(__file__).resolve().parent

# =============================================================================
# v36_byTeacher compact motion-GRU architecture, v8
# =============================================================================
# The train/eval entrypoint may install the controlled reference-assisted MS1
# coarse prior, but the GRU itself never receives a reference coordinate.
#
# GRU input groups:
#   1) MS1 visual localization XY             2 -> 128
#   2) temporal visual mean                  512 -> 128
#   3) first visual difference               512 -> 128
#   4) previous motion [v,a,sin(theta),cos]   4 -> 128
# Concatenate: 4 x 128 = 512 input dimensions.
# Previous hidden state is supplied separately to GRUCell as a 256-d state.
ARCHITECTURE_NAME = (
    "V36_byTeacher_CompactMotionGRU_MSXY_TemporalMean_FirstDiff_"
    "PrevMotionInfo_H256_v8_nativeA"
)

BACKBONE_KEY = os.environ.get(
    "UAVSAT_BACKBONE", "mobilenet_v3_small"
).strip().lower()
if BACKBONE_KEY not in BACKBONE_SPECS:
    raise ValueError(
        "UAVSAT_BACKBONE must be one of %s"
        % sorted(BACKBONE_SPECS)
    )
BACKBONE_NAME, CLIP_DIM = BACKBONE_SPECS[
    BACKBONE_KEY
]

BACKBONE_OUTPUT_DIR = (
    PROJECT_ROOT / "output" / BACKBONE_KEY
)
RUN_TAG = os.environ.get(
    "UAVSAT_RUN_TAG", ""
).strip()
OUTPUT_DIR = (
    BACKBONE_OUTPUT_DIR / "experiments" / RUN_TAG
    if RUN_TAG
    else BACKBONE_OUTPUT_DIR
    / "compact_motion_gru_v8_nativeA"
)
CHECKPOINT_DIR = (
    BACKBONE_OUTPUT_DIR / "checkpoints"
)
VISUAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / f"visual_retrieval_A_only_{BACKBONE_KEY}.pt"
)
TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / f"compact_motion_gru_A_native_v8_{BACKBONE_KEY}.pt"
)
LATEST_TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / f"compact_motion_gru_A_native_v8_{BACKBONE_KEY}_latest.pt"
)
FEATURE_CACHE_DIR = (
    BACKBONE_OUTPUT_DIR / "feature_cache"
)

WAYPOINT_DIR = (
    PROJECT_ROOT.parent / "route_waypoints"
)
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}

# -----------------------------------------------------------------------------
# Visual retrieval
# -----------------------------------------------------------------------------
MEANSHIFT_ITERATIONS = int(
    os.environ.get(
        "UAVSAT_MS_ITERATIONS",
        str(MEANSHIFT_ITERATIONS),
    )
)
MEANSHIFT_BANDWIDTH_M = float(
    os.environ.get(
        "UAVSAT_MS_BANDWIDTH_M",
        str(MEANSHIFT_BANDWIDTH_M),
    )
)
MEANSHIFT_SCORE_TAU = float(
    os.environ.get(
        "UAVSAT_MS_SCORE_TAU",
        str(MEANSHIFT_SCORE_TAU),
    )
)
MEANSHIFT_MODE_BETA = float(
    os.environ.get(
        "UAVSAT_MS_MODE_BETA",
        str(MEANSHIFT_MODE_BETA),
    )
)

MS1_BASE_GRID_SIZE = 6
MS1_FORWARD_ROWS = 3
MS1_FORWARD_COLS = 6
MS1_CANDIDATE_COUNT = (
    MS1_FORWARD_ROWS * MS1_FORWARD_COLS
)
MS2_GRID_SIZE = 6

LOCAL_PRIOR_JITTER_M = float(
    os.environ.get(
        "UAVSAT_VISUAL_TRAIN_JITTER_M",
        "12.0",
    )
)

# -----------------------------------------------------------------------------
# Compact motion GRU
# -----------------------------------------------------------------------------
RNN_HIDDEN_DIM = int(
    os.environ.get(
        "UAVSAT_RNN_HIDDEN", "256"
    )
)
RNN_FEATURE_DIM = int(
    os.environ.get(
        "UAVSAT_RNN_FEATURE", "128"
    )
)
RNN_DROPOUT = float(
    os.environ.get(
        "UAVSAT_RNN_DROPOUT", "0.0"
    )
)

RNN_MS_XY_DIM = 2
RNN_TEMPORAL_MEAN_DIM = EMBED_DIM
RNN_FIRST_DIFFERENCE_DIM = EMBED_DIM
RNN_PREVIOUS_MOTION_DIM = 4
RNN_PROJECTED_GROUPS = 4
RNN_COMBINED_INPUT_DIM = RNN_FEATURE_DIM * RNN_PROJECTED_GROUPS

POSITION_INPUT_SCALE_M = float(
    os.environ.get(
        "UAVSAT_POSITION_INPUT_SCALE_M",
        "50.0",
    )
)
STEP_INPUT_SCALE_M = float(
    os.environ.get(
        "UAVSAT_STEP_INPUT_SCALE_M",
        "10.0",
    )
)

TEMPORAL_EPOCHS = int(
    os.environ.get(
        "UAVSAT_TEMPORAL_EPOCHS", "60"
    )
)
TEMPORAL_LR = float(
    os.environ.get(
        "UAVSAT_TEMPORAL_LR", "3e-4"
    )
)
TEMPORAL_WEIGHT_DECAY = float(
    os.environ.get(
        "UAVSAT_TEMPORAL_WEIGHT_DECAY",
        "1e-4",
    )
)
TBPTT_STEPS = int(
    os.environ.get(
        "UAVSAT_TBPTT_STEPS", "32"
    )
)
GRAD_CLIP_NORM = float(
    os.environ.get(
        "UAVSAT_GRAD_CLIP_NORM", "5.0"
    )
)
EARLY_STOP_PATIENCE = int(
    os.environ.get(
        "UAVSAT_EARLY_STOP_PATIENCE",
        "12",
    )
)
EARLY_STOP_MIN_EPOCH = int(
    os.environ.get(
        "UAVSAT_EARLY_STOP_MIN_EPOCH",
        "15",
    )
)
EARLY_STOP_MIN_DELTA = float(
    os.environ.get(
        "UAVSAT_EARLY_STOP_MIN_DELTA",
        "0.01",
    )
)

LOSS_DELTA = float(
    os.environ.get(
        "UAVSAT_LOSS_DELTA", "5.0"
    )
)
LOSS_SPEED = float(
    os.environ.get(
        "UAVSAT_LOSS_SPEED", "1.0"
    )
)
LOSS_ACCELERATION = float(
    os.environ.get(
        "UAVSAT_LOSS_ACCELERATION",
        "0.25",
    )
)
LOSS_HEADING = float(
    os.environ.get(
        "UAVSAT_LOSS_HEADING", "2.0"
    )
)
LOSS_HEADING_DELTA = float(
    os.environ.get(
        "UAVSAT_LOSS_HEADING_DELTA",
        "0.5",
    )
)

# -----------------------------------------------------------------------------
# Standard XY Kalman filter
# -----------------------------------------------------------------------------
KALMAN_INIT_POSITION_VAR = float(
    os.environ.get(
        "UAVSAT_KF_INIT_POS_VAR", "25.0"
    )
)
KALMAN_INIT_VELOCITY_VAR = float(
    os.environ.get(
        "UAVSAT_KF_INIT_VEL_VAR", "25.0"
    )
)
KALMAN_Q_POSITION = float(
    os.environ.get(
        "UAVSAT_KF_Q_POS", "1.0"
    )
)
KALMAN_Q_VELOCITY = float(
    os.environ.get(
        "UAVSAT_KF_Q_VEL", "1.0"
    )
)
KALMAN_R_POSITION = float(
    os.environ.get(
        "UAVSAT_KF_R_POS", "9.0"
    )
)

# Disable inherited legacy controlled-protocol switches. The controlled
# reference-assisted MS1 prior, when used, is installed explicitly by
# train_multirate_a.py and is kept separate from the estimator itself.
CONTROLLED_GT_PRIOR = False
CONTROLLED_GT_PRIOR_JITTER_M = 0.0
CONTROLLED_FINAL_PROGRESS_CAP_TO_GT = False
CONTROLLED_GT_MOTION_ENVELOPE = False
CONTROLLED_PACE_ASSIST = False

TEMPORAL_TRAIN_ROUTES = ["route_A"]
TEMPORAL_VALIDATION_ROUTE = "route_C"
TEMPORAL_TEST_ROUTE = "route_B"

LATENCY_WARMUP_FRAMES = int(
    os.environ.get(
        "UAVSAT_LATENCY_WARMUP_FRAMES",
        "30",
    )
)
