from pathlib import Path
from config_base import *

PROJECT_ROOT = Path(__file__).resolve().parent

ARCHITECTURE_NAME = (
    "V36_byTeacher_MSPreviousPosition_2Frame_GRUVisualMeasurementVariance_"
    "NoSatContext_Polynomial_Kalman_v3"
)
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "controlled_referenceprior_forward3x6_ms_previous_position_2frame_gru_visual_measurement_variance_nosat_v3_A_only.pt"
)
LATEST_TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "controlled_referenceprior_forward3x6_ms_previous_position_2frame_gru_visual_measurement_variance_nosat_v3_A_only_latest.pt"
)
FEATURE_CACHE_DIR = OUTPUT_DIR / "feature_cache"

# ---------------------------------------------------------------------------
# Teacher-requested inter-frame hand-off
# ---------------------------------------------------------------------------
# For frame t, the newly arrived UAV image re-localizes the previous Kalman
# output X_(t-1) by MeanShift. X_(t-1)^MS is only the GRU previous-position cue;
# it never overwrites the external Kalman posterior/covariance.
TEACHER_MEANSHIFT_FEEDBACK = True
TEACHER_FEEDBACK_PRESERVE_KALMAN_VELOCITY = True
TEACHER_FEEDBACK_USE_FORWARD_3X6 = True

# ---------------------------------------------------------------------------
# GRU visual measurement
# ---------------------------------------------------------------------------
# No current-SoftMS + correction path and no Kalman-prior + correction path.
# The learned current measurement is predicted from the teacher-requested
# re-localized previous position:
#       z_t = X_(t-1)^MS + measurement_step_head(h_t)
MEANSHIFT_POSITION_AS_GRU_INPUT = True
DIRECT_SOFTMS_MEASUREMENT = False
USE_GRU_VISUAL_MEASUREMENT_HEAD = True
GRU_VISUAL_MEASUREMENT_PROGRESS_RANGE_M = 16.0
GRU_VISUAL_MEASUREMENT_CROSS_RANGE_M = 10.0
GRU_VISUAL_MEASUREMENT_INIT_PROGRESS_M = 3.0

# ---------------------------------------------------------------------------
# Visual uncertainty
# ---------------------------------------------------------------------------
# SoftMS spread is only an internal ambiguity cue. variance_head alone outputs
# the R_t used by Kalman. A lower initial/minimum variance lets a well-supervised
# GRU measurement actually correct the motion prior instead of being ignored.
MEANSHIFT_VARIANCE_AS_GRU_INPUT = True
DIRECT_SOFTMS_VARIANCE = False
USE_LEARNED_VARIANCE_HEAD = True
GRU_VISUAL_VARIANCE_INIT_M2 = 4.0
KALMAN_R_MIN_VAR = 1.0
KALMAN_R_MAX_VAR = 400.0

# MeanShift/local-posterior confidence must not independently alter Kalman R.
KALMAN_USE_MS_CONFIDENCE = False
VISUAL_CONFIDENCE_FLOOR = 1.0
VISUAL_CONFIDENCE_CEIL = 1.0
KALMAN_NIS_CONFIDENCE_BOOST = 0.0
KALMAN_NIS_MAX_R_SCALE = 1.0
ACQ_LOW_CONF_VARIANCE_GAIN = 0.0

# ---------------------------------------------------------------------------
# Kalman trust balance
# ---------------------------------------------------------------------------
# v2 was too conservative: the learned measurement was supervised toward the
# reference position but the posterior was allowed to move only a few metres.
# Increase process uncertainty and correction room while retaining bounded,
# non-teleporting updates.
KALMAN_Q_PROGRESS = 0.80
KALMAN_Q_CROSS = 0.25
KALMAN_Q_VELOCITY = 0.40
KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M = 16.0
KALMAN_MAX_MEASUREMENT_INNOVATION_CROSS_M = 10.0
KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M = 7.0
KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M = 4.0
KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME = 2.0
KALMAN_FINAL_STEP_SLACK_M = 1.0
KALMAN_FINAL_STEP_MAX_M = 8.0

# ---------------------------------------------------------------------------
# Compact temporal state
# ---------------------------------------------------------------------------
EXPERIMENT_FRAME_COUNT = 2
USE_SATELLITE_CONTEXT_IN_GRU = False

# SoftMS spread(2) + current SoftMS-prior innovation(2)
# + current SoftMS-X_(t-1)^MS displacement(2) + previous velocity(2)
# + heading residual/turn-rate(2).
RNN_NUMERIC_DIM = 10

# ---------------------------------------------------------------------------
# Training balance
# ---------------------------------------------------------------------------
# Reference position is supervision, not the answer passed to the estimator.
# Give the learned visual measurement the strongest direct objective; retain
# motion/heading supervision but reduce competition from the polynomial loss.
LOSS_MEASUREMENT = 4.0
LOSS_NEXT_STEP = 2.0
LOSS_VARIANCE_NLL = 0.02
TEMPORAL_LR = 5e-5
RNN_DROPOUT = 0.05
TBPTT_STEPS = 64
GRAD_CLIP_NORM = 3.0
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.02
EARLY_STOP_MIN_EPOCH = 25

# ---------------------------------------------------------------------------
# Forward-search accuracy/speed ablation
# ---------------------------------------------------------------------------
# Thesis/default setting remains 3x6=18. The same trained checkpoint can be
# evaluated with 4x6=24, 5x6=30 and 6x6=36.
FORWARD_SEARCH_ROWS = 3
FORWARD_SEARCH_COLS = 6
FORWARD_SEARCH_CANDIDATE_COUNT = FORWARD_SEARCH_ROWS * FORWARD_SEARCH_COLS
FORWARD_SEARCH_EXPERIMENT_ROWS = (3, 4, 5, 6)
LATENCY_WARMUP_FRAMES = 30

CONTROLLED_PROTOCOL_NAME = (
    "reference-point+smooth-jitter_forward3x6_MS-previous-position-to-GRU_"
    "2frame_GRU-visual-measurement-and-variance_no-sat-context_"
    "measurement-trusting-polynomial-Kalman_v3"
)

WAYPOINT_DIR = PROJECT_ROOT.parent / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}
