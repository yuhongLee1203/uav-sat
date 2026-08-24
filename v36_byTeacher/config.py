from pathlib import Path
from config_base import *

PROJECT_ROOT = Path(__file__).resolve().parent

ARCHITECTURE_NAME = (
    "V36_byTeacher_MSPreviousPosition_2Frame_GRUVisualMeasurementVariance_"
    "NoSatContext_Polynomial_Kalman_v2"
)
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "controlled_referenceprior_forward3x6_ms_previous_position_2frame_gru_visual_measurement_variance_nosat_A_only.pt"
)
LATEST_TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "controlled_referenceprior_forward3x6_ms_previous_position_2frame_gru_visual_measurement_variance_nosat_A_only_latest.pt"
)
FEATURE_CACHE_DIR = OUTPUT_DIR / "feature_cache"

# ---------------------------------------------------------------------------
# Teacher-requested inter-frame hand-off
# ---------------------------------------------------------------------------
# For frame t, the newly arrived UAV image re-localizes the previous Kalman
# output X_(t-1) by MeanShift.  X_(t-1)^MS is the GRU's previous-position cue.
# It never overwrites kf.x or kf.P.
TEACHER_MEANSHIFT_FEEDBACK = True
TEACHER_FEEDBACK_PRESERVE_KALMAN_VELOCITY = True
TEACHER_FEEDBACK_USE_FORWARD_3X6 = True

# ---------------------------------------------------------------------------
# GRU visual measurement
# ---------------------------------------------------------------------------
# IMPORTANT: the old equation
#       z_t = current_SoftMS + correction_head
# is NOT used.
# The current GRU visual measurement is instead predicted forward from the
# re-localized previous position:
#       z_t = X_(t-1)^MS + measurement_step_head(h_t)
# Therefore MeanShift supplies the previous-position reference and the GRU
# learns the current visual displacement.  This also keeps the Kalman prior
# independent from the GRU visual-measurement base.
MEANSHIFT_POSITION_AS_GRU_INPUT = True
DIRECT_SOFTMS_MEASUREMENT = False
USE_GRU_VISUAL_MEASUREMENT_HEAD = True
GRU_VISUAL_MEASUREMENT_PROGRESS_RANGE_M = 12.0
GRU_VISUAL_MEASUREMENT_CROSS_RANGE_M = 8.0
GRU_VISUAL_MEASUREMENT_INIT_PROGRESS_M = 3.0

# ---------------------------------------------------------------------------
# Visual uncertainty
# ---------------------------------------------------------------------------
# SoftMS mode-space spread is only an internal ambiguity cue for the GRU.  It is
# not sent to Kalman and is not algebraically added to variance_head output.
# variance_head alone produces the measurement variance R_t used by Kalman.
MEANSHIFT_VARIANCE_AS_GRU_INPUT = True
DIRECT_SOFTMS_VARIANCE = False
USE_LEARNED_VARIANCE_HEAD = True
GRU_VISUAL_VARIANCE_INIT_M2 = 9.0

# Kalman must not consume MeanShift/local-posterior confidence.  The compatibility
# confidence argument is fixed to one by robust_tracker.py.  Disable NIS-based R
# inflation as well so learned variance remains the effective measurement R_t.
KALMAN_USE_MS_CONFIDENCE = False
VISUAL_CONFIDENCE_FLOOR = 1.0
VISUAL_CONFIDENCE_CEIL = 1.0
KALMAN_NIS_CONFIDENCE_BOOST = 0.0
KALMAN_NIS_MAX_R_SCALE = 1.0
ACQ_LOW_CONF_VARIANCE_GAIN = 0.0

# Give the learned GRU measurement enough room to disagree with the motion prior
# before the normal bounded posterior correction is applied.
KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M = 12.0
KALMAN_MAX_MEASUREMENT_INNOVATION_CROSS_M = 8.0
KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M = 4.0
KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M = 2.5

# ---------------------------------------------------------------------------
# Compact temporal state
# ---------------------------------------------------------------------------
# Current + previous UAV embeddings provide temporal mean and first difference.
# The second embedding difference is removed; physical acceleration is predicted
# explicitly by motion_head and learned through the polynomial next-step target.
EXPERIMENT_FRAME_COUNT = 2

# Satellite context is intentionally excluded from the main GRU.  Satellite
# embeddings are still used normally by UAV/SAT retrieval and MeanShift.
USE_SATELLITE_CONTEXT_IN_GRU = False

# Numeric GRU state:
# SoftMS mode-space spread(2)
# + current SoftMS - Kalman prior(2)
# + current SoftMS - re-localized previous position(2)
# + previous velocity(2)
# + previous heading residual(1) + turn rate(1) = 10.
RNN_NUMERIC_DIM = 10

# The measurement head now has an independent job and needs a stronger direct
# position objective.  Variance remains supervised by Gaussian NLL.
LOSS_MEASUREMENT = 2.0
LOSS_VARIANCE_NLL = 0.05
TEMPORAL_LR = 1e-4

# ---------------------------------------------------------------------------
# Forward-search accuracy/speed ablation
# ---------------------------------------------------------------------------
# Thesis/default setting remains 3x6=18. During evaluation the same trained
# temporal checkpoint can be tested with 4x6=24, 5x6=30 and 6x6=36 so only the
# inference candidate count changes.
FORWARD_SEARCH_ROWS = 3
FORWARD_SEARCH_COLS = 6
FORWARD_SEARCH_CANDIDATE_COUNT = FORWARD_SEARCH_ROWS * FORWARD_SEARCH_COLS
FORWARD_SEARCH_EXPERIMENT_ROWS = (3, 4, 5, 6)
LATENCY_WARMUP_FRAMES = 30

CONTROLLED_PROTOCOL_NAME = (
    "reference-point+smooth-jitter_forward3x6_MS-previous-position-to-GRU_"
    "2frame_GRU-visual-measurement-and-variance_no-sat-context_"
    "polynomial_Kalman_no-MS-confidence"
)

# Waypoints stay in the repository-level route_waypoints folder.
WAYPOINT_DIR = PROJECT_ROOT.parent / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}
