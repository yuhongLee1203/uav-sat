from pathlib import Path
from config_base import *

PROJECT_ROOT = Path(__file__).resolve().parent

ARCHITECTURE_NAME = (
    "V36_byTeacher_DirectSoftMS_2Frame_GRU_Polynomial_Kalman"
)
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "controlled_referenceprior_forward3x6_direct_softms_2frame_teacher_feedback_gru_A_only.pt"
)
LATEST_TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "controlled_referenceprior_forward3x6_direct_softms_2frame_teacher_feedback_gru_A_only_latest.pt"
)
FEATURE_CACHE_DIR = OUTPUT_DIR / "feature_cache"

# ---------------------------------------------------------------------------
# Teacher-requested inter-frame hand-off
# ---------------------------------------------------------------------------
# X_t remains the external Kalman posterior. When the next UAV image arrives,
# MeanShift around X_t is computed again and that result is supplied to the GRU
# as the previous localization input. It never overwrites kf.x or kf.P.
TEACHER_MEANSHIFT_FEEDBACK = True
TEACHER_FEEDBACK_PRESERVE_KALMAN_VELOCITY = True
TEACHER_FEEDBACK_USE_FORWARD_3X6 = True

# ---------------------------------------------------------------------------
# Clean visual-measurement responsibility
# ---------------------------------------------------------------------------
# SoftMS anchor is the visual measurement z_t directly.
# No GRU correction head is applied to the MeanShift result.
DIRECT_SOFTMS_MEASUREMENT = True
USE_LEARNED_MEASUREMENT_CORRECTION = False

# SoftMS mode-space response variance is the Kalman measurement covariance R_t
# directly. The old variance head is intentionally removed so uncertainty is
# not estimated twice from the same visual response.
DIRECT_SOFTMS_VARIANCE = True
USE_LEARNED_VARIANCE_HEAD = False

# ---------------------------------------------------------------------------
# Compact temporal state
# ---------------------------------------------------------------------------
# Two UAV embeddings are sufficient for the explicit current visual change:
# mean(z_{t-1}, z_t) and z_t-z_{t-1}. The old second-order embedding difference
# is removed; physical acceleration is still predicted by the motion head and
# trained end-to-end through the polynomial next-step objective.
EXPERIMENT_FRAME_COUNT = 2

# Numeric GRU state:
# response variance(2) + current visual innovation(2)
# + current SoftMS - teacher-feedback previous localization(2)
# + previous velocity(2) + heading residual(1) + turn rate(1) = 10.
RNN_NUMERIC_DIM = 10

# The temporal network no longer predicts visual position correction or visual
# variance. Those objectives therefore must not contribute to temporal loss.
LOSS_MEASUREMENT = 0.0
LOSS_VARIANCE_NLL = 0.0

# Keep the existing motion/heading supervision. A slightly smaller temporal LR
# is used because this compact recurrent model converges quickly.
TEMPORAL_LR = 1e-4

CONTROLLED_PROTOCOL_NAME = (
    "reference-point+smooth-jitter_forward3x6_direct-SoftMS-measurement_"
    "direct-SoftMS-variance_2frame_GRU_polynomial_Kalman_"
    "with_next-frame_MeanShift_previous-position_feedback"
)

# Waypoints stay in the repository-level route_waypoints folder.
WAYPOINT_DIR = PROJECT_ROOT.parent / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}
