from pathlib import Path
from config_base import *

PROJECT_ROOT = Path(__file__).resolve().parent
ARCHITECTURE_NAME = "V36_byTeacher_MeanShiftFeedback_GRU_Polynomial_Kalman"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "controlled_referenceprior_forward3x6_teacher_feedback_state_gru_A_only.pt"
LATEST_TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "controlled_referenceprior_forward3x6_teacher_feedback_state_gru_A_only_latest.pt"
FEATURE_CACHE_DIR = OUTPUT_DIR / "feature_cache"

# v36_byTeacher keeps the same route reference-point protocol, Forward-3x6,
# Soft Mean-Shift, GRU polynomial and external Kalman settings as v36.
# The only architectural change is the inter-frame GRU state hand-off:
#   X_t(output) = Kalman(GRU(MeanShift(...)))
#   X_t^MS      = MeanShift(current next-frame image around X_t(output))
#   X_{t+1}     = Kalman(GRU(previous_position=X_t^MS))
# X_t(output) remains both the reported position and the external Kalman's
# posterior state. X_t^MS replaces only the GRU previous-position input; it
# must not overwrite kf.x or kf.P before the next Kalman prediction.
TEACHER_MEANSHIFT_FEEDBACK = True
TEACHER_FEEDBACK_PRESERVE_KALMAN_VELOCITY = True
TEACHER_FEEDBACK_USE_FORWARD_3X6 = True

CONTROLLED_PROTOCOL_NAME = (
    "reference-point+smooth-jitter_forward3x6_SoftMS_3frame_causal_heading_"
    "GRU_Polynomial_Kalman_with_next-frame_MeanShift_previous-position_feedback"
)

# Waypoints stay in the repository-level route_waypoints folder.
WAYPOINT_DIR = PROJECT_ROOT.parent / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}
