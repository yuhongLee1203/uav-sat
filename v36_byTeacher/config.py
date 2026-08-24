from pathlib import Path
from config_base import *

PROJECT_ROOT = Path(__file__).resolve().parent

ARCHITECTURE_NAME = (
    "V36_byTeacher_MeanShiftEvidence_2Frame_GRUVisualMeasurementVariance_Polynomial_Kalman"
)
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "controlled_referenceprior_forward3x6_ms_evidence_2frame_gru_visual_measurement_variance_A_only.pt"
)
LATEST_TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "controlled_referenceprior_forward3x6_ms_evidence_2frame_gru_visual_measurement_variance_A_only_latest.pt"
)
FEATURE_CACHE_DIR = OUTPUT_DIR / "feature_cache"

# ---------------------------------------------------------------------------
# Teacher-requested inter-frame hand-off
# ---------------------------------------------------------------------------
# X_t remains the external Kalman posterior. When the next UAV image arrives,
# MeanShift around X_t is computed again and that result becomes the GRU's
# previous-localization evidence. It never overwrites kf.x or kf.P.
TEACHER_MEANSHIFT_FEEDBACK = True
TEACHER_FEEDBACK_PRESERVE_KALMAN_VELOCITY = True
TEACHER_FEEDBACK_USE_FORWARD_3X6 = True

# ---------------------------------------------------------------------------
# Visual measurement responsibility
# ---------------------------------------------------------------------------
# Current SoftMS position MUST be sent into the GRU, but it is evidence only.
# The old equation
#       visual_measurement = SoftMS + correction_head
# is disabled.
# The historical correction_head is now interpreted as a learned visual
# measurement-innovation head around the Kalman motion prior:
#       visual_measurement = predicted_se + measurement_head(hidden)
# Thus MeanShift is never algebraically added to the head output.
MEANSHIFT_POSITION_AS_GRU_INPUT = True
DIRECT_SOFTMS_MEASUREMENT = False
USE_GRU_VISUAL_MEASUREMENT_HEAD = True
GRU_VISUAL_MEASUREMENT_PROGRESS_RANGE_M = 12.0
GRU_VISUAL_MEASUREMENT_CROSS_RANGE_M = 8.0

# ---------------------------------------------------------------------------
# Visual uncertainty responsibility
# ---------------------------------------------------------------------------
# SoftMS mode-space spread remains a useful cue for ambiguity, so it is retained
# as a GRU input feature. It is NOT added to the final variance.
# variance_head alone predicts the Kalman measurement covariance R_t.
MEANSHIFT_VARIANCE_AS_GRU_INPUT = True
DIRECT_SOFTMS_VARIANCE = False
USE_LEARNED_VARIANCE_HEAD = True
GRU_VISUAL_VARIANCE_INIT_M2 = 25.0

# Since variance_head is the final visual uncertainty, do not let the old
# SoftMS-derived visual-confidence path rescale R_t a second time inside Kalman.
VISUAL_CONFIDENCE_FLOOR = 1.0
VISUAL_CONFIDENCE_CEIL = 1.0
ACQ_LOW_CONF_VARIANCE_GAIN = 0.0

# ---------------------------------------------------------------------------
# Compact temporal state
# ---------------------------------------------------------------------------
# Use only current + previous UAV embeddings. Their mean and first embedding
# difference are explicit inputs. The old second-order embedding difference is
# removed; physical acceleration is still predicted by motion_head and trained
# through the polynomial next-step objective.
EXPERIMENT_FRAME_COUNT = 2

# Numeric GRU state:
# SoftMS mode-space spread(2)
# + current SoftMS - Kalman prior(2)
# + current SoftMS - re-localized previous position(2)
# + previous velocity(2)
# + previous heading residual(1) + turn rate(1) = 10.
RNN_NUMERIC_DIM = 10

# The GRU now owns visual measurement and uncertainty, therefore both must be
# supervised. Measurement uses the reference position; variance uses Gaussian
# NLL from the measurement residual.
LOSS_MEASUREMENT = 1.0
LOSS_VARIANCE_NLL = 0.05

# Keep the existing motion/heading supervision. The compact model converges
# quickly, so use the more conservative temporal learning rate.
TEMPORAL_LR = 1e-4

CONTROLLED_PROTOCOL_NAME = (
    "reference-point+smooth-jitter_forward3x6_SoftMS-as-GRU-evidence_"
    "2frame_GRU-visual-measurement-and-variance_polynomial_Kalman_"
    "with_next-frame_MeanShift_previous-position-feedback"
)

# Waypoints stay in the repository-level route_waypoints folder.
WAYPOINT_DIR = PROJECT_ROOT.parent / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}
