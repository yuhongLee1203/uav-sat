from pathlib import Path
from config_base import *

PROJECT_ROOT = Path(__file__).resolve().parent

ARCHITECTURE_NAME = (
    "V36_byTeacher_MeanShiftEvidence_2Frame_GRUVisualMeasurementVariance_NoSatContext_Polynomial_Kalman"
)
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "controlled_referenceprior_forward3x6_ms_evidence_2frame_gru_visual_measurement_variance_nosat_A_only.pt"
)
LATEST_TEMPORAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "controlled_referenceprior_forward3x6_ms_evidence_2frame_gru_visual_measurement_variance_nosat_A_only_latest.pt"
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
# Current SoftMS position is sent into the GRU as evidence only. The old
#       visual_measurement = SoftMS + correction_head
# path is removed. The historical correction_head now acts as a GRU visual
# measurement-innovation head around the Kalman motion prior:
#       visual_measurement = predicted_se + measurement_head(hidden)
# SoftMS is therefore never algebraically added to the head output.
MEANSHIFT_POSITION_AS_GRU_INPUT = True
DIRECT_SOFTMS_MEASUREMENT = False
USE_GRU_VISUAL_MEASUREMENT_HEAD = True
GRU_VISUAL_MEASUREMENT_PROGRESS_RANGE_M = 12.0
GRU_VISUAL_MEASUREMENT_CROSS_RANGE_M = 8.0

# ---------------------------------------------------------------------------
# Visual uncertainty responsibility
# ---------------------------------------------------------------------------
# SoftMS mode-space spread is retained only as an ambiguity cue inside GRU.
# It is NOT sent to Kalman and is NOT added to variance_head output.
# variance_head alone predicts the final Kalman measurement covariance R_t.
MEANSHIFT_VARIANCE_AS_GRU_INPUT = True
DIRECT_SOFTMS_VARIANCE = False
USE_LEARNED_VARIANCE_HEAD = True
GRU_VISUAL_VARIANCE_INIT_M2 = 25.0

# Kalman must not consume MeanShift/local-posterior confidence. The legacy
# update API still has an acquisition_confidence argument for compatibility,
# but fixing both bounds to 1 makes its value constant and removes any dynamic
# MeanShift-confidence influence on R, innovation gates, or posterior limits.
KALMAN_USE_MS_CONFIDENCE = False
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

# Satellite context is intentionally not part of the main GRU. In this
# controlled single-window protocol, the posterior-weighted SAT embedding is
# strongly scene/route specific and can encourage Route-A memorization. The
# visual retrieval/MeanShift stage still uses SAT embeddings normally.
USE_SATELLITE_CONTEXT_IN_GRU = False

# Numeric GRU state:
# SoftMS mode-space spread(2)
# + current SoftMS - Kalman prior(2)
# + current SoftMS - re-localized previous position(2)
# + previous velocity(2)
# + previous heading residual(1) + turn rate(1) = 10.
RNN_NUMERIC_DIM = 10

# GRU owns visual measurement and uncertainty, so both are supervised.
# Measurement uses the reference position. Variance uses Gaussian NLL from the
# learned measurement residual; SoftMS variance is not part of this output.
LOSS_MEASUREMENT = 1.0
LOSS_VARIANCE_NLL = 0.05

# Compact recurrent model converges quickly; use a conservative learning rate.
TEMPORAL_LR = 1e-4

CONTROLLED_PROTOCOL_NAME = (
    "reference-point+smooth-jitter_forward3x6_SoftMS-as-GRU-evidence_"
    "2frame_GRU-visual-measurement-and-variance_no-sat-context_"
    "polynomial_Kalman_no-MS-confidence_with-next-frame-MS-feedback"
)

# Waypoints stay in the repository-level route_waypoints folder.
WAYPOINT_DIR = PROJECT_ROOT.parent / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}
