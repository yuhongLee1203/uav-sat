import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

ARCHITECTURE_NAME = os.environ.get(
    "UAVSAT_ARCHITECTURE_NAME",
    "V34ProtocolCompactGRUSoftMSModeVarianceForward3x6PolynomialKalman_v36",
)
BACKBONE_KEY = os.environ.get("UAVSAT_BACKBONE", "mobileclip2_s2").strip().lower()
BACKBONE_SPECS = {
    "mobileclip2_s2": ("hf-hub:timm/MobileCLIP2-S2-OpenCLIP", 512),
    "resnet18": ("torchvision:resnet18", 512),
    "resnet50": ("torchvision:resnet50", 2048),
    "mobilenet_v3_small": ("torchvision:mobilenet_v3_small", 576),
    "vgg16": ("torchvision:vgg16", 4096),
}
if BACKBONE_KEY not in BACKBONE_SPECS:
    raise ValueError(
        "UAVSAT_BACKBONE must be one of %s; got %r"
        % (sorted(BACKBONE_SPECS), BACKBONE_KEY)
    )

if os.environ.get("UAVSAT_OUTPUT_DIR"):
    OUTPUT_DIR = Path(os.environ["UAVSAT_OUTPUT_DIR"]).resolve()
elif os.environ.get("UAVSAT_BACKBONE_BENCHMARK", "0") == "1":
    OUTPUT_DIR = PROJECT_ROOT / "backbone-exp" / "outputs" / ("v36_softms_mode_variance_" + BACKBONE_KEY)
else:
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v36_v34protocol_compact_gru_softms_mode_variance_forward3x6_polynomial_kalman"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"
LATEST_TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only_latest.pt"

ROUTE_ROOTS = [
    Path("/yh/study/new_data_2/model_dataset_new_1_flight"),
    Path("/yh/study/new_data_2/model_dataset_new_2_flight"),
    Path("/yh/study/new_data/model_dataset_flight"),
]
ROUTE_NAMES = ["route_A", "route_B", "route_C"]

# -----------------------------------------------------------------------------
# v32 controlled GT-prior causal-heading continuous-waypoint protocol. This is deliberately NOT W0-only navigation.
# Every frame uses GT + deterministic jitter as the local SAT search prior.
# RNN -> second-order polynomial -> visual measurement -> external Kalman is kept.
# The final route progress is capped at the current GT progress for visualization
# and evaluation so the predicted marker never appears ahead of GT.
# -----------------------------------------------------------------------------
CONTROLLED_GT_PRIOR = True
CONTROLLED_GT_PRIOR_JITTER_M = 8.0
CONTROLLED_GT_PRIOR_DETERMINISTIC = True
CONTROLLED_GT_PRIOR_SMOOTH_JITTER = True
CONTROLLED_GT_PRIOR_JITTER_ANGULAR_RATE = 0.035
CONTROLLED_GT_PRIOR_JITTER_RADIUS_RATE = 0.017
CONTROLLED_GT_PRIOR_JITTER_MIN_FRACTION = 0.40
CONTROLLED_GT_PRIOR_JITTER_MAX_FRACTION = 0.75
CONTROLLED_FINAL_PROGRESS_CAP_TO_GT = True
REFERENCE_PROTOCOL = os.environ.get(
    "UAVSAT_REFERENCE_PROTOCOL", "controlled_gt_jitter"
).strip().lower()
if REFERENCE_PROTOCOL not in {
    "controlled_gt_jitter", "frame_reference", "route_reference",
    "scheduled_route_reference",
}:
    raise ValueError(
        "UAVSAT_REFERENCE_PROTOCOL must be controlled_gt_jitter, frame_reference, "
        "route_reference, or scheduled_route_reference"
    )
FRAME_REFERENCE_SUPERVISION = REFERENCE_PROTOCOL in {
    "frame_reference", "route_reference", "scheduled_route_reference"
}
ROUTE_REFERENCE_ONLY = REFERENCE_PROTOCOL == "route_reference"
SCHEDULED_ROUTE_REFERENCE = REFERENCE_PROTOCOL == "scheduled_route_reference"
# Evaluation/inference must not read the current frame's GT coordinate or use a
# GT-derived motion/progress cap in either route-only protocol.
NO_GT_INFERENCE = ROUTE_REFERENCE_ONLY or SCHEDULED_ROUTE_REFERENCE
CONTROLLED_PROTOCOL_NAME = (
    "route_reference_motion_prior_forward3x6_SoftMS_3frame_GRU_"
    "next_position_polynomial_Kalman"
    if ROUTE_REFERENCE_ONLY
    else (
        "scheduled_route_reference_forward3x6_SoftMS_3frame_GRU_"
        "next_position_polynomial_Kalman"
        if SCHEDULED_ROUTE_REFERENCE
        else
        "frame_indexed_reference_forward3x6_SoftMS_3frame_GRU_"
        "next_position_polynomial_Kalman"
        if FRAME_REFERENCE_SUPERVISION
        else "GT+smooth-jitter_controlled_forward3x6_SoftMS_local_prior_3frame_"
             "causal_heading_continuous_waypoint_RNN_polynomial_Kalman"
    )
)

WAYPOINT_DIR = PROJECT_ROOT / "route_waypoints"
WAYPOINT_FILES = {
    "route_A": WAYPOINT_DIR / "route_A_waypoints.json",
    "route_B": WAYPOINT_DIR / "route_B_waypoints.json",
    "route_C": WAYPOINT_DIR / "route_C_waypoints.json",
}
DENSE_ROUTE_REFERENCE_DIR = Path(
    os.environ.get(
        "UAVSAT_DENSE_ROUTE_REFERENCE_DIR",
        str(PROJECT_ROOT / "frame-reference-exp" / "references"),
    )
).resolve()

SAT_IMAGE = Path(
    "/yh/study/sim_data/sim_competition_crop_check/"
    "sim_map_competition_roi_crop.png"
)
SAT_JSON = Path(
    "/yh/study/sim_data/sim_competition_crop_check/"
    "sim_map_competition_roi_crop_worldfile_epsg3826.json"
)

# -----------------------------------------------------------------------------
# Local UAV<->SAT retrieval (checkpoint-compatible with the previous local model)
# -----------------------------------------------------------------------------
BACKBONE_NAME, CLIP_DIM = BACKBONE_SPECS[BACKBONE_KEY]
EMBED_DIM = 512
BACKBONE_HEAD_DIM = 512
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
VISUAL_CACHE_BATCH_SIZE = int(os.environ.get("UAVSAT_VISUAL_CACHE_BATCH_SIZE", "256"))
VISUAL_EARLY_STOPPING_PATIENCE = 8
VISUAL_LABEL_SMOOTHING = 0.05
VISUAL_COORD_LOSS_WEIGHT = 0.25

MEANSHIFT_SCORE_TAU = 0.30
MEANSHIFT_BANDWIDTH_M = 8.0
MEANSHIFT_ITERATIONS = 3
# All patches seed SoftMS and no fixed Top-K is used. Seeds converging to the
# same basin are consolidated before the final coordinate aggregation.
MEANSHIFT_MODE_BETA = 12.0
# Shifted seeds whose converged coordinates are within this radius represent
# one Mean-Shift basin.  They are consolidated before the final coordinate
# aggregation; this is not a fixed Top-K.
MEANSHIFT_MODE_MERGE_RADIUS_M = 2.0
GRID_SIZE = 6
CANDIDATE_COUNT = 36
LOCAL_PRIOR_JITTER_M = 12.0
CANDIDATE_CAPTURE_RADIUS_M = 7.5
MIN_TRAIN_CAPTURE_RATE = 0.95

# -----------------------------------------------------------------------------
# v32 controlled local acquisition. Exactly one 6x6 SAT window is used per frame
# and its center is current-frame GT + bounded deterministic jitter. The legacy
# acquisition fields stay for checkpoint/code compatibility but the bank size is 1.
# -----------------------------------------------------------------------------
ACQ_HYPOTHESIS_COUNT = (
    int(os.environ.get("UAVSAT_ROUTE_REFERENCE_HYPOTHESES", "13"))
    if ROUTE_REFERENCE_ONLY else 1
)
ACQ_LOCAL_GRID_SIZE = 6
ROUTE_REFERENCE_BANK_RADIUS_M = float(
    os.environ.get("UAVSAT_ROUTE_REFERENCE_BANK_RADIUS_M", "60.0")
)

# v33 forward-only local visual search. The original 6x6 geometry is built only
# to determine which half lies ahead of the causal estimated heading. Only the
# selected 3x6=18 forward candidates are encoded/scored by the visual model.
FORWARD_ONLY_LOCAL_SEARCH = os.environ.get("UAVSAT_EXPERIMENT_FORWARD_ONLY", "1") == "1"
FORWARD_SEARCH_ROWS = 3
FORWARD_SEARCH_COLS = 6
FORWARD_SEARCH_CANDIDATE_COUNT = FORWARD_SEARCH_ROWS * FORWARD_SEARCH_COLS
FORWARD_SEARCH_ORIGIN_BACKSHIFT_M = float(
    os.environ.get("UAVSAT_FORWARD_ORIGIN_BACKSHIFT_M", "0.0")
)
ACQ_MIN_RADIUS_M = 0.0
ACQ_BASE_RADIUS_M = 0.0
ACQ_MAX_RADIUS_M = 0.0
ACQ_STD_GAIN = 0.0
ACQ_SPEED_HORIZON_FRAMES = 0.0
ACQ_LOW_CONFIDENCE_GAIN_M = 0.0
ACQ_INITIAL_CONFIDENCE = 0.55
ACQ_VISUAL_TEMPERATURE = 0.42
ACQ_LOCAL_PRIOR_SIGMA_M = 18.0
ACQ_LOCAL_PRIOR_WEIGHT = 0.55
ACQ_HYPOTHESIS_MOTION_PRIOR_WEIGHT = 0.20
ACQ_RAW_VISUAL_EVIDENCE_WEIGHT = (
    float(os.environ.get("UAVSAT_ACQ_RAW_VISUAL_EVIDENCE_WEIGHT", "2.0"))
    if ROUTE_REFERENCE_ONLY else 0.0
)
ACQ_SCORER_TEMPERATURE = 0.75
ACQ_POSTERIOR_EPS = 1e-8
ACQ_MAX_RESPONSE_VARIANCE_M2 = 900.0
ACQ_LOW_CONF_VARIANCE_GAIN = 18.0
ACQ_NUMERIC_DIM = 8

# Controlled protocol: GT+jitter remains active in training, validation and B/C
# evaluation. There is no autonomous acquisition/re-localization claim in v30.
ACQ_TRAIN_PROGRESS_OFFSET_M = 0.0
ACQ_TRAIN_CROSS_OFFSET_M = 0.0
ACQ_TEACHER_FINAL = 1.0
ACQ_TEACHER_DECAY_EPOCHS = 24

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_GUARD_FRAMES = 16

# -----------------------------------------------------------------------------
# Continuous route coordinates. s is progress on the ordered waypoint polyline;
# e is signed cross-track displacement. waypoint frame_index is not an input.
# -----------------------------------------------------------------------------
WAYPOINT_MIN_LEG_LENGTH_M = 1.0
ROUTE_PROGRESS_SCALE_M = 100.0
ROUTE_CROSS_TRACK_SCALE_M = 25.0
ROUTE_REMAINING_SCALE_M = 100.0
ROUTE_STEP_SCALE_M = 10.0

# -----------------------------------------------------------------------------
# Three-frame recurrent motion state.
# -----------------------------------------------------------------------------
TEMPORAL_WINDOW_FRAMES = 3
RNN_HIDDEN_DIM = 256
RNN_FEATURE_DIM = 128
# Only simplify the main GRU input. Everything else remains the v34 protocol:
# response variance(2) + visual innovation(2) + previous velocity(2)
# + previous heading residual/turn-rate(2).
RNN_NUMERIC_DIM = 8
RNN_DROPOUT = 0.10
MAX_FORWARD_SPEED_M_PER_FRAME = 14.0
MAX_CROSS_SPEED_M_PER_FRAME = 5.0
MAX_FINAL_CROSS_TRACK_M = 10.0
MAX_FORWARD_ACCEL_M_PER_FRAME2 = 5.0
MAX_CROSS_ACCEL_M_PER_FRAME2 = 4.0
MAX_POLYNOMIAL_STEP_M_PER_FRAME = 14.0
MAX_MEASUREMENT_CORRECTION_PARALLEL_M = 4.0
MAX_MEASUREMENT_CORRECTION_CROSS_M = 4.0


# -----------------------------------------------------------------------------
# Explicit heading / turn state. The GRU predicts heading residual relative to
# the known smooth waypoint-route tangent and an angular rate. These states are
# used by the second-order polynomial itself, not only logged for visualization.
# -----------------------------------------------------------------------------
MAX_HEADING_RESIDUAL_DEG = 70.0
MAX_TURN_RATE_DEG_PER_FRAME = 12.0
HEADING_STATE_EMA_ALPHA = 0.35
TURN_RATE_EMA_ALPHA = 0.30
MAX_HEADING_DELTA_DEG_PER_FRAME = 5.0
MAX_TURN_RATE_DELTA_DEG_PER_FRAME2 = 5.0

# Core angular supervision. Keep heading because an angle is periodic and is
# directly used to rotate the second-order polynomial displacement.
LOSS_HEADING = 1.25

# Turn-rate is derived again from consecutive reference-position headings and is
# not directly used by the polynomial rotation in the current implementation.
# Disable it as an independent supervised objective.
LOSS_TURN_RATE = 0.0
EARLY_SCORE_HEADING_WEIGHT = 0.03

# -----------------------------------------------------------------------------
# Sequential temporal training. TBPTT detaches hidden state; the local SAT prior
# remains controlled by current-frame GT+jitter on every frame.
# -----------------------------------------------------------------------------
TEMPORAL_EPOCHS = 60
TEMPORAL_LR = 2e-4
TEMPORAL_WEIGHT_DECAY = 1e-3
TBPTT_STEPS = 32
GRAD_CLIP_NORM = 5.0
MOTION_WARMUP_EPOCHS = 6
TEACHER_RATIO_FINAL = 1.0
TEACHER_DECAY_EPOCHS = 24
TRAIN_CENTER_JITTER_M = 6.0
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 0.05
EARLY_STOP_MIN_EPOCH = 18

# -----------------------------------------------------------------------------
# Clean temporal supervision.
#
# The temporal targets are all derived from the same ordered reference-position
# sequence. Therefore the primary objective should supervise the quantities that
# are directly needed by the estimator:
#   1) visual position measurement,
#   2) final heading-aware next-step displacement,
#   3) heading,
#   4) measurement uncertainty.
#
# Velocity is retained only as a weak auxiliary constraint so the velocity head
# remains interpretable. Acceleration, turn-rate, speed, route-progress and the
# hand-crafted cross-motion penalty are disabled as independent loss terms.
# The acceleration head is still trained end-to-end through LOSS_NEXT_STEP,
# because next_step = heading_rotate(v + 0.5*a).
# -----------------------------------------------------------------------------
# Only one local hypothesis is used in the current protocol, so categorical
# acquisition cross-entropy is degenerate (one class) and contributes no
# useful supervision.
LOSS_ACQUISITION = 0.0
LOSS_MEASUREMENT = 1.00
LOSS_NEXT_STEP = 3.00

# Weak auxiliary supervision only; target is derived from frame-to-frame
# reference displacement rather than an independent velocity sensor.
LOSS_VELOCITY = 0.25

# No independent acceleration sensor/label. Let acceleration be learned through
# the end-to-end next-step displacement objective.
LOSS_ACCELERATION = 0.0

# Redundant with the forward component of LOSS_NEXT_STEP.
LOSS_SPEED = 0.0

# Hand-crafted preference for near-zero lateral motion; disable it so the data
# and the next-step target determine lateral motion.
LOSS_CROSS_MOTION_REG = 0.0

# Required for the learned measurement variance used by Kalman update.
LOSS_VARIANCE_NLL = 0.05

# Redundant with the longitudinal component of LOSS_MEASUREMENT.
LOSS_PROGRESS = 0.0

# Frame-reference experiment: the two primary Smooth-L1 objectives are
# (1) current visual measurement -> current frame reference position, and
# (2) frame-t polynomial prediction -> frame-(t+1) reference position.
if FRAME_REFERENCE_SUPERVISION:
    LOSS_MEASUREMENT = 3.0
    LOSS_NEXT_STEP = 3.0
    LOSS_VELOCITY = 0.10
if ROUTE_REFERENCE_ONLY:
    # Route-A GT labels the correct route-bank hypothesis during training;
    # inference receives no GT and uses the learned acquisition scorer.
    LOSS_ACQUISITION = 1.0

# -----------------------------------------------------------------------------
# External Kalman [s,e,vs,ve]. Position estimate is allowed to move backwards
# when a new image corrects an earlier over-prediction; physical forward motion
# is represented by non-negative v_s, not by clamping the estimated position.
# -----------------------------------------------------------------------------
KALMAN_INIT_PROGRESS_VAR = 16.0
KALMAN_INIT_CROSS_VAR = 9.0
KALMAN_INIT_VELOCITY_VAR = 16.0
KALMAN_Q_PROGRESS = 0.20
KALMAN_Q_CROSS = 0.08
KALMAN_Q_VELOCITY = 0.20
KALMAN_R_MIN_VAR = 4.00
KALMAN_R_MAX_VAR = 2500.0
KALMAN_NIS_SOFT_THRESHOLD = 9.21
KALMAN_NIS_MAX_R_SCALE = 40.0
KALMAN_NIS_CONFIDENCE_BOOST = 2.0

# v32 continuous-waypoint no-jump controlled estimator. The single-hypothesis acquisition score is
# never used as confidence because with one hypothesis it is identically 1.
# Confidence is derived from the 6x6 local posterior itself.
VISUAL_CONFIDENCE_FLOOR = 0.08
VISUAL_CONFIDENCE_CEIL = 0.95
VISUAL_CONFIDENCE_MARGIN_SCALE = 0.12
VISUAL_CONFIDENCE_VARIANCE_SCALE_M2 = 80.0

# Learned RNN motion is rate-limited before entering the inertial polynomial.
# This preserves acceleration/inertia rather than allowing frame-to-frame state jumps.
MOTION_VELOCITY_EMA_ALPHA = 0.55
MOTION_ACCELERATION_EMA_ALPHA = 0.35
MAX_MOTION_VELOCITY_DELTA_M_PER_FRAME = 2.0
MAX_MOTION_ACCEL_DELTA_M_PER_FRAME2 = 1.5
MOTION_POLYNOMIAL_STEP_EMA_ALPHA = 0.60
MAX_POLYNOMIAL_STEP_DELTA_M_PER_FRAME = 2.5

# Robust constrained Kalman. Visual measurements may correct the polynomial prior,
# but cannot teleport the posterior to a different local patch in one frame.
KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M = 5.0
KALMAN_MAX_MEASUREMENT_INNOVATION_CROSS_M = 3.0
KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M = 3.0
KALMAN_MAX_POSTERIOR_CORRECTION_CROSS_M = 1.75
KALMAN_MAX_VELOCITY_CORRECTION_M_PER_FRAME = 1.25
KALMAN_FINAL_STEP_SLACK_M = 0.00
KALMAN_FINAL_STEP_MIN_M = 0.00
KALMAN_FINAL_STEP_MAX_M = 7.00

# Smooth the lateral route frame around waypoint corners so a nonzero cross-track
# estimate does not rotate discontinuously when s crosses a waypoint boundary.
ROUTE_FRAME_SMOOTH_RADIUS_M = 24.0
ROUTE_FRAME_POSTTURN_ONLY = True
ROUTE_PROJECTION_SWITCH_RADIUS_M = 24.0
ROUTE_PROJECTION_SWITCH_MARGIN_M = 2.0
ROUTE_PROJECTION_PROGRESS_STEP_FACTOR = 1.75
ROUTE_PROJECTION_PROGRESS_STEP_SLACK_M = 1.0

# Composite early-stop score for the controlled local-prior validation protocol.
EARLY_SCORE_SPEED_WEIGHT = 2.0
EARLY_SCORE_PROGRESS_WEIGHT = 0.15
EARLY_SCORE_MISS_WEIGHT = 0.06

JUMP_TOLERANCE_M = 5.0
LEGACY_STEP_THRESHOLD_M = 3.0
VIDEO_FPS = 12.0
VIDEO_WIDTH = 1800
VIDEO_HEIGHT = 900

SEED = 2033
DEVICE = "cuda"
FEATURE_CACHE_DTYPE = "float16"

# Optional end-to-end timing. The timer starts at an already transformed UAV
# tensor and stops after the external Kalman produces final XY.
MEASURE_END_TO_END_LATENCY = os.environ.get("UAVSAT_MEASURE_LATENCY", "0") == "1"
LATENCY_WARMUP_FRAMES = int(os.environ.get("UAVSAT_LATENCY_WARMUP", "30"))

# v36-exp controlled ablations. Defaults reproduce the complete V36 exactly.
EXPERIMENT_VARIANT = os.environ.get("UAVSAT_EXPERIMENT_VARIANT", "full_v36")
EXPERIMENT_ANCHOR = os.environ.get("UAVSAT_EXPERIMENT_ANCHOR", "softms")
EXPERIMENT_FRAME_COUNT = int(os.environ.get("UAVSAT_EXPERIMENT_FRAME_COUNT", "3"))
EXPERIMENT_MOTION = os.environ.get("UAVSAT_EXPERIMENT_MOTION", "quadratic")
EXPERIMENT_KALMAN = os.environ.get("UAVSAT_EXPERIMENT_KALMAN", "learned")
EXPERIMENT_DISABLE_GRU = os.environ.get("UAVSAT_EXPERIMENT_DISABLE_GRU", "0") == "1"
EXPERIMENT_FIXED_VARIANCE_M2 = float(os.environ.get("UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2", "25.0"))
FEATURE_CACHE_DIR = Path(
    os.environ.get("UAVSAT_FEATURE_CACHE_DIR", str(OUTPUT_DIR / "feature_cache"))
).resolve()

if EXPERIMENT_ANCHOR not in {"softms", "weighted_centroid"}:
    raise ValueError("UAVSAT_EXPERIMENT_ANCHOR must be softms or weighted_centroid")
if EXPERIMENT_FRAME_COUNT not in {1, 2, 3}:
    raise ValueError("UAVSAT_EXPERIMENT_FRAME_COUNT must be 1, 2, or 3")
if EXPERIMENT_MOTION not in {"none", "velocity", "quadratic"}:
    raise ValueError("UAVSAT_EXPERIMENT_MOTION must be none, velocity, or quadratic")
if EXPERIMENT_KALMAN not in {"none", "fixed", "learned"}:
    raise ValueError("UAVSAT_EXPERIMENT_KALMAN must be none, fixed, or learned")

# -----------------------------------------------------------------------------
# v32 controlled pace envelope. This controlled protocol intentionally uses the
# current GT trajectory only as a speed/turn safety envelope, because the user
# requested that the displayed prediction never outrun or pre-turn the GT.
# It is not an autonomous-navigation protocol.
# -----------------------------------------------------------------------------
CONTROLLED_GT_MOTION_ENVELOPE = True
CONTROLLED_MAX_STEP_RATIO = 1.25
CONTROLLED_PACE_ASSIST = True
CONTROLLED_PACE_MIN_RATIO = 0.92
CONTROLLED_PACE_CATCHUP_GAIN = 0.08
CONTROLLED_PACE_MAX_EXTRA_M = 0.75
CONTROLLED_MIN_STEP_ALLOWANCE_M = 0.0
CONTROLLED_CAUSAL_HEADING = True
CONTROLLED_NO_PRETURN = True
