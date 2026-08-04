"""Configuration for leakage-free, turn-aware straight-line UAV tracking."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "straight_line_hardms_v2"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

DATASETS_ROOT = Path("/yh/study/new_data_2")
ROUTE_ROOTS = [
    DATASETS_ROOT / "model_dataset_new_1_flight",  # Route A
    DATASETS_ROOT / "model_dataset_new_2_flight",  # Route B
    Path("/yh/study/new_data/model_dataset_flight"),  # Route C
]
ROUTE_NAMES = ["route_A", "route_B", "route_C"]

SAT_IMAGE = Path(
    "/yh/study/sim_data/sim_competition_crop_check/"
    "sim_map_competition_roi_crop.png"
)
SAT_JSON = Path(
    "/yh/study/sim_data/sim_competition_crop_check/"
    "sim_map_competition_roi_crop_worldfile_epsg3826.json"
)
VISUAL_CHECKPOINT = Path(
    "/yh/study/UAV_GPS_allmap_imgonly4/outputs/"
    "basinrank_B_fixed_hardms/checkpoints/best.pt"
)

# Frozen retrieval model.
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
SAT_MPP = 0.3
CANDIDATE_SPACING_M = SAT_STRIDE * SAT_MPP
USE_COORD_ENCODER = False
USE_QAH_MS_RELATION = False
USE_BASIN_RANK_MS = False

# IMPORTANT: sampling is based only on image/frame order, never on future GT.
# 1 means every UAV frame. Set 3 only when you explicitly want a faster test.
TRACK_FRAME_STRIDE = 1
HISTORY = 5
LINE_FIT_HISTORY = 6
MAX_FRAME_ID_GAP = 20

# Candidate search. Recovery uses several overlapping search centres, forming a
# corridor between the current prediction and the last reliable position.
GRID_SIZE = 6
GRID_SIZE_MEDIUM = 10
GRID_SIZE_RECOVERY = 14
MAX_SEARCH_CENTERS = 3

# Fixed Hard Mean-Shift produces several spatial modes instead of one early
# irreversible Top-1 decision.
MEANSHIFT_SCORE_TAU = 0.30
MEANSHIFT_BANDWIDTH_M = 8.0
MEANSHIFT_ITERATIONS = 3
TOP_MODES = 4
MODE_NMS_RADIUS_M = 12.0
MODE_LOCAL_RADIUS_M = 16.0

# The visual posterior remains dominant. The straight-line prior is deliberately
# weak, especially during recovery, so a real turn can overcome the old heading.
MOTION_PRIOR_WEIGHT_NORMAL = 0.35
MOTION_PRIOR_WEIGHT_MEDIUM = 0.18
MOTION_PRIOR_WEIGHT_RECOVERY = 0.05
MOTION_SIGMA_ALONG_M = 18.0
MOTION_SIGMA_CROSS_M = 8.0
RECOVERY_SIGMA_ALONG_M = 45.0
RECOVERY_SIGMA_CROSS_M = 30.0

# State is [x, y, vx, vy], where velocity is metres per original frame ID.
POSITION_STD_M = 3.0
VELOCITY_STD_M_PER_FRAME = 2.0
PROCESS_POSITION_STD_M = 0.9
PROCESS_VELOCITY_STD_M_PER_FRAME = 0.45
MAX_POSITION_STD_M = 80.0
MAX_SPEED_M_PER_FRAME = 7.0
MIN_NONZERO_SPEED_M_PER_FRAME = 0.05

# Robust position/velocity correction. These limits scale with the real frame
# interval dt, rather than assuming every selected node has dt=1.
MEASUREMENT_STD_MIN_M = 3.0
MEASUREMENT_STD_MAX_M = 32.0
MAX_ALONG_CORRECTION_M_PER_FRAME = 6.0
MAX_CROSS_CORRECTION_M_PER_FRAME = 4.0
RECOVERY_MAX_CORRECTION_M_PER_FRAME = 14.0
VELOCITY_OBSERVATION_GAIN = 0.35
VELOCITY_HISTORY_BLEND = 0.15
MAX_TURN_DEG_PER_FRAME = 22.0
RECOVERY_MAX_TURN_DEG_PER_FRAME = 75.0

# Mode selection score.
VISUAL_SCORE_WEIGHT = 1.0
MOTION_SCORE_WEIGHT_NORMAL = 0.55
MOTION_SCORE_WEIGHT_MEDIUM = 0.25
MOTION_SCORE_WEIGHT_RECOVERY = 0.08
TEMPORAL_MODE_WEIGHT = 0.20
SPATIAL_STD_PENALTY = 0.25

# Grid-size-invariant confidence. Do not threshold absolute Top-1 probability,
# because 0.08 is unreasonable for a 14x14 or corridor candidate set.
RELIABLE_CONFIDENCE = 0.48
WEAK_CONFIDENCE = 0.28
GOOD_STREAK_TO_SHRINK = 3
LOST_STREAK_MEDIUM = 1
LOST_STREAK_RECOVERY = 3
BOUNDARY_RATIO = 0.82

# Evaluation only. GT is never used to choose frames or update the state.
JUMP_TOLERANCE_M = 2.0

DEVICE = "cuda"
NUM_WORKERS = 6
SEED = 2027