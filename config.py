"""Configuration for straight-line prior UAV--satellite tracking."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "straight_line_hardms"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

DATASETS_ROOT = Path("/yh/study/new_data_2")
ROUTE_ROOTS = [
    DATASETS_ROOT / "model_dataset_new_1_flight",  # Route A
    DATASETS_ROOT / "model_dataset_new_2_flight",  # Route B
    Path("/yh/study/new_data/model_dataset_flight"),  # Route C
]
ROUTE_NAMES = ["route_A", "route_B", "route_C"]

SAT_IMAGE = Path("/yh/study/sim_data/sim_competition_crop_check/sim_map_competition_roi_crop.png")
SAT_JSON = Path("/yh/study/sim_data/sim_competition_crop_check/sim_map_competition_roi_crop_worldfile_epsg3826.json")
VISUAL_CHECKPOINT = Path(
    "/yh/study/UAV_GPS_allmap_imgonly4/outputs/"
    "basinrank_B_fixed_hardms/checkpoints/best.pt"
)

# Frozen visual localizer.
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

# Candidate windows. Keep 6x6 as the normal protocol, then expand only when
# confidence is poor or the selected mode lies at the grid boundary.
GRID_SIZE = 6
GRID_SIZE_MEDIUM = 10
GRID_SIZE_LARGE = 14
GRID_SIZE_RECOVERY = 18

MEANSHIFT_SCORE_TAU = 0.30
MEANSHIFT_BANDWIDTH_M = 8.0
MEANSHIFT_ITERATIONS = 3

# Five GT nodes are used only for oracle initialization.
HISTORY = 5
TEMPORAL_POSITION_MERGE_M = 0.5

# Straight-line motion model.
LINE_FIT_HISTORY = 5
VELOCITY_EMA = 0.25
MAX_TURN_DEG_PER_NODE = 18.0
MIN_SPEED_RATIO = 0.55
MAX_SPEED_RATIO = 1.45
ABSOLUTE_MAX_SPEED_M_PER_NODE = 18.0

# Anisotropic motion prior. Cross-track motion is penalized more strongly than
# along-track motion, which suppresses left/right field-row hopping.
MOTION_SIGMA_ALONG_M = 13.0
MOTION_SIGMA_CROSS_M = 4.5
RECOVERY_SIGMA_ALONG_M = 20.0
RECOVERY_SIGMA_CROSS_M = 9.0

# Continuous visual correction. These are hard safety limits on a single
# correction, not arbitrary jump thresholds used as the primary estimator.
MAX_UPDATE_ALPHA = 0.65
MIN_CONFIDENT_ALPHA = 0.05
MAX_ALONG_CORRECTION_M = 8.0
MAX_CROSS_CORRECTION_M = 4.0
EXTREME_INNOVATION_M = 35.0

# Confidence/recovery logic.
HIGH_ENTROPY = 0.82
LOW_PEAK_PROB = 0.08
LOW_MARGIN_PROB = 0.015
GOOD_STREAK_TO_SHRINK = 3
LOST_STREAK_MEDIUM = 1
LOST_STREAK_LARGE = 3
LOST_STREAK_RECOVERY = 6

# Covariance used for diagnostics and candidate expansion.
POSITION_STD_M = 4.0
VELOCITY_STD_M = 3.0
PROCESS_POSITION_STD_M = 1.2
PROCESS_VELOCITY_STD_M = 1.0
MAX_POSITION_STD_M = 30.0

# Evaluation only.
JUMP_TOLERANCE_M = 2.0
CAPTURE_RADIUS_M = 7.0
DEVICE = "cuda"
NUM_WORKERS = 6
SEED = 2027