"""Configuration for temporal prior-guided local UAV--satellite tracking."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "temporal_prior_hardms"
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

# Frozen visual localizer: exactly the archived 6x6 P320/S32 HardMS setting.
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
# The archived 6x6 checkpoint was trained without coordinate features.
USE_COORD_ENCODER = False
USE_QAH_MS_RELATION = False
USE_BASIN_RANK_MS = False

GRID_SIZE = 6
RECOVERY_GRID_SIZE = 10
RECOVERY_MOTION_SIGMA_M = 8.0
RECOVERY_ACCELERATION_M = 3.0
RECOVERY_TRAIN_PROB = 0.30
MEANSHIFT_SCORE_TAU = 0.30
MEANSHIFT_BANDWIDTH_M = 8.0
MEANSHIFT_ITERATIONS = 3

# Motion prior and temporal correction.
HISTORY = 5
SEQUENCE_LENGTH = 32
SEQUENCE_STRIDE = 16
MOTION_SIGMA_M = 10.0
MOTION_SIGMA_MIN_M = 6.0
MOTION_SIGMA_MAX_M = 18.0
VELOCITY_SCALE_M = 8.0
# Route-A learned motion residuals did not transfer to B/C. The first robust
# temporal protocol therefore uses a deterministic constant-velocity prior;
# the TCN only learns whether the frozen visual measurement is trustworthy.
USE_LEARNED_MOTION_RESIDUAL = False
MOTION_RESIDUAL_MAX_M = 2.0
GATE_MAX_ALPHA = 0.35
TEMPORAL_POSITION_MERGE_M = 0.5
POSITION_SCALE_M = 20.0
JUMP_THRESHOLD_M = 15.0

TCN_HIDDEN = 128
TCN_DROPOUT = 0.10
EPOCHS = 45
LR = 2e-4
WEIGHT_DECAY = 1e-3
NUM_WORKERS = 6
SEED = 2027
DEVICE = "cuda"

# L = final + 0.5 motion + 0.25 gate + 0.5 relative + 0.1 acceleration.
LOSS_MOTION = 0.5
LOSS_GATE = 1.0
LOSS_RELATIVE = 0.5
LOSS_ACCELERATION = 0.1

# Each route is split by one contiguous acquisition segment, never by random
# frames. The guard separates both temporal sequences and neighbouring map area.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.20
TEST_FRACTION = 0.10
SPLIT_GUARD_NODES = HISTORY
