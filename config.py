
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "temporal_prior_hardms"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "temporal_motion_retriever.pt"

DATASETS_ROOT = Path("/yh/study/new_data_2")
ROUTE_ROOTS = [
    DATASETS_ROOT / "model_dataset_new_1_flight",
    DATASETS_ROOT / "model_dataset_new_2_flight",
    Path("/yh/study/new_data/model_dataset_flight"),
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

# Frozen visual retrieval model.  These values must stay compatible with best.pt.
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
MOTION_SPATIAL_SIZE = 4  # retained only for checkpoint/API compatibility

# Frozen retrieval baselines.
MEANSHIFT_SCORE_TAU = 0.30
MEANSHIFT_BANDWIDTH_M = 8.0
MEANSHIFT_ITERATIONS = 3

# ---------------------------------------------------------------------------
# Proposed model: Residual Second-Order Temporal Lattice CRF (RTL-CRF)
# ---------------------------------------------------------------------------
# The experiment uses a controlled noisy local prior.  GT is used only to make
# this prior and to compute supervision; GT is never an input token to the CRF.
GRID_SIZE = 6
LOCAL_PRIOR_JITTER_M = 12.0
TEMPORAL_WINDOW = 5
WINDOW_STRIDE = 1

TOKEN_DIM = 192
TRANSITION_HIDDEN = 64
TEMPORAL_DROPOUT = 0.10
POSITION_SCALE_M = 10.0

# Optimisation.
EPOCHS = 40
LR = 2e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP_NORM = 5.0
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 64
SEED = 2027
DEVICE = "cuda"
FEATURE_CACHE_DTYPE = "float16"
EARLY_STOPPING_PATIENCE = 12

# Sequence-level CRF and coordinate objectives.
LOSS_CRF = 1.0
LOSS_FINAL_COORD = 1.0
LOSS_PATH_COORD = 0.35

# Contiguous route splits prevent random-frame leakage.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_GUARD_FRAMES = TEMPORAL_WINDOW

# Evaluation / checkpoint selection.
JUMP_TOLERANCE_M = 3.0
VAL_RPE_WEIGHT = 0.25
VAL_JUMP_WEIGHT = 0.02
VAL_STATIONARY_WEIGHT = 0.05