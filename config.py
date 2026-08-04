"""Configuration for learned temporal UAV--satellite localization.

The final method is a trainable temporal architecture.  The archived visual
checkpoint remains frozen and supplies UAV/SAT embeddings; every temporal
motion, candidate re-ranking, uncertainty and correction term is learned.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "temporal_prior_hardms"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "temporal_motion_retriever.pt"

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

# Frozen visual retrieval model.
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

# Archived HardMS is evaluated only as a baseline.
MEANSHIFT_SCORE_TAU = 0.30
MEANSHIFT_BANDWIDTH_M = 8.0
MEANSHIFT_ITERATIONS = 3

# ---------------------------------------------------------------------------
# Proposed architecture: Temporal Motion-Conditioned Retrieval (TMCR)
# ---------------------------------------------------------------------------
# Five frames are used only to initialise the temporal state.  Evaluation is
# closed-loop after these frames; no later GT position is used as input.
HISTORY = 5
SEQUENCE_LENGTH = 24
SEQUENCE_STRIDE = 8
FRAME_STRIDE = 1

# A single fixed candidate lattice is used.  There is no hand-written dynamic
# recovery rule.  The temporal model learns to keep its motion centre inside it.
GRID_SIZE = 15

# Frozen spatial features are pooled before all-pairs correlation.
MOTION_SPATIAL_SIZE = 4
MOTION_CORR_CHANNELS = 64
MOTION_FEATURE_DIM = 128
GLOBAL_PAIR_DIM = 128
TEMPORAL_HIDDEN = 256
CANDIDATE_TOKEN_DIM = 256
TEMPORAL_DROPOUT = 0.10
POSITION_SCALE_M = 20.0
DT_SCALE = 10.0
MOTION_LOGVAR_MIN = -5.0
MOTION_LOGVAR_MAX = 5.0
TEMPORAL_TEMPERATURE = 1.0

# Paper ablations. Change one flag at a time and retrain.
USE_SPATIAL_CORRELATION = True
USE_TEMPORAL_CANDIDATE_DECODER = True
USE_LEARNED_UNCERTAINTY_GATE = True

# Scheduled sampling is a training protocol, not an inference aid.  Early
# epochs centre some candidate windows on GT+jitter so that the decoder learns
# a meaningful posterior; the probability decays toward closed-loop training.
TEACHER_FORCING_START = 0.90
TEACHER_FORCING_END = 0.10
TEACHER_JITTER_M = 18.0

# Optimisation.
EPOCHS = 60
LR = 2e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP_NORM = 5.0
NUM_WORKERS = 4
SEED = 2027
DEVICE = "cuda"

# Loss: each term corresponds to a learned component of the architecture.
LOSS_COORD = 1.0
LOSS_CANDIDATE = 0.5
LOSS_MOTION_NLL = 0.5
LOSS_RELATIVE = 0.35
LOSS_ACCELERATION = 0.10
LOSS_UNCERTAINTY = 0.05

# Contiguous route splits prevent random-frame leakage.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_GUARD_FRAMES = HISTORY

# Evaluation.
JUMP_TOLERANCE_M = 3.0
EVAL_BATCH_SIZE = 64
FEATURE_CACHE_DTYPE = "float16"