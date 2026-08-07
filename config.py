from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
# Keep this ablation in a separate directory so the original /10 result is not
# overwritten.  Training/evaluation protocol remains Route A -> Route B/C.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "strict_train_A_test_BC_no_position_scale_w4"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / "visual_retrieval_A_only.pt"
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / "rtl_crf_A_only.pt"
DATASETS_ROOT = Path("/yh/study/new_data_2")
ROUTE_ROOTS = [
    DATASETS_ROOT / "model_dataset_new_1_flight",
    DATASETS_ROOT / "model_dataset_new_2_flight",
    Path("/yh/study/new_data/model_dataset_flight"),
]
ROUTE_NAMES = ["route_A", "route_B", "route_C"]
TRAIN_ROUTE_NAMES = ["route_A"]
EVAL_ROUTE_NAMES = ["route_B", "route_C"]
SAT_IMAGE = Path(
    "/yh/study/sim_data/sim_competition_crop_check/"
    "sim_map_competition_roi_crop.png"
)
SAT_JSON = Path(
    "/yh/study/sim_data/sim_competition_crop_check/"
    "sim_map_competition_roi_crop_worldfile_epsg3826.json"
)
# Public pretrained backbone only. No task-specific .pt is loaded here.
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
MOTION_SPATIAL_SIZE = 4

# A-only retrieval-head training.
VISUAL_EPOCHS = 30
VISUAL_LR = 3e-4
VISUAL_WEIGHT_DECAY = 1e-3
VISUAL_BATCH_SIZE = 64
VISUAL_CACHE_BATCH_SIZE = 64
VISUAL_EARLY_STOPPING_PATIENCE = 8
VISUAL_LABEL_SMOOTHING = 0.05
VISUAL_COORD_LOSS_WEIGHT = 0.25

# Retrieval baselines.
MEANSHIFT_SCORE_TAU = 0.30
MEANSHIFT_BANDWIDTH_M = 8.0
MEANSHIFT_ITERATIONS = 3

# RTL-CRF: fixed 4-frame second-order experiment.
# Four consecutive positions provide two overlapping second-order T2 terms:
# (t-3,t-2,t-1) and (t-2,t-1,t), while preserving the same model.
GRID_SIZE = 6
LOCAL_PRIOR_JITTER_M = 12.0
CANDIDATE_CAPTURE_RADIUS_M = 7.5
MIN_TRAIN_CAPTURE_RATE = 0.95
TEMPORAL_WINDOW = 4
WINDOW_STRIDE = 1
TOKEN_DIM = 192
TRANSITION_HIDDEN = 64
TEMPORAL_DROPOUT = 0.10
# No POSITION_SCALE_M here. Spatial/motion features are fed to RTL-CRF in
# their original meter-based values for the no-scaling ablation.

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

LOSS_CRF = 1.0
LOSS_FINAL_COORD = 1.0
LOSS_PATH_COORD = 0.35

# Contiguous Route-A split.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
# Keep the same split boundary as the existing 5-frame experiment so the
# 3-frame vs 5-frame comparison changes only temporal context length.
SPLIT_GUARD_FRAMES = 5

JUMP_TOLERANCE_M = 3.0
VAL_RPE_WEIGHT = 0.25
VAL_JUMP_WEIGHT = 0.02
VAL_STATIONARY_WEIGHT = 0.05
