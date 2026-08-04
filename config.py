"""Configuration for oracle-candidate temporal smoothing diagnostics.

This is intentionally not a closed-loop tracker.  GT (optionally with a
bounded deterministic jitter) is used only to centre the local satellite
candidate window.  The temporal filter never receives GT after the five-frame
initialisation.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "temporal_prior_hardms"

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

# ---------------------------------------------------------------------------
# Current experiment: correct local candidates first, temporal smoothing next.
# ---------------------------------------------------------------------------
# "gt": exact GT centre.  "gt_jitter": GT plus deterministic bounded noise.
CANDIDATE_CENTER_MODE = "gt_jitter"
GT_JITTER_MAX_M = 12.0
GRID_SIZE = 6
TRACK_FRAME_STRIDE = 1
HISTORY = 5
LINE_FIT_HISTORY = 7
MAX_FRAME_ID_GAP = 20

# Fixed Hard Mean-Shift.  Top modes are retained only so the temporal predictor
# may choose a motion-consistent visual hypothesis; mode 1 is still reported as
# the raw HardMS baseline.
MEANSHIFT_SCORE_TAU = 0.30
MEANSHIFT_BANDWIDTH_M = 8.0
MEANSHIFT_ITERATIONS = 3
TOP_MODES = 4
MODE_NMS_RADIUS_M = 12.0
MODE_LOCAL_RADIUS_M = 16.0
BOUNDARY_RATIO = 0.82

# No map-motion prior is injected into retrieval in this diagnostic stage.
MOTION_PRIOR_WEIGHT = 0.0
MOTION_SIGMA_ALONG_M = 30.0
MOTION_SIGMA_CROSS_M = 30.0

# Temporal mode selection.  This is a soft score, never a hard accept/reject.
MODE_VISUAL_WEIGHT = 1.0
MODE_MOTION_WEIGHT = 0.22
MODE_CROSS_WEIGHT = 0.18
MODE_MAX_DISTANCE_UNITS = 4.0

# State [x, y, vx, vy], velocity in metres per original frame ID.
POSITION_STD_M = 3.0
VELOCITY_STD_M_PER_FRAME = 1.5
PROCESS_POSITION_STD_M = 0.6
PROCESS_VELOCITY_STD_M_PER_FRAME = 0.25
MAX_SPEED_M_PER_FRAME = 7.0
MIN_NONZERO_SPEED_M_PER_FRAME = 0.03

# Causal alpha-beta straight-line smoother.
ALPHA_ALONG_MIN = 0.34
ALPHA_ALONG_MAX = 0.72
ALPHA_CROSS_MIN = 0.10
ALPHA_CROSS_MAX = 0.34
BETA_MIN = 0.05
BETA_MAX = 0.24
MAX_ALONG_CORRECTION_M_PER_FRAME = 12.0
MAX_CROSS_CORRECTION_M_PER_FRAME = 5.0
INNOVATION_HUBER_M = 14.0
LINE_HUBER_M = 7.0
VELOCITY_LINE_BLEND = 0.30
MAX_TURN_DEG_PER_FRAME = 28.0

# Confidence is used continuously; it never decides whether GT is in the grid.
CONFIDENCE_FLOOR = 0.08

# Evaluation only.
STATIONARY_GT_STEP_M = 0.50
JUMP_TOLERANCE_M = 2.0

DEVICE = "cuda"
NUM_WORKERS = 6
SEED = 2027