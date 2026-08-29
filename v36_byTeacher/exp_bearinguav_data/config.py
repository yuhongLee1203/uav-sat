"""v36_byTeacher configuration for same-scene BearingUAV routes."""

import importlib.util
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
spec = importlib.util.spec_from_file_location("_original_v36_config", PARENT / "config.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
for name, value in vars(original).items():
    if not name.startswith("__"):
        globals()[name] = value

# Same-scene public-data protocol:
#   train_1 + train_2 -> training
#   val_1             -> validation
# All routes come from city-A and use one satellite image / one georeference.
# The routes are sparse irregular polylines across the map; diagonal legs and
# crossings between different routes are allowed.
PROJECT_ROOT = HERE
GENERATED_ROOT = HERE / "generated_routes_2train_1val_samecity"
TRAIN_ROUTE_NAMES = ["train_1", "train_2"]
VALIDATION_ROUTE_NAMES = ["val_1"]
TEST_ROUTE_NAMES = []
ROUTE_NAMES = TRAIN_ROUTE_NAMES + VALIDATION_ROUTE_NAMES
ROUTE_ROOTS = [GENERATED_ROOT / name for name in ROUTE_NAMES]
WAYPOINT_DIR = GENERATED_ROOT / "waypoints"
WAYPOINT_FILES = {name: WAYPOINT_DIR / (name + "_waypoints.json") for name in ROUTE_NAMES}
SAT_IMAGE = GENERATED_ROOT / "bearing_citya.jpg"
SAT_JSON = GENERATED_ROOT / "bearing_citya_geo.json"

# The pseudo-routes use real samples near simplified planned centerlines. Keep
# enough cross-track state room for this public-data adapter; this is not a
# validation-label-dependent correction.
MAX_FINAL_CROSS_TRACK_M = float(os.environ.get("BEARING_MAX_CROSS_TRACK_M", "25.0"))

# New protocol ID prevents the previous banded 1000-frame data, checkpoints and
# feature caches from being silently reused.
BEARING_DATA_PROTOCOL = "actualpose_samecity_irregular_2train1val_max600_v6"
BACKBONE_OUTPUT_DIR = HERE / "output" / BACKBONE_KEY
DEFAULT_OUTPUT_DIR = BACKBONE_OUTPUT_DIR / (str(EXPERIMENT_FRAME_COUNT) + "frame")
RUN_TAG = os.environ.get("UAVSAT_RUN_TAG", "").strip()
OUTPUT_DIR = BACKBONE_OUTPUT_DIR / "experiments" / RUN_TAG if RUN_TAG else DEFAULT_OUTPUT_DIR
CHECKPOINT_DIR = BACKBONE_OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / (
    "visual_retrieval_2train_1val_" + BACKBONE_KEY + "_" + BEARING_DATA_PROTOCOL + ".pt"
)
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / (
    "multiroute2train1val_referenceprior_forward3x6_ms_previous_position_"
    + str(EXPERIMENT_FRAME_COUNT) + "frame_" + BACKBONE_KEY
    + "_direct_softms_gru_motion_heading_learned_variance_multirate_native_plus_stride"
    + str(TEMPORAL_EXTRA_A_STRIDE) + "_v7_" + BEARING_DATA_PROTOCOL + ".pt"
)
LATEST_TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / (
    "multiroute2train1val_referenceprior_forward3x6_ms_previous_position_"
    + str(EXPERIMENT_FRAME_COUNT) + "frame_" + BACKBONE_KEY
    + "_direct_softms_gru_motion_heading_learned_variance_multirate_native_plus_stride"
    + str(TEMPORAL_EXTRA_A_STRIDE) + "_v7_" + BEARING_DATA_PROTOCOL + "_latest.pt"
)
FEATURE_CACHE_DIR = BACKBONE_OUTPUT_DIR / ("feature_cache_" + BEARING_DATA_PROTOCOL)
