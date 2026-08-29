"""v36_byTeacher configuration for corrected BearingUAV actual-pose routes."""

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

# Data-only substitution. Every generated frame label is the selected
# BearingUAV sample's actual source position. The pseudo-sequence is ordered by
# progress along a multi-segment planned route, uses variable physical frame
# spacing, and places a waypoint at every traversed 90-degree turn.
PROJECT_ROOT = HERE
GENERATED_ROOT = HERE / "generated_routes_3train_1val_1test"
TRAIN_ROUTE_NAMES = ["train_1", "train_2", "train_3"]
VALIDATION_ROUTE_NAMES = ["val_1"]
TEST_ROUTE_NAMES = ["test_1"]
ROUTE_NAMES = TRAIN_ROUTE_NAMES + VALIDATION_ROUTE_NAMES + TEST_ROUTE_NAMES
ROUTE_ROOTS = [GENERATED_ROOT / name for name in ROUTE_NAMES]
WAYPOINT_DIR = GENERATED_ROOT / "waypoints"
WAYPOINT_FILES = {name: WAYPOINT_DIR / (name + "_waypoints.json") for name in ROUTE_NAMES}
SAT_IMAGE = GENERATED_ROOT / "bearing_cities_abc.jpg"
SAT_JSON = GENERATED_ROOT / "bearing_cities_abc_geo.json"

# A BearingUAV sample may lie several metres away from the simplified planned
# route centerline. The original 10 m cross-track clamp was designed for the
# real flight routes and caused public-data errors to pin at exactly 10 m.
MAX_FINAL_CROSS_TRACK_M = float(os.environ.get("BEARING_MAX_CROSS_TRACK_M", "25.0"))

# Version all data-derived artifacts. v4 is the exact-count, multi-turn,
# route-specific speed-profile protocol: train_1/2/3 are slow/medium/fast and
# val/test effective step rates lie between the training rates.
BEARING_DATA_PROTOCOL = "actualpose_multiturn_1000_speedprofile_v4"
BACKBONE_OUTPUT_DIR = HERE / "output" / BACKBONE_KEY
DEFAULT_OUTPUT_DIR = BACKBONE_OUTPUT_DIR / (str(EXPERIMENT_FRAME_COUNT) + "frame")
RUN_TAG = os.environ.get("UAVSAT_RUN_TAG", "").strip()
OUTPUT_DIR = BACKBONE_OUTPUT_DIR / "experiments" / RUN_TAG if RUN_TAG else DEFAULT_OUTPUT_DIR
CHECKPOINT_DIR = BACKBONE_OUTPUT_DIR / "checkpoints"
VISUAL_CHECKPOINT = CHECKPOINT_DIR / (
    "visual_retrieval_3train_1val_" + BACKBONE_KEY + "_" + BEARING_DATA_PROTOCOL + ".pt"
)
TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / (
    "multiroute3train1val_referenceprior_forward3x6_ms_previous_position_"
    + str(EXPERIMENT_FRAME_COUNT) + "frame_" + BACKBONE_KEY
    + "_direct_softms_gru_motion_heading_learned_variance_multirate_A_native_plus_stride"
    + str(TEMPORAL_EXTRA_A_STRIDE) + "_v7_" + BEARING_DATA_PROTOCOL + ".pt"
)
LATEST_TEMPORAL_CHECKPOINT = CHECKPOINT_DIR / (
    "multiroute3train1val_referenceprior_forward3x6_ms_previous_position_"
    + str(EXPERIMENT_FRAME_COUNT) + "frame_" + BACKBONE_KEY
    + "_direct_softms_gru_motion_heading_learned_variance_multirate_A_native_plus_stride"
    + str(TEMPORAL_EXTRA_A_STRIDE) + "_v7_" + BEARING_DATA_PROTOCOL + "_latest.pt"
)
FEATURE_CACHE_DIR = BACKBONE_OUTPUT_DIR / ("feature_cache_" + BEARING_DATA_PROTOCOL)
