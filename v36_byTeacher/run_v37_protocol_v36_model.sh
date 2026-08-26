#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
V37_DIR="$ROOT/v37"
BACKBONE="${UAVSAT_BACKBONE:-mobilenet_v3_small}"
GPU="${GPU:-0}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-220}"
OUT="$HERE/output/$BACKBONE/v37_protocol_v36_model"
LOG="$OUT/train_eval.log"

mkdir -p "$OUT"

if [[ ! -d "$V37_DIR" ]]; then
  echo "missing v37 directory: $V37_DIR" >&2
  exit 2
fi

echo "[1/2] rebuild v37 scheduled references (offline sparse route anchors, direct XY)"
(
  cd "$V37_DIR"
  UAVSAT_BACKBONE="$BACKBONE" \
  UAVSAT_REFERENCE_PROTOCOL=scheduled_route_reference \
  python3 scripts/build_scheduled_references.py
)

echo "[2/2] run exact v37 non-model protocol with only the v36 temporal/visual model module substituted"
CUDA_VISIBLE_DEVICES="$GPU" \
V36_DIR="$HERE" \
V37_DIR="$V37_DIR" \
UAVSAT_BACKBONE="$BACKBONE" \
UAVSAT_REFERENCE_PROTOCOL=scheduled_route_reference \
UAVSAT_TRAIN_ROUTES=route_B,route_C \
UAVSAT_VALIDATION_ROUTES=route_A \
UAVSAT_EVAL_ROUTES=route_A \
UAVSAT_OUTPUT_DIR="$OUT" \
VISUAL_EPOCHS="$VISUAL_EPOCHS" \
TEMPORAL_EPOCHS="$TEMPORAL_EPOCHS" \
python3 -u - <<'PY' 2>&1 | tee "$LOG"
import importlib.util
import os
import sys
from pathlib import Path

v36_dir = Path(os.environ["V36_DIR"]).resolve()
v37_dir = Path(os.environ["V37_DIR"]).resolve()

# Load the v37 configuration first. This keeps the v37 data protocol, scheduled
# reference implementation, direct reference-XY search centre, B+C/A split,
# forward 4x6 geometry, visual training, losses, optimizer settings, TBPTT
# schedule, motion stabilizers and constrained external Kalman.
sys.path.insert(0, str(v37_dir))
import config

# The ONLY intended architectural substitution is the v36_byTeacher model.
# v36 uses two explicit UAV frames and a 10-D numeric state. Keep those model
# dimensions while leaving the surrounding v37 pipeline untouched.
config.ARCHITECTURE_NAME = (
    "V36_byTeacher_2FrameModel_on_v37_"
    "ScheduledReference_BCtoA_Forward4x6"
)
config.EXPERIMENT_FRAME_COUNT = 2
config.RNN_NUMERIC_DIM = 10

# Run the bounded branch of the v36 model so the model-facing dynamics match
# the bounded v37 estimator instead of the old v36 pure/unbounded experiment.
config.MANUAL_DYNAMICS_CONSTRAINTS = True
config.GRU_VISUAL_MEASUREMENT_PROGRESS_RANGE_M = float(
    getattr(config, "MAX_MEASUREMENT_CORRECTION_PARALLEL_M", 4.0)
)
config.GRU_VISUAL_MEASUREMENT_CROSS_RANGE_M = float(
    getattr(config, "MAX_MEASUREMENT_CORRECTION_CROSS_M", 4.0)
)
config.GRU_VISUAL_MEASUREMENT_INIT_PROGRESS_M = 0.0
config.GRU_VISUAL_VARIANCE_INIT_M2 = 4.0
config.RNN_HEADING_INPUT_SCALE_DEG = float(config.MAX_HEADING_RESIDUAL_DEG)
config.RNN_TURN_RATE_INPUT_SCALE_DEG_PER_FRAME = float(
    config.MAX_TURN_RATE_DEG_PER_FRAME
)

# Load v36_byTeacher/visual_model.py under the module name expected by v37.
# Its AllMapGeoCLIP is checkpoint-compatible; its ThreeFrameRouteStateGRU alias
# points to the v36 two-frame teacher model. v37 visual_localizer and tracker are
# imported only AFTER this substitution, so every non-model component is v37.
spec = importlib.util.spec_from_file_location(
    "visual_model", v36_dir / "visual_model.py"
)
visual_model = importlib.util.module_from_spec(spec)
sys.modules["visual_model"] = visual_model
spec.loader.exec_module(visual_model)

import robust_tracker

print("=== v37 protocol / v36 model control experiment ===", flush=True)
print("model module:", v36_dir / "visual_model.py", flush=True)
print("train routes:", config.TRAIN_ROUTES, flush=True)
print("validation routes:", config.VALIDATION_ROUTES, flush=True)
print("reference protocol:", config.REFERENCE_PROTOCOL, flush=True)
print("reference directory:", config.DENSE_ROUTE_REFERENCE_DIR, flush=True)
print(
    "search: %dx%d (%d candidates)"
    % (
        int(config.FORWARD_SEARCH_ROWS),
        int(config.FORWARD_SEARCH_COLS),
        int(config.FORWARD_SEARCH_CANDIDATE_COUNT),
    ),
    flush=True,
)
print(
    "temporal: lr=%g wd=%g tbptt=%d grad_clip=%g"
    % (
        float(config.TEMPORAL_LR),
        float(config.TEMPORAL_WEIGHT_DECAY),
        int(config.TBPTT_STEPS),
        float(config.GRAD_CLIP_NORM),
    ),
    flush=True,
)
print(
    "losses: measurement=%g next_step=%g velocity=%g heading=%g variance=%g"
    % (
        float(config.LOSS_MEASUREMENT),
        float(config.LOSS_NEXT_STEP),
        float(config.LOSS_VELOCITY),
        float(config.LOSS_HEADING),
        float(config.LOSS_VARIANCE_NLL),
    ),
    flush=True,
)

sys.argv = [
    "robust_tracker.py",
    "--mode", "train_eval",
    "--visual-epochs", os.environ["VISUAL_EPOCHS"],
    "--temporal-epochs", os.environ["TEMPORAL_EPOCHS"],
    "--jitter-m", "0",
]
robust_tracker.main()
PY

echo "done: $OUT"
