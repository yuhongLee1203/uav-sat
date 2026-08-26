#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
V37_DIR="$ROOT/v37"
BACKBONE="${UAVSAT_BACKBONE:-mobilenet_v3_small}"
GPU="${GPU:-0}"
VISUAL_EPOCHS="${VISUAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-220}"
RESET_RUN="${RESET_RUN:-1}"
OUT="$HERE/output/$BACKBONE/v37_exact_protocol_v36_architecture"
LOG="$OUT/train_eval.log"

if [[ ! -d "$V37_DIR" ]]; then
  echo "missing v37 directory: $V37_DIR" >&2
  exit 2
fi

# A fair control must not inherit any visual or temporal task checkpoint from a
# previous v36 experiment. v37 itself trains the visual task heads from the
# pretrained backbone and then trains temporal B+C -> A.
if [[ "$RESET_RUN" == "1" ]]; then
  rm -rf "$OUT"
fi
mkdir -p "$OUT"

echo "[1/2] build the exact v37 scheduled references"
(
  cd "$V37_DIR"
  UAVSAT_BACKBONE="$BACKBONE" \
  UAVSAT_REFERENCE_PROTOCOL=scheduled_route_reference \
  python3 scripts/build_scheduled_references.py
)

echo "[2/2] exact v37 data/search/training/filter pipeline + v36 learned architecture only"
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

# -------------------------------------------------------------------------
# IMPORTANT CONTROL RULE
# -------------------------------------------------------------------------
# Import v37 config first and keep ALL non-model protocol values from v37:
#   data roots/split, B+C -> A route assignment,
#   scheduled-reference manifests,
#   forward 4x6 candidate construction,
#   visual training and validation,
#   visual/temporal losses,
#   optimizer, LR, weight decay, TBPTT and gradient clipping,
#   temporal chunk schedule / resume behavior,
#   motion-state stabilization,
#   external Kalman predict/update constraints,
#   validation/evaluation code.
# Only values that are required by the v36 learned module itself may differ.
sys.path.insert(0, str(v37_dir))
import config

protocol_keys = [
    "ROUTE_ROOTS",
    "ROUTE_NAMES",
    "TRAIN_ROUTES",
    "VALIDATION_ROUTES",
    "EVAL_ROUTES",
    "REFERENCE_PROTOCOL",
    "DENSE_ROUTE_REFERENCE_DIR",
    "FORWARD_SEARCH_ROWS",
    "FORWARD_SEARCH_COLS",
    "FORWARD_SEARCH_CANDIDATE_COUNT",
    "SAT_STRIDE",
    "MEANSHIFT_SCORE_TAU",
    "MEANSHIFT_BANDWIDTH_M",
    "MEANSHIFT_ITERATIONS",
    "MEANSHIFT_MODE_BETA",
    "VISUAL_LR",
    "VISUAL_WEIGHT_DECAY",
    "VISUAL_BATCH_SIZE",
    "TEMPORAL_LR",
    "TEMPORAL_WEIGHT_DECAY",
    "TBPTT_STEPS",
    "GRAD_CLIP_NORM",
    "LOSS_MEASUREMENT",
    "LOSS_NEXT_STEP",
    "LOSS_VELOCITY",
    "LOSS_ACCELERATION",
    "LOSS_HEADING",
    "LOSS_TURN_RATE",
    "LOSS_VARIANCE_NLL",
    "KALMAN_Q_PROGRESS",
    "KALMAN_Q_CROSS",
    "KALMAN_Q_VELOCITY",
]
protocol_before = {key: getattr(config, key) for key in protocol_keys}

# These are the ONLY intentional differences. They describe the v36 learned
# architecture and are needed for its input/head dimensions and pure learned
# output semantics. They do not change v37 data construction or trainer code.
config.ARCHITECTURE_NAME = (
    "V36_byTeacher_2FramePureModel_on_EXACT_v37_"
    "ScheduledReference_BCtoA_Forward4x6"
)
config.EXPERIMENT_FRAME_COUNT = 2
config.RNN_NUMERIC_DIM = 10
config.MANUAL_DYNAMICS_CONSTRAINTS = False
config.GRU_VISUAL_VARIANCE_INIT_M2 = 4.0
config.RNN_HEADING_INPUT_SCALE_DEG = float(config.MAX_HEADING_RESIDUAL_DEG)
config.RNN_TURN_RATE_INPUT_SCALE_DEG_PER_FRAME = float(
    config.MAX_TURN_RATE_DEG_PER_FRAME
)

# Fail immediately if anything outside the learned architecture was changed.
protocol_after = {key: getattr(config, key) for key in protocol_keys}
for key in protocol_keys:
    before = protocol_before[key]
    after = protocol_after[key]
    if str(before) != str(after):
        raise RuntimeError(
            f"non-model protocol changed unexpectedly: {key}: {before!r} -> {after!r}"
        )

if list(config.TRAIN_ROUTES) != ["route_B", "route_C"]:
    raise RuntimeError(f"expected v37 train routes B+C, got {config.TRAIN_ROUTES}")
if list(config.VALIDATION_ROUTES) != ["route_A"]:
    raise RuntimeError(f"expected v37 validation route A, got {config.VALIDATION_ROUTES}")
if list(config.EVAL_ROUTES) != ["route_A"]:
    raise RuntimeError(f"expected v37 eval route A, got {config.EVAL_ROUTES}")
if str(config.REFERENCE_PROTOCOL) != "scheduled_route_reference":
    raise RuntimeError(f"expected scheduled_route_reference, got {config.REFERENCE_PROTOCOL}")
if int(config.FORWARD_SEARCH_ROWS) != 4 or int(config.FORWARD_SEARCH_COLS) != 6:
    raise RuntimeError(
        "exact v37 protocol requires forward 4x6, got "
        f"{config.FORWARD_SEARCH_ROWS}x{config.FORWARD_SEARCH_COLS}"
    )

# Inject only v36_byTeacher's learned model module under the module name that
# v37 visual_localizer/robust_tracker import. The v36 AllMapGeoCLIP retrieval
# class is checkpoint-compatible with v37; the material architecture difference
# is the v36 two-frame recurrent model and its heads.
spec = importlib.util.spec_from_file_location(
    "visual_model", v36_dir / "visual_model.py"
)
visual_model = importlib.util.module_from_spec(spec)
sys.modules["visual_model"] = visual_model
spec.loader.exec_module(visual_model)

# From this point onward ALL pipeline code is v37 code.
import robust_tracker

print("=== FAIR CONTROL: exact v37 protocol / v36 architecture only ===", flush=True)
print("pipeline module:", Path(robust_tracker.__file__).resolve(), flush=True)
print("model module:", v36_dir / "visual_model.py", flush=True)
print("train routes:", config.TRAIN_ROUTES, flush=True)
print("validation routes:", config.VALIDATION_ROUTES, flush=True)
print("eval routes:", config.EVAL_ROUTES, flush=True)
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
    "visual trainer: epochs=%s lr=%g wd=%g batch=%d"
    % (
        os.environ["VISUAL_EPOCHS"],
        float(config.VISUAL_LR),
        float(config.VISUAL_WEIGHT_DECAY),
        int(config.VISUAL_BATCH_SIZE),
    ),
    flush=True,
)
print(
    "temporal trainer: target_epochs=%s lr=%g wd=%g tbptt=%d grad_clip=%g"
    % (
        os.environ["TEMPORAL_EPOCHS"],
        float(config.TEMPORAL_LR),
        float(config.TEMPORAL_WEIGHT_DECAY),
        int(config.TBPTT_STEPS),
        float(config.GRAD_CLIP_NORM),
    ),
    flush=True,
)
print(
    "losses: measurement=%g next_step=%g velocity=%g acceleration=%g "
    "heading=%g turn_rate=%g variance=%g"
    % (
        float(config.LOSS_MEASUREMENT),
        float(config.LOSS_NEXT_STEP),
        float(config.LOSS_VELOCITY),
        float(config.LOSS_ACCELERATION),
        float(config.LOSS_HEADING),
        float(config.LOSS_TURN_RATE),
        float(config.LOSS_VARIANCE_NLL),
    ),
    flush=True,
)
print(
    "architecture-only differences: frame_count=2 rnn_numeric_dim=10 "
    "v36 pure learned heads",
    flush=True,
)

# Match v37/run_gpu0.sh: train visual heads from scratch, train temporal model,
# then evaluate. Do NOT pass --reuse-visual or --resume-temporal.
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
