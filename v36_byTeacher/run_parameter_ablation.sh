#!/usr/bin/env bash
set -euo pipefail

# v36_byTeacher v8r1 METHOD-LEVEL spatial / MeanShift sensitivity study.
#
# This is intentionally different from GPU5:
#   GPU5 = learned GRU-input component ablation (retrain each removed branch).
#   GPU6 = fixed algorithmic design choices around local search / MeanShift.
#
# These GPU6 knobs are NOT learned by the network. They define which satellite
# candidates are exposed to the visual matcher and how MeanShift aggregates
# nearby visual modes. Therefore the already-trained FULL v8r1 checkpoint can
# be reused and each case is EVAL ONLY.
#
# Normal full baseline (do not rerun here):
#   MS1 = strict forward 3x6 selected from 6x6
#   MS2 = centered 6x6
#   MeanShift bandwidth = 8 m
#   MeanShift mode-merge radius = 2 m
#   MeanShift iterations = 3
#
# Default GPU6 cases (GROUP=all):
#   Search geometry:
#     - MS1 full 6x6 (remove hard forward restriction)
#     - MS1 forward 1x6
#     - MS1 forward 2x6
#       [baseline = forward 3x6]
#     - MS2 4x4 / 8x8
#       [baseline = 6x6]
#   MeanShift spatial scale:
#     - bandwidth 4 m / 12 m
#       [baseline = 8 m]
#     - mode-merge radius 1 m / 4 m
#       [baseline = 2 m]
#
# Optional lower-priority convergence sensitivity:
#   bash run_parameter_ablation.sh iterations
#     - MeanShift iterations 1 / 5 [baseline = 3]
#
# Outputs are isolated under:
#   output/<backbone>/method_ablation/<case>/...

GROUP="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS="${UAVSAT_CPU_THREADS:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

case "${GROUP}" in
  all|search|meanshift|iterations) ;;
  *)
    echo "usage: bash run_parameter_ablation.sh {all|search|meanshift|iterations}" >&2
    exit 2
    ;;
esac

export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found." >&2
  exit 2
fi

"${PYTHON_BIN}" -m py_compile \
  config.py data.py visual_model.py visual_localizer.py \
  robust_tracker_base.py robust_tracker.py train_multirate_a.py

echo "METHOD ablation: EVAL ONLY; CPU threads=${CPU_THREADS}" >&2

run_case() {
  local tag="$1"
  local ms1_mode="$2"
  local ms1_rows="$3"
  local ms2_grid="$4"
  local bandwidth="$5"
  local merge_radius="$6"
  local iterations="$7"

  echo
  echo "================================================================================================"
  echo "METHOD ABLATION CASE: ${tag}"
  echo "  mode                    = EVAL ONLY (reuse FULL v8r1 checkpoint)"
  echo "  MS1 search mode         = ${ms1_mode}"
  echo "  MS1 forward rows        = ${ms1_rows} (6 candidates per row when forward)"
  echo "  MS2 grid                = ${ms2_grid}x${ms2_grid}"
  echo "  MeanShift bandwidth     = ${bandwidth} m"
  echo "  MeanShift merge radius  = ${merge_radius} m"
  echo "  MeanShift iterations    = ${iterations}"
  echo "================================================================================================"

  UAVSAT_GRU_ABLATION=full \
  UAVSAT_METHOD_ABLATION_TAG="${tag}" \
  UAVSAT_METHOD_MS1_MODE="${ms1_mode}" \
  UAVSAT_METHOD_MS1_ROWS="${ms1_rows}" \
  UAVSAT_METHOD_MS2_GRID="${ms2_grid}" \
  UAVSAT_MS_BANDWIDTH_M="${bandwidth}" \
  UAVSAT_METHOD_MS_MERGE_RADIUS_M="${merge_radius}" \
  UAVSAT_MS_ITERATIONS="${iterations}" \
  UAVSAT_CPU_THREADS="${CPU_THREADS}" \
  "${PYTHON_BIN}" - <<'PY'
import os
import shutil
import sys
from pathlib import Path

import torch

import config

threads = max(1, int(os.environ.get("UAVSAT_CPU_THREADS", "1")))
torch.set_num_threads(threads)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

tag = os.environ["UAVSAT_METHOD_ABLATION_TAG"].strip()
ms1_mode = os.environ["UAVSAT_METHOD_MS1_MODE"].strip().lower()
ms1_rows = int(os.environ["UAVSAT_METHOD_MS1_ROWS"])
ms2_grid = int(os.environ["UAVSAT_METHOD_MS2_GRID"])
merge_radius = float(os.environ["UAVSAT_METHOD_MS_MERGE_RADIUS_M"])

if not tag or any(
    ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    for ch in tag
):
    raise SystemExit("ERROR: invalid method-ablation tag=%r" % tag)
if config.GRU_ABLATION != "full":
    raise SystemExit("ERROR: GPU6 method study must keep the FULL GRU")
if ms1_mode not in {"forward", "full"}:
    raise SystemExit("ERROR: MS1 mode must be forward or full")
if ms1_rows not in {1, 2, 3}:
    raise SystemExit("ERROR: MS1 forward rows must be 1, 2, or 3")
if ms2_grid not in {4, 6, 8}:
    raise SystemExit("ERROR: MS2 grid must be 4, 6, or 8")
if merge_radius <= 0.0:
    raise SystemExit("ERROR: MeanShift merge radius must be positive")

# -------------------------------------------------------------------------
# Reuse FULL v8r1 weights. Prefer the validation-selected checkpoint; if the
# full run is still in progress, use its latest checkpoint for preliminary
# sensitivity results.
# -------------------------------------------------------------------------
shared_backbone_root = Path(config.BACKBONE_OUTPUT_DIR)
shared_checkpoint_dir = Path(config.CHECKPOINT_DIR)
shared_visual_checkpoint = Path(config.VISUAL_CHECKPOINT)
shared_feature_cache = Path(config.FEATURE_CACHE_DIR)

best_checkpoint = (
    shared_checkpoint_dir
    / f"reference_prior_compact_gru_A_native_v8r1_full_{config.BACKBONE_KEY}.pt"
)
latest_checkpoint = (
    shared_checkpoint_dir
    / f"reference_prior_compact_gru_A_native_v8r1_full_{config.BACKBONE_KEY}_latest.pt"
)
if best_checkpoint.exists():
    source_checkpoint = best_checkpoint
    source_kind = "best"
elif latest_checkpoint.exists():
    source_checkpoint = latest_checkpoint
    source_kind = "latest/in-progress"
else:
    raise SystemExit(
        "ERROR: FULL v8r1 checkpoint not found. Checked:\n  %s\n  %s"
        % (best_checkpoint, latest_checkpoint)
    )

# -------------------------------------------------------------------------
# Set ONLY method-level inference geometry / MeanShift knobs.
# Keep network architecture, learned weights, Kalman parameters, MS2 Gaussian
# prior, visual model, and all training hyperparameters at the full baseline.
# -------------------------------------------------------------------------
config.MS1_BASE_GRID_SIZE = 6
config.MS1_FORWARD_ROWS = ms1_rows
config.MS1_FORWARD_COLS = 6
config.MS1_CANDIDATE_COUNT = ms1_rows * 6 if ms1_mode == "forward" else 36
config.MS2_GRID_SIZE = ms2_grid
config.MEANSHIFT_BANDWIDTH_M = float(os.environ["UAVSAT_MS_BANDWIDTH_M"])
config.MEANSHIFT_MODE_MERGE_RADIUS_M = merge_radius
config.MEANSHIFT_ITERATIONS = int(os.environ["UAVSAT_MS_ITERATIONS"])

# Isolate outputs/checkpoints from GPU0/GPU5 and from every other GPU6 case.
case_root = shared_backbone_root / "method_ablation" / tag
case_checkpoint_dir = case_root / "checkpoints"
case_checkpoint_dir.mkdir(parents=True, exist_ok=True)
config.BACKBONE_OUTPUT_DIR = case_root
config.CHECKPOINT_DIR = case_checkpoint_dir
config.VISUAL_CHECKPOINT = shared_visual_checkpoint
config.FEATURE_CACHE_DIR = shared_feature_cache

# train_multirate_a.py expects the full-v8r1 best checkpoint name in its current
# checkpoint directory when --mode eval is used. Copy the selected shared source
# to that isolated expected location; no training is performed.
destination_checkpoint = (
    case_checkpoint_dir
    / f"reference_prior_compact_gru_A_native_v8r1_full_{config.BACKBONE_KEY}.pt"
)
if (
    not destination_checkpoint.exists()
    or destination_checkpoint.stat().st_size != source_checkpoint.stat().st_size
    or destination_checkpoint.stat().st_mtime_ns < source_checkpoint.stat().st_mtime_ns
):
    shutil.copy2(source_checkpoint, destination_checkpoint)

# Import only after all runtime config values and isolated paths are ready.
import train_multirate_a as train


def _select_nearest_forward_rows(full_centers, heading_rad, rows):
    """Return nearest positive forward rows from a regular 6x6 lattice."""
    batch = int(full_centers.shape[0])
    keep = int(rows) * 6
    geometric_center = full_centers.mean(dim=1, keepdim=True)
    relative = full_centers - geometric_center

    headings = torch.as_tensor(
        heading_rad,
        dtype=full_centers.dtype,
        device=full_centers.device,
    ).reshape(-1)
    if headings.numel() == 1 and batch > 1:
        headings = headings.expand(batch)
    if headings.numel() != batch:
        raise ValueError("heading count must match center batch size")

    cos_h = torch.cos(headings)
    sin_h = torch.sin(headings)
    use_x = cos_h.abs() >= sin_h.abs()
    sign = torch.where(
        use_x,
        torch.where(cos_h >= 0, torch.ones_like(cos_h), -torch.ones_like(cos_h)),
        torch.where(sin_h >= 0, torch.ones_like(sin_h), -torch.ones_like(sin_h)),
    )
    longitudinal = torch.where(
        use_x[:, None], relative[:, :, 0], relative[:, :, 1]
    ) * sign[:, None]
    lateral = torch.where(
        use_x[:, None], relative[:, :, 1], relative[:, :, 0]
    )

    forward_mask = longitudinal > 0.0
    if not bool(torch.all(forward_mask.sum(dim=1) >= keep)):
        raise RuntimeError("not enough positive forward candidates")

    huge = torch.full_like(longitudinal, 1e9)
    nearest_cost = torch.where(forward_mask, longitudinal, huge)
    selected = torch.topk(
        nearest_cost,
        k=keep,
        dim=1,
        largest=False,
        sorted=False,
    ).indices

    selected_long = torch.gather(longitudinal, 1, selected)
    selected_lat = torch.gather(lateral, 1, selected)
    order = torch.argsort(selected_long * 1000.0 + selected_lat, dim=1)
    return torch.gather(selected, 1, order)


@torch.no_grad()
def _method_ms1_search(visual, uav_clip, prior_xy, heading_rad, device):
    """GPU6-only MS1 geometry variant; GPU5/default code is untouched."""
    center = train.rt._tensor_xy(prior_xy, device)
    full = visual.candidate_batch(
        uav_clip=uav_clip,
        center_xy=center,
        grid_size=6,
    )

    if ms1_mode == "full":
        return full

    selected = _select_nearest_forward_rows(
        full.centers,
        heading_rad,
        ms1_rows,
    )
    batch_idx = torch.arange(
        full.centers.shape[0], device=full.centers.device
    )[:, None]
    indices = full.indices[batch_idx, selected]
    centers = full.centers[batch_idx, selected]
    z_sat = full.z_sat[batch_idx, selected]
    raw_logits = full.raw_logits[batch_idx, selected]
    raw_prob = torch.softmax(
        raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1
    )
    top1_idx = raw_logits.argmax(dim=1)
    raw_top1_xy = centers[
        torch.arange(centers.shape[0], device=centers.device), top1_idx
    ]

    softms_xy, softms_support, _, _, mode_weights, _ = train.rt.soft_mean_shift(
        raw_logits,
        centers,
        config.MEANSHIFT_SCORE_TAU,
        config.MEANSHIFT_BANDWIDTH_M,
        config.MEANSHIFT_ITERATIONS,
        config.MEANSHIFT_MODE_BETA,
    )
    return train.rt.CandidateBatch(
        indices=indices,
        centers=centers,
        z_uav=full.z_uav,
        z_sat=z_sat,
        raw_logits=raw_logits,
        raw_prob=raw_prob,
        raw_top1_xy=raw_top1_xy,
        softms_xy=softms_xy,
        softms_support=softms_support,
        softms_mode_count=(mode_weights > 0).sum(dim=1),
    )


# Override only inside this GPU6 Python process.
train.rt.ms1_forward_search = _method_ms1_search

print("CPU THREAD LIMIT:", threads, flush=True)
print("SOURCE CHECKPOINT KIND:", source_kind, flush=True)
print("SOURCE FULL CHECKPOINT:", source_checkpoint, flush=True)
print("ISOLATED CASE ROOT:", case_root, flush=True)
print("GRU: FULL, weights unchanged", flush=True)
print("MS1 MODE:", ms1_mode, flush=True)
print("MS1 CANDIDATES:", config.MS1_CANDIDATE_COUNT, flush=True)
print("MS2 GRID:", f"{config.MS2_GRID_SIZE}x{config.MS2_GRID_SIZE}", flush=True)
print("MS BANDWIDTH:", config.MEANSHIFT_BANDWIDTH_M, flush=True)
print("MS MERGE RADIUS:", config.MEANSHIFT_MODE_MERGE_RADIUS_M, flush=True)
print("MS ITERATIONS:", config.MEANSHIFT_ITERATIONS, flush=True)
print("MODE: eval only", flush=True)

sys.argv = ["train_multirate_a.py", "--mode", "eval"]
train.main()
PY
}

run_search_group() {
  # Structural check: is the hard forward restriction itself useful?
  run_case "ms1_full_6x6"       full    3 6 8.0 2.0 3

  # Forward-depth sensitivity. Baseline is forward 3x6.
  run_case "ms1_forward_1x6"    forward 1 6 8.0 2.0 3
  run_case "ms1_forward_2x6"    forward 2 6 8.0 2.0 3

  # Second-stage local refinement area. Baseline is 6x6.
  run_case "ms2_grid_4x4"       forward 3 4 8.0 2.0 3
  run_case "ms2_grid_8x8"       forward 3 8 8.0 2.0 3
}

run_meanshift_group() {
  # Spatial kernel scale. Baseline bandwidth is 8 m.
  run_case "ms_bandwidth_4m"     forward 3 6 4.0 2.0 3
  run_case "ms_bandwidth_12m"    forward 3 6 12.0 2.0 3

  # Basin-consolidation distance. Baseline merge radius is 2 m.
  run_case "ms_merge_radius_1m"  forward 3 6 8.0 1.0 3
  run_case "ms_merge_radius_4m"  forward 3 6 8.0 4.0 3
}

run_iterations_group() {
  # Lower-priority convergence sensitivity. Baseline is 3 iterations.
  run_case "ms_iterations_1"     forward 3 6 8.0 2.0 1
  run_case "ms_iterations_5"     forward 3 6 8.0 2.0 5
}

case "${GROUP}" in
  search) run_search_group ;;
  meanshift) run_meanshift_group ;;
  iterations) run_iterations_group ;;
  all)
    run_search_group
    run_meanshift_group
    ;;
esac

echo
echo "All requested METHOD-level ablations completed."
