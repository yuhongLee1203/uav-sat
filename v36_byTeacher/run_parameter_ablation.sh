#!/usr/bin/env bash
set -euo pipefail

# v36_byTeacher v8r1 one-variable-at-a-time spatial sensitivity study.
#
# Fixed baseline for every case unless the case explicitly changes one item:
#   MS1 = strict forward 3x6
#   MS2 = centered 6x6
#   MeanShift bandwidth = 8 m
#   MeanShift mode-merge radius = 2 m (FIXED; not part of this study)
#   MeanShift iterations = 3 (FIXED)
#   GRU / Kalman / visual weights = fixed FULL v8r1
#
# Experiments:
#   ms1:
#     forward 3x6 [baseline], 4x6, 5x6, 6x6, 7x6
#     Only MS1 forward depth changes. Lateral width stays exactly 6 candidates.
#
#   ms2:
#     5x5, 6x6 [baseline], 7x7
#     MS1 stays forward 3x6; only the second-stage centered grid changes.
#
#   meanshift:
#     bandwidth 4, 8 [baseline], 12, 16 m
#     MS1 stays 3x6 and MS2 stays 6x6; only MeanShift bandwidth changes.
#
# All cases are EVAL ONLY and reuse the same FULL v8r1 checkpoint.
# Outputs are isolated under:
#   output/<backbone>/method_ablation/<case>/...

GROUP="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS="${UAVSAT_CPU_THREADS:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

case "${GROUP}" in
  all|ms1|ms2|meanshift|search) ;;
  *)
    echo "usage: bash run_parameter_ablation.sh {all|ms1|ms2|meanshift|search}" >&2
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

echo "METHOD sensitivity: EVAL ONLY; CPU threads=${CPU_THREADS}" >&2

run_case() {
  local tag="$1"
  local ms1_rows="$2"
  local ms2_grid="$3"
  local bandwidth="$4"

  echo
  echo "================================================================================================"
  echo "METHOD SENSITIVITY CASE: ${tag}"
  echo "  mode                    = EVAL ONLY (same FULL v8r1 weights)"
  echo "  MS1 forward support     = ${ms1_rows}x6"
  echo "  MS2 centered grid       = ${ms2_grid}x${ms2_grid}"
  echo "  MeanShift bandwidth     = ${bandwidth} m"
  echo "  MeanShift merge radius  = 2.0 m (fixed)"
  echo "  MeanShift iterations    = 3 (fixed)"
  echo "================================================================================================"

  UAVSAT_GRU_ABLATION=full \
  UAVSAT_METHOD_ABLATION_TAG="${tag}" \
  UAVSAT_METHOD_MS1_ROWS="${ms1_rows}" \
  UAVSAT_METHOD_MS2_GRID="${ms2_grid}" \
  UAVSAT_MS_BANDWIDTH_M="${bandwidth}" \
  UAVSAT_METHOD_MS_MERGE_RADIUS_M="2.0" \
  UAVSAT_MS_ITERATIONS="3" \
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
ms1_rows = int(os.environ["UAVSAT_METHOD_MS1_ROWS"])
ms2_grid = int(os.environ["UAVSAT_METHOD_MS2_GRID"])
merge_radius = float(os.environ["UAVSAT_METHOD_MS_MERGE_RADIUS_M"])

if not tag or any(
    ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    for ch in tag
):
    raise SystemExit("ERROR: invalid method-sensitivity tag=%r" % tag)
if config.GRU_ABLATION != "full":
    raise SystemExit("ERROR: method sensitivity must keep the FULL GRU")
if ms1_rows not in {3, 4, 5, 6, 7}:
    raise SystemExit("ERROR: MS1 forward rows must be one of 3, 4, 5, 6, 7")
if ms2_grid not in {5, 6, 7}:
    raise SystemExit("ERROR: MS2 grid must be one of 5, 6, 7")
if merge_radius <= 0.0:
    raise SystemExit("ERROR: MeanShift merge radius must be positive")

# Reuse one fixed FULL v8r1 checkpoint for all sensitivity cases.
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

# Change only the requested spatial variable. Everything learned stays fixed.
config.MS1_FORWARD_ROWS = ms1_rows
config.MS1_FORWARD_COLS = 6
config.MS1_CANDIDATE_COUNT = ms1_rows * 6
config.MS2_GRID_SIZE = ms2_grid
config.MEANSHIFT_BANDWIDTH_M = float(os.environ["UAVSAT_MS_BANDWIDTH_M"])
config.MEANSHIFT_MODE_MERGE_RADIUS_M = merge_radius
config.MEANSHIFT_ITERATIONS = int(os.environ["UAVSAT_MS_ITERATIONS"])

case_root = shared_backbone_root / "method_ablation" / tag
case_checkpoint_dir = case_root / "checkpoints"
case_checkpoint_dir.mkdir(parents=True, exist_ok=True)
config.BACKBONE_OUTPUT_DIR = case_root
config.CHECKPOINT_DIR = case_checkpoint_dir
config.VISUAL_CHECKPOINT = shared_visual_checkpoint
config.FEATURE_CACHE_DIR = shared_feature_cache

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

# Import after runtime config and isolated output paths are ready.
import train_multirate_a as train


def _select_forward_rectangle(full_centers, heading_rad, rows, lateral_cols=6):
    """Select exactly rows x lateral_cols nearest forward lattice positions.

    For 3x6 this reproduces the original strict forward half of a 6x6 grid.
    For deeper supports, the temporary square source grid is expanded only so
    enough forward rows exist; the returned support is still exactly Nx6.
    """
    batch = int(full_centers.shape[0])
    headings = torch.as_tensor(
        heading_rad,
        dtype=full_centers.dtype,
        device=full_centers.device,
    ).reshape(-1)
    if headings.numel() == 1 and batch > 1:
        headings = headings.expand(batch)
    if headings.numel() != batch:
        raise ValueError("heading count must match center batch size")

    selected_rows = []
    for b in range(batch):
        centers = full_centers[b]
        relative = centers - centers.mean(dim=0, keepdim=True)
        cos_h = torch.cos(headings[b])
        sin_h = torch.sin(headings[b])
        use_x = bool((cos_h.abs() >= sin_h.abs()).item())
        if use_x:
            sign = 1.0 if float(cos_h.item()) >= 0.0 else -1.0
            longitudinal = relative[:, 0] * sign
            lateral = relative[:, 1]
        else:
            sign = 1.0 if float(sin_h.item()) >= 0.0 else -1.0
            longitudinal = relative[:, 1] * sign
            lateral = relative[:, 0]

        positive_idx = torch.nonzero(longitudinal > 0.0, as_tuple=False).flatten()
        if positive_idx.numel() < rows * lateral_cols:
            raise RuntimeError(
                f"not enough forward candidates: need {rows*lateral_cols}, "
                f"have {positive_idx.numel()}"
            )

        # Group equal regular-grid longitudinal levels at 1 mm precision.
        qlong = torch.round(longitudinal * 1000.0) / 1000.0
        levels = torch.unique(qlong[positive_idx])
        levels = torch.sort(levels).values
        if levels.numel() < rows:
            raise RuntimeError(
                f"regular-grid grouping found only {levels.numel()} forward rows; "
                f"need {rows}"
            )

        chosen = []
        for level in levels[:rows]:
            row_idx = torch.nonzero(
                (qlong - level).abs() <= 0.0005,
                as_tuple=False,
            ).flatten()
            if row_idx.numel() < lateral_cols:
                raise RuntimeError(
                    f"forward row has only {row_idx.numel()} lateral candidates; "
                    f"need {lateral_cols}"
                )
            nearest_lat = torch.topk(
                lateral[row_idx].abs(),
                k=lateral_cols,
                largest=False,
                sorted=False,
            ).indices
            row_idx = row_idx[nearest_lat]
            row_idx = row_idx[torch.argsort(lateral[row_idx])]
            chosen.append(row_idx)

        selected_rows.append(torch.cat(chosen, dim=0))

    return torch.stack(selected_rows, dim=0)


@torch.no_grad()
def _method_ms1_search(visual, uav_clip, prior_xy, heading_rad, device):
    """GPU6-only exact forward Nx6 MS1 support; default/GPU5 code untouched."""
    center = train.rt._tensor_xy(prior_xy, device)

    # An even GxG regular grid contains G/2 forward rows around its geometric
    # centre. Therefore G=2*N is the smallest source grid that can supply Nx6.
    source_grid = max(6, 2 * int(ms1_rows))
    full = visual.candidate_batch(
        uav_clip=uav_clip,
        center_xy=center,
        grid_size=source_grid,
    )

    selected = _select_forward_rectangle(
        full.centers,
        heading_rad,
        ms1_rows,
        lateral_cols=6,
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


train.rt.ms1_forward_search = _method_ms1_search

print("CPU THREAD LIMIT:", threads, flush=True)
print("SOURCE CHECKPOINT KIND:", source_kind, flush=True)
print("SOURCE FULL CHECKPOINT:", source_checkpoint, flush=True)
print("ISOLATED CASE ROOT:", case_root, flush=True)
print("GRU: FULL, weights unchanged", flush=True)
print("MS1 SUPPORT:", f"{ms1_rows}x6", flush=True)
print("MS1 CANDIDATES:", config.MS1_CANDIDATE_COUNT, flush=True)
print("MS2 GRID:", f"{config.MS2_GRID_SIZE}x{config.MS2_GRID_SIZE}", flush=True)
print("MS BANDWIDTH:", config.MEANSHIFT_BANDWIDTH_M, flush=True)
print("MS MERGE RADIUS:", config.MEANSHIFT_MODE_MERGE_RADIUS_M, "(fixed)", flush=True)
print("MS ITERATIONS:", config.MEANSHIFT_ITERATIONS, "(fixed)", flush=True)
print("MODE: eval only", flush=True)

sys.argv = ["train_multirate_a.py", "--mode", "eval"]
train.main()
PY
}

run_ms1_group() {
  # ONE variable changes: MS1 forward depth. Everything else is baseline.
  run_case "ms1_forward_3x6_baseline" 3 6 8.0
  run_case "ms1_forward_4x6"          4 6 8.0
  run_case "ms1_forward_5x6"          5 6 8.0
  run_case "ms1_forward_6x6"          6 6 8.0
  run_case "ms1_forward_7x6"          7 6 8.0
}

run_ms2_group() {
  # ONE variable changes: MS2 centered search-window size. MS1 remains 3x6.
  run_case "ms2_grid_5x5"             3 5 8.0
  run_case "ms2_grid_6x6_baseline"    3 6 8.0
  run_case "ms2_grid_7x7"             3 7 8.0
}

run_meanshift_group() {
  # ONE variable changes: MeanShift spatial bandwidth. MS1=3x6, MS2=6x6.
  run_case "ms_bandwidth_4m"           3 6 4.0
  run_case "ms_bandwidth_8m_baseline"  3 6 8.0
  run_case "ms_bandwidth_12m"          3 6 12.0
  run_case "ms_bandwidth_16m"          3 6 16.0
}

case "${GROUP}" in
  ms1) run_ms1_group ;;
  ms2) run_ms2_group ;;
  meanshift) run_meanshift_group ;;
  search)
    run_ms1_group
    run_ms2_group
    ;;
  all)
    run_ms1_group
    run_ms2_group
    run_meanshift_group
    ;;
esac

echo
echo "All requested one-variable-at-a-time sensitivity cases completed."
