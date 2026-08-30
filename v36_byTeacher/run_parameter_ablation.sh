#!/usr/bin/env bash
set -euo pipefail

# v36_byTeacher v8r1 one-variable-at-a-time spatial sensitivity + latency study.
#
# Fixed baseline for every case unless the case explicitly changes one item:
#   MS1 = strict forward 3x6
#   MS2 = centered 6x6
#   MeanShift bandwidth = 8 m
#   MeanShift mode-merge radius = 2 m (FIXED; not part of this study)
#   MeanShift iterations = 3 (FIXED; not part of this study)
#   GRU / Kalman / visual weights = fixed FULL v8r1
#
# Experiments kept for the paper:
#   ms1       : forward 3x6 [baseline], 4x6, 5x6, 6x6, 7x6
#   ms2       : centered 5x5, 6x6 [baseline], 7x7
#   meanshift : bandwidth 4, 8 [baseline], 12, 16 m
#
# Latency is measured on exactly the same B/C evaluation frames used for
# accuracy. We report synchronized wall-clock latency for the major stages and
# separate reduction latency for:
#   (a) MeanShift final averaging over converged active modes; and
#   (b) direct weighted averaging over every MS2 candidate.
# This makes the accuracy/latency trade-off explicit without assuming in
# advance that one aggregation strategy is faster.
#
# Deliberately NOT tested here:
#   - MeanShift merge-radius sensitivity
#   - MeanShift iteration-count sensitivity
#   - old mixed multi-variable parameter table
#
# All cases are EVAL ONLY and reuse the same FULL v8r1 checkpoint.

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

echo "METHOD sensitivity + latency: EVAL ONLY; CPU threads=${CPU_THREADS}" >&2

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
  echo "  latency                 = synchronized per-frame B/C averages"
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
import csv
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
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

import train_multirate_a as train


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _wall_start():
    _sync()
    return time.perf_counter()


def _wall_elapsed_ms(start):
    _sync()
    return (time.perf_counter() - start) * 1000.0


def _reduction_time_ms(fn):
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        value = fn()
        end_event.record()
        end_event.synchronize()
        return value, float(start_event.elapsed_time(end_event))
    start = time.perf_counter()
    value = fn()
    return value, float((time.perf_counter() - start) * 1000.0)


ROUTE_STATS = defaultdict(lambda: defaultdict(list))


def _append(route_name, key, value):
    ROUTE_STATS[str(route_name)][str(key)].append(float(value))


def _mean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)) if values.size else float("nan")


def _p90(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.percentile(values, 90.0)) if values.size else float("nan")


def _lsr15(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values <= 15.0) * 100.0) if values.size else float("nan")


def _select_forward_rectangle(full_centers, heading_rad, rows, lateral_cols=6):
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

        qlong = torch.round(longitudinal * 1000.0) / 1000.0
        levels = torch.sort(torch.unique(qlong[positive_idx])).values
        if levels.numel() < rows:
            raise RuntimeError(
                f"regular-grid grouping found only {levels.numel()} forward rows; need {rows}"
            )

        chosen = []
        for level in levels[:rows]:
            row_idx = torch.nonzero(
                (qlong - level).abs() <= 0.0005,
                as_tuple=False,
            ).flatten()
            if row_idx.numel() < lateral_cols:
                raise RuntimeError(
                    f"forward row has only {row_idx.numel()} lateral candidates; need {lateral_cols}"
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


def _raw_candidate_batch(visual, uav_clip, center_xy, grid_size):
    indices = train.rt.regular_grid_indices(
        visual.gallery["xy"],
        visual.gallery["pixel"],
        visual.pixel_index,
        center_xy,
        int(grid_size),
        config.SAT_STRIDE,
        visual.device,
    )
    centers = visual.gallery["xy"][indices]
    satellite_clip = visual.gallery["clip_feat"][indices]
    z_uav = visual.model.encode_uav_from_clip(uav_clip)
    z_sat = visual.model.encode_sat_from_clip(
        satellite_clip.reshape(-1, satellite_clip.shape[-1]),
        centers.reshape(-1, 2),
    ).reshape(centers.shape[0], centers.shape[1], -1)
    raw_logits = visual.model.logit_scale.exp().clamp(max=100.0) * (
        z_uav[:, None] * z_sat
    ).sum(dim=2)
    raw_prob = torch.softmax(
        raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1
    )
    raw_index = raw_logits.argmax(dim=1)
    raw_top1_xy = centers[
        torch.arange(centers.shape[0], device=visual.device), raw_index
    ]
    zeros_xy = torch.zeros(centers.shape[0], 2, device=centers.device)
    zeros_scalar = torch.zeros(centers.shape[0], device=centers.device)
    zeros_count = torch.zeros(
        centers.shape[0], dtype=torch.long, device=centers.device
    )
    return train.rt.CandidateBatch(
        indices=indices,
        centers=centers,
        z_uav=z_uav,
        z_sat=z_sat,
        raw_logits=raw_logits,
        raw_prob=raw_prob,
        raw_top1_xy=raw_top1_xy,
        softms_xy=zeros_xy,
        softms_support=zeros_scalar,
        softms_mode_count=zeros_count,
    )


def _run_meanshift_with_timing(logits, centers):
    start = _wall_start()
    soft_xy, soft_support, modes, density, mode_weights, compact = train.rt.soft_mean_shift(
        logits,
        centers,
        config.MEANSHIFT_SCORE_TAU,
        config.MEANSHIFT_BANDWIDTH_M,
        config.MEANSHIFT_ITERATIONS,
        config.MEANSHIFT_MODE_BETA,
    )
    total_ms = _wall_elapsed_ms(start)

    final_ms = []
    active_counts = []
    for b in range(int(mode_weights.shape[0])):
        mask = mode_weights[b] > 0
        active_weights = mode_weights[b, mask]
        active_modes = modes[b, mask]
        active_counts.append(float(active_weights.numel()))
        _, reduction_ms = _reduction_time_ms(
            lambda aw=active_weights, am=active_modes: (aw[:, None] * am).sum(dim=0)
        )
        final_ms.append(float(reduction_ms))

    return (
        soft_xy,
        soft_support,
        mode_weights,
        {
            "total_ms": float(total_ms),
            "final_active_average_ms": float(np.mean(final_ms)),
            "active_modes": float(np.mean(active_counts)),
        },
    )


@torch.no_grad()
def _method_ms1_search(visual, uav_clip, prior_xy, heading_rad, device):
    total_start = _wall_start()
    center = train.rt._tensor_xy(prior_xy, device)
    source_grid = max(6, 2 * int(ms1_rows))

    full_indices = train.rt.regular_grid_indices(
        visual.gallery["xy"],
        visual.gallery["pixel"],
        visual.pixel_index,
        center,
        source_grid,
        config.SAT_STRIDE,
        visual.device,
    )
    full_centers = visual.gallery["xy"][full_indices]
    selected = _select_forward_rectangle(
        full_centers,
        heading_rad,
        ms1_rows,
        lateral_cols=6,
    )
    batch_idx = torch.arange(
        full_centers.shape[0], device=full_centers.device
    )[:, None]
    indices = full_indices[batch_idx, selected]
    centers = visual.gallery["xy"][indices]
    satellite_clip = visual.gallery["clip_feat"][indices]
    z_uav = visual.model.encode_uav_from_clip(uav_clip)
    z_sat = visual.model.encode_sat_from_clip(
        satellite_clip.reshape(-1, satellite_clip.shape[-1]),
        centers.reshape(-1, 2),
    ).reshape(centers.shape[0], centers.shape[1], -1)
    raw_logits = visual.model.logit_scale.exp().clamp(max=100.0) * (
        z_uav[:, None] * z_sat
    ).sum(dim=2)
    raw_prob = torch.softmax(
        raw_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1
    )
    top1_idx = raw_logits.argmax(dim=1)
    raw_top1_xy = centers[
        torch.arange(centers.shape[0], device=centers.device), top1_idx
    ]

    softms_xy, softms_support, mode_weights, ms_timing = _run_meanshift_with_timing(
        raw_logits, centers
    )
    batch = train.rt.CandidateBatch(
        indices=indices,
        centers=centers,
        z_uav=z_uav,
        z_sat=z_sat,
        raw_logits=raw_logits,
        raw_prob=raw_prob,
        raw_top1_xy=raw_top1_xy,
        softms_xy=softms_xy,
        softms_support=softms_support,
        softms_mode_count=(mode_weights > 0).sum(dim=1),
    )
    batch.method_total_latency_ms = _wall_elapsed_ms(total_start)
    batch.meanshift_total_latency_ms = ms_timing["total_ms"]
    batch.final_active_average_latency_ms = ms_timing["final_active_average_ms"]
    batch.active_mode_count = ms_timing["active_modes"]
    return batch


@torch.no_grad()
def _timed_ms2_center_search(visual, uav_clip, kalman_xy, device):
    start = _wall_start()
    batch = _raw_candidate_batch(
        visual=visual,
        uav_clip=uav_clip,
        center_xy=train.rt._tensor_xy(kalman_xy, device),
        grid_size=int(config.MS2_GRID_SIZE),
    )
    batch.center_search_latency_ms = _wall_elapsed_ms(start)
    return batch


@torch.no_grad()
def _timed_kalman_conditioned_ms2(visual, uav_clip, kalman_xy, device):
    batch = _timed_ms2_center_search(
        visual=visual,
        uav_clip=uav_clip,
        kalman_xy=kalman_xy,
        device=device,
    )
    kalman_center = train.rt._tensor_xy(kalman_xy, device)
    sigma = max(float(config.MS2_KALMAN_PRIOR_SIGMA_M), 1e-6)
    prior_weight = float(config.MS2_KALMAN_PRIOR_WEIGHT)

    displacement = batch.centers - kalman_center[:, None, :]
    distance_squared = displacement.square().sum(dim=2)
    spatial_log_prior = (
        -0.5 * prior_weight * distance_squared / (sigma * sigma)
    )
    posterior_logits = batch.raw_logits + spatial_log_prior
    posterior_prob = torch.softmax(
        posterior_logits / float(config.MEANSHIFT_SCORE_TAU), dim=1
    )
    posterior_index = posterior_logits.argmax(dim=1)
    posterior_top1_xy = batch.centers[
        torch.arange(batch.centers.shape[0], device=device), posterior_index
    ]

    weighted_all_xy, weighted_all_ms = _reduction_time_ms(
        lambda: (posterior_prob[:, :, None] * batch.centers).sum(dim=1)
    )

    softms_xy, softms_support, mode_weights, ms_timing = _run_meanshift_with_timing(
        posterior_logits, batch.centers
    )

    result = train.rt.CandidateBatch(
        indices=batch.indices,
        centers=batch.centers,
        z_uav=batch.z_uav,
        z_sat=batch.z_sat,
        raw_logits=posterior_logits,
        raw_prob=posterior_prob,
        raw_top1_xy=posterior_top1_xy,
        softms_xy=softms_xy,
        softms_support=softms_support,
        softms_mode_count=(mode_weights > 0).sum(dim=1),
    )
    result.center_search_latency_ms = float(batch.center_search_latency_ms)
    result.meanshift_total_latency_ms = float(ms_timing["total_ms"])
    result.final_active_average_latency_ms = float(
        ms_timing["final_active_average_ms"]
    )
    result.all_candidate_average_latency_ms = float(weighted_all_ms)
    result.active_mode_count = float(ms_timing["active_modes"])
    result.weighted_all_xy = weighted_all_xy
    return result


train.rt.ms1_forward_search = _method_ms1_search
train._kalman_conditioned_ms2 = _timed_kalman_conditioned_ms2
_original_forward_frame = train.rt._forward_frame


@torch.no_grad()
def _timed_forward_frame(model, visual, uav_clip, state, device):
    route_name = str(state.get("reference_route_name", "unknown"))
    reference_index = int(state.get("reference_index", 0))
    reference_xy = None
    if route_name in train._REFERENCE_PRIORS:
        sequence = train._REFERENCE_PRIORS[route_name]
        if 0 <= reference_index < len(sequence):
            reference_xy = np.asarray(sequence[reference_index], dtype=np.float64)

    frame_start = _wall_start()
    frame = _original_forward_frame(model, visual, uav_clip, state, device)
    frame_latency_ms = _wall_elapsed_ms(frame_start)

    ms1 = frame["ms1"]
    ms2 = frame["ms2"]
    _append(route_name, "e2e_latency_ms", frame_latency_ms)
    _append(route_name, "ms1_latency_ms", ms1.method_total_latency_ms)
    _append(route_name, "ms1_meanshift_latency_ms", ms1.meanshift_total_latency_ms)
    _append(route_name, "ms2_center_search_latency_ms", ms2.center_search_latency_ms)
    _append(route_name, "ms2_meanshift_latency_ms", ms2.meanshift_total_latency_ms)
    _append(
        route_name,
        "ms2_final_active_average_latency_ms",
        ms2.final_active_average_latency_ms,
    )
    _append(
        route_name,
        "ms2_all_candidate_average_latency_ms",
        ms2.all_candidate_average_latency_ms,
    )
    _append(route_name, "ms2_active_modes", ms2.active_mode_count)

    if reference_xy is not None:
        ms_xy = ms2.softms_xy[0].detach().cpu().numpy().astype(np.float64)
        all_xy = ms2.weighted_all_xy[0].detach().cpu().numpy().astype(np.float64)
        _append(route_name, "meanshift_error_m", np.linalg.norm(ms_xy - reference_xy))
        _append(route_name, "weighted_all_error_m", np.linalg.norm(all_xy - reference_xy))

    return frame


train.rt._forward_frame = _timed_forward_frame

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
print("MODE: eval only + synchronized latency", flush=True)

sys.argv = ["train_multirate_a.py", "--mode", "eval"]
train.main()

summary_rows = []
for route_name in ("route_C", "route_B"):
    stats = ROUTE_STATS.get(route_name, {})
    if not stats:
        continue
    ms_errors = stats.get("meanshift_error_m", [])
    all_errors = stats.get("weighted_all_error_m", [])
    row = {
        "Case": tag,
        "Route": route_name,
        "MS1_Support": f"{ms1_rows}x6",
        "MS2_Grid": f"{ms2_grid}x{ms2_grid}",
        "MeanShift_Bandwidth_m": float(config.MEANSHIFT_BANDWIDTH_M),
        "MeanShift_MLE_m": _mean(ms_errors),
        "MeanShift_P90_m": _p90(ms_errors),
        "MeanShift_LSR15_pct": _lsr15(ms_errors),
        "WeightedAll_MLE_m": _mean(all_errors),
        "WeightedAll_P90_m": _p90(all_errors),
        "WeightedAll_LSR15_pct": _lsr15(all_errors),
        "E2E_Latency_ms": _mean(stats.get("e2e_latency_ms", [])),
        "MS1_Latency_ms": _mean(stats.get("ms1_latency_ms", [])),
        "MS1_MeanShift_Latency_ms": _mean(
            stats.get("ms1_meanshift_latency_ms", [])
        ),
        "MS2_CenterSearch_Latency_ms": _mean(
            stats.get("ms2_center_search_latency_ms", [])
        ),
        "MS2_MeanShift_Total_Latency_ms": _mean(
            stats.get("ms2_meanshift_latency_ms", [])
        ),
        "MS2_FinalActiveAverage_Latency_ms": _mean(
            stats.get("ms2_final_active_average_latency_ms", [])
        ),
        "MS2_AllCandidateAverage_Latency_ms": _mean(
            stats.get("ms2_all_candidate_average_latency_ms", [])
        ),
        "MS2_ActiveModes_Avg": _mean(stats.get("ms2_active_modes", [])),
    }
    summary_rows.append(row)

if len(summary_rows) == 2:
    avg_row = {"Case": tag, "Route": "B+C average"}
    text_keys = {"Case", "Route", "MS1_Support", "MS2_Grid"}
    for key in summary_rows[0].keys():
        if key in text_keys:
            continue
        avg_row[key] = float(
            (float(summary_rows[0][key]) + float(summary_rows[1][key])) / 2.0
        )
    avg_row["MS1_Support"] = f"{ms1_rows}x6"
    avg_row["MS2_Grid"] = f"{ms2_grid}x{ms2_grid}"
    summary_rows.append(avg_row)

summary_path = case_root / "accuracy_latency_summary.csv"
if summary_rows:
    fieldnames = list(summary_rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 96)
    print("ACCURACY + LATENCY SUMMARY")
    for row in summary_rows:
        print(
            "%s: MS MLE=%.3fm P90=%.3fm LSR15=%.2f%% | "
            "WeightedAll MLE=%.3fm P90=%.3fm LSR15=%.2f%% | "
            "E2E=%.3fms MS1=%.3fms CenterMS=%.3fms "
            "MS2MeanShift=%.3fms FinalActiveAvg=%.6fms "
            "AllCandidateAvg=%.6fms ActiveModes=%.2f"
            % (
                row["Route"],
                row["MeanShift_MLE_m"],
                row["MeanShift_P90_m"],
                row["MeanShift_LSR15_pct"],
                row["WeightedAll_MLE_m"],
                row["WeightedAll_P90_m"],
                row["WeightedAll_LSR15_pct"],
                row["E2E_Latency_ms"],
                row["MS1_Latency_ms"],
                row["MS2_CenterSearch_Latency_ms"],
                row["MS2_MeanShift_Total_Latency_ms"],
                row["MS2_FinalActiveAverage_Latency_ms"],
                row["MS2_AllCandidateAverage_Latency_ms"],
                row["MS2_ActiveModes_Avg"],
            )
        )
    print("CSV:", summary_path)
    print("=" * 96)
PY
}

run_ms1_group() {
  run_case "ms1_forward_3x6_baseline" 3 6 8.0
  run_case "ms1_forward_4x6"          4 6 8.0
  run_case "ms1_forward_5x6"          5 6 8.0
  run_case "ms1_forward_6x6"          6 6 8.0
  run_case "ms1_forward_7x6"          7 6 8.0
}

run_ms2_group() {
  run_case "ms2_grid_5x5"             3 5 8.0
  run_case "ms2_grid_6x6_baseline"    3 6 8.0
  run_case "ms2_grid_7x7"             3 7 8.0
}

run_meanshift_group() {
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
echo "All requested one-variable-at-a-time accuracy/latency cases completed."
