#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

MODE="${1:-all}"
BACKBONE="${UAVSAT_BACKBONE:-mobilenet_v3_small}"
JITTER_M="${JITTER_M:-8}"
VISUAL_EPOCHS_RUN="${VISUAL_EPOCHS_RUN:-30}"
TEMPORAL_EPOCHS_RUN="${TEMPORAL_EPOCHS_RUN:-60}"
PATIENCE_RUN="${PATIENCE_RUN:-15}"
FORCE_VISUAL="${FORCE_VISUAL:-0}"
TIMING_GPU="${TIMING_GPU:-0}"

BACKBONE_ROOT="$HERE/output/$BACKBONE"
LOG_DIR="$BACKBONE_ROOT/logs"
VISUAL_CKPT="$BACKBONE_ROOT/checkpoints/visual_retrieval_A_only_${BACKBONE}.pt"
mkdir -p "$BACKBONE_ROOT/checkpoints" "$BACKBONE_ROOT/feature_cache" "$LOG_DIR"

export BACKBONE JITTER_M VISUAL_EPOCHS_RUN TEMPORAL_EPOCHS_RUN PATIENCE_RUN

train_visual() {
  if [[ "$FORCE_VISUAL" != "1" && -f "$VISUAL_CKPT" ]]; then
    echo "[visual] reuse $VISUAL_CKPT"
    return
  fi

  echo "[visual] GPU0: train Route-A-only $BACKBONE visual model"
  CUDA_VISIBLE_DEVICES=0 \
  UAVSAT_BACKBONE="$BACKBONE" \
  UAVSAT_EXPERIMENT_FRAME_COUNT=2 \
  python3 -u - <<'PY' 2>&1 | tee "$LOG_DIR/00_visual_train.log"
import os
import config
from robust_tracker_base import resolve_device, set_seed
from visual_localizer import train_visual_retrieval_a_only

set_seed(config.SEED)
device = resolve_device()
train_visual_retrieval_a_only(
    device=device,
    epochs=int(os.environ["VISUAL_EPOCHS_RUN"]),
    jitter_m=float(os.environ["JITTER_M"]),
    resume=False,
)
print("visual checkpoint:", config.VISUAL_CHECKPOINT, flush=True)
PY
}

prepare_feature_cache() {
  echo "[cache] GPU0: prebuild A/B/C MobileNet backbone caches once before parallel jobs"
  CUDA_VISIBLE_DEVICES=0 \
  UAVSAT_BACKBONE="$BACKBONE" \
  UAVSAT_EXPERIMENT_FRAME_COUNT=2 \
  python3 -u - <<'PY' 2>&1 | tee "$LOG_DIR/01_prepare_feature_cache.log"
import config
import robust_tracker as rt
from visual_localizer import FrozenVisualLocalizer

rt.set_seed(config.SEED)
device = rt.resolve_device()
visual = FrozenVisualLocalizer(device)
for route_name, root in zip(config.ROUTE_NAMES, config.ROUTE_ROOTS):
    cache = rt.build_route_cache(route_name, root, visual, device)
    print(f"cache ready: {route_name}, frames={len(cache)}", flush=True)
print("feature cache root:", config.FEATURE_CACHE_DIR, flush=True)
PY
}

train_temporal_parallel() {
  echo "[temporal] GPU0: 2-frame main method"
  echo "[temporal] GPU1: 1-frame visual-input ablation"

  CUDA_VISIBLE_DEVICES=0 \
  UAVSAT_BACKBONE="$BACKBONE" \
  UAVSAT_EXPERIMENT_FRAME_COUNT=2 \
  python3 -u robust_tracker.py \
    --mode train \
    --reuse-visual \
    --temporal-epochs "$TEMPORAL_EPOCHS_RUN" \
    --patience "$PATIENCE_RUN" \
    --jitter-m "$JITTER_M" \
    --forward-rows 3 \
    > >(tee "$LOG_DIR/10_train_2frame.log") 2>&1 &
  pid_2frame=$!

  CUDA_VISIBLE_DEVICES=1 \
  UAVSAT_BACKBONE="$BACKBONE" \
  UAVSAT_EXPERIMENT_FRAME_COUNT=1 \
  python3 -u robust_tracker.py \
    --mode train \
    --reuse-visual \
    --temporal-epochs "$TEMPORAL_EPOCHS_RUN" \
    --patience "$PATIENCE_RUN" \
    --jitter-m "$JITTER_M" \
    --forward-rows 3 \
    > >(tee "$LOG_DIR/11_train_1frame.log") 2>&1 &
  pid_1frame=$!

  wait "$pid_2frame"
  wait "$pid_1frame"
  echo "[temporal] both temporal trainings finished"
}

run_full_eval() {
  local gpu="$1"
  local frames="$2"
  local rows="$3"
  local suffix="${4:-parallel}"
  local log="$LOG_DIR/full_${frames}frame_${rows}x6_${suffix}.log"
  echo "[eval] GPU${gpu}: full ${frames}-frame ${rows}x6"
  CUDA_VISIBLE_DEVICES="$gpu" \
  UAVSAT_BACKBONE="$BACKBONE" \
  UAVSAT_EXPERIMENT_FRAME_COUNT="$frames" \
  python3 -u robust_tracker.py \
    --mode eval \
    --reuse-visual \
    --jitter-m "$JITTER_M" \
    --forward-rows "$rows" \
    --measure-latency \
    > >(tee "$log") 2>&1 &
  LAST_PID=$!
}

run_meanshift_eval() {
  local gpu="$1"
  local rows="$2"
  local suffix="${3:-parallel}"
  local log="$LOG_DIR/meanshift_${rows}x6_${suffix}.log"
  echo "[eval] GPU${gpu}: MeanShift-only ${rows}x6"
  CUDA_VISIBLE_DEVICES="$gpu" \
  UAVSAT_BACKBONE="$BACKBONE" \
  UAVSAT_EXPERIMENT_FRAME_COUNT=2 \
  MS_ROWS="$rows" \
  python3 -u - <<'PY' > >(tee "$log") 2>&1 &
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import config
import robust_tracker as rt
import robust_tracker_base as b
from visual_localizer import FrozenVisualLocalizer

rows = int(os.environ["MS_ROWS"])
jitter = float(os.environ["JITTER_M"])
config.LOCAL_PRIOR_JITTER_M = jitter
config.CONTROLLED_GT_PRIOR_JITTER_M = jitter
rt._set_forward_rows(rows)
rt.set_seed(config.SEED)
device = rt.resolve_device()
visual = FrozenVisualLocalizer(device)
warmup = int(getattr(config, "LATENCY_WARMUP_FRAMES", 30))

out_dir = Path(config.BACKBONE_OUTPUT_DIR) / "meanshift_only"
out_dir.mkdir(parents=True, exist_ok=True)
summary = {
    "method": "MeanShift-only",
    "backbone": str(config.BACKBONE_KEY),
    "forward_rows": rows,
    "candidate_count": rows * 6,
    "jitter_m": jitter,
    "timing_definition": (
        "cached UAV backbone feature -> UAV projector + selected SAT projector + "
        "cosine similarity + Soft MeanShift; excludes image I/O/preprocessing/UAV backbone"
    ),
    "routes": {},
}

for route_name in ["route_B", "route_C"]:
    route_index = config.ROUTE_NAMES.index(route_name)
    cache = rt.build_route_cache(
        route_name, config.ROUTE_ROOTS[route_index], visual, device
    )
    route = rt.WaypointRoute(
        rt.load_waypoint_xy(route_name, visual.origin_lat, visual.origin_lon)
    )
    gt_state = rt.build_gt_route_state(cache, route)
    errors = []
    timing_ms = []
    frame_rows = []

    for index in range(len(cache)):
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        controlled_center_se, controlled_prior_xy, controlled_jitter_xy = (
            b.controlled_gt_prior_se(cache, route, gt_state, index)
        )
        center_xy = torch.as_tensor(
            np.asarray(controlled_prior_xy, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        ).reshape(1, 2)
        heading_rad = route.route_heading_rad(float(controlled_center_se[0]))

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        candidate = rt.forward_rows_candidate_batch(
            visual=visual,
            uav_clip=uav_clip,
            center_xy=center_xy,
            heading_rad=heading_rad,
            grid_size=int(config.ACQ_LOCAL_GRID_SIZE),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        pred_xy = candidate.softms_xy[0].detach().cpu().numpy()
        reference_xy = cache.gt_xy[index].cpu().numpy()
        error = float(np.linalg.norm(pred_xy - reference_xy))
        errors.append(error)
        if index >= warmup:
            timing_ms.append(elapsed_ms)
        frame_rows.append(
            {
                "frame_id": int(cache.frame_ids[index]),
                "reference_x": float(reference_xy[0]),
                "reference_y": float(reference_xy[1]),
                "meanshift_x": float(pred_xy[0]),
                "meanshift_y": float(pred_xy[1]),
                "error_m": error,
                "latency_ms": float(elapsed_ms),
            }
        )

    csv_path = out_dir / f"{route_name}_meanshift_forward{rows}x6_frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)

    result = b.metric_summary(errors)
    error_array = np.asarray(errors, dtype=np.float64)
    q90 = float(np.quantile(error_array, 0.90))
    tail = error_array[error_array >= q90]
    result["CVaR90_m"] = float(np.mean(tail)) if tail.size else q90

    steady = np.asarray(timing_ms, dtype=np.float64)
    if steady.size == 0:
        steady = np.asarray([r["latency_ms"] for r in frame_rows], dtype=np.float64)
    mean_ms = float(np.mean(steady))
    result["CoreTiming"] = {
        "mean_ms": mean_ms,
        "median_ms": float(np.median(steady)),
        "p90_ms": float(np.quantile(steady, 0.90)),
        "fps": float(1000.0 / max(mean_ms, 1e-12)),
        "samples": int(steady.size),
    }
    result["CSV"] = str(csv_path)
    summary["routes"][route_name] = result
    print(
        f"MeanShift-only {route_name} {rows}x6: "
        f"MLE={result['MLE_m']:.3f}m P90={result['P90_m']:.3f}m "
        f"core={mean_ms:.3f}ms FPS={result['CoreTiming']['fps']:.2f}",
        flush=True,
    )

summary_path = out_dir / f"meanshift_forward{rows}x6_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("summary:", summary_path, flush=True)
PY
  LAST_PID=$!
}

benchmark_backbone() {
  local gpu="${1:-$TIMING_GPU}"
  echo "[latency] GPU${gpu}: batch-1 UAV backbone latency for $BACKBONE"
  CUDA_VISIBLE_DEVICES="$gpu" \
  UAVSAT_BACKBONE="$BACKBONE" \
  UAVSAT_EXPERIMENT_FRAME_COUNT=2 \
  python3 -u - <<'PY' 2>&1 | tee "$LOG_DIR/backbone_latency.log"
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
import robust_tracker as rt
from data import image_transform
from visual_localizer import FrozenVisualLocalizer

rt.set_seed(config.SEED)
device = rt.resolve_device()
visual = FrozenVisualLocalizer(device)
cache = rt.build_route_cache("route_B", config.ROUTE_ROOTS[1], visual, device)
transform = image_transform(False, source="uav")
warmup = int(getattr(config, "LATENCY_WARMUP_FRAMES", 30))
max_samples = min(len(cache.image_paths), 300)
times = []

for index, path in enumerate(cache.image_paths[:max_samples]):
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    _ = visual.encode_uav_clip(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if index >= warmup:
        times.append(elapsed_ms)

values = np.asarray(times, dtype=np.float64)
if values.size == 0:
    raise RuntimeError("not enough frames for backbone latency benchmark")
mean_ms = float(np.mean(values))
result = {
    "backbone": str(config.BACKBONE_KEY),
    "batch_size": 1,
    "definition": "UAV backbone only; disk I/O and Python/PIL preprocessing excluded",
    "samples": int(values.size),
    "mean_ms": mean_ms,
    "median_ms": float(np.median(values)),
    "p90_ms": float(np.quantile(values, 0.90)),
    "fps": float(1000.0 / max(mean_ms, 1e-12)),
}
out = Path(config.BACKBONE_OUTPUT_DIR) / "backbone_latency.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2), flush=True)
print("summary:", out, flush=True)
PY
}

collect_results() {
  echo "[collect] build one accuracy/latency CSV"
  UAVSAT_BACKBONE="$BACKBONE" \
  UAVSAT_EXPERIMENT_FRAME_COUNT=2 \
  python3 -u - <<'PY'
import csv
import json
from pathlib import Path

import numpy as np
import config

root = Path(config.BACKBONE_OUTPUT_DIR)
backbone_path = root / "backbone_latency.json"
backbone = json.loads(backbone_path.read_text(encoding="utf-8")) if backbone_path.exists() else {}
backbone_ms = float(backbone.get("mean_ms", float("nan")))
rows = []

metric_keys = [
    "MLE_m", "MedLE_m", "P90_m", "P95_m", "P99_m", "CVaR90_m",
    "LSR@5_pct", "LSR@10_pct", "LSR@15_pct", "LSR@20_pct",
]

def cvar90_from_csv(path, error_field):
    if not path or not Path(path).exists():
        return float("nan")
    values = []
    with Path(path).open("r", encoding="utf-8") as f:
        for item in csv.DictReader(f):
            values.append(float(item[error_field]))
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    q90 = float(np.quantile(arr, 0.90))
    tail = arr[arr >= q90]
    return float(np.mean(tail)) if tail.size else q90

def add_row(method, frames, search_rows, route, result, core, error_field):
    core_ms = float(core.get("mean_ms", float("nan")))
    full_ms = backbone_ms + core_ms if np.isfinite(backbone_ms) and np.isfinite(core_ms) else float("nan")
    row = {
        "method": method,
        "frame_count": frames,
        "search": f"{search_rows}x6",
        "candidate_count": int(search_rows * 6),
        "route": route,
        "backbone": str(config.BACKBONE_KEY),
        "backbone_ms": backbone_ms,
        "core_ms": core_ms,
        "core_fps": float(core.get("fps", float("nan"))),
        "estimated_full_ms": full_ms,
        "estimated_full_fps": (1000.0 / full_ms if np.isfinite(full_ms) and full_ms > 0 else float("nan")),
    }
    for key in metric_keys:
        row[key] = float(result.get(key, float("nan")))
    if not np.isfinite(row["CVaR90_m"]):
        row["CVaR90_m"] = cvar90_from_csv(result.get("CSV"), error_field)
    rows.append(row)

# Complete 2-frame method: candidate-count accuracy/speed ablation.
for search_rows in (3, 4, 5, 6):
    path = root / "2frame" / f"robust_tracker_summary_forward{search_rows}x6.json"
    if not path.exists():
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    block = payload.get("results", {}).get(f"{search_rows}x6", {})
    for route, result in block.items():
        add_row(
            "Full-GRU-Polynomial-Kalman", 2, search_rows, route, result,
            result.get("TrackingCoreTiming", {}), "error_final_m"
        )

# 1-frame visual-input ablation at the thesis/default 3x6 search.
path = root / "1frame" / "robust_tracker_summary_forward3x6.json"
if path.exists():
    payload = json.loads(path.read_text(encoding="utf-8"))
    block = payload.get("results", {}).get("3x6", {})
    for route, result in block.items():
        add_row(
            "Full-GRU-Polynomial-Kalman", 1, 3, route, result,
            result.get("TrackingCoreTiming", {}), "error_final_m"
        )

# Raw MeanShift baseline / ablation.
for search_rows in (3, 4, 5, 6):
    path = root / "meanshift_only" / f"meanshift_forward{search_rows}x6_summary.json"
    if not path.exists():
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    for route, result in payload.get("routes", {}).items():
        add_row(
            "MeanShift-only", 0, search_rows, route, result,
            result.get("CoreTiming", {}), "error_m"
        )

# B/C arithmetic mean rows for quick thesis-table preparation.
base_rows = list(rows)
groups = {}
for row in base_rows:
    key = (
        row["method"], row["frame_count"], row["search"],
        row["candidate_count"], row["backbone"]
    )
    groups.setdefault(key, []).append(row)
for key, group in groups.items():
    if len(group) < 2:
        continue
    method, frames, search, candidates, backbone_name = key
    mean_row = {
        "method": method,
        "frame_count": frames,
        "search": search,
        "candidate_count": candidates,
        "route": "mean_BC",
        "backbone": backbone_name,
    }
    numeric_fields = [
        key for key in group[0]
        if key not in {"method", "frame_count", "search", "candidate_count", "route", "backbone"}
    ]
    for field in numeric_fields:
        values = np.asarray([float(item[field]) for item in group], dtype=np.float64)
        mean_row[field] = float(np.nanmean(values))
    rows.append(mean_row)

out = root / "experiment_summary.csv"
if not rows:
    raise RuntimeError("no experiment summaries found")
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("combined summary:", out)
for row in rows:
    if row["route"] == "mean_BC":
        print(
            f"{row['method']:28s} frame={row['frame_count']} search={row['search']} "
            f"MLE={row['MLE_m']:.3f}m P90={row['P90_m']:.3f}m "
            f"full≈{row['estimated_full_ms']:.3f}ms FPS≈{row['estimated_full_fps']:.2f}"
        )
PY
}

run_eval_parallel() {
  # Wave 1: use all seven available GPUs for the expensive accuracy runs.
  run_full_eval 0 2 3 parallel; p0=$LAST_PID
  run_full_eval 1 2 4 parallel; p1=$LAST_PID
  run_full_eval 2 2 5 parallel; p2=$LAST_PID
  run_full_eval 3 2 6 parallel; p3=$LAST_PID
  run_full_eval 4 1 3 parallel; p4=$LAST_PID
  run_meanshift_eval 5 3 parallel; p5=$LAST_PID
  run_meanshift_eval 6 4 parallel; p6=$LAST_PID
  wait "$p0"; wait "$p1"; wait "$p2"; wait "$p3"; wait "$p4"; wait "$p5"; wait "$p6"

  # Wave 2: remaining MeanShift sizes + backbone benchmark.
  run_meanshift_eval 5 5 parallel; p5=$LAST_PID
  run_meanshift_eval 6 6 parallel; p6=$LAST_PID
  benchmark_backbone 4 & p4=$!
  wait "$p4"; wait "$p5"; wait "$p6"

  collect_results
}

run_timing_serial() {
  echo "[timing] publication timing sweep: all methods on the same physical GPU${TIMING_GPU}"
  benchmark_backbone "$TIMING_GPU"

  for rows in 3 4 5 6; do
    run_full_eval "$TIMING_GPU" 2 "$rows" timing
    pid=$LAST_PID
    wait "$pid"
  done

  run_full_eval "$TIMING_GPU" 1 3 timing
  pid=$LAST_PID
  wait "$pid"

  for rows in 3 4 5 6; do
    run_meanshift_eval "$TIMING_GPU" "$rows" timing
    pid=$LAST_PID
    wait "$pid"
  done

  collect_results
}

case "$MODE" in
  visual)
    train_visual
    prepare_feature_cache
    ;;
  train)
    train_visual
    prepare_feature_cache
    train_temporal_parallel
    ;;
  eval)
    prepare_feature_cache
    run_eval_parallel
    ;;
  timing)
    prepare_feature_cache
    run_timing_serial
    ;;
  all)
    train_visual
    prepare_feature_cache
    train_temporal_parallel
    run_eval_parallel
    ;;
  collect)
    collect_results
    ;;
  *)
    echo "usage: $0 [all|visual|train|eval|timing|collect]" >&2
    exit 2
    ;;
esac
