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

# The machines available for this project are GPU 0, 5 and 6.  Accuracy jobs
# are parallelized across those three GPUs.  Publication timing should still be
# measured serially on TIMING_GPU so FPS comparisons use one physical device.
GPU_MAIN="${GPU_MAIN:-0}"
GPU_ABLATION="${GPU_ABLATION:-5}"
GPU_BASELINE="${GPU_BASELINE:-6}"

BACKBONE_ROOT="$HERE/output/$BACKBONE"
LOG_DIR="$BACKBONE_ROOT/logs"
CHECKPOINT_DIR="$BACKBONE_ROOT/checkpoints"
FEATURE_CACHE_DIR="$BACKBONE_ROOT/feature_cache"
VISUAL_CKPT="$CHECKPOINT_DIR/visual_retrieval_A_only_${BACKBONE}.pt"
mkdir -p "$CHECKPOINT_DIR" "$FEATURE_CACHE_DIR" "$LOG_DIR"

export BACKBONE JITTER_M VISUAL_EPOCHS_RUN TEMPORAL_EPOCHS_RUN PATIENCE_RUN

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

ensure_visual() {
  # The visual checkpoint contains the satellite backbone gallery.  Therefore
  # once this file exists, every temporal/baseline/ablation process loads the
  # same SAT gallery instead of rebuilding satellite patches/features.
  exec 9>"$BACKBONE_ROOT/.visual_cache.lock"
  flock 9
  if [[ "$FORCE_VISUAL" != "1" && -f "$VISUAL_CKPT" ]]; then
    log "visual: reuse $VISUAL_CKPT (SAT gallery already embedded)"
    flock -u 9
    exec 9>&-
    return
  fi

  log "visual: GPU${GPU_MAIN} Route-A-only training; SAT gallery will be built once"
  CUDA_VISIBLE_DEVICES="$GPU_MAIN" \
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
print("satellite gallery is stored inside this checkpoint and reused later", flush=True)
PY
  flock -u 9
  exec 9>&-
}

prepare_shared_cache() {
  # Route A/B/C UAV backbone features are persistent on disk.  Do this before
  # starting any parallel job so GPU 0/5/6 never race to build the same cache.
  exec 8>"$BACKBONE_ROOT/.route_cache.lock"
  flock 8
  log "cache: GPU${GPU_MAIN} prebuild/reuse Route A/B/C UAV backbone cache"
  CUDA_VISIBLE_DEVICES="$GPU_MAIN" \
  UAVSAT_BACKBONE="$BACKBONE" \
  UAVSAT_EXPERIMENT_FRAME_COUNT=2 \
  python3 -u - <<'PY' 2>&1 | tee "$LOG_DIR/01_prepare_shared_cache.log"
import config
import robust_tracker as rt
from visual_localizer import FrozenVisualLocalizer

rt.set_seed(config.SEED)
device = rt.resolve_device()
visual = FrozenVisualLocalizer(device)
print("SAT gallery source: visual checkpoint", config.VISUAL_CHECKPOINT, flush=True)
for route_name, root in zip(config.ROUTE_NAMES, config.ROUTE_ROOTS):
    cache = rt.build_route_cache(route_name, root, visual, device)
    print(
        f"route cache ready: {route_name}, frames={len(cache)}, "
        f"feature_cache={config.FEATURE_CACHE_DIR}",
        flush=True,
    )
PY
  flock -u 8
  exec 8>&-
}

train_temporal_one() {
  local gpu="$1"
  local frames="$2"
  local log_file="$LOG_DIR/10_train_${frames}frame_multirate.log"
  log "train: GPU${gpu} ${frames}-frame MultiRate Route-A native+stride temporal model"
  CUDA_VISIBLE_DEVICES="$gpu" \
  UAVSAT_BACKBONE="$BACKBONE" \
  UAVSAT_EXPERIMENT_FRAME_COUNT="$frames" \
  python3 -u train_multirate_a.py \
    --temporal-epochs "$TEMPORAL_EPOCHS_RUN" \
    --patience "$PATIENCE_RUN" \
    --jitter-m "$JITTER_M" \
    --forward-rows 3 \
    > >(tee "$log_file") 2>&1
}

train_temporal_parallel() {
  # Only frame-count ablation needs a second temporal training.  All remaining
  # ablations reuse the trained 2-frame checkpoint at inference time.
  train_temporal_one "$GPU_MAIN" 2 & p_main=$!
  train_temporal_one "$GPU_ABLATION" 1 & p_one=$!
  wait "$p_main"
  wait "$p_one"
  log "train: 2-frame and 1-frame temporal checkpoints finished"
}

run_decoder_baselines() {
  local gpu="${1:-$GPU_BASELINE}"
  local rows="${2:-3}"
  local suffix="${3:-accuracy}"
  local log_file="$LOG_DIR/20_decoder_baselines_${rows}x6_${suffix}.log"
  log "baseline: GPU${gpu} Top-1 + Weighted Centroid + SoftMS from one shared candidate pass"

  CUDA_VISIBLE_DEVICES="$gpu" \
  UAVSAT_BACKBONE="$BACKBONE" \
  UAVSAT_EXPERIMENT_FRAME_COUNT=2 \
  UAVSAT_RUN_TAG="decoder_baselines_${rows}x6_${suffix}" \
  DECODER_ROWS="$rows" \
  python3 -u - <<'PY' 2>&1 | tee "$log_file"
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

rows = int(os.environ["DECODER_ROWS"])
rt._set_forward_rows(rows)
config.LOCAL_PRIOR_JITTER_M = float(os.environ["JITTER_M"])
config.CONTROLLED_GT_PRIOR_JITTER_M = float(os.environ["JITTER_M"])
rt.set_seed(config.SEED)
device = rt.resolve_device()
visual = FrozenVisualLocalizer(device)
warmup = int(getattr(config, "LATENCY_WARMUP_FRAMES", 30))
out_dir = Path(config.OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

summary = {
    "backbone": str(config.BACKBONE_KEY),
    "search": f"{rows}x6",
    "candidate_count": rows * 6,
    "train_routes": ["route_A"],
    "eval_routes": ["route_B", "route_C"],
    "methods": {},
    "timing_definition": (
        "cached UAV backbone feature -> selected SAT projection + cosine scoring + "
        "Top1/weighted/SoftMS decoder; image I/O and UAV backbone excluded"
    ),
}

for route_name in ["route_B", "route_C"]:
    route_index = config.ROUTE_NAMES.index(route_name)
    cache = rt.build_route_cache(route_name, config.ROUTE_ROOTS[route_index], visual, device)
    route = rt.WaypointRoute(rt.load_waypoint_xy(route_name, visual.origin_lat, visual.origin_lon))
    gt_state = rt.build_gt_route_state(cache, route)

    errors = {"top1": [], "weighted_centroid": [], "softms": []}
    timing_ms = []
    frame_rows = []

    for index in range(len(cache)):
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        controlled_center_se, controlled_prior_xy, _ = b.controlled_gt_prior_se(
            cache, route, gt_state, index
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
        weighted_xy = (
            candidate.raw_prob[:, :, None] * candidate.centers
        ).sum(dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if index >= warmup:
            timing_ms.append(elapsed_ms)

        reference_xy = cache.gt_xy[index].cpu().numpy()
        predictions = {
            "top1": candidate.raw_top1_xy[0].detach().cpu().numpy(),
            "weighted_centroid": weighted_xy[0].detach().cpu().numpy(),
            "softms": candidate.softms_xy[0].detach().cpu().numpy(),
        }
        row = {
            "frame_id": int(cache.frame_ids[index]),
            "reference_x": float(reference_xy[0]),
            "reference_y": float(reference_xy[1]),
            "latency_shared_candidate_ms": float(elapsed_ms),
        }
        for name, pred in predictions.items():
            error = float(np.linalg.norm(pred - reference_xy))
            errors[name].append(error)
            row[f"{name}_x"] = float(pred[0])
            row[f"{name}_y"] = float(pred[1])
            row[f"{name}_error_m"] = error
        frame_rows.append(row)

    csv_path = out_dir / f"{route_name}_decoder_baselines_{rows}x6.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)

    steady = np.asarray(timing_ms, dtype=np.float64)
    shared_ms = float(np.mean(steady)) if steady.size else float("nan")
    for name, values in errors.items():
        result = b.metric_summary(values)
        arr = np.asarray(values, dtype=np.float64)
        q90 = float(np.quantile(arr, 0.90))
        tail = arr[arr >= q90]
        result["CVaR90_m"] = float(np.mean(tail)) if tail.size else q90
        result["shared_candidate_ms"] = shared_ms
        result["shared_candidate_fps"] = float(1000.0 / shared_ms) if np.isfinite(shared_ms) and shared_ms > 0 else float("nan")
        result["CSV"] = str(csv_path)
        summary["methods"].setdefault(name, {})[route_name] = result
        print(
            f"{name} {route_name} {rows}x6: MLE={result['MLE_m']:.3f}m "
            f"P90={result['P90_m']:.3f}m LSR@15={result['LSR@15_pct']:.2f}%",
            flush=True,
        )

summary_path = out_dir / "decoder_baselines_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("summary:", summary_path, flush=True)
PY
}

benchmark_backbone() {
  local gpu="${1:-$GPU_BASELINE}"
  local log_file="$LOG_DIR/21_backbone_latency_gpu${gpu}.log"
  log "latency: GPU${gpu} batch-1 UAV backbone benchmark"
  CUDA_VISIBLE_DEVICES="$gpu" \
  UAVSAT_BACKBONE="$BACKBONE" \
  UAVSAT_EXPERIMENT_FRAME_COUNT=2 \
  UAVSAT_RUN_TAG="backbone_latency" \
  python3 -u - <<'PY' 2>&1 | tee "$log_file"
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
    "definition": "UAV backbone only; disk I/O and PIL preprocessing excluded",
    "samples": int(values.size),
    "mean_ms": mean_ms,
    "median_ms": float(np.median(values)),
    "p90_ms": float(np.quantile(values, 0.90)),
    "fps": float(1000.0 / max(mean_ms, 1e-12)),
}
out = Path(config.OUTPUT_DIR) / "backbone_latency.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2), flush=True)
print("summary:", out, flush=True)
PY
}

run_tracker_variant() {
  local gpu="$1"
  local tag="$2"
  local frames="$3"
  local rows="$4"
  local anchor="$5"
  local motion="$6"
  local kalman="$7"
  local disable_gru="$8"
  local fixed_variance="${9:-25.0}"
  local timing="${10:-0}"
  local latency_arg=()
  if [[ "$timing" == "1" ]]; then
    latency_arg=(--measure-latency)
  fi

  log "eval: GPU${gpu} tag=${tag} frames=${frames} search=${rows}x6 anchor=${anchor} motion=${motion} kalman=${kalman} disable_gru=${disable_gru}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  UAVSAT_BACKBONE="$BACKBONE" \
  UAVSAT_EXPERIMENT_FRAME_COUNT="$frames" \
  UAVSAT_EXPERIMENT_VARIANT="$tag" \
  UAVSAT_EXPERIMENT_ANCHOR="$anchor" \
  UAVSAT_EXPERIMENT_MOTION="$motion" \
  UAVSAT_EXPERIMENT_KALMAN="$kalman" \
  UAVSAT_EXPERIMENT_DISABLE_GRU="$disable_gru" \
  UAVSAT_EXPERIMENT_FIXED_VARIANCE_M2="$fixed_variance" \
  UAVSAT_RUN_TAG="$tag" \
  python3 -u robust_tracker.py \
    --mode eval \
    --reuse-visual \
    --jitter-m "$JITTER_M" \
    --forward-rows "$rows" \
    "${latency_arg[@]}" \
    > >(tee "$LOG_DIR/eval_${tag}.log") 2>&1
}

run_ablation_parallel() {
  # Wave 1: strongest architecture ablations.
  run_tracker_variant "$GPU_MAIN" main_2frame_3x6 2 3 softms quadratic learned 0 25 0 & p0=$!
  run_tracker_variant "$GPU_ABLATION" frame_1 1 3 softms quadratic learned 0 25 0 & p5=$!
  run_tracker_variant "$GPU_BASELINE" no_gru 2 3 softms quadratic learned 1 25 0 & p6=$!
  wait "$p0"; wait "$p5"; wait "$p6"

  # Wave 2: inertial polynomial contribution.
  run_tracker_variant "$GPU_MAIN" motion_velocity 2 3 softms velocity learned 0 25 0 & p0=$!
  run_tracker_variant "$GPU_ABLATION" motion_none 2 3 softms none learned 0 25 0 & p5=$!
  run_tracker_variant "$GPU_BASELINE" kalman_none 2 3 softms quadratic none 0 25 0 & p6=$!
  wait "$p0"; wait "$p5"; wait "$p6"

  # Wave 3: uncertainty/Kalman and decoder-anchor contribution.
  run_tracker_variant "$GPU_MAIN" kalman_fixedR 2 3 softms quadratic fixed 0 25 0 & p0=$!
  run_tracker_variant "$GPU_ABLATION" weighted_anchor_full 2 3 weighted_centroid quadratic learned 0 25 0 & p5=$!
  run_tracker_variant "$GPU_BASELINE" search_4x6 2 4 softms quadratic learned 0 25 0 & p6=$!
  wait "$p0"; wait "$p5"; wait "$p6"

  # Wave 4: candidate-count accuracy trade-off.
  run_tracker_variant "$GPU_MAIN" search_5x6 2 5 softms quadratic learned 0 25 0 & p0=$!
  run_tracker_variant "$GPU_ABLATION" search_6x6 2 6 softms quadratic learned 0 25 0 & p5=$!
  wait "$p0"; wait "$p5"
}

run_accuracy_pipeline() {
  # Decoder baselines need no temporal checkpoint, so GPU6 can work while the
  # two temporal models are training on GPU0/GPU5.
  run_decoder_baselines "$GPU_BASELINE" 3 accuracy & p_decoder=$!
  train_temporal_parallel
  wait "$p_decoder"
  run_ablation_parallel
  collect_results
}

run_timing_serial() {
  # Do not compare FPS measured on different GPUs.  Run the publication timing
  # sweep serially on one device after all checkpoints/caches already exist.
  benchmark_backbone "$TIMING_GPU"
  run_decoder_baselines "$TIMING_GPU" 3 timing
  run_tracker_variant "$TIMING_GPU" timing_main_3x6 2 3 softms quadratic learned 0 25 1
  run_tracker_variant "$TIMING_GPU" timing_search_4x6 2 4 softms quadratic learned 0 25 1
  run_tracker_variant "$TIMING_GPU" timing_search_5x6 2 5 softms quadratic learned 0 25 1
  run_tracker_variant "$TIMING_GPU" timing_search_6x6 2 6 softms quadratic learned 0 25 1
  collect_results
}

collect_results() {
  log "collect: scan all tagged experiment summaries"
  UAVSAT_BACKBONE="$BACKBONE" \
  python3 -u - <<'PY'
import csv
import json
from pathlib import Path

import config

root = Path(config.BACKBONE_OUTPUT_DIR)
experiments = root / "experiments"
rows = []

for path in sorted(experiments.glob("*/robust_tracker_summary_forward*x6.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    tag = path.parent.name
    for search, route_map in payload.get("results", {}).items():
        for route_name, result in route_map.items():
            timing = result.get("TrackingCoreTiming", {})
            rows.append({
                "tag": tag,
                "type": "tracker",
                "search": search,
                "route": route_name,
                "backbone": str(config.BACKBONE_KEY),
                "MLE_m": result.get("MLE_m"),
                "MedLE_m": result.get("MedLE_m"),
                "P90_m": result.get("P90_m"),
                "P95_m": result.get("P95_m"),
                "P99_m": result.get("P99_m"),
                "LSR@5_pct": result.get("LSR@5_pct"),
                "LSR@10_pct": result.get("LSR@10_pct"),
                "LSR@15_pct": result.get("LSR@15_pct"),
                "LSR@20_pct": result.get("LSR@20_pct"),
                "core_ms": timing.get("mean_ms"),
                "core_fps": timing.get("fps"),
                "source": str(path),
            })

for path in sorted(experiments.glob("decoder_baselines_*/*decoder_baselines_summary.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    search = payload.get("search", "3x6")
    for method, route_map in payload.get("methods", {}).items():
        for route_name, result in route_map.items():
            rows.append({
                "tag": method,
                "type": "decoder_baseline",
                "search": search,
                "route": route_name,
                "backbone": str(config.BACKBONE_KEY),
                "MLE_m": result.get("MLE_m"),
                "MedLE_m": result.get("MedLE_m"),
                "P90_m": result.get("P90_m"),
                "P95_m": result.get("P95_m"),
                "P99_m": result.get("P99_m"),
                "LSR@5_pct": result.get("LSR@5_pct"),
                "LSR@10_pct": result.get("LSR@10_pct"),
                "LSR@15_pct": result.get("LSR@15_pct"),
                "LSR@20_pct": result.get("LSR@20_pct"),
                "core_ms": result.get("shared_candidate_ms"),
                "core_fps": result.get("shared_candidate_fps"),
                "source": str(path),
            })

out = root / "experiment_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "tag", "type", "search", "route", "backbone",
    "MLE_m", "MedLE_m", "P90_m", "P95_m", "P99_m",
    "LSR@5_pct", "LSR@10_pct", "LSR@15_pct", "LSR@20_pct",
    "core_ms", "core_fps", "source",
]
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"collected {len(rows)} rows -> {out}")
PY
}

print_plan() {
  cat <<EOF
v36_byTeacher experiment suite
  backbone      : $BACKBONE
  GPU main      : $GPU_MAIN
  GPU ablation  : $GPU_ABLATION
  GPU baseline  : $GPU_BASELINE
  timing GPU    : $TIMING_GPU
  visual ckpt   : $VISUAL_CKPT
  feature cache : $FEATURE_CACHE_DIR

Recommended thesis experiments produced by 'all':
  - Route A only: visual + temporal training
  - Route B and Route C: evaluation, reported separately
  - Decoder baseline: Top-1 / Weighted Centroid / SoftMS
  - Temporal input: 1-frame vs 2-frame
  - GRU: full vs no-GRU
  - Motion prior: quadratic vs velocity-only vs no learned motion
  - Kalman: learned variance vs fixed R vs no Kalman
  - Anchor in full tracker: SoftMS vs weighted centroid
  - Forward candidate count: 3x6 / 4x6 / 5x6 / 6x6

Cache policy:
  - SAT gallery is built only when the visual checkpoint is missing/forced.
  - The visual checkpoint embeds that SAT gallery and all later jobs reuse it.
  - Route A/B/C UAV backbone features are prebuilt once before parallel jobs.
  - 1-frame/2-frame and every inference ablation share the same visual/cache root.
EOF
}

case "$MODE" in
  plan)
    print_plan
    ;;
  prepare)
    ensure_visual
    prepare_shared_cache
    ;;
  visual)
    ensure_visual
    ;;
  cache)
    ensure_visual
    prepare_shared_cache
    ;;
  train)
    ensure_visual
    prepare_shared_cache
    train_temporal_parallel
    ;;
  eval)
    ensure_visual
    prepare_shared_cache
    run_decoder_baselines "$GPU_BASELINE" 3 accuracy
    run_ablation_parallel
    collect_results
    ;;
  timing)
    ensure_visual
    prepare_shared_cache
    run_timing_serial
    ;;
  collect)
    collect_results
    ;;
  all)
    print_plan
    ensure_visual
    prepare_shared_cache
    run_accuracy_pipeline
    ;;
  *)
    echo "usage: $0 [plan|prepare|visual|cache|train|eval|timing|collect|all]" >&2
    exit 2
    ;;
esac
