#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU="${GPU:-0}"
ROUTE="${ROUTE:-route_B}"
WINDOW="${WINDOW:-5}"
WARMUP="${WARMUP:-30}"
SAMPLES="${SAMPLES:-200}"
FIRST_REPEATS="${FIRST_REPEATS:-20}"

OUT_DIR="outputs/full_pipeline_latency"
mkdir -p "${OUT_DIR}"

BACKBONES=(
  mobileclip2_s2
  vgg16
  resnet18
  mobilenet_v3_small
)

echo "======================================================================"
echo "FULL MODEL INPUT -> FINAL GPS LATENCY BENCHMARK"
echo "======================================================================"
echo "physical GPU       : ${GPU}"
echo "route              : ${ROUTE}"
echo "temporal window    : ${WINDOW}"
echo "warm-up outputs    : ${WARMUP}"
echo "timed outputs      : ${SAMPLES}"
echo "first-GPS repeats  : ${FIRST_REPEATS}"
echo
echo "All backbones are benchmarked SEQUENTIALLY on the SAME GPU."
echo "This avoids GPU contention and keeps the comparison fair."
echo
echo "Main timed interval:"
echo "prepared UAV tensor -> full localization model -> final GPS lat/lon"
echo "======================================================================"

for backbone in "${BACKBONES[@]}"; do
  echo
  echo "######################################################################"
  echo "BACKBONE: ${backbone}"
  echo "######################################################################"

  CUDA_VISIBLE_DEVICES="${GPU}" \
  OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 \
  RTL_TEMPORAL_WINDOW="${WINDOW}" \
  RTL_BACKBONE="${backbone}" \
  python3 benchmark_backbone_speed.py \
    --backbone "${backbone}" \
    --route "${ROUTE}" \
    --window "${WINDOW}" \
    --warmup "${WARMUP}" \
    --samples "${SAMPLES}" \
    --first-output-repeats "${FIRST_REPEATS}" \
    --jitter-m 12 \
    --output "${OUT_DIR}/${backbone}_w${WINDOW}_${ROUTE}.json"
done

echo
echo "======================================================================"
echo "FINAL COMPLETE-PIPELINE COMPARISON"
echo "======================================================================"

WINDOW="${WINDOW}" ROUTE="${ROUTE}" python3 - <<'PY'
import json
import os
from pathlib import Path

window = int(os.environ["WINDOW"])
route = os.environ["ROUTE"]

root = Path("outputs/full_pipeline_latency")

backbones = [
    "mobileclip2_s2",
    "vgg16",
    "resnet18",
    "mobilenet_v3_small",
]

rows = []

for backbone in backbones:
    path = root / f"{backbone}_w{window}_{route}.json"

    if not path.exists():
        raise FileNotFoundError(path)

    result = json.loads(
        path.read_text(encoding="utf-8")
    )

    first = result["first_gps_output"]["latency"]
    steady = result["steady_state"]["model_input_to_gps"]
    visual = result["steady_state"]["visual_stage"]
    temporal = result["steady_state"]["temporal_plus_gps"]

    rows.append({
        "backbone": backbone,
        "first_gps_ms": first["mean_ms"],
        "steady_mean_ms": steady["mean_ms"],
        "steady_median_ms": steady["median_ms"],
        "steady_p95_ms": steady["p95_ms"],
        "gps_outputs_per_second": steady[
            "gps_outputs_per_second"
        ],
        "visual_mean_ms": visual["mean_ms"],
        "temporal_gps_mean_ms": temporal["mean_ms"],
    })

print()
print(
    f"{'backbone':<24}"
    f"{'first GPS ms':>15}"
    f"{'E2E mean ms':>15}"
    f"{'E2E P95 ms':>15}"
    f"{'GPS FPS':>12}"
)
print("-" * 81)

for row in rows:
    print(
        f"{row['backbone']:<24}"
        f"{row['first_gps_ms']:>15.3f}"
        f"{row['steady_mean_ms']:>15.3f}"
        f"{row['steady_p95_ms']:>15.3f}"
        f"{row['gps_outputs_per_second']:>12.2f}"
    )

comparison = {
    "definition": (
        "prepared UAV tensor entering full localization model "
        "to final GPS latitude/longitude"
    ),
    "temporal_window": window,
    "route": route,
    "rows": rows,
    "important": (
        "gps_outputs_per_second is complete steady-state localization "
        "throughput, NOT backbone-only FPS."
    ),
}

comparison_path = (
    root
    / f"comparison_full_pipeline_w{window}_{route}.json"
)

comparison_path.write_text(
    json.dumps(
        comparison,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print(f"saved: {comparison_path}")
PY

echo
echo "======================================================================"
echo "DONE"
echo "======================================================================"
