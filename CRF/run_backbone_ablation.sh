#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ============================================================================
# True end-to-end benchmark:
# prepared UAV tensor -> complete localization model -> final GPS lat/lon
#
# NO RETRAINING.
# Reuses the checkpoints already produced by the backbone ablation.
# ============================================================================

GPU="${GPU:-0}"
WINDOW="${WINDOW:-5}"
WARMUP="${WARMUP:-30}"
MAX_TIMED_OUTPUTS="${MAX_TIMED_OUTPUTS:-0}"
JITTER_M="${JITTER_M:-12}"

OUT_DIR="outputs/full_pipeline_latency"
mkdir -p "${OUT_DIR}"

BACKBONES=(
  mobileclip2_s2
  vgg16
  resnet18
  mobilenet_v3_small
)

ROUTES=(
  route_B
  route_C
)

echo "============================================================================"
echo "TRUE FULL PIPELINE LATENCY + ACCURACY BENCHMARK"
echo "============================================================================"
echo "GPU                    : ${GPU}"
echo "Temporal window        : ${WINDOW}"
echo "Warm-up GPS outputs    : ${WARMUP}"
echo "Max timed outputs      : ${MAX_TIMED_OUTPUTS} (0 = all after warm-up)"
echo "Jitter                 : ${JITTER_M} m"
echo
echo "Timed steady-state interval:"
echo "prepared UAV tensor"
echo " -> backbone"
echo " -> UAV/SAT heads"
echo " -> 6x6 retrieval"
echo " -> HardMS"
echo " -> T2-only RTL-CRF"
echo " -> Correction Gate"
echo " -> final XY"
echo " -> GPS latitude/longitude"
echo
echo "Image disk I/O / preprocessing and one-time gallery creation are NOT timed."
echo "All backbones run SEQUENTIALLY on the same GPU."
echo "============================================================================"

for backbone in "${BACKBONES[@]}"; do
  for route in "${ROUTES[@]}"; do
    echo
    echo "############################################################################"
    echo "BACKBONE=${backbone} ROUTE=${route}"
    echo "############################################################################"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    RTL_TEMPORAL_WINDOW="${WINDOW}" \
    RTL_BACKBONE="${backbone}" \
    python3 benchmark_backbone_speed.py \
      --backbone "${backbone}" \
      --route "${route}" \
      --window "${WINDOW}" \
      --warmup "${WARMUP}" \
      --max-timed-outputs "${MAX_TIMED_OUTPUTS}" \
      --jitter-m "${JITTER_M}" \
      --output "${OUT_DIR}/${backbone}_w${WINDOW}_${route}.json"
  done
done

echo
echo "============================================================================"
echo "FINAL B/C COMPLETE-PIPELINE COMPARISON"
echo "============================================================================"

WINDOW="${WINDOW}" python3 - <<'PY'
import json
import os
from pathlib import Path

window = int(os.environ["WINDOW"])
root = Path("outputs/full_pipeline_latency")

backbones = [
    "mobileclip2_s2",
    "vgg16",
    "resnet18",
    "mobilenet_v3_small",
]

routes = [
    "route_B",
]

rows = []

for backbone in backbones:
    row = {
        "backbone": backbone,
    }

    for route in routes:
        path = root / (
            f"{backbone}_w{window}_{route}.json"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        result = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        accuracy = result["accuracy"]
        steady = result["latency"][
            "steady_state_after_warmup"
        ]

        prefix = "B" if route == "route_B" else "C"

        row[f"{prefix}_MLE_m"] = accuracy[
            "MLE_m"
        ]
        row[f"{prefix}_P90_m"] = accuracy[
            "P90_m"
        ]
        row[f"{prefix}_RPE_m"] = accuracy[
            "RPE_m"
        ]
        row[f"{prefix}_Jump_pct"] = accuracy[
            "JumpRate_pct"
        ]
        row[f"{prefix}_E2E_mean_ms"] = steady[
            "mean_ms"
        ]
        row[f"{prefix}_E2E_P95_ms"] = steady[
            "p95_ms"
        ]
        row[f"{prefix}_GPS_FPS"] = steady[
            "gps_outputs_per_second"
        ]
        row[f"{prefix}_First_GPS_ms"] = result[
            "latency"
        ]["first_gps_output_ms"]

    row["Mean_MLE_BC_m"] = (
        row["B_MLE_m"]
        + row["C_MLE_m"]
    ) / 2.0

    row["Mean_E2E_ms_BC"] = (
        row["B_E2E_mean_ms"]
        + row["C_E2E_mean_ms"]
    ) / 2.0

    row["Mean_GPS_FPS_BC"] = (
        row["B_GPS_FPS"]
        + row["C_GPS_FPS"]
    ) / 2.0

    rows.append(row)

print()
print(
    f"{'backbone':<22}"
    f"{'B MLE':>9}"
    f"{'B RPE':>9}"
    f"{'B Jump%':>10}"
    f"{'B ms':>10}"
    f"{'B FPS':>9}"
    f"{'C MLE':>9}"
    f"{'C RPE':>9}"
    f"{'C Jump%':>10}"
    f"{'C ms':>10}"
    f"{'C FPS':>9}"
)

print("-" * 126)

for row in rows:
    print(
        f"{row['backbone']:<22}"
        f"{row['B_MLE_m']:>9.3f}"
        f"{row['B_RPE_m']:>9.3f}"
        f"{row['B_Jump_pct']:>10.3f}"
        f"{row['B_E2E_mean_ms']:>10.3f}"
        f"{row['B_GPS_FPS']:>9.2f}"
        f"{row['C_MLE_m']:>9.3f}"
        f"{row['C_RPE_m']:>9.3f}"
        f"{row['C_Jump_pct']:>10.3f}"
        f"{row['C_E2E_mean_ms']:>10.3f}"
        f"{row['C_GPS_FPS']:>9.2f}"
    )

comparison = {
    "definition": (
        "prepared UAV tensor -> visual backbone -> retrieval heads -> "
        "6x6 candidate selection -> cosine retrieval -> Fixed HardMS -> "
        "T2-only RTL-CRF -> Correction Gate -> final XY -> GPS lat/lon"
    ),
    "excluded_from_latency": [
        "image disk I/O",
        "PIL/torchvision preprocessing",
        "checkpoint/model loading",
        "one-time satellite backbone-gallery construction",
    ],
    "temporal_window": window,
    "rows": rows,
}

comparison_path = (
    root
    / f"comparison_full_pipeline_w{window}_BC.json"
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
print(
    f"saved: {comparison_path}"
)
PY

echo
echo "============================================================================"
echo "DONE"
echo "============================================================================"
echo "Main comparison:"
echo "  ${OUT_DIR}/comparison_full_pipeline_w${WINDOW}_BC.json"
echo
echo "Per-frame CSV files:"
echo "  ${OUT_DIR}/*.frames.csv"
echo "============================================================================"
