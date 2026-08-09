#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BASE_VISUAL="outputs/strict_train_A_test_BC_no_position_scale/checkpoints/visual_retrieval_A_only.pt"

compare_results() {
python3 - <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOTS = {
    3: Path("outputs/strict_train_A_test_BC_t2only_w3"),
    4: Path("outputs/strict_train_A_test_BC_t2only_w4"),
    5: Path("outputs/strict_train_A_test_BC_t2only_w5"),
}
JUMP_TOLERANCE_M = 3.0

for window, root in ROOTS.items():
    for route in ("route_B", "route_C"):
        path = root / f"{route}_robust_frames.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Finish all T2-only 3/4/5 runs first."
            )


def read_route(window, route):
    path = ROOTS[window] / f"{route}_robust_frames.csv"
    frame = pd.read_csv(path)
    required = {
        "frame_id", "gt_x", "gt_y", "temporal_x", "temporal_y"
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path}: missing columns {missing}")
    return frame[[
        "frame_id", "gt_x", "gt_y", "temporal_x", "temporal_y"
    ]].rename(columns={
        "temporal_x": f"temporal_x_w{window}",
        "temporal_y": f"temporal_y_w{window}",
    })


def metrics(prediction, gt):
    prediction = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    error = np.linalg.norm(prediction - gt, axis=1)

    if len(prediction) > 1:
        pred_step = np.diff(prediction, axis=0)
        gt_step = np.diff(gt, axis=0)
        rpe = np.linalg.norm(pred_step - gt_step, axis=1)
        gt_step_length = np.linalg.norm(gt_step, axis=1)
        jump_threshold = (
            float(np.percentile(gt_step_length, 99)) + JUMP_TOLERANCE_M
        )
        jump_rate = float(
            (np.linalg.norm(pred_step, axis=1) > jump_threshold).mean() * 100.0
        )
        rpe_mean = float(rpe.mean())
    else:
        jump_rate = 0.0
        rpe_mean = 0.0

    return {
        "frames": int(len(error)),
        "MLE_m": float(error.mean()),
        "P90_m": float(np.percentile(error, 90)),
        "RPE_m": rpe_mean,
        "JumpRate_pct": jump_rate,
        "LSR@5_pct": float((error <= 5.0).mean() * 100.0),
        "LSR@10_pct": float((error <= 10.0).mean() * 100.0),
        "LSR@15_pct": float((error <= 15.0).mean() * 100.0),
    }


summary = {
    "method": "T2-only Residual Second-Order Temporal Lattice CRF",
    "comparison": "3 vs 4 vs 5 frames on common evaluation frame IDs",
    "routes": {},
}

print("=" * 112)
print("T2-ONLY RTL-CRF: 3-FRAME vs 4-FRAME vs 5-FRAME — COMMON-FRAME COMPARISON")
print("=" * 112)

for route in ("route_B", "route_C"):
    merged = read_route(3, route)
    for window in (4, 5):
        other = read_route(window, route)
        merged = merged.merge(
            other[[
                "frame_id",
                f"temporal_x_w{window}",
                f"temporal_y_w{window}",
            ]],
            on="frame_id",
            how="inner",
            validate="one_to_one",
        )
    merged = merged.sort_values("frame_id").reset_index(drop=True)
    if merged.empty:
        raise RuntimeError(f"{route}: no common frames across 3/4/5 runs")

    gt = merged[["gt_x", "gt_y"]].to_numpy(np.float64)
    route_result = {}
    for window in (3, 4, 5):
        prediction = merged[[
            f"temporal_x_w{window}",
            f"temporal_y_w{window}",
        ]].to_numpy(np.float64)
        route_result[f"window_{window}"] = metrics(prediction, gt)

    summary["routes"][route] = route_result

    print()
    print(f"{route.upper()}  common frames={len(merged)}")
    print("-" * 112)
    print(
        f"{'metric':<22}"
        f"{'3-frame':>16}"
        f"{'4-frame':>16}"
        f"{'5-frame':>16}"
        f"{'best':>16}"
    )

    metric_names = (
        "MLE_m",
        "P90_m",
        "RPE_m",
        "JumpRate_pct",
        "LSR@5_pct",
        "LSR@10_pct",
        "LSR@15_pct",
    )

    lower_is_better = {
        "MLE_m", "P90_m", "RPE_m", "JumpRate_pct"
    }

    for metric in metric_names:
        values = {
            w: float(route_result[f"window_{w}"][metric])
            for w in (3, 4, 5)
        }
        if metric in lower_is_better:
            best = min(values, key=values.get)
        else:
            best = max(values, key=values.get)
        print(
            f"{metric:<22}"
            f"{values[3]:>16.4f}"
            f"{values[4]:>16.4f}"
            f"{values[5]:>16.4f}"
            f"{str(best) + '-frame':>16}"
        )

output = Path("outputs/t2only_temporal_window_3_4_5_comparison.json")
output.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print()
print("=" * 112)
print(f"comparison saved to: {output}")
print("=" * 112)
PY
}

if [[ "${1:-}" == "--compare-t2only" ]]; then
    compare_results
    exit 0
fi

WINDOW="${RTL_TEMPORAL_WINDOW:-4}"
if [[ "${WINDOW}" != "3" && "${WINDOW}" != "4" && "${WINDOW}" != "5" ]]; then
    echo "ERROR: RTL_TEMPORAL_WINDOW must be 3, 4, or 5; got '${WINDOW}'" >&2
    exit 2
fi

OUTPUT_DIR="outputs/strict_train_A_test_BC_t2only_w${WINDOW}"
TARGET_VISUAL="${OUTPUT_DIR}/checkpoints/visual_retrieval_A_only.pt"

if [[ ! -f "${BASE_VISUAL}" ]]; then
    echo "ERROR: missing the existing A-only visual checkpoint:" >&2
    echo "  ${BASE_VISUAL}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}/checkpoints"

# Each concurrent GPU run receives its own physical copy.  This is important
# because visual_localizer.py writes the resumed checkpoint back at the end of
# the zero-epoch visual stage; sharing one file across 3 GPUs would create a
# write race.
cp -p "${BASE_VISUAL}" "${TARGET_VISUAL}"

echo "============================================================================"
echo "T2-ONLY SECOND-ORDER RTL-CRF"
echo "Temporal window: ${WINDOW} frames"
echo "No learned T1 factor"
echo "Output directory: ${OUTPUT_DIR}"
echo "Visual checkpoint copy: ${TARGET_VISUAL}"
echo "============================================================================"

HAS_RESUME=0
for arg in "$@"; do
    if [[ "${arg}" == "--resume" ]]; then
        HAS_RESUME=1
        break
    fi
done

if [[ ${HAS_RESUME} -eq 1 ]]; then
    python3 robust_tracker.py "$@"
else
    python3 robust_tracker.py --resume "$@"
fi

# If this happens to be the last of the three concurrent jobs, print the
# common-frame comparison immediately.  Otherwise use --compare-t2only later.
if [[ -f "outputs/strict_train_A_test_BC_t2only_w3/route_B_robust_frames.csv" \
   && -f "outputs/strict_train_A_test_BC_t2only_w3/route_C_robust_frames.csv" \
   && -f "outputs/strict_train_A_test_BC_t2only_w4/route_B_robust_frames.csv" \
   && -f "outputs/strict_train_A_test_BC_t2only_w4/route_C_robust_frames.csv" \
   && -f "outputs/strict_train_A_test_BC_t2only_w5/route_B_robust_frames.csv" \
   && -f "outputs/strict_train_A_test_BC_t2only_w5/route_C_robust_frames.csv" ]]; then
    compare_results
fi
