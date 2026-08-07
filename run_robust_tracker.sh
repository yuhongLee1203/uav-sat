#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ===========================================================================
# Fixed 4-frame RTL-CRF experiment
# ===========================================================================
#
# Fair temporal-window comparison:
#
#   3-frame:
#     outputs/strict_train_A_test_BC_no_position_scale_w3/
#
#   4-frame:
#     outputs/strict_train_A_test_BC_no_position_scale_w4/
#
#   5-frame:
#     outputs/strict_train_A_test_BC_no_position_scale/
#
# All variants:
#   - reuse the exact same Route-A-only visual retrieval checkpoint
#   - train temporal model only on Route A
#   - evaluate on unseen Route B and Route C
#   - use raw-meter features (no POSITION_SCALE_M=/10)
#   - use the same split guard (=5)
#
# The only intended architecture difference is temporal context length:
#
#   3 frames -> 1 T2 term
#   4 frames -> 2 overlapping T2 terms
#   5 frames -> 3 overlapping T2 terms
#
# Existing 3-frame and 5-frame results are NEVER deleted or overwritten.
# ===========================================================================

BASE_5_DIR="outputs/strict_train_A_test_BC_no_position_scale"
W3_DIR="outputs/strict_train_A_test_BC_no_position_scale_w3"
W4_DIR="outputs/strict_train_A_test_BC_no_position_scale_w4"

BASE_VISUAL="${BASE_5_DIR}/checkpoints/visual_retrieval_A_only.pt"
W4_VISUAL="${W4_DIR}/checkpoints/visual_retrieval_A_only.pt"

mkdir -p "${W4_DIR}/checkpoints"

if [[ ! -f "${BASE_VISUAL}" ]]; then
    echo "ERROR: missing the existing Route-A-only visual checkpoint:" >&2
    echo "  ${BASE_VISUAL}" >&2
    echo "The 4-frame run must reuse the exact visual checkpoint from the 5-frame run." >&2
    exit 2
fi

# Copy exactly the same visual checkpoint into the 4-frame experiment folder.
cp -p "${BASE_VISUAL}" "${W4_VISUAL}"

echo "============================================================================"
echo "FIXED 4-FRAME SECOND-ORDER RTL-CRF"
echo "============================================================================"
echo "Visual checkpoint:"
echo "  ${BASE_VISUAL}"
echo "4-frame output:"
echo "  ${W4_DIR}"
echo
echo "Expected temporal structure:"
echo "  T1(frame0, frame1)"
echo "  T2(frame0, frame1, frame2)"
echo "  T2(frame1, frame2, frame3)"
echo "============================================================================"

# robust_tracker.py enters the visual-training function in train/train_eval mode.
# With --visual-epochs 0, --resume is needed so it loads the copied visual
# checkpoint rather than attempting an empty zero-epoch visual training run.
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

# ===========================================================================
# Common-frame temporal-window comparison
# ===========================================================================
#
# 3/4/5 windows have different warm-up lengths.  Therefore we do NOT compare
# their summary JSON values blindly.  Instead, metrics below are recomputed
# only on frame IDs present in all available runs.
# ===========================================================================

if [[ -f "${BASE_5_DIR}/route_B_robust_frames.csv" \
   && -f "${BASE_5_DIR}/route_C_robust_frames.csv" \
   && -f "${W4_DIR}/route_B_robust_frames.csv" \
   && -f "${W4_DIR}/route_C_robust_frames.csv" ]]; then

python3 - <<'PY'
from pathlib import Path
import json

import numpy as np
import pandas as pd

W5 = Path("outputs/strict_train_A_test_BC_no_position_scale")
W4 = Path("outputs/strict_train_A_test_BC_no_position_scale_w4")
W3 = Path("outputs/strict_train_A_test_BC_no_position_scale_w3")

JUMP_TOLERANCE_M = 3.0


def read_prediction(directory: Path, route: str, suffix: str) -> pd.DataFrame:
    path = directory / f"{route}_robust_frames.csv"
    frame = pd.read_csv(path)

    required = {
        "frame_id",
        "gt_x",
        "gt_y",
        "temporal_x",
        "temporal_y",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path}: missing columns {missing}")

    return frame[
        [
            "frame_id",
            "gt_x",
            "gt_y",
            "temporal_x",
            "temporal_y",
        ]
    ].rename(
        columns={
            "temporal_x": f"temporal_x_{suffix}",
            "temporal_y": f"temporal_y_{suffix}",
        }
    )


def metric_block(prediction: np.ndarray, gt: np.ndarray):
    prediction = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    error = np.linalg.norm(prediction - gt, axis=1)

    if len(prediction) > 1:
        pred_step = np.diff(prediction, axis=0)
        gt_step = np.diff(gt, axis=0)

        rpe_each = np.linalg.norm(pred_step - gt_step, axis=1)
        gt_step_length = np.linalg.norm(gt_step, axis=1)

        jump_threshold = (
            float(np.percentile(gt_step_length, 99))
            + JUMP_TOLERANCE_M
        )
        jump_rate = float(
            (
                np.linalg.norm(pred_step, axis=1)
                > jump_threshold
            ).mean()
            * 100.0
        )
        rpe = float(rpe_each.mean())
    else:
        rpe = 0.0
        jump_rate = 0.0

    return {
        "frames": int(len(error)),
        "MLE_m": float(error.mean()),
        "P90_m": float(np.percentile(error, 90)),
        "RPE_m": rpe,
        "JumpRate_pct": jump_rate,
        "LSR@5_pct": float((error <= 5.0).mean() * 100.0),
        "LSR@10_pct": float((error <= 10.0).mean() * 100.0),
        "LSR@15_pct": float((error <= 15.0).mean() * 100.0),
    }


have_w3 = (
    (W3 / "route_B_robust_frames.csv").exists()
    and (W3 / "route_C_robust_frames.csv").exists()
)

comparison = {
    "comparison": (
        "4-frame RTL-CRF vs existing 5-frame RTL-CRF"
        + (" and existing 3-frame RTL-CRF" if have_w3 else "")
    ),
    "fairness": (
        "All metrics are recomputed on common frame IDs. "
        "The visual retrieval checkpoint, Route-A training protocol, "
        "Route-B/C testing protocol, split guard, and raw-meter feature "
        "configuration are unchanged; temporal window length is the intended "
        "difference."
    ),
    "routes": {},
}

print()
print("=" * 110)
if have_w3:
    print("3-FRAME vs 4-FRAME vs 5-FRAME RTL-CRF — COMMON-FRAME COMPARISON")
else:
    print("4-FRAME vs 5-FRAME RTL-CRF — COMMON-FRAME COMPARISON")
print("=" * 110)

for route in ("route_B", "route_C"):
    d5 = read_prediction(W5, route, "w5")
    d4 = read_prediction(W4, route, "w4")

    common = d5.merge(
        d4[
            [
                "frame_id",
                "temporal_x_w4",
                "temporal_y_w4",
            ]
        ],
        on="frame_id",
        how="inner",
        validate="one_to_one",
    )

    if have_w3:
        d3 = read_prediction(W3, route, "w3")
        common = common.merge(
            d3[
                [
                    "frame_id",
                    "temporal_x_w3",
                    "temporal_y_w3",
                ]
            ],
            on="frame_id",
            how="inner",
            validate="one_to_one",
        )

    common = common.sort_values("frame_id").reset_index(drop=True)

    if common.empty:
        raise RuntimeError(f"{route}: no common evaluation frames")

    gt = common[["gt_x", "gt_y"]].to_numpy(np.float64)

    p5 = common[
        ["temporal_x_w5", "temporal_y_w5"]
    ].to_numpy(np.float64)
    p4 = common[
        ["temporal_x_w4", "temporal_y_w4"]
    ].to_numpy(np.float64)

    m5 = metric_block(p5, gt)
    m4 = metric_block(p4, gt)

    route_result = {
        "window_5": m5,
        "window_4": m4,
    }

    if have_w3:
        p3 = common[
            ["temporal_x_w3", "temporal_y_w3"]
        ].to_numpy(np.float64)
        m3 = metric_block(p3, gt)
        route_result["window_3"] = m3

    comparison["routes"][route] = route_result

    print()
    print(f"{route.upper()}  common frames={len(common)}")
    print("-" * 110)

    metrics = (
        "MLE_m",
        "P90_m",
        "RPE_m",
        "JumpRate_pct",
        "LSR@5_pct",
        "LSR@10_pct",
        "LSR@15_pct",
    )

    if have_w3:
        print(
            f"{'metric':<22}"
            f"{'3-frame':>16}"
            f"{'4-frame':>16}"
            f"{'5-frame':>16}"
            f"{'4 - 5':>16}"
        )
        for key in metrics:
            a3 = float(route_result["window_3"][key])
            a4 = float(route_result["window_4"][key])
            a5 = float(route_result["window_5"][key])
            print(
                f"{key:<22}"
                f"{a3:>16.4f}"
                f"{a4:>16.4f}"
                f"{a5:>16.4f}"
                f"{(a4-a5):>16.4f}"
            )
    else:
        print(
            f"{'metric':<22}"
            f"{'4-frame':>16}"
            f"{'5-frame':>16}"
            f"{'4 - 5':>16}"
        )
        for key in metrics:
            a4 = float(route_result["window_4"][key])
            a5 = float(route_result["window_5"][key])
            print(
                f"{key:<22}"
                f"{a4:>16.4f}"
                f"{a5:>16.4f}"
                f"{(a4-a5):>16.4f}"
            )

comparison_path = W4 / "comparison_temporal_window_3_4_5.json"
comparison_path.write_text(
    json.dumps(
        comparison,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("=" * 110)
print(f"comparison saved to: {comparison_path}")
print("=" * 110)
PY

else
    echo
    echo "NOTE: 4-frame training/evaluation finished, but automatic comparison was skipped."
    echo "Required existing 5-frame CSVs:"
    echo "  ${BASE_5_DIR}/route_B_robust_frames.csv"
    echo "  ${BASE_5_DIR}/route_C_robust_frames.csv"
    echo "Required new 4-frame CSVs:"
    echo "  ${W4_DIR}/route_B_robust_frames.csv"
    echo "  ${W4_DIR}/route_C_robust_frames.csv"
fi
