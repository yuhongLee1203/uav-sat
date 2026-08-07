#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Fixed 3-frame RTL-CRF experiment
# ---------------------------------------------------------------------------
# Fair comparison against the already completed 5-frame no-position-scale run:
#   - same Route-A-only visual checkpoint
#   - same Route A train/validation boundary
#   - same Route B/C evaluation protocol
#   - same jitter and model hyperparameters
#   - same raw-meter (no /10) features
#   - only TEMPORAL_WINDOW changes: 5 -> 3
#
# Existing 5-frame results are kept at:
#   outputs/strict_train_A_test_BC_no_position_scale/
#
# New 3-frame results are written to:
#   outputs/strict_train_A_test_BC_no_position_scale_w3/

BASE_5_DIR="outputs/strict_train_A_test_BC_no_position_scale"
W3_DIR="outputs/strict_train_A_test_BC_no_position_scale_w3"

BASE_VISUAL="${BASE_5_DIR}/checkpoints/visual_retrieval_A_only.pt"
W3_VISUAL="${W3_DIR}/checkpoints/visual_retrieval_A_only.pt"

mkdir -p "${W3_DIR}/checkpoints"

if [[ ! -f "${BASE_VISUAL}" ]]; then
    echo "ERROR: missing the existing Route-A-only visual checkpoint:" >&2
    echo "  ${BASE_VISUAL}" >&2
    echo "The 3-frame ablation must reuse the exact visual checkpoint from the 5-frame run." >&2
    exit 2
fi

# Always copy the exact same visual checkpoint before a fresh 3-frame run.
# This never modifies the completed 5-frame result directory.
cp -p "${BASE_VISUAL}" "${W3_VISUAL}"

echo "============================================================"
echo "Fixed 3-frame second-order RTL-CRF"
echo "Visual checkpoint reused from 5-frame experiment:"
echo "  ${BASE_VISUAL}"
echo "  -> ${W3_VISUAL}"
echo "3-frame output:"
echo "  ${W3_DIR}"
echo "============================================================"

# robust_tracker.py currently enters visual training whenever mode is train/train_eval.
# --resume is therefore required when --visual-epochs 0 is used, so it loads
# the copied best visual checkpoint instead of creating an empty zero-epoch run.
HAS_RESUME=0
for arg in "$@"; do
    if [[ "$arg" == "--resume" ]]; then
        HAS_RESUME=1
        break
    fi
done

if [[ ${HAS_RESUME} -eq 1 ]]; then
    python3 robust_tracker.py "$@"
else
    python3 robust_tracker.py --resume "$@"
fi

# ---------------------------------------------------------------------------
# Automatically compare 3-frame and existing 5-frame results on COMMON frames.
# This avoids giving the 3-frame run two extra evaluation frames simply because
# its temporal warm-up is shorter.
# ---------------------------------------------------------------------------
if [[ -f "${BASE_5_DIR}/route_B_robust_frames.csv" \
   && -f "${BASE_5_DIR}/route_C_robust_frames.csv" \
   && -f "${W3_DIR}/route_B_robust_frames.csv" \
   && -f "${W3_DIR}/route_C_robust_frames.csv" ]]; then

python3 - <<'PY'
from pathlib import Path
import json

import numpy as np
import pandas as pd

BASE = Path("outputs/strict_train_A_test_BC_no_position_scale")
W3 = Path("outputs/strict_train_A_test_BC_no_position_scale_w3")

def metric_block(prediction, gt):
    prediction = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    error = np.linalg.norm(prediction - gt, axis=1)

    if len(prediction) > 1:
        predicted_step = np.diff(prediction, axis=0)
        gt_step = np.diff(gt, axis=0)
        rpe = np.linalg.norm(predicted_step - gt_step, axis=1)
        gt_step_length = np.linalg.norm(gt_step, axis=1)
        jump_threshold = float(np.percentile(gt_step_length, 99)) + 3.0
        jump_rate = float(
            (np.linalg.norm(predicted_step, axis=1) > jump_threshold).mean() * 100.0
        )
    else:
        rpe = np.zeros(1, dtype=np.float64)
        jump_rate = 0.0

    return {
        "frames": int(len(error)),
        "MLE_m": float(error.mean()),
        "P90_m": float(np.percentile(error, 90)),
        "RPE_m": float(rpe.mean()),
        "JumpRate_pct": float(jump_rate),
        "LSR@5_pct": float((error <= 5.0).mean() * 100.0),
        "LSR@10_pct": float((error <= 10.0).mean() * 100.0),
        "LSR@15_pct": float((error <= 15.0).mean() * 100.0),
    }

comparison = {
    "comparison": "3-frame vs existing 5-frame RTL-CRF",
    "fairness": (
        "Metrics are recomputed only on frame IDs shared by both runs. "
        "Both runs use the same Route-A-only visual checkpoint and the same "
        "Route-A train/validation boundary; only temporal window length differs."
    ),
    "routes": {},
}

print()
print("=" * 92)
print("3-FRAME vs EXISTING 5-FRAME RTL-CRF — COMMON-FRAME COMPARISON")
print("=" * 92)

for route in ("route_B", "route_C"):
    old = pd.read_csv(BASE / f"{route}_robust_frames.csv")
    new = pd.read_csv(W3 / f"{route}_robust_frames.csv")

    common = old[
        [
            "frame_id",
            "gt_x",
            "gt_y",
            "temporal_x",
            "temporal_y",
        ]
    ].merge(
        new[
            [
                "frame_id",
                "temporal_x",
                "temporal_y",
            ]
        ],
        on="frame_id",
        suffixes=("_w5", "_w3"),
        how="inner",
        validate="one_to_one",
    ).sort_values("frame_id")

    if common.empty:
        raise RuntimeError(f"{route}: no common frames between 3-frame and 5-frame runs")

    gt = common[["gt_x", "gt_y"]].to_numpy(np.float64)
    pred5 = common[["temporal_x_w5", "temporal_y_w5"]].to_numpy(np.float64)
    pred3 = common[["temporal_x_w3", "temporal_y_w3"]].to_numpy(np.float64)

    m5 = metric_block(pred5, gt)
    m3 = metric_block(pred3, gt)

    comparison["routes"][route] = {
        "window_5": m5,
        "window_3": m3,
    }

    print()
    print(route.upper(), f" common frames={len(common)}")
    print("-" * 92)
    print(f"{'metric':<22}{'5-frame':>16}{'3-frame':>16}{'3 - 5':>16}")
    for key in (
        "MLE_m",
        "P90_m",
        "RPE_m",
        "JumpRate_pct",
        "LSR@5_pct",
        "LSR@10_pct",
        "LSR@15_pct",
    ):
        a = float(m5[key])
        b = float(m3[key])
        print(f"{key:<22}{a:>16.4f}{b:>16.4f}{(b-a):>16.4f}")

comparison_path = W3 / "comparison_3frame_vs_5frame.json"
comparison_path.write_text(
    json.dumps(comparison, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print()
print("=" * 92)
print(f"comparison saved to: {comparison_path}")
print("=" * 92)
PY

else
    echo
    echo "NOTE: 3-frame training/evaluation finished, but automatic comparison was skipped"
    echo "because one or more existing 5-frame per-frame CSV files are missing."
    echo "Expected:"
    echo "  ${BASE_5_DIR}/route_B_robust_frames.csv"
    echo "  ${BASE_5_DIR}/route_C_robust_frames.csv"
fi
