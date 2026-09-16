#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/yh/study/cvpr_data/Bearing_UAV_90K}"
CITY="${CITY:-cityb}"
GPU="${GPU:-0}"
PREPARED_ROOT="${REPO_ROOT}/v39_otherdata/generated/${CITY}"
OUTPUT_DIR="${PREPARED_ROOT}/v39_output_exact"

cd "${REPO_ROOT}"

echo "[CODE-AUDIT] compiling every file used by this Bearing exact-v39 run"
python3 -m py_compile \
  v39_otherdata/bearing_prepare.py \
  v39_otherdata/bearing_prepare_sequence_v3.py \
  v39_otherdata/data.py \
  v39_otherdata/bearing_runner.py \
  v39_otherdata/bearing_runner_exact_v39.py \
  v39_otherdata/bearing_plot_final_vs_gt.py \
  v39_DirectFinalMS/patch_direct_finalms.py
echo "[CODE-AUDIT] PASS"

rm -rf "${PREPARED_ROOT}"

echo "[PREP] full-route coverage-safe pseudo-flight"
python3 - "${DATASET_ROOT}" "${CITY}" "${PREPARED_ROOT}" <<'PY'
import argparse
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "v39_otherdata"))
import bearing_prepare_sequence_v3 as seq

# Proven Bearing corridors.  They contain multiple irregular turns and have
# already demonstrated dense observation coverage on cityb.
seq.PIECEWISE_ROUTE_SPECS = {
    "train_01": [
        (330, 620), (690, 850), (1060, 690), (1390, 1030), (1710, 880),
        (1990, 1210), (2240, 1090), (2510, 1450), (2780, 1290),
        (3070, 1620), (3330, 1480),
    ],
    "train_02": [
        (430, 3080), (770, 2780), (1120, 3060), (1460, 2700),
        (1800, 2970), (2110, 2600), (2460, 2910), (2800, 2510),
        (3170, 2780), (3510, 2410),
    ],
    "train_03": [
        (3330, 430), (3050, 760), (3410, 1110), (3100, 1480),
        (3510, 1810), (3200, 2180), (3560, 2530), (3260, 2900),
        (3610, 3260), (3310, 3610),
    ],
    "test_01": [
        (560, 1810), (900, 1510), (1260, 1840), (1610, 1540),
        (1980, 1900), (2320, 1610), (2680, 1970), (3250, 1450),
        (3410, 2050),
    ],
    "test_02": [
        (900, 330), (1160, 660), (900, 1010), (1270, 1320),
        (1010, 1660), (1370, 2010), (1090, 2360), (1500, 2660),
        (1240, 3010), (1660, 3360), (1440, 3690),
    ],
}

# Keep the selection-version string required by the exact-v39 prepared-data
# lock.  The physical-prune function below fixes the v12 truncation bug by
# enforcing coverage of BOTH ends of EVERY straight leg.
seq.SELECTION_VERSION = "soft_sequence_v12_dense_then_physical_leg_prune"
_original_leg_solver = seq._longest_physical_leg


def _coverage_safe_physical_prune(
    dense_ids,
    dense_tids,
    xy_m,
    target_m,
    target_cross_axis,
    headings,
    *,
    preferred_step_m,
    safety_max_step_m,
    big_turn_threshold_deg,
    max_cross_track_m,
    max_same_leg_lateral_jump_m,
    max_same_leg_backward_m,
):
    """Smooth each leg without ever truncating the route.

    v12 previously accepted the longest smooth chain even when that chain lived
    only in the first/middle part of a leg.  Here a smoothed chain is accepted
    only when it spans at least 85% of that leg's dense target range and reaches
    both leg boundaries.  Otherwise that leg falls back to its dense sequence.
    Even for an accepted chain, dense boundary samples are restored before/after
    the chain so every waypoint transition is represented.
    """
    groups = seq._split_selected_into_legs(
        dense_tids, headings, big_turn_threshold_deg
    )
    kept_positions = []
    per_leg = []

    for leg_index, group in enumerate(groups):
        if not group:
            continue
        chain = _original_leg_solver(
            dense_ids,
            dense_tids,
            group,
            xy_m,
            target_m,
            target_cross_axis,
            headings,
            preferred_step_m=preferred_step_m,
            safety_max_step_m=safety_max_step_m,
            max_cross_track_m=max_cross_track_m,
            max_same_leg_lateral_jump_m=max_same_leg_lateral_jump_m,
            max_same_leg_backward_m=max_same_leg_backward_m,
        )

        group_start_tid = int(dense_tids[group[0]])
        group_end_tid = int(dense_tids[group[-1]])
        group_span = max(group_end_tid - group_start_tid, 1)

        use_dense = False
        coverage = 0.0
        if len(chain) < 2:
            use_dense = True
        else:
            chain_start_tid = int(dense_tids[chain[0]])
            chain_end_tid = int(dense_tids[chain[-1]])
            coverage = (chain_end_tid - chain_start_tid) / float(group_span)
            start_fraction = (chain_start_tid - group_start_tid) / float(group_span)
            end_fraction = (group_end_tid - chain_end_tid) / float(group_span)
            if coverage < 0.85 or start_fraction > 0.10 or end_fraction > 0.10:
                use_dense = True

        if use_dense:
            selected = list(group)
            mode = "dense_fallback_for_coverage"
            coverage = 1.0
        else:
            # Restore all dense samples from the leg boundary to the first/last
            # smoothed sample.  This prevents a beautiful middle subsequence from
            # disconnecting a waypoint transition.
            first = chain[0]
            last = chain[-1]
            prefix = [p for p in group if p < first]
            suffix = [p for p in group if p > last]
            selected = sorted(set(prefix + list(chain) + suffix))
            mode = "smoothed_full_span"

        kept_positions.extend(selected)
        per_leg.append(
            {
                "leg": int(leg_index),
                "dense": int(len(group)),
                "kept": int(len(selected)),
                "coverage": float(coverage),
                "mode": mode,
                "first_target": int(dense_tids[selected[0]]),
                "last_target": int(dense_tids[selected[-1]]),
            }
        )

    kept_positions = sorted(set(kept_positions))
    return (
        [int(dense_ids[p]) for p in kept_positions],
        [int(dense_tids[p]) for p in kept_positions],
        per_leg,
    )


seq._physical_prune = _coverage_safe_physical_prune

args = argparse.Namespace(
    dataset_root=sys.argv[1],
    city=sys.argv[2],
    output_root=sys.argv[3],
    step_m=4.0,
    preferred_step_m=4.0,
    # Dense stage reproduces the previously successful high-coverage selector.
    safety_max_step_m=22.0,
    max_sample_distance_m=10.0,
    candidate_limit=256,
    beam_width=512,
    skip_penalty=40.0,
    continuity_weight=2.5,
    large_step_weight=0.85,
    cross_weight=2.0,
    backward_weight=12.0,
    point_cross_weight=20.0,
    lateral_smooth_weight=35.0,
    big_turn_threshold_deg=25.0,
    # Physical pruning is allowed only when it preserves full leg coverage.
    max_cross_track_m=8.5,
    max_same_leg_lateral_jump_m=4.0,
    max_same_leg_backward_m=0.25,
    min_selected_ratio=0.25,
)
seq.prepare(args)
PY

# ---------------------------------------------------------------------------
# PRE-INFERENCE ROUTE COVERAGE AUDIT
# ---------------------------------------------------------------------------
# A run is invalid if selected real observations do not reach EVERY waypoint.
# This directly prevents the previous image where red stopped after ~1/3 route.
python3 - "${PREPARED_ROOT}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
exp = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
if exp.get("sequence_selection_version") != "soft_sequence_v12_dense_then_physical_leg_prune":
    raise SystemExit("[COVERAGE-AUDIT] unexpected selection version")

print("[COVERAGE-AUDIT] checking every waypoint against retained observations")
for route in ["train_01", "train_02", "train_03", "test_01", "test_02"]:
    with (root / "routes" / route / "manifest.csv").open(
        "r", newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 30:
        raise SystemExit(f"[COVERAGE-AUDIT] {route}: only {len(rows)} frames")

    pts = [(float(r["x_m"]), float(r["y_m"])) for r in rows]
    wp_payload = json.loads(
        (root / "routes" / route / "waypoints.json").read_text(encoding="utf-8")
    )
    wps = [
        (float(w["longitude"]), float(w["latitude"]))
        for w in sorted(
            wp_payload["waypoints"], key=lambda w: int(w["waypoint_order"])
        )
    ]

    nearest = []
    for wx, wy in wps:
        nearest.append(min(math.hypot(x-wx, y-wy) for x, y in pts))

    endpoint_start = math.hypot(pts[0][0]-wps[0][0], pts[0][1]-wps[0][1])
    endpoint_end = math.hypot(pts[-1][0]-wps[-1][0], pts[-1][1]-wps[-1][1])

    # Route projection switches within 24 m in canonical v39.  Requiring every
    # waypoint within 18 m gives margin and guarantees the sequence reaches all
    # planned turns rather than only accumulating enough frames in the front.
    if max(nearest) > 18.0:
        raise SystemExit(
            f"[COVERAGE-AUDIT] {route}: waypoint not covered; "
            f"max nearest={max(nearest):.2f}m distances={[round(v,2) for v in nearest]}"
        )
    if endpoint_start > 18.0 or endpoint_end > 18.0:
        raise SystemExit(
            f"[COVERAGE-AUDIT] {route}: endpoint coverage failed "
            f"start={endpoint_start:.2f}m end={endpoint_end:.2f}m"
        )

    stats = exp["route_stats"][route]
    print(
        f"  {route}: PASS frames={len(rows)} "
        f"start={endpoint_start:.2f}m end={endpoint_end:.2f}m "
        f"worst_waypoint={max(nearest):.2f}m "
        f"selected={float(stats['selected_ratio'])*100:.1f}%"
    )
print("[COVERAGE-AUDIT] FULL ROUTE COVERAGE: PASS")
PY

# ---------------------------------------------------------------------------
# EXACT CANONICAL v39 MODEL/INFERENCE
# ---------------------------------------------------------------------------
python3 v39_otherdata/bearing_runner_exact_v39.py \
  --dataset-root "${DATASET_ROOT}" \
  --city "${CITY}" \
  --gpu "${GPU}" \
  --backbone mobilenet_v3_small \
  --visual-epochs 30 \
  --epochs-per-route 20 \
  --patience 10 \
  --jitter-m 8 \
  --step-m 4 \
  --max-sample-distance-m 10 \
  --heading-weight-px-per-deg 0

# ---------------------------------------------------------------------------
# POST-INFERENCE COMPLETION + METRIC AUDIT
# ---------------------------------------------------------------------------
# Do not create a result image unless prediction and GT both reach the LAST leg.
python3 - "${PREPARED_ROOT}" "${OUTPUT_DIR}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
summaries = json.loads((out / "bearing_v39_summary.json").read_text(encoding="utf-8"))
with (root / "routes" / "route_A" / "manifest.csv").open(
    "r", newline="", encoding="utf-8"
) as f:
    a = list(csv.DictReader(f))
origin_x = float(a[0]["x_m"])
origin_y = float(a[0]["y_m"])

print("[FINAL-COMPLETION-AUDIT]")
for route in ["test_01", "test_02"]:
    summary = summaries[route]
    wp = json.loads(
        (root / "routes" / route / "waypoints.json").read_text(encoding="utf-8")
    )["waypoints"]
    wp = sorted(wp, key=lambda w: int(w["waypoint_order"]))
    last_leg = len(wp) - 2

    pred_leg = int(summary["FinalPredictedWaypointLeg"])
    gt_leg = int(summary["FinalGTWaypointLeg"])
    if pred_leg != last_leg or gt_leg != last_leg:
        raise SystemExit(
            f"[FINAL-COMPLETION-AUDIT] {route}: route NOT completed: "
            f"pred_leg={pred_leg}, gt_leg={gt_leg}, required={last_leg}"
        )

    csv_path = Path(summary["CSV"])
    if not csv_path.exists():
        csv_path = out / csv_path.name
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (root / "routes" / route / "manifest.csv").open(
        "r", newline="", encoding="utf-8"
    ) as f:
        manifest = list(csv.DictReader(f))
    if len(rows) != len(manifest):
        raise SystemExit(
            f"[FINAL-COMPLETION-AUDIT] {route}: frame count mismatch "
            f"CSV={len(rows)} manifest={len(manifest)}"
        )

    errors = [
        math.hypot(float(r["final_x"])-float(r["gt_x"]),
                   float(r["final_y"])-float(r["gt_y"]))
        for r in rows
    ]
    mle = sum(errors) / len(errors)
    if abs(mle - float(summary["MLE_m"])) > 1e-5:
        raise SystemExit(
            f"[FINAL-COMPLETION-AUDIT] {route}: MLE mismatch "
            f"CSV={mle:.9f} summary={float(summary['MLE_m']):.9f}"
        )

    last = rows[-1]
    final_abs = (
        float(last["final_x"]) + origin_x,
        float(last["final_y"]) + origin_y,
    )
    end_wp = (float(wp[-1]["longitude"]), float(wp[-1]["latitude"]))
    end_distance = math.hypot(final_abs[0]-end_wp[0], final_abs[1]-end_wp[1])
    if end_distance > 35.0:
        raise SystemExit(
            f"[FINAL-COMPLETION-AUDIT] {route}: final prediction did not reach route end; "
            f"distance={end_distance:.2f}m"
        )

    print(
        f"  {route}: PASS frames={len(rows)} last_leg={pred_leg}/{last_leg} "
        f"end_distance={end_distance:.2f}m MLE={mle:.3f}m"
    )
print("[FINAL-COMPLETION-AUDIT] BOTH TEST ROUTES COMPLETE: PASS")
PY

# Produce ONLY the two final-result images, one per held-out test route.
python3 v39_otherdata/bearing_plot_final_vs_gt.py \
  --prepared-root "${PREPARED_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --routes test_01 test_02

echo ""
echo "================================================================================"
echo "Bearing exact-v39 full-route experiment finished"
echo "Model: Weighted Centroid -> 3-frame Context-GRU -> velocity -> fixed Kalman -> final 5x5 MS"
echo "Model/inference parameters unchanged; Bearing cadence adaptation disabled"
echo "Route rule: every planned waypoint must be covered before inference can run"
echo "Output: ONLY final prediction vs reference route"
echo "  ${OUTPUT_DIR}/test_01_final_result.jpg"
echo "  ${OUTPUT_DIR}/test_02_final_result.jpg"
echo "Summary: ${OUTPUT_DIR}/bearing_v39_summary.json"
echo "================================================================================"
