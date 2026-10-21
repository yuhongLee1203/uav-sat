"""Build ID96 episodes against predeclared routes, never test-GT waypoints.

The c4m metadata has independent geotagged images, not recorded video.  GT is
used offline to choose nearby observations for a fixed route and later for
metrics; the waypoint inputs themselves are fixed before any held-out sample
is selected.  This remains a *synthetic* episode benchmark, not official
Bearing-UAV Table 6 or a real flight trajectory.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v39_otherdata"))
import bearing_prepare as base
import bearing_prepare_official96_temporal as official
from bearing_multicity_routes import OFFICIAL_TEST_ROUTES

VERSION = "official_rsi96_seed42_fixed_waypoint_temporal_v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=("citya", "cityb", "cityc", "cityd"), required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="City prepared directory; must not already exist")
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/yh/study/cvpr_data/Bearing_UAV_90K"))
    parser.add_argument("--max-target-distance-m", type=float, default=40.0)
    parser.add_argument("--resume-incomplete", action="store_true",
                        help="Retry this script's incomplete output after a sparse split error")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.max_target_distance_m <= 0:
        parser.error("max-target-distance-m must be positive")
    if output.exists():
        if not args.resume_incomplete or (output / "experiment.json").exists():
            parser.error("output already exists; refusing to overwrite: " + str(output))
        existing = {str(path.relative_to(output)) for path in output.rglob("*")
                    if path.is_file()}
        allowed = {"routes/train_01/manifest.csv", "routes/train_01/waypoints.json"}
        if not existing.issubset(allowed):
            parser.error("incomplete output has unexpected files: " + str(sorted(existing - allowed)))
    if output.name != "prepared" or output.parent.name != args.city:
        parser.error("output must end in <root>/<city>/prepared")

    fixed_points = {
        1: ("val_01", base.ROUTE_SPECS["train_02"]),
        2: ("test_01", OFFICIAL_TEST_ROUTES[args.city]["test_01"]),
        3: ("test_02", OFFICIAL_TEST_ROUTES[args.city]["test_02"]),
    }
    original = official.select_spatial_episode

    def select_fixed(rows, *, preferred_step_m, max_frame_step_m,
                     target_frames, route_slot, used_paths):
        name, points = fixed_points[route_slot]
        planned = base._scale_route(points, base.REFERENCE_SIZE, base.REFERENCE_SIZE)
        selected, audit = official.select_episode(
            rows, planned, step_m=12.0,
            max_target_distance_m=float(args.max_target_distance_m),
            max_frame_step_m=100.0, used_paths=used_paths)
        audit.update({"construction": "fixed_predeclared_waypoint_route",
                      "waypoint_source": name,
                      "waypoints_derived_from_selected_GT": False})
        return selected, planned, audit

    prep_args = official.build_parser().parse_args([
        "--dataset-root", str(args.dataset_root.resolve()),
        "--city", args.city,
        "--output-root", str(output),
        "--step-m", "12",
        "--max-target-distance-m", str(args.max_target_distance_m),
        "--max-frame-step-m", "100",
    ])
    official.select_spatial_episode = select_fixed
    try:
        official.prepare(prep_args)
    finally:
        official.select_spatial_episode = original

    experiment = output / "experiment.json"
    data = json.loads(experiment.read_text())
    data["sequence_selection_version"] = VERSION
    data["heldout_waypoints_from_GT"] = False
    data["episode_construction_uses_GT_offline"] = True
    data["official_table6_reproduction"] = False
    data["max_target_distance_m"] = float(args.max_target_distance_m)
    experiment.write_text(json.dumps(data, indent=2) + "\n")
    (output / "fixed_route_audit.json").write_text(json.dumps({
        "version": VERSION,
        "city": args.city,
        "metadata_csv": data["metadata_csv"],
        "official_split_seed": data["official_split_seed"],
        "test_waypoints_derived_from_test_GT": False,
        "episode_construction_uses_GT_offline": True,
        "max_target_distance_m": float(args.max_target_distance_m),
        "route_stats": data["route_stats"],
    }, indent=2) + "\n")
    print("[V40 FIXED ROUTES]", output, flush=True)


if __name__ == "__main__":
    main()
