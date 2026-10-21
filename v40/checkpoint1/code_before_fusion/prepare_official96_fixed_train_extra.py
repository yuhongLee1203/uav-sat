"""Additional train-split episodes on predeclared v39 route geometry."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v39_otherdata"))
import bearing_prepare as base
import bearing_prepare_official96_temporal as official


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=("citya", "cityb", "cityc", "cityd"), required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Root ending in <city>; must not already exist")
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/yh/study/cvpr_data/Bearing_UAV_90K"))
    parser.add_argument("--shifted", action="store_true",
                        help="Predeclared off-route training episodes with varied frame spacing")
    parser.add_argument("--grid", action="store_true",
                        help="Twelve predeclared horizontal/vertical training routes across the map")
    args = parser.parse_args()
    if args.shifted and args.grid:
        parser.error("choose either shifted routes or grid routes")
    output = args.output_root.resolve()
    if output.exists() or output.name != args.city:
        parser.error("output must be a new directory ending in city name")
    prepared = args.prepared_root.resolve() / args.city / "prepared"
    fixed_audit = json.loads((prepared / "fixed_route_audit.json").read_text())
    if fixed_audit["version"] != "official_rsi96_seed42_fixed_waypoint_temporal_v1":
        parser.error("prepared route protocol mismatch")

    metadata = pd.read_csv(fixed_audit["metadata_csv"])
    metadata["_official_index"] = np.arange(len(metadata), dtype=np.int64)
    train_indices = set(map(int, official.official_split_indices(len(metadata))["train"]))
    city_rows = base._city_rows(metadata, args.city)
    city_rows["_city_source_index"] = np.arange(len(city_rows), dtype=np.int64)
    pool = city_rows[city_rows["_official_index"].isin(train_indices)].reset_index(drop=True)
    dataset_root = args.dataset_root.resolve()
    basename_index = base._build_basename_index(dataset_root, args.city)
    with (prepared / "routes" / "train_01" / "manifest.csv").open(newline="") as file:
        used_image_paths = {row["image_path"] for row in csv.DictReader(file)}
    used_metadata_paths = set()
    for value in pool["target_path"].astype(str):
        if str(base._resolve_image_path(value, dataset_root, args.city, basename_index)) in used_image_paths:
            used_metadata_paths.add(value)

    audits = {}
    all_indices = set()
    specs = (("train_02_shiftp20_step20", "train_02", 20.0, 20.0),
             ("train_02_shiftn20_step25", "train_02", -20.0, 25.0),
             ("train_03_shiftp25_step30", "train_03", 25.0, 30.0),
             ("train_03_shiftn25_step30", "train_03", -25.0, 30.0)) if args.shifted else (
             ("train_03_step20", "train_03", 0.0, 20.0),
             ("train_03_step30", "train_03", 0.0, 30.0))
    route_plans = dict(base.ROUTE_SPECS)
    if args.grid:
        specs = []
        for axis in ("h", "v"):
            for number, line in enumerate(np.linspace(512.0, 3584.0, 6)):
                name = "train_grid_%s%02d" % (axis, number)
                points = [(256.0, line), (3840.0, line)] if axis == "h" else [
                    (line, 256.0), (line, 3840.0)]
                route_plans[name] = points if number % 2 == 0 else list(reversed(points))
                specs.append((name, name, 20.0 if number % 2 == 0 else -20.0,
                              20.0 if number % 3 == 0 else 30.0))
    for name, route_name, offset_m, step in specs:
        planned = base._scale_route(route_plans[route_name],
                                    base.REFERENCE_SIZE, base.REFERENCE_SIZE)
        tangent = np.gradient(planned, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        # This shift is fixed by the experiment design, never by an image GT.
        observation_path = planned + normal * (offset_m / base.MPP)
        selected, audit = official.select_episode(
            pool, observation_path, step_m=step, max_target_distance_m=40.0,
            max_frame_step_m=100.0, used_paths=used_metadata_paths)
        indices = set(map(int, selected["_official_index"]))
        if not indices <= train_indices or indices & all_indices:
            raise RuntimeError("extra route has split leakage or duplicate rows")
        all_indices.update(indices)
        paths = [base._resolve_image_path(value, dataset_root, args.city, basename_index)
                 for value in selected["target_path"]]
        base._write_route(output / name, name, planned, selected, paths)
        audits[name] = {**audit, "official_split": "train",
                        "predeclared_observation_offset_m": offset_m,
                        "waypoints_derived_from_selected_GT": False}
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps({
        "city": args.city,
        "no_cross_split_samples": True,
        "no_duplicate_extra_rows": True,
        "predeclared_train_waypoints": True,
        "official_split_seed": 42,
        "episodes": audits,
    }, indent=2) + "\n")
    print("[V40 FIXED TRAIN EXTRA]", output, json.dumps(audits), flush=True)


if __name__ == "__main__":
    main()
