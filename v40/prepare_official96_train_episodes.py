"""Build train-only pseudo-episodes from the official c4m/ID96 train split.

These are ordered independent Bearing-UAV observations, not recorded video.
No validation/test image can enter the additional training episodes.
"""
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
    parser.add_argument("--base-suite", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/yh/study/cvpr_data/Bearing_UAV_90K"))
    parser.add_argument("--target-frames", type=int, default=120)
    args = parser.parse_args()
    if args.target_frames < 24:
        parser.error("target-frames must be >=24")
    prepared = args.base_suite / args.city / "prepared"
    audit = json.loads((prepared / "official96_split_audit.json").read_text())
    if (audit.get("adapter_version") != official.ADAPTER_VERSION
            or audit.get("official_dataset_id") != 96
            or not audit.get("no_cross_split_samples")):
        raise RuntimeError("base suite is not the audited official96 temporal adapter")
    metadata = pd.read_csv(base._find_metadata(args.dataset_root))
    metadata["_official_index"] = np.arange(len(metadata), dtype=np.int64)
    train_indices = set(official.official_split_indices(len(metadata))["train"].tolist())
    city_rows = base._city_rows(metadata, args.city)
    city_rows["_city_source_index"] = np.arange(len(city_rows), dtype=np.int64)
    pool = city_rows[city_rows["_official_index"].isin(train_indices)].reset_index(drop=True)
    used = set()
    with (prepared / "routes" / "train_01" / "manifest.csv").open(newline="") as f:
        used.update(row["source_target_path"] for row in csv.DictReader(f))
    basename_index = base._build_basename_index(args.dataset_root, args.city)
    summaries = {}
    for slot, step in enumerate((8.0, 20.0, 30.0), start=1):
        name = "train_spatial_%02d" % int(step)
        selected, planned, episode_audit = official.select_spatial_episode(
            pool, preferred_step_m=step, max_frame_step_m=45.0,
            target_frames=args.target_frames, route_slot=slot,
            used_paths=used)
        if not set(selected["_official_index"].astype(int)).issubset(train_indices):
            raise RuntimeError("non-train official row selected")
        paths = [base._resolve_image_path(value, args.dataset_root, args.city,
                                          basename_index)
                 for value in selected["target_path"]]
        destination = args.output_root / args.city / name
        if destination.exists():
            raise FileExistsError("refusing to overwrite existing episode: " + str(destination))
        base._write_route(destination, name, planned, selected, paths)
        summaries[name] = episode_audit
        print("[OFFICIAL96 TRAIN EPISODE]", args.city, name,
              json.dumps(episode_audit), flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / args.city / "audit.json").write_text(json.dumps({
        "city": args.city,
        "source_split": "c4m official ID96 seed42 train only",
        "base_suite": str(args.base_suite),
        "original_train_images_excluded": True,
        "no_cross_split_samples": True,
        "episode_stats": summaries,
        "warning": "Pseudo-temporal ordering of independent images, not UAV video",
    }, indent=2))


if __name__ == "__main__":
    main()
