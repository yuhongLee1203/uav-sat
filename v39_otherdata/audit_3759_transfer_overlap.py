#!/usr/bin/env python3
"""Audit source-training image reuse in a different official96 split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def paths(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        return {row["image_path"] for row in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"source_training_split": "soft_sequence_v13_multicity_auto_train_fullroute",
              "target_split": "official_rsi96_seed42_temporal_v1", "cities": {}}
    for city in ("citya", "cityb", "cityc", "cityd"):
        source = paths(args.source / city / "prepared/routes/train_01/manifest.csv")
        target = {}
        for route in ("val_01", "test_01", "test_02"):
            selected = paths(args.official / city / "prepared/routes" / route / "manifest.csv")
            target[route] = {"frames": len(selected), "source_train_overlap": len(source & selected),
                             "overlapping_image_paths": sorted(source & selected)}
        report["cities"][city] = {"source_train_frames": len(source), "target_routes": target}
    report["total_test_overlap"] = sum(
        report["cities"][city]["target_routes"][route]["source_train_overlap"]
        for city in report["cities"] for route in ("test_01", "test_02"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("[3759 SPLIT OVERLAP] test frames reused in source training:",
          report["total_test_overlap"], flush=True)


if __name__ == "__main__":
    main()
