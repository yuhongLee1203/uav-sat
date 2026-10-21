"""Verify changing validation GT changes metrics but never predicted positions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import image_only_citya as exp
from train_multicity_routes import evaluate_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    exp.configure("citya")
    if (exp.config.GT_FRONT18_ORACLE or exp.config.GT_FINAL_MS_ORACLE_CENTER):
        raise RuntimeError("GT-independence audit cannot run with GT oracle enabled")
    exp.verify_visual_checkpoint_split()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    exp.verify_temporal_correction_bounds(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = exp.ThreeFrameRouteStateGRU().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    original_builder = exp.tr.build_route_cache
    reports = []
    predictions = []
    for corrupt in (False, True):
        def build(*positional, **kwargs):
            cache = original_builder(*positional, **kwargs)
            if corrupt:
                cache.gt_xy = cache.gt_xy + torch.tensor(
                    [1234.0, -987.0], dtype=cache.gt_xy.dtype,
                    device=cache.gt_xy.device)
            return cache
        exp.tr.build_route_cache = build
        try:
            result = evaluate_validation("citya", model, device,
                                         args.output / ("shifted_gt" if corrupt else "original_gt"))
        finally:
            exp.tr.build_route_cache = original_builder
        with open(result["CSV"], newline="") as file:
            rows = list(csv.DictReader(file))
        predictions.append(np.array([[float(row[k]) for k in
                                      ("final_x", "final_y", "prior_center_x", "prior_center_y")]
                                     for row in rows]))
        reports.append(float(result["MLE_m"]))
    delta = float(np.max(np.abs(predictions[0] - predictions[1])))
    report = {"prediction_and_search_center_max_difference_m": delta,
              "original_gt_MLE_m": reports[0], "shifted_gt_MLE_m": reports[1],
              "pass": delta < 1e-6 and abs(reports[0] - reports[1]) > 100,
              "validation_only": True, "GT_shift_m": [1234, -987]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["pass"]:
        raise RuntimeError("GT independence audit failed")


if __name__ == "__main__":
    main()
