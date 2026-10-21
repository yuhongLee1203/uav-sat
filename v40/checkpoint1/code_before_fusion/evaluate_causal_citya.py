"""Evaluate a saved temporal checkpoint with the corrected GT-free causal path.

GT is read only after each prediction for metrics. This does not train or
select a checkpoint. It also reports a route-time baseline for transparency.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import image_only_citya as exp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exp.configure()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp.config.OUTPUT_DIR = args.output.resolve()
    exp.config.MEASURE_END_TO_END_LATENCY = True
    args.output.mkdir(parents=True, exist_ok=True)
    visual = exp.FrozenVisualLocalizer(device)
    model = exp.new_model(device)
    checkpoint = args.checkpoint or exp.config.TEMPORAL_CHECKPOINT
    if args.checkpoint:
        saved = torch.load(checkpoint, map_location="cpu")
        if saved.get("architecture") != exp.config.ARCHITECTURE_NAME:
            raise RuntimeError("checkpoint architecture mismatch")
        model.load_state_dict(saved["model"], strict=True)
    model.eval()
    report = {
        "checkpoint": str(checkpoint),
        "training": "none in this command",
        "reference_protocol": "causal route_reference; no current-frame GT input",
        "acquisition": "7 local hypotheses within +/-15 m; each scores forward 18 patches",
        "route_duration_prior": False,
    }
    for name in ("route_B", "route_C"):
        route_idx = exp.config.ROUTE_NAMES.index(name)
        route = exp.tr.WaypointRoute(exp.tr.load_waypoint_xy(
            name, visual.origin_lat, visual.origin_lon))
        cache = exp.tr.build_route_cache(
            name, exp.config.ROUTE_ROOTS[route_idx], visual, device)
        summary = exp.tr.run_route_inference(
            name, visual, model, cache, route, device)
        # This is a diagnostic baseline, not an allowed image-only predictor:
        # it assumes the entire planned episode length and frame count are known.
        schedule_xy = np.asarray([
            route.xy_from_se(route.total_length_m * i / max(len(cache) - 1, 1), 0.0)
            for i in range(len(cache))
        ])
        errors = np.linalg.norm(schedule_xy - cache.gt_xy.numpy(), axis=1)
        report[name] = {
            "model": summary,
            "route_duration_only_baseline": {
                "MLE_m": float(errors.mean()),
                "P90_m": float(np.quantile(errors, 0.9)),
                "LSR@15_pct": float(np.mean(errors <= 15.0) * 100.0),
                "warning": "Uses full route duration; not image-only and not a fair navigation baseline",
            },
        }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
    lines = ["# Causal City A held-out evaluation", "",
             "Checkpoint: `%s`" % checkpoint, "",
             "| Route | Model MLE (m) | Model LSR@15 | Route-duration-only MLE (m)* |",
             "|---|---:|---:|---:|"]
    for name in ("route_B", "route_C"):
        row = report[name]
        lines.append("| %s | %.3f | %.2f%% | %.3f |" % (
            name, row["model"]["MLE_m"], row["model"]["LSR@15_pct"],
            row["route_duration_only_baseline"]["MLE_m"]))
    lines += ["", "*Diagnostic only: route duration is not an image input and can nearly solve these synthetic routes by itself."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
