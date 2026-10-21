"""Evaluate an existing temporal checkpoint on causal prepared routes.

Current-frame GT is used only by metric calculations, never localization.
"""
import argparse
import json
from pathlib import Path

import torch

import image_only_citya as exp
from train_multicity_routes import check_city, evaluate_city


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", choices=("citya", "cityb", "cityc", "cityd"), required=True)
    parser.add_argument("--temporal-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    counts = {city: check_city(city) for city in args.cities}
    exp.configure(args.cities[0])
    if args.temporal_checkpoint:
        model = exp.ThreeFrameRouteStateGRU().to(device)
        payload = torch.load(args.temporal_checkpoint, map_location="cpu")
        if payload.get("architecture") != exp.config.ARCHITECTURE_NAME:
            raise RuntimeError("temporal checkpoint architecture mismatch")
        model.load_state_dict(payload["model"], strict=True)
    else:
        model = exp.new_model(device)
    args.output.mkdir(parents=True, exist_ok=True)
    results = {city: evaluate_city(city, model, device, args.output / "eval")
               for city in args.cities}
    report = {
        "cities": args.cities,
        "counts_train_test01_test02": counts,
        "temporal_checkpoint": str(args.temporal_checkpoint) if args.temporal_checkpoint else "shared_ABCD_warm_start",
        "test_GT_used_by_model": False,
        "inference_profile": {
            "backbone": exp.config.BACKBONE_KEY,
            "search_hypotheses": exp.config.ACQ_HYPOTHESIS_COUNT,
            "search_bank_radius_m": exp.config.ROUTE_REFERENCE_BANK_RADIUS_M,
            "raw_visual_weight": exp.config.ACQ_RAW_VISUAL_EVIDENCE_WEIGHT,
            "motion_prior_weight": exp.config.ACQ_HYPOTHESIS_MOTION_PRIOR_WEIGHT,
            "kalman_final_step_max_m": exp.config.KALMAN_FINAL_STEP_MAX_M,
            "kalman_visual_step_relax": exp.config.NO_GT_ALLOW_VISUAL_STEP_RELAX,
            "kalman_visual_relax_extra_m": exp.config.NO_GT_VISUAL_RELAX_EXTRA_M,
            "kalman_measurement_innovation_progress_m": exp.config.KALMAN_MAX_MEASUREMENT_INNOVATION_PROGRESS_M,
            "kalman_posterior_correction_progress_m": exp.config.KALMAN_MAX_POSTERIOR_CORRECTION_PROGRESS_M,
            "kalman_max_backward_correction_m": exp.config.NO_GT_MAX_BACKWARD_CORRECTION_M,
        },
        "results": results,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
    for city in args.cities:
        for route in ("test_01", "test_02"):
            row = results[city][route]
            print(city, route, "MLE=%.3fm" % row["MLE_m"], flush=True)


if __name__ == "__main__":
    main()
