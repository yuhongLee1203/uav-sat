"""One-shot held-out evaluation for the predeclared-waypoint ID96 adapter."""
import argparse
import json
import os
from pathlib import Path

import torch

import image_only_citya as exp
from train_multicity_routes import evaluate_city


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--city", choices=("citya", "cityb", "cityc", "cityd"), default="citya")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tta-rotations", default="0")
    parser.add_argument("--tta-flips", action="store_true")
    args = parser.parse_args()
    rotations = tuple(int(value) for value in args.tta_rotations.split(","))
    if not rotations or any(value not in (0, 1, 2, 3) for value in rotations):
        parser.error("tta rotations must be 0,1,2,3")
    if rotations != (0,) or args.tta_flips:
        if not os.environ.get("V40_FEATURE_CACHE_ROOT"):
            parser.error("TTA requires an isolated V40_FEATURE_CACHE_ROOT")
        original = exp.FrozenVisualLocalizer.encode_uav_clip

        def encode_with_tta(self, image):
            transformed = [torch.rot90(image, turns, (-2, -1)) for turns in rotations]
            if args.tta_flips:
                transformed.extend(torch.flip(item, (-1,)) for item in transformed[:])
            return torch.stack([original(self, item) for item in transformed]).mean(dim=0)

        exp.FrozenVisualLocalizer.encode_uav_clip = encode_with_tta
    args.output.mkdir(parents=True, exist_ok=True)
    exp.configure(args.city)
    if (exp.config.GT_FRONT18_ORACLE or exp.config.GT_FINAL_MS_ORACLE_CENTER
            or exp.config.GT_FRONT18_PATCH_ORACLE
            or exp.config.GT_FINAL_MS_PATCH_ORACLE):
        raise RuntimeError("held-out test forbids current-frame GT oracle switches")
    exp.verify_visual_checkpoint_split()
    prepared = json.loads((exp.PREPARED / "experiment.json").read_text())
    if (prepared.get("heldout_waypoints_from_GT") is not False
            or prepared.get("sequence_selection_version")
            != "official_rsi96_seed42_fixed_waypoint_temporal_v1"):
        raise RuntimeError("held-out waypoints are not independently predeclared")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    exp.verify_temporal_correction_bounds(checkpoint)
    if checkpoint.get("architecture") != exp.config.ARCHITECTURE_NAME:
        raise RuntimeError("checkpoint does not match the v39 architecture")
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = exp.ThreeFrameRouteStateGRU().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    rows = evaluate_city(args.city, model, device, args.output)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "city": args.city,
        "prepared_protocol": prepared["sequence_selection_version"],
        "heldout_waypoints_from_GT": False,
        "episode_construction_uses_GT_offline": True,
        "current_frame_GT_model_input": False,
        "official_table6_reproduction": False,
        "tta_rotations": rotations,
        "tta_flips": args.tta_flips,
        "visual_checkpoint": str(exp.config.VISUAL_CHECKPOINT.resolve()),
        "environment": {key: value for key, value in os.environ.items()
                        if key.startswith(("V40_", "UAVSAT_", "MS_"))},
        "results": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    for name, value in rows.items():
        print("[V40 FIXED TEST]", args.city, name, "MLE=%.3fm" % value["MLE_m"], flush=True)


if __name__ == "__main__":
    main()
