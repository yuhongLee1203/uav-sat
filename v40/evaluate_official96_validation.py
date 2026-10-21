"""Evaluate a trained v39-topology temporal checkpoint on held-out City A val.

All tuning must use this validation episode; the test routes are not loaded.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

import image_only_citya as exp
from train_multicity_routes import evaluate_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--city", choices=("citya", "cityb", "cityc", "cityd"), default="citya")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tta-rotations", default="0",
                        help="Comma-separated image quarter-turns; validation-only preprocessing")
    parser.add_argument("--tta-flips", action="store_true",
                        help="Average horizontal reflections of the selected rotations")
    parser.add_argument("--tta-scales", default="1.0",
                        help="Comma-separated center-crop fractions; 1.0 keeps the full UAV image")
    args = parser.parse_args()
    rotations = tuple(int(value) for value in args.tta_rotations.split(","))
    if not rotations or any(value not in (0, 1, 2, 3) for value in rotations):
        parser.error("tta rotations must be 0,1,2,3")
    scales = tuple(float(value) for value in args.tta_scales.split(","))
    if not scales or any(value <= 0 or value > 1 for value in scales):
        parser.error("tta scales must be in (0, 1]")
    if rotations != (0,) or args.tta_flips or scales != (1.0,):
        if not os.environ.get("V40_FEATURE_CACHE_ROOT"):
            parser.error("TTA requires an isolated V40_FEATURE_CACHE_ROOT")
        original = exp.FrozenVisualLocalizer.encode_uav_clip

        def encode_with_tta(self, image):
            transformed = []
            height, width = image.shape[-2:]
            for scale in scales:
                if scale == 1.0:
                    scaled = image
                else:
                    crop_h = max(1, round(height * scale))
                    crop_w = max(1, round(width * scale))
                    top = (height - crop_h) // 2
                    left = (width - crop_w) // 2
                    scaled = F.interpolate(
                        image[..., top:top + crop_h, left:left + crop_w],
                        size=(height, width), mode="bilinear", align_corners=False)
                for turns in rotations:
                    item = torch.rot90(scaled, turns, (-2, -1))
                    transformed.append(item)
                    if args.tta_flips:
                        transformed.append(torch.flip(item, (-1,)))
            return torch.stack([original(self, item) for item in transformed]).mean(dim=0)

        exp.FrozenVisualLocalizer.encode_uav_clip = encode_with_tta
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp.configure(args.city)
    exp.verify_visual_checkpoint_split()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    exp.verify_temporal_correction_bounds(checkpoint)
    if checkpoint.get("architecture") != exp.config.ARCHITECTURE_NAME:
        raise RuntimeError("Temporal checkpoint architecture does not match v39 topology")
    model = exp.ThreeFrameRouteStateGRU().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {
        "temporal_checkpoint": str(args.checkpoint.resolve()),
        "visual_checkpoint": str(exp.config.VISUAL_CHECKPOINT.resolve()),
        "tta_rotations": rotations,
        "tta_flips": args.tta_flips,
        "tta_scales": scales,
        "environment": {key: value for key, value in os.environ.items()
                        if key.startswith(("V40_", "UAVSAT_", "MS_"))},
        "validation_only": True,
        "GT_front18_oracle": bool(exp.config.GT_FRONT18_ORACLE),
        "GT_front18_patch_oracle": bool(exp.config.GT_FRONT18_PATCH_ORACLE),
        "image_guided_front18": bool(exp.config.IMAGE_GUIDED_FRONT18),
        "GT_final_MS_center_oracle": bool(exp.config.GT_FINAL_MS_ORACLE_CENTER),
        "GT_final_MS_patch_oracle": bool(exp.config.GT_FINAL_MS_PATCH_ORACLE),
    }
    (args.output / "run_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n")
    summary = evaluate_validation(args.city, model, device, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "visual_checkpoint": settings["visual_checkpoint"],
        "run_settings": str((args.output / "run_settings.json").resolve()),
        "city": args.city,
        "validation_only": True,
        "GT_front18_oracle": bool(exp.config.GT_FRONT18_ORACLE),
        "GT_front18_patch_oracle": bool(exp.config.GT_FRONT18_PATCH_ORACLE),
        "image_guided_front18": bool(exp.config.IMAGE_GUIDED_FRONT18),
        "GT_final_MS_center_oracle": bool(exp.config.GT_FINAL_MS_ORACLE_CENTER),
        "GT_final_MS_patch_oracle": bool(exp.config.GT_FINAL_MS_PATCH_ORACLE),
        "valid_no_GT_inference_result": not any((exp.config.GT_FRONT18_ORACLE,
                                                   exp.config.GT_FINAL_MS_ORACLE_CENTER)),
        "current_frame_backbone": True,
        "tta_rotations": rotations,
        "tta_flips": args.tta_flips,
        "tta_scales": scales,
        "current_frame_GT_model_input": bool(exp.config.GT_FRONT18_ORACLE
                                               or exp.config.GT_FINAL_MS_ORACLE_CENTER),
        "MLE_m": summary["MLE_m"],
        "P90_m": summary["P90_m"],
        "LSR@15_pct": summary["LSR@15_pct"],
        "SelectedCandidateCapture_pct": summary["SelectedCandidateCapture_pct"],
        "BankCandidateCapture_pct": summary["BankCandidateCapture_pct"],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("[V40 VALIDATION ONLY]", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
