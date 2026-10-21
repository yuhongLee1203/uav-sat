"""Train-val-only MeanShift parameter audit; never reads prepared route tests."""
import argparse
import json
from pathlib import Path

import torch

import image_only_citya as exp
from visual_localizer import FrozenVisualLocalizer, soft_mean_shift


@torch.no_grad()
def collect(visual, cache, indices, device):
    all_logits, all_centers, all_gt = [], [], []
    gallery = visual.gallery
    model = visual.model
    for start in range(0, len(indices), 64):
        batch = indices[start:start + 64]
        candidate = cache["candidate_indices"][batch].to(device)
        centers = gallery["xy"][candidate]
        sat_clip = gallery["clip_feat"][candidate].float()
        uav_clip = cache["uav_clip"][batch].to(device).float()
        z_uav = model.encode_uav_from_clip(uav_clip)
        z_sat = model.encode_sat_from_clip(
            sat_clip.reshape(-1, sat_clip.shape[-1]),
            centers.reshape(-1, 2)).reshape(len(batch), candidate.shape[1], -1)
        logits = model.logit_scale.exp().clamp(max=100.0) * (
            z_uav[:, None] * z_sat).sum(dim=2)
        all_logits.append(logits.cpu())
        all_centers.append(centers.cpu())
        all_gt.append(cache["gt_xy"][batch])
    return torch.cat(all_logits), torch.cat(all_centers), torch.cat(all_gt)


@torch.no_grad()
def score(rows, tau, bandwidth, beta, device):
    logits, centers, gt = rows
    errors = []
    for start in range(0, len(gt), 32):
        end = start + 32
        xy, *_ = soft_mean_shift(
            logits[start:end].to(device), centers[start:end].to(device),
            tau, bandwidth, 3, beta)
        errors.append(torch.linalg.norm(xy.cpu() - gt[start:end], dim=1))
    error = torch.cat(errors)
    return {"MLE_m": float(error.mean()), "P90_m": float(torch.quantile(error, 0.9))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=("citya", "cityb", "cityc", "cityd"), required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    exp.configure(args.city)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual = FrozenVisualLocalizer(device)
    cache = torch.load(args.feature_cache, map_location="cpu")
    length = len(cache["image_paths"])
    start = int(length * 0.9)
    end = start + (length - start) // 2
    calibration = collect(visual, cache, list(range(start, end)), device)
    confirmation = collect(visual, cache, list(range(end, length)), device)
    rows = []
    for tau in (0.15, 0.30, 0.60, 1.0):
        for bandwidth in (4.0, 6.0, 8.0, 12.0):
            for beta in (4.0, 12.0):
                candidate = {"tau": tau, "bandwidth_m": bandwidth, "beta": beta}
                candidate["calibration"] = score(calibration, tau, bandwidth, beta, device)
                rows.append(candidate)
    rows.sort(key=lambda row: row["calibration"]["MLE_m"])
    best = rows[0]
    best["confirmation"] = score(
        confirmation, best["tau"], best["bandwidth_m"], best["beta"], device)
    default = {"tau": 0.30, "bandwidth_m": 8.0, "beta": 12.0}
    default["confirmation"] = score(confirmation, 0.30, 8.0, 12.0, device)
    output = {"city": args.city, "feature_cache": str(args.feature_cache),
              "calibration_count": len(calibration[2]),
              "confirmation_count": len(confirmation[2]),
              "best": best, "default": default, "all_calibration": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print("[MS CALIBRATION]", json.dumps({"best": best, "default": default}), flush=True)


if __name__ == "__main__":
    main()
