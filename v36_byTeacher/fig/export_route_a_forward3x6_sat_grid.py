#!/usr/bin/env python3
"""Export the real forward 3x6 SAT candidates for one Route-A UAV frame."""
from pathlib import Path
import sys
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from data import RouteDataset, SatPatchGallery, crop_satellite
from visual_localizer import build_pixel_index, regular_grid_indices

ROUTE_ROOT = Path("/yh/study/new_data_2/model_dataset_new_1_flight")
FRAME_ID = int(os.environ.get("UAVSAT_FIG_FRAME", "800"))


def main():
    frame_tag = f"{FRAME_ID:04d}"
    output_dir = ROOT / "fig" / f"sat3x6_routeA_frame{frame_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = RouteDataset(ROUTE_ROOT, train=False)
    sample_index = next(
        index for index, sample in enumerate(dataset.samples)
        if int(sample["frame_id"]) == FRAME_ID
    )
    sample = dataset.samples[sample_index]
    prior_xy = torch.tensor([[sample["x_meter"], sample["y_meter"]]], dtype=torch.float32)
    next_sample = dataset.samples[min(sample_index + 1, len(dataset.samples) - 1)]
    delta = torch.tensor(
        [next_sample["x_meter"] - sample["x_meter"], next_sample["y_meter"] - sample["y_meter"]],
        dtype=torch.float32,
    )
    heading = torch.atan2(delta[1], delta[0])
    unit = torch.tensor([[torch.cos(heading), torch.sin(heading)]])
    grid_center = prior_xy - float(config.FORWARD_SEARCH_ORIGIN_BACKSHIFT_M) * unit

    gallery = SatPatchGallery(origin_lat=dataset.origin_lat, origin_lon=dataset.origin_lon)
    gallery_xy = torch.tensor([[row["x_meter"], row["y_meter"]] for row in gallery.samples], dtype=torch.float32)
    gallery_pixel = torch.tensor([[row["pixel_x"], row["pixel_y"]] for row in gallery.samples], dtype=torch.float32)
    full_indices = regular_grid_indices(
        gallery_xy, gallery_pixel, build_pixel_index(gallery_pixel), grid_center,
        grid_size=6, stride=int(config.SAT_STRIDE), device=torch.device("cpu"),
    )
    full_centers = gallery_xy[full_indices]
    projection = ((full_centers - grid_center[:, None]) * unit[:, None]).sum(dim=2)
    selected = torch.topk(projection, k=18, dim=1, largest=True, sorted=False).indices
    selected_indices = torch.gather(full_indices, 1, selected)
    selected_centers = gallery_xy[selected_indices]
    cross = torch.tensor([[-unit[0, 1], unit[0, 0]]])
    forward = ((selected_centers - grid_center[:, None]) * unit[:, None]).sum(dim=2)
    lateral = ((selected_centers - grid_center[:, None]) * cross[:, None]).sum(dim=2)
    order = torch.argsort(-forward * 1000.0 + lateral, dim=1)
    selected_indices = torch.gather(selected_indices, 1, order)[0].tolist()

    # A forward 3x6 window is anchored laterally on the UAV prior: the UAV
    # must sit below the middle gap (between columns 3 and 4), rather than be
    # displaced toward either side by the gallery's global 32-pixel lattice.
    # Re-anchor the discrete lattice by whole stride steps before rendering.
    # This only changes the explanatory figure; it does not alter training.
    selected_pixels_for_anchor = np.asarray([
        [gallery.samples[int(index)]["pixel_x"], gallery.samples[int(index)]["pixel_y"]]
        for index in selected_indices
    ], dtype=float)
    image_direction = np.asarray([
        next_sample["pixel_x"] - sample["pixel_x"],
        next_sample["pixel_y"] - sample["pixel_y"],
    ], dtype=float)
    image_norm = max(float(np.linalg.norm(image_direction)), 1e-6)
    image_lateral = np.asarray([-image_direction[1], image_direction[0]]) / image_norm
    uav_pixel = np.asarray([sample["pixel_x"], sample["pixel_y"]], dtype=float)
    lateral_error = float(np.dot(uav_pixel - selected_pixels_for_anchor.mean(axis=0), image_lateral))
    lateral_shift = round(lateral_error / float(config.SAT_STRIDE))
    desired_pixels = selected_pixels_for_anchor + lateral_shift * float(config.SAT_STRIDE) * image_lateral
    pixel_index = {
        (int(round(row["pixel_x"])), int(round(row["pixel_y"]))): index
        for index, row in enumerate(gallery.samples)
    }
    reanchored = []
    for pixel_x, pixel_y in desired_pixels:
        key = (int(round(pixel_x / config.SAT_STRIDE)) * config.SAT_STRIDE,
               int(round(pixel_y / config.SAT_STRIDE)) * config.SAT_STRIDE)
        if key not in pixel_index:
            reanchored = []
            break
        reanchored.append(pixel_index[key])
    if len(reanchored) == 18:
        selected_indices = reanchored

    patch_size = int(config.SAT_CROP_SIZE)
    canvas = Image.new("RGB", (6 * patch_size, 3 * patch_size), "white")
    with Image.open(config.SAT_IMAGE) as satellite:
        satellite = satellite.convert("RGB")
        for index, gallery_index in enumerate(selected_indices):
            center = gallery.samples[int(gallery_index)]
            patch = crop_satellite(satellite, center["pixel_x"], center["pixel_y"], patch_size)
            row, col = divmod(index, 6)
            patch.save(output_dir / f"sat_r{row + 1}_c{col + 1}.png")
            canvas.paste(patch, (col * patch_size, row * patch_size))

        # One continuous north-up map covering the union of all 18 overlapping
        # 320x320 candidate patches. This is the actual 3x6 search footprint,
        # not a mosaic of duplicated patches.
        selected_pixels = np.asarray([
            [gallery.samples[int(index)]["pixel_x"], gallery.samples[int(index)]["pixel_y"]]
            for index in selected_indices
        ], dtype=float)
        half = patch_size // 2
        left = int(np.floor(selected_pixels[:, 0].min())) - half
        right = int(np.ceil(selected_pixels[:, 0].max())) + half
        top = int(np.floor(selected_pixels[:, 1].min())) - half
        bottom = int(np.ceil(selected_pixels[:, 1].max())) + half
        coverage = satellite.crop((left, top, right, bottom))
        coverage.save(ROOT / "fig" / f"sat_routeA_frame{frame_tag}_forward3x6_coverage_map.png")

        gridded = coverage.copy()
        grid_draw = ImageDraw.Draw(gridded)
        width, height = gridded.size
        for column in range(7):
            x = round(column * width / 6)
            grid_draw.line((x, 0, x, height), fill="white", width=3)
        for row in range(4):
            y = round(row * height / 3)
            grid_draw.line((0, y, width, y), fill="white", width=3)
        gridded.save(ROOT / "fig" / f"sat_routeA_frame{frame_tag}_forward3x6_coverage_map_grid.png")

        # Actual forward 3x6 subset: draw its 18 real overlapping patch
        # footprints and centres on one continuous satellite crop.
        forward_left = int(np.floor(selected_pixels[:, 0].min())) - half
        forward_right = int(np.ceil(selected_pixels[:, 0].max())) + half
        forward_top = int(np.floor(selected_pixels[:, 1].min())) - half
        forward_bottom = int(np.ceil(selected_pixels[:, 1].max())) + half
        forward_map = satellite.crop((forward_left, forward_top, forward_right, forward_bottom))
        forward_draw = ImageDraw.Draw(forward_map)
        for pixel_x, pixel_y in selected_pixels:
            x = int(round(pixel_x - forward_left))
            y = int(round(pixel_y - forward_top))
            forward_draw.rectangle((x - half, y - half, x + half, y + half), outline="white", width=1)
            forward_draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#FFD60A", outline="black", width=1)
        # Actual UAV position and its measured frame-to-frame ground direction.
        uav_x = float(sample["pixel_x"] - forward_left)
        uav_y = float(sample["pixel_y"] - forward_top)
        direction_x = float(next_sample["pixel_x"] - sample["pixel_x"])
        direction_y = float(next_sample["pixel_y"] - sample["pixel_y"])
        direction_norm = max((direction_x ** 2 + direction_y ** 2) ** 0.5, 1e-6)
        arrow_length = 90.0
        arrow_x = uav_x + arrow_length * direction_x / direction_norm
        arrow_y = uav_y + arrow_length * direction_y / direction_norm
        forward_draw.line((uav_x, uav_y, arrow_x, arrow_y), fill="#FF2D2D", width=6)
        arrow_angle = np.arctan2(direction_y, direction_x)
        head = 18.0
        for offset in (2.55, -2.55):
            forward_draw.line(
                (arrow_x, arrow_y, arrow_x + head * np.cos(arrow_angle + offset), arrow_y + head * np.sin(arrow_angle + offset)),
                fill="#FF2D2D", width=6,
            )
        forward_draw.ellipse((uav_x - 9, uav_y - 9, uav_x + 9, uav_y + 9), fill="#FF2D2D", outline="white", width=2)
        forward_map.save(ROOT / "fig" / f"sat_routeA_frame{frame_tag}_forward3x6_patch_footprints.png")

        # Full UAV-centred 6x6 lattice: 36 overlapping 320x320 patches at a
        # 32-pixel stride. Crop their union once so overlap is fused naturally.
        full_indices_list = full_indices[0].tolist()
        full_pixels = np.asarray([
            [gallery.samples[int(index)]["pixel_x"], gallery.samples[int(index)]["pixel_y"]]
            for index in full_indices_list
        ], dtype=float)
        full_left = int(np.floor(full_pixels[:, 0].min())) - half
        full_right = int(np.ceil(full_pixels[:, 0].max())) + half
        full_top = int(np.floor(full_pixels[:, 1].min())) - half
        full_bottom = int(np.ceil(full_pixels[:, 1].max())) + half
        full_map = satellite.crop((full_left, full_top, full_right, full_bottom))
        full_map.save(ROOT / "fig" / f"sat_routeA_frame{frame_tag}_uav_centered_6x6_coverage.png")

        overlay = full_map.copy()
        overlay_draw = ImageDraw.Draw(overlay)
        # Every white rectangle is one true 320x320 SAT patch. The 36 patch
        # centres are displaced by stride=32, therefore their footprints overlap.
        for pixel_x, pixel_y in full_pixels:
            x = int(round(pixel_x - full_left))
            y = int(round(pixel_y - full_top))
            overlay_draw.rectangle((x - half, y - half, x + half, y + half), outline="white", width=1)
        # Mark the centre of every one of the 36 overlapping 320x320 patches.
        for pixel_x, pixel_y in full_pixels:
            x = int(round(pixel_x - full_left))
            y = int(round(pixel_y - full_top))
            overlay_draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#FFD60A", outline="black", width=1)
        overlay.save(ROOT / "fig" / f"sat_routeA_frame{frame_tag}_uav_centered_6x6_patch_footprints.png")

    draw = ImageDraw.Draw(canvas)
    for row in range(4):
        y = row * patch_size
        draw.line((0, y, 6 * patch_size, y), fill="white", width=4)
    for col in range(7):
        x = col * patch_size
        draw.line((x, 0, x, 3 * patch_size), fill="white", width=4)
    canvas.save(ROOT / "fig" / f"sat_routeA_frame{frame_tag}_forward3x6_grid.png")


if __name__ == "__main__":
    main()
