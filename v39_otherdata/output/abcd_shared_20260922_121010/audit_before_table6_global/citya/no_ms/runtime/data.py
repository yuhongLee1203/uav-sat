"""Bearing-UAV adapter implementing the small data API used by v36.

Coordinates are synthetic map coordinates: longitude := absolute x metres,
latitude := absolute y metres. No geographic projection or UAV yaw is provided
to the network; this convention only lets the unchanged v36 route code reuse its
meters_from_latlon() interface.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config

Image.MAX_IMAGE_PIXELS = None


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def meters_from_latlon(lat, lon, origin_lat, origin_lon):
    return float(lon) - float(origin_lon), float(lat) - float(origin_lat)


class SatGeoMapper:
    def __init__(self, bounds_json, sat_image_path):
        meta = load_json(bounds_json)
        if meta.get("mode") != "bearing_uav_pixel_meter":
            raise ValueError("v39 Bearing adapter requires mode=bearing_uav_pixel_meter")
        self.mpp = float(meta["mpp"])
        with Image.open(sat_image_path) as image:
            self.width, self.height = image.size

    def latlon_to_pixel(self, lat, lon):
        return float(lon) / self.mpp, float(lat) / self.mpp

    def pixel_to_latlon(self, x, y):
        return float(y) * self.mpp, float(x) * self.mpp


def crop_satellite(sat_image, pixel_x, pixel_y, crop_size):
    half = crop_size // 2
    left = int(round(pixel_x)) - half
    top = int(round(pixel_y)) - half
    crop = Image.new("RGB", (crop_size, crop_size))
    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(left + crop_size, sat_image.width)
    src_bottom = min(top + crop_size, sat_image.height)
    if src_right > src_left and src_bottom > src_top:
        patch = sat_image.crop((src_left, src_top, src_right, src_bottom))
        crop.paste(patch, (src_left - left, src_top - top))
    return crop


def image_transform(train=False, source="uav"):
    # Bearing-UAV target patches are already 256x256. Preserve UAV pixels exactly;
    # satellite crops follow v36 and are resized to the backbone's 256x256 input.
    if source == "uav":
        return transforms.Compose(
            [transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)]
        )
    return transforms.Compose(
        [
            transforms.Resize((int(config.IMAGE_SIZE), int(config.IMAGE_SIZE))),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ]
    )


class RouteDataset(Dataset):
    def __init__(
        self,
        root,
        sat_image_path=None,
        sat_json_path=None,
        origin_lat=None,
        origin_lon=None,
        train=True,
        include_frame_ids=None,
    ):
        self.root = Path(root)
        manifest = self.root / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError("Missing prepared Bearing route manifest: %s" % manifest)
        with manifest.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if include_frame_ids is not None:
            allow = {int(v) for v in include_frame_ids}
            rows = [r for r in rows if int(r["frame_id"]) in allow]
        if not rows:
            raise ValueError("No samples in %s" % manifest)
        self.samples = rows
        self.origin_lat = float(origin_lat) if origin_lat is not None else float(rows[0]["y_m"])
        self.origin_lon = float(origin_lon) if origin_lon is not None else float(rows[0]["x_m"])
        self.transform = image_transform(train, source="uav")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        if image.size != (int(config.IMAGE_SIZE), int(config.IMAGE_SIZE)):
            raise ValueError(
                "Bearing UAV image must already be %dx%d because v39 does not resize UAV input; got %s for %s"
                % (config.IMAGE_SIZE, config.IMAGE_SIZE, image.size, image_path)
            )
        x_m = float(row["x_m"])
        y_m = float(row["y_m"])
        x_rel, y_rel = meters_from_latlon(y_m, x_m, self.origin_lat, self.origin_lon)
        return {
            "uav": self.transform(image),
            "xy": torch.tensor([x_rel, y_rel], dtype=torch.float32),
            "raw_xy": torch.tensor([x_rel, y_rel], dtype=torch.float32),
            "latlon": torch.tensor([y_m, x_m], dtype=torch.float32),
            "pixel": torch.tensor([float(row["x_px"]), float(row["y_px"])], dtype=torch.float32),
            "yaw": torch.tensor(float(row.get("yaw_deg", 0.0)), dtype=torch.float32),
            "altitude": torch.tensor(float("nan"), dtype=torch.float32),
            "frame_id": str(row["frame_id"]),
            "timestamp_ns": torch.tensor(int(row["timestamp_ns"]), dtype=torch.long),
            "image_path": str(image_path),
        }


class SatPatchGallery(Dataset):
    def __init__(
        self,
        sat_image_path=None,
        sat_json_path=None,
        origin_lat=None,
        origin_lon=None,
        crop_size=None,
        stride=None,
    ):
        sat_image_path = Path(sat_image_path or config.SAT_IMAGE)
        sat_json_path = Path(sat_json_path or config.SAT_JSON)
        self.sat_image = Image.open(sat_image_path).convert("RGB")
        self.mapper = SatGeoMapper(sat_json_path, sat_image_path)
        self.origin_lat = float(origin_lat)
        self.origin_lon = float(origin_lon)
        self.crop_size = int(crop_size or config.SAT_CROP_SIZE)
        self.stride = int(stride or config.SAT_STRIDE)
        self.transform = image_transform(False, source="sat")
        self.samples = self._build_samples()

    def _axis_centers(self, size):
        half = self.crop_size // 2
        centers = list(range(half, max(half + 1, size - half), self.stride))
        edge = size - half - 1
        if centers[-1] != edge:
            centers.append(edge)
        return centers

    def _build_samples(self):
        rows = []
        for y in self._axis_centers(self.sat_image.height):
            for x in self._axis_centers(self.sat_image.width):
                lat, lon = self.mapper.pixel_to_latlon(x, y)
                xm, ym = meters_from_latlon(lat, lon, self.origin_lat, self.origin_lon)
                rows.append((float(x), float(y), float(lat), float(lon), float(xm), float(ym)))
        return rows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, lat, lon, xm, ym = self.samples[idx]
        patch = crop_satellite(self.sat_image, x, y, self.crop_size)
        return {
            "sat": self.transform(patch),
            "xy": torch.tensor([xm, ym], dtype=torch.float32),
            "latlon": torch.tensor([lat, lon], dtype=torch.float32),
            "pixel": torch.tensor([x, y], dtype=torch.float32),
            "index": torch.tensor(idx, dtype=torch.long),
        }
