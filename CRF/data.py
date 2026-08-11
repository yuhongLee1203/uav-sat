import json
import math
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config

Image.MAX_IMAGE_PIXELS = None

CLIP_MEAN = (0.0, 0.0, 0.0)
CLIP_STD = (1.0, 1.0, 1.0)


class CenterMaxSquareCrop:
    def __call__(self, img):
        width, height = img.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return img.crop((left, top, left + side, top + side))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def frame_index(path):
    match = re.search(r"vi_(\d+)\.", Path(path).name)
    if not match:
        raise ValueError(f"Cannot parse frame index from {path}")
    return int(match.group(1))


def sensor_path(root):
    root = Path(root)
    yaw_path = root / "sensor_with_yaw.json"
    return yaw_path if yaw_path.exists() else root / "sensor.json"


def timestamps_from_sensor(path):
    data = load_json(path)
    timestamps = data.get("timestamp")
    if not isinstance(timestamps, list):
        raise ValueError(f"{path} must contain a timestamp list")
    return timestamps


def bearing_degrees(lat1, lon1, lat2, lon2):
    lat1 = math.radians(float(lat1))
    lat2 = math.radians(float(lat2))
    dlon = math.radians(float(lon2) - float(lon1))
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def meters_from_latlon(lat, lon, origin_lat, origin_lon):
    earth_radius_m = 6378137.0
    origin_lat_rad = math.radians(float(origin_lat))
    x = math.radians(float(lon) - float(origin_lon)) * earth_radius_m * math.cos(origin_lat_rad)
    y = math.radians(float(lat) - float(origin_lat)) * earth_radius_m
    return x, y


def twd97_from_latlon(lat, lon):
    a = 6378137.0
    b = 6356752.314245
    lon0 = math.radians(121.0)
    k0 = 0.9999
    dx = 250000.0
    lat = math.radians(float(lat))
    lon = math.radians(float(lon))
    e = math.sqrt(1.0 - (b * b) / (a * a))
    e2 = e * e / (1.0 - e * e)
    n = (a - b) / (a + b)
    nu = a / math.sqrt(1.0 - (e * math.sin(lat)) ** 2)
    p = lon - lon0
    A = a * (1 - n + 5 / 4 * (n**2 - n**3) + 81 / 64 * (n**4 - n**5))
    B = 3 * a * n / 2 * (1 - n + 7 / 8 * (n**2 - n**3) + 55 / 64 * (n**4 - n**5))
    C = 15 * a * n**2 / 16 * (1 - n + 3 / 4 * (n**2 - n**3))
    D = 35 * a * n**3 / 48 * (1 - n + 11 / 16 * (n**2 - n**3))
    E = 315 * a * n**4 / 51 * (1 - n)
    S = A * lat - B * math.sin(2 * lat) + C * math.sin(4 * lat) - D * math.sin(6 * lat) + E * math.sin(8 * lat)
    K1 = S * k0
    K2 = k0 * nu * math.sin(2 * lat) / 4
    K3 = (
        k0
        * nu
        * math.sin(lat)
        * (math.cos(lat) ** 3)
        / 24
        * (5 - math.tan(lat) ** 2 + 9 * e2 * math.cos(lat) ** 2 + 4 * (e2**2) * math.cos(lat) ** 4)
    )
    y = K1 + K2 * (p**2) + K3 * (p**4)
    K4 = k0 * nu * math.cos(lat)
    K5 = k0 * nu * (math.cos(lat) ** 3) / 6 * (1 - math.tan(lat) ** 2 + e2 * math.cos(lat) ** 2)
    x = K4 * p + K5 * (p**3) + dx
    return x, y


def latlon_from_twd97(x, y):
    lat, lon = 23.45, 120.28
    for _ in range(8):
        px, py = twd97_from_latlon(lat, lon)
        lat += (float(y) - py) / 110800.0
        lon += (float(x) - px) / (102300.0 * math.cos(math.radians(lat)))
    return lat, lon


class SatGeoMapper:
    def __init__(self, bounds_json, sat_image_path):
        meta = load_json(bounds_json)
        self.world_file = meta.get("world_file")
        bounds = meta.get("geo_bounds") or meta.get("satellite")
        if self.world_file is None:
            if bounds is None:
                raise ValueError(f"{bounds_json} must contain geo_bounds, satellite bounds, or world_file")
            self.top_left = bounds["top_left"]
            self.bottom_right = bounds["bottom_right"]
        else:
            self.A = float(self.world_file["A"])
            self.D = float(self.world_file["D"])
            self.B = float(self.world_file["B"])
            self.E = float(self.world_file["E"])
            self.C = float(self.world_file["C"])
            self.F = float(self.world_file["F"])
            det = self.A * self.E - self.B * self.D
            if abs(det) < 1e-12:
                raise ValueError(f"Invalid world_file affine transform in {bounds_json}")
            self.inv_A = self.E / det
            self.inv_B = -self.B / det
            self.inv_D = -self.D / det
            self.inv_E = self.A / det
        with Image.open(sat_image_path) as img:
            self.width, self.height = img.size

    def latlon_to_pixel(self, lat, lon):
        if self.world_file is not None:
            x_world, y_world = twd97_from_latlon(lat, lon)
            dx = x_world - self.C
            dy = y_world - self.F
            x = self.inv_A * dx + self.inv_B * dy
            y = self.inv_D * dx + self.inv_E * dy
            return x, y
        left = float(self.top_left["longitude"])
        right = float(self.bottom_right["longitude"])
        top = float(self.top_left["latitude"])
        bottom = float(self.bottom_right["latitude"])
        x = (float(lon) - left) / (right - left) * (self.width - 1)
        y = (top - float(lat)) / (top - bottom) * (self.height - 1)
        return x, y

    def pixel_to_latlon(self, x, y):
        if self.world_file is not None:
            x_world = self.A * float(x) + self.B * float(y) + self.C
            y_world = self.D * float(x) + self.E * float(y) + self.F
            return latlon_from_twd97(x_world, y_world)
        left = float(self.top_left["longitude"])
        right = float(self.bottom_right["longitude"])
        top = float(self.top_left["latitude"])
        bottom = float(self.bottom_right["latitude"])
        lon = left + (float(x) / (self.width - 1)) * (right - left)
        lat = top - (float(y) / (self.height - 1)) * (top - bottom)
        return lat, lon


def crop_satellite(sat_image, pixel_x, pixel_y, crop_size):
    half = crop_size // 2
    left = int(round(pixel_x)) - half
    top = int(round(pixel_y)) - half
    crop = Image.new("RGB", (crop_size, crop_size))
    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(left + crop_size, sat_image.width)
    src_bottom = min(top + crop_size, sat_image.height)
    if src_right <= src_left or src_bottom <= src_top:
        return crop
    patch = sat_image.crop((src_left, src_top, src_right, src_bottom))
    crop.paste(patch, (src_left - left, src_top - top))
    return crop


def image_transform(train, source="uav"):
    center_crop_size = getattr(config, "UAV_CENTER_CROP_SIZE", None)
    resize_after_crop = getattr(config, "UAV_RESIZE_AFTER_CROP", None)
    use_uav_max_square_crop = source == "uav" and bool(getattr(config, "UAV_CENTER_MAX_SQUARE_CROP", False))
    use_uav_center_crop = source == "uav" and center_crop_size is not None and not use_uav_max_square_crop
    uav_crop = None
    if use_uav_max_square_crop:
        uav_crop = CenterMaxSquareCrop()
    elif use_uav_center_crop:
        uav_crop = transforms.CenterCrop(int(center_crop_size))
    if train and bool(getattr(config, "TRAIN_UAV_AUGMENT", True)):
        if use_uav_center_crop or use_uav_max_square_crop:
            ops = [uav_crop]
            if resize_after_crop is not None:
                ops.append(transforms.Resize((int(resize_after_crop), int(resize_after_crop))))
            ops.extend(
                [
                    transforms.RandomApply(
                        [transforms.RandomAffine(degrees=4, translate=(0.02, 0.02), scale=(0.98, 1.02), fill=0)],
                        p=0.35,
                    ),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(CLIP_MEAN, CLIP_STD),
                ]
            )
            return transforms.Compose(
                ops
            )
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=8, translate=(0.03, 0.03), scale=(0.96, 1.04), fill=0)],
                    p=0.5,
                ),
                transforms.RandomPerspective(distortion_scale=0.12, p=0.25),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04)],
                    p=0.5,
                ),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
    if use_uav_center_crop or use_uav_max_square_crop:
        ops = [uav_crop]
        if resize_after_crop is not None:
            ops.append(transforms.Resize((int(resize_after_crop), int(resize_after_crop))))
        ops.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
        return transforms.Compose(
            ops
        )
    return transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


class RouteDataset(Dataset):
    def __init__(
        self,
        root,
        sat_image_path=config.SAT_IMAGE,
        sat_json_path=config.SAT_JSON,
        origin_lat=None,
        origin_lon=None,
        train=True,
        include_frame_ids=None,
    ):
        self.root = Path(root)
        self.vi_dir = self.root / "vi"
        if not self.vi_dir.exists():
            self.vi_dir = self.root / "images"
        self.timestamps = timestamps_from_sensor(sensor_path(self.root))
        self.origin_lat = float(origin_lat if origin_lat is not None else self.timestamps[0]["latitude"])
        self.origin_lon = float(origin_lon if origin_lon is not None else self.timestamps[0]["longitude"])
        self.mapper = SatGeoMapper(sat_json_path, sat_image_path)
        self.transform = image_transform(train, source="uav")
        self.include_frame_ids = set(int(v) for v in include_frame_ids) if include_frame_ids is not None else None
        self.min_altitude_m = getattr(config, "MIN_ALTITUDE_M", None)
        self.altitude_field = str(getattr(config, "ALTITUDE_FIELD", "altitude"))
        self.samples = self._build_samples()

    def _image_paths(self):
        paths = [p for p in self.vi_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        return sorted(paths, key=frame_index)

    def _yaw_for_index(self, idx):
        item = self.timestamps[idx]
        if "yaw" in item:
            return float(item["yaw"])
        if idx < len(self.timestamps) - 1:
            cur = self.timestamps[idx]
            nxt = self.timestamps[idx + 1]
        else:
            cur = self.timestamps[idx - 1]
            nxt = self.timestamps[idx]
        return bearing_degrees(cur["latitude"], cur["longitude"], nxt["latitude"], nxt["longitude"])

    def _build_samples(self):
        samples = []
        for path in self._image_paths():
            idx = frame_index(path)
            if idx >= len(self.timestamps):
                continue
            if self.include_frame_ids is not None and idx not in self.include_frame_ids:
                continue
            item = self.timestamps[idx]
            altitude = item.get(self.altitude_field, item.get("altitude", None))
            altitude = float(altitude) if altitude is not None else None
            if self.min_altitude_m is not None:
                if altitude is None or altitude < float(self.min_altitude_m):
                    continue
            lat = float(item["latitude"])
            lon = float(item["longitude"])
            pixel_x, pixel_y = self.mapper.latlon_to_pixel(lat, lon)
            if pixel_x < 0 or pixel_x >= self.mapper.width or pixel_y < 0 or pixel_y >= self.mapper.height:
                continue
            x_meter, y_meter = meters_from_latlon(lat, lon, self.origin_lat, self.origin_lon)
            samples.append(
                {
                    "image_path": str(path),
                    "frame_id": idx,
                    "lat": lat,
                    "lon": lon,
                    "yaw": self._yaw_for_index(idx),
                    "altitude": altitude,
                    "x_meter": x_meter,
                    "y_meter": y_meter,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                }
            )
        if not samples:
            raise ValueError(f"No samples found under {self.root}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        uav = Image.open(sample["image_path"]).convert("RGB")
        return {
            "uav": self.transform(uav),
            "xy": torch.tensor([sample["x_meter"], sample["y_meter"]], dtype=torch.float32),
            "latlon": torch.tensor([sample["lat"], sample["lon"]], dtype=torch.float32),
            "pixel": torch.tensor([sample["pixel_x"], sample["pixel_y"]], dtype=torch.float32),
            "yaw": torch.tensor(sample["yaw"], dtype=torch.float32),
            "altitude": torch.tensor(float(sample["altitude"]) if sample["altitude"] is not None else float("nan"), dtype=torch.float32),
            "frame_id": str(sample["frame_id"]),
            "image_path": sample["image_path"],
        }


class SatPatchGallery(Dataset):
    def __init__(
        self,
        sat_image_path=config.SAT_IMAGE,
        sat_json_path=config.SAT_JSON,
        origin_lat=None,
        origin_lon=None,
        crop_size=config.SAT_CROP_SIZE,
        stride=config.SAT_STRIDE,
    ):
        self.sat_image = Image.open(sat_image_path).convert("RGB")
        self.mapper = SatGeoMapper(sat_json_path, sat_image_path)
        self.origin_lat = float(origin_lat)
        self.origin_lon = float(origin_lon)
        self.crop_size = int(crop_size)
        self.stride = int(stride)
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
        xs = self._axis_centers(self.sat_image.width)
        ys = self._axis_centers(self.sat_image.height)
        samples = []
        for y in ys:
            for x in xs:
                lat, lon = self.mapper.pixel_to_latlon(x, y)
                xm, ym = meters_from_latlon(lat, lon, self.origin_lat, self.origin_lon)
                samples.append(
                    {
                        "pixel_x": float(x),
                        "pixel_y": float(y),
                        "lat": float(lat),
                        "lon": float(lon),
                        "x_meter": float(xm),
                        "y_meter": float(ym),
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        patch = crop_satellite(self.sat_image, s["pixel_x"], s["pixel_y"], self.crop_size)
        return {
            "sat": self.transform(patch),
            "xy": torch.tensor([s["x_meter"], s["y_meter"]], dtype=torch.float32),
            "latlon": torch.tensor([s["lat"], s["lon"]], dtype=torch.float32),
            "pixel": torch.tensor([s["pixel_x"], s["pixel_y"]], dtype=torch.float32),
            "index": torch.tensor(idx, dtype=torch.long),
        }
