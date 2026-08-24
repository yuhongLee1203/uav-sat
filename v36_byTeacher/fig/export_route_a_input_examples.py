#!/usr/bin/env python3
"""Export one real UAV frame and the exact model input crops used by v36_byTeacher."""
from pathlib import Path
import sys
import os

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from data import SatGeoMapper, crop_satellite, interpolate_sampled_gps, sensor_path, timestamps_from_sensor

ROUTE_ROOT = Path("/yh/study/new_data_2/model_dataset_new_1_flight")
FRAME_ID = int(os.environ.get("UAVSAT_FIG_FRAME", "800"))


def main():
    output_dir = ROOT / "fig"
    output_dir.mkdir(parents=True, exist_ok=True)
    uav_path = ROUTE_ROOT / "vi" / f"vi_{FRAME_ID:06d}.jpg"
    timestamps = timestamps_from_sensor(sensor_path(ROUTE_ROOT))
    latitude, longitude = interpolate_sampled_gps(timestamps)[FRAME_ID]
    mapper = SatGeoMapper(config.SAT_JSON, config.SAT_IMAGE)
    pixel_x, pixel_y = mapper.latlon_to_pixel(latitude, longitude)

    with Image.open(uav_path) as image:
        uav = image.convert("RGB")
        uav.save(output_dir / f"uav_routeA_frame{FRAME_ID:04d}_original.png")
        width, height = uav.size
        side = int(config.UAV_CENTER_CROP_SIZE)
        left = (width - side) // 2
        top = (height - side) // 2
        uav.crop((left, top, left + side, top + side)).save(
            output_dir / f"uav_routeA_frame{FRAME_ID:04d}_center_crop256.png"
        )

    with Image.open(config.SAT_IMAGE) as image:
        crop_satellite(image.convert("RGB"), pixel_x, pixel_y, int(config.SAT_CROP_SIZE)).save(
            output_dir / f"sat_routeA_frame{FRAME_ID:04d}_gt_patch320.png"
        )


if __name__ == "__main__":
    main()
