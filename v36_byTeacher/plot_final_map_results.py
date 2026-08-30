#!/usr/bin/env python3
"""Compatibility entry point for final map plots of the autonomous tracker."""

import argparse
from pathlib import Path

import config
from plot_final_trajectory import plot_route


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routes", nargs="+", choices=config.ROUTE_NAMES, default=["route_C", "route_B"]
    )
    parser.add_argument("--show-intermediate", action="store_true")
    parser.add_argument("--padding-px", type=float, default=140.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or (Path(config.OUTPUT_DIR) / "plots_final_map")
    for route_name in args.routes:
        csv_path = Path(config.OUTPUT_DIR) / (
            f"{route_name}_autonomous_ms1_kf_gru_ms2_frames.csv"
        )
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        plot_route(
            route_name,
            csv_path,
            output_dir / f"{route_name}_autonomous_final_map.png",
            show_intermediate=bool(args.show_intermediate),
            padding_px=float(args.padding_px),
        )


if __name__ == "__main__":
    main()
