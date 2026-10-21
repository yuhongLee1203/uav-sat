#!/usr/bin/env python3
"""Combine measured v39 GT-protocol ablations with synthetic-route proxies.

Official Bearing-Naver SR/SPL/NE require closed-loop navigation logs. The
prepared c4m episodes contain independent geotagged frames, so the route
metrics computed here are explicitly offline trajectory proxies.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from build_iclr_ablation_tables import _csv_for, _navigation_keys

VARIANTS = ("full", "no_gru", "no_kalman", "no_ms", "frames1", "frames2",
            "grid4", "grid5", "grid7", "grid8")


def path_length(points: list[tuple[float, float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def route_proxy(prepared: Path, output: Path, route: str, summary: dict) -> dict:
    with _csv_for(summary, output).open(newline="", encoding="utf-8") as handle:
        frames = list(csv.DictReader(handle))
    if not frames:
        raise RuntimeError(f"No frames in {output}/{route}")
    waypoints = json.loads((prepared / "routes" / route / "waypoints.json").read_text())
    if waypoints.get("coordinate_system") != "bearing_rsi_map_meters":
        raise RuntimeError(f"Unexpected waypoint coordinate system: {prepared}/{route}")
    planned = [(float(p["latitude"]), float(p["longitude"]))
               for p in waypoints["waypoints"]]
    predicted = [(float(r["final_x"]), float(r["final_y"])) for r in frames]
    endpoint = math.dist(predicted[-1],
                         (float(frames[-1]["gt_x"]), float(frames[-1]["gt_y"])))
    success = endpoint <= 20.0
    optimal = path_length(planned)
    actual = path_length(predicted)
    spl = optimal / max(optimal, actual, 1e-9) if success else 0.0
    return {"route": route, "frames": len(frames), "NE_m": endpoint,
            "SR@20": int(success), "SPL": spl,
            "LSR@3_hits": sum(float(r["error_final_m"]) <= 3.0 for r in frames),
            "LSR@5_hits": sum(float(r["error_final_m"]) <= 5.0 for r in frames),
            "planned_path_m": optimal, "predicted_path_m": actual}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--cities", nargs="+", default=["citya", "cityb", "cityc", "cityd"])
    args = parser.parse_args()
    root = args.suite_root.resolve()
    localization = json.loads((root / "paper_ablation_results.json").read_text())
    overlap_path = root / "source_train_overlap_audit.json"
    overlap = json.loads(overlap_path.read_text()) if overlap_path.is_file() else None
    if set(args.cities) != set(localization["cities"]):
        raise RuntimeError("Localization table city set differs from requested navigation city set")
    rows = []
    route_rows = []
    for variant in VARIANTS:
        routes = []
        for city in args.cities:
            output = root / city / "variants" / variant
            manifest = json.loads((output / "experiment_manifest.json").read_text())
            if manifest["reference_protocol"] != "controlled_gt_jitter" or not manifest["uses_controlled_gt_reference"]:
                raise RuntimeError(f"{city}/{variant} did not run the requested GT protocol")
            prepared = root / city / "prepared"
            summaries = json.loads((output / "bearing_v39_summary.json").read_text())
            for route, key in _navigation_keys(summaries):
                result = route_proxy(prepared, output, route, summaries[key])
                result.update({"city": city, "variant": variant})
                routes.append(result)
                route_rows.append(result)
        base = localization["rows"][variant]
        total_frames = sum(r["frames"] for r in routes)
        rows.append({
            "variant": variant, "frames": base["frames"], "routes": len(routes),
            "R@1_star_pct": base["R@1*_pct"],
            "LSR@3_pct": 100 * sum(r["LSR@3_hits"] for r in routes) / total_frames,
            "LSR@5_pct": 100 * sum(r["LSR@5_hits"] for r in routes) / total_frames,
            "LSR@15_pct": base["LSR@15_pct"], "HSR@15_pct": base["HSR@15_pct"],
            "MLE_m": base["MLE_m"], "MHE_deg": base["MHE_deg"],
            "SR@20_proxy_pct": 100 * sum(r["SR@20"] for r in routes) / len(routes),
            "SPL_proxy_pct": 100 * sum(r["SPL"] for r in routes) / len(routes),
            "NE_proxy_m": sum(r["NE_m"] for r in routes) / len(routes),
        })
    payload = {
        "protocol": "official96_seed42_reassembled_synthetic_temporal_GT_controlled_jitter",
        "cities": args.cities,
        "R@1_star_definition": "Same-quadrant/sign success derived from final continuous XY; not native four-tile retrieval Recall@1.",
        "navigation_definition": "Offline synthetic-episode endpoint and predicted-path proxy; not closed-loop Bearing-Naver SR/SPL/NE.",
        "source_train_overlap_audit": overlap,
        "rows": rows, "route_rows": route_rows,
    }
    (root / "official96_gt_all_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (root / "official96_gt_all_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# c4m Official96 reassembled temporal episodes, GT-assisted v39 ablation", "",
             "R@1* is a same-quadrant proxy from final XY. Official Recall@1 needs a",
             "dedicated four-reference-tile ranking output. SR@20/SPL/NE below are offline",
             "synthetic-route proxies, because these images are not a recorded flight or",
             "closed-loop navigation rollout. GT enters inference through controlled jitter.", "",
             "| Variant | R@1* ↑ | LSR@3 ↑ | LSR@5 ↑ | LSR@15 ↑ | HSR@15 ↑ | MLE m ↓ | MHE deg ↓ | SR@20 proxy ↑ | SPL proxy ↑ | NE proxy m ↓ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['R@1_star_pct']:.2f}% | "
                     f"{r['LSR@3_pct']:.2f}% | {r['LSR@5_pct']:.2f}% | "
                     f"{r['LSR@15_pct']:.2f}% | {r['HSR@15_pct']:.2f}% | "
                     f"{r['MLE_m']:.3f} | {r['MHE_deg']:.2f} | "
                     f"{r['SR@20_proxy_pct']:.2f}% | {r['SPL_proxy_pct']:.2f}% | "
                     f"{r['NE_proxy_m']:.3f} |")
    if overlap is not None:
        lines += ["", f"Source-training images reused in Official96 test episodes: "
                  f"{overlap['total_test_overlap']}. This transfer is not an independent "
                  "held-out test; see `source_train_overlap_audit.json`."]
    (root / "official96_gt_all_metrics.md").write_text("\n".join(lines) + "\n")
    print("[GT ALL METRICS]", root / "official96_gt_all_metrics.md")


if __name__ == "__main__":
    main()
