#!/usr/bin/env python3
"""Aggregate our v39 results and official Bearing-UAV reruns on identical route frames."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _find_ours_csv(out: Path, route: str, summary: dict) -> Path:
    p = Path(str(summary.get("CSV", "")))
    if p.exists():
        return p
    q = out / p.name
    if q.exists():
        return q
    matches = sorted(out.glob(f"{route}_*_frames.csv"))
    if not matches:
        raise FileNotFoundError(f"No ours CSV for {out.parent.name}/{route}")
    return matches[-1]


def _write_csv(path: Path, rows: List[dict]) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _ours_route(generated_root: Path, city: str, route: str) -> dict:
    out = generated_root / city / "v39_output_bearing_adapted"
    summary = json.loads((out / "bearing_v39_summary.json").read_text(encoding="utf-8"))[route]
    rows = _read_csv(_find_ours_csv(out, route, summary))
    errors = np.asarray([
        math.hypot(float(r["final_x"]) - float(r["gt_x"]),
                   float(r["final_y"]) - float(r["gt_y"]))
        for r in rows
    ], dtype=np.float64)

    paper_path = out / "bearing_paper_metrics.json"
    recall = None
    if paper_path.exists():
        paper = json.loads(paper_path.read_text(encoding="utf-8"))
        recall = paper["routes"][route].get("Recall@1_derived_same_quadrant_pct")

    return {
        "frames": int(len(errors)),
        "Recall@1_pct": float(recall) if recall is not None else None,
        "MLE_m": float(errors.mean()),
        "MedLE_m": float(np.median(errors)),
        "P90_m": float(np.percentile(errors, 90)),
        "LSR@5_pct": float(100.0 * np.mean(errors <= 5.0)),
        "LSR@10_pct": float(100.0 * np.mean(errors <= 10.0)),
        "LSR@15_pct": float(100.0 * np.mean(errors <= 15.0)),
        "LSR@20_pct": float(100.0 * np.mean(errors <= 20.0)),
        "HSR@15_pct": None,
        "MHE_deg": None,
        "MedHE_deg": None,
        "errors": errors.tolist(),
    }


def export(generated_root: Path, official_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    ours_all: List[float] = []
    bearing_all: List[float] = []
    bearing_head_all: List[float] = []
    ours_recall_good = 0.0
    bearing_recall_good = 0.0
    total_ours = 0
    total_bearing = 0

    for city in ("citya", "cityb", "cityc", "cityd"):
        official_payload = json.loads(
            (official_root / city / "official_bearinguav_same_route.json").read_text(encoding="utf-8")
        )
        for route in ("test_01", "test_02"):
            ours = _ours_route(generated_root, city, route)
            off = official_payload["routes"][route]
            if int(ours["frames"]) != int(off["frames"]):
                raise RuntimeError(
                    f"{city}/{route}: frame mismatch ours={ours['frames']} official={off['frames']}"
                )

            rows.append({
                "city": city,
                "route": route,
                "method": "Ours v39 Bearing-adapted",
                "result_source": "rerun on selected route frames",
                "frames": ours["frames"],
                "Recall@1_pct": ours["Recall@1_pct"],
                "MLE_m": ours["MLE_m"],
                "MedLE_m": ours["MedLE_m"],
                "P90_m": ours["P90_m"],
                "LSR@5_pct": ours["LSR@5_pct"],
                "LSR@10_pct": ours["LSR@10_pct"],
                "LSR@15_pct": ours["LSR@15_pct"],
                "LSR@20_pct": ours["LSR@20_pct"],
                "HSR@15_pct": "N/A",
                "MHE_deg": "N/A",
                "MedHE_deg": "N/A",
                "protocol": "temporal controlled-local-prior refinement",
            })
            rows.append({
                "city": city,
                "route": route,
                "method": "Bearing-UAV official VGG-16",
                "result_source": "official code+checkpoint rerun on identical selected route frames",
                "frames": off["frames"],
                "Recall@1_pct": off["Recall@1_pct"],
                "MLE_m": off["MLE_m"],
                "MedLE_m": off["MedLE_m"],
                "P90_m": off["P90_m"],
                "LSR@5_pct": off["LSR@5_pct"],
                "LSR@10_pct": off["LSR@10_pct"],
                "LSR@15_pct": off["LSR@15_pct"],
                "LSR@20_pct": off["LSR@20_pct"],
                "HSR@15_pct": off["HSR@15_pct"],
                "MHE_deg": off["MHE_deg"],
                "MedHE_deg": off["MedHE_deg"],
                "protocol": "single-frame four-neighbour RST pose regression",
            })

            ours_all.extend(ours["errors"])
            bearing_all.extend(off["distance_errors_m"])
            bearing_head_all.extend(off["heading_errors_deg"])
            if ours["Recall@1_pct"] is not None:
                ours_recall_good += float(ours["Recall@1_pct"]) * int(ours["frames"]) / 100.0
            bearing_recall_good += float(off["Recall@1_pct"]) * int(off["frames"]) / 100.0
            total_ours += int(ours["frames"])
            total_bearing += int(off["frames"])

    ours_arr = np.asarray(ours_all, dtype=np.float64)
    off_arr = np.asarray(bearing_all, dtype=np.float64)
    head_arr = np.asarray(bearing_head_all, dtype=np.float64)

    pooled = [
        {
            "method": "Ours v39 Bearing-adapted",
            "evaluation": "same 8 route frame sets",
            "frames": int(len(ours_arr)),
            "Recall@1_pct": float(100.0 * ours_recall_good / max(total_ours, 1)),
            "MLE_m": float(ours_arr.mean()),
            "MedLE_m": float(np.median(ours_arr)),
            "P90_m": float(np.percentile(ours_arr, 90)),
            "LSR@5_pct": float(100.0 * np.mean(ours_arr <= 5.0)),
            "LSR@10_pct": float(100.0 * np.mean(ours_arr <= 10.0)),
            "LSR@15_pct": float(100.0 * np.mean(ours_arr <= 15.0)),
            "LSR@20_pct": float(100.0 * np.mean(ours_arr <= 20.0)),
            "HSR@15_pct": "N/A",
            "MHE_deg": "N/A",
            "MedHE_deg": "N/A",
            "protocol": "temporal controlled-local-prior refinement",
        },
        {
            "method": "Bearing-UAV official VGG-16",
            "evaluation": "same 8 route frame sets",
            "frames": int(len(off_arr)),
            "Recall@1_pct": float(100.0 * bearing_recall_good / max(total_bearing, 1)),
            "MLE_m": float(off_arr.mean()),
            "MedLE_m": float(np.median(off_arr)),
            "P90_m": float(np.percentile(off_arr, 90)),
            "LSR@5_pct": float(100.0 * np.mean(off_arr <= 5.0)),
            "LSR@10_pct": float(100.0 * np.mean(off_arr <= 10.0)),
            "LSR@15_pct": float(100.0 * np.mean(off_arr <= 15.0)),
            "LSR@20_pct": float(100.0 * np.mean(off_arr <= 20.0)),
            "HSR@15_pct": float(100.0 * np.mean(head_arr <= 15.0)),
            "MHE_deg": float(head_arr.mean()),
            "MedHE_deg": float(np.median(head_arr)),
            "protocol": "single-frame four-neighbour RST pose regression",
        },
    ]

    _write_csv(output_dir / "same_route_route_level.csv", rows)
    _write_csv(output_dir / "same_route_pooled.csv", pooled)
    payload = {
        "comparison_type": "same selected Bearing-UAV route frames and GT",
        "stronger_than_published-only_reference": True,
        "remaining_protocol_difference": (
            "Our v39 consumes temporal history and a controlled local prior; official Bearing-UAV "
            "regresses position/heading from the current UAV image and four neighbouring satellite tiles."
        ),
        "route_level": rows,
        "pooled": pooled,
        "external_baseline_note": (
            "University-1652, SUES-200, DenseUAV and GTA-UAV are not rerun here because the public "
            "Bearing-UAV repository does not ship its adapted implementations/checkpoints. Their "
            "paper-reported values remain in the published-reference table only."
        ),
    }
    (output_dir / "same_route_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("[SAME-ROUTE-COMPARISON] PASS", flush=True)
    for row in pooled:
        print(
            f"  {row['method']}: MLE={row['MLE_m']:.3f}m "
            f"MedLE={row['MedLE_m']:.3f}m LSR15={row['LSR@15_pct']:.2f}%",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generated-root", required=True)
    p.add_argument("--official-result-root", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    export(Path(a.generated_root).resolve(), Path(a.official_result_root).resolve(), Path(a.output_dir).resolve())


if __name__ == "__main__":
    main()
