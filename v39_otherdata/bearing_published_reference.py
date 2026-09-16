#!/usr/bin/env python3
"""Export Bearing-UAV paper reference tables next to our multi-city results.

Important: University-1652, SUES-200, DenseUAV and GTA-UAV implementations are
NOT shipped in the public Bearing-UAV repository.  Their rows below are the
numbers REPORTED by the Bearing-UAV paper, not re-runs performed by this repo.
The output marks this explicitly so a thesis/paper cannot accidentally describe
published reference numbers as newly reproduced experiments.

Sources
-------
Main-paper UAV-view comparison (CVPR 2026):
  University-1652, SUES-200, DenseUAV, GTA-UAV,
  Bearing-UAV VGG-16, Bearing-UAV VGG-16 + weather augmentation.

Supplementary Table 2 (UAV view):
  Bearing-UAV ResNet18, ViT-Small, MobileNet-V3-Small, VGG-16 with
  Recall@1, LSR@15, HSR@15, MLE, MedLE, MHE and MedHE.

Our row uses the current four-city / eight-route output.  Its MLE, MedLE and
LSR use the same mathematical definitions, but the evaluation protocol differs:
v39 is controlled-local-prior temporal refinement on pseudo-flight sequences,
whereas Bearing-UAV performs four-adjacent-RST pose regression.  Therefore this
script never labels our row as an apples-to-apples reproduced benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


MAIN_UAV_ROWS = [
    {
        "method": "University-1652",
        "kind": "published external baseline",
        "Recall@1_pct": 60.20,
        "LSR@15_pct": 15.11,
        "HSR@15_pct": None,
        "MLE_m": 33.15,
        "MedLE_m": None,
        "MHE_deg": None,
        "MedHE_deg": None,
        "SR@20_pct": 0.00,
        "SPL_pct": None,
        "NE_m": 602.96,
        "source": "Bearing-UAV CVPR 2026 main comparison table, UAV view",
    },
    {
        "method": "SUES-200",
        "kind": "published external baseline",
        "Recall@1_pct": 66.60,
        "LSR@15_pct": 15.76,
        "HSR@15_pct": None,
        "MLE_m": 30.83,
        "MedLE_m": None,
        "MHE_deg": None,
        "MedHE_deg": None,
        "SR@20_pct": 0.00,
        "SPL_pct": None,
        "NE_m": 618.85,
        "source": "Bearing-UAV CVPR 2026 main comparison table, UAV view",
    },
    {
        "method": "DenseUAV",
        "kind": "published external baseline",
        "Recall@1_pct": 73.43,
        "LSR@15_pct": 16.54,
        "HSR@15_pct": None,
        "MLE_m": 28.79,
        "MedLE_m": None,
        "MHE_deg": None,
        "MedHE_deg": None,
        "SR@20_pct": 0.00,
        "SPL_pct": None,
        "NE_m": 651.93,
        "source": "Bearing-UAV CVPR 2026 main comparison table, UAV view",
    },
    {
        "method": "GTA-UAV",
        "kind": "published external baseline",
        "Recall@1_pct": 70.71,
        "LSR@15_pct": 27.96,
        "HSR@15_pct": None,
        "MLE_m": 28.43,
        "MedLE_m": None,
        "MHE_deg": None,
        "MedHE_deg": None,
        "SR@20_pct": 0.00,
        "SPL_pct": None,
        "NE_m": 661.91,
        "source": "Bearing-UAV CVPR 2026 main comparison table, UAV view",
    },
    {
        "method": "Bearing-UAV VGG-16",
        "kind": "published Bearing-UAV",
        "Recall@1_pct": 83.17,
        "LSR@15_pct": 89.36,
        "HSR@15_pct": 77.21,
        "MLE_m": 8.61,
        "MedLE_m": 7.30,
        "MHE_deg": 12.90,
        "MedHE_deg": 7.20,
        "SR@20_pct": 50.00,
        "SPL_pct": None,
        "NE_m": 275.61,
        "source": "Bearing-UAV CVPR 2026 main table + supplementary Table 2, UAV view",
    },
    {
        "method": "Bearing-UAV VGG-16 + weather augmentation",
        "kind": "published Bearing-UAV",
        "Recall@1_pct": 86.52,
        "LSR@15_pct": 92.88,
        "HSR@15_pct": None,
        "MLE_m": 7.48,
        "MedLE_m": None,
        "MHE_deg": 9.63,
        "MedHE_deg": None,
        "SR@20_pct": 25.00,
        "SPL_pct": None,
        "NE_m": 248.77,
        "source": "Bearing-UAV CVPR 2026 main comparison/weather table, UAV view",
    },
]


SUPPLEMENT_BACKBONES = [
    {
        "method": "Bearing-UAV ResNet18",
        "Recall@1_pct": 75.34,
        "LSR@15_pct": 71.83,
        "HSR@15_pct": 46.83,
        "MLE_m": 12.09,
        "MedLE_m": 10.56,
        "MHE_deg": 26.52,
        "MedHE_deg": 16.16,
    },
    {
        "method": "Bearing-UAV ViT-Small",
        "Recall@1_pct": 81.39,
        "LSR@15_pct": 85.47,
        "HSR@15_pct": 51.57,
        "MLE_m": 9.46,
        "MedLE_m": 8.02,
        "MHE_deg": 24.89,
        "MedHE_deg": 14.45,
    },
    {
        "method": "Bearing-UAV MobileNet-V3-Small",
        "Recall@1_pct": 79.76,
        "LSR@15_pct": 81.20,
        "HSR@15_pct": 64.94,
        "MLE_m": 10.34,
        "MedLE_m": 8.72,
        "MHE_deg": 19.53,
        "MedHE_deg": 10.14,
    },
    {
        "method": "Bearing-UAV VGG-16",
        "Recall@1_pct": 83.17,
        "LSR@15_pct": 89.36,
        "HSR@15_pct": 77.21,
        "MLE_m": 8.61,
        "MedLE_m": 7.30,
        "MHE_deg": 12.90,
        "MedHE_deg": 7.20,
    },
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _find_result_csv(out: Path, route: str, summary: dict) -> Path:
    p = Path(str(summary.get("CSV", "")))
    if p.exists():
        return p
    q = out / p.name
    if q.exists():
        return q
    matches = sorted(out.glob(f"{route}_*_frames.csv"))
    if not matches:
        raise FileNotFoundError(f"No frame CSV for {out.parent.name}/{route}")
    return matches[-1]


def _our_pooled(generated_root: Path) -> dict:
    errors: List[float] = []
    route_count = 0
    frame_count = 0
    for city in ("citya", "cityb", "cityc", "cityd"):
        out = generated_root / city / "v39_output_bearing_adapted"
        summary_path = out / "bearing_v39_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"Missing current result: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for route in ("test_01", "test_02"):
            rows = _read_csv(_find_result_csv(out, route, summary[route]))
            route_count += 1
            frame_count += len(rows)
            errors.extend(
                math.hypot(
                    float(r["final_x"]) - float(r["gt_x"]),
                    float(r["final_y"]) - float(r["gt_y"]),
                )
                for r in rows
            )

    arr = np.asarray(errors, dtype=np.float64)
    paper_agg_path = generated_root / "bearing_paper_comparison_multicity.json"
    derived_recall = None
    if paper_agg_path.exists():
        payload = json.loads(paper_agg_path.read_text(encoding="utf-8"))
        derived_recall = payload.get("weighted_Recall@1_derived_same_quadrant_pct")

    return {
        "method": "Ours v39 Bearing-adapted",
        "kind": "current experiment; different protocol",
        "Recall@1_pct": float(derived_recall) if derived_recall is not None else None,
        "LSR@15_pct": float(100.0 * np.mean(arr <= 15.0)),
        "HSR@15_pct": None,
        "MLE_m": float(arr.mean()),
        "MedLE_m": float(np.median(arr)),
        "MHE_deg": None,
        "MedHE_deg": None,
        "SR@20_pct": None,
        "SPL_pct": None,
        "NE_m": None,
        "source": (
            f"current four-city/eight-route run; {frame_count} frames, {route_count} routes; "
            "controlled-local-prior temporal refinement"
        ),
    }


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def export(generated_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ours = _our_pooled(generated_root)

    main_rows = [dict(row) for row in MAIN_UAV_ROWS]
    main_json = {
        "table": "Bearing-UAV CVPR 2026 published UAV-view reference rows",
        "reproduced_by_this_repo": False,
        "reason": (
            "The public Bearing-UAV repository does not ship the external baseline "
            "implementations for University-1652, SUES-200, DenseUAV or GTA-UAV."
        ),
        "rows": main_rows,
    }
    (output_dir / "bearinguav_published_uav_reference.json").write_text(
        json.dumps(main_json, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "bearinguav_published_uav_reference.csv", main_rows)

    supplement_rows = []
    for row in SUPPLEMENT_BACKBONES:
        item = dict(row)
        item["source"] = "Bearing-UAV CVPR 2026 supplementary Table 2, UAV view"
        supplement_rows.append(item)
    (output_dir / "bearinguav_backbone_supplement.json").write_text(
        json.dumps({"rows": supplement_rows}, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "bearinguav_backbone_supplement.csv", supplement_rows)

    comparison_rows = main_rows + [ours]
    comparison = {
        "rows": comparison_rows,
        "comparison_warning": (
            "Our MLE/MedLE/LSR@15 use the same mathematical definitions, but the "
            "protocol is not identical. v39 uses a controlled local prior and temporal "
            "pseudo-flight refinement; Bearing-UAV published rows use its own RST pose-"
            "regression/retrieval evaluation. Do not claim an apples-to-apples win."
        ),
        "our_recall_note": (
            "Our Recall@1 value is derived with Bearing-UAV's same-quadrant/sign decision "
            "criterion from a continuous final position, not from a native RST retrieval head."
        ),
    }
    (output_dir / "ours_vs_bearinguav_published.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "ours_vs_bearinguav_published.csv", comparison_rows)

    print("[PUBLISHED-REFERENCE] exported paper reference tables", flush=True)
    print(f"  our pooled MLE   = {ours['MLE_m']:.3f} m", flush=True)
    print(f"  our pooled MedLE = {ours['MedLE_m']:.3f} m", flush=True)
    print(f"  our LSR@15       = {ours['LSR@15_pct']:.2f}%", flush=True)
    if ours["Recall@1_pct"] is not None:
        print(f"  our Recall@1*    = {ours['Recall@1_pct']:.2f}% (derived)", flush=True)
    print("[PUBLISHED-REFERENCE] external rows are REPORTED values, not re-runs", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    export(Path(args.generated_root).resolve(), Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
