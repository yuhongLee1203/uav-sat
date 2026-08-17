#!/usr/bin/env python3
"""Collect V36 backbone results without averaging route-level percentages."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


THRESHOLDS = (3, 5, 10, 15)


def read_errors(output_dir: Path, route: str) -> np.ndarray:
    files = sorted(output_dir.glob(f"{route}_*_frames.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        return np.empty(0, dtype=np.float64)
    with files[-1].open(newline="", encoding="utf-8") as handle:
        return np.asarray(
            [float(row["error_final_m"]) for row in csv.DictReader(handle)], dtype=np.float64
        )


def error_metrics(errors: np.ndarray) -> dict:
    if not errors.size:
        return {"MLE_m": None, "P90_m": None, **{f"LSR@{t}_pct": None for t in THRESHOLDS}}
    values = {"MLE_m": float(np.mean(errors)), "P90_m": float(np.quantile(errors, 0.90))}
    values.update({f"LSR@{t}_pct": float(np.mean(errors <= t) * 100.0) for t in THRESHOLDS})
    return values


def combined_timing(summary: dict) -> dict:
    timings = [summary.get(route, {}).get("EndToEndTiming") for route in ("route_B", "route_C")]
    timings = [item for item in timings if item]
    if not timings:
        return {"E2E_mean_ms": None, "E2E_p90_ms": None, "E2E_p95_ms": None, "E2E_samples": 0}
    weights = np.asarray([max(1, int(item.get("samples", 0))) for item in timings], dtype=np.float64)
    def weighted(key):
        return float(np.average([float(item[key]) for item in timings], weights=weights))
    return {
        "E2E_mean_ms": weighted("mean_ms"),
        "E2E_p90_ms": weighted("p90_ms"),
        "E2E_p95_ms": weighted("p95_ms"),
        "E2E_samples": int(np.sum(weights)),
    }


def number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backbones", nargs="+", required=True)
    args = parser.parse_args()

    rows = []
    for backbone in args.backbones:
        output_dir = args.output_root / f"v36_{backbone}"
        summary_path = output_dir / "robust_tracker_summary.json"
        row = {"backbone": backbone, "status": "incomplete", "output_dir": str(output_dir)}
        if not summary_path.exists():
            rows.append(row)
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            b_errors = read_errors(output_dir, "route_B")
            c_errors = read_errors(output_dir, "route_C")
            all_errors = np.concatenate((b_errors, c_errors))
            row.update({"status": "complete", "B_frames": int(b_errors.size), "C_frames": int(c_errors.size)})
            for prefix, errors in (("B", b_errors), ("C", c_errors), ("BC", all_errors)):
                row.update({f"{prefix}_{key}": value for key, value in error_metrics(errors).items()})
            row.update(combined_timing(summary))
            row["timing_definition"] = (
                "prepared UAV tensor -> backbone -> visual retrieval/GRU -> external RouteKalman -> final XY"
            )
        except (OSError, ValueError, KeyError) as exc:
            row["status"] = f"invalid: {exc}"
        rows.append(row)

    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "v36_backbone_comparison.csv"
    keys = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "v36_backbone_comparison.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_path = args.output_root / "v36_backbone_comparison.md"
    lines = [
        "# V36 backbone comparison",
        "",
        "Error is pooled over Route B+C frames (not an average of two route averages). "
        "Latency is a sample-count-weighted B+C value. Cache is excluded from latency: it is used "
        "only for training/evaluation feature reuse, while the timed path runs the real backbone.",
        "",
        "| Backbone | Status | MLE (m) | P90 (m) | LSR@3 | LSR@5 | LSR@10 | LSR@15 | E2E mean (ms) | E2E p95 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {backbone} | {status} | {mle} | {p90} | {l3} | {l5} | {l10} | {l15} | {mean} | {p95} |".format(
                backbone=row["backbone"], status=row["status"],
                mle=number(row.get("BC_MLE_m")), p90=number(row.get("BC_P90_m")),
                l3=number(row.get("BC_LSR@3_pct"), 2), l5=number(row.get("BC_LSR@5_pct"), 2),
                l10=number(row.get("BC_LSR@10_pct"), 2), l15=number(row.get("BC_LSR@15_pct"), 2),
                mean=number(row.get("E2E_mean_ms"), 2), p95=number(row.get("E2E_p95_ms"), 2),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"CSV : {csv_path}")
    print(f"JSON: {args.output_root / 'v36_backbone_comparison.json'}")
    print(f"MD  : {md_path}")


if __name__ == "__main__":
    main()
