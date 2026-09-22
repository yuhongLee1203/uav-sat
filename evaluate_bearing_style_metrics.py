#!/usr/bin/env python3
"""Recompute Bearing-UAV-style metrics from existing *_frames.csv results.

This is post-processing only: no model training or inference is run.
Default metrics intentionally exclude P90 and jump-rate.

Bearing-UAV compatible metrics used here:
  LSR@15 : percentage of frames with localization error <= 15 m
  HSR@15 : percentage of frames with circular heading error <= 15 deg
  MLE    : mean localization error (m)
  MedLE  : median localization error (m)
  MHE    : mean circular heading error (deg)
  MedHE  : median circular heading error (deg)

Recall@1 is intentionally NOT reported because Bearing-UAV defines it as
retrieving, among four adjacent RSTs, the RST whose location is closest to
the UVP. The v39 3x6 local-search tracker uses a different candidate protocol,
so relabeling its local top-1 as Bearing-UAV Recall@1 would not be comparable.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def circular_error_deg(pred, gt):
    p, g = fnum(pred), fnum(gt)
    if p is None or g is None:
        return None
    return abs((p - g + 180.0) % 360.0 - 180.0)


def localization_error(row):
    e = fnum(row.get("error_final_m"))
    if e is not None:
        return abs(e)
    px, py = fnum(row.get("final_x")), fnum(row.get("final_y"))
    gx, gy = fnum(row.get("gt_x")), fnum(row.get("gt_y"))
    if None not in (px, py, gx, gy):
        return math.hypot(px - gx, py - gy)
    return None


def heading_error(row):
    # Prefer recomputation from headings so the 0/360 boundary is always correct.
    e = circular_error_deg(row.get("estimated_heading_deg"), row.get("gt_heading_deg"))
    if e is not None:
        return e
    raw = fnum(row.get("heading_error_deg"))
    if raw is None:
        return None
    return abs((raw + 180.0) % 360.0 - 180.0)


def pct(values, threshold):
    if not values:
        return None
    return 100.0 * sum(v <= threshold for v in values) / len(values)


def mean(values):
    return statistics.fmean(values) if values else None


def med(values):
    return statistics.median(values) if values else None


def fmt(v):
    return "" if v is None else f"{v:.2f}"


def collect(root: Path):
    grouped = defaultdict(lambda: {"loc": [], "head": [], "files": []})
    frame_files = sorted(root.rglob("*_frames.csv"))
    for path in frame_files:
        loc, head = [], []
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    le = localization_error(row)
                    he = heading_error(row)
                    if le is not None:
                        loc.append(le)
                    if he is not None:
                        head.append(he)
        except Exception as exc:
            print(f"[skip] {path}: {exc}")
            continue
        if not loc:
            continue
        key = path.parent
        grouped[key]["loc"].extend(loc)
        grouped[key]["head"].extend(head)
        grouped[key]["files"].append(path)
    return grouped


def infer_labels(root: Path, folder: Path):
    rel = folder.relative_to(root)
    parts = rel.parts
    run = parts[0] if parts else folder.name
    city = next((p for p in parts if p.lower() in {"citya", "cityb", "cityc", "cityd"}), "")
    method = parts[-1] if parts else folder.name
    return run, city, method, str(rel)


def write_outputs(root: Path, output_base: Path, grouped):
    rows = []
    for folder, data in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        loc, head = data["loc"], data["head"]
        run, city, method, rel = infer_labels(root, folder)
        rows.append({
            "Run": run,
            "City": city,
            "Method": method,
            "Path": rel,
            "LSR@15↑": pct(loc, 15.0),
            "HSR@15↑": pct(head, 15.0),
            "MLE↓": mean(loc),
            "MedLE↓": med(loc),
            "MHE↓": mean(head),
            "MedHE↓": med(head),
            "N_loc": len(loc),
            "N_heading": len(head),
        })

    output_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_base.with_suffix(".csv")
    md_path = output_base.with_suffix(".md")

    fields = ["Run", "City", "Method", "Path", "LSR@15↑", "HSR@15↑", "MLE↓", "MedLE↓", "MHE↓", "MedHE↓", "N_loc", "N_heading"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (fmt(v) if isinstance(v, float) else v) for k, v in r.items()})

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# Bearing-UAV-style evaluation\n\n")
        fh.write("Metrics: LSR@15 (<=15 m), HSR@15 (<=15 deg), MLE/MedLE, MHE/MedHE. P90 and jump-rate are intentionally omitted.\n\n")
        fh.write("| Run | City | Method | LSR@15↑ | HSR@15↑ | MLE↓ | MedLE↓ | MHE↓ | MedHE↓ | N |\n")
        fh.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            fh.write(
                f"| {r['Run']} | {r['City']} | {r['Method']} | {fmt(r['LSR@15↑'])} | {fmt(r['HSR@15↑'])} | "
                f"{fmt(r['MLE↓'])} | {fmt(r['MedLE↓'])} | {fmt(r['MHE↓'])} | {fmt(r['MedHE↓'])} | {r['N_loc']} |\n"
            )

    print(f"Found {sum(len(v['files']) for v in grouped.values())} frame CSV files in {len(grouped)} experiment folders.")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    if not rows:
        raise SystemExit("No usable *_frames.csv files found under the selected root.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="paper_results", help="Root containing existing *_frames.csv result files")
    ap.add_argument("--output", default=None, help="Output base path without extension")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Result root does not exist: {root}")
    output = Path(args.output).resolve() if args.output else root / "bearing_style_metrics"
    grouped = collect(root)
    write_outputs(root, output, grouped)


if __name__ == "__main__":
    main()
