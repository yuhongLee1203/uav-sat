#!/usr/bin/env python3
"""Collect official Bearing-UAV Sat/UAV test CSVs into paper-ready tables."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


CITY = {"34bc": "CityA", "36bc": "CityB", "37bc": "CityC", "38bc": "CityD"}


def newest(root: Path) -> Path:
    xs = sorted(root.glob("test_results*/test_results96bc.csv"), key=lambda p: p.stat().st_mtime)
    if not xs:
        raise FileNotFoundError(f"official test_results96bc.csv missing below {root}")
    return xs[-1]


def metrics(df: pd.DataFrame) -> dict:
    recall = (np.sign(df.x_pred.to_numpy()) == np.sign(df.x_norm.to_numpy())) & \
             (np.sign(df.y_pred.to_numpy()) == np.sign(df.y_norm.to_numpy()))
    d = df.distance_error.to_numpy(float); h = df.angle_error.to_numpy(float)
    return {"Samples": len(df), "Recall@1_pct": 100 * recall.mean(),
            "LSR@15_pct": 100 * (d <= 15).mean(), "HSR@15_pct": 100 * (h <= 15).mean(),
            "MLE_m": d.mean(), "MedLE_m": np.median(d),
            "MHE_deg": h.mean(), "MedHE_deg": np.median(h)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sat-root"); p.add_argument("--uav-root")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    paths = {}
    if a.sat_root: paths["Sat"] = newest(Path(a.sat_root))
    if a.uav_root: paths["UAV"] = newest(Path(a.uav_root))
    if not paths: raise SystemExit("provide --sat-root and/or --uav-root")
    rows = []
    for view, path in paths.items():
        df = pd.read_csv(path, dtype={"rsi_id": str})
        df["rsi_id"] = df.rsi_id.str.lower()
        for rid, city in CITY.items():
            rows.append({"City": city, "View": view, **metrics(df[df.rsi_id == rid])})
        rows.append({"City": "ALL", "View": view, **metrics(df)})
        raw = out / "raw" / view.lower(); raw.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, raw / path.name)
        for figure in path.parent.glob("*.jpg"):
            shutil.copy2(figure, raw / figure.name)
        mae = path.parent / "test_mae.json"
        if mae.exists(): shutil.copy2(mae, raw / mae.name)
    with (out / "official_geo_by_city.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (out / "official_geo_by_city.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md = ["# Official Bearing-UAV geo-localization", "",
          "| City | View | Recall@1 | LSR@15 | HSR@15 | MLE m | MedLE m | MHE deg | MedHE deg |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append("| {City} | {View} | {Recall@1_pct:.3f} | {LSR@15_pct:.3f} | {HSR@15_pct:.3f} | {MLE_m:.3f} | {MedLE_m:.3f} | {MHE_deg:.3f} | {MedHE_deg:.3f} |".format(**r))
    (out / "OFFICIAL_GEO_TABLE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[OFFICIAL GEO TABLE] {out}")


if __name__ == "__main__":
    main()
