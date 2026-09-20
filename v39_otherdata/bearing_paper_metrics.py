#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import bearing_prepare as bearing

CITIES = ("citya", "cityb", "cityc", "cityd")
NAV_TO_INTERNAL = {"nav50": "test_01", "nav51": "test_02"}

# UAV-view numbers from Bearing-UAV main Table 2; VGG-16 MedLE/MedHE from
# supplementary Table 8. Navigation is kept in a separate table because our
# current experiment is offline route replay rather than Bearing-Naver.
LOCALIZATION_LITERATURE = [
    {"Method": "University-1652", "Recall@1_UAV_pct": 60.20, "LSR@15_UAV_pct": 15.11,
     "HSR@15_UAV_pct": None, "MLE_UAV_m": 33.15, "MedLE_UAV_m": None,
     "MHE_UAV_deg": None, "MedHE_UAV_deg": None},
    {"Method": "SUES-200", "Recall@1_UAV_pct": 66.60, "LSR@15_UAV_pct": 15.76,
     "HSR@15_UAV_pct": None, "MLE_UAV_m": 30.83, "MedLE_UAV_m": None,
     "MHE_UAV_deg": None, "MedHE_UAV_deg": None},
    {"Method": "DenseUAV", "Recall@1_UAV_pct": 73.43, "LSR@15_UAV_pct": 16.54,
     "HSR@15_UAV_pct": None, "MLE_UAV_m": 28.79, "MedLE_UAV_m": None,
     "MHE_UAV_deg": None, "MedHE_UAV_deg": None},
    {"Method": "GTA-UAV", "Recall@1_UAV_pct": 70.71, "LSR@15_UAV_pct": 27.96,
     "HSR@15_UAV_pct": None, "MLE_UAV_m": 28.43, "MedLE_UAV_m": None,
     "MHE_UAV_deg": None, "MedHE_UAV_deg": None},
    {"Method": "Bearing-UAV (VGG-16)", "Recall@1_UAV_pct": 83.17,
     "LSR@15_UAV_pct": 89.36, "HSR@15_UAV_pct": 77.21,
     "MLE_UAV_m": 8.61, "MedLE_UAV_m": 7.30,
     "MHE_UAV_deg": 12.90, "MedHE_UAV_deg": 7.20},
]

NAVIGATION_LITERATURE = [
    {"Method": "University-1652", "SR@20_UAV_pct": 0.00, "SPL_UAV_pct": 0.00, "NE_UAV_m": 602.96},
    {"Method": "SUES-200", "SR@20_UAV_pct": 0.00, "SPL_UAV_pct": 0.00, "NE_UAV_m": 618.85},
    {"Method": "DenseUAV", "SR@20_UAV_pct": 0.00, "SPL_UAV_pct": 0.00, "NE_UAV_m": 651.93},
    {"Method": "GTA-UAV", "SR@20_UAV_pct": 0.00, "SPL_UAV_pct": 0.00, "NE_UAV_m": 661.91},
    {"Method": "Bearing-UAV (VGG-16)", "SR@20_UAV_pct": 50.00, "SPL_UAV_pct": 29.82, "NE_UAV_m": 275.61},
    {"Method": "Yours", "SR@20_UAV_pct": None, "SPL_UAV_pct": None, "NE_UAV_m": None},
]


def q(a, p): return float(np.percentile(np.asarray(a, float), p))
def plen(x, y): return float(np.hypot(np.diff(x), np.diff(y)).sum())

def read_csv(p):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_csv(full, nav):
    prefix = "route_B" if nav == "nav50" else "route_C"
    m = sorted(full.glob(prefix + "_*_frames.csv"))
    if not m:
        raise FileNotFoundError(f"{full}: {nav} frames csv missing")
    return m[-1]


def metrics(rows):
    err = np.asarray([float(r["error_final_m"]) for r in rows])
    he = np.asarray([abs(float(r["heading_error_deg"])) for r in rows if r.get("heading_error_deg", "") != ""])
    gx = np.asarray([float(r["gt_x"]) for r in rows]); gy = np.asarray([float(r["gt_y"]) for r in rows])
    px = np.asarray([float(r["final_x"]) for r in rows]); py = np.asarray([float(r["final_y"]) for r in rows])
    lat = np.asarray([float(r["end_to_end_latency_ms"]) for r in rows if r.get("end_to_end_latency_ms", "") != ""])
    ne = float(math.hypot(px[-1] - gx[-1], py[-1] - gy[-1])); sr = float(ne <= 20.0)
    gl = plen(gx, gy); pl = plen(px, py); spl = sr * gl / max(gl, pl, 1e-9)
    return {
        "Frames": len(rows), "MLE_m": float(err.mean()), "MedLE_m": float(np.median(err)),
        "P90_m": q(err, 90), "P95_m": q(err, 95), "P99_m": q(err, 99),
        "LSR@5_pct": float((err <= 5).mean() * 100), "LSR@10_pct": float((err <= 10).mean() * 100),
        "LSR@15_pct": float((err <= 15).mean() * 100), "LSR@20_pct": float((err <= 20).mean() * 100),
        "MHE_deg": float(he.mean()) if len(he) else None,
        "MedHE_deg": float(np.median(he)) if len(he) else None,
        "HSR@15_pct": float((he <= 15).mean() * 100) if len(he) else None,
        "NE_m_route_replay": ne, "SR@20_pct_route_replay": 100 * sr,
        "SPL_pct_route_replay": 100 * spl, "GTPath_m": gl, "PredPath_m": pl,
        "JumpRate_pct": 100 * sum(int(float(r.get("abnormal_jump", "0") or 0)) != 0 for r in rows) / len(rows),
        "MaxFinalStep_m": max(float(r["final_step_m"]) for r in rows),
        "InferenceMean_ms": float(lat.mean()) if len(lat) else None,
        "FPS": float(1000 / lat.mean()) if len(lat) and lat.mean() > 0 else None,
    }


def four_rst_recall(prepared_root: Path, nav: str, result_rows) -> float:
    """Derived Bearing-UAV Recall@1 decision from our continuous final position.

    Bearing-UAV defines Recall@1 as choosing, among four adjacent RSTs, the RST
    closest to the UVP.  Its four RSTs correspond to the four quadrants around
    the RSB center.  We map our continuous prediction back to the same quadrant
    decision and compare it with the GT quadrant from official metadata.
    """
    exp = json.loads((prepared_root / "experiment.json").read_text(encoding="utf-8"))
    metadata = pd.read_csv(exp["metadata_csv"])
    city_rows = bearing._city_rows(metadata, exp["city"])
    manifest = read_csv(prepared_root / "routes" / NAV_TO_INTERNAL[nav] / "manifest.csv")
    if len(manifest) != len(result_rows):
        raise RuntimeError(f"{exp['city']} {nav}: manifest/result length mismatch")
    good = 0
    for man, pred in zip(manifest, result_rows):
        meta = city_rows.iloc[int(man["source_index"])]
        cx = float(meta["block_x"]) * bearing.PATCH_SIZE + bearing.PATCH_SIZE
        cy = float(meta["block_y"]) * bearing.PATCH_SIZE + bearing.PATCH_SIZE
        gt_dx = float(meta["x_norm"]) * bearing.PATCH_SIZE
        gt_dy = float(meta["y_norm"]) * bearing.PATCH_SIZE
        pred_px = float(pred["final_x"]) / float(exp["mpp"])
        pred_py = float(pred["final_y"]) / float(exp["mpp"])
        gt_quadrant = (gt_dx >= 0.0, gt_dy >= 0.0)
        pred_quadrant = (pred_px - cx >= 0.0, pred_py - cy >= 0.0)
        good += int(gt_quadrant == pred_quadrant)
    return 100.0 * good / max(len(result_rows), 1)


def md(headers, rows):
    def f(v):
        if v is None: return "—"
        if isinstance(v, float): return f"{v:.3f}"
        return str(v)
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ] + ["| " + " | ".join(f(r.get(h)) for h in headers) + " |" for r in rows])


def write_csv(p, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def checkpoint_size_mb(root: Path):
    total = 0
    files = []
    for city in CITIES:
        candidates = [root / city / "train_frames3" / "checkpoints", root / city / "train_full" / "checkpoints"]
        ckdir = next((p for p in candidates if p.is_dir()), None)
        if ckdir is None: continue
        for name in ("visual_retrieval_A_only.pt", "controlled_gtprior_forward3x6_continuous_waypoint_state_gru_A_only.pt"):
            p = ckdir / name
            if p.is_file():
                total += p.stat().st_size
                files.append(str(p))
    # Average per city, because the same architecture is trained independently per city.
    return (total / max(len(CITIES), 1)) / (1024.0 ** 2), files


def main():
    a = argparse.ArgumentParser(); a.add_argument("--suite-root", required=True); a.add_argument("--output-dir")
    x = a.parse_args(); root = Path(x.suite_root).resolve(); out = Path(x.output_dir or root / "paper_benchmark")
    out.mkdir(parents=True, exist_ok=True)
    routes = []; pooled = []; recall_weighted = 0.0; recall_frames = 0
    for city in CITIES:
        full = root / city / "variants" / "full"
        if not (full / "bearing_v39_summary.json").is_file(): raise RuntimeError(f"missing summary: {city}")
        prepared = root / city / "prepared"
        for nav in ("nav50", "nav51"):
            cp = find_csv(full, nav); rows = read_csv(cp); pooled += rows
            recall = four_rst_recall(prepared, nav, rows)
            recall_weighted += recall * len(rows); recall_frames += len(rows)
            r = {"City": city, "Route": nav, **metrics(rows), "Recall@1_4RST_derived_pct": recall, "CSV": str(cp)}
            routes.append(r)
    pm = metrics(pooled)
    recall_all = recall_weighted / max(recall_frames, 1)
    ours = {
        "Method": "Yours (Forward-18 + GRU + Kalman + SoftMS)",
        "Recall@1_UAV_pct": recall_all,
        "LSR@15_UAV_pct": pm["LSR@15_pct"], "HSR@15_UAV_pct": pm["HSR@15_pct"],
        "MLE_UAV_m": pm["MLE_m"], "MedLE_UAV_m": pm["MedLE_m"],
        "MHE_UAV_deg": pm["MHE_deg"], "MedHE_UAV_deg": pm["MedHE_deg"],
    }
    cities = []
    for c in CITIES:
        rr = [r for r in routes if r["City"] == c]
        weights = np.asarray([r["Frames"] for r in rr], dtype=float)
        cities.append({
            "City": c,
            "Recall@1_4RST_derived_pct": float(np.average([r["Recall@1_4RST_derived_pct"] for r in rr], weights=weights)),
            "MLE_m": float(np.average([r["MLE_m"] for r in rr], weights=weights)),
            "MedLE_m": float(np.average([r["MedLE_m"] for r in rr], weights=weights)),
            "LSR@15_pct": float(np.average([r["LSR@15_pct"] for r in rr], weights=weights)),
            "MHE_deg": float(np.average([r["MHE_deg"] for r in rr], weights=weights)),
            "MedHE_deg": float(np.average([r["MedHE_deg"] for r in rr], weights=weights)),
            "HSR@15_pct": float(np.average([r["HSR@15_pct"] for r in rr], weights=weights)),
        })
    comparison = LOCALIZATION_LITERATURE + [ours]
    model_mb, ckpts = checkpoint_size_mb(root)
    efficiency = [{
        "Method": "Yours",
        "OnDiskModelSize_MB_per_city": model_mb,
        "EndToEndInferenceMean_ms": pm["InferenceMean_ms"],
        "FPS": pm["FPS"],
        "GFLOPs": None,
        "Note": "GFLOPs not reported until the complete cached-SAT + UAV + temporal pipeline is profiled with the same input convention as baselines.",
    }]
    replay = [{
        "Method": "Yours route replay diagnostic",
        "SR@20_pct_route_replay": float(np.mean([r["SR@20_pct_route_replay"] for r in routes])),
        "SPL_pct_route_replay": float(np.mean([r["SPL_pct_route_replay"] for r in routes])),
        "NE_m_route_replay": float(np.mean([r["NE_m_route_replay"] for r in routes])),
    }]
    payload = {
        "suite": str(root), "ours_main": ours, "per_route": routes, "per_city": cities,
        "localization_literature_comparison": comparison,
        "bearing_naver_navigation_reference": NAVIGATION_LITERATURE,
        "route_replay_navigation_diagnostic": replay,
        "efficiency": efficiency,
        "checkpoint_files": ckpts,
        "fairness": {
            "Recall@1": "Derived with the same four-adjacent-RST quadrant decision from our continuous prediction; marked derived because our network is not an RST retrieval classifier.",
            "navigation": "Our route-replay SR/SPL/NE are not inserted into Bearing-Naver closed-loop comparison.",
            "localization_heading": "Computed directly from raw per-frame predictions; no display smoothing is used.",
        },
    }
    (out / "bearing_paper_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out / "table_route_metrics.csv", routes); write_csv(out / "table_city_metrics.csv", cities)
    write_csv(out / "table_localization_literature_comparison.csv", comparison)
    write_csv(out / "table_navigation_reference.csv", NAVIGATION_LITERATURE)
    write_csv(out / "table_route_replay_navigation.csv", replay); write_csv(out / "table_efficiency.csv", efficiency)

    lines = [
        "# Bearing-UAV aligned paper tables", "",
        "## A. Localization + heading (paper-facing)", "",
        md(["Method", "Recall@1_UAV_pct", "LSR@15_UAV_pct", "HSR@15_UAV_pct", "MLE_UAV_m", "MedLE_UAV_m", "MHE_UAV_deg", "MedHE_UAV_deg"], comparison), "",
        "## B. Multi-city results", "",
        md(["City", "Recall@1_4RST_derived_pct", "MLE_m", "MedLE_m", "LSR@15_pct", "MHE_deg", "MedHE_deg", "HSR@15_pct"], cities), "",
        "## C. Bearing-Naver navigation reference (do not insert route-replay values here)", "",
        md(["Method", "SR@20_UAV_pct", "SPL_UAV_pct", "NE_UAV_m"], NAVIGATION_LITERATURE), "",
        "## D. Route-replay navigation diagnostics", "",
        md(["City", "Route", "NE_m_route_replay", "SR@20_pct_route_replay", "SPL_pct_route_replay", "JumpRate_pct", "MaxFinalStep_m"], routes), "",
        "## E. Efficiency", "", md(["Method", "OnDiskModelSize_MB_per_city", "EndToEndInferenceMean_ms", "FPS", "GFLOPs"], efficiency), "",
        "## Protocol notes", "",
        "- Recall@1 for ours is explicitly marked 4-RST derived: our continuous final XY is mapped to Bearing-UAV's four-adjacent-RST decision.",
        "- Route-replay SR/SPL/NE are diagnostics only and are not Bearing-Naver closed-loop results.",
        "- MLE/MedLE/LSR@15/MHE/MedHE/HSR@15 are computed from raw per-frame outputs.",
    ]
    (out / "PAPER_TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[PAPER BENCHMARK DONE]", out); print(json.dumps(ours, indent=2))

if __name__ == "__main__": main()
