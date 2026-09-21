#!/usr/bin/env python3
"""Benchmark only the final coordinate reduction for visual decoders.

MeanShift timing deliberately starts after convergence and basin consolidation:
it reduces two converged modes to one XY.  Weighted reduces all 36 patch
locations.  Feature extraction, candidate scoring and MeanShift iterations are
excluded from both measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch


def one(coords: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (coords * weights[:, None]).sum(dim=0)


def measure(coords, weights, repeats, warmup, device):
    for _ in range(warmup):
        one(coords, weights)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        if device.type == "cuda":
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            one(coords, weights)
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        else:
            start = time.perf_counter_ns()
            one(coords, weights)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    values = torch.tensor(samples, dtype=torch.float64)
    return {
        "latency_mean_ms": float(values.mean()),
        "latency_median_ms": float(values.median()),
        "latency_p90_ms": float(torch.quantile(values, 0.90)),
        "latency_stdev_ms": float(statistics.pstdev(samples)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--repeats", type=int, default=5000)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--seed", type=int, default=2033)
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rows = []
    for decoder, count, scope in (
        ("Weighted", 36, "36 scored patch coordinates"),
        ("MeanShift", 2, "2 converged modes; convergence excluded"),
    ):
        coords = torch.randn(count, 2, device=device)
        weights = torch.softmax(torch.randn(count, device=device), dim=0)
        rows.append({"Decoder": decoder, "Inputs": count, "Scope": scope,
                     **measure(coords, weights, args.repeats, args.warmup, device)})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    payload = {"device": str(device), "repeats": args.repeats,
               "scope": "aggregation-only; MeanShift convergence excluded", "rows": rows}
    (out / "decoder_aggregation_latency.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out / "decoder_aggregation_latency.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    md = ["# Decoder aggregation-only latency", "",
          "MeanShift convergence, feature extraction, search and scoring are excluded.", "",
          "| Decoder | Inputs | Mean ms | Median ms | P90 ms |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['Decoder']} | {r['Inputs']} | {r['latency_mean_ms']:.6f} | {r['latency_median_ms']:.6f} | {r['latency_p90_ms']:.6f} |")
    (out / "DECODER_LATENCY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
