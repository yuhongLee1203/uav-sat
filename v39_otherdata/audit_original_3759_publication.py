#!/usr/bin/env python3
"""Paired-route audit of the original GT-assisted 3.759 m inference replay."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

CITIES = ("citya", "cityb", "cityc", "cityd")
ROUTES = ("test_01", "test_02")
VARIANTS = ("full", "no_gru", "no_kalman", "no_ms", "frames1", "frames2",
            "grid4", "grid5", "grid7", "grid8")


def load_rows(root):
    data = {}
    for city in CITIES:
        for route in ROUTES:
            key = f"{city}/{route}"
            data[key] = {}
            for variant in VARIANTS:
                folder = root / city / "variants" / variant
                summary = json.loads((folder / "bearing_v39_summary.json").read_text())
                path = folder / Path(summary[route]["CSV"]).name
                with path.open(newline="") as handle:
                    data[key][variant] = list(csv.DictReader(handle))
            if len({len(rows) for rows in data[key].values()}) != 1:
                raise ValueError(f"Unpaired variant rows: {key}")
    if sum(len(d["full"]) for d in data.values()) != 1482:
        raise ValueError("Expected 1,482 original test frames")
    return data


def errors(rows):
    return np.asarray([float(r["error_final_m"]) for r in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", required=True, type=Path)
    root = parser.parse_args().suite_root.resolve()
    gate = json.loads((root / "original_3759_all_metrics.json").read_text())
    if gate["historical_paper_value_check"]["status"] != "PASS":
        raise ValueError("Historical reproduction gate did not pass")
    data = load_rows(root)
    keys = list(data)
    rng = np.random.default_rng(2033)
    samples = rng.integers(0, 8, size=(100000, 8))
    comparisons = {}
    for variant in VARIANTS[1:]:
        counts = np.asarray([len(data[key]["full"]) for key in keys])
        route_means = np.asarray([
            np.mean(errors(data[key]["full"]) - errors(data[key][variant]))
            for key in keys
        ])
        bootstrap = (route_means[samples] * counts[samples]).sum(1) / counts[samples].sum(1)
        comparisons[variant] = {
            "full_minus_variant_MLE_m": float(np.dot(route_means, counts) / counts.sum()),
            "route_bootstrap_95CI_m": [float(x) for x in np.quantile(bootstrap, [.025, .975])],
            "routes_full_better": int(np.count_nonzero(route_means < 0)),
        }
    full_rows = [row for key in keys for row in data[key]["full"]]
    prior = np.asarray([
        math.hypot(float(r["prior_center_x"])-float(r["gt_x"]),
                   float(r["prior_center_y"])-float(r["gt_y"]))
        for r in full_rows
    ])
    gt_count = sum(r["controlled_gt_prior"] == "1" for r in full_rows)
    teacher_count = sum(r["acquisition_selected_index"] == r["acquisition_target_index"]
                        for r in full_rows)
    result = {
        "frames": len(full_rows), "routes": len(keys), "gt_prior_frames": gt_count,
        "historical_exact_printed_match": gate["historical_paper_value_check"]["exact_printed_match"],
        "historical_exact_mismatches": gate["historical_paper_value_check"]["exact_mismatches"],
        "teacher_target_matches": teacher_count,
        "prior_MLE_m": float(prior.mean()),
        "prior_LSR_at_5_pct": float(100*np.mean(prior <= 5)),
        "bootstrap_seed": 2033, "bootstrap_replicates": 100000,
        "bootstrap_unit": "whole route, frame-count-weighted pooled MLE",
        "comparisons": comparisons,
    }
    (root / "publication_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Publication audit: original 3.759 m replay", "",
        "**Verdict:** Report these results only as a GT-assisted localization ablation with "
        "the exact test-time reference protocol disclosed. They do not establish no-GT "
        "localization or closed-loop navigation performance.", "",
        "Historical-value proximity check: PASS; exact printed-value match: "
        + ("YES" if gate["historical_paper_value_check"]["exact_printed_match"] else "NO")
        + ". Test set: 1,482 frames, eight synthetic routes, City A/B/C/D.",
        *[f"Exact mismatch: {x}" for x in gate["historical_paper_value_check"]["exact_mismatches"]], "",
        "## Test-time GT use", "",
        f"Current-frame GT centers the local search prior on {gt_count:,}/1,482 frames. "
        f"The selected acquisition hypothesis matches the GT target hypothesis on "
        f"{teacher_count:,}/1,482 frames. The GT-centered prior alone has "
        f"{prior.mean():.3f} m MLE and {100*np.mean(prior<=5):.2f}% LSR@5.",
        "The inference path also passes current GT progress and step bounds to the motion "
        "estimator. A per-frame final-cap event flag of zero does not mean GT was absent.", "",
        "## Paired MLE differences", "",
        "Full minus comparison, in metres; negative favors Full. The 95% interval "
        "resamples the eight whole routes, preserving route frame counts.", "",
        "| Comparison | Difference | Route bootstrap 95% interval | Routes Full better |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS[1:]:
        c = comparisons[variant]
        lo, hi = c["route_bootstrap_95CI_m"]
        lines.append(f"| {variant} | {c['full_minus_variant_MLE_m']:+.4f} | "
                     f"[{lo:+.4f}, {hi:+.4f}] | {c['routes_full_better']}/8 |")
    lines += [
        "", "## Paper-facing conclusions", "",
        "- Three frames has lower MLE than one and two frames on every route.",
        "- Full and w/o Kalman have comparable MLE; the paired interval crosses zero.",
        "- The original 6x6 Full window is slower and less accurate than 7x7 and 8x8 "
        "in this test. Both larger windows have lower MLE on all eight routes.",
        "- LSR@15 saturates at 100% for several rows. Official heading, Recall@1, "
        "SR@20, SPL and NE are unavailable in this evaluation.", "",
        "A changed Full configuration chosen after viewing these test results is exploratory. "
        "A new best-model claim needs selection on training/validation data and evaluation "
        "on a fresh unseen test split.", "",
    ]
    path = root / "publication_audit.md"
    path.write_text("\n".join(lines))
    print(path)


if __name__ == "__main__":
    main()
