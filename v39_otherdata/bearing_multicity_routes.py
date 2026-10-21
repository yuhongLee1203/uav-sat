#!/usr/bin/env python3
"""Reproducible multi-city route specifications for Bearing-UAV-90K.

Testing uses the two navigation waypoint files published by the official
Bearing-UAV repository for each of the four cities.  This gives exactly two
held-out test routes per city without inventing evaluation trajectories.

Training routes remain separate pseudo-flight routes.  They use the smoother
long-leg templates already defined in bearing_prepare.py; the v39 runner only
uses train_01 for the canonical A-only temporal training protocol.

Turn convention
---------------
For reporting/auditing we use heading change modulo 360 degrees:
    delta = (heading_next - heading_previous) % 360
A meaningful turn must therefore lie in [20, 350] degrees.  Values <20 or >350
are treated as near-straight continuation points, not as turns.  This keeps the
user-requested 20..350-degree interpretation while preserving the official
waypoints exactly and keeping the experiment reproducible.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

import bearing_prepare as base

Point = Tuple[int, int]
RouteSpecs = Dict[str, List[Point]]

OFFICIAL_TEST_ROUTE_SOURCE: Mapping[str, Mapping[str, str]] = {
    "citya": {
        "test_01": "Bearing-UAV official wps34bc_50.json",
        "test_02": "Bearing-UAV official wps34bc_51.json",
    },
    "cityb": {
        "test_01": "Bearing-UAV official wps36bc_50.json",
        "test_02": "Bearing-UAV official wps36bc_51.json",
    },
    "cityc": {
        "test_01": "Bearing-UAV official wps37bc_50.json",
        "test_02": "Bearing-UAV official wps37bc_51.json",
    },
    "cityd": {
        "test_01": "Bearing-UAV official wps38bc_50.json",
        "test_02": "Bearing-UAV official wps38bc_51.json",
    },
}

# Coordinates copied from the official Bearing-UAV navigation waypoint files.
# They are in the canonical 4096x4096 RSI pixel coordinate system used by the
# local Bearing_UAV_90K dataset.
OFFICIAL_TEST_ROUTES: Mapping[str, Mapping[str, List[Point]]] = {
    "citya": {
        "test_01": [
            (1669, 3393), (2225, 3573), (2417, 3257), (2618, 3509),
            (2603, 3637), (2594, 3803), (2771, 3816), (2957, 3785),
            (3340, 3540), (3363, 3378), (3395, 3224), (3412, 3102),
            (3246, 3084),
        ],
        "test_02": [
            (520, 260), (780, 460), (980, 780), (660, 1080),
            (1120, 1460), (850, 1820), (1380, 2060), (1000, 2350),
            (1480, 2620), (1180, 2900),
        ],
    },
    "cityb": {
        "test_01": [
            (1669, 3393), (2225, 3573), (2417, 3257), (2618, 3509),
            (2603, 3637), (2594, 3803), (2771, 3816), (2957, 3785),
            (3340, 3540), (3363, 3378), (3395, 3224), (3412, 3102),
            (3246, 3084),
        ],
        "test_02": [
            (350, 3900), (600, 3600), (850, 3700), (1100, 3350),
            (1350, 3400), (1600, 3100), (1850, 3150), (2100, 2800),
            (2350, 2900), (2600, 2550), (2850, 2650),
        ],
    },
    "cityc": {
        "test_01": [
            (1669, 3393), (2225, 3573), (2417, 3257), (2618, 3509),
            (2603, 3637), (2594, 3803), (2771, 3816), (2957, 3785),
            (3340, 3540), (3363, 3378), (3395, 3224), (3412, 3102),
            (3246, 3084),
        ],
        "test_02": [
            (800, 1500), (1100, 1780), (1450, 2000), (1750, 1900),
            (2050, 2150), (2380, 2350), (2700, 2250), (3000, 2100),
            (3320, 2750), (3650, 2450), (3950, 3300),
        ],
    },
    "cityd": {
        "test_01": [
            (1669, 3393), (2225, 3573), (2417, 3257), (2618, 3509),
            (2603, 3637), (2594, 3803), (2771, 3816), (2957, 3785),
            (3340, 3540), (3363, 3378), (3395, 3224), (3412, 3102),
            (3246, 3084),
        ],
        "test_02": [
            (800, 1500), (1100, 1780), (1450, 2000), (1750, 1900),
            (2050, 2150), (2380, 2350), (2700, 2250), (3000, 2100),
            (3320, 2750), (3650, 2450), (3950, 3300),
        ],
    },
}


def route_specs_for_city(city: str) -> RouteSpecs:
    city = str(city).lower()
    if city not in OFFICIAL_TEST_ROUTES:
        raise ValueError(f"Unsupported Bearing city: {city}")

    # Keep training templates separate from official held-out test trajectories.
    specs: RouteSpecs = {
        name: [tuple(map(int, p)) for p in base.ROUTE_SPECS[name]]
        for name in base.TRAIN_ROUTES
    }
    for name in base.TEST_ROUTES:
        specs[name] = [tuple(map(int, p)) for p in OFFICIAL_TEST_ROUTES[city][name]]
    return specs


def _headings(points: Sequence[Point]) -> List[float]:
    values: List[float] = []
    for a, b in zip(points[:-1], points[1:]):
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        if abs(dx) + abs(dy) <= 1e-9:
            raise ValueError("Route contains duplicate consecutive waypoints")
        values.append((math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0)
    return values


def turn_deltas_mod360(points: Sequence[Point]) -> List[float]:
    headings = _headings(points)
    return [
        (float(b) - float(a)) % 360.0
        for a, b in zip(headings[:-1], headings[1:])
    ]


def meaningful_turns(points: Sequence[Point]) -> List[float]:
    return [d for d in turn_deltas_mod360(points) if 20.0 <= d <= 350.0]


def shortest_turn_magnitude(delta_mod360: float) -> float:
    d = float(delta_mod360) % 360.0
    return min(d, 360.0 - d)


def audit_test_turn_diversity(city: str, specs: Mapping[str, Sequence[Point]]) -> dict:
    """Reject the old near-90-degree hand-built zig-zag evaluation pattern.

    We do NOT demand that every individual route cover every angle bin; the two
    official routes in one city are evaluated together.  The city-level pair
    must contain multiple meaningful turns and cover at least four broad
    modulo-360 turn bins.  No more than 65% of meaningful turns may sit in the
    75..105-degree near-right-angle band (using shortest-turn magnitude).
    """
    city = str(city).lower()
    report = {"city": city, "routes": {}}
    combined: List[float] = []

    for route in base.TEST_ROUTES:
        points = specs[route]
        all_deltas = turn_deltas_mod360(points)
        major = meaningful_turns(points)
        if len(points) < 6:
            raise RuntimeError(f"{city}/{route}: too few waypoints ({len(points)})")
        if len(major) < 4:
            raise RuntimeError(
                f"{city}/{route}: only {len(major)} meaningful 20..350deg turns"
            )
        combined.extend(major)
        report["routes"][route] = {
            "source": OFFICIAL_TEST_ROUTE_SOURCE[city][route],
            "waypoints": len(points),
            "turn_delta_mod360_deg": [round(v, 1) for v in all_deltas],
            "meaningful_20_350_deg": [round(v, 1) for v in major],
        }

    # Six broad bins across the requested modulo-360 range.
    edges = (20.0, 60.0, 120.0, 180.0, 240.0, 300.0, 350.000001)
    occupied = set()
    for value in combined:
        for i in range(len(edges) - 1):
            if edges[i] <= value < edges[i + 1]:
                occupied.add(i)
                break

    near_90 = sum(
        75.0 <= shortest_turn_magnitude(v) <= 105.0
        for v in combined
    )
    near_90_fraction = near_90 / max(len(combined), 1)

    if len(occupied) < 4:
        raise RuntimeError(
            f"{city}: official test pair covers only {len(occupied)} broad turn bins"
        )
    if near_90_fraction > 0.65:
        raise RuntimeError(
            f"{city}: too many near-90deg turns ({100*near_90_fraction:.1f}%)"
        )

    report["combined"] = {
        "meaningful_turn_count": len(combined),
        "occupied_turn_bins": sorted(int(v) for v in occupied),
        "near_90_fraction": near_90_fraction,
        "rule": "meaningful delta in [20,350] deg; >=4 bins across two tests; <=65% near 90deg",
    }
    return report
