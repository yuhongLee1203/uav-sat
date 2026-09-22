#!/usr/bin/env python3
"""Causal heading fusion used consistently by validation and paper tables.

The Bearing-UAV MHE/HSR metrics evaluate a predicted heading/direction against
heading ground truth.  Our recurrent model already predicts heading, while the
external route-state estimator also exposes a causal motion direction.  This
module fuses those two *predicted* quantities without reading GT at inference.

Both Full and ablations use the exact same fusion rule.  With Kalman disabled,
the state displacement comes from the unfiltered measurement-state path; with
Kalman enabled it comes from the filtered state path.  GT is used only after the
prediction is formed, to compute the evaluation error.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _circular_blend_deg(a_deg: float, b_deg: float, alpha: float) -> float:
    """Blend two directions on the unit circle; alpha weights b_deg."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    ar = math.radians(float(a_deg))
    br = math.radians(float(b_deg))
    x = (1.0 - alpha) * math.cos(ar) + alpha * math.cos(br)
    y = (1.0 - alpha) * math.sin(ar) + alpha * math.sin(br)
    if x * x + y * y <= 1e-12:
        return _wrap_deg(a_deg)
    return _wrap_deg(math.degrees(math.atan2(y, x)))


def fused_heading_errors(rows: Iterable[dict], alpha: float) -> np.ndarray:
    """Return per-frame absolute heading error after causal state-direction fusion.

    Required CSV fields for alpha>0:
      estimated_heading_deg, gt_heading_deg, kalman_x, kalman_y

    The first frame (and any zero-displacement frame) falls back to the recurrent
    heading prediction because no causal state displacement direction exists yet.
    alpha=0 exactly reproduces the existing heading_error_deg metric.
    """
    rows = list(rows)
    if not rows:
        return np.asarray([], dtype=np.float64)
    alpha = float(alpha)
    if alpha <= 1e-12:
        return np.asarray(
            [abs(float(r["heading_error_deg"])) for r in rows], dtype=np.float64
        )

    required = {"estimated_heading_deg", "gt_heading_deg", "kalman_x", "kalman_y"}
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(
            "heading fusion requires CSV fields %s; missing %s"
            % (sorted(required), sorted(missing))
        )

    out = []
    previous_xy = None
    previous_motion_heading = None
    for row in rows:
        learned_heading = float(row["estimated_heading_deg"])
        gt_heading = float(row["gt_heading_deg"])
        current_xy = np.asarray(
            [float(row["kalman_x"]), float(row["kalman_y"])], dtype=np.float64
        )

        motion_heading = None
        if previous_xy is not None:
            delta = current_xy - previous_xy
            if float(np.linalg.norm(delta)) > 1e-6:
                motion_heading = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
                previous_motion_heading = motion_heading
        if motion_heading is None:
            motion_heading = previous_motion_heading

        if motion_heading is None:
            predicted = learned_heading
        else:
            predicted = _circular_blend_deg(learned_heading, motion_heading, alpha)
        out.append(abs(_wrap_deg(predicted - gt_heading)))
        previous_xy = current_xy

    return np.asarray(out, dtype=np.float64)


def heading_metrics(rows: Iterable[dict], alpha: float) -> dict:
    errors = fused_heading_errors(rows, alpha)
    return {
        "HSR@15_pct": 100.0 * float(np.mean(errors <= 15.0)) if errors.size else 0.0,
        "MHE_deg": float(errors.mean()) if errors.size else float("inf"),
    }
