#!/usr/bin/env python3
"""Causal heading fusion used consistently by validation and paper tables.

Bearing-UAV HSR/MHE compare a predicted direction with direction ground truth.
The recurrent head is the primary direction predictor.  The causal estimator
motion direction is used only as a consistency correction; when both predicted
quantities strongly disagree, the fusion automatically falls back toward the
recurrent prediction.  No GT enters the prediction rule.
"""
from __future__ import annotations

import math
import os
from typing import Iterable

import numpy as np


def _wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _circular_blend_deg(a_deg: float, b_deg: float, alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    ar = math.radians(float(a_deg))
    br = math.radians(float(b_deg))
    x = (1.0 - alpha) * math.cos(ar) + alpha * math.cos(br)
    y = (1.0 - alpha) * math.sin(ar) + alpha * math.sin(br)
    if x * x + y * y <= 1e-12:
        return _wrap_deg(a_deg)
    return _wrap_deg(math.degrees(math.atan2(y, x)))


def _agreement_alpha(learned_deg: float, motion_deg: float, alpha: float) -> float:
    """Smoothly suppress a motion correction when two causal predictors disagree."""
    limit = float(os.environ.get("BEARING_HEADING_FUSION_DISAGREEMENT_DEG", "60.0"))
    power = float(os.environ.get("BEARING_HEADING_FUSION_AGREEMENT_POWER", "2.0"))
    limit = max(limit, 1e-3)
    disagreement = abs(_wrap_deg(float(motion_deg) - float(learned_deg)))
    agreement = float(np.clip(1.0 - disagreement / limit, 0.0, 1.0))
    return float(np.clip(alpha, 0.0, 1.0)) * (agreement ** max(power, 0.0))


def fused_heading_errors(rows: Iterable[dict], alpha: float) -> np.ndarray:
    """Per-frame absolute heading error after causal agreement-gated fusion.

    Required CSV fields for alpha>0:
      estimated_heading_deg, gt_heading_deg, kalman_x, kalman_y

    Full and every ablation use exactly the same rule.  The first/zero-motion
    frame falls back to the recurrent heading.  GT is read only after the final
    heading prediction has been formed, solely to compute the metric.
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
            effective_alpha = _agreement_alpha(
                learned_heading, motion_heading, alpha
            )
            predicted = _circular_blend_deg(
                learned_heading, motion_heading, effective_alpha
            )
        out.append(abs(_wrap_deg(predicted - gt_heading)))
        previous_xy = current_xy

    return np.asarray(out, dtype=np.float64)


def heading_metrics(rows: Iterable[dict], alpha: float) -> dict:
    errors = fused_heading_errors(rows, alpha)
    return {
        "HSR@15_pct": 100.0 * float(np.mean(errors <= 15.0)) if errors.size else 0.0,
        "MHE_deg": float(errors.mean()) if errors.size else float("inf"),
    }
