#!/usr/bin/env python3
"""Train-route-derived limits for the v40 audit of v39 GT reference runs."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


def build_profile(prepared_root: Path) -> dict:
    """Read train_01 only. Held-out test geometry never sets model limits."""
    path = Path(prepared_root) / 'routes' / 'train_01' / 'manifest.csv'
    with path.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 3:
        raise ValueError(f'Need at least three training frames: {path}')
    xy = np.array([[float(r['x_m']), float(r['y_m'])] for r in rows])
    if not np.isfinite(xy).all():
        raise ValueError(f'Nonfinite training coordinates: {path}')
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    maximum = float(steps.max())
    if maximum <= 0:
        raise ValueError(f'Zero-motion training route: {path}')
    # A single signed axis cannot move farther than the observed XY norm.
    # Twice the observed maximum leaves space for noisy visual corrections.
    bound = 2.0 * maximum
    yaw = np.radians(np.array([float(r['yaw_deg']) for r in rows]))
    yaw_delta = np.abs(np.arctan2(np.sin(np.diff(yaw)), np.cos(np.diff(yaw))))
    source_ids = [int(r['source_index']) for r in rows]
    profile = {
        'source': 'train_01_manifest_only',
        'train_frames': len(rows),
        'train_step_max_m': maximum,
        'train_step_p95_m': float(np.quantile(steps, 0.95)),
        'train_yaw_delta_p95_deg': math.degrees(float(np.quantile(yaw_delta, 0.95))),
        'nonconsecutive_source_pairs_pct': 100.0 * sum(abs(b-a) != 1 for a,b in zip(source_ids, source_ids[1:])) / len(steps),
        'max_forward_speed_m_per_frame': bound,
        'max_polynomial_step_m_per_frame': bound,
        'max_measurement_correction_parallel_m': bound,
        'kalman_max_measurement_innovation_progress_m': bound,
        'kalman_max_posterior_correction_progress_m': bound,
        'kalman_max_velocity_correction_m_per_frame': bound,
        'kalman_final_step_max_m': bound,
        'max_cross_speed_m_per_frame': bound,
        'max_forward_accel_m_per_frame2': bound,
        'max_cross_accel_m_per_frame2': bound,
        'max_measurement_correction_cross_m': bound,
        'kalman_max_measurement_innovation_cross_m': bound,
        'kalman_max_posterior_correction_cross_m': bound,
        'max_motion_velocity_delta_m_per_frame': bound,
        'max_polynomial_step_delta_m_per_frame': bound,
        'max_final_cross_track_m': 4.0 * maximum,
        'route_frame_smooth_radius_m': 0.0,
        'controlled_final_progress_cap_to_gt': False,
    }
    return profile
