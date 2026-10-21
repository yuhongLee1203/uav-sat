"""Unified controlled protocol for the final v36_byTeacher experiments.

Formal protocol
---------------
* Stage-1 visual model is trained once on Route A from scratch.
* The formal local-search prior is a reproducible fixed-radius perturbation:
  exactly 8.0 m by default, with direction changing deterministically by frame.
* All main architecture comparisons use the same full centered 6x6 candidate
  lattice, SoftMS decoder, bandwidth=8 m, score temperature tau=0.30.
* Route A trains temporal G. Routes B/C are evaluation only.
* Before a formal run, candidate capture is measured from actual gallery
  geometry. A run that requires the formal 6x6 prior is rejected if capture is
  below the configured minimum; this prevents reporting an experiment where
  the correct local support is systematically outside the search window.
* There is no jitter=0 oracle in formal tables.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import config
import robust_tracker_base as rb
from visual_localizer import regular_grid_indices, soft_mean_shift

MAIN_JITTER_M = float(os.environ.get("UAVSAT_MAIN_JITTER_M", "8.0"))
MAIN_GRID_SIZE = int(os.environ.get("UAVSAT_MAIN_GRID_SIZE", "6"))
MAIN_BANDWIDTH_M = float(os.environ.get("UAVSAT_MAIN_BANDWIDTH_M", "8.0"))
MAIN_TAU = float(os.environ.get("UAVSAT_MAIN_TAU", "0.30"))
MAIN_SEED = int(os.environ.get("UAVSAT_MAIN_SEED", "2026"))
MAIN_CAPTURE_MIN_RATE = float(os.environ.get("UAVSAT_CAPTURE_MIN_RATE", "0.95"))
MAIN_DECODER = "softms"
ROUTE_TO_INDEX = {"route_A": 0, "route_B": 1, "route_C": 2}


def route_index(route_name: str) -> int:
    if route_name not in ROUTE_TO_INDEX:
        raise ValueError("unknown route: %s" % route_name)
    return ROUTE_TO_INDEX[route_name]


def fixed_radial_jitter(index: int, route_name: str, magnitude_m: float = MAIN_JITTER_M, seed: int = MAIN_SEED) -> np.ndarray:
    magnitude = float(magnitude_m)
    if magnitude <= 0.0:
        return np.zeros(2, dtype=np.float64)
    ridx = route_index(route_name)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    phase = (
        float(index) * golden_angle
        + float(ridx) * 0.8115781021773633
        + float(int(seed) % 100000) * 0.00017320508075688773
    ) % (2.0 * math.pi)
    return magnitude * np.asarray([math.cos(phase), math.sin(phase)], dtype=np.float64)


def fixed_radial_jitter_tensor(length: int, route_name: str, magnitude_m: float = MAIN_JITTER_M, seed: int = MAIN_SEED, device: Optional[torch.device] = None) -> torch.Tensor:
    rows = [fixed_radial_jitter(i, route_name, magnitude_m=magnitude_m, seed=seed) for i in range(int(length))]
    return torch.as_tensor(
        np.asarray(rows, dtype=np.float32),
        dtype=torch.float32,
        device=device if device is not None else torch.device("cpu"),
    )


def search_center(reference_xy, index: int, route_name: str, magnitude_m: float = MAIN_JITTER_M, seed: int = MAIN_SEED) -> np.ndarray:
    reference = np.asarray(reference_xy, dtype=np.float64).reshape(2)
    return reference + fixed_radial_jitter(index, route_name, magnitude_m, seed)


def xy_tensor(xy, device):
    return torch.as_tensor(xy, dtype=torch.float32, device=device).reshape(1, 2)


class StandardXYKalman:
    """Standard constant-velocity XY Kalman filter shared by all eight methods."""

    def __init__(self, initial_xy):
        p = np.asarray(initial_xy, dtype=np.float64).reshape(2)
        self.x = np.asarray([p[0], p[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([
            float(config.KALMAN_INIT_POSITION_VAR),
            float(config.KALMAN_INIT_POSITION_VAR),
            float(config.KALMAN_INIT_VELOCITY_VAR),
            float(config.KALMAN_INIT_VELOCITY_VAR),
        ])
        self.F = np.asarray([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.H = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self.Q = np.diag([
            float(config.KALMAN_Q_POSITION),
            float(config.KALMAN_Q_POSITION),
            float(config.KALMAN_Q_VELOCITY),
            float(config.KALMAN_Q_VELOCITY),
        ])

    def step(self, measurement_xy, variance_xy):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        z = np.asarray(measurement_xy, dtype=np.float64).reshape(2)
        var = np.asarray(variance_xy, dtype=np.float64).reshape(2)
        base_r = float(config.KALMAN_R_POSITION)
        R = np.diag(np.maximum(var, base_r))
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ innovation
        eye = np.eye(4, dtype=np.float64)
        ikh = eye - K @ self.H
        self.P = ikh @ self.P @ ikh.T + K @ R @ K.T
        return self.x[:2].copy()


def make_state(initial_xy, initial_velocity_xy=None):
    kf = StandardXYKalman(initial_xy)
    if initial_velocity_xy is not None:
        kf.x[2:] = np.asarray(initial_velocity_xy, dtype=np.float64).reshape(2)
    return {"kalman": kf, "hidden": None, "previous_z": None}


def apply_k(state, xy, variance, device):
    filtered = state["kalman"].step(
        xy[0].detach().cpu().numpy(),
        variance[0].detach().cpu().numpy(),
    )
    return xy_tensor(filtered, device)


def apply_g(model, xy, variance, z_uav, state):
    out = model.forward_step(
        stage_xy=xy,
        variance_xy=variance,
        z_uav=z_uav,
        previous_z_uav=state["previous_z"],
        hidden=state["hidden"],
    )
    state["hidden"] = out.hidden
    return out.corrected_xy, out


@torch.no_grad()
def raw_candidates(visual, uav_clip, center_xy, grid_size=MAIN_GRID_SIZE):
    grid_size = int(grid_size)
    indices = regular_grid_indices(
        visual.gallery["xy"],
        visual.gallery["pixel"],
        visual.pixel_index,
        xy_tensor(center_xy, visual.device),
        grid_size,
        int(config.SAT_STRIDE),
        visual.device,
    )
    centers = visual.gallery["xy"][indices]
    satellite_clip = visual.gallery["clip_feat"][indices]
    z_uav = visual.model.encode_uav_from_clip(uav_clip)
    z_sat = visual.model.encode_sat_from_clip(
        satellite_clip.reshape(-1, satellite_clip.shape[-1]),
        centers.reshape(-1, 2),
    ).reshape(centers.shape[0], centers.shape[1], -1)
    logits = visual.model.logit_scale.exp().clamp(max=100.0) * (
        z_uav[:, None] * z_sat
    ).sum(dim=2)
    return {
        "indices": indices,
        "centers": centers,
        "z_uav": z_uav,
        "z_sat": z_sat,
        "logits": logits,
        "candidate_count": int(centers.shape[1]),
    }


@torch.no_grad()
def decode_visual(visual, uav_clip, center_xy, grid_size=MAIN_GRID_SIZE, decoder=MAIN_DECODER, bandwidth_m=MAIN_BANDWIDTH_M, tau=MAIN_TAU):
    raw = raw_candidates(visual, uav_clip, center_xy, grid_size=grid_size)
    centers = raw["centers"]
    logits = raw["logits"]
    probability = torch.softmax(logits / max(float(tau), 1e-6), dim=1)

    modes = density = mode_weights = None
    if decoder == "softms":
        xy, support, modes, density, mode_weights, _ = soft_mean_shift(
            logits,
            centers,
            float(tau),
            float(bandwidth_m),
            int(config.MEANSHIFT_ITERATIONS),
            float(config.MEANSHIFT_MODE_BETA),
        )
        mode_count = (mode_weights > 0).sum(dim=1)
    elif decoder == "weighted":
        xy = (probability[:, :, None] * centers).sum(dim=1)
        support = probability.max(dim=1).values
        mode_count = torch.zeros(centers.shape[0], dtype=torch.long, device=centers.device)
    else:
        raise ValueError("formal decoder must be softms or weighted; got %s" % decoder)

    diff = centers - xy[:, None, :]
    variance = (probability[:, :, None] * diff.square()).sum(dim=1).clamp_min(1e-3)
    raw.update({
        "xy": xy,
        "variance": variance,
        "support": support,
        "probability": probability,
        "modes": modes,
        "density": density,
        "mode_weights": mode_weights,
        "mode_count": mode_count,
    })
    return raw


@torch.no_grad()
def capture_report(visual, cache, route_name: str, jitter_m: float = MAIN_JITTER_M, grid_size: int = MAIN_GRID_SIZE, seed: int = MAIN_SEED):
    gt = cache.gt_xy.to(visual.device, dtype=torch.float32)
    jitter = fixed_radial_jitter_tensor(
        len(cache), route_name, magnitude_m=float(jitter_m), seed=int(seed), device=visual.device
    )
    prior = gt + jitter
    indices = regular_grid_indices(
        visual.gallery["xy"],
        visual.gallery["pixel"],
        visual.pixel_index,
        prior,
        int(grid_size),
        int(config.SAT_STRIDE),
        visual.device,
    )
    centers = visual.gallery["xy"][indices]
    nearest = torch.linalg.norm(centers - gt[:, None, :], dim=2).min(dim=1).values
    captured = nearest <= float(config.CANDIDATE_CAPTURE_RADIUS_M)
    return {
        "Route": route_name,
        "Jitter_m": float(jitter_m),
        "GridSize": int(grid_size),
        "CandidateCount": int(grid_size) * int(grid_size),
        "CaptureRadius_m": float(config.CANDIDATE_CAPTURE_RADIUS_M),
        "CandidateCaptureRate_pct": float(captured.float().mean().item() * 100.0),
        "NearestCandidateMean_m": float(nearest.mean().item()),
        "NearestCandidateP95_m": float(torch.quantile(nearest, 0.95).item()),
        "NearestCandidateMax_m": float(nearest.max().item()),
    }


def assert_capture(report, minimum_rate=MAIN_CAPTURE_MIN_RATE):
    rate = float(report["CandidateCaptureRate_pct"]) / 100.0
    if rate + 1e-12 < float(minimum_rate):
        raise RuntimeError(
            "candidate capture is too low for a formal run: route=%s jitter=%.3fm grid=%dx%d capture=%.2f%% < %.2f%%"
            % (
                report["Route"],
                report["Jitter_m"],
                report["GridSize"],
                report["GridSize"],
                report["CandidateCaptureRate_pct"],
                100.0 * float(minimum_rate),
            )
        )


def build_cache(route_name, visual, device):
    idx = config.ROUTE_NAMES.index(route_name)
    return rb.build_route_cache(route_name, config.ROUTE_ROOTS[idx], visual, device)


def protocol_metadata():
    return {
        "protocol_name": "unified_fixed_radial_8m_v1",
        "main_jitter_m": float(MAIN_JITTER_M),
        "jitter_definition": "fixed radial magnitude; deterministic changing direction",
        "main_grid_size": int(MAIN_GRID_SIZE),
        "main_candidate_count": int(MAIN_GRID_SIZE) ** 2,
        "decoder": MAIN_DECODER,
        "meanshift_bandwidth_m": float(MAIN_BANDWIDTH_M),
        "score_temperature_tau": float(MAIN_TAU),
        "seed": int(MAIN_SEED),
        "minimum_candidate_capture_rate_pct": float(MAIN_CAPTURE_MIN_RATE) * 100.0,
        "formal_oracle_zero_m": False,
        "train_route": "route_A",
        "test_routes": ["route_B", "route_C"],
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
