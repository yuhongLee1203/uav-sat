
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config
from data import RouteDataset, crop_satellite
from robust_tracker import contiguous_splits, deterministic_jitter
from visual_localizer import FrozenVisualLocalizer
from visual_model import TemporalLatticeCRF


def npy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


def softmax_np(value, axis=-1):
    value = np.asarray(value, dtype=np.float64)
    value = value - value.max(axis=axis, keepdims=True)
    value = np.exp(value)
    return value / np.clip(value.sum(axis=axis, keepdims=True), 1e-12, None)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def find_sample_index(dataset: RouteDataset, frame_id: int) -> int:
    for index, sample in enumerate(dataset.samples):
        if int(sample["frame_id"]) == int(frame_id):
            return index
    raise ValueError("frame_id={} not found".format(frame_id))


def csv_auto_frame(route_name: str, mode: str) -> Optional[int]:
    path = config.OUTPUT_DIR / "{}_robust_frames.csv".format(route_name)
    if not path.exists():
        return None
    rows = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            try:
                gt = np.array([float(row["gt_x"]), float(row["gt_y"])])
                hard = np.array([float(row["hardms_x"]), float(row["hardms_y"])])
                final = np.array([float(row["temporal_x"]), float(row["temporal_y"])])
                hard_error = float(np.linalg.norm(hard - gt))
                final_error = float(np.linalg.norm(final - gt))
                rows.append({
                    "frame_id": int(float(row["frame_id"])),
                    "hard_error": hard_error,
                    "final_error": final_error,
                    "improvement": hard_error - final_error,
                })
            except (KeyError, TypeError, ValueError):
                pass
    if not rows:
        return None
    key = {
        "best_improvement": "improvement",
        "largest_hardms_error": "hard_error",
        "largest_final_error": "final_error",
    }[mode]
    return max(rows, key=lambda item: item[key])["frame_id"]


def select_end_index(dataset, route_name, frame_id, end_index, split_name, auto_frame):
    window = int(config.TEMPORAL_WINDOW)
    if frame_id is not None:
        result = find_sample_index(dataset, frame_id)
    elif end_index is not None:
        result = int(end_index)
    else:
        chosen = csv_auto_frame(route_name, auto_frame)
        if chosen is not None:
            result = find_sample_index(dataset, chosen)
        else:
            segment = contiguous_splits(len(dataset))[split_name]
            result = max(segment.start + window - 1, (segment.start + segment.end) // 2)
    if result < window - 1 or result >= len(dataset):
        raise ValueError("invalid end index {} for length {}".format(result, len(dataset)))
    return result


def grid_layout(pixels: np.ndarray, grid_size: int):
    """Map candidate indices back to their real 6x6 pixel layout."""
    pixels = np.asarray(pixels)
    xs = np.round(pixels[:, 0]).astype(int)
    ys = np.round(pixels[:, 1]).astype(int)
    unique_x = sorted(set(xs.tolist()))
    unique_y = sorted(set(ys.tolist()))
    regular = len(unique_x) == grid_size and len(unique_y) == grid_size
    grid = np.full((grid_size, grid_size), -1, dtype=int)
    if regular:
        xmap = {v: i for i, v in enumerate(unique_x)}
        ymap = {v: i for i, v in enumerate(unique_y)}
        for index, (x, y) in enumerate(zip(xs, ys)):
            row, col = ymap[int(y)], xmap[int(x)]
            if grid[row, col] != -1:
                regular = False
                break
            grid[row, col] = index
        if (grid < 0).any():
            regular = False
    if not regular:
        grid = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    return grid, regular


def to_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.asarray(values)[grid]


def local_crop(sat_image: Image.Image, pixels: np.ndarray, margin=200):
    pixels = np.asarray(pixels)
    left = max(0, int(np.floor(pixels[:, 0].min())) - margin)
    right = min(sat_image.width, int(np.ceil(pixels[:, 0].max())) + margin)
    top = max(0, int(np.floor(pixels[:, 1].min())) - margin)
    bottom = min(sat_image.height, int(np.ceil(pixels[:, 1].max())) + margin)
    return sat_image.crop((left, top, right, bottom)), (left, top, right, bottom)


def shift_pixels(pixels: np.ndarray, box):
    result = np.asarray(pixels, dtype=np.float64).copy()
    result[:, 0] -= box[0]
    result[:, 1] -= box[1]
    return result



def meter_xy_to_pixel_exact(xy, origin_lat, origin_lon, mapper):
    """Invert data.meters_from_latlon and use the repository map transform."""
    points = np.asarray(xy, dtype=np.float64)
    one = points.ndim == 1
    if one:
        points = points[None]
    earth_radius_m = 6378137.0
    origin_lat_rad = math.radians(float(origin_lat))
    result = []
    for x_meter, y_meter in points:
        lat = float(origin_lat) + math.degrees(float(y_meter) / earth_radius_m)
        lon = float(origin_lon) + math.degrees(
            float(x_meter) / (earth_radius_m * math.cos(origin_lat_rad))
        )
        result.append(mapper.latlon_to_pixel(lat, lon))
    result = np.asarray(result, dtype=np.float64)
    return result[0] if one else result

def local_xy_to_pixel(candidate_xy, candidate_pixel, query_xy):
    design = np.concatenate([candidate_xy, np.ones((len(candidate_xy), 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(design, candidate_pixel, rcond=None)
    query = np.asarray(query_xy)
    if query.ndim == 1:
        query = query[None]
    return np.concatenate([query, np.ones((len(query), 1))], axis=1) @ coef


def model_intermediates(model, z_uav, z_sat, raw_logits, raw_prob, centers, frame_ids, hardms):
    emission, token = model._emissions(z_uav, z_sat, raw_logits, raw_prob, centers)
    dt = (frame_ids[:, 1:] - frame_ids[:, :-1]).float().clamp_min(1.0)
    first = model._first_transition_score(centers[:, 0], centers[:, 1], dt[:, 0])
    alpha = emission[:, 0, :, None] + emission[:, 1, None, :] + first
    alpha_history = [alpha]
    second = []
    for time in range(2, centers.shape[1]):
        transition = model._second_transition_score(
            centers[:, time - 2], centers[:, time - 1], centers[:, time],
            dt[:, time - 2], dt[:, time - 1]
        )
        second.append(transition)
        score = alpha[:, :, :, None] + transition + emission[:, time, None, None, :]
        alpha = torch.logsumexp(score, dim=1)
        alpha_history.append(alpha)
    official = model(z_uav, z_sat, raw_logits, raw_prob, centers, frame_ids, hardms, None)
    return {
        "official": official,
        "token": token,
        "emission": emission,
        "dt": dt,
        "first": first,
        "second": second,
        "alpha_history": alpha_history,
    }


def viterbi_path(emission: torch.Tensor, first: torch.Tensor, second: Sequence[torch.Tensor]):
    """Representative MAP path for visualization; not the official sum-product output."""
    time_count, candidate_count = emission.shape
    delta = emission[0, :, None] + emission[1, None, :] + first
    pointers = []
    for time in range(2, time_count):
        score = delta[:, :, None] + second[time - 2] + emission[time, None, None, :]
        delta, pointer = score.max(dim=0)
        pointers.append(pointer)
    best = int(delta.reshape(-1).argmax().item())
    path = [-1] * time_count
    path[-2] = best // candidate_count
    path[-1] = best % candidate_count
    best_score = float(delta[path[-2], path[-1]].item())
    for time in range(time_count - 1, 1, -1):
        path[time - 2] = int(pointers[time - 2][path[time - 1], path[time]].item())
    return path, best_score


def forward_probabilities(emission: np.ndarray, alpha_history: Sequence[np.ndarray]):
    result = [softmax_np(emission[0])]
    result.append(softmax_np(np.logaddexp.reduce(alpha_history[0], axis=0)))
    for alpha in alpha_history[1:]:
        result.append(softmax_np(np.logaddexp.reduce(alpha, axis=0)))
    return result


def heatmap(axis, values, grid, title, selected=None, decimals=2):
    matrix = to_grid(values, grid)
    image = axis.imshow(matrix, interpolation="nearest")
    axis.set_title(title)
    axis.set_xticks(range(matrix.shape[1]))
    axis.set_yticks(range(matrix.shape[0]))
    selected = set([] if selected is None else [int(v) for v in selected])
    low, high = float(matrix.min()), float(matrix.max())
    scale = max(high - low, 1e-8)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            index = int(grid[row, col])
            star = "*" if index in selected else ""
            norm = (matrix[row, col] - low) / scale
            axis.text(col, row, "{}{}\n{:.{}f}".format(index, star, matrix[row, col], decimals),
                      ha="center", va="center", fontsize=5.5,
                      color="white" if norm > 0.55 else "black")
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def plot_inputs(image_paths, frame_ids, out, dpi):
    fig, axes = plt.subplots(1, len(image_paths), figsize=(18, 4))
    for time, (path, frame_id) in enumerate(zip(image_paths, frame_ids)):
        axes[time].imshow(Image.open(path).convert("RGB"))
        axes[time].set_title("t{} / frame {}".format(time - len(image_paths) + 1, frame_id))
        axes[time].axis("off")
    fig.suptitle("Stage 1 — The actual five UAV frames sent to RTL-CRF")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_lattices(sat_image, candidate_pixels, gt_pixels, prior_pixels, frame_ids, out, dpi):
    fig, axes = plt.subplots(1, len(frame_ids), figsize=(21, 5))
    for time, axis in enumerate(axes):
        crop, box = local_crop(sat_image, candidate_pixels[time], margin=200)
        cp = shift_pixels(candidate_pixels[time], box)
        gp = shift_pixels(gt_pixels[time:time + 1], box)[0]
        pp = shift_pixels(prior_pixels[time:time + 1], box)[0]
        axis.imshow(crop)
        axis.scatter(cp[:, 0], cp[:, 1], s=25, facecolors="none", edgecolors="white")
        for index, point in enumerate(cp):
            axis.text(point[0], point[1], str(index), fontsize=5, ha="center", va="center")
        axis.scatter([gp[0]], [gp[1]], marker="*", s=140, label="GT")
        axis.scatter([pp[0]], [pp[1]], marker="x", s=80, label="GT+jitter prior")
        axis.set_title("frame {}\n36 nodes = 36 real 2D locations".format(frame_ids[time]))
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Stage 2 — The temporal state at each time is a real 2D 6x6 candidate lattice")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_patch_mapping(sat_image, candidate_pixels, grid, out, dpi):
    size = grid.shape[0]
    fig = plt.figure(figsize=(17, 10))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0])
    patches = outer[0].subgridspec(size, size, wspace=0.02, hspace=0.02)
    for row in range(size):
        for col in range(size):
            axis = fig.add_subplot(patches[row, col])
            index = int(grid[row, col])
            pixel = candidate_pixels[index]
            patch = crop_satellite(sat_image, float(pixel[0]), float(pixel[1]), int(config.SAT_CROP_SIZE))
            axis.imshow(patch)
            axis.text(0.03, 0.09, str(index), transform=axis.transAxes, fontsize=9,
                      bbox={"facecolor": "white", "alpha": 0.75, "pad": 1})
            axis.axis("off")
    axis = fig.add_subplot(outer[1])
    axis.axis("off")
    lines = [
        "2D grid -> flattened candidate axis", "",
        "The drawing of 36 nodes in one vertical column", "does NOT remove 2D spatial information.", ""
    ]
    for row in range(size):
        lines.append("  ".join("({},{}) -> {:02d}".format(row, col, int(grid[row, col])) for col in range(size)))
    lines += [
        "", "Every node stores:",
        "  • a real satellite patch", "  • map coordinate (x, y)",
        "  • gallery pixel coordinate", "  • embedding and similarity score", "",
        "Transitions use coordinate differences:", "  Δx = x_next - x_previous", "  Δy = y_next - y_previous", "",
        "They never use index difference such as 21 - 20."
    ]
    axis.text(0.01, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)
    fig.suptitle("Stage 3 — Exact correspondence between the 2D current-frame patch grid and nodes 0..35")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_five(values, grids, frame_ids, path, out, title, dpi):
    fig, axes = plt.subplots(1, len(frame_ids), figsize=(22, 5))
    for time, axis in enumerate(axes):
        heatmap(axis, values[time], grids[time], "frame {}".format(frame_ids[time]), [path[time]])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_first(first, centers0, centers1, grid1, prev_index, curr_index, out, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    image = axes[0].imshow(first, interpolation="nearest", aspect="auto")
    axes[0].scatter([curr_index], [prev_index], marker="*", s=140)
    axes[0].set_title("All 36×36 first-order scores")
    axes[0].set_xlabel("current candidate index")
    axes[0].set_ylabel("previous candidate index")
    plt.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    heatmap(axes[1], first[prev_index], grid1,
            "Current candidates given previous node {}".format(prev_index), [curr_index])
    p0, p1 = centers0[prev_index], centers1[curr_index]
    axes[2].scatter(centers0[:, 0], centers0[:, 1], s=15, alpha=0.3, label="previous lattice")
    axes[2].scatter(centers1[:, 0], centers1[:, 1], s=15, alpha=0.3, label="current lattice")
    axes[2].annotate("", xy=p1, xytext=p0, arrowprops={"arrowstyle": "->", "linewidth": 2})
    axes[2].scatter([p0[0]], [p0[1]], s=100, marker="o")
    axes[2].scatter([p1[0]], [p1[1]], s=140, marker="*")
    delta = p1 - p0
    axes[2].set_title("Real 2D displacement\nΔx={:.2f} m, Δy={:.2f} m, distance={:.2f} m".format(
        delta[0], delta[1], np.linalg.norm(delta)))
    axes[2].set_aspect("equal", adjustable="datalim")
    axes[2].legend(fontsize=8)
    fig.suptitle("Stage 6 — First-order score: the model evaluates a real 2D movement between two frames")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_second(second, centers0, centers1, centers2, grid2, i0, i1, i2, out, dpi):
    conditional = second[i0, i1]
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
    heatmap(axes[0], conditional, grid2,
            "All current nodes given previous pair {} -> {}".format(i0, i1), [i2])
    p0, p1, p2 = centers0[i0], centers1[i1], centers2[i2]
    v01, v12 = p1 - p0, p2 - p1
    acceleration = v12 - v01
    axes[1].scatter(centers0[:, 0], centers0[:, 1], s=12, alpha=0.2)
    axes[1].scatter(centers1[:, 0], centers1[:, 1], s=12, alpha=0.2)
    axes[1].scatter(centers2[:, 0], centers2[:, 1], s=12, alpha=0.2)
    axes[1].plot([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], marker="o")
    axes[1].annotate("", xy=p1, xytext=p0, arrowprops={"arrowstyle": "->"})
    axes[1].annotate("", xy=p2, xytext=p1, arrowprops={"arrowstyle": "->"})
    axes[1].set_title("Representative 2D triple\nv01=({:.2f},{:.2f}), v12=({:.2f},{:.2f})\nacceleration=({:.2f},{:.2f})".format(
        v01[0], v01[1], v12[0], v12[1], acceleration[0], acceleration[1]))
    axes[1].set_aspect("equal", adjustable="datalim")
    top = np.argsort(conditional)[-10:][::-1]
    text = ["fixed history: node {} -> node {}".format(i0, i1), "", "Top possible current nodes:"]
    for rank, index in enumerate(top, 1):
        cv = centers2[index] - p1
        ca = cv - v01
        cosine = float(np.dot(v01, cv) / max(np.linalg.norm(v01) * np.linalg.norm(cv), 1e-8))
        text.append("{:02d}. node {:02d} score={:7.3f} |a|={:6.2f} cos={:6.3f}".format(
            rank, index, conditional[index], np.linalg.norm(ca), cosine))
    axes[2].axis("off")
    axes[2].text(0.01, 0.98, "\n".join(text), va="top", family="monospace", fontsize=9)
    fig.suptitle("Stage 7 — Second-order score: compare two consecutive 2D velocity vectors")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_dp(emission, alpha_history, grids, frame_ids, path, out, dpi):
    accumulated = forward_probabilities(emission, alpha_history)
    fig, axes = plt.subplots(2, len(frame_ids), figsize=(22, 9))
    for time in range(len(frame_ids)):
        heatmap(axes[0, time], softmax_np(emission[time]), grids[time],
                "frame {}\nlocal emission".format(frame_ids[time]), [path[time]])
        heatmap(axes[1, time], accumulated[time], grids[time],
                "frame {}\nforward accumulated".format(frame_ids[time]), [path[time]])
    fig.suptitle("Stage 8 — Dynamic programming accumulates visual and motion evidence across time\nTop: single-frame evidence. Bottom: evidence after preceding frames are integrated.")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_final(sat_image, candidate_pixels, grid, raw_prob, emission,
               posterior, gt_pixel, hard_pixel, path_pixel, final_pixel, gate, out, dpi):
    fig, axes = plt.subplots(1, 4, figsize=(23, 5.7))
    heatmap(axes[0], raw_prob, grid, "Raw retrieval probability")
    heatmap(axes[1], softmax_np(emission), grid, "Learned emission probability")
    heatmap(axes[2], posterior, grid, "Official CRF path posterior")
    crop, box = local_crop(sat_image, candidate_pixels, margin=250)
    cp = shift_pixels(candidate_pixels, box)
    points = np.stack([hard_pixel, path_pixel, final_pixel, gt_pixel])
    points = shift_pixels(points, box)
    axes[3].imshow(crop)
    axes[3].scatter(cp[:, 0], cp[:, 1], s=24, facecolors="none", edgecolors="white")
    labels = ["HardMS", "path expectation", "final RTL-CRF", "GT"]
    markers = ["s", "o", "*", "x"]
    sizes = [80, 90, 180, 110]
    for point, label, marker, size in zip(points, labels, markers, sizes):
        axes[3].scatter([point[0]], [point[1]], marker=marker, s=size, label=label)
    axes[3].plot(points[:2, 0], points[:2, 1], linestyle="--")
    axes[3].set_title("final = HardMS + {:.3f} × (path − HardMS)".format(gate))
    axes[3].legend(fontsize=8)
    axes[3].axis("off")
    fig.suptitle("Stage 9 — Final posterior, continuous weighted expectation, and HardMS-anchored fusion")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(sat_image, candidate_pixels, gt_px, hard_px, vit_px, final_px, out, dpi):
    all_pixels = candidate_pixels.reshape(-1, 2)
    trajectory_pixels = np.concatenate(
        [
            np.asarray(gt_px).reshape(-1, 2),
            np.asarray(hard_px).reshape(-1, 2),
            np.asarray(vit_px).reshape(-1, 2),
            np.asarray(final_px).reshape(1, 2),
        ],
        axis=0,
    )
    crop, box = local_crop(
        sat_image,
        np.concatenate([all_pixels, trajectory_pixels], axis=0),
        margin=300,
    )
    gt_px, hard_px, vit_px = shift_pixels(gt_px, box), shift_pixels(hard_px, box), shift_pixels(vit_px, box)
    final_px = shift_pixels(final_px[None], box)[0]
    fig, axis = plt.subplots(figsize=(11, 8))
    axis.imshow(crop)
    axis.plot(gt_px[:, 0], gt_px[:, 1], marker="o", label="GT 5-frame path")
    axis.plot(hard_px[:, 0], hard_px[:, 1], marker="s", linestyle="--", label="per-frame HardMS")
    axis.plot(vit_px[:, 0], vit_px[:, 1], marker="o", label="diagnostic Viterbi MAP path")
    axis.scatter([final_px[0]], [final_px[1]], marker="*", s=190, label="official final output at t")
    axis.set_title("Stage 10 — The flattened temporal states mapped back to the real 2D satellite map")
    axis.legend()
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def contact_sheet(paths, out):
    images = [Image.open(path).convert("RGB") for path in paths]
    width = 800
    thumbs = [image.resize((width, int(image.height * width / image.width))) for image in images]
    columns = 2
    rows = int(math.ceil(len(thumbs) / columns))
    height = max(image.height for image in thumbs)
    canvas = Image.new("RGB", (columns * width, rows * height), "white")
    for index, image in enumerate(thumbs):
        canvas.paste(image, ((index % columns) * width, (index // columns) * height))
    canvas.save(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=config.ROUTE_NAMES, default="route_A")
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--split", choices=("test", "all"), default="test")
    parser.add_argument("--auto-frame", choices=("best_improvement", "largest_hardms_error", "largest_final_error"), default="best_improvement")
    parser.add_argument("--jitter-m", type=float, default=config.LOCAL_PRIOR_JITTER_M)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--output-root", type=Path, default=config.OUTPUT_DIR / "stage_debug")
    args = parser.parse_args()

    route_index = config.ROUTE_NAMES.index(args.route)
    route_root = config.ROUTE_ROOTS[route_index]
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    print("device:", device)
    print("loading frozen retrieval model...")
    visual = FrozenVisualLocalizer(device)
    dataset = RouteDataset(route_root, train=False, origin_lat=visual.origin_lat, origin_lon=visual.origin_lon)
    end = select_end_index(dataset, args.route, args.frame_id, args.end_index, args.split, args.auto_frame)
    window = int(config.TEMPORAL_WINDOW)
    indices = list(range(end - window + 1, end + 1))
    items = [dataset[index] for index in indices]
    frame_ids_list = [int(item["frame_id"]) for item in items]
    image_paths = [item["image_path"] for item in items]
    print("dataset indices:", indices)
    print("frame ids:", frame_ids_list)

    uav = torch.stack([item["uav"] for item in items]).to(device)
    gt_xy = torch.stack([item["xy"].float() for item in items])
    gt_pixel = torch.stack([item["pixel"].float() for item in items])
    jitter = deterministic_jitter(len(dataset), route_index, float(args.jitter_m))[indices]
    prior_xy = gt_xy + jitter

    print("running 5-frame frozen retrieval...")
    with torch.no_grad():
        candidate = visual.candidate_batch(visual.encode_uav_clip(uav), prior_xy.to(device), int(config.GRID_SIZE))

    print("loading RTL-CRF checkpoint...")
    checkpoint = torch.load(config.TEMPORAL_CHECKPOINT, map_location=device)
    if checkpoint.get("architecture") != "ResidualSecondOrderTemporalLatticeCRF":
        raise RuntimeError("checkpoint is not RTL-CRF: {}".format(checkpoint.get("architecture")))
    model = TemporalLatticeCRF().to(device)
    model.load_state_dict(checkpoint.get("best_model") or checkpoint["model"], strict=True)
    model.eval()

    frame_ids = torch.tensor(frame_ids_list, dtype=torch.long, device=device)[None]
    with torch.no_grad():
        debug = model_intermediates(
            model, candidate.z_uav[None], candidate.z_sat[None], candidate.raw_logits[None],
            candidate.raw_prob[None], candidate.centers[None], frame_ids, candidate.hardms_xy[None]
        )

    official = debug["official"]
    emission = debug["emission"][0]
    first = debug["first"][0]
    second = [value[0] for value in debug["second"]]
    alphas = [value[0] for value in debug["alpha_history"]]
    path, path_score = viterbi_path(emission, first, second)

    centers = npy(candidate.centers)
    pixels = npy(visual.gallery["pixel"][candidate.indices])
    raw_logits = npy(candidate.raw_logits)
    raw_prob = npy(candidate.raw_prob)
    emission_np = npy(emission)
    first_np = npy(first)
    second_np = [npy(value) for value in second]
    alpha_np = [npy(value) for value in alphas]
    posterior = npy(official.path_probability[0])
    hard = npy(candidate.hardms_xy)
    raw_top1 = npy(candidate.raw_top1_xy)
    gt_xy_np, gt_pixel_np, prior_xy_np = npy(gt_xy), npy(gt_pixel), npy(prior_xy)
    path_xy = npy(official.path_expectation[0])
    final_xy = npy(official.final_xy[0])
    gate = float(official.correction_gate[0].item())
    viterbi_xy = np.stack([centers[time, index] for time, index in enumerate(path)])

    grids, regular = [], []
    for time in range(window):
        grid, is_regular = grid_layout(pixels[time], int(config.GRID_SIZE))
        grids.append(grid)
        regular.append(bool(is_regular))

    prior_pixels = meter_xy_to_pixel_exact(
        prior_xy_np, visual.origin_lat, visual.origin_lon, dataset.mapper
    )
    hard_pixels_exact = meter_xy_to_pixel_exact(
        hard, visual.origin_lat, visual.origin_lon, dataset.mapper
    )
    path_pixel_exact = meter_xy_to_pixel_exact(
        path_xy, visual.origin_lat, visual.origin_lon, dataset.mapper
    )
    final_pixel_exact = meter_xy_to_pixel_exact(
        final_xy, visual.origin_lat, visual.origin_lon, dataset.mapper
    )
    viterbi_pixels_exact = meter_xy_to_pixel_exact(
        viterbi_xy, visual.origin_lat, visual.origin_lon, dataset.mapper
    )
    out_dir = args.output_root / "{}_frame_{}".format(args.route, frame_ids_list[-1])
    ensure_dir(out_dir)
    sat_image = Image.open(config.SAT_IMAGE).convert("RGB")
    generated = []

    jobs = [
        ("01_actual_uav_window.png", lambda out: plot_inputs(image_paths, frame_ids_list, out, args.dpi)),
        ("02_real_2d_candidate_lattices.png", lambda out: plot_lattices(sat_image, pixels, gt_pixel_np, prior_pixels, frame_ids_list, out, args.dpi)),
        ("03_patch_grid_to_flattened_nodes.png", lambda out: plot_patch_mapping(sat_image, pixels[-1], grids[-1], out, args.dpi)),
        ("04_raw_cosine_logits.png", lambda out: plot_five(raw_logits, grids, frame_ids_list, path, out,
             "Stage 4 — Raw MobileCLIP+MLP cosine logits; no Top-1 collapse", args.dpi)),
        ("05_learned_emission_scores.png", lambda out: plot_five(emission_np, grids, frame_ids_list, path, out,
             "Stage 5 — Learned emission potentials after feature and score calibration", args.dpi)),
        ("06_first_order_transition.png", lambda out: plot_first(first_np, centers[0], centers[1], grids[1], path[0], path[1], out, args.dpi)),
        ("07_second_order_transition.png", lambda out: plot_second(second_np[-1], centers[-3], centers[-2], centers[-1], grids[-1], path[-3], path[-2], path[-1], out, args.dpi)),
        ("08_dynamic_programming_accumulation.png", lambda out: plot_dp(emission_np, alpha_np, grids, frame_ids_list, path, out, args.dpi)),
        ("09_final_posterior_and_fusion.png", lambda out: plot_final(sat_image, pixels[-1], grids[-1], raw_prob[-1], emission_np[-1], posterior,
             gt_pixel_np[-1], hard_pixels_exact[-1], path_pixel_exact, final_pixel_exact, gate, out, args.dpi)),
        ("10_real_2d_window_trajectory.png", lambda out: plot_trajectory(sat_image, pixels, gt_pixel_np, hard_pixels_exact, viterbi_pixels_exact, final_pixel_exact, out, args.dpi)),
    ]
    for filename, function in jobs:
        output = out_dir / filename
        print("writing", output)
        function(output)
        generated.append(output)
    contact_sheet(generated, out_dir / "11_all_stages_contact_sheet.png")

    np.savez_compressed(
        out_dir / "rtl_crf_intermediate_tensors.npz",
        frame_ids=np.asarray(frame_ids_list), dataset_indices=np.asarray(indices),
        gt_xy=gt_xy_np, prior_xy=prior_xy_np,
        candidate_indices=npy(candidate.indices).astype(int), candidate_centers=centers,
        candidate_pixels=pixels, raw_logits=raw_logits, raw_probability=raw_prob,
        emission_logits=emission_np, first_transition_score=first_np,
        second_transition_score=np.stack(second_np), alpha_history=np.stack(alpha_np),
        path_probability=posterior, raw_top1_xy=raw_top1, hardms_xy=hard,
        viterbi_path_indices=np.asarray(path), viterbi_path_xy=viterbi_xy,
        path_expectation_xy=path_xy, final_xy=final_xy, correction_gate=np.asarray([gate]),
    )
    report = {
        "route": args.route,
        "dataset_indices": indices,
        "frame_ids": frame_ids_list,
        "regular_2d_lattice_by_frame": regular,
        "tensor_shapes": {
            "raw_logits": list(raw_logits.shape),
            "emission_logits": list(emission_np.shape),
            "first_transition": list(first_np.shape),
            "second_transition_each": [list(value.shape) for value in second_np],
            "alpha_each": [list(value.shape) for value in alpha_np],
            "path_probability": list(posterior.shape),
        },
        "interpretation": {
            "time": "one ordered axis of five frames",
            "space_per_time": "36 states, each storing a true 2D map coordinate",
            "flattening": "only candidate indexing; spatial calculations use x and y",
            "official_inference": "log-sum-exp sum-product over all paths",
            "viterbi_path": "diagnostic representative path only",
        },
        "result": {
            "viterbi_path_indices": path,
            "viterbi_score": path_score,
            "hardms_xy": hard[-1].tolist(),
            "path_expectation_xy": path_xy.tolist(),
            "final_xy": final_xy.tolist(),
            "gt_xy": gt_xy_np[-1].tolist(),
            "correction_gate": gate,
            "hardms_error_m": float(np.linalg.norm(hard[-1] - gt_xy_np[-1])),
            "path_error_m": float(np.linalg.norm(path_xy - gt_xy_np[-1])),
            "final_error_m": float(np.linalg.norm(final_xy - gt_xy_np[-1])),
        },
    }
    with (out_dir / "rtl_crf_stage_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nDone:", out_dir)
    print("HardMS error: {:.3f} m".format(report["result"]["hardms_error_m"]))
    print("Path expectation error: {:.3f} m".format(report["result"]["path_error_m"]))
    print("Final RTL-CRF error: {:.3f} m".format(report["result"]["final_error_m"]))
    print("Correction gate: {:.4f}".format(gate))
    print("Viterbi is diagnostic only; official output uses the full path posterior.")


if __name__ == "__main__":
    main()