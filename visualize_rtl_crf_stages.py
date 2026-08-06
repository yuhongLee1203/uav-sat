
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
import matplotlib.font_manager as font_manager
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



def forward_probabilities(emission: np.ndarray, alpha_history: Sequence[np.ndarray]):
    """把每一步 alpha 轉成『目前幀 36 個候選的累積機率』。"""
    result = [softmax_np(emission[0])]
    result.append(softmax_np(np.logaddexp.reduce(alpha_history[0], axis=0)))
    for alpha in alpha_history[1:]:
        result.append(softmax_np(np.logaddexp.reduce(alpha, axis=0)))
    return result


def configure_chinese_font(font_path: Optional[str] = None):
    """優先使用系統中的繁體中文／CJK 字型。"""
    candidates = []
    if font_path:
        candidates.append(Path(font_path))
    candidates.extend(
        [
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCJKtc-Regular.otf"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
            Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            name = font_manager.FontProperties(fname=str(candidate)).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            print("使用中文字型:", candidate)
            return str(candidate)
    plt.rcParams["axes.unicode_minus"] = False
    print("警告：找不到 CJK 中文字型。圖仍會產生，但中文字可能顯示成方框。")
    print("可使用 --font-path /你的/NotoSansCJK-Regular.ttc")
    return None


def heatmap(
    axis,
    values,
    grid,
    title,
    selected=None,
    decimals=2,
    colorbar_label=None,
):
    """把攤平的 36 個值依照真實 6×6 排列畫回去。"""
    matrix = to_grid(values, grid)
    image = axis.imshow(matrix, interpolation="nearest")
    axis.set_title(title, fontsize=10)
    axis.set_xticks(range(matrix.shape[1]))
    axis.set_yticks(range(matrix.shape[0]))
    axis.set_xlabel("候選格欄")
    axis.set_ylabel("候選格列")
    selected_set = set([] if selected is None else [int(v) for v in selected])
    low, high = float(np.nanmin(matrix)), float(np.nanmax(matrix))
    scale = max(high - low, 1e-8)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            index = int(grid[row, col])
            mark = "★" if index in selected_set else ""
            norm = (float(matrix[row, col]) - low) / scale
            axis.text(
                col,
                row,
                "{}{}\n{:.{}f}".format(index, mark, matrix[row, col], decimals),
                ha="center",
                va="center",
                fontsize=5.4,
                color="white" if norm > 0.55 else "black",
            )
    bar = plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    if colorbar_label:
        bar.set_label(colorbar_label)


def frame_name(time_index: int, frame_count: int, frame_id: int) -> str:
    relative = time_index - frame_count + 1
    return "t{}／影格 {}".format(relative if relative != 0 else "", frame_id)


def plot_inputs(image_paths, frame_ids, out, dpi):
    fig, axes = plt.subplots(1, len(image_paths), figsize=(19, 4.3))
    for time, (path, frame_id) in enumerate(zip(image_paths, frame_ids)):
        axes[time].imshow(Image.open(path).convert("RGB"))
        relative = time - len(image_paths) + 1
        relative_text = "t" if relative == 0 else "t{}".format(relative)
        axes[time].set_title("{}／影格 {}".format(relative_text, frame_id))
        axes[time].axis("off")
    fig.suptitle(
        "圖 1：RTL-CRF 實際使用的連續五張 UAV 圖片\n"
        "前四張提供時序資訊；模型最後只輸出第 5 張（t）的定位",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_lattice_outline(axis, points, color, label=None, alpha=0.75):
    """用候選點的外框表示一幀 6×6 候選格在全域圖上的範圍。"""
    pts = np.asarray(points)
    left, right = pts[:, 0].min(), pts[:, 0].max()
    top, bottom = pts[:, 1].min(), pts[:, 1].max()
    xs = [left, right, right, left, left]
    ys = [top, top, bottom, bottom, top]
    axis.plot(xs, ys, linewidth=1.5, alpha=alpha, color=color, label=label)


def plot_lattices(
    sat_image,
    candidate_pixels,
    gt_pixels,
    prior_pixels,
    frame_ids,
    out,
    dpi,
):
    """
    上半部：所有幀共用同一裁切座標，才能比較 GT 與 prior 真實軌跡。
    下半部：每幀各自裁切，僅用來看 GT 在該幀 6×6 候選格內的位置。
    """
    frame_count = len(frame_ids)
    all_points = np.concatenate(
        [
            candidate_pixels.reshape(-1, 2),
            np.asarray(gt_pixels).reshape(-1, 2),
            np.asarray(prior_pixels).reshape(-1, 2),
        ],
        axis=0,
    )
    common_crop, common_box = local_crop(sat_image, all_points, margin=260)
    common_candidates = [
        shift_pixels(candidate_pixels[t], common_box) for t in range(frame_count)
    ]
    common_gt = shift_pixels(gt_pixels, common_box)
    common_prior = shift_pixels(prior_pixels, common_box)

    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, frame_count, height_ratios=[1.35, 1.0])

    top_axis = fig.add_subplot(gs[0, :])
    top_axis.imshow(common_crop)
    colors = [plt.cm.tab10(index) for index in range(frame_count)]
    for time in range(frame_count):
        cp = common_candidates[time]
        draw_lattice_outline(
            top_axis,
            cp,
            colors[time],
            label="影格 {} 的 6×6 候選範圍".format(frame_ids[time]),
        )
        top_axis.scatter(
            cp[:, 0],
            cp[:, 1],
            s=8,
            alpha=0.20,
            color=colors[time],
        )

    top_axis.plot(
        common_gt[:, 0],
        common_gt[:, 1],
        marker="o",
        linewidth=2.5,
        label="真正 GT 五幀軌跡",
    )
    top_axis.plot(
        common_prior[:, 0],
        common_prior[:, 1],
        marker="x",
        linestyle="--",
        linewidth=2.0,
        label="GT + 每幀獨立 jitter 的 prior 軌跡",
    )
    for time, frame_id in enumerate(frame_ids):
        top_axis.text(
            common_gt[time, 0],
            common_gt[time, 1],
            " GT{}".format(frame_id),
            fontsize=8,
        )
        top_axis.text(
            common_prior[time, 0],
            common_prior[time, 1],
            " prior{}".format(frame_id),
            fontsize=8,
        )
        top_axis.annotate(
            "",
            xy=(common_prior[time, 0], common_prior[time, 1]),
            xytext=(common_gt[time, 0], common_gt[time, 1]),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0, "alpha": 0.65},
        )
    top_axis.set_title(
        "上半部：固定同一張衛星圖與同一座標系\n"
        "GT 應呈現真實移動；prior 因每幀 jitter 而可能亂跳。箭頭是該幀 GT → noisy prior 的偏移",
        fontsize=13,
    )
    top_axis.legend(fontsize=8, ncol=2, loc="best")
    top_axis.axis("off")

    local_axes = []
    for time in range(frame_count):
        axis = fig.add_subplot(gs[1, time])
        local_axes.append(axis)
        crop, box = local_crop(sat_image, candidate_pixels[time], margin=180)
        cp = shift_pixels(candidate_pixels[time], box)
        gp = shift_pixels(gt_pixels[time : time + 1], box)[0]
        pp = shift_pixels(prior_pixels[time : time + 1], box)[0]
        axis.imshow(crop)
        axis.scatter(
            cp[:, 0],
            cp[:, 1],
            s=24,
            facecolors="none",
            edgecolors="white",
            linewidths=0.8,
        )
        for index, point in enumerate(cp):
            axis.text(
                point[0],
                point[1],
                str(index),
                fontsize=5,
                ha="center",
                va="center",
            )
        axis.scatter([gp[0]], [gp[1]], marker="*", s=150, label="GT")
        axis.scatter([pp[0]], [pp[1]], marker="x", s=90, label="GT+jitter prior")
        axis.set_title(
            "影格 {}\n此小圖以本幀候選格重新裁切".format(frame_ids[time]),
            fontsize=9,
        )
        axis.axis("off")

    handles, labels = local_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(
        "圖 2：GT、GT+jitter prior 與每幀 6×6 候選格的真實關係\n"
        "注意：下排五張小圖各有自己的裁切原點，不能用星星在小圖內的位置直接比較 GT 軌跡",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_patch_mapping(sat_image, candidate_pixels, grid, out, dpi):
    size = grid.shape[0]
    fig = plt.figure(figsize=(18, 10))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.1])
    patches = outer[0].subgridspec(size, size, wspace=0.02, hspace=0.02)

    for row in range(size):
        for col in range(size):
            axis = fig.add_subplot(patches[row, col])
            index = int(grid[row, col])
            pixel = candidate_pixels[index]
            patch = crop_satellite(
                sat_image,
                float(pixel[0]),
                float(pixel[1]),
                int(config.SAT_CROP_SIZE),
            )
            axis.imshow(patch)
            axis.text(
                0.03,
                0.09,
                "候選 {}".format(index),
                transform=axis.transAxes,
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.78, "pad": 1},
            )
            axis.axis("off")

    text_axis = fig.add_subplot(outer[1])
    text_axis.axis("off")
    lines = [
        "這張圖只是在解釋資料如何存進 tensor：",
        "",
        "左邊是真正的 6×6 衛星 patch 排列。",
        "程式為了計算，把它依序編成候選 0～35。",
        "",
        "『攤平成 36 個節點』不代表空間變成一維。",
        "每個候選仍保留：",
        "  1. 衛星 patch 圖片",
        "  2. 真實地圖座標 (x, y)",
        "  3. 衛星大圖 pixel 座標",
        "  4. SAT embedding 與相似度",
        "",
        "格子與候選編號對照：",
    ]
    for row in range(size):
        lines.append(
            "  ".join(
                "({},{})→{:02d}".format(row, col, int(grid[row, col]))
                for col in range(size)
            )
        )
    lines.extend(
        [
            "",
            "時序模型計算移動時使用真實座標：",
            "  Δx = 下一候選 x − 前一候選 x",
            "  Δy = 下一候選 y − 前一候選 y",
            "",
            "不會使用『候選編號 21 − 候選編號 20』",
            "來代表空間移動。",
        ]
    )
    text_axis.text(
        0.01,
        0.98,
        "\n".join(lines),
        va="top",
        fontsize=10,
        linespacing=1.35,
    )
    fig.suptitle(
        "圖 3：真正 2D 的 6×6 patches，如何在程式中編成候選 0～35",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_five_heatmaps(
    values,
    grids,
    frame_ids,
    selected_indices,
    out,
    title,
    subtitle,
    colorbar_label,
    dpi,
):
    frame_count = len(frame_ids)
    fig, axes = plt.subplots(1, frame_count, figsize=(22, 5.2))
    for time, axis in enumerate(axes):
        heatmap(
            axis,
            values[time],
            grids[time],
            "影格 {}\n★＝此張圖最高分候選 {}".format(
                frame_ids[time], int(selected_indices[time])
            ),
            selected=[selected_indices[time]],
            colorbar_label=colorbar_label,
        )
    fig.suptitle("{}\n{}".format(title, subtitle), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def choose_first_pair(alpha0: np.ndarray) -> Tuple[int, int]:
    flat = int(np.asarray(alpha0).reshape(-1).argmax())
    candidate_count = alpha0.shape[1]
    return flat // candidate_count, flat % candidate_count


def choose_last_triple(
    alpha_before_last: np.ndarray,
    second_last: np.ndarray,
    emission_last: np.ndarray,
) -> Tuple[int, int, int]:
    scores = (
        np.asarray(alpha_before_last)[:, :, None]
        + np.asarray(second_last)
        + np.asarray(emission_last)[None, None, :]
    )
    flat = int(scores.reshape(-1).argmax())
    n1, n2 = scores.shape[1], scores.shape[2]
    i0 = flat // (n1 * n2)
    remainder = flat % (n1 * n2)
    i1 = remainder // n2
    i2 = remainder % n2
    return i0, i1, i2


def plot_first(
    first,
    centers0,
    centers1,
    grid1,
    prev_index,
    curr_index,
    frame0,
    frame1,
    out,
    dpi,
):
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.8))

    image = axes[0].imshow(first, interpolation="nearest", aspect="auto")
    axes[0].scatter([curr_index], [prev_index], marker="*", s=150)
    axes[0].set_title(
        "完整 36×36 一階移動分數表\n"
        "縱軸＝前一幀候選，橫軸＝下一幀候選",
        fontsize=10,
    )
    axes[0].set_xlabel("影格 {} 的候選編號".format(frame1))
    axes[0].set_ylabel("影格 {} 的候選編號".format(frame0))
    plt.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    heatmap(
        axes[1],
        first[prev_index],
        grid1,
        "固定前一幀候選 {} 後\n下一幀 36 個候選各自的移動合理性".format(prev_index),
        selected=[curr_index],
        colorbar_label="一階移動分數",
    )

    p0, p1 = centers0[prev_index], centers1[curr_index]
    axes[2].scatter(
        centers0[:, 0],
        centers0[:, 1],
        s=14,
        alpha=0.25,
        label="影格 {} 的 36 候選".format(frame0),
    )
    axes[2].scatter(
        centers1[:, 0],
        centers1[:, 1],
        s=14,
        alpha=0.25,
        label="影格 {} 的 36 候選".format(frame1),
    )
    axes[2].annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops={"arrowstyle": "->", "linewidth": 2},
    )
    axes[2].scatter([p0[0]], [p0[1]], s=100, marker="o")
    axes[2].scatter([p1[0]], [p1[1]], s=150, marker="*")
    delta = p1 - p0
    axes[2].set_title(
        "★所代表的真實 2D 移動\n"
        "Δx={:.2f} m，Δy={:.2f} m，距離={:.2f} m".format(
            delta[0], delta[1], np.linalg.norm(delta)
        ),
        fontsize=10,
    )
    axes[2].set_aspect("equal", adjustable="datalim")
    axes[2].legend(fontsize=8)

    fig.suptitle(
        "圖 6：一階時序分數＝模型判斷『相鄰兩幀這樣移動是否合理』\n"
        "★是前兩幀累積總分最高的候選配對，只用來指出分數表中的一個實際例子",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_second(
    second,
    centers0,
    centers1,
    centers2,
    grid2,
    i0,
    i1,
    i2,
    frame0,
    frame1,
    frame2,
    out,
    dpi,
):
    conditional = second[i0, i1]
    fig, axes = plt.subplots(1, 3, figsize=(22, 5.8))

    heatmap(
        axes[0],
        conditional,
        grid2,
        "固定歷史候選 {} → {} 後\n"
        "目前幀 36 候選的二階運動分數".format(i0, i1),
        selected=[i2],
        colorbar_label="二階運動分數",
    )

    p0, p1, p2 = centers0[i0], centers1[i1], centers2[i2]
    v01, v12 = p1 - p0, p2 - p1
    acceleration = v12 - v01
    axes[1].scatter(centers0[:, 0], centers0[:, 1], s=12, alpha=0.18)
    axes[1].scatter(centers1[:, 0], centers1[:, 1], s=12, alpha=0.18)
    axes[1].scatter(centers2[:, 0], centers2[:, 1], s=12, alpha=0.18)
    axes[1].plot(
        [p0[0], p1[0], p2[0]],
        [p0[1], p1[1], p2[1]],
        marker="o",
        linewidth=2,
    )
    axes[1].annotate("", xy=p1, xytext=p0, arrowprops={"arrowstyle": "->"})
    axes[1].annotate("", xy=p2, xytext=p1, arrowprops={"arrowstyle": "->"})
    axes[1].set_title(
        "★三候選在真實 2D 地圖中的移動\n"
        "{}→{}：v₀₁=({:.2f},{:.2f})\n"
        "{}→{}：v₁₂=({:.2f},{:.2f})\n"
        "速度改變 v₁₂−v₀₁=({:.2f},{:.2f})".format(
            frame0,
            frame1,
            v01[0],
            v01[1],
            frame1,
            frame2,
            v12[0],
            v12[1],
            acceleration[0],
            acceleration[1],
        ),
        fontsize=9,
    )
    axes[1].set_aspect("equal", adjustable="datalim")

    top = np.argsort(conditional)[-10:][::-1]
    text = [
        "固定前兩個位置：候選 {} → 候選 {}".format(i0, i1),
        "",
        "目前幀分數最高的 10 個候選：",
        "分數同時考慮兩段 2D 速度、方向變化與加速度",
        "",
    ]
    for rank, index in enumerate(top, 1):
        current_velocity = centers2[index] - p1
        current_acceleration = current_velocity - v01
        cosine = float(
            np.dot(v01, current_velocity)
            / max(np.linalg.norm(v01) * np.linalg.norm(current_velocity), 1e-8)
        )
        text.append(
            "{:02d}. 候選 {:02d}  分數={:7.3f}  |速度改變|={:6.2f}  方向cos={:6.3f}".format(
                rank,
                index,
                conditional[index],
                np.linalg.norm(current_acceleration),
                cosine,
            )
        )
    axes[2].axis("off")
    axes[2].text(0.01, 0.98, "\n".join(text), va="top", fontsize=9)

    fig.suptitle(
        "圖 7：二階時序分數＝模型比較連續兩段 2D 移動，壓低突然左→右→左或前→後→前的組合\n"
        "★是最後一次 CRF 更新中總分最高的三候選組合；正式輸出仍會整合全部候選組合",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_dp(emission, alpha_history, grids, frame_ids, out, dpi):
    accumulated = forward_probabilities(emission, alpha_history)
    frame_count = len(frame_ids)
    fig, axes = plt.subplots(2, frame_count, figsize=(22, 9.2))

    for time in range(frame_count):
        local_probability = softmax_np(emission[time])
        local_best = int(local_probability.argmax())
        accumulated_best = int(accumulated[time].argmax())

        heatmap(
            axes[0, time],
            local_probability,
            grids[time],
            "影格 {}\n只看本幀：最高候選 {}".format(frame_ids[time], local_best),
            selected=[local_best],
            colorbar_label="本幀候選機率",
        )
        heatmap(
            axes[1, time],
            accumulated[time],
            grids[time],
            "影格 {}\n加入前面時序：最高候選 {}".format(
                frame_ids[time], accumulated_best
            ),
            selected=[accumulated_best],
            colorbar_label="累積候選機率",
        )

    fig.suptitle(
        "圖 8：CRF 如何逐幀合併資訊\n"
        "上排＝只看當前 UAV/SAT 的單幀候選可信度；下排＝再加入前面影格與一階／二階移動分數後的累積可信度\n"
        "同一影格上下兩排的高峰若改變，表示時序資訊正在修正單幀判斷",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_final(
    sat_image,
    candidate_pixels,
    grid,
    raw_prob,
    emission,
    posterior,
    gt_pixel,
    hard_pixel,
    path_pixel,
    final_pixel,
    gate,
    out,
    dpi,
):
    fig, axes = plt.subplots(1, 4, figsize=(23, 5.9))

    raw_best = int(np.asarray(raw_prob).argmax())
    emission_probability = softmax_np(emission)
    emission_best = int(emission_probability.argmax())
    posterior_best = int(np.asarray(posterior).argmax())

    heatmap(
        axes[0],
        raw_prob,
        grid,
        "目前第 5 幀\n原始檢索機率\n最高候選 {}".format(raw_best),
        selected=[raw_best],
        colorbar_label="原始檢索機率",
    )
    heatmap(
        axes[1],
        emission_probability,
        grid,
        "目前第 5 幀\n模型校正後單幀機率\n最高候選 {}".format(emission_best),
        selected=[emission_best],
        colorbar_label="校正後單幀機率",
    )
    heatmap(
        axes[2],
        posterior,
        grid,
        "正式 CRF 最終後驗機率\n已整合五幀與所有候選路徑\n最高候選 {}".format(posterior_best),
        selected=[posterior_best],
        colorbar_label="五幀整合後機率",
    )

    all_for_crop = np.concatenate(
        [
            np.asarray(candidate_pixels).reshape(-1, 2),
            np.stack([gt_pixel, hard_pixel, path_pixel, final_pixel]),
        ],
        axis=0,
    )
    crop, box = local_crop(sat_image, all_for_crop, margin=250)
    cp = shift_pixels(candidate_pixels, box)
    points = shift_pixels(
        np.stack([hard_pixel, path_pixel, final_pixel, gt_pixel]), box
    )
    axes[3].imshow(crop)
    axes[3].scatter(
        cp[:, 0],
        cp[:, 1],
        s=24,
        facecolors="none",
        edgecolors="white",
        label="目前幀 36 個候選中心",
    )
    labels = [
        "Fixed HardMS 單幀錨點",
        "CRF 後驗加權後的連續位置",
        "最終 RTL-CRF 位置",
        "目前幀 GT",
    ]
    markers = ["s", "o", "*", "x"]
    sizes = [85, 95, 185, 115]
    for point, label, marker, size in zip(points, labels, markers, sizes):
        axes[3].scatter(
            [point[0]], [point[1]], marker=marker, s=size, label=label
        )
    axes[3].plot(
        points[:2, 0],
        points[:2, 1],
        linestyle="--",
        linewidth=1.4,
    )
    axes[3].set_title(
        "座標如何合併\n"
        "最終位置＝HardMS＋{:.3f}×（CRF連續位置−HardMS）".format(gate),
        fontsize=10,
    )
    axes[3].legend(fontsize=7)
    axes[3].axis("off")

    fig.suptitle(
        "圖 9：目前第 5 幀從原始檢索 → 單幀校正 → 五幀 CRF 後驗 → 連續座標 → 最終融合\n"
        "第三張 6×6 才是正式時序推論後的 36 候選機率；不是 Viterbi，也不是只選一條路徑",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(
    sat_image,
    candidate_pixels,
    gt_px,
    prior_px,
    hard_px,
    path_px,
    final_px,
    frame_ids,
    out,
    dpi,
):
    all_pixels = np.concatenate(
        [
            np.asarray(candidate_pixels).reshape(-1, 2),
            np.asarray(gt_px).reshape(-1, 2),
            np.asarray(prior_px).reshape(-1, 2),
            np.asarray(hard_px).reshape(-1, 2),
            np.asarray(path_px).reshape(1, 2),
            np.asarray(final_px).reshape(1, 2),
        ],
        axis=0,
    )
    crop, box = local_crop(sat_image, all_pixels, margin=320)
    gt_local = shift_pixels(gt_px, box)
    prior_local = shift_pixels(prior_px, box)
    hard_local = shift_pixels(hard_px, box)
    path_local = shift_pixels(np.asarray(path_px)[None], box)[0]
    final_local = shift_pixels(np.asarray(final_px)[None], box)[0]
    current_candidates = shift_pixels(candidate_pixels[-1], box)

    fig, axis = plt.subplots(figsize=(12, 9))
    axis.imshow(crop)
    axis.scatter(
        current_candidates[:, 0],
        current_candidates[:, 1],
        s=24,
        facecolors="none",
        edgecolors="white",
        alpha=0.75,
        label="目前第 5 幀的 36 個候選中心",
    )
    axis.plot(
        gt_local[:, 0],
        gt_local[:, 1],
        marker="o",
        linewidth=2.5,
        label="真正 GT 五幀軌跡",
    )
    axis.plot(
        prior_local[:, 0],
        prior_local[:, 1],
        marker="x",
        linestyle="--",
        linewidth=1.8,
        label="GT+jitter prior 五幀軌跡",
    )
    axis.plot(
        hard_local[:, 0],
        hard_local[:, 1],
        marker="s",
        linestyle=":",
        linewidth=1.8,
        label="每幀 Fixed HardMS 單幀位置",
    )
    axis.scatter(
        [path_local[0]],
        [path_local[1]],
        marker="o",
        s=130,
        label="目前第 5 幀 CRF 後驗加權連續位置",
    )
    axis.scatter(
        [final_local[0]],
        [final_local[1]],
        marker="*",
        s=220,
        label="目前第 5 幀正式 RTL-CRF 最終位置",
    )
    for time, frame_id in enumerate(frame_ids):
        axis.text(gt_local[time, 0], gt_local[time, 1], " GT{}".format(frame_id), fontsize=8)
        axis.text(
            prior_local[time, 0],
            prior_local[time, 1],
            " prior{}".format(frame_id),
            fontsize=8,
        )

    axis.text(
        0.02,
        0.02,
        "重要：一個 5-frame window 的 RTL-CRF 正式輸出只有目前第 5 幀的位置。\n"
        "前四幀是用來建立時序證據，不會在這次 forward 中各自再輸出一個 RTL-CRF final。\n"
        "本圖已移除 Viterbi；因為正式模型不使用 Viterbi 產生最終定位。",
        transform=axis.transAxes,
        fontsize=10,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.82, "pad": 6},
    )
    axis.set_title(
        "圖 10：把所有重要位置放回同一張固定衛星圖\n"
        "這張才能正確比較 GT、noisy prior、HardMS 與目前幀最終 RTL-CRF 位置",
        fontsize=14,
    )
    axis.legend(fontsize=8, loc="best")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def contact_sheet(paths, out):
    images = [Image.open(path).convert("RGB") for path in paths]
    width = 900
    thumbs = [
        image.resize((width, int(image.height * width / image.width)))
        for image in images
    ]
    columns = 2
    rows = int(math.ceil(len(thumbs) / columns))
    height = max(image.height for image in thumbs)
    canvas = Image.new("RGB", (columns * width, rows * height), "white")
    for index, image in enumerate(thumbs):
        canvas.paste(
            image,
            ((index % columns) * width, (index // columns) * height),
        )
    canvas.save(out)


def write_readme(out_dir: Path):
    content = """RTL-CRF 圖 1～10 閱讀順序

圖 1：實際輸入的連續五張 UAV 圖片。正式輸出是第 5 張的位置。
圖 2：上排固定同一座標系，比較真實 GT 與每幀獨立 jitter 的 prior；下排才是各幀重新置中的局部 6×6 候選格。
圖 3：解釋 6×6 patch 如何編成候選 0～35；只是 tensor 索引，二維座標沒有消失。
圖 4：每幀 MobileCLIP+MLP 得到的原始 36 候選相似度。
圖 5：RTL-CRF 的單點評分網路校正後，每幀 36 候選的分數。
圖 6：相鄰兩幀 36×36 種移動組合的一階合理性分數。
圖 7：固定前兩個候選後，目前 36 候選形成三幀運動的二階合理性分數；它看兩段速度、方向與速度改變。
圖 8：上排只看本幀；下排加入前面影格與時序移動後的累積候選機率。
圖 9：目前第 5 幀正式 CRF 後驗、後驗加權連續座標，以及與 HardMS 的 learned gate 融合。
圖 10：全部重要位置放回同一張固定衛星圖；不包含 Viterbi，因為正式模型不使用 Viterbi。

正式模型重點：
- 時間是一條 5-frame 順序。
- 每個時間點都有 36 個真實 2D 候選位置。
- 模型用 log-sum-exp 整合全部可能候選路徑，不是只選一條 Viterbi path。
- 最後取得第 5 幀 36 個候選的後驗機率。
- 後驗機率乘上 36 個候選中心並加總，得到連續二維位置。
"""
    (out_dir / "00_圖1到圖10閱讀說明.txt").write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="產生 RTL-CRF 圖 1～10 的完整真實中間結果。"
    )
    parser.add_argument("--route", choices=config.ROUTE_NAMES, default="route_A")
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--split", choices=("test", "all"), default="test")
    parser.add_argument(
        "--auto-frame",
        choices=(
            "best_improvement",
            "largest_hardms_error",
            "largest_final_error",
        ),
        default="best_improvement",
    )
    parser.add_argument(
        "--jitter-m",
        type=float,
        default=config.LOCAL_PRIOR_JITTER_M,
    )
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--font-path", type=str, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=config.OUTPUT_DIR / "stage_debug_v3",
    )
    args = parser.parse_args()

    configure_chinese_font(args.font_path)

    route_index = config.ROUTE_NAMES.index(args.route)
    route_root = config.ROUTE_ROOTS[route_index]
    device = torch.device(
        args.device
        if torch.cuda.is_available() and str(args.device).startswith("cuda")
        else "cpu"
    )
    print("device:", device)
    print("route:", args.route)
    print("載入 frozen retrieval model...")
    visual = FrozenVisualLocalizer(device)

    dataset = RouteDataset(
        route_root,
        train=False,
        origin_lat=visual.origin_lat,
        origin_lon=visual.origin_lon,
    )
    end = select_end_index(
        dataset,
        args.route,
        args.frame_id,
        args.end_index,
        args.split,
        args.auto_frame,
    )
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
    jitter = deterministic_jitter(
        len(dataset), route_index, float(args.jitter_m)
    )[indices]
    prior_xy = gt_xy + jitter

    print("執行五幀 Frozen MobileCLIP+MLP retrieval...")
    with torch.no_grad():
        candidate = visual.candidate_batch(
            visual.encode_uav_clip(uav),
            prior_xy.to(device),
            int(config.GRID_SIZE),
        )

    print("載入 RTL-CRF checkpoint...")
    checkpoint = torch.load(config.TEMPORAL_CHECKPOINT, map_location=device)
    if checkpoint.get("architecture") != "ResidualSecondOrderTemporalLatticeCRF":
        raise RuntimeError(
            "checkpoint 不是目前 RTL-CRF：{}".format(
                checkpoint.get("architecture")
            )
        )

    model = TemporalLatticeCRF().to(device)
    model.load_state_dict(
        checkpoint.get("best_model") or checkpoint["model"],
        strict=True,
    )
    model.eval()

    frame_ids_tensor = torch.tensor(
        frame_ids_list,
        dtype=torch.long,
        device=device,
    )[None]

    with torch.no_grad():
        debug = model_intermediates(
            model,
            candidate.z_uav[None],
            candidate.z_sat[None],
            candidate.raw_logits[None],
            candidate.raw_prob[None],
            candidate.centers[None],
            frame_ids_tensor,
            candidate.hardms_xy[None],
        )

    official = debug["official"]
    emission = debug["emission"][0]
    first = debug["first"][0]
    second = [value[0] for value in debug["second"]]
    alphas = [value[0] for value in debug["alpha_history"]]

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
    gt_xy_np = npy(gt_xy)
    gt_pixel_np = npy(gt_pixel)
    prior_xy_np = npy(prior_xy)
    path_xy = npy(official.path_expectation[0])
    final_xy = npy(official.final_xy[0])
    gate = float(official.correction_gate[0].item())

    grids, regular = [], []
    for time in range(window):
        grid, is_regular = grid_layout(
            pixels[time],
            int(config.GRID_SIZE),
        )
        grids.append(grid)
        regular.append(bool(is_regular))

    prior_pixels = meter_xy_to_pixel_exact(
        prior_xy_np,
        visual.origin_lat,
        visual.origin_lon,
        dataset.mapper,
    )
    hard_pixels_exact = meter_xy_to_pixel_exact(
        hard,
        visual.origin_lat,
        visual.origin_lon,
        dataset.mapper,
    )
    path_pixel_exact = meter_xy_to_pixel_exact(
        path_xy,
        visual.origin_lat,
        visual.origin_lon,
        dataset.mapper,
    )
    final_pixel_exact = meter_xy_to_pixel_exact(
        final_xy,
        visual.origin_lat,
        visual.origin_lon,
        dataset.mapper,
    )

    raw_best_indices = raw_logits.argmax(axis=1)
    emission_best_indices = emission_np.argmax(axis=1)
    first_i0, first_i1 = choose_first_pair(alpha_np[0])
    second_i0, second_i1, second_i2 = choose_last_triple(
        alpha_np[-2],
        second_np[-1],
        emission_np[-1],
    )

    out_dir = (
        args.output_root
        / "{}_frame_{}".format(args.route, frame_ids_list[-1])
    )
    ensure_dir(out_dir)
    write_readme(out_dir)
    sat_image = Image.open(config.SAT_IMAGE).convert("RGB")

    generated = []
    jobs = [
        (
            "01_actual_uav_window.png",
            lambda out: plot_inputs(
                image_paths,
                frame_ids_list,
                out,
                args.dpi,
            ),
        ),
        (
            "02_gt_prior_and_candidate_lattices.png",
            lambda out: plot_lattices(
                sat_image,
                pixels,
                gt_pixel_np,
                prior_pixels,
                frame_ids_list,
                out,
                args.dpi,
            ),
        ),
        (
            "03_patch_grid_to_candidate_indices.png",
            lambda out: plot_patch_mapping(
                sat_image,
                pixels[-1],
                grids[-1],
                out,
                args.dpi,
            ),
        ),
        (
            "04_raw_cosine_similarity.png",
            lambda out: plot_five_heatmaps(
                raw_logits,
                grids,
                frame_ids_list,
                raw_best_indices,
                out,
                "圖 4：MobileCLIP+MLP 對每幀 36 個衛星 patches 的原始 cosine 相似度",
                "★只是標示該幀原始最高分；此時 RTL-CRF 尚未整合時間，也沒有提前把其他 35 候選丟掉",
                "原始 cosine 分數",
                args.dpi,
            ),
        ),
        (
            "05_learned_candidate_scores.png",
            lambda out: plot_five_heatmaps(
                emission_np,
                grids,
                frame_ids_list,
                emission_best_indices,
                out,
                "圖 5：RTL-CRF 單點評分網路校正後的 36 候選分數",
                "分數結合 UAV/SAT embedding、原始分數、機率與候選相對 2D 位置；這一步仍是逐幀候選評分",
                "模型校正後分數",
                args.dpi,
            ),
        ),
        (
            "06_first_order_motion_scores.png",
            lambda out: plot_first(
                first_np,
                centers[0],
                centers[1],
                grids[1],
                first_i0,
                first_i1,
                frame_ids_list[0],
                frame_ids_list[1],
                out,
                args.dpi,
            ),
        ),
        (
            "07_second_order_motion_scores.png",
            lambda out: plot_second(
                second_np[-1],
                centers[-3],
                centers[-2],
                centers[-1],
                grids[-1],
                second_i0,
                second_i1,
                second_i2,
                frame_ids_list[-3],
                frame_ids_list[-2],
                frame_ids_list[-1],
                out,
                args.dpi,
            ),
        ),
        (
            "08_crf_forward_accumulation.png",
            lambda out: plot_dp(
                emission_np,
                alpha_np,
                grids,
                frame_ids_list,
                out,
                args.dpi,
            ),
        ),
        (
            "09_final_posterior_and_fusion.png",
            lambda out: plot_final(
                sat_image,
                pixels[-1],
                grids[-1],
                raw_prob[-1],
                emission_np[-1],
                posterior,
                gt_pixel_np[-1],
                hard_pixels_exact[-1],
                path_pixel_exact,
                final_pixel_exact,
                gate,
                out,
                args.dpi,
            ),
        ),
        (
            "10_all_positions_on_one_map.png",
            lambda out: plot_trajectory(
                sat_image,
                pixels,
                gt_pixel_np,
                prior_pixels,
                hard_pixels_exact,
                path_pixel_exact,
                final_pixel_exact,
                frame_ids_list,
                out,
                args.dpi,
            ),
        ),
    ]

    for filename, function in jobs:
        output = out_dir / filename
        print("writing", output)
        function(output)
        generated.append(output)

    contact_sheet(
        generated,
        out_dir / "11_all_stages_contact_sheet.png",
    )

    np.savez_compressed(
        out_dir / "rtl_crf_intermediate_tensors.npz",
        frame_ids=np.asarray(frame_ids_list),
        dataset_indices=np.asarray(indices),
        gt_xy=gt_xy_np,
        prior_xy=prior_xy_np,
        candidate_indices=npy(candidate.indices).astype(int),
        candidate_centers=centers,
        candidate_pixels=pixels,
        raw_logits=raw_logits,
        raw_probability=raw_prob,
        emission_logits=emission_np,
        first_transition_score=first_np,
        second_transition_score=np.stack(second_np),
        alpha_history=np.stack(alpha_np),
        path_probability=posterior,
        raw_top1_xy=raw_top1,
        hardms_xy=hard,
        path_expectation_xy=path_xy,
        final_xy=final_xy,
        correction_gate=np.asarray([gate]),
        first_example_pair=np.asarray([first_i0, first_i1]),
        second_example_triple=np.asarray(
            [second_i0, second_i1, second_i2]
        ),
    )

    report = {
        "method": "ResidualSecondOrderTemporalLatticeCRF",
        "route": args.route,
        "dataset_indices": indices,
        "frame_ids": frame_ids_list,
        "regular_2d_lattice_by_frame": regular,
        "jitter_m": float(args.jitter_m),
        "tensor_shapes": {
            "raw_logits": list(raw_logits.shape),
            "emission_logits": list(emission_np.shape),
            "first_transition": list(first_np.shape),
            "second_transition_each": [
                list(value.shape) for value in second_np
            ],
            "alpha_each": [list(value.shape) for value in alpha_np],
            "path_probability": list(posterior.shape),
        },
        "interpretation": {
            "time": "五個依序排列的時間點",
            "space_per_time": "每個時間點有36個候選，每個候選保存真實二維座標",
            "flattening": "6x6攤平成0到35只用於tensor索引",
            "official_inference": "log-sum-exp整合全部可能候選路徑",
            "viterbi": "此新版完全不計算也不顯示Viterbi",
            "figure_2": "上半部共用固定地圖座標；下半部各自以本幀候選格重新裁切",
        },
        "result": {
            "hardms_xy": hard[-1].tolist(),
            "path_expectation_xy": path_xy.tolist(),
            "final_xy": final_xy.tolist(),
            "gt_xy": gt_xy_np[-1].tolist(),
            "correction_gate": gate,
            "hardms_error_m": float(
                np.linalg.norm(hard[-1] - gt_xy_np[-1])
            ),
            "path_error_m": float(
                np.linalg.norm(path_xy - gt_xy_np[-1])
            ),
            "final_error_m": float(
                np.linalg.norm(final_xy - gt_xy_np[-1])
            ),
        },
        "generated_images": [str(path) for path in generated],
    }
    with (out_dir / "rtl_crf_stage_report.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("")
    print("完成：", out_dir)
    print("HardMS 誤差：{:.3f} m".format(report["result"]["hardms_error_m"]))
    print(
        "CRF 後驗加權位置誤差：{:.3f} m".format(
            report["result"]["path_error_m"]
        )
    )
    print(
        "最終 RTL-CRF 誤差：{:.3f} m".format(
            report["result"]["final_error_m"]
        )
    )
    print("Correction gate：{:.4f}".format(gate))
    print("新版圖 1～10 已完全移除 Viterbi。")


if __name__ == "__main__":
    main()
