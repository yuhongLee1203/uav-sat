#!/usr/bin/env python3
"""Export the four missing FieldAnchorFINAL result figures from real model artifacts.

Inputs are the versioned V36 final checkpoint, its Route-B/C per-frame outputs,
the original UAV frames and the original satellite orthomosaic.  Figure 23
additionally reruns the frozen visual checkpoint for three selected frames.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures_v1"
RUN = ROOT / "outputs" / "v36_v34protocol_compact_gru_softms_mode_variance_forward3x6_polynomial_kalman"
CHECKPOINT = RUN / "checkpoints" / "visual_retrieval_A_only.pt"
SAT = Path("/yh/study/sim_data/sim_competition_crop_check/sim_map_competition_roi_crop.png")
CSV_NAME = "controlled_gtprior_forward3x6_continuous_waypoint_rnn_polynomial_kalman_frames.csv"

INK, MUTED = "#20252B", "#6B7280"
# White is reserved for the GT star because the satellite imagery contains
# dark fields: the earlier black reference marker was not legible.
GT, VISUAL, FINAL, PRIOR = "#F7F7F7", "#D55E00", "#009E73", "#7E57C2"


def rows_for(route: str) -> list[dict[str, float | str]]:
    path = RUN / f"{route}_{CSV_NAME}"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key not in {"protocol", "image_path"}:
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
    return rows


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=350, bbox_inches="tight", pad_inches=.04, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=.04, facecolor="white")
    plt.close(fig)


def turn_indices(rows: list[dict], count: int, separation: int = 24) -> list[int]:
    scores = np.asarray([abs(float(r["gt_turn_rate_deg_per_frame"])) for r in rows])
    order = np.argsort(scores)[::-1]
    chosen: list[int] = []
    for index in order:
        if index < 12 or index >= len(rows) - 12:
            continue
        if scores[index] <= 1e-6:
            break
        if all(abs(int(index) - old) >= separation for old in chosen):
            chosen.append(int(index))
        if len(chosen) == count:
            break
    return sorted(chosen)


def figure21(route_b: list[dict], route_c: list[dict]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 5.3), sharex=False, constrained_layout=True)
    for axis, (route, rows) in zip(axes, (("Route B", route_b), ("Route C", route_c))):
        frame = np.asarray([r["frame_id"] for r in rows])
        gt = np.asarray([[r["gt_x"], r["gt_y"]] for r in rows])
        ms = np.asarray([[r["softms_x"], r["softms_y"]] for r in rows])
        final = np.asarray([[r["final_x"], r["final_y"]] for r in rows])
        ms_error = np.linalg.norm(ms - gt, axis=1)
        final_error = np.linalg.norm(final - gt, axis=1)
        axis.plot(frame, ms_error, color=VISUAL, linewidth=.75, alpha=.78, label="MS visual estimate")
        axis.plot(frame, final_error, color=FINAL, linewidth=.95, label="Kalman estimate (MS + temporal constraint)")
        for index in turn_indices(rows, 4, separation=55):
            axis.axvline(frame[index], color="#CC6677", linestyle="--", linewidth=.65, alpha=.75)
        axis.axhline(5, color="#6B7280", linestyle=":", linewidth=.8, label="5 m threshold")
        ymax = max(np.quantile(np.r_[ms_error, final_error], .995) * 1.12, 20)
        axis.set(title=route, ylabel="Euclidean error (m)", ylim=(0, ymax))
        axis.grid(axis="y", linewidth=.45, color="#D8DDE3")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    axes[1].set_xlabel("Frame index")
    fig.suptitle("Per-frame error trace: MS versus Kalman estimate", fontsize=12, fontweight="bold")
    fig.text(.5, .006, "Dashed guides mark representative reference turns; this diagnostic compares the visual MS estimate with the temporally constrained Kalman estimate.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig21_per_frame_localization_error")


def figure22(route_b: list[dict], route_c: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.7), constrained_layout=True)
    combined = route_b + route_c
    gt = np.asarray([[r["gt_x"], r["gt_y"]] for r in combined])
    ms = np.asarray([[r["softms_x"], r["softms_y"]] for r in combined])
    final = np.asarray([[r["final_x"], r["final_y"]] for r in combined])
    ms_error = np.linalg.norm(ms - gt, axis=1)
    final_error = np.linalg.norm(final - gt, axis=1)
    all_rows = [("MS visual estimate", ms_error, VISUAL), ("Kalman estimate (MS + temporal constraint)", final_error, FINAL)]
    xmax = max(np.quantile(values, .995) for _, values, _ in all_rows)
    xmax = max(15., math.ceil(float(xmax) / 5.) * 5.)
    for label, values, color in all_rows:
        values = np.sort(np.asarray(values, dtype=float))
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=label,
                color=color, linewidth=1.8)
    for threshold in (3, 5, 10, 15):
        if threshold <= xmax:
            ax.axvline(threshold, color="#9AA3AD", linewidth=.7, linestyle="--")
            ax.text(threshold, .025, f"{threshold} m", rotation=90, ha="right", va="bottom", fontsize=7, color=MUTED)
    ax.set(xlim=(0, xmax), ylim=(0, 1.01), xlabel="Localization error (m)", ylabel="Fraction of held-out frames at or below error")
    ax.set_title("Cumulative localization-error distribution (Routes B + C)", fontsize=12, fontweight="bold")
    ax.grid(linewidth=.45, color="#D8DDE3")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig22_error_ecdf")


def figure10(route_b: list[dict], route_c: list[dict], gallery_xy: np.ndarray, gallery_pixel: np.ndarray) -> None:
    """Full route-scale evidence: connected chronological trajectories, no grid markers."""
    Image.MAX_IMAGE_PIXELS = None
    satellite = Image.open(SAT).convert("RGB")
    to_pixel = affine_xy_to_pixel(gallery_xy, gallery_pixel)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.1), constrained_layout=True)
    for axis, (route, rows) in zip(axes, (("Route B", route_b), ("Route C", route_c))):
        reference_xy = np.asarray([[r["gt_x"], r["gt_y"]] for r in rows])
        final_xy = np.asarray([[r["final_x"], r["final_y"]] for r in rows])
        pixels = to_pixel(np.r_[reference_xy, final_xy])
        crop = crop_extent(satellite, pixels, margin=260)
        origin = np.asarray(crop[:2])
        axis.imshow(satellite.crop(crop))
        ref = to_pixel(reference_xy) - origin
        final = to_pixel(final_xy) - origin
        axis.plot(ref[:, 0], ref[:, 1], color=GT, linewidth=1.45, label="reference trajectory", zorder=3)
        axis.plot(final[:, 0], final[:, 1], color=FINAL, linewidth=1.15, label="final KF trajectory", zorder=4)
        axis.plot(ref[0, 0], ref[0, 1], "o", ms=6, color="#4C78A8", markeredgecolor="white", markeredgewidth=.75, label="start", zorder=5)
        axis.plot(ref[-1, 0], ref[-1, 1], ">", ms=7, color="#4C78A8", markeredgecolor="white", markeredgewidth=.75, label="end", zorder=5)
        axis.set_title(route, fontsize=10, fontweight="bold")
        axis.axis("off")
    axes[0].legend(loc="upper right", fontsize=7, framealpha=.88)
    fig.suptitle("Full held-out trajectories: chronological reference versus final KF", fontsize=12, fontweight="bold")
    fig.text(.5, .008, "Lines connect consecutive recorded frames. Start/end markers describe route direction; satellite-grid centres are intentionally not drawn.", ha="center", fontsize=8, color=MUTED)
    save(fig, "fig10_full_trajectory")


def load_visual_responses(cases: list[tuple[str, int]], all_rows: dict[str, list[dict]]):
    """Re-evaluate selected actual frames with the frozen visual checkpoint."""
    os.environ["UAVSAT_OUTPUT_DIR"] = str(RUN)
    os.environ["UAVSAT_BACKBONE"] = "mobileclip2_s2"
    v34 = str(ROOT / "v34")
    if v34 not in sys.path:
        sys.path.insert(0, v34)
    import config  # noqa: PLC0415
    # v34 supplies checkpoint-compatible inference utilities.  Its historical
    # output path is fixed in config.py, so bind those utilities to the final
    # V36 artifact explicitly before importing the localizer.
    config.VISUAL_CHECKPOINT = CHECKPOINT
    from visual_localizer import FrozenVisualLocalizer  # noqa: PLC0415
    from robust_tracker import forward_3x6_candidate_batch  # noqa: PLC0415

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    visual = FrozenVisualLocalizer(device)
    results = []
    for route, index in cases:
        cache = torch.load(RUN / "feature_cache" / f"{route}_uav_clip.pt", map_location="cpu")
        row = all_rows[route][index]
        frame_ids = cache["frame_ids"].numpy().astype(int)
        cache_index = int(np.where(frame_ids == int(row["frame_id"]))[0][0])
        clip = cache["uav_clip"][cache_index:cache_index + 1].float().to(device)
        center = torch.tensor([[row["prior_center_x"], row["prior_center_y"]]], dtype=torch.float32, device=device)
        heading = torch.tensor([math.radians(float(row["search_heading_deg"]))], dtype=torch.float32, device=device)
        batch = forward_3x6_candidate_batch(visual, clip, center, heading)
        pixels = visual.gallery["pixel"][batch.indices[0]].detach().cpu().numpy()
        results.append({
            "route": route, "index": index, "row": row,
            "centers": batch.centers[0].detach().cpu().numpy(),
            "probability": batch.raw_prob[0].detach().cpu().numpy(),
            "softms": batch.softms_xy[0].detach().cpu().numpy(),
            "top1": batch.raw_top1_xy[0].detach().cpu().numpy(),
            "pixels": pixels,
            "gallery_xy": visual.gallery["xy"].detach().cpu().numpy(),
            "gallery_pixel": visual.gallery["pixel"].detach().cpu().numpy(),
        })
    return results


def affine_xy_to_pixel(gallery_xy: np.ndarray, gallery_pixel: np.ndarray):
    design = np.c_[gallery_xy, np.ones(len(gallery_xy))]
    coeff, *_ = np.linalg.lstsq(design, gallery_pixel, rcond=None)
    return lambda xy: np.asarray(xy, dtype=float) @ coeff[:2] + coeff[2]


def crop_extent(image: Image.Image, points: np.ndarray, margin: int = 210):
    x0 = max(0, int(np.floor(points[:, 0].min())) - margin)
    y0 = max(0, int(np.floor(points[:, 1].min())) - margin)
    x1 = min(image.width, int(np.ceil(points[:, 0].max())) + margin)
    y1 = min(image.height, int(np.ceil(points[:, 1].max())) + margin)
    return (x0, y0, x1, y1)


def figure23(responses: list[dict]) -> None:
    Image.MAX_IMAGE_PIXELS = None
    satellite = Image.open(SAT).convert("RGB")
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 10.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.28]})
    labels = ["concentrated response", "diffuse response", "response at a turn"]
    for row_index, (response, label) in enumerate(zip(responses, labels)):
        row = response["row"]
        query = Image.open(str(row["image_path"])).convert("RGB")
        axes[row_index, 0].imshow(query)
        axes[row_index, 0].set_title(f"{label}: {response['route'].replace('_', ' ').title()}, frame {int(row['frame_id'])}", fontsize=9, loc="left", fontweight="bold")
        axes[row_index, 0].axis("off")

        to_pixel = affine_xy_to_pixel(response["gallery_xy"], response["gallery_pixel"])
        candidate_px = response["pixels"]
        # Include GT/MS in the crop as well as all candidates.  Cropping only
        # the grid could leave the reference marker isolated and unexplained.
        gt_absolute = to_pixel([[row["gt_x"], row["gt_y"]]])[0]
        ms_absolute = to_pixel([response["softms"]])[0]
        crop = crop_extent(satellite, np.vstack([candidate_px, gt_absolute, ms_absolute]), margin=150)
        axes[row_index, 1].imshow(satellite.crop(crop))
        local = candidate_px - np.asarray(crop[:2])
        size = 38
        scatter = axes[row_index, 1].scatter(local[:, 0], local[:, 1], c=response["probability"],
                                               cmap="magma", norm=Normalize(0, max(.02, response["probability"].max())),
                                               s=size, marker="s", edgecolors="white", linewidths=.45, zorder=3)
        gt = gt_absolute - np.asarray(crop[:2])
        soft = ms_absolute - np.asarray(crop[:2])
        axes[row_index, 1].plot(gt[0], gt[1], marker="*", ms=15, color=GT, markeredgecolor="#111827", markeredgewidth=1.15, label="GT reference position", zorder=6)
        axes[row_index, 1].plot(soft[0], soft[1], marker="o", ms=9, color=VISUAL, markeredgecolor="white", markeredgewidth=1.0, label="MS visual position", zorder=7)
        axes[row_index, 1].annotate("GT", xy=gt, xytext=(9, 7), textcoords="offset points", color="white", fontsize=7.5, weight="bold", zorder=8)
        axes[row_index, 1].annotate("MS", xy=soft, xytext=(9, -12), textcoords="offset points", color="white", fontsize=7.5, weight="bold", zorder=8)
        axes[row_index, 1].set_title("3×6 candidate positions; colour = visual matching probability", fontsize=8.5, loc="left")
        axes[row_index, 1].axis("off")
        if row_index == 0:
            axes[row_index, 1].legend(frameon=True, fontsize=7, loc="upper right")
            fig.colorbar(scatter, ax=axes[:, 1], fraction=.022, pad=.01, label="candidate probability")
    fig.suptitle("What a single-frame visual match produces", fontsize=12, fontweight="bold")
    fig.text(.5, .006, "Left: UAV query frame. Right: 18 local satellite candidates. White star = GT; orange circle = MS calculated from candidate probabilities.", ha="center", fontsize=8, color=MUTED)
    save(fig, "fig23_visual_response_cases")


def figure24(route_b: list[dict], route_c: list[dict], gallery_xy: np.ndarray, gallery_pixel: np.ndarray) -> None:
    Image.MAX_IMAGE_PIXELS = None
    satellite = Image.open(SAT).convert("RGB")
    to_pixel = affine_xy_to_pixel(gallery_xy, gallery_pixel)
    selection = [("Route B", route_b, i) for i in turn_indices(route_b, 2, 50)]
    selection += [("Route C", route_c, i) for i in turn_indices(route_c, 2, 50)]
    selection = selection[:4]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.2), constrained_layout=True)
    for axis, (route, rows, index) in zip(axes.flat, selection):
        lo, hi = max(0, index - 60), min(len(rows), index + 61)
        segment = rows[lo:hi]
        gt_xy = np.array([[r["gt_x"], r["gt_y"]] for r in segment])
        visual_xy = np.array([[r["softms_x"], r["softms_y"]] for r in segment])
        final_xy = np.array([[r["final_x"], r["final_y"]] for r in segment])
        points = to_pixel(np.r_[gt_xy, visual_xy, final_xy])
        crop = crop_extent(satellite, points, margin=180)
        origin = np.asarray(crop[:2])
        axis.imshow(satellite.crop(crop))
        for values, colour, text, width in ((gt_xy, GT, "reference", 2.15), (visual_xy, VISUAL, "MS visual", 1.2), (final_xy, FINAL, "Kalman estimate", 1.65)):
            pix = to_pixel(values) - origin
            axis.plot(pix[:, 0], pix[:, 1], color=colour, linewidth=width, label=text, alpha=.95)
        ref_pix = to_pixel(gt_xy) - origin
        axis.scatter(ref_pix[::10, 0], ref_pix[::10, 1], s=10, color="#B7BEC7", edgecolor="white", linewidth=.25, zorder=4, label="reference samples")
        midpoint = min(len(ref_pix) - 8, max(8, index - lo))
        axis.annotate("", xy=ref_pix[midpoint + 7], xytext=ref_pix[midpoint],
                      arrowprops=dict(arrowstyle="->", color=GT, lw=1.3), zorder=6)
        axis.plot(ref_pix[0, 0], ref_pix[0, 1], "o", ms=5.5, color="#4C78A8", markeredgecolor="white", markeredgewidth=.6, zorder=7, label="segment start")
        current = to_pixel([[rows[index]["gt_x"], rows[index]["gt_y"]]])[0] - origin
        axis.plot(current[0], current[1], "*", ms=11, color="white", markeredgecolor=GT, markeredgewidth=1.0, label="turn frame")
        axis.plot(ref_pix[-1, 0], ref_pix[-1, 1], ">", ms=6, color="#4C78A8", markeredgecolor="white", markeredgewidth=.6, zorder=7, label="segment end")
        axis.set_title(f"{route}, frame {int(rows[index]['frame_id'])} (turn rate {rows[index]['gt_turn_rate_deg_per_frame']:.1f}°/frame)", fontsize=8.5, loc="left", fontweight="bold")
        axis.axis("off")
    axes.flat[0].legend(loc="upper right", fontsize=7, framealpha=.88)
    fig.suptitle("Before–turn–after localization continuity", fontsize=12, fontweight="bold")
    fig.text(.5, .006, "Dots sample the reference trajectory every 10 frames; the arrow indicates motion direction. All traces come from the final Route-B/C model output.", ha="center", fontsize=8, color=MUTED)
    save(fig, "fig24_turn_closeups")


def choose_cases(all_rows: dict[str, list[dict]]) -> list[tuple[str, int]]:
    b, c = all_rows["route_B"], all_rows["route_C"]
    # Fig. 23 is evidence for local visual matching.  Each shown frame must
    # therefore have GT inside the candidate bank; otherwise it is actually a
    # capture-range failure example and answers a different question.
    def captured(rows):
        return [i for i in range(15, len(rows) - 15) if float(rows[i]["candidate_capture"]) > .5]

    def visual_error(row):
        return math.hypot(float(row["softms_x"]) - float(row["gt_x"]), float(row["softms_y"]) - float(row["gt_y"]))

    easy = min(captured(b), key=lambda i: visual_error(b[i]))
    ambiguous = max(captured(c), key=lambda i: float(c[i]["visual_entropy"]))
    near_turn = max(captured(b), key=lambda i: abs(float(b[i]["gt_turn_rate_deg_per_frame"])))
    return [("route_B", easy), ("route_C", ambiguous), ("route_B", near_turn)]


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    rows = {route: rows_for(route) for route in ("route_B", "route_C")}
    figure21(rows["route_B"], rows["route_C"])
    figure22(rows["route_B"], rows["route_C"])
    cases = choose_cases(rows)
    print("rerunning frozen visual model for", [(r, int(rows[r][i]["frame_id"])) for r, i in cases], flush=True)
    responses = load_visual_responses(cases, rows)
    figure10(rows["route_B"], rows["route_C"], responses[0]["gallery_xy"], responses[0]["gallery_pixel"])
    figure23(responses)
    figure24(rows["route_B"], rows["route_C"], responses[0]["gallery_xy"], responses[0]["gallery_pixel"])
    print(f"wrote FieldAnchorFINAL result figures to {OUT}", flush=True)


if __name__ == "__main__":
    main()
