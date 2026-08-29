"""Train/evaluate v36_byTeacher using only native-speed Route A.

Training:
  * Route-A at the recorded/original temporal rate only.
Validation:
  * Route-C for checkpoint selection / early stopping.
Test:
  * Route-B only after checkpoint selection.

This entry point also installs the discrete-lattice MS1 forward-selector fix.
The 6x6 candidate window is generated around the nearest satellite lattice
anchor, while the autonomous predicted center is continuous. Forward rows must
therefore be defined relative to that lattice anchor, not the continuous point.
"""

import argparse

import torch

import config
import robust_tracker as rt
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only


def _lattice_forward_3x6(full_centers, center_xy, heading_rad):
    """Return exactly the center-adjacent forward 18 lattice candidates.

    The base 6x6 grid is axis-aligned. We first identify the actual satellite
    lattice anchor nearest the continuous autonomous center. The dominant
    heading axis then chooses the longitudinal grid direction. Candidates are
    ranked from the anchor plane forward, so cardinal cases are exactly:

        east/north -> 0, +1, +2 rows
        west/south -> 0, -1, -2 rows

    If the gallery is at an image boundary and the regular grid fallback is not
    perfectly rectangular, the nearest candidates behind the anchor plane are
    used only to fill the fixed 18-candidate tensor instead of crashing.
    """

    batch = int(full_centers.shape[0])
    headings = torch.as_tensor(
        heading_rad,
        dtype=full_centers.dtype,
        device=full_centers.device,
    ).reshape(-1)
    if headings.numel() == 1 and batch > 1:
        headings = headings.expand(batch)
    if headings.numel() != batch:
        raise ValueError("heading count must match center batch size")

    # regular_grid_indices() itself snaps the continuous search center to the
    # nearest satellite lattice anchor. Recover that same anchor from the 6x6
    # candidate tensor and measure the forward half-grid from it.
    distance2 = (full_centers - center_xy[:, None, :]).square().sum(dim=2)
    anchor_local = distance2.argmin(dim=1)
    anchor_xy = full_centers[
        torch.arange(batch, device=full_centers.device), anchor_local
    ]
    relative = full_centers - anchor_xy[:, None, :]

    cos_h = torch.cos(headings)
    sin_h = torch.sin(headings)
    use_x = cos_h.abs() >= sin_h.abs()
    sign_x = torch.where(
        cos_h >= 0,
        torch.ones_like(cos_h),
        -torch.ones_like(cos_h),
    )
    sign_y = torch.where(
        sin_h >= 0,
        torch.ones_like(sin_h),
        -torch.ones_like(sin_h),
    )

    primary = torch.where(
        use_x[:, None],
        relative[:, :, 0] * sign_x[:, None],
        relative[:, :, 1] * sign_y[:, None],
    )
    secondary = torch.where(
        use_x[:, None],
        relative[:, :, 1],
        relative[:, :, 0],
    )

    # Prefer candidates on/in front of the anchor plane, ordered by distance
    # from that plane. Behind-plane candidates get a large penalty and are only
    # possible as a map-boundary fallback. This always returns exactly 18.
    behind = primary < -1e-4
    forward_cost = primary.abs() + behind.to(primary.dtype) * 1.0e6
    selected_local = torch.topk(
        forward_cost,
        k=int(config.MS1_CANDIDATE_COUNT),
        dim=1,
        largest=False,
        sorted=False,
    ).indices

    chosen_primary = torch.gather(primary, 1, selected_local)
    chosen_secondary = torch.gather(secondary, 1, selected_local)
    ordering_key = chosen_primary * 1000.0 + chosen_secondary
    order = torch.argsort(ordering_key, dim=1)
    return torch.gather(selected_local, 1, order)


# forward_3x6_candidate_batch resolves this module-global function from the
# robust_tracker module at call time. Install the corrected lattice selector
# before any train/eval rollout starts.
rt._nearest_forward_3x6 = _lattice_forward_3x6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    parser.add_argument(
        "--temporal-epochs", type=int, default=int(config.TEMPORAL_EPOCHS)
    )
    parser.add_argument(
        "--patience", type=int, default=int(config.EARLY_STOP_PATIENCE)
    )
    parser.add_argument(
        "--jitter-m", type=float, default=float(config.LOCAL_PRIOR_JITTER_M)
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--train-visual-if-missing",
        action="store_true",
        help="train the Route-A-only visual retrieval model if missing",
    )
    parser.add_argument("--measure-latency", action="store_true")
    return parser.parse_args()


def ensure_visual(device, args):
    if not config.VISUAL_CHECKPOINT.exists():
        if not args.train_visual_if_missing:
            raise FileNotFoundError(
                f"visual checkpoint not found: {config.VISUAL_CHECKPOINT}; "
                "pass --train-visual-if-missing to train it"
            )
        train_visual_retrieval_a_only(
            device=device,
            epochs=int(config.VISUAL_EPOCHS),
            jitter_m=float(args.jitter_m),
            resume=False,
        )
    return FrozenVisualLocalizer(device)


def train_native_a(visual, device, args):
    route_a_cache = rt.build_route_cache(
        "route_A", config.ROUTE_ROOTS[0], visual, device
    )
    c_index = config.ROUTE_NAMES.index("route_C")
    route_c_cache = rt.build_route_cache(
        "route_C", config.ROUTE_ROOTS[c_index], visual, device
    )
    return rt.train_temporal_route_a(
        visual=visual,
        route_a_cache=route_a_cache,
        route_c_cache=route_c_cache,
        device=device,
        epochs=int(args.temporal_epochs),
        patience_limit=int(args.patience),
        resume=bool(args.resume),
    )


def evaluate_c_and_b(visual, device, args):
    model = rt.load_temporal_model(device)
    for route_name in ("route_C", "route_B"):
        route_index = config.ROUTE_NAMES.index(route_name)
        cache = rt.build_route_cache(
            route_name, config.ROUTE_ROOTS[route_index], visual, device
        )
        summary = rt.run_route_inference(
            route_name,
            visual,
            model,
            cache,
            device,
            save_csv=True,
            measure_latency=bool(args.measure_latency),
        )
        role = "validation" if route_name == "route_C" else "test"
        print(
            f"{route_name} ({role}): "
            f"prior={summary['Prior_MLE_m']:.3f}m "
            f"MS1={summary['MS1_MLE_m']:.3f}m "
            f"KF={summary['Kalman_MLE_m']:.3f}m "
            f"MS2/final={summary['MLE_m']:.3f}m "
            f"P90={summary['P90_m']:.3f}m "
            f"LSR@15={summary['LSR@15_pct']:.2f}%",
            flush=True,
        )


def main():
    args = parse_args()
    rt.set_seed(config.SEED)
    device = rt.resolve_device()
    visual = ensure_visual(device, args)

    print("=" * 96, flush=True)
    print("Temporal training: Route-A ORIGINAL SPEED ONLY", flush=True)
    print("Route-C validation; Route-B final untouched test", flush=True)
    print("MS1 lattice-forward selector: FIXED 18 candidates", flush=True)
    print("=" * 96, flush=True)

    if args.mode in ("train", "all"):
        _, best = train_native_a(visual, device, args)
        print(f"best Route-C validation MLE={best:.3f}m", flush=True)

    if args.mode in ("eval", "all"):
        evaluate_c_and_b(visual, device, args)


if __name__ == "__main__":
    main()
