"""Train/evaluate v36_byTeacher using only native-speed Route A.

Training:
  * Route-A at the recorded/original temporal rate only.
Validation:
  * Route-C for checkpoint selection / early stopping.
Test:
  * Route-B only after checkpoint selection.

MS1 and MS2 have deliberately different spatial roles:
  * MS1 searches ONLY the forward half (3x6=18) of the predicted 6x6 window.
  * Kalman fuses the MS1 measurement with its prior state.
  * MS2 then opens a NEW full 6x6 window CENTERED on the Kalman result.

Temporal v6:
  * GRU still receives current MS1 and temporal visual/state evidence.
  * After bootstrap, the motion baseline is previous polynomial Delta, not the
    forward-only MS1 displacement. MS1 therefore provides correction evidence
    without redefining the UAV speed every frame.
"""

import argparse

import torch

import config
import robust_tracker as rt
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only


def _strict_forward_half_3x6(full_centers, center_xy, heading_rad):
    """Select the true forward HALF of the regular 6x6 lattice.

    The regular 6x6 window has six discrete rows/columns. Because six is even,
    its geometric center lies BETWEEN the two middle lattice rows/columns. We
    therefore split the lattice into an exact rear 3x6 half and forward 3x6
    half instead of treating the nearest lattice anchor as the center row.

    The dominant component of heading selects which map axis is longitudinal:
      east  -> right  three columns
      west  -> left   three columns
      north -> upper/positive three rows in metric Y
      south -> lower/negative three rows in metric Y

    This always returns exactly 18 candidates. The continuous predicted center
    is used to choose the base 6x6 window, but it does NOT turn MS1 into a
    centered search. MS2 is the only centered full-6x6 search after Kalman.
    """

    del center_xy

    batch = int(full_centers.shape[0])
    keep = int(config.MS1_CANDIDATE_COUNT)
    if int(full_centers.shape[1]) != 36 or keep != 18:
        raise RuntimeError(
            "strict MS1 requires a 6x6 base window and exactly 18 forward candidates"
        )

    headings = torch.as_tensor(
        heading_rad,
        dtype=full_centers.dtype,
        device=full_centers.device,
    ).reshape(-1)
    if headings.numel() == 1 and batch > 1:
        headings = headings.expand(batch)
    if headings.numel() != batch:
        raise ValueError("heading count must match center batch size")

    geometric_center = full_centers.mean(dim=1, keepdim=True)
    relative = full_centers - geometric_center

    cos_h = torch.cos(headings)
    sin_h = torch.sin(headings)
    use_x = cos_h.abs() >= sin_h.abs()
    longitudinal_sign = torch.where(
        use_x,
        torch.where(cos_h >= 0, torch.ones_like(cos_h), -torch.ones_like(cos_h)),
        torch.where(sin_h >= 0, torch.ones_like(sin_h), -torch.ones_like(sin_h)),
    )

    longitudinal = torch.where(
        use_x[:, None],
        relative[:, :, 0],
        relative[:, :, 1],
    ) * longitudinal_sign[:, None]

    selected_local = torch.topk(
        longitudinal,
        k=keep,
        dim=1,
        largest=True,
        sorted=False,
    ).indices

    chosen_longitudinal = torch.gather(longitudinal, 1, selected_local)
    lateral = torch.where(
        use_x[:, None],
        relative[:, :, 1],
        relative[:, :, 0],
    )
    chosen_lateral = torch.gather(lateral, 1, selected_local)
    ordering_key = chosen_longitudinal * 1000.0 + chosen_lateral
    order = torch.argsort(ordering_key, dim=1)
    return torch.gather(selected_local, 1, order)


rt._nearest_forward_3x6 = _strict_forward_half_3x6


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
    print("MS1 = STRICT FORWARD HALF 3x6 from predicted position", flush=True)
    print("Kalman = fuse prior state + MS1 measurement", flush=True)
    print("MS2 = NEW CENTERED FULL 6x6 around Kalman output", flush=True)
    print("GRU v6 = bootstrap from MS1 once, then previous-Delta motion baseline", flush=True)
    print("=" * 96, flush=True)

    if args.mode in ("train", "all"):
        _, best = train_native_a(visual, device, args)
        print(f"best Route-C validation MLE={best:.3f}m", flush=True)

    if args.mode in ("eval", "all"):
        evaluate_c_and_b(visual, device, args)


if __name__ == "__main__":
    main()
