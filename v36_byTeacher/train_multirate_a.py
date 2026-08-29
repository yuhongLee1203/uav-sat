"""Train/evaluate v36_byTeacher using only native-speed Route A.

Training:
  * Route-A at the recorded/original temporal rate only.
Validation:
  * Route-C for checkpoint selection / early stopping.
Test:
  * Route-B only after checkpoint selection.

Spatial roles are fixed and must not be mixed:
  * MS1 searches ONLY the strict forward half (3x6=18) of the predicted 6x6.
  * Kalman fuses the MS1 position measurement with its prior state.
  * MS2 opens a NEW full 6x6 CENTERED on the Kalman result.

This entry point also adds validation-only oracle diagnostics.  Reference
positions are read only AFTER each complete frame prediction, and are used only
to answer whether the correct position was geometrically present in the MS1/MS2
candidate support.  They never alter a search center or estimator state.
"""

import argparse

import numpy as np
import torch

import config
import robust_tracker as rt
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only


def _strict_forward_half_3x6(full_centers, center_xy, heading_rad):
    """Select the true forward HALF of the regular 6x6 lattice.

    Six is even, so the geometric center lies between the two middle lattice
    rows/columns.  The dominant heading axis therefore splits the predicted
    6x6 into an exact rear 3x6 half and forward 3x6 half:

      east  -> right  three columns
      west  -> left   three columns
      north -> positive-Y three rows
      south -> negative-Y three rows

    MS1 is intentionally NOT a centered search.  MS2 is the only full centered
    6x6 search, and it is created after Kalman fusion.
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


# forward_3x6_candidate_batch resolves this module-global function at call time.
rt._nearest_forward_3x6 = _strict_forward_half_3x6


def _mean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)) if values.size else float("inf")


@torch.no_grad()
def _evaluate_route_with_candidate_diagnostics(route_name, visual, model, cache, device):
    """Validation inference plus post-prediction candidate-support diagnostics."""

    model.eval()
    state = rt._initial_temporal_state(route_name, visual, device)

    final_errors = []
    prior_errors = []
    previous_final_errors = []
    ms1_errors = []
    kalman_errors = []
    ms1_oracle = []
    ms2_oracle = []

    first_over_20 = None
    first_over_50 = None
    first_ms1_miss = None
    first_ms2_miss = None
    capture_radius = float(getattr(config, "CANDIDATE_CAPTURE_RADIUS_M", 7.5))

    for index in range(len(cache)):
        uav_clip = cache.uav_clip[index : index + 1].to(device).float()
        frame = rt._forward_frame(model, visual, uav_clip, state, device)

        # Reference is deliberately read only after the complete frame output
        # (MS1 -> Kalman/GRU -> MS2 -> final) already exists.
        reference_xy = cache.gt_xy[index].cpu().numpy().astype(np.float64)
        final_xy = frame["final_xy"]
        ms1_xy = frame["ms1_xy"][0].detach().cpu().numpy().astype(np.float64)

        prior_error = float(np.linalg.norm(frame["prior_xy"] - reference_xy))
        previous_final_error = float(
            np.linalg.norm(frame["previous_final_xy"] - reference_xy)
        )
        ms1_error = float(np.linalg.norm(ms1_xy - reference_xy))
        kalman_error = float(np.linalg.norm(frame["kalman_xy"] - reference_xy))
        final_error = float(np.linalg.norm(final_xy - reference_xy))

        ms1_centers = frame["ms1"].centers[0].detach().cpu().numpy().astype(np.float64)
        ms2_centers = frame["ms2"].centers[0].detach().cpu().numpy().astype(np.float64)
        ms1_min = float(np.linalg.norm(ms1_centers - reference_xy[None, :], axis=1).min())
        ms2_min = float(np.linalg.norm(ms2_centers - reference_xy[None, :], axis=1).min())

        previous_final_errors.append(previous_final_error)
        prior_errors.append(prior_error)
        ms1_errors.append(ms1_error)
        kalman_errors.append(kalman_error)
        final_errors.append(final_error)
        ms1_oracle.append(ms1_min)
        ms2_oracle.append(ms2_min)

        if first_over_20 is None and final_error > 20.0:
            first_over_20 = int(index)
        if first_over_50 is None and final_error > 50.0:
            first_over_50 = int(index)
        if first_ms1_miss is None and ms1_min > capture_radius:
            first_ms1_miss = int(index)
        if first_ms2_miss is None and ms2_min > capture_radius:
            first_ms2_miss = int(index)

        rt._advance_state(state, frame)
        rt._detach_state(state)

    summary = rt.metric_summary(final_errors)
    ms1_oracle_np = np.asarray(ms1_oracle, dtype=np.float64)
    ms2_oracle_np = np.asarray(ms2_oracle, dtype=np.float64)
    summary.update(
        {
            "Architecture": rt.ARCHITECTURE_NAME,
            "Route": route_name,
            "PreviousFinalToCurrentRef_MLE_m": _mean(previous_final_errors),
            "Prior_MLE_m": _mean(prior_errors),
            "MS1_MLE_m": _mean(ms1_errors),
            "Kalman_MLE_m": _mean(kalman_errors),
            "MS2_Final_MLE_m": _mean(final_errors),
            "FirstFinalErrorOver20mFrame": first_over_20,
            "FirstFinalErrorOver50mFrame": first_over_50,
            "MS1_OracleMin_MLE_m": _mean(ms1_oracle),
            "MS2_OracleMin_MLE_m": _mean(ms2_oracle),
            "MS1_CandidateCapture_pct": float(np.mean(ms1_oracle_np <= capture_radius) * 100.0),
            "MS2_CandidateCapture_pct": float(np.mean(ms2_oracle_np <= capture_radius) * 100.0),
            "FirstMS1CandidateMissFrame": first_ms1_miss,
            "FirstMS2CandidateMissFrame": first_ms2_miss,
            "CandidateCaptureRadius_m": capture_radius,
        }
    )

    print(
        "Cdiag: "
        f"MS1oracle={summary['MS1_OracleMin_MLE_m']:.2f}m "
        f"MS1cap={summary['MS1_CandidateCapture_pct']:.1f}% "
        f"MS1miss={summary['FirstMS1CandidateMissFrame']} | "
        f"MS2oracle={summary['MS2_OracleMin_MLE_m']:.2f}m "
        f"MS2cap={summary['MS2_CandidateCapture_pct']:.1f}% "
        f"MS2miss={summary['FirstMS2CandidateMissFrame']} | "
        f"final20={summary['FirstFinalErrorOver20mFrame']}",
        flush=True,
    )
    return summary


# train_temporal_route_a() looks up evaluate_route in the robust_tracker module
# at runtime. Replace only validation evaluation; estimator/search logic remains
# unchanged.
rt.evaluate_route = _evaluate_route_with_candidate_diagnostics


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
    print("Cdiag = post-prediction candidate oracle/capture diagnostics only", flush=True)
    print("=" * 96, flush=True)

    if args.mode in ("train", "all"):
        _, best = train_native_a(visual, device, args)
        print(f"best Route-C validation MLE={best:.3f}m", flush=True)

    if args.mode in ("eval", "all"):
        evaluate_c_and_b(visual, device, args)


if __name__ == "__main__":
    main()
