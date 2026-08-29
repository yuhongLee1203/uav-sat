"""Train/evaluate the autonomous temporal model with native-speed Route-A only.

Training:   Route-A at the original recorded frame rate and speed only.
Validation: Route-C for checkpoint selection / early stopping.
Test:       Route-B only after checkpoint selection.

No stride-2 / 2x-speed Route-A sequence is used in this entry point.
"""

import argparse
import json

import torch

import config
import robust_tracker as rt
from visual_localizer import FrozenVisualLocalizer, train_visual_retrieval_a_only

TRAINING_MODE = "route_A_native_only"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["train", "eval", "all"], default="all"
    )
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
        help="train the Route-A-only visual retrieval model if its checkpoint is absent",
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


def train_native_route_a(visual, route_a_cache, route_c_cache, device, args):
    model = rt.ThreeFrameRouteStateGRU().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config.TEMPORAL_LR),
        weight_decay=float(config.TEMPORAL_WEIGHT_DECAY),
    )

    start_epoch = 1
    best_score = float("inf")
    best_state = None
    patience = 0

    if args.resume and config.LATEST_TEMPORAL_CHECKPOINT.exists():
        payload = torch.load(config.LATEST_TEMPORAL_CHECKPOINT, map_location="cpu")
        if payload.get("architecture") != rt.ARCHITECTURE_NAME:
            raise RuntimeError(
                "resume architecture mismatch: %r" % payload.get("architecture")
            )
        if payload.get("training_mode") != TRAINING_MODE:
            raise RuntimeError(
                "refusing to resume an older multirate/stride checkpoint; "
                "run without --resume for native-only Route-A training"
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload.get("best_score", best_score))
        best_state = payload.get("best_model")
        patience = int(payload.get("patience", 0))

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "Temporal training: Route-A native speed ONLY; "
        "validation=Route-C; Route-B is reserved for final test",
        flush=True,
    )

    for epoch in range(start_epoch, int(args.temporal_epochs) + 1):
        model.train()
        native_loss = rt._training_sequence_loss(
            model, optimizer, visual, route_a_cache, "route_A", device
        )

        val = rt.evaluate_route("route_C", visual, model, route_c_cache, device)
        score = float(val["MLE_m"])
        improved = score < best_score - float(config.EARLY_STOP_MIN_DELTA)

        if improved:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
            torch.save(
                {
                    "architecture": rt.ARCHITECTURE_NAME,
                    "training_mode": TRAINING_MODE,
                    "model": best_state,
                    "epoch": epoch,
                    "validation_route": "route_C",
                    "validation": val,
                    "training_routes": ["route_A"],
                    "reference_usage": (
                        "training targets and post-prediction metrics only; "
                        "no reference-centered runtime search or motion caps"
                    ),
                },
                config.TEMPORAL_CHECKPOINT,
            )
        else:
            patience += 1

        torch.save(
            {
                "architecture": rt.ARCHITECTURE_NAME,
                "training_mode": TRAINING_MODE,
                "model": model.state_dict(),
                "best_model": best_state,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "patience": patience,
            },
            config.LATEST_TEMPORAL_CHECKPOINT,
        )

        print(
            f"epoch={epoch:03d}/{args.temporal_epochs} "
            f"A_loss={native_loss:.5f} "
            f"C_val_MLE={val['MLE_m']:.3f}m "
            f"C_val_P90={val['P90_m']:.3f}m "
            f"best={best_score:.3f} "
            f"patience={patience}/{args.patience}",
            flush=True,
        )

        if (
            epoch >= int(config.EARLY_STOP_MIN_EPOCH)
            and patience >= int(args.patience)
        ):
            print("early stopping", flush=True)
            break

    if best_state is None:
        raise RuntimeError("native Route-A training did not produce a checkpoint")

    model.load_state_dict(best_state)
    return model, best_score


def evaluate_c_and_b(visual, device, measure_latency=False):
    model = rt.load_temporal_model(device)
    results = {
        "architecture": rt.ARCHITECTURE_NAME,
        "training_mode": TRAINING_MODE,
        "train": ["route_A"],
        "validation": "route_C",
        "test": "route_B",
        "results": {},
    }

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
            measure_latency=bool(measure_latency),
        )
        results["results"][route_name] = summary
        role = "validation" if route_name == "route_C" else "test"
        print(
            f"{route_name} ({role}): MLE={summary['MLE_m']:.3f}m "
            f"P90={summary['P90_m']:.3f}m "
            f"LSR@15={summary['LSR@15_pct']:.2f}%",
            flush=True,
        )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = config.OUTPUT_DIR / "autonomous_ms1_kf_gru_ms2_summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"summary: {summary_path}", flush=True)


def main():
    args = parse_args()
    rt.set_seed(config.SEED)
    device = rt.resolve_device()

    print("=" * 96, flush=True)
    print("Route-A native-speed temporal training (NO stride-2 / 2x-speed data)", flush=True)
    print("Route-C validation; Route-B final test", flush=True)
    print("=" * 96, flush=True)

    visual = ensure_visual(device, args)

    if args.mode in ("train", "all"):
        route_a_cache = rt.build_route_cache(
            "route_A", config.ROUTE_ROOTS[0], visual, device
        )
        c_index = config.ROUTE_NAMES.index("route_C")
        route_c_cache = rt.build_route_cache(
            "route_C", config.ROUTE_ROOTS[c_index], visual, device
        )
        _, best = train_native_route_a(
            visual, route_a_cache, route_c_cache, device, args
        )
        print(f"best Route-C validation MLE={best:.3f}m", flush=True)

    if args.mode in ("eval", "all"):
        evaluate_c_and_b(
            visual, device, measure_latency=bool(args.measure_latency)
        )


if __name__ == "__main__":
    main()
