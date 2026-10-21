#!/usr/bin/env python3
"""Exit successfully when a city checkpoint has the shared visual weights/gallery."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def state(payload):
    value = payload.get("model")
    if not isinstance(value, dict):
        raise ValueError("visual checkpoint has no model state")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-checkpoint", type=Path, required=True)
    parser.add_argument("--shared-checkpoint", type=Path, required=True)
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    if not args.city_checkpoint.is_file() or not args.shared_checkpoint.is_file():
        raise SystemExit(1)
    city = torch.load(args.city_checkpoint, map_location="cpu")
    shared = torch.load(args.shared_checkpoint, map_location="cpu")
    if city.get("gallery_city") != args.city:
        raise SystemExit(1)
    city_state, shared_state = state(city), state(shared)
    if city_state.keys() != shared_state.keys():
        raise SystemExit(1)
    if not all(torch.equal(city_state[key], shared_state[key]) for key in city_state):
        raise SystemExit(1)
    print(f"[GALLERY REUSE] {args.city}: learned visual state and city gallery match")


if __name__ == "__main__":
    main()
