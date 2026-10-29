#!/usr/bin/env python3
"""Adapt raw human object-relative intent to a tabletop Hand-E reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from demo2gripper.hande_embodiment import (
    HandEEmbodimentConfig,
    adapt_human_intent_to_hande,
    adapted_intent_payload,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-intent", type=Path, required=True)
    parser.add_argument("--pickup-json", type=Path, required=True)
    parser.add_argument("--hand-joints", type=Path, required=True)
    parser.add_argument("--t-base-camera", type=Path, required=True)
    parser.add_argument(
        "--support-normal-base",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("NX", "NY", "NZ"),
    )
    parser.add_argument("--min-support-frames", type=int, default=5)
    parser.add_argument("--min-palm-planar-component", type=float, default=0.25)
    parser.add_argument("--max-palm-planar-angle-p95-deg", type=float, default=10.0)
    parser.add_argument("--max-grasp-offset-m", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "raw_intent": args.raw_intent.expanduser().resolve(),
        "pickup_json": args.pickup_json.expanduser().resolve(),
        "hand_joints": args.hand_joints.expanduser().resolve(),
        "T_base_camera": args.t_base_camera.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{name} does not exist: {path}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite existing output: {output}")

    raw_intent = _read_json(paths["raw_intent"])
    pickup = _read_json(paths["pickup_json"])
    joints = np.load(paths["hand_joints"], allow_pickle=False)
    T_base_camera = np.load(paths["T_base_camera"], allow_pickle=False)
    config = HandEEmbodimentConfig(
        min_support_frames=args.min_support_frames,
        min_palm_planar_component=args.min_palm_planar_component,
        max_palm_planar_angle_p95_deg=args.max_palm_planar_angle_p95_deg,
        max_grasp_offset_m=args.max_grasp_offset_m,
    )
    result = adapt_human_intent_to_hande(
        raw_intent=raw_intent,
        pickup=pickup,
        hand_joints_camera_m=joints,
        T_base_camera=T_base_camera,
        support_normal_base=np.asarray(args.support_normal_base, dtype=float),
        config=config,
    )
    provenance = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    payload = adapted_intent_payload(
        result,
        raw_intent=raw_intent,
        provenance=provenance,
        config=config,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"robot-adapted Hand-E intent: {output}")
    print(
        "critical frames "
        f"{result.critical_grasp_interval[0]}:{result.critical_grasp_interval[1]}, "
        f"palm planar p95 {result.palm_planar_angle_p95_deg:.2f} deg"
    )
    print(f"closing axis base: {result.adapted_closing_axis_base.tolist()}")
    print(f"approach axis base: {result.adapted_approach_axis_base.tolist()}")


if __name__ == "__main__":
    main()
