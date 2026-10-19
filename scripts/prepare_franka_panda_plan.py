#!/usr/bin/env python3
"""Compose a live camera pose and object-frame grasp into stock Panda Hand targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from demo2gripper.franka import (
    make_tcp_targets,
    panda_hand_targets,
    validate_rigid_transform,
)


def _load_camera_object(path: Path, frame_index: int) -> tuple[np.ndarray, int]:
    poses = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if poses.shape == (4, 4):
        if frame_index not in (-1, 0):
            raise ValueError("a single T_camera_object pose only supports frame 0 or -1")
        return validate_rigid_transform(poses, name="T_camera_object"), 0
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError(
            "T_camera_object must have shape (4,4) or (F,4,4), "
            f"got {poses.shape}"
        )
    try:
        selected = poses[frame_index]
    except IndexError as error:
        raise ValueError(
            f"frame index {frame_index} is outside {len(poses)} poses"
        ) from error
    return validate_rigid_transform(selected, name="T_camera_object"), frame_index % len(poses)


def _matrices_to_lists(values: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
    return {name: transform.tolist() for name, transform in values.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--grasp-json", type=Path, required=True)
    parser.add_argument(
        "--t-base-camera",
        type=Path,
        required=True,
        help="Measured camera-to-panda_link0 transform as a 4x4 .npy",
    )
    parser.add_argument(
        "--t-camera-object",
        type=Path,
        required=True,
        help="FoundationPose output from the live robot camera",
    )
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--approach-distance-m", type=float)
    parser.add_argument("--lift-height-m", type=float, default=0.03)
    parser.add_argument("--tcp-offset-m", type=float, default=0.1034)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grasp = json.loads(args.grasp_json.read_text(encoding="utf-8"))
    object_tcp = validate_rigid_transform(
        np.asarray(grasp["T_object_gripper_tcp"], dtype=float),
        name="T_object_gripper_tcp",
    )
    base_camera = validate_rigid_transform(
        np.load(args.t_base_camera, allow_pickle=False), name="T_base_camera"
    )
    camera_object, selected_frame = _load_camera_object(
        args.t_camera_object, args.frame_index
    )
    base_object = base_camera @ camera_object
    approach_distance_m = (
        float(grasp["approach_distance_m"])
        if args.approach_distance_m is None
        else args.approach_distance_m
    )
    tcp_targets = make_tcp_targets(
        base_object,
        object_tcp,
        approach_distance_m=approach_distance_m,
        lift_height_m=args.lift_height_m,
    )
    hand_targets = panda_hand_targets(
        tcp_targets, tcp_offset_m=args.tcp_offset_m
    )

    payload = {
        "schema_version": 1,
        "robot_profile": "Franka Panda with stock Franka Hand",
        "base_frame": "panda_link0",
        "command_frame": "panda_hand",
        "frame_convention": "T_parent_child, column vectors, metres",
        "selected_camera_frame": selected_frame,
        "candidate_index": grasp.get("candidate_index"),
        "T_base_camera": base_camera.tolist(),
        "T_camera_object": camera_object.tolist(),
        "T_base_object": base_object.tolist(),
        "T_object_gripper_tcp": object_tcp.tolist(),
        "canonical_tcp_targets": _matrices_to_lists(tcp_targets),
        "panda_hand_targets": _matrices_to_lists(hand_targets),
        "approach_distance_m": approach_distance_m,
        "lift_height_m": args.lift_height_m,
        "tcp_offset_m": args.tcp_offset_m,
        "pregrasp_width_m": grasp.get("pregrasp_width_m"),
        "demonstrated_contact_width_m": grasp.get("demonstrated_contact_width_m"),
        "execution_order": ["open", "pregrasp", "grasp", "force_grasp", "lift", "hold"],
        "provenance": {
            "grasp_json": str(args.grasp_json.resolve()),
            "T_base_camera": str(args.t_base_camera.resolve()),
            "T_camera_object": str(args.t_camera_object.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"candidate: {payload['candidate_index']}")
    for name, transform in tcp_targets.items():
        xyz = transform[:3, 3]
        print(f"{name:8s} TCP xyz: {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
