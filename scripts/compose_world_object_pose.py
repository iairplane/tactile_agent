#!/usr/bin/env python3
"""Compose a known-object pose estimate with camera extrinsics for Isaac Lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _load_transform(path: Path, name: str) -> np.ndarray:
    transform = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}")
    return transform


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t-camera-object",
        type=Path,
        required=True,
        help="Object-to-camera estimate shaped (F,4,4) or one (4,4) transform",
    )
    parser.add_argument(
        "--t-world-camera",
        type=Path,
        required=True,
        help="Calibrated camera-to-world transform shaped (4,4)",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help="Frame from a pose sequence; -1 selects the last frame",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    camera_object = np.asarray(
        np.load(args.t_camera_object, allow_pickle=False), dtype=float
    )
    if camera_object.shape == (4, 4):
        frame_index = 0
        selected = camera_object
    elif camera_object.ndim == 3 and camera_object.shape[1:] == (4, 4):
        try:
            selected = camera_object[args.frame_index]
        except IndexError as error:
            raise ValueError(
                f"frame index {args.frame_index} is outside {len(camera_object)} poses"
            ) from error
        frame_index = args.frame_index % len(camera_object)
    else:
        raise ValueError(
            "T_camera_object must have shape (4,4) or (F,4,4), "
            f"got {camera_object.shape}"
        )

    world_camera = _load_transform(args.t_world_camera, "T_world_camera")
    world_object = world_camera @ selected
    pose_xyzw = np.r_[
        world_object[:3, 3], Rotation.from_matrix(world_object[:3, :3]).as_quat()
    ]
    payload = {
        "schema_version": 1,
        "frame_index": frame_index,
        "pose_convention": "T_world_object; quaternion XYZW",
        "T_world_object": world_object.tolist(),
        "pose_xyzw": pose_xyzw.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("pose_xyzw:", " ".join(f"{value:.9g}" for value in pose_xyzw))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
