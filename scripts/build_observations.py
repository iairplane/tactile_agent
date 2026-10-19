#!/usr/bin/env python3
"""Pack tracked object poses and hand joints into the canonical observation NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t-camera-object",
        required=True,
        type=Path,
        help=".npy array shaped (F, 4, 4); maps object-frame points to camera frame",
    )
    parser.add_argument(
        "--hand-joints-camera-m",
        required=True,
        type=Path,
        help=".npy array shaped (F, 21, 3), in camera coordinates and metres",
    )
    parser.add_argument(
        "--confidence",
        required=True,
        type=Path,
        help=".npy array shaped (F, 21), with values in [0, 1]",
    )
    parser.add_argument(
        "--timestamps-s",
        required=True,
        type=Path,
        help=".npy array shaped (F,), strictly increasing and in seconds",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _load(path: Path, name: str) -> np.ndarray:
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot load {name} from {path}: {exc}") from exc
    return np.asarray(array)


def _require_shape(array: np.ndarray, expected: tuple[int, ...], name: str) -> None:
    if array.shape != expected:
        raise SystemExit(f"{name} must have shape {expected}, got {array.shape}")


def main() -> int:
    args = parse_args()
    transforms = _load(args.t_camera_object, "T_camera_object")
    joints = _load(args.hand_joints_camera_m, "hand_joints_camera_m")
    confidence = _load(args.confidence, "hand_joint_confidence")
    timestamps = _load(args.timestamps_s, "timestamps_s")

    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise SystemExit(
            f"T_camera_object must have shape (F, 4, 4), got {transforms.shape}"
        )
    frame_count = transforms.shape[0]
    if frame_count == 0:
        raise SystemExit("the sequence must contain at least one frame")
    _require_shape(joints, (frame_count, 21, 3), "hand_joints_camera_m")
    _require_shape(confidence, (frame_count, 21), "hand_joint_confidence")
    _require_shape(timestamps, (frame_count,), "timestamps_s")

    arrays = {
        "T_camera_object": transforms,
        "hand_joints_camera_m": joints,
        "hand_joint_confidence": confidence,
        "timestamps_s": timestamps,
    }
    for name, array in arrays.items():
        if not np.issubdtype(array.dtype, np.number):
            raise SystemExit(f"{name} must be numeric, got dtype {array.dtype}")
        if not np.all(np.isfinite(array)):
            raise SystemExit(f"{name} contains NaN or infinity")

    expected_last_row = np.broadcast_to(
        np.array([0.0, 0.0, 0.0, 1.0]), (frame_count, 4)
    )
    if not np.allclose(transforms[:, 3, :], expected_last_row, atol=1e-6):
        raise SystemExit("T_camera_object has a non-homogeneous last row")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise SystemExit("hand_joint_confidence values must be in [0, 1]")
    if frame_count > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise SystemExit("timestamps_s must be strictly increasing")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        T_camera_object=transforms.astype(np.float64, copy=False),
        hand_joints_camera_m=joints.astype(np.float64, copy=False),
        hand_joint_confidence=confidence.astype(np.float64, copy=False),
        timestamps_s=timestamps.astype(np.float64, copy=False),
    )
    print(f"wrote {frame_count} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
