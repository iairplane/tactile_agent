#!/usr/bin/env python3
"""Evaluate a camera-frame object-pose estimate against simulation ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_transform(path: Path, name: str) -> np.ndarray:
    transform = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}")
    return transform


def _select_estimate(path: Path, frame_index: int) -> tuple[np.ndarray, int]:
    estimates = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if estimates.shape == (4, 4):
        return estimates, 0
    if estimates.ndim != 3 or estimates.shape[1:] != (4, 4):
        raise ValueError(
            "T_camera_object must have shape (4,4) or (F,4,4), "
            f"got {estimates.shape}"
        )
    try:
        estimate = estimates[frame_index]
    except IndexError as error:
        raise ValueError(
            f"frame index {frame_index} is outside {len(estimates)} poses"
        ) from error
    return estimate, frame_index % len(estimates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t-camera-object",
        type=Path,
        required=True,
        help="Estimated transform shaped (4,4) or pose sequence shaped (F,4,4)",
    )
    parser.add_argument(
        "--t-world-camera",
        type=Path,
        required=True,
        help="Calibrated camera-to-world transform shaped (4,4)",
    )
    parser.add_argument(
        "--t-world-object-gt",
        type=Path,
        required=True,
        help="Simulation ground-truth object-to-world transform shaped (4,4)",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help="Frame from a pose sequence; -1 selects the last frame",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    camera_object, frame_index = _select_estimate(
        args.t_camera_object, args.frame_index
    )
    world_camera = _load_transform(args.t_world_camera, "T_world_camera")
    world_object_gt = _load_transform(args.t_world_object_gt, "T_world_object_gt")
    world_object_est = world_camera @ camera_object

    translation_error_m = float(
        np.linalg.norm(world_object_est[:3, 3] - world_object_gt[:3, 3])
    )
    relative_rotation = (
        world_object_gt[:3, :3].T @ world_object_est[:3, :3]
    )
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    rotation_error_deg = float(np.degrees(np.arccos(cosine)))

    payload = {
        "schema_version": 1,
        "frame_index": frame_index,
        "pose_convention": "T_world_object = T_world_camera @ T_camera_object",
        "T_world_object_est": world_object_est.tolist(),
        "T_world_object_gt": world_object_gt.tolist(),
        "translation_error_m": translation_error_m,
        "translation_error_mm": translation_error_m * 1000.0,
        "rotation_error_deg": rotation_error_deg,
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
