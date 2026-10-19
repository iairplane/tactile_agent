#!/usr/bin/env python3
"""Convert GraspGenX's Isaac grasp YAML into the canonical candidate NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def _grasp_number(item: tuple[str, object]) -> int:
    name = item[0]
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid GraspGenX grasp key: {name}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pregrasp-width-m",
        type=float,
        required=True,
        help="Total jaw opening before closing; GraspGenX does not predict contact width",
    )
    parser.add_argument(
        "--tcp-offset-m",
        type=float,
        required=True,
        help="Gripper-base to finger-center TCP offset along canonical +Z",
    )
    args = parser.parse_args()
    if args.pregrasp_width_m <= 0.0:
        raise ValueError("--pregrasp-width-m must be positive")

    payload = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    if payload.get("format") != "isaac_grasp":
        raise ValueError("input is not GraspGenX isaac_grasp YAML")
    entries = sorted(payload.get("grasps", {}).items(), key=_grasp_number)
    if not entries:
        raise ValueError("input contains no grasps")

    poses = []
    scores = []
    for name, entry in entries:
        position = np.asarray(entry["position"], dtype=float)
        orientation = entry["orientation"]
        quaternion_wxyz = np.r_[float(orientation["w"]), orientation["xyz"]]
        if position.shape != (3,) or quaternion_wxyz.shape != (4,):
            raise ValueError(f"invalid pose fields in {name}")
        quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
        pose[:3, 3] = position
        pose[:3, 3] += pose[:3, 2] * args.tcp_offset_m
        poses.append(pose)
        scores.append(float(entry["confidence"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        poses=np.asarray(poses),
        scores=np.asarray(scores),
        widths_m=np.full(len(poses), args.pregrasp_width_m),
    )


if __name__ == "__main__":
    main()
