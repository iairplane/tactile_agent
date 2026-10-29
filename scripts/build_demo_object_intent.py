#!/usr/bin/env python3
"""Build a no-CAD object template and raw human object-frame grasp intent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from demo2gripper.demo_object_intent import (
    DemoObjectIntentConfig,
    build_demo_object_intent,
    write_demo_object_intent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickup-json", type=Path, required=True)
    parser.add_argument("--pickup-track", type=Path, required=True)
    parser.add_argument("--hand-joints", type=Path, required=True)
    parser.add_argument("--hand-confidence", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--precontact-object-points", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-tip-confidence", type=float, default=0.4)
    parser.add_argument("--min-wrist-confidence", type=float, default=0.25)
    parser.add_argument("--min-support-frames", type=int, default=3)
    parser.add_argument(
        "--critical-tail-frames",
        type=int,
        default=None,
        help=(
            "for legacy pickup JSON without critical_grasp_interval, use this many "
            "terminal stable-contact frames for grasp SE(3)"
        ),
    )
    parser.add_argument(
        "--write-observations",
        action="store_true",
        help="also write a standard translation-only observations.npz archive",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pickup = json.loads(args.pickup_json.read_text(encoding="utf-8"))
    with np.load(args.pickup_track, allow_pickle=False) as archive:
        required = {"object_centers_camera_m", "timestamps_s"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                "pickup track is missing: " + ", ".join(sorted(missing))
            )
        centers = np.asarray(archive["object_centers_camera_m"], dtype=float)
        track_timestamps = np.asarray(archive["timestamps_s"], dtype=float)

    artifacts = build_demo_object_intent(
        pickup=pickup,
        object_centers_camera_m=centers,
        track_timestamps_s=track_timestamps,
        hand_joints_camera_m=np.load(args.hand_joints, allow_pickle=False),
        hand_joint_confidence=np.load(args.hand_confidence, allow_pickle=False),
        timestamps_s=np.load(args.timestamps, allow_pickle=False),
        precontact_object_points_camera_m=np.load(
            args.precontact_object_points, allow_pickle=False
        ),
        config=DemoObjectIntentConfig(
            min_tip_confidence=args.min_tip_confidence,
            min_wrist_confidence=args.min_wrist_confidence,
            min_support_frames=args.min_support_frames,
            critical_grasp_tail_frames=args.critical_tail_frames,
        ),
    )
    manifest = write_demo_object_intent(
        artifacts,
        args.output_dir,
        provenance={
            "pickup_json": str(args.pickup_json.resolve()),
            "pickup_track": str(args.pickup_track.resolve()),
            "hand_joints": str(args.hand_joints.resolve()),
            "hand_confidence": str(args.hand_confidence.resolve()),
            "timestamps": str(args.timestamps.resolve()),
            "precontact_object_points": str(
                args.precontact_object_points.resolve()
            ),
        },
        write_observations=args.write_observations,
    )
    intent = artifacts.intent
    print(
        f"template: {args.output_dir / 'object_points_m.npy'} "
        f"({len(artifacts.object_points_m)} points, {manifest['frame_id']})"
    )
    print(
        f"intent: {args.output_dir / 'grasp_intent.json'} "
        f"(validated frames {artifacts.validated_contact_interval[0]}:"
        f"{artifacts.validated_contact_interval[1]}, critical frames "
        f"{intent.stable_interval[0]}:{intent.stable_interval[1]}, "
        f"support {intent.support_frames}, width {intent.width_m:.4f} m)"
    )


if __name__ == "__main__":
    main()
