#!/usr/bin/env python3
"""Send one recorded RGB-D frame through the remote pose contract (no robot I/O)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo2gripper.f2g_pose_service import (  # noqa: E402
    PoseServiceClient,
    RemotePoseProvider,
)
from demo2gripper.pose_observation import PoseObservationGuardConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8765")
    parser.add_argument("--rgb", required=True, type=Path)
    parser.add_argument("--depth", required=True, type=Path)
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="metres per stored depth unit (use 0.001 for millimetre PNG)",
    )
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--frame-id", default="recorded-frame")
    parser.add_argument("--mask-source", default="recorded_binary_mask")
    parser.add_argument("--mask-source-score", type=float)
    parser.add_argument("--timeout-s", type=float, default=0.25)
    parser.add_argument("--max-staleness-s", type=float, default=0.50)
    parser.add_argument("--max-server-latency-s", type=float, default=0.30)
    parser.add_argument("--max-roundtrip-latency-s", type=float, default=0.40)
    parser.add_argument("--auth-token-env", default="D2G_POSE_TOKEN")
    parser.add_argument(
        "--allow-fake-backend",
        action="store_true",
        help="accept the CPU test backend in this offline query",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_image(path: Path, *, rgb: bool) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("PNG/JPEG input requires OpenCV; use .npy arrays instead") from error
    flag = cv2.IMREAD_COLOR if rgb else cv2.IMREAD_UNCHANGED
    value = cv2.imread(str(path), flag)
    if value is None:
        raise RuntimeError(f"could not read image {path}")
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB) if rgb else value


def load_intrinsics(path: Path) -> np.ndarray:
    return (
        np.load(path, allow_pickle=False)
        if path.suffix.lower() == ".npy"
        else np.loadtxt(path)
    )


def main() -> int:
    args = parse_args()
    rgb = load_image(args.rgb, rgb=True)
    depth = load_image(args.depth, rgb=False).astype(np.float32) * args.depth_scale
    mask = load_image(args.mask, rgb=False)
    K = load_intrinsics(args.intrinsics)
    token = os.environ.get(args.auth_token_env) or None
    allowed_backends = {"f2g_pose"}
    if args.allow_fake_backend:
        allowed_backends.add("fake_pose")
    client = PoseServiceClient(
        args.endpoint, timeout_s=args.timeout_s, auth_token=token
    )
    provider = RemotePoseProvider(
        client,
        guard_config=PoseObservationGuardConfig(
            max_staleness_s=args.max_staleness_s,
            max_server_latency_s=args.max_server_latency_s,
            max_roundtrip_latency_s=args.max_roundtrip_latency_s,
            allowed_backends=frozenset(allowed_backends),
        ),
    )
    captured_at = time.monotonic()
    observation = provider.estimate(
        frame_id=args.frame_id,
        capture_timestamp_s=captured_at,
        rgb=rgb,
        depth_m=depth,
        K=K,
        mask=mask,
        mask_source=args.mask_source,
        mask_source_score=args.mask_source_score,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(observation.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
