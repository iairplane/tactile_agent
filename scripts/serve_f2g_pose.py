#!/usr/bin/env python3
"""Serve pose-only F2G-Pose inference over the versioned RGB-D HTTP contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo2gripper.f2g_pose_service import (  # noqa: E402
    F2GPoseBackend,
    FakePoseBackend,
    ForegroundDepthGateConfig,
    MaskGateConfig,
    PoseInferenceService,
    create_pose_http_server,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--backend", choices=("f2g_pose", "fake_pose"), default="f2g_pose")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--radio-repository", type=Path)
    parser.add_argument("--radio-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--f2g-root",
        type=Path,
        help="official F2G-Pose checkout, used only to record its git revision",
    )
    parser.add_argument(
        "--auth-token-env",
        default="D2G_POSE_TOKEN",
        help="environment variable holding an optional bearer token",
    )
    parser.add_argument(
        "--allow-fake-backend",
        action="store_true",
        help="required acknowledgement for the CPU-only test backend",
    )
    parser.add_argument("--min-mask-pixels", type=int, default=100)
    parser.add_argument("--min-valid-depth-pixels", type=int, default=100)
    parser.add_argument("--min-valid-depth-fraction", type=float, default=0.20)
    parser.add_argument("--min-depth-cluster-pixels", type=int, default=80)
    parser.add_argument("--min-depth-cluster-fraction", type=float, default=0.25)
    parser.add_argument("--depth-cluster-gap-m", type=float, default=0.008)
    parser.add_argument("--max-depth-cluster-span-m", type=float, default=0.050)
    parser.add_argument("--background-tolerance-m", type=float, default=0.010)
    parser.add_argument("--background-separation-m", type=float, default=0.012)
    parser.add_argument("--max-background-fraction", type=float, default=0.50)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path | None) -> str:
    if root is None:
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def main() -> int:
    args = parse_args()
    if args.backend == "fake_pose":
        if not args.allow_fake_backend:
            raise SystemExit("--backend fake_pose requires --allow-fake-backend")
        backend = FakePoseBackend()
    else:
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise SystemExit("--checkpoint must name the installed F2G-Pose weights")
        backend = F2GPoseBackend(
            args.checkpoint,
            config=args.config,
            radio_repository=args.radio_repository,
            radio_checkpoint=args.radio_checkpoint,
            device=args.device,
            code_revision=git_revision(args.f2g_root),
            checkpoint_sha256=sha256(args.checkpoint),
        )
    service = PoseInferenceService(
        backend,
        mask_config=MaskGateConfig(min_pixels=args.min_mask_pixels),
        depth_config=ForegroundDepthGateConfig(
            min_valid_mask_pixels=args.min_valid_depth_pixels,
            min_valid_mask_fraction=args.min_valid_depth_fraction,
            cluster_gap_m=args.depth_cluster_gap_m,
            min_cluster_pixels=args.min_depth_cluster_pixels,
            min_cluster_fraction=args.min_depth_cluster_fraction,
            max_cluster_span_m=args.max_depth_cluster_span_m,
            background_match_tolerance_m=args.background_tolerance_m,
            min_background_separation_m=args.background_separation_m,
            max_mask_near_background_fraction=args.max_background_fraction,
        ),
    )
    token = os.environ.get(args.auth_token_env) or None
    if args.host not in {"127.0.0.1", "localhost", "::1"} and token is None:
        raise SystemExit(
            f"non-loopback --host requires a bearer token in {args.auth_token_env}"
        )
    server = create_pose_http_server(
        args.host, args.port, service, auth_token=token
    )
    print(
        json.dumps(
            {
                **service.health(),
                "listen": f"http://{args.host}:{server.server_port}",
                "authentication": "bearer" if token is not None else "none",
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
