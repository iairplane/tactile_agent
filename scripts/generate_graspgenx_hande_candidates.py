#!/usr/bin/env python3
"""Generate object-frame Hand-E grasp candidates with local locked GraspGenX.

This is an offline perception adapter.  It never imports a robot driver and
cannot send robot or gripper commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from demo2gripper.graspgenx_adapter import (  # noqa: E402
    FRAME_CONVENTION,
    GRIPPER_NAME,
    canonical_grasp_poses_to_object_tcp,
    center_object_points,
    hande_base_to_canonical_tcp,
    load_hande_dimensions,
    save_candidates_npz,
    save_provenance_json,
    validate_object_points,
)

DEFAULT_GRASPGENX_ROOT = PROJECT_ROOT / "third_party" / "locked" / "grasp_candidates"
DEFAULT_GRIPPER_DESCRIPTIONS_ROOT = PROJECT_ROOT / "third_party" / "models" / "gripper_descriptions"
DEFAULT_HANDE_CONFIG = (
    DEFAULT_GRIPPER_DESCRIPTIONS_ROOT
    / "gripper_descriptions"
    / "assets"
    / "x_grippers"
    / GRIPPER_NAME
    / "config.json"
)
DEFAULT_CHECKPOINTS = PROJECT_ROOT / "third_party" / "models" / "graspgenx_checkpoints" / "release"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-points", type=Path, required=True, help="(N,3) NPY in metres")
    parser.add_argument(
        "--object-frame-id",
        required=True,
        help="Stable identity of the object coordinate frame used by the NPY",
    )
    parser.add_argument("--output", type=Path, required=True, help="Canonical candidates.npz")
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=None,
        help="Default: <output stem>.provenance.json",
    )
    parser.add_argument("--graspgenx-root", type=Path, default=DEFAULT_GRASPGENX_ROOT)
    parser.add_argument(
        "--gripper-descriptions-root",
        type=Path,
        default=DEFAULT_GRIPPER_DESCRIPTIONS_ROOT,
    )
    parser.add_argument("--hande-config", type=Path, default=DEFAULT_HANDE_CONFIG)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--gen-pth", default=None, help="Optional generator checkpoint filename")
    parser.add_argument(
        "--dis-pth", default=None, help="Optional discriminator checkpoint filename"
    )
    parser.add_argument("--tcp-offset-m", type=float, default=None)
    parser.add_argument("--pregrasp-width-m", type=float, default=None)
    parser.add_argument("--grasp-threshold", type=float, default=-1.0)
    parser.add_argument("--num-grasps", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--min-grasps", type=int, default=40)
    parser.add_argument("--max-tries", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _resolve_checkpoint(checkpoint_dir: Path, explicit_name: str | None) -> Path:
    if explicit_name is not None:
        path = checkpoint_dir / explicit_name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = list(checkpoint_dir.glob("epoch_*.pth"))
    if not candidates:
        candidates = list(checkpoint_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir}")

    def epoch(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(candidates, key=epoch)


def _load_backend(args: argparse.Namespace):
    graspgenx_root = args.graspgenx_root.resolve()
    gripper_root = args.gripper_descriptions_root.resolve()
    checkpoints = args.checkpoints.resolve()
    if not (graspgenx_root / "graspgenx").is_dir():
        raise FileNotFoundError(f"invalid GraspGenX checkout: {graspgenx_root}")
    if not (gripper_root / "gripper_descriptions").is_dir():
        raise FileNotFoundError(f"invalid gripper_descriptions checkout: {gripper_root}")
    if not checkpoints.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoints}")

    os.environ["GRASPGENX_GRIPPER_CFG_DIR"] = str(gripper_root)
    os.environ["GRASPGENX_CHECKPOINT_DIR"] = str(checkpoints.parent)
    sys.path.insert(0, str(graspgenx_root))
    sys.path.insert(0, str(gripper_root))

    import torch
    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.utils.checkpoint_io import load_model_cfg

    if not torch.cuda.is_available():
        raise RuntimeError("GraspGenX requires CUDA, but torch.cuda.is_available() is false")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_cfg = load_model_cfg(
        str(checkpoints / "gen"),
        str(checkpoints / "dis"),
        args.gen_pth,
        args.dis_pth,
    )
    sampler = GraspGenXSampler(model_cfg, GRIPPER_NAME)
    return torch, GraspGenXSampler, sampler


def main() -> None:
    args = _parse_args()
    if args.num_grasps <= 0 or args.top_k <= 0 or args.min_grasps <= 0 or args.max_tries <= 0:
        raise ValueError("inference count arguments must be positive")

    object_frame_id = args.object_frame_id.strip()
    if not object_frame_id:
        raise ValueError("--object-frame-id must be non-empty")
    input_path = args.object_points.resolve()
    points = validate_object_points(np.load(input_path, allow_pickle=False))
    centered_points, T_object_centered = center_object_points(points)
    tcp_offset_m, pregrasp_width_m, dimension_sources = load_hande_dimensions(
        args.hande_config,
        tcp_offset_override_m=args.tcp_offset_m,
        pregrasp_width_override_m=args.pregrasp_width_m,
    )

    torch, sampler_type, sampler = _load_backend(args)
    with torch.inference_mode():
        poses_tensor, scores_tensor = sampler_type.run_inference(
            centered_points,
            sampler,
            grasp_threshold=args.grasp_threshold,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.top_k,
            min_grasps=args.min_grasps,
            max_tries=args.max_tries,
            remove_outliers=False,
        )
    if len(poses_tensor) == 0:
        raise RuntimeError("GraspGenX returned no candidates")
    # GraspGenX output is its canonical grasp frame (+X closing, +Z
    # approach), not the Hand-E URDF's native base_link frame.
    poses_centered_canonical = poses_tensor.detach().cpu().numpy()
    scores = scores_tensor.detach().cpu().numpy()
    poses_object_tcp = canonical_grasp_poses_to_object_tcp(
        poses_centered_canonical,
        T_object_centered,
        tcp_offset_m,
    )

    output_path = args.output.resolve()
    save_candidates_npz(
        output_path,
        poses_object_tcp,
        scores,
        pregrasp_width_m,
        object_frame_id=object_frame_id,
        base_to_tcp_offset_m=tcp_offset_m,
    )
    provenance_path = (
        args.provenance_output.resolve()
        if args.provenance_output is not None
        else output_path.with_suffix(".provenance.json")
    )
    gen_checkpoint = _resolve_checkpoint(args.checkpoints.resolve() / "gen", args.gen_pth)
    dis_checkpoint = _resolve_checkpoint(args.checkpoints.resolve() / "dis", args.dis_pth)
    provenance = {
        "schema_version": 1,
        "artifact_type": "graspgenx_candidates",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "frame_id": object_frame_id,
        "object_frame_id": object_frame_id,
        "frame_convention": FRAME_CONVENTION,
        "gripper_name": GRIPPER_NAME,
        "input": {
            "object_points_path": str(input_path),
            "object_points_sha256": _sha256(input_path),
            "point_count": int(len(points)),
            "units": "m",
        },
        "centering": {
            "applied_before_inference": True,
            "point_cloud_center_object_m": T_object_centered[:3, 3].tolist(),
            "T_object_centered": T_object_centered.tolist(),
            "restored_before_tcp_conversion": True,
        },
        "gripper": {
            "config_path": str(args.hande_config.resolve()),
            "config_sha256": _sha256(args.hande_config.resolve()),
            "tcp_offset_m": tcp_offset_m,
            "pregrasp_width_m": pregrasp_width_m,
            "dimension_sources": dimension_sources,
            "T_hande_base_canonical_tcp": hande_base_to_canonical_tcp(
                tcp_offset_m
            ).tolist(),
        },
        "backend": {
            "name": "NVlabs/GraspGenX",
            "source_root": str(args.graspgenx_root.resolve()),
            "git_revision": _git_revision(args.graspgenx_root.resolve()),
            "generator_checkpoint": str(gen_checkpoint),
            "generator_checkpoint_sha256": _sha256(gen_checkpoint),
            "discriminator_checkpoint": str(dis_checkpoint),
            "discriminator_checkpoint_sha256": _sha256(dis_checkpoint),
        },
        "inference": {
            "sampler_output_frame": "graspgenx_canonical",
            "sampler_output_axes": "+X closing, +Z approach",
            "native_hande_base_rotation_applied_to_sampler_output": False,
            "grasp_threshold": args.grasp_threshold,
            "num_grasps": args.num_grasps,
            "top_k": args.top_k,
            "min_grasps": args.min_grasps,
            "max_tries": args.max_tries,
            "remove_outliers": False,
            "seed": args.seed,
        },
        "output": {
            "candidates_path": str(output_path),
            "candidates_sha256": _sha256(output_path),
            "candidate_count": int(len(poses_object_tcp)),
            "npz_keys": [
                "poses",
                "scores",
                "widths_m",
                "gripper_name",
                "object_frame_id",
                "base_to_tcp_offset_m",
                "frame_convention",
            ],
        },
    }
    save_provenance_json(provenance_path, provenance)
    print(
        json.dumps(
            {
                "candidates": str(output_path),
                "provenance": str(provenance_path),
                "candidate_count": len(poses_object_tcp),
                "tcp_offset_m": tcp_offset_m,
                "pregrasp_width_m": pregrasp_width_m,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
