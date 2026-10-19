#!/usr/bin/env python3
"""Run the original NVlabs FoundationPose register-and-track pipeline on RGB-D frames."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument(
        "--depth-dir", type=Path, required=True, help="Float32 .npy depth in metres"
    )
    parser.add_argument("--intrinsics", type=Path, required=True, help="Camera K as a 3x3 .npy")
    parser.add_argument(
        "--object-mask-dir",
        type=Path,
        required=True,
        help="Object mask PNGs; FoundationPose uses the first frame for registration",
    )
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-scale", type=float, default=1.0, help="Mesh units to metres")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--foundationpose-root",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "locked" / "known_object_pose_fallback",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--registration-refine-iterations", type=int, default=5)
    parser.add_argument("--tracking-refine-iterations", type=int, default=2)
    return parser.parse_args()


def numeric_files(directory: Path, suffix: str) -> list[Path]:
    files = [path for path in directory.iterdir() if path.suffix.lower() == suffix]
    try:
        return sorted(files, key=lambda path: int(path.stem))
    except ValueError as exc:
        raise ValueError(f"Frame stems must be integers in {directory}") from exc


def main() -> int:
    args = parse_args()
    if args.mesh_scale <= 0:
        raise ValueError("--mesh-scale must be positive")

    rgb_paths = numeric_files(args.rgb_dir, ".png")
    depth_by_stem = {path.stem: path for path in numeric_files(args.depth_dir, ".npy")}
    mask_by_stem = {path.stem: path for path in numeric_files(args.object_mask_dir, ".png")}
    if not rgb_paths:
        raise ValueError(f"No PNG frames found in {args.rgb_dir}")
    if any(path.stem not in depth_by_stem for path in rgb_paths):
        raise ValueError("Every RGB frame needs a same-stem depth array")
    first_mask_path = mask_by_stem.get(rgb_paths[0].stem)
    if first_mask_path is None:
        raise ValueError(f"No first-frame object mask for {rgb_paths[0].name}")

    intrinsics = np.asarray(np.load(args.intrinsics, allow_pickle=False), dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Camera intrinsics must have shape (3, 3), got {intrinsics.shape}")

    foundationpose_root = args.foundationpose_root.resolve()
    sys.path.insert(0, str(foundationpose_root))

    import nvdiffrast.torch as dr
    import torch
    import trimesh
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    from Utils import set_logging_format, set_seed

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("The original FoundationPose backend requires a CUDA device")
    torch.cuda.set_device(device)
    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(str(args.mesh), force="mesh")
    mesh.apply_scale(args.mesh_scale)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext(device=device)
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug=0,
        debug_dir=str(args.output_dir / "debug"),
    )

    first_mask = cv2.imread(str(first_mask_path), cv2.IMREAD_GRAYSCALE)
    if first_mask is None or not np.any(first_mask):
        raise ValueError(f"First-frame object mask is empty or unreadable: {first_mask_path}")

    poses: list[np.ndarray] = []
    frame_records: list[dict[str, object]] = []
    registration_score: float | None = None
    image_shape: tuple[int, int] | None = None

    for frame_index, rgb_path in enumerate(rgb_paths):
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read RGB frame: {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth_path = depth_by_stem[rgb_path.stem]
        depth = np.asarray(np.load(depth_path, allow_pickle=False), dtype=np.float32)
        if image_shape is None:
            image_shape = rgb.shape[:2]
            if first_mask.shape != image_shape:
                raise ValueError(
                    f"First mask shape {first_mask.shape} does not match RGB {image_shape}"
                )
        if rgb.shape[:2] != image_shape or depth.shape != image_shape:
            raise ValueError(f"RGB/depth shape mismatch at frame {rgb_path.name}")

        started = time.perf_counter()
        if frame_index == 0:
            pose = estimator.register(
                K=intrinsics,
                rgb=rgb,
                depth=depth,
                ob_mask=first_mask > 0,
                iteration=args.registration_refine_iterations,
            )
            phase = "register"
            registration_score = float(estimator.scores[0].detach().cpu())
        else:
            pose = estimator.track_one(
                rgb=rgb,
                depth=depth,
                K=intrinsics,
                iteration=args.tracking_refine_iterations,
            )
            phase = "track"
        elapsed_s = time.perf_counter() - started

        pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)
        if not np.isfinite(pose).all():
            raise RuntimeError(f"FoundationPose returned a non-finite pose for {rgb_path.name}")
        poses.append(pose)
        frame_records.append(
            {
                "frame_index": frame_index,
                "rgb": rgb_path.name,
                "depth": depth_path.name,
                "phase": phase,
                "elapsed_s": elapsed_s,
            }
        )
        translation = pose[:3, 3]
        print(
            f"{frame_index + 1}/{len(rgb_paths)} {phase:8s} "
            f"t=({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}) m "
            f"{elapsed_s:.2f}s"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pose_array = np.stack(poses).astype(np.float32, copy=False)
    np.save(args.output_dir / "T_camera_object.npy", pose_array, allow_pickle=False)

    manifest = {
        "backend": "NVlabs/FoundationPose (original PyTorch implementation)",
        "display_name": "FoundationPose",
        "pose_convention": (
            "T_camera_object maps metre-scaled CAD coordinates into camera coordinates"
        ),
        "translation_unit": "metre",
        "registration": "first-frame RGB-D + object mask + CAD; no input pose",
        "registration_score_raw": registration_score,
        "rgb_dir": str(args.rgb_dir.resolve()),
        "depth_dir": str(args.depth_dir.resolve()),
        "intrinsics": str(args.intrinsics.resolve()),
        "first_mask": str(first_mask_path.resolve()),
        "mesh": str(args.mesh.resolve()),
        "mesh_scale": args.mesh_scale,
        "foundationpose_root": str(foundationpose_root),
        "device": str(device),
        "registration_refine_iterations": args.registration_refine_iterations,
        "tracking_refine_iterations": args.tracking_refine_iterations,
        "shape": {"T_camera_object": list(pose_array.shape)},
        "frames": frame_records,
    }
    manifest_path = args.output_dir / "foundationpose_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(poses)} object poses to {args.output_dir / 'T_camera_object.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
