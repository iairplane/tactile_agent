#!/usr/bin/env python3
"""Estimate independent WAPR poses for a numeric RGB-D sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import trimesh

from estimate_wapr_pose import save_pose_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--object-mask-dir", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def numeric_files(directory: Path, suffix: str) -> list[Path]:
    paths = [path for path in directory.iterdir() if path.suffix.lower() == suffix]
    try:
        return sorted(paths, key=lambda path: int(path.stem))
    except ValueError as error:
        raise ValueError(f"Frame filename stems must be integers in {directory}") from error


def main() -> int:
    args = parse_args()
    if args.mesh_scale <= 0:
        raise ValueError("--mesh-scale must be positive")

    rgb_paths = numeric_files(args.rgb_dir, ".png")
    depth_by_stem = {path.stem: path for path in numeric_files(args.depth_dir, ".npy")}
    mask_by_stem = {
        path.stem: path for path in numeric_files(args.object_mask_dir, ".png")
    }
    if not rgb_paths:
        raise ValueError(f"No RGB frames found in {args.rgb_dir}")
    if any(path.stem not in depth_by_stem or path.stem not in mask_by_stem for path in rgb_paths):
        raise ValueError("Every RGB frame needs a same-stem depth array and object mask")

    intrinsics = np.asarray(np.load(args.intrinsics, allow_pickle=False), dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Intrinsics must have shape (3, 3), got {intrinsics.shape}")

    mesh = trimesh.load(str(args.mesh), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected one triangle mesh: {args.mesh}")
    mesh.apply_scale(args.mesh_scale)
    diameter_m = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))

    from wapr import WAPREstimator

    setup_started = time.perf_counter()
    estimator = WAPREstimator(device=args.device)
    prepared_mesh = estimator.prepare_meshes([mesh])[0]
    setup_s = time.perf_counter() - setup_started

    overlay_dir = args.output_dir / "pose_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    poses: list[np.ndarray] = []
    scores: list[float] = []
    records: list[dict[str, object]] = []
    for frame_index, rgb_path in enumerate(rgb_paths):
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = np.asarray(
            np.load(depth_by_stem[rgb_path.stem], allow_pickle=False), dtype=np.float32
        )
        mask = cv2.imread(str(mask_by_stem[rgb_path.stem]), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None or not np.any(mask):
            raise ValueError(f"Unreadable RGB or empty object mask for {rgb_path.name}")
        if depth.shape != bgr.shape[:2] or mask.shape != bgr.shape[:2]:
            raise ValueError(f"RGB, depth, and mask shapes differ for {rgb_path.name}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        result = estimator.estimate_one_category_one_instance(
            rgb, depth, intrinsics, prepared_mesh, diameter_m, mask=mask
        )
        elapsed_s = time.perf_counter() - started
        pose = np.asarray(result["pose_4x4"], dtype=np.float32).reshape(4, 4)
        score = float(result["score_6d"])
        if not np.isfinite(pose).all() or not np.isfinite(score):
            raise RuntimeError(f"WAPR returned an invalid result for {rgb_path.name}")

        poses.append(pose)
        scores.append(score)
        overlay_path = overlay_dir / rgb_path.name
        save_pose_overlay(bgr, mesh, pose, intrinsics, diameter_m, score, overlay_path)
        records.append(
            {
                "frame_index": frame_index,
                "rgb": rgb_path.name,
                "depth": depth_by_stem[rgb_path.stem].name,
                "mask": mask_by_stem[rgb_path.stem].name,
                "score_6d": score,
                "elapsed_s": elapsed_s,
            }
        )
        translation = pose[:3, 3]
        print(
            f"{frame_index + 1}/{len(rgb_paths)} {rgb_path.name} "
            f"score={score:.3f} t=({translation[0]:+.3f}, "
            f"{translation[1]:+.3f}, {translation[2]:+.3f})m {elapsed_s:.2f}s"
        )

    pose_array = np.stack(poses).astype(np.float32, copy=False)
    score_array = np.asarray(scores, dtype=np.float32)
    np.save(args.output_dir / "T_camera_object.npy", pose_array, allow_pickle=False)
    np.save(args.output_dir / "confidence.npy", score_array, allow_pickle=False)
    manifest = {
        "backend": "WangYuLin-SEU/WAPR official WAPREstimator",
        "display_name": "WAPR per-frame RGB-D 6D pose",
        "pose_ground_truth_used": False,
        "pose_convention": "T_camera_object maps metric CAD coordinates into camera coordinates",
        "translation_unit": "metre",
        "mesh": str(args.mesh.resolve()),
        "mesh_scale": args.mesh_scale,
        "device": args.device,
        "model_and_mesh_setup_s": setup_s,
        "shape": {
            "T_camera_object": list(pose_array.shape),
            "confidence": list(score_array.shape),
        },
        "frames": records,
    }
    (args.output_dir / "wapr_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(poses)} WAPR poses to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
