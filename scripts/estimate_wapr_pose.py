#!/usr/bin/env python3
"""Estimate one object pose from RGB-D, mask, and CAD with official WAPR.

The saved ``T_camera_object`` maps the scaled input-mesh frame into the
OpenCV camera frame (+X right, +Y down, +Z forward). Translations are metres.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rgb", type=Path, required=True, help="Input RGB image")
    parser.add_argument(
        "--depth", type=Path, required=True, help="Float32 .npy depth image in metres"
    )
    parser.add_argument(
        "--intrinsics", type=Path, required=True, help="OpenCV camera K as a 3x3 .npy"
    )
    parser.add_argument("--object-mask", type=Path, required=True, help="Object mask image")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-scale", type=float, default=1.0, help="Mesh units to metres")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def project_points(
    points_object: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    rotation_vector, _ = cv2.Rodrigues(pose[:3, :3].astype(np.float64))
    pixels, _ = cv2.projectPoints(
        np.asarray(points_object, dtype=np.float64),
        rotation_vector,
        pose[:3, 3].astype(np.float64),
        intrinsics.astype(np.float64),
        None,
    )
    return np.rint(pixels.reshape(-1, 2)).astype(np.int32)


def save_pose_overlay(
    bgr: np.ndarray,
    mesh: trimesh.Trimesh,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    diameter_m: float,
    score_6d: float,
    output_path: Path,
) -> None:
    overlay = bgr.copy()
    lower, upper = np.asarray(mesh.bounds, dtype=np.float64)
    corners = np.asarray(
        [
            [upper[0] if index & 1 else lower[0],
             upper[1] if index & 2 else lower[1],
             upper[2] if index & 4 else lower[2]]
            for index in range(8)
        ],
        dtype=np.float64,
    )
    corner_pixels = project_points(corners, pose, intrinsics)
    for start in range(8):
        for bit in (1, 2, 4):
            end = start ^ bit
            if start < end:
                cv2.line(
                    overlay,
                    tuple(corner_pixels[start]),
                    tuple(corner_pixels[end]),
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )

    axis_length = 0.35 * diameter_m
    axes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float64,
    )
    axis_pixels = project_points(axes, pose, intrinsics)
    origin = tuple(axis_pixels[0])
    axis_styles = (("X", (0, 0, 255)), ("Y", (0, 255, 0)), ("Z", (255, 0, 0)))
    for endpoint, (label, color) in zip(axis_pixels[1:], axis_styles):
        point = tuple(endpoint)
        cv2.arrowedLine(overlay, origin, point, color, 3, cv2.LINE_AA, tipLength=0.12)
        cv2.putText(
            overlay,
            label,
            point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    translation = pose[:3, 3]
    labels = (
        f"WAPR object->camera | score={score_6d:.3f}",
        f"t=({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}) m",
    )
    for index, label in enumerate(labels):
        cv2.putText(
            overlay,
            label,
            (12, 26 + 24 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"Could not write pose overlay: {output_path}")


def main() -> int:
    args = parse_args()
    if args.mesh_scale <= 0:
        raise ValueError("--mesh-scale must be positive")

    bgr = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read RGB image: {args.rgb}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_m = np.asarray(np.load(args.depth, allow_pickle=False), dtype=np.float32)
    intrinsics = np.asarray(
        np.load(args.intrinsics, allow_pickle=False), dtype=np.float32
    )
    mask = cv2.imread(str(args.object_mask), cv2.IMREAD_GRAYSCALE)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Camera intrinsics must have shape (3, 3), got {intrinsics.shape}")
    if depth_m.shape != rgb.shape[:2]:
        raise ValueError(
            f"Depth shape {depth_m.shape} does not match RGB shape {rgb.shape[:2]}"
        )
    if mask is None or mask.shape != rgb.shape[:2] or not np.any(mask):
        raise ValueError("Object mask must be readable, non-empty, and match the RGB image")

    mesh = trimesh.load(str(args.mesh), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected one triangle mesh: {args.mesh}")
    mesh.apply_scale(args.mesh_scale)
    diameter_m = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    if not np.isfinite(diameter_m) or diameter_m <= 0:
        raise ValueError("Scaled mesh must have a finite positive diameter")

    from wapr import WAPREstimator

    setup_started = time.perf_counter()
    estimator = WAPREstimator(device=args.device)
    model_setup_s = time.perf_counter() - setup_started

    mesh_started = time.perf_counter()
    prepared_mesh = estimator.prepare_meshes([mesh])[0]
    mesh_setup_s = time.perf_counter() - mesh_started

    inference_started = time.perf_counter()
    result = estimator.estimate_one_category_one_instance(
        rgb,
        depth_m,
        intrinsics,
        prepared_mesh,
        diameter_m,
        mask=mask,
    )
    inference_s = time.perf_counter() - inference_started
    pose = np.asarray(result["pose_4x4"], dtype=np.float32).reshape(4, 4)
    if not np.isfinite(pose).all():
        raise RuntimeError("WAPR returned a non-finite pose")
    score_6d = float(result["score_6d"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pose_array = pose[None]
    pose_path = args.output_dir / "T_camera_object.npy"
    np.save(pose_path, pose_array, allow_pickle=False)
    overlay_path = args.output_dir / "pose_overlay.png"
    save_pose_overlay(
        bgr,
        mesh,
        pose,
        intrinsics,
        diameter_m,
        score_6d,
        overlay_path,
    )

    manifest = {
        "backend": "WangYuLin-SEU/WAPR official WAPREstimator",
        "input_protocol": "single RGB-D frame + object mask + metric CAD mesh",
        "pose_ground_truth_used": False,
        "pose_convention": (
            "T_camera_object maps the scaled input-mesh frame into the OpenCV "
            "camera frame (+X right, +Y down, +Z forward)"
        ),
        "translation_unit": "metre",
        "rgb": str(args.rgb.resolve()),
        "depth_m": str(args.depth.resolve()),
        "intrinsics": str(args.intrinsics.resolve()),
        "object_mask": str(args.object_mask.resolve()),
        "mesh": str(args.mesh.resolve()),
        "mesh_scale_to_metres": args.mesh_scale,
        "diameter_m": diameter_m,
        "diameter_definition": "diagonal of the scaled mesh axis-aligned bounds",
        "device": args.device,
        "score_6d": score_6d,
        "timing_s": {
            "model_setup": model_setup_s,
            "mesh_setup": mesh_setup_s,
            "inference": inference_s,
        },
        "shape": {"T_camera_object": list(pose_array.shape)},
        "outputs": {
            "T_camera_object": pose_path.name,
            "pose_overlay": overlay_path.name,
        },
    }
    manifest_path = args.output_dir / "wapr_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote WAPR object-to-camera pose, score={score_6d:.3f}, "
        f"t=({pose[0, 3]:.4f}, {pose[1, 3]:.4f}, {pose[2, 3]:.4f}) m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
