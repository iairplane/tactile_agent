#!/usr/bin/env python3
"""Register then track one known CAD object through paired RGB-D frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rgb-dir", type=Path, required=True, help="Numeric RGB PNG frames.")
    parser.add_argument(
        "--depth-dir", type=Path, required=True, help="Numeric float depth .npy frames in metres."
    )
    parser.add_argument("--intrinsics", type=Path, required=True, help="Camera K as a 3x3 .npy.")
    parser.add_argument("--first-mask", type=Path, required=True, help="First-frame mask PNG.")
    parser.add_argument("--cad", type=Path, required=True, help="Object CAD mesh (OBJ or PLY).")
    parser.add_argument("--refine-model-path", type=Path, required=True)
    parser.add_argument("--score-model-path", type=Path, required=True)
    parser.add_argument(
        "--library",
        type=Path,
        help="libfoundation_pose_nvidia.so; omit to use upstream auto-discovery.",
    )
    parser.add_argument("--engine-cache-dir", type=Path, default=Path("engine_cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mesh-unit-scale",
        type=float,
        default=1.0,
        help="Scale from CAD units to metres (use 0.001 for millimetres).",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device index after CUDA_VISIBLE_DEVICES remapping.",
    )
    parser.add_argument("--n-hypotheses", type=int, default=252)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Optional RefineNet micro-batch; must divide n-hypotheses.",
    )
    return parser.parse_args()


def _numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Frame filename stem must be an integer: {path.name}") from exc


def discover_pairs(rgb_dir: Path, depth_dir: Path) -> list[tuple[Path, Path]]:
    if not rgb_dir.is_dir():
        raise ValueError(f"RGB directory does not exist: {rgb_dir}")
    if not depth_dir.is_dir():
        raise ValueError(f"Depth directory does not exist: {depth_dir}")

    rgb = {path.stem: path for path in rgb_dir.iterdir() if path.suffix.lower() == ".png"}
    depth = {path.stem: path for path in depth_dir.iterdir() if path.suffix.lower() == ".npy"}
    if not rgb:
        raise ValueError(f"No RGB PNG frames found in {rgb_dir}")
    if set(rgb) != set(depth):
        missing_depth = sorted(set(rgb) - set(depth))
        missing_rgb = sorted(set(depth) - set(rgb))
        raise ValueError(
            "RGB/depth stems must match exactly; "
            f"missing depth={missing_depth}, missing RGB={missing_rgb}"
        )

    stems = sorted(rgb, key=lambda stem: _numeric_stem(rgb[stem]))
    frame_numbers = [_numeric_stem(rgb[stem]) for stem in stems]
    if len(set(frame_numbers)) != len(frame_numbers):
        raise ValueError("Frame numbers are not unique")
    expected = list(range(frame_numbers[0], frame_numbers[-1] + 1))
    if frame_numbers != expected:
        raise ValueError(
            f"Frame numbers must be contiguous; got {frame_numbers[0]}.."
            f"{frame_numbers[-1]} with {len(frame_numbers)} pairs"
        )
    return [(rgb[stem], depth[stem]) for stem in stems]


def load_intrinsics(path: Path) -> np.ndarray:
    intrinsics = np.load(path, allow_pickle=False)
    if intrinsics.shape != (3, 3) or not np.issubdtype(intrinsics.dtype, np.number):
        raise ValueError(f"Camera intrinsics must be a numeric (3, 3) array: {path}")
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if not np.isfinite(intrinsics).all() or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("Camera intrinsics must be finite with positive fx and fy")
    if not np.allclose(intrinsics[2], (0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError("Camera intrinsics last row must be [0, 0, 1]")
    return np.ascontiguousarray(intrinsics)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError(f"RGB input must be an RGB PNG: {path}")
        rgb = np.asarray(image, dtype=np.uint8)
    return np.ascontiguousarray(rgb)


def load_depth(path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    depth = np.load(path, allow_pickle=False)
    if depth.shape != image_shape:
        raise ValueError(
            f"Depth shape {depth.shape} does not match RGB shape {image_shape}: {path}"
        )
    if not np.issubdtype(depth.dtype, np.floating):
        raise ValueError(f"Depth must use a floating dtype and contain metres: {path}")
    depth = np.asarray(depth, dtype=np.float32)
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError(f"Depth must be finite and non-negative: {path}")
    return np.ascontiguousarray(depth)


def load_mask(path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        mask = np.asarray(image)
    if mask.ndim != 2 or mask.shape != image_shape:
        raise ValueError(
            f"First-frame mask must be single-channel with shape {image_shape}: {path}"
        )
    mask = np.ascontiguousarray(mask != 0, dtype=np.uint8) * 255
    if not np.any(mask):
        raise ValueError("First-frame mask is empty")
    return mask


def require_file(path: Path | None, label: str) -> None:
    if path is not None and not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")


def validate_args(args: argparse.Namespace) -> None:
    require_file(args.intrinsics, "Camera intrinsics")
    require_file(args.first_mask, "First-frame mask")
    require_file(args.cad, "CAD mesh")
    require_file(args.refine_model_path, "RefineNet weights")
    require_file(args.score_model_path, "ScoreNet weights")
    require_file(args.library, "FoundationPose library")
    if args.mesh_unit_scale <= 0:
        raise ValueError("mesh-unit-scale must be positive")
    if args.device_id < 0:
        raise ValueError("device-id must be non-negative")
    if args.n_hypotheses <= 0:
        raise ValueError("n-hypotheses must be positive")
    if args.batch_size is not None and (
        args.batch_size <= 0 or args.n_hypotheses % args.batch_size != 0
    ):
        raise ValueError("batch-size must be positive and divide n-hypotheses")


def validate_estimate(pose: np.ndarray, confidence: float, frame_name: str) -> None:
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise RuntimeError(f"FoundationPose returned an invalid pose for {frame_name}")
    if not np.isfinite(confidence):
        raise RuntimeError(f"FoundationPose returned non-finite confidence for {frame_name}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    pairs = discover_pairs(args.rgb_dir, args.depth_dir)
    intrinsics = load_intrinsics(args.intrinsics)

    first_rgb = load_rgb(pairs[0][0])
    height, width = first_rgb.shape[:2]
    first_depth = load_depth(pairs[0][1], (height, width))
    first_mask = load_mask(args.first_mask, (height, width))

    from foundation_pose_nvidia import (
        Estimator,
        EstimatorOptions,
        RgbdFrame,
        RuntimeConfig,
        load_library,
    )

    args.engine_cache_dir.mkdir(parents=True, exist_ok=True)
    options = EstimatorOptions.from_env(
        args.cad,
        refine_model_path=args.refine_model_path,
        score_model_path=args.score_model_path,
        engine_cache_dir=args.engine_cache_dir,
        device_id=args.device_id,
        mesh_unit_scale=args.mesh_unit_scale,
    )
    config = RuntimeConfig(
        max_image_width=width,
        max_image_height=height,
        n_hypotheses=args.n_hypotheses,
        batch_size=args.batch_size,
    )
    library = load_library(args.library, print_build_info=True)

    poses: list[np.ndarray] = []
    confidences: list[float] = []
    records: list[dict[str, Any]] = []
    with Estimator(
        options,
        config,
        library=library,
        prepare_batch=args.n_hypotheses,
    ) as estimator:
        for frame_idx, (rgb_path, depth_path) in enumerate(pairs):
            if frame_idx == 0:
                rgb = first_rgb
                depth = first_depth
                result = estimator.register(RgbdFrame(rgb, depth, intrinsics, first_mask))
                phase = "register"
            else:
                rgb = load_rgb(rgb_path)
                if rgb.shape[:2] != (height, width):
                    raise ValueError(
                        f"All RGB frames must have shape {(height, width)}: "
                        f"{rgb_path} has {rgb.shape[:2]}"
                    )
                depth = load_depth(depth_path, (height, width))
                result = estimator.track(RgbdFrame(rgb, depth, intrinsics))
                phase = "track"

            validate_estimate(result.pose, result.score, rgb_path.name)
            poses.append(np.asarray(result.pose, dtype=np.float32))
            confidences.append(float(result.score))
            records.append(
                {
                    "frame_idx": frame_idx,
                    "rgb": rgb_path.name,
                    "depth": depth_path.name,
                    "phase": phase,
                    "confidence": float(result.score),
                    "elapsed_s": float(result.elapsed_s),
                }
            )
            translation = result.pose[:3, 3]
            print(
                f"{frame_idx + 1}/{len(pairs)} {phase:8s} "
                f"score={result.score:.4f} "
                f"t=({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}) m"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pose_array = np.stack(poses).astype(np.float32, copy=False)
    confidence_array = np.asarray(confidences, dtype=np.float32)
    np.save(args.output_dir / "T_camera_object.npy", pose_array, allow_pickle=False)
    np.save(args.output_dir / "confidence.npy", confidence_array, allow_pickle=False)
    manifest = {
        "backend": "foundation_pose_nvidia",
        "pose_convention": "T_camera_object maps object-frame coordinates into camera frame",
        "translation_unit": "metre",
        "cad": str(args.cad.resolve()),
        "mesh_unit_scale": args.mesh_unit_scale,
        "intrinsics": str(args.intrinsics.resolve()),
        "shape": {
            "T_camera_object": list(pose_array.shape),
            "confidence": list(confidence_array.shape),
        },
        "frames": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(pairs)} poses to {args.output_dir / 'T_camera_object.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
