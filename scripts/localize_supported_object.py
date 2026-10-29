#!/usr/bin/env python3
"""Localize one stationary object above a known support in aligned RGB-D frames.

The output is a review-only base-frame translation anchor.  This command never sends
robot or gripper commands and does not estimate object orientation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from demo2gripper.supported_object import (
    FrameEvidence,
    SupportedObjectConfig,
    localize_supported_object,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing K.npy, rgb/, and aligned metre-valued depth/",
    )
    parser.add_argument("--t-base-camera", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--workspace-min-m",
        required=True,
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Inclusive lower XYZ bounds in the FR3 base frame",
    )
    parser.add_argument(
        "--workspace-max-m",
        required=True,
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Inclusive upper XYZ bounds in the FR3 base frame",
    )
    support = parser.add_mutually_exclusive_group(required=True)
    support.add_argument(
        "--support-z-m",
        type=float,
        help="Known top surface Z in the FR3 base frame",
    )
    support.add_argument(
        "--support-band-m",
        type=float,
        nargs=2,
        metavar=("Z_MIN", "Z_MAX"),
        help="Measured support Z band; its upper edge is used conservatively",
    )
    parser.add_argument(
        "--min-height-above-support-m",
        required=True,
        type=float,
        help="Minimum accepted object-point height above the support top",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=1.5)
    parser.add_argument("--min-component-points", type=int, default=150)
    parser.add_argument("--min-dominance-ratio", type=float, default=1.8)
    parser.add_argument("--morphology-radius-px", type=int, default=1)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--max-center-deviation-m", type=float, default=0.008)
    parser.add_argument("--max-point-count-cv", type=float, default=0.25)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output directory after localization passes",
    )
    return parser.parse_args()


def _numeric_paths(directory: Path, suffixes: set[str]) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"directory does not exist: {directory}")
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    try:
        paths.sort(key=lambda path: int(path.stem))
    except ValueError as exc:
        raise ValueError(f"frame filename stems must be integers in {directory}") from exc
    if not paths:
        raise ValueError(f"no matching frames in {directory}")
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError(f"duplicate numeric frame stems in {directory}")
    return paths


def _load_inputs(
    input_dir: Path,
) -> tuple[list[Path], list[Path], list[np.ndarray], np.ndarray, np.ndarray]:
    intrinsics_path = input_dir / "K.npy"
    if not intrinsics_path.is_file():
        raise ValueError(f"missing intrinsics: {intrinsics_path}")
    rgb_paths = _numeric_paths(input_dir / "rgb", IMAGE_SUFFIXES)
    depth_paths = _numeric_paths(input_dir / "depth", {".npy"})
    if [path.stem for path in rgb_paths] != [path.stem for path in depth_paths]:
        raise ValueError("RGB and depth frame stems differ")

    depths: list[np.ndarray] = []
    final_rgb: np.ndarray | None = None
    expected_shape: tuple[int, int] | None = None
    for rgb_path, depth_path in zip(rgb_paths, depth_paths, strict=True):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"could not read RGB frame: {rgb_path}")
        depth = np.load(depth_path, allow_pickle=False)
        if depth.ndim != 2:
            raise ValueError(f"depth frame must be two-dimensional: {depth_path}")
        if rgb.shape[:2] != depth.shape:
            raise ValueError(
                f"RGB/depth shape mismatch for frame {rgb_path.stem}: "
                f"{rgb.shape[:2]} versus {depth.shape}"
            )
        if expected_shape is None:
            expected_shape = depth.shape
        if depth.shape != expected_shape:
            raise ValueError(f"image shape changed at frame {rgb_path.stem}")
        depths.append(depth)
        final_rgb = rgb
    assert final_rgb is not None
    intrinsics = np.load(intrinsics_path, allow_pickle=False)
    return rgb_paths, depth_paths, depths, intrinsics, final_rgb


def _component_json(frame: FrameEvidence) -> dict[str, object]:
    component = frame.selected
    return {
        "selected_component_label": component.label,
        "eligible_component_count": frame.eligible_component_count,
        "point_count": component.point_count,
        "second_largest_point_count": frame.second_largest_point_count,
        "dominance_ratio": frame.dominance_ratio,
        "center_base_m": component.center_base_m.tolist(),
        "bounds_base_m": {
            "min": component.bounds_min_base_m.tolist(),
            "max": component.bounds_max_base_m.tolist(),
        },
        "robust_bounds_p01_p99_base_m": {
            "min": component.robust_bounds_min_base_m.tolist(),
            "max": component.robust_bounds_max_base_m.tolist(),
        },
        "bbox_xywh_px": list(component.bbox_xywh),
        "center_uv_px": component.center_uv_px.tolist(),
    }


def _draw_overlay(rgb: np.ndarray, mask: np.ndarray, final_frame: FrameEvidence) -> np.ndarray:
    overlay = rgb.copy()
    tint = np.zeros_like(rgb)
    tint[:, :, 1] = 255
    selected = mask.astype(bool)
    overlay[selected] = cv2.addWeighted(rgb[selected], 0.35, tint[selected], 0.65, 0.0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    x, y, width, height = final_frame.selected.bbox_xywh
    cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), (255, 0, 255), 2)
    u, v = np.rint(final_frame.selected.center_uv_px).astype(int)
    cv2.drawMarker(overlay, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.putText(
        overlay,
        f"supported object: {final_frame.selected.point_count} depth points",
        (max(4, x), max(22, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def _write_outputs(
    output_dir: Path,
    report: dict[str, object],
    mask: np.ndarray,
    overlay: np.ndarray,
    selected_points_base_m: dict[str, np.ndarray],
    *,
    overwrite: bool,
) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists() and not overwrite:
        raise ValueError(f"output directory already exists; use --overwrite: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        with (staging / "live_object_anchor.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        if not cv2.imwrite(str(staging / "final_object_mask.png"), mask.astype(np.uint8) * 255):
            raise RuntimeError("failed to write final object mask")
        if not cv2.imwrite(str(staging / "final_object_overlay.png"), overlay):
            raise RuntimeError("failed to write final object overlay")
        np.savez_compressed(
            staging / "selected_object_points_base_m.npz", **selected_points_base_m
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run(args: argparse.Namespace) -> Path:
    input_dir = args.input_dir.resolve()
    rgb_paths, depth_paths, depths, intrinsics, final_rgb = _load_inputs(input_dir)
    T_base_camera = np.load(args.t_base_camera, allow_pickle=False)
    if args.support_band_m is not None:
        support_min, support_max = args.support_band_m
        if not np.isfinite([support_min, support_max]).all() or support_min >= support_max:
            raise ValueError("support band must contain finite values satisfying min < max")
        support_top = support_max
        support_specification: dict[str, object] = {
            "mode": "band_upper_edge",
            "band_base_z_m": [support_min, support_max],
            "effective_top_base_z_m": support_top,
        }
    else:
        support_top = args.support_z_m
        support_specification = {
            "mode": "known_plane",
            "top_base_z_m": support_top,
        }

    config = SupportedObjectConfig(
        workspace_min_base_m=np.asarray(args.workspace_min_m, dtype=float),
        workspace_max_base_m=np.asarray(args.workspace_max_m, dtype=float),
        support_top_base_z_m=float(support_top),
        min_height_above_support_m=args.min_height_above_support_m,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        min_component_points=args.min_component_points,
        min_dominance_ratio=args.min_dominance_ratio,
        morphology_radius_px=args.morphology_radius_px,
        min_frames=args.min_frames,
        max_center_deviation_m=args.max_center_deviation_m,
        max_point_count_cv=args.max_point_count_cv,
    )
    result = localize_supported_object(depths, intrinsics, T_base_camera, config)

    frames = []
    selected_points_base_m: dict[str, np.ndarray] = {}
    for index, (rgb_path, depth_path, evidence) in enumerate(
        zip(rgb_paths, depth_paths, result.frames, strict=True)
    ):
        points_key = f"frame_{index:06d}"
        selected_points_base_m[points_key] = evidence.selected.points_base_m
        frame = {
            "index": index,
            "stem": rgb_path.stem,
            "rgb": str(rgb_path.resolve()),
            "depth": str(depth_path.resolve()),
            "selected_points_base_key": points_key,
        }
        frame.update(_component_json(evidence))
        frames.append(frame)
    finite_dominance = [
        frame.dominance_ratio
        for frame in result.frames
        if frame.dominance_ratio is not None
    ]
    report: dict[str, object] = {
        "schema": "demo2gripper.live_supported_object_anchor.v1",
        "status": "accepted",
        "artifact_role": "review_only_translation_anchor",
        "authorized_for_robot_execution": False,
        "coordinate_frame": "FR3 base frame from T_base_camera",
        "anchor_semantics": (
            "coordinate-wise median of per-frame medians of the visible, "
            "above-support object surface; not a 6D pose or geometric-volume center"
        ),
        "center_base_m": result.center_base_m.tolist(),
        "robust_bounds_p01_p99_base_m": {
            "min": result.bounds_min_base_m.tolist(),
            "max": result.bounds_max_base_m.tolist(),
        },
        "stability": {
            "frame_count": len(result.frames),
            "center_rms_deviation_m": result.center_rms_deviation_m,
            "center_max_deviation_m": result.center_max_deviation_m,
            "point_count_cv": result.point_count_cv,
            "minimum_finite_dominance_ratio": (
                min(finite_dominance) if finite_dominance else None
            ),
        },
        "quality": {
            "accepted": True,
            "all_input_frames_accepted": True,
            "dominance_check": "passed",
            "cross_frame_center_check": "passed",
            "point_count_stability_check": "passed",
        },
        "support": support_specification,
        "parameters": {
            "workspace_min_base_m": list(args.workspace_min_m),
            "workspace_max_base_m": list(args.workspace_max_m),
            "min_height_above_support_m": args.min_height_above_support_m,
            "effective_object_z_threshold_base_m": (
                support_top + args.min_height_above_support_m
            ),
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
            "min_component_points": args.min_component_points,
            "min_dominance_ratio": args.min_dominance_ratio,
            "morphology_radius_px": args.morphology_radius_px,
            "min_frames": args.min_frames,
            "max_center_deviation_m": args.max_center_deviation_m,
            "max_point_count_cv": args.max_point_count_cv,
        },
        "inputs": {
            "input_dir": str(input_dir),
            "intrinsics": str((input_dir / "K.npy").resolve()),
            "T_base_camera": str(args.t_base_camera.resolve()),
            "T_base_camera_value": np.asarray(T_base_camera, dtype=float).tolist(),
        },
        "frames": frames,
        "outputs": {
            "mask": "final_object_mask.png",
            "overlay": "final_object_overlay.png",
            "selected_points_base": "selected_object_points_base_m.npz",
        },
    }
    overlay = _draw_overlay(final_rgb, result.final_mask, result.frames[-1])
    _write_outputs(
        args.output_dir,
        report,
        result.final_mask,
        overlay,
        selected_points_base_m,
        overwrite=args.overwrite,
    )
    return args.output_dir.resolve() / "live_object_anchor.json"


def main() -> None:
    args = parse_args()
    try:
        output = run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"localization rejected: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"accepted stable object anchor: {output}")


if __name__ == "__main__":
    main()
