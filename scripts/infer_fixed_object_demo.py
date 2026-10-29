#!/usr/bin/env python3
"""Infer a review-only grasp from one fixed-object RGB-D hand demonstration.

This adapter deliberately has no CAD/object-pose input.  It treats the calibrated
camera optical frame as the fixed reference for one recording, reconstructs the
visible object surface from clean frames, reuses the core pinch-intent estimator,
and reports canonical TCP targets in the FR3 base frame.  It never emits a robot
command-frame target and never authorizes execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from demo2gripper.fixed_object import (
    backproject_masked_depth,
    cloud_motion_metrics,
    infer_fixed_camera_intent,
    voxel_downsample,
)
from demo2gripper.franka import make_tcp_targets, validate_rigid_transform
from demo2gripper.geometry import transform_points
from demo2gripper.intent import INDEX_TIP, THUMB_TIP, WRIST, IntentConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
HAND_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rgb-dir", required=True, type=Path)
    parser.add_argument("--depth-dir", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--object-mask-dir", required=True, type=Path)
    parser.add_argument("--hand-mask-dir", required=True, type=Path)
    parser.add_argument("--hand-joints-camera-m", required=True, type=Path)
    parser.add_argument("--hand-confidence", required=True, type=Path)
    parser.add_argument("--hand-keypoints-2d", type=Path)
    parser.add_argument("--timestamps-s", required=True, type=Path)
    parser.add_argument("--t-base-camera", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-prefix-frames", type=int, default=45)
    parser.add_argument("--reference-suffix-frames", type=int, default=45)
    parser.add_argument("--mask-erode-px", type=int, default=2)
    parser.add_argument("--hand-exclusion-px", type=int, default=8)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=1.5)
    parser.add_argument("--voxel-size-m", type=float, default=0.003)
    parser.add_argument("--min-cloud-points", type=int, default=200)
    parser.add_argument("--max-contact-distance-m", type=float, default=0.020)
    parser.add_argument("--min-joint-confidence", type=float, default=0.5)
    parser.add_argument("--min-pinch-width-m", type=float, default=0.005)
    parser.add_argument("--max-jaw-width-m", type=float, default=0.050)
    parser.add_argument("--pregrasp-clearance-m", type=float, default=0.004)
    parser.add_argument("--min-stable-frames", type=int, default=10)
    parser.add_argument("--min-stable-duration-s", type=float, default=0.5)
    parser.add_argument("--approach-distance-m", type=float, default=0.10)
    parser.add_argument("--lift-height-m", type=float, default=0.05)
    parser.add_argument("--max-object-centroid-shift-m", type=float, default=0.010)
    parser.add_argument("--max-object-median-residual-m", type=float, default=0.008)
    parser.add_argument("--max-object-p95-residual-m", type=float, default=0.025)
    return parser.parse_args()


def numeric_paths(directory: Path, suffixes: set[str]) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"directory does not exist: {directory}")
    paths = [path for path in directory.iterdir() if path.suffix.lower() in suffixes]
    try:
        paths.sort(key=lambda path: int(path.stem))
    except ValueError as exc:
        raise ValueError(f"frame filename stems must be integers in {directory}") from exc
    if not paths:
        raise ValueError(f"no matching frames in {directory}")
    return paths


def indexed_paths(directory: Path, suffixes: set[str], frame_stems: list[str]) -> list[Path]:
    available = {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }
    missing = [stem for stem in frame_stems if stem not in available]
    if missing:
        raise ValueError(f"missing frames in {directory}: {', '.join(missing[:5])}")
    return [available[stem] for stem in frame_stems]


def morphology(mask: np.ndarray, radius: int, operation: str) -> np.ndarray:
    if radius == 0:
        return mask > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    code = cv2.MORPH_ERODE if operation == "erode" else cv2.MORPH_DILATE
    return cv2.morphologyEx((mask > 0).astype(np.uint8), code, kernel) > 0


def build_reference_cloud(
    frame_indices: range,
    depth_paths: list[Path],
    object_mask_paths: list[Path],
    hand_mask_paths: list[Path],
    intrinsics: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[int]]:
    frame_clouds: list[np.ndarray] = []
    point_counts: list[int] = []
    expected_shape: tuple[int, int] | None = None
    for frame_index in frame_indices:
        depth = np.asarray(np.load(depth_paths[frame_index], allow_pickle=False), dtype=float)
        object_mask = cv2.imread(str(object_mask_paths[frame_index]), cv2.IMREAD_GRAYSCALE)
        hand_mask = cv2.imread(str(hand_mask_paths[frame_index]), cv2.IMREAD_GRAYSCALE)
        if object_mask is None or hand_mask is None:
            raise ValueError(f"could not read masks for frame {frame_index}")
        if expected_shape is None:
            expected_shape = depth.shape
        shapes_match = object_mask.shape == depth.shape and hand_mask.shape == depth.shape
        if depth.shape != expected_shape or not shapes_match:
            raise ValueError(f"RGB-D/mask shape mismatch at frame {frame_index}")

        clean_object = morphology(object_mask, args.mask_erode_px, "erode")
        excluded_hand = morphology(hand_mask, args.hand_exclusion_px, "dilate")
        points = backproject_masked_depth(
            depth,
            intrinsics,
            clean_object,
            exclude_mask=excluded_hand,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
        point_counts.append(len(points))
        if len(points):
            frame_clouds.append(voxel_downsample(points, args.voxel_size_m))

    if not frame_clouds:
        return np.empty((0, 3), dtype=float), point_counts
    return voxel_downsample(np.concatenate(frame_clouds), args.voxel_size_m), point_counts


def angular_residuals(directions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(directions, axis=1)
    valid = norms > 1e-9
    if not np.any(valid):
        return np.asarray([np.inf])
    unit = directions[valid] / norms[valid, None]
    return np.arccos(np.clip(unit @ reference, -1.0, 1.0))


def quality_metrics(
    joints: np.ndarray,
    object_points: np.ndarray,
    start: int,
    end: int,
    closing_axis: np.ndarray,
    approach_axis: np.ndarray,
) -> dict[str, float]:
    tree = cKDTree(object_points)
    tips = joints[start:end, [THUMB_TIP, INDEX_TIP]]
    distances, indices = tree.query(tips.reshape(-1, 3), k=1)
    projected = object_points[indices].reshape(-1, 2, 3)
    centers = projected.mean(axis=1)
    center_median = np.median(centers, axis=0)
    center_residuals = np.linalg.norm(centers - center_median, axis=1)
    closing = projected[:, 1] - projected[:, 0]
    approach = centers - joints[start:end, WRIST]
    widths = np.abs(closing @ closing_axis)
    closing_angles = angular_residuals(closing, closing_axis)
    approach_angles = angular_residuals(approach, approach_axis)
    return {
        "max_tip_surface_distance_m": float(np.max(distances)),
        "contact_center_rms_m": float(np.sqrt(np.mean(center_residuals**2))),
        "contact_center_max_m": float(np.max(center_residuals)),
        "width_median_m": float(np.median(widths)),
        "width_p95_m": float(np.quantile(widths, 0.95)),
        "width_mad_m": float(np.median(np.abs(widths - np.median(widths)))),
        "closing_angle_p95_deg": float(np.rad2deg(np.quantile(closing_angles, 0.95))),
        "approach_angle_p95_deg": float(np.rad2deg(np.quantile(approach_angles, 0.95))),
    }


def project(points: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    pixels = np.full((len(points), 2), np.nan)
    homogeneous = points[valid] @ intrinsics.T
    pixels[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, valid


def pixel(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def draw_preview(
    rgb_path: Path,
    object_mask_path: Path,
    hand_mask_path: Path,
    keypoints: np.ndarray | None,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    camera_tcp: np.ndarray,
    contact_points: np.ndarray,
    approach_distance_m: float,
    base_tcp: np.ndarray,
    width_m: float,
    output_path: Path,
) -> None:
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    object_mask = cv2.imread(str(object_mask_path), cv2.IMREAD_GRAYSCALE)
    hand_mask = cv2.imread(str(hand_mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or object_mask is None or hand_mask is None:
        raise ValueError("could not load preview RGB/masks")
    for mask, color in ((object_mask, (255, 180, 20)), (hand_mask, (220, 60, 220))):
        active = mask > 0
        tint = np.asarray(color, dtype=np.float32)
        image[active] = np.clip(0.58 * image[active] + 0.42 * tint, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 2)

    if keypoints is not None:
        valid = np.isfinite(keypoints).all(axis=1)
        for first, second in HAND_EDGES:
            confident = confidence[first] >= 0.5 and confidence[second] >= 0.5
            if valid[first] and valid[second] and confident:
                cv2.line(image, pixel(keypoints[first]), pixel(keypoints[second]), (0, 240, 255), 2)
        for index, point in enumerate(keypoints):
            if valid[index] and confidence[index] >= 0.5:
                color = (0, 0, 255) if index in (THUMB_TIP, INDEX_TIP) else (255, 230, 80)
                cv2.circle(image, pixel(point), 4, color, -1, cv2.LINE_AA)

    contact_pixels, contact_valid = project(contact_points, intrinsics)
    if np.all(contact_valid):
        cv2.line(image, pixel(contact_pixels[0]), pixel(contact_pixels[1]), (20, 255, 20), 4)
        for point in contact_pixels:
            cv2.circle(image, pixel(point), 7, (20, 255, 20), -1, cv2.LINE_AA)

    center = camera_tcp[:3, 3]
    axes = np.vstack(
        [
            center,
            center + 0.04 * camera_tcp[:3, 0],
            center + 0.04 * camera_tcp[:3, 1],
            center + 0.04 * camera_tcp[:3, 2],
            center - approach_distance_m * camera_tcp[:3, 2],
        ]
    )
    axis_pixels, axis_valid = project(axes, intrinsics)
    if axis_valid[0]:
        origin = pixel(axis_pixels[0])
        for index, color in ((1, (0, 0, 255)), (2, (0, 220, 0)), (3, (255, 80, 40))):
            if axis_valid[index]:
                cv2.arrowedLine(image, origin, pixel(axis_pixels[index]), color, 3, tipLength=0.2)
        if axis_valid[4]:
            cv2.arrowedLine(
                image, pixel(axis_pixels[4]), origin, (255, 255, 255), 3, tipLength=0.16
            )

    cv2.rectangle(image, (0, 0), (image.shape[1], 82), (15, 15, 15), -1)
    cv2.putText(
        image,
        "FIXED OBJECT / NO CAD / REVIEW ONLY",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (40, 230, 255),
        2,
        cv2.LINE_AA,
    )
    xyz = base_tcp[:3, 3]
    cv2.putText(
        image,
        f"base TCP xyz = {xyz[0]:+.3f} {xyz[1]:+.3f} {xyz[2]:+.3f} m",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"demonstrated width = {width_m * 1000:.1f} mm",
        (12, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"could not write preview {output_path}")


def matrices_to_lists(values: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
    return {name: matrix.tolist() for name, matrix in values.items()}


def main() -> int:
    args = parse_args()
    for name in (
        "reference_prefix_frames",
        "reference_suffix_frames",
        "min_cloud_points",
        "min_stable_frames",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("mask_erode_px", "hand_exclusion_px"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rgb_paths = numeric_paths(args.rgb_dir, IMAGE_SUFFIXES)
    stems = [path.stem for path in rgb_paths]
    depth_paths = indexed_paths(args.depth_dir, {".npy"}, stems)
    object_mask_paths = indexed_paths(args.object_mask_dir, {".png"}, stems)
    hand_mask_paths = indexed_paths(args.hand_mask_dir, {".png"}, stems)
    frame_count = len(rgb_paths)
    if args.reference_prefix_frames + args.reference_suffix_frames > frame_count:
        raise ValueError("reference prefix + suffix frames exceed the recording length")

    intrinsics = np.asarray(np.load(args.intrinsics, allow_pickle=False), dtype=float)
    joints = np.asarray(np.load(args.hand_joints_camera_m, allow_pickle=False), dtype=float)
    confidence = np.asarray(np.load(args.hand_confidence, allow_pickle=False), dtype=float)
    timestamps = np.asarray(np.load(args.timestamps_s, allow_pickle=False), dtype=float)
    keypoints = (
        np.asarray(np.load(args.hand_keypoints_2d, allow_pickle=False), dtype=float)
        if args.hand_keypoints_2d
        else None
    )
    if joints.shape != (frame_count, 21, 3):
        raise ValueError(f"hand joints must have shape ({frame_count}, 21, 3), got {joints.shape}")
    if confidence.shape != (frame_count, 21):
        raise ValueError(
            f"hand confidence must have shape ({frame_count}, 21), got {confidence.shape}"
        )
    if timestamps.shape != (frame_count,):
        raise ValueError(f"timestamps must have shape ({frame_count},), got {timestamps.shape}")
    if keypoints is not None and keypoints.shape != (frame_count, 21, 2):
        raise ValueError(f"hand keypoints must have shape ({frame_count}, 21, 2)")

    prefix_indices = range(args.reference_prefix_frames)
    suffix_indices = range(frame_count - args.reference_suffix_frames, frame_count)
    prefix_cloud, prefix_counts = build_reference_cloud(
        prefix_indices,
        depth_paths,
        object_mask_paths,
        hand_mask_paths,
        intrinsics,
        args,
    )
    suffix_cloud, suffix_counts = build_reference_cloud(
        suffix_indices,
        depth_paths,
        object_mask_paths,
        hand_mask_paths,
        intrinsics,
        args,
    )
    if len(prefix_cloud) < args.min_cloud_points or len(suffix_cloud) < args.min_cloud_points:
        raise ValueError(
            "insufficient clean object depth in reference frames: "
            f"prefix={len(prefix_cloud)}, suffix={len(suffix_cloud)}"
        )

    motion = cloud_motion_metrics(prefix_cloud, suffix_cloud)
    if motion.centroid_shift_m > args.max_object_centroid_shift_m:
        raise ValueError(
            f"object appears to move: centroid shift {motion.centroid_shift_m * 1000:.1f} mm"
        )
    if motion.median_nearest_distance_m > args.max_object_median_residual_m:
        raise ValueError(
            "object reference clouds disagree: median residual "
            f"{motion.median_nearest_distance_m * 1000:.1f} mm"
        )
    if motion.p95_nearest_distance_m > args.max_object_p95_residual_m:
        raise ValueError(
            "object reference clouds disagree: p95 residual "
            f"{motion.p95_nearest_distance_m * 1000:.1f} mm"
        )

    object_points = voxel_downsample(
        np.concatenate((prefix_cloud, suffix_cloud)), args.voxel_size_m
    )
    np.save(args.output_dir / "object_points_camera_m.npy", object_points, allow_pickle=False)

    intent = infer_fixed_camera_intent(
        joints,
        confidence,
        timestamps,
        object_points,
        IntentConfig(
            max_tip_object_distance_m=args.max_contact_distance_m,
            min_joint_confidence=args.min_joint_confidence,
            min_pinch_distance_m=args.min_pinch_width_m,
            max_pinch_distance_m=args.max_jaw_width_m,
            min_stable_frames=args.min_stable_frames,
        ),
    )
    start, end = intent.stable_interval
    stable_duration_s = float(timestamps[end - 1] - timestamps[start])
    if stable_duration_s < args.min_stable_duration_s:
        raise ValueError(
            f"stable contact lasts {stable_duration_s:.3f}s, below "
            f"{args.min_stable_duration_s:.3f}s"
        )

    camera_tcp = validate_rigid_transform(
        np.asarray(intent.T_object_gripper_tcp, dtype=float), name="T_camera_gripper_tcp"
    )
    base_camera = validate_rigid_transform(
        np.load(args.t_base_camera, allow_pickle=False), name="T_base_camera"
    )
    base_tcp = base_camera @ camera_tcp
    quality = quality_metrics(
        joints,
        object_points,
        start,
        end,
        camera_tcp[:3, 0],
        camera_tcp[:3, 2],
    )
    required_opening_m = quality["width_p95_m"] + args.pregrasp_clearance_m
    if required_opening_m > args.max_jaw_width_m:
        raise ValueError(
            f"demonstrated opening needs {required_opening_m * 1000:.1f} mm including "
            f"clearance, above Hand-E limit {args.max_jaw_width_m * 1000:.1f} mm"
        )

    tcp_targets = make_tcp_targets(
        base_camera,
        camera_tcp,
        approach_distance_m=args.approach_distance_m,
        lift_height_m=args.lift_height_m,
    )
    contact_camera = np.asarray(intent.contact_points_object_m, dtype=float)
    contact_base = transform_points(base_camera, contact_camera)
    stable_midpoint = (start + end - 1) // 2
    preview_path = args.output_dir / "preview.png"
    draw_preview(
        rgb_paths[stable_midpoint],
        object_mask_paths[stable_midpoint],
        hand_mask_paths[stable_midpoint],
        None if keypoints is None else keypoints[stable_midpoint],
        confidence[stable_midpoint],
        intrinsics,
        camera_tcp,
        contact_camera,
        args.approach_distance_m,
        base_tcp,
        intent.width_m,
        preview_path,
    )

    payload = {
        "schema": "demo2gripper.fixed_object_no_cad.v1",
        "mode": "fixed_object_no_cad",
        "status": "review_only",
        "execution_authorized": False,
        "reference_frame": "OpenCV color optical frame fixed for this recording",
        "base_frame": "FR3 O/base from T_base_camera calibration contract",
        "frame_convention": "T_parent_child, column vectors, metres; TCP +X closing, +Z approach",
        "T_base_camera": base_camera.tolist(),
        "T_camera_gripper_tcp": camera_tcp.tolist(),
        "T_base_gripper_tcp": base_tcp.tolist(),
        "canonical_tcp_targets_base": matrices_to_lists(tcp_targets),
        "contact_points_camera_m": contact_camera.tolist(),
        "contact_points_base_m": contact_base.tolist(),
        "closing_axis_camera": camera_tcp[:3, 0].tolist(),
        "approach_axis_camera": camera_tcp[:3, 2].tolist(),
        "demonstrated_contact_width_m": float(intent.width_m),
        "pregrasp_width_m": float(required_opening_m),
        "approach_distance_m": args.approach_distance_m,
        "lift_height_m": args.lift_height_m,
        "stable_interval": [start, end],
        "stable_duration_s": stable_duration_s,
        "support_frames": intent.support_frames,
        "preview_frame": stable_midpoint,
        "quality": {
            **quality,
            "object_centroid_shift_m": motion.centroid_shift_m,
            "object_cloud_median_residual_m": motion.median_nearest_distance_m,
            "object_cloud_p95_residual_m": motion.p95_nearest_distance_m,
            "object_point_count": len(object_points),
            "prefix_valid_depth_per_frame_min": min(prefix_counts),
            "suffix_valid_depth_per_frame_min": min(suffix_counts),
        },
        "limitations": [
            "The object and D435 must remain fixed for the whole recording.",
            "Contact is inferred from RGB-D fingertip proximity; force contact is not measured.",
            "The current T_base_camera is operator-accepted best-effort calibration.",
            "Hand-E flange-to-TCP and collision geometry are intentionally deferred.",
            "These canonical TCP targets must not be sent to the robot.",
        ],
        "artifacts": {
            "preview": preview_path.name,
            "object_points_camera_m": "object_points_camera_m.npy",
        },
        "provenance": {
            "rgb_dir": str(args.rgb_dir.resolve()),
            "depth_dir": str(args.depth_dir.resolve()),
            "object_mask_dir": str(args.object_mask_dir.resolve()),
            "hand_mask_dir": str(args.hand_mask_dir.resolve()),
            "hand_joints_camera_m": str(args.hand_joints_camera_m.resolve()),
            "hand_confidence": str(args.hand_confidence.resolve()),
            "timestamps_s": str(args.timestamps_s.resolve()),
            "T_base_camera": str(args.t_base_camera.resolve()),
        },
    }
    result_path = args.output_dir / "fixed_object_grasp.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"fixed-object grasp: frames [{start}, {end}), width={intent.width_m * 1000:.1f} mm, "
        f"base xyz={base_tcp[0, 3]:.4f} {base_tcp[1, 3]:.4f} {base_tcp[2, 3]:.4f} m"
    )
    print(f"review-only result: {result_path}")
    print(f"preview: {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
