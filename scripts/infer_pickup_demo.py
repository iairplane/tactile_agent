#!/usr/bin/env python3
"""Infer and validate a review-only CAD-free pickup demonstration.

The adapter reconstructs a clean object cloud before contact, cuts grasp
inference off before the detected lift, validates a sustained object lift and
hand/object motion coupling, and writes canonical TCP poses in camera and FR3
base coordinates.  It never creates a gripper command-frame target and never
authorizes robot execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from demo2gripper.fixed_object import backproject_masked_depth, voxel_downsample
from demo2gripper.franka import make_tcp_targets, validate_rigid_transform
from demo2gripper.geometry import robust_direction_mean, robust_mean, transform_points
from demo2gripper.intent import INDEX_TIP, THUMB_TIP, WRIST, IntentConfig
from demo2gripper.pickup import (
    align_hand_joints_to_reference_translation,
    assess_approach_for_fixed_object_replay,
    detect_pickup_events,
    infer_pre_lift_pickup_intent,
    measure_hand_object_coupling,
    measure_prelift_object_motion,
    validate_contact_before_lift,
)

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
    parser.add_argument("--reference-prefix-frames", type=int, default=90)
    parser.add_argument("--post-placement-frames", type=int, default=45)
    parser.add_argument("--pre-lift-guard-frames", type=int, default=2)
    parser.add_argument("--mask-erode-px", type=int, default=2)
    parser.add_argument("--reference-hand-exclusion-px", type=int, default=8)
    parser.add_argument("--tracking-hand-exclusion-px", type=int, default=4)
    parser.add_argument("--tracking-depth-trim-m", type=float, default=0.08)
    parser.add_argument("--support-ring-inner-px", type=int, default=5)
    parser.add_argument("--support-ring-outer-px", type=int, default=30)
    parser.add_argument("--min-object-height-above-support-m", type=float, default=0.005)
    parser.add_argument("--width-slice-lower-height-fraction", type=float, default=0.60)
    parser.add_argument("--width-slice-upper-height-fraction", type=float, default=0.90)
    parser.add_argument("--max-support-height-motion-p95-m", type=float, default=0.005)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=1.5)
    parser.add_argument("--voxel-size-m", type=float, default=0.003)
    parser.add_argument("--min-reference-cloud-points", type=int, default=200)
    parser.add_argument("--min-tracking-points", type=int, default=500)
    parser.add_argument("--max-contact-distance-m", type=float, default=0.020)
    parser.add_argument("--min-joint-confidence", type=float, default=0.5)
    parser.add_argument("--min-pinch-width-m", type=float, default=0.005)
    parser.add_argument("--max-jaw-width-m", type=float, default=0.050)
    parser.add_argument("--pregrasp-clearance-m", type=float, default=0.004)
    parser.add_argument("--min-jaw-width-margin-m", type=float, default=0.003)
    parser.add_argument("--min-stable-frames", type=int, default=10)
    parser.add_argument("--min-stable-duration-s", type=float, default=0.5)
    parser.add_argument("--max-contact-lift-gap-frames", type=int, default=3)
    parser.add_argument("--approach-lookback-frames", type=int, default=30)
    parser.add_argument("--approach-guard-frames", type=int, default=4)
    parser.add_argument("--min-approach-frames", type=int, default=15)
    parser.add_argument("--min-approach-trajectory-displacement-m", type=float, default=0.010)
    parser.add_argument("--max-approach-trajectory-residual-p95-m", type=float, default=0.015)
    parser.add_argument("--min-approach-signal-to-residual", type=float, default=3.0)
    parser.add_argument("--min-approach-window-count", type=int, default=3)
    parser.add_argument("--max-approach-window-angle-p95-deg", type=float, default=15.0)
    parser.add_argument("--max-reference-motion-p95-m", type=float, default=0.005)
    parser.add_argument("--max-prelift-object-shift-m", type=float, default=0.005)
    parser.add_argument("--prelift-shift-noise-multiplier", type=float, default=6.0)
    parser.add_argument("--min-lift-onset-m", type=float, default=0.008)
    parser.add_argument("--min-lift-onset-frames", type=int, default=5)
    parser.add_argument("--min-airborne-height-m", type=float, default=0.030)
    parser.add_argument("--min-airborne-frames", type=int, default=15)
    parser.add_argument("--min-airborne-duration-s", type=float, default=0.5)
    parser.add_argument("--min-coupling-frames", type=int, default=15)
    parser.add_argument("--max-relative-motion-p95-m", type=float, default=0.015)
    parser.add_argument("--max-hand-object-distance-m", type=float, default=0.040)
    parser.add_argument("--min-moving-step-pairs", type=int, default=10)
    parser.add_argument("--min-velocity-cosine", type=float, default=0.70)
    parser.add_argument("--max-step-mismatch-p95-m", type=float, default=0.010)
    parser.add_argument("--max-contact-center-rms-m", type=float, default=0.010)
    parser.add_argument("--max-raw-closing-angle-p95-deg", type=float, default=15.0)
    parser.add_argument("--approach-distance-m", type=float, default=0.10)
    parser.add_argument("--lift-height-m", type=float, default=0.05)
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


def indexed_paths(directory: Path, suffixes: set[str], stems: list[str]) -> list[Path]:
    available = {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }
    missing = [stem for stem in stems if stem not in available]
    if missing:
        raise ValueError(f"missing frames in {directory}: {', '.join(missing[:5])}")
    return [available[stem] for stem in stems]


def morphology(mask: np.ndarray, radius: int, operation: str) -> np.ndarray:
    if radius == 0:
        return mask > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    code = cv2.MORPH_ERODE if operation == "erode" else cv2.MORPH_DILATE
    return cv2.morphologyEx((mask > 0).astype(np.uint8), code, kernel) > 0


def load_clean_object_points(
    frame_index: int,
    depth_paths: list[Path],
    object_mask_paths: list[Path],
    hand_mask_paths: list[Path],
    intrinsics: np.ndarray,
    *,
    mask_erode_px: int,
    hand_exclusion_px: int,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    depth = np.asarray(np.load(depth_paths[frame_index], allow_pickle=False), dtype=float)
    object_mask = cv2.imread(str(object_mask_paths[frame_index]), cv2.IMREAD_GRAYSCALE)
    hand_mask = cv2.imread(str(hand_mask_paths[frame_index]), cv2.IMREAD_GRAYSCALE)
    if object_mask is None or hand_mask is None:
        raise ValueError(f"could not read masks for frame {frame_index}")
    if object_mask.shape != depth.shape or hand_mask.shape != depth.shape:
        raise ValueError(f"RGB-D/mask shape mismatch at frame {frame_index}")
    clean_object = morphology(object_mask, mask_erode_px, "erode")
    excluded_hand = morphology(hand_mask, hand_exclusion_px, "dilate")
    return backproject_masked_depth(
        depth,
        intrinsics,
        clean_object,
        exclude_mask=excluded_hand,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )


def build_reference_cloud(
    frame_count: int,
    depth_paths: list[Path],
    object_mask_paths: list[Path],
    hand_mask_paths: list[Path],
    intrinsics: np.ndarray,
    base_camera: np.ndarray,
    support_height_base_m: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    clouds: list[np.ndarray] = []
    counts: list[int] = []
    for frame_index in range(frame_count):
        points = load_clean_object_points(
            frame_index,
            depth_paths,
            object_mask_paths,
            hand_mask_paths,
            intrinsics,
            mask_erode_px=args.mask_erode_px,
            hand_exclusion_px=args.reference_hand_exclusion_px,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
        if len(points):
            points_base = transform_points(base_camera, points)
            points = points[
                points_base[:, 2]
                >= support_height_base_m + args.min_object_height_above_support_m
            ]
        counts.append(len(points))
        if len(points):
            clouds.append(voxel_downsample(points, args.voxel_size_m))
    if not clouds:
        return np.empty((0, 3), dtype=float), [], counts
    combined = voxel_downsample(np.concatenate(clouds), args.voxel_size_m)
    return combined, clouds, counts


def estimate_support_height(
    frame_count: int,
    depth_paths: list[Path],
    object_mask_paths: list[Path],
    hand_mask_paths: list[Path],
    intrinsics: np.ndarray,
    base_camera: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, float, list[float]]:
    if args.support_ring_outer_px <= args.support_ring_inner_px:
        raise ValueError("support ring outer radius must exceed its inner radius")
    outer_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * args.support_ring_outer_px + 1, 2 * args.support_ring_outer_px + 1),
    )
    inner_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * args.support_ring_inner_px + 1, 2 * args.support_ring_inner_px + 1),
    )
    per_frame: list[float] = []
    for frame_index in range(frame_count):
        depth = np.asarray(np.load(depth_paths[frame_index], allow_pickle=False), dtype=float)
        object_mask = cv2.imread(str(object_mask_paths[frame_index]), cv2.IMREAD_GRAYSCALE)
        hand_mask = cv2.imread(str(hand_mask_paths[frame_index]), cv2.IMREAD_GRAYSCALE)
        if object_mask is None or hand_mask is None:
            raise ValueError(f"could not read masks for frame {frame_index}")
        object_binary = (object_mask > 0).astype(np.uint8)
        ring = (cv2.dilate(object_binary, outer_kernel) > 0) & ~(
            cv2.dilate(object_binary, inner_kernel) > 0
        )
        ring &= ~morphology(hand_mask, args.reference_hand_exclusion_px, "dilate")
        points = backproject_masked_depth(
            depth,
            intrinsics,
            ring,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
        if len(points) >= args.min_tracking_points:
            per_frame.append(float(np.median(transform_points(base_camera, points)[:, 2])))
    if len(per_frame) < 15:
        raise ValueError("insufficient local support-plane depth in the reference prefix")
    support_height = float(np.median(per_frame))
    residual_p95 = float(np.quantile(np.abs(np.asarray(per_frame) - support_height), 0.95))
    if residual_p95 > args.max_support_height_motion_p95_m:
        raise ValueError(
            "local support height is not stable: p95 deviation "
            f"{residual_p95 * 1000.0:.1f} mm"
        )
    return support_height, residual_p95, per_frame


def largest_component_points(
    depth: np.ndarray,
    object_mask: np.ndarray,
    hand_mask: np.ndarray,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    clean = morphology(object_mask, args.mask_erode_px, "erode")
    clean &= ~morphology(hand_mask, args.tracking_hand_exclusion_px, "dilate")
    component_count, labels, _, _ = cv2.connectedComponentsWithStats(
        clean.astype(np.uint8), 8
    )
    best = np.empty((0, 3), dtype=float)
    for label in range(1, component_count):
        points = backproject_masked_depth(
            depth,
            intrinsics,
            labels == label,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
        if len(points):
            median_depth = np.median(points[:, 2])
            points = points[
                np.abs(points[:, 2] - median_depth) <= args.tracking_depth_trim_m
            ]
        if len(points) > len(best):
            best = points
    return best


def track_object(
    depth_paths: list[Path],
    object_mask_paths: list[Path],
    hand_mask_paths: list[Path],
    intrinsics: np.ndarray,
    base_camera: np.ndarray,
    support_height_base_m: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count = len(depth_paths)
    camera_centers = np.full((frame_count, 3), np.nan, dtype=float)
    base_centers = np.full_like(camera_centers, np.nan)
    point_counts = np.zeros(frame_count, dtype=np.int64)
    for frame_index in range(frame_count):
        depth = np.asarray(np.load(depth_paths[frame_index], allow_pickle=False), dtype=float)
        object_mask = cv2.imread(str(object_mask_paths[frame_index]), cv2.IMREAD_GRAYSCALE)
        hand_mask = cv2.imread(str(hand_mask_paths[frame_index]), cv2.IMREAD_GRAYSCALE)
        if object_mask is None or hand_mask is None:
            raise ValueError(f"could not read masks for frame {frame_index}")
        points = largest_component_points(
            depth, object_mask, hand_mask, intrinsics, args
        )
        if len(points):
            all_base_points = transform_points(base_camera, points)
            keep = (
                all_base_points[:, 2]
                >= support_height_base_m + args.min_object_height_above_support_m
            )
            points = points[keep]
            all_base_points = all_base_points[keep]
        point_counts[frame_index] = len(points)
        if len(points) >= args.min_tracking_points:
            camera_centers[frame_index] = robust_mean(points)
            base_centers[frame_index] = transform_points(
                base_camera, camera_centers[frame_index][None]
            )[0]
    return camera_centers, base_centers, point_counts


def angular_p95(directions: np.ndarray, reference: np.ndarray) -> float:
    norms = np.linalg.norm(directions, axis=1)
    valid = norms > 1e-9
    if not np.any(valid):
        return float("inf")
    unit = directions[valid] / norms[valid, None]
    angles = np.arccos(np.clip(unit @ reference, -1.0, 1.0))
    return float(np.rad2deg(np.quantile(angles, 0.95)))


def contact_quality(
    joints: np.ndarray,
    confidence: np.ndarray,
    object_points: np.ndarray,
    start: int,
    end: int,
    camera_tcp: np.ndarray,
) -> dict[str, float | list[float]]:
    tips = joints[start:end, [THUMB_TIP, INDEX_TIP]]
    widths = np.linalg.norm(tips[:, 1] - tips[:, 0], axis=1)
    midpoints = np.mean(tips, axis=1)
    midpoint = np.median(midpoints, axis=0)
    midpoint_residual = np.linalg.norm(midpoints - midpoint, axis=1)
    raw_closing = tips[:, 1] - tips[:, 0]
    raw_closing_axis = robust_direction_mean(raw_closing, name="raw closing axis")
    raw_approach = midpoints - joints[start:end, WRIST]
    tree = cKDTree(object_points)
    tip_distances = tree.query(tips.reshape(-1, 3), k=1)[0]
    return {
        "max_tip_to_reference_surface_m": float(np.max(tip_distances)),
        "tip_to_reference_surface_p95_m": float(np.quantile(tip_distances, 0.95)),
        "contact_midpoint_rms_m": float(np.sqrt(np.mean(midpoint_residual**2))),
        "contact_midpoint_max_m": float(np.max(midpoint_residual)),
        "pinch_width_q05_m": float(np.quantile(widths, 0.05)),
        "pinch_width_median_m": float(np.median(widths)),
        "pinch_width_p95_m": float(np.quantile(widths, 0.95)),
        "pinch_width_mad_m": float(np.median(np.abs(widths - np.median(widths)))),
        "minimum_tip_confidence": float(
            np.min(confidence[start:end, [THUMB_TIP, INDEX_TIP]])
        ),
        "raw_closing_axis_camera": raw_closing_axis.tolist(),
        "raw_closing_angle_p95_deg": angular_p95(raw_closing, raw_closing_axis),
        "wrist_to_pinch_angle_p95_deg": angular_p95(raw_approach, camera_tcp[:3, 2]),
        "wrist_confidence_min": float(np.min(confidence[start:end, WRIST])),
        "wrist_confidence_median": float(np.median(confidence[start:end, WRIST])),
    }


def reference_spans(clouds: list[np.ndarray], axis: np.ndarray) -> np.ndarray:
    spans = []
    for cloud in clouds:
        projected = cloud @ axis
        spans.append(float(np.quantile(projected, 0.99) - np.quantile(projected, 0.01)))
    return np.asarray(spans, dtype=float)


def single_view_reference_spans(
    frame_count: int,
    depth_paths: list[Path],
    object_mask_paths: list[Path],
    hand_mask_paths: list[Path],
    intrinsics: np.ndarray,
    axis: np.ndarray,
    base_camera: np.ndarray,
    support_height_base_m: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure an upper-slice single-view p01-p99 span diagnostic.

    The cloud used for contact proximity is eroded to avoid mask-edge depth
    leakage.  This diagnostic uses the full silhouette and robust percentiles,
    then excludes the support-level region where reflections dominate.  It is
    not a validated width at the demonstrated grasp plane.
    """

    spans: list[float] = []
    heights: list[float] = []
    for frame_index in range(frame_count):
        points = load_clean_object_points(
            frame_index,
            depth_paths,
            object_mask_paths,
            hand_mask_paths,
            intrinsics,
            mask_erode_px=0,
            hand_exclusion_px=args.reference_hand_exclusion_px,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
        if len(points) < args.min_tracking_points:
            continue
        median_depth = np.median(points[:, 2])
        points = points[
            np.abs(points[:, 2] - median_depth) <= args.tracking_depth_trim_m
        ]
        points_base = transform_points(base_camera, points)
        height = points_base[:, 2] - support_height_base_m
        object_height = float(np.quantile(height, 0.99))
        keep = (
            (height >= args.width_slice_lower_height_fraction * object_height)
            & (height <= args.width_slice_upper_height_fraction * object_height)
        )
        points = points[keep]
        if len(points) < args.min_tracking_points:
            continue
        projected = points @ axis
        spans.append(float(np.quantile(projected, 0.99) - np.quantile(projected, 0.01)))
        heights.append(object_height)
    if not spans:
        raise ValueError("no valid un-eroded reference frames for object-width review")
    return np.asarray(spans, dtype=float), np.asarray(heights, dtype=float)


def project(points: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    pixels = np.full((len(points), 2), np.nan)
    homogeneous = points[valid] @ intrinsics.T
    pixels[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, valid


def pixel(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def draw_masks_and_hand(
    image: np.ndarray,
    object_mask_path: Path,
    hand_mask_path: Path,
    keypoints: np.ndarray | None,
    confidence: np.ndarray,
) -> None:
    object_mask = cv2.imread(str(object_mask_path), cv2.IMREAD_GRAYSCALE)
    hand_mask = cv2.imread(str(hand_mask_path), cv2.IMREAD_GRAYSCALE)
    if object_mask is None or hand_mask is None:
        raise ValueError("could not load preview masks")
    for mask, color in ((object_mask, (255, 180, 20)), (hand_mask, (220, 60, 220))):
        active = mask > 0
        image[active] = np.clip(
            0.58 * image[active] + 0.42 * np.asarray(color, dtype=np.float32), 0, 255
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 2)
    if keypoints is None:
        return
    valid = np.isfinite(keypoints).all(axis=1)
    for first, second in HAND_EDGES:
        if (
            valid[first]
            and valid[second]
            and confidence[first] >= 0.5
            and confidence[second] >= 0.5
        ):
            cv2.line(image, pixel(keypoints[first]), pixel(keypoints[second]), (0, 240, 255), 2)
    for index, point in enumerate(keypoints):
        if valid[index] and confidence[index] >= 0.5:
            color = (0, 0, 255) if index in (THUMB_TIP, INDEX_TIP) else (255, 230, 80)
            cv2.circle(image, pixel(point), 4, color, -1, cv2.LINE_AA)


def draw_preview(
    rgb_paths: list[Path],
    object_mask_paths: list[Path],
    hand_mask_paths: list[Path],
    keypoints: np.ndarray | None,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    camera_tcp: np.ndarray,
    observed_tips: np.ndarray,
    contact_frame: int,
    peak_frame: int,
    peak_lift_m: float,
    base_tcp: np.ndarray,
    nominal_opening_fit: bool,
    preliminary_opening_m: float,
    fixed_object_replay_valid: bool,
    prelift_shift_p95_m: float,
    approach_window_angle_p95_deg: float | None,
    output_path: Path,
) -> None:
    panels: list[np.ndarray] = []
    for frame_index, label in ((contact_frame, "PRE-LIFT CONTACT"), (peak_frame, "PEAK LIFT")):
        image = cv2.imread(str(rgb_paths[frame_index]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not load preview RGB frame {frame_index}")
        draw_masks_and_hand(
            image,
            object_mask_paths[frame_index],
            hand_mask_paths[frame_index],
            None if keypoints is None else keypoints[frame_index],
            confidence[frame_index],
        )
        cv2.rectangle(image, (0, 0), (image.shape[1], 35), (15, 15, 15), -1)
        cv2.putText(
            image,
            f"{label}  frame {frame_index}",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (40, 230, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(image)

    tip_pixels, tip_valid = project(observed_tips, intrinsics)
    if np.all(tip_valid):
        cv2.line(panels[0], pixel(tip_pixels[0]), pixel(tip_pixels[1]), (20, 255, 20), 4)
        for point in tip_pixels:
            cv2.circle(panels[0], pixel(point), 7, (20, 255, 20), -1, cv2.LINE_AA)
    center = camera_tcp[:3, 3]
    axes = np.vstack([center, center + 0.04 * camera_tcp[:3, 0], center + 0.04 * camera_tcp[:3, 2]])
    axis_pixels, axis_valid = project(axes, intrinsics)
    if np.all(axis_valid):
        origin = pixel(axis_pixels[0])
        cv2.arrowedLine(panels[0], origin, pixel(axis_pixels[1]), (0, 0, 255), 3, tipLength=0.2)
        cv2.arrowedLine(panels[0], origin, pixel(axis_pixels[2]), (255, 80, 40), 3, tipLength=0.2)

    combined = np.hstack(panels)
    header = np.full((130, combined.shape[1], 3), 15, dtype=np.uint8)
    xyz = base_tcp[:3, 3]
    feasibility = (
        "UPPER-SLICE FIT / GRASP WIDTH UNKNOWN"
        if nominal_opening_fit
        else "NO UPPER-SLICE FIT / GRASP WIDTH UNKNOWN"
    )
    approach_window_label = (
        "n/a"
        if approach_window_angle_p95_deg is None
        else f"{approach_window_angle_p95_deg:.1f} deg"
    )
    cv2.putText(
        header,
        "PICKUP EVENT VALIDATED / REVIEW ONLY / NO ROBOT EXECUTION",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (40, 230, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        (
            "REPLAY POSE GATES PASS / EXECUTION STILL DISALLOWED"
            if fixed_object_replay_valid
            else "REPLAY TARGET WITHHELD"
        )
        + f" | shift p95={prelift_shift_p95_m * 1000:.1f} mm"
        + f" | approach-window p95={approach_window_label}",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 230, 80) if fixed_object_replay_valid else (50, 120, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        f"demonstrated grasp xyz={xyz[0]:+.3f} {xyz[1]:+.3f} {xyz[2]:+.3f} m | "
        f"peak lift={peak_lift_m * 1000:.1f} mm",
        (12, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        f"prelim upper-slice opening={preliminary_opening_m * 1000:.1f} mm | "
        f"{feasibility}",
        (12, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (50, 200, 255) if nominal_opening_fit else (60, 80, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output_path), np.vstack((header, combined))):
        raise RuntimeError(f"could not write preview {output_path}")


def matrices_to_lists(values: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
    return {name: matrix.tolist() for name, matrix in values.items()}


def main() -> int:
    args = parse_args()
    for name in (
        "reference_prefix_frames",
        "post_placement_frames",
        "min_reference_cloud_points",
        "min_tracking_points",
        "min_stable_frames",
        "approach_lookback_frames",
        "min_approach_frames",
        "min_approach_window_count",
        "min_lift_onset_frames",
        "min_airborne_frames",
        "min_coupling_frames",
        "min_moving_step_pairs",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "pre_lift_guard_frames",
        "approach_guard_frames",
        "max_contact_lift_gap_frames",
        "mask_erode_px",
        "reference_hand_exclusion_px",
        "tracking_hand_exclusion_px",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not (
        0.0
        < args.width_slice_lower_height_fraction
        < args.width_slice_upper_height_fraction
        < 1.0
    ):
        raise ValueError("width slice fractions must satisfy 0 < lower < upper < 1")
    if args.approach_lookback_frames < args.min_approach_frames:
        raise ValueError("approach lookback must be at least min approach frames")
    for name in (
        "min_approach_trajectory_displacement_m",
        "max_approach_trajectory_residual_p95_m",
        "min_approach_signal_to_residual",
        "max_approach_window_angle_p95_deg",
        "max_prelift_object_shift_m",
        "prelift_shift_noise_multiplier",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rgb_paths = numeric_paths(args.rgb_dir, IMAGE_SUFFIXES)
    stems = [path.stem for path in rgb_paths]
    depth_paths = indexed_paths(args.depth_dir, {".npy"}, stems)
    object_mask_paths = indexed_paths(args.object_mask_dir, {".png"}, stems)
    hand_mask_paths = indexed_paths(args.hand_mask_dir, {".png"}, stems)
    frame_count = len(rgb_paths)
    if args.reference_prefix_frames >= frame_count:
        raise ValueError("reference prefix must be shorter than the recording")
    if args.post_placement_frames > frame_count:
        raise ValueError("post-placement window exceeds the recording")

    intrinsics = np.asarray(np.load(args.intrinsics, allow_pickle=False), dtype=float)
    joints = np.asarray(np.load(args.hand_joints_camera_m, allow_pickle=False), dtype=float)
    confidence = np.asarray(np.load(args.hand_confidence, allow_pickle=False), dtype=float)
    timestamps = np.asarray(np.load(args.timestamps_s, allow_pickle=False), dtype=float)
    base_camera = validate_rigid_transform(
        np.load(args.t_base_camera, allow_pickle=False), name="T_base_camera"
    )
    keypoints = (
        np.asarray(np.load(args.hand_keypoints_2d, allow_pickle=False), dtype=float)
        if args.hand_keypoints_2d
        else None
    )
    if joints.shape != (frame_count, 21, 3):
        raise ValueError(f"hand joints must have shape ({frame_count}, 21, 3), got {joints.shape}")
    if confidence.shape != (frame_count, 21):
        raise ValueError(f"hand confidence must have shape ({frame_count}, 21)")
    if timestamps.shape != (frame_count,):
        raise ValueError(f"timestamps must have shape ({frame_count},)")
    if keypoints is not None and keypoints.shape != (frame_count, 21, 2):
        raise ValueError(f"hand keypoints must have shape ({frame_count}, 21, 2)")

    support_height, support_height_residual_p95, support_height_samples = (
        estimate_support_height(
            args.reference_prefix_frames,
            depth_paths,
            object_mask_paths,
            hand_mask_paths,
            intrinsics,
            base_camera,
            args,
        )
    )
    reference_cloud, reference_frame_clouds, reference_counts = build_reference_cloud(
        args.reference_prefix_frames,
        depth_paths,
        object_mask_paths,
        hand_mask_paths,
        intrinsics,
        base_camera,
        support_height,
        args,
    )
    if len(reference_cloud) < args.min_reference_cloud_points:
        raise ValueError(
            f"insufficient pre-contact object depth: {len(reference_cloud)} points"
        )
    np.save(
        args.output_dir / "precontact_object_points_camera_m.npy",
        reference_cloud,
        allow_pickle=False,
    )

    camera_track, base_track, track_counts = track_object(
        depth_paths,
        object_mask_paths,
        hand_mask_paths,
        intrinsics,
        base_camera,
        support_height,
        args,
    )
    events = detect_pickup_events(
        base_track,
        timestamps,
        reference_frame_count=args.reference_prefix_frames,
        max_reference_motion_p95_m=args.max_reference_motion_p95_m,
        min_lift_onset_m=args.min_lift_onset_m,
        min_onset_frames=args.min_lift_onset_frames,
        min_airborne_height_m=args.min_airborne_height_m,
        min_airborne_frames=args.min_airborne_frames,
        min_airborne_duration_s=args.min_airborne_duration_s,
    )
    contact_cutoff = events.lift_onset_frame - args.pre_lift_guard_frames
    if contact_cutoff <= args.reference_prefix_frames:
        raise ValueError("pre-lift contact window ends inside the reference prefix")

    camera_base = np.linalg.inv(base_camera)
    reference_center_camera = transform_points(
        camera_base, events.reference_center_base_m[None]
    )[0]
    aligned_joints, object_track_valid, object_displacements_camera = (
        align_hand_joints_to_reference_translation(
            joints,
            camera_track,
            reference_center_camera,
        )
    )
    aligned_confidence = confidence.copy()
    aligned_confidence[~object_track_valid] = 0.0

    intent = infer_pre_lift_pickup_intent(
        aligned_joints[:contact_cutoff],
        aligned_confidence[:contact_cutoff],
        timestamps[:contact_cutoff],
        reference_cloud,
        IntentConfig(
            max_tip_object_distance_m=args.max_contact_distance_m,
            min_joint_confidence=args.min_joint_confidence,
            min_pinch_distance_m=args.min_pinch_width_m,
            max_pinch_distance_m=args.max_jaw_width_m,
            min_stable_frames=args.min_stable_frames,
        ),
        approach_lookback_frames=args.approach_lookback_frames,
        approach_guard_frames=args.approach_guard_frames,
        min_approach_frames=args.min_approach_frames,
    )
    start, end = intent.stable_interval
    validate_contact_before_lift(
        intent.stable_interval,
        events.lift_onset_frame,
        max_gap_frames=args.max_contact_lift_gap_frames,
    )
    stable_duration = float(timestamps[end - 1] - timestamps[start])
    if stable_duration < args.min_stable_duration_s:
        raise ValueError(
            f"stable contact lasts {stable_duration:.3f}s, below "
            f"{args.min_stable_duration_s:.3f}s"
        )
    prelift_motion = measure_prelift_object_motion(
        base_track,
        events.reference_center_base_m,
        intent.stable_interval,
        reference_motion_p95_m=events.reference_motion_p95_m,
        max_shift_m=args.max_prelift_object_shift_m,
        noise_multiplier=args.prelift_shift_noise_multiplier,
    )

    aligned_camera_tcp = validate_rigid_transform(
        intent.T_camera_gripper_tcp, name="T_camera_gripper_tcp"
    )
    representative_shift_camera = (
        base_camera[:3, :3].T @ prelift_motion.shift_vector_base_m
    )
    camera_tcp = aligned_camera_tcp.copy()
    camera_tcp[:3, 3] += representative_shift_camera
    camera_tcp = validate_rigid_transform(
        camera_tcp, name="translation_compensated_T_camera_gripper_tcp"
    )
    base_tcp = base_camera @ camera_tcp
    quality = contact_quality(
        aligned_joints,
        aligned_confidence,
        reference_cloud,
        start,
        end,
        aligned_camera_tcp,
    )
    if quality["contact_midpoint_rms_m"] > args.max_contact_center_rms_m:
        raise ValueError("pre-lift pinch midpoint is not stable enough")
    if quality["raw_closing_angle_p95_deg"] > args.max_raw_closing_angle_p95_deg:
        raise ValueError("pre-lift closing direction is not stable enough")
    approach_replay = assess_approach_for_fixed_object_replay(
        intent,
        min_trajectory_displacement_m=args.min_approach_trajectory_displacement_m,
        max_trajectory_residual_p95_m=(
            args.max_approach_trajectory_residual_p95_m
        ),
        min_signal_to_residual=args.min_approach_signal_to_residual,
        min_window_count=args.min_approach_window_count,
        max_window_angle_p95_deg=args.max_approach_window_angle_p95_deg,
    )
    approach_signal_to_residual = approach_replay.signal_to_residual_ratio
    approach_axis_window_stable = approach_replay.window_stable
    replay_gate_failures = list(approach_replay.gate_failures)
    if not prelift_motion.replay_target_valid:
        replay_gate_failures.insert(0, "prelift_object_shift")
    fixed_object_replay_valid = not replay_gate_failures

    pinch_camera = np.mean(joints[:, [THUMB_TIP, INDEX_TIP]], axis=1)
    pinch_base = transform_points(base_camera, pinch_camera)
    pinch_confidence = np.min(confidence[:, [THUMB_TIP, INDEX_TIP]], axis=1)
    coupling = measure_hand_object_coupling(
        base_track,
        pinch_base,
        pinch_confidence,
        (events.lift_onset_frame, events.airborne_interval[1]),
        min_confidence=args.min_joint_confidence,
        min_frames=args.min_coupling_frames,
        max_offset_residual_m=args.max_relative_motion_p95_m,
    )
    if coupling.relative_motion_p95_m > args.max_relative_motion_p95_m:
        raise ValueError(
            "lifted object does not retain a stable offset to the pinch center: "
            f"p95={coupling.relative_motion_p95_m * 1000.0:.1f} mm, limit="
            f"{args.max_relative_motion_p95_m * 1000.0:.1f} mm, "
            f"valid_frames={coupling.valid_frames}"
        )
    if coupling.median_center_distance_m > args.max_hand_object_distance_m:
        raise ValueError(
            "lifted object is too far from the tracked pinch center: "
            f"median={coupling.median_center_distance_m * 1000.0:.1f} mm, limit="
            f"{args.max_hand_object_distance_m * 1000.0:.1f} mm"
        )
    if coupling.moving_step_pairs < args.min_moving_step_pairs:
        raise ValueError(
            "insufficient moving frame pairs to validate hand/object coupling: "
            f"{coupling.moving_step_pairs}, need {args.min_moving_step_pairs}"
        )
    if coupling.velocity_cosine_median < args.min_velocity_cosine:
        raise ValueError(
            "object and hand motion directions do not agree: median cosine="
            f"{coupling.velocity_cosine_median:.3f}, need {args.min_velocity_cosine:.3f}"
        )
    if coupling.step_displacement_mismatch_p95_m > args.max_step_mismatch_p95_m:
        raise ValueError(
            "object and hand step displacements disagree: p95="
            f"{coupling.step_displacement_mismatch_p95_m * 1000.0:.1f} mm, limit="
            f"{args.max_step_mismatch_p95_m * 1000.0:.1f} mm"
        )

    eroded_spans = reference_spans(reference_frame_clouds, camera_tcp[:3, 0])
    spans, estimated_heights = single_view_reference_spans(
        args.reference_prefix_frames,
        depth_paths,
        object_mask_paths,
        hand_mask_paths,
        intrinsics,
        camera_tcp[:3, 0],
        base_camera,
        support_height,
        args,
    )
    preliminary_object_span = float(np.quantile(spans, 0.95))
    full_contact_cloud_span = float(np.quantile(eroded_spans, 0.95))
    reflection_contamination_suspected = (
        full_contact_cloud_span - preliminary_object_span >= 0.010
    )
    preliminary_opening = preliminary_object_span + args.pregrasp_clearance_m
    nominal_width_fit = preliminary_opening <= args.max_jaw_width_m
    width_margin = args.max_jaw_width_m - preliminary_opening
    nominal_width_fit_with_margin = width_margin >= args.min_jaw_width_margin_m
    tcp_height_above_support = float(base_tcp[2, 3] - support_height)
    tcp_targets = (
        make_tcp_targets(
            base_camera,
            camera_tcp,
            approach_distance_m=args.approach_distance_m,
            lift_height_m=args.lift_height_m,
        )
        if fixed_object_replay_valid
        else None
    )
    observed_tips_camera = (
        intent.observed_fingertip_points_camera_m + representative_shift_camera
    )
    observed_tips_base = transform_points(base_camera, observed_tips_camera)

    suffix = base_track[frame_count - args.post_placement_frames :]
    suffix_valid = np.isfinite(suffix).all(axis=1)
    post_placement_center = (
        robust_mean(suffix[suffix_valid])
        if np.any(suffix_valid)
        else np.full(3, np.nan)
    )
    post_placement_shift = float(
        np.linalg.norm(post_placement_center - events.reference_center_base_m)
    )

    np.savez_compressed(
        args.output_dir / "pickup_track.npz",
        object_centers_camera_m=camera_track,
        object_centers_base_m=base_track,
        object_displacements_from_reference_camera_m=object_displacements_camera,
        object_valid_depth_point_count=track_counts,
        pinch_centers_base_m=pinch_base,
        pinch_confidence=pinch_confidence,
        vertical_displacement_m=events.vertical_displacement_m,
        timestamps_s=timestamps,
    )
    contact_frame = (start + end - 1) // 2
    preview_path = args.output_dir / "preview.png"
    draw_preview(
        rgb_paths,
        object_mask_paths,
        hand_mask_paths,
        keypoints,
        confidence,
        intrinsics,
        camera_tcp,
        observed_tips_camera,
        contact_frame,
        events.peak_frame,
        events.peak_lift_m,
        base_tcp,
        nominal_width_fit_with_margin,
        preliminary_opening,
        fixed_object_replay_valid,
        prelift_motion.contact_shift_p95_m,
        intent.approach_window_angle_p95_deg,
        preview_path,
    )

    airborne_start, airborne_end = events.airborne_interval
    critical_grasp_start = max(start, end - args.min_stable_frames)
    validation_config = {
        name: value
        for name, value in sorted(vars(args).items())
        if not isinstance(value, Path)
    }
    payload = {
        "schema": "demo2gripper.cad_free_pickup.v2",
        "mode": "cad_free_pickup_demonstration",
        "status": "review_only",
        "pickup_demonstration_validated": True,
        "execution_authorized": False,
        "robot_execution_ready": False,
        "fixed_object_replay_valid": fixed_object_replay_valid,
        "robot_grasp_width_feasible": None,
        "robot_command_frame_available": False,
        "grasp_plane_width_validated": False,
        "support_clearance_validated": False,
        "reflection_contamination_suspected": reflection_contamination_suspected,
        "nominal_jaw_limit_fit": nominal_width_fit,
        "nominal_upper_slice_opening_fit_with_margin": nominal_width_fit_with_margin,
        "reference_frame": "OpenCV color optical frame at the demonstrated contact pose",
        "base_frame": "FR3 O/base from T_base_camera calibration contract",
        "frame_convention": (
            "T_parent_child, column vectors, metres; demonstrated grasp frame +X "
            "closing, +Z stable-contact wrist-to-pinch approach"
        ),
        "contact_inference_frame": "translation-aligned reference object frame",
        "translation_compensation_applied": True,
        "reference_object_center_camera_m": reference_center_camera.tolist(),
        "representative_contact_shift_camera_m": representative_shift_camera.tolist(),
        "object_track_valid_frames_before_lift": int(
            np.count_nonzero(object_track_valid[:contact_cutoff])
        ),
        "T_base_camera": base_camera.tolist(),
        "T_camera_demonstrated_grasp_frame": camera_tcp.tolist(),
        "T_base_demonstrated_grasp_frame": base_tcp.tolist(),
        "canonical_tcp_targets_base": (
            None if tcp_targets is None else matrices_to_lists(tcp_targets)
        ),
        "canonical_tcp_targets_withheld_reason": (
            None
            if fixed_object_replay_valid
            else ", ".join(replay_gate_failures)
        ),
        "fixed_object_replay_gate_failures": replay_gate_failures,
        "observed_fingertip_points_camera_m": observed_tips_camera.tolist(),
        "observed_fingertip_points_base_m": observed_tips_base.tolist(),
        "closing_axis_camera": camera_tcp[:3, 0].tolist(),
        "approach_axis_camera": camera_tcp[:3, 2].tolist(),
        "approach_axis_source": "stable_contact_wrist_to_pinch",
        "demonstrated_pinch_width_m": intent.demonstrated_pinch_width_m,
        "human_pinch_is_not_robot_width": True,
        "preliminary_single_view_upper_slice_span_p95_m": preliminary_object_span,
        "width_estimate_status": "preliminary_single_view_support_filtered",
        "estimated_object_height_median_m": float(np.median(estimated_heights)),
        "estimated_object_height_p95_m": float(np.quantile(estimated_heights, 0.95)),
        "full_contact_cloud_span_p95_m_not_for_width": full_contact_cloud_span,
        "object_span_method": (
            f"p01-p99 projection in the {args.width_slice_lower_height_fraction:.2f}H-"
            f"{args.width_slice_upper_height_fraction:.2f}H support-relative slice of "
            "each un-eroded, depth-trimmed reference mask; p95 across frames"
        ),
        "pregrasp_clearance_m": args.pregrasp_clearance_m,
        "preliminary_upper_slice_opening_m": preliminary_opening,
        "recommended_pregrasp_opening_m": args.max_jaw_width_m,
        "max_configured_jaw_width_m": args.max_jaw_width_m,
        "required_jaw_width_margin_m": args.min_jaw_width_margin_m,
        "jaw_width_margin_m": width_margin,
        "tcp_height_above_local_support_m": tcp_height_above_support,
        "stable_contact_interval": [start, end],
        "stable_contact_duration_s": stable_duration,
        "critical_grasp_interval": [critical_grasp_start, end],
        "critical_grasp_interval_source": (
            "terminal_min_stable_frames_of_stable_contact_interval"
        ),
        "approach_trajectory_interval": (
            None
            if intent.approach_interval is None
            else list(intent.approach_interval)
        ),
        "approach_trajectory_available": intent.approach_interval is not None,
        "approach_trajectory_role": "phase_diagnostic_and_fixed_replay_gate_only",
        "approach_trajectory_displacement_m": intent.approach_trajectory_displacement_m,
        "approach_trajectory_residual_p95_m": intent.approach_trajectory_residual_p95_m,
        "approach_signal_to_residual_ratio": approach_signal_to_residual,
        "approach_window_angle_p95_deg": intent.approach_window_angle_p95_deg,
        "approach_window_angle_max_deg": intent.approach_window_angle_max_deg,
        "approach_window_count": intent.approach_window_count,
        "approach_axis_window_stable": approach_axis_window_stable,
        "prelift_contact_object_center_base_m": (
            prelift_motion.contact_center_base_m.tolist()
        ),
        "prelift_object_shift_vector_base_m": prelift_motion.shift_vector_base_m.tolist(),
        "prelift_contact_center_shift_m": prelift_motion.contact_center_shift_m,
        "prelift_contact_shift_p95_m": prelift_motion.contact_shift_p95_m,
        "effective_prelift_shift_limit_m": prelift_motion.effective_shift_limit_m,
        "grasp_origin_offset_from_contact_object_center_base_m": (
            base_tcp[:3, 3] - prelift_motion.contact_center_base_m
        ).tolist(),
        "contact_cutoff_frame": contact_cutoff,
        "lift_onset_frame": events.lift_onset_frame,
        "airborne_interval": [airborne_start, airborne_end],
        "airborne_duration_s": events.airborne_duration_s,
        "peak_lift_frame": events.peak_frame,
        "peak_lift_m": events.peak_lift_m,
        "post_placement_center_base_m": post_placement_center.tolist(),
        "post_placement_shift_from_reference_m": post_placement_shift,
        "quality": {
            **quality,
            "reference_object_point_count": len(reference_cloud),
            "reference_valid_depth_per_frame_min": min(reference_counts),
            "reference_center_base_m": events.reference_center_base_m.tolist(),
            "local_support_height_base_m": support_height,
            "local_support_height_motion_p95_m": support_height_residual_p95,
            "local_support_height_sample_frames": len(support_height_samples),
            "reference_cloud_min_height_above_support_m": (
                args.min_object_height_above_support_m
            ),
            "width_slice_height_fraction": [
                args.width_slice_lower_height_fraction,
                args.width_slice_upper_height_fraction,
            ],
            "reference_center_motion_p95_m": events.reference_motion_p95_m,
            "reference_vertical_sigma_m": events.reference_vertical_sigma_m,
            "effective_lift_onset_threshold_m": events.effective_lift_onset_m,
            "coupling_interval": [events.lift_onset_frame, events.airborne_interval[1]],
            "coupling_valid_consensus_frames": coupling.valid_frames,
            "hand_object_median_offset_base_m": coupling.median_offset_base_m.tolist(),
            "hand_object_center_distance_median_m": coupling.median_center_distance_m,
            "hand_object_relative_motion_median_m": coupling.relative_motion_median_m,
            "hand_object_relative_motion_p95_m": coupling.relative_motion_p95_m,
            "hand_object_moving_step_pairs": coupling.moving_step_pairs,
            "hand_object_velocity_cosine_median": coupling.velocity_cosine_median,
            "hand_object_step_mismatch_p95_m": coupling.step_displacement_mismatch_p95_m,
        },
        "validation_config": validation_config,
        "interpretation": [
            "The human pickup is validated by sustained base-frame lift and "
            "hand/object motion coupling.",
            "Free-space approach-trajectory quality does not participate in pickup "
            "validation; it only controls whether a direct fixed-object replay target "
            "is emitted.",
            "Hand/object coupling uses a robust offset consensus from lift onset through "
            "the sustained airborne interval so transient occlusion keypoint jumps do not "
            "override the majority trajectory evidence.",
            "The reported transform is a canonical human demonstrated-grasp frame, not a "
            "measured Hand-E TCP or robot command target.",
            "Observed fingertips are not claimed to be measured force-contact points.",
            "Support-level reflection is suppressed for tracking and the upper-slice width "
            "diagnostic, but residual reflection contamination remains suspected.",
            "The human fingertip aperture includes anatomy and offsets, so it does not set "
            "the robot's required opening.",
            "The upper-slice object span plus clearance is only a preliminary single-view "
            "opening diagnostic; width at the demonstrated grasp plane is unknown.",
            "Direct fixed-object replay is withheld when the object slides before lift "
            "or the free-space approach fit is unavailable, short, curved, noisy, or "
            "window-sensitive.",
            "The grasp-frame +Z axis comes from the stable-contact wrist-to-pinch "
            "relation; the free-space fitted trajectory does not define the critical "
            "grasp pose.",
            "The final object location is reported but is not used to retarget this grasp.",
        ],
        "limitations": [
            "No CAD model, force contact, force closure, IK, collision, or "
            "reachability check is available.",
            "A new object location requires live template registration before "
            "any plan can be prepared.",
            "The current T_base_camera is operator-accepted best-effort calibration.",
            "Hand-E flange-to-TCP and collision geometry remain unmeasured.",
            "The free-space trajectory is a phase/replay diagnostic and is not a "
            "validated robot approach axis on this recording.",
            "The demonstrated TCP is close to the support surface, so finger/support "
            "clearance must be checked with the measured Hand-E geometry.",
            "The demonstrated grasp frame and any derived review poses must not be sent "
            "to the robot.",
        ],
        "artifacts": {
            "preview": preview_path.name,
            "precontact_object_points_camera_m": "precontact_object_points_camera_m.npy",
            "pickup_track": "pickup_track.npz",
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
    result_path = args.output_dir / "pickup_grasp.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"pickup validated: contact=[{start}, {end}), lift={events.peak_lift_m * 1000:.1f} mm, "
        f"base xyz={base_tcp[0, 3]:.4f} {base_tcp[1, 3]:.4f} {base_tcp[2, 3]:.4f} m"
    )
    print(
        f"single-view opening diagnostic={preliminary_opening * 1000:.1f} mm, "
        f"limit={args.max_jaw_width_m * 1000:.1f} mm, "
        f"nominal_fit={nominal_width_fit_with_margin}"
    )
    approach_window_summary = (
        "n/a"
        if intent.approach_window_angle_p95_deg is None
        else f"{intent.approach_window_angle_p95_deg:.1f} deg"
    )
    print(
        f"fixed-object replay valid={fixed_object_replay_valid}; "
        f"pre-lift shift p95={prelift_motion.contact_shift_p95_m * 1000:.1f} mm, "
        f"approach window p95={approach_window_summary}"
    )
    print(f"review-only result: {result_path}")
    print(f"preview: {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
