#!/usr/bin/env python3
"""Create a deterministic synthetic RGB-D contract demo for visualization.

This is not perception output and is not evidence of model accuracy.  It provides
geometrically consistent images, masks, metric hand joints, object poses, and
grasp candidates so the complete visualization/data plumbing can be exercised.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np


FRAME_COUNT = 120
HEIGHT = 480
WIDTH = 640
FPS = 30.0
CONTACT_START = 30
CONTACT_END = 100
BOX_HALF_EXTENTS_M = np.array([0.040, 0.055, 0.030], dtype=np.float64)

HAND_BONES = (
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
    (5, 9),
    (9, 13),
    (13, 17),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rotation_y(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def project(points_camera_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    pixels_h = points_camera_m @ intrinsics.T
    return pixels_h[:, :2] / pixels_h[:, 2:3]


def cuboid_corners(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.asarray(list(product(*zip(lower, upper, strict=True))), dtype=np.float64)


def box_surface_points(half_extents: np.ndarray, samples: int = 31) -> np.ndarray:
    coordinates = [
        np.linspace(-half_extents[axis], half_extents[axis], samples)
        for axis in range(3)
    ]
    faces: list[np.ndarray] = []
    for fixed_axis in range(3):
        free_axes = [axis for axis in range(3) if axis != fixed_axis]
        first, second = np.meshgrid(
            coordinates[free_axes[0]], coordinates[free_axes[1]], indexing="ij"
        )
        for side in (-1.0, 1.0):
            face = np.zeros((samples * samples, 3), dtype=np.float64)
            face[:, fixed_axis] = side * half_extents[fixed_axis]
            face[:, free_axes[0]] = first.ravel()
            face[:, free_axes[1]] = second.ravel()
            faces.append(face)
    return np.unique(np.concatenate(faces, axis=0), axis=0)


def base_hand_joints_object_m() -> np.ndarray:
    """Return one plausible right-hand MANO21 layout around a box pinch."""

    joints = np.zeros((21, 3), dtype=np.float64)
    joints[0] = [0.000, 0.115, -0.130]

    joints[1] = [-0.035, 0.090, -0.110]
    joints[2] = [-0.052, 0.061, -0.082]
    joints[3] = [-0.052, 0.028, -0.043]
    joints[4] = [-BOX_HALF_EXTENTS_M[0], 0.000, 0.000]

    joints[5] = [-0.018, 0.085, -0.100]
    joints[6] = [0.019, 0.058, -0.076]
    joints[7] = [0.037, 0.027, -0.041]
    joints[8] = [BOX_HALF_EXTENTS_M[0], 0.000, 0.000]

    joints[9] = [0.004, 0.090, -0.100]
    joints[10] = [0.032, 0.069, -0.076]
    joints[11] = [0.050, 0.056, -0.056]
    joints[12] = [0.057, 0.047, -0.045]

    joints[13] = [0.025, 0.094, -0.101]
    joints[14] = [0.047, 0.078, -0.081]
    joints[15] = [0.062, 0.070, -0.065]
    joints[16] = [0.068, 0.064, -0.055]

    joints[17] = [0.045, 0.100, -0.106]
    joints[18] = [0.063, 0.091, -0.086]
    joints[19] = [0.075, 0.087, -0.073]
    joints[20] = [0.081, 0.083, -0.063]
    return joints


def hand_joints_object_m(frame_idx: int) -> np.ndarray:
    joints = base_hand_joints_object_m()
    phase = 2.0 * np.pi * frame_idx / FRAME_COUNT
    joints[:, 0] += 0.0007 * np.sin(phase)

    if CONTACT_START <= frame_idx < CONTACT_END:
        # Keep each tip on its corresponding side face while adding small,
        # deterministic in-plane motion typical of a held pinch.
        tip_motion = np.array(
            [0.0, 0.00045 * np.sin(2.3 * phase), 0.00035 * np.cos(1.7 * phase)]
        )
        joints[4] = np.array([-BOX_HALF_EXTENTS_M[0], 0.0, 0.0]) + tip_motion
        joints[8] = np.array([BOX_HALF_EXTENTS_M[0], 0.0, 0.0]) - tip_motion
        return joints

    if frame_idx < CONTACT_START:
        progress = frame_idx / float(CONTACT_START - 1)
    else:
        progress = (FRAME_COUNT - 1 - frame_idx) / float(FRAME_COUNT - CONTACT_END - 1)
    progress = float(np.clip(progress, 0.0, 1.0))
    eased = progress * progress * (3.0 - 2.0 * progress)
    body_offset = (1.0 - eased) * np.array([0.0, 0.018, -0.035])
    joints += body_offset

    thumb_far = np.array([-0.078, -0.012, -0.090])
    thumb_near = np.array([-0.066, -0.004, -0.065])
    index_far = np.array([0.078, -0.012, -0.090])
    index_near = np.array([0.066, -0.004, -0.065])
    joints[4] = (1.0 - eased) * thumb_far + eased * thumb_near
    joints[8] = (1.0 - eased) * index_far + eased * index_near
    joints[3] = 0.58 * joints[3] + 0.42 * joints[4]
    joints[7] = 0.58 * joints[7] + 0.42 * joints[8]
    return joints


def object_transform(frame_idx: int) -> np.ndarray:
    phase = 2.0 * np.pi * frame_idx / FRAME_COUNT
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_z(0.025 * np.sin(phase)) @ rotation_y(
        -0.075 + 0.15 * frame_idx / (FRAME_COUNT - 1)
    )
    transform[:3, 3] = np.array(
        [0.006 * np.sin(phase), 0.003 * np.sin(2.0 * phase), 0.720 + 0.004 * np.cos(phase)]
    )
    return transform


def make_background() -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(HEIGHT, dtype=np.float32)[:, None]
    frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :, 0] = np.clip(72.0 + 0.045 * rows, 0, 255).astype(np.uint8)
    frame[:, :, 1] = np.clip(78.0 + 0.050 * rows, 0, 255).astype(np.uint8)
    frame[:, :, 2] = np.clip(84.0 + 0.055 * rows, 0, 255).astype(np.uint8)
    depth = np.repeat((1.42 - 0.00035 * rows), WIDTH, axis=1).astype(np.float32)
    cv2.line(frame, (0, 374), (WIDTH - 1, 374), (91, 96, 100), 2, cv2.LINE_AA)
    return frame, depth


def convex_mask(points_px: np.ndarray) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    hull = cv2.convexHull(np.rint(points_px).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255, cv2.LINE_AA)
    return mask


def render_object(
    frame: np.ndarray,
    depth: np.ndarray,
    transform: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = -BOX_HALF_EXTENTS_M
    upper = BOX_HALF_EXTENTS_M
    corners_object = cuboid_corners(lower, upper)
    corners_camera = transform_points(transform, corners_object)
    corners_px = project(corners_camera, intrinsics)
    object_mask = convex_mask(corners_px)

    faces: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
    face_colours = ((76, 150, 220), (82, 160, 230), (68, 137, 207))
    for axis in range(3):
        for side_index, side in enumerate((-1.0, 1.0)):
            indices = np.flatnonzero(np.isclose(corners_object[:, axis], side * upper[axis]))
            hull = cv2.convexHull(np.rint(corners_px[indices]).astype(np.int32))
            colour = tuple(min(channel + 8 * side_index, 255) for channel in face_colours[axis])
            faces.append((float(np.mean(corners_camera[indices, 2])), hull, colour))
    for _, hull, colour in sorted(faces, key=lambda item: item[0], reverse=True):
        cv2.fillConvexPoly(frame, hull, colour, cv2.LINE_AA)
    for _, hull, _ in faces:
        cv2.polylines(frame, [hull], True, (42, 80, 119), 1, cv2.LINE_AA)

    part_lower = np.array([-0.0405, -0.0560, -0.0305])
    part_upper = np.array([0.0405, -0.0330, 0.0305])
    part_camera = transform_points(transform, cuboid_corners(part_lower, part_upper))
    part_mask = convex_mask(project(part_camera, intrinsics))
    frame[part_mask > 0] = (73, 197, 154)
    part_contours, _ = cv2.findContours(part_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, part_contours, -1, (38, 112, 84), 1, cv2.LINE_AA)

    object_front_depth = float(np.min(corners_camera[:, 2]))
    depth[object_mask > 0] = object_front_depth
    depth[part_mask > 0] = float(np.min(part_camera[:, 2]))
    return object_mask, part_mask


def render_hand(
    frame: np.ndarray,
    depth: np.ndarray,
    joints_camera_m: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    pixels = np.rint(project(joints_camera_m, intrinsics)).astype(np.int32)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    palm_hull = cv2.convexHull(pixels[[0, 1, 5, 9, 13, 17]])
    cv2.fillConvexPoly(mask, palm_hull, 255, cv2.LINE_AA)
    for first, second in HAND_BONES:
        cv2.line(mask, tuple(pixels[first]), tuple(pixels[second]), 255, 13, cv2.LINE_AA)
    for pixel in pixels:
        cv2.circle(mask, tuple(pixel), 7, 255, -1, cv2.LINE_AA)

    frame[mask > 0] = (137, 183, 224)
    for first, second in HAND_BONES:
        cv2.line(
            frame,
            tuple(pixels[first]),
            tuple(pixels[second]),
            (111, 153, 198),
            3,
            cv2.LINE_AA,
        )
    for joint_index, pixel in enumerate(pixels):
        colour = (40, 215, 248) if joint_index in (4, 8) else (198, 220, 238)
        radius = 5 if joint_index in (4, 8) else 3
        cv2.circle(frame, tuple(pixel), radius, colour, -1, cv2.LINE_AA)

    depth[mask > 0] = np.minimum(
        depth[mask > 0], np.float32(np.median(joints_camera_m[:, 2]))
    )
    return mask


def orthonormal_pose(center: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = center
    return transform


def grasp_candidates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    approach = -base_hand_joints_object_m()[0]
    best = orthonormal_pose(np.zeros(3), np.array([1.0, 0.0, 0.0]), approach)
    poses = np.repeat(best[None, ...], 5, axis=0)
    poses[1, :3, 3] = [0.012, 0.001, 0.0]
    poses[2] = orthonormal_pose(
        np.array([0.0, 0.0, 0.0]), np.array([0.92, 0.38, 0.0]), approach
    )
    poses[3, :3, 3] = [0.041, 0.0, 0.0]
    poses[4] = orthonormal_pose(
        np.array([0.0, -0.018, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.25, 1.0]),
    )
    scores = np.array([0.94, 0.97, 0.91, 0.99, 0.86], dtype=np.float32)
    widths_m = np.array([0.078, 0.078, 0.078, 0.078, 0.110], dtype=np.float32)
    return poses, scores, widths_m


def write_example(output_dir: Path) -> None:
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    segmentation_dir = output_dir / "segmentation"
    mask_dirs = {obj_id: segmentation_dir / "masks" / str(obj_id) for obj_id in (1, 2, 3)}
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    for directory in mask_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    intrinsics = np.array(
        [[600.0, 0.0, WIDTH / 2.0], [0.0, 600.0, HEIGHT / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    np.save(output_dir / "K.npy", intrinsics, allow_pickle=False)

    transforms = np.empty((FRAME_COUNT, 4, 4), dtype=np.float64)
    joints_camera = np.empty((FRAME_COUNT, 21, 3), dtype=np.float64)
    hand_confidence = np.full((FRAME_COUNT, 21), 0.97, dtype=np.float32)
    object_confidence = np.empty(FRAME_COUNT, dtype=np.float32)
    manifest_frames: list[dict[str, object]] = []

    for frame_idx in range(FRAME_COUNT):
        name = f"{frame_idx:06d}"
        transform = object_transform(frame_idx)
        joints_object = hand_joints_object_m(frame_idx)
        joints = transform_points(transform, joints_object)
        transforms[frame_idx] = transform
        joints_camera[frame_idx] = joints
        object_confidence[frame_idx] = 0.965 + 0.012 * np.sin(
            2.0 * np.pi * frame_idx / FRAME_COUNT
        )

        frame, depth = make_background()
        object_mask, part_mask = render_object(frame, depth, transform, intrinsics)
        hand_mask = render_hand(frame, depth, joints, intrinsics)
        object_mask[hand_mask > 0] = 0
        part_mask[hand_mask > 0] = 0

        cv2.imwrite(str(rgb_dir / f"{name}.png"), frame)
        np.save(depth_dir / f"{name}.npy", depth, allow_pickle=False)
        for obj_id, mask in ((1, hand_mask), (2, object_mask), (3, part_mask)):
            cv2.imwrite(str(mask_dirs[obj_id] / f"{name}.png"), mask)
        manifest_frames.append(
            {
                "frame_idx": frame_idx,
                "source": f"{name}.png",
                "masks": {
                    str(obj_id): f"masks/{obj_id}/{name}.png" for obj_id in (1, 2, 3)
                },
            }
        )

    np.savez_compressed(
        output_dir / "observations.npz",
        T_camera_object=transforms,
        hand_joints_camera_m=joints_camera,
        hand_joint_confidence=hand_confidence,
        timestamps_s=np.arange(FRAME_COUNT, dtype=np.float64) / FPS,
    )
    np.save(
        output_dir / "object_points_m.npy",
        box_surface_points(BOX_HALF_EXTENTS_M),
        allow_pickle=False,
    )
    poses, scores, widths_m = grasp_candidates()
    np.savez_compressed(
        output_dir / "candidates.npz", poses=poses, scores=scores, widths_m=widths_m
    )
    np.save(output_dir / "object_pose_confidence.npy", object_confidence, allow_pickle=False)

    manifest = {
        "backend": "synthetic_contract_demo",
        "synthetic": True,
        "purpose": "visualization_and_interface_contract_only",
        "mask_encoding": {"dtype": "uint8", "background": 0, "foreground": 255},
        "image_size_hw": [HEIGHT, WIDTH],
        "fps": FPS,
        "stable_contact_interval_half_open": [CONTACT_START, CONTACT_END],
        "objects": [
            {"obj_id": 1, "name": "right_hand", "source": "synthetic_geometry"},
            {"obj_id": 2, "name": "target_object", "source": "synthetic_geometry"},
            {"obj_id": 3, "name": "top_part", "source": "synthetic_geometry"},
        ],
        "frames": manifest_frames,
    }
    (segmentation_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    write_example(args.output_dir)
    print(
        f"Wrote synthetic contract demo: {FRAME_COUNT} frames at {WIDTH}x{HEIGHT} "
        f"to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
