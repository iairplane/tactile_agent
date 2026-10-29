"""Geometry helpers for a fixed-object, CAD-free demonstration.

The object and calibrated camera must remain stationary for the complete capture.
The camera optical frame is the reference frame for this diagnostic path; no CAD
object frame is invented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .intent import IntentConfig, infer_grasp_intent
from .models import GraspIntent, ObservationSequence


@dataclass(frozen=True)
class CloudMotionMetrics:
    """Nearest-neighbour evidence that two fixed-camera point clouds agree."""

    centroid_shift_m: float
    median_nearest_distance_m: float
    p95_nearest_distance_m: float


def validate_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=float)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {intrinsics.shape}")
    if not np.isfinite(intrinsics).all():
        raise ValueError("intrinsics contain NaN or infinity")
    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        raise ValueError("intrinsics focal lengths must be positive")
    if not np.allclose(intrinsics[2], [0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("intrinsics last row must be [0, 0, 1]")
    return intrinsics


def backproject_masked_depth(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    object_mask: np.ndarray,
    *,
    exclude_mask: np.ndarray | None = None,
    min_depth_m: float = 0.1,
    max_depth_m: float = 2.0,
) -> np.ndarray:
    """Backproject valid masked depth into the OpenCV color optical frame."""

    depth_m = np.asarray(depth_m, dtype=float)
    object_mask = np.asarray(object_mask, dtype=bool)
    intrinsics = validate_intrinsics(intrinsics)
    if depth_m.ndim != 2:
        raise ValueError(f"depth_m must be two-dimensional, got {depth_m.shape}")
    if object_mask.shape != depth_m.shape:
        raise ValueError(
            f"object_mask shape {object_mask.shape} does not match depth {depth_m.shape}"
        )
    if exclude_mask is not None:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)
        if exclude_mask.shape != depth_m.shape:
            raise ValueError(
                f"exclude_mask shape {exclude_mask.shape} does not match depth {depth_m.shape}"
            )
    if not 0.0 < min_depth_m < max_depth_m:
        raise ValueError("depth bounds must satisfy 0 < min_depth_m < max_depth_m")

    valid = (
        object_mask
        & np.isfinite(depth_m)
        & (depth_m >= min_depth_m)
        & (depth_m <= max_depth_m)
    )
    if exclude_mask is not None:
        valid &= ~exclude_mask
    rows, columns = np.nonzero(valid)
    if len(rows) == 0:
        return np.empty((0, 3), dtype=np.float64)

    z = depth_m[rows, columns]
    x = (columns.astype(float) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows.astype(float) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return np.column_stack((x, y, z))


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Average points falling in the same metric voxel."""

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points contain NaN or infinity")
    if voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m must be positive")
    if len(points) == 0:
        return points.astype(np.float64, copy=True)

    voxel_index = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(voxel_index, axis=0, return_inverse=True)
    sums = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
    counts = np.bincount(inverse).astype(np.float64)
    np.add.at(sums, inverse, points)
    return sums / counts[:, None]


def cloud_motion_metrics(first: np.ndarray, second: np.ndarray) -> CloudMotionMetrics:
    """Measure fixed-camera agreement with symmetric nearest-neighbour residuals."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    for name, points in (("first", first), ("second", second)):
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(f"{name} cloud must have non-empty shape (N, 3)")
        if not np.isfinite(points).all():
            raise ValueError(f"{name} cloud contains NaN or infinity")

    first_to_second = cKDTree(second).query(first, k=1)[0]
    second_to_first = cKDTree(first).query(second, k=1)[0]
    residuals = np.concatenate((first_to_second, second_to_first))
    centroid_shift = np.linalg.norm(np.median(first, axis=0) - np.median(second, axis=0))
    return CloudMotionMetrics(
        centroid_shift_m=float(centroid_shift),
        median_nearest_distance_m=float(np.median(residuals)),
        p95_nearest_distance_m=float(np.quantile(residuals, 0.95)),
    )


def make_fixed_camera_observation(
    hand_joints_camera_m: np.ndarray,
    hand_joint_confidence: np.ndarray,
    timestamps_s: np.ndarray,
    object_points_camera_m: np.ndarray,
) -> ObservationSequence:
    """Create the existing intent input with camera optical as a fixed reference."""

    joints = np.asarray(hand_joints_camera_m, dtype=float)
    confidence = np.asarray(hand_joint_confidence, dtype=float)
    timestamps = np.asarray(timestamps_s, dtype=float)
    object_points = np.asarray(object_points_camera_m, dtype=float)
    if joints.ndim != 3 or joints.shape[1:] != (21, 3):
        raise ValueError(f"hand joints must have shape (F, 21, 3), got {joints.shape}")
    frame_count = len(joints)
    if confidence.shape != (frame_count, 21):
        raise ValueError(
            f"hand confidence must have shape ({frame_count}, 21), got {confidence.shape}"
        )
    if timestamps.shape != (frame_count,):
        raise ValueError(f"timestamps must have shape ({frame_count},), got {timestamps.shape}")
    if frame_count == 0:
        raise ValueError("demonstration must contain at least one frame")
    if not np.isfinite(joints).all() or not np.isfinite(confidence).all():
        raise ValueError("hand arrays contain NaN or infinity")
    if not np.isfinite(timestamps).all() or (frame_count > 1 and np.any(np.diff(timestamps) <= 0)):
        raise ValueError("timestamps must be finite and strictly increasing")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("hand confidence values must lie in [0, 1]")
    if object_points.ndim != 2 or object_points.shape[1] != 3 or len(object_points) == 0:
        raise ValueError("object point cloud must have non-empty shape (N, 3)")
    if not np.isfinite(object_points).all():
        raise ValueError("object point cloud contains NaN or infinity")

    fixed_reference = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    return ObservationSequence(
        T_camera_object=fixed_reference,
        hand_joints_camera_m=joints,
        hand_joint_confidence=confidence,
        timestamps_s=timestamps,
        object_points_m=object_points,
    )


def infer_fixed_camera_intent(
    hand_joints_camera_m: np.ndarray,
    hand_joint_confidence: np.ndarray,
    timestamps_s: np.ndarray,
    object_points_camera_m: np.ndarray,
    config: IntentConfig = IntentConfig(),
) -> GraspIntent:
    """Infer a grasp whose numeric pose is ``T_camera_gripper_tcp``."""

    observation = make_fixed_camera_observation(
        hand_joints_camera_m,
        hand_joint_confidence,
        timestamps_s,
        object_points_camera_m,
    )
    return infer_grasp_intent(observation, config)
