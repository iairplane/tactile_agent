"""Small geometry helpers using the T_parent_child convention."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def invert_transforms(transforms: np.ndarray) -> np.ndarray:
    """Invert one transform or a batch of transforms."""

    transforms = np.asarray(transforms, dtype=float)
    rotation = transforms[..., :3, :3]
    translation = transforms[..., :3, 3]
    inverse = np.zeros_like(transforms, dtype=float)
    inverse[..., :3, :3] = np.swapaxes(rotation, -1, -2)
    inverse[..., :3, 3] = -np.einsum(
        "...ij,...j->...i", inverse[..., :3, :3], translation
    )
    inverse[..., 3, 3] = 1.0
    return inverse


def transform_points(transforms: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply T_parent_child to point sets shaped ``(..., N, 3)``."""

    transforms = np.asarray(transforms, dtype=float)
    points = np.asarray(points, dtype=float)
    rotation = transforms[..., :3, :3]
    translation = transforms[..., :3, 3]
    return np.einsum("...ij,...nj->...ni", rotation, points) + translation[..., None, :]


def normalize(vector: np.ndarray, *, name: str = "vector") -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError(f"cannot normalize zero-length {name}")
    return np.asarray(vector, dtype=float) / norm


def continuous_symmetry_gauge_rotation(
    symmetry_axis: int,
    contact_center: np.ndarray,
    closing_direction: np.ndarray,
    approach_direction: np.ndarray,
) -> np.ndarray:
    """Rotate an axial-symmetry gauge to a fixed transverse reference.

    The first usable transverse direction is chosen in the order contact
    center, closing direction, then approach direction. For Z symmetry the
    fixed reference is +X.
    """

    if symmetry_axis not in (0, 1, 2):
        raise ValueError("symmetry_axis must be 0, 1, or 2")
    axis = np.eye(3)[symmetry_axis]
    reference_index = 1 if symmetry_axis == 0 else 0
    reference = np.eye(3)[reference_index]

    for feature in (contact_center, closing_direction, approach_direction):
        transverse = np.asarray(feature, dtype=float) - np.dot(feature, axis) * axis
        if np.linalg.norm(transverse) <= 1e-8:
            continue
        transverse = normalize(transverse)
        source_basis = np.column_stack(
            (transverse, np.cross(axis, transverse), axis)
        )
        reference_basis = np.column_stack(
            (reference, np.cross(axis, reference), axis)
        )
        return reference_basis @ source_basis.T
    raise ValueError("cannot determine a gauge around the continuous symmetry axis")


def canonicalize_continuous_symmetry_pose(
    pose: np.ndarray, symmetry_axis: int
) -> np.ndarray:
    """Return a gauge-fixed copy of an object-frame TCP pose."""

    pose = np.asarray(pose, dtype=float)
    gauge = continuous_symmetry_gauge_rotation(
        symmetry_axis,
        pose[:3, 3],
        pose[:3, 0],
        pose[:3, 2],
    )
    canonical = pose.copy()
    canonical[:3, :3] = gauge @ pose[:3, :3]
    canonical[:3, 3] = gauge @ pose[:3, 3]
    return canonical


def robust_mean(samples: np.ndarray) -> np.ndarray:
    """Mean after rejecting samples outside three median absolute deviations."""

    samples = np.asarray(samples, dtype=float)
    center = np.median(samples, axis=0)
    residuals = np.linalg.norm(samples - center, axis=-1)
    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual)))
    cutoff = median_residual + 3.0 * max(mad, 1e-9)
    inliers = samples[residuals <= cutoff]
    return np.mean(inliers, axis=0)


def robust_direction_mean(directions: np.ndarray, *, name: str) -> np.ndarray:
    directions = np.asarray(directions, dtype=float)
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    reference = normalize(np.median(unit, axis=0), name=name)
    angles = np.arccos(np.clip(unit @ reference, -1.0, 1.0))
    median_angle = float(np.median(angles))
    mad = float(np.median(np.abs(angles - median_angle)))
    cutoff = median_angle + 3.0 * max(mad, 1e-9)
    return normalize(np.mean(unit[angles <= cutoff], axis=0), name=name)


def project_to_point_cloud(
    query_points: np.ndarray, point_cloud: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-point distances and projected points."""

    tree = cKDTree(np.asarray(point_cloud, dtype=float))
    distances, indices = tree.query(np.asarray(query_points, dtype=float), k=1)
    return np.asarray(distances), np.asarray(point_cloud, dtype=float)[indices]


def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """Return the earliest longest half-open interval containing only True."""

    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], mask, [False]))
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    starts = transitions[::2]
    ends = transitions[1::2]
    if len(starts) == 0:
        return 0, 0
    best = int(np.argmax(ends - starts))
    return int(starts[best]), int(ends[best])
