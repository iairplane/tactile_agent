"""Deterministic fixtures for exercising the pipeline without model weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _box_surface_points(half_extents: np.ndarray, samples: int = 17) -> np.ndarray:
    coordinates = [
        np.linspace(-half_extents[axis], half_extents[axis], samples)
        for axis in range(3)
    ]
    points: list[np.ndarray] = []
    for fixed_axis in range(3):
        free_axes = [axis for axis in range(3) if axis != fixed_axis]
        first, second = np.meshgrid(
            coordinates[free_axes[0]], coordinates[free_axes[1]], indexing="ij"
        )
        for side in (-1.0, 1.0):
            face = np.zeros((samples * samples, 3), dtype=float)
            face[:, fixed_axis] = side * half_extents[fixed_axis]
            face[:, free_axes[0]] = first.ravel()
            face[:, free_axes[1]] = second.ravel()
            points.append(face)
    return np.unique(np.concatenate(points, axis=0), axis=0)


def _rotation_y(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=float,
    )


def write_synthetic_inputs(output_dir: str | Path) -> tuple[Path, Path, Path]:
    """Write a moving-camera pinch sequence and grasp candidates."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 12
    object_points = _box_surface_points(np.array([0.03, 0.025, 0.04]))
    transforms = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)
    joints_camera = np.zeros((frame_count, 21, 3), dtype=float)
    confidence = np.full((frame_count, 21), 0.95, dtype=float)

    rng = np.random.default_rng(7)
    for frame in range(frame_count):
        transforms[frame, :3, :3] = _rotation_y(-0.06 + 0.01 * frame)
        transforms[frame, :3, 3] = np.array(
            [0.32 + 0.002 * frame, -0.04, 0.72 - 0.001 * frame]
        )

        joints_object = np.zeros((21, 3), dtype=float)
        joints_object[:] = np.array([0.0, 0.0, -0.10])
        joints_object[:, 1] += np.linspace(-0.025, 0.025, 21)
        joints_object[0] = np.array([0.0, 0.0, -0.13])

        if 3 <= frame < 10:
            noise = rng.normal(scale=0.00035, size=(2, 3))
            joints_object[4] = np.array([-0.03, 0.0, 0.0]) + noise[0]
            joints_object[8] = np.array([0.03, 0.0, 0.0]) + noise[1]
        else:
            joints_object[4] = np.array([-0.045, 0.0, -0.06])
            joints_object[8] = np.array([0.045, 0.0, -0.06])

        rotation = transforms[frame, :3, :3]
        translation = transforms[frame, :3, 3]
        joints_camera[frame] = joints_object @ rotation.T + translation

    observations_path = output_dir / "observations.npz"
    np.savez_compressed(
        observations_path,
        T_camera_object=transforms,
        hand_joints_camera_m=joints_camera,
        hand_joint_confidence=confidence,
        timestamps_s=np.arange(frame_count, dtype=float) / 30.0,
    )
    object_points_path = output_dir / "object_points_m.npy"
    np.save(object_points_path, object_points)

    poses = np.repeat(np.eye(4)[None, ...], 5, axis=0)
    poses[0, :3, 3] = np.array([0.001, 0.0, 0.0])
    poses[1, :3, 3] = np.array([0.035, 0.0, 0.0])
    poses[2, :3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    poses[3, :3, 3] = np.array([0.0, 0.0, 0.0])
    poses[4, :3, :3] = np.diag([-1.0, -1.0, 1.0])

    candidates_path = output_dir / "candidates.npz"
    np.savez_compressed(
        candidates_path,
        poses=poses,
        scores=np.array([0.90, 1.00, 0.82, 0.95, 0.84]),
        widths_m=np.array([0.061, 0.060, 0.060, 0.11, 0.061]),
    )
    return observations_path, object_points_path, candidates_path
