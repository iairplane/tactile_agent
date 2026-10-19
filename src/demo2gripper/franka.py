"""Franka target poses derived from an object-frame grasp."""

from __future__ import annotations

import math

import numpy as np


def validate_rigid_transform(transform: np.ndarray, *, name: str) -> np.ndarray:
    """Return a validated 4x4 rigid transform."""

    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}")
    if not np.isfinite(transform).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return transform


def translation_transform(translation_m: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(translation_m, dtype=float).reshape(3)
    return transform


def rotation_z_transform(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    transform = np.eye(4)
    transform[:3, :3] = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return transform


def make_tcp_targets(
    T_base_object: np.ndarray,
    T_object_gripper_tcp: np.ndarray,
    *,
    approach_distance_m: float,
    lift_height_m: float,
) -> dict[str, np.ndarray]:
    """Create canonical TCP targets for approach, grasp, and vertical lift."""

    if approach_distance_m <= 0.0:
        raise ValueError("approach_distance_m must be positive")
    if lift_height_m <= 0.0:
        raise ValueError("lift_height_m must be positive")
    base_object = validate_rigid_transform(T_base_object, name="T_base_object")
    object_tcp = validate_rigid_transform(
        T_object_gripper_tcp, name="T_object_gripper_tcp"
    )
    grasp = base_object @ object_tcp
    pregrasp = grasp @ translation_transform([0.0, 0.0, -approach_distance_m])
    lift = grasp.copy()
    lift[:3, 3] += [0.0, 0.0, lift_height_m]
    return {"pregrasp": pregrasp, "grasp": grasp, "lift": lift}


def canonical_tcp_to_panda_hand(
    T_base_tcp: np.ndarray, *, tcp_offset_m: float = 0.1034
) -> np.ndarray:
    """Map the canonical +X-closing TCP to the stock Panda ``panda_hand`` frame."""

    if tcp_offset_m <= 0.0:
        raise ValueError("tcp_offset_m must be positive")
    base_tcp = validate_rigid_transform(T_base_tcp, name="T_base_tcp")
    return (
        base_tcp
        @ rotation_z_transform(-0.5 * math.pi)
        @ translation_transform([0.0, 0.0, -tcp_offset_m])
    )


def panda_hand_targets(
    tcp_targets: dict[str, np.ndarray], *, tcp_offset_m: float = 0.1034
) -> dict[str, np.ndarray]:
    return {
        name: canonical_tcp_to_panda_hand(transform, tcp_offset_m=tcp_offset_m)
        for name, transform in tcp_targets.items()
    }
