"""Pure pose math shared by the ROS MoveIt integration tools."""

from __future__ import annotations

import math
from collections.abc import Sequence

QuaternionXYZW = tuple[float, float, float, float]


def normalize_quaternion_xyzw(values: Sequence[float]) -> QuaternionXYZW:
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("quaternion must contain four finite XYZW values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("quaternion must be non-zero")
    return (
        values[0] / norm,
        values[1] / norm,
        values[2] / norm,
        values[3] / norm,
    )


def quaternion_angle_rad(first_xyzw: QuaternionXYZW, second_xyzw: QuaternionXYZW) -> float:
    dot = abs(sum(a * b for a, b in zip(first_xyzw, second_xyzw, strict=True)))
    return 2.0 * math.acos(min(1.0, max(0.0, dot)))


def quaternion_multiply_xyzw(
    first: QuaternionXYZW, second: QuaternionXYZW
) -> QuaternionXYZW:
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return normalize_quaternion_xyzw(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


def quaternion_conjugate_xyzw(value: QuaternionXYZW) -> QuaternionXYZW:
    return (-value[0], -value[1], -value[2], value[3])


def rotate_vector(
    orientation_xyzw: QuaternionXYZW, vector: Sequence[float]
) -> list[float]:
    if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
        raise ValueError("vector must contain three finite values")
    qx, qy, qz, qw = orientation_xyzw
    vx, vy, vz = vector
    # Expanded q * [v, 0] * conjugate(q), avoiding an invalid zero quaternion.
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    ]


def compose_pose(
    parent_position: Sequence[float],
    parent_orientation: QuaternionXYZW,
    child_position: Sequence[float],
    child_orientation: QuaternionXYZW,
) -> tuple[list[float], QuaternionXYZW]:
    if len(parent_position) != 3 or not all(
        math.isfinite(value) for value in parent_position
    ):
        raise ValueError("parent position must contain three finite values")
    rotated_child = rotate_vector(parent_orientation, child_position)
    position = [
        parent + child
        for parent, child in zip(parent_position, rotated_child, strict=True)
    ]
    orientation = quaternion_multiply_xyzw(parent_orientation, child_orientation)
    return position, orientation


def link8_pose_from_configured_ee(
    ee_position: Sequence[float],
    ee_orientation: QuaternionXYZW,
    link8_to_ee_position: Sequence[float],
    link8_to_ee_orientation: QuaternionXYZW,
) -> tuple[list[float], QuaternionXYZW]:
    if len(ee_position) != 3 or not all(math.isfinite(value) for value in ee_position):
        raise ValueError("EE position must contain three finite values")
    link8_orientation = quaternion_multiply_xyzw(
        ee_orientation, quaternion_conjugate_xyzw(link8_to_ee_orientation)
    )
    offset_base = rotate_vector(link8_orientation, link8_to_ee_position)
    link8_position = [
        ee - offset for ee, offset in zip(ee_position, offset_base, strict=True)
    ]
    return link8_position, link8_orientation


def pose_delta(
    start_position: Sequence[float],
    start_orientation: QuaternionXYZW,
    target_position: Sequence[float],
    target_orientation: QuaternionXYZW,
) -> tuple[float, float]:
    if len(start_position) != 3 or len(target_position) != 3:
        raise ValueError("positions must each contain three values")
    translation_m = math.sqrt(
        sum(
            (target - start) ** 2
            for start, target in zip(start_position, target_position, strict=True)
        )
    )
    rotation_rad = quaternion_angle_rad(start_orientation, target_orientation)
    return translation_m, rotation_rad


def classify_measured_pose_outcome(
    *,
    start_position: Sequence[float],
    start_orientation: QuaternionXYZW,
    measured_position: Sequence[float],
    measured_orientation: QuaternionXYZW,
    target_position_error_m: float,
    target_rotation_error_rad: float,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    stationary_translation_m: float = 0.0005,
    stationary_rotation_rad: float = math.radians(0.5),
) -> tuple[str, float, float]:
    """Classify measured feedback after an action server reported success.

    ROS action success only says that the action server completed its protocol.  A
    controller which failed to enter the hardware command mode can still consume a
    trajectory for its full duration and return success while the robot stays put.
    Endpoint feedback therefore remains the authority for physical completion.

    The stationary thresholds are diagnostic tolerances for encoder/TF noise.  They
    do not authorize motion and do not relax the target endpoint tolerances.
    """

    finite_values = (
        target_position_error_m,
        target_rotation_error_rad,
        position_tolerance_m,
        orientation_tolerance_rad,
        stationary_translation_m,
        stationary_rotation_rad,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in finite_values):
        raise ValueError("pose outcome tolerances and errors must be finite and non-negative")

    measured_translation_m, measured_rotation_rad = pose_delta(
        start_position,
        start_orientation,
        measured_position,
        measured_orientation,
    )
    if (
        target_position_error_m <= position_tolerance_m
        and target_rotation_error_rad <= orientation_tolerance_rad
    ):
        return (
            "measured_target_reached",
            measured_translation_m,
            measured_rotation_rad,
        )
    if (
        measured_translation_m <= stationary_translation_m
        and measured_rotation_rad <= stationary_rotation_rad
    ):
        return (
            "controller_reported_success_without_measured_motion",
            measured_translation_m,
            measured_rotation_rad,
        )
    return (
        "measured_endpoint_mismatch",
        measured_translation_m,
        measured_rotation_rad,
    )
