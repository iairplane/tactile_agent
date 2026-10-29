"""Pure geometry for an axial press-and-twist medicine-bottle cap motion.

The module deliberately has no robot, ROS, gripper, or tactile dependencies.  It
defines a strict frame contract and produces bounded ``CartesianPose`` values
that can be reviewed by a higher-level task compiler.  Producing these values
does not authorize their execution.

Coordinates follow the repository convention: right-handed, metres, column
vectors, and ``T_parent_child``.  ``bottle_axis_base`` points from the bottle
base toward the cap.  PRESS therefore moves along the negative axis, while an
axial LIFT moves along the positive axis.  Quaternions are serialized XYZW.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cartesian_pose_sequence import (
    MAX_SEGMENT_ROTATION_RAD,
    MAX_SEGMENT_TRANSLATION_M,
    MAX_WAYPOINTS,
    POSE_SEQUENCE_SCHEMA,
    CartesianPose,
    validate_pose_sequence_motion,
)
from .franka import validate_rigid_transform

BOTTLE_CAP_FRAME_SCHEMA = "demo2gripper.medicine_bottle_cap_frame.v1"
PRESS_TWIST_GEOMETRY_SCHEMA = "demo2gripper.medicine_bottle_press_twist_geometry.v1"
AXIAL_LIFT_GEOMETRY_SCHEMA = "demo2gripper.medicine_bottle_axial_lift_geometry.v1"

FRAME_CONVENTION = "T_parent_child; right-handed; column vectors"
AXIS_DIRECTION = "bottle_base_to_cap"
UNIT_VECTOR_ATOL = 1e-6
AXIS_ALIGNMENT_ATOL = 1e-6

# A lift is deliberately sampled more finely than the generic 50 mm Cartesian
# segment cap.  This is a geometry bound, not a claim that a path is collision
# free or safe to execute.
MAX_AXIAL_LIFT_STEP_M = 0.020


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_scalar(value: object, *, name: str) -> float:
    result = _finite_scalar(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _unit_vector(values: object, *, name: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must contain exactly three finite numbers")
    raw_object = np.asarray(values, dtype=object)
    if raw_object.shape != (3,) or any(
        isinstance(value, (bool, np.bool_)) for value in raw_object
    ):
        raise ValueError(f"{name} must contain exactly three finite numbers")
    try:
        vector = raw_object.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly three finite numbers") from exc
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain exactly three finite numbers")
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=UNIT_VECTOR_ATOL):
        raise ValueError(f"{name} must be a unit vector; norm is {norm:.9g}")
    normalized = vector / norm
    return tuple(float(value) for value in normalized)


def _readonly_transform(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {raw.shape}")
    if any(isinstance(item, (bool, np.bool_)) for item in raw.flat):
        raise ValueError(f"{name} must contain only finite numbers")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite numbers") from exc
    transform = validate_rigid_transform(numeric, name=name).copy()
    transform.setflags(write=False)
    return transform


def _require_axis_alignment(
    transform: np.ndarray,
    *,
    bottle_axis_base: Sequence[float],
    bottle_axis_tcp: Sequence[float],
    name: str,
) -> None:
    represented = transform[:3, :3] @ np.asarray(bottle_axis_tcp, dtype=float)
    error = float(np.linalg.norm(represented - np.asarray(bottle_axis_base, dtype=float)))
    if error > AXIS_ALIGNMENT_ATOL:
        raise ValueError(
            f"{name} orientation does not map bottle_axis_tcp onto bottle_axis_base "
            f"(vector error {error:.6g})"
        )


@dataclass(frozen=True, eq=False)
class BottleCapFrame:
    """Reviewed cap-centred TCP frame and its signed bottle axis.

    ``T_base_cap_tcp`` is the TCP pose at the cap centre before pressing.  Its
    origin is assumed to lie on the cap's rotation axis.  ``bottle_axis_tcp``
    makes the otherwise ambiguous local TCP direction explicit and must map to
    ``bottle_axis_base`` through the transform rotation.
    """

    base_frame: str
    cap_frame: str
    tcp_frame: str
    T_base_cap_tcp: np.ndarray
    bottle_axis_base: tuple[float, float, float]
    bottle_axis_tcp: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "base_frame", _nonempty_string(self.base_frame, name="base_frame")
        )
        object.__setattr__(
            self, "cap_frame", _nonempty_string(self.cap_frame, name="cap_frame")
        )
        object.__setattr__(
            self, "tcp_frame", _nonempty_string(self.tcp_frame, name="tcp_frame")
        )
        if len({self.base_frame, self.cap_frame, self.tcp_frame}) != 3:
            raise ValueError("base_frame, cap_frame, and tcp_frame must be distinct")
        transform = _readonly_transform(self.T_base_cap_tcp, name="T_base_cap_tcp")
        axis_base = _unit_vector(self.bottle_axis_base, name="bottle_axis_base")
        axis_tcp = _unit_vector(self.bottle_axis_tcp, name="bottle_axis_tcp")
        _require_axis_alignment(
            transform,
            bottle_axis_base=axis_base,
            bottle_axis_tcp=axis_tcp,
            name="T_base_cap_tcp",
        )
        object.__setattr__(self, "T_base_cap_tcp", transform)
        object.__setattr__(self, "bottle_axis_base", axis_base)
        object.__setattr__(self, "bottle_axis_tcp", axis_tcp)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": BOTTLE_CAP_FRAME_SCHEMA,
            "transform_convention": FRAME_CONVENTION,
            "units": "metres",
            "axis_direction": AXIS_DIRECTION,
            "base_frame": self.base_frame,
            "cap_frame": self.cap_frame,
            "tcp_frame": self.tcp_frame,
            "T_base_cap_tcp": self.T_base_cap_tcp.tolist(),
            "bottle_axis_base": list(self.bottle_axis_base),
            "bottle_axis_tcp": list(self.bottle_axis_tcp),
        }


_FRAME_KEYS = frozenset(
    {
        "schema",
        "transform_convention",
        "units",
        "axis_direction",
        "base_frame",
        "cap_frame",
        "tcp_frame",
        "T_base_cap_tcp",
        "bottle_axis_base",
        "bottle_axis_tcp",
    }
)


def parse_bottle_cap_frame_payload(
    payload: object, *, expected_base_frame: str | None = None
) -> BottleCapFrame:
    """Parse the strict JSON-compatible bottle/cap frame contract."""

    if not isinstance(payload, Mapping):
        raise ValueError("bottle-cap frame payload must be an object")
    keys = set(payload)
    if any(not isinstance(key, str) for key in keys):
        raise ValueError("bottle-cap frame payload field names must be strings")
    missing = sorted(_FRAME_KEYS - keys)
    unknown = sorted(keys - _FRAME_KEYS)
    if missing:
        raise ValueError(f"bottle-cap frame payload is missing fields: {missing}")
    if unknown:
        raise ValueError(f"bottle-cap frame payload has unknown fields: {unknown}")
    if payload.get("schema") != BOTTLE_CAP_FRAME_SCHEMA:
        raise ValueError(f"bottle-cap frame schema must be {BOTTLE_CAP_FRAME_SCHEMA!r}")
    if payload.get("transform_convention") != FRAME_CONVENTION:
        raise ValueError(f"transform_convention must be {FRAME_CONVENTION!r}")
    if payload.get("units") != "metres":
        raise ValueError("bottle-cap frame units must be 'metres'")
    if payload.get("axis_direction") != AXIS_DIRECTION:
        raise ValueError(f"axis_direction must be {AXIS_DIRECTION!r}")
    base_frame = _nonempty_string(payload.get("base_frame"), name="base_frame")
    if expected_base_frame is not None and base_frame != expected_base_frame:
        raise ValueError(
            f"bottle-cap base_frame {base_frame!r} does not match "
            f"expected base frame {expected_base_frame!r}"
        )
    return BottleCapFrame(
        base_frame=base_frame,
        cap_frame=payload.get("cap_frame"),  # type: ignore[arg-type]
        tcp_frame=payload.get("tcp_frame"),  # type: ignore[arg-type]
        T_base_cap_tcp=payload.get("T_base_cap_tcp"),  # type: ignore[arg-type]
        bottle_axis_base=payload.get("bottle_axis_base"),  # type: ignore[arg-type]
        bottle_axis_tcp=payload.get("bottle_axis_tcp"),  # type: ignore[arg-type]
    )


def _axis_angle_rotation(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    identity = np.eye(3)
    return identity + math.sin(angle_rad) * skew + (1.0 - math.cos(angle_rad)) * (
        skew @ skew
    )


def _rotation_matrix_to_quaternion_xyzw(
    rotation: np.ndarray, *, reference: Sequence[float] | None = None
) -> tuple[float, float, float, float]:
    """Convert a proper rotation to a deterministic, optionally continuous XYZW q."""

    matrix = np.asarray(rotation, dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = matrix
    validate_rigid_transform(transform, name="pose rotation")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=float,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            values = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            values = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            values = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    values /= np.linalg.norm(values)
    if reference is None:
        if values[3] < 0.0:
            values *= -1.0
    elif float(np.dot(values, np.asarray(reference, dtype=float))) < 0.0:
        values *= -1.0
    return tuple(float(value) for value in values)


def _pose_from_transform(
    transform: np.ndarray, *, reference_quaternion: Sequence[float] | None = None
) -> CartesianPose:
    checked = validate_rigid_transform(transform, name="T_base_tcp waypoint")
    return CartesianPose(
        position_m=tuple(float(value) for value in checked[:3, 3]),
        orientation_xyzw=_rotation_matrix_to_quaternion_xyzw(
            checked[:3, :3], reference=reference_quaternion
        ),
    )


def _pose_sequence_payload(
    waypoints: Sequence[CartesianPose], *, base_frame: str
) -> dict[str, Any]:
    return {
        "schema": POSE_SEQUENCE_SCHEMA,
        "base_frame": base_frame,
        "waypoints": [waypoint.as_dict() for waypoint in waypoints],
    }


@dataclass(frozen=True)
class PressTwistGeometry:
    frame: BottleCapFrame
    initial_pose: CartesianPose
    press_waypoints: tuple[CartesianPose, ...]
    twist_waypoints: tuple[CartesianPose, ...]
    press_distance_m: float
    twist_step_angle_rad: float
    twist_steps: int
    thread_pitch_m_per_revolution: float
    motion_validation: Mapping[str, Any]

    @property
    def waypoints(self) -> tuple[CartesianPose, ...]:
        return self.press_waypoints + self.twist_waypoints

    def as_dict(self) -> dict[str, Any]:
        pitch_step = self.thread_pitch_m_per_revolution * abs(
            self.twist_step_angle_rad
        ) / (2.0 * math.pi)
        return {
            "schema": PRESS_TWIST_GEOMETRY_SCHEMA,
            "status": "geometry_only",
            "execution_authorized": False,
            "hardware_execution_enabled": False,
            "frame": self.frame.as_dict(),
            "pose_sequences": {
                "PRESS": _pose_sequence_payload(
                    self.press_waypoints, base_frame=self.frame.base_frame
                ),
                "TWIST": _pose_sequence_payload(
                    self.twist_waypoints, base_frame=self.frame.base_frame
                ),
            },
            "parameters": {
                "press_distance_m": self.press_distance_m,
                "twist_step_angle_rad": self.twist_step_angle_rad,
                "twist_step_angle_deg": math.degrees(self.twist_step_angle_rad),
                "twist_steps": self.twist_steps,
                "total_signed_twist_angle_rad": (
                    self.twist_steps * self.twist_step_angle_rad
                ),
                "thread_pitch_m_per_revolution": self.thread_pitch_m_per_revolution,
                "axial_pitch_displacement_per_step_m": pitch_step,
            },
            "geometric_criteria": {
                "PRESS": {
                    "translation_direction": "negative_bottle_axis_base",
                    "target_axial_offset_from_initial_m": -self.press_distance_m,
                    "lateral_offset_from_cap_axis_m": 0.0,
                    "orientation_change_rad": 0.0,
                },
                "TWIST": {
                    "rotation_axis": "positive_bottle_axis_base",
                    "max_abs_rotation_per_step_rad": MAX_SEGMENT_ROTATION_RAD,
                    "coaxial_lateral_drift_m": 0.0,
                    "pitch_rule": (
                        "positive bottle-axis displacement equals pitch times "
                        "absolute accumulated turns"
                    ),
                    "final_axial_offset_from_initial_m": (
                        -self.press_distance_m + pitch_step * self.twist_steps
                    ),
                },
            },
            "cartesian_motion_validation": dict(self.motion_validation),
        }


def build_press_twist_geometry(
    frame: BottleCapFrame,
    *,
    press_distance_m: float,
    twist_step_angle_rad: float,
    twist_steps: int,
    thread_pitch_m_per_revolution: float = 0.0,
) -> PressTwistGeometry:
    """Build one PRESS waypoint followed by bounded, coaxial TWIST waypoints.

    A positive twist angle follows the right-hand rule around
    ``bottle_axis_base``.  Thread pitch is a non-negative magnitude: when it is
    non-zero, every twist step moves outward along the positive bottle axis by
    ``pitch * abs(step_angle) / (2*pi)`` regardless of twist direction.
    """

    if not isinstance(frame, BottleCapFrame):
        raise TypeError("frame must be a BottleCapFrame")
    press_distance = _positive_scalar(press_distance_m, name="press_distance_m")
    if press_distance > MAX_SEGMENT_TRANSLATION_M + 1e-12:
        raise ValueError(
            f"press_distance_m exceeds the {MAX_SEGMENT_TRANSLATION_M:.3f} m "
            "Cartesian segment cap"
        )
    step_angle = _finite_scalar(twist_step_angle_rad, name="twist_step_angle_rad")
    if abs(step_angle) <= 1e-12:
        raise ValueError("twist_step_angle_rad must be non-zero")
    if abs(step_angle) > MAX_SEGMENT_ROTATION_RAD + 1e-12:
        raise ValueError(
            "absolute twist_step_angle_rad must be at most "
            f"{MAX_SEGMENT_ROTATION_RAD:.12g} rad (20 deg)"
        )
    if isinstance(twist_steps, (bool, np.bool_)) or not isinstance(
        twist_steps, (int, np.integer)
    ):
        raise ValueError("twist_steps must be a positive integer")
    steps = int(twist_steps)
    if steps <= 0:
        raise ValueError("twist_steps must be a positive integer")
    if 1 + steps > MAX_WAYPOINTS:
        raise ValueError(
            f"PRESS plus TWIST must contain at most {MAX_WAYPOINTS} waypoints"
        )
    pitch = _finite_scalar(
        thread_pitch_m_per_revolution, name="thread_pitch_m_per_revolution"
    )
    if pitch < 0.0:
        raise ValueError("thread_pitch_m_per_revolution must be non-negative")

    initial = frame.T_base_cap_tcp
    axis = np.asarray(frame.bottle_axis_base, dtype=float)
    press_transform = initial.copy()
    press_transform[:3, 3] = initial[:3, 3] - press_distance * axis
    initial_pose = _pose_from_transform(initial)
    press_pose = _pose_from_transform(
        press_transform, reference_quaternion=initial_pose.orientation_xyzw
    )

    pitch_per_step = pitch * abs(step_angle) / (2.0 * math.pi)
    twist_poses: list[CartesianPose] = []
    previous_quaternion = press_pose.orientation_xyzw
    for step_index in range(1, steps + 1):
        accumulated_angle = step_index * step_angle
        transform = press_transform.copy()
        transform[:3, :3] = (
            _axis_angle_rotation(axis, accumulated_angle) @ initial[:3, :3]
        )
        transform[:3, 3] = press_transform[:3, 3] + (
            step_index * pitch_per_step * axis
        )
        _require_axis_alignment(
            transform,
            bottle_axis_base=frame.bottle_axis_base,
            bottle_axis_tcp=frame.bottle_axis_tcp,
            name=f"TWIST waypoint {step_index}",
        )
        pose = _pose_from_transform(transform, reference_quaternion=previous_quaternion)
        twist_poses.append(pose)
        previous_quaternion = pose.orientation_xyzw

    all_waypoints = (press_pose, *twist_poses)
    metrics = validate_pose_sequence_motion(initial_pose, all_waypoints)
    return PressTwistGeometry(
        frame=frame,
        initial_pose=initial_pose,
        press_waypoints=(press_pose,),
        twist_waypoints=tuple(twist_poses),
        press_distance_m=press_distance,
        twist_step_angle_rad=step_angle,
        twist_steps=steps,
        thread_pitch_m_per_revolution=pitch,
        motion_validation=metrics,
    )


@dataclass(frozen=True)
class AxialLiftGeometry:
    frame: BottleCapFrame
    start_pose: CartesianPose
    lift_waypoints: tuple[CartesianPose, ...]
    lift_distance_m: float
    max_step_m: float
    motion_validation: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": AXIAL_LIFT_GEOMETRY_SCHEMA,
            "status": "geometry_only",
            "execution_authorized": False,
            "hardware_execution_enabled": False,
            "frame": self.frame.as_dict(),
            "pose_sequences": {
                "LIFT": _pose_sequence_payload(
                    self.lift_waypoints, base_frame=self.frame.base_frame
                )
            },
            "parameters": {
                "lift_distance_m": self.lift_distance_m,
                "max_step_m": self.max_step_m,
                "waypoint_count": len(self.lift_waypoints),
            },
            "geometric_criteria": {
                "LIFT": {
                    "translation_direction": "positive_bottle_axis_base",
                    "target_axial_offset_from_start_m": self.lift_distance_m,
                    "strictly_monotonic_axial_motion": True,
                    "lateral_offset_from_cap_axis_m": 0.0,
                    "orientation_change_rad": 0.0,
                    "max_segment_translation_m": self.max_step_m,
                }
            },
            "cartesian_motion_validation": dict(self.motion_validation),
        }


def build_axial_lift_geometry(
    frame: BottleCapFrame,
    *,
    T_base_start_tcp: object,
    lift_distance_m: float,
    max_step_m: float = MAX_AXIAL_LIFT_STEP_M,
) -> AxialLiftGeometry:
    """Sample a fixed-orientation lift along the positive bottle axis."""

    if not isinstance(frame, BottleCapFrame):
        raise TypeError("frame must be a BottleCapFrame")
    start = _readonly_transform(T_base_start_tcp, name="T_base_start_tcp")
    _require_axis_alignment(
        start,
        bottle_axis_base=frame.bottle_axis_base,
        bottle_axis_tcp=frame.bottle_axis_tcp,
        name="T_base_start_tcp",
    )
    distance = _positive_scalar(lift_distance_m, name="lift_distance_m")
    step_limit = _positive_scalar(max_step_m, name="max_step_m")
    if step_limit > MAX_AXIAL_LIFT_STEP_M + 1e-12:
        raise ValueError(
            f"max_step_m is capped at {MAX_AXIAL_LIFT_STEP_M:.3f} m for axial lift"
        )
    count = int(math.ceil(distance / step_limit))
    if count > MAX_WAYPOINTS:
        raise ValueError(
            f"axial lift needs {count} waypoints and exceeds the {MAX_WAYPOINTS}-waypoint cap"
        )
    actual_step = distance / count
    if actual_step > MAX_SEGMENT_TRANSLATION_M + 1e-12:
        raise ValueError("axial lift segment exceeds the Cartesian translation cap")

    axis = np.asarray(frame.bottle_axis_base, dtype=float)
    start_pose = _pose_from_transform(start)
    waypoints: list[CartesianPose] = []
    for step_index in range(1, count + 1):
        transform = start.copy()
        transform[:3, 3] = start[:3, 3] + step_index * actual_step * axis
        waypoints.append(
            _pose_from_transform(
                transform, reference_quaternion=start_pose.orientation_xyzw
            )
        )
    metrics = validate_pose_sequence_motion(start_pose, waypoints)
    return AxialLiftGeometry(
        frame=frame,
        start_pose=start_pose,
        lift_waypoints=tuple(waypoints),
        lift_distance_m=distance,
        max_step_m=step_limit,
        motion_validation=metrics,
    )


def cartesian_pose_to_transform(pose: CartesianPose) -> np.ndarray:
    """Convert a generated ``CartesianPose`` back to ``T_base_tcp``."""

    if not isinstance(pose, CartesianPose):
        raise TypeError("pose must be a CartesianPose")
    x, y, z, w = pose.orientation_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("pose quaternion must be finite and non-zero")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    transform = np.eye(4)
    transform[:3, :3] = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
    transform[:3, 3] = pose.position_m
    return validate_rigid_transform(transform, name="T_base_tcp from CartesianPose")


__all__ = [
    "AXIAL_LIFT_GEOMETRY_SCHEMA",
    "AXIS_DIRECTION",
    "BOTTLE_CAP_FRAME_SCHEMA",
    "FRAME_CONVENTION",
    "MAX_AXIAL_LIFT_STEP_M",
    "PRESS_TWIST_GEOMETRY_SCHEMA",
    "AxialLiftGeometry",
    "BottleCapFrame",
    "PressTwistGeometry",
    "build_axial_lift_geometry",
    "build_press_twist_geometry",
    "cartesian_pose_to_transform",
    "parse_bottle_cap_frame_payload",
]
