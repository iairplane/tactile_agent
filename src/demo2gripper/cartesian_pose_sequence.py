"""Pure validation for bounded configured-EE Cartesian pose sequences."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from demo2gripper.pose_motion import (
    QuaternionXYZW,
    link8_pose_from_configured_ee,
    normalize_quaternion_xyzw,
    pose_delta,
)

POSE_SEQUENCE_SCHEMA = "demo2gripper.configured_ee_pose_sequence.v1"
MAX_WAYPOINTS = 32
MAX_SEGMENT_TRANSLATION_M = 0.05
MAX_SEGMENT_ROTATION_RAD = math.radians(20.0)
MAX_TOTAL_CARTESIAN_LENGTH_M = 0.5
MAX_CARTESIAN_SPEED_M_S = 0.025


@dataclass(frozen=True)
class CartesianPose:
    """One absolute pose in a declared base frame."""

    position_m: tuple[float, float, float]
    orientation_xyzw: QuaternionXYZW

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "position_m": list(self.position_m),
            "orientation_xyzw": list(self.orientation_xyzw),
        }


def _finite_vector(values: object, *, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError(f"{name} must contain only finite numbers")
    return tuple(float(value) for value in values)


def parse_pose_sequence_payload(
    payload: object, *, expected_base_frame: str
) -> list[CartesianPose]:
    """Parse the strict configured-EE waypoint JSON contract."""

    if not isinstance(payload, Mapping):
        raise ValueError("pose-sequence JSON must be an object")
    if payload.get("schema") != POSE_SEQUENCE_SCHEMA:
        raise ValueError(f"pose-sequence schema must be {POSE_SEQUENCE_SCHEMA!r}")
    base_frame = payload.get("base_frame")
    if base_frame != expected_base_frame:
        raise ValueError(
            f"pose-sequence base_frame {base_frame!r} does not match "
            f"--base-frame {expected_base_frame!r}"
        )
    raw_waypoints = payload.get("waypoints")
    if not isinstance(raw_waypoints, list):
        raise ValueError("pose-sequence waypoints must be an array")
    if not 1 <= len(raw_waypoints) <= MAX_WAYPOINTS:
        raise ValueError(f"pose-sequence must contain 1..{MAX_WAYPOINTS} waypoints")

    waypoints: list[CartesianPose] = []
    for index, raw_waypoint in enumerate(raw_waypoints):
        if not isinstance(raw_waypoint, Mapping):
            raise ValueError(f"waypoint {index} must be an object")
        position = _finite_vector(
            raw_waypoint.get("position_m"),
            length=3,
            name=f"waypoint {index} position_m",
        )
        raw_orientation = _finite_vector(
            raw_waypoint.get("orientation_xyzw"),
            length=4,
            name=f"waypoint {index} orientation_xyzw",
        )
        orientation = normalize_quaternion_xyzw(raw_orientation)
        waypoints.append(
            CartesianPose(
                position_m=(position[0], position[1], position[2]),
                orientation_xyzw=orientation,
            )
        )
    return waypoints


def load_pose_sequence(path: Path, *, expected_base_frame: str) -> list[CartesianPose]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read pose-sequence JSON {path}: {exc}") from exc
    return parse_pose_sequence_payload(payload, expected_base_frame=expected_base_frame)


def validate_cartesian_speed_m_s(value: float) -> float:
    speed = float(value)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("max Cartesian speed must be positive and finite")
    if speed > MAX_CARTESIAN_SPEED_M_S:
        raise ValueError(
            f"max Cartesian speed is capped at {MAX_CARTESIAN_SPEED_M_S:.3f} m/s"
        )
    return speed


def validate_pose_sequence_motion(
    start: CartesianPose,
    waypoints: Sequence[CartesianPose],
) -> dict[str, Any]:
    """Validate current-to-first and inter-waypoint configured-EE motion."""

    if not 1 <= len(waypoints) <= MAX_WAYPOINTS:
        raise ValueError(f"pose sequence must contain 1..{MAX_WAYPOINTS} waypoints")

    segment_metrics: list[dict[str, float | int]] = []
    total_length_m = 0.0
    previous = start
    for index, waypoint in enumerate(waypoints):
        translation_m, rotation_rad = pose_delta(
            previous.position_m,
            previous.orientation_xyzw,
            waypoint.position_m,
            waypoint.orientation_xyzw,
        )
        if translation_m > MAX_SEGMENT_TRANSLATION_M + 1e-12:
            raise ValueError(
                f"segment {index} translation {translation_m:.6f} m exceeds "
                f"the {MAX_SEGMENT_TRANSLATION_M:.3f} m cap"
            )
        if rotation_rad > MAX_SEGMENT_ROTATION_RAD + 1e-12:
            raise ValueError(
                f"segment {index} rotation {math.degrees(rotation_rad):.3f} deg "
                f"exceeds the {math.degrees(MAX_SEGMENT_ROTATION_RAD):.1f} deg cap"
            )
        total_length_m += translation_m
        segment_metrics.append(
            {
                "segment_index": index,
                "translation_m": translation_m,
                "rotation_rad": rotation_rad,
                "rotation_deg": math.degrees(rotation_rad),
            }
        )
        previous = waypoint

    if total_length_m > MAX_TOTAL_CARTESIAN_LENGTH_M + 1e-12:
        raise ValueError(
            f"total Cartesian length {total_length_m:.6f} m exceeds "
            f"the {MAX_TOTAL_CARTESIAN_LENGTH_M:.3f} m cap"
        )
    return {
        "waypoint_count": len(waypoints),
        "segment_count": len(segment_metrics),
        "segment_metrics": segment_metrics,
        "total_cartesian_length_m": total_length_m,
        "limits": {
            "max_waypoints": MAX_WAYPOINTS,
            "max_segment_translation_m": MAX_SEGMENT_TRANSLATION_M,
            "max_segment_rotation_rad": MAX_SEGMENT_ROTATION_RAD,
            "max_segment_rotation_deg": math.degrees(MAX_SEGMENT_ROTATION_RAD),
            "max_total_cartesian_length_m": MAX_TOTAL_CARTESIAN_LENGTH_M,
        },
    }


def configured_ee_to_link8_waypoints(
    waypoints: Sequence[CartesianPose],
    *,
    link8_to_configured_ee_position_m: Sequence[float],
    link8_to_configured_ee_orientation_xyzw: Sequence[float],
) -> list[CartesianPose]:
    """Convert absolute configured-EE poses to absolute link8 poses."""

    offset_position = _finite_vector(
        link8_to_configured_ee_position_m,
        length=3,
        name="T_link8_configured_ee translation",
    )
    raw_offset_orientation = _finite_vector(
        link8_to_configured_ee_orientation_xyzw,
        length=4,
        name="T_link8_configured_ee orientation",
    )
    offset_orientation = normalize_quaternion_xyzw(raw_offset_orientation)
    converted: list[CartesianPose] = []
    for waypoint in waypoints:
        position, orientation = link8_pose_from_configured_ee(
            waypoint.position_m,
            waypoint.orientation_xyzw,
            offset_position,
            offset_orientation,
        )
        converted.append(
            CartesianPose(tuple(position), normalize_quaternion_xyzw(orientation))
        )
    return converted


__all__ = [
    "MAX_CARTESIAN_SPEED_M_S",
    "MAX_SEGMENT_ROTATION_RAD",
    "MAX_SEGMENT_TRANSLATION_M",
    "MAX_TOTAL_CARTESIAN_LENGTH_M",
    "MAX_WAYPOINTS",
    "POSE_SEQUENCE_SCHEMA",
    "CartesianPose",
    "configured_ee_to_link8_waypoints",
    "load_pose_sequence",
    "parse_pose_sequence_payload",
    "validate_cartesian_speed_m_s",
    "validate_pose_sequence_motion",
]
