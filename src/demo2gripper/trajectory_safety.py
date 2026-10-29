"""Pure fail-closed safety checks for bounded FR3 joint trajectories."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# Source: franka_description/robots/fr3/joint_limits.yaml in the deployed
# franka_ros2 workspace.  These are hardware position limits, not reduced
# experiment limits.
FR3_JOINT_POSITION_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "fr3_joint1": (-2.9007, 2.9007),
    "fr3_joint2": (-1.8361, 1.8361),
    "fr3_joint3": (-2.9007, 2.9007),
    "fr3_joint4": (-3.0770, -0.1169),
    "fr3_joint5": (-2.8763, 2.8763),
    "fr3_joint6": (0.4398, 4.6216),
    "fr3_joint7": (-3.0508, 3.0508),
}

DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD = 0.05
DEFAULT_MAX_JOINT_SWEEP_RAD = 2.0
DEFAULT_MAX_TOTAL_JOINT_SWEEP_RAD = 7.0


def _positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def validate_fr3_trajectory_audit_thresholds(
    *,
    min_joint_limit_margin_rad: float,
    max_joint_sweep_rad: float,
    max_total_joint_sweep_rad: float,
) -> dict[str, float]:
    """Validate thresholds without allowing the reviewed defaults to be weakened."""

    thresholds = {
        "min_joint_limit_margin_rad": _positive_finite(
            min_joint_limit_margin_rad, name="min_joint_limit_margin_rad"
        ),
        "max_joint_sweep_rad": _positive_finite(
            max_joint_sweep_rad, name="max_joint_sweep_rad"
        ),
        "max_total_joint_sweep_rad": _positive_finite(
            max_total_joint_sweep_rad, name="max_total_joint_sweep_rad"
        ),
    }
    if thresholds["min_joint_limit_margin_rad"] < DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD:
        raise ValueError(
            "min_joint_limit_margin_rad cannot weaken the reviewed "
            f"{DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD:.3f} rad minimum"
        )
    if thresholds["max_joint_sweep_rad"] > DEFAULT_MAX_JOINT_SWEEP_RAD:
        raise ValueError(
            "max_joint_sweep_rad cannot weaken the reviewed "
            f"{DEFAULT_MAX_JOINT_SWEEP_RAD:.3f} rad cap"
        )
    if thresholds["max_total_joint_sweep_rad"] > DEFAULT_MAX_TOTAL_JOINT_SWEEP_RAD:
        raise ValueError(
            "max_total_joint_sweep_rad cannot weaken the reviewed "
            f"{DEFAULT_MAX_TOTAL_JOINT_SWEEP_RAD:.3f} rad cap"
        )
    return thresholds


def audit_fr3_joint_trajectory(
    trajectory: Mapping[str, Any],
    *,
    min_joint_limit_margin_rad: float = DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD,
    max_joint_sweep_rad: float = DEFAULT_MAX_JOINT_SWEEP_RAD,
    max_total_joint_sweep_rad: float = DEFAULT_MAX_TOTAL_JOINT_SWEEP_RAD,
    enforce_joint_sweep_guards: bool = True,
) -> dict[str, Any]:
    """Audit every point of one complete FR3 arm trajectory.

    The input is the JSON-compatible dictionary emitted by the MoveIt tools.
    Missing joints, malformed points, non-finite positions and threshold
    violations all fail closed.  The returned evidence is serializable and can
    be included verbatim in a plan artifact.
    """

    if not isinstance(enforce_joint_sweep_guards, bool):
        raise ValueError("enforce_joint_sweep_guards must be a boolean")
    thresholds = validate_fr3_trajectory_audit_thresholds(
        min_joint_limit_margin_rad=min_joint_limit_margin_rad,
        max_joint_sweep_rad=max_joint_sweep_rad,
        max_total_joint_sweep_rad=max_total_joint_sweep_rad,
    )
    expected_names = list(FR3_JOINT_POSITION_LIMITS_RAD)
    joint_names = trajectory.get("joint_names")
    points = trajectory.get("points")
    violations: list[str] = []

    if not isinstance(joint_names, list) or any(
        not isinstance(name, str) or not name for name in joint_names
    ):
        violations.append("trajectory joint_names must be a non-empty string array")
        joint_names = []
    elif len(set(joint_names)) != len(joint_names):
        violations.append("trajectory joint_names contain duplicates")
    elif set(joint_names) != set(expected_names):
        missing = sorted(set(expected_names) - set(joint_names))
        extra = sorted(set(joint_names) - set(expected_names))
        violations.append(
            f"trajectory joint set differs from FR3 arm; missing={missing}, extra={extra}"
        )

    if not isinstance(points, list) or len(points) < 2:
        violations.append("trajectory must contain at least two points")
        points = []

    positions_by_joint: dict[str, list[float]] = {name: [] for name in joint_names}
    previous_time = -math.inf
    for point_index, point in enumerate(points):
        if not isinstance(point, Mapping):
            violations.append(f"trajectory point {point_index} is not an object")
            continue
        positions = point.get("positions")
        if not isinstance(positions, list) or len(positions) != len(joint_names):
            violations.append(
                f"trajectory point {point_index} positions do not match joint_names"
            )
            continue
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in positions
        ):
            violations.append(f"trajectory point {point_index} positions are not finite")
            continue
        time_from_start_s = point.get("time_from_start_s")
        if not isinstance(time_from_start_s, (int, float)) or not math.isfinite(
            time_from_start_s
        ):
            violations.append(f"trajectory point {point_index} time is not finite")
        elif time_from_start_s < 0.0 or time_from_start_s <= previous_time:
            violations.append(
                f"trajectory point {point_index} time is not strictly increasing"
            )
        else:
            previous_time = float(time_from_start_s)
        for name, value in zip(joint_names, positions, strict=True):
            positions_by_joint[name].append(float(value))

    joint_metrics: dict[str, dict[str, Any]] = {}
    total_sweep_rad = 0.0
    if not violations:
        for name in expected_names:
            lower, upper = FR3_JOINT_POSITION_LIMITS_RAD[name]
            positions = positions_by_joint[name]
            minimum = min(positions)
            maximum = max(positions)
            margin = min(minimum - lower, upper - maximum)
            sweep = maximum - minimum
            endpoint_displacement = abs(positions[-1] - positions[0])
            limit_ok = margin >= thresholds["min_joint_limit_margin_rad"]
            sweep_ok = (
                sweep <= thresholds["max_joint_sweep_rad"]
                if enforce_joint_sweep_guards
                else None
            )
            joint_metrics[name] = {
                "hardware_lower_rad": lower,
                "hardware_upper_rad": upper,
                "trajectory_min_rad": minimum,
                "trajectory_max_rad": maximum,
                "minimum_hardware_limit_margin_rad": margin,
                "sweep_rad": sweep,
                "endpoint_displacement_rad": endpoint_displacement,
                "limit_margin_passed": limit_ok,
                "sweep_passed": sweep_ok,
            }
            total_sweep_rad += sweep
            if not limit_ok:
                violations.append(
                    f"{name} hardware-limit margin {margin:.6f} rad is below "
                    f"{thresholds['min_joint_limit_margin_rad']:.6f} rad"
                )
            if sweep_ok is False:
                violations.append(
                    f"{name} sweep {sweep:.6f} rad exceeds "
                    f"{thresholds['max_joint_sweep_rad']:.6f} rad"
                )

        if (
            enforce_joint_sweep_guards
            and total_sweep_rad > thresholds["max_total_joint_sweep_rad"]
        ):
            violations.append(
                f"total joint sweep {total_sweep_rad:.6f} rad exceeds "
                f"{thresholds['max_total_joint_sweep_rad']:.6f} rad"
            )

    return {
        "schema": "demo2gripper.fr3_joint_trajectory_audit.v1",
        "accepted": not violations,
        "joint_limits_source": "franka_description/robots/fr3/joint_limits.yaml",
        "thresholds": thresholds,
        "joint_sweep_guards_enabled": enforce_joint_sweep_guards,
        "joint_metrics": joint_metrics,
        "total_joint_sweep_rad": total_sweep_rad if joint_metrics else None,
        "violations": violations,
    }


__all__ = [
    "DEFAULT_MAX_JOINT_SWEEP_RAD",
    "DEFAULT_MAX_TOTAL_JOINT_SWEEP_RAD",
    "DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD",
    "FR3_JOINT_POSITION_LIMITS_RAD",
    "audit_fr3_joint_trajectory",
    "validate_fr3_trajectory_audit_thresholds",
]
