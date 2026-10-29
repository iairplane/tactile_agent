"""Plan-only Hand-E side-grasp candidates for the demonstrated test tube.

This module is deliberately hardware independent.  It turns the reviewed
camera-frame relation into a family of canonical TCP poses, but it never opens
ROS, MoveIt, a camera, or a robot driver.  MoveIt results can be attached later
and ranked using their audited joint trajectories.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .franka import validate_rigid_transform

RELATION_SCHEMA = "demo2gripper.test_tube_current_image_relation.v1"
CANDIDATE_SET_SCHEMA = "demo2gripper.test_tube_pregrasp_candidates.v1"
MOVEIT_PLAN_SCHEMA = "demo2gripper.moveit_pose_goal.v2"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value: object, *, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _unit(value: object, *, name: str) -> np.ndarray:
    result = _finite_vector(value, size=3, name=name)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be non-zero")
    return result / norm


def _positive(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _fraction(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def rotation_matrix_to_quaternion_xyzw(rotation: object) -> np.ndarray:
    """Return a normalized XYZW quaternion with a non-negative scalar part."""

    matrix = np.asarray(rotation, dtype=float)
    transform = np.eye(4)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    transform[:3, :3] = matrix
    validate_rigid_transform(transform, name="rotation")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def verify_relation_provenance(
    relation: Mapping[str, Any], *, repository_root: str | Path
) -> list[dict[str, str]]:
    """Verify every path/hash pair declared by ``current_relation.json``."""

    provenance = relation.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("current relation must declare non-empty provenance")
    root = Path(repository_root).resolve()
    verified: list[dict[str, str]] = []
    for name, artifact in provenance.items():
        if not isinstance(name, str) or not isinstance(artifact, Mapping):
            raise ValueError("current relation provenance entries must be named objects")
        raw_path = artifact.get("path")
        expected = artifact.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"provenance.{name}.path is missing")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"provenance.{name}.sha256 is invalid")
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"provenance artifact does not exist: {path}")
        actual = sha256_file(path)
        if actual != expected.lower():
            raise ValueError(
                f"provenance hash mismatch for {name}: expected {expected}, got {actual}"
            )
        verified.append({"name": name, "path": str(path), "sha256": actual})
    return verified


def validate_current_relation(relation: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the reviewed source-tube relation and reject inconsistent inputs."""

    if relation.get("schema") != RELATION_SCHEMA:
        raise ValueError(f"current relation schema must be {RELATION_SCHEMA!r}")
    if relation.get("execution_authorized") is not False:
        raise ValueError("current relation must remain explicitly plan-only")
    if relation.get("hardware_commands_dispatched") is not False:
        raise ValueError("current relation claims that hardware commands were dispatched")

    source = relation.get("current_source_tube")
    grasp = relation.get("transferred_side_grasp")
    bottom_line = relation.get("bottom_line")
    if not isinstance(source, Mapping) or not isinstance(grasp, Mapping):
        raise ValueError("current relation lacks source-tube or transferred-grasp data")
    if not isinstance(bottom_line, Mapping):
        raise ValueError("current relation lacks its observability decision")
    if bottom_line.get("source_axis_and_grasp_observable") is not True:
        raise ValueError("source axis and transferred grasp were not marked observable")

    endpoints = np.asarray(source.get("axis_closed_or_lower_visible_to_open_uv_px"), dtype=float)
    if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all():
        raise ValueError("source image axis must contain two finite image points")
    image_span = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    if image_span <= 20.0:
        raise ValueError("source image axis is too short to transfer an axial grasp")

    axial_fraction = _fraction(
        grasp.get("demo_axial_fraction_closed_to_open"),
        name="demo axial grasp fraction",
    )
    fraction_uncertainty = _positive(
        grasp.get("demo_axial_fraction_uncertainty"),
        name="demo axial fraction uncertainty",
    )
    if fraction_uncertainty >= 0.5:
        raise ValueError("demo axial fraction uncertainty must be below 0.5")
    transferred_uv = _finite_vector(
        grasp.get("current_grasp_centre_uv_px"), size=2, name="current grasp centre"
    )
    expected_uv = endpoints[0] + axial_fraction * (endpoints[1] - endpoints[0])
    transfer_residual_px = float(np.linalg.norm(transferred_uv - expected_uv))
    if transfer_residual_px > 1.0:
        raise ValueError(
            "transferred grasp centre is inconsistent with the demonstration axial fraction"
        )

    depth = grasp.get("depth_at_grasp")
    depth_observation = source.get("depth_observation")
    if not isinstance(depth, Mapping) or not isinstance(depth_observation, Mapping):
        raise ValueError("current relation lacks metric grasp/axis observations")
    grasp_camera = _finite_vector(
        depth.get("xyz_camera_m"), size=3, name="camera-frame grasp point"
    )
    axis_camera = _unit(
        depth_observation.get("axis_camera_unit_closed_to_open"),
        name="camera-frame tube axis",
    )
    residual_m = _positive(
        depth_observation.get("orthogonal_line_fit_rms_m"),
        name="tube-axis fit residual",
    )
    if residual_m > 0.005:
        raise ValueError("camera-frame tube-axis fit residual exceeds 5 mm")

    return {
        "axial_fraction_closed_to_open": axial_fraction,
        "axial_fraction_uncertainty": fraction_uncertainty,
        "transferred_grasp_uv_px": transferred_uv,
        "expected_grasp_uv_px": expected_uv,
        "transfer_residual_px": transfer_residual_px,
        "grasp_camera_m": grasp_camera,
        "tube_axis_camera_closed_to_open": axis_camera,
        "tube_axis_fit_rms_m": residual_m,
    }


def _reference_transverse(axis: np.ndarray) -> np.ndarray:
    basis = np.eye(3)
    seed = basis[int(np.argmin(np.abs(basis @ axis)))]
    transverse = seed - float(seed @ axis) * axis
    return transverse / np.linalg.norm(transverse)


def _roll_angles(step_deg: float) -> list[float]:
    step = _positive(step_deg, name="roll_step_deg")
    count_float = 360.0 / step
    count = int(round(count_float))
    if count < 4 or count > 72 or not math.isclose(count * step, 360.0, abs_tol=1e-7):
        raise ValueError("roll_step_deg must divide 360 and produce 4..72 samples")
    return [index * step for index in range(count)]


def generate_pregrasp_candidates(
    relation: Mapping[str, Any],
    T_base_camera: object,
    *,
    approach_distance_m: float = 0.060,
    roll_step_deg: float = 30.0,
    include_tube_axis_flip: bool = True,
    workspace_min_m: Sequence[float] = (0.20, -0.50, 0.04),
    workspace_max_m: Sequence[float] = (0.90, 0.60, 0.80),
) -> dict[str, Any]:
    """Create a deterministic family of canonical Hand-E TCP pregrasps.

    Local ``+X`` is the Hand-E jaw-closing axis, local ``+Y`` is aligned with
    either direction of the cylindrical tube axis, and local ``+Z`` points from
    pregrasp toward the tube.  The latter two choices leave one free roll around
    the tube, which is sampled for downstream MoveIt ranking.
    """

    extracted = validate_current_relation(relation)
    base_camera = validate_rigid_transform(
        np.asarray(T_base_camera, dtype=float), name="T_base_camera"
    )
    approach_distance = _positive(approach_distance_m, name="approach_distance_m")
    minimum = _finite_vector(workspace_min_m, size=3, name="workspace_min_m")
    maximum = _finite_vector(workspace_max_m, size=3, name="workspace_max_m")
    if np.any(maximum <= minimum):
        raise ValueError("workspace_max_m must exceed workspace_min_m on every axis")

    grasp_camera = extracted["grasp_camera_m"]
    grasp_base = base_camera[:3, :3] @ grasp_camera + base_camera[:3, 3]
    axis_base = base_camera[:3, :3] @ extracted["tube_axis_camera_closed_to_open"]
    axis_base = axis_base / np.linalg.norm(axis_base)
    roll_angles = _roll_angles(roll_step_deg)
    signs = (1, -1) if include_tube_axis_flip else (1,)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for sign in signs:
        local_y = sign * axis_base
        closing_zero = _reference_transverse(local_y)
        approach_zero = np.cross(closing_zero, local_y)
        approach_zero /= np.linalg.norm(approach_zero)
        for roll_deg in roll_angles:
            roll_rad = math.radians(roll_deg)
            closing = math.cos(roll_rad) * closing_zero - math.sin(roll_rad) * approach_zero
            closing /= np.linalg.norm(closing)
            approach = np.cross(closing, local_y)
            approach /= np.linalg.norm(approach)
            rotation = np.column_stack((closing, local_y, approach))
            transform_grasp = np.eye(4)
            transform_grasp[:3, :3] = rotation
            transform_grasp[:3, 3] = grasp_base
            validate_rigid_transform(transform_grasp, name="T_base_tcp_grasp")

            transform_pregrasp = transform_grasp.copy()
            transform_pregrasp[:3, 3] -= approach_distance * approach
            candidate_id = f"axis_{'p' if sign > 0 else 'n'}_roll_{int(round(roll_deg)):03d}"
            workspace_ok = bool(
                np.all(transform_pregrasp[:3, 3] >= minimum)
                and np.all(transform_pregrasp[:3, 3] <= maximum)
                and np.all(grasp_base >= minimum)
                and np.all(grasp_base <= maximum)
            )
            record = {
                "id": candidate_id,
                "tube_axis_sign": sign,
                "roll_deg": roll_deg,
                "T_base_tcp_grasp": transform_grasp.tolist(),
                "T_base_tcp_pregrasp": transform_pregrasp.tolist(),
                "pregrasp_position_base_m": transform_pregrasp[:3, 3].tolist(),
                "pregrasp_orientation_xyzw": rotation_matrix_to_quaternion_xyzw(rotation).tolist(),
                "closing_axis_base": closing.tolist(),
                "tube_axis_base": local_y.tolist(),
                "approach_axis_base": approach.tolist(),
                "constraints": {
                    "abs_closing_dot_tube": abs(float(closing @ axis_base)),
                    "abs_approach_dot_tube": abs(float(approach @ axis_base)),
                    "rotation_determinant": float(np.linalg.det(rotation)),
                    "pregrasp_retreat_m": float(
                        np.linalg.norm(transform_pregrasp[:3, 3] - grasp_base)
                    ),
                    "workspace_passed": workspace_ok,
                },
                "plan_only": True,
                "may_execute": False,
            }
            if workspace_ok:
                candidates.append(record)
            else:
                rejected.append(
                    {
                        "id": candidate_id,
                        "reason": "grasp or pregrasp lies outside configured workspace",
                        "pregrasp_position_base_m": transform_pregrasp[:3, 3].tolist(),
                    }
                )

    if not candidates:
        raise ValueError("all roll candidates were rejected by the workspace bounds")
    return {
        "schema": CANDIDATE_SET_SCHEMA,
        "status": "geometry_candidates_plan_only",
        "execution_authorized": False,
        "hardware_commands_dispatched": False,
        "transform_convention": "T_parent_child; metres; right-handed",
        "canonical_tcp_convention": ("+X jaw closing; +Y tube axis; +Z from pregrasp toward tube"),
        "demonstration_transfer": {
            "axial_fraction_closed_to_open": extracted["axial_fraction_closed_to_open"],
            "axial_fraction_uncertainty": extracted["axial_fraction_uncertainty"],
            "current_grasp_uv_px": extracted["transferred_grasp_uv_px"].tolist(),
            "fraction_transfer_residual_px": extracted["transfer_residual_px"],
            "metric_grasp_source": "current_relation.depth_at_grasp.xyz_camera_m",
        },
        "metric_geometry": {
            "T_base_camera": base_camera.tolist(),
            "grasp_position_camera_m": grasp_camera.tolist(),
            "grasp_position_base_m": grasp_base.tolist(),
            "tube_axis_camera_closed_to_open": extracted[
                "tube_axis_camera_closed_to_open"
            ].tolist(),
            "tube_axis_base_closed_to_open": axis_base.tolist(),
            "tube_axis_fit_rms_m": extracted["tube_axis_fit_rms_m"],
            "approach_distance_m": approach_distance,
        },
        "sampling": {
            "roll_step_deg": float(roll_step_deg),
            "tube_axis_signs": list(signs),
            "generated_count": len(signs) * len(roll_angles),
            "workspace_accepted_count": len(candidates),
            "workspace_rejected_count": len(rejected),
        },
        "workspace_bounds_base_m": {"min": minimum.tolist(), "max": maximum.tolist()},
        "candidates": candidates,
        "rejected_geometry_candidates": rejected,
        "selected_plan_only_candidate_id": None,
        "execution_blockers": [
            "this artifact is plan-only and never authorizes hardware execution",
            "MoveIt models the arm but not the installed Hand-E, cup, tube, or rack",
            (
                "transparent-tube metric depth and the supplied camera extrinsic "
                "require independent validation"
            ),
            "destination socket metric pose remains unresolved",
        ],
    }


def build_moveit_plan_only_command(
    candidate: Mapping[str, Any],
    *,
    planner_script: str | Path,
    ros_wrapper: str | Path,
    output_path: str | Path,
    T_link8_tcp: object,
    planning_time_s: float = 4.0,
    planning_attempts: int = 3,
) -> list[str]:
    """Build the exact argv for the existing planner without an execute flag."""

    if candidate.get("plan_only") is not True or candidate.get("may_execute") is not False:
        raise ValueError("candidate is not explicitly plan-only")
    position = _finite_vector(
        candidate.get("pregrasp_position_base_m"), size=3, name="pregrasp position"
    )
    quaternion = _finite_vector(
        candidate.get("pregrasp_orientation_xyzw"), size=4, name="pregrasp quaternion"
    )
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm <= 1e-9:
        raise ValueError("pregrasp quaternion must be non-zero")
    quaternion /= quaternion_norm
    link8_tcp = validate_rigid_transform(np.asarray(T_link8_tcp, dtype=float), name="T_link8_tcp")
    link8_tcp_quaternion = rotation_matrix_to_quaternion_xyzw(link8_tcp[:3, :3])
    planning_time = _positive(planning_time_s, name="planning_time_s")
    if isinstance(planning_attempts, bool) or not isinstance(planning_attempts, int):
        raise ValueError("planning_attempts must be an integer")
    if not 1 <= planning_attempts <= 10:
        raise ValueError("planning_attempts must be in [1, 10]")

    command = [
        str(Path(ros_wrapper)),
        str(Path(planner_script)),
        "--target-position-m",
        *(f"{value:.12g}" for value in position),
        "--target-orientation-xyzw",
        *(f"{value:.12g}" for value in quaternion),
        "--target-frame",
        "configured-ee",
        "--link8-to-configured-ee-position-m",
        *(f"{value:.12g}" for value in link8_tcp[:3, 3]),
        "--link8-to-configured-ee-orientation-xyzw",
        *(f"{value:.12g}" for value in link8_tcp_quaternion),
        "--planning-time-s",
        f"{planning_time:.12g}",
        "--planning-attempts",
        str(planning_attempts),
        "--max-translation-m",
        "0.6",
        "--max-rotation-rad",
        repr(math.pi),
        "--output",
        str(Path(output_path)),
    ]
    if "--execute" in command:
        raise AssertionError("plan-only command unexpectedly contains --execute")
    return command


def assess_moveit_plan(candidate: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one MoveIt result against its candidate and expose rank metrics."""

    reasons: list[str] = []
    if plan.get("schema") != MOVEIT_PLAN_SCHEMA:
        reasons.append("unexpected MoveIt plan schema")
    if plan.get("mode") != "plan_only":
        reasons.append("MoveIt result is not plan_only")
    if plan.get("input_target_frame") != "configured-ee":
        reasons.append("MoveIt result target frame is not configured-ee")
    if plan.get("arm_only_collision_model") is not True:
        reasons.append("MoveIt result does not identify the arm-only collision model")
    if plan.get("execution_requested") is not False:
        reasons.append("MoveIt result does not prove execution_requested=false")
    if plan.get("execution_dispatched") is not False:
        reasons.append("MoveIt result does not prove execution_dispatched=false")

    expected_position = _finite_vector(
        candidate.get("pregrasp_position_base_m"), size=3, name="candidate position"
    )
    expected_quaternion = _finite_vector(
        candidate.get("pregrasp_orientation_xyzw"), size=4, name="candidate quaternion"
    )
    try:
        actual_position = _finite_vector(
            plan.get("input_target_position_base_m"), size=3, name="plan target position"
        )
        actual_quaternion = _finite_vector(
            plan.get("input_target_orientation_xyzw"),
            size=4,
            name="plan target quaternion",
        )
        position_error = float(np.linalg.norm(actual_position - expected_position))
        actual_norm = float(np.linalg.norm(actual_quaternion))
        expected_norm = float(np.linalg.norm(expected_quaternion))
        if actual_norm <= 1e-9 or expected_norm <= 1e-9:
            raise ValueError("plan and candidate quaternions must be non-zero")
        q1 = actual_quaternion / actual_norm
        q2 = expected_quaternion / expected_norm
        orientation_error = 2.0 * math.acos(min(1.0, max(0.0, abs(float(q1 @ q2)))))
        if position_error > 1e-6:
            reasons.append("MoveIt target position does not match the candidate")
        if orientation_error > 1e-6:
            reasons.append("MoveIt target orientation does not match the candidate")
    except ValueError as exc:
        position_error = None
        orientation_error = None
        reasons.append(str(exc))

    audit = plan.get("trajectory_audit")
    if not isinstance(audit, Mapping):
        reasons.append("MoveIt result lacks a trajectory audit")
        audit = {}
    metrics = audit.get("joint_metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}
    sweeps = [
        float(value["sweep_rad"])
        for value in metrics.values()
        if isinstance(value, Mapping)
        and isinstance(value.get("sweep_rad"), (int, float))
        and math.isfinite(float(value["sweep_rad"]))
    ]
    margins = [
        float(value["minimum_hardware_limit_margin_rad"])
        for value in metrics.values()
        if isinstance(value, Mapping)
        and isinstance(value.get("minimum_hardware_limit_margin_rad"), (int, float))
        and math.isfinite(float(value["minimum_hardware_limit_margin_rad"]))
    ]
    total_sweep = audit.get("total_joint_sweep_rad")
    total_sweep_value = (
        float(total_sweep)
        if isinstance(total_sweep, (int, float)) and math.isfinite(float(total_sweep))
        else math.inf
    )
    moveit_found = plan.get("moveit_planning_succeeded") is True
    audit_accepted = audit.get("accepted") is True
    accepted = (
        not reasons and moveit_found and audit_accepted and plan.get("planning_succeeded") is True
    )
    return {
        "candidate_id": candidate.get("id"),
        "result_valid": not reasons,
        "accepted": accepted,
        "moveit_ik_collision_plan_found": moveit_found,
        "trajectory_audit_accepted": audit_accepted,
        "target_position_error_m": position_error,
        "target_orientation_error_rad": orientation_error,
        "total_joint_sweep_rad": None if math.isinf(total_sweep_value) else total_sweep_value,
        "max_single_joint_sweep_rad": max(sweeps) if sweeps else None,
        "minimum_joint_limit_margin_rad": min(margins) if margins else None,
        "planning_time_s": plan.get("planning_time_s"),
        "rejection_reasons": reasons
        + ([] if moveit_found else ["MoveIt found no IK/collision-valid plan"])
        + ([] if audit_accepted else ["trajectory audit rejected the plan"]),
    }


def rank_plan_assessments(assessments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank accepted plans by bounded motion, then retain failures for evidence."""

    def finite_or_inf(value: object) -> float:
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
        return math.inf

    def descending_finite_or_inf(value: object) -> float:
        if isinstance(value, (int, float)) and math.isfinite(value):
            return -float(value)
        return math.inf

    def key(item: Mapping[str, Any]) -> tuple[float, ...]:
        return (
            0.0 if item.get("accepted") is True else 1.0,
            0.0 if item.get("moveit_ik_collision_plan_found") is True else 1.0,
            finite_or_inf(item.get("total_joint_sweep_rad")),
            finite_or_inf(item.get("max_single_joint_sweep_rad")),
            descending_finite_or_inf(item.get("minimum_joint_limit_margin_rad")),
            finite_or_inf(item.get("planning_time_s")),
        )

    return [dict(item) for item in sorted(assessments, key=key)]


__all__ = [
    "CANDIDATE_SET_SCHEMA",
    "MOVEIT_PLAN_SCHEMA",
    "RELATION_SCHEMA",
    "assess_moveit_plan",
    "build_moveit_plan_only_command",
    "generate_pregrasp_candidates",
    "rank_plan_assessments",
    "rotation_matrix_to_quaternion_xyzw",
    "sha256_file",
    "validate_current_relation",
    "verify_relation_provenance",
]
