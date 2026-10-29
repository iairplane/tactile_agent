"""Review-only Hand-E candidates for a circular cup rim.

A circular rim determines its centre and axis, but it does not determine a yaw
about that axis.  This module therefore ignores the transverse axes in the live
rim transform and makes every radial choice explicit.  It only composes pose
candidates for later MoveIt plan-only evaluation; it has no ROS or hardware
side effects.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .franka import validate_rigid_transform

CIRCULAR_RIM_CANDIDATE_SCHEMA = "demo2gripper.circular_rim_candidates.v1"
DEFAULT_AZIMUTH_OFFSETS_DEG = (0.0, 90.0, -90.0, 180.0)


def _unit_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def _project_radial(value: np.ndarray, axis: np.ndarray, *, name: str) -> np.ndarray:
    vector = _unit_vector(value, name=name)
    projected = vector - float(np.dot(vector, axis)) * axis
    norm = float(np.linalg.norm(projected))
    if norm <= 1e-6:
        raise ValueError(f"{name} is parallel to the live cup axis")
    return projected / norm


def _minimal_axis_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the unique shortest rotation from ``source`` to ``target``."""

    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine > 1.0 - 1e-10:
        return np.eye(3)
    if cosine < -1.0 + 1e-8:
        raise ValueError(
            "demo and live cup axes are antiparallel; radial transport is ambiguous"
        )
    cross = np.cross(source, target)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    sine_squared = float(np.dot(cross, cross))
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine_squared)


def _deterministic_reference(axis: np.ndarray) -> np.ndarray:
    basis = np.eye(3)
    seed = basis[int(np.argmin(np.abs(basis @ axis)))]
    return _project_radial(seed, axis, name="deterministic base-axis reference")


def _validated_azimuths(values: Sequence[float]) -> tuple[float, ...]:
    angles = tuple(float(value) for value in values)
    if not 1 <= len(angles) <= 12:
        raise ValueError("azimuth_offsets_deg must contain between 1 and 12 values")
    if not np.isfinite(angles).all():
        raise ValueError("azimuth_offsets_deg must be finite")
    normalized = np.mod(angles, 360.0)
    for index, angle in enumerate(normalized):
        if any(
            math.isclose(float(angle), float(previous), abs_tol=1e-8)
            for previous in normalized[:index]
        ):
            raise ValueError("azimuth_offsets_deg contains equivalent duplicate angles")
    return angles


def _rim_gauge(
    center: np.ndarray,
    axis: np.ndarray,
    reference_radial: np.ndarray,
    azimuth_deg: float,
) -> np.ndarray:
    reference_tangent = np.cross(axis, reference_radial)
    angle = math.radians(azimuth_deg)
    radial = math.cos(angle) * reference_radial + math.sin(angle) * reference_tangent
    radial = _project_radial(radial, axis, name="candidate radial")
    tangent = _unit_vector(np.cross(axis, radial), name="candidate tangent")
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((radial, tangent, axis))
    transform[:3, 3] = center
    return validate_rigid_transform(transform, name="T_base_rim_gauge")


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def generate_circular_rim_candidates(
    *,
    T_base_live_rim: np.ndarray,
    T_rim_human_pinch_tcp: np.ndarray,
    T_link8_tcp: np.ndarray,
    approach_distance_m: float,
    azimuth_offsets_deg: Sequence[float] = DEFAULT_AZIMUTH_OFFSETS_DEG,
    T_base_demo_rim: np.ndarray | None = None,
    reference_radial_base: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compose explicit radial candidates without inventing circular-rim yaw.

    ``T_base_live_rim`` contributes only its translation and local ``+Z`` cup
    axis.  Its local ``+X/+Y`` axes are intentionally ignored.  When a demo rim
    is supplied, its interaction radial is parallel-transported by the shortest
    rotation between demo and live cup axes.  That transported vector is only an
    agent preference and is never described as observed live object yaw.
    """

    live_rim = validate_rigid_transform(T_base_live_rim, name="T_base_live_rim")
    relation = validate_rigid_transform(
        T_rim_human_pinch_tcp, name="T_rim_human_pinch_tcp"
    )
    link8_tcp = validate_rigid_transform(T_link8_tcp, name="T_link8_tcp")
    if not np.isfinite(approach_distance_m) or approach_distance_m <= 0.0:
        raise ValueError("approach_distance_m must be positive and finite")
    angles = _validated_azimuths(azimuth_offsets_deg)
    if T_base_demo_rim is not None and reference_radial_base is not None:
        raise ValueError(
            "supply at most one of T_base_demo_rim and reference_radial_base"
        )

    relation_radial = relation[:2, 3]
    radial_distance = float(np.linalg.norm(relation_radial))
    if radial_distance <= 1e-6:
        raise ValueError("T_rim_human_pinch_tcp does not define a rim radial contact")
    if relation[0, 3] <= 0.0 or abs(float(relation[1, 3])) > max(
        1e-5, 0.01 * radial_distance
    ):
        raise ValueError(
            "rim relation must use local +X toward the demonstrated pinch"
        )

    center = live_rim[:3, 3].copy()
    axis = live_rim[:3, 2].copy()
    if reference_radial_base is not None:
        reference = _project_radial(
            reference_radial_base, axis, name="reference_radial_base"
        )
        reference_source = "explicit_base_radial_preference"
    elif T_base_demo_rim is not None:
        demo_rim = validate_rigid_transform(
            T_base_demo_rim, name="T_base_demo_rim"
        )
        alignment = _minimal_axis_alignment(demo_rim[:3, 2], axis)
        reference = _project_radial(
            alignment @ demo_rim[:3, 0],
            axis,
            name="parallel-transported demonstration radial",
        )
        reference_source = "parallel_transported_demonstration_interaction_radial"
    else:
        reference = _deterministic_reference(axis)
        reference_source = "deterministic_base_axis_gauge"

    inverse_link8_tcp = np.linalg.inv(link8_tcp)
    candidates: list[dict[str, Any]] = []
    for index, azimuth_deg in enumerate(angles):
        rim_gauge = _rim_gauge(center, axis, reference, azimuth_deg)
        contact_tcp = validate_rigid_transform(
            rim_gauge @ relation, name=f"candidate[{index}].T_base_contact_tcp"
        )
        pregrasp_tcp = contact_tcp.copy()
        pregrasp_tcp[:3, 3] -= contact_tcp[:3, 2] * approach_distance_m
        pregrasp_tcp = validate_rigid_transform(
            pregrasp_tcp, name=f"candidate[{index}].T_base_pregrasp_tcp"
        )
        contact_link8 = validate_rigid_transform(
            contact_tcp @ inverse_link8_tcp,
            name=f"candidate[{index}].T_base_contact_link8",
        )
        pregrasp_link8 = validate_rigid_transform(
            pregrasp_tcp @ inverse_link8_tcp,
            name=f"candidate[{index}].T_base_pregrasp_link8",
        )
        candidates.append(
            {
                "candidate_id": f"rim_azimuth_{index:02d}",
                "azimuth_offset_deg": azimuth_deg,
                "azimuth_normalized_deg": float(azimuth_deg % 360.0),
                "azimuth_authority": (
                    "agentic_plan_only_choice_about_unobservable_cup_axis"
                ),
                "copied_live_object_yaw": False,
                "radial_axis_base": rim_gauge[:3, 0].tolist(),
                "tangent_axis_base": rim_gauge[:3, 1].tolist(),
                "cup_axis_base": axis.tolist(),
                "closing_to_radial_unsigned_angle_deg": _angle_deg(
                    contact_tcp[:3, 0], rim_gauge[:3, 0]
                ),
                "T_base_rim_gauge": rim_gauge.tolist(),
                "T_base_contact_tcp": contact_tcp.tolist(),
                "T_base_pregrasp_tcp": pregrasp_tcp.tolist(),
                "T_base_contact_link8": contact_link8.tolist(),
                "T_base_pregrasp_link8": pregrasp_link8.tolist(),
                "moveit_plan_only": {
                    "tool_link": "fr3_link8",
                    "target_order": ["pregrasp", "contact"],
                    "T_base_link8_targets": {
                        "pregrasp": pregrasp_link8.tolist(),
                        "contact": contact_link8.tolist(),
                    },
                    "execute": False,
                },
            }
        )

    return {
        "schema": CIRCULAR_RIM_CANDIDATE_SCHEMA,
        "status": "review_only",
        "execution_authorized": False,
        "hardware_execution_enabled": False,
        "frame_convention": (
            "right-handed metres and column vectors; T_parent_child; rim +X is "
            "the selected contact radial, +Y tangent, +Z cup opening axis; "
            "canonical TCP +X closes and +Z approaches"
        ),
        "symmetry_handling": {
            "object_symmetry": "continuous_rotation_about_live_cup_axis",
            "live_rim_center_consumed": True,
            "live_rim_axis_consumed": True,
            "live_rim_transverse_axes_consumed": False,
            "live_object_yaw_observable": False,
            "reference_radial_source": reference_source,
            "reference_radial_base": reference.tolist(),
            "reference_is_agent_preference_not_object_yaw": True,
        },
        "inputs": {
            "live_rim_center_base_m": center.tolist(),
            "live_cup_axis_base": axis.tolist(),
            "T_rim_human_pinch_tcp": relation.tolist(),
            "T_link8_tcp": link8_tcp.tolist(),
            "approach_distance_m": float(approach_distance_m),
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "required_plan_only_evaluation": [
            "current-state IK for pregrasp and contact",
            "self and planning-scene collision checks",
            "pregrasp trajectory feasibility",
            "bounded contact approach feasibility and rim clearance",
            "rank reachable candidates with an explicit agent policy",
        ],
        "blocking_conditions": [
            "no candidate has passed MoveIt plan-only evaluation",
            "no candidate has been selected by the agent policy",
            "operator review and execution-specific gates remain outstanding",
        ],
    }


__all__ = [
    "CIRCULAR_RIM_CANDIDATE_SCHEMA",
    "DEFAULT_AZIMUTH_OFFSETS_DEG",
    "generate_circular_rim_candidates",
]
