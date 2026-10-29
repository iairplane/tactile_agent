"""Ground an object-relative pinch into live targets for online agent planning.

The one-shot demonstration supplies the side-pinch relation: a base-aligned
object-to-pinch translation and a parallel-jaw closing axis.  The agent supplies
approach and lift goals for the current scene.  This module combines those quantities
with a live object translation anchor.  It does not perform IK, collision checking,
command-frame conversion, or robot execution.

The canonical TCP convention matches the rest of the repository: local ``+X`` is the
jaw closing axis and local ``+Z`` points toward the object.  The default agent policy
uses base ``-Z`` for approach and base ``+Z`` for a 40 mm lift.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .franka import validate_rigid_transform
from .grounding import ground_position_offsets_base
from .task_skill import (
    GraspRelationMode,
    JsonSerializable,
    ObjectGraspRelation,
    ground_object_grasp,
)


class ApproachSource(StrEnum):
    """Evidence used for the candidate's canonical ``+Z`` approach axis."""

    AGENT_SELECTED = "agent_selected"
    SELECTED_OBJECT_RELATIVE_SE3 = "selected_object_relative_se3"
    DEMONSTRATED_STABLE = "demonstrated_stable"
    BASE_VERTICAL_FALLBACK = "base_vertical_fallback"


def _vector3(value: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _unit_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    vector = _vector3(value, name=name)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def _canonical_rotation(
    closing_axis_base: np.ndarray,
    approach_axis_base: np.ndarray,
    *,
    preserve_closing_axis: bool = False,
) -> np.ndarray:
    """Build a right-handed canonical TCP rotation from two measured axes."""

    approach = _unit_vector(approach_axis_base, name="approach_axis_base")
    closing = _unit_vector(closing_axis_base, name="closing_axis_base")
    if preserve_closing_axis:
        approach = approach - float(np.dot(approach, closing)) * closing
        projected_norm = float(np.linalg.norm(approach))
    else:
        closing = closing - float(np.dot(closing, approach)) * approach
        projected_norm = float(np.linalg.norm(closing))
    if projected_norm <= 1e-6:
        raise ValueError("closing_axis_base is parallel to the selected approach axis")
    if preserve_closing_axis:
        approach /= projected_norm
    else:
        closing /= projected_norm
    lateral = np.cross(approach, closing)
    lateral /= np.linalg.norm(lateral)
    rotation = np.column_stack((closing, lateral, approach))
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise ValueError("closing and approach axes do not define a right-handed pose")
    return rotation


@dataclass(frozen=True)
class GroundedGraspCandidate(JsonSerializable):
    """Review-only canonical TCP poses grounded at the current object anchor."""

    T_base_pregrasp_tcp: np.ndarray
    T_base_grasp_tcp: np.ndarray
    approach_source: ApproachSource
    approach_distance_m: float
    jaw_opening_m: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "approach_source", ApproachSource(self.approach_source))
        pregrasp = validate_rigid_transform(
            self.T_base_pregrasp_tcp, name="T_base_pregrasp_tcp"
        )
        grasp = validate_rigid_transform(self.T_base_grasp_tcp, name="T_base_grasp_tcp")
        object.__setattr__(self, "T_base_pregrasp_tcp", pregrasp)
        object.__setattr__(self, "T_base_grasp_tcp", grasp)
        if not np.isfinite(self.approach_distance_m) or self.approach_distance_m <= 0.0:
            raise ValueError("approach_distance_m must be positive and finite")
        measured_distance = float(
            np.linalg.norm(pregrasp[:3, 3] - grasp[:3, 3])
        )
        if not np.isclose(measured_distance, self.approach_distance_m, atol=1e-8):
            raise ValueError("pregrasp-to-grasp distance does not match approach_distance_m")
        if self.jaw_opening_m is not None and (
            not np.isfinite(self.jaw_opening_m) or self.jaw_opening_m <= 0.0
        ):
            raise ValueError("jaw_opening_m must be positive and finite or None")

    @property
    def closing_axis_base(self) -> np.ndarray:
        return self.T_base_grasp_tcp[:3, 0].copy()

    @property
    def approach_axis_base(self) -> np.ndarray:
        return self.T_base_grasp_tcp[:3, 2].copy()


@dataclass(frozen=True)
class GroundedPickPhases(JsonSerializable):
    """Position/pose targets for separately planned approach, grasp, and lift phases."""

    candidate: GroundedGraspCandidate
    approach_poses_base: np.ndarray
    lift_positions_base_m: np.ndarray

    def __post_init__(self) -> None:
        approach = np.asarray(self.approach_poses_base, dtype=float)
        if approach.shape != (2, 4, 4):
            raise ValueError("approach_poses_base must have shape (2, 4, 4)")
        for index, pose in enumerate(approach):
            validate_rigid_transform(pose, name=f"approach_poses_base[{index}]")
        if not np.allclose(approach[0], self.candidate.T_base_pregrasp_tcp):
            raise ValueError("approach must start at the candidate pregrasp pose")
        if not np.allclose(approach[-1], self.candidate.T_base_grasp_tcp):
            raise ValueError("approach must end at the candidate grasp pose")
        lift = np.asarray(self.lift_positions_base_m, dtype=float)
        if lift.ndim != 2 or lift.shape[1:] != (3,) or len(lift) < 2:
            raise ValueError("lift_positions_base_m must have shape (N, 3), N >= 2")
        if not np.isfinite(lift).all():
            raise ValueError("lift_positions_base_m contains non-finite values")
        if not np.allclose(lift[0], self.candidate.T_base_grasp_tcp[:3, 3], atol=1e-8):
            raise ValueError("lift must start at the candidate grasp position")
        object.__setattr__(self, "approach_poses_base", approach)
        object.__setattr__(self, "lift_positions_base_m", lift)


def make_object_relative_se3_grasp_candidate(
    *,
    T_base_object: np.ndarray,
    grasp_relation: ObjectGraspRelation,
) -> GroundedGraspCandidate:
    """Ground a selected SE(3) grasp without changing its approach orientation.

    The final pose is exactly ``T_base_object @ T_object_gripper_tcp``.  The
    pregrasp retreats by the selected approach distance along the canonical TCP
    local ``-Z`` direction; no agent-provided approach axis is accepted here.
    """

    if grasp_relation.mode is not GraspRelationMode.SE3:
        raise ValueError("object-relative SE(3) grounding requires an se3 relation")
    if grasp_relation.approach_distance_m is None:
        raise ValueError("se3 grasp relation has no approach_distance_m")
    grounded = ground_object_grasp(T_base_object, grasp_relation)
    grasp_pose = grounded.require_se3()
    pregrasp_pose = grasp_pose.copy()
    pregrasp_pose[:3, 3] -= (
        grasp_pose[:3, 2] * grasp_relation.approach_distance_m
    )
    return GroundedGraspCandidate(
        T_base_pregrasp_tcp=pregrasp_pose,
        T_base_grasp_tcp=grasp_pose,
        approach_source=ApproachSource.SELECTED_OBJECT_RELATIVE_SE3,
        approach_distance_m=grasp_relation.approach_distance_m,
        jaw_opening_m=grasp_relation.jaw_opening_m,
    )


def make_base_aligned_grasp_candidate(
    *,
    live_object_anchor_base_m: np.ndarray,
    grasp_relation: ObjectGraspRelation,
    closing_axis_base: np.ndarray | None = None,
    closing_axis_stable: bool | None = None,
    demonstrated_approach_axis_base: np.ndarray | None = None,
    demonstrated_approach_stable: bool = False,
    approach_distance_m: float,
    min_downward_approach_alignment: float | None = None,
    agent_approach_axis_base: np.ndarray | None = None,
) -> GroundedGraspCandidate:
    """Create live pregrasp/grasp poses from a demonstrated side-pinch relation.

    ``base_aligned_planar`` relations carry their own demonstrated jaw axis, so the
    agent only chooses an approach axis.  The translation-only arguments remain for
    legacy artifacts: they require a separately supplied stable closing axis and may
    still select a demonstrated approach or the old vertical fallback.
    """

    if not isinstance(demonstrated_approach_stable, bool):
        raise ValueError("demonstrated_approach_stable must be a bool")
    if not np.isfinite(approach_distance_m) or approach_distance_m <= 0.0:
        raise ValueError("approach_distance_m must be positive and finite")
    if min_downward_approach_alignment is not None and not (
        np.isfinite(min_downward_approach_alignment)
        and 0.0 < min_downward_approach_alignment <= 1.0
    ):
        raise ValueError("min_downward_approach_alignment must lie in (0, 1]")
    if grasp_relation.mode not in {
        GraspRelationMode.TRANSLATION_ONLY,
        GraspRelationMode.BASE_ALIGNED_PLANAR,
    }:
        raise ValueError(
            "base-aligned candidate requires a translation_only or "
            "base_aligned_planar grasp relation"
        )

    live_anchor = _vector3(
        live_object_anchor_base_m, name="live_object_anchor_base_m"
    )
    T_base_object_translation = np.eye(4, dtype=float)
    T_base_object_translation[:3, 3] = live_anchor
    grounded = ground_object_grasp(T_base_object_translation, grasp_relation)

    if grasp_relation.mode is GraspRelationMode.BASE_ALIGNED_PLANAR:
        relation_closing_axis = grounded.require_closing_axis()
        if closing_axis_base is not None:
            supplied_closing_axis = _unit_vector(
                closing_axis_base, name="closing_axis_base"
            )
            if not np.isclose(
                abs(float(np.dot(relation_closing_axis, supplied_closing_axis))),
                1.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "closing_axis_base conflicts with the base_aligned_planar relation"
                )
        selected_closing_axis = relation_closing_axis
    else:
        if not isinstance(closing_axis_stable, bool):
            raise ValueError(
                "legacy translation_only grounding requires closing_axis_stable"
            )
        if not closing_axis_stable:
            raise ValueError("a stable closing axis is required")
        if closing_axis_base is None:
            raise ValueError(
                "legacy translation_only grounding requires closing_axis_base"
            )
        selected_closing_axis = _unit_vector(
            closing_axis_base, name="closing_axis_base"
        )

    if (
        agent_approach_axis_base is None
        and grasp_relation.mode is GraspRelationMode.BASE_ALIGNED_PLANAR
    ):
        agent_approach_axis_base = np.array([0.0, 0.0, -1.0])

    if agent_approach_axis_base is not None:
        approach_axis = _unit_vector(
            agent_approach_axis_base, name="agent_approach_axis_base"
        )
        approach_source = ApproachSource.AGENT_SELECTED
    else:
        use_demonstrated_approach = demonstrated_approach_stable
        demonstrated_approach: np.ndarray | None = None
        if demonstrated_approach_stable:
            if demonstrated_approach_axis_base is None:
                raise ValueError(
                    "a stable demonstrated approach requires "
                    "demonstrated_approach_axis_base"
                )
            demonstrated_approach = _unit_vector(
                demonstrated_approach_axis_base,
                name="demonstrated_approach_axis_base",
            )
            if min_downward_approach_alignment is not None:
                downward_alignment = float(
                    np.dot(demonstrated_approach, np.array([0.0, 0.0, -1.0]))
                )
                use_demonstrated_approach = (
                    downward_alignment >= min_downward_approach_alignment
                )

        if use_demonstrated_approach:
            assert demonstrated_approach is not None
            approach_axis = demonstrated_approach
            approach_source = ApproachSource.DEMONSTRATED_STABLE
        else:
            approach_axis = np.array([0.0, 0.0, -1.0])
            approach_source = ApproachSource.BASE_VERTICAL_FALLBACK

    rotation = _canonical_rotation(
        selected_closing_axis,
        approach_axis,
        preserve_closing_axis=(
            grasp_relation.mode is GraspRelationMode.BASE_ALIGNED_PLANAR
        ),
    )
    grasp_pose = np.eye(4, dtype=float)
    grasp_pose[:3, :3] = rotation
    grasp_pose[:3, 3] = grounded.position_base_m
    pregrasp_pose = grasp_pose.copy()
    pregrasp_pose[:3, 3] -= rotation[:, 2] * approach_distance_m
    return GroundedGraspCandidate(
        T_base_pregrasp_tcp=pregrasp_pose,
        T_base_grasp_tcp=grasp_pose,
        approach_source=approach_source,
        approach_distance_m=float(approach_distance_m),
        jaw_opening_m=grasp_relation.jaw_opening_m,
    )


def make_agent_planned_pick_phases(
    candidate: GroundedGraspCandidate,
    *,
    agent_lift_delta_base_m: np.ndarray | None = None,
) -> GroundedPickPhases:
    """Build phase endpoints from the agent's current-scene approach and lift goals."""

    lift_delta = (
        np.array([0.0, 0.0, 0.04], dtype=float)
        if agent_lift_delta_base_m is None
        else _vector3(agent_lift_delta_base_m, name="agent_lift_delta_base_m")
    )
    if float(lift_delta[2]) <= 0.0:
        raise ValueError("agent_lift_delta_base_m must have a positive base-Z component")
    if float(np.linalg.norm(lift_delta)) <= 1e-9:
        raise ValueError("agent_lift_delta_base_m must be non-zero")
    grasp_position = candidate.T_base_grasp_tcp[:3, 3]
    return GroundedPickPhases(
        candidate=candidate,
        approach_poses_base=np.stack(
            (candidate.T_base_pregrasp_tcp, candidate.T_base_grasp_tcp)
        ),
        lift_positions_base_m=np.stack(
            (grasp_position, grasp_position + lift_delta)
        ),
    )


def make_grounded_pick_phases(
    candidate: GroundedGraspCandidate,
    *,
    demonstrated_lift_offsets_base_m: np.ndarray,
) -> GroundedPickPhases:
    """Anchor the demonstrated lift at the live grasp and keep phases separate."""

    offsets = np.asarray(demonstrated_lift_offsets_base_m, dtype=float)
    if offsets.ndim != 2 or offsets.shape[1:] != (3,) or len(offsets) < 2:
        raise ValueError(
            "demonstrated_lift_offsets_base_m must have shape (N, 3), N >= 2"
        )
    if not np.isfinite(offsets).all():
        raise ValueError("demonstrated_lift_offsets_base_m contains non-finite values")
    if not np.allclose(offsets[0], np.zeros(3), atol=1e-8):
        raise ValueError("demonstrated lift offsets must start at the grasp anchor")
    grounded_lift = ground_position_offsets_base(
        offsets,
        live_anchor_position_m=candidate.T_base_grasp_tcp[:3, 3],
        anchor_kind="grasp_translation",
    )
    return GroundedPickPhases(
        candidate=candidate,
        approach_poses_base=np.stack(
            (candidate.T_base_pregrasp_tcp, candidate.T_base_grasp_tcp)
        ),
        lift_positions_base_m=grounded_lift.positions_base_m,
    )
