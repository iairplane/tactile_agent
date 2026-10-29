"""Declarative contract for an agentic, stage-based manipulation skill.

The contract keeps demonstration interpretation separate from robot execution.
In the v3 contract, a demonstration provides the semantic stage sequence and
the side-pinch endpoint constraint.  The runtime agent creates all motion-stage
targets from the current scene and MoveIt creates the executable trajectories.
Demonstrated hand paths may be retained in compiler diagnostics, but they are
not referenced by or consumed through :class:`TaskSkill`.

Intervals are half-open.  Transforms follow the repository convention
``T_parent_child`` and use metres, right-handed coordinates, and column vectors.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum, StrEnum
from typing import Any

import numpy as np

from .franka import validate_rigid_transform


class StageName(StrEnum):
    """Canonical order of the demonstrated manipulation stages."""

    APPROACH = "APPROACH"
    PRESHAPE = "PRESHAPE"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSPORT = "TRANSPORT"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"


class AnchorFrame(StrEnum):
    """Runtime reference used to ground a stage in the current scene.

    The waypoint representation declares how values are encoded; this enum
    declares which live reference the agent must supply before planning.  The
    four short names are retained for generic and existing skill definitions.
    """

    BASE = "base"
    OBJECT = "object"
    SUPPORT = "support"
    TOOL = "tool"
    OBJECT_CENTER_BASE_ALIGNED = "object_center_base_aligned"
    GRASP_TRANSLATION = "grasp_translation"
    RUNTIME_STAGE_START = "runtime_stage_start"
    PLACEMENT = "placement"
    RELEASE_POSE = "release_pose"


class WaypointRepresentation(StrEnum):
    """On-disk numeric representation of a stage's waypoint artifact."""

    POSITION_OFFSETS_BASE_M = "position_offsets_base_m"
    POSES_SE3 = "poses_se3"


class PlannerMode(StrEnum):
    """Executor family requested by a stage, without selecting an implementation."""

    MOTION_PLANNING = "motion_planning"
    CARTESIAN = "cartesian"
    GRIPPER_ACTION = "gripper_action"


class StageTargetPolicy(StrEnum):
    """Authority that supplies a stage target at runtime.

    ``legacy_demonstration_path`` exists only so v2 artifacts can still be read.
    It is rejected by the v3 task contract.
    """

    ONLINE_AGENT_PLAN = "online_agent_plan"
    OBJECT_RELATIVE_GRASP = "object_relative_grasp"
    TASK_INTENT = "task_intent"
    LEGACY_DEMONSTRATION_PATH = "legacy_demonstration_path"


class FailurePolicy(StrEnum):
    """Agent action when a stage's guard or completion check fails."""

    HOLD = "hold"
    REOBSERVE_REPLAN = "reobserve_replan"
    RETREAT_REOBSERVE_RETRY = "retreat_reobserve_retry"
    ABORT = "abort"


class GraspRelationMode(StrEnum):
    """Amount of object-relative grasp geometry supported by evidence."""

    TRANSLATION_ONLY = "translation_only"
    BASE_ALIGNED_PLANAR = "base_aligned_planar"
    SE3 = "se3"


class GraspOffsetFrame(StrEnum):
    """Frame in which the object-center-to-grasp offset is expressed."""

    BASE_ALIGNED = "base_aligned"
    OBJECT = "object"


class SkillStatus(StrEnum):
    """Review state of a compiled task skill."""

    REVIEW_ONLY = "review_only"
    READY = "ready"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class JsonSerializable:
    """JSON-safe conversion shared by the task-skill dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _validate_frame_interval(interval: tuple[int, int], *, name: str) -> None:
    if len(interval) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    start, end = interval
    if isinstance(start, bool) or isinstance(end, bool):
        raise ValueError(f"{name} values must be integers")
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"{name} values must be integers")
    if start < 0 or end <= start:
        raise ValueError(f"{name} must be a non-empty half-open interval")


def _validate_time_interval(interval: tuple[float, float], *, name: str) -> None:
    values = np.asarray(interval, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain two finite values")
    if values[0] < 0.0 or values[1] <= values[0]:
        raise ValueError(f"{name} must be a non-empty half-open interval")


def _validate_conditions(conditions: tuple[str, ...], *, name: str) -> None:
    if not conditions:
        raise ValueError(f"{name} must contain at least one predicate")
    if any(not isinstance(item, str) or not item.strip() for item in conditions):
        raise ValueError(f"{name} predicates must be non-empty strings")


@dataclass(frozen=True)
class WaypointArtifactRef(JsonSerializable):
    """Reference to demonstrated waypoints stored outside the skill JSON.

    ``uri`` may be a repository-relative file path or another artifact URI.
    ``key`` selects an array within a container such as an NPZ file.
    ``representation`` prevents position-only offsets from being consumed as
    complete tool poses.
    """

    uri: str
    representation: WaypointRepresentation
    key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "representation", WaypointRepresentation(self.representation)
        )
        if not isinstance(self.uri, str) or not self.uri.strip():
            raise ValueError("waypoint artifact uri must be a non-empty string")
        if self.key is not None and (not isinstance(self.key, str) or not self.key.strip()):
            raise ValueError("waypoint artifact key must be a non-empty string or None")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WaypointArtifactRef:
        return cls(
            uri=str(data["uri"]),
            representation=WaypointRepresentation(data["representation"]),
            key=data.get("key"),
        )


@dataclass(frozen=True)
class SkillStage(JsonSerializable):
    """One demonstrated stage and its closed-loop execution contract."""

    name: StageName
    frame_interval: tuple[int, int]
    time_interval_s: tuple[float, float]
    anchor: AnchorFrame
    planner_mode: PlannerMode
    target_policy: StageTargetPolicy
    waypoint_artifact: WaypointArtifactRef | None
    preconditions: tuple[str, ...]
    completion: tuple[str, ...]
    failure_policy: FailurePolicy
    optional: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", StageName(self.name))
        object.__setattr__(self, "anchor", AnchorFrame(self.anchor))
        object.__setattr__(self, "planner_mode", PlannerMode(self.planner_mode))
        object.__setattr__(self, "target_policy", StageTargetPolicy(self.target_policy))
        object.__setattr__(self, "failure_policy", FailurePolicy(self.failure_policy))
        _validate_frame_interval(self.frame_interval, name=f"{self.name.value} frame_interval")
        _validate_time_interval(self.time_interval_s, name=f"{self.name.value} time_interval_s")
        _validate_conditions(self.preconditions, name=f"{self.name.value} preconditions")
        _validate_conditions(self.completion, name=f"{self.name.value} completion")
        if not isinstance(self.optional, bool):
            raise ValueError("optional must be a bool")
        if self.name is StageName.GRASP:
            if self.target_policy is not StageTargetPolicy.OBJECT_RELATIVE_GRASP:
                raise ValueError("GRASP must use the object-relative grasp target policy")
        elif self.target_policy is StageTargetPolicy.OBJECT_RELATIVE_GRASP:
            raise ValueError("object-relative grasp target policy is only valid for GRASP")
        if self.name is StageName.RELEASE:
            if self.target_policy is not StageTargetPolicy.TASK_INTENT:
                raise ValueError("RELEASE must use the task-intent target policy")
        elif self.target_policy is StageTargetPolicy.TASK_INTENT:
            raise ValueError("task-intent target policy is only valid for RELEASE")
        if (
            self.target_policy is StageTargetPolicy.ONLINE_AGENT_PLAN
            and self.waypoint_artifact is not None
        ):
            raise ValueError(
                "online agent-planned stages must not carry executable waypoints"
            )
        if (
            self.target_policy
            in {
                StageTargetPolicy.OBJECT_RELATIVE_GRASP,
                StageTargetPolicy.TASK_INTENT,
            }
            and self.waypoint_artifact is not None
        ):
            raise ValueError(
                "semantic endpoint/action stages must not carry executable waypoints"
            )
        if (
            self.target_policy is StageTargetPolicy.LEGACY_DEMONSTRATION_PATH
            and self.waypoint_artifact is None
        ):
            raise ValueError("legacy demonstration-path stages require a waypoint artifact")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillStage:
        artifact = data.get("waypoint_artifact")
        name = StageName(data["name"])
        if "target_policy" in data:
            target_policy = StageTargetPolicy(data["target_policy"])
        elif name is StageName.GRASP:
            target_policy = StageTargetPolicy.OBJECT_RELATIVE_GRASP
        elif name is StageName.RELEASE:
            target_policy = StageTargetPolicy.TASK_INTENT
        elif artifact is not None:
            target_policy = StageTargetPolicy.LEGACY_DEMONSTRATION_PATH
        else:
            target_policy = StageTargetPolicy.ONLINE_AGENT_PLAN
        return cls(
            name=name,
            frame_interval=tuple(int(value) for value in data["frame_interval"]),
            time_interval_s=tuple(float(value) for value in data["time_interval_s"]),
            anchor=AnchorFrame(data["anchor"]),
            planner_mode=PlannerMode(data["planner_mode"]),
            target_policy=target_policy,
            waypoint_artifact=(
                None if artifact is None else WaypointArtifactRef.from_dict(artifact)
            ),
            preconditions=tuple(str(value) for value in data["preconditions"]),
            completion=tuple(str(value) for value in data["completion"]),
            failure_policy=FailurePolicy(data["failure_policy"]),
            optional=data.get("optional", False),
        )


@dataclass(frozen=True)
class ObjectGraspRelation(JsonSerializable):
    """Grasp evidence expressed relative to the object.

    ``base_aligned_planar`` is the legacy side-pinch contract: it stores a
    base-aligned offset from the observed object center and an undirected jaw
    line in the base XY plane.  It deliberately contains no approach direction
    or TCP pose; robot adaptation and the runtime agent resolve those pieces.
    It transfers with object translation, but not object rotation.

    ``translation_only`` is retained for v2 compatibility and carries no jaw
    direction. ``se3`` is the current contract: it stores the selected robot
    TCP pose, pregrasp distance, object-frame identity, and gripper identity,
    and therefore requires a matching observed object pose.
    """

    mode: GraspRelationMode
    offset_from_object_center_m: np.ndarray
    offset_frame: GraspOffsetFrame
    R_object_gripper_tcp: np.ndarray | None = None
    closing_axis: np.ndarray | None = None
    jaw_opening_m: float | None = None
    approach_distance_m: float | None = None
    object_frame_id: str | None = None
    gripper_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", GraspRelationMode(self.mode))
        object.__setattr__(self, "offset_frame", GraspOffsetFrame(self.offset_frame))
        offset = np.asarray(self.offset_from_object_center_m, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("offset_from_object_center_m must contain three finite values")
        object.__setattr__(self, "offset_from_object_center_m", offset)

        if self.mode is GraspRelationMode.TRANSLATION_ONLY:
            if self.offset_frame is not GraspOffsetFrame.BASE_ALIGNED:
                raise ValueError("translation_only grasp offset must be base_aligned")
            if self.R_object_gripper_tcp is not None:
                raise ValueError("translation_only grasp relation must not contain a rotation")
            if self.closing_axis is not None:
                raise ValueError("translation_only grasp relation must not contain a closing axis")
        elif self.mode is GraspRelationMode.BASE_ALIGNED_PLANAR:
            if self.offset_frame is not GraspOffsetFrame.BASE_ALIGNED:
                raise ValueError("base_aligned_planar grasp offset must be base_aligned")
            if self.R_object_gripper_tcp is not None:
                raise ValueError("base_aligned_planar relation must not contain a rotation")
            if self.closing_axis is None:
                raise ValueError("base_aligned_planar relation requires closing_axis")
            closing = np.asarray(self.closing_axis, dtype=float)
            if closing.shape != (3,) or not np.isfinite(closing).all():
                raise ValueError("closing_axis must contain three finite values")
            if abs(float(closing[2])) > 1e-6:
                raise ValueError("base_aligned_planar closing_axis must lie in the base XY plane")
            norm = float(np.linalg.norm(closing[:2]))
            if norm <= 1e-9:
                raise ValueError("closing_axis must be non-zero")
            closing = closing / norm
            closing[2] = 0.0
            # A parallel-jaw closing line is invariant under a 180-degree flip.
            # Store one deterministic representative so theta and theta + pi
            # serialize and compare identically.
            if closing[0] < -1e-12 or (
                abs(float(closing[0])) <= 1e-12 and closing[1] < 0.0
            ):
                closing = -closing
            object.__setattr__(self, "closing_axis", closing)
        else:
            if self.offset_frame is not GraspOffsetFrame.OBJECT:
                raise ValueError("se3 grasp offset must be expressed in the object frame")
            if self.R_object_gripper_tcp is None:
                raise ValueError("se3 grasp relation requires R_object_gripper_tcp")
            rotation = np.asarray(self.R_object_gripper_tcp, dtype=float)
            if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
                raise ValueError("R_object_gripper_tcp must be a finite 3x3 matrix")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
                raise ValueError("R_object_gripper_tcp must be orthonormal")
            if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
                raise ValueError("R_object_gripper_tcp determinant must be +1")
            object.__setattr__(self, "R_object_gripper_tcp", rotation)
            if self.closing_axis is not None:
                raise ValueError("se3 grasp relation encodes its closing axis in the rotation")

            if (
                not isinstance(self.object_frame_id, str)
                or not self.object_frame_id.strip()
            ):
                raise ValueError("se3 grasp relation requires object_frame_id")
            if not isinstance(self.gripper_name, str) or not self.gripper_name.strip():
                raise ValueError("se3 grasp relation requires gripper_name")

        if self.mode is not GraspRelationMode.SE3 and (
            self.object_frame_id is not None or self.gripper_name is not None
        ):
            raise ValueError(
                "object_frame_id and gripper_name are only valid for se3 relations"
            )

        if self.jaw_opening_m is not None:
            if not np.isfinite(self.jaw_opening_m) or self.jaw_opening_m <= 0.0:
                raise ValueError("jaw_opening_m must be positive and finite or None")
        if self.approach_distance_m is not None:
            if (
                not np.isfinite(self.approach_distance_m)
                or self.approach_distance_m <= 0.0
            ):
                raise ValueError(
                    "approach_distance_m must be positive and finite or None"
                )
        if self.mode is GraspRelationMode.SE3 and self.approach_distance_m is None:
            raise ValueError("se3 grasp relation requires approach_distance_m")
        if self.mode is not GraspRelationMode.SE3 and self.approach_distance_m is not None:
            raise ValueError("approach_distance_m is only valid for se3 relations")

    @classmethod
    def translation_only(
        cls,
        offset_from_object_center_m: np.ndarray,
        *,
        jaw_opening_m: float | None = None,
    ) -> ObjectGraspRelation:
        return cls(
            mode=GraspRelationMode.TRANSLATION_ONLY,
            offset_from_object_center_m=offset_from_object_center_m,
            offset_frame=GraspOffsetFrame.BASE_ALIGNED,
            jaw_opening_m=jaw_opening_m,
        )

    @classmethod
    def se3(
        cls,
        offset_from_object_center_m: np.ndarray,
        R_object_gripper_tcp: np.ndarray,
        *,
        approach_distance_m: float,
        object_frame_id: str,
        gripper_name: str,
        jaw_opening_m: float | None = None,
    ) -> ObjectGraspRelation:
        return cls(
            mode=GraspRelationMode.SE3,
            offset_from_object_center_m=offset_from_object_center_m,
            offset_frame=GraspOffsetFrame.OBJECT,
            R_object_gripper_tcp=R_object_gripper_tcp,
            jaw_opening_m=jaw_opening_m,
            approach_distance_m=approach_distance_m,
            object_frame_id=object_frame_id,
            gripper_name=gripper_name,
        )

    @classmethod
    def base_aligned_planar(
        cls,
        offset_from_object_center_m: np.ndarray,
        closing_axis_base: np.ndarray,
        *,
        jaw_opening_m: float | None = None,
    ) -> ObjectGraspRelation:
        """Store a translation-anchor-relative pinch center and base-plane jaw line."""

        return cls(
            mode=GraspRelationMode.BASE_ALIGNED_PLANAR,
            offset_from_object_center_m=offset_from_object_center_m,
            offset_frame=GraspOffsetFrame.BASE_ALIGNED,
            closing_axis=closing_axis_base,
            jaw_opening_m=jaw_opening_m,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectGraspRelation:
        rotation = data.get("R_object_gripper_tcp")
        return cls(
            mode=GraspRelationMode(data["mode"]),
            offset_from_object_center_m=np.asarray(
                data["offset_from_object_center_m"], dtype=float
            ),
            offset_frame=GraspOffsetFrame(data["offset_frame"]),
            R_object_gripper_tcp=(
                None if rotation is None else np.asarray(rotation, dtype=float)
            ),
            closing_axis=(
                None
                if data.get("closing_axis") is None
                else np.asarray(data["closing_axis"], dtype=float)
            ),
            jaw_opening_m=(
                None if data.get("jaw_opening_m") is None else float(data["jaw_opening_m"])
            ),
            approach_distance_m=(
                None
                if data.get("approach_distance_m") is None
                else float(data["approach_distance_m"])
            ),
            object_frame_id=data.get("object_frame_id"),
            gripper_name=data.get("gripper_name"),
        )


@dataclass(frozen=True)
class GroundedGraspTarget(JsonSerializable):
    """Current-scene grounding of an object-relative grasp relation.

    ``T_base_gripper_tcp`` remains ``None`` for a planar relation because the
    demonstration does not select the robot approach axis or encode the Hand-E
    TCP mapping. This is an endpoint constraint for the runtime agent, not proof
    of IK, collision, width, or execution feasibility.
    """

    position_base_m: np.ndarray
    T_base_gripper_tcp: np.ndarray | None
    relation_mode: GraspRelationMode
    closing_axis_base: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_mode", GraspRelationMode(self.relation_mode))
        position = np.asarray(self.position_base_m, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("position_base_m must contain three finite values")
        object.__setattr__(self, "position_base_m", position)

        if self.relation_mode is GraspRelationMode.TRANSLATION_ONLY:
            if self.T_base_gripper_tcp is not None:
                raise ValueError("translation_only grounded target must not contain an SE(3) pose")
            if self.closing_axis_base is not None:
                raise ValueError("translation_only grounded target has no closing axis")
        elif self.relation_mode is GraspRelationMode.BASE_ALIGNED_PLANAR:
            if self.T_base_gripper_tcp is not None:
                raise ValueError("base_aligned_planar target must not contain an SE(3) pose")
            if self.closing_axis_base is None:
                raise ValueError("base_aligned_planar target requires a closing axis")
            closing = np.asarray(self.closing_axis_base, dtype=float)
            if closing.shape != (3,) or not np.isfinite(closing).all():
                raise ValueError("closing_axis_base must contain three finite values")
            if abs(float(closing[2])) > 1e-6:
                raise ValueError("closing_axis_base must lie in the base XY plane")
            if not np.isclose(np.linalg.norm(closing), 1.0, atol=1e-6):
                raise ValueError("closing_axis_base must be a unit vector")
            object.__setattr__(self, "closing_axis_base", closing)
        elif self.T_base_gripper_tcp is None:
            raise ValueError("se3 grounded target requires T_base_gripper_tcp")
        else:
            pose = validate_rigid_transform(
                self.T_base_gripper_tcp, name="T_base_gripper_tcp"
            )
            if not np.allclose(pose[:3, 3], position, atol=1e-8):
                raise ValueError("T_base_gripper_tcp translation must match position_base_m")
            object.__setattr__(self, "T_base_gripper_tcp", pose)
            object.__setattr__(self, "closing_axis_base", pose[:3, 0].copy())

    @property
    def orientation_resolved(self) -> bool:
        return self.T_base_gripper_tcp is not None

    def require_se3(self) -> np.ndarray:
        if self.T_base_gripper_tcp is None:
            raise ValueError("grasp orientation is unresolved; an SE(3) target is unavailable")
        return self.T_base_gripper_tcp.copy()

    def require_closing_axis(self) -> np.ndarray:
        if self.closing_axis_base is None:
            raise ValueError("grasp closing axis is unresolved")
        return self.closing_axis_base.copy()


def ground_object_grasp(
    T_base_object: np.ndarray, relation: ObjectGraspRelation
) -> GroundedGraspTarget:
    """Ground an object-relative grasp without inventing missing orientation."""

    base_object = validate_rigid_transform(T_base_object, name="T_base_object")
    if relation.mode is GraspRelationMode.TRANSLATION_ONLY:
        position_base = (
            base_object[:3, 3] + relation.offset_from_object_center_m
        )
        return GroundedGraspTarget(
            position_base_m=position_base,
            T_base_gripper_tcp=None,
            relation_mode=relation.mode,
        )

    if relation.mode is GraspRelationMode.BASE_ALIGNED_PLANAR:
        position_base = base_object[:3, 3] + relation.offset_from_object_center_m
        return GroundedGraspTarget(
            position_base_m=position_base,
            T_base_gripper_tcp=None,
            relation_mode=relation.mode,
            closing_axis_base=relation.closing_axis,
        )

    object_tcp = np.eye(4, dtype=float)
    object_tcp[:3, :3] = relation.R_object_gripper_tcp
    object_tcp[:3, 3] = relation.offset_from_object_center_m
    position_base = (
        base_object[:3, :3] @ relation.offset_from_object_center_m
        + base_object[:3, 3]
    )
    return GroundedGraspTarget(
        position_base_m=position_base,
        T_base_gripper_tcp=base_object @ object_tcp,
        relation_mode=relation.mode,
        closing_axis_base=(base_object @ object_tcp)[:3, 0],
    )


@dataclass(frozen=True)
class TaskSkill(JsonSerializable):
    """Ordered skill graph compiled from one human demonstration."""

    skill_id: str
    stages: tuple[SkillStage, ...]
    grasp_relation: ObjectGraspRelation | None
    schema_version: str = "demo2gripper.agentic_task_skill.v3"
    status: SkillStatus = SkillStatus.REVIEW_ONLY
    execution_authorized: bool = False
    robot_execution_ready: bool = False
    blocking_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SkillStatus(self.status))
        if isinstance(self.blocking_conditions, (str, bytes)):
            raise ValueError("blocking_conditions must be a sequence of strings")
        object.__setattr__(self, "blocking_conditions", tuple(self.blocking_conditions))
        if not isinstance(self.skill_id, str) or not self.skill_id.strip():
            raise ValueError("skill_id must be a non-empty string")
        if not self.stages:
            raise ValueError("stages must not be empty")
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique")
        order = {name: index for index, name in enumerate(StageName)}
        if [order[name] for name in names] != sorted(order[name] for name in names):
            raise ValueError("stages must follow the canonical manipulation order")
        if StageName.GRASP in names and self.grasp_relation is None:
            raise ValueError("a GRASP stage requires grasp_relation")
        if StageName.GRASP not in names and self.grasp_relation is not None:
            raise ValueError("grasp_relation requires a GRASP stage")
        if (
            self.schema_version == "demo2gripper.agentic_task_skill.v3"
            and self.grasp_relation is not None
            and self.grasp_relation.mode is GraspRelationMode.TRANSLATION_ONLY
        ):
            raise ValueError(
                "v3 GRASP requires a closing-axis constraint or a complete SE(3) relation"
            )
        if self.schema_version == "demo2gripper.agentic_task_skill.v3":
            if any(
                stage.target_policy is StageTargetPolicy.LEGACY_DEMONSTRATION_PATH
                for stage in self.stages
            ):
                raise ValueError("v3 stages must not replay a demonstration path")
            if any(stage.waypoint_artifact is not None for stage in self.stages):
                raise ValueError("v3 stages must not reference demonstration waypoints")
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be a non-empty string")
        if not isinstance(self.execution_authorized, bool):
            raise ValueError("execution_authorized must be a bool")
        if not isinstance(self.robot_execution_ready, bool):
            raise ValueError("robot_execution_ready must be a bool")
        if any(
            not isinstance(condition, str) or not condition.strip()
            for condition in self.blocking_conditions
        ):
            raise ValueError("blocking_conditions must contain non-empty strings")
        if self.execution_authorized and not self.robot_execution_ready:
            raise ValueError("execution authorization requires robot_execution_ready")
        if self.execution_authorized and self.blocking_conditions:
            raise ValueError("execution authorization requires no blocking_conditions")
        if self.execution_authorized and self.status is not SkillStatus.READY:
            raise ValueError("execution authorization requires status=ready")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSkill:
        relation = data.get("grasp_relation")
        return cls(
            skill_id=str(data["skill_id"]),
            stages=tuple(SkillStage.from_dict(stage) for stage in data["stages"]),
            grasp_relation=(
                None if relation is None else ObjectGraspRelation.from_dict(relation)
            ),
            schema_version=str(
                data.get("schema_version", "demo2gripper.agentic_task_skill.v3")
            ),
            status=SkillStatus(data.get("status", SkillStatus.REVIEW_ONLY.value)),
            execution_authorized=data.get("execution_authorized", False),
            robot_execution_ready=data.get("robot_execution_ready", False),
            blocking_conditions=tuple(data.get("blocking_conditions", ())),
        )

