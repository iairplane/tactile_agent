import json

import numpy as np
import pytest

from demo2gripper.task_skill import (
    AnchorFrame,
    FailurePolicy,
    GraspOffsetFrame,
    GraspRelationMode,
    ObjectGraspRelation,
    PlannerMode,
    SkillStage,
    SkillStatus,
    StageName,
    StageTargetPolicy,
    TaskSkill,
    WaypointArtifactRef,
    WaypointRepresentation,
    ground_object_grasp,
)


def make_stage(name: StageName, index: int) -> SkillStage:
    anchors = {
        StageName.APPROACH: AnchorFrame.GRASP_TRANSLATION,
        StageName.PRESHAPE: AnchorFrame.GRASP_TRANSLATION,
        StageName.GRASP: AnchorFrame.GRASP_TRANSLATION,
        StageName.LIFT: AnchorFrame.RUNTIME_STAGE_START,
        StageName.TRANSPORT: AnchorFrame.RUNTIME_STAGE_START,
        StageName.PLACE: AnchorFrame.PLACEMENT,
        StageName.RELEASE: AnchorFrame.RELEASE_POSE,
        StageName.RETREAT: AnchorFrame.RELEASE_POSE,
    }
    return SkillStage(
        name=name,
        frame_interval=(10 * index, 10 * index + 10),
        time_interval_s=(index / 3.0, (index + 1) / 3.0),
        anchor=anchors[name],
        planner_mode=(
            PlannerMode.GRIPPER_ACTION
            if name in {StageName.GRASP, StageName.RELEASE}
            else PlannerMode.MOTION_PLANNING
        ),
        target_policy=(
            StageTargetPolicy.OBJECT_RELATIVE_GRASP
            if name is StageName.GRASP
            else (
                StageTargetPolicy.TASK_INTENT
                if name is StageName.RELEASE
                else StageTargetPolicy.ONLINE_AGENT_PLAN
            )
        ),
        waypoint_artifact=None,
        preconditions=(f"{name.value.lower()}_ready",),
        completion=(f"{name.value.lower()}_complete",),
        failure_policy=FailurePolicy.REOBSERVE_REPLAN,
        optional=name is StageName.TRANSPORT,
    )


def test_full_stage_skill_is_json_serializable_and_round_trips() -> None:
    stages = tuple(make_stage(name, index) for index, name in enumerate(StageName))
    relation = ObjectGraspRelation.base_aligned_planar(
        np.array([0.01, -0.02, 0.03]),
        np.array([1.0, 1.0, 0.0]),
        jaw_opening_m=0.04,
    )
    skill = TaskSkill(
        skill_id="dice_pickup_v0",
        stages=stages,
        grasp_relation=relation,
        blocking_conditions=("robot_executor_unavailable",),
    )

    payload = json.loads(skill.to_json())
    restored = TaskSkill.from_dict(payload)

    assert [stage.name for stage in restored.stages] == list(StageName)
    assert restored.stages[4].optional
    assert restored.stages[2].anchor is AnchorFrame.GRASP_TRANSLATION
    assert all(stage.waypoint_artifact is None for stage in restored.stages)
    assert all(
        stage.target_policy is StageTargetPolicy.ONLINE_AGENT_PLAN
        for stage in restored.stages
        if stage.name not in {StageName.GRASP, StageName.RELEASE}
    )
    assert restored.grasp_relation.mode is GraspRelationMode.BASE_ALIGNED_PLANAR
    assert restored.grasp_relation.offset_frame is GraspOffsetFrame.BASE_ALIGNED
    assert restored.grasp_relation.R_object_gripper_tcp is None
    np.testing.assert_allclose(
        restored.grasp_relation.closing_axis,
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
    )
    np.testing.assert_allclose(
        restored.grasp_relation.offset_from_object_center_m,
        relation.offset_from_object_center_m,
    )
    assert restored.status is SkillStatus.REVIEW_ONLY
    assert not restored.execution_authorized
    assert not restored.robot_execution_ready
    assert restored.blocking_conditions == ("robot_executor_unavailable",)


def test_task_skill_authorization_requires_ready_robot_and_no_blockers() -> None:
    stage = make_stage(StageName.APPROACH, 0)

    authorized = TaskSkill(
        skill_id="authorized",
        stages=(stage,),
        grasp_relation=None,
        status=SkillStatus.READY,
        execution_authorized=True,
        robot_execution_ready=True,
    )
    restored = TaskSkill.from_dict(json.loads(authorized.to_json()))
    assert restored.execution_authorized
    assert restored.robot_execution_ready
    assert restored.blocking_conditions == ()

    with pytest.raises(ValueError, match="robot_execution_ready"):
        TaskSkill(
            skill_id="robot_not_ready",
            stages=(stage,),
            grasp_relation=None,
            execution_authorized=True,
        )
    with pytest.raises(ValueError, match="no blocking_conditions"):
        TaskSkill(
            skill_id="blocked",
            stages=(stage,),
            grasp_relation=None,
            execution_authorized=True,
            robot_execution_ready=True,
            blocking_conditions=("operator_review_pending",),
        )
    with pytest.raises(ValueError, match="sequence of strings"):
        TaskSkill(
            skill_id="invalid_blockers",
            stages=(stage,),
            grasp_relation=None,
            blocking_conditions="not_a_tuple",
        )
    with pytest.raises(ValueError, match="status=ready"):
        TaskSkill(
            skill_id="review_only",
            stages=(stage,),
            grasp_relation=None,
            execution_authorized=True,
            robot_execution_ready=True,
        )


def test_translation_only_grounding_ignores_unknown_object_rotation() -> None:
    T_base_object = np.eye(4)
    T_base_object[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    T_base_object[:3, 3] = [0.5, 0.1, 0.2]
    relation = ObjectGraspRelation.translation_only(np.array([0.02, 0.0, 0.03]))

    target = ground_object_grasp(T_base_object, relation)

    np.testing.assert_allclose(target.position_base_m, [0.52, 0.1, 0.23])
    assert target.T_base_gripper_tcp is None
    assert not target.orientation_resolved
    with pytest.raises(ValueError, match="orientation is unresolved"):
        target.require_se3()


def test_planar_side_pinch_grounding_preserves_base_aligned_undirected_axis() -> None:
    T_base_object = np.eye(4)
    T_base_object[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    T_base_object[:3, 3] = [0.5, 0.1, 0.2]
    relation = ObjectGraspRelation.base_aligned_planar(
        np.array([0.02, -0.01, 0.03]), np.array([-2.0, -1.0, 0.0])
    )

    target = ground_object_grasp(T_base_object, relation)

    np.testing.assert_allclose(target.position_base_m, [0.52, 0.09, 0.23])
    np.testing.assert_allclose(
        target.require_closing_axis(), np.array([2.0, 1.0, 0.0]) / np.sqrt(5.0)
    )
    assert not target.orientation_resolved
    with pytest.raises(ValueError, match="orientation is unresolved"):
        target.require_se3()


def test_planar_jaw_line_is_invariant_to_axis_reversal() -> None:
    positive = ObjectGraspRelation.base_aligned_planar(
        np.zeros(3), np.array([0.3, 0.4, 0.0])
    )
    reversed_axis = ObjectGraspRelation.base_aligned_planar(
        np.zeros(3), np.array([-0.3, -0.4, 0.0])
    )

    np.testing.assert_allclose(positive.closing_axis, reversed_axis.closing_axis)


def test_se3_grounding_composes_object_relative_pose() -> None:
    T_base_object = np.eye(4)
    T_base_object[:3, 3] = [0.4, -0.1, 0.05]
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    relation = ObjectGraspRelation.se3(
        np.array([0.01, 0.02, 0.03]),
        rotation,
        approach_distance_m=0.08,
        object_frame_id="die_v1",
        gripper_name="robotiq_hande",
        jaw_opening_m=0.035,
    )

    target = ground_object_grasp(T_base_object, relation)

    expected = np.eye(4)
    expected[:3, :3] = rotation
    expected[:3, 3] = [0.41, -0.08, 0.08]
    assert target.orientation_resolved
    np.testing.assert_allclose(target.require_se3(), expected)


@pytest.mark.parametrize(
    ("mode", "rotation", "message"),
    [
        (GraspRelationMode.TRANSLATION_ONLY, np.eye(3), "must not contain a rotation"),
        (GraspRelationMode.SE3, None, "requires R_object_gripper_tcp"),
    ],
)
def test_grasp_relation_rejects_inconsistent_orientation_evidence(
    mode: GraspRelationMode, rotation: np.ndarray | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ObjectGraspRelation(
            mode=mode,
            offset_from_object_center_m=np.zeros(3),
            offset_frame=(
                GraspOffsetFrame.BASE_ALIGNED
                if mode is GraspRelationMode.TRANSLATION_ONLY
                else GraspOffsetFrame.OBJECT
            ),
            R_object_gripper_tcp=rotation,
            approach_distance_m=(0.08 if mode is GraspRelationMode.SE3 else None),
            object_frame_id=("die_v1" if mode is GraspRelationMode.SE3 else None),
            gripper_name=("robotiq_hande" if mode is GraspRelationMode.SE3 else None),
        )


def test_grasp_relation_rejects_offset_frame_inconsistent_with_mode() -> None:
    with pytest.raises(ValueError, match="must be base_aligned"):
        ObjectGraspRelation(
            mode=GraspRelationMode.TRANSLATION_ONLY,
            offset_from_object_center_m=np.zeros(3),
            offset_frame=GraspOffsetFrame.OBJECT,
        )

    with pytest.raises(ValueError, match="object frame"):
        ObjectGraspRelation(
            mode=GraspRelationMode.SE3,
            offset_from_object_center_m=np.zeros(3),
            offset_frame=GraspOffsetFrame.BASE_ALIGNED,
            R_object_gripper_tcp=np.eye(3),
            approach_distance_m=0.08,
            object_frame_id="die_v1",
            gripper_name="robotiq_hande",
        )


def test_task_skill_rejects_missing_grasp_relation_and_out_of_order_stages() -> None:
    grasp = make_stage(StageName.GRASP, 0)
    lift = make_stage(StageName.LIFT, 1)

    with pytest.raises(ValueError, match="GRASP stage requires"):
        TaskSkill(skill_id="missing_relation", stages=(grasp, lift), grasp_relation=None)

    relation = ObjectGraspRelation.base_aligned_planar(
        np.zeros(3), np.array([1.0, 0.0, 0.0])
    )
    with pytest.raises(ValueError, match="canonical manipulation order"):
        TaskSkill(
            skill_id="wrong_order",
            stages=(lift, grasp),
            grasp_relation=relation,
        )


def test_stage_rejects_invalid_intervals_and_empty_guards() -> None:
    kwargs = {
        "name": StageName.APPROACH,
        "frame_interval": (4, 4),
        "time_interval_s": (0.0, 1.0),
        "anchor": AnchorFrame.BASE,
        "planner_mode": PlannerMode.MOTION_PLANNING,
        "target_policy": StageTargetPolicy.ONLINE_AGENT_PLAN,
        "waypoint_artifact": None,
        "preconditions": ("object_pose_fresh",),
        "completion": ("pregrasp_reached",),
        "failure_policy": FailurePolicy.HOLD,
    }
    with pytest.raises(ValueError, match="non-empty half-open interval"):
        SkillStage(**kwargs)

    kwargs["frame_interval"] = (0, 4)
    kwargs["preconditions"] = ()
    with pytest.raises(ValueError, match="at least one predicate"):
        SkillStage(**kwargs)


def test_v3_rejects_a_legacy_demonstration_path() -> None:
    stage = SkillStage(
        name=StageName.APPROACH,
        frame_interval=(0, 10),
        time_interval_s=(0.0, 1.0),
        anchor=AnchorFrame.GRASP_TRANSLATION,
        planner_mode=PlannerMode.MOTION_PLANNING,
        target_policy=StageTargetPolicy.LEGACY_DEMONSTRATION_PATH,
        waypoint_artifact=WaypointArtifactRef(
            "trajectory.npz",
            WaypointRepresentation.POSITION_OFFSETS_BASE_M,
            "approach_position_offsets_base_m",
        ),
        preconditions=("scene_current",),
        completion=("pregrasp_reached",),
        failure_policy=FailurePolicy.REOBSERVE_REPLAN,
    )

    with pytest.raises(ValueError, match="must not replay"):
        TaskSkill(skill_id="legacy_path", stages=(stage,), grasp_relation=None)

    legacy = TaskSkill(
        skill_id="legacy_path",
        stages=(stage,),
        grasp_relation=None,
        schema_version="demo2gripper.agentic_task_skill.v2",
    )
    assert legacy.stages[0].waypoint_artifact is not None
