import pytest

from demo2gripper.agent_loop import (
    AgentAction,
    AgentFeedback,
    AgentLoop,
    AgentState,
)
from demo2gripper.task_skill import (
    AnchorFrame,
    FailurePolicy,
    PlannerMode,
    SkillStage,
    SkillStatus,
    StageName,
    StageTargetPolicy,
    TaskSkill,
)

READY = frozenset({"stage_ready"})
COMPLETE = frozenset({"stage_complete"})


def make_stage(
    name: StageName,
    index: int,
    *,
    anchor: AnchorFrame = AnchorFrame.OBJECT_CENTER_BASE_ALIGNED,
    optional: bool = False,
    failure_policy: FailurePolicy = FailurePolicy.REOBSERVE_REPLAN,
    planner_mode: PlannerMode = PlannerMode.MOTION_PLANNING,
) -> SkillStage:
    return SkillStage(
        name=name,
        frame_interval=(index * 10, index * 10 + 10),
        time_interval_s=(float(index), float(index + 1)),
        anchor=anchor,
        planner_mode=planner_mode,
        target_policy=StageTargetPolicy.ONLINE_AGENT_PLAN,
        waypoint_artifact=None,
        preconditions=("stage_ready",),
        completion=("stage_complete",),
        failure_policy=failure_policy,
        optional=optional,
    )


def make_skill(
    *stages: SkillStage,
    status: SkillStatus = SkillStatus.READY,
    execution_authorized: bool = True,
    robot_execution_ready: bool = True,
    blocking_conditions: tuple[str, ...] = (),
) -> TaskSkill:
    return TaskSkill(
        skill_id="test_skill",
        stages=stages,
        grasp_relation=None,
        status=status,
        execution_authorized=execution_authorized,
        robot_execution_ready=robot_execution_ready,
        blocking_conditions=blocking_conditions,
    )


def drive_to_plan(loop: AgentLoop) -> None:
    decision = loop.step(AgentFeedback(perception_fresh=True))
    assert (decision.action, decision.state) == (AgentAction.GROUND, AgentState.GROUND)

    decision = loop.step(
        AgentFeedback(
            perception_fresh=True,
            grounded=True,
            satisfied_predicates=READY,
        )
    )
    assert (decision.action, decision.state) == (AgentAction.PLAN, AgentState.PLAN)


def drive_to_execute(loop: AgentLoop) -> None:
    drive_to_plan(loop)
    decision = loop.step(
        AgentFeedback(
            perception_fresh=True,
            plan_valid=True,
            satisfied_predicates=READY,
        )
    )
    assert (decision.action, decision.state) == (
        AgentAction.EXECUTE,
        AgentState.EXECUTE,
    )


def test_happy_path_advances_in_order_and_completes() -> None:
    approach = make_stage(StageName.APPROACH, 0)
    lift = make_stage(StageName.LIFT, 1, anchor=AnchorFrame.RUNTIME_STAGE_START)
    loop = AgentLoop(make_skill(approach, lift))

    assert loop.current_stage is approach
    drive_to_execute(loop)
    decision = loop.step(AgentFeedback())
    assert decision.action is AgentAction.HOLD
    assert decision.reason == "execution result is pending"

    decision = loop.step(AgentFeedback(execution_succeeded=True))
    assert (decision.action, decision.state) == (AgentAction.HOLD, AgentState.VERIFY)
    decision = loop.step(AgentFeedback(satisfied_predicates=COMPLETE))
    assert decision.action is AgentAction.ADVANCE
    assert (decision.stage_index, decision.stage_name) == (0, StageName.APPROACH)
    assert loop.stage_index == 1
    assert loop.current_stage is lift
    assert loop.state is AgentState.OBSERVE

    drive_to_execute(loop)
    loop.step(AgentFeedback(execution_succeeded=True))
    decision = loop.step(AgentFeedback(satisfied_predicates=COMPLETE))
    assert decision.action is AgentAction.COMPLETE
    assert decision.terminal
    assert loop.done
    assert loop.current_stage is None
    with pytest.raises(RuntimeError, match="terminal decision"):
        loop.step(AgentFeedback())


def test_declared_predicates_gate_planning_execution_and_completion() -> None:
    loop = AgentLoop(make_skill(make_stage(StageName.APPROACH, 0)))
    loop.step(AgentFeedback(perception_fresh=True))

    decision = loop.step(AgentFeedback(perception_fresh=True, grounded=True))
    assert (decision.action, decision.state) == (AgentAction.HOLD, AgentState.GROUND)
    assert "stage_ready" in decision.reason

    decision = loop.step(
        AgentFeedback(
            perception_fresh=True,
            grounded=True,
            satisfied_predicates=READY,
        )
    )
    assert decision.action is AgentAction.PLAN
    decision = loop.step(AgentFeedback(perception_fresh=True, plan_valid=True))
    assert (decision.action, decision.state) == (AgentAction.HOLD, AgentState.PLAN)
    assert "stage_ready" in decision.reason

    decision = loop.step(
        AgentFeedback(
            perception_fresh=True,
            plan_valid=True,
            satisfied_predicates=READY,
        )
    )
    assert decision.action is AgentAction.EXECUTE
    loop.step(AgentFeedback(execution_succeeded=True))
    decision = loop.step(AgentFeedback())
    assert (decision.action, decision.state) == (AgentAction.HOLD, AgentState.VERIFY)
    assert "stage_complete" in decision.reason
    assert loop.step(AgentFeedback(satisfied_predicates=COMPLETE)).action is AgentAction.COMPLETE


@pytest.mark.parametrize(
    ("skill_kwargs", "expected_action"),
    [
        (
            {
                "status": SkillStatus.REVIEW_ONLY,
                "execution_authorized": False,
                "robot_execution_ready": True,
            },
            AgentAction.REVIEW_REQUIRED,
        ),
        (
            {
                "status": SkillStatus.READY,
                "execution_authorized": False,
                "robot_execution_ready": True,
            },
            AgentAction.REVIEW_REQUIRED,
        ),
        (
            {
                "status": SkillStatus.READY,
                "execution_authorized": False,
                "robot_execution_ready": False,
            },
            AgentAction.HOLD,
        ),
    ],
)
def test_unapproved_or_unready_skill_never_emits_execute(
    skill_kwargs: dict[str, object], expected_action: AgentAction
) -> None:
    stage = make_stage(StageName.APPROACH, 0)
    loop = AgentLoop(make_skill(stage, **skill_kwargs))  # type: ignore[arg-type]
    drive_to_plan(loop)

    decision = loop.step(
        AgentFeedback(
            perception_fresh=True,
            plan_valid=True,
            satisfied_predicates=READY,
        )
    )

    assert decision.action is expected_action
    assert decision.action is not AgentAction.EXECUTE
    assert decision.state is AgentState.PLAN


@pytest.mark.parametrize(
    "anchor",
    [
        AnchorFrame.OBJECT_CENTER_BASE_ALIGNED,
        AnchorFrame.GRASP_TRANSLATION,
        AnchorFrame.OBJECT,
    ],
)
def test_invalidated_grounding_anchor_cancels_execution_before_reobserve(
    anchor: AnchorFrame,
) -> None:
    loop = AgentLoop(make_skill(make_stage(StageName.APPROACH, 0, anchor=anchor)))
    drive_to_execute(loop)

    decision = loop.step(
        AgentFeedback(
            perception_fresh=True,
            grounding_anchor_invalidated=True,
        )
    )
    assert (decision.action, decision.state) == (
        AgentAction.CANCEL_AND_HOLD,
        AgentState.CANCEL,
    )

    decision = loop.step(AgentFeedback(perception_fresh=True))
    assert (decision.action, decision.state) == (
        AgentAction.CANCEL_AND_HOLD,
        AgentState.CANCEL,
    )
    decision = loop.step(
        AgentFeedback(perception_fresh=True, execution_stopped=True)
    )
    assert (decision.action, decision.state) == (
        AgentAction.OBSERVE,
        AgentState.OBSERVE,
    )


def test_invalidated_anchor_before_execution_regrounds_then_replans() -> None:
    loop = AgentLoop(make_skill(make_stage(StageName.APPROACH, 0)))
    drive_to_plan(loop)

    decision = loop.step(
        AgentFeedback(
            perception_fresh=True,
            grounding_anchor_invalidated=True,
        )
    )

    assert (decision.action, decision.state) == (AgentAction.REPLAN, AgentState.GROUND)
    decision = loop.step(
        AgentFeedback(
            perception_fresh=True,
            grounded=True,
            satisfied_predicates=READY,
        )
    )
    assert (decision.action, decision.state) == (AgentAction.PLAN, AgentState.PLAN)


@pytest.mark.parametrize(
    ("name", "anchor"),
    [
        (StageName.LIFT, AnchorFrame.RUNTIME_STAGE_START),
        (StageName.PLACE, AnchorFrame.PLACEMENT),
    ],
)
def test_expected_object_motion_does_not_invalidate_lift_or_place(
    name: StageName, anchor: AnchorFrame
) -> None:
    loop = AgentLoop(make_skill(make_stage(name, 0, anchor=anchor)))
    drive_to_execute(loop)

    decision = loop.step(AgentFeedback(grounding_anchor_invalidated=True))

    assert (decision.action, decision.state) == (AgentAction.HOLD, AgentState.EXECUTE)


@pytest.mark.parametrize(
    ("policy", "expected_action", "expected_state", "terminal"),
    [
        (FailurePolicy.HOLD, AgentAction.HOLD, AgentState.PLAN, False),
        (
            FailurePolicy.REOBSERVE_REPLAN,
            AgentAction.OBSERVE,
            AgentState.OBSERVE,
            False,
        ),
        (
            FailurePolicy.RETREAT_REOBSERVE_RETRY,
            AgentAction.RECOVERY_RETREAT,
            AgentState.RECOVERY,
            False,
        ),
        (FailurePolicy.ABORT, AgentAction.ABORT, AgentState.PLAN, True),
    ],
)
def test_stage_failure_policy_controls_recovery(
    policy: FailurePolicy,
    expected_action: AgentAction,
    expected_state: AgentState,
    terminal: bool,
) -> None:
    stage = make_stage(StageName.APPROACH, 0, failure_policy=policy)
    loop = AgentLoop(make_skill(stage))
    drive_to_plan(loop)

    decision = loop.step(AgentFeedback(recoverable_failure=True))

    assert (decision.action, decision.state) == (expected_action, expected_state)
    assert decision.terminal is terminal
    assert loop.aborted is terminal


def test_recovery_retreat_must_complete_before_reobservation() -> None:
    stage = make_stage(
        StageName.APPROACH,
        0,
        failure_policy=FailurePolicy.RETREAT_REOBSERVE_RETRY,
    )
    loop = AgentLoop(make_skill(stage))
    drive_to_plan(loop)
    decision = loop.step(AgentFeedback(recoverable_failure=True))
    assert (decision.action, decision.state) == (
        AgentAction.RECOVERY_RETREAT,
        AgentState.RECOVERY,
    )

    decision = loop.step(AgentFeedback(perception_fresh=True))
    assert (decision.action, decision.state) == (
        AgentAction.RECOVERY_RETREAT,
        AgentState.RECOVERY,
    )
    decision = loop.step(AgentFeedback(recovery_completed=True))
    assert (decision.action, decision.state) == (
        AgentAction.OBSERVE,
        AgentState.OBSERVE,
    )


@pytest.mark.parametrize(
    ("skill_kwargs", "expected_action"),
    [
        (
            {
                "status": SkillStatus.REVIEW_ONLY,
                "execution_authorized": False,
                "robot_execution_ready": False,
            },
            AgentAction.REVIEW_REQUIRED,
        ),
        (
            {
                "status": SkillStatus.READY,
                "execution_authorized": False,
                "robot_execution_ready": True,
            },
            AgentAction.REVIEW_REQUIRED,
        ),
        (
            {
                "status": SkillStatus.READY,
                "execution_authorized": False,
                "robot_execution_ready": False,
            },
            AgentAction.HOLD,
        ),
    ],
)
def test_recovery_retreat_obeys_the_same_authorization_gate_as_execution(
    skill_kwargs: dict[str, object], expected_action: AgentAction
) -> None:
    stage = make_stage(
        StageName.PRESHAPE,
        0,
        failure_policy=FailurePolicy.RETREAT_REOBSERVE_RETRY,
    )
    loop = AgentLoop(make_skill(stage, **skill_kwargs))  # type: ignore[arg-type]
    drive_to_plan(loop)

    decision = loop.step(AgentFeedback(recoverable_failure=True))

    assert decision.action is expected_action
    assert decision.action not in {
        AgentAction.EXECUTE,
        AgentAction.RECOVERY_RETREAT,
    }
    assert decision.state is AgentState.PLAN
    assert loop.state is AgentState.PLAN


def test_optional_stage_requires_explicit_skip_and_required_stage_cannot_skip() -> None:
    transport = make_stage(
        StageName.TRANSPORT,
        0,
        anchor=AnchorFrame.RUNTIME_STAGE_START,
        optional=True,
    )
    place = make_stage(StageName.PLACE, 1, anchor=AnchorFrame.PLACEMENT)
    loop = AgentLoop(make_skill(transport, place))

    assert loop.step(AgentFeedback()).action is AgentAction.OBSERVE
    decision = loop.step(AgentFeedback(), skip_optional=True)
    assert decision.action is AgentAction.ADVANCE
    assert "explicitly skipped" in decision.reason
    assert loop.current_stage is place

    with pytest.raises(ValueError, match="cannot skip required stage PLACE"):
        loop.step(AgentFeedback(), skip_optional=True)


def test_feedback_rejects_invalid_signals() -> None:
    with pytest.raises(ValueError, match="grounding_anchor_invalidated must be a bool"):
        AgentFeedback(grounding_anchor_invalidated=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a frozenset"):
        AgentFeedback(satisfied_predicates={"ready"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        AgentFeedback(satisfied_predicates=frozenset({""}))
