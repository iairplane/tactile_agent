"""Plan-only decision loop for stage-based manipulation skills.

The loop sequences a :class:`~demo2gripper.task_skill.TaskSkill` through
observation, target generation/grounding, planning, execution acknowledgement,
and completion verification. For a GRASP stage, grounding resolves the stored
side-pinch relation. For a motion stage, it means asking the runtime agent to
generate a target from current scene and task state. It emits declarative
decisions only: this module does not load a robot driver, send a trajectory, or
command a gripper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .task_skill import (
    AnchorFrame,
    FailurePolicy,
    SkillStage,
    SkillStatus,
    StageName,
    TaskSkill,
)


class AgentState(StrEnum):
    """Closed-loop state within the current skill stage."""

    OBSERVE = "OBSERVE"
    GROUND = "GROUND"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    CANCEL = "CANCEL"
    RECOVERY = "RECOVERY"


class AgentAction(StrEnum):
    """Declarative action requested from the surrounding runtime."""

    OBSERVE = "observe"
    GROUND = "ground"
    PLAN = "plan"
    EXECUTE = "execute"
    ADVANCE = "advance"
    REPLAN = "replan"
    HOLD = "hold"
    REVIEW_REQUIRED = "review_required"
    CANCEL_AND_HOLD = "cancel_and_hold"
    RECOVERY_RETREAT = "recovery_retreat"
    ABORT = "abort"
    COMPLETE = "complete"


@dataclass(frozen=True)
class AgentFeedback:
    """Structured, externally established evidence for one loop step.

    Predicate names are matched against the active stage's declared
    ``preconditions`` and ``completion`` sets. A pending operation is not a
    failure; callers set ``recoverable_failure`` only after one is reported.
    """

    perception_fresh: bool = False
    grounded: bool = False
    plan_valid: bool = False
    execution_succeeded: bool = False
    satisfied_predicates: frozenset[str] = frozenset()
    grounding_anchor_invalidated: bool = False
    recoverable_failure: bool = False
    execution_stopped: bool = False
    recovery_completed: bool = False

    def __post_init__(self) -> None:
        boolean_fields = (
            "perception_fresh",
            "grounded",
            "plan_valid",
            "execution_succeeded",
            "grounding_anchor_invalidated",
            "recoverable_failure",
            "execution_stopped",
            "recovery_completed",
        )
        for name in boolean_fields:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")

        predicates = self.satisfied_predicates
        if not isinstance(predicates, frozenset):
            raise ValueError("satisfied_predicates must be a frozenset")
        if any(not isinstance(item, str) or not item.strip() for item in predicates):
            raise ValueError("satisfied_predicates must contain only non-empty strings")


@dataclass(frozen=True)
class AgentDecision:
    """One plan-only decision and the state it leaves the loop in."""

    action: AgentAction
    state: AgentState
    stage_index: int
    stage_name: StageName
    reason: str
    terminal: bool = False


class AgentLoop:
    """Advance through a task skill without issuing robot commands."""

    _MOTION_ACTIONS = frozenset(
        {
            AgentAction.EXECUTE,
            AgentAction.RECOVERY_RETREAT,
        }
    )

    _INVALIDATABLE_ANCHORS = frozenset(
        {
            AnchorFrame.OBJECT_CENTER_BASE_ALIGNED,
            AnchorFrame.GRASP_TRANSLATION,
            AnchorFrame.OBJECT,
        }
    )

    def __init__(self, skill: TaskSkill) -> None:
        if not isinstance(skill, TaskSkill):
            raise TypeError("skill must be a TaskSkill")
        self.skill = skill
        self.state = AgentState.OBSERVE
        self.stage_index = 0
        self.done = False
        self.aborted = False

    @property
    def current_stage(self) -> SkillStage | None:
        """Return the active stage, or ``None`` after a terminal decision."""

        if self.done or self.aborted:
            return None
        return self.skill.stages[self.stage_index]

    def step(
        self, feedback: AgentFeedback, *, skip_optional: bool = False
    ) -> AgentDecision:
        """Consume current evidence and select the next declarative action.

        ``skip_optional`` is the only way to bypass a stage. It is accepted only
        before work starts; asking to skip a required stage is rejected.
        """

        if not isinstance(feedback, AgentFeedback):
            raise TypeError("feedback must be AgentFeedback")
        if self.done or self.aborted:
            raise RuntimeError("agent loop has already reached a terminal decision")

        stage = self.skill.stages[self.stage_index]
        if self.state is AgentState.CANCEL:
            return self._wait_for_cancel(stage, feedback)
        if self.state is AgentState.RECOVERY:
            return self._wait_for_recovery(stage, feedback)

        if skip_optional:
            if self.state is not AgentState.OBSERVE:
                raise RuntimeError("an optional stage can only be skipped before it starts")
            if not stage.optional:
                raise ValueError(f"cannot skip required stage {stage.name.value}")
            return self._advance(stage, skipped=True)

        if self._anchor_invalidated(stage, feedback):
            return self._handle_anchor_invalidation(stage, feedback)

        if feedback.recoverable_failure:
            return self._handle_failure(stage)

        if self.state is AgentState.OBSERVE:
            if not feedback.perception_fresh:
                return self._decision(
                    stage,
                    AgentAction.OBSERVE,
                    "waiting for fresh perception",
                )
            self.state = AgentState.GROUND
            return self._decision(
                stage,
                AgentAction.GROUND,
                "fresh perception is available; generate or ground the stage target",
            )

        if self.state is AgentState.GROUND:
            if not feedback.perception_fresh:
                self.state = AgentState.OBSERVE
                return self._decision(
                    stage,
                    AgentAction.OBSERVE,
                    "perception became stale before grounding completed",
                )
            if not feedback.grounded:
                return self._decision(
                    stage,
                    AgentAction.GROUND,
                    "waiting for a grounded stage target",
                )
            missing = self._missing(stage.preconditions, feedback)
            if missing:
                return self._decision(
                    stage,
                    AgentAction.HOLD,
                    self._missing_reason("stage preconditions", missing),
                )
            self.state = AgentState.PLAN
            return self._decision(
                stage,
                AgentAction.PLAN,
                "grounding and stage preconditions are satisfied; request a plan",
            )

        if self.state is AgentState.PLAN:
            if not feedback.perception_fresh:
                self.state = AgentState.OBSERVE
                return self._decision(
                    stage,
                    AgentAction.OBSERVE,
                    "perception became stale before a valid plan was established",
                )
            missing = self._missing(stage.preconditions, feedback)
            if missing:
                return self._decision(
                    stage,
                    AgentAction.HOLD,
                    self._missing_reason("stage preconditions", missing),
                )
            if not feedback.plan_valid:
                return self._decision(
                    stage,
                    AgentAction.PLAN,
                    "waiting for a valid plan",
                )
            authorization = self._authorization_decision(stage)
            if authorization is not None:
                return authorization
            self.state = AgentState.EXECUTE
            return self._decision(
                stage,
                AgentAction.EXECUTE,
                "a valid plan is available for the authorized current stage",
            )

        if self.state is AgentState.EXECUTE:
            if not feedback.execution_succeeded:
                return self._decision(
                    stage,
                    AgentAction.HOLD,
                    "execution result is pending",
                )
            self.state = AgentState.VERIFY
            return self._decision(
                stage,
                AgentAction.HOLD,
                "execution succeeded; hold while stage completion is verified",
            )

        missing = self._missing(stage.completion, feedback)
        if missing:
            return self._decision(
                stage,
                AgentAction.HOLD,
                self._missing_reason("stage completion predicates", missing),
            )
        return self._advance(stage, skipped=False)

    @staticmethod
    def _missing(
        required: tuple[str, ...], feedback: AgentFeedback
    ) -> tuple[str, ...]:
        return tuple(sorted(set(required) - feedback.satisfied_predicates))

    @staticmethod
    def _missing_reason(kind: str, missing: tuple[str, ...]) -> str:
        return f"waiting for {kind}: {', '.join(missing)}"

    def _authorization_decision(self, stage: SkillStage) -> AgentDecision | None:
        block = self._authorization_block()
        if block is None:
            return None
        action, reason = block
        return self._decision(stage, action, reason)

    def _authorization_block(self) -> tuple[AgentAction, str] | None:
        """Return the fail-closed result for any action that can start motion."""

        if self.skill.status is SkillStatus.REVIEW_ONLY:
            return (
                AgentAction.REVIEW_REQUIRED,
                "skill status is review_only; hold until an execution-ready skill is supplied",
            )
        if not self.skill.robot_execution_ready:
            return (
                AgentAction.HOLD,
                "robot execution is not ready; execution remains blocked",
            )
        if not self.skill.execution_authorized:
            reason = "skill execution has not been authorized"
            if self.skill.blocking_conditions:
                reason += ": " + ", ".join(self.skill.blocking_conditions)
            return AgentAction.REVIEW_REQUIRED, reason
        if self.skill.blocking_conditions:
            return (
                AgentAction.REVIEW_REQUIRED,
                "skill has blocking conditions: " + ", ".join(self.skill.blocking_conditions),
            )
        return None

    def _anchor_invalidated(
        self, stage: SkillStage, feedback: AgentFeedback
    ) -> bool:
        return (
            feedback.grounding_anchor_invalidated
            and stage.anchor in self._INVALIDATABLE_ANCHORS
            and self.state
            in {
                AgentState.GROUND,
                AgentState.PLAN,
                AgentState.EXECUTE,
                AgentState.VERIFY,
            }
        )

    def _handle_anchor_invalidation(
        self, stage: SkillStage, feedback: AgentFeedback
    ) -> AgentDecision:
        if self.state is AgentState.EXECUTE:
            self.state = AgentState.CANCEL
            return self._decision(
                stage,
                AgentAction.CANCEL_AND_HOLD,
                "grounding anchor changed during execution; cancel and wait for a stop",
            )
        if not feedback.perception_fresh:
            self.state = AgentState.OBSERVE
            return self._decision(
                stage,
                AgentAction.OBSERVE,
                "grounding anchor changed and perception is stale; observe before regrounding",
            )
        self.state = AgentState.GROUND
        return self._decision(
            stage,
            AgentAction.REPLAN,
            "grounding anchor changed; reground before planning again",
        )

    def _wait_for_cancel(
        self, stage: SkillStage, feedback: AgentFeedback
    ) -> AgentDecision:
        if not feedback.execution_stopped:
            return self._decision(
                stage,
                AgentAction.CANCEL_AND_HOLD,
                "cancellation requested; waiting for confirmed execution stop",
            )
        self.state = AgentState.OBSERVE
        return self._decision(
            stage,
            AgentAction.OBSERVE,
            "execution stop confirmed; observe before regrounding",
        )

    def _wait_for_recovery(
        self, stage: SkillStage, feedback: AgentFeedback
    ) -> AgentDecision:
        if not feedback.recovery_completed:
            authorization = self._authorization_decision(stage)
            if authorization is not None:
                return authorization
            return self._decision(
                stage,
                AgentAction.RECOVERY_RETREAT,
                "waiting for the recovery retreat to complete",
            )
        self.state = AgentState.OBSERVE
        return self._decision(
            stage,
            AgentAction.OBSERVE,
            "recovery retreat completed; reobserve before retrying the stage",
        )

    def _handle_failure(self, stage: SkillStage) -> AgentDecision:
        policy = stage.failure_policy
        if policy is FailurePolicy.HOLD:
            return self._decision(
                stage,
                AgentAction.HOLD,
                "stage failure reported; failure policy requires holding position",
            )
        if policy is FailurePolicy.REOBSERVE_REPLAN:
            self.state = AgentState.OBSERVE
            return self._decision(
                stage,
                AgentAction.OBSERVE,
                "stage failure reported; reobserve before regrounding and replanning",
            )
        if policy is FailurePolicy.RETREAT_REOBSERVE_RETRY:
            authorization = self._authorization_decision(stage)
            if authorization is not None:
                return authorization
            self.state = AgentState.RECOVERY
            return self._decision(
                stage,
                AgentAction.RECOVERY_RETREAT,
                "stage failure reported; request a recovery retreat before retrying",
            )

        self.aborted = True
        return self._decision(
            stage,
            AgentAction.ABORT,
            "stage failure reported; failure policy requires aborting the skill",
            terminal=True,
        )

    def _advance(self, stage: SkillStage, *, skipped: bool) -> AgentDecision:
        completed_index = self.stage_index
        if self.stage_index + 1 == len(self.skill.stages):
            self.done = True
            reason = (
                "final optional stage was explicitly skipped; skill is complete"
                if skipped
                else "final stage completion predicates are satisfied; skill is complete"
            )
            return self._decision(
                stage,
                AgentAction.COMPLETE,
                reason,
                terminal=True,
                stage_index=completed_index,
            )

        self.stage_index += 1
        self.state = AgentState.OBSERVE
        reason = (
            "optional stage explicitly skipped; advance to the next stage"
            if skipped
            else "stage completion predicates are satisfied; advance to the next stage"
        )
        return self._decision(
            stage,
            AgentAction.ADVANCE,
            reason,
            stage_index=completed_index,
        )

    def _decision(
        self,
        stage: SkillStage,
        action: AgentAction,
        reason: str,
        *,
        terminal: bool = False,
        stage_index: int | None = None,
    ) -> AgentDecision:
        if action in self._MOTION_ACTIONS and self._authorization_block() is not None:
            raise RuntimeError(
                f"internal error: unauthorized motion action {action.value} escaped its gate"
            )
        return AgentDecision(
            action=action,
            state=self.state,
            stage_index=self.stage_index if stage_index is None else stage_index,
            stage_name=stage.name,
            reason=reason,
            terminal=terminal,
        )
