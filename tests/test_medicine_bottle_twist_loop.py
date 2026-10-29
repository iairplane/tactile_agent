from __future__ import annotations

import math
from collections import deque
from types import SimpleNamespace

import pytest

from demo2gripper.medicine_bottle_twist_loop import (
    MedicineBottleTwistLoopConfig,
    TwistLoopPhase,
    TwistLoopStatus,
    TwistLoopStopReason,
    run_medicine_bottle_twist_loop,
)
from demo2gripper.tactile_twist_supervisor import TactileTwistAction


class FakeIO:
    def __init__(self, release_answers: list[bool] | None = None) -> None:
        self.calls: list[tuple[str, TwistLoopPhase, float | None]] = []
        self._release_answers = deque(release_answers or [])

    def press_engage(self, *, context) -> None:
        self.calls.append(("press_engage", context.phase, None))

    def execute_twist_step(self, *, angle_rad, context) -> None:
        self.calls.append(("execute_twist_step", context.phase, angle_rad))

    def execute_twist_batch(self, *, angle_rad, step_count, context) -> None:
        self.calls.append(
            ("execute_twist_batch", context.phase, step_count * angle_rad)
        )

    def collect_force_window(self, *, context):
        self.calls.append(("collect_force_window", context.phase, None))
        return object()

    def unload(self, *, context) -> None:
        self.calls.append(("unload", context.phase, None))

    def reset(self, *, angle_rad, context) -> None:
        self.calls.append(("reset", context.phase, angle_rad))

    def regrasp(self, *, context) -> None:
        self.calls.append(("regrasp", context.phase, None))

    def release_verify(self, *, context) -> bool:
        self.calls.append(("release_verify", context.phase, None))
        return self._release_answers.popleft()


class FakeEvaluator:
    def __init__(self, actions: list[TactileTwistAction]) -> None:
        self.actions = deque(actions)
        self.rebaseline_phases: list[TwistLoopPhase] = []

    def evaluate(self, observation, *, context):
        action = self.actions.popleft()
        return SimpleNamespace(evaluation=SimpleNamespace(action=action))

    def rebaseline(self, observation, *, context) -> None:
        self.rebaseline_phases.append(context.phase)


def _config(**changes) -> MedicineBottleTwistLoopConfig:
    values = {
        "twist_step_angle_rad": math.radians(10.0),
        "wrist_travel_limit_rad": math.radians(25.0),
        "max_twist_steps": 10,
        "max_regrasp_cycles": 3,
        "max_press_retries": 2,
        "max_press_confirm_windows": 3,
        "max_unload_windows": 2,
    }
    values.update(changes)
    return MedicineBottleTwistLoopConfig(**values)


def test_pre_press_baseline_and_two_stable_press_windows_precede_twist() -> None:
    io = FakeIO([True])
    evaluator = FakeEvaluator(
        [
            TactileTwistAction.HOLD_FOR_ENGAGEMENT,
            TactileTwistAction.START_TWIST,
            TactileTwistAction.RELEASE_CANDIDATE,
        ]
    )

    result = run_medicine_bottle_twist_loop(io, evaluator, _config())

    assert result.completed
    assert result.status is TwistLoopStatus.RELEASE_VERIFIED
    assert result.twist_step_count == 1
    assert evaluator.rebaseline_phases == [TwistLoopPhase.PRE_PRESS_BASELINE]
    assert [call[:2] for call in io.calls] == [
        ("collect_force_window", TwistLoopPhase.PRE_PRESS_BASELINE),
        ("press_engage", TwistLoopPhase.PRESS_ENGAGE),
        ("collect_force_window", TwistLoopPhase.PRESS_VERIFY),
        ("collect_force_window", TwistLoopPhase.PRESS_VERIFY),
        ("execute_twist_step", TwistLoopPhase.TWIST),
        ("collect_force_window", TwistLoopPhase.TWIST),
        ("release_verify", TwistLoopPhase.TWIST),
    ]


def test_press_more_requests_another_bounded_increment_without_probe_twist() -> None:
    io = FakeIO([True])
    evaluator = FakeEvaluator(
        [
            TactileTwistAction.PRESS_MORE,
            TactileTwistAction.HOLD_FOR_ENGAGEMENT,
            TactileTwistAction.START_TWIST,
            TactileTwistAction.RELEASE_CANDIDATE,
        ]
    )

    result = run_medicine_bottle_twist_loop(io, evaluator, _config())

    assert result.completed
    names = [name for name, _, _ in io.calls]
    assert names.count("press_engage") == 2
    assert names.count("execute_twist_step") == 1
    first_twist = names.index("execute_twist_step")
    assert all(name != "execute_twist_step" for name in names[:first_twist])


def test_geometry_ignores_early_reset_and_resets_before_wrist_overshoot() -> None:
    io = FakeIO([True])
    evaluator = FakeEvaluator(
        [
            TactileTwistAction.START_TWIST,
            TactileTwistAction.WRIST_RESET_REGRASP,
            TactileTwistAction.TWIST_CONTINUE,
            TactileTwistAction.UNLOAD_CONFIRMED,
            TactileTwistAction.START_TWIST,
            TactileTwistAction.RELEASE_CANDIDATE,
        ]
    )

    result = run_medicine_bottle_twist_loop(io, evaluator, _config())

    assert result.completed
    assert result.regrasp_count == 1
    assert result.twist_step_count == 3
    twists = [
        value for name, _, value in io.calls if name == "execute_twist_step"
    ]
    assert twists == pytest.approx([math.radians(10.0)] * 3)
    resets = [value for name, _, value in io.calls if name == "reset"]
    assert resets == pytest.approx([-math.radians(20.0)])
    assert evaluator.rebaseline_phases == [
        TwistLoopPhase.PRE_PRESS_BASELINE,
        TwistLoopPhase.PRE_PRESS_BASELINE,
    ]


def test_pending_release_collects_second_window_before_external_verification() -> None:
    io = FakeIO([True])
    evaluator = FakeEvaluator(
        [
            TactileTwistAction.START_TWIST,
            TactileTwistAction.HOLD_FOR_ENGAGEMENT,
            TactileTwistAction.RELEASE_CANDIDATE,
        ]
    )

    result = run_medicine_bottle_twist_loop(io, evaluator, _config())

    assert result.completed
    twist_windows = [
        call
        for call in io.calls
        if call[:2] == ("collect_force_window", TwistLoopPhase.TWIST)
    ]
    assert len(twist_windows) == 2
    assert [name for name, _, _ in io.calls].count("release_verify") == 1


def test_rejected_release_at_wrist_boundary_never_twists_past_limit() -> None:
    io = FakeIO([False])
    evaluator = FakeEvaluator(
        [
            TactileTwistAction.START_TWIST,
            TactileTwistAction.TWIST_CONTINUE,
            TactileTwistAction.RELEASE_CANDIDATE,
        ]
    )

    result = run_medicine_bottle_twist_loop(
        io,
        evaluator,
        _config(max_regrasp_cycles=0),
    )

    assert not result.completed
    assert result.reason is TwistLoopStopReason.REGRASP_BUDGET_EXHAUSTED
    assert result.twist_step_count == 2
    assert [name for name, _, _ in io.calls].count("execute_twist_step") == 2


def test_press_increment_budget_exhaustion_stops_before_rotation() -> None:
    io = FakeIO()
    evaluator = FakeEvaluator([TactileTwistAction.PRESS_MORE] * 3)

    result = run_medicine_bottle_twist_loop(io, evaluator, _config())

    assert not result.completed
    assert result.reason is TwistLoopStopReason.PRESS_ENGAGEMENT_NOT_VERIFIED
    names = [name for name, _, _ in io.calls]
    assert names.count("press_engage") == 3
    assert "execute_twist_step" not in names


def test_wrist_segment_microsteps_are_batched_before_one_tactile_window() -> None:
    io = FakeIO([True])
    evaluator = FakeEvaluator(
        [TactileTwistAction.START_TWIST, TactileTwistAction.RELEASE_CANDIDATE]
    )

    result = run_medicine_bottle_twist_loop(
        io,
        evaluator,
        _config(
            wrist_travel_limit_rad=math.radians(60.0),
            twist_batch_steps=6,
        ),
    )

    assert result.completed
    assert result.twist_step_count == 6
    batches = [call for call in io.calls if call[0] == "execute_twist_batch"]
    assert batches == [
        ("execute_twist_batch", TwistLoopPhase.TWIST, pytest.approx(math.radians(60)))
    ]
    assert [call[0] for call in io.calls].count("collect_force_window") == 3


@pytest.mark.parametrize("angle_deg", [4.99, 10.01])
def test_twist_microstep_remains_geometrically_bounded(angle_deg: float) -> None:
    with pytest.raises(ValueError, match=r"\[5, 10\]"):
        _config(twist_step_angle_rad=math.radians(angle_deg))
