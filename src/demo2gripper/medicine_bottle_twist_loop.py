"""Deterministic microstep loop for a press-and-turn medicine-bottle cap.

The module owns no hardware.  A hardware runner supplies a
``MedicineBottleTwistIO`` implementation, while tests supply a fake with the
same atomic operations.  Every twist command is followed by fresh tactile
evidence and a decision from :mod:`demo2gripper.tactile_twist_supervisor`.

Every segment begins with a pre-press baseline and an axial tactile gate::

    pre-press baseline -> bounded press increment(s)
                       -> verify stable bilateral axial reaction -> formal twist

    unload -> verify tangential unloading -> reset wrist -> regrasp
           -> new pre-press baseline -> press gate -> formal twist

The configured step, wrist-reset, and regrasp budgets bound the episode.  A
verified tactile release candidate completes the loop immediately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from .tactile_twist_supervisor import (
    FingerWrenchWindow,
    TactileEvaluationMode,
    TactileTwistAction,
    TactileTwistBaseline,
    TactileTwistConfig,
    TactileTwistMemory,
    TactileTwistResult,
    build_tactile_twist_baseline,
    evaluate_tactile_twist,
)

MIN_TWIST_MICROSTEP_RAD = math.radians(5.0)
MAX_TWIST_MICROSTEP_RAD = math.radians(10.0)
MAX_REGRASP_CYCLES = 64
MAX_PRESS_RETRIES = 3


class TwistLoopPhase(StrEnum):
    PRE_PRESS_BASELINE = "pre_press_baseline"
    PRESS_ENGAGE = "press_engage"
    PRESS_VERIFY = "press_verify"
    TWIST = "twist"
    UNLOAD_CHECK = "unload_check"


class TwistLoopStatus(StrEnum):
    RELEASE_VERIFIED = "release_verified"
    STOPPED = "stopped"


class TwistLoopStopReason(StrEnum):
    RELEASE_VERIFIED = "release_verified"
    TWIST_STEP_BUDGET_EXHAUSTED = "twist_step_budget_exhausted"
    REGRASP_BUDGET_EXHAUSTED = "regrasp_budget_exhausted"
    INVALID_TACTILE_EVIDENCE = "invalid_tactile_evidence"
    ENGAGEMENT_NOT_RECOVERED = "engagement_not_recovered"
    SLIP_OR_IMBALANCE = "slip_or_imbalance"
    UNLOAD_NOT_VERIFIED = "unload_not_verified"
    PRESS_ENGAGEMENT_NOT_VERIFIED = "press_engagement_not_verified"


@dataclass(frozen=True)
class MedicineBottleTwistLoopConfig:
    """Bounds for one deterministic cap-removal attempt."""

    twist_step_angle_rad: float
    wrist_travel_limit_rad: float
    max_twist_steps: int
    max_regrasp_cycles: int = MAX_REGRASP_CYCLES
    max_press_retries: int = 2
    max_press_confirm_windows: int = 3
    max_unload_windows: int = 3
    twist_batch_steps: int = 1

    def __post_init__(self) -> None:
        angle = float(self.twist_step_angle_rad)
        limit = float(self.wrist_travel_limit_rad)
        if not math.isfinite(angle) or not (
            MIN_TWIST_MICROSTEP_RAD - 1e-12
            <= abs(angle)
            <= MAX_TWIST_MICROSTEP_RAD + 1e-12
        ):
            raise ValueError("absolute twist_step_angle_rad must lie in [5, 10] degrees")
        if not math.isfinite(limit) or limit < abs(angle):
            raise ValueError(
                "wrist_travel_limit_rad must be finite and at least one microstep"
            )
        for name in (
            "max_twist_steps",
            "max_press_confirm_windows",
            "max_unload_windows",
            "twist_batch_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        retries = self.max_press_retries
        if (
            isinstance(retries, bool)
            or not isinstance(retries, int)
            or not 0 <= retries <= MAX_PRESS_RETRIES
        ):
            raise ValueError(
                f"max_press_retries must be an integer in [0, {MAX_PRESS_RETRIES}]"
            )
        cycles = self.max_regrasp_cycles
        if (
            isinstance(cycles, bool)
            or not isinstance(cycles, int)
            or not 0 <= cycles <= MAX_REGRASP_CYCLES
        ):
            raise ValueError(
                f"max_regrasp_cycles must be an integer in [0, {MAX_REGRASP_CYCLES}]"
            )
        if abs(angle) * self.twist_batch_steps > limit + 1e-12:
            raise ValueError(
                "twist_batch_steps times the microstep angle cannot exceed "
                "wrist_travel_limit_rad"
            )


@dataclass(frozen=True)
class TwistLoopContext:
    """State attached to every port and evaluator call."""

    phase: TwistLoopPhase
    twist_step_count: int
    segment_step_count: int
    regrasp_count: int
    press_retry_count: int
    segment_wrist_travel_rad: float
    total_signed_twist_rad: float


@dataclass(frozen=True)
class TactileForceObservation:
    left: FingerWrenchWindow
    right: FingerWrenchWindow
    now_monotonic_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.left, FingerWrenchWindow) or not isinstance(
            self.right, FingerWrenchWindow
        ):
            raise TypeError("left and right must be FingerWrenchWindow instances")
        now = float(self.now_monotonic_s)
        if not math.isfinite(now):
            raise ValueError("now_monotonic_s must be finite")
        object.__setattr__(self, "now_monotonic_s", now)


class MedicineBottleTwistIO(Protocol):
    """The atomic operations that a hardware runner must provide.

    Implementations should raise when a command was rejected or its measured
    completion predicate failed.  This pure loop never retries an unconfirmed
    physical command.
    """

    def press_engage(self, *, context: TwistLoopContext) -> None: ...

    def execute_twist_step(
        self, *, angle_rad: float, context: TwistLoopContext
    ) -> None: ...

    def execute_twist_batch(
        self, *, angle_rad: float, step_count: int, context: TwistLoopContext
    ) -> None: ...

    def collect_force_window(
        self, *, context: TwistLoopContext
    ) -> TactileForceObservation: ...

    def unload(self, *, context: TwistLoopContext) -> None: ...

    def reset(self, *, angle_rad: float, context: TwistLoopContext) -> None: ...

    def regrasp(self, *, context: TwistLoopContext) -> None: ...

    def release_verify(self, *, context: TwistLoopContext) -> bool: ...


class TwistEvaluator(Protocol):
    def evaluate(
        self, observation: TactileForceObservation, *, context: TwistLoopContext
    ) -> TactileTwistResult: ...

    def rebaseline(
        self, observation: TactileForceObservation, *, context: TwistLoopContext
    ) -> None: ...


class TactileTwistSupervisorEvaluator:
    """Stateful adapter around the pure tactile twist supervisor.

    The adapter updates only supervisor memory and its per-segment baseline.  It
    has no reference to the arm, gripper, ROS, USB, or an SDK device.
    """

    def __init__(
        self,
        *,
        bottle_axis_task: np.ndarray,
        axis_point_task_m: np.ndarray,
        baseline: TactileTwistBaseline | None,
        config: TactileTwistConfig,
        commanded_twist_sign: int,
        wrist_travel_limit_rad: float,
    ) -> None:
        if commanded_twist_sign not in (-1, 1):
            raise ValueError("commanded_twist_sign must be -1 or +1")
        limit = float(wrist_travel_limit_rad)
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("wrist_travel_limit_rad must be positive and finite")
        axis = np.array(bottle_axis_task, dtype=float, copy=True)
        point = np.array(axis_point_task_m, dtype=float, copy=True)
        if axis.shape != (3,) or not np.isfinite(axis).all():
            raise ValueError("bottle_axis_task must contain three finite values")
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            raise ValueError("bottle_axis_task must be non-zero")
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("axis_point_task_m must contain three finite values")
        axis = axis / norm
        axis.setflags(write=False)
        point.setflags(write=False)
        self._axis = axis
        self._point = point
        self._baseline = baseline
        self._config = config
        self._sign = commanded_twist_sign
        self._wrist_limit = limit
        self._memory = TactileTwistMemory()

    @property
    def baseline(self) -> TactileTwistBaseline:
        if self._baseline is None:
            raise RuntimeError("tactile twist baseline has not been established")
        return self._baseline

    @property
    def memory(self) -> TactileTwistMemory:
        return self._memory

    def evaluate(
        self, observation: TactileForceObservation, *, context: TwistLoopContext
    ) -> TactileTwistResult:
        if self._baseline is None:
            raise RuntimeError("rebaseline() must run before tactile evaluation")
        result = evaluate_tactile_twist(
            observation.left,
            observation.right,
            bottle_axis_task=self._axis,
            axis_point_task_m=self._point,
            commanded_twist_sign=self._sign,
            wrist_travel_rad=context.segment_wrist_travel_rad,
            wrist_travel_limit_rad=self._wrist_limit,
            baseline=self._baseline,
            config=self._config,
            mode=(
                TactileEvaluationMode.PRESS
                if context.phase is TwistLoopPhase.PRESS_VERIFY
                else TactileEvaluationMode.UNLOAD
                if context.phase is TwistLoopPhase.UNLOAD_CHECK
                else TactileEvaluationMode.TWIST
            ),
            now_s=observation.now_monotonic_s,
            memory=self._memory,
        )
        self._memory = result.memory
        return result

    def rebaseline(
        self, observation: TactileForceObservation, *, context: TwistLoopContext
    ) -> None:
        if context.phase is not TwistLoopPhase.PRE_PRESS_BASELINE:
            raise ValueError("a new tactile baseline is allowed only before axial press")
        self._baseline = build_tactile_twist_baseline(
            observation.left,
            observation.right,
            bottle_axis_task=self._axis,
            axis_point_task_m=self._point,
            config=self._config,
            now_s=observation.now_monotonic_s,
        )
        self._memory = TactileTwistMemory()


@dataclass(frozen=True)
class TwistLoopEvent:
    operation: str
    context: TwistLoopContext
    angle_rad: float | None = None
    tactile_action: TactileTwistAction | None = None
    release_verified: bool | None = None


@dataclass(frozen=True)
class MedicineBottleTwistLoopResult:
    status: TwistLoopStatus
    reason: TwistLoopStopReason
    twist_step_count: int
    regrasp_count: int
    total_signed_twist_rad: float
    release_verify_attempts: int
    events: tuple[TwistLoopEvent, ...]
    last_evaluation: TactileTwistResult | None

    @property
    def completed(self) -> bool:
        return self.status is TwistLoopStatus.RELEASE_VERIFIED


def _context(
    phase: TwistLoopPhase,
    *,
    steps: int,
    segment_steps: int,
    regrasps: int,
    press_retries: int,
    segment_travel: float,
    total_twist: float,
) -> TwistLoopContext:
    return TwistLoopContext(
        phase=phase,
        twist_step_count=steps,
        segment_step_count=segment_steps,
        regrasp_count=regrasps,
        press_retry_count=press_retries,
        segment_wrist_travel_rad=segment_travel,
        total_signed_twist_rad=total_twist,
    )


def _result(
    *,
    status: TwistLoopStatus,
    reason: TwistLoopStopReason,
    steps: int,
    regrasps: int,
    total_twist: float,
    release_attempts: int,
    events: list[TwistLoopEvent],
    last_evaluation: TactileTwistResult | None,
) -> MedicineBottleTwistLoopResult:
    return MedicineBottleTwistLoopResult(
        status=status,
        reason=reason,
        twist_step_count=steps,
        regrasp_count=regrasps,
        total_signed_twist_rad=total_twist,
        release_verify_attempts=release_attempts,
        events=tuple(events),
        last_evaluation=last_evaluation,
    )


def run_medicine_bottle_twist_loop(
    io: MedicineBottleTwistIO,
    evaluator: TwistEvaluator,
    config: MedicineBottleTwistLoopConfig,
) -> MedicineBottleTwistLoopResult:
    """Run one bounded, tactile-supervised press-and-turn episode.

    Each segment starts with a stationary pre-press baseline.  One or more
    bounded axial press increments must produce ``START_TWIST`` before any
    rotation begins.  During formal rotation tactile evidence supervises
    stability, imbalance, wrist reset, and release; it does not generate the
    commanded rotation angle.  The function is bounded by both
    ``max_twist_steps`` and the configured regrasp cap.
    """

    if not isinstance(config, MedicineBottleTwistLoopConfig):
        raise TypeError("config must be MedicineBottleTwistLoopConfig")
    events: list[TwistLoopEvent] = []
    steps = 0
    segment_steps = 0
    regrasps = 0
    press_retries = 0
    segment_travel = 0.0
    total_twist = 0.0
    release_attempts = 0
    last: TactileTwistResult | None = None

    def stop(reason: TwistLoopStopReason) -> MedicineBottleTwistLoopResult:
        return _result(
            status=TwistLoopStatus.STOPPED,
            reason=reason,
            steps=steps,
            regrasps=regrasps,
            total_twist=total_twist,
            release_attempts=release_attempts,
            events=events,
            last_evaluation=last,
        )

    def current_context(phase: TwistLoopPhase) -> TwistLoopContext:
        return _context(
            phase,
            steps=steps,
            segment_steps=segment_steps,
            regrasps=regrasps,
            press_retries=press_retries,
            segment_travel=segment_travel,
            total_twist=total_twist,
        )

    def execute_microsteps(phase: TwistLoopPhase) -> TwistLoopContext:
        nonlocal steps, segment_steps, segment_travel, total_twist
        angle_rad = config.twist_step_angle_rad
        remaining_budget = config.max_twist_steps - steps
        remaining_wrist = max(
            0.0, config.wrist_travel_limit_rad - segment_travel
        )
        wrist_steps = max(
            1,
            int(math.floor((remaining_wrist + 1e-12) / abs(angle_rad))),
        )
        batch_steps = min(
            config.twist_batch_steps,
            remaining_budget,
            wrist_steps,
        )
        before = current_context(phase)
        if batch_steps == 1:
            io.execute_twist_step(angle_rad=angle_rad, context=before)
            operation = "execute_twist_step"
        else:
            execute_batch = getattr(io, "execute_twist_batch", None)
            if execute_batch is None:
                for offset in range(batch_steps):
                    fallback = _context(
                        phase,
                        steps=steps + offset,
                        segment_steps=segment_steps + offset,
                        regrasps=regrasps,
                        press_retries=press_retries,
                        segment_travel=segment_travel + offset * abs(angle_rad),
                        total_twist=total_twist + offset * angle_rad,
                    )
                    io.execute_twist_step(angle_rad=angle_rad, context=fallback)
            else:
                execute_batch(
                    angle_rad=angle_rad,
                    step_count=batch_steps,
                    context=before,
                )
            operation = "execute_twist_batch"
        steps += batch_steps
        segment_steps += batch_steps
        segment_travel += batch_steps * abs(angle_rad)
        total_twist += batch_steps * angle_rad
        after = current_context(phase)
        events.append(
            TwistLoopEvent(
                operation=operation,
                context=after,
                angle_rad=batch_steps * angle_rad,
            )
        )
        return after

    def collect_decision(context: TwistLoopContext) -> TactileTwistResult:
        """Evaluate exactly one fresh post-command tactile window."""

        nonlocal last
        observation = io.collect_force_window(context=context)
        last = evaluator.evaluate(observation, context=context)
        events.append(
            TwistLoopEvent(
                operation="collect_force_window",
                context=context,
                tactile_action=last.evaluation.action,
            )
        )
        return last

    needs_press_gate = True
    while steps < config.max_twist_steps:
        if needs_press_gate:
            baseline_context = current_context(TwistLoopPhase.PRE_PRESS_BASELINE)
            baseline_observation = io.collect_force_window(context=baseline_context)
            events.append(
                TwistLoopEvent(
                    operation="collect_force_window", context=baseline_context
                )
            )
            evaluator.rebaseline(baseline_observation, context=baseline_context)
            events.append(
                TwistLoopEvent(operation="rebaseline", context=baseline_context)
            )
            press_verified = False
            for press_retries in range(config.max_press_retries + 1):
                press_context = current_context(TwistLoopPhase.PRESS_ENGAGE)
                io.press_engage(context=press_context)
                events.append(
                    TwistLoopEvent(operation="press_engage", context=press_context)
                )
                verify_context = current_context(TwistLoopPhase.PRESS_VERIFY)
                request_more_press = False
                for _ in range(config.max_press_confirm_windows):
                    decision = collect_decision(verify_context)
                    action = decision.evaluation.action
                    if action is TactileTwistAction.HOLD_INVALID:
                        return stop(TwistLoopStopReason.INVALID_TACTILE_EVIDENCE)
                    if action is TactileTwistAction.SLIP_RECOVERY:
                        return stop(TwistLoopStopReason.SLIP_OR_IMBALANCE)
                    if action is TactileTwistAction.START_TWIST:
                        press_verified = True
                        events.append(
                            TwistLoopEvent(
                                operation="press_engagement_verified",
                                context=verify_context,
                                tactile_action=action,
                            )
                        )
                        break
                    if action is TactileTwistAction.PRESS_MORE:
                        request_more_press = True
                        break
                    if action is not TactileTwistAction.HOLD_FOR_ENGAGEMENT:
                        return stop(TwistLoopStopReason.PRESS_ENGAGEMENT_NOT_VERIFIED)
                if press_verified:
                    break
                if not request_more_press:
                    return stop(TwistLoopStopReason.PRESS_ENGAGEMENT_NOT_VERIFIED)
            if not press_verified:
                return stop(TwistLoopStopReason.PRESS_ENGAGEMENT_NOT_VERIFIED)
            needs_press_gate = False

        context = execute_microsteps(TwistLoopPhase.TWIST)
        decision = collect_decision(context)
        last = decision
        action = last.evaluation.action
        if action is TactileTwistAction.HOLD_INVALID:
            return stop(TwistLoopStopReason.INVALID_TACTILE_EVIDENCE)
        if action is TactileTwistAction.SLIP_RECOVERY:
            return stop(TwistLoopStopReason.SLIP_OR_IMBALANCE)
        if action is TactileTwistAction.HOLD_FOR_ENGAGEMENT:
            for _ in range(config.max_press_confirm_windows - 1):
                decision = collect_decision(context)
                action = decision.evaluation.action
                if action is not TactileTwistAction.HOLD_FOR_ENGAGEMENT:
                    break
            if action is TactileTwistAction.HOLD_FOR_ENGAGEMENT:
                return stop(TwistLoopStopReason.ENGAGEMENT_NOT_RECOVERED)
            if action is TactileTwistAction.HOLD_INVALID:
                return stop(TwistLoopStopReason.INVALID_TACTILE_EVIDENCE)
            if action is TactileTwistAction.SLIP_RECOVERY:
                return stop(TwistLoopStopReason.SLIP_OR_IMBALANCE)
        if action is TactileTwistAction.RELEASE_CANDIDATE:
            verified = io.release_verify(context=context)
            if not isinstance(verified, bool):
                raise TypeError("release_verify must return bool")
            release_attempts += 1
            events.append(
                TwistLoopEvent(
                    operation="release_verify",
                    context=context,
                    tactile_action=action,
                    release_verified=verified,
                )
            )
            if verified:
                return _result(
                    status=TwistLoopStatus.RELEASE_VERIFIED,
                    reason=TwistLoopStopReason.RELEASE_VERIFIED,
                    steps=steps,
                    regrasps=regrasps,
                    total_twist=total_twist,
                    release_attempts=release_attempts,
                    events=events,
                    last_evaluation=last,
                )
            # A rejected release check does not authorize another step past a
            # wrist boundary.  Fall through as ordinary continued twist so
            # the geometric boundary below either performs a bounded regrasp
            # or terminates when that budget is exhausted.
            action = TactileTwistAction.TWIST_CONTINUE
        if (
            steps >= config.max_twist_steps
            and action is TactileTwistAction.TWIST_CONTINUE
        ):
            return stop(TwistLoopStopReason.TWIST_STEP_BUDGET_EXHAUSTED)
        geometry_boundary = (
            segment_travel + abs(config.twist_step_angle_rad)
            > config.wrist_travel_limit_rad + 1e-12
        )
        if (
            action is TactileTwistAction.WRIST_RESET_REGRASP
            and not geometry_boundary
        ):
            action = TactileTwistAction.TWIST_CONTINUE
        elif action is TactileTwistAction.TWIST_CONTINUE and geometry_boundary:
            action = TactileTwistAction.WRIST_RESET_REGRASP
        if action is TactileTwistAction.TWIST_CONTINUE:
            continue
        if action is not TactileTwistAction.WRIST_RESET_REGRASP:
            raise AssertionError(f"unhandled tactile action: {action}")

        if regrasps >= config.max_regrasp_cycles:
            return stop(TwistLoopStopReason.REGRASP_BUDGET_EXHAUSTED)

        unload_context = current_context(TwistLoopPhase.UNLOAD_CHECK)
        io.unload(context=unload_context)
        events.append(TwistLoopEvent(operation="unload", context=unload_context))

        unloaded = False
        for _ in range(config.max_unload_windows):
            observation = io.collect_force_window(context=unload_context)
            last = evaluator.evaluate(observation, context=unload_context)
            unload_action = last.evaluation.action
            events.append(
                TwistLoopEvent(
                    operation="collect_force_window",
                    context=unload_context,
                    tactile_action=unload_action,
                )
            )
            if unload_action is TactileTwistAction.HOLD_INVALID:
                return stop(TwistLoopStopReason.INVALID_TACTILE_EVIDENCE)
            if unload_action is TactileTwistAction.SLIP_RECOVERY:
                return stop(TwistLoopStopReason.SLIP_OR_IMBALANCE)
            if unload_action is TactileTwistAction.UNLOAD_CONFIRMED:
                unloaded = True
                break
        if not unloaded:
            return stop(TwistLoopStopReason.UNLOAD_NOT_VERIFIED)

        reset_angle = -math.copysign(segment_travel, config.twist_step_angle_rad)
        io.reset(angle_rad=reset_angle, context=unload_context)
        events.append(
            TwistLoopEvent(
                operation="reset", context=unload_context, angle_rad=reset_angle
            )
        )
        io.regrasp(context=unload_context)
        regrasps += 1
        segment_steps = 0
        segment_travel = 0.0
        press_retries = 0
        regrasp_context = current_context(TwistLoopPhase.PRE_PRESS_BASELINE)
        events.append(TwistLoopEvent(operation="regrasp", context=regrasp_context))
        last = None
        needs_press_gate = True

    return stop(TwistLoopStopReason.TWIST_STEP_BUDGET_EXHAUSTED)


__all__ = [
    "MAX_REGRASP_CYCLES",
    "MAX_PRESS_RETRIES",
    "MAX_TWIST_MICROSTEP_RAD",
    "MIN_TWIST_MICROSTEP_RAD",
    "MedicineBottleTwistIO",
    "MedicineBottleTwistLoopConfig",
    "MedicineBottleTwistLoopResult",
    "TactileForceObservation",
    "TactileTwistSupervisorEvaluator",
    "TwistEvaluator",
    "TwistLoopContext",
    "TwistLoopEvent",
    "TwistLoopPhase",
    "TwistLoopStatus",
    "TwistLoopStopReason",
    "run_medicine_bottle_twist_loop",
]
