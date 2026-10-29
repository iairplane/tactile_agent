from __future__ import annotations

import math

import numpy as np

from demo2gripper.medicine_bottle_tactile_presence import (
    MedicineBottlePresenceConfig,
    TactilePresenceSupervisorEvaluator,
)
from demo2gripper.medicine_bottle_twist_loop import (
    TactileForceObservation,
    TwistLoopContext,
    TwistLoopPhase,
)
from demo2gripper.tactile_twist_supervisor import (
    FingerWrenchWindow,
    TactileTwistAction,
)


def _context(
    phase: TwistLoopPhase,
    *,
    regrasps: int = 0,
    wrist_fraction: float = 0.0,
    total_twist_deg: float = 0.0,
) -> TwistLoopContext:
    return TwistLoopContext(
        phase=phase,
        twist_step_count=0,
        segment_step_count=0,
        regrasp_count=regrasps,
        press_retry_count=0,
        segment_wrist_travel_rad=wrist_fraction * math.radians(45),
        total_signed_twist_rad=math.radians(total_twist_deg),
    )


def _observation(left_fz: float, right_fz: float, *, start: float):
    count = 8
    host = start + 0.01 * np.arange(count)

    def finger(side: str, value: float) -> FingerWrenchWindow:
        wrench = np.zeros((count, 6), dtype=float)
        wrench[:, 2] = value
        return FingerWrenchWindow(
            calibration_id=f"{side}-cal",
            task_frame_id="raw-fz-presence",
            wrench_task=wrench,
            contact_position_task_m=np.array([0.02, 0.0, 0.0]),
            sdk_time_s=1000.0 * host,
            capture_monotonic_s=host,
            valid=np.ones(count, dtype=bool),
        )

    return TactileForceObservation(
        finger("left", left_fz),
        finger("right", right_fz),
        float(host[-1] + 0.005),
    )


def _evaluator() -> TactilePresenceSupervisorEvaluator:
    return TactilePresenceSupervisorEvaluator(
        config=MedicineBottlePresenceConfig(
            profile_id="presence",
            min_unique_samples=4,
            max_age_s=0.2,
            max_pair_skew_s=0.05,
            min_window_span_s=0.06,
            contact_retain_fz=0.03,
            press_engage_fz=0.02,
            press_retain_fz=0.01,
            unload_return_fz=0.02,
            release_min_total_twist_rad=math.radians(360),
            release_resistance_peak_fraction=0.35,
            release_min_peak_fz=0.02,
            release_required_consecutive_windows=2,
        ),
        wrist_travel_limit_rad=math.radians(45),
    )


def test_presence_mode_accepts_one_pad_press_and_keeps_fixed_twist() -> None:
    evaluator = _evaluator()
    evaluator.rebaseline(
        _observation(0.30, 0.0, start=1.0),
        context=_context(TwistLoopPhase.PRE_PRESS_BASELINE),
    )

    press = evaluator.evaluate(
        _observation(0.34, 0.0, start=2.0),
        context=_context(TwistLoopPhase.PRESS_VERIFY),
    )
    twist = evaluator.evaluate(
        _observation(0.28, 0.0, start=3.0),
        context=_context(TwistLoopPhase.TWIST, wrist_fraction=0.5),
    )

    assert press.evaluation.action is TactileTwistAction.START_TWIST
    assert twist.evaluation.action is TactileTwistAction.TWIST_CONTINUE
    assert "any_pad_tactile_presence_retained" in twist.evaluation.predicates


def test_presence_mode_stops_only_after_both_pads_lose_presence() -> None:
    evaluator = _evaluator()
    evaluator.rebaseline(
        _observation(0.30, 0.0, start=1.0),
        context=_context(TwistLoopPhase.PRE_PRESS_BASELINE),
    )
    evaluator.evaluate(
        _observation(0.34, 0.0, start=2.0),
        context=_context(TwistLoopPhase.PRESS_VERIFY),
    )

    lost = evaluator.evaluate(
        _observation(0.0, 0.0, start=3.0),
        context=_context(TwistLoopPhase.TWIST),
    )
    assert lost.evaluation.action is TactileTwistAction.SLIP_RECOVERY


def test_presence_mode_uses_proxy_peak_drop_after_first_revolution_for_release() -> None:
    evaluator = _evaluator()
    evaluator.rebaseline(
        _observation(0.30, 0.0, start=1.0),
        context=_context(TwistLoopPhase.PRE_PRESS_BASELINE),
    )
    evaluator.evaluate(
        _observation(0.40, 0.0, start=2.0),
        context=_context(TwistLoopPhase.PRESS_VERIFY),
    )

    before_upfeed = evaluator.evaluate(
        _observation(0.50, 0.0, start=3.0),
        context=_context(
            TwistLoopPhase.TWIST,
            regrasps=5,
            wrist_fraction=0.5,
            total_twist_deg=350.0,
        ),
    )
    upfeed_peak = evaluator.evaluate(
        _observation(0.50, 0.0, start=4.0),
        context=_context(
            TwistLoopPhase.TWIST,
            regrasps=6,
            wrist_fraction=0.5,
            total_twist_deg=370.0,
        ),
    )
    first_low = evaluator.evaluate(
        _observation(0.34, 0.0, start=5.0),
        context=_context(
            TwistLoopPhase.TWIST,
            regrasps=6,
            wrist_fraction=0.5,
            total_twist_deg=380.0,
        ),
    )
    result = evaluator.evaluate(
        _observation(0.34, 0.0, start=6.0),
        context=_context(
            TwistLoopPhase.TWIST,
            regrasps=6,
            wrist_fraction=0.5,
            total_twist_deg=380.0,
        ),
    )

    assert before_upfeed.evaluation.action is TactileTwistAction.TWIST_CONTINUE
    assert upfeed_peak.evaluation.action is TactileTwistAction.TWIST_CONTINUE
    assert first_low.evaluation.action is TactileTwistAction.HOLD_FOR_ENGAGEMENT
    assert result.evaluation.action is TactileTwistAction.RELEASE_CANDIDATE
    assert "thread_release_proxy_drop_confirmed" in result.evaluation.predicates


def test_regrasp_count_alone_never_requests_release_before_one_turn() -> None:
    evaluator = _evaluator()
    evaluator.rebaseline(
        _observation(0.30, 0.0, start=1.0),
        context=_context(TwistLoopPhase.PRE_PRESS_BASELINE),
    )
    evaluator.evaluate(
        _observation(0.40, 0.0, start=2.0),
        context=_context(TwistLoopPhase.PRESS_VERIFY),
    )

    result = evaluator.evaluate(
        _observation(0.50, 0.0, start=3.0),
        context=_context(
            TwistLoopPhase.TWIST,
            regrasps=17,
            wrist_fraction=1.0,
            total_twist_deg=180.0,
        ),
    )

    assert result.evaluation.action is TactileTwistAction.WRIST_RESET_REGRASP


def test_unload_accepts_fresh_window_even_while_gel_fz_is_recovering() -> None:
    evaluator = _evaluator()
    evaluator.rebaseline(
        _observation(0.30, 0.0, start=1.0),
        context=_context(TwistLoopPhase.PRE_PRESS_BASELINE),
    )

    result = evaluator.evaluate(
        _observation(0.20, 0.0, start=2.0),
        context=_context(TwistLoopPhase.UNLOAD_CHECK),
    )

    assert result.evaluation.action is TactileTwistAction.UNLOAD_CONFIRMED
    assert "fresh_window_after_successful_gripper_open" in result.evaluation.predicates
