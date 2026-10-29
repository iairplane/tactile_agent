import numpy as np
import pytest

from demo2gripper.tactile_twist_supervisor import (
    FingerWrenchWindow,
    TactileEvaluationMode,
    TactileTwistAction,
    TactileTwistConfig,
    TactileTwistMemory,
    TactileTwistState,
    build_tactile_twist_baseline,
    evaluate_tactile_twist,
)

AXIS = np.array([0.0, 0.0, 1.0])
AXIS_POINT = np.zeros(3)
NOISE = np.array([-0.02, 0.01, -0.01, 0.02, 0.0])


def config() -> TactileTwistConfig:
    return TactileTwistConfig(
        profile_id="press-turn-cap-test.v1",
        min_unique_samples=4,
        max_age_s=0.20,
        max_pair_skew_s=0.03,
        min_window_span_s=0.06,
        engage_drive_z=5.0,
        retain_drive_z=2.0,
        engage_moment_z=5.0,
        retain_moment_z=2.0,
        engage_finger_z=2.5,
        retain_finger_z=1.0,
        max_imbalance_enter=0.40,
        max_imbalance_retain=0.65,
        release_drive_peak_fraction=0.25,
        release_moment_peak_fraction=0.25,
        release_min_peak_z=6.0,
        release_consecutive_windows=2,
        wrist_reset_fraction=0.90,
    )


def finger_window(
    side: str,
    tangential_level: float,
    *,
    capture_end_s: float,
    sdk_offset: float,
    torque_level: float = 0.0,
    axial_level: float = 0.0,
    repeated_sdk: bool = False,
    valid: bool = True,
) -> FingerWrenchWindow:
    if side == "left":
        position = np.array([0.020, 0.0, 0.0])
        tangent = np.array([0.0, 1.0, 0.0])
        calibration_id = "left-cal"
    elif side == "right":
        position = np.array([-0.020, 0.0, 0.0])
        tangent = np.array([0.0, -1.0, 0.0])
        calibration_id = "right-cal"
    else:
        raise AssertionError(side)
    values = tangential_level + NOISE
    wrench = np.zeros((len(values), 6), dtype=float)
    wrench[:, :3] = values[:, None] * tangent[None, :]
    wrench[:, 2] = axial_level + 0.5 * NOISE
    wrench[:, 5] = torque_level + 0.25 * NOISE
    sdk_time = (
        np.full(len(values), sdk_offset)
        if repeated_sdk
        else sdk_offset + np.arange(len(values), dtype=float)
    )
    capture = np.linspace(capture_end_s - 0.16, capture_end_s, len(values))
    return FingerWrenchWindow(
        calibration_id=calibration_id,
        task_frame_id="cap_task",
        wrench_task=wrench,
        contact_position_task_m=position,
        sdk_time_s=sdk_time,
        capture_monotonic_s=capture,
        valid=np.full(len(values), valid),
    )


def pair(
    left_level: float,
    right_level: float,
    *,
    capture_end_s: float,
    sdk_offset: float,
    left_axial: float = 0.0,
    right_axial: float = 0.0,
) -> tuple[FingerWrenchWindow, FingerWrenchWindow]:
    return (
        finger_window(
            "left",
            left_level,
            capture_end_s=capture_end_s,
            sdk_offset=sdk_offset,
            torque_level=0.2 * left_level,
            axial_level=left_axial,
        ),
        finger_window(
            "right",
            right_level,
            capture_end_s=capture_end_s - 0.005,
            sdk_offset=sdk_offset + 100.0,
            torque_level=0.2 * right_level,
            axial_level=right_axial,
        ),
    )


def baseline():
    left, right = pair(0.0, 0.0, capture_end_s=8.98, sdk_offset=0.0)
    return build_tactile_twist_baseline(
        left,
        right,
        bottle_axis_task=AXIS,
        axis_point_task_m=AXIS_POINT,
        config=config(),
        now_s=9.0,
    )


def confirmed_press_memory():
    reference = baseline()
    memory = None
    for capture_end, sdk_offset in ((9.18, 10.0), (9.38, 20.0)):
        left, right = pair(
            0.0,
            0.0,
            capture_end_s=capture_end,
            sdk_offset=sdk_offset,
            left_axial=1.0,
            right_axial=1.0,
        )
        result = evaluate_tactile_twist(
            left,
            right,
            bottle_axis_task=AXIS,
            axis_point_task_m=AXIS_POINT,
            commanded_twist_sign=1,
            wrist_travel_rad=0.0,
            wrist_travel_limit_rad=1.0,
            baseline=reference,
            config=config(),
            mode=TactileEvaluationMode.PRESS,
            now_s=capture_end + 0.02,
            memory=memory,
        )
        memory = result.memory
    assert result.evaluation.action is TactileTwistAction.START_TWIST
    return reference, memory


def evaluate(
    left_level: float,
    right_level: float,
    *,
    capture_end_s: float = 10.18,
    sdk_offset: float = 30.0,
    wrist_fraction: float = 0.20,
    memory: TactileTwistMemory | None = None,
    axial_level: float = 1.0,
    mode: TactileEvaluationMode = TactileEvaluationMode.TWIST,
):
    reference, confirmed = confirmed_press_memory()
    if memory is None and mode is TactileEvaluationMode.TWIST:
        memory = confirmed
    left, right = pair(
        left_level,
        right_level,
        capture_end_s=capture_end_s,
        sdk_offset=sdk_offset,
        left_axial=axial_level,
        right_axial=axial_level,
    )
    return evaluate_tactile_twist(
        left,
        right,
        bottle_axis_task=AXIS,
        axis_point_task_m=AXIS_POINT,
        commanded_twist_sign=1,
        wrist_travel_rad=wrist_fraction,
        wrist_travel_limit_rad=1.0,
        baseline=reference,
        config=config(),
        mode=mode,
        now_s=capture_end_s + 0.02,
        memory=memory,
    )


def test_axial_press_not_tangential_probe_gates_start_twist() -> None:
    reference = baseline()
    memory = None
    actions = []
    for capture_end, sdk_offset in ((9.18, 10.0), (9.38, 20.0)):
        left, right = pair(
            0.0,
            0.0,
            capture_end_s=capture_end,
            sdk_offset=sdk_offset,
            left_axial=1.0,
            right_axial=1.0,
        )
        result = evaluate_tactile_twist(
            left,
            right,
            bottle_axis_task=AXIS,
            axis_point_task_m=AXIS_POINT,
            commanded_twist_sign=1,
            wrist_travel_rad=0.0,
            wrist_travel_limit_rad=1.0,
            baseline=reference,
            config=config(),
            mode=TactileEvaluationMode.PRESS,
            now_s=capture_end + 0.02,
            memory=memory,
        )
        memory = result.memory
        actions.append(result.evaluation.action)

    assert actions == [
        TactileTwistAction.HOLD_FOR_ENGAGEMENT,
        TactileTwistAction.START_TWIST,
    ]
    assert result.evaluation.state is TactileTwistState.PRESS_ENGAGED
    assert result.memory.press_confirmed
    assert result.evaluation.drive_z < config().engage_drive_z
    assert "axial_press_engaged" in result.evaluation.predicates


def test_insufficient_axial_reaction_requests_another_bounded_press() -> None:
    result = evaluate(
        0.0,
        0.0,
        axial_level=0.02,
        mode=TactileEvaluationMode.PRESS,
        memory=TactileTwistMemory(),
    )

    assert result.evaluation.state is TactileTwistState.PRESS_NOT_ENGAGED
    assert result.evaluation.action is TactileTwistAction.PRESS_MORE


def test_formal_twist_is_blocked_without_prior_press_confirmation() -> None:
    result = evaluate(0.08, 0.08, memory=TactileTwistMemory())

    assert result.evaluation.state is TactileTwistState.PRESS_NOT_ENGAGED
    assert result.evaluation.action is TactileTwistAction.PRESS_MORE


def test_tangential_drive_supervises_formal_twist_after_press_gate() -> None:
    result = evaluate(0.08, 0.08)
    outcome = result.evaluation

    assert outcome.state is TactileTwistState.TWIST_ENGAGED
    assert outcome.action is TactileTwistAction.TWIST_CONTINUE
    assert outcome.signed_tangential_drive == pytest.approx(0.16)
    assert outcome.force_axis_moment == pytest.approx(0.0032)
    assert outcome.wrench_axis_torque == pytest.approx(0.0352)
    assert outcome.load_imbalance == pytest.approx(0.0)
    assert "tangential_engagement_stable" in outcome.predicates
    assert "axial_press_retained" in outcome.predicates
    assert result.memory.engaged_windows == 1


def test_high_stable_tangential_drive_does_not_choose_twist_angle_or_regrasp() -> None:
    result = evaluate(1.0, 1.0, wrist_fraction=0.20)

    assert result.evaluation.state is TactileTwistState.TWIST_ENGAGED
    assert result.evaluation.action is TactileTwistAction.TWIST_CONTINUE
    assert "axial_press_retained" in result.evaluation.predicates
    assert "wrist_regrasp_boundary_reached" not in result.evaluation.predicates


def test_wrist_boundary_requests_regrasp_only_while_tangentially_engaged() -> None:
    engaged = evaluate(0.08, 0.08, wrist_fraction=0.91)
    not_engaged = evaluate(0.0, 0.0, wrist_fraction=0.99)

    assert engaged.evaluation.action is TactileTwistAction.WRIST_RESET_REGRASP
    assert "wrist_regrasp_boundary_reached" in engaged.evaluation.predicates
    assert "axial_press_retained" in engaged.evaluation.predicates
    assert not_engaged.evaluation.state is TactileTwistState.NO_ENGAGEMENT
    assert not_engaged.evaluation.action is TactileTwistAction.HOLD_FOR_ENGAGEMENT


def test_one_finger_load_loss_requests_slip_recovery() -> None:
    result = evaluate(1.0, 0.08)

    assert result.evaluation.state is TactileTwistState.SLIP_OR_IMBALANCED
    assert result.evaluation.action is TactileTwistAction.SLIP_RECOVERY
    assert result.evaluation.load_imbalance > config().max_imbalance_retain


def test_opposing_finger_tangential_direction_requests_slip_recovery() -> None:
    result = evaluate(1.0, -0.8)

    assert result.evaluation.state is TactileTwistState.SLIP_OR_IMBALANCED
    assert result.evaluation.action is TactileTwistAction.SLIP_RECOVERY
    assert "tangential_direction_matches_command" not in result.evaluation.predicates


def test_engagement_hysteresis_keeps_a_weaker_but_coherent_window() -> None:
    first = evaluate(0.06, 0.06)
    second = evaluate(
        0.025,
        0.025,
        capture_end_s=10.38,
        sdk_offset=20.0,
        memory=first.memory,
    )

    assert second.evaluation.drive_z < config().engage_drive_z
    assert second.evaluation.state is TactileTwistState.TWIST_ENGAGED
    assert second.evaluation.action is TactileTwistAction.TWIST_CONTINUE


def test_release_requires_prior_peak_and_two_fresh_low_windows() -> None:
    engaged = evaluate(1.0, 1.0)
    first_low = evaluate(
        0.03,
        0.03,
        capture_end_s=10.38,
        sdk_offset=20.0,
        memory=engaged.memory,
    )
    second_low = evaluate(
        0.02,
        0.02,
        capture_end_s=10.58,
        sdk_offset=30.0,
        memory=first_low.memory,
    )

    assert first_low.evaluation.state is TactileTwistState.NO_ENGAGEMENT
    assert first_low.memory.release_low_windows == 1
    assert second_low.evaluation.state is TactileTwistState.RELEASE_CANDIDATE
    assert second_low.evaluation.action is TactileTwistAction.RELEASE_CANDIDATE
    assert "drive_and_axis_moment_dropped_from_peak" in second_low.evaluation.predicates


def test_low_signal_without_prior_engagement_is_not_release() -> None:
    result = evaluate(0.02, 0.02)

    assert result.evaluation.state is TactileTwistState.NO_ENGAGEMENT
    assert result.evaluation.action is TactileTwistAction.HOLD_FOR_ENGAGEMENT


def test_replayed_window_is_invalid_and_does_not_advance_memory() -> None:
    first = evaluate(1.0, 1.0)
    replay = evaluate(1.0, 1.0, memory=first.memory)

    assert replay.evaluation.state is TactileTwistState.INVALID
    assert replay.evaluation.action is TactileTwistAction.HOLD_INVALID
    assert replay.memory == first.memory
    assert "replayed" in replay.evaluation.reasons[0]


def test_press_window_cannot_reuse_pre_press_baseline_samples() -> None:
    reference = baseline()
    left, right = pair(
        0.0,
        0.0,
        capture_end_s=9.08,
        sdk_offset=10.0,
        left_axial=1.0,
        right_axial=1.0,
    )

    result = evaluate_tactile_twist(
        left,
        right,
        bottle_axis_task=AXIS,
        axis_point_task_m=AXIS_POINT,
        commanded_twist_sign=1,
        wrist_travel_rad=0.0,
        wrist_travel_limit_rad=1.0,
        baseline=reference,
        config=config(),
        mode=TactileEvaluationMode.PRESS,
        now_s=9.10,
    )

    assert result.evaluation.action is TactileTwistAction.HOLD_INVALID
    assert "overlaps the pre-press baseline" in result.evaluation.reasons[0]


def test_second_press_confirmation_window_must_not_overlap_first() -> None:
    reference = baseline()
    first_left, first_right = pair(
        0.0,
        0.0,
        capture_end_s=9.18,
        sdk_offset=10.0,
        left_axial=1.0,
        right_axial=1.0,
    )
    first = evaluate_tactile_twist(
        first_left,
        first_right,
        bottle_axis_task=AXIS,
        axis_point_task_m=AXIS_POINT,
        commanded_twist_sign=1,
        wrist_travel_rad=0.0,
        wrist_travel_limit_rad=1.0,
        baseline=reference,
        config=config(),
        mode=TactileEvaluationMode.PRESS,
        now_s=9.20,
    )
    overlap_left, overlap_right = pair(
        0.0,
        0.0,
        capture_end_s=9.28,
        sdk_offset=20.0,
        left_axial=1.0,
        right_axial=1.0,
    )

    overlap = evaluate_tactile_twist(
        overlap_left,
        overlap_right,
        bottle_axis_task=AXIS,
        axis_point_task_m=AXIS_POINT,
        commanded_twist_sign=1,
        wrist_travel_rad=0.0,
        wrist_travel_limit_rad=1.0,
        baseline=reference,
        config=config(),
        mode=TactileEvaluationMode.PRESS,
        now_s=9.30,
        memory=first.memory,
    )

    assert first.evaluation.action is TactileTwistAction.HOLD_FOR_ENGAGEMENT
    assert overlap.evaluation.action is TactileTwistAction.HOLD_INVALID
    assert "overlaps previously consumed evidence" in overlap.evaluation.reasons[0]
    assert overlap.memory == first.memory


def test_stale_or_cached_force_window_is_invalid() -> None:
    left = finger_window(
        "left", 1.0, capture_end_s=9.0, sdk_offset=10.0
    )
    right = finger_window(
        "right",
        1.0,
        capture_end_s=9.0,
        sdk_offset=20.0,
        repeated_sdk=True,
    )
    result = evaluate_tactile_twist(
        left,
        right,
        bottle_axis_task=AXIS,
        axis_point_task_m=AXIS_POINT,
        commanded_twist_sign=1,
        wrist_travel_rad=0.0,
        wrist_travel_limit_rad=1.0,
        baseline=baseline(),
        config=config(),
        mode=TactileEvaluationMode.TWIST,
        now_s=10.0,
    )

    assert result.evaluation.state is TactileTwistState.INVALID
    assert result.evaluation.action is TactileTwistAction.HOLD_INVALID
    assert any("fresh unique SDK samples" in reason for reason in result.evaluation.reasons)


def test_calibration_change_invalidates_window() -> None:
    left, right = pair(1.0, 1.0, capture_end_s=10.18, sdk_offset=10.0)
    changed = FingerWrenchWindow(
        calibration_id="different-left-cal",
        task_frame_id=left.task_frame_id,
        wrench_task=left.wrench_task,
        contact_position_task_m=left.contact_position_task_m,
        sdk_time_s=left.sdk_time_s,
        capture_monotonic_s=left.capture_monotonic_s,
        valid=left.valid,
    )
    result = evaluate_tactile_twist(
        changed,
        right,
        bottle_axis_task=AXIS,
        axis_point_task_m=AXIS_POINT,
        commanded_twist_sign=1,
        wrist_travel_rad=0.0,
        wrist_travel_limit_rad=1.0,
        baseline=baseline(),
        config=config(),
        mode=TactileEvaluationMode.TWIST,
        now_s=10.20,
    )

    assert result.evaluation.state is TactileTwistState.INVALID
    assert "calibration ids changed" in result.evaluation.reasons[0]


def test_exactly_constant_baseline_is_rejected_for_relative_thresholding() -> None:
    left, right = pair(0.0, 0.0, capture_end_s=9.98, sdk_offset=0.0)
    zero_wrench = np.zeros_like(left.wrench_task)
    left_constant = FingerWrenchWindow(
        calibration_id=left.calibration_id,
        task_frame_id=left.task_frame_id,
        wrench_task=zero_wrench,
        contact_position_task_m=left.contact_position_task_m,
        sdk_time_s=left.sdk_time_s,
        capture_monotonic_s=left.capture_monotonic_s,
        valid=left.valid,
    )
    right_constant = FingerWrenchWindow(
        calibration_id=right.calibration_id,
        task_frame_id=right.task_frame_id,
        wrench_task=zero_wrench,
        contact_position_task_m=right.contact_position_task_m,
        sdk_time_s=right.sdk_time_s,
        capture_monotonic_s=right.capture_monotonic_s,
        valid=right.valid,
    )

    with pytest.raises(ValueError, match="no measurable variation"):
        build_tactile_twist_baseline(
            left_constant,
            right_constant,
            bottle_axis_task=AXIS,
            axis_point_task_m=AXIS_POINT,
            config=config(),
            now_s=10.0,
        )
