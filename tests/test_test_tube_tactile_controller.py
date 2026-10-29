from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from demo2gripper.gf220_live import GF220ClassifierWindows
from demo2gripper.gf220_wrench_adapter import AdaptedGF220Wrenches
from demo2gripper.tactile_contact_classifier import FingerRimClassifierWindow
from demo2gripper.tactile_rim_probe import TimedArrayWindow
from demo2gripper.tactile_twist_supervisor import FingerWrenchWindow
from demo2gripper.test_tube_contact_bootstrap import (
    TestTubeContactBootstrapConfig as ContactBootstrapConfig,
)
from demo2gripper.test_tube_contact_bootstrap import (
    TestTubeContactDecision as ContactDecision,
)
from demo2gripper.test_tube_contact_bootstrap import (
    TestTubeContactProfile as ContactProfile,
)
from demo2gripper.test_tube_contact_bootstrap import TestTubeContactState as ContactState
from demo2gripper.test_tube_contact_bootstrap import (
    TestTubeFingerContactCalibration as FingerContactCalibration,
)
from demo2gripper.test_tube_predicates import (
    ContactStatus,
    JamAssessment,
    ReceiptHeader,
    insertion_progress_receipt_from_evaluation,
    tactile_axis_receipt_from_evaluation,
    tactile_contact_receipt_from_evaluation,
)
from demo2gripper.test_tube_tactile_controller import (
    TestTubeFingerImageProfile as FingerImageProfile,
)
from demo2gripper.test_tube_tactile_controller import (
    TestTubeInsertionJamProfile as InsertionJamProfile,
)
from demo2gripper.test_tube_tactile_controller import (
    TestTubeTactileAction as TactileAction,
)
from demo2gripper.test_tube_tactile_controller import (
    TestTubeTactileControllerProfile as ControllerProfile,
)
from demo2gripper.test_tube_tactile_controller import (
    TestTubeTactileMemory as TactileMemory,
)
from demo2gripper.test_tube_tactile_controller import TestTubeTactileMode as TactileMode
from demo2gripper.test_tube_tactile_controller import evaluate_test_tube_tactile

SESSION = "tube-session-001"
CONTACT_PROFILE_ID = "tube-contact-profile-001"
SERIALS = ("serial-a", "serial-b")
CALIBRATIONS = ("cal-a", "cal-b")
IMAGE_SHAPE = (48, 64, 3)


def _timed(values: np.ndarray, start: float) -> TimedArrayWindow:
    count = len(values)
    times = start + np.arange(count, dtype=float) * 0.01
    return TimedArrayWindow(values, times, times, np.ones(count, dtype=bool))


def _patch_image(
    *,
    center: tuple[float, float] = (32.0, 24.0),
    angle_deg: float = 0.0,
) -> np.ndarray:
    yy, xx = np.mgrid[: IMAGE_SHAPE[0], : IMAGE_SHAPE[1]]
    angle = np.deg2rad(angle_deg)
    x = (xx - center[0]) * np.cos(angle) + (yy - center[1]) * np.sin(angle)
    y = -(xx - center[0]) * np.sin(angle) + (yy - center[1]) * np.cos(angle)
    image = np.full(IMAGE_SHAPE, 100, dtype=np.uint8)
    image[(np.abs(x) <= 18.0) & (np.abs(y) <= 3.0)] = 190
    return image


def _windows(
    *,
    start: float,
    angles: tuple[float, float] = (0.0, 0.0),
    force_levels: tuple[float, float] = (4.0, 4.0),
    centers: tuple[tuple[float, float], tuple[float, float]] = (
        (32.0, 24.0),
        (32.0, 24.0),
    ),
) -> GF220ClassifierWindows:
    fingers = []
    for index in range(2):
        frame = _patch_image(center=centers[index], angle_deg=angles[index])
        images = np.repeat(frame[None], 4, axis=0)
        force = np.zeros((4, 6), dtype=float)
        force[:, 2] = force_levels[index]
        fingers.append(
            FingerRimClassifierWindow(
                calibration_id=CALIBRATIONS[index],
                warped=_timed(images, start),
                diff=_timed(images, start),
                force6d=_timed(force, start),
            )
        )
    return GF220ClassifierWindows(
        sensor_names=("left", "right"),
        serials=SERIALS,
        windows=(fingers[0], fingers[1]),
        start_skew_s=0.0,
        end_skew_s=0.0,
        collected_monotonic_s=start + 0.03,
    )


def _contact_profile() -> ContactProfile:
    config = ContactBootstrapConfig(
        min_unique_samples=4,
        min_calibration_windows_per_stage=3,
        contact_confirmation_windows=2,
        loss_confirmation_windows=2,
    )
    fingers = tuple(
        FingerContactCalibration(
            serial=serial,
            calibration_id=calibration,
            unloaded_fz_sensor_units=0.0,
            unloaded_noise_sensor_units=0.02,
            positive_fz_sensor_units=4.0,
            positive_delta_sensor_units=4.0,
            positive_support_windows=3,
            positive_total_windows=3,
            contact_response_qualified=True,
            last_bootstrap_sdk_time_s=2.0,
            last_bootstrap_capture_time_s=2.0,
        )
        for serial, calibration in zip(SERIALS, CALIBRATIONS, strict=True)
    )
    return ContactProfile(
        profile_id=CONTACT_PROFILE_ID,
        session_id=SESSION,
        session_evidence_sha256="a" * 64,
        fingers=(fingers[0], fingers[1]),
        engage_threshold_sensor_units=1.0,
        retain_threshold_sensor_units=0.4,
        config=config,
    )


def _controller_profile(
    *,
    axis_qualified: bool = True,
    slip_qualified: bool = False,
    jam_qualified: bool = True,
    axis_confirmation_windows: int = 4,
    slip_confirmation_windows: int = 2,
) -> ControllerProfile:
    baseline = np.full(IMAGE_SHAPE, 100, dtype=np.uint8)
    roi = np.ones(IMAGE_SHAPE[:2], dtype=bool)
    fingers = tuple(
        FingerImageProfile(
            serial=serial,
            calibration_id=calibration,
            unloaded_warped=baseline,
            roi_mask=roi,
            R_tcp_pad=np.eye(3),
            residual_threshold=20.0,
            min_contact_area_px=80,
        )
        for serial, calibration in zip(SERIALS, CALIBRATIONS, strict=True)
    )
    jam = InsertionJamProfile(
        profile_id="tube-insert-wrench-v1",
        task_frame_id="tube_insert",
        insertion_axis_task=np.array([0.0, 0.0, 1.0]),
        reaction_sign=1,
        resistance_baseline_signal=0.0,
        resistance_noise_signal=0.5,
        minimum_resistance_delta_signal=5.0,
        runtime_qualified=jam_qualified,
    )
    return ControllerProfile(
        profile_id="tube-tactile-controller-v1",
        session_id=SESSION,
        contact_profile=_contact_profile(),
        fingers=(fingers[0], fingers[1]),
        retained_width_min_m=0.010,
        retained_width_max_m=0.025,
        axis_runtime_qualified=axis_qualified,
        slip_runtime_qualified=slip_qualified,
        jam=jam,
        axis_confirmation_windows=axis_confirmation_windows,
        slip_confirmation_windows=slip_confirmation_windows,
    )


def _contact(
    windows: GF220ClassifierWindows,
    *,
    active: tuple[str, ...] = SERIALS,
    present: bool = True,
) -> ContactDecision:
    sdk = []
    host = []
    for serial in SERIALS:
        force = windows.for_serial(serial).force6d
        assert force is not None
        sdk.append((serial, float(force.sdk_time_s[-1])))
        host.append((serial, float(force.capture_monotonic_s[-1])))
    state = ContactState(
        session_id=SESSION,
        profile_id=CONTACT_PROFILE_ID,
        contact_present=present,
        contact_windows=2 if present else 0,
        loss_windows=0 if present else 2,
        last_sdk_time_s_by_serial=tuple(sdk),
        last_capture_time_s_by_serial=tuple(host),
    )
    return ContactDecision(
        valid=True,
        contact_present=present,
        decision="CONTACT_PRESENT" if present else "CONTACT_ABSENT",
        max_abs_fz_delta_sensor_units=4.0 if present else 0.0,
        active_serials=active,
        reason="test receipt",
        next_state=state,
    )


def _evaluate(
    windows: GF220ClassifierWindows,
    *,
    mode: TactileMode,
    profile: ControllerProfile | None = None,
    memory: TactileMemory = TactileMemory(),
    contact: ContactDecision | None = None,
    **kwargs: object,
):
    return evaluate_test_tube_tactile(
        mode=mode,
        windows=windows,
        contact=contact or _contact(windows),
        profile=profile or _controller_profile(),
        memory=memory,
        now_s=windows.collected_monotonic_s + 0.01,
        T_base_tcp=np.eye(4),
        target_axis_base=np.array([1.0, 0.0, 0.0]),
        **kwargs,
    )


def _wrenches(
    windows: GF220ClassifierWindows,
    *,
    axial_each: float,
) -> AdaptedGF220Wrenches:
    result = []
    for index, serial in enumerate(SERIALS):
        source = windows.for_serial(serial).force6d
        assert source is not None
        wrench = np.zeros((len(source.values), 6), dtype=float)
        wrench[:, 2] = axial_each
        result.append(
            FingerWrenchWindow(
                calibration_id=CALIBRATIONS[index],
                task_frame_id="tube_insert",
                wrench_task=wrench,
                contact_position_task_m=np.array(
                    [0.02 if index == 0 else -0.02, 0.0, 0.0]
                ),
                sdk_time_s=source.sdk_time_s,
                capture_monotonic_s=source.capture_monotonic_s,
                valid=source.valid,
            )
        )
    return AdaptedGF220Wrenches(
        profile_id="tube-insert-wrench-v1",
        task_frame_id="tube_insert",
        left_serial=SERIALS[0],
        right_serial=SERIALS[1],
        left=result[0],
        right=result[1],
        provenance_by_serial={SERIALS[0]: {}, SERIALS[1]: {}},
    )


def test_bilateral_contact_and_retained_width_require_two_windows() -> None:
    first_windows = _windows(start=3.0)
    first = _evaluate(
        first_windows,
        mode=TactileMode.GRASP,
        gripper_width_m=0.015,
    )
    second_windows = _windows(start=3.1)
    second = _evaluate(
        second_windows,
        mode=TactileMode.GRASP,
        memory=first.next_memory,
        gripper_width_m=0.015,
    )

    assert first.valid and first.primitives.bilateral_contact
    assert first.action is TactileAction.WAIT
    assert second.action is TactileAction.GRASP_READY
    assert "grasp_established" in second.promoted_predicates


def test_one_pad_contact_never_claims_a_grasp() -> None:
    windows = _windows(start=3.0, force_levels=(4.0, 0.0))
    result = _evaluate(
        windows,
        mode=TactileMode.GRASP,
        contact=_contact(windows, active=(SERIALS[0],)),
        gripper_width_m=0.015,
    )

    assert result.valid and result.primitives.contact_present
    assert not result.primitives.bilateral_contact
    assert "grasp_established" not in result.promoted_predicates
    assert result.action is TactileAction.WAIT


def test_dual_pad_axis_is_fused_in_base_and_debounced() -> None:
    profile = _controller_profile(axis_confirmation_windows=4)
    memory = TactileMemory()
    actions = []
    result = None
    for index in range(4):
        windows = _windows(start=3.0 + 0.1 * index)
        result = _evaluate(
            windows,
            mode=TactileMode.PIVOT,
            profile=profile,
            memory=memory,
        )
        memory = result.next_memory
        actions.append(result.action)

    assert result is not None
    assert result.primitives.axis_base == pytest.approx((1.0, 0.0, 0.0))
    assert result.primitives.target_axis_error_deg == pytest.approx(0.0)
    assert actions == [
        TactileAction.PIVOT_OPEN_STEP,
        TactileAction.PIVOT_OPEN_STEP,
        TactileAction.PIVOT_OPEN_STEP,
        TactileAction.PIVOT_READY,
    ]
    assert "tactile_relative_angle_reached" in result.promoted_predicates


def test_debounced_controller_evaluation_converts_to_bound_axis_receipt() -> None:
    profile = _controller_profile(axis_confirmation_windows=2)
    first = _evaluate(
        _windows(start=3.0),
        mode=TactileMode.PIVOT,
        profile=profile,
    )
    second = _evaluate(
        _windows(start=3.1),
        mode=TactileMode.PIVOT,
        profile=profile,
        memory=first.next_memory,
    )

    receipt = tactile_axis_receipt_from_evaluation(
        evaluation=second,
        profile=profile,
        provenance_id="geometry-1",
    )

    assert receipt.profile_id == profile.profile_id
    assert receipt.session_id == SESSION
    assert receipt.source_evidence_id == second.evidence_id
    assert receipt.axis_tcp == pytest.approx((1.0, 0.0, 0.0))
    assert receipt.axis_base == pytest.approx((1.0, 0.0, 0.0))
    assert receipt.target_confirmation_count == 2
    assert receipt.target_confirmed is True

    contact = tactile_contact_receipt_from_evaluation(
        header=ReceiptHeader(
            task_id="tube-task",
            session_id=SESSION,
            provenance_id="geometry-1",
            receipt_id="tactile-contact-1",
            producer_id="dual-gf220-controller",
            observed_at_s=3.14,
        ),
        evaluation=second,
        profile=profile,
    )
    assert contact.status is ContactStatus.PRESENT
    assert contact.bilateral_contact is True
    assert contact.axis_evaluation == receipt


def test_unqualified_axis_stays_shadow_and_holds() -> None:
    profile = _controller_profile(axis_qualified=False, axis_confirmation_windows=1)
    windows = _windows(start=3.0)
    result = _evaluate(windows, mode=TactileMode.PIVOT, profile=profile)

    assert result.valid and result.primitives.target_axis_candidate
    assert result.action is TactileAction.HOLD_INVALID
    assert "tactile_relative_angle_reached" in result.shadow_predicates
    assert "tactile_relative_angle_reached" not in result.promoted_predicates


def test_incoherent_pad_axes_do_not_produce_a_3d_axis() -> None:
    windows = _windows(start=3.0, angles=(0.0, 90.0))
    result = _evaluate(windows, mode=TactileMode.PIVOT)

    assert result.valid
    assert not result.primitives.axis_valid
    assert result.primitives.pad_axis_disagreement_deg is not None
    assert result.primitives.pad_axis_disagreement_deg > 80.0
    assert result.action is TactileAction.WAIT


def test_image_shift_plus_loss_of_one_pad_promotes_only_qualified_slip() -> None:
    profile = _controller_profile(slip_qualified=True, slip_confirmation_windows=1)
    first_windows = _windows(start=3.0)
    first = _evaluate(
        first_windows,
        mode=TactileMode.HOLD,
        profile=profile,
    )
    second_windows = _windows(
        start=3.1,
        centers=((40.0, 24.0), (40.0, 24.0)),
        force_levels=(4.0, 0.0),
    )
    second = _evaluate(
        second_windows,
        mode=TactileMode.HOLD,
        profile=profile,
        memory=first.next_memory,
        contact=_contact(second_windows, active=(SERIALS[0],)),
    )

    assert second.primitives.slip_candidate
    assert "slip_detected" in second.promoted_predicates
    assert second.action is TactileAction.RECLAMP_STEP


def test_low_progress_and_relative_resistance_need_two_windows_for_jam() -> None:
    profile = _controller_profile(jam_qualified=True)
    first_windows = _windows(start=3.0)
    first = _evaluate(
        first_windows,
        mode=TactileMode.INSERT,
        profile=profile,
        adapted_wrenches=_wrenches(first_windows, axial_each=4.0),
        commanded_axial_step_m=0.0015,
        measured_axial_progress_m=0.0001,
    )
    second_windows = _windows(start=3.1)
    second = _evaluate(
        second_windows,
        mode=TactileMode.INSERT,
        profile=profile,
        memory=first.next_memory,
        adapted_wrenches=_wrenches(second_windows, axial_each=4.0),
        commanded_axial_step_m=0.0015,
        measured_axial_progress_m=0.0001,
    )

    assert first.primitives.insertion_jam_candidate
    assert first.action is TactileAction.WAIT
    assert second.action is TactileAction.RETRACT_REALIGN
    assert "insertion_jam" in second.promoted_predicates


def test_healthy_progress_is_no_jam_and_requests_next_step() -> None:
    windows = _windows(start=3.0)
    result = _evaluate(
        windows,
        mode=TactileMode.INSERT,
        adapted_wrenches=_wrenches(windows, axial_each=1.0),
        commanded_axial_step_m=0.0015,
        measured_axial_progress_m=0.0014,
    )

    assert not result.primitives.insertion_jam_candidate
    assert "no_jam" in result.promoted_predicates
    assert result.action is TactileAction.INSERT_STEP

    receipt = insertion_progress_receipt_from_evaluation(
        header=ReceiptHeader(
            task_id="tube-task",
            session_id=SESSION,
            provenance_id="geometry-1",
            receipt_id="insertion-1",
            producer_id="tube-insertion-monitor",
            observed_at_s=3.04,
        ),
        evaluation=result,
        profile=_controller_profile(),
        measured_depth_m=0.004,
        target_depth_m=0.010,
        depth_tolerance_m=0.001,
        axial_progress_m=0.0014,
        minimum_progress_m=0.000375,
    )
    assert receipt.jam_assessment is JamAssessment.CLEAR
    assert receipt.runtime_jam_qualified is True


def test_unqualified_jam_candidate_is_shadow_and_cannot_advance() -> None:
    profile = _controller_profile(jam_qualified=False)
    windows = _windows(start=3.0)
    result = _evaluate(
        windows,
        mode=TactileMode.INSERT,
        profile=profile,
        adapted_wrenches=_wrenches(windows, axial_each=4.0),
        commanded_axial_step_m=0.0015,
        measured_axial_progress_m=0.0001,
    )

    assert "insertion_jam" in result.shadow_predicates
    assert "insertion_jam" not in result.promoted_predicates
    assert result.action is TactileAction.HOLD_INVALID


def test_replayed_image_window_fails_closed_without_advancing_memory() -> None:
    windows = _windows(start=3.0)
    first = _evaluate(windows, mode=TactileMode.HOLD)
    replay = _evaluate(
        windows,
        mode=TactileMode.HOLD,
        memory=first.next_memory,
    )

    assert not replay.valid
    assert replay.action is TactileAction.HOLD_INVALID
    assert "replays" in replay.reason
    assert replay.next_memory == first.next_memory


def test_contact_receipt_must_be_bound_to_same_window() -> None:
    windows = _windows(start=3.0)
    foreign_windows = _windows(start=3.1)
    result = _evaluate(
        windows,
        mode=TactileMode.HOLD,
        contact=_contact(foreign_windows),
    )

    assert not result.valid
    assert result.action is TactileAction.HOLD_INVALID
    assert "not bound" in result.reason


def test_contact_receipt_state_cannot_disagree_with_decision() -> None:
    windows = _windows(start=3.0)
    contact = _contact(windows)
    contradictory = replace(
        contact,
        next_state=replace(contact.next_state, contact_present=False),
    )
    result = _evaluate(
        windows,
        mode=TactileMode.HOLD,
        contact=contradictory,
    )

    assert not result.valid
    assert "disagrees" in result.reason


def test_qualified_pivot_requires_tcp_and_target_geometry() -> None:
    windows = _windows(start=3.0)
    result = evaluate_test_tube_tactile(
        mode=TactileMode.PIVOT,
        windows=windows,
        contact=_contact(windows),
        profile=_controller_profile(axis_qualified=True),
        now_s=windows.collected_monotonic_s + 0.01,
    )

    assert not result.valid
    assert result.action is TactileAction.HOLD_INVALID
    assert "requires T_base_tcp" in result.reason
    assert "requires target_axis_base" in result.reason


def test_insertion_wrench_evidence_must_match_the_tactile_window() -> None:
    windows = _windows(start=3.0)
    foreign = _windows(start=3.1)
    result = _evaluate(
        windows,
        mode=TactileMode.INSERT,
        adapted_wrenches=_wrenches(foreign, axial_each=4.0),
        commanded_axial_step_m=0.0015,
        measured_axial_progress_m=0.0001,
    )

    assert not result.valid
    assert result.action is TactileAction.HOLD_INVALID
    assert "not bound" in result.reason


def test_invalid_insertion_progress_fails_closed() -> None:
    windows = _windows(start=3.0)
    result = _evaluate(
        windows,
        mode=TactileMode.INSERT,
        adapted_wrenches=_wrenches(windows, axial_each=4.0),
        commanded_axial_step_m=0.0015,
        measured_axial_progress_m=-0.0001,
    )

    assert not result.valid
    assert result.action is TactileAction.HOLD_INVALID
    assert "non-negative" in result.reason


def test_release_requires_open_gripper_and_confirmed_contact_absence() -> None:
    windows = _windows(start=3.0, force_levels=(0.0, 0.0))
    result = _evaluate(
        windows,
        mode=TactileMode.RELEASE,
        contact=_contact(windows, active=(), present=False),
        gripper_open_confirmed=True,
    )

    assert result.valid
    assert result.action is TactileAction.RELEASE_READY


def test_profile_rejects_reflection_as_pad_rotation() -> None:
    profile = _controller_profile()
    with pytest.raises(ValueError, match="proper rotation"):
        replace(
            profile.fingers[0],
            R_tcp_pad=np.diag([1.0, 1.0, -1.0]),
        )
