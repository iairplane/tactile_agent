from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from demo2gripper.gf220_live import GF220Calibration, GF220ClassifierWindows
from demo2gripper.tactile_contact_classifier import FingerRimClassifierWindow
from demo2gripper.tactile_rim_probe import TimedArrayWindow
from demo2gripper.test_tube_closed_loop import GripperFeedback
from demo2gripper.test_tube_preflight import TubeState
from demo2gripper.test_tube_session_tactile import (
    SESSION_BOOTSTRAP_SCHEMA,
    SessionBootstrapTestTubeTactileInterpreter,
    SessionTactileConfig,
    SessionTactileError,
    create_session_bootstrap_tactile_interpreter,
)

SERIALS = ("gf-a", "gf-b")
SHAPE = (64, 64, 3)


def _calibrations() -> tuple[GF220Calibration, GF220Calibration]:
    image = np.full(SHAPE, 100, dtype=np.uint8)
    values = tuple(
        GF220Calibration.create(
            serial=serial,
            warp_sha256=("a" if index == 0 else "b") * 64,
            unloaded_warped=image,
        )
        for index, serial in enumerate(SERIALS)
    )
    return values[0], values[1]


def _contact_image(
    angle_deg: float | None, center: tuple[float, float] = (32.0, 32.0)
) -> np.ndarray:
    image = np.full(SHAPE, 100, dtype=np.uint8)
    if angle_deg is None:
        return image
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    angle = math.radians(angle_deg)
    along = (xx - center[0]) * math.cos(angle) + (yy - center[1]) * math.sin(angle)
    across = -(xx - center[0]) * math.sin(angle) + (yy - center[1]) * math.cos(angle)
    image[(np.abs(along) <= 22.0) & (np.abs(across) <= 3.0)] = 190
    return image


def _timed(
    values: np.ndarray, start: float, sample_period_s: float = 0.01
) -> TimedArrayWindow:
    count = len(values)
    times = start + np.arange(count, dtype=float) * sample_period_s
    return TimedArrayWindow(values, times, times, np.ones(count, dtype=bool))


def _windows(
    calibrations: tuple[GF220Calibration, GF220Calibration],
    *,
    start: float,
    forces: tuple[float, float] = (0.0, 0.0),
    angles: tuple[float | None, float | None] = (None, None),
    centers: tuple[tuple[float, float], tuple[float, float]] = (
        (32.0, 32.0),
        (32.0, 32.0),
    ),
    sample_period_s: float = 0.01,
) -> GF220ClassifierWindows:
    fingers = []
    for index, calibration in enumerate(calibrations):
        image = _contact_image(angles[index], centers[index])
        images = np.repeat(image[None], 4, axis=0)
        force = np.zeros((4, 6), dtype=float)
        force[:, 2] = forces[index]
        fingers.append(
            FingerRimClassifierWindow(
                calibration_id=calibration.calibration_id,
                warped=_timed(images, start, sample_period_s),
                diff=_timed(images, start, sample_period_s),
                force6d=_timed(force, start, sample_period_s),
            )
        )
    return GF220ClassifierWindows(
        sensor_names=("a", "b"),
        serials=SERIALS,
        windows=(fingers[0], fingers[1]),
        start_skew_s=0.0,
        end_skew_s=0.0,
        collected_monotonic_s=start + 3.0 * sample_period_s,
    )


def _feedback(
    interpreter: SessionBootstrapTestTubeTactileInterpreter,
    windows: GF220ClassifierWindows,
    state: TubeState,
    *,
    command: float | None = None,
    progress: float | None = None,
):
    return interpreter.evaluate(
        windows=windows,
        state=state,
        now_s=windows.collected_monotonic_s + 0.01,
        T_base_tcp=np.eye(4),
        gripper=GripperFeedback(0.050, False, True),
        commanded_insertion_step_m=command,
        measured_insertion_progress_m=progress,
    )


def _ready(tmp_path: Path) -> tuple[
    SessionBootstrapTestTubeTactileInterpreter,
    tuple[GF220Calibration, GF220Calibration],
    float,
]:
    calibrations = _calibrations()
    interpreter = SessionBootstrapTestTubeTactileInterpreter(
        calibrations=calibrations,
        output_dir=tmp_path,
        config=SessionTactileConfig(
            unloaded_windows=3,
            contact_confirmation_windows=1,
            bilateral_confirmation_windows=2,
            pivot_confirmation_windows=2,
            rotation_stop_confirmation_windows=2,
            insertion_event_confirmation_windows=2,
            minimum_successful_steps_before_seat=2,
        ),
    )
    start = 1.0
    assert not _feedback(
        interpreter, _windows(calibrations, start=start), TubeState.OBSERVE
    ).valid
    start += 0.1
    assert not _feedback(
        interpreter, _windows(calibrations, start=start), TubeState.OBSERVE
    ).valid
    start += 0.1
    assert _feedback(
        interpreter, _windows(calibrations, start=start), TubeState.OBSERVE
    ).valid
    return interpreter, calibrations, start + 0.1


def test_window_freshness_uses_latest_common_endpoint(tmp_path: Path) -> None:
    calibrations = _calibrations()
    interpreter = SessionBootstrapTestTubeTactileInterpreter(
        calibrations=calibrations,
        output_dir=tmp_path,
        config=SessionTactileConfig(unloaded_windows=1, max_age_s=0.25),
    )
    # Four frames at 12.5 Hz span 240 ms.  At evaluation time the first frame
    # is 250 ms old while the common FORCE6D/image endpoint is only 10 ms old.
    windows = _windows(calibrations, start=1.0, sample_period_s=0.08)

    result = _feedback(interpreter, windows, TubeState.OBSERVE)

    assert result.valid
    assert result.observed_at_s == pytest.approx(1.24)
    assert interpreter.session_ready


def test_window_with_stale_latest_common_endpoint_is_rejected(tmp_path: Path) -> None:
    calibrations = _calibrations()
    interpreter = SessionBootstrapTestTubeTactileInterpreter(
        calibrations=calibrations,
        output_dir=tmp_path,
        config=SessionTactileConfig(unloaded_windows=1, max_age_s=0.25),
    )
    windows = _windows(calibrations, start=1.0)

    result = interpreter.evaluate(
        windows=windows,
        state=TubeState.OBSERVE,
        now_s=1.29,
        T_base_tcp=np.eye(4),
        gripper=GripperFeedback(0.050, False, True),
        commanded_insertion_step_m=None,
        measured_insertion_progress_m=None,
    )

    assert not result.valid
    audit = json.loads((tmp_path / "tactile_interpreter/latest.json").read_text())
    assert audit["blockers"] == [f"{SERIALS[0]}: tactile window is stale"]


def test_session_bootstrap_derives_first_contact_and_bilateral_gate(tmp_path: Path) -> None:
    interpreter, calibrations, start = _ready(tmp_path)

    one = _feedback(
        interpreter,
        _windows(calibrations, start=start, forces=(4.0, 0.0), angles=(0.0, None)),
        TubeState.SIDE_GRASP,
    )
    start += 0.1
    both_first = _feedback(
        interpreter,
        _windows(calibrations, start=start, forces=(4.0, 4.0), angles=(0.0, 0.0)),
        TubeState.SIDE_GRASP,
    )
    start += 0.1
    both_second = _feedback(
        interpreter,
        _windows(calibrations, start=start, forces=(4.0, 4.0), angles=(0.0, 0.0)),
        TubeState.SIDE_GRASP,
    )

    assert one.valid and one.contact_present and one.single_contact_side == "left"
    assert not one.bilateral_contact
    assert not both_first.bilateral_contact
    assert both_second.bilateral_contact
    receipt = json.loads(
        (tmp_path / "tactile_interpreter/session_bootstrap.json").read_text()
    )
    assert receipt["schema"] == SESSION_BOOTSTRAP_SCHEMA
    assert receipt["status"] == "runtime_session_ready"
    assert all(item["force_engage_threshold"] > 0.0 for item in receipt["pads"])
    assert all(len(item["image_reference_sha256"]) == 64 for item in receipt["pads"])
    assert receipt["hardware_commands_dispatched"] is False


def test_light_optical_contact_does_not_require_force6d_and_edges_are_not_ready(
    tmp_path: Path,
) -> None:
    interpreter, calibrations, start = _ready(tmp_path)

    first = _feedback(
        interpreter,
        _windows(
            calibrations,
            start=start,
            forces=(0.0, 0.0),
            angles=(0.0, 0.0),
        ),
        TubeState.SIDE_GRASP,
    )
    start += 0.1
    centered = _feedback(
        interpreter,
        _windows(
            calibrations,
            start=start,
            forces=(0.0, 0.0),
            angles=(0.0, 0.0),
        ),
        TubeState.SIDE_GRASP,
    )

    assert first.contact_present
    assert not first.bilateral_contact
    assert centered.bilateral_contact
    assert centered.bilateral_contact_interior
    audit = json.loads((tmp_path / "tactile_interpreter/latest.json").read_text())
    assert audit["signals"]["optical_contact_serials"] == list(SERIALS)
    assert audit["signals"]["force_qualified_contact_serials"] == []

    start += 0.1
    edge_first = _feedback(
        interpreter,
        _windows(
            calibrations,
            start=start,
            forces=(0.0, 0.0),
            angles=(0.0, 0.0),
            centers=((32.0, 3.0), (32.0, 61.0)),
        ),
        TubeState.SIDE_GRASP,
    )
    start += 0.1
    edge_second = _feedback(
        interpreter,
        _windows(
            calibrations,
            start=start,
            forces=(0.0, 0.0),
            angles=(0.0, 0.0),
            centers=((32.0, 3.0), (32.0, 61.0)),
        ),
        TubeState.SIDE_GRASP,
    )
    assert edge_first.bilateral_contact
    assert edge_second.bilateral_contact
    assert not edge_first.bilateral_contact_interior
    assert not edge_second.bilateral_contact_interior


def test_pivot_and_rotation_stop_are_debounced_session_events(tmp_path: Path) -> None:
    interpreter, calibrations, start = _ready(tmp_path)
    for state in (TubeState.SIDE_GRASP, TubeState.EXTRACT):
        for _ in range(2):
            result = _feedback(
                interpreter,
                _windows(
                    calibrations,
                    start=start,
                    forces=(4.0, 4.0),
                    angles=(0.0, 0.0),
                ),
                state,
            )
            start += 0.1
    assert result.bilateral_contact

    first = _feedback(
        interpreter,
        _windows(calibrations, start=start, forces=(4.0, 4.0), angles=(45.0, 45.0)),
        TubeState.PIVOT_EVALUATE,
    )
    start += 0.1
    second = _feedback(
        interpreter,
        _windows(calibrations, start=start, forces=(4.0, 4.0), angles=(45.0, 45.0)),
        TubeState.PIVOT_EVALUATE,
    )
    start += 0.1
    reclamp_one = _feedback(
        interpreter,
        _windows(calibrations, start=start, forces=(4.0, 4.0), angles=(45.0, 45.0)),
        TubeState.RECLAMP,
    )
    start += 0.1
    reclamp_two = _feedback(
        interpreter,
        _windows(calibrations, start=start, forces=(4.0, 4.0), angles=(45.0, 45.0)),
        TubeState.RECLAMP,
    )

    assert not first.relative_angle_target_reached
    assert second.relative_angle_target_reached
    assert not reclamp_one.rotation_stopped
    assert reclamp_two.rotation_stopped


def test_pivot_contact_loss_reclamps_but_normal_axis_motion_is_not_slip(
    tmp_path: Path,
) -> None:
    interpreter, calibrations, start = _ready(tmp_path)
    for _ in range(2):
        held = _feedback(
            interpreter,
            _windows(
                calibrations,
                start=start,
                forces=(4.0, 4.0),
                angles=(0.0, 0.0),
            ),
            TubeState.EXTRACT,
        )
        start += 0.1
    assert held.bilateral_contact

    # A few pixels of patch-centre motion accompanies the intended slow pivot;
    # the pivot-specific threshold is deliberately larger than the steady-hold
    # slip threshold, so this alone must not request a reclamp.
    normal_pivot = _feedback(
        interpreter,
        _windows(
            calibrations,
            start=start,
            forces=(4.0, 4.0),
            angles=(20.0, 20.0),
            centers=((35.0, 32.0), (35.0, 32.0)),
        ),
        TubeState.PIVOT_EVALUATE,
    )
    start += 0.1
    assert not normal_pivot.slip_detected

    first_loss = _feedback(
        interpreter,
        _windows(
            calibrations,
            start=start,
            forces=(4.0, 0.0),
            angles=(20.0, None),
            centers=((35.0, 32.0), (35.0, 32.0)),
        ),
        TubeState.PIVOT_EVALUATE,
    )
    start += 0.1
    confirmed_loss = _feedback(
        interpreter,
        _windows(
            calibrations,
            start=start,
            forces=(4.0, 0.0),
            angles=(20.0, None),
            centers=((35.0, 32.0), (35.0, 32.0)),
        ),
        TubeState.PIVOT_EVALUATE,
    )

    assert not first_loss.slip_detected
    assert confirmed_loss.slip_detected
    audit = json.loads((tmp_path / "tactile_interpreter/latest.json").read_text())
    assert audit["signals"]["contact_patch_lost"] is True
    assert audit["signals"]["pivot_contact_monitoring"] is True


def _gravity_direction_ready(
    tmp_path: Path,
) -> tuple[
    SessionBootstrapTestTubeTactileInterpreter,
    tuple[GF220Calibration, GF220Calibration],
    float,
]:
    calibrations = _calibrations()
    interpreter = SessionBootstrapTestTubeTactileInterpreter(
        calibrations=calibrations,
        output_dir=tmp_path,
        config=SessionTactileConfig(
            unloaded_windows=1,
            contact_confirmation_windows=1,
            bilateral_confirmation_windows=1,
            pivot_confirmation_windows=2,
            # With identity T_base_tcp, +Y lever crossed with base -Z gravity
            # gives a negative moment around canonical jaw +X.
            gravity_pivot_lever_tcp=(0.0, 1.0, 0.0),
        ),
    )
    start = 1.0
    assert _feedback(
        interpreter, _windows(calibrations, start=start), TubeState.OBSERVE
    ).valid
    start += 0.1
    held = _feedback(
        interpreter,
        _windows(
            calibrations,
            start=start,
            forces=(4.0, 4.0),
            angles=(0.0, 0.0),
        ),
        TubeState.EXTRACT,
    )
    assert held.bilateral_contact
    return interpreter, calibrations, start + 0.1


def test_gravity_pivot_requires_expected_signed_pad_rotation(tmp_path: Path) -> None:
    interpreter, calibrations, start = _gravity_direction_ready(tmp_path)

    # Standard opposed pads mirror the same physical negative TCP-X rotation:
    # the left image line moves +45 degrees and the right line -45 degrees.
    first = _feedback(
        interpreter,
        _windows(
            calibrations,
            start=start,
            forces=(4.0, 4.0),
            angles=(45.0, -45.0),
        ),
        TubeState.PIVOT_EVALUATE,
    )
    start += 0.1
    second = _feedback(
        interpreter,
        _windows(
            calibrations,
            start=start,
            forces=(4.0, 4.0),
            angles=(45.0, -45.0),
        ),
        TubeState.PIVOT_EVALUATE,
    )

    assert not first.relative_angle_target_reached
    assert second.relative_angle_target_reached
    assert not second.slip_detected
    audit = json.loads((tmp_path / "tactile_interpreter/latest.json").read_text())
    signals = audit["signals"]
    assert signals["gravity_pivot_expected_sign"] == -1
    assert signals["gravity_pivot_observed_sign"] == -1
    assert signals["gravity_pivot_direction_matches"] is True
    assert signals["gravity_pivot_direction_fault"] is False


def test_opposite_gravity_pivot_direction_immediately_requests_reclamp(
    tmp_path: Path,
) -> None:
    interpreter, calibrations, start = _gravity_direction_ready(tmp_path)

    # The opposite mirrored image motion is a positive TCP-X rotation while
    # gravity predicts negative.  It reuses the controller's immediate
    # slip/reclamp precedence and can never promote the 45-degree target.
    result = _feedback(
        interpreter,
        _windows(
            calibrations,
            start=start,
            forces=(4.0, 4.0),
            angles=(-10.0, 10.0),
        ),
        TubeState.PIVOT_EVALUATE,
    )

    assert result.slip_detected
    assert not result.relative_angle_target_reached
    audit = json.loads((tmp_path / "tactile_interpreter/latest.json").read_text())
    signals = audit["signals"]
    assert signals["gravity_pivot_expected_sign"] == -1
    assert signals["gravity_pivot_observed_sign"] == 1
    assert signals["gravity_pivot_direction_fault"] is True
    assert signals["gravity_pivot_direction_fault_reclamp"] is True


def test_axial_patch_motion_in_a_hold_state_is_debounced_as_slip(tmp_path: Path) -> None:
    interpreter, calibrations, start = _ready(tmp_path)
    for center in (32.0, 32.0, 38.0, 44.0):
        result = _feedback(
            interpreter,
            _windows(
                calibrations,
                start=start,
                forces=(4.0, 4.0),
                angles=(0.0, 0.0),
                centers=((center, 32.0), (center, 32.0)),
            ),
            TubeState.EXTRACT,
        )
        start += 0.1

    assert result.slip_detected


def test_level_entry_reanchors_slip_after_extraction_patch_motion(
    tmp_path: Path,
) -> None:
    interpreter, calibrations, start = _ready(tmp_path)
    for center in (32.0, 38.0, 44.0):
        extracted = _feedback(
            interpreter,
            _windows(
                calibrations,
                start=start,
                forces=(4.0, 4.0),
                angles=(0.0, 0.0),
                centers=((center, 32.0), (center, 32.0)),
            ),
            TubeState.EXTRACT,
        )
        start += 0.1

    assert extracted.slip_detected
    leveled = _feedback(
        interpreter,
        _windows(
            calibrations,
            start=start,
            forces=(4.0, 4.0),
            angles=(0.0, 0.0),
            centers=((44.0, 32.0), (44.0, 32.0)),
        ),
        TubeState.LEVEL_HELD_TUBE,
    )

    assert leveled.bilateral_contact_interior
    assert not leveled.slip_detected
    audit = json.loads((tmp_path / "tactile_interpreter/latest.json").read_text())
    assert audit["signals"]["slip_reference_reset_on_level_entry"] is True
    assert audit["signals"]["slip_windows"] == 0


def test_insertion_reference_requires_consecutive_bilateral_windows(tmp_path: Path) -> None:
    interpreter, calibrations, start = _ready(tmp_path)
    for _ in range(2):
        _feedback(
            interpreter,
            _windows(
                calibrations,
                start=start,
                forces=(4.0, 4.0),
                angles=(0.0, 0.0),
            ),
            TubeState.SIDE_GRASP,
        )
        start += 0.1
    for _ in range(2):
        result = _feedback(
            interpreter,
            _windows(
                calibrations,
                start=start,
                forces=(4.0, 4.0),
                angles=(0.0, 0.0),
            ),
            TubeState.ALIGN_SOCKET,
        )
        start += 0.1
    assert not result.insertion_reference_ready

    result = _feedback(
        interpreter,
        _windows(
            calibrations,
            start=start,
            forces=(4.0, 0.0),
            angles=(0.0, None),
        ),
        TubeState.ALIGN_SOCKET,
    )
    start += 0.1
    assert not result.insertion_reference_ready

    readiness = []
    for _ in range(4):
        result = _feedback(
            interpreter,
            _windows(
                calibrations,
                start=start,
                forces=(4.0, 4.0),
                angles=(0.0, 0.0),
            ),
            TubeState.ALIGN_SOCKET,
        )
        readiness.append(result.insertion_reference_ready)
        start += 0.1
    assert readiness == [False, False, False, True]


def test_contact_patch_loss_in_a_hold_state_is_debounced_as_slip(tmp_path: Path) -> None:
    interpreter, calibrations, start = _ready(tmp_path)
    for _ in range(2):
        result = _feedback(
            interpreter,
            _windows(
                calibrations,
                start=start,
                forces=(4.0, 4.0),
                angles=(0.0, 0.0),
            ),
            TubeState.EXTRACT,
        )
        start += 0.1
    assert result.bilateral_contact and not result.slip_detected

    for _ in range(2):
        result = _feedback(
            interpreter,
            _windows(
                calibrations,
                start=start,
                forces=(4.0, 0.0),
                angles=(0.0, None),
            ),
            TubeState.EXTRACT,
        )
        start += 0.1

    assert result.slip_detected


def test_insertion_seat_and_jam_are_progress_resistance_events(tmp_path: Path) -> None:
    interpreter, calibrations, start = _ready(tmp_path / "seat")
    # Establish bilateral contact and a same-session pre-insertion load baseline.
    setup_states = (
        TubeState.SIDE_GRASP,
        TubeState.SIDE_GRASP,
        TubeState.ALIGN_SOCKET,
        TubeState.ALIGN_SOCKET,
        TubeState.ALIGN_SOCKET,
    )
    insertion_reference_readiness = []
    for state in setup_states:
        result = _feedback(
            interpreter,
            _windows(calibrations, start=start, forces=(4.0, 4.0), angles=(0.0, 0.0)),
            state,
        )
        if state is TubeState.ALIGN_SOCKET:
            insertion_reference_readiness.append(result.insertion_reference_ready)
        start += 0.1
    assert insertion_reference_readiness == [False, False, True]
    for _ in range(2):
        _feedback(
            interpreter,
            _windows(calibrations, start=start, forces=(4.0, 4.0), angles=(0.0, 0.0)),
            TubeState.INSERT_MICROSTEP,
            command=0.0015,
            progress=0.0015,
        )
        start += 0.1
        healthy = _feedback(
            interpreter,
            _windows(calibrations, start=start, forces=(4.0, 4.0), angles=(0.0, 0.0)),
            TubeState.INSERT_EVALUATE,
            command=0.0015,
            progress=0.0015,
        )
        start += 0.1
        assert not healthy.insertion_jam and not healthy.insertion_seated
    seat_first = _feedback(
        interpreter,
        _windows(calibrations, start=start, forces=(8.0, 8.0), angles=(0.0, 0.0)),
        TubeState.INSERT_EVALUATE,
        command=0.0015,
        progress=0.0001,
    )
    start += 0.1
    seat_second = _feedback(
        interpreter,
        _windows(calibrations, start=start, forces=(8.0, 8.0), angles=(0.0, 0.0)),
        TubeState.INSERT_EVALUATE,
        command=0.0015,
        progress=0.0001,
    )
    assert seat_first.insertion_resistance_high and not seat_first.insertion_seated
    assert seat_second.insertion_seated and not seat_second.insertion_jam

    jam_interpreter, jam_calibrations, start = _ready(tmp_path / "jam")
    for state in setup_states:
        _feedback(
            jam_interpreter,
            _windows(jam_calibrations, start=start, forces=(4.0, 4.0), angles=(0.0, 0.0)),
            state,
        )
        start += 0.1
    jam_first = _feedback(
        jam_interpreter,
        _windows(jam_calibrations, start=start, forces=(10.0, 4.2), angles=(0.0, 0.0)),
        TubeState.INSERT_EVALUATE,
        command=0.0015,
        progress=0.0001,
    )
    start += 0.1
    jam_second = _feedback(
        jam_interpreter,
        _windows(jam_calibrations, start=start, forces=(10.0, 4.2), angles=(0.0, 0.0)),
        TubeState.INSERT_EVALUATE,
        command=0.0015,
        progress=0.0001,
    )
    assert not jam_first.insertion_jam
    assert jam_second.insertion_jam and not jam_second.insertion_seated


def test_confirmed_contact_patch_loss_invalidates_insertion_reference(
    tmp_path: Path,
) -> None:
    interpreter, calibrations, start = _ready(tmp_path)
    for state in (
        TubeState.SIDE_GRASP,
        TubeState.SIDE_GRASP,
        TubeState.ALIGN_SOCKET,
        TubeState.ALIGN_SOCKET,
        TubeState.ALIGN_SOCKET,
    ):
        ready = _feedback(
            interpreter,
            _windows(
                calibrations,
                start=start,
                forces=(4.0, 4.0),
                angles=(0.0, 0.0),
            ),
            state,
        )
        start += 0.1
    assert ready.insertion_reference_ready

    first_loss = _feedback(
        interpreter,
        _windows(
            calibrations,
            start=start,
            forces=(4.0, 0.0),
            angles=(0.0, None),
        ),
        TubeState.INSERT_EVALUATE,
        command=0.0015,
        progress=0.0015,
    )
    start += 0.1
    confirmed_loss = _feedback(
        interpreter,
        _windows(
            calibrations,
            start=start,
            forces=(4.0, 0.0),
            angles=(0.0, None),
        ),
        TubeState.INSERT_EVALUATE,
        command=0.0015,
        progress=0.0015,
    )

    assert not first_loss.slip_detected
    assert first_loss.insertion_reference_ready
    assert confirmed_loss.slip_detected
    assert not confirmed_loss.insertion_reference_ready
    assert not confirmed_loss.insertion_jam
    assert not confirmed_loss.insertion_seated


def test_factory_rejects_prefilled_or_unknown_threshold_keys(tmp_path: Path) -> None:
    with pytest.raises(SessionTactileError, match="unknown session tactile config"):
        create_session_bootstrap_tactile_interpreter(
            config={"force_threshold": 3.0},
            calibrations=_calibrations(),
            output_dir=tmp_path,
        )

