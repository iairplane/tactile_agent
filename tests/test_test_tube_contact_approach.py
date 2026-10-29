from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from demo2gripper.gf220_live import GF220Calibration, GF220ClassifierWindows
from demo2gripper.tactile_contact_classifier import FingerRimClassifierWindow
from demo2gripper.tactile_rim_probe import TimedArrayWindow
from demo2gripper.test_tube_contact_approach import (
    ContactApproachMemory,
    evaluate_contact_window,
    fit_unloaded_contact_profile,
    make_approach_schedule,
)


def _calibration(serial: str, value: int) -> GF220Calibration:
    return GF220Calibration.create(
        serial=serial,
        warp_sha256=("01" if serial == "a" else "02") * 32,
        unloaded_warped=np.full((64, 64, 3), value, dtype=np.uint8),
    )


def _finger_window(
    calibration_id: str,
    *,
    start: float,
    base: int,
    contact: bool,
    seed: int,
) -> FingerRimClassifierWindow:
    rng = np.random.default_rng(seed)
    warped = np.clip(
        base + rng.integers(-1, 2, size=(5, 64, 64, 3)), 0, 255
    ).astype(np.uint8)
    diff = np.clip(
        127 + rng.integers(-1, 2, size=(5, 64, 64, 3)), 0, 255
    ).astype(np.uint8)
    if contact:
        warped[:, 23:41, 26:39, :] = np.clip(
            warped[:, 23:41, 26:39, :].astype(int) + 55, 0, 255
        ).astype(np.uint8)
        diff[:, 23:41, 26:39, :] = 205
    sdk = start + np.arange(5, dtype=float) * 0.02
    host = 100.0 + sdk
    valid = np.ones(5, dtype=bool)
    warped_window = TimedArrayWindow(warped, sdk, host, valid)
    diff_window = TimedArrayWindow(diff, sdk, host, valid)
    return FingerRimClassifierWindow(calibration_id, warped_window, diff_window, None)


def _pair(
    calibrations: tuple[GF220Calibration, GF220Calibration],
    *,
    start: float,
    contacts: tuple[bool, bool] = (False, False),
    seed: int = 1,
) -> GF220ClassifierWindows:
    windows = tuple(
        _finger_window(
            calibration.calibration_id,
            start=start + index * 0.001,
            base=90 + index * 10,
            contact=contacts[index],
            seed=seed + index,
        )
        for index, calibration in enumerate(calibrations)
    )
    return GF220ClassifierWindows(
        sensor_names=("finger_a", "finger_b"),
        serials=("a", "b"),
        windows=(windows[0], windows[1]),
        start_skew_s=0.001,
        end_skew_s=0.001,
        collected_monotonic_s=200.0 + start,
    )


def _profile():
    calibrations = (_calibration("a", 90), _calibration("b", 100))
    unloaded = tuple(
        _pair(calibrations, start=1.0 + index * 0.20, seed=10 + index * 10)
        for index in range(3)
    )
    return calibrations, fit_unloaded_contact_profile(
        session_id="session-a", calibrations=calibrations, windows=unloaded
    )


def test_schedule_is_bounded_and_ends_at_grasp() -> None:
    schedule = make_approach_schedule(
        np.array([0.5, 0.1, 0.2]), np.array([0.0, 0.0, 2.0]), 0.010
    )
    assert len(schedule) == 3
    assert np.allclose(schedule[0], [0.5, 0.1, 0.194])
    assert np.allclose(schedule[1], [0.5, 0.1, 0.198])
    assert np.allclose(schedule[2], [0.5, 0.1, 0.2])
    with pytest.raises(ValueError, match=r"0\.040"):
        make_approach_schedule(np.ones(3), np.array([1.0, 0.0, 0.0]), 0.041)

    capped = make_approach_schedule(
        np.array([0.5, 0.1, 0.2]),
        np.array([0.0, 0.0, 1.0]),
        0.0377,
        maximum_steps=9,
    )
    assert len(capped) == 9
    assert np.allclose(capped[-1], [0.5, 0.1, 0.1983])
    assert capped[-1][2] < 0.2


def test_first_single_pad_contact_stops_and_never_reports_grasp_ready() -> None:
    calibrations, profile = _profile()
    clear = evaluate_contact_window(
        _pair(calibrations, start=2.0, contacts=(False, False), seed=100), profile
    )
    assert clear.valid
    assert clear.action == "advance_clear"
    assert not clear.stop_forward

    contact = evaluate_contact_window(
        _pair(calibrations, start=2.2, contacts=(True, False), seed=120),
        profile,
        clear.next_memory,
    )
    assert contact.stop_forward
    assert contact.action == "one_pad_contact_hold"
    assert contact.active_serials == ("a",)
    assert not contact.grasp_ready

    # Contact disappearing may never restart forward motion in this session.
    disappeared = evaluate_contact_window(
        _pair(calibrations, start=2.4, contacts=(False, False), seed=140),
        profile,
        contact.next_memory,
    )
    assert disappeared.stop_forward
    assert disappeared.action == "one_pad_contact_hold"
    assert not disappeared.grasp_ready


def test_grasp_ready_requires_two_fresh_bilateral_windows() -> None:
    calibrations, profile = _profile()
    first = evaluate_contact_window(
        _pair(calibrations, start=2.0, contacts=(True, True), seed=200), profile
    )
    assert first.bilateral_contact
    assert first.stop_forward
    assert first.action == "one_pad_contact_hold"
    assert not first.grasp_ready

    second = evaluate_contact_window(
        _pair(calibrations, start=2.2, contacts=(True, True), seed=220),
        profile,
        first.next_memory,
    )
    assert second.action == "grasp_ready"
    assert second.grasp_ready
    assert second.next_memory.bilateral_windows == 2


def test_replayed_image_window_fails_closed() -> None:
    calibrations, profile = _profile()
    window = _pair(calibrations, start=2.0, contacts=(False, False), seed=300)
    first = evaluate_contact_window(window, profile)
    replay = evaluate_contact_window(window, profile, first.next_memory)
    assert not replay.valid
    assert replay.action == "hold_invalid"
    assert replay.stop_forward
    assert "overlaps" in replay.reason


def test_dry_run_does_not_open_sensors_or_spawn_moveit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run_test_tube_contact_approach.py"
    spec = importlib.util.spec_from_file_location("contact_approach_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class ForbiddenSource:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dry run opened GF220")

    monkeypatch.setattr(module, "GF220LiveSource", ForbiddenSource)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dry run spawned MoveIt")),
    )
    output = tmp_path / "dry"
    args = module.parse_args(
        [
            "--target-grasp-position-m",
            "0.5",
            "0.1",
            "0.2",
            "--target-grasp-orientation-xyzw",
            "0",
            "0",
            "0",
            "1",
            "--approach-axis-base",
            "0",
            "0",
            "1",
            "--remaining-distance-m",
            "0.012",
            "--output-dir",
            str(output),
        ]
    )
    assert module.run(args) == 0
    report = json.loads((output / "run_report.json").read_text())
    assert report["status"] == "dry_run_complete"
    assert report["hardware_commands_dispatched"] is False
    assert report["scheduled_step_count"] == 3


def test_stopped_memory_is_immutable_semantically() -> None:
    memory = ContactApproachMemory(forward_stopped=True)
    assert memory.forward_stopped


def test_execute_flow_checks_stationary_contact_before_first_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run_test_tube_contact_approach.py"
    spec = importlib.util.spec_from_file_location("contact_approach_execute_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calibrations = (_calibration("a", 90), _calibration("b", 100))
    samples = iter(
        [
            _pair(calibrations, start=1.0, seed=10),
            _pair(calibrations, start=1.2, seed=20),
            _pair(calibrations, start=1.4, seed=30),
            _pair(calibrations, start=1.6, contacts=(True, False), seed=40),
            _pair(calibrations, start=1.8, contacts=(True, False), seed=50),
        ]
    )

    class FakeSource:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def calibrate_unloaded(self, *, timeout_s: float):
            return calibrations

        def collect_classifier_windows(self, not_before_s: float, samples: int, timeout: float):
            assert samples == 5
            return next(globals_iter)

    globals_iter = samples
    monkeypatch.setattr(module, "GF220LiveSource", FakeSource)
    monkeypatch.setattr(
        module,
        "_run_move",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("preflight contact must prevent MoveIt")
        ),
    )
    output = tmp_path / "execute_fake"
    args = module.parse_args(
        [
            "--target-grasp-position-m",
            "0.5",
            "0.1",
            "0.2",
            "--target-grasp-orientation-xyzw",
            "0",
            "0",
            "0",
            "1",
            "--approach-axis-base",
            "0",
            "0",
            "1",
            "--remaining-distance-m",
            "0.012",
            "--maximum-steps",
            "3",
            "--output-dir",
            str(output),
            "--execute",
            "--assert-pads-unloaded",
            "--accept-arm-only-collision-model",
        ]
    )
    assert module.run(args) == 0
    report = json.loads((output / "run_report.json").read_text())
    assert report["status"] == "one_pad_contact_hold"
    assert report["hardware_commands_dispatched"] is False
    assert report["steps"] == []
    assert (output / "contact_preflight.json").is_file()
    assert (output / "contact_confirmation.json").is_file()


def test_move_command_audit_cap_is_only_half_millimetre_above_step(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run_test_tube_contact_approach.py"
    spec = importlib.util.spec_from_file_location("contact_approach_command_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(
        [
            "--target-grasp-position-m",
            "0.5",
            "0.1",
            "0.2",
            "--target-grasp-orientation-xyzw",
            "0",
            "0",
            "0",
            "1",
            "--approach-axis-base",
            "0",
            "0",
            "1",
            "--remaining-distance-m",
            "0.004",
        ]
    )
    command = module._move_command(
        args,
        target_position=np.array([0.5, 0.1, 0.2]),
        target_orientation=(0.0, 0.0, 0.0, 1.0),
        link8_to_tcp_position=[0.0, 0.0, 0.19],
        link8_to_tcp_orientation=(0.0, 0.0, 0.0, 1.0),
        output=tmp_path / "move.json",
    )
    index = command.index("--max-translation-m")
    assert float(command[index + 1]) == pytest.approx(0.0045)
