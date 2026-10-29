from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from demo2gripper.gf220_live import GF220Calibration, GF220ClassifierWindows
from demo2gripper.tactile_contact_classifier import FingerRimClassifierWindow
from demo2gripper.tactile_rim_probe import TimedArrayWindow
from demo2gripper.test_tube_contact_approach import fit_unloaded_contact_profile
from demo2gripper.test_tube_tactile_close import (
    evaluate_tactile_close_window,
    make_staged_close_schedule,
)


def _calibration(serial: str, value: int) -> GF220Calibration:
    return GF220Calibration.create(
        serial=serial,
        warp_sha256=("01" if serial == "a" else "02") * 32,
        unloaded_warped=np.full((64, 64, 3), value, dtype=np.uint8),
    )


def _finger(
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
        warped[:, 22:42, 25:40] = np.clip(
            warped[:, 22:42, 25:40].astype(int) + 55, 0, 255
        ).astype(np.uint8)
        diff[:, 22:42, 25:40] = 205
    sdk = start + np.arange(5) * 0.02
    host = 100.0 + sdk
    valid = np.ones(5, dtype=bool)
    return FingerRimClassifierWindow(
        calibration_id,
        TimedArrayWindow(warped, sdk, host, valid),
        TimedArrayWindow(diff, sdk, host, valid),
        None,
    )


def _pair(
    calibrations: tuple[GF220Calibration, GF220Calibration],
    *,
    start: float,
    contacts: tuple[bool, bool] = (False, False),
    seed: int = 1,
) -> GF220ClassifierWindows:
    windows = tuple(
        _finger(
            calibration.calibration_id,
            start=start + index * 0.001,
            base=90 + index * 10,
            contact=contacts[index],
            seed=seed + index,
        )
        for index, calibration in enumerate(calibrations)
    )
    return GF220ClassifierWindows(
        sensor_names=("left", "right"),
        serials=("a", "b"),
        windows=(windows[0], windows[1]),
        start_skew_s=0.001,
        end_skew_s=0.001,
        collected_monotonic_s=200.0 + start,
    )


def _profile():
    calibrations = (_calibration("a", 90), _calibration("b", 100))
    profile = fit_unloaded_contact_profile(
        session_id="close-session",
        calibrations=calibrations,
        windows=tuple(
            _pair(calibrations, start=1.0 + index * 0.2, seed=10 + index * 10)
            for index in range(3)
        ),
    )
    return calibrations, profile


def test_close_schedule_uses_five_then_two_millimetre_caps() -> None:
    schedule = make_staged_close_schedule(49.4, minimum_opening_mm=10.0)
    assert schedule[:5] == pytest.approx((44.4, 39.4, 34.4, 29.4, 25.0))
    assert schedule[-1] == pytest.approx(10.0)
    coarse = np.diff(np.asarray((49.4, *schedule[:5])))
    fine = np.diff(np.asarray(schedule[4:]))
    assert np.all(np.abs(coarse) <= 5.0 + 1e-9)
    assert np.all(np.abs(fine) <= 2.0 + 1e-9)


def test_two_bilateral_windows_are_required_for_grasp_ready() -> None:
    calibrations, profile = _profile()
    first = evaluate_tactile_close_window(
        _pair(calibrations, start=2.0, contacts=(True, True), seed=100), profile
    )
    assert first.action == "hold_confirm"
    assert not first.grasp_ready
    second = evaluate_tactile_close_window(
        _pair(calibrations, start=2.2, contacts=(True, True), seed=120),
        profile,
        first.next_memory,
    )
    assert second.action == "grasp_ready"
    assert second.grasp_ready


def test_persistent_same_single_pad_reports_needs_centering() -> None:
    calibrations, profile = _profile()
    first = evaluate_tactile_close_window(
        _pair(calibrations, start=2.0, contacts=(True, False), seed=200), profile
    )
    assert first.action == "hold_confirm"
    second = evaluate_tactile_close_window(
        _pair(calibrations, start=2.2, contacts=(True, False), seed=220),
        profile,
        first.next_memory,
    )
    assert second.action == "needs_centering"
    assert second.needs_centering
    assert not second.grasp_ready


def test_switching_single_pad_does_not_fake_persistence() -> None:
    calibrations, profile = _profile()
    first = evaluate_tactile_close_window(
        _pair(calibrations, start=2.0, contacts=(True, False), seed=300), profile
    )
    switched = evaluate_tactile_close_window(
        _pair(calibrations, start=2.2, contacts=(False, True), seed=320),
        profile,
        first.next_memory,
    )
    assert switched.action == "hold_confirm"
    assert switched.next_memory.single_windows == 1


def _load_script():
    script = Path(__file__).resolve().parents[1] / "scripts/run_test_tube_tactile_close.py"
    spec = importlib.util.spec_from_file_location("test_tube_tactile_close_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_dry_run_opens_no_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()

    class ForbiddenSource:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dry run opened GF220")

    monkeypatch.setattr(module, "GF220LiveSource", ForbiddenSource)
    monkeypatch.setattr(module, "SerialModbusRTU", ForbiddenSource)
    output = tmp_path / "dry"
    args = module.parse_args(["--output-dir", str(output)])
    assert module.run(args) == 0
    report = json.loads((output / "run_report.json").read_text())
    assert report["status"] == "dry_run_complete"
    assert report["hardware_command_attempted"] is False
    assert report["configured_opening_schedule_mm"][0] == pytest.approx(23.0)


def test_fake_execute_finishes_after_one_close_and_bilateral_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    calibrations = (_calibration("a", 90), _calibration("b", 100))
    window_iter = iter(
        [
            _pair(calibrations, start=1.0, seed=10),
            _pair(calibrations, start=1.2, seed=20),
            _pair(calibrations, start=1.4, seed=30),
            _pair(calibrations, start=1.6, seed=40),
            _pair(calibrations, start=1.8, contacts=(True, True), seed=50),
            _pair(calibrations, start=2.0, contacts=(True, True), seed=60),
        ]
    )

    class FakeSource:
        def __init__(self, *args, **kwargs):
            pass

        def start(self, *, timeout_s: float):
            return self

        def close(self):
            return None

        def calibrate_unloaded(self, *, timeout_s: float):
            return calibrations

        def collect_classifier_windows(self, not_before_s: float, samples: int, timeout: float):
            return next(window_iter)

    initial_state = {
        "activated": True,
        "fault_code": 0,
        "opening_width_mm": 25.0,
        "object_status": {"grasped": False},
    }

    class FakeTransport:
        def __init__(self, *args, **kwargs):
            pass

        def read_feedback(self):
            return initial_state

        def close(self):
            pass

    commanded: list[float] = []

    def fake_command(transport, args, opening_mm: float):
        commanded.append(opening_mm)
        return {
            **initial_state,
            "opening_width_mm": opening_mm,
        }

    monkeypatch.setattr(module, "GF220LiveSource", FakeSource)
    monkeypatch.setattr(module, "SerialModbusRTU", FakeTransport)
    monkeypatch.setattr(module, "feedback_payload", lambda value, command=None: value)
    monkeypatch.setattr(module, "_command_opening", fake_command)
    output = tmp_path / "execute"
    fake_port = tmp_path / "hande-port"
    fake_port.touch()
    args = module.parse_args(
        [
            "--output-dir",
            str(output),
            "--execute",
            "--assert-pads-unloaded",
            "--assert-tube-between-fingers",
            "--port",
            str(fake_port),
        ]
    )
    assert module.run(args) == 0
    report = json.loads((output / "run_report.json").read_text())
    assert report["status"] == "grasp_ready"
    assert report["final_action"] == "hold_grasp_no_extraction"
    assert len(report["steps"]) == 1
    assert commanded == [23.0]
    assert report["steps"][0]["bilateral_confirmation"]
