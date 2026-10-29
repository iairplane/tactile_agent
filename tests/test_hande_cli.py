from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import demo2gripper.hande_cli as hande_cli
from demo2gripper.hande_cli import (
    HandECliError,
    execute_motion,
    feedback_payload,
    read_status,
    validate_motion_limits,
)


def feedback(*, object_status: int, closure_mm: float) -> SimpleNamespace:
    return SimpleNamespace(
        status=(1, 1, 3, object_status),
        position_mm=closure_mm,
        requested_position_mm=closure_mm,
        current=12,
        fault_code=0,
    )


def state(*, object_status: int, closure_mm: float) -> SimpleNamespace:
    return SimpleNamespace(
        connected=True,
        position=closure_mm / 50.0,
        position_mm=closure_mm,
        requested_position=closure_mm / 50.0,
        current=12,
        object_status=object_status,
        fault_code=0,
        timestamp=123.0,
    )


def test_status_path_is_strictly_read_only() -> None:
    calls: list[object] = []

    class ReadOnlyHand:
        def __init__(self, **kwargs):
            calls.append(("construct", kwargs))

        def read_feedback(self):
            calls.append("read_feedback")
            return feedback(object_status=3, closure_mm=0.0)

        def close(self):
            calls.append("close_transport")

        def stop(self):
            pytest.fail("status must not write a stop command")

        def move(self, *args, **kwargs):
            pytest.fail("status must not write a motion command")

        def initGripper(self, *args, **kwargs):  # noqa: N802 - vendor API spelling
            pytest.fail("status must not activate the Hand-E")

    result = read_status(
        port="/dev/serial/by-id/test",
        sdk_factory=ReadOnlyHand,
        modbus_timeout_s=0.25,
    )

    assert calls == [
        (
            "construct",
            {"port": "/dev/serial/by-id/test", "autoInit": False, "timeout": 0.25},
        ),
        "read_feedback",
        "close_transport",
    ]
    assert result["object_status"]["meaning"] == "motion_complete_no_object"
    assert result["object_status"]["fully_open"]


def test_cli_stdout_remains_one_json_object_when_sdk_prints(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    class NoisyReadOnlyHand:
        def __init__(self, **kwargs):
            print("serial retry diagnostic")

        def read_feedback(self):
            return feedback(object_status=3, closure_mm=0.0)

        def close(self):
            pass

    monkeypatch.setattr(
        hande_cli,
        "_load_hande_types",
        lambda _: (object, NoisyReadOnlyHand),
    )

    assert (
        hande_cli.main(
            [
                "status",
                "--port",
                "/dev/serial/by-id/test",
                "--lerobot-src",
                str(tmp_path),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"]
    assert captured.out.count("\n") == 1
    assert "serial retry diagnostic" in captured.err


def test_object_status_two_is_grasped_and_three_is_not() -> None:
    grasped = feedback_payload(feedback(object_status=2, closure_mm=24.0), command="close")
    empty = feedback_payload(feedback(object_status=3, closure_mm=50.0), command="close")

    assert grasped["object_status"] == {
        "code": 2,
        "meaning": "grasped_object_detected_closing",
        "grasped": True,
        "fully_closed_without_object": False,
        "fully_open": False,
    }
    assert empty["object_status"]["code"] == 3
    assert empty["object_status"]["meaning"] == "fully_closed_no_object"
    assert not empty["object_status"]["grasped"]
    assert empty["object_status"]["fully_closed_without_object"]


def test_measured_closed_endpoint_counts_as_empty_completion() -> None:
    measured_closed_mm = 47.656
    empty = feedback_payload(
        feedback(object_status=3, closure_mm=measured_closed_mm), command="close"
    )
    contact = feedback_payload(
        feedback(object_status=2, closure_mm=measured_closed_mm), command="close"
    )

    assert empty["object_status"]["meaning"] == "fully_closed_no_object"
    assert empty["object_status"]["fully_closed_without_object"]
    assert not empty["object_status"]["grasped"]
    assert contact["object_status"]["grasped"]
    assert not contact["object_status"]["fully_closed_without_object"]


@pytest.mark.parametrize(
    ("command", "terminal_state", "target"),
    [
        ("open", state(object_status=3, closure_mm=0.0), 0.0),
        ("close", state(object_status=2, closure_mm=23.0), 1.0),
        ("close", state(object_status=3, closure_mm=47.656), 1.0),
    ],
)
def test_motion_uses_corrected_driver_without_auto_activation_by_default(
    command: str,
    terminal_state: SimpleNamespace,
    target: float,
) -> None:
    instances = []

    class Driver:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.commands = []
            self.disconnected = False
            instances.append(self)

        def connect(self):
            return True

        def command(self, value):
            self.commands.append(value)

        def get_state(self):
            return terminal_state

        def disconnect(self):
            self.disconnected = True

    result = execute_motion(
        command=command,
        port="/dev/serial/by-id/test",
        driver_factory=Driver,
        speed=50,
        force=40,
        timeout_s=1.0,
        modbus_timeout_s=0.25,
        activate=False,
    )

    driver = instances[0]
    assert driver.options["auto_activate"] is False
    assert driver.options["binary_latch"] is False
    assert driver.options["speed"] == 50
    assert driver.options["force"] == 40
    assert driver.commands == [target]
    assert driver.disconnected
    assert result["object_status"]["grasped"] is (terminal_state.object_status == 2)


def test_inactive_driver_fails_before_sending_motion() -> None:
    class InactiveDriver:
        _connect_error = RuntimeError("Hand-E is not activated")

        def __init__(self, **kwargs):
            self.commands = []

        def connect(self):
            return False

        def command(self, value):
            self.commands.append(value)

        def disconnect(self):
            pass

    with pytest.raises(HandECliError, match="not activated"):
        execute_motion(
            command="close",
            port="/dev/serial/by-id/test",
            driver_factory=InactiveDriver,
            speed=50,
            force=40,
            timeout_s=1.0,
            modbus_timeout_s=0.25,
            activate=False,
        )


def test_absolute_opening_move_uses_normalized_closure_and_reports_target() -> None:
    instances = []

    class Driver:
        def __init__(self, **kwargs):
            self.commands = []
            instances.append(self)

        def connect(self):
            return True

        def command(self, value):
            self.commands.append(value)

        def get_state(self):
            return state(object_status=3, closure_mm=42.0)

        def disconnect(self):
            pass

    result = execute_motion(
        command="move",
        opening_mm=8.0,
        port="/dev/serial/by-id/test",
        driver_factory=Driver,
        speed=10,
        force=20,
        timeout_s=1.0,
        modbus_timeout_s=0.25,
        activate=False,
    )

    assert instances[0].commands == [pytest.approx(0.84)]
    assert result["requested_opening_width_mm"] == pytest.approx(8.0)
    assert result["requested_closure_mm"] == pytest.approx(42.0)


@pytest.mark.parametrize("opening_mm", [-0.1, 50.1, float("nan")])
def test_absolute_opening_move_rejects_invalid_target(opening_mm: float) -> None:
    with pytest.raises(HandECliError):
        execute_motion(
            command="move",
            opening_mm=opening_mm,
            port="/dev/serial/by-id/test",
            driver_factory=object,
            speed=10,
            force=20,
            timeout_s=1.0,
            modbus_timeout_s=0.25,
            activate=False,
        )


@pytest.mark.parametrize(("speed", "force"), [(101, 40), (50, 71), (-1, 40), (50, -1)])
def test_conservative_register_caps_are_enforced(speed: int, force: int) -> None:
    with pytest.raises(HandECliError):
        validate_motion_limits(speed, force)
