from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demo2gripper.hande_modbus_cli import ACTIVE_STOP_ACTION


def _load_script():
    script = Path(__file__).resolve().parents[1] / "scripts/run_test_tube_continuous_close.py"
    spec = importlib.util.spec_from_file_location("run_test_tube_continuous_close", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_never_opens_hardware(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()

    class Forbidden:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dry run opened hardware")

    monkeypatch.setattr(module, "GF220LiveSource", Forbidden)
    monkeypatch.setattr(module, "SerialModbusRTU", Forbidden)
    output = tmp_path / "dry"
    args = module.parse_args(["--output-dir", str(output)])
    assert module.run(args) == 0
    report = json.loads((output / "run_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "dry_run_complete"
    assert report["hardware_commands_dispatched"] is False


def test_execute_requires_explicit_scene_assertions() -> None:
    module = _load_script()
    with pytest.raises(ValueError, match="assert-pads-unloaded"):
        module.parse_args(["--execute"])


def test_stop_clears_goto_and_confirms_stationary(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()

    class FakeTransport:
        def __init__(self) -> None:
            self.action_values: list[int] = []
            self.feedback = iter(
                (
                    SimpleNamespace(
                        activation_bit=1,
                        go_to_bit=0,
                        activation_status=3,
                        object_status=0,
                        fault_code=0,
                        requested_count=200,
                        position_count=100,
                        current=4,
                    ),
                    SimpleNamespace(
                        activation_bit=1,
                        go_to_bit=0,
                        activation_status=3,
                        object_status=0,
                        fault_code=0,
                        requested_count=200,
                        position_count=100,
                        current=4,
                    ),
                    SimpleNamespace(
                        activation_bit=1,
                        go_to_bit=0,
                        activation_status=3,
                        object_status=0,
                        fault_code=0,
                        requested_count=200,
                        position_count=100,
                        current=4,
                    ),
                )
            )

        def stop_motion(self) -> None:
            self.action_values.append(ACTIVE_STOP_ACTION)

        def read_feedback(self):
            return next(self.feedback)

    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    transport = FakeTransport()
    state, reads = module._stop_and_confirm(transport, timeout_s=0.5)
    assert transport.action_values == [0x0100]
    assert reads == 3
    assert state["go_to_bit"] == 0
    assert state["opening_width_mm"] == pytest.approx(50.0 - 100 * 0.1953125)


def test_post_stop_requires_two_bilateral_windows() -> None:
    module = _load_script()
    bilateral_one = SimpleNamespace(
        valid=True,
        grasp_ready=False,
        needs_centering=False,
        active_serials=("a", "b"),
    )
    bilateral_two = SimpleNamespace(
        valid=True,
        grasp_ready=True,
        needs_centering=False,
        active_serials=("a", "b"),
    )
    assert module._post_stop_result([bilateral_one, bilateral_two]) == (
        "grasp_ready",
        "hold_grasp_no_extraction",
    )


def test_continuous_close_limits_force_and_speed() -> None:
    module = _load_script()
    with pytest.raises(ValueError, match="force"):
        module.parse_args(["--force", "21"])
    with pytest.raises(ValueError, match="speed"):
        module.parse_args(["--speed", "71"])
