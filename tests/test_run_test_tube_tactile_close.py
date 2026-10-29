from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_script():
    script = Path(__file__).resolve().parents[1] / "scripts/run_test_tube_tactile_close.py"
    spec = importlib.util.spec_from_file_location("run_test_tube_tactile_close", script)
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


def test_execute_requires_both_scene_assertions() -> None:
    module = _load_script()
    with pytest.raises(ValueError, match="assert-pads-unloaded"):
        module.parse_args(["--execute"])
    with pytest.raises(ValueError, match="assert-tube-between-open-fingers"):
        module.parse_args(["--execute", "--assert-pads-unloaded"])


def test_schedule_is_descending_and_skips_reopening_targets() -> None:
    module = _load_script()
    assert module._remaining_schedule((42.0, 34.0, 28.0), 35.0) == (34.0, 28.0)
    with pytest.raises(ValueError, match="strictly descending"):
        module.parse_args(["--opening-schedule-mm", "30", "31"])
