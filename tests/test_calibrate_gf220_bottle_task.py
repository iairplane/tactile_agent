from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import calibrate_gf220_bottle_task as calibration


class _FakeSource:
    instances: list[_FakeSource] = []

    def __init__(self, *args, **kwargs) -> None:
        self.closed = False
        self.__class__.instances.append(self)

    def start(self, *, timeout_s: float) -> None:
        return None

    def calibrate_unloaded(self, *, timeout_s: float) -> list[object]:
        return [
            SimpleNamespace(
                serial="left",
                calibration_id="left-cal",
                warp_sha256="left-warp",
                baseline_sha256="left-baseline",
            ),
            SimpleNamespace(
                serial="right",
                calibration_id="right-cal",
                warp_sha256="right-warp",
                baseline_sha256="right-baseline",
            ),
        ]

    def close(self) -> None:
        self.closed = True


def _run_until_descent_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    recovery_open_fails: bool = False,
    recovery_retract_fails: bool = False,
) -> list[str]:
    events: list[str] = []
    _FakeSource.instances.clear()
    monkeypatch.setattr(calibration, "GF220LiveSource", _FakeSource)
    monkeypatch.setattr(
        calibration,
        "sample",
        lambda source, out, label: {"sensors": {}},
    )

    def fake_gripper(out: Path, label: str, opening_mm: float) -> None:
        events.append(f"gripper:{label}")
        if label == "recovery_open" and recovery_open_fails:
            raise RuntimeError("recovery open failed")

    def fake_move_sequence(
        out: Path, label: str, poses: list[dict[str, object]]
    ) -> None:
        events.append(f"move:{label}")
        if label == "descend_around_cap":
            raise RuntimeError("original descent failure")
        if label == "recovery_retract" and recovery_retract_fails:
            raise RuntimeError("recovery retract failed")

    monkeypatch.setattr(calibration, "gripper", fake_gripper)
    monkeypatch.setattr(calibration, "move_sequence", fake_move_sequence)
    monkeypatch.setattr(
        sys,
        "argv",
        ["calibrate_gf220_bottle_task.py", str(tmp_path / "run")],
    )

    with pytest.raises(RuntimeError, match="original descent failure"):
        calibration.main()

    assert _FakeSource.instances[0].closed is True
    return events


def test_descent_dispatch_failure_opens_then_retracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _run_until_descent_failure(tmp_path, monkeypatch)

    assert events == [
        "gripper:open_before_calibration",
        "move:descend_around_cap",
        "gripper:recovery_open",
        "move:recovery_retract",
    ]


def test_recovery_failures_do_not_mask_descent_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _run_until_descent_failure(
        tmp_path,
        monkeypatch,
        recovery_open_fails=True,
        recovery_retract_fails=True,
    )

    assert events[-2:] == ["gripper:recovery_open", "move:recovery_retract"]
    run = tmp_path / "run"
    assert (run / "recovery_open_error.json").is_file()
    assert (run / "recovery_retract_error.json").is_file()
