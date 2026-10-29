from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "retry_test_tube_shallow_grasp_live",
    ROOT / "scripts" / "retry_test_tube_shallow_grasp_live.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_probe_targets_are_exactly_five_and_ten_mm_along_axis() -> None:
    start = np.eye(4)
    axis = np.asarray([0.01, -0.02, -1.0])
    axis /= np.linalg.norm(axis)
    first, second = MODULE.probe_targets(start, axis)
    np.testing.assert_allclose(first[:3, 3], axis * 0.005)
    np.testing.assert_allclose(second[:3, 3], axis * 0.010)
    np.testing.assert_allclose(first[:3, :3], np.eye(3))
    np.testing.assert_allclose(second[:3, :3], np.eye(3))


@pytest.mark.parametrize("axis", ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
def test_probe_targets_reject_non_downward_or_zero_axis(axis: list[float]) -> None:
    with pytest.raises(ValueError):
        MODULE.probe_targets(np.eye(4), np.asarray(axis))


def test_centres_must_move_toward_pad_middle_on_both_sides() -> None:
    old = {"left": (100.0, 214.0), "right": (100.0, 9.0)}
    current = {
        "left": {
            "image_active": True,
            "center_uv": (100.0, 180.0),
            "image_height_px": 240,
        },
        "right": {
            "image_active": True,
            "center_uv": (100.0, 45.0),
            "image_height_px": 240,
        },
    }
    passed, improvement = MODULE.centres_moved_inward(old, current)
    assert passed
    assert improvement["left"] >= MODULE.MIN_INWARD_SHIFT_PX
    assert improvement["right"] >= MODULE.MIN_INWARD_SHIFT_PX


def test_one_edge_staying_outward_rejects_grasp() -> None:
    old = {"left": (100.0, 214.0), "right": (100.0, 9.0)}
    current = {
        "left": {
            "image_active": True,
            "center_uv": (100.0, 180.0),
            "image_height_px": 240,
        },
        "right": {
            "image_active": True,
            "center_uv": (100.0, 7.0),
            "image_height_px": 240,
        },
    }
    passed, _ = MODULE.centres_moved_inward(old, current)
    assert not passed


def test_live_start_accepts_restart_cleared_gobj_with_recorded_tactile_audit() -> None:
    source = MODULE.ShallowSource(
        task_dir=Path("/tmp/source"),
        task_id="task",
        terminal_tcp=np.eye(4),
        terminal_width_m=0.018,
        approach_axis_base=np.asarray([0.0, 0.0, -1.0]),
        old_centres_uv={"left": (0.0, 214.0), "right": (0.0, 9.0)},
        terminal_audit={"provenance": "immutable_prior_run_terminal_audit"},
        artifacts_sha256="0" * 64,
    )
    feedback = {
        "width_m": 0.018,
        "moving": False,
        "fault_free": True,
        "object_detected": False,
    }
    validated = MODULE.validate_live_start(source, np.eye(4), feedback)
    assert validated["width_m"] == 0.018
    assert validated["object_detected"] is False
    assert "restart_volatile" in validated["object_detected_role"]
    assert source.terminal_audit["provenance"] == "immutable_prior_run_terminal_audit"


def test_gf220_calibrates_before_first_collect(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Source:
        calibrated = False

        def calibrate_unloaded(self, *, timeout_s: float) -> str:
            assert timeout_s == 15.0
            events.append("calibrate")
            self.calibrated = True
            return "calibrations"

    source = Source()

    def collect(candidate: Source, *, frames: int, timeout_s: float) -> str:
        assert candidate is source
        assert candidate.calibrated
        assert frames == 5
        assert timeout_s == 1.0
        events.append("collect")
        return f"window-{len(events)}"

    monkeypatch.setattr(MODULE.contact, "_collect", collect)
    calibrations, windows = MODULE.calibrate_then_collect_unloaded(source)
    assert calibrations == "calibrations"
    assert len(windows) == 3
    assert events == ["calibrate", "collect", "collect", "collect"]


def test_live_start_still_requires_stationary_matching_width() -> None:
    source = MODULE.ShallowSource(
        task_dir=Path("/tmp/source"),
        task_id="task",
        terminal_tcp=np.eye(4),
        terminal_width_m=0.018,
        approach_axis_base=np.asarray([0.0, 0.0, -1.0]),
        old_centres_uv={"left": (0.0, 214.0), "right": (0.0, 9.0)},
        terminal_audit={},
        artifacts_sha256="0" * 64,
    )
    feedback = {
        "width_m": 0.018,
        "moving": True,
        "fault_free": True,
        "object_detected": False,
    }
    with pytest.raises(RuntimeError, match="no longer matches"):
        MODULE.validate_live_start(source, np.eye(4), feedback)


def test_open_resume_live_requires_matching_open_state() -> None:
    source = MODULE.ShallowSource(
        task_dir=Path("/tmp/source"),
        task_id="task",
        terminal_tcp=np.eye(4),
        terminal_width_m=0.018,
        approach_axis_base=np.asarray([0.0, 0.0, -1.0]),
        old_centres_uv={"left": (0.0, 214.0), "right": (0.0, 9.0)},
        terminal_audit={},
        artifacts_sha256="0" * 64,
    )
    resume = MODULE.OpenResumeSource(
        report_path=Path("/tmp/prior/run_report.json"),
        session_id="prior",
        opened_width_m=0.0494140625,
        report_sha256="1" * 64,
        error_message="sync timeout",
    )
    feedback = {
        "width_m": 0.0494140625,
        "moving": False,
        "fault_free": True,
        "open_confirmed": False,
        "object_detected": False,
    }
    validated = MODULE.validate_open_resume_live(
        source, resume, np.eye(4), feedback
    )
    assert validated["width_m"] == pytest.approx(0.0494140625)
    assert validated["open_confirmed"] is False
    assert "restart_volatile" in validated["open_confirmed_role"]


def test_open_resume_live_still_rejects_width_below_48mm() -> None:
    source = MODULE.ShallowSource(
        task_dir=Path("/tmp/source"),
        task_id="task",
        terminal_tcp=np.eye(4),
        terminal_width_m=0.018,
        approach_axis_base=np.asarray([0.0, 0.0, -1.0]),
        old_centres_uv={"left": (0.0, 214.0), "right": (0.0, 9.0)},
        terminal_audit={},
        artifacts_sha256="0" * 64,
    )
    resume = MODULE.OpenResumeSource(
        report_path=Path("/tmp/prior/run_report.json"),
        session_id="prior",
        opened_width_m=0.0494140625,
        report_sha256="1" * 64,
        error_message="sync timeout",
    )
    feedback = {
        "width_m": 0.0479,
        "moving": False,
        "fault_free": True,
        "open_confirmed": False,
        "object_detected": False,
    }
    with pytest.raises(RuntimeError, match="open resume state"):
        MODULE.validate_open_resume_live(source, resume, np.eye(4), feedback)


def test_gf220_pair_sync_timeout_gets_exactly_one_source_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    instances: list[object] = []

    class Source:
        def __init__(self, _path: Path, *, config: object) -> None:
            self.index = len(instances) + 1
            self.calibrated = False
            instances.append(self)
            assert config.max_pair_skew_s == 0.08
            assert config.max_frame_age_s == 0.4

        def start(self, *, timeout_s: float) -> None:
            assert timeout_s == 15.0
            events.append(f"start-{self.index}")

        def calibrate_unloaded(self, *, timeout_s: float) -> str:
            assert timeout_s == 15.0
            self.calibrated = True
            events.append(f"calibrate-{self.index}")
            return f"calibrations-{self.index}"

        def close(self) -> None:
            events.append(f"close-{self.index}")

    def collect(source: Source, *, frames: int, timeout_s: float) -> str:
        assert source.calibrated
        assert frames == 5
        assert timeout_s == 1.0
        events.append(f"collect-{source.index}")
        if source.index == 1:
            raise TimeoutError("could not collect synchronized GF220 windows")
        return f"window-{source.index}-{events.count(f'collect-{source.index}')}"

    monkeypatch.setattr(MODULE.contact, "GF220LiveSource", Source)
    monkeypatch.setattr(MODULE.contact, "_collect", collect)
    source, calibrations, windows, attempts = MODULE.start_calibrated_gf220(
        Path("/tmp/profile.json")
    )
    assert source is instances[1]
    assert calibrations == "calibrations-2"
    assert len(windows) == 3
    assert [item["status"] for item in attempts] == ["timeout", "ready"]
    assert len(instances) == 2
    assert events.count("close-1") == 1


def test_current_run_contract_builds_ten_mm_bounded_probe() -> None:
    run = ROOT / "runs/live/test_tube_autonomous_20261024_201608"
    if not run.is_dir():
        pytest.skip("current reviewed real-run artifact is not present")
    source = MODULE.load_shallow_source(run)
    first, second = MODULE.probe_targets(
        source.terminal_tcp, source.approach_axis_base
    )
    assert MODULE.pose_error(source.terminal_tcp, first)[0] == pytest.approx(0.005)
    assert MODULE.pose_error(source.terminal_tcp, second)[0] == pytest.approx(0.010)
    assert len(source.old_centres_uv) == 2


def test_current_failed_run_is_exact_post_open_resume_contract() -> None:
    source_run = ROOT / "runs/live/test_tube_autonomous_20261024_201608"
    resume_run = ROOT / "runs/live/test_tube_shallow_live_retry_20261024_202602"
    config = ROOT / "configs/test_tube_closed_loop.demo07.json"
    if not source_run.is_dir() or not resume_run.is_dir():
        pytest.skip("reviewed source/resume artifacts are not present")
    source = MODULE.load_shallow_source(source_run)
    resume = MODULE.load_open_resume_source(
        resume_run,
        source=source,
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
    )
    assert resume.opened_width_m == pytest.approx(0.0494140625)
    assert "synchronized GF220" in resume.error_message

