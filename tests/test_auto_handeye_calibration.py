from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_auto_handeye_calibration as runner


def _write_inputs(tmp_path: Path, *, pose_count: int = 2) -> dict[str, Path]:
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "schema": "demo2gripper.configured_ee_pose_sequence.v1",
                "base_frame": "fr3_link0",
                "waypoints": [
                    {
                        "position_m": [0.60 + 0.02 * index, 0.10, 0.28],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                    for index in range(pose_count)
                ],
            }
        ),
        encoding="utf-8",
    )
    board = tmp_path / "board.json"
    board.write_text("{}\n", encoding="utf-8")
    transform = np.eye(4)
    transform[2, 3] = 0.190
    transform_path = tmp_path / "T_link8_tcp.npy"
    np.save(transform_path, transform, allow_pickle=False)
    return {"targets": targets, "board": board, "transform": transform_path}


def _args(
    tmp_path: Path,
    *,
    execute: bool,
    pose_count: int = 2,
    target_samples: int = 1,
):
    inputs = _write_inputs(tmp_path, pose_count=pose_count)
    values = [
        "--targets",
        str(inputs["targets"]),
        "--output-dir",
        str(tmp_path / "run"),
        "--serial",
        "camera-123",
        "--board-metadata",
        str(inputs["board"]),
        "--session-id",
        "test-session",
        "--target-samples",
        str(target_samples),
        "--t-link8-configured-ee",
        str(inputs["transform"]),
        "--moveit-runner",
        "/bin/true",
        "--moveit-script",
        "/bin/true",
        "--capture-python",
        sys.executable,
        "--capture-script",
        "/bin/true",
        "--settle-s",
        "0",
    ]
    if execute:
        values.extend(("--execute", "--accept-arm-only-collision-model"))
    return runner.build_parser().parse_args(values)


def _motion_payload(*, execute: bool, accepted: bool = True) -> dict:
    payload = {
        "schema": "demo2gripper.moveit_pose_goal.v2",
        "mode": "execute" if execute else "plan_only",
        "moveit_error_code": 1,
        "trajectory": {"point_count": 4},
        "trajectory_audit": {"accepted": accepted},
        "planning_succeeded": accepted,
        "execution_dispatched": execute and accepted,
    }
    if execute and accepted:
        payload.update(
            {
                "execution_policy": "plan_then_audit_then_execute_exact_trajectory",
                "execution_status": 4,
                "execution_error_code": 1,
                "execution_succeeded": True,
            }
        )
    return payload


def _calibration_endpoint_payload() -> dict:
    payload = _motion_payload(execute=True)
    payload.update(
        {
            "input_target_frame": "configured-ee",
            "moveit_planning_succeeded": True,
            "trajectory_audit": {"accepted": True, "violations": []},
            "execution_succeeded": False,
            "execution_failure_reason": None,
            "measurement_error": None,
            "measured_end_position_base_m": [0.46, 0.13, 0.39],
            "measured_end_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "measured_end_configured_ee_position_base_m": [0.56, 0.18, 0.24],
            "measured_end_configured_ee_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "measured_target_position_error_m": 0.0047,
            "measured_target_rotation_error_rad": 0.033,
            "measured_input_target_position_error_m": 0.0068,
            "measured_input_target_rotation_error_rad": 0.033,
        }
    )
    return payload


def _write_motion(command: list[str], payload: dict) -> None:
    output = Path(command[command.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload), encoding="utf-8")


def _write_sample(command: list[str]) -> None:
    output = Path(command[command.index("--output-dir") + 1])
    output.mkdir(parents=True)
    for name in ("T_base_flange.npy", "T_camera_board.npy"):
        np.save(output / name, np.eye(4), allow_pickle=False)
    for name in ("rgb.png", "overlay.png"):
        (output / name).write_bytes(b"test")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "demo2gripper.eye_to_hand_sample.v1",
                "session_id": "test-session",
                "valid_for_handeye_calibration": True,
            }
        ),
        encoding="utf-8",
    )


def _completed(command: list[str], returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, "", "")


def test_plan_only_is_default_and_never_captures(tmp_path, monkeypatch) -> None:
    args = _args(tmp_path, execute=False, target_samples=24)
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, log_prefix):
        del cwd, log_prefix
        command = list(command)
        commands.append(command)
        _write_motion(command, _motion_payload(execute=False))
        return _completed(command)

    monkeypatch.setattr(runner, "_run_process", fake_run)
    assert runner.run(args) == 0
    assert len(commands) == 2
    assert all("--output" in command for command in commands)
    assert all("--output-dir" not in command for command in commands)
    assert all("--execute" not in command for command in commands)
    report = json.loads((tmp_path / "run/run_report.json").read_text())
    assert report["mode"] == "plan_only"
    assert report["status"] == "review_ready"
    assert report["planned_target_count"] == 2


def test_execute_captures_only_after_strict_motion_success(tmp_path, monkeypatch) -> None:
    args = _args(tmp_path, execute=True)
    args.position_tolerance_m = 0.004
    args.orientation_tolerance_rad = 0.06
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, log_prefix):
        del cwd, log_prefix
        command = list(command)
        commands.append(command)
        if "--output" in command:
            _write_motion(command, _motion_payload(execute=True))
        else:
            _write_sample(command)
        return _completed(command)

    monkeypatch.setattr(runner, "_run_process", fake_run)
    assert runner.run(args) == 0
    assert len(commands) == 2
    move, capture = commands
    assert move[-2:] == ["--execute", "--accept-arm-only-collision-model"]
    assert move[move.index("--velocity-scale") + 1] == "0.02"
    assert move[move.index("--position-tolerance-m") + 1] == "0.004"
    assert move[move.index("--orientation-tolerance-rad") + 1] == "0.06"
    assert move[move.index("--min-joint-limit-margin-rad") + 1] == "0.1"
    assert "--ros-robot-state-topic" in capture
    report = json.loads((tmp_path / "run/run_report.json").read_text())
    assert report["status"] == "complete"
    assert report["accepted_sample_count"] == 1
    assert report["motion"]["position_tolerance_m"] == 0.004
    assert report["motion"]["orientation_tolerance_rad"] == 0.06


def test_calibration_accepts_bounded_endpoint_miss_and_captures_actual_pose(
    tmp_path, monkeypatch
) -> None:
    args = _args(tmp_path, execute=True)
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, log_prefix):
        del cwd, log_prefix
        command = list(command)
        commands.append(command)
        if "--output" in command:
            _write_motion(command, _calibration_endpoint_payload())
            return _completed(command, 9)
        _write_sample(command)
        return _completed(command)

    monkeypatch.setattr(runner, "_run_process", fake_run)
    assert runner.run(args) == 0
    assert len(commands) == 2
    report = json.loads((tmp_path / "run/run_report.json").read_text())
    assert report["status"] == "complete"
    assert report["accepted_sample_count"] == 1
    assert report["attempts"][0]["motion_acceptance"] == "actual_pose_for_calibration"
    assert report["attempts"][0]["measured_input_target_position_error_m"] == 0.0068


def test_calibration_rejects_configured_ee_miss_over_endpoint_gate() -> None:
    payload = _calibration_endpoint_payload()
    payload["measured_input_target_position_error_m"] = 0.030001
    assert runner.calibration_execution_acceptable(payload) is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("position_tolerance_m", 0.0),
        ("position_tolerance_m", 0.011),
        ("orientation_tolerance_rad", float("inf")),
        ("orientation_tolerance_rad", 0.101),
    ],
)
def test_pose_tolerances_are_positive_finite_and_capped(
    tmp_path, name: str, value: float
) -> None:
    args = _args(tmp_path, execute=False)
    setattr(args, name, value)
    with pytest.raises(ValueError, match=name):
        runner.validate_args(args)


def test_undispatched_motion_failure_skips_target_without_relaxing_gates(
    tmp_path, monkeypatch
) -> None:
    args = _args(tmp_path, execute=True)
    commands: list[list[str]] = []
    motion_count = 0

    def fake_run(command, *, cwd, log_prefix):
        nonlocal motion_count
        del cwd, log_prefix
        command = list(command)
        commands.append(command)
        if "--output" in command:
            accepted = motion_count > 0
            motion_count += 1
            _write_motion(command, _motion_payload(execute=True, accepted=accepted))
            return _completed(command, 0 if accepted else 9)
        _write_sample(command)
        return _completed(command)

    monkeypatch.setattr(runner, "_run_process", fake_run)
    assert runner.run(args) == 0
    moves = [command for command in commands if "--output" in command]
    captures = [command for command in commands if "--output-dir" in command]
    assert len(moves) == 2
    assert len(captures) == 1
    for flag in (
        "--velocity-scale",
        "--acceleration-scale",
        "--min-joint-limit-margin-rad",
        "--max-joint-sweep-rad",
        "--max-total-joint-sweep-rad",
    ):
        assert moves[0][moves[0].index(flag) + 1] == moves[1][moves[1].index(flag) + 1]
    report = json.loads((tmp_path / "run/run_report.json").read_text())
    assert report["attempts"][0]["motion_status"] == "skipped"
    assert report["attempts"][1]["capture_status"] == "accepted"
    assert report["safety"]["thresholds_relaxed_after_failure"] is False


def test_capture_failure_skips_sample_and_continues(tmp_path, monkeypatch) -> None:
    args = _args(tmp_path, execute=True)
    capture_count = 0

    def fake_run(command, *, cwd, log_prefix):
        nonlocal capture_count
        del cwd, log_prefix
        command = list(command)
        if "--output" in command:
            _write_motion(command, _motion_payload(execute=True))
            return _completed(command)
        capture_count += 1
        if capture_count == 1:
            return _completed(command, 1)
        _write_sample(command)
        return _completed(command)

    monkeypatch.setattr(runner, "_run_process", fake_run)
    assert runner.run(args) == 0
    report = json.loads((tmp_path / "run/run_report.json").read_text())
    assert report["attempts"][0]["capture_status"] == "skipped"
    assert report["attempts"][1]["capture_status"] == "accepted"
    assert report["accepted_sample_count"] == 1


def test_dispatched_execution_failure_aborts_before_another_target(tmp_path, monkeypatch) -> None:
    args = _args(tmp_path, execute=True)
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, log_prefix):
        del cwd, log_prefix
        command = list(command)
        commands.append(command)
        payload = _motion_payload(execute=True)
        payload["execution_succeeded"] = False
        _write_motion(command, payload)
        return _completed(command, 9)

    monkeypatch.setattr(runner, "_run_process", fake_run)
    assert runner.run(args) == 3
    assert len(commands) == 1
    report = json.loads((tmp_path / "run/run_report.json").read_text())
    assert report["status"] == "aborted_after_motion_failure"
    assert report["attempts"][0]["motion_status"] == "failed_after_dispatch"


def test_missing_execution_artifact_aborts_with_unknown_dispatch_state(
    tmp_path, monkeypatch
) -> None:
    args = _args(tmp_path, execute=True)

    def fake_run(command, *, cwd, log_prefix):
        del cwd, log_prefix
        return _completed(list(command), 7)

    monkeypatch.setattr(runner, "_run_process", fake_run)
    assert runner.run(args) == 3
    report = json.loads((tmp_path / "run/run_report.json").read_text())
    assert report["status"] == "aborted_after_motion_failure"
    assert report["attempts"][0]["motion_status"] == "failed_with_unknown_dispatch_state"
