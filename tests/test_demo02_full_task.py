from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo2gripper.autonomous_pick import PickStage
from scripts.run_demo02_full_task import (
    FullTaskStage,
    _append_stage,
    _pick_command,
    _pick_report_ready,
    _pose_sequence_command,
    pose_sequence_result_succeeded,
)


def normalized(tmp_path: Path) -> dict:
    transform = np.eye(4)
    transform[2, 3] = 0.190
    transform_path = tmp_path / "T_link8_tcp.npy"
    np.save(transform_path, transform)
    return {
        "pick_config": tmp_path / "pick_config.json",
        "commands": {
            "pick": ["python", "scripts/run_autonomous_pick.py"],
            "pose_sequence": [
                "scripts/run_ros_python.sh",
                "scripts/moveit_cartesian_pose_sequence.py",
            ],
        },
        "pose_sequence_max_step_m": 0.005,
        "pose_sequence_speed_m_s": 0.015,
        "pick": {
            "t_link8_tcp": transform_path,
            "velocity_scale": 0.03,
            "acceleration_scale": 0.03,
            "min_joint_limit_margin_rad": 0.05,
            "max_joint_sweep_rad": 2.0,
            "max_total_joint_sweep_rad": 7.0,
        },
    }


def sequence_result(*, mode: str = "execute") -> dict:
    return {
        "schema": "demo2gripper.moveit_cartesian_pose_sequence.v1",
        "mode": mode,
        "moveit_error_code": 1,
        "path_fraction": 1.0,
        "trajectory": {"point_count": 8},
        "trajectory_audit": {"accepted": True},
        "planning_succeeded": True,
        "execution_policy": "plan_then_audit_then_execute_exact_trajectory",
        "execution_dispatched": True,
        "execution_status": 4,
        "execution_error_code": 1,
        "execution_succeeded": True,
    }


def test_pick_command_only_enables_physical_execution_with_both_flags(
    tmp_path: Path,
) -> None:
    config = normalized(tmp_path)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "pick"

    review = _pick_command(
        config, output_dir=output_dir, input_dir=input_dir, execute=False
    )
    execute = _pick_command(
        config, output_dir=output_dir, input_dir=input_dir, execute=True
    )

    assert review[:2] == ["python", "scripts/run_autonomous_pick.py"]
    assert review[review.index("--config") + 1] == str(config["pick_config"])
    assert review[review.index("--input-dir") + 1] == str(input_dir.resolve())
    assert "--execute" not in review
    assert "--accept-arm-only-collision-model" not in review
    assert execute[-2:] == ["--execute", "--accept-arm-only-collision-model"]


def test_pose_sequence_command_preserves_full_trajectory_audit_and_acknowledgement(
    tmp_path: Path,
) -> None:
    config = normalized(tmp_path)
    input_path = tmp_path / "lid_open.json"
    output_path = tmp_path / "result.json"

    command = _pose_sequence_command(
        config, input_path, output=output_path, execute=True
    )

    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert command[:2] == [
        "scripts/run_ros_python.sh",
        "scripts/moveit_cartesian_pose_sequence.py",
    ]
    assert value("--min-joint-limit-margin-rad") == "0.05"
    assert value("--max-joint-sweep-rad") == "2.0"
    assert value("--max-total-joint-sweep-rad") == "7.0"
    assert value("--max-cartesian-speed-m-s") == "0.015"
    assert value("--base-frame") == "fr3_link0"
    assert command[-2:] == ["--execute", "--accept-arm-only-collision-model"]

    review = _pose_sequence_command(
        config, input_path, output=output_path, execute=False
    )
    assert "--execute" not in review
    assert "--accept-arm-only-collision-model" not in review


def test_pose_sequence_result_requires_complete_audited_exact_execution() -> None:
    accepted = sequence_result()
    assert pose_sequence_result_succeeded(accepted, executed=True)

    for key, rejected_value in (
        ("path_fraction", 0.999),
        ("planning_succeeded", False),
        ("execution_dispatched", False),
        ("execution_succeeded", False),
    ):
        rejected = dict(accepted)
        rejected[key] = rejected_value
        assert not pose_sequence_result_succeeded(rejected, executed=True)

    plan_only = sequence_result(mode="plan_only")
    assert pose_sequence_result_succeeded(plan_only, executed=False)
    assert not pose_sequence_result_succeeded(plan_only, executed=True)


def test_continuation_is_gated_on_complete_executed_pick() -> None:
    stages = [
        {"name": stage.value, "status": "passed"}
        for stage in PickStage
    ]
    report = {
        "status": "succeeded",
        "object_state": "held_after_robot_specific_short_lift",
        "stages": stages,
    }

    assert _pick_report_ready(report, executed=True) == (True, "accepted")

    failed_lift = json.loads(json.dumps(report))
    failed_lift["stages"][-1]["status"] = "failed"
    assert _pick_report_ready(failed_lift, executed=True)[0] is False

    unverified_hold = dict(report)
    unverified_hold["object_state"] = "unknown"
    assert _pick_report_ready(unverified_hold, executed=True)[0] is False

    review = {
        "status": "review_ready",
        "stages": stages[:3],
    }
    assert _pick_report_ready(review, executed=False) == (True, "accepted")
    assert _pick_report_ready(review, executed=True)[0] is False


def test_full_stage_report_cannot_skip_a_stage(tmp_path: Path) -> None:
    report_path = tmp_path / "stage_report.json"
    report = {
        "stages": [
            {"name": stage.value, "status": "passed"}
            for stage in PickStage
        ],
        "updated_at": "test",
    }

    with pytest.raises(ValueError, match="expected TRANSPORT"):
        _append_stage(
            report,
            report_path,
            FullTaskStage.PLACE,
            status="passed",
            details={},
            started_at="test",
        )

    _append_stage(
        report,
        report_path,
        FullTaskStage.TRANSPORT,
        status="passed",
        details={"guard": "object hold verified"},
        started_at="test",
    )
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["last_completed_stage"] == "TRANSPORT"
    assert written["stages"][-1]["details"]["guard"] == "object hold verified"
