from __future__ import annotations

import importlib.util
import json
import shlex
from pathlib import Path

import numpy as np
import pytest


def _module(script):
    path = Path(__file__).resolve().parents[1] / "scripts" / script
    spec = importlib.util.spec_from_file_location(script.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prior_run(tmp_path):
    root = tmp_path / "prior run"
    task = root / "task"
    (task / "steps").mkdir(parents=True)
    (task / "run_report.json").write_text(json.dumps({
        "task_id": "test-task", "status": "held", "final_state": "HOLD", "step_count": 4,
    }))
    steps = []
    for index, z in enumerate((.25, .23, .21, .21)):
        pose = np.eye(4)
        pose[:3, 3] = [.5, .1, z]
        final = index == 3
        steps.append({
            "index": index,
            "decision": {
                "previous_state": "SIDE_GRASP", "next_state": "HOLD" if final else "SIDE_GRASP",
                "action": {"kind": "stop_and_hold" if final else "servo_tcp",
                           "reason": "SIDE_GRASP timeout exceeded" if final else "approach"},
            },
            "dispatch": {"status": "held" if final else "completed", "receipt": (
                {"held": True, "arm": {"accepted": True}} if final else {"accepted": True}
            )},
            "observation": {
                "robot": {"T_base_tcp": pose.tolist(), "fault": None, "motion_complete": True},
                "gripper": {"width_m": .049, "moving": False, "fault_free": True,
                            "object_detected": False},
                "tactile": {"valid": True, "contact_present": False, "bilateral_contact": False},
            },
        })
    for index, step in enumerate(steps):
        (task / "steps" / f"{index:04}.json").write_text(json.dumps(step))
    return root, steps


def _save_steps(root, steps):
    for index, step in enumerate(steps):
        (root / "task/steps" / f"{index:04}.json").write_text(json.dumps(step))


def test_recovery_uses_only_actual_reverse_poses_and_keeps_source_unchanged(prior_run):
    module = _module("run_test_tube_closed_loop.py")
    root, steps = prior_run
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    path, provenance = module.load_startup_recovery(root, task_id="test-task")
    assert len(path) == 3
    assert provenance["retained_source_step_indices"] == [3, 1, 0]
    assert provenance["total_distance_m"] == pytest.approx(.04)
    assert provenance["maximum_rise_m"] == pytest.approx(.04)
    for actual, index in zip(path, (3, 1, 0)):
        np.testing.assert_array_equal(actual, steps[index]["observation"]["robot"]["T_base_tcp"])
    assert provenance["gripper_release_commanded"] is False
    assert provenance["terminal_feedback_motion_complete_before_hold"] is True
    assert provenance["terminal_stop_and_hold_acknowledged"] is True
    assert provenance["live_stationary_start_required"] is True
    assert all(p.read_bytes() == data for p, data in before.items())


def test_recovery_accepts_pre_stop_motion_snapshot_after_acknowledged_hold(prior_run):
    module = _module("run_test_tube_closed_loop.py")
    root, steps = prior_run
    steps[-1]["observation"]["robot"]["motion_complete"] = False
    _save_steps(root, steps)

    path, provenance = module.load_startup_recovery(root, task_id="test-task")

    assert len(path) == 3
    assert provenance["terminal_feedback_motion_complete_before_hold"] is False
    assert provenance["terminal_stop_and_hold_acknowledged"] is True
    assert provenance["live_stationary_start_required"] is True


@pytest.mark.parametrize("invalid", [
    "task", "closed", "dispatch", "contact", "earlier_contact", "fault",
    "downward", "distance", "missing",
])
def test_recovery_rejects_ineligible_history(prior_run, invalid):
    module = _module("run_test_tube_closed_loop.py")
    root, steps = prior_run
    if invalid == "closed":
        steps[1]["decision"]["action"]["kind"] = "start_continuous_close"
    elif invalid == "dispatch":
        steps[0]["dispatch"]["status"] = "failed"
    elif invalid == "contact":
        steps[-1]["observation"]["tactile"]["contact_present"] = True
    elif invalid == "earlier_contact":
        steps[1]["observation"]["tactile"]["contact_present"] = True
    elif invalid == "fault":
        steps[-1]["observation"]["robot"]["fault"] = "hardware_fault"
    elif invalid == "downward":
        steps[1]["observation"]["robot"]["T_base_tcp"][2][3] = .205
    elif invalid == "distance":
        steps[0]["observation"]["robot"]["T_base_tcp"][2][3] = .34
    _save_steps(root, steps)
    if invalid == "missing":
        (root / "task/steps/0001.json").unlink()
    with pytest.raises(module.ClosedLoopContractError):
        module.load_startup_recovery(
            root, task_id="different-task" if invalid == "task" else "test-task"
        )


def test_recovery_cli_is_forwarded_as_one_quoted_argument(prior_run, tmp_path):
    root, _ = prior_run
    task = _module("run_test_tube_closed_loop.py")
    stack = _module("run_test_tube_live_stack.py")
    args = task.parse_args(["--config", "task.json", "--output-dir", "out",
                            "--recover-from-run", str(root)])
    assert args.recover_from_run == root and not args.execute
    args = stack.parse_args(["--stack", "stack.json", "--output-dir", "out",
                             "--recover-from-run", str(root)])
    assert args.recover_from_run == root and not args.execute
    command = stack._task_command(
        task={}, task_config=tmp_path / "task.json", task_output=tmp_path / "out",
        accept_arm_only_collision_model=True, recover_from_run=root,
    )
    tokens = shlex.split(command[2])
    assert tokens[tokens.index("--recover-from-run") + 1] == str(root.resolve())
