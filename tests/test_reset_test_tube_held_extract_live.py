from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reset_test_tube_held_extract_live",
    ROOT / "scripts" / "reset_test_tube_held_extract_live.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def pose(z: float, rotation_y_deg: float = 0.0) -> list[list[float]]:
    value = np.eye(4)
    angle = np.deg2rad(rotation_y_deg)
    value[:3, :3] = [
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ]
    value[:3, 3] = [0.51, 0.19, z]
    return value.tolist()


def observation(z: float, rotation_y_deg: float = 0.0) -> dict:
    return {
        "robot": {
            "T_base_tcp": pose(z, rotation_y_deg),
            "motion_complete": True,
            "fault": None,
        },
        "gripper": {
            "width_m": 0.018,
            "moving": False,
            "fault_free": True,
            "object_detected": False,
        },
        "tactile": {
            "valid": True,
            "contact_present": True,
            "bilateral_contact": True,
            "bilateral_contact_interior": True,
        },
    }


def write_run(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    steps_dir = task / "steps"
    steps_dir.mkdir(parents=True)
    specifications = (
        ("SIDE_GRASP", "EXTRACT", "servo_tcp", "completed", 0.160),
        ("EXTRACT", "EXTRACT", "servo_tcp", "completed", 0.200),
        ("EXTRACT", "LEVEL_HELD_TUBE", "servo_tcp", "completed", 0.240),
        ("LEVEL_HELD_TUBE", "HOLD", "stop_and_hold", "held", 0.241),
    )
    for index, (previous, next_state, kind, status, z) in enumerate(specifications):
        receipt = {"accepted": True}
        if kind == "stop_and_hold":
            receipt = {
                "held": True,
                "arm": {"accepted": True},
                "gripper": {"accepted": True},
            }
        step = {
            "index": index,
            "observation": observation(z),
            "decision": {
                "previous_state": previous,
                "next_state": next_state,
                "action": {"kind": kind},
            },
            "dispatch": {"status": status, "receipt": receipt},
        }
        (steps_dir / f"{index:04d}.json").write_text(json.dumps(step))
    report = {
        "schema": "demo2gripper.test_tube_closed_loop_report.v1",
        "mode": "execute",
        "status": "held",
        "final_state": "HOLD",
        "hardware_commands_dispatched": True,
        "task_id": "tube",
        "step_count": len(specifications),
        "last_step": str(steps_dir / "0003.json"),
    }
    (task / "run_report.json").write_text(json.dumps(report))
    return tmp_path


def test_loads_strict_level_hold_and_builds_reverse_measured_path(
    tmp_path: Path,
) -> None:
    source = MODULE.load_held_extract_source(write_run(tmp_path))
    assert source.held_width_m == pytest.approx(0.018)
    assert source.terminal_tcp[2, 3] == pytest.approx(0.241)
    assert source.contact_tcp[2, 3] == pytest.approx(0.160)
    assert source.reverse_path[0][2, 3] == pytest.approx(0.241)
    assert source.reverse_path[-1][2, 3] == pytest.approx(0.160)
    assert all(
        second[2, 3] <= first[2, 3] + 0.003
        for first, second in zip(source.reverse_path, source.reverse_path[1:])
    )


def test_rejects_terminal_hold_without_bilateral_interior_contact(
    tmp_path: Path,
) -> None:
    run = write_run(tmp_path)
    terminal = run / "task/steps/0003.json"
    payload = json.loads(terminal.read_text())
    payload["observation"]["tactile"]["bilateral_contact_interior"] = False
    terminal.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="bilateral LEVEL_HELD_TUBE"):
        MODULE.load_held_extract_source(run)


def test_live_gate_requires_unchanged_pose_and_width(tmp_path: Path) -> None:
    source = MODULE.load_held_extract_source(write_run(tmp_path))
    feedback = {
        "width_m": 0.018,
        "moving": False,
        "fault_free": True,
        "object_detected": False,
    }
    validated = MODULE.validate_live_start(source, source.terminal_tcp, feedback)
    assert validated["translation_error_m"] == pytest.approx(0.0)
    feedback["width_m"] = 0.030
    with pytest.raises(RuntimeError, match="Hand-E differs"):
        MODULE.validate_live_start(source, source.terminal_tcp, feedback)


def test_current_resident_hold_satisfies_reset_contract() -> None:
    run = ROOT / "runs/live/test_tube_autonomous_20261024_204745"
    if not run.is_dir():
        pytest.skip("reviewed resident held run is unavailable")
    source = MODULE.load_held_extract_source(run)
    assert source.held_width_m == pytest.approx(0.0181640625)
    assert source.terminal_tcp[2, 3] > source.contact_tcp[2, 3] + 0.075


def write_failed_closed_run(tmp_path: Path, *, error_serial: str) -> Path:
    task = tmp_path / "task"
    steps_dir = task / "steps"
    steps_dir.mkdir(parents=True)
    specifications = (
        ("SIDE_GRASP", "EXTRACT", 0.160, 0.0),
        ("EXTRACT", "EXTRACT", 0.200, 0.0),
        ("EXTRACT", "LEVEL_HELD_TUBE", 0.240, 0.0),
        ("LEVEL_HELD_TUBE", "LEVEL_HELD_TUBE", 0.241, 10.0),
        ("LEVEL_HELD_TUBE", "LEVEL_HELD_TUBE", 0.241, 20.0),
        ("LEVEL_HELD_TUBE", "LEVEL_HELD_TUBE", 0.241, 30.0),
    )
    for index, (previous, next_state, z, angle) in enumerate(specifications):
        step = {
            "index": index,
            "observation": observation(z, angle),
            "decision": {
                "previous_state": previous,
                "next_state": next_state,
                "action": {"kind": "servo_tcp"},
            },
            "dispatch": {
                "status": "completed",
                "receipt": {
                    "accepted": True,
                    "target": pose(z, angle + 10.0) if index == 5 else pose(z, angle),
                },
            },
        }
        (steps_dir / f"{index:04d}.json").write_text(json.dumps(step))
    error = f"GF220 acquisition failed for {error_serial}"
    report = {
        "schema": "demo2gripper.test_tube_closed_loop_report.v1",
        "mode": "execute",
        "status": "failed_closed",
        "final_state": "HOLD",
        "error": error,
        "hold_receipt": {
            "held": True,
            "arm": {"accepted": True, "reason": f"runner failure: {error}"},
            "gripper": {"accepted": True, "reason": f"runner failure: {error}"},
        },
        "hardware_commands_dispatched": True,
        "task_id": "tube",
        "step_count": len(specifications),
        "last_step": str(steps_dir / "0005.json"),
    }
    (task / "run_report.json").write_text(json.dumps(report))
    return tmp_path


def test_failed_closed_gf220_path_reverses_level_before_descent(
    tmp_path: Path,
) -> None:
    serials = ("left", "right")
    source = MODULE.load_held_extract_source(
        write_failed_closed_run(tmp_path, error_serial="left"),
        gf220_serials=serials,
    )
    assert source.source_status == "failed_closed"
    assert source.retained_step_indices == (-1, 5, 4, 3, 2, 1, 0)
    assert source.level_return_step_indices == (5, 4, 3)
    assert MODULE.retry.pose_error(source.terminal_tcp, np.asarray(pose(0.241, 40.0)))[1] < 1e-9
    last_extract_position = source.retained_step_indices.index(2)
    assert all(
        abs(item[2, 3] - source.last_extract_tcp[2, 3]) <= 0.008
        for item in source.reverse_path[: last_extract_position + 1]
    )
    assert source.reverse_path[last_extract_position + 1][2, 3] < 0.21


def test_failed_closed_rejects_unconfigured_sensor_error(tmp_path: Path) -> None:
    run = write_failed_closed_run(tmp_path, error_serial="not-configured")
    with pytest.raises(ValueError, match="healthy stopped bilateral"):
        MODULE.load_held_extract_source(run, gf220_serials=("left", "right"))


def test_current_runner_failed_closed_satisfies_reset_contract() -> None:
    run = ROOT / "runs/live/test_tube_autonomous_20261024_205603"
    if not run.is_dir():
        pytest.skip("reviewed runner failed-closed run is unavailable")
    config = json.loads(
        (ROOT / "configs/test_tube_closed_loop.demo07.json").read_text()
    )
    source = MODULE.load_held_extract_source(
        run,
        gf220_serials=MODULE.configured_gf220_serials(config),
    )
    assert source.source_status == "failed_closed"
    assert source.level_return_step_indices
    last_extract_position = source.retained_step_indices.index(64)
    assert all(
        abs(item[2, 3] - source.last_extract_tcp[2, 3]) <= 0.008
        for item in source.reverse_path[: last_extract_position + 1]
    )


def write_extract_uncertainty_hold(tmp_path: Path, *, reason: str) -> Path:
    run = write_run(tmp_path)
    terminal = run / "task/steps/0003.json"
    payload = json.loads(terminal.read_text())
    payload["decision"]["previous_state"] = "EXTRACT"
    payload["decision"]["action"]["reason"] = reason
    payload["dispatch"]["receipt"]["arm"]["reason"] = reason
    payload["dispatch"]["receipt"]["gripper"]["reason"] = reason
    terminal.write_text(json.dumps(payload))
    return run


def test_extract_uncertainty_hold_skips_level_recovery(tmp_path: Path) -> None:
    source = MODULE.load_held_extract_source(
        write_extract_uncertainty_hold(
            tmp_path, reason="tube_position_uncertainty_too_large"
        )
    )
    assert source.terminal_previous_state == "EXTRACT"
    assert source.level_return_step_indices == ()
    assert source.last_extract_step_index == 2
    assert source.reverse_path[0][2, 3] == pytest.approx(0.241)
    assert source.reverse_path[-1][2, 3] == pytest.approx(0.160)


def test_extract_hold_rejects_any_other_reason(tmp_path: Path) -> None:
    run = write_extract_uncertainty_hold(tmp_path, reason="some_other_reason")
    with pytest.raises(ValueError, match="healthy stopped bilateral"):
        MODULE.load_held_extract_source(run)


def test_current_extract_uncertainty_hold_satisfies_reset_contract() -> None:
    run = ROOT / "runs/live/test_tube_autonomous_20261024_210234"
    if not run.is_dir():
        pytest.skip("reviewed EXTRACT uncertainty hold is unavailable")
    source = MODULE.load_held_extract_source(run)
    assert source.source_status == "held"
    assert source.terminal_previous_state == "EXTRACT"
    assert source.level_return_step_indices == ()
    assert source.held_width_m == pytest.approx(0.01875)

