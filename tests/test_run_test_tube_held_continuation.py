from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_test_tube_held_continuation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_test_tube_held_continuation", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retention_requires_bilateral_warped_contact() -> None:
    module = _load_script()
    hande = SimpleNamespace(object_detected=True)
    assert not module._retained_hold(
        {"valid": True, "bilateral_contact": False}, hande
    )
    assert module._retained_hold(
        {"valid": True, "bilateral_contact": True},
        SimpleNamespace(object_detected=False),
    )
    assert not module._retained_hold(
        {"valid": False, "bilateral_contact": True}, hande
    )


def test_default_extraction_is_eighty_mm() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--config",
            "unused.json",
            "--baseline-dir",
            "unused-baseline",
            "--source-step",
            "unused-step.json",
            "--output-dir",
            "unused-output",
        ]
    )
    assert args.extract_mm == pytest.approx(80.0)
    target = module.extraction_target(np.eye(4), args.extract_mm)
    assert target[2, 3] == pytest.approx(0.080)


def test_extraction_distance_is_bounded() -> None:
    module = _load_script()
    with pytest.raises(ValueError, match=r"\[40, 120\]"):
        module.extraction_target(np.eye(4), 121.0)


def _write_level_resume_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    held = tmp_path / "held" / "task"
    steps = held / "steps"
    steps.mkdir(parents=True)
    level = np.eye(4)
    level[:3, :3] = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    step0 = {
        "index": 0,
        "decision": {
            "action": {
                "kind": "servo_tcp",
                "parameters": {
                    "purpose": "rigid_held_tube_level",
                    "T_base_tcp_live_target": level.tolist(),
                },
            }
        },
    }
    step1 = {
        "index": 1,
        "decision": {
            "previous_state": "LEVEL_HELD_TUBE",
            "next_state": "HOLD",
            "action": {
                "kind": "stop_and_hold",
                "reason": "LEVEL_HELD_TUBE timeout exceeded",
            },
        },
        "dispatch": {"status": "held", "receipt": {"held": True}},
        "observation": {
            "robot": {"T_base_tcp": np.eye(4).tolist()},
            "tube_pose": {"T_base_object": np.eye(4).tolist()},
            "gripper": {"width_m": 0.018, "fault_free": True, "moving": False},
            "tactile": {
                "valid": True,
                "bilateral_contact": True,
                "bilateral_contact_interior": True,
            },
        },
    }
    for index, value in enumerate((step0, step1)):
        (steps / f"{index:04d}.json").write_text(json.dumps(value))
    (held / "run_report.json").write_text(
        json.dumps(
            {
                "schema": "demo2gripper.test_tube_closed_loop_report.v1",
                "mode": "execute",
                "status": "held",
                "final_state": "HOLD",
                "hardware_commands_dispatched": True,
                "step_count": 2,
            }
        )
    )
    lifted = level.copy()
    lifted[2, 3] = 0.05
    quaternion = [0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)]
    receipt = tmp_path / "escape.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "demo2gripper.moveit_pose_goal.v2",
                "mode": "execute",
                "input_target_frame": "configured-ee",
                "planning_succeeded": True,
                "execution_requested": True,
                "execution_dispatched": True,
                "execution_succeeded": True,
                "execution_outcome": "measured_target_reached",
                "trajectory_audit": {"accepted": True, "violations": []},
                "start_configured_ee_position_base_m": [0.0, 0.0, 0.0],
                "start_configured_ee_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "input_target_position_base_m": [0.0, 0.0, 0.05],
                "input_target_orientation_xyzw": quaternion,
                "measured_end_configured_ee_position_base_m": [0.0, 0.0, 0.05],
                "measured_end_configured_ee_orientation_xyzw": quaternion,
            }
        )
    )
    return held.parent, receipt


def test_level_escape_resume_reconstructs_rigid_relation(tmp_path: Path) -> None:
    module = _load_script()
    held, receipt = _write_level_resume_artifacts(tmp_path)
    result = module.load_level_escape_resume(
        held, receipt, horizontal_tolerance_deg=4.0
    )
    assert result.held_width_m == pytest.approx(0.018)
    assert result.horizontal_error_deg == pytest.approx(0.0, abs=1e-8)
    assert np.allclose(result.T_tube_tcp, np.eye(4))


def test_level_escape_resume_rejects_unrelated_start(tmp_path: Path) -> None:
    module = _load_script()
    held, receipt = _write_level_resume_artifacts(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["start_configured_ee_position_base_m"] = [0.01, 0.0, 0.0]
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="did not start"):
        module.load_level_escape_resume(
            held, receipt, horizontal_tolerance_deg=4.0
        )


def _coarse_shortfall_receipt(module, target: np.ndarray | None = None) -> dict:
    target = np.eye(4) if target is None else target
    measured = target.copy()
    measured[0, 3] += 0.007
    return {
        "schema": "demo2gripper.moveit_pose_goal.v2",
        "mode": "execute",
        "input_target_frame": "configured-ee",
        "planning_succeeded": True,
        "execution_requested": True,
        "execution_dispatched": True,
        "execution_action_reported_success": True,
        "execution_succeeded": False,
        "execution_outcome": "measured_endpoint_mismatch",
        "execution_failure_reason": module.COARSE_ENDPOINT_MISMATCH_REASON,
        "measurement_error": None,
        "moveit_error_code": 1,
        "execution_error_code": 1,
        "trajectory_audit": {"accepted": True, "violations": []},
        "measured_input_target_position_error_m": 0.007,
        "measured_input_target_rotation_error_rad": 0.0,
        "input_target_position_base_m": target[:3, 3].tolist(),
        "input_target_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "measured_end_configured_ee_position_base_m": measured[:3, 3].tolist(),
        "measured_end_configured_ee_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def test_coarse_endpoint_shortfall_allows_only_audited_ten_mm_settle() -> None:
    module = _load_script()
    target = np.eye(4)
    _, measured = module._recoverable_coarse_endpoint_shortfall(
        _coarse_shortfall_receipt(module, target),
        expected_target=target,
    )
    assert measured[0, 3] == pytest.approx(0.007)

    too_far = _coarse_shortfall_receipt(module, target)
    too_far["measured_input_target_position_error_m"] = 0.011
    with pytest.raises(ValueError, match="not eligible"):
        module._recoverable_coarse_endpoint_shortfall(
            too_far,
            expected_target=target,
        )


def test_failed_continuation_resume_requires_exact_standoff_failure(
    tmp_path: Path,
) -> None:
    module = _load_script()
    held, escape = _write_level_resume_artifacts(tmp_path)
    level = module.load_level_escape_resume(
        held, escape, horizontal_tolerance_deg=4.0
    )
    socket_receipt = tmp_path / "socket.json"
    socket_receipt.write_text("{}")
    prior = tmp_path / "prior"
    (prior / "arm").mkdir(parents=True)
    coarse = _coarse_shortfall_receipt(module)
    (prior / "arm" / "socket_standoff.json").write_text(json.dumps(coarse))
    report = {
        "schema": "demo2gripper.test_tube_held_continuation.v1",
        "mode": "execute",
        "status": "failed_hold",
        "hardware_commands_dispatched": True,
        "config_sha256": "config-hash",
        "baseline_warped_reference_sha256": "baseline-hash",
        "error": {
            "type": "LiveIOError",
            "message": module.COARSE_ENDPOINT_MISMATCH_REASON,
        },
        "final_arm_stop": {
            "accepted": True,
            "reason": "continuation finished",
        },
        "final_gripper_stop": {
            "accepted": True,
            "reason": "continuation failed",
        },
        "level_escape_resume": {
            "held_report": str(level.held_report),
            "escape_receipt": str(level.escape_receipt),
            "artifacts_sha256": level.artifacts_sha256,
        },
        "prepared_socket_evidence": {"receipt": str(socket_receipt)},
        "stages": [{"name": "transfer_clear", "status": "complete"}],
    }
    (prior / "run_report.json").write_text(json.dumps(report))
    resumed = module.load_failed_continuation_resume(
        prior,
        level_resume=level,
        config_sha256="config-hash",
        baseline_sha256="baseline-hash",
        socket_receipt=socket_receipt,
    )
    assert resumed.measured_tcp[0, 3] == pytest.approx(0.007)

    report["stages"].append({"name": "release", "status": "complete"})
    (prior / "run_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="already reached release"):
        module.load_failed_continuation_resume(
            prior,
            level_resume=level,
            config_sha256="config-hash",
            baseline_sha256="baseline-hash",
            socket_receipt=socket_receipt,
        )
