from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def _load_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/retreat_test_tube_pregrasp_for_retry.py"
    )
    spec = importlib.util.spec_from_file_location(
        "retreat_test_tube_pregrasp_for_retry", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pose() -> np.ndarray:
    value = np.eye(4)
    value[:3, 3] = [0.558, 0.192, 0.226]
    return value


def _source_run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    task = root / "task"
    steps = task / "steps"
    steps.mkdir(parents=True)
    report = {
        "schema": "demo2gripper.test_tube_closed_loop_report.v1",
        "mode": "execute",
        "status": "running",
        "final_state": "APPROACH_TUBE",
        "task_id": "test-tube-demo07-seven-stage-online",
    }
    (task / "run_report.json").write_text(json.dumps(report))
    target = _pose()
    approach = {
        "index": 2,
        "observation": {
            "robot": {"T_base_tcp": _pose().tolist()},
            "gripper": {
                "width_m": 0.0494,
                "moving": False,
                "fault_free": True,
                "object_detected": False,
            },
        },
        "decision": {
            "previous_state": "OBSERVE",
            "next_state": "APPROACH_TUBE",
            "action": {
                "kind": "move_tcp",
                "parameters": {
                    "path": "top_approach",
                    "T_base_tcp_target": target.tolist(),
                },
            },
        },
    }
    last = {
        "index": 3,
        "observation": approach["observation"],
        "decision": {
            "previous_state": "APPROACH_TUBE",
            "next_state": "APPROACH_TUBE",
            "action": {"kind": "pause_arm", "parameters": {}},
        },
    }
    (steps / "0002.json").write_text(json.dumps(approach))
    (steps / "0003.json").write_text(json.dumps(last))
    return root


def test_loads_exact_top_pregrasp_and_last_open_state(tmp_path: Path) -> None:
    module = _load_script()
    target, report, source, last = module.load_pregrasp_reference(
        _source_run(tmp_path)
    )
    assert target == pytest.approx(_pose())
    assert report["final_state"] == "APPROACH_TUBE"
    assert source["index"] == 2
    assert last["index"] == 3


def test_retreat_is_base_z_only_and_preserves_orientation() -> None:
    module = _load_script()
    current = _pose()
    current[:3, :3] = Rotation.from_euler("xyz", [1, -2, 8], degrees=True).as_matrix()
    target = module.build_retreat_target(current, retreat_mm=60)
    assert target[:3, :3] == pytest.approx(current[:3, :3])
    assert target[:2, 3] == pytest.approx(current[:2, 3])
    assert target[2, 3] == pytest.approx(current[2, 3] + 0.060)


def test_live_start_uses_eight_mm_and_two_degree_envelope() -> None:
    module = _load_script()
    reference = _pose()
    accepted = reference.copy()
    accepted[:3, 3] += [0.004, -0.003, 0.001]
    accepted[:3, :3] = Rotation.from_euler("z", 1.5, degrees=True).as_matrix()
    evidence = module.validate_live_start(reference, accepted)
    assert evidence["distance_mm"] == pytest.approx(26**0.5)
    assert evidence["rotation_error_deg"] == pytest.approx(1.5)
    too_far = reference.copy()
    too_far[0, 3] += 0.0081
    with pytest.raises(RuntimeError, match="recorded top-pregrasp"):
        module.validate_live_start(reference, too_far)
    too_rotated = reference.copy()
    too_rotated[:3, :3] = Rotation.from_euler("z", 2.1, degrees=True).as_matrix()
    with pytest.raises(RuntimeError, match="rotated"):
        module.validate_live_start(reference, too_rotated)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"width_m": 0.047}, "not open"),
        ({"moving": True}, "still moving"),
        ({"object_detected": True}, "reports an object"),
        ({"fault_free": False}, "not fault-free"),
    ],
)
def test_open_gripper_gate(change, match) -> None:
    module = _load_script()
    feedback = {
        "width_m": 0.0494,
        "moving": False,
        "fault_free": True,
        "object_detected": False,
    }
    feedback.update(change)
    with pytest.raises(RuntimeError, match=match):
        module.validate_open_gripper(feedback)


def test_execute_requires_collision_model_ack() -> None:
    module = _load_script()
    with pytest.raises(SystemExit):
        module.parse_args([
            "--config", "unused.json",
            "--source-run", "unused",
            "--output-dir", "unused-output",
            "--execute",
        ])
