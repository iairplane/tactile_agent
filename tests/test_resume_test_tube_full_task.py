import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture(scope="module")
def recovery():
    path = Path(__file__).resolve().parents[1] / "scripts/resume_test_tube_full_task.py"
    spec = importlib.util.spec_from_file_location("resume_test_tube_full_task", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pose_errors_measure_translation_rotation_and_quaternion_equivalence(recovery):
    identity = [0, 0, 0, 1]
    distance, angle = recovery.pose_errors(
        [.003, .004, 0], Rotation.from_euler("z", 45, degrees=True).as_quat(),
        [0, 0, 0], identity,
    )
    assert distance == pytest.approx(.005)
    assert angle == pytest.approx(45)
    assert recovery.pose_errors([0, 0, 0], identity, [0, 0, 0], [0, 0, 0, -1]) == (0, 0)
    with pytest.raises(ValueError, match="finite"):
        recovery.pose_errors([np.nan, 0, 0], identity, [0, 0, 0], identity)


def test_recovery_rotates_tcp_offset_before_base_vertical_translation(recovery):
    quaternion = Rotation.from_euler("y", 90, degrees=True).as_quat()
    target = recovery.recovery_waypoint([.4, -.1, .2], quaternion)
    # The 190 mm TCP offset follows link8 local Z; the 60 mm reset follows base Z.
    np.testing.assert_allclose(target["position_m"], [.59, -.1, .26], atol=1e-12)
    np.testing.assert_allclose(target["orientation_xyzw"], quaternion)


def test_motion_success_requires_measured_endpoint_and_kept_wrist(recovery):
    target = {"position_m": [.4, -.1, .26], "orientation_xyzw": [0, 0, 0, 1]}
    outcome = {"execution_succeeded": True, "measured_end_configured_ee": dict(target)}
    np.testing.assert_allclose(
        recovery.require_motion_success(outcome, target), target["position_m"]
    )
    for invalid in ({"execution_succeeded": False}, {"execution_succeeded": True}):
        with pytest.raises(RuntimeError):
            recovery.require_motion_success(invalid, target)
    for bad_position in ([.409, -.1, .26], [np.nan, -.1, .26], [.4, -.1]):
        outcome["measured_end_configured_ee"] = {**target, "position_m": bad_position}
        with pytest.raises(RuntimeError):
            recovery.require_motion_success(outcome, target)
    outcome["measured_end_configured_ee"] = {
        **target, "orientation_xyzw": Rotation.from_euler("x", 3, degrees=True).as_quat().tolist(),
    }
    with pytest.raises(RuntimeError, match="wrist orientation"):
        recovery.require_motion_success(outcome, target)


def test_complete_requires_stack_and_nested_task_completion(recovery):
    complete = {
        "status": "complete", "task_report": {"status": "complete", "final_state": "COMPLETE"},
    }
    assert recovery.full_task_complete(complete)
    for incomplete in (
        {}, {"status": "complete"}, {"status": "complete", "task_report": None},
        {**complete, "status": "held"},
        {"status": "complete", "task_report": {"status": "held", "final_state": "COMPLETE"}},
        {"status": "complete", "task_report": {"status": "complete", "final_state": "EXTRACT"}},
    ):
        assert not recovery.full_task_complete(incomplete)
