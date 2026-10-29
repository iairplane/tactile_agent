from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.capture_handeye_sample import robot_reader_command, summarize_robot_states
from scripts.ros_franka_state_reader import franka_state_to_dict, pose_to_column_major


def _pose(
    translation: tuple[float, float, float],
    quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> SimpleNamespace:
    return SimpleNamespace(
        position=SimpleNamespace(x=translation[0], y=translation[1], z=translation[2]),
        orientation=SimpleNamespace(
            x=quaternion_xyzw[0],
            y=quaternion_xyzw[1],
            z=quaternion_xyzw[2],
            w=quaternion_xyzw[3],
        ),
    )


class _Errors:
    def __init__(self, **values: bool) -> None:
        self._values = {"cartesian_reflex": False, "joint_reflex": False, **values}
        for name, value in self._values.items():
            setattr(self, name, value)

    def get_fields_and_field_types(self) -> dict[str, str]:
        return {name: "boolean" for name in self._values}


def _message(time_s: float) -> SimpleNamespace:
    yaw = math.radians(90.0)
    return SimpleNamespace(
        o_t_ee=SimpleNamespace(
            pose=_pose((0.4, -0.2, 0.5), (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
        ),
        f_t_ee=SimpleNamespace(pose=_pose((0.0, 0.0, 0.19))),
        measured_joint_state=SimpleNamespace(position=[0.1] * 7, velocity=[0.0] * 7),
        time=time_s,
        robot_mode=1,
        current_errors=_Errors(),
    )


def test_pose_conversion_is_libfranka_column_major() -> None:
    column_major = pose_to_column_major(_pose((1.0, 2.0, 3.0)))
    matrix = np.asarray(column_major).reshape(4, 4, order="F")
    expected = np.asarray(
        [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    )
    np.testing.assert_allclose(matrix, expected)


def test_ros_state_has_identical_base_flange_semantics_to_existing_summarizer() -> None:
    states = [franka_state_to_dict(_message(1.0 + index * 0.001)) for index in range(4)]
    output = "\n".join(json.dumps(state) for state in states)
    args = argparse.Namespace(
        min_robot_states=4,
        max_abs_dq_rad_s=0.01,
        max_pose_translation_m=0.0005,
        max_pose_rotation_deg=0.1,
        robot_reader_reviewed=True,
        robot_reader_kind="ros_topic",
        ros_state_reader=Path("scripts/ros_franka_state_reader.py"),
        ros_python=Path("/usr/bin/python3"),
        ros_robot_state_topic="/franka_robot_state_broadcaster/robot_state",
        echo_robot_state=Path("unused"),
        robot_hostname="unused",
    )
    base_flange, summary = summarize_robot_states((output, "", 0), args)

    expected_base_ee = np.asarray(states[0]["O_T_EE"]).reshape(4, 4, order="F")
    expected_flange_ee = np.asarray(states[0]["F_T_EE"]).reshape(4, 4, order="F")
    np.testing.assert_allclose(base_flange, expected_base_ee @ np.linalg.inv(expected_flange_ee))
    assert summary["source"] == "ROS 2 franka_robot_state_broadcaster read-only subscription"
    assert summary["ros_topic"] == "/franka_robot_state_broadcaster/robot_state"
    assert summary["current_errors_seen"] == []


def test_ros_state_preserves_active_errors_for_fail_closed_capture() -> None:
    message = _message(1.0)
    message.current_errors = _Errors(cartesian_reflex=True)
    assert franka_state_to_dict(message)["current_errors"] == ["cartesian_reflex"]


def test_capture_launches_reviewed_ros_reader_with_enough_post_exposure_states() -> None:
    args = argparse.Namespace(
        robot_reader_kind="ros_topic",
        ros_python=Path("/usr/bin/python3"),
        ros_state_reader=Path("scripts/ros_franka_state_reader.py"),
        ros_robot_state_topic="/franka_robot_state_broadcaster/robot_state",
        min_robot_states=90,
        min_robot_states_before_exposure=10,
        min_robot_states_after_exposure=10,
        robot_timeout_s=15.0,
    )
    command = robot_reader_command(args)
    assert command[:4] == [
        "/usr/bin/python3",
        "scripts/ros_franka_state_reader.py",
        "--topic",
        "/franka_robot_state_broadcaster/robot_state",
    ]
    count = int(command[command.index("--count") + 1])
    assert count == 120
