#!/usr/bin/python3
"""Launch only the FR3 MoveIt Servo node against an existing MoveIt stack.

This process does not publish a motion command.  The caller must select Servo's
TWIST command type before publishing ``/servo_node/delta_twist_cmds``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "fr3_moveit_servo.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--use-fake-hardware", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Servo config must be a mapping: {path}")
    return value


def generate_launch_description(
    *, config: Path = DEFAULT_CONFIG, robot_ip: str = "172.16.0.2", fake: bool = False
):
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.substitutions import Command, FindExecutable
    from launch_ros.actions import Node
    from launch_ros.parameter_descriptions import ParameterValue

    servo_params = load_yaml(config)
    description_share = Path(get_package_share_directory("franka_description"))
    moveit_share = Path(get_package_share_directory("franka_fr3_moveit_config"))
    urdf = description_share / "robots" / "fr3" / "fr3.urdf.xacro"
    srdf = description_share / "robots" / "fr3" / "fr3.srdf.xacro"

    fake_text = "true" if fake else "false"
    robot_description = {
        "robot_description": ParameterValue(
            Command(
                [
                    FindExecutable(name="xacro"),
                    " ",
                    str(urdf),
                    " ros2_control:=false",
                    " hand:=false",
                    " arm_id:=fr3",
                    " robot_ip:=",
                    robot_ip,
                    " use_fake_hardware:=",
                    fake_text,
                    " fake_sensor_commands:=false",
                ]
            ),
            value_type=str,
        )
    }
    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(
            Command(
                [
                    FindExecutable(name="xacro"),
                    " ",
                    str(srdf),
                    " hand:=false",
                    " arm_id:=fr3",
                ]
            ),
            value_type=str,
        )
    }
    kinematics = {
        "robot_description_kinematics": load_yaml(moveit_share / "config" / "kinematics.yaml")
    }
    joint_limits = {
        "robot_description_planning": load_yaml(
            moveit_share / "config" / "fr3_joint_limits.yaml"
        )
    }

    servo_node = Node(
        package="moveit_servo",
        executable="servo_node",
        name="servo_node",
        namespace="/",
        output="screen",
        parameters=[
            {"moveit_servo": servo_params},
            {"update_period": float(servo_params["publish_period"])},
            {"planning_group_name": str(servo_params["move_group_name"])},
            robot_description,
            robot_description_semantic,
            kinematics,
            joint_limits,
        ],
    )
    return LaunchDescription([servo_node])


def main() -> int:
    from launch import LaunchService

    args = parse_args()
    service = LaunchService(argv=[])
    service.include_launch_description(
        generate_launch_description(
            config=args.config.resolve(),
            robot_ip=args.robot_ip,
            fake=args.use_fake_hardware,
        )
    )
    return service.run()


if __name__ == "__main__":
    raise SystemExit(main())
