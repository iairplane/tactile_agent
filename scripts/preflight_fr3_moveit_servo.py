#!/usr/bin/python3
"""Fail-closed static/runtime preflight for the FR3 MoveIt Servo boundary.

No Twist or trajectory command is published.  ``--configure-twist`` only calls
Servo's command-type service and verifies that TWIST was accepted.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "fr3_moveit_servo.yaml"
EXPECTED = {
    "move_group_name": "fr3_arm",
    "cartesian_command_in_topic": "~/delta_twist_cmds",
    "command_out_topic": "/fr3_arm_controller/joint_trajectory",
    "command_out_type": "trajectory_msgs/JointTrajectory",
    "joint_topic": "/joint_states",
    "monitored_planning_scene_topic": "/monitored_planning_scene",
    "command_in_type": "speed_units",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--configure-twist", action="store_true")
    parser.add_argument("--require-realtime", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=4.0)
    return parser.parse_args()


def load_and_validate(path: Path) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if not path.is_file():
        return {}, [f"servo_config_missing:{path}"]
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        return {}, [f"servo_config_unreadable:{error}"]
    if not isinstance(config, dict):
        return {}, ["servo_config_not_mapping"]
    for key, expected in EXPECTED.items():
        if config.get(key) != expected:
            blockers.append(f"servo_config_{key}_must_equal:{expected}")
    for key in ("check_collisions", "publish_joint_positions"):
        if config.get(key) is not True:
            blockers.append(f"servo_config_{key}_must_be_true")
    if config.get("publish_joint_velocities") is not False:
        blockers.append("servo_joint_velocities_must_be_false_for_position_streaming")
    if config.get("is_primary_planning_scene_monitor") is not False:
        blockers.append("servo_must_use_move_group_as_primary_planning_scene_monitor")
    for key in ("publish_period", "incoming_command_timeout", "collision_check_rate"):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            blockers.append(f"servo_config_{key}_must_be_positive")
    return config, blockers


def package_blockers() -> list[str]:
    try:
        from ament_index_python.packages import get_package_prefix
    except ImportError:
        return ["ament_index_python_unavailable"]
    blockers = []
    for package in ("moveit_servo", "franka_fr3_moveit_config", "franka_description"):
        try:
            get_package_prefix(package)
        except Exception:
            blockers.append(f"ros_package_missing:{package}")
    return blockers


def _controllers() -> str:
    completed = subprocess.run(
        ["ros2", "control", "list_controllers"],
        capture_output=True,
        text=True,
        timeout=4.0,
        check=False,
    )
    # ros2_control CLI emits ANSI colour sequences even when stdout is captured.
    # Strip them before matching controller names and lifecycle states.
    return re.sub(r"\x1b\[[0-9;]*[mK]", "", completed.stdout + completed.stderr)


def _runtime_graph_blockers(
    node: Any,
    *,
    nodes: set[str],
    pause_service_ready: bool,
    switch_service_ready: bool,
    require_switch_service: bool,
) -> list[str]:
    """Return the currently missing Servo graph endpoints.

    ROS graph discovery is asynchronous.  Keeping this check separate lets the
    caller sample the complete graph repeatedly instead of treating the first
    appearance of the node names as proof that all topic endpoints exist.
    """

    blockers: list[str] = []
    if "/servo_node" not in nodes:
        blockers.append("servo_node_not_running")
    if "/move_group" not in nodes:
        blockers.append("move_group_not_running")
    if node.count_subscribers("/servo_node/delta_twist_cmds") < 1:
        blockers.append("servo_twist_subscription_missing")
    if node.count_publishers("/servo_node/status") < 1:
        blockers.append("servo_status_publisher_missing")
    if node.count_publishers("/fr3_arm_controller/joint_trajectory") < 1:
        blockers.append("servo_command_out_publisher_missing")
    if node.count_subscribers("/fr3_arm_controller/joint_trajectory") < 1:
        blockers.append("trajectory_controller_subscription_missing")
    if node.count_publishers("/joint_states") < 1:
        blockers.append("joint_states_publisher_missing")
    if node.count_publishers("/monitored_planning_scene") < 1:
        blockers.append("monitored_planning_scene_publisher_missing")
    if not pause_service_ready:
        blockers.append("servo_pause_service_missing")
    if require_switch_service and not switch_service_ready:
        blockers.append("servo_switch_command_type_service_missing")
    return blockers


def _wait_for_runtime_graph(
    node: Any,
    *,
    rclpy_module: Any,
    pause_client: Any,
    switch_client: Any | None,
    timeout_s: float,
) -> list[str]:
    """Wait up to ``timeout_s`` for one complete, internally consistent graph."""

    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        rclpy_module.spin_once(node, timeout_sec=min(0.05, remaining))
        nodes = {
            f"{namespace.rstrip('/')}/{name}".replace("//", "/")
            for name, namespace in node.get_node_names_and_namespaces()
        }
        blockers = _runtime_graph_blockers(
            node,
            nodes=nodes,
            pause_service_ready=pause_client.service_is_ready(),
            switch_service_ready=(
                switch_client is None or switch_client.service_is_ready()
            ),
            require_switch_service=switch_client is not None,
        )
        if not blockers or time.monotonic() >= deadline:
            return blockers


def runtime_blockers(*, configure_twist: bool, timeout_s: float) -> list[str]:
    import rclpy
    from moveit_msgs.srv import ServoCommandType
    from rclpy.node import Node
    from std_srvs.srv import SetBool

    blockers: list[str] = []
    rclpy.init(args=None)
    node = Node(f"fr3_servo_preflight_{int(time.time() * 1000)}")
    try:
        pause_client = node.create_client(SetBool, "/servo_node/pause_servo")
        switch_client = (
            node.create_client(ServoCommandType, "/servo_node/switch_command_type")
            if configure_twist
            else None
        )
        blockers.extend(
            _wait_for_runtime_graph(
                node,
                rclpy_module=rclpy,
                pause_client=pause_client,
                switch_client=switch_client,
                timeout_s=timeout_s,
            )
        )

        try:
            controller_text = _controllers()
        except (OSError, subprocess.SubprocessError) as error:
            blockers.append(f"controller_query_failed:{error}")
        else:
            if not any(
                line.startswith("fr3_arm_controller ") and line.rstrip().endswith(" active")
                for line in controller_text.splitlines()
            ):
                blockers.append("fr3_arm_controller_not_active")

        if configure_twist and "servo_switch_command_type_service_missing" not in blockers:
            if switch_client is not None:
                request = ServoCommandType.Request()
                request.command_type = ServoCommandType.Request.TWIST
                future = switch_client.call_async(request)
                rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
                if not future.done() or future.result() is None:
                    blockers.append("servo_twist_mode_call_timed_out")
                elif future.result().success is not True:
                    blockers.append("servo_twist_mode_rejected")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return blockers


def main() -> int:
    args = parse_args()
    if args.configure_twist and not args.runtime:
        raise SystemExit("--configure-twist requires --runtime")
    config, blockers = load_and_validate(args.config.resolve())
    blockers.extend(package_blockers())
    if args.require_realtime and "realtime" not in platform.release().lower():
        blockers.append(f"realtime_kernel_required:{platform.release()}")
    if args.runtime and not package_blockers():
        blockers.extend(
            runtime_blockers(
                configure_twist=args.configure_twist,
                timeout_s=args.timeout_s,
            )
        )
    report = {
        "ready": not blockers,
        "hardware_commands_dispatched": False,
        "twist_mode_configured": bool(args.configure_twist and not blockers),
        "config": str(args.config.resolve()),
        "move_group_name": config.get("move_group_name"),
        "command_in": "/servo_node/delta_twist_cmds",
        "command_out": config.get("command_out_topic"),
        "collision_checking_enabled": config.get("check_collisions") is True,
        "collision_model_scope": "fr3_arm_only_no_hande",
        "blockers": sorted(set(blockers)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
