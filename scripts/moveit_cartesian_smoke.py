#!/usr/bin/python3
"""Plan or execute one bounded straight-line FR3 Cartesian translation.

This hardware-integration tool keeps the current ``fr3_link8`` orientation and
asks MoveIt to interpolate a collision-checked Cartesian line.  Every returned
joint path is audited before it can be sent unchanged to ExecuteTrajectory.
Planning is the default.  Execution needs an explicit acknowledgement because
the installed Hand-E is not represented by the current arm-only MoveIt collision
model.

Run with the ROS system Python after sourcing ROS and the local franka_ros2
overlay.  This is a staged integration tool, not a TaskSkill executor.
"""

from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo2gripper.trajectory_safety import (  # noqa: E402
    DEFAULT_MAX_JOINT_SWEEP_RAD,
    DEFAULT_MAX_TOTAL_JOINT_SWEEP_RAD,
    DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD,
    audit_fr3_joint_trajectory,
    validate_fr3_trajectory_audit_thresholds,
)

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

MAX_TRANSLATION_M = 0.160
MAX_CARTESIAN_SPEED_M_S = 0.030


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--delta-x-m", type=float, default=0.0)
    parser.add_argument("--delta-y-m", type=float, default=0.0)
    parser.add_argument("--delta-z-m", type=float, default=0.0)
    parser.add_argument("--base-frame", default="fr3_link0")
    parser.add_argument("--tool-link", default="fr3_link8")
    parser.add_argument("--group", default="fr3_arm")
    parser.add_argument("--service-name", default="/compute_cartesian_path")
    parser.add_argument("--action-name", default="/execute_trajectory")
    parser.add_argument("--server-timeout-s", type=float, default=15.0)
    parser.add_argument("--max-step-m", type=float, default=0.005)
    parser.add_argument("--jump-threshold", type=float, default=2.0)
    parser.add_argument("--revolute-jump-threshold-rad", type=float, default=0.10)
    parser.add_argument("--velocity-scale", type=float, default=0.05)
    parser.add_argument("--acceleration-scale", type=float, default=0.05)
    parser.add_argument("--max-cartesian-speed-m-s", type=float, default=0.025)
    parser.add_argument(
        "--min-joint-limit-margin-rad",
        type=float,
        default=DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD,
        help="minimum distance of every planned point from every FR3 hardware limit",
    )
    parser.add_argument(
        "--max-joint-sweep-rad",
        type=float,
        default=DEFAULT_MAX_JOINT_SWEEP_RAD,
        help="maximum allowed min-to-max sweep of any one joint in the complete path",
    )
    parser.add_argument(
        "--max-total-joint-sweep-rad",
        type=float,
        default=DEFAULT_MAX_TOTAL_JOINT_SWEEP_RAD,
        help="maximum allowed sum of all seven joint sweeps in the complete path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/fr3_cartesian_smoke_plan.json"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="audit and execute the exact fully computed Cartesian path",
    )
    parser.add_argument(
        "--accept-arm-only-collision-model",
        action="store_true",
        help="Acknowledge that the installed Hand-E is absent from the collision model",
    )
    parser.add_argument(
        "--allow-generic-kernel",
        action="store_true",
        help="Allow execution although the current kernel name lacks 'realtime'",
    )
    return parser.parse_args()


def requested_translation(args: argparse.Namespace) -> tuple[float, float, float]:
    return (args.delta_x_m, args.delta_y_m, args.delta_z_m)


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "server_timeout_s",
        "max_step_m",
        "jump_threshold",
        "revolute_jump_threshold_rad",
        "velocity_scale",
        "acceleration_scale",
        "max_cartesian_speed_m_s",
        "min_joint_limit_margin_rad",
        "max_joint_sweep_rad",
        "max_total_joint_sweep_rad",
    )
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")

    delta = requested_translation(args)
    if not all(math.isfinite(value) for value in delta):
        raise ValueError("requested translation must be finite")
    distance = math.sqrt(sum(value * value for value in delta))
    if distance <= 0.0:
        raise ValueError("requested translation must be non-zero")
    if distance > MAX_TRANSLATION_M:
        raise ValueError(
            f"translation norm {distance:.6f} m exceeds the {MAX_TRANSLATION_M:.3f} m cap"
        )
    if args.max_step_m > 0.010:
        raise ValueError("--max-step-m is capped at 0.010 m")
    for name in ("velocity_scale", "acceleration_scale"):
        if getattr(args, name) > 0.05:
            raise ValueError(f"--{name.replace('_', '-')} is capped at 0.05")
    if args.max_cartesian_speed_m_s > MAX_CARTESIAN_SPEED_M_S:
        raise ValueError(
            "--max-cartesian-speed-m-s is capped at "
            f"{MAX_CARTESIAN_SPEED_M_S:.3f} m/s"
        )
    validate_fr3_trajectory_audit_thresholds(
        min_joint_limit_margin_rad=args.min_joint_limit_margin_rad,
        max_joint_sweep_rad=args.max_joint_sweep_rad,
        max_total_joint_sweep_rad=args.max_total_joint_sweep_rad,
    )
    if args.execute and not args.accept_arm_only_collision_model:
        raise ValueError("--execute requires --accept-arm-only-collision-model")
    if args.execute and "realtime" not in platform.release().lower():
        if not args.allow_generic_kernel:
            raise ValueError(
                "execution on this kernel requires --allow-generic-kernel; "
                f"current kernel is {platform.release()}"
            )


def duration_seconds(value: object) -> float:
    return float(value.sec) + float(value.nanosec) * 1e-9


def trajectory_dict(trajectory: object) -> dict:
    joint_trajectory = trajectory.joint_trajectory
    return {
        "joint_names": list(joint_trajectory.joint_names),
        "point_count": len(joint_trajectory.points),
        "points": [
            {
                "positions": list(point.positions),
                "velocities": list(point.velocities),
                "accelerations": list(point.accelerations),
                "time_from_start_s": duration_seconds(point.time_from_start),
            }
            for point in joint_trajectory.points
        ],
    }


def wait_for_transform(
    node: Node,
    buffer: Buffer,
    *,
    base_frame: str,
    tool_link: str,
    timeout_s: float,
) -> object:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if buffer.can_transform(base_frame, tool_link, Time()):
            break
    return buffer.lookup_transform(
        base_frame,
        tool_link,
        Time(),
        timeout=Duration(seconds=0.1),
    )


def make_request(args: argparse.Namespace, transform: object) -> GetCartesianPath.Request:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    delta = requested_translation(args)

    target = Pose()
    target.position.x = translation.x + delta[0]
    target.position.y = translation.y + delta[1]
    target.position.z = translation.z + delta[2]
    target.orientation = rotation

    request = GetCartesianPath.Request()
    request.header.frame_id = args.base_frame
    request.start_state.is_diff = True
    request.group_name = args.group
    request.link_name = args.tool_link
    request.waypoints = [target]
    request.max_step = args.max_step_m
    request.jump_threshold = args.jump_threshold
    request.prismatic_jump_threshold = 0.0
    request.revolute_jump_threshold = args.revolute_jump_threshold_rad
    request.avoid_collisions = True
    request.max_velocity_scaling_factor = args.velocity_scale
    request.max_acceleration_scaling_factor = args.acceleration_scale
    request.cartesian_speed_limited_link = args.tool_link
    request.max_cartesian_speed = args.max_cartesian_speed_m_s
    return request


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"argument error: {exc}", file=sys.stderr)
        return 2

    rclpy.init()
    node = Node("demo2gripper_cartesian_smoke")
    listener = None
    try:
        buffer = Buffer()
        listener = TransformListener(buffer, node)
        try:
            start_transform = wait_for_transform(
                node,
                buffer,
                base_frame=args.base_frame,
                tool_link=args.tool_link,
                timeout_s=args.server_timeout_s,
            )
        except TransformException as exc:
            node.get_logger().error(f"TF lookup failed: {exc}")
            return 3

        delta = requested_translation(args)
        distance = math.sqrt(sum(value * value for value in delta))
        node.get_logger().info(
            "requesting fixed-orientation Cartesian translation "
            f"[{delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}] m "
            f"({distance * 1000.0:.1f} mm); "
            f"mode={'execute' if args.execute else 'plan_only'}"
        )

        client = node.create_client(GetCartesianPath, args.service_name)
        if not client.wait_for_service(timeout_sec=args.server_timeout_s):
            node.get_logger().error(f"Cartesian path service unavailable: {args.service_name}")
            return 4
        future = client.call_async(make_request(args, start_transform))
        rclpy.spin_until_future_complete(node, future)
        response = future.result()
        if response is None:
            node.get_logger().error("Cartesian path service returned no response")
            return 5

        trajectory = trajectory_dict(response.solution)
        moveit_planning_succeeded = (
            response.error_code.val == MoveItErrorCodes.SUCCESS
            and response.fraction >= 1.0 - 1e-9
            and trajectory["point_count"] > 1
        )
        trajectory_audit = audit_fr3_joint_trajectory(
            trajectory,
            min_joint_limit_margin_rad=args.min_joint_limit_margin_rad,
            max_joint_sweep_rad=args.max_joint_sweep_rad,
            max_total_joint_sweep_rad=args.max_total_joint_sweep_rad,
        )
        planning_succeeded = moveit_planning_succeeded and trajectory_audit["accepted"]

        execution_dispatched = False
        execution_status = None
        execution_error_code = None
        execution_error_message = None
        execution_failure_reason = None
        if args.execute:
            if not moveit_planning_succeeded:
                execution_failure_reason = "MoveIt did not compute a complete Cartesian path"
            elif not trajectory_audit["accepted"]:
                execution_failure_reason = "planned Cartesian path failed the FR3 joint audit"
            else:
                action = ActionClient(node, ExecuteTrajectory, args.action_name)
                if not action.wait_for_server(timeout_sec=args.server_timeout_s):
                    execution_failure_reason = (
                        f"trajectory execution action unavailable: {args.action_name}"
                    )
                else:
                    goal = ExecuteTrajectory.Goal()
                    goal.trajectory = response.solution
                    goal.controller_names = ["fr3_arm_controller"]
                    goal_future = action.send_goal_async(goal)
                    rclpy.spin_until_future_complete(node, goal_future)
                    handle = goal_future.result()
                    if handle is None or not handle.accepted:
                        execution_failure_reason = (
                            "trajectory execution goal was rejected"
                        )
                    else:
                        execution_dispatched = True
                        result_future = handle.get_result_async()
                        rclpy.spin_until_future_complete(node, result_future)
                        wrapped = result_future.result()
                        if wrapped is None:
                            execution_failure_reason = (
                                "trajectory execution returned no result"
                            )
                        else:
                            execution_status = wrapped.status
                            execution_error_code = wrapped.result.error_code.val
                            execution_error_message = (
                                wrapped.result.error_code.message
                            )

        end_transform = None
        measurement_error = None
        if execution_dispatched:
            try:
                end_transform = wait_for_transform(
                    node,
                    buffer,
                    base_frame=args.base_frame,
                    tool_link=args.tool_link,
                    timeout_s=args.server_timeout_s,
                )
            except TransformException as exc:
                measurement_error = f"post-execution TF lookup failed: {exc}"
                node.get_logger().error(measurement_error)

        start = start_transform.transform.translation
        end = end_transform.transform.translation if end_transform is not None else None
        end_rotation = end_transform.transform.rotation if end_transform is not None else None
        execution_succeeded = (
            planning_succeeded
            and execution_dispatched
            and execution_status == GoalStatus.STATUS_SUCCEEDED
            and execution_error_code == MoveItErrorCodes.SUCCESS
            and end_transform is not None
        )
        summary = {
            "schema": "demo2gripper.moveit_cartesian_smoke.v2",
            "mode": "execute" if args.execute else "plan_only",
            "arm_only_collision_model": True,
            "base_frame": args.base_frame,
            "tool_link": args.tool_link,
            "group": args.group,
            "requested_translation_base_m": list(delta),
            "translation_norm_m": distance,
            "start_position_base_m": [start.x, start.y, start.z],
            "held_orientation_xyzw": [
                start_transform.transform.rotation.x,
                start_transform.transform.rotation.y,
                start_transform.transform.rotation.z,
                start_transform.transform.rotation.w,
            ],
            "target_position_base_m": [
                start.x + delta[0],
                start.y + delta[1],
                start.z + delta[2],
            ],
            "moveit_error_code": response.error_code.val,
            "moveit_error_message": response.error_code.message,
            "path_fraction": response.fraction,
            "trajectory": trajectory,
            "moveit_planning_succeeded": moveit_planning_succeeded,
            "trajectory_audit": trajectory_audit,
            "planning_succeeded": planning_succeeded,
            "execution_requested": args.execute,
            "execution_policy": "plan_then_audit_then_execute_exact_trajectory",
            "execution_dispatched": execution_dispatched,
            "execution_status": execution_status,
            "execution_error_code": execution_error_code,
            "execution_error_message": execution_error_message,
            "execution_failure_reason": execution_failure_reason,
            "execution_succeeded": execution_succeeded if args.execute else None,
            "measurement_error": measurement_error,
            "measured_end_position_base_m": (
                [end.x, end.y, end.z] if end is not None else None
            ),
            "measured_end_orientation_xyzw": (
                [
                    end_rotation.x,
                    end_rotation.y,
                    end_rotation.z,
                    end_rotation.w,
                ]
                if end_rotation is not None
                else None
            ),
            "measured_translation_base_m": (
                [
                    end.x - start.x,
                    end.y - start.y,
                    end.z - start.z,
                ]
                if end is not None
                else None
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        headline = {key: value for key, value in summary.items() if key != "trajectory"}
        print(json.dumps(headline, indent=2))
        print(f"trajectory points: {trajectory['point_count']}; saved: {args.output}")
        if not planning_succeeded:
            if trajectory_audit["violations"]:
                node.get_logger().error(
                    "trajectory audit rejected the Cartesian path: "
                    + "; ".join(trajectory_audit["violations"])
                )
            else:
                node.get_logger().error(
                    "MoveIt did not compute the complete Cartesian path"
                )
            return 10
        if args.execute and not execution_succeeded:
            node.get_logger().error(
                execution_failure_reason or "trajectory execution did not succeed"
            )
            return 11
        return 0
    finally:
        if listener is not None:
            listener.unregister()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
