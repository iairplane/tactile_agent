#!/usr/bin/python3
"""Plan or execute one bounded FR3 lift while holding the current orientation.

This is a hardware-integration smoke test, not a TaskSkill executor.  It uses the
current ``fr3_link8`` pose and requests at most 20 mm vertical motion through the
MoveIt ``MoveGroup`` action.  Planning is the default; execution needs explicit
flags because the current robot model does not include the installed Hand-E.

Run this script with the ROS system Python after sourcing ROS and the local
franka_ros2 overlay.  Do not run it from the repository's Python virtualenv.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformException, TransformListener

MAX_LIFT_M = 0.020


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-z-m", type=float, default=MAX_LIFT_M)
    parser.add_argument("--base-frame", default="fr3_link0")
    parser.add_argument("--tool-link", default="fr3_link8")
    parser.add_argument("--group", default="fr3_arm")
    parser.add_argument("--action-name", default="/move_action")
    parser.add_argument("--planning-time-s", type=float, default=5.0)
    parser.add_argument("--server-timeout-s", type=float, default=15.0)
    parser.add_argument("--velocity-scale", type=float, default=0.05)
    parser.add_argument("--acceleration-scale", type=float, default=0.05)
    parser.add_argument("--position-tolerance-m", type=float, default=0.001)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.02)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/fr3_lift_smoke_plan.json"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute through MoveIt instead of returning a plan only",
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


def validate_args(args: argparse.Namespace) -> None:
    finite_positive = (
        "planning_time_s",
        "server_timeout_s",
        "velocity_scale",
        "acceleration_scale",
        "position_tolerance_m",
        "orientation_tolerance_rad",
    )
    for name in finite_positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if not math.isfinite(args.delta_z_m) or args.delta_z_m == 0.0:
        raise ValueError("--delta-z-m must be finite and non-zero")
    if abs(args.delta_z_m) > MAX_LIFT_M:
        raise ValueError(f"absolute --delta-z-m is capped at {MAX_LIFT_M:.3f} m")
    for name in ("velocity_scale", "acceleration_scale"):
        if getattr(args, name) > 0.05:
            raise ValueError(f"--{name.replace('_', '-')} is capped at 0.05")
    if args.execute and not args.accept_arm_only_collision_model:
        raise ValueError("--execute requires --accept-arm-only-collision-model")
    if args.execute and "realtime" not in platform.release().lower():
        if not args.allow_generic_kernel:
            raise ValueError(
                "execution on this kernel requires --allow-generic-kernel; "
                f"current kernel is {platform.release()}"
            )


def make_goal(args: argparse.Namespace, transform: object) -> MoveGroup.Goal:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    target = Pose()
    target.position.x = translation.x
    target.position.y = translation.y
    target.position.z = translation.z + args.delta_z_m
    target.orientation = rotation

    region = SolidPrimitive()
    region.type = SolidPrimitive.BOX
    diameter = 2.0 * args.position_tolerance_m
    region.dimensions = [diameter, diameter, diameter]
    region_pose = Pose()
    region_pose.position = target.position
    region_pose.orientation.w = 1.0

    position = PositionConstraint()
    position.header.frame_id = args.base_frame
    position.link_name = args.tool_link
    position.constraint_region.primitives = [region]
    position.constraint_region.primitive_poses = [region_pose]
    position.weight = 1.0

    orientation = OrientationConstraint()
    orientation.header.frame_id = args.base_frame
    orientation.link_name = args.tool_link
    orientation.orientation = target.orientation
    orientation.absolute_x_axis_tolerance = args.orientation_tolerance_rad
    orientation.absolute_y_axis_tolerance = args.orientation_tolerance_rad
    orientation.absolute_z_axis_tolerance = args.orientation_tolerance_rad
    orientation.parameterization = OrientationConstraint.ROTATION_VECTOR
    orientation.weight = 1.0

    constraints = Constraints()
    constraints.name = "bounded_agentic_lift_smoke"
    constraints.position_constraints = [position]
    constraints.orientation_constraints = [orientation]

    goal = MoveGroup.Goal()
    goal.request.group_name = args.group
    goal.request.num_planning_attempts = 5
    goal.request.allowed_planning_time = args.planning_time_s
    goal.request.max_velocity_scaling_factor = args.velocity_scale
    goal.request.max_acceleration_scaling_factor = args.acceleration_scale
    goal.request.workspace_parameters.header.frame_id = args.base_frame
    goal.request.workspace_parameters.min_corner.x = -2.0
    goal.request.workspace_parameters.min_corner.y = -2.0
    goal.request.workspace_parameters.min_corner.z = -0.5
    goal.request.workspace_parameters.max_corner.x = 2.0
    goal.request.workspace_parameters.max_corner.y = 2.0
    goal.request.workspace_parameters.max_corner.z = 2.0
    goal.request.start_state.is_diff = True
    goal.request.goal_constraints = [constraints]
    goal.planning_options.plan_only = not args.execute
    goal.planning_options.look_around = False
    goal.planning_options.replan = False
    goal.planning_options.planning_scene_diff.is_diff = True
    goal.planning_options.planning_scene_diff.robot_state.is_diff = True
    return goal


def duration_seconds(value: object) -> float:
    return float(value.sec) + float(value.nanosec) * 1e-9


def result_summary(args: argparse.Namespace, transform: object, wrapped: object) -> dict:
    result = wrapped.result
    trajectory = result.planned_trajectory.joint_trajectory
    return {
        "schema": "demo2gripper.moveit_lift_smoke.v1",
        "mode": "execute" if args.execute else "plan_only",
        "arm_only_collision_model": True,
        "base_frame": args.base_frame,
        "tool_link": args.tool_link,
        "group": args.group,
        "requested_translation_base_m": [0.0, 0.0, args.delta_z_m],
        "start_position_base_m": [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ],
        "held_orientation_xyzw": [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ],
        "moveit_error_code": result.error_code.val,
        "moveit_error_message": result.error_code.message,
        "action_status": wrapped.status,
        "planning_time_s": result.planning_time,
        "trajectory": {
            "joint_names": list(trajectory.joint_names),
            "point_count": len(trajectory.points),
            "points": [
                {
                    "positions": list(point.positions),
                    "velocities": list(point.velocities),
                    "accelerations": list(point.accelerations),
                    "time_from_start_s": duration_seconds(point.time_from_start),
                }
                for point in trajectory.points
            ],
        },
    }


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"argument error: {exc}", file=sys.stderr)
        return 2

    rclpy.init()
    node = Node("demo2gripper_lift_smoke")
    listener = None
    try:
        buffer = Buffer()
        listener = TransformListener(buffer, node)
        deadline = time.monotonic() + args.server_timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if buffer.can_transform(args.base_frame, args.tool_link, Time()):
                break
        try:
            transform = buffer.lookup_transform(
                args.base_frame,
                args.tool_link,
                Time(),
                timeout=Duration(seconds=0.1),
            )
        except TransformException as exc:
            node.get_logger().error(f"TF lookup failed: {exc}")
            return 3

        node.get_logger().info(
            f"current {args.tool_link} z={transform.transform.translation.z:.4f} m; "
            f"requesting {args.delta_z_m * 1000.0:+.1f} mm; "
            f"mode={'execute' if args.execute else 'plan_only'}"
        )
        client = ActionClient(node, MoveGroup, args.action_name)
        if not client.wait_for_server(timeout_sec=args.server_timeout_s):
            node.get_logger().error(f"MoveGroup action unavailable: {args.action_name}")
            return 4

        goal_future = client.send_goal_async(make_goal(args, transform))
        rclpy.spin_until_future_complete(node, goal_future)
        handle = goal_future.result()
        if handle is None or not handle.accepted:
            node.get_logger().error("MoveGroup rejected the lift smoke goal")
            return 5

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            node.get_logger().error("MoveGroup returned no result")
            return 6

        summary = result_summary(args, transform, wrapped)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        succeeded = (
            wrapped.status == GoalStatus.STATUS_SUCCEEDED
            and wrapped.result.error_code.val == MoveItErrorCodes.SUCCESS
            and summary["trajectory"]["point_count"] > 0
        )
        headline = {
            key: value for key, value in summary.items() if key != "trajectory"
        }
        print(json.dumps(headline, indent=2))
        print(
            f"trajectory points: {summary['trajectory']['point_count']}; "
            f"saved: {args.output}"
        )
        if not succeeded:
            node.get_logger().error("MoveIt did not produce a successful non-empty trajectory")
            return 7
        return 0
    finally:
        if listener is not None:
            listener.unregister()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
