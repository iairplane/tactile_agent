#!/usr/bin/python3
"""Plan or execute one bounded absolute configured-EE Cartesian pose sequence.

The input waypoints are absolute configured-end-effector poses in ``--base-frame``.
Every pose is converted to ``fr3_link8`` with the explicitly supplied
``T_link8_configured_ee`` before all waypoints are submitted in one
GetCartesianPath request.  The complete returned FR3 joint path is audited and,
only when explicitly requested, that exact path is sent to ExecuteTrajectory.

Planning is the default.  This tool never commands the gripper.
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

from demo2gripper.cartesian_pose_sequence import (  # noqa: E402
    MAX_CARTESIAN_SPEED_M_S,
    CartesianPose,
    configured_ee_to_link8_waypoints,
    load_pose_sequence,
    validate_cartesian_speed_m_s,
    validate_pose_sequence_motion,
)
from demo2gripper.pose_motion import (  # noqa: E402
    compose_pose,
    normalize_quaternion_xyzw,
    pose_delta,
)
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

MAX_SCALING_FACTOR = 0.08
MAX_CARTESIAN_STEP_M = 0.010


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--link8-to-configured-ee-position-m",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="translation of the configured EE expressed in fr3_link8",
    )
    parser.add_argument(
        "--link8-to-configured-ee-orientation-xyzw",
        type=float,
        nargs=4,
        required=True,
        metavar=("QX", "QY", "QZ", "QW"),
        help="orientation of the configured EE expressed in fr3_link8",
    )
    parser.add_argument("--base-frame", default="fr3_link0")
    parser.add_argument("--tool-link", default="fr3_link8")
    parser.add_argument("--group", default="fr3_arm")
    parser.add_argument("--service-name", default="/compute_cartesian_path")
    parser.add_argument("--action-name", default="/execute_trajectory")
    parser.add_argument("--controller-name", default="fr3_arm_controller")
    parser.add_argument("--server-timeout-s", type=float, default=15.0)
    parser.add_argument("--max-step-m", type=float, default=0.005)
    parser.add_argument("--jump-threshold", type=float, default=2.0)
    parser.add_argument("--revolute-jump-threshold-rad", type=float, default=0.10)
    parser.add_argument("--velocity-scale", type=float, default=0.03)
    parser.add_argument("--acceleration-scale", type=float, default=0.03)
    parser.add_argument(
        "--max-cartesian-speed-m-s",
        type=float,
        default=MAX_CARTESIAN_SPEED_M_S,
    )
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
        "--execute",
        action="store_true",
        help="audit and execute the exact fully computed Cartesian path",
    )
    parser.add_argument(
        "--accept-arm-only-collision-model",
        action="store_true",
        help="acknowledge that the installed Hand-E is absent from the collision model",
    )
    parser.add_argument(
        "--allow-generic-kernel",
        action="store_true",
        help="allow execution although the current kernel name lacks 'realtime'",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "server_timeout_s",
        "max_step_m",
        "jump_threshold",
        "revolute_jump_threshold_rad",
        "velocity_scale",
        "acceleration_scale",
        "min_joint_limit_margin_rad",
        "max_joint_sweep_rad",
        "max_total_joint_sweep_rad",
    )
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if args.max_step_m > MAX_CARTESIAN_STEP_M:
        raise ValueError(f"--max-step-m is capped at {MAX_CARTESIAN_STEP_M:.3f} m")
    for name in ("velocity_scale", "acceleration_scale"):
        if getattr(args, name) > MAX_SCALING_FACTOR:
            raise ValueError(
                f"--{name.replace('_', '-')} is capped at {MAX_SCALING_FACTOR:.2f}"
            )
    validate_cartesian_speed_m_s(args.max_cartesian_speed_m_s)
    normalize_quaternion_xyzw(args.link8_to_configured_ee_orientation_xyzw)
    if not all(
        math.isfinite(value) for value in args.link8_to_configured_ee_position_m
    ):
        raise ValueError("T_link8_configured_ee translation must be finite")
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


def pose_from_transform(transform: object) -> CartesianPose:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return CartesianPose(
        position_m=(translation.x, translation.y, translation.z),
        orientation_xyzw=normalize_quaternion_xyzw(
            (rotation.x, rotation.y, rotation.z, rotation.w)
        ),
    )


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


def make_request(
    args: argparse.Namespace, waypoints: list[CartesianPose]
) -> GetCartesianPath.Request:
    poses: list[Pose] = []
    for waypoint in waypoints:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = waypoint.position_m
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = waypoint.orientation_xyzw
        poses.append(pose)

    request = GetCartesianPath.Request()
    request.header.frame_id = args.base_frame
    request.start_state.is_diff = True
    request.group_name = args.group
    request.link_name = args.tool_link
    request.waypoints = poses
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


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        configured_ee_waypoints = load_pose_sequence(
            args.input, expected_base_frame=args.base_frame
        )
        link8_waypoints = configured_ee_to_link8_waypoints(
            configured_ee_waypoints,
            link8_to_configured_ee_position_m=(
                args.link8_to_configured_ee_position_m
            ),
            link8_to_configured_ee_orientation_xyzw=(
                args.link8_to_configured_ee_orientation_xyzw
            ),
        )
    except ValueError as exc:
        print(f"argument/input error: {exc}", file=sys.stderr)
        return 2

    offset_position = tuple(args.link8_to_configured_ee_position_m)
    offset_orientation = normalize_quaternion_xyzw(
        args.link8_to_configured_ee_orientation_xyzw
    )

    rclpy.init()
    node = Node("demo2gripper_cartesian_pose_sequence")
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

        start_link8 = pose_from_transform(start_transform)
        start_ee_position, start_ee_orientation = compose_pose(
            start_link8.position_m,
            start_link8.orientation_xyzw,
            offset_position,
            offset_orientation,
        )
        start_configured_ee = CartesianPose(
            tuple(start_ee_position), start_ee_orientation
        )
        try:
            sequence_validation = validate_pose_sequence_motion(
                start_configured_ee, configured_ee_waypoints
            )
        except ValueError as exc:
            summary = {
                "schema": "demo2gripper.moveit_cartesian_pose_sequence.v1",
                "mode": "execute" if args.execute else "plan_only",
                "input": str(args.input),
                "base_frame": args.base_frame,
                "tool_link": args.tool_link,
                "configured_ee_waypoints": [
                    waypoint.as_dict() for waypoint in configured_ee_waypoints
                ],
                "start_configured_ee": start_configured_ee.as_dict(),
                "planning_succeeded": False,
                "execution_requested": args.execute,
                "execution_dispatched": False,
                "execution_succeeded": False if args.execute else None,
                "failure_reason": f"bounded pose-sequence check failed: {exc}",
            }
            write_summary(args.output, summary)
            node.get_logger().error(summary["failure_reason"])
            return 4

        node.get_logger().info(
            f"requesting {len(link8_waypoints)} absolute configured-EE Cartesian "
            f"waypoints ({sequence_validation['total_cartesian_length_m'] * 1000.0:.1f} "
            f"mm total); mode={'execute' if args.execute else 'plan_only'}"
        )
        client = node.create_client(GetCartesianPath, args.service_name)
        if not client.wait_for_service(timeout_sec=args.server_timeout_s):
            node.get_logger().error(
                f"Cartesian path service unavailable: {args.service_name}"
            )
            return 5
        future = client.call_async(make_request(args, link8_waypoints))
        rclpy.spin_until_future_complete(node, future)
        response = future.result()
        if response is None:
            node.get_logger().error("Cartesian path service returned no response")
            return 6

        trajectory = trajectory_dict(response.solution)
        complete_path = response.fraction == 1.0
        moveit_planning_succeeded = (
            response.error_code.val == MoveItErrorCodes.SUCCESS
            and complete_path
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
                execution_failure_reason = (
                    "MoveIt did not compute a complete Cartesian path with fraction 1"
                )
            elif not trajectory_audit["accepted"]:
                execution_failure_reason = (
                    "planned Cartesian path failed the FR3 joint audit"
                )
            else:
                action = ActionClient(node, ExecuteTrajectory, args.action_name)
                if not action.wait_for_server(timeout_sec=args.server_timeout_s):
                    execution_failure_reason = (
                        f"trajectory execution action unavailable: {args.action_name}"
                    )
                else:
                    goal = ExecuteTrajectory.Goal()
                    goal.trajectory = response.solution
                    goal.controller_names = [args.controller_name]
                    goal_future = action.send_goal_async(goal)
                    rclpy.spin_until_future_complete(node, goal_future)
                    handle = goal_future.result()
                    if handle is None or not handle.accepted:
                        execution_failure_reason = "trajectory execution goal was rejected"
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
                            execution_error_message = wrapped.result.error_code.message

        measured_link8 = None
        measured_configured_ee = None
        measured_final_position_error_m = None
        measured_final_rotation_error_rad = None
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
                measured_link8 = pose_from_transform(end_transform)
                measured_ee_position, measured_ee_orientation = compose_pose(
                    measured_link8.position_m,
                    measured_link8.orientation_xyzw,
                    offset_position,
                    offset_orientation,
                )
                measured_configured_ee = CartesianPose(
                    tuple(measured_ee_position), measured_ee_orientation
                )
                final_target = configured_ee_waypoints[-1]
                (
                    measured_final_position_error_m,
                    measured_final_rotation_error_rad,
                ) = pose_delta(
                    measured_configured_ee.position_m,
                    measured_configured_ee.orientation_xyzw,
                    final_target.position_m,
                    final_target.orientation_xyzw,
                )
            except TransformException as exc:
                measurement_error = f"post-execution TF lookup failed: {exc}"
                node.get_logger().error(measurement_error)

        execution_succeeded = (
            planning_succeeded
            and execution_dispatched
            and execution_status == GoalStatus.STATUS_SUCCEEDED
            and execution_error_code == MoveItErrorCodes.SUCCESS
            and measured_configured_ee is not None
        )
        summary = {
            "schema": "demo2gripper.moveit_cartesian_pose_sequence.v1",
            "mode": "execute" if args.execute else "plan_only",
            "input": str(args.input),
            "input_pose_semantics": f"T_{args.base_frame}_configured_ee",
            "moveit_pose_semantics": f"T_{args.base_frame}_{args.tool_link}",
            "arm_only_collision_model": True,
            "base_frame": args.base_frame,
            "tool_link": args.tool_link,
            "group": args.group,
            "start_link8": start_link8.as_dict(),
            "start_configured_ee": start_configured_ee.as_dict(),
            "link8_to_configured_ee": {
                "translation_m": list(offset_position),
                "orientation_xyzw": list(offset_orientation),
            },
            "configured_ee_waypoints": [
                waypoint.as_dict() for waypoint in configured_ee_waypoints
            ],
            "moveit_link8_waypoints": [
                waypoint.as_dict() for waypoint in link8_waypoints
            ],
            "sequence_validation": sequence_validation,
            "max_step_m": args.max_step_m,
            "velocity_scale": args.velocity_scale,
            "acceleration_scale": args.acceleration_scale,
            "max_cartesian_speed_m_s": args.max_cartesian_speed_m_s,
            "moveit_error_code": response.error_code.val,
            "moveit_error_message": response.error_code.message,
            "path_fraction": response.fraction,
            "complete_path_required": True,
            "trajectory": trajectory,
            "moveit_planning_succeeded": moveit_planning_succeeded,
            "trajectory_audit": trajectory_audit,
            "planning_succeeded": planning_succeeded,
            "execution_requested": args.execute,
            "execution_policy": "plan_then_audit_then_execute_exact_trajectory",
            "execution_dispatched": execution_dispatched,
            "execution_action_name": args.action_name,
            "execution_controller_name": args.controller_name,
            "execution_status": execution_status,
            "execution_error_code": execution_error_code,
            "execution_error_message": execution_error_message,
            "execution_failure_reason": execution_failure_reason,
            "execution_succeeded": execution_succeeded if args.execute else None,
            "measurement_error": measurement_error,
            "measured_end_link8": (
                measured_link8.as_dict() if measured_link8 is not None else None
            ),
            "measured_end_configured_ee": (
                measured_configured_ee.as_dict()
                if measured_configured_ee is not None
                else None
            ),
            "measured_final_position_error_m": measured_final_position_error_m,
            "measured_final_rotation_error_rad": measured_final_rotation_error_rad,
        }
        write_summary(args.output, summary)

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
