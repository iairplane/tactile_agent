#!/usr/bin/python3
"""Plan or execute one bounded absolute FR3 tool pose goal.

The target is expressed in ``--base-frame`` as an XYZ position in metres and an
XYZW quaternion.  It can name ``fr3_link8`` directly or the configured end
effector; the latter is converted to a link8 pose with an explicit rigid offset.
MoveIt performs IK, arm collision checking and trajectory planning.  Every returned
joint path is audited before it can be sent unchanged to ExecuteTrajectory.  Planning
is the default.  Execution needs an explicit acknowledgement because the installed
Hand-E is absent from the current arm-only collision model.

Run with the ROS system Python after sourcing ROS and the local franka_ros2
overlay.  This is a low-speed integration tool for a staged task executor; it
does not make contact decisions and it does not command the gripper.
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

from demo2gripper.pose_motion import (
    QuaternionXYZW,
    classify_measured_pose_outcome,
    compose_pose,
    link8_pose_from_configured_ee,
    normalize_quaternion_xyzw,
    pose_delta,
    quaternion_angle_rad,
)
from demo2gripper.trajectory_safety import (
    DEFAULT_MAX_JOINT_SWEEP_RAD,
    DEFAULT_MAX_TOTAL_JOINT_SWEEP_RAD,
    DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD,
    audit_fr3_joint_trajectory,
    validate_fr3_trajectory_audit_thresholds,
)

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
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

DEFAULT_MAX_TRANSLATION_M = 0.160
HARD_MAX_TRANSLATION_M = 0.600
DEFAULT_MAX_ROTATION_RAD = math.radians(60.0)
HARD_MAX_ROTATION_RAD = math.pi
MAX_VELOCITY_SCALE = 0.05
MAX_ACCELERATION_SCALE = 0.05
MAX_POSITION_TOLERANCE_M = 0.003
MAX_ORIENTATION_TOLERANCE_RAD = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--target-position-m",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="absolute target position of --target-frame in --base-frame",
    )
    parser.add_argument(
        "--target-orientation-xyzw",
        type=float,
        nargs=4,
        required=True,
        metavar=("QX", "QY", "QZ", "QW"),
        help="absolute target orientation of --target-frame in --base-frame",
    )
    parser.add_argument(
        "--target-frame",
        choices=("link8", "configured-ee"),
        default="link8",
        help=(
            "frame represented by the target pose; configured-ee is converted "
            "to fr3_link8 using the supplied rigid transform"
        ),
    )
    parser.add_argument(
        "--link8-to-configured-ee-position-m",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.190),
        metavar=("X", "Y", "Z"),
        help="translation of the configured EE expressed in fr3_link8",
    )
    parser.add_argument(
        "--link8-to-configured-ee-orientation-xyzw",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 1.0),
        metavar=("QX", "QY", "QZ", "QW"),
        help="orientation of the configured EE expressed in fr3_link8",
    )
    parser.add_argument("--base-frame", default="fr3_link0")
    parser.add_argument("--tool-link", default="fr3_link8")
    parser.add_argument("--group", default="fr3_arm")
    parser.add_argument("--action-name", default="/move_action")
    parser.add_argument("--execute-action-name", default="/execute_trajectory")
    parser.add_argument("--controller-name", default="fr3_arm_controller")
    parser.add_argument("--planning-time-s", type=float, default=8.0)
    parser.add_argument("--planning-attempts", type=int, default=5)
    parser.add_argument("--server-timeout-s", type=float, default=15.0)
    parser.add_argument("--velocity-scale", type=float, default=0.03)
    parser.add_argument("--acceleration-scale", type=float, default=0.03)
    parser.add_argument(
        "--max-translation-m",
        type=float,
        default=DEFAULT_MAX_TRANSLATION_M,
        help="per-command current-to-target translation guard",
    )
    parser.add_argument(
        "--max-rotation-rad",
        type=float,
        default=DEFAULT_MAX_ROTATION_RAD,
        help="per-command current-to-target shortest rotation guard",
    )
    parser.add_argument(
        "--disable-pose-delta-guard",
        action="store_true",
        help=(
            "disable only the current-to-target translation and rotation guard; "
            "rigid-pose validation, MoveIt collision planning, joint-path audit, "
            "and measured endpoint checks remain active"
        ),
    )
    parser.add_argument("--position-tolerance-m", type=float, default=0.001)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.02)
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
        help="maximum allowed min-to-max sweep of any one joint in the complete plan",
    )
    parser.add_argument(
        "--max-total-joint-sweep-rad",
        type=float,
        default=DEFAULT_MAX_TOTAL_JOINT_SWEEP_RAD,
        help="maximum allowed sum of all seven joint sweeps in the complete plan",
    )
    parser.add_argument(
        "--disable-joint-sweep-guards",
        action="store_true",
        help=(
            "disable only per-joint and total sweep magnitude guards; malformed or "
            "non-finite trajectories and hardware joint-limit margin violations "
            "remain blocking"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/fr3_pose_goal_plan.json"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="plan, audit, then execute the exact accepted trajectory",
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
    finite_positive = (
        "planning_time_s",
        "server_timeout_s",
        "velocity_scale",
        "acceleration_scale",
        "max_translation_m",
        "max_rotation_rad",
        "position_tolerance_m",
        "orientation_tolerance_rad",
        "min_joint_limit_margin_rad",
        "max_joint_sweep_rad",
        "max_total_joint_sweep_rad",
    )
    for name in finite_positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if args.planning_attempts <= 0 or args.planning_attempts > 10:
        raise ValueError("--planning-attempts must be in [1, 10]")
    if args.max_translation_m > HARD_MAX_TRANSLATION_M:
        raise ValueError(
            f"--max-translation-m is capped at {HARD_MAX_TRANSLATION_M:.3f} m"
        )
    if args.max_rotation_rad > HARD_MAX_ROTATION_RAD:
        raise ValueError(
            f"--max-rotation-rad is capped at {HARD_MAX_ROTATION_RAD:.6f} rad"
        )
    if not all(math.isfinite(value) for value in args.target_position_m):
        raise ValueError("target position must contain three finite values")
    normalize_quaternion_xyzw(args.target_orientation_xyzw)
    if not all(
        math.isfinite(value) for value in args.link8_to_configured_ee_position_m
    ):
        raise ValueError("link8-to-configured-ee position must contain finite values")
    normalize_quaternion_xyzw(args.link8_to_configured_ee_orientation_xyzw)
    if args.velocity_scale > MAX_VELOCITY_SCALE:
        raise ValueError(
            f"--velocity-scale is capped at {MAX_VELOCITY_SCALE:.2f}"
        )
    if args.acceleration_scale > MAX_ACCELERATION_SCALE:
        raise ValueError(
            f"--acceleration-scale is capped at {MAX_ACCELERATION_SCALE:.2f}"
        )
    if args.position_tolerance_m > MAX_POSITION_TOLERANCE_M:
        raise ValueError(
            "--position-tolerance-m is capped at "
            f"{MAX_POSITION_TOLERANCE_M:.3f} m"
        )
    if args.orientation_tolerance_rad > MAX_ORIENTATION_TOLERANCE_RAD:
        raise ValueError(
            "--orientation-tolerance-rad is capped at "
            f"{MAX_ORIENTATION_TOLERANCE_RAD:.3f} rad"
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


def pose_from_transform(transform: object) -> tuple[list[float], QuaternionXYZW]:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    position = [translation.x, translation.y, translation.z]
    orientation = normalize_quaternion_xyzw(
        [rotation.x, rotation.y, rotation.z, rotation.w]
    )
    return position, orientation


def validate_pose_delta(
    start_position: list[float],
    start_orientation: QuaternionXYZW,
    target_position: list[float],
    target_orientation: QuaternionXYZW,
    *,
    max_translation_m: float,
    max_rotation_rad: float,
) -> tuple[float, float]:
    translation_m, rotation_rad = pose_delta(
        start_position, start_orientation, target_position, target_orientation
    )
    if translation_m > max_translation_m:
        raise ValueError(
            f"translation {translation_m:.6f} m exceeds the "
            f"{max_translation_m:.3f} m command cap"
        )
    if rotation_rad > max_rotation_rad:
        raise ValueError(
            f"rotation {math.degrees(rotation_rad):.3f} deg exceeds the "
            f"{math.degrees(max_rotation_rad):.1f} deg command cap"
        )
    if translation_m <= 1e-6 and rotation_rad <= 1e-5:
        raise ValueError("target pose is indistinguishable from the current pose")
    return translation_m, rotation_rad


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


def make_goal(
    args: argparse.Namespace,
    target_position: list[float],
    target_orientation: QuaternionXYZW,
) -> MoveGroup.Goal:
    target = Pose()
    target.position.x, target.position.y, target.position.z = target_position
    (
        target.orientation.x,
        target.orientation.y,
        target.orientation.z,
        target.orientation.w,
    ) = target_orientation

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
    constraints.name = "bounded_absolute_link8_pose"
    constraints.position_constraints = [position]
    constraints.orientation_constraints = [orientation]

    goal = MoveGroup.Goal()
    goal.request.group_name = args.group
    goal.request.num_planning_attempts = args.planning_attempts
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
    # Always ask MoveGroup for a plan only.  If execution was requested, the
    # exact returned trajectory is audited and then sent to ExecuteTrajectory.
    # This prevents MoveGroup from planning and executing before the caller can
    # inspect the actual joint path.
    goal.planning_options.plan_only = True
    goal.planning_options.look_around = False
    goal.planning_options.replan = False
    goal.planning_options.planning_scene_diff.is_diff = True
    goal.planning_options.planning_scene_diff.robot_state.is_diff = True
    return goal


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


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"argument error: {exc}", file=sys.stderr)
        return 2

    input_target_position = list(args.target_position_m)
    input_target_orientation = normalize_quaternion_xyzw(
        args.target_orientation_xyzw
    )
    link8_to_ee_position = list(args.link8_to_configured_ee_position_m)
    link8_to_ee_orientation = normalize_quaternion_xyzw(
        args.link8_to_configured_ee_orientation_xyzw
    )
    if args.target_frame == "configured-ee":
        target_position, target_orientation = link8_pose_from_configured_ee(
            input_target_position,
            input_target_orientation,
            link8_to_ee_position,
            link8_to_ee_orientation,
        )
    else:
        target_position = input_target_position
        target_orientation = input_target_orientation

    rclpy.init()
    node = Node("demo2gripper_pose_goal")
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

        start_position, start_orientation = pose_from_transform(start_transform)
        start_ee_position, start_ee_orientation = compose_pose(
            start_position,
            start_orientation,
            link8_to_ee_position,
            link8_to_ee_orientation,
        )
        try:
            if args.disable_pose_delta_guard:
                translation_m, rotation_rad = pose_delta(
                    start_position,
                    start_orientation,
                    target_position,
                    target_orientation,
                )
                if args.target_frame == "configured-ee":
                    input_translation_m, input_rotation_rad = pose_delta(
                        start_ee_position,
                        start_ee_orientation,
                        input_target_position,
                        input_target_orientation,
                    )
                else:
                    input_translation_m = translation_m
                    input_rotation_rad = rotation_rad
            else:
                translation_m, rotation_rad = validate_pose_delta(
                    start_position,
                    start_orientation,
                    target_position,
                    target_orientation,
                    max_translation_m=args.max_translation_m,
                    max_rotation_rad=args.max_rotation_rad,
                )
                if args.target_frame == "configured-ee":
                    input_translation_m, input_rotation_rad = validate_pose_delta(
                        start_ee_position,
                        start_ee_orientation,
                        input_target_position,
                        input_target_orientation,
                        max_translation_m=args.max_translation_m,
                        max_rotation_rad=args.max_rotation_rad,
                    )
                else:
                    input_translation_m = translation_m
                    input_rotation_rad = rotation_rad
        except ValueError as exc:
            node.get_logger().error(f"bounded-pose check failed: {exc}")
            return 4

        node.get_logger().info(
            f"requesting absolute {args.target_frame} pose as an {args.tool_link} goal: "
            f"translation={translation_m * 1000.0:.1f} mm, "
            f"rotation={math.degrees(rotation_rad):.1f} deg; "
            f"mode={'execute' if args.execute else 'plan_only'}"
        )
        client = ActionClient(node, MoveGroup, args.action_name)
        if not client.wait_for_server(timeout_sec=args.server_timeout_s):
            node.get_logger().error(f"MoveGroup action unavailable: {args.action_name}")
            return 5

        goal_future = client.send_goal_async(
            make_goal(args, target_position, target_orientation)
        )
        rclpy.spin_until_future_complete(node, goal_future)
        handle = goal_future.result()
        if handle is None or not handle.accepted:
            node.get_logger().error("MoveGroup rejected the absolute pose goal")
            return 6

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            node.get_logger().error("MoveGroup returned no result")
            return 7

        result = wrapped.result
        trajectory = trajectory_dict(result.planned_trajectory)
        moveit_planning_succeeded = (
            wrapped.status == GoalStatus.STATUS_SUCCEEDED
            and result.error_code.val == MoveItErrorCodes.SUCCESS
            and trajectory["point_count"] > 1
        )
        trajectory_audit = audit_fr3_joint_trajectory(
            trajectory,
            min_joint_limit_margin_rad=args.min_joint_limit_margin_rad,
            max_joint_sweep_rad=args.max_joint_sweep_rad,
            max_total_joint_sweep_rad=args.max_total_joint_sweep_rad,
            enforce_joint_sweep_guards=not args.disable_joint_sweep_guards,
        )
        planning_succeeded = moveit_planning_succeeded and trajectory_audit["accepted"]

        execution_dispatched = False
        execution_status = None
        execution_error_code = None
        execution_error_message = None
        execution_failure_reason = None
        if args.execute:
            if not moveit_planning_succeeded:
                execution_failure_reason = "MoveIt did not produce a complete plan"
            elif not trajectory_audit["accepted"]:
                execution_failure_reason = "planned trajectory failed the FR3 joint audit"
            else:
                execution_client = ActionClient(
                    node, ExecuteTrajectory, args.execute_action_name
                )
                if not execution_client.wait_for_server(
                    timeout_sec=args.server_timeout_s
                ):
                    execution_failure_reason = (
                        "trajectory execution action unavailable: "
                        f"{args.execute_action_name}"
                    )
                else:
                    execution_goal = ExecuteTrajectory.Goal()
                    execution_goal.trajectory = result.planned_trajectory
                    execution_goal.controller_names = [args.controller_name]
                    execution_goal_future = execution_client.send_goal_async(
                        execution_goal
                    )
                    rclpy.spin_until_future_complete(node, execution_goal_future)
                    execution_handle = execution_goal_future.result()
                    if execution_handle is None or not execution_handle.accepted:
                        execution_failure_reason = (
                            "trajectory execution goal was rejected"
                        )
                    else:
                        execution_dispatched = True
                        execution_result_future = execution_handle.get_result_async()
                        rclpy.spin_until_future_complete(
                            node, execution_result_future
                        )
                        execution_wrapped = execution_result_future.result()
                        if execution_wrapped is None:
                            execution_failure_reason = (
                                "trajectory execution returned no result"
                            )
                        else:
                            execution_status = execution_wrapped.status
                            execution_error_code = (
                                execution_wrapped.result.error_code.val
                            )
                            execution_error_message = (
                                execution_wrapped.result.error_code.message
                            )

        measured_position = None
        measured_orientation = None
        measured_ee_position = None
        measured_ee_orientation = None
        measured_target_position_error_m = None
        measured_target_rotation_error_rad = None
        measured_input_target_position_error_m = None
        measured_input_target_rotation_error_rad = None
        measured_motion_translation_m = None
        measured_motion_rotation_rad = None
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
                measured_position, measured_orientation = pose_from_transform(
                    end_transform
                )
                measured_ee_position, measured_ee_orientation = compose_pose(
                    measured_position,
                    measured_orientation,
                    link8_to_ee_position,
                    link8_to_ee_orientation,
                )
                measured_target_position_error_m = math.sqrt(
                    sum(
                        (measured - target) ** 2
                        for measured, target in zip(
                            measured_position, target_position, strict=True
                        )
                    )
                )
                measured_target_rotation_error_rad = quaternion_angle_rad(
                    measured_orientation, target_orientation
                )
                measured_input_position = (
                    measured_ee_position
                    if args.target_frame == "configured-ee"
                    else measured_position
                )
                measured_input_orientation = (
                    measured_ee_orientation
                    if args.target_frame == "configured-ee"
                    else measured_orientation
                )
                measured_input_target_position_error_m = math.sqrt(
                    sum(
                        (measured - target) ** 2
                        for measured, target in zip(
                            measured_input_position,
                            input_target_position,
                            strict=True,
                        )
                    )
                )
                measured_input_target_rotation_error_rad = quaternion_angle_rad(
                    measured_input_orientation, input_target_orientation
                )
            except TransformException as exc:
                measurement_error = f"post-execution TF lookup failed: {exc}"
                node.get_logger().error(measurement_error)

        execution_action_reported_success = (
            execution_status == GoalStatus.STATUS_SUCCEEDED
            and execution_error_code == MoveItErrorCodes.SUCCESS
        )
        execution_outcome = None
        if args.execute:
            if not moveit_planning_succeeded:
                execution_outcome = "moveit_planning_failed"
            elif not trajectory_audit["accepted"]:
                execution_outcome = "trajectory_audit_rejected"
            elif not execution_dispatched:
                execution_outcome = "execution_not_dispatched"
            elif not execution_action_reported_success:
                execution_outcome = "execution_action_failed"
            elif (
                measured_position is None
                or measured_orientation is None
                or measured_target_position_error_m is None
                or measured_target_rotation_error_rad is None
            ):
                execution_outcome = "post_execution_measurement_unavailable"
                if execution_failure_reason is None:
                    execution_failure_reason = (
                        measurement_error
                        or "trajectory action reported success but endpoint feedback is unavailable"
                    )
            else:
                (
                    execution_outcome,
                    measured_motion_translation_m,
                    measured_motion_rotation_rad,
                ) = classify_measured_pose_outcome(
                    start_position=start_position,
                    start_orientation=start_orientation,
                    measured_position=measured_position,
                    measured_orientation=measured_orientation,
                    target_position_error_m=measured_target_position_error_m,
                    target_rotation_error_rad=measured_target_rotation_error_rad,
                    position_tolerance_m=MAX_POSITION_TOLERANCE_M,
                    orientation_tolerance_rad=MAX_ORIENTATION_TOLERANCE_RAD,
                )
                if (
                    execution_outcome
                    == "controller_reported_success_without_measured_motion"
                ):
                    execution_failure_reason = (
                        "trajectory action reported success, but measured TCP stayed at "
                        "the start pose; the hardware command mode or controller path is "
                        "likely inactive"
                    )
                elif execution_outcome == "measured_endpoint_mismatch":
                    execution_failure_reason = (
                        "trajectory action reported success, but measured TCP did not "
                        "reach the requested endpoint"
                    )

        execution_succeeded = (
            planning_succeeded
            and execution_action_reported_success
            and execution_outcome == "measured_target_reached"
        )
        summary = {
            "schema": "demo2gripper.moveit_pose_goal.v2",
            "mode": "execute" if args.execute else "plan_only",
            "input_target_frame": args.target_frame,
            "input_target_semantics": (
                f"T_{args.base_frame}_configured_ee"
                if args.target_frame == "configured-ee"
                else f"T_{args.base_frame}_{args.tool_link}"
            ),
            "moveit_target_semantics": f"T_{args.base_frame}_{args.tool_link}",
            "arm_only_collision_model": True,
            "base_frame": args.base_frame,
            "tool_link": args.tool_link,
            "group": args.group,
            "start_position_base_m": start_position,
            "start_orientation_xyzw": list(start_orientation),
            "start_configured_ee_position_base_m": start_ee_position,
            "start_configured_ee_orientation_xyzw": list(start_ee_orientation),
            "input_target_position_base_m": input_target_position,
            "input_target_orientation_xyzw": list(input_target_orientation),
            "link8_to_configured_ee": {
                "translation_m": link8_to_ee_position,
                "orientation_xyzw": list(link8_to_ee_orientation),
            },
            "target_position_base_m": target_position,
            "target_orientation_xyzw": list(target_orientation),
            "input_requested_translation_norm_m": input_translation_m,
            "input_requested_rotation_rad": input_rotation_rad,
            "requested_translation_norm_m": translation_m,
            "requested_rotation_rad": rotation_rad,
            "requested_rotation_deg": math.degrees(rotation_rad),
            "position_tolerance_m": args.position_tolerance_m,
            "orientation_tolerance_rad": args.orientation_tolerance_rad,
            "velocity_scale": args.velocity_scale,
            "acceleration_scale": args.acceleration_scale,
            "max_translation_m": args.max_translation_m,
            "max_rotation_rad": args.max_rotation_rad,
            "pose_delta_guard_enabled": not args.disable_pose_delta_guard,
            "pose_delta_guard_disabled_by_operator_config": (
                args.disable_pose_delta_guard
            ),
            "joint_sweep_guards_enabled": not args.disable_joint_sweep_guards,
            "joint_sweep_guards_disabled_by_operator_config": (
                args.disable_joint_sweep_guards
            ),
            "moveit_error_code": result.error_code.val,
            "moveit_error_message": result.error_code.message,
            "planning_action_status": wrapped.status,
            "action_status": (
                execution_status if args.execute else wrapped.status
            ),
            "planning_time_s": result.planning_time,
            "trajectory": trajectory,
            "moveit_planning_succeeded": moveit_planning_succeeded,
            "trajectory_audit": trajectory_audit,
            "planning_succeeded": planning_succeeded,
            "execution_requested": args.execute,
            "execution_policy": "plan_then_audit_then_execute_exact_trajectory",
            "execution_dispatched": execution_dispatched,
            "execution_action_name": args.execute_action_name,
            "execution_controller_name": args.controller_name,
            "execution_status": execution_status,
            "execution_error_code": execution_error_code,
            "execution_error_message": execution_error_message,
            "execution_action_reported_success": execution_action_reported_success,
            "execution_outcome": execution_outcome,
            "execution_failure_reason": execution_failure_reason,
            "execution_succeeded": execution_succeeded if args.execute else None,
            "measurement_error": measurement_error,
            "measured_end_position_base_m": measured_position,
            "measured_end_orientation_xyzw": (
                list(measured_orientation) if measured_orientation is not None else None
            ),
            "measured_end_configured_ee_position_base_m": measured_ee_position,
            "measured_end_configured_ee_orientation_xyzw": (
                list(measured_ee_orientation)
                if measured_ee_orientation is not None
                else None
            ),
            "measured_target_position_error_m": measured_target_position_error_m,
            "measured_target_rotation_error_rad": measured_target_rotation_error_rad,
            "measured_input_target_position_error_m": (
                measured_input_target_position_error_m
            ),
            "measured_input_target_rotation_error_rad": (
                measured_input_target_rotation_error_rad
            ),
            "measured_motion_translation_m": measured_motion_translation_m,
            "measured_motion_rotation_rad": measured_motion_rotation_rad,
        }
        succeeded = execution_succeeded if args.execute else planning_succeeded
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        headline = {key: value for key, value in summary.items() if key != "trajectory"}
        print(json.dumps(headline, indent=2))
        print(f"trajectory points: {trajectory['point_count']}; saved: {args.output}")
        if not succeeded:
            if trajectory_audit["violations"]:
                node.get_logger().error(
                    "trajectory audit rejected the plan: "
                    + "; ".join(trajectory_audit["violations"])
                )
            else:
                node.get_logger().error(
                    execution_failure_reason
                    or "MoveIt did not produce a successful executable trajectory"
                )
            return 9
        return 0
    finally:
        if listener is not None:
            listener.unregister()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
