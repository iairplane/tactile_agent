#!/usr/bin/python3
"""Plan or execute one bounded absolute FR3 arm joint goal.

MoveIt always plans first.  The complete returned trajectory is audited before
the exact trajectory can be sent to ExecuteTrajectory.  Planning is the
default; execution requires the arm-only collision-model acknowledgement.
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

from demo2gripper.trajectory_safety import (
    DEFAULT_MAX_JOINT_SWEEP_RAD,
    DEFAULT_MAX_TOTAL_JOINT_SWEEP_RAD,
    DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD,
    FR3_JOINT_POSITION_LIMITS_RAD,
    audit_fr3_joint_trajectory,
    validate_fr3_trajectory_audit_thresholds,
)

import rclpy
from action_msgs.msg import GoalStatus
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINT_NAMES = tuple(FR3_JOINT_POSITION_LIMITS_RAD)
MAX_VELOCITY_SCALE = 0.05
MAX_ACCELERATION_SCALE = 0.05
MAX_JOINT_TOLERANCE_RAD = 0.01
MAX_MEASURED_ERROR_RAD = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--target-joint-positions-rad",
        type=float,
        nargs=7,
        required=True,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    )
    parser.add_argument("--group", default="fr3_arm")
    parser.add_argument("--action-name", default="/move_action")
    parser.add_argument("--execute-action-name", default="/execute_trajectory")
    parser.add_argument("--controller-name", default="fr3_arm_controller")
    parser.add_argument("--joint-state-topic", default="/joint_states")
    parser.add_argument("--planning-time-s", type=float, default=8.0)
    parser.add_argument("--planning-attempts", type=int, default=5)
    parser.add_argument("--server-timeout-s", type=float, default=15.0)
    parser.add_argument("--velocity-scale", type=float, default=0.02)
    parser.add_argument("--acceleration-scale", type=float, default=0.02)
    parser.add_argument("--joint-tolerance-rad", type=float, default=0.001)
    parser.add_argument("--measured-tolerance-rad", type=float, default=0.01)
    parser.add_argument(
        "--min-joint-limit-margin-rad",
        type=float,
        default=DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD,
    )
    parser.add_argument(
        "--max-joint-sweep-rad",
        type=float,
        default=DEFAULT_MAX_JOINT_SWEEP_RAD,
    )
    parser.add_argument(
        "--max-total-joint-sweep-rad",
        type=float,
        default=DEFAULT_MAX_TOTAL_JOINT_SWEEP_RAD,
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/fr3_joint_goal_plan.json")
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-arm-only-collision-model", action="store_true")
    parser.add_argument("--allow-generic-kernel", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    target = list(args.target_joint_positions_rad)
    if not all(math.isfinite(value) for value in target):
        raise ValueError("target joint positions must be finite")
    if args.planning_attempts <= 0 or args.planning_attempts > 10:
        raise ValueError("--planning-attempts must be in [1, 10]")
    for name in (
        "planning_time_s",
        "server_timeout_s",
        "velocity_scale",
        "acceleration_scale",
        "joint_tolerance_rad",
        "measured_tolerance_rad",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if args.velocity_scale > MAX_VELOCITY_SCALE:
        raise ValueError(f"--velocity-scale is capped at {MAX_VELOCITY_SCALE:.2f}")
    if args.acceleration_scale > MAX_ACCELERATION_SCALE:
        raise ValueError(
            f"--acceleration-scale is capped at {MAX_ACCELERATION_SCALE:.2f}"
        )
    if args.joint_tolerance_rad > MAX_JOINT_TOLERANCE_RAD:
        raise ValueError(
            f"--joint-tolerance-rad is capped at {MAX_JOINT_TOLERANCE_RAD:.3f}"
        )
    if args.measured_tolerance_rad > MAX_MEASURED_ERROR_RAD:
        raise ValueError(
            f"--measured-tolerance-rad is capped at {MAX_MEASURED_ERROR_RAD:.3f}"
        )
    validate_fr3_trajectory_audit_thresholds(
        min_joint_limit_margin_rad=args.min_joint_limit_margin_rad,
        max_joint_sweep_rad=args.max_joint_sweep_rad,
        max_total_joint_sweep_rad=args.max_total_joint_sweep_rad,
    )
    for name, position in zip(JOINT_NAMES, target, strict=True):
        lower, upper = FR3_JOINT_POSITION_LIMITS_RAD[name]
        margin = min(position - lower, upper - position)
        if margin < args.min_joint_limit_margin_rad:
            raise ValueError(
                f"{name} target hardware-limit margin {margin:.6f} rad is below "
                f"{args.min_joint_limit_margin_rad:.6f} rad"
            )
    if args.execute and not args.accept_arm_only_collision_model:
        raise ValueError("--execute requires --accept-arm-only-collision-model")
    if args.execute and "realtime" not in platform.release().lower():
        if not args.allow_generic_kernel:
            raise ValueError(
                "execution on this kernel requires --allow-generic-kernel; "
                f"current kernel is {platform.release()}"
            )


def make_goal(args: argparse.Namespace) -> MoveGroup.Goal:
    constraints = Constraints()
    constraints.name = "bounded_absolute_fr3_joint_goal"
    for name, position in zip(
        JOINT_NAMES, args.target_joint_positions_rad, strict=True
    ):
        joint = JointConstraint()
        joint.joint_name = name
        joint.position = position
        joint.tolerance_above = args.joint_tolerance_rad
        joint.tolerance_below = args.joint_tolerance_rad
        joint.weight = 1.0
        constraints.joint_constraints.append(joint)

    goal = MoveGroup.Goal()
    goal.request.group_name = args.group
    goal.request.num_planning_attempts = args.planning_attempts
    goal.request.allowed_planning_time = args.planning_time_s
    goal.request.max_velocity_scaling_factor = args.velocity_scale
    goal.request.max_acceleration_scaling_factor = args.acceleration_scale
    goal.request.start_state.is_diff = True
    goal.request.goal_constraints = [constraints]
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


def wait_for_joint_positions(
    node: Node, *, topic: str, timeout_s: float
) -> list[float] | None:
    latest: dict[str, float] = {}

    def callback(message: JointState) -> None:
        latest.update(zip(message.name, message.position, strict=False))

    subscription = node.create_subscription(JointState, topic, callback, 10)
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline and not all(
            name in latest for name in JOINT_NAMES
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
        if not all(name in latest for name in JOINT_NAMES):
            return None
        return [latest[name] for name in JOINT_NAMES]
    finally:
        node.destroy_subscription(subscription)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"argument error: {exc}", file=sys.stderr)
        return 2

    target = list(args.target_joint_positions_rad)
    rclpy.init()
    node = Node("demo2gripper_joint_goal")
    try:
        client = ActionClient(node, MoveGroup, args.action_name)
        if not client.wait_for_server(timeout_sec=args.server_timeout_s):
            node.get_logger().error(f"MoveGroup action unavailable: {args.action_name}")
            return 3

        node.get_logger().info(
            "requesting absolute FR3 joint goal; "
            f"mode={'execute' if args.execute else 'plan_only'}"
        )
        goal_future = client.send_goal_async(make_goal(args))
        rclpy.spin_until_future_complete(node, goal_future)
        handle = goal_future.result()
        if handle is None or not handle.accepted:
            node.get_logger().error("MoveGroup rejected the absolute joint goal")
            return 4

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            node.get_logger().error("MoveGroup returned no result")
            return 5

        result = wrapped.result
        trajectory = trajectory_dict(result.planned_trajectory)
        moveit_ok = (
            wrapped.status == GoalStatus.STATUS_SUCCEEDED
            and result.error_code.val == MoveItErrorCodes.SUCCESS
            and trajectory["point_count"] > 1
        )
        audit = audit_fr3_joint_trajectory(
            trajectory,
            min_joint_limit_margin_rad=args.min_joint_limit_margin_rad,
            max_joint_sweep_rad=args.max_joint_sweep_rad,
            max_total_joint_sweep_rad=args.max_total_joint_sweep_rad,
        )
        endpoint_error = None
        endpoint_ok = False
        if trajectory["points"] and set(trajectory["joint_names"]) == set(JOINT_NAMES):
            final = dict(
                zip(
                    trajectory["joint_names"],
                    trajectory["points"][-1]["positions"],
                    strict=True,
                )
            )
            endpoint_error = max(
                abs(final[name] - position)
                for name, position in zip(JOINT_NAMES, target, strict=True)
            )
            endpoint_ok = endpoint_error <= args.joint_tolerance_rad
        planning_succeeded = moveit_ok and audit["accepted"] and endpoint_ok

        execution_dispatched = False
        execution_status = None
        execution_error_code = None
        execution_error_message = None
        execution_failure_reason = None
        measured = None
        measured_error = None
        if args.execute:
            if not planning_succeeded:
                execution_failure_reason = "joint-goal plan failed its offline gates"
            else:
                execution_client = ActionClient(
                    node, ExecuteTrajectory, args.execute_action_name
                )
                if not execution_client.wait_for_server(
                    timeout_sec=args.server_timeout_s
                ):
                    execution_failure_reason = (
                        f"trajectory execution action unavailable: {args.execute_action_name}"
                    )
                else:
                    execution_goal = ExecuteTrajectory.Goal()
                    execution_goal.trajectory = result.planned_trajectory
                    execution_goal.controller_names = [args.controller_name]
                    future = execution_client.send_goal_async(execution_goal)
                    rclpy.spin_until_future_complete(node, future)
                    execution_handle = future.result()
                    if execution_handle is None or not execution_handle.accepted:
                        execution_failure_reason = "trajectory execution goal was rejected"
                    else:
                        execution_dispatched = True
                        future = execution_handle.get_result_async()
                        rclpy.spin_until_future_complete(node, future)
                        execution_wrapped = future.result()
                        if execution_wrapped is None:
                            execution_failure_reason = "trajectory execution returned no result"
                        else:
                            execution_status = execution_wrapped.status
                            execution_error_code = execution_wrapped.result.error_code.val
                            execution_error_message = execution_wrapped.result.error_code.message
                            measured = wait_for_joint_positions(
                                node,
                                topic=args.joint_state_topic,
                                timeout_s=args.server_timeout_s,
                            )
                            if measured is not None:
                                measured_error = max(
                                    abs(actual - requested)
                                    for actual, requested in zip(
                                        measured, target, strict=True
                                    )
                                )

        execution_succeeded = (
            planning_succeeded
            and execution_dispatched
            and execution_status == GoalStatus.STATUS_SUCCEEDED
            and execution_error_code == MoveItErrorCodes.SUCCESS
            and measured_error is not None
            and measured_error <= args.measured_tolerance_rad
        )
        summary = {
            "schema": "demo2gripper.moveit_joint_goal.v1",
            "mode": "execute" if args.execute else "plan_only",
            "group": args.group,
            "joint_names": list(JOINT_NAMES),
            "target_joint_positions_rad": target,
            "joint_tolerance_rad": args.joint_tolerance_rad,
            "measured_tolerance_rad": args.measured_tolerance_rad,
            "velocity_scale": args.velocity_scale,
            "acceleration_scale": args.acceleration_scale,
            "moveit_error_code": result.error_code.val,
            "moveit_error_message": result.error_code.message,
            "planning_action_status": wrapped.status,
            "planning_time_s": result.planning_time,
            "trajectory": trajectory,
            "trajectory_audit": audit,
            "planned_endpoint_max_error_rad": endpoint_error,
            "moveit_planning_succeeded": moveit_ok,
            "planning_succeeded": planning_succeeded,
            "execution_requested": args.execute,
            "execution_policy": "plan_then_audit_then_execute_exact_trajectory",
            "execution_dispatched": execution_dispatched,
            "execution_status": execution_status,
            "execution_error_code": execution_error_code,
            "execution_error_message": execution_error_message,
            "execution_failure_reason": execution_failure_reason,
            "measured_joint_positions_rad": measured,
            "measured_target_max_error_rad": measured_error,
            "execution_succeeded": execution_succeeded if args.execute else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        headline = {key: value for key, value in summary.items() if key != "trajectory"}
        print(json.dumps(headline, indent=2))
        print(f"trajectory points: {trajectory['point_count']}; saved: {args.output}")
        succeeded = execution_succeeded if args.execute else planning_succeeded
        if not succeeded:
            node.get_logger().error(
                execution_failure_reason
                or "; ".join(audit["violations"])
                or "MoveIt joint-goal plan failed"
            )
            return 9
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
