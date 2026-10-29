#!/usr/bin/python3
"""Stream read-only Franka ROS state as libfranka-compatible JSON lines.

This helper only creates a subscription.  It exists so the RealSense/OpenCV
calibration process can keep using its Python environment while ROS 2 runs in
the system Python environment.  Each output object contains the subset consumed
by ``capture_handeye_sample.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Any

DEFAULT_TOPIC = "/franka_robot_state_broadcaster/robot_state"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.topic.startswith("/"):
        raise ValueError("--topic must be an absolute ROS topic")
    if args.count < 2:
        raise ValueError("--count must be at least 2")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0.0:
        raise ValueError("--timeout-s must be finite and positive")


def _quaternion_rotation_xyzw(values: tuple[float, float, float, float]) -> list[list[float]]:
    x, y, z, w = (float(value) for value in values)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("ROS pose has an invalid quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def pose_to_column_major(pose: Any) -> list[float]:
    """Convert a geometry_msgs/Pose-like value to libfranka's column-major matrix."""
    rotation = _quaternion_rotation_xyzw(
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
    )
    translation = [float(pose.position.x), float(pose.position.y), float(pose.position.z)]
    if not all(math.isfinite(value) for value in translation):
        raise ValueError("ROS pose has a non-finite translation")
    matrix = [
        [rotation[0][0], rotation[0][1], rotation[0][2], translation[0]],
        [rotation[1][0], rotation[1][1], rotation[1][2], translation[1]],
        [rotation[2][0], rotation[2][1], rotation[2][2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return [matrix[row][column] for column in range(4) for row in range(4)]


def active_error_names(errors: Any) -> list[str]:
    fields = errors.get_fields_and_field_types()
    return sorted(name for name in fields if bool(getattr(errors, name)))


def franka_state_to_dict(message: Any) -> dict[str, Any]:
    """Return the exact semantic fields used by the existing stationarity checks."""
    q = [float(value) for value in message.measured_joint_state.position]
    dq = [float(value) for value in message.measured_joint_state.velocity]
    if len(q) != 7 or len(dq) != 7 or not all(math.isfinite(value) for value in q + dq):
        raise ValueError("ROS Franka state must contain seven finite measured q/dq values")
    robot_time_ms = float(message.time) * 1e3
    if not math.isfinite(robot_time_ms):
        raise ValueError("ROS Franka state has an invalid robot time")
    return {
        "O_T_EE": pose_to_column_major(message.o_t_ee.pose),
        "F_T_EE": pose_to_column_major(message.f_t_ee.pose),
        "q": q,
        "dq": dq,
        "time": robot_time_ms,
        "robot_mode": int(message.robot_mode),
        "current_errors": active_error_names(message.current_errors),
    }


def run(args: argparse.Namespace) -> int:
    # Import only here: unit tests exercise conversion under the non-ROS camera Python.
    import rclpy
    from franka_msgs.msg import FrankaRobotState
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    rclpy.init(args=None)
    node = rclpy.create_node("handeye_read_only_state_reader")
    received = 0
    failure: Exception | None = None

    def callback(message: FrankaRobotState) -> None:
        nonlocal received, failure
        try:
            print(json.dumps(franka_state_to_dict(message), separators=(",", ":")), flush=True)
            received += 1
        except Exception as exc:  # Stop rather than emit a partially trusted state.
            failure = exc

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(10, args.count),
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    subscription = node.create_subscription(FrankaRobotState, args.topic, callback, qos)
    _ = subscription
    deadline = time.monotonic() + args.timeout_s
    try:
        while received < args.count and failure is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if failure is not None:
        raise RuntimeError(f"invalid Franka ROS state: {failure}") from failure
    if received < args.count:
        raise RuntimeError(
            f"timed out after {args.timeout_s:.3f}s waiting for {args.count} states on "
            f"{args.topic}; received {received}"
        )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
