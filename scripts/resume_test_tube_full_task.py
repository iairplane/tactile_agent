#!/usr/bin/env python3
"""Reset a supported, unchanged grasp and run the complete resident task.

Requires fresh scene evidence that the tube is still supported in its cup.
Default mode writes a plan only and opens no hardware. Source ROS Jazzy and the
Franka overlay before execution; the caller's Python/environment is inherited.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from demo2gripper.autonomous_pick import atomic_write_json  # noqa: E402

PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AQ00O97B-if00-port0"


def pose_errors(position, quaternion, expected_position, expected_quaternion):
    position, expected_position = np.asarray(position), np.asarray(expected_position)
    if (
        position.shape != (3,)
        or expected_position.shape != (3,)
        or not np.isfinite(position).all()
        or not np.isfinite(expected_position).all()
    ):
        raise ValueError("positions must contain three finite values")
    rotation = Rotation.from_quat(quaternion) * Rotation.from_quat(expected_quaternion).inv()
    return float(np.linalg.norm(position - expected_position)), float(
        rotation.magnitude() * 180 / np.pi
    )


def recovery_waypoint(position, quaternion):
    pose_errors(position, quaternion, position, quaternion)
    tcp = np.asarray(position, dtype=float) + Rotation.from_quat(quaternion).apply([0, 0, 0.19])
    return {
        "position_m": (tcp + [0, 0, 0.06]).tolist(),
        "orientation_xyzw": Rotation.from_quat(quaternion).as_quat().tolist(),
    }


def require_motion_success(outcome, target):
    if outcome.get("execution_succeeded") is not True:
        raise RuntimeError("recovery trajectory did not succeed")
    measured = outcome.get("measured_end_configured_ee")
    if not isinstance(measured, dict):
        raise RuntimeError("recovery trajectory lacks measured TCP endpoint")
    position = np.asarray(measured.get("position_m"), dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise RuntimeError("recovery trajectory has invalid measured TCP endpoint")
    if np.linalg.norm(position - np.asarray(target["position_m"])) > 0.008:
        raise RuntimeError("measured recovery TCP missed the target by more than 8 mm")
    _, angle = pose_errors(
        position, measured["orientation_xyzw"], target["position_m"], target["orientation_xyzw"]
    )
    if angle > 2:
        raise RuntimeError("recovery changed wrist orientation by more than 2 degrees")
    return position


def full_task_complete(stack_report):
    task = stack_report.get("task_report")
    return bool(
        stack_report.get("status") == "complete"
        and isinstance(task, dict)
        and task.get("status") == "complete"
        and task.get("final_state") == "COMPLETE"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--close-run", required=True, type=Path)
    parser.add_argument("--stack", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-arm-only-collision-model", action="store_true")
    parser.add_argument("--accept-provisional-camera-extrinsic", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and not args.accept_arm_only_collision_model:
        parser.error("execution requires --accept-arm-only-collision-model")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.close_run, args.stack = args.close_run.resolve(), args.stack.resolve()
    args.output_dir = args.output_dir.resolve()
    close = json.loads((args.close_run / "run_report.json").read_text())
    reference = json.loads((args.close_run / "measured_lift_target.json").read_text())
    if close.get("status") != "grasp_ready":
        raise ValueError("close-run must record grasp_ready")
    if reference.get("frame") != "fr3_link0" or reference.get("link") != "fr3_link8":
        raise ValueError("saved pre-lift measurement must be fr3_link0 -> fr3_link8")
    planned = recovery_waypoint(reference["position_m"], reference["quaternion_xyzw"])
    if not args.stack.is_file():
        raise ValueError(f"stack manifest is missing: {args.stack}")
    json.loads(args.stack.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "demo2gripper.test_tube_supported_reset_resume.v1",
        "status": "planned",
        "close_run": str(args.close_run),
        "stack": str(args.stack),
        "full_task_complete": False,
        "hardware_commands_dispatched": False,
        "stages": [
            "verify_unchanged_supported_grasp",
            "open_49_5_mm",
            "vertical_reset_60_mm",
            "run_full_resident_task",
        ],
        "planned_recovery_tcp": planned,
        "receipts": {},
    }
    report_path = args.output_dir / "run_report.json"
    atomic_write_json(report_path, report)
    if not args.execute:
        print(json.dumps(report))
        return 0

    import rclpy
    from action_msgs.srv import CancelGoal
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformException, TransformListener

    rclpy.init()
    node = rclpy.create_node("test_tube_supported_reset_resume")
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    cancel = node.create_client(CancelGoal, "/execute_trajectory/_action/cancel_goal")
    child = None

    def save(stage):
        report["active_stage"] = stage
        atomic_write_json(report_path, report)

    def current_pose():
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                tf = buffer.lookup_transform("fr3_link0", "fr3_link8", Time())
            except TransformException:
                continue
            stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            age = node.get_clock().now().nanoseconds * 1e-9 - stamp
            if not -0.05 <= age <= 0.5:
                continue
            p, q = tf.transform.translation, tf.transform.rotation
            return [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]
        raise RuntimeError("fresh current link8 transform unavailable")

    def require_unchanged():
        p, q = current_pose()
        distance, angle = pose_errors(p, q, reference["position_m"], reference["quaternion_xyzw"])
        if distance > 0.0025 or angle > 2:
            raise RuntimeError(
                "supported grasp moved since recorded close: "
                f"{distance * 1000:.2f} mm, {angle:.2f} deg"
            )
        report["receipts"]["pose_check"] = {
            "position_m": p,
            "quaternion_xyzw": q,
            "position_error_m": distance,
            "rotation_error_deg": angle,
        }
        return p, q

    def run_child(stage, argv, timeout):
        nonlocal child
        save(stage)
        log_path = args.output_dir / f"{stage}.log"
        report["receipts"][stage] = {"argv": argv, "log": str(log_path), "status": "running"}
        atomic_write_json(report_path, report)
        with log_path.open("w") as log:
            child = subprocess.Popen(
                argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            deadline = time.monotonic() + timeout
            while child.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{stage} timed out")
                rclpy.spin_once(node, timeout_sec=0.05)
            code = child.returncode
        report["receipts"][stage].update(status="finished", returncode=code)
        atomic_write_json(report_path, report)
        if code:
            raise RuntimeError(f"{stage} failed ({code}); see {log_path}")
        return log_path

    def gripper_status(stage):
        path = run_child(
            stage,
            [sys.executable, str(ROOT / "scripts/hande_light_cli.py"), "status", "--port", PORT],
            8,
        )
        state = json.loads(path.read_text())["state"]
        if int(state["fault_code"]) == 9:
            path = run_child(
                stage + "_retry",
                [
                    sys.executable,
                    str(ROOT / "scripts/hande_light_cli.py"),
                    "status",
                    "--port",
                    PORT,
                ],
                8,
            )
            state = json.loads(path.read_text())["state"]
        if state.get("activated") is not True or int(state["fault_code"]) != 0:
            raise RuntimeError("Hand-E is not activated and fault-free")
        return state

    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("recovery interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        report["status"] = "running"
        save("verify_unchanged_supported_grasp")
        require_unchanged()
        initial = gripper_status("initial_gripper_status")
        if (
            float(initial["opening_width_mm"]) < 48
            and abs(
                float(initial["opening_width_mm"]) - float(close["measured_stop_opening_mm"])
            ) > 1.5
        ):
            raise RuntimeError("held opening changed since recorded close")
        if not cancel.wait_for_service(timeout_sec=3):
            raise RuntimeError("trajectory cancellation service unavailable")
        require_unchanged()
        report["hardware_commands_dispatched"] = True
        run_child(
            "open_gripper",
            [
                sys.executable,
                str(ROOT / "scripts/hande_light_cli.py"),
                "move",
                "--port",
                PORT,
                "--opening-mm",
                "49.5",
                "--speed",
                "45",
                "--force",
                "8",
                "--timeout-s",
                "8",
            ],
            15,
        )
        opened = gripper_status("verify_open")
        if float(opened["opening_width_mm"]) < 48 or int(opened["object_status"]["code"]) != 3:
            raise RuntimeError("Hand-E did not reach stationary opening >=48 mm")
        report["receipts"]["open_feedback"] = opened
        p, q = require_unchanged()
        target = recovery_waypoint(p, q)
        midpoint = dict(target)
        midpoint["position_m"] = list(target["position_m"])
        midpoint["position_m"][2] -= .03
        target_path = args.output_dir / "recovery_target.json"
        atomic_write_json(
            target_path,
            {
                "schema": "demo2gripper.configured_ee_pose_sequence.v1",
                "base_frame": "fr3_link0",
                "waypoints": [midpoint, target],
            },
        )
        motion_path = args.output_dir / "recovery_motion.json"
        run_child(
            "vertical_reset",
            [
                sys.executable,
                str(ROOT / "scripts/moveit_cartesian_pose_sequence.py"),
                "--input",
                str(target_path),
                "--output",
                str(motion_path),
                "--link8-to-configured-ee-position-m",
                "0",
                "0",
                ".19",
                "--link8-to-configured-ee-orientation-xyzw",
                "0",
                "0",
                "0",
                "1",
                "--velocity-scale",
                ".03",
                "--acceleration-scale",
                ".03",
                "--max-cartesian-speed-m-s",
                ".015",
                "--execute",
                "--accept-arm-only-collision-model",
            ],
            90,
        )
        outcome = json.loads(motion_path.read_text())
        actual = require_motion_success(outcome, target)
        report["receipts"]["measured_recovery_tcp"] = actual.tolist()
        argv = [
            sys.executable,
            str(ROOT / "scripts/run_test_tube_live_stack.py"),
            "--stack",
            str(args.stack),
            "--output-dir",
            str(args.output_dir / "full_task"),
            "--execute",
            "--accept-arm-only-collision-model",
        ]
        if args.accept_provisional_camera_extrinsic:
            argv.append("--accept-provisional-camera-extrinsic")
        run_child("full_task", argv, 1800)
        result = json.loads((args.output_dir / "full_task/stack_report.json").read_text())
        report["stack_report"] = result
        if not full_task_complete(result):
            raise RuntimeError("resident task did not report COMPLETE")
        report.update(status="complete", full_task_complete=True)
        return 0
    except BaseException as error:
        report.update(status="held", error=f"{type(error).__name__}: {error}")
        # Stop the producer before cancelling server-side goals: cancellation
        # during planning must not leave a child free to submit a later goal.
        if child is not None:
            # The stack launcher can have a task child: stop the entire stage
            # process group, including a descendant left by a failed launcher.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=3)
        if report["hardware_commands_dispatched"]:
            try:
                future = cancel.call_async(CancelGoal.Request())
                rclpy.spin_until_future_complete(node, future, timeout_sec=3)
                response = future.result() if future.done() else None
                report["cancel_receipt"] = {
                    "response_received": response is not None,
                    "return_code": None if response is None else response.return_code,
                    "goals_canceling": None if response is None else len(response.goals_canceling),
                }
            except Exception as stop_error:
                report["cancel_error"] = str(stop_error)
            try:
                from demo2gripper.hande_modbus_cli import SerialModbusRTU

                transport = SerialModbusRTU(PORT, 0.12)
                try:
                    transport.stop_motion()
                    report["gripper_stop_sent"] = True
                finally:
                    transport.close()
            except Exception as stop_error:
                report["gripper_stop_error"] = str(stop_error)
        return 2
    finally:
        atomic_write_json(report_path, report)
        signal.signal(signal.SIGTERM, previous_term)
        listener.unregister()
        node.destroy_node()
        rclpy.shutdown()
        print(json.dumps(report))


if __name__ == "__main__":
    raise SystemExit(main())
