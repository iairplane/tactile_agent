#!/usr/bin/env python3
"""Return the interrupted test-tube grasp to its supported start for a retry.

The current recovery case is deliberately narrow: a reviewed fast-probe run
defines the original contact TCP, while the live robot may be 25--60 mm above
that point with a partially levelled wrist.  Execution first restores the
contact orientation at the current safe height, descends vertically to the
recorded contact point, opens the Hand-E, and retreats in base +Z.

Default mode is file-only.  Physical commands require both ``--execute`` and
``--accept-arm-only-collision-model``.  This script never enables FCI, unlocks
joints, or activates the gripper.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import signal
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from demo2gripper.autonomous_pick import atomic_write_json  # noqa: E402


def _matrix(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation is not right handed")
    return result.copy()


def _pose_error(measured: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    measured = _matrix(measured, "measured pose")
    target = _matrix(target, "target pose")
    distance = float(np.linalg.norm(measured[:3, 3] - target[:3, 3]))
    rotation = Rotation.from_matrix(measured[:3, :3] @ target[:3, :3].T)
    return distance, math.degrees(float(rotation.magnitude()))


def load_contact_reference(close_run: Path) -> tuple[np.ndarray, float, dict[str, Any]]:
    path = close_run.resolve() / "run_report.json"
    if not path.is_file():
        raise ValueError(f"fast-probe report is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "demo2gripper.test_tube_fast_probe.v1":
        raise ValueError("close-run is not a test-tube fast-probe report")
    if report.get("status") not in {"bilateral_contact", "grasp_ready"}:
        raise ValueError("close-run did not finish with a retained tactile grasp")
    contact = _matrix(report.get("T_base_tcp_final"), "contact T_base_tcp")
    final_gripper = report.get("final_gripper")
    if not isinstance(final_gripper, Mapping):
        raise ValueError("close-run lacks final gripper feedback")
    width = float(final_gripper.get("width_m", math.nan))
    if not math.isfinite(width) or not 0.005 <= width <= 0.045:
        raise ValueError("close-run retained width is invalid")
    return contact, width, report


def build_reset_targets(
    contact: np.ndarray,
    current: np.ndarray,
    *,
    retreat_mm: float = 80.0,
) -> dict[str, np.ndarray]:
    contact = _matrix(contact, "contact pose")
    current = _matrix(current, "current pose")
    if not math.isfinite(retreat_mm) or not 40.0 <= retreat_mm <= 120.0:
        raise ValueError("retreat_mm must lie in [40, 120]")
    restored_high = contact.copy()
    restored_high[2, 3] = current[2, 3]
    retreat = contact.copy()
    retreat[2, 3] += retreat_mm / 1000.0
    return {
        "restore_contact_orientation_at_high_pose": restored_high,
        "lower_to_supported_contact": contact,
        "retreat_after_release": retreat,
    }


def validate_live_start(
    contact: np.ndarray,
    current: np.ndarray,
    *,
    min_lift_mm: float = 25.0,
    max_lift_mm: float = 60.0,
    max_xy_error_mm: float = 8.0,
    max_rotation_error_deg: float = 45.0,
) -> dict[str, float]:
    contact = _matrix(contact, "contact pose")
    current = _matrix(current, "current pose")
    values = (min_lift_mm, max_lift_mm, max_xy_error_mm, max_rotation_error_deg)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("live-start bounds must be positive and finite")
    if min_lift_mm >= max_lift_mm:
        raise ValueError("min_lift_mm must be below max_lift_mm")
    delta = current[:3, 3] - contact[:3, 3]
    lift_mm = float(delta[2] * 1000.0)
    xy_error_mm = float(np.linalg.norm(delta[:2]) * 1000.0)
    _, rotation_error_deg = _pose_error(current, contact)
    if not min_lift_mm <= lift_mm <= max_lift_mm:
        raise RuntimeError(
            f"current TCP lift {lift_mm:.2f} mm is outside the reviewed reset envelope"
        )
    if xy_error_mm > max_xy_error_mm:
        raise RuntimeError(
            f"current TCP XY drift {xy_error_mm:.2f} mm exceeds the reset envelope"
        )
    if rotation_error_deg > max_rotation_error_deg:
        raise RuntimeError(
            "current wrist rotation differs too far from the recorded contact pose: "
            f"{rotation_error_deg:.2f} deg"
        )
    return {
        "lift_mm": lift_mm,
        "xy_error_mm": xy_error_mm,
        "rotation_error_deg": rotation_error_deg,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--close-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retreat-mm", type=float, default=80.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-arm-only-collision-model", action="store_true")
    args = parser.parse_args(argv)
    build_reset_targets(np.eye(4), np.eye(4), retreat_mm=args.retreat_mm)
    if args.execute and not args.accept_arm_only_collision_model:
        parser.error("--execute requires --accept-arm-only-collision-model")
    return args


def run(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    close_run = args.close_run.resolve()
    output = args.output_dir.resolve()
    if not config_path.is_file():
        raise ValueError(f"task config is missing: {config_path}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite a non-empty output directory")
    output.mkdir(parents=True, exist_ok=True)

    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    live = config.get("live_adapter")
    if not isinstance(live, Mapping):
        raise ValueError("task config lacks live_adapter")
    robot = copy.deepcopy(live.get("robot"))
    tactile = live.get("tactile")
    if not isinstance(robot, dict) or not isinstance(tactile, Mapping):
        raise ValueError("task config lacks robot/tactile adapters")
    for key, ceiling in (("max_linear_m_s", 0.010), ("max_angular_rad_s", 0.12)):
        value = float(robot["servo"][key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"invalid robot.servo.{key}")
        robot["servo"][key] = min(value, ceiling)

    contact, expected_width_m, source_report = load_contact_reference(close_run)
    report: dict[str, Any] = {
        "schema": "demo2gripper.test_tube_retry_reset.v1",
        "status": "prepared",
        "mode": "execute" if args.execute else "dry_run",
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "close_run": str(close_run),
        "close_run_status": source_report["status"],
        "T_base_tcp_contact": contact.tolist(),
        "expected_held_width_m": expected_width_m,
        "retreat_mm": float(args.retreat_mm),
        "ros_domain_id": 0,
        "servo_limits": robot["servo"],
        "arm_only_collision_model_accepted": bool(
            args.accept_arm_only_collision_model
        ),
        "hardware_commands_dispatched": False,
        "stages": [],
    }
    report_path = output / "run_report.json"
    atomic_write_json(report_path, report)
    if not args.execute:
        report.update(
            status="dry_run_complete",
            planned_stages=[
                "restore_contact_orientation_at_high_pose",
                "lower_to_supported_contact",
                "open_hande_49_5_mm",
                f"retreat_base_z_{float(args.retreat_mm):g}_mm",
            ],
        )
        atomic_write_json(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path)}))
        return 0

    # The workstation's interactive shell can use another ROS domain.  This
    # process is intentionally pinned before importing/constructing rclpy IO.
    os.environ["ROS_DOMAIN_ID"] = "0"
    import run_test_tube_continuous_close as contact_io

    from demo2gripper.test_tube_live_io import (
        HandEModbusRuntime,
        RosMoveItServoRuntime,
        _load_artifact_transform,
    )

    arm = gripper = transport = None
    original_handlers: dict[int, Any] = {}

    def save() -> None:
        atomic_write_json(report_path, report)

    def persist_stage(name: str, **payload: Any) -> None:
        report["stages"].append(
            {"name": name, "at_monotonic_s": time.monotonic(), **payload}
        )
        save()

    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"received signal {signum}")

    def healthy_arm():
        state = arm.state()
        if state.fault:
            raise RuntimeError(f"arm fault: {state.fault}")
        return state

    def healthy_gripper(*, require_held_width: bool) -> Any:
        feedback = gripper.feedback()
        if not feedback.fault_free:
            raise RuntimeError("Hand-E fault during reset")
        if feedback.moving and require_held_width:
            raise RuntimeError("Hand-E unexpectedly moving during held reset")
        if require_held_width and abs(feedback.width_m - expected_width_m) > 0.003:
            raise RuntimeError("Hand-E opening drifted from the recorded held width")
        return feedback

    def servo_stage(name: str, target: np.ndarray, *, timeout_s: float) -> np.ndarray:
        target = _matrix(target, f"{name} target")
        receipt = dict(
            arm.servo_target(
                target=target,
                action_id=name,
                position_tolerance_m=0.0015,
            )
        )
        report["hardware_commands_dispatched"] = True
        save()
        observations: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = healthy_arm()
            feedback = healthy_gripper(require_held_width=name != "retreat_after_release")
            distance_m, rotation_error_deg = _pose_error(state.T_base_tcp, target)
            observations.append(
                {
                    "at_monotonic_s": time.monotonic(),
                    "distance_m": distance_m,
                    "rotation_error_deg": rotation_error_deg,
                    "motion_complete": state.motion_complete,
                    "gripper": asdict(feedback),
                }
            )
            if distance_m <= 0.002 and rotation_error_deg <= 2.0 and state.motion_complete:
                arm.stop(reason=f"{name}: target reached")
                persist_stage(
                    name,
                    status="complete",
                    target=target.tolist(),
                    receipt=receipt,
                    T_base_tcp_final=state.T_base_tcp.tolist(),
                    observations=observations,
                )
                return state.T_base_tcp.copy()
            time.sleep(0.025)
        arm.stop(reason=f"{name}: timeout")
        raise TimeoutError(f"{name} did not reach target")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            original_handlers[signum] = signal.signal(signum, interrupted)
        transport = contact_io.SerialModbusRTU(
            str(tactile["hande_port"]), float(tactile.get("modbus_timeout_s", 0.12))
        )
        initial_raw = contact_io._read_healthy_initial_state(transport)
        gripper = HandEModbusRuntime(transport)
        initial_gripper = healthy_gripper(require_held_width=True)
        tcp = _load_artifact_transform(robot, "tcp")
        arm = RosMoveItServoRuntime(
            config=robot,
            T_link8_tcp=tcp,
            output_dir=output,
        )
        initial_arm = healthy_arm()
        start_evidence = validate_live_start(contact, initial_arm.T_base_tcp)
        targets = build_reset_targets(
            contact,
            initial_arm.T_base_tcp,
            retreat_mm=float(args.retreat_mm),
        )
        report.update(
            status="executing",
            initial_hande_state=initial_raw,
            initial_gripper=asdict(initial_gripper),
            initial_T_base_tcp=initial_arm.T_base_tcp.tolist(),
            live_start_evidence=start_evidence,
            targets={key: value.tolist() for key, value in targets.items()},
        )
        save()

        servo_stage(
            "restore_contact_orientation_at_high_pose",
            targets["restore_contact_orientation_at_high_pose"],
            timeout_s=20.0,
        )
        servo_stage(
            "lower_to_supported_contact",
            targets["lower_to_supported_contact"],
            timeout_s=15.0,
        )

        before_open = healthy_arm()
        distance_m, rotation_error_deg = _pose_error(before_open.T_base_tcp, contact)
        if distance_m > 0.002 or rotation_error_deg > 2.0:
            raise RuntimeError("TCP left the supported contact target before release")
        open_receipt = dict(
            gripper.move_width(
                width_m=0.0495,
                speed=45,
                force=8,
                action_id="open_for_retry",
            )
        )
        report["hardware_commands_dispatched"] = True
        save()
        deadline = time.monotonic() + 8.0
        open_feedback = None
        while time.monotonic() < deadline:
            healthy_arm()
            open_feedback = healthy_gripper(require_held_width=False)
            if open_feedback.open_confirmed and not open_feedback.moving:
                break
            time.sleep(0.025)
        else:
            raise TimeoutError("Hand-E opening to 49.5 mm was not confirmed")
        persist_stage(
            "open_hande_49_5_mm",
            status="complete",
            receipt=open_receipt,
            gripper=asdict(open_feedback),
        )

        servo_stage(
            "retreat_after_release",
            targets["retreat_after_release"],
            timeout_s=15.0,
        )
        report.update(status="complete", reset_ready_for_new_observation=True)
        save()
        return 0
    except BaseException as error:
        report.update(
            status="held",
            error={"type": type(error).__name__, "message": str(error)},
        )
        if arm is not None or gripper is not None:
            report["hardware_commands_dispatched"] = True
        if arm is not None:
            try:
                report["arm_stop"] = dict(arm.stop(reason="retry reset failed"))
            except BaseException as stop_error:
                report["arm_stop_error"] = str(stop_error)
        if gripper is not None:
            try:
                report["gripper_stop"] = dict(gripper.stop(reason="retry reset failed"))
            except BaseException as stop_error:
                report["gripper_stop_error"] = str(stop_error)
        save()
        return 2
    finally:
        for resource in (arm, gripper):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as close_error:
                report.setdefault("cleanup_errors", []).append(str(close_error))
        if transport is not None and gripper is None:
            transport.close()
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
        save()
        print(json.dumps({"status": report["status"], "report": str(report_path)}))


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
