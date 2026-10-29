#!/usr/bin/env python3
"""Plan or execute reviewed eye-to-hand poses and capture strict RGB-D samples.

The input is the repository's absolute configured-EE pose-sequence JSON.  In
the default plan-only mode every target is sent to the existing MoveIt pose
tool for reachability and trajectory-audit evidence, but the arm is not moved
and no camera sample is taken.  With ``--execute`` each target is planned,
audited, and executed by that same tool; a sample is then captured from the
read-only Franka ROS state topic after a short settling interval.

Planning/audit rejection and camera-quality rejection skip only that target.
An execution that was dispatched but did not complete aborts the run because
the physical robot state can no longer be assumed safe for the next target.
This orchestrator never relaxes a motion or calibration threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from demo2gripper.autonomous_pick import (  # noqa: E402
    atomic_write_json,
    pose_result_succeeded,
    rotation_matrix_to_quaternion_xyzw,
)
from demo2gripper.cartesian_pose_sequence import (  # noqa: E402
    CartesianPose,
    load_pose_sequence,
)
from demo2gripper.franka import validate_rigid_transform  # noqa: E402
from demo2gripper.pose_motion import pose_delta  # noqa: E402

REPORT_SCHEMA = "demo2gripper.auto_handeye_calibration_run.v1"
SAMPLE_SCHEMA = "demo2gripper.eye_to_hand_sample.v1"
BASE_FRAME = "fr3_link0"
CALIBRATION_LINK8_ENDPOINT_POSITION_LIMIT_M = 0.010
CALIBRATION_CONFIGURED_EE_ENDPOINT_POSITION_LIMIT_M = 0.030
CALIBRATION_ENDPOINT_ROTATION_LIMIT_RAD = 0.10
EXECUTION_POLICY = "plan_then_audit_then_execute_exact_trajectory"
DEFAULT_MOVEIT_RUNNER = REPOSITORY_ROOT / "scripts/run_ros_python.sh"
DEFAULT_MOVEIT_SCRIPT = REPOSITORY_ROOT / "scripts/moveit_pose_goal.py"
DEFAULT_CAPTURE_SCRIPT = REPOSITORY_ROOT / "scripts/capture_handeye_sample.py"
DEFAULT_LINK8_CONFIGURED_EE = REPOSITORY_ROOT / "calibration/T_link8_canonical_tcp.npy"
DEFAULT_ROBOT_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--targets",
        type=Path,
        required=True,
        help="Reviewed configured-EE pose-sequence JSON in fr3_link0",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--serial", required=True, help="Exact RealSense serial number")
    parser.add_argument("--board-metadata", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--target-samples", type=int, default=24)
    parser.add_argument(
        "--t-link8-configured-ee",
        type=Path,
        default=DEFAULT_LINK8_CONFIGURED_EE,
    )
    parser.add_argument("--moveit-runner", type=Path, default=DEFAULT_MOVEIT_RUNNER)
    parser.add_argument("--moveit-script", type=Path, default=DEFAULT_MOVEIT_SCRIPT)
    parser.add_argument("--capture-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--capture-script", type=Path, default=DEFAULT_CAPTURE_SCRIPT)
    parser.add_argument("--ros-robot-state-topic", default=DEFAULT_ROBOT_STATE_TOPIC)
    parser.add_argument("--settle-s", type=float, default=0.75)

    parser.add_argument("--velocity-scale", type=float, default=0.02)
    parser.add_argument("--acceleration-scale", type=float, default=0.02)
    parser.add_argument("--max-translation-m", type=float, default=0.10)
    parser.add_argument("--max-rotation-rad", type=float, default=math.radians(30.0))
    parser.add_argument("--position-tolerance-m", type=float, default=0.001)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.02)
    parser.add_argument("--min-joint-limit-margin-rad", type=float, default=0.10)
    parser.add_argument("--max-joint-sweep-rad", type=float, default=0.70)
    parser.add_argument("--max-total-joint-sweep-rad", type=float, default=2.0)

    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--accept-arm-only-collision-model",
        action="store_true",
        help="Acknowledge that Hand-E and the calibration plate are absent from MoveIt",
    )
    return parser


def _existing_file(path: Path, *, name: str, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} not found: {resolved}")
    if executable and not resolved.stat().st_mode & 0o111:
        raise ValueError(f"{name} is not executable: {resolved}")
    return resolved


def _positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def validate_args(args: argparse.Namespace) -> dict[str, Any]:
    args.targets = _existing_file(args.targets, name="targets")
    args.board_metadata = _existing_file(args.board_metadata, name="board metadata")
    args.t_link8_configured_ee = _existing_file(
        args.t_link8_configured_ee, name="T_link8_configured_ee"
    )
    args.moveit_runner = _existing_file(args.moveit_runner, name="MoveIt runner", executable=True)
    args.moveit_script = _existing_file(args.moveit_script, name="MoveIt pose script")
    args.capture_python = _existing_file(
        args.capture_python, name="capture Python", executable=True
    )
    args.capture_script = _existing_file(args.capture_script, name="capture script")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.target_samples <= 0:
        raise ValueError("target_samples must be positive")
    if not args.session_id or not args.session_id.strip():
        raise ValueError("session_id must be non-empty")
    if not args.ros_robot_state_topic.startswith("/"):
        raise ValueError("ros_robot_state_topic must be absolute")
    if not math.isfinite(args.settle_s) or not 0.0 <= args.settle_s <= 10.0:
        raise ValueError("settle_s must lie in [0, 10]")

    for name in (
        "velocity_scale",
        "acceleration_scale",
        "max_translation_m",
        "max_rotation_rad",
        "position_tolerance_m",
        "orientation_tolerance_rad",
        "min_joint_limit_margin_rad",
        "max_joint_sweep_rad",
        "max_total_joint_sweep_rad",
    ):
        _positive_finite(getattr(args, name), name=name)
    if args.velocity_scale > 0.05 or args.acceleration_scale > 0.05:
        raise ValueError("velocity and acceleration scaling are capped at 0.05")
    if args.position_tolerance_m > 0.01:
        raise ValueError("position_tolerance_m is capped at 0.01 m")
    if args.orientation_tolerance_rad > 0.1:
        raise ValueError("orientation_tolerance_rad is capped at 0.1 rad")
    if args.min_joint_limit_margin_rad < 0.05:
        raise ValueError("min_joint_limit_margin_rad cannot weaken the 0.05 rad gate")
    if args.max_joint_sweep_rad > 2.0:
        raise ValueError("max_joint_sweep_rad cannot weaken the 2.0 rad gate")
    if args.max_total_joint_sweep_rad > 7.0:
        raise ValueError("max_total_joint_sweep_rad cannot weaken the 7.0 rad gate")
    if args.execute and not args.accept_arm_only_collision_model:
        raise ValueError("--execute requires --accept-arm-only-collision-model")

    poses = load_pose_sequence(args.targets, expected_base_frame=BASE_FRAME)
    if args.execute and len(poses) < args.target_samples:
        raise ValueError(
            f"targets contain {len(poses)} poses, fewer than target_samples {args.target_samples}"
        )
    for index, (first, second) in enumerate(zip(poses, poses[1:], strict=False)):
        translation_m, rotation_rad = pose_delta(
            first.position_m,
            first.orientation_xyzw,
            second.position_m,
            second.orientation_xyzw,
        )
        if translation_m > args.max_translation_m + 1e-12:
            raise ValueError(
                f"targets segment {index}->{index + 1} translation "
                f"{translation_m:.6f} m exceeds max_translation_m"
            )
        if rotation_rad > args.max_rotation_rad + 1e-12:
            raise ValueError(
                f"targets segment {index}->{index + 1} rotation "
                f"{math.degrees(rotation_rad):.3f} deg exceeds max_rotation_rad"
            )

    link8_configured_ee = validate_rigid_transform(
        np.load(args.t_link8_configured_ee, allow_pickle=False),
        name="T_link8_configured_ee",
    )
    return {"poses": poses, "T_link8_configured_ee": link8_configured_ee}


def build_move_command(
    args: argparse.Namespace,
    target: CartesianPose,
    *,
    output: Path,
) -> list[str]:
    transform = validate_rigid_transform(
        np.load(args.t_link8_configured_ee, allow_pickle=False),
        name="T_link8_configured_ee",
    )
    offset_orientation = rotation_matrix_to_quaternion_xyzw(transform[:3, :3])
    command = [
        str(args.moveit_runner),
        str(args.moveit_script),
        "--target-position-m",
        *(str(value) for value in target.position_m),
        "--target-orientation-xyzw",
        *(str(value) for value in target.orientation_xyzw),
        "--target-frame",
        "configured-ee",
        "--link8-to-configured-ee-position-m",
        *(str(float(value)) for value in transform[:3, 3]),
        "--link8-to-configured-ee-orientation-xyzw",
        *(str(value) for value in offset_orientation),
        "--base-frame",
        BASE_FRAME,
        "--velocity-scale",
        str(args.velocity_scale),
        "--acceleration-scale",
        str(args.acceleration_scale),
        "--max-translation-m",
        str(args.max_translation_m),
        "--max-rotation-rad",
        str(args.max_rotation_rad),
        "--position-tolerance-m",
        str(args.position_tolerance_m),
        "--orientation-tolerance-rad",
        str(args.orientation_tolerance_rad),
        "--min-joint-limit-margin-rad",
        str(args.min_joint_limit_margin_rad),
        "--max-joint-sweep-rad",
        str(args.max_joint_sweep_rad),
        "--max-total-joint-sweep-rad",
        str(args.max_total_joint_sweep_rad),
        "--output",
        str(output),
    ]
    if args.execute:
        command.extend(("--execute", "--accept-arm-only-collision-model"))
    return command


def build_capture_command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> list[str]:
    return [
        str(args.capture_python),
        str(args.capture_script),
        "--serial",
        args.serial,
        "--board-metadata",
        str(args.board_metadata),
        "--output-dir",
        str(output_dir),
        "--session-id",
        args.session_id,
        "--ros-robot-state-topic",
        args.ros_robot_state_topic,
    ]


def _run_process(
    command: Sequence[str], *, cwd: Path, log_prefix: Path
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(list(command), cwd=cwd, text=True, capture_output=True, check=False)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_prefix.with_suffix(".stdout.log").write_text(completed.stdout, encoding="utf-8")
    log_prefix.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
    return completed


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _finite_vector(payload: dict[str, Any], key: str, length: int) -> bool:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != length:
        return False
    try:
        return all(math.isfinite(float(component)) for component in value)
    except (TypeError, ValueError):
        return False


def calibration_execution_acceptable(payload: dict[str, Any]) -> bool:
    """Accept a completed calibration move using its measured pose.

    The shared pose tool intentionally keeps a 3 mm task-execution contract.
    Hand-eye calibration does not need to hit the requested pose exactly: the
    following capture records the robot's actual stationary flange transform.
    This fallback therefore keeps every planning, trajectory-audit, controller,
    and measurement gate.  The sanity bounds are 10 mm at link8, 0.10 rad,
    and 30 mm at the configured EE to account for its 190 mm lever arm.
    """

    trajectory = payload.get("trajectory")
    audit = payload.get("trajectory_audit")
    if not isinstance(trajectory, dict) or not isinstance(audit, dict):
        return False
    if (
        payload.get("schema") != "demo2gripper.moveit_pose_goal.v2"
        or payload.get("mode") != "execute"
        or payload.get("input_target_frame") != "configured-ee"
        or payload.get("moveit_error_code") != 1
        or payload.get("moveit_planning_succeeded") is not True
        or payload.get("planning_succeeded") is not True
        or trajectory.get("point_count", 0) <= 1
        or audit.get("accepted") is not True
        or audit.get("violations") not in (None, [])
        or payload.get("execution_policy") != EXECUTION_POLICY
        or payload.get("execution_dispatched") is not True
        or payload.get("execution_status") != 4
        or payload.get("execution_error_code") != 1
        or payload.get("execution_failure_reason") not in (None, "")
        or payload.get("measurement_error") not in (None, "")
    ):
        return False

    for key, length in (
        ("measured_end_position_base_m", 3),
        ("measured_end_orientation_xyzw", 4),
        ("measured_end_configured_ee_position_base_m", 3),
        ("measured_end_configured_ee_orientation_xyzw", 4),
    ):
        if not _finite_vector(payload, key, length):
            return False

    for key in (
        "measured_end_orientation_xyzw",
        "measured_end_configured_ee_orientation_xyzw",
    ):
        norm = math.sqrt(sum(float(value) ** 2 for value in payload[key]))
        if not 0.99 <= norm <= 1.01:
            return False

    try:
        link8_position_error = float(payload["measured_target_position_error_m"])
        link8_rotation_error = float(payload["measured_target_rotation_error_rad"])
        configured_ee_position_error = float(
            payload["measured_input_target_position_error_m"]
        )
        configured_ee_rotation_error = float(
            payload["measured_input_target_rotation_error_rad"]
        )
    except (KeyError, TypeError, ValueError):
        return False
    errors = (
        link8_position_error,
        link8_rotation_error,
        configured_ee_position_error,
        configured_ee_rotation_error,
    )
    return (
        all(math.isfinite(value) and value >= 0.0 for value in errors)
        and link8_position_error <= CALIBRATION_LINK8_ENDPOINT_POSITION_LIMIT_M
        and configured_ee_position_error
        <= CALIBRATION_CONFIGURED_EE_ENDPOINT_POSITION_LIMIT_M
        and link8_rotation_error <= CALIBRATION_ENDPOINT_ROTATION_LIMIT_RAD
        and configured_ee_rotation_error <= CALIBRATION_ENDPOINT_ROTATION_LIMIT_RAD
    )


def sample_result_succeeded(sample_dir: Path, *, session_id: str) -> tuple[bool, str]:
    manifest_path = sample_dir / "manifest.json"
    required = ("T_base_flange.npy", "T_camera_board.npy", "rgb.png", "overlay.png")
    if not manifest_path.is_file():
        return False, "capture produced no manifest"
    try:
        manifest = _read_json(manifest_path)
    except Exception as exc:
        return False, f"invalid sample manifest: {exc}"
    if manifest.get("schema") != SAMPLE_SCHEMA:
        return False, "sample manifest schema mismatch"
    if manifest.get("session_id") != session_id:
        return False, "sample session_id mismatch"
    if manifest.get("valid_for_handeye_calibration") is not True:
        return False, "sample is not valid_for_handeye_calibration"
    missing = [name for name in required if not (sample_dir / name).is_file()]
    if missing:
        return False, f"sample is missing required files: {missing}"
    return True, "accepted"


def _new_report(args: argparse.Namespace, pose_count: int) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "updated_utc": datetime.now(UTC).isoformat(),
        "mode": "execute" if args.execute else "plan_only",
        "status": "running",
        "targets": str(args.targets),
        "target_pose_count": pose_count,
        "target_samples": args.target_samples,
        "accepted_sample_count": 0,
        "accepted_samples": [],
        "attempts": [],
        "safety": {
            "motion_policy": EXECUTION_POLICY,
            "arm_only_collision_model_acknowledged": (
                args.accept_arm_only_collision_model if args.execute else False
            ),
            "capture_robot_access": "read-only ROS 2 state subscription",
            "calibration_endpoint_gate": {
                "link8_position_limit_m": CALIBRATION_LINK8_ENDPOINT_POSITION_LIMIT_M,
                "configured_ee_position_limit_m": (
                    CALIBRATION_CONFIGURED_EE_ENDPOINT_POSITION_LIMIT_M
                ),
                "rotation_limit_rad": CALIBRATION_ENDPOINT_ROTATION_LIMIT_RAD,
                "semantics": "capture the measured stationary pose after a successful action",
            },
            "thresholds_relaxed_after_failure": False,
        },
        "motion": {
            "velocity_scale": args.velocity_scale,
            "acceleration_scale": args.acceleration_scale,
            "max_translation_m": args.max_translation_m,
            "max_rotation_rad": args.max_rotation_rad,
            "position_tolerance_m": args.position_tolerance_m,
            "orientation_tolerance_rad": args.orientation_tolerance_rad,
            "min_joint_limit_margin_rad": args.min_joint_limit_margin_rad,
            "max_joint_sweep_rad": args.max_joint_sweep_rad,
            "max_total_joint_sweep_rad": args.max_total_joint_sweep_rad,
        },
    }


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    report["updated_utc"] = datetime.now(UTC).isoformat()
    atomic_write_json(output_dir / "run_report.json", report)


def run(args: argparse.Namespace) -> int:
    normalized = validate_args(args)
    poses: list[CartesianPose] = normalized["poses"]
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "motions").mkdir()
    (args.output_dir / "logs").mkdir()
    if args.execute:
        (args.output_dir / "candidates").mkdir()
    report = _new_report(args, len(poses))
    _write_report(args.output_dir, report)

    try:
        for target_index, target in enumerate(poses):
            if args.execute and report["accepted_sample_count"] >= args.target_samples:
                break
            stem = f"target_{target_index:03d}"
            motion_path = args.output_dir / "motions" / f"{stem}.json"
            motion_command = build_move_command(args, target, output=motion_path)
            attempt: dict[str, Any] = {
                "target_index": target_index,
                "target": target.as_dict(),
                "motion_artifact": str(motion_path),
                "motion_command": motion_command,
                "motion_status": "running",
                "capture_status": "not_requested",
            }
            report["attempts"].append(attempt)
            _write_report(args.output_dir, report)

            motion_process = _run_process(
                motion_command,
                cwd=REPOSITORY_ROOT,
                log_prefix=args.output_dir / "logs" / f"{stem}.motion",
            )
            attempt["motion_returncode"] = motion_process.returncode
            motion_payload: dict[str, Any] | None = None
            if motion_path.is_file():
                try:
                    motion_payload = _read_json(motion_path)
                except Exception as exc:
                    attempt["motion_error"] = f"invalid motion artifact: {exc}"
            else:
                attempt["motion_error"] = "motion tool produced no JSON artifact"

            strict_motion_ok = (
                motion_process.returncode == 0
                and motion_payload is not None
                and pose_result_succeeded(motion_payload, executed=args.execute)
            )
            calibration_motion_ok = (
                args.execute
                and motion_process.returncode == 9
                and motion_payload is not None
                and calibration_execution_acceptable(motion_payload)
            )
            motion_ok = strict_motion_ok or calibration_motion_ok
            if not motion_ok:
                dispatch_known = motion_payload is not None and isinstance(
                    motion_payload.get("execution_dispatched"), bool
                )
                dispatched = bool(
                    dispatch_known and motion_payload.get("execution_dispatched") is True
                )
                attempt["motion_status"] = "failed_after_dispatch" if dispatched else "skipped"
                attempt.setdefault(
                    "motion_error",
                    "MoveIt plan/audit/execute result did not pass the strict contract",
                )
                _write_report(args.output_dir, report)
                if args.execute and (dispatched or not dispatch_known):
                    attempt["motion_status"] = (
                        "failed_after_dispatch"
                        if dispatched
                        else "failed_with_unknown_dispatch_state"
                    )
                    report["status"] = "aborted_after_motion_failure"
                    _write_report(args.output_dir, report)
                    return 3
                continue

            attempt["motion_status"] = "executed" if args.execute else "planned"
            if calibration_motion_ok:
                attempt["motion_acceptance"] = "actual_pose_for_calibration"
                for key in (
                    "measured_target_position_error_m",
                    "measured_target_rotation_error_rad",
                    "measured_input_target_position_error_m",
                    "measured_input_target_rotation_error_rad",
                ):
                    attempt[key] = motion_payload[key]
            if not args.execute:
                _write_report(args.output_dir, report)
                continue

            if args.settle_s:
                time.sleep(args.settle_s)
            # Keep the target index so a failed/partial capture can never block a
            # later target by occupying the next contiguous sample directory.
            # The solver naturally sorts sparse candidate directory names.
            sample_index = target_index
            sample_dir = args.output_dir / "candidates" / f"{sample_index:03d}"
            capture_command = build_capture_command(args, output_dir=sample_dir)
            attempt["capture_command"] = capture_command
            attempt["capture_output"] = str(sample_dir)
            attempt["capture_status"] = "running"
            _write_report(args.output_dir, report)
            capture_process = _run_process(
                capture_command,
                cwd=REPOSITORY_ROOT,
                log_prefix=args.output_dir / "logs" / f"{stem}.capture",
            )
            attempt["capture_returncode"] = capture_process.returncode
            sample_ok, sample_reason = sample_result_succeeded(
                sample_dir, session_id=args.session_id
            )
            if capture_process.returncode != 0 or not sample_ok:
                attempt["capture_status"] = "skipped"
                attempt["capture_error"] = sample_reason
                _write_report(args.output_dir, report)
                continue

            attempt["capture_status"] = "accepted"
            report["accepted_samples"].append(
                {
                    "sample_index": sample_index,
                    "target_index": target_index,
                    "path": str(sample_dir),
                }
            )
            report["accepted_sample_count"] = len(report["accepted_samples"])
            _write_report(args.output_dir, report)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        _write_report(args.output_dir, report)
        return 130
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_report(args.output_dir, report)
        raise

    if args.execute:
        complete = report["accepted_sample_count"] >= args.target_samples
        report["status"] = "complete" if complete else "insufficient_samples"
        _write_report(args.output_dir, report)
        return 0 if complete else 1

    planned = sum(attempt["motion_status"] == "planned" for attempt in report["attempts"])
    report["planned_target_count"] = planned
    report["status"] = "review_ready" if planned == len(poses) else "review_incomplete"
    _write_report(args.output_dir, report)
    return 0 if planned == len(poses) else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
