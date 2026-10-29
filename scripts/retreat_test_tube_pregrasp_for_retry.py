#!/usr/bin/env python3
"""Clear the D435 view after an interrupted test-tube top-pregrasp.

This recovery is intentionally narrow.  It reads the exact top-pregrasp target
from a resident test-tube run, requires the live TCP to remain within 8 mm and
2 degrees of that target, requires a stationary open and empty Hand-E, and then
moves only 60 mm along base +Z without changing the measured wrist orientation.

Default mode is file-only.  Physical commands require both ``--execute`` and
``--accept-arm-only-collision-model``.  The script never enables FCI, unlocks
joints, activates the gripper, or commands a close.
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


def _task_dir(source_run: Path) -> Path:
    root = source_run.resolve()
    if (root / "task" / "run_report.json").is_file():
        return root / "task"
    if (root / "run_report.json").is_file():
        return root
    raise ValueError(f"resident test-tube report is missing below {root}")


def load_pregrasp_reference(
    source_run: Path,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    task = _task_dir(source_run)
    report = json.loads((task / "run_report.json").read_text(encoding="utf-8"))
    if report.get("schema") != "demo2gripper.test_tube_closed_loop_report.v1":
        raise ValueError("source run is not a resident test-tube report")
    if report.get("mode") != "execute":
        raise ValueError("source run did not open the real execution path")
    steps = sorted((task / "steps").glob("*.json"))
    if not steps:
        raise ValueError("source run has no atomic step evidence")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in steps]
    candidates = []
    for payload in payloads:
        decision = payload.get("decision")
        action = decision.get("action") if isinstance(decision, Mapping) else None
        parameters = action.get("parameters") if isinstance(action, Mapping) else None
        if (
            isinstance(parameters, Mapping)
            and action.get("kind") == "move_tcp"
            and parameters.get("path") == "top_approach"
            and decision.get("next_state") == "APPROACH_TUBE"
        ):
            candidates.append(payload)
    if len(candidates) != 1:
        raise ValueError("source run must contain exactly one recorded top-pregrasp target")
    reference_step = candidates[0]
    target = _matrix(
        reference_step["decision"]["action"]["parameters"].get("T_base_tcp_target"),
        "recorded top-pregrasp target",
    )
    last = payloads[-1]
    decision = last.get("decision")
    observation = last.get("observation")
    if not isinstance(decision, Mapping) or not isinstance(observation, Mapping):
        raise ValueError("last source step is malformed")
    if (
        decision.get("previous_state") != "APPROACH_TUBE"
        or decision.get("next_state") not in {"APPROACH_TUBE", "HOLD"}
    ):
        raise ValueError("source run did not stop at the reviewed top-pregrasp phase")
    robot = observation.get("robot")
    gripper = observation.get("gripper")
    if not isinstance(robot, Mapping) or not isinstance(gripper, Mapping):
        raise ValueError("last source step lacks robot/gripper evidence")
    recorded = _matrix(robot.get("T_base_tcp"), "last recorded TCP")
    validate_live_start(target, recorded)
    validate_open_gripper(gripper)
    return target, report, reference_step, last


def validate_live_start(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    max_distance_mm: float = 8.0,
    max_rotation_deg: float = 2.0,
) -> dict[str, float]:
    distance_m, rotation_deg = _pose_error(current, reference)
    if not math.isfinite(max_distance_mm) or max_distance_mm <= 0.0:
        raise ValueError("max_distance_mm must be positive and finite")
    if not math.isfinite(max_rotation_deg) or max_rotation_deg <= 0.0:
        raise ValueError("max_rotation_deg must be positive and finite")
    distance_mm = distance_m * 1000.0
    if distance_mm > max_distance_mm:
        raise RuntimeError(
            f"live TCP is {distance_mm:.2f} mm from the recorded top-pregrasp target"
        )
    if rotation_deg > max_rotation_deg:
        raise RuntimeError(
            f"live TCP is rotated {rotation_deg:.2f} deg from the recorded top-pregrasp target"
        )
    return {"distance_mm": distance_mm, "rotation_error_deg": rotation_deg}


def validate_open_gripper(feedback: object) -> dict[str, Any]:
    values = asdict(feedback) if not isinstance(feedback, Mapping) else dict(feedback)
    width_m = float(values.get("width_m", math.nan))
    if values.get("fault_free") is not True:
        raise RuntimeError("Hand-E feedback is not fault-free")
    if values.get("moving") is True:
        raise RuntimeError("Hand-E is still moving")
    if values.get("object_detected") is True:
        raise RuntimeError("Hand-E reports an object while an empty open hand is required")
    if not math.isfinite(width_m) or width_m < 0.048:
        raise RuntimeError(f"Hand-E is not open: measured width {width_m!r} m")
    return values


def build_retreat_target(current: np.ndarray, *, retreat_mm: float = 60.0) -> np.ndarray:
    current = _matrix(current, "current TCP")
    if not math.isfinite(retreat_mm) or not 40.0 <= retreat_mm <= 80.0:
        raise ValueError("retreat_mm must lie in [40, 80]")
    target = current.copy()
    target[2, 3] += retreat_mm / 1000.0
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retreat-mm", type=float, default=60.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-arm-only-collision-model", action="store_true")
    args = parser.parse_args(argv)
    build_retreat_target(np.eye(4), retreat_mm=args.retreat_mm)
    if args.execute and not args.accept_arm_only_collision_model:
        parser.error("--execute requires --accept-arm-only-collision-model")
    return args


def run(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    source_run = args.source_run.resolve()
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

    reference, source_report, source_step, last_step = load_pregrasp_reference(source_run)
    nominal_target = build_retreat_target(reference, retreat_mm=float(args.retreat_mm))
    report: dict[str, Any] = {
        "schema": "demo2gripper.test_tube_pregrasp_retreat.v1",
        "status": "prepared",
        "mode": "execute" if args.execute else "dry_run",
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_run": str(source_run),
        "source_task_id": source_report.get("task_id"),
        "source_pregrasp_step": int(source_step["index"]),
        "source_last_step": int(last_step["index"]),
        "T_base_tcp_recorded_pregrasp": reference.tolist(),
        "T_base_tcp_nominal_retreat": nominal_target.tolist(),
        "retreat_mm": float(args.retreat_mm),
        "start_limits": {"translation_mm": 8.0, "rotation_deg": 2.0},
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
            planned_stage=f"retreat_base_z_{float(args.retreat_mm):g}_mm",
        )
        atomic_write_json(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path)}))
        return 0

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

    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"received signal {signum}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            original_handlers[signum] = signal.signal(signum, interrupted)
        transport = contact_io.SerialModbusRTU(
            str(tactile["hande_port"]), float(tactile.get("modbus_timeout_s", 0.12))
        )
        initial_raw = contact_io._read_healthy_initial_state(transport)
        gripper = HandEModbusRuntime(transport)
        initial_gripper = gripper.feedback()
        validate_open_gripper(initial_gripper)
        tcp = _load_artifact_transform(robot, "tcp")
        arm = RosMoveItServoRuntime(config=robot, T_link8_tcp=tcp, output_dir=output)
        initial_arm = arm.state()
        if initial_arm.fault:
            raise RuntimeError(f"arm fault: {initial_arm.fault}")
        start_evidence = validate_live_start(reference, initial_arm.T_base_tcp)
        target = build_retreat_target(
            initial_arm.T_base_tcp, retreat_mm=float(args.retreat_mm)
        )
        report.update(
            status="executing",
            initial_hande_state=initial_raw,
            initial_gripper=asdict(initial_gripper),
            initial_T_base_tcp=initial_arm.T_base_tcp.tolist(),
            live_start_evidence=start_evidence,
            T_base_tcp_target=target.tolist(),
        )
        save()

        receipt = dict(
            arm.servo_target(
                target=target,
                action_id="retreat_pregrasp_base_z",
                position_tolerance_m=0.0015,
            )
        )
        report["hardware_commands_dispatched"] = True
        save()
        observations: list[dict[str, Any]] = []
        deadline = time.monotonic() + 15.0
        final_state = None
        while time.monotonic() < deadline:
            state = arm.state()
            if state.fault:
                raise RuntimeError(f"arm fault during pregrasp retreat: {state.fault}")
            feedback = gripper.feedback()
            validate_open_gripper(feedback)
            progress_mm = float(
                (state.T_base_tcp[2, 3] - initial_arm.T_base_tcp[2, 3]) * 1000.0
            )
            xy_drift_mm = float(
                np.linalg.norm(
                    state.T_base_tcp[:2, 3] - initial_arm.T_base_tcp[:2, 3]
                )
                * 1000.0
            )
            _, start_rotation_deg = _pose_error(state.T_base_tcp, initial_arm.T_base_tcp)
            distance_m, target_rotation_deg = _pose_error(state.T_base_tcp, target)
            observations.append(
                {
                    "at_monotonic_s": time.monotonic(),
                    "progress_mm": progress_mm,
                    "xy_drift_mm": xy_drift_mm,
                    "rotation_from_start_deg": start_rotation_deg,
                    "target_distance_m": distance_m,
                    "target_rotation_deg": target_rotation_deg,
                    "motion_complete": state.motion_complete,
                    "gripper": asdict(feedback),
                }
            )
            if (
                progress_mm < -2.0
                or progress_mm > float(args.retreat_mm) + 3.0
                or xy_drift_mm > 3.0
                or start_rotation_deg > 2.0
            ):
                raise RuntimeError("measured TCP left the reviewed base +Z retreat corridor")
            if distance_m <= 0.002 and target_rotation_deg <= 2.0 and state.motion_complete:
                final_state = state
                break
            time.sleep(0.025)
        if final_state is None:
            raise TimeoutError("pregrasp retreat did not reach its target")
        arm.stop(reason="pregrasp retry retreat complete")
        report["stages"].append(
            {
                "name": "retreat_pregrasp_base_z",
                "status": "complete",
                "receipt": receipt,
                "T_base_tcp_final": final_state.T_base_tcp.tolist(),
                "observations": observations,
            }
        )
        report["status"] = "complete"
        save()
        print(json.dumps({"status": report["status"], "report": str(report_path)}))
        return 0
    except Exception as error:
        if arm is not None:
            try:
                arm.stop(reason=f"pregrasp retreat failed: {error}")
            except Exception:
                pass
        if gripper is not None:
            try:
                gripper.stop(reason=f"pregrasp retreat failed: {error}")
            except Exception:
                pass
        report.update(status="held", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
        if arm is not None:
            arm.close()
        if transport is not None:
            transport.close()


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
