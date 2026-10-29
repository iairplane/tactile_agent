#!/usr/bin/env python3
"""Open and vertically clear the exact reviewed empty two-probe HOLD.

Default mode is artifact-only. Physical commands require both ``--execute``
and ``--accept-arm-only-collision-model``. The live path accepts only the
immutable failed run named by ``--failed-run``, verifies the current TCP and
empty 10 mm Hand-E state, opens to 49.5 mm, then moves exactly 80 mm along
base +Z without changing orientation. It never performs another grasp.
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
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import retry_test_tube_shallow_grasp_live as retry  # noqa: E402
import run_test_tube_continuous_close as contact  # noqa: E402

from demo2gripper.test_tube_live_io import (  # noqa: E402
    HandEModbusRuntime,
    RosMoveItServoRuntime,
    _load_artifact_transform,
)

EMPTY_PROBE_ERROR = "bilateral inward tactile-image grasp not found within 10 mm"
RETREAT_M = 0.080


@dataclass(frozen=True)
class EmptyProbeSource:
    report_path: Path
    session_id: str
    deepest_tcp: np.ndarray
    closed_width_m: float
    report_sha256: str


def _report_path(path: Path) -> Path:
    root = path.expanduser().resolve()
    return root / "run_report.json" if root.is_dir() else root


def load_empty_probe_source(
    failed_run: Path, *, config_sha256: str
) -> EmptyProbeSource:
    path = _report_path(failed_run)
    report = retry._read_json(path)
    error = report.get("error", {})
    if (
        report.get("schema") != "demo2gripper.test_tube_shallow_live_retry.v1"
        or report.get("mode") != "execute"
        or report.get("status") != "failed_hold"
        or report.get("config_sha256") != config_sha256
        or not isinstance(error, Mapping)
        or error.get("type") != "RuntimeError"
        or error.get("message") != EMPTY_PROBE_ERROR
        or report.get("maximum_probe_travel_mm") != 10.0
    ):
        raise ValueError("failed run is not the exact reviewed empty two-probe HOLD")
    stages = report.get("stages")
    if not isinstance(stages, list):
        raise ValueError("failed run has no stage history")
    required = [
        "probe_1_move",
        "probe_1_close",
        "probe_1_reopen_for_deeper_retry",
        "probe_2_move",
        "probe_2_close",
    ]
    names = [str(item.get("name", "")) for item in stages]
    positions = []
    for name in required:
        if names.count(name) != 1:
            raise ValueError(f"failed run does not contain exactly one {name}")
        positions.append(names.index(name))
    if positions != sorted(positions) or positions[-1] != len(stages) - 1:
        raise ValueError("failed run did not stop immediately after the second probe")
    move = stages[positions[-2]]
    close = stages[positions[-1]]
    feedback = close.get("gripper", {})
    width = float(feedback.get("width_m", math.nan))
    if (
        move.get("status") != "complete"
        or close.get("status") != "retry"
        or close.get("trigger") != "minimum_opening"
        or close.get("bilateral_image_confirmed") is not False
        or close.get("retained") is not False
        or feedback.get("fault_free") is not True
        or feedback.get("moving") is not False
        or feedback.get("object_detected") is not False
        or not 0.008 <= width <= 0.013
    ):
        raise ValueError("second probe is not the reviewed stationary empty close")
    confirmations = close.get("confirmations")
    if (
        not isinstance(confirmations, list)
        or len(confirmations) != 2
        or any(
            item.get("decision", {}).get("valid") is not True
            or item.get("decision", {}).get("active_serials") != []
            or item.get("decision", {}).get("grasp_ready") is not False
            or any(
                pad.get("image_active") is not False
                for pad in item.get("geometry", {}).values()
            )
            for item in confirmations
        )
    ):
        raise ValueError("second probe lacks two stationary no-contact windows")
    deepest = retry._matrix(move.get("T_base_tcp"), "deepest probe TCP")
    commanded = retry._matrix(
        move.get("receipt", {}).get("target"), "deepest probe command"
    )
    distance, angle = retry.pose_error(deepest, commanded)
    if distance > 0.002 or angle > 2.0:
        raise ValueError("deepest probe did not reach its committed target")
    if (
        report.get("final_arm_stop", {}).get("accepted") is not True
        or report.get("final_gripper_stop", {}).get("accepted") is not True
    ):
        raise ValueError("failed run lacks confirmed final arm/gripper stops")
    session_id = report.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("failed run lacks session_id")
    return EmptyProbeSource(
        report_path=path,
        session_id=session_id,
        deepest_tcp=deepest,
        closed_width_m=width,
        report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def validate_live_empty_probe(
    source: EmptyProbeSource, current_tcp: np.ndarray, feedback: object
) -> dict[str, Any]:
    values = asdict(feedback) if not isinstance(feedback, Mapping) else dict(feedback)
    distance, angle = retry.pose_error(current_tcp, source.deepest_tcp)
    width = float(values.get("width_m", math.nan))
    if distance > 0.003 or angle > 2.0:
        raise RuntimeError("live TCP differs from the recorded deepest empty probe")
    if (
        values.get("fault_free") is not True
        or values.get("moving") is True
        or values.get("object_detected") is True
        or not math.isfinite(width)
        or abs(width - source.closed_width_m) > 0.003
    ):
        raise RuntimeError("live Hand-E differs from the recorded empty close")
    return {
        "translation_error_m": distance,
        "rotation_error_deg": angle,
        "width_m": width,
        "object_detected": bool(values.get("object_detected")),
    }


def retreat_target(start: np.ndarray) -> np.ndarray:
    result = retry._matrix(start, "retreat start")
    result[2, 3] += RETREAT_M
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--failed-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-arm-only-collision-model", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and not args.accept_arm_only_collision_model:
        parser.error("--execute requires --accept-arm-only-collision-model")
    return args


def run(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"config is missing: {config_path}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite a non-empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes)
    live = config["live_adapter"]
    robot = copy.deepcopy(live["robot"])
    tactile = live["tactile"]
    robot["servo"]["max_linear_m_s"] = min(
        float(robot["servo"]["max_linear_m_s"]), 0.010
    )
    robot["servo"]["max_angular_rad_s"] = min(
        float(robot["servo"]["max_angular_rad_s"]), 0.12
    )
    source = load_empty_probe_source(args.failed_run, config_sha256=config_sha256)
    planned_target = retreat_target(source.deepest_tcp)
    report: dict[str, Any] = {
        "schema": "demo2gripper.test_tube_empty_probe_reset.v1",
        "session_id": f"empty-probe-reset-{uuid.uuid4().hex}",
        "mode": "execute" if args.execute else "dry_run",
        "status": "prepared",
        "config": str(config_path),
        "config_sha256": config_sha256,
        "failed_run": str(source.report_path.parent),
        "failed_report_sha256": source.report_sha256,
        "failed_session_id": source.session_id,
        "recorded_deepest_T_base_tcp": source.deepest_tcp.tolist(),
        "recorded_closed_width_m": source.closed_width_m,
        "planned_retreat_T_base_tcp": planned_target.tolist(),
        "retreat_base_z_mm": 80.0,
        "orientation_change": False,
        "arm_only_collision_model_accepted": bool(
            args.accept_arm_only_collision_model
        ),
        "hardware_commands_dispatched": False,
        "stages": [],
    }
    report_path = output / "run_report.json"

    def save() -> None:
        contact._atomic_json(report_path, report)

    save()
    if not args.execute:
        report.update(
            status="dry_run_complete",
            planned_stages=[
                "verify_exact_deepest_empty_probe_hold",
                "open_hande_49.5mm",
                "retreat_base_plus_z_80mm_orientation_fixed",
            ],
        )
        save()
        print(json.dumps({"status": report["status"], "report": str(report_path)}))
        return 0

    os.environ["ROS_DOMAIN_ID"] = "0"
    arm = gripper = transport = None
    handlers: dict[int, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"received signal {signum}")

    def stage(name: str, **values: Any) -> None:
        report["stages"].append(
            {"name": name, "at_monotonic_s": time.monotonic(), **values}
        )
        save()

    def healthy_arm() -> Any:
        state = arm.state()
        if state.fault:
            raise RuntimeError(f"arm fault: {state.fault}")
        return state

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupted)
        transport = contact.SerialModbusRTU(
            tactile["hande_port"], float(tactile.get("modbus_timeout_s", 0.12))
        )
        initial_hande = contact._read_healthy_initial_state(transport)
        gripper = HandEModbusRuntime(transport)
        arm = RosMoveItServoRuntime(
            config=robot,
            T_link8_tcp=_load_artifact_transform(robot, "tcp"),
            output_dir=output,
        )
        state = healthy_arm()
        deadline = time.monotonic() + 2.0
        while not state.motion_complete and time.monotonic() < deadline:
            time.sleep(0.025)
            state = healthy_arm()
        if not state.motion_complete:
            raise RuntimeError("arm is not stationary at the deepest probe")
        feedback = gripper.feedback()
        start_validation = validate_live_empty_probe(
            source, state.T_base_tcp, feedback
        )
        stage(
            "verified_deepest_empty_probe_hold",
            status="complete",
            validation=start_validation,
            initial_hande_state=initial_hande,
            gripper=asdict(feedback),
        )

        receipt = dict(
            gripper.move_width(
                width_m=0.0495,
                speed=45,
                force=8,
                action_id="empty_probe_reset_open",
            )
        )
        report["hardware_commands_dispatched"] = True
        save()
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            healthy_arm()
            feedback = gripper.feedback()
            if not feedback.fault_free:
                raise RuntimeError("Hand-E fault during reset opening")
            if feedback.width_m >= 0.048 and not feedback.moving:
                break
            time.sleep(0.025)
        else:
            raise TimeoutError("Hand-E did not open to at least 48 mm")
        contact._stop_and_confirm(transport, timeout_s=1.0)
        stage(
            "opened_hande_49.5mm",
            status="complete",
            receipt=receipt,
            gripper=asdict(gripper.feedback()),
        )

        start = healthy_arm().T_base_tcp.copy()
        if retry.pose_error(start, source.deepest_tcp)[0] > 0.003:
            raise RuntimeError("TCP changed before the reset retreat")
        target = retreat_target(start)
        receipt = dict(arm.servo_target(target=target, action_id="reset_retreat_80mm"))
        report["hardware_commands_dispatched"] = True
        save()
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            state = healthy_arm()
            feedback = gripper.feedback()
            if not feedback.fault_free or feedback.moving or feedback.width_m < 0.048:
                arm.stop(reason="Hand-E state changed during reset retreat")
                raise RuntimeError("Hand-E open state was lost during reset retreat")
            delta = state.T_base_tcp[:3, 3] - start[:3, 3]
            progress = float(delta[2])
            lateral = float(np.linalg.norm(delta[:2]))
            _, angle = retry.pose_error(state.T_base_tcp, target)
            if progress < -0.002 or progress > RETREAT_M + 0.002 or lateral > 0.003 or angle > 2.0:
                arm.stop(reason="reset retreat corridor violation")
                raise RuntimeError("reset retreat left the bounded base +Z corridor")
            distance, _ = retry.pose_error(state.T_base_tcp, target)
            if distance <= 0.002 and state.motion_complete:
                arm.stop(reason="empty probe reset retreat complete")
                stage(
                    "retreated_base_plus_z_80mm",
                    status="complete",
                    receipt=receipt,
                    target=target.tolist(),
                    T_base_tcp=state.T_base_tcp.tolist(),
                    gripper=asdict(feedback),
                )
                report["status"] = "reset_complete"
                break
            time.sleep(0.025)
        else:
            arm.stop(reason="empty probe reset retreat timeout")
            raise TimeoutError("80 mm reset retreat did not reach its target")
    except BaseException as error:
        report["status"] = "failed_hold"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        for label, resource in (("arm", arm), ("gripper", gripper)):
            if resource is not None:
                try:
                    report[f"final_{label}_stop"] = dict(
                        resource.stop(reason="empty probe reset finished")
                    )
                except BaseException as error:
                    report[f"{label}_stop_error"] = str(error)
                    report["status"] = "failed_hold"
        for resource in (arm, gripper):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    report.setdefault("cleanup_errors", []).append(str(error))
        if transport is not None and gripper is None:
            transport.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        save()
    print(json.dumps({"status": report["status"], "report": str(report_path)}))
    return 0 if report["status"] == "reset_complete" else 2


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
