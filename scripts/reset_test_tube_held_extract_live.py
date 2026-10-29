#!/usr/bin/env python3
"""Return one provenance-checked held test tube to its source and retreat.

The default mode is artifact-only.  Physical commands require ``--execute``
and the reduced-collision acknowledgement.  The live path accepts only a
resident closed-loop run which ended in a confirmed ``LEVEL_HELD_TUBE`` hold,
revalidates the unchanged TCP and Hand-E width, follows the recorded extraction
TCP samples in reverse while keeping the grasp closed, opens at the recorded
contact pose, then follows the same samples forward to clear the source.
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
from collections.abc import Mapping, Sequence
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

REPORT_SCHEMA = "demo2gripper.test_tube_closed_loop_report.v1"
RESET_SCHEMA = "demo2gripper.test_tube_held_extract_reset.v1"


@dataclass(frozen=True)
class HeldExtractSource:
    report_path: Path
    task_id: str
    source_status: str
    failure_error: str | None
    terminal_previous_state: str
    terminal_tcp: np.ndarray
    last_extract_tcp: np.ndarray
    contact_tcp: np.ndarray
    held_width_m: float
    reverse_path: tuple[np.ndarray, ...]
    retained_step_indices: tuple[int, ...]
    last_extract_step_index: int
    level_return_step_indices: tuple[int, ...]
    artifacts_sha256: str


def _task_dir(path: Path) -> Path:
    root = path.expanduser().resolve()
    if (root / "task" / "run_report.json").is_file():
        return root / "task"
    if (root / "run_report.json").is_file():
        return root
    raise ValueError(f"held run has no run_report.json: {root}")


def _observation_tcp(step: Mapping[str, Any], label: str) -> np.ndarray:
    try:
        value = step["observation"]["robot"]["T_base_tcp"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{label} lacks measured T_base_tcp") from error
    return retry._matrix(value, f"{label} measured T_base_tcp")


def _strict_contact(observation: Mapping[str, Any]) -> bool:
    tactile = observation.get("tactile", {})
    return bool(
        tactile.get("valid") is True
        and tactile.get("contact_present") is True
        and tactile.get("bilateral_contact") is True
        and tactile.get("bilateral_contact_interior") is True
    )


def _sparsify_reverse_path(
    terminal: np.ndarray,
    measured_forward: Sequence[tuple[int, np.ndarray]],
    *,
    minimum_spacing_m: float = 0.005,
    minimum_rotation_deg: float = 3.0,
    required_indices: frozenset[int] = frozenset(),
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    if not measured_forward:
        raise ValueError("held run has no measured extraction trajectory")
    candidates = [(-1, terminal), *reversed(measured_forward)]
    retained: list[np.ndarray] = []
    indices: list[int] = []
    for index, pose in candidates:
        translation_m = rotation_deg = math.inf
        if retained:
            translation_m, rotation_deg = retry.pose_error(retained[-1], pose)
        if (
            not retained
            or index in required_indices
            or translation_m >= minimum_spacing_m
            or rotation_deg >= minimum_rotation_deg
        ):
            retained.append(pose.copy())
            indices.append(index)
    contact_index, contact_pose = measured_forward[0]
    if not np.allclose(retained[-1], contact_pose, atol=1e-9, rtol=0.0):
        retained.append(contact_pose.copy())
        indices.append(contact_index)
    return tuple(retained), tuple(indices)


def load_held_extract_source(
    path: Path, *, gf220_serials: Sequence[str] | None = None
) -> HeldExtractSource:
    task = _task_dir(path)
    report_path = task / "run_report.json"
    report = retry._read_json(report_path)
    source_status = report.get("status")
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("mode") != "execute"
        or source_status not in {"held", "failed_closed"}
        or report.get("final_state") != "HOLD"
        or report.get("hardware_commands_dispatched") is not True
    ):
        raise ValueError("source is not an executed resident held or failed-closed run")
    task_id = report.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("held run lacks task_id")
    files = sorted((task / "steps").glob("*.json"))
    if not files or report.get("step_count") != len(files):
        raise ValueError("held run step history is incomplete")
    steps = [retry._read_json(item) for item in files]
    if any(step.get("index") != index for index, step in enumerate(steps)):
        raise ValueError("held run step indices are not contiguous")
    last_step = report.get("last_step")
    if not isinstance(last_step, str) or Path(last_step).name != files[-1].name:
        raise ValueError("held run last_step does not name the terminal step artifact")

    terminal = steps[-1]
    decision = terminal.get("decision", {})
    action = decision.get("action", {})
    dispatch = terminal.get("dispatch", {})
    receipt = dispatch.get("receipt", {})
    observation = terminal.get("observation", {})
    gripper = observation.get("gripper", {})
    robot = observation.get("robot", {})
    width = float(gripper.get("width_m", math.nan))
    observed_terminal_tcp = _observation_tcp(terminal, "terminal hold")
    failure_error: str | None = None
    if source_status == "held":
        standard_level_hold = bool(
            decision.get("previous_state") == "LEVEL_HELD_TUBE"
            and decision.get("next_state") == "HOLD"
            and action.get("kind") == "stop_and_hold"
        )
        extract_uncertainty_hold = bool(
            decision.get("previous_state") == "EXTRACT"
            and decision.get("next_state") == "HOLD"
            and action.get("kind") == "stop_and_hold"
            and action.get("reason") == "tube_position_uncertainty_too_large"
            and receipt.get("arm", {}).get("reason")
            == "tube_position_uncertainty_too_large"
            and receipt.get("gripper", {}).get("reason")
            == "tube_position_uncertainty_too_large"
        )
        terminal_contract = bool(
            (standard_level_hold or extract_uncertainty_hold)
            and dispatch.get("status") == "held"
            and receipt.get("held") is True
            and receipt.get("arm", {}).get("accepted") is True
            and receipt.get("gripper", {}).get("accepted") is True
        )
    else:
        serials = tuple(str(value) for value in (gf220_serials or ()))
        failure_error = report.get("error")
        accepted_errors = {f"GF220 acquisition failed for {serial}" for serial in serials}
        hold = report.get("hold_receipt", {})
        terminal_contract = bool(
            serials
            and failure_error in accepted_errors
            and decision.get("previous_state") == "LEVEL_HELD_TUBE"
            and decision.get("next_state") == "LEVEL_HELD_TUBE"
            and action.get("kind") == "servo_tcp"
            and dispatch.get("status") == "completed"
            and receipt.get("accepted") is True
            and hold.get("held") is True
            and hold.get("arm", {}).get("accepted") is True
            and hold.get("gripper", {}).get("accepted") is True
            and hold.get("arm", {}).get("reason") == f"runner failure: {failure_error}"
            and hold.get("gripper", {}).get("reason") == f"runner failure: {failure_error}"
        )
        try:
            accepted_terminal_tcp = retry._matrix(
                receipt.get("target"), "terminal completed Servo receipt target"
            )
        except (TypeError, ValueError):
            terminal_contract = False
            accepted_terminal_tcp = observed_terminal_tcp
    if (
        not terminal_contract
        or not _strict_contact(observation)
        or gripper.get("fault_free") is not True
        or gripper.get("moving") is not False
        or not math.isfinite(width)
        or not 0.005 <= width <= 0.040
        or robot.get("fault") is not None
    ):
        raise ValueError(
            "terminal state is not a healthy stopped bilateral LEVEL_HELD_TUBE hold"
        )
    # A runner-level failure happens while acquiring the *next* observation.
    # The last completed Servo receipt therefore post-dates the terminal step's
    # measured observation.  Its rigid, accepted target is the only admissible
    # live-start candidate for failed_closed; an ordinary held/HOLD terminal
    # continues to use its measured observation.
    terminal_tcp = (
        accepted_terminal_tcp
        if source_status == "failed_closed"
        else observed_terminal_tcp
    )

    contact_steps: list[Mapping[str, Any]] = []
    for step in steps:
        step_decision = step.get("decision", {})
        step_action = step_decision.get("action", {})
        if (
            step_decision.get("previous_state") == "SIDE_GRASP"
            and step_decision.get("next_state") == "EXTRACT"
            and step_action.get("kind") == "servo_tcp"
            and step.get("dispatch", {}).get("status") == "completed"
            and step.get("dispatch", {}).get("receipt", {}).get("accepted") is True
            and _strict_contact(step.get("observation", {}))
        ):
            contact_steps.append(step)
    if len(contact_steps) != 1:
        raise ValueError("held run must contain exactly one strict grasp-to-extract transition")
    contact_step = contact_steps[0]
    contact_index = int(contact_step["index"])

    measured_forward: list[tuple[int, np.ndarray]] = []
    extract_indices: list[int] = []
    for step in steps[contact_index:]:
        previous = step.get("decision", {}).get("previous_state")
        if previous not in {"SIDE_GRASP", "EXTRACT", "LEVEL_HELD_TUBE"}:
            raise ValueError("held trajectory left grasp/extract/level states before stopping")
        if not _strict_contact(step.get("observation", {})):
            raise ValueError("recorded return trajectory contains unverified contact")
        index = int(step["index"])
        measured_forward.append(
            (index, _observation_tcp(step, f"step {step['index']}"))
        )
        step_action = step.get("decision", {}).get("action", {})
        if (
            previous == "EXTRACT"
            and step_action.get("kind") == "servo_tcp"
            and step.get("dispatch", {}).get("status") == "completed"
            and step.get("dispatch", {}).get("receipt", {}).get("accepted") is True
        ):
            extract_indices.append(index)
    if not extract_indices:
        raise ValueError("held run has no completed extraction trajectory")
    last_extract_index = extract_indices[-1]
    last_extract_tcp = next(
        pose for index, pose in measured_forward if index == last_extract_index
    )
    reverse_path, retained_indices = _sparsify_reverse_path(
        terminal_tcp,
        measured_forward,
        required_indices=frozenset({last_extract_index}),
    )
    contact_tcp = measured_forward[0][1]
    try:
        last_extract_position = retained_indices.index(last_extract_index)
    except ValueError as error:
        raise ValueError("reverse path omitted the final EXTRACT pose") from error
    terminal_previous_state = str(decision.get("previous_state"))
    level_return_indices = (
        tuple(
            index
            for index in retained_indices[1:last_extract_position]
            if index > last_extract_index
        )
        if terminal_previous_state == "LEVEL_HELD_TUBE"
        else ()
    )
    if source_status == "failed_closed":
        if last_extract_position < 2 or not level_return_indices:
            raise ValueError("failed-closed return lacks reverse LEVEL waypoints")
        if any(
            abs(reverse_path[index][2, 3] - last_extract_tcp[2, 3]) > 0.008
            for index in range(last_extract_position + 1)
        ):
            raise ValueError("reverse LEVEL path changes height before orientation recovery")
    total = sum(
        float(np.linalg.norm(second[:3, 3] - first[:3, 3]))
        for first, second in zip(reverse_path, reverse_path[1:])
    )
    vertical_return = float(terminal_tcp[2, 3] - contact_tcp[2, 3])
    lateral = max(
        float(np.linalg.norm(pose[:2, 3] - contact_tcp[:2, 3]))
        for pose in reverse_path
    )
    upward_jumps = [
        float(second[2, 3] - first[2, 3])
        for first, second in zip(reverse_path, reverse_path[1:])
    ]
    rotation_jumps = [
        retry.pose_error(first, second)[1]
        for first, second in zip(reverse_path, reverse_path[1:])
    ]
    if (
        not 0.040 <= vertical_return <= 0.120
        or not 0.040 <= total <= 0.160
        or lateral > 0.020
        or any(value > 0.003 for value in upward_jumps)
        or any(value > 12.0 for value in rotation_jumps)
    ):
        raise ValueError("recorded extraction return path exceeds its bounded corridor")

    digest = hashlib.sha256(report_path.read_bytes())
    for item in files:
        digest.update(item.read_bytes())
    return HeldExtractSource(
        report_path=report_path,
        task_id=task_id,
        source_status=str(source_status),
        failure_error=failure_error,
        terminal_previous_state=terminal_previous_state,
        terminal_tcp=terminal_tcp,
        last_extract_tcp=last_extract_tcp,
        contact_tcp=contact_tcp,
        held_width_m=width,
        reverse_path=reverse_path,
        retained_step_indices=retained_indices,
        last_extract_step_index=last_extract_index,
        level_return_step_indices=level_return_indices,
        artifacts_sha256=digest.hexdigest(),
    )


def validate_live_start(
    source: HeldExtractSource, current_tcp: np.ndarray, feedback: object
) -> dict[str, Any]:
    values = asdict(feedback) if not isinstance(feedback, Mapping) else dict(feedback)
    translation_m, rotation_deg = retry.pose_error(current_tcp, source.terminal_tcp)
    width = float(values.get("width_m", math.nan))
    if translation_m > 0.003 or rotation_deg > 2.0:
        raise RuntimeError("current TCP differs from the recorded terminal hold")
    if (
        values.get("fault_free") is not True
        or values.get("moving") is not False
        or not math.isfinite(width)
        or abs(width - source.held_width_m) > 0.003
    ):
        raise RuntimeError("current Hand-E differs from the recorded terminal hold")
    return {
        "translation_error_m": translation_m,
        "rotation_error_deg": rotation_deg,
        "width_m": width,
        "object_detected": bool(values.get("object_detected")),
    }


def configured_gf220_serials(config: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        raw_profile = config["live_adapter"]["tactile"]["gf220_profile"]
    except (KeyError, TypeError) as error:
        raise ValueError("config lacks live_adapter.tactile.gf220_profile") from error
    profile_path = Path(str(raw_profile)).expanduser()
    if not profile_path.is_absolute():
        profile_path = (ROOT / profile_path).resolve()
    profile = retry._read_json(profile_path)
    sensors = profile.get("sensors")
    if not isinstance(sensors, list):
        raise ValueError("GF220 profile sensors must be a list")
    serials = tuple(
        str(item.get("sdk_serial", ""))
        for item in sensors
        if isinstance(item, Mapping)
    )
    if len(serials) != 2 or any(not serial for serial in serials) or len(set(serials)) != 2:
        raise ValueError("GF220 profile must name two distinct sdk_serial values")
    return serials


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--held-run", type=Path, required=True)
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
    config = json.loads(config_bytes)
    live = config["live_adapter"]
    source = load_held_extract_source(
        args.held_run,
        gf220_serials=configured_gf220_serials(config),
    )
    robot = copy.deepcopy(live["robot"])
    tactile = live["tactile"]
    robot["servo"]["max_linear_m_s"] = min(
        float(robot["servo"]["max_linear_m_s"]), 0.010
    )
    robot["servo"]["max_angular_rad_s"] = min(
        float(robot["servo"]["max_angular_rad_s"]), 0.12
    )
    last_extract_position = source.retained_step_indices.index(source.last_extract_step_index)
    if source.terminal_previous_state == "EXTRACT":
        level_recovery_path = (source.reverse_path[0],)
        extraction_descent_path = source.reverse_path
    else:
        level_recovery_path = source.reverse_path[: last_extract_position + 1]
        extraction_descent_path = source.reverse_path[last_extract_position:]
    return_path = tuple(reversed(extraction_descent_path))
    report: dict[str, Any] = {
        "schema": RESET_SCHEMA,
        "mode": "execute" if args.execute else "dry_run",
        "status": "prepared",
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "held_run": str(source.report_path.parent),
        "held_artifacts_sha256": source.artifacts_sha256,
        "task_id": source.task_id,
        "source_status": source.source_status,
        "source_failure_error": source.failure_error,
        "source_terminal_previous_state": source.terminal_previous_state,
        "recorded_terminal_T_base_tcp": source.terminal_tcp.tolist(),
        "recorded_last_extract_T_base_tcp": source.last_extract_tcp.tolist(),
        "recorded_contact_T_base_tcp": source.contact_tcp.tolist(),
        "recorded_held_width_m": source.held_width_m,
        "retained_step_indices": list(source.retained_step_indices),
        "last_extract_step_index": source.last_extract_step_index,
        "reverse_level_step_indices": list(source.level_return_step_indices),
        "reverse_path_T_base_tcp": [pose.tolist() for pose in source.reverse_path],
        "level_recovery_path_T_base_tcp": [
            pose.tolist() for pose in level_recovery_path
        ],
        "extraction_descent_path_T_base_tcp": [
            pose.tolist() for pose in extraction_descent_path
        ],
        "return_path_T_base_tcp": [pose.tolist() for pose in return_path],
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
        planned_stages = ["verify_unchanged_terminal_hold"]
        if source.terminal_previous_state == "LEVEL_HELD_TUBE":
            planned_stages.append(
                "reverse_recorded_leveling_at_extracted_height_with_grasp_closed"
            )
        planned_stages.extend(
            [
                "reverse_recorded_extraction_descent_with_grasp_closed",
                "open_hande_49.5mm_at_recorded_contact_pose",
                "follow_recorded_extraction_to_last_extract_pose_with_gripper_open",
            ]
        )
        report.update(
            status="dry_run_complete",
            planned_stages=planned_stages,
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

    def follow_path(name: str, path: Sequence[np.ndarray], *, require_closed: bool) -> None:
        observations: list[dict[str, Any]] = []
        for index, target in enumerate(path[1:], start=1):
            receipt = dict(
                arm.servo_target(target=target, action_id=f"{name}_{index:02d}")
            )
            report["hardware_commands_dispatched"] = True
            save()
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                state = healthy_arm()
                feedback = gripper.feedback()
                width = float(feedback.width_m)
                if not feedback.fault_free or feedback.moving:
                    arm.stop(reason=f"{name}: Hand-E state changed")
                    raise RuntimeError(f"Hand-E state changed during {name}")
                if require_closed and abs(width - source.held_width_m) > 0.003:
                    arm.stop(reason=f"{name}: held width changed")
                    raise RuntimeError("held width changed during reverse extraction")
                if not require_closed and width < 0.048:
                    arm.stop(reason=f"{name}: open width changed")
                    raise RuntimeError("open Hand-E state changed during retreat")
                distance_m, rotation_deg = retry.pose_error(state.T_base_tcp, target)
                if distance_m <= 0.002 and rotation_deg <= 2.0 and state.motion_complete:
                    observations.append(
                        {
                            "waypoint": index,
                            "target": target.tolist(),
                            "receipt": receipt,
                            "T_base_tcp": state.T_base_tcp.tolist(),
                            "width_m": width,
                        }
                    )
                    break
                time.sleep(0.025)
            else:
                arm.stop(reason=f"{name}: waypoint timeout")
                raise TimeoutError(f"{name} waypoint {index} did not converge")
        arm.stop(reason=f"{name} complete")
        stage(name, status="complete", observations=observations)

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
            raise RuntimeError("arm is not stationary at the recorded terminal hold")
        feedback = gripper.feedback()
        stage(
            "verified_terminal_hold",
            status="complete",
            validation=validate_live_start(source, state.T_base_tcp, feedback),
            initial_hande_state=initial_hande,
            gripper=asdict(feedback),
        )

        if source.terminal_previous_state == "LEVEL_HELD_TUBE":
            follow_path(
                "reversed_recorded_leveling",
                level_recovery_path,
                require_closed=True,
            )
        follow_path(
            "reversed_recorded_extraction_descent",
            extraction_descent_path,
            require_closed=True,
        )
        state = healthy_arm()
        distance_m, rotation_deg = retry.pose_error(state.T_base_tcp, source.contact_tcp)
        if distance_m > 0.002 or rotation_deg > 2.0:
            raise RuntimeError("TCP is not at the recorded supported contact pose")
        receipt = dict(
            gripper.move_width(
                width_m=0.0495,
                speed=45,
                force=8,
                action_id="held_extract_reset_open",
            )
        )
        report["hardware_commands_dispatched"] = True
        save()
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            healthy_arm()
            feedback = gripper.feedback()
            if not feedback.fault_free:
                raise RuntimeError("Hand-E fault during release at the source")
            if feedback.width_m >= 0.048 and not feedback.moving:
                break
            time.sleep(0.025)
        else:
            raise TimeoutError("Hand-E did not open to at least 48 mm")
        contact._stop_and_confirm(transport, timeout_s=1.0)
        stage(
            "opened_at_recorded_contact_pose",
            status="complete",
            receipt=receipt,
            gripper=asdict(gripper.feedback()),
        )
        follow_path(
            "retreated_along_recorded_extraction",
            return_path,
            require_closed=False,
        )
        report["status"] = "reset_complete"
    except BaseException as error:
        report["status"] = "failed_hold"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        for label, resource in (("arm", arm), ("gripper", gripper)):
            if resource is not None:
                try:
                    report[f"final_{label}_stop"] = dict(
                        resource.stop(reason="held extraction reset finished")
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
