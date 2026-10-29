#!/usr/bin/env python3
"""Retry an edge-biased tube grasp with two bounded tactile image probes.

The source run must be the exact resident HOLD where Hand-E gOBJ stopped the
close.  Its per-pad audit must prove bilateral, edge-biased WARPED image
contact.  Live execution stops and opens Hand-E, learns a new unloaded image
baseline, then performs at most two automatic 5 mm probes along the recorded
approach axis.  Every close is continuous and is stopped by the first image
contact or gOBJ.  A grasp qualifies only after two stationary bilateral image
windows whose contact centres moved inward.  Only then may the arm lift 80 mm
while retaining bilateral image contact.

Default mode is file-only.  Physical commands require both ``--execute`` and
``--accept-arm-only-collision-model``.  This script never enables FCI or
unlocks the robot.
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
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_test_tube_continuous_close as contact  # noqa: E402

from demo2gripper.test_tube_live_io import (  # noqa: E402
    HandEModbusRuntime,
    RosMoveItServoRuntime,
    _load_artifact_transform,
)
from demo2gripper.test_tube_tactile_supervisor import (  # noqa: E402
    compensated_residual,
    residual_contact_geometry,
)

SHALLOW_REASON = (
    "Hand-E object status stopped the close before fresh finger-pad tactile contact; "
    "grasp is too shallow and requires a deeper retry"
)
PROBE_STEP_M = 0.005
MAX_PROBE_TRAVEL_M = 0.010
LIFT_M = 0.080
MIN_INWARD_SHIFT_PX = 5.0


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


def pose_error(measured: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    measured = _matrix(measured, "measured pose")
    target = _matrix(target, "target pose")
    distance = float(np.linalg.norm(measured[:3, 3] - target[:3, 3]))
    rotation = Rotation.from_matrix(measured[:3, :3] @ target[:3, :3].T)
    return distance, math.degrees(float(rotation.magnitude()))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _task_dir(path: Path) -> Path:
    root = path.expanduser().resolve()
    if (root / "task" / "run_report.json").is_file():
        return root / "task"
    if (root / "run_report.json").is_file():
        return root
    raise ValueError(f"resident task report is missing below {root}")


@dataclass(frozen=True)
class ShallowSource:
    task_dir: Path
    task_id: str
    terminal_tcp: np.ndarray
    terminal_width_m: float
    approach_axis_base: np.ndarray
    old_centres_uv: dict[str, tuple[float, float]]
    terminal_audit: dict[str, Any]
    artifacts_sha256: str


@dataclass(frozen=True)
class OpenResumeSource:
    report_path: Path
    session_id: str
    opened_width_m: float
    report_sha256: str
    error_message: str


def load_shallow_source(source_run: Path) -> ShallowSource:
    task = _task_dir(source_run)
    report_path = task / "run_report.json"
    report = _read_json(report_path)
    if report.get("schema") != "demo2gripper.test_tube_closed_loop_report.v1":
        raise ValueError("source is not a resident test-tube run")
    if (
        report.get("mode") != "execute"
        or report.get("status") != "held"
        or report.get("final_state") != "HOLD"
    ):
        raise ValueError("source is not a held real execution")
    files = sorted((task / "steps").glob("*.json"))
    if not files or report.get("step_count") != len(files):
        raise ValueError("source step history is incomplete")
    steps = [_read_json(path) for path in files]
    if any(item.get("index") != index for index, item in enumerate(steps)):
        raise ValueError("source step indices are not contiguous")
    last = steps[-1]
    decision = last.get("decision", {})
    action = decision.get("action", {}) if isinstance(decision, Mapping) else {}
    dispatch = last.get("dispatch", {})
    receipt = dispatch.get("receipt", {}) if isinstance(dispatch, Mapping) else {}
    if (
        decision.get("previous_state") != "SIDE_GRASP"
        or decision.get("next_state") != "HOLD"
        or action.get("kind") != "stop_and_hold"
        or action.get("reason") != SHALLOW_REASON
        or dispatch.get("status") != "held"
        or receipt.get("held") is not True
        or receipt.get("arm", {}).get("accepted") is not True
        or receipt.get("gripper", {}).get("accepted") is not True
    ):
        raise ValueError("source is not the exact reviewed shallow-gOBJ HOLD")
    observation = last.get("observation", {})
    robot = observation.get("robot", {}) if isinstance(observation, Mapping) else {}
    gripper = observation.get("gripper", {}) if isinstance(observation, Mapping) else {}
    tactile = observation.get("tactile", {}) if isinstance(observation, Mapping) else {}
    width = float(gripper.get("width_m", math.nan))
    if (
        robot.get("fault") is not None
        or gripper.get("fault_free") is not True
        or gripper.get("moving") is not False
        or gripper.get("object_detected") is not True
        or not 0.010 <= width <= 0.030
        or tactile.get("valid") is not True
    ):
        raise ValueError("terminal robot/Hand-E feedback is not the shallow HOLD")
    terminal = _matrix(robot.get("T_base_tcp"), "terminal TCP")

    audit_path = task / "tactile_interpreter" / "latest.json"
    audit = _read_json(audit_path)
    pads = audit.get("pads")
    audit_feedback = audit.get("feedback", {})
    if (
        audit.get("schema") != "demo2gripper.test_tube_session_tactile_audit.v1"
        or not isinstance(pads, list)
        or len(pads) != 2
        or not isinstance(audit_feedback, Mapping)
        or not math.isclose(
            float(audit_feedback.get("observed_at_s", math.nan)),
            float(tactile.get("observed_at_s", math.nan)),
            abs_tol=1e-6,
            rel_tol=0.0,
        )
        or not all(
            isinstance(item, Mapping)
            and item.get("image_active") is True
            and float(item.get("area_px", 0.0)) >= 116.0
            and isinstance(item.get("center_uv"), list)
            and len(item["center_uv"]) == 2
            for item in pads
        )
    ):
        raise ValueError("terminal audit lacks fresh bilateral tactile-image contact")
    old_centres = {
        str(item["serial"]): (float(item["center_uv"][0]), float(item["center_uv"][1]))
        for item in pads
    }

    top = final = None
    for step in steps:
        item_decision = step.get("decision", {})
        item_action = item_decision.get("action", {}) if isinstance(item_decision, Mapping) else {}
        parameters = item_action.get("parameters", {}) if isinstance(item_action, Mapping) else {}
        if (
            item_action.get("kind") == "move_tcp"
            and parameters.get("path") == "top_approach"
        ):
            top = _matrix(parameters.get("T_base_tcp_target"), "top-pregrasp target")
        if parameters.get("final_contact_target") is not None:
            final = _matrix(parameters["final_contact_target"], "final-contact target")
    if top is None or final is None:
        raise ValueError("source run lacks the committed approach geometry")
    vector = final[:3, 3] - top[:3, 3]
    distance = float(np.linalg.norm(vector))
    if not 0.030 <= distance <= 0.080:
        raise ValueError("committed approach distance is outside [30, 80] mm")
    axis = vector / distance
    if float(axis @ np.asarray([0.0, 0.0, -1.0])) < 0.95:
        raise ValueError("committed approach is not predominantly base -Z")
    if pose_error(terminal, final)[0] > 0.005:
        raise ValueError("terminal TCP is not near the committed contact target")

    digest = hashlib.sha256(report_path.read_bytes())
    digest.update(audit_path.read_bytes())
    for path in files:
        digest.update(path.read_bytes())
    task_id = report.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("source run lacks task_id")
    return ShallowSource(
        task_dir=task,
        task_id=task_id,
        terminal_tcp=terminal,
        terminal_width_m=width,
        approach_axis_base=axis,
        old_centres_uv=old_centres,
        terminal_audit=audit,
        artifacts_sha256=digest.hexdigest(),
    )


def load_open_resume_source(
    resume_run: Path,
    *,
    source: ShallowSource,
    config_sha256: str,
) -> OpenResumeSource:
    root = resume_run.expanduser().resolve()
    path = root / "run_report.json" if root.is_dir() else root
    report = _read_json(path)
    if (
        report.get("schema") != "demo2gripper.test_tube_shallow_live_retry.v1"
        or report.get("mode") != "execute"
        or report.get("status") != "failed_hold"
        or report.get("source_task_id") != source.task_id
        or report.get("source_artifacts_sha256") != source.artifacts_sha256
        or report.get("config_sha256") != config_sha256
    ):
        raise ValueError("resume report does not match this source run and config")
    if pose_error(
        _matrix(report.get("terminal_T_base_tcp"), "resume terminal TCP"),
        source.terminal_tcp,
    )[0] > 1e-6:
        raise ValueError("resume report terminal TCP differs from its source run")
    error = report.get("error", {})
    message = str(error.get("message", "")) if isinstance(error, Mapping) else ""
    if (
        not isinstance(error, Mapping)
        or error.get("type") != "TimeoutError"
        or "synchronized GF220" not in message
    ):
        raise ValueError("resume report did not stop at the reviewed GF220 sync timeout")
    stages = report.get("stages")
    if not isinstance(stages, list):
        raise ValueError("resume report stages are missing")
    opens = [item for item in stages if item.get("name") == "open_before_tactile_probe"]
    if len(opens) != 1 or any(
        str(item.get("name", "")).startswith("probe_") for item in stages
    ):
        raise ValueError("resume report is not the exact post-open pre-probe stop")
    opened = opens[0]
    receipt = opened.get("receipt", {})
    feedback = opened.get("gripper", {})
    width = float(feedback.get("width_m", math.nan))
    if (
        opened.get("status") != "complete"
        or receipt.get("accepted") is not True
        or not math.isclose(float(receipt.get("target_width_m", math.nan)), 0.0495)
        or feedback.get("open_confirmed") is not True
        or feedback.get("fault_free") is not True
        or feedback.get("moving") is not False
        or not 0.047 <= width <= 0.050
        or report.get("final_arm_stop", {}).get("accepted") is not True
        or report.get("final_gripper_stop", {}).get("accepted") is not True
    ):
        raise ValueError("resume report lacks a completed, stopped 49.5 mm opening")
    session_id = report.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("resume report lacks session_id")
    return OpenResumeSource(
        report_path=path,
        session_id=session_id,
        opened_width_m=width,
        report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        error_message=message,
    )


def probe_targets(
    start: np.ndarray, approach_axis_base: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    start = _matrix(start, "probe start")
    axis = np.asarray(approach_axis_base, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("approach axis must be a finite 3-vector")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        raise ValueError("approach axis must be non-zero")
    axis /= norm
    if float(axis @ np.asarray([0.0, 0.0, -1.0])) < 0.95:
        raise ValueError("probe axis is not predominantly base -Z")
    values = []
    for travel in (PROBE_STEP_M, MAX_PROBE_TRAVEL_M):
        target = start.copy()
        target[:3, 3] += axis * travel
        values.append(target)
    return values[0], values[1]


def validate_live_start(
    source: ShallowSource, current_tcp: np.ndarray, feedback: object
) -> dict[str, Any]:
    values = asdict(feedback) if not isinstance(feedback, Mapping) else dict(feedback)
    distance, rotation = pose_error(current_tcp, source.terminal_tcp)
    width = float(values.get("width_m", math.nan))
    if distance > 0.003 or rotation > 2.0:
        raise RuntimeError("live TCP differs from the recorded shallow HOLD")
    if (
        values.get("fault_free") is not True
        or values.get("moving") is True
        or not math.isfinite(width)
        or abs(width - source.terminal_width_m) > 0.003
    ):
        raise RuntimeError("live Hand-E no longer matches the recorded shallow HOLD")
    return {
        "translation_error_m": distance,
        "rotation_error_deg": rotation,
        "width_m": width,
        "object_detected": bool(values.get("object_detected")),
        "object_detected_role": "restart_volatile_corroboration_not_a_start_gate",
    }


def validate_open_resume_live(
    source: ShallowSource,
    resume: OpenResumeSource,
    current_tcp: np.ndarray,
    feedback: object,
) -> dict[str, Any]:
    values = asdict(feedback) if not isinstance(feedback, Mapping) else dict(feedback)
    distance, rotation = pose_error(current_tcp, source.terminal_tcp)
    width = float(values.get("width_m", math.nan))
    if distance > 0.003 or rotation > 2.0:
        raise RuntimeError("live TCP differs from the recorded post-open pose")
    if (
        values.get("fault_free") is not True
        or values.get("moving") is True
        or not math.isfinite(width)
        or width < 0.048
        or abs(width - resume.opened_width_m) > 0.003
    ):
        raise RuntimeError("live Hand-E no longer matches the recorded open resume state")
    return {
        "translation_error_m": distance,
        "rotation_error_deg": rotation,
        "width_m": width,
        "prior_opened_width_m": resume.opened_width_m,
        "open_confirmed": bool(values.get("open_confirmed")),
        "open_confirmed_role": "restart_volatile_corroboration_not_a_resume_gate",
        "object_detected": bool(values.get("object_detected")),
    }


def image_geometry(window: object, profile: object) -> dict[str, dict[str, Any]]:
    """Return WARPED-only contact area/centre for each fresh dual-pad window."""

    sources = dict(zip(window.serials, window.windows, strict=True))
    results: dict[str, dict[str, Any]] = {}
    for finger in profile.fingers:
        source = sources[finger.serial]
        contacts = []
        for frame in np.asarray(source.warped.values):
            residual = compensated_residual(
                frame, finger.warped_reference, finger.roi_mask
            )
            geometry = residual_contact_geometry(
                residual,
                finger.roi_mask,
                residual_threshold=finger.warped_residual_threshold,
                min_contact_area_px=finger.min_component_area_px,
                min_orientation_elongation=1.35,
                min_orientation_concentration=0.25,
            )
            if geometry.contact_candidate:
                contacts.append(geometry)
        centres = [item.center_uv_px for item in contacts if item.center_uv_px is not None]
        centre = None
        if centres:
            median = np.median(np.asarray(centres, dtype=float), axis=0)
            centre = (float(median[0]), float(median[1]))
        results[finger.serial] = {
            "image_active": len(contacts) / len(source.warped.values)
            >= profile.contact_frame_fraction,
            "area_px": (
                float(np.median([item.area_px for item in contacts]))
                if contacts
                else 0.0
            ),
            "center_uv": centre,
            "image_height_px": int(source.warped.values.shape[1]),
        }
    return results


def centres_moved_inward(
    old: Mapping[str, tuple[float, float]], current: Mapping[str, Mapping[str, Any]]
) -> tuple[bool, dict[str, float]]:
    improvements: dict[str, float] = {}
    for serial, old_center in old.items():
        item = current.get(serial)
        if item is None or item.get("center_uv") is None or item.get("image_active") is not True:
            return False, improvements
        new_center = item["center_uv"]
        middle_v = (float(item["image_height_px"]) - 1.0) / 2.0
        improvement = abs(old_center[1] - middle_v) - abs(float(new_center[1]) - middle_v)
        improvements[serial] = improvement
    return all(value >= MIN_INWARD_SHIFT_PX for value in improvements.values()), improvements


def calibrate_then_collect_unloaded(source: object) -> tuple[Any, tuple[Any, ...]]:
    """Honor GF220LiveSource's required calibration-before-collection order."""

    calibrations = source.calibrate_unloaded(timeout_s=15.0)
    windows = tuple(
        contact._collect(source, frames=5, timeout_s=1.0) for _ in range(3)
    )
    return calibrations, windows


def _gf220_sync_timeout(error: TimeoutError) -> bool:
    message = str(error)
    return "synchronized GF220" in message or "fresh pair" in message


def start_calibrated_gf220(
    profile_path: Path,
) -> tuple[Any, Any, tuple[Any, ...], list[dict[str, Any]]]:
    """Start/calibrate GF220, with exactly one restart on pair-sync timeout."""

    attempts: list[dict[str, Any]] = []
    for attempt in (1, 2):
        source = contact.GF220LiveSource(
            profile_path,
            config=contact.GF220LiveConfig(
                buffer_frames=40,
                warmup_frames=8,
                warp_validation_frames=2,
                max_frame_age_s=0.4,
                max_pair_skew_s=0.08,
            ),
        )
        try:
            source.start(timeout_s=15.0)
            calibrations, windows = calibrate_then_collect_unloaded(source)
            attempts.append({"attempt": attempt, "status": "ready"})
            return source, calibrations, windows, attempts
        except TimeoutError as error:
            attempts.append(
                {"attempt": attempt, "status": "timeout", "message": str(error)}
            )
            source.close()
            if attempt == 2 or not _gf220_sync_timeout(error):
                raise
    raise AssertionError("unreachable GF220 retry state")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-after-open-run", type=Path)
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
        raise ValueError(f"task config is missing: {config_path}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite a non-empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes)
    live = config["live_adapter"]
    robot = copy.deepcopy(live["robot"])
    tactile = live["tactile"]
    for key, ceiling in (("max_linear_m_s", 0.010), ("max_angular_rad_s", 0.12)):
        value = float(robot["servo"][key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"invalid robot.servo.{key}")
        robot["servo"][key] = min(value, ceiling)
    source_evidence = load_shallow_source(args.source_run)
    resume_evidence = (
        load_open_resume_source(
            args.resume_after_open_run,
            source=source_evidence,
            config_sha256=config_sha256,
        )
        if args.resume_after_open_run is not None
        else None
    )
    targets = probe_targets(
        source_evidence.terminal_tcp, source_evidence.approach_axis_base
    )
    report: dict[str, Any] = {
        "schema": "demo2gripper.test_tube_shallow_live_retry.v1",
        "session_id": f"shallow-live-retry-{uuid.uuid4().hex}",
        "mode": "execute" if args.execute else "dry_run",
        "status": "prepared",
        "config": str(config_path),
        "config_sha256": config_sha256,
        "source_run": str(args.source_run.expanduser().resolve()),
        "source_task_id": source_evidence.task_id,
        "source_artifacts_sha256": source_evidence.artifacts_sha256,
        "terminal_T_base_tcp": source_evidence.terminal_tcp.tolist(),
        "terminal_width_m": source_evidence.terminal_width_m,
        "hande_object_detected_role": "recorded_true_but_restart_volatile_corroboration",
        "hande_open_confirmed_role": "restart_volatile_corroboration_not_a_resume_gate",
        "pre_open_tactile_evidence": "immutable_prior_run_terminal_audit_not_fresh",
        "resume_after_open": (
            None
            if resume_evidence is None
            else {
                "report": str(resume_evidence.report_path),
                "session_id": resume_evidence.session_id,
                "opened_width_m": resume_evidence.opened_width_m,
                "report_sha256": resume_evidence.report_sha256,
                "error_message": resume_evidence.error_message,
            }
        ),
        "approach_axis_base": source_evidence.approach_axis_base.tolist(),
        "probe_targets": [target.tolist() for target in targets],
        "probe_step_mm": 5.0,
        "maximum_probe_travel_mm": 10.0,
        "lift_mm": 80.0,
        "contact_gate": "two_stationary_bilateral_WARPED_image_windows",
        "force6d_role": "not_used_by_this_narrow_image-contact_recovery",
        "minimum_inward_center_shift_px": MIN_INWARD_SHIFT_PX,
        "arm_only_collision_model_accepted": bool(
            args.accept_arm_only_collision_model
        ),
        "servo_limits": robot["servo"],
        "hardware_commands_dispatched": False,
        "observations": [],
        "stages": [],
    }
    report_path = output / "run_report.json"
    contact._atomic_json(report_path, report)
    if not args.execute:
        start_stages = (
            ["verify_exact_live_open_state_and_prior_open_receipt"]
            if resume_evidence is not None
            else [
                "verify_exact_live_pose_width_health_and_prior_tactile_audit",
                "stop_and_open",
            ]
        )
        report.update(
            status="dry_run_complete",
            planned_stages=start_stages
            + [
                "start_then_calibrate_unloaded_before_any_GF220_collect",
                "one_bounded_GF220_restart_on_pair_sync_timeout",
                "probe_5mm_close_confirm",
                "optional_probe_10mm_close_confirm",
                "retained_lift_80mm",
            ],
        )
        contact._atomic_json(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path)}))
        return 0

    os.environ["ROS_DOMAIN_ID"] = "0"
    arm = gripper = tactile_source = transport = None
    reference_centres = dict(source_evidence.old_centres_uv)
    handlers: dict[int, Any] = {}
    deferred: list[tuple[str, Any, Any]] = []

    def save() -> None:
        contact._atomic_json(report_path, report)

    def stage(name: str, **values: Any) -> None:
        report["stages"].append(
            {"name": name, "at_monotonic_s": time.monotonic(), **values}
        )
        save()

    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"received signal {signum}")

    def healthy_arm() -> Any:
        state = arm.state()
        if state.fault:
            raise RuntimeError(f"arm fault: {state.fault}")
        return state

    def observe(stem: str, memory: Any) -> tuple[Any, Any, Any, Any]:
        decision, memory, _, window = contact._observe(
            source=tactile_source,
            profile=profile,
            memory=memory,
            output_dir=output,
            stem=stem,
            timeout_s=0.75,
            persist=False,
        )
        geometry = image_geometry(window, profile)
        deferred.append((stem, decision, window))
        report["observations"].append(
            {
                "stem": stem,
                "at_monotonic_s": time.monotonic(),
                "decision": asdict(decision),
                "image_geometry": geometry,
            }
        )
        if not decision.valid:
            raise RuntimeError("invalid or stale dual-GF220 image window")
        return decision, memory, geometry, window

    def wait_open(action_id: str) -> Any:
        receipt = dict(
            gripper.move_width(width_m=0.0495, speed=45, force=8, action_id=action_id)
        )
        report["hardware_commands_dispatched"] = True
        save()
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            healthy_arm()
            feedback = gripper.feedback()
            if not feedback.fault_free:
                raise RuntimeError("Hand-E fault while opening")
            if feedback.open_confirmed and not feedback.moving:
                return receipt, feedback
            time.sleep(0.025)
        raise TimeoutError("Hand-E opening to 49.5 mm was not confirmed")

    def move_open(name: str, target: np.ndarray, memory: Any) -> Any:
        start = healthy_arm().T_base_tcp.copy()
        receipt = dict(arm.servo_target(target=target, action_id=name))
        report["hardware_commands_dispatched"] = True
        save()
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            decision, memory, _, _ = observe(f"{name}_{len(report['observations']):03d}", memory)
            state = healthy_arm()
            feedback = gripper.feedback()
            if decision.active_serials or feedback.object_detected:
                arm.stop(reason=f"{name}: unexpected open-finger contact")
                raise RuntimeError("contact occurred during open-finger axial probe")
            distance, angle = pose_error(state.T_base_tcp, target)
            travel = target[:3, 3] - start[:3, 3]
            length = float(np.linalg.norm(travel))
            axis = travel / length
            delta = state.T_base_tcp[:3, 3] - start[:3, 3]
            progress = float(delta @ axis)
            lateral = float(np.linalg.norm(delta - progress * axis))
            if progress < -0.002 or progress > length + 0.002 or lateral > 0.003:
                arm.stop(reason=f"{name}: corridor violation")
                raise RuntimeError("probe motion left its bounded axial corridor")
            if distance <= 0.002 and angle <= 2.0 and state.motion_complete:
                arm.stop(reason=f"{name}: target reached")
                stage(
                    name,
                    status="complete",
                    receipt=receipt,
                    T_base_tcp=state.T_base_tcp.tolist(),
                )
                return memory
        arm.stop(reason=f"{name}: timeout")
        raise TimeoutError(f"{name} did not reach its target")

    def close_and_confirm(attempt: int, memory: Any) -> tuple[bool, Any, dict[str, Any]]:
        receipt = dict(
            gripper.move_width(
                width_m=0.010,
                speed=45,
                force=8,
                action_id=f"probe_{attempt}_continuous_close",
            )
        )
        report["hardware_commands_dispatched"] = True
        save()
        deadline = time.monotonic() + 4.0
        trigger = "timeout"
        moving_windows = []
        while time.monotonic() < deadline:
            decision, memory, geometry, _ = observe(
                f"probe_{attempt}_close_{len(moving_windows):03d}", memory
            )
            feedback = gripper.feedback()
            moving_windows.append({"decision": asdict(decision), "geometry": geometry})
            if decision.active_serials:
                trigger = "first_tactile_image_contact"
                break
            if feedback.object_detected:
                trigger = "hande_object_before_image"
                break
            if not feedback.moving and feedback.width_m <= 0.0105:
                trigger = "minimum_opening"
                break
        stopped, reads = contact._stop_and_confirm(transport, timeout_s=1.0)
        confirmations = []
        confirm_memory = contact.TactileCloseMemory()
        for index in range(2):
            decision, confirm_memory, geometry, _ = observe(
                f"probe_{attempt}_stopped_{index}", confirm_memory
            )
            confirmations.append((decision, geometry))
        bilateral = bool(
            len(confirmations) == 2
            and all(len(item[0].active_serials) == 2 for item in confirmations)
            and confirmations[-1][0].grasp_ready
        )
        geometry = confirmations[-1][1]
        inward, improvements = centres_moved_inward(
            reference_centres, geometry
        )
        feedback = gripper.feedback()
        retained = bool(
            bilateral
            and inward
            and feedback.fault_free
            and not feedback.moving
            and feedback.object_detected
            and 0.005 <= feedback.width_m <= 0.045
        )
        evidence = {
            "receipt": receipt,
            "trigger": trigger,
            "stopped_hande": stopped,
            "stop_feedback_reads": reads,
            "moving_windows": moving_windows,
            "confirmations": [
                {"decision": asdict(item[0]), "geometry": item[1]}
                for item in confirmations
            ],
            "bilateral_image_confirmed": bilateral,
            "centres_moved_inward": inward,
            "centre_improvement_px": improvements,
            "retained": retained,
            "gripper": asdict(feedback),
        }
        stage(f"probe_{attempt}_close", status="retained" if retained else "retry", **evidence)
        return retained, memory, evidence

    def lift_with_contact(memory: Any) -> Any:
        start = healthy_arm().T_base_tcp.copy()
        target = start.copy()
        target[2, 3] += LIFT_M
        receipt = dict(arm.servo_target(target=target, action_id="retained_lift_80mm"))
        report["hardware_commands_dispatched"] = True
        save()
        loss_windows = 0
        observations = []
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            decision, memory, geometry, _ = observe(
                f"lift_{len(observations):03d}", memory
            )
            state = healthy_arm()
            feedback = gripper.feedback()
            bilateral = len(decision.active_serials) == 2
            loss_windows = 0 if bilateral else loss_windows + 1
            distance, angle = pose_error(state.T_base_tcp, target)
            observations.append(
                {
                    "T_base_tcp": state.T_base_tcp.tolist(),
                    "motion_complete": state.motion_complete,
                    "gripper": asdict(feedback),
                    "decision": asdict(decision),
                    "geometry": geometry,
                }
            )
            if (
                not feedback.fault_free
                or feedback.moving
                or not feedback.object_detected
                or loss_windows >= 2
            ):
                arm.stop(reason="retained lift lost tactile hold")
                raise RuntimeError("bilateral tactile-image hold was lost during lift")
            if distance <= 0.002 and angle <= 2.0 and state.motion_complete:
                arm.stop(reason="retained lift reached")
                stage(
                    "retained_lift_80mm",
                    status="complete",
                    receipt=receipt,
                    target=target.tolist(),
                    T_base_tcp_final=state.T_base_tcp.tolist(),
                    observations=observations,
                )
                return memory
        arm.stop(reason="retained lift timeout")
        raise TimeoutError("80 mm retained lift did not reach its target")

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
        contact._read_healthy_initial_state(transport)
        initial_arm = healthy_arm()
        if not initial_arm.motion_complete:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not initial_arm.motion_complete:
                time.sleep(0.025)
                initial_arm = healthy_arm()
        if not initial_arm.motion_complete:
            raise RuntimeError("arm is not stationary at the retry start pose")
        initial_gripper = gripper.feedback()
        if resume_evidence is None:
            live_start = validate_live_start(
                source_evidence, initial_arm.T_base_tcp, initial_gripper
            )
            stage(
                "verified_shallow_hold",
                status="complete",
                initial_hande_state=initial_hande,
                live_start=live_start,
                gripper=asdict(initial_gripper),
                recorded_tactile=source_evidence.terminal_audit,
            )

            contact._stop_and_confirm(transport, timeout_s=1.0)
            open_receipt, opened = wait_open("open_before_tactile_probe")
            stage(
                "open_before_tactile_probe",
                status="complete",
                receipt=open_receipt,
                gripper=asdict(opened),
            )
        else:
            live_start = validate_open_resume_live(
                source_evidence,
                resume_evidence,
                initial_arm.T_base_tcp,
                initial_gripper,
            )
            stage(
                "verified_resume_after_open",
                status="complete",
                initial_hande_state=initial_hande,
                live_start=live_start,
                gripper=asdict(initial_gripper),
                prior_report=str(resume_evidence.report_path),
                prior_report_sha256=resume_evidence.report_sha256,
                repeated_open_command=False,
            )

        profile_path = Path(tactile["gf220_profile"])
        if not profile_path.is_absolute():
            profile_path = ROOT / profile_path
        tactile_source, calibrations, unloaded, gf220_attempts = (
            start_calibrated_gf220(profile_path)
        )
        profile = contact.fit_unloaded_contact_profile(
            session_id=report["session_id"],
            calibrations=calibrations,
            windows=unloaded,
            min_unique_frames=2,
            contact_frame_fraction=0.5,
        )
        contact._save_profile(output, profile)
        for index, window in enumerate(unloaded):
            contact._save_window(output / f"unloaded_window_{index:02d}.npz", window)
        stage(
            "calibrated_unloaded_tactile",
            status="complete",
            api_order="start_then_calibrate_unloaded_then_collect",
            source_attempts=gf220_attempts,
            maximum_source_restarts=1,
            max_pair_skew_s=0.08,
            max_frame_age_s=0.4,
            contact_reference_centres_uv=reference_centres,
            contact_reference_provenance="immutable_prior_run_terminal_audit",
        )
        memory = contact.TactileCloseMemory()
        empty, memory, _, _ = observe("unloaded_preflight", memory)
        if empty.active_serials:
            raise RuntimeError("finger pads are not unloaded after opening")
        origin = healthy_arm().T_base_tcp.copy()
        probe_targets_live = probe_targets(origin, source_evidence.approach_axis_base)

        retained = False
        for attempt, target in enumerate(probe_targets_live, start=1):
            memory = move_open(f"probe_{attempt}_move", target, memory)
            retained, memory, _ = close_and_confirm(attempt, memory)
            if retained:
                break
            if attempt < len(probe_targets_live):
                _, reopened = wait_open(f"probe_{attempt}_reopen_for_deeper_retry")
                stage(
                    f"probe_{attempt}_reopen_for_deeper_retry",
                    status="complete",
                    gripper=asdict(reopened),
                )
                memory = contact.TactileCloseMemory()
                empty, memory, _, _ = observe(f"probe_{attempt}_reopened_empty", memory)
                if empty.active_serials:
                    raise RuntimeError("contact persisted after reopening")
        if not retained:
            raise RuntimeError("bilateral inward tactile-image grasp not found within 10 mm")
        memory = lift_with_contact(contact.TactileCloseMemory())
        final_checks = []
        for index in range(2):
            decision, memory, geometry, _ = observe(f"post_lift_{index}", memory)
            final_checks.append(
                {"decision": asdict(decision), "geometry": geometry}
            )
        if not all(len(item["decision"]["active_serials"]) == 2 for item in final_checks):
            raise RuntimeError("post-lift bilateral tactile image was not retained")
        report["post_lift_confirmations"] = final_checks
        report["T_base_tcp_final"] = healthy_arm().T_base_tcp.tolist()
        report["final_gripper"] = asdict(gripper.feedback())
        report["status"] = "grasp_ready_lifted"
    except BaseException as error:
        report["status"] = "failed_hold"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        for label, resource in (("arm", arm), ("gripper", gripper)):
            if resource is not None:
                try:
                    report[f"final_{label}_stop"] = dict(
                        resource.stop(reason="shallow live retry finished")
                    )
                except BaseException as error:
                    report[f"{label}_stop_error"] = str(error)
                    report["status"] = "failed_hold"
        for stem, decision, window in deferred:
            contact._persist_observation(
                output, stem=stem, decision=decision, window=window
            )
        for resource in (arm, gripper, tactile_source):
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
    return 0 if report["status"] == "grasp_ready_lifted" else 2


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
