#!/usr/bin/env python3
"""Run the fixed demo-02 pick, place, and lid-manipulation state machine.

The existing autonomous-pick runner owns ``OBSERVE`` through ``LIFT``.  This
orchestrator imports its atomic report and, only after a fully successful executed
pick, advances through the fixed continuation:

``TRANSPORT -> PLACE -> RELEASE -> RETREAT -> LID_APPROACH -> LID_GRASP ->
LID_OPEN -> HOLD -> LID_CLOSE -> LID_RELEASE -> FINAL_RETREAT``.

Planning/review is the default.  Physical commands require both ``--execute`` and
``--accept-arm-only-collision-model``.  Every motion tool plans first, audits the
complete FR3 joint path, and executes that exact path only when requested.
"""

from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import math
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from demo2gripper.autonomous_pick import (
    PickStage,
    atomic_write_json,
    cartesian_result_succeeded,
    hande_open_succeeded,
    hande_width_snapshot_valid,
    measured_pose_within_tolerance,
    pose_result_succeeded,
    quaternion_xyzw_to_rotation_matrix,
    rotation_matrix_to_quaternion_xyzw,
)
from demo2gripper.cartesian_pose_sequence import (
    MAX_CARTESIAN_SPEED_M_S,
    CartesianPose,
    load_pose_sequence,
)
from demo2gripper.franka import validate_rigid_transform
from demo2gripper.pose_motion import pose_delta
from scripts.run_autonomous_pick import (
    StageFailure,
    _artifact_summary,
    _configured_ee_pose_command,
    _hande_command,
    _motion_options,
    _read_json,
    _run_command,
    _trajectory_audit_options,
)
from scripts.run_autonomous_pick import (
    _validate_config as _validate_pick_config,
)

SCHEMA = "demo2gripper.demo02_full_task_config.v1"
REPORT_SCHEMA = "demo2gripper.demo02_full_task_report.v1"
BASE_FRAME = "fr3_link0"
POSE_MATCH_POSITION_TOLERANCE_M = 0.001
POSE_MATCH_ROTATION_TOLERANCE_RAD = math.radians(1.0)


class FullTaskStage(StrEnum):
    TRANSPORT = "TRANSPORT"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    LID_APPROACH = "LID_APPROACH"
    LID_GRASP = "LID_GRASP"
    LID_OPEN = "LID_OPEN"
    HOLD = "HOLD"
    LID_CLOSE = "LID_CLOSE"
    LID_RELEASE = "LID_RELEASE"
    FINAL_RETREAT = "FINAL_RETREAT"


FULL_STAGE_ORDER = tuple(stage.value for stage in PickStage) + tuple(
    stage.value for stage in FullTaskStage
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Pass an existing RGB-D capture to the autonomous-pick subprocess",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--accept-arm-only-collision-model",
        action="store_true",
        help="Acknowledge that the installed Hand-E is absent from the MoveIt model",
    )
    return parser.parse_args()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_path(value: Any, *, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _command(config: Mapping[str, Any], name: str) -> list[str]:
    commands = config.get("commands")
    if not isinstance(commands, Mapping):
        raise ValueError("config.commands must be an object")
    value = commands.get(name)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(token, str) or not token for token in value)
    ):
        raise ValueError(f"config.commands.{name} must be a non-empty string array")
    return list(value)


def _vector3(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return vector


def _pose_transform(pose: CartesianPose) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = quaternion_xyzw_to_rotation_matrix(pose.orientation_xyzw)
    transform[:3, 3] = pose.position_m
    return validate_rigid_transform(transform, name="T_base_configured_ee")


def _single_pose(path: Path, *, name: str) -> CartesianPose:
    poses = load_pose_sequence(path, expected_base_frame=BASE_FRAME)
    if len(poses) != 1:
        raise ValueError(f"{name} must contain exactly one waypoint")
    return poses[0]


def _poses_match(
    first: CartesianPose,
    second: CartesianPose,
    *,
    position_tolerance_m: float = POSE_MATCH_POSITION_TOLERANCE_M,
    rotation_tolerance_rad: float = POSE_MATCH_ROTATION_TOLERANCE_RAD,
) -> bool:
    translation, rotation = pose_delta(
        first.position_m,
        first.orientation_xyzw,
        second.position_m,
        second.orientation_xyzw,
    )
    return translation <= position_tolerance_m and rotation <= rotation_tolerance_rad


def _validate_config(
    config: Mapping[str, Any], *, config_path: Path, execute: bool
) -> dict[str, Any]:
    if config.get("schema") != SCHEMA:
        raise ValueError(f"config schema must be {SCHEMA!r}")
    base = config_path.parent.resolve()
    pick_config_path = _resolve_path(config.get("pick_config"), base=base, name="pick_config")
    if not pick_config_path.is_file():
        raise ValueError(f"missing pick_config: {pick_config_path}")
    pick_config = _read_json(pick_config_path)
    pick = _validate_pick_config(pick_config, config_path=pick_config_path, execute=execute)

    waypoint_values = config.get("waypoints")
    motion = config.get("motion")
    lid_gripper = config.get("lid_gripper", {})
    if not isinstance(waypoint_values, Mapping):
        raise ValueError("config.waypoints must be an object")
    if not isinstance(motion, Mapping):
        raise ValueError("config.motion must be an object")
    if not isinstance(lid_gripper, Mapping):
        raise ValueError("config.lid_gripper must be an object")

    waypoint_names = (
        "transport_tcp",
        "placement_tcp",
        "lid_precontact_tcp",
        "lid_contact_tcp",
        "lid_open_waypoints",
        "lid_close_waypoints",
    )
    waypoint_paths = {
        name: _resolve_path(waypoint_values.get(name), base=base, name=f"waypoints.{name}")
        for name in waypoint_names
    }
    for name, path in waypoint_paths.items():
        if not path.is_file():
            raise ValueError(f"missing waypoints.{name}: {path}")

    transport = _single_pose(waypoint_paths["transport_tcp"], name="transport_tcp")
    placement = _single_pose(waypoint_paths["placement_tcp"], name="placement_tcp")
    lid_precontact = _single_pose(waypoint_paths["lid_precontact_tcp"], name="lid_precontact_tcp")
    lid_contact = _single_pose(waypoint_paths["lid_contact_tcp"], name="lid_contact_tcp")
    lid_open = load_pose_sequence(
        waypoint_paths["lid_open_waypoints"], expected_base_frame=BASE_FRAME
    )
    lid_close = load_pose_sequence(
        waypoint_paths["lid_close_waypoints"], expected_base_frame=BASE_FRAME
    )

    raw_place_deltas = motion.get("place_deltas_base_m")
    if not isinstance(raw_place_deltas, list) or not raw_place_deltas:
        raise ValueError("motion.place_deltas_base_m must be a non-empty array")
    place_deltas = [
        _vector3(value, name=f"motion.place_deltas_base_m[{index}]")
        for index, value in enumerate(raw_place_deltas)
    ]
    if len(place_deltas) > 8:
        raise ValueError("motion.place_deltas_base_m is capped at 8 segments")
    for index, delta in enumerate(place_deltas):
        if np.linalg.norm(delta) > 0.05 + 1e-12:
            raise ValueError(f"place delta {index} exceeds the 0.05 m segment cap")
        if delta[2] >= 0.0 or np.linalg.norm(delta[:2]) > 1e-6:
            raise ValueError("place deltas must be strictly downward in base Z")
    place_total = np.sum(place_deltas, axis=0)
    expected_placement_position = np.asarray(transport.position_m) + place_total
    if not np.allclose(
        expected_placement_position,
        placement.position_m,
        atol=POSE_MATCH_POSITION_TOLERANCE_M,
        rtol=0.0,
    ):
        raise ValueError("transport target plus place deltas does not reach placement_tcp")
    _, placement_rotation = pose_delta(
        transport.position_m,
        transport.orientation_xyzw,
        transport.position_m,
        placement.orientation_xyzw,
    )
    if placement_rotation > POSE_MATCH_ROTATION_TOLERANCE_RAD:
        raise ValueError("TRANSPORT and PLACE must preserve the grasp orientation")

    retreat_delta = _vector3(
        motion.get("retreat_delta_base_m", [0.0, 0.0, 0.08]),
        name="motion.retreat_delta_base_m",
    )
    if (
        retreat_delta[2] <= 0.0
        or np.linalg.norm(retreat_delta[:2]) > 1e-6
        or np.linalg.norm(retreat_delta) > 0.10 + 1e-12
    ):
        raise ValueError("retreat delta must be base +Z only and no longer than 0.10 m")

    lid_approach_delta = np.asarray(lid_contact.position_m) - np.asarray(lid_precontact.position_m)
    _, lid_approach_rotation = pose_delta(
        lid_precontact.position_m,
        lid_precontact.orientation_xyzw,
        lid_precontact.position_m,
        lid_contact.orientation_xyzw,
    )
    if (
        np.linalg.norm(lid_approach_delta) <= 0.0
        or np.linalg.norm(lid_approach_delta) > 0.10 + 1e-12
    ):
        raise ValueError("lid precontact-to-contact translation must lie in (0, 0.10] m")
    if lid_approach_rotation > POSE_MATCH_ROTATION_TOLERANCE_RAD:
        raise ValueError("lid contact approach must hold orientation")
    if not _poses_match(lid_open[0], lid_contact):
        raise ValueError("lid-open sequence must start at lid_contact_tcp")
    if not _poses_match(lid_close[0], lid_open[-1]):
        raise ValueError("lid-close sequence must start at the final lid-open waypoint")
    if not _poses_match(lid_close[-1], lid_contact):
        raise ValueError("lid-close sequence must finish at lid_contact_tcp")
    if len(lid_close) != len(lid_open) or any(
        not _poses_match(close_pose, open_pose)
        for close_pose, open_pose in zip(lid_close, reversed(lid_open), strict=True)
    ):
        raise ValueError("lid-close waypoints must reverse the validated lid-open path")

    hold_duration_s = float(motion.get("hold_duration_s", 1.0))
    if not math.isfinite(hold_duration_s) or not 0.0 <= hold_duration_s <= 10.0:
        raise ValueError("motion.hold_duration_s must lie in [0, 10] seconds")
    sequence_speed = float(motion.get("pose_sequence_speed_m_s", MAX_CARTESIAN_SPEED_M_S))
    if not math.isfinite(sequence_speed) or not 0.0 < sequence_speed <= MAX_CARTESIAN_SPEED_M_S:
        raise ValueError(
            f"motion.pose_sequence_speed_m_s must lie in (0, {MAX_CARTESIAN_SPEED_M_S}]"
        )
    sequence_step = float(motion.get("pose_sequence_max_step_m", 0.005))
    if not math.isfinite(sequence_step) or not 0.0 < sequence_step <= 0.010:
        raise ValueError("motion.pose_sequence_max_step_m must lie in (0, 0.010]")

    lid_min_opening_m = float(lid_gripper.get("min_contact_opening_m", 0.002))
    lid_max_opening_m = float(lid_gripper.get("max_contact_opening_m", 0.048))
    if not (
        math.isfinite(lid_min_opening_m)
        and math.isfinite(lid_max_opening_m)
        and 0.0 < lid_min_opening_m < lid_max_opening_m < 0.050
    ):
        raise ValueError("lid contact opening range must lie strictly inside (0, 0.050) m")

    return {
        "repo_root": pick["repo_root"],
        "pick_config": pick_config_path,
        "pick": pick,
        "commands": {
            "pick": _command(config, "pick"),
            "pose_sequence": _command(config, "pose_sequence"),
        },
        "waypoint_paths": waypoint_paths,
        "transport": transport,
        "placement": placement,
        "lid_precontact": lid_precontact,
        "lid_contact": lid_contact,
        "lid_open": lid_open,
        "lid_close": lid_close,
        "place_deltas": place_deltas,
        "retreat_delta": retreat_delta,
        "lid_approach_delta": lid_approach_delta,
        "hold_duration_s": hold_duration_s,
        "pose_sequence_speed_m_s": sequence_speed,
        "pose_sequence_max_step_m": sequence_step,
        "lid_min_opening_m": lid_min_opening_m,
        "lid_max_opening_m": lid_max_opening_m,
    }


def _execution_flags(execute: bool) -> list[str]:
    return ["--execute", "--accept-arm-only-collision-model"] if execute else []


def _pick_command(
    normalized: Mapping[str, Any],
    *,
    output_dir: Path,
    input_dir: Path | None,
    execute: bool,
) -> list[str]:
    command = [
        *normalized["commands"]["pick"],
        "--config",
        str(normalized["pick_config"]),
        "--output-dir",
        str(output_dir),
    ]
    if input_dir is not None:
        command.extend(("--input-dir", str(input_dir.resolve())))
    command.extend(_execution_flags(execute))
    return command


def _pose_command(
    normalized: Mapping[str, Any],
    target: CartesianPose,
    *,
    output: Path,
    execute: bool,
) -> list[str]:
    T_link8_tcp = np.load(normalized["pick"]["t_link8_tcp"], allow_pickle=False)
    command = _configured_ee_pose_command(normalized["pick"], _pose_transform(target), T_link8_tcp)
    return [*command, "--output", str(output), *_execution_flags(execute)]


def _cartesian_command(
    normalized: Mapping[str, Any],
    delta: np.ndarray,
    *,
    output: Path,
    execute: bool,
) -> list[str]:
    vector = _vector3(delta, name="Cartesian delta")
    return [
        *normalized["pick"]["commands"]["cartesian"],
        "--delta-x-m",
        str(vector[0]),
        "--delta-y-m",
        str(vector[1]),
        "--delta-z-m",
        str(vector[2]),
        *_motion_options(normalized["pick"]),
        *_trajectory_audit_options(normalized["pick"]),
        "--max-cartesian-speed-m-s",
        str(normalized["pick"]["max_cartesian_speed_m_s"]),
        "--output",
        str(output),
        *_execution_flags(execute),
    ]


def _pose_sequence_command(
    normalized: Mapping[str, Any],
    input_path: Path,
    *,
    output: Path,
    execute: bool,
) -> list[str]:
    T_link8_tcp = validate_rigid_transform(
        np.load(normalized["pick"]["t_link8_tcp"], allow_pickle=False),
        name="T_link8_configured_ee",
    )
    quaternion = rotation_matrix_to_quaternion_xyzw(T_link8_tcp[:3, :3])
    return [
        *normalized["commands"]["pose_sequence"],
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--link8-to-configured-ee-position-m",
        *(str(value) for value in T_link8_tcp[:3, 3]),
        "--link8-to-configured-ee-orientation-xyzw",
        *(str(value) for value in quaternion),
        "--base-frame",
        BASE_FRAME,
        "--max-step-m",
        str(normalized["pose_sequence_max_step_m"]),
        "--velocity-scale",
        str(normalized["pick"]["velocity_scale"]),
        "--acceleration-scale",
        str(normalized["pick"]["acceleration_scale"]),
        "--max-cartesian-speed-m-s",
        str(normalized["pose_sequence_speed_m_s"]),
        *_trajectory_audit_options(normalized["pick"]),
        *_execution_flags(execute),
    ]


def pose_sequence_result_succeeded(payload: Mapping[str, Any], *, executed: bool) -> bool:
    trajectory = payload.get("trajectory")
    audit = payload.get("trajectory_audit")
    planned = (
        payload.get("schema") == "demo2gripper.moveit_cartesian_pose_sequence.v1"
        and payload.get("moveit_error_code") == 1
        and float(payload.get("path_fraction", 0.0)) == 1.0
        and isinstance(trajectory, Mapping)
        and int(trajectory.get("point_count", 0)) > 1
        and isinstance(audit, Mapping)
        and audit.get("accepted") is True
        and payload.get("planning_succeeded") is True
    )
    if not executed:
        return planned and payload.get("mode") == "plan_only"
    return (
        planned
        and payload.get("mode") == "execute"
        and payload.get("execution_policy") == "plan_then_audit_then_execute_exact_trajectory"
        and payload.get("execution_dispatched") is True
        and payload.get("execution_status") == 4
        and payload.get("execution_error_code") == 1
        and payload.get("execution_succeeded") is True
    )


def _lid_contact_verified(
    payload: Mapping[str, Any], *, min_opening_m: float, max_opening_m: float
) -> bool:
    state = payload.get("state")
    object_status = state.get("object_status") if isinstance(state, Mapping) else None
    opening_mm = state.get("opening_width_mm") if isinstance(state, Mapping) else None
    return (
        payload.get("ok") is True
        and isinstance(state, Mapping)
        and isinstance(object_status, Mapping)
        and object_status.get("code") == 2
        and object_status.get("grasped") is True
        and state.get("fault_code") == 0
        and isinstance(opening_mm, (int, float))
        and min_opening_m <= float(opening_mm) * 1e-3 <= max_opening_m
    )


def _lid_width_snapshot_valid(
    payload: Mapping[str, Any],
    *,
    min_opening_m: float,
    max_opening_m: float,
    reference_opening_m: float,
    max_drift_m: float = 0.002,
) -> bool:
    state = payload.get("state")
    opening_mm = state.get("opening_width_mm") if isinstance(state, Mapping) else None
    if (
        payload.get("ok") is not True
        or not isinstance(state, Mapping)
        or state.get("fault_code") != 0
        or not isinstance(opening_mm, (int, float))
    ):
        return False
    opening_m = float(opening_mm) * 1e-3
    return (
        min_opening_m <= opening_m <= max_opening_m
        and math.isfinite(reference_opening_m)
        and abs(opening_m - reference_opening_m) <= max_drift_m
    )


def _assert_stage_prefix(stages: Sequence[Mapping[str, Any]]) -> None:
    if len(stages) > len(FULL_STAGE_ORDER):
        raise ValueError("full-task report contains too many stages")
    for index, record in enumerate(stages):
        if record.get("name") != FULL_STAGE_ORDER[index]:
            raise ValueError(
                f"expected stage {FULL_STAGE_ORDER[index]}, got {record.get('name')!r}"
            )
        status = record.get("status")
        if status not in {"passed", "failed"}:
            raise ValueError(f"invalid stage status {status!r}")
        if status == "failed" and index != len(stages) - 1:
            raise ValueError("a failed stage must terminate the report")


def _append_stage(
    report: dict[str, Any],
    report_path: Path,
    stage: FullTaskStage,
    *,
    status: str,
    details: Mapping[str, Any],
    started_at: str,
) -> None:
    expected = FULL_STAGE_ORDER[len(report["stages"])]
    if stage.value != expected:
        raise ValueError(f"cannot append {stage.value}; expected {expected}")
    report["stages"].append(
        {
            "name": stage.value,
            "status": status,
            "started_at": started_at,
            "finished_at": _now(),
            "details": dict(details),
        }
    )
    _assert_stage_prefix(report["stages"])
    if status == "passed":
        report["last_completed_stage"] = stage.value
    report["updated_at"] = _now()
    atomic_write_json(report_path, report)


def _merge_pick_report(
    report: dict[str, Any], report_path: Path, pick_report: Mapping[str, Any]
) -> None:
    raw_stages = pick_report.get("stages")
    if not isinstance(raw_stages, list):
        raise StageFailure("autonomous-pick report has no stage array")
    for raw in raw_stages:
        if not isinstance(raw, Mapping):
            raise StageFailure("autonomous-pick stage entry is not an object")
        expected = FULL_STAGE_ORDER[len(report["stages"])]
        if raw.get("name") != expected:
            raise StageFailure(
                f"autonomous-pick report expected {expected}, got {raw.get('name')!r}"
            )
        report["stages"].append(dict(raw))
        if raw.get("status") == "passed":
            report["last_completed_stage"] = raw["name"]
        report["updated_at"] = _now()
        _assert_stage_prefix(report["stages"])
        atomic_write_json(report_path, report)


def _run_process(
    command: Sequence[str], *, cwd: Path, log_prefix: Path
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(list(command), cwd=cwd, text=True, capture_output=True, check=False)
    log_prefix.with_suffix(".stdout.log").write_text(completed.stdout, encoding="utf-8")
    log_prefix.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
    return completed


def _run_pose(
    normalized: Mapping[str, Any],
    target: CartesianPose,
    *,
    output_dir: Path,
    artifact_name: str,
) -> dict[str, Any]:
    artifact = output_dir / f"{artifact_name}.json"
    _run_command(
        _pose_command(normalized, target, output=artifact, execute=True),
        cwd=normalized["repo_root"],
        log_prefix=output_dir / artifact_name,
    )
    payload = _read_json(artifact)
    if not pose_result_succeeded(payload, executed=True):
        raise StageFailure(f"{artifact_name} pose execution did not succeed")
    within_tolerance, errors = measured_pose_within_tolerance(
        payload, normalized["pick"]["thresholds"]
    )
    if not within_tolerance:
        raise StageFailure(f"{artifact_name} measured pose missed tolerance: {errors}")
    return {**_artifact_summary(artifact, payload), "measured_pose_errors": errors}


def _run_cartesian(
    normalized: Mapping[str, Any],
    delta: np.ndarray,
    *,
    output_dir: Path,
    artifact_name: str,
) -> dict[str, Any]:
    artifact = output_dir / f"{artifact_name}.json"
    _run_command(
        _cartesian_command(normalized, delta, output=artifact, execute=True),
        cwd=normalized["repo_root"],
        log_prefix=output_dir / artifact_name,
    )
    payload = _read_json(artifact)
    if not cartesian_result_succeeded(payload, executed=True):
        raise StageFailure(f"{artifact_name} Cartesian execution did not succeed")
    measured = _vector3(
        payload.get("measured_translation_base_m"),
        name=f"{artifact_name} measured translation",
    )
    error = float(np.linalg.norm(measured - delta))
    if error > normalized["pick"]["thresholds"].max_measured_position_error_m:
        raise StageFailure(f"{artifact_name} measured translation error is {error:.6f} m")
    return {
        **_artifact_summary(artifact, payload),
        "requested_delta_base_m": delta.tolist(),
        "measured_translation_error_m": error,
    }


def _run_sequence(
    normalized: Mapping[str, Any],
    input_path: Path,
    *,
    output_dir: Path,
    artifact_name: str,
) -> dict[str, Any]:
    artifact = output_dir / f"{artifact_name}.json"
    _run_command(
        _pose_sequence_command(normalized, input_path, output=artifact, execute=True),
        cwd=normalized["repo_root"],
        log_prefix=output_dir / artifact_name,
    )
    payload = _read_json(artifact)
    if not pose_sequence_result_succeeded(payload, executed=True):
        raise StageFailure(f"{artifact_name} pose-sequence execution did not succeed")
    position_error = payload.get("measured_final_position_error_m")
    rotation_error = payload.get("measured_final_rotation_error_rad")
    thresholds = normalized["pick"]["thresholds"]
    if (
        not isinstance(position_error, (int, float))
        or not math.isfinite(float(position_error))
        or float(position_error) > thresholds.max_measured_tcp_position_error_m
        or not isinstance(rotation_error, (int, float))
        or not math.isfinite(float(rotation_error))
        or math.degrees(float(rotation_error))
        > thresholds.max_measured_orientation_error_deg
    ):
        raise StageFailure(
            f"{artifact_name} measured final configured-EE pose missed tolerance"
        )
    return {
        **_artifact_summary(artifact, payload),
        "measured_final_position_error_m": float(position_error),
        "measured_final_rotation_error_rad": float(rotation_error),
    }


def _run_hande(
    normalized: Mapping[str, Any],
    action: str,
    *,
    output_dir: Path,
    artifact_name: str,
) -> tuple[dict[str, Any], Path]:
    result = _run_command(
        _hande_command(normalized["pick"], action),
        cwd=normalized["repo_root"],
        log_prefix=output_dir / artifact_name,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise StageFailure(f"{artifact_name} Hand-E output is not an object")
    artifact = output_dir / f"{artifact_name}.json"
    atomic_write_json(artifact, payload)
    return payload, artifact


def _pick_report_ready(payload: Mapping[str, Any], *, executed: bool) -> tuple[bool, str]:
    stages = payload.get("stages")
    if not isinstance(stages, list):
        return False, "autonomous-pick report has no stages"
    expected_count = len(PickStage) if executed else 3
    if len(stages) != expected_count:
        return False, f"autonomous-pick report has {len(stages)} stages, expected {expected_count}"
    for index, record in enumerate(stages):
        if (
            not isinstance(record, Mapping)
            or record.get("name") != tuple(stage.value for stage in PickStage)[index]
            or record.get("status") != "passed"
        ):
            return False, "autonomous-pick stage prefix is not fully passed"
    expected_status = "succeeded" if executed else "review_ready"
    if payload.get("status") != expected_status:
        return False, f"autonomous-pick status is not {expected_status}"
    if executed and payload.get("object_state") != "held_after_robot_specific_short_lift":
        return False, "autonomous-pick did not verify the object held after lift"
    return True, "accepted"


def _recovery_policy(stage: FullTaskStage) -> dict[str, Any]:
    if stage in {FullTaskStage.TRANSPORT, FullTaskStage.PLACE, FullTaskStage.RELEASE}:
        return {
            "policy": "hold_position_and_stop",
            "object_state": "possibly_held",
            "automatic_motion_attempted": False,
        }
    if stage in {
        FullTaskStage.LID_GRASP,
        FullTaskStage.LID_OPEN,
        FullTaskStage.HOLD,
        FullTaskStage.LID_CLOSE,
        FullTaskStage.LID_RELEASE,
    }:
        return {
            "policy": "hold_position_and_stop",
            "lid_state": "unknown_or_possibly_held",
            "automatic_motion_attempted": False,
        }
    return {
        "policy": "stop_without_additional_hardware_command",
        "automatic_motion_attempted": False,
    }


def _targets_summary(normalized: Mapping[str, Any]) -> dict[str, Any]:
    def pose_dict(pose: CartesianPose) -> dict[str, list[float]]:
        return pose.as_dict()

    return {
        "base_frame": BASE_FRAME,
        "transport_tcp": pose_dict(normalized["transport"]),
        "placement_tcp": pose_dict(normalized["placement"]),
        "place_deltas_base_m": [delta.tolist() for delta in normalized["place_deltas"]],
        "retreat_delta_base_m": normalized["retreat_delta"].tolist(),
        "lid_precontact_tcp": pose_dict(normalized["lid_precontact"]),
        "lid_contact_tcp": pose_dict(normalized["lid_contact"]),
        "lid_approach_delta_base_m": normalized["lid_approach_delta"].tolist(),
        "lid_open_waypoint_count": len(normalized["lid_open"]),
        "lid_close_waypoint_count": len(normalized["lid_close"]),
        "hold_duration_s": normalized["hold_duration_s"],
        "waypoint_artifacts": {
            name: str(path) for name, path in normalized["waypoint_paths"].items()
        },
        "trajectory_audit_thresholds": {
            "min_joint_limit_margin_rad": normalized["pick"]["min_joint_limit_margin_rad"],
            "max_joint_sweep_rad": normalized["pick"]["max_joint_sweep_rad"],
            "max_total_joint_sweep_rad": normalized["pick"]["max_total_joint_sweep_rad"],
        },
    }


def run(args: argparse.Namespace) -> int:
    if args.execute and not args.accept_arm_only_collision_model:
        raise ValueError("--execute requires --accept-arm-only-collision-model")
    if not args.execute and args.accept_arm_only_collision_model:
        raise ValueError("collision-model acknowledgement is only valid with --execute")
    config_path = args.config.expanduser().resolve()
    normalized = _validate_config(
        _read_json(config_path), config_path=config_path, execute=args.execute
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    report_path = output_dir / "stage_report.json"
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "running",
        "mode": "execute" if args.execute else "dry_run_review",
        "authorized_for_robot_execution": bool(args.execute),
        "arm_only_collision_model_acknowledged": bool(args.accept_arm_only_collision_model),
        "config": str(config_path),
        "output_dir": str(output_dir),
        "started_at": _now(),
        "updated_at": _now(),
        "stage_order": list(FULL_STAGE_ORDER),
        "stages": [],
        "resolved_targets": _targets_summary(normalized),
    }
    atomic_write_json(report_path, report)

    pick_dir = output_dir / "pick"
    pick_command = _pick_command(
        normalized,
        output_dir=pick_dir,
        input_dir=args.input_dir,
        execute=args.execute,
    )
    try:
        pick_result = _run_process(
            pick_command,
            cwd=normalized["repo_root"],
            log_prefix=output_dir / "autonomous_pick",
        )
    except Exception as exc:
        report["status"] = "failed"
        report["failed_stage"] = PickStage.OBSERVE.value
        report["failure"] = {
            "error_type": type(exc).__name__,
            "error": f"could not start autonomous-pick subprocess: {exc}",
        }
        report["finished_at"] = _now()
        atomic_write_json(report_path, report)
        return 2
    pick_report_path = pick_dir / "stage_report.json"
    if not pick_report_path.is_file():
        report["status"] = "failed"
        report["failed_stage"] = PickStage.OBSERVE.value
        report["failure"] = {
            "error": "autonomous-pick subprocess produced no stage report",
            "returncode": pick_result.returncode,
        }
        report["finished_at"] = _now()
        atomic_write_json(report_path, report)
        return 2
    try:
        pick_report = _read_json(pick_report_path)
        _merge_pick_report(report, report_path, pick_report)
    except Exception as exc:
        report["status"] = "failed"
        report["failed_stage"] = FULL_STAGE_ORDER[len(report["stages"])]
        report["failure"] = {
            "error_type": type(exc).__name__,
            "error": f"invalid autonomous-pick report: {exc}",
            "returncode": pick_result.returncode,
        }
        report["finished_at"] = _now()
        atomic_write_json(report_path, report)
        return 2
    ready, reason = _pick_report_ready(pick_report, executed=args.execute)
    report["pick_report"] = _artifact_summary(pick_report_path, pick_report)
    if pick_result.returncode != 0 or not ready:
        report["status"] = "failed"
        report["failed_stage"] = pick_report.get("failed_stage")
        report["failure"] = {
            "error": reason,
            "returncode": pick_result.returncode,
            "pick_failure": pick_report.get("failure"),
        }
        if "recovery" in pick_report:
            report["recovery"] = pick_report["recovery"]
        report["finished_at"] = _now()
        atomic_write_json(report_path, report)
        return 2

    if not args.execute:
        report["status"] = "review_ready"
        report["review_scope"] = (
            "OBSERVE/GROUND/PREGRASP reachability plus strict static validation of every "
            "continuation target and waypoint transition; continuation trajectories are "
            "planned from measured state immediately before each physical stage"
        )
        report["not_executed"] = list(FULL_STAGE_ORDER[len(report["stages"]) :])
        report["finished_at"] = _now()
        atomic_write_json(report_path, report)
        print(f"full-task review ready: {report_path}")
        return 0

    active_stage: FullTaskStage | None = None
    active_started = _now()
    try:
        close_payload = _read_json(pick_dir / "hande_close.json")
        close_opening_m = float(close_payload["state"]["opening_width_mm"]) * 1e-3

        active_stage = FullTaskStage.TRANSPORT
        active_started = _now()
        motion = _run_pose(
            normalized,
            normalized["transport"],
            output_dir=output_dir,
            artifact_name="transport_execution",
        )
        held_payload, held_path = _run_hande(
            normalized,
            "status",
            output_dir=output_dir,
            artifact_name="hande_post_transport",
        )
        if not hande_width_snapshot_valid(
            held_payload,
            normalized["pick"]["thresholds"],
            reference_opening_m=close_opening_m,
            max_drift_m=0.002,
        ):
            raise StageFailure("object hold verification failed after TRANSPORT")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "motion": motion,
                "object_hold": _artifact_summary(held_path, held_payload),
            },
            started_at=active_started,
        )

        active_stage = FullTaskStage.PLACE
        active_started = _now()
        place_segments = [
            _run_cartesian(
                normalized,
                delta,
                output_dir=output_dir,
                artifact_name=f"place_segment_{index:02d}",
            )
            for index, delta in enumerate(normalized["place_deltas"], start=1)
        ]
        held_payload, held_path = _run_hande(
            normalized,
            "status",
            output_dir=output_dir,
            artifact_name="hande_pre_release",
        )
        if not hande_width_snapshot_valid(
            held_payload,
            normalized["pick"]["thresholds"],
            reference_opening_m=close_opening_m,
            max_drift_m=0.002,
        ):
            raise StageFailure("object hold verification failed before RELEASE")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "segments": place_segments,
                "object_hold": _artifact_summary(held_path, held_payload),
                "placement_predicate": "commanded TCP endpoint reached while width remained stable",
            },
            started_at=active_started,
        )

        active_stage = FullTaskStage.RELEASE
        active_started = _now()
        release_payload, release_path = _run_hande(
            normalized,
            "open",
            output_dir=output_dir,
            artifact_name="hande_release_object",
        )
        if not hande_open_succeeded(release_payload, normalized["pick"]["thresholds"]):
            raise StageFailure("Hand-E did not fully release the placed object")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=_artifact_summary(release_path, release_payload),
            started_at=active_started,
        )

        active_stage = FullTaskStage.RETREAT
        active_started = _now()
        retreat = _run_cartesian(
            normalized,
            normalized["retreat_delta"],
            output_dir=output_dir,
            artifact_name="placement_retreat",
        )
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=retreat,
            started_at=active_started,
        )

        active_stage = FullTaskStage.LID_APPROACH
        active_started = _now()
        lid_approach = _run_pose(
            normalized,
            normalized["lid_precontact"],
            output_dir=output_dir,
            artifact_name="lid_precontact_execution",
        )
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=lid_approach,
            started_at=active_started,
        )

        active_stage = FullTaskStage.LID_GRASP
        active_started = _now()
        contact_motion = _run_cartesian(
            normalized,
            normalized["lid_approach_delta"],
            output_dir=output_dir,
            artifact_name="lid_contact_execution",
        )
        lid_close_payload, lid_close_path = _run_hande(
            normalized,
            "close",
            output_dir=output_dir,
            artifact_name="hande_grasp_lid",
        )
        if not _lid_contact_verified(
            lid_close_payload,
            min_opening_m=normalized["lid_min_opening_m"],
            max_opening_m=normalized["lid_max_opening_m"],
        ):
            raise StageFailure("Hand-E close did not verify lid contact")
        lid_opening_m = float(lid_close_payload["state"]["opening_width_mm"]) * 1e-3
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "contact_motion": contact_motion,
                "gripper": _artifact_summary(lid_close_path, lid_close_payload),
            },
            started_at=active_started,
        )

        active_stage = FullTaskStage.LID_OPEN
        active_started = _now()
        lid_open = _run_sequence(
            normalized,
            normalized["waypoint_paths"]["lid_open_waypoints"],
            output_dir=output_dir,
            artifact_name="lid_open_execution",
        )
        lid_open_hold, lid_open_hold_path = _run_hande(
            normalized,
            "status",
            output_dir=output_dir,
            artifact_name="hande_post_lid_open",
        )
        if not _lid_width_snapshot_valid(
            lid_open_hold,
            min_opening_m=normalized["lid_min_opening_m"],
            max_opening_m=normalized["lid_max_opening_m"],
            reference_opening_m=lid_opening_m,
        ):
            raise StageFailure("lid hold verification failed after LID_OPEN")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "motion": lid_open,
                "lid_hold": _artifact_summary(lid_open_hold_path, lid_open_hold),
            },
            started_at=active_started,
        )

        active_stage = FullTaskStage.HOLD
        active_started = _now()
        time.sleep(normalized["hold_duration_s"])
        lid_hold, lid_hold_path = _run_hande(
            normalized,
            "status",
            output_dir=output_dir,
            artifact_name="hande_post_hold",
        )
        if not _lid_width_snapshot_valid(
            lid_hold,
            min_opening_m=normalized["lid_min_opening_m"],
            max_opening_m=normalized["lid_max_opening_m"],
            reference_opening_m=lid_opening_m,
        ):
            raise StageFailure("lid hold verification failed during HOLD")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "duration_s": normalized["hold_duration_s"],
                "predicate": "deterministic elapsed hold interval",
                "lid_hold": _artifact_summary(lid_hold_path, lid_hold),
            },
            started_at=active_started,
        )

        active_stage = FullTaskStage.LID_CLOSE
        active_started = _now()
        lid_close = _run_sequence(
            normalized,
            normalized["waypoint_paths"]["lid_close_waypoints"],
            output_dir=output_dir,
            artifact_name="lid_close_execution",
        )
        lid_closed_hold, lid_closed_hold_path = _run_hande(
            normalized,
            "status",
            output_dir=output_dir,
            artifact_name="hande_post_lid_close",
        )
        if not _lid_width_snapshot_valid(
            lid_closed_hold,
            min_opening_m=normalized["lid_min_opening_m"],
            max_opening_m=normalized["lid_max_opening_m"],
            reference_opening_m=lid_opening_m,
        ):
            raise StageFailure("lid hold verification failed after LID_CLOSE")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "motion": lid_close,
                "lid_hold": _artifact_summary(lid_closed_hold_path, lid_closed_hold),
            },
            started_at=active_started,
        )

        active_stage = FullTaskStage.LID_RELEASE
        active_started = _now()
        lid_release_payload, lid_release_path = _run_hande(
            normalized,
            "open",
            output_dir=output_dir,
            artifact_name="hande_release_lid",
        )
        if not hande_open_succeeded(lid_release_payload, normalized["pick"]["thresholds"]):
            raise StageFailure("Hand-E did not fully release the lid")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=_artifact_summary(lid_release_path, lid_release_payload),
            started_at=active_started,
        )

        active_stage = FullTaskStage.FINAL_RETREAT
        active_started = _now()
        final_retreat = _run_cartesian(
            normalized,
            -normalized["lid_approach_delta"],
            output_dir=output_dir,
            artifact_name="lid_final_retreat",
        )
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=final_retreat,
            started_at=active_started,
        )

        report["status"] = "succeeded"
        report["finished_at"] = _now()
        report["object_state"] = "released_at_placement_tcp"
        report["lid_state"] = "closed_and_released"
        atomic_write_json(report_path, report)
        print(f"demo-02 full task succeeded: {report_path}")
        return 0
    except Exception as exc:
        if active_stage is not None and len(report["stages"]) < len(FULL_STAGE_ORDER):
            expected = FULL_STAGE_ORDER[len(report["stages"])]
            if active_stage.value == expected:
                _append_stage(
                    report,
                    report_path,
                    active_stage,
                    status="failed",
                    details={"error_type": type(exc).__name__, "error": str(exc)},
                    started_at=active_started,
                )
        report["status"] = "failed"
        report["failed_stage"] = active_stage.value if active_stage is not None else None
        report["failure"] = {"error_type": type(exc).__name__, "error": str(exc)}
        if active_stage is not None:
            report["recovery"] = _recovery_policy(active_stage)
        report["finished_at"] = _now()
        atomic_write_json(report_path, report)
        print(
            f"demo-02 full task failed at {report['failed_stage']}: {exc}",
            file=sys.stderr,
        )
        return 2


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception:  # pragma: no cover - preserve integration diagnostics
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
