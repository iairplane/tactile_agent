#!/usr/bin/env python3
"""Plan or run one continuous tactile-supervised medicine-cap task.

Planning is the default and never opens a robot, gripper, or GF220 device.  It
validates every input, resolves the cap-centred configured-EE poses, and writes
the exact subprocess command templates for review.  Physical execution needs
both ``--execute`` and ``--accept-arm-only-collision-model`` and is refused on
a kernel whose release name does not contain ``realtime``.

The physical path delegates arm motion to the existing audited MoveIt pose and
Cartesian-sequence programs, finger motion to ``hande_cli.py``, and tactile
acquisition to ``GF220LiveSource`` plus ``gf220_wrench_adapter``.  The twist
policy itself lives in ``medicine_bottle_twist_loop``: collect a pre-press
baseline, incrementally press until axial engagement is stable, then twist and,
when needed, unload/reset/regrasp before establishing the press gate again.
"""

from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from demo2gripper.autonomous_pick import rotation_matrix_to_quaternion_xyzw
from demo2gripper.cartesian_pose_sequence import CartesianPose, POSE_SEQUENCE_SCHEMA
from demo2gripper.franka import validate_rigid_transform
from demo2gripper.gf220_live import GF220LiveConfig, GF220LiveSource, GF220ManualWarpProfile
from demo2gripper.gf220_wrench_adapter import (
    GF220WrenchAdapterConfig,
    GF220WrenchSensorCalibration,
    adapt_gf220_force6d_windows,
)
from demo2gripper.medicine_bottle import (
    BottleCapFrame,
    cartesian_pose_to_transform,
    parse_bottle_cap_frame_payload,
)
from demo2gripper.medicine_bottle_twist_loop import (
    MedicineBottleTwistLoopConfig,
    TactileForceObservation,
    TwistLoopContext,
    TwistLoopPhase,
    run_medicine_bottle_twist_loop,
)
from demo2gripper.medicine_bottle_tactile_bootstrap import (
    TactileBootstrapQualityError,
    bootstrap_medicine_bottle_tactile_presence,
)
from demo2gripper.medicine_bottle_tactile_presence import (
    TactilePresenceSupervisorEvaluator,
    release_resistance_limit_sensor_units,
)
from demo2gripper.pose_motion import normalize_quaternion_xyzw
from demo2gripper.tactile_twist_supervisor import TactileTwistConfig

CONFIG_SCHEMA = "demo2gripper.medicine_bottle_task_config.v1"
REPORT_SCHEMA = "demo2gripper.medicine_bottle_task_report.v1"
ACTION_LOG_SCHEMA = "demo2gripper.medicine_bottle_action_log.v1"
PLAN_SCHEMA = "demo2gripper.medicine_bottle_resolved_plan.v1"
WRENCH_PROFILE_SCHEMA = "demo2gripper.gf220_wrench_transform_profile.v1"
LIVE_CALIBRATION_SOURCE = "GF220LiveSource.calibrate_unloaded"
CONTACT_WAYPOINT_MAX_SEGMENT_M = 0.025
BOOTSTRAP_ATTEMPTS_SCHEMA = (
    "demo2gripper.medicine_bottle_tactile_bootstrap_attempts.v1"
)
BOOTSTRAP_RETRYABLE_QUALITY_REASONS = frozenset(
    {
        "axial_press_tactile_response_absent",
    }
)
MAX_BOOTSTRAP_RETRY_OPENING_DRIFT_MM = 2.0

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "repo_root",
        "inputs",
        "commands",
        "grasp",
        "press",
        "twist",
        "wrist",
        "regrasp",
        "release_verify",
        "hande",
        "gf220",
        "motion",
    }
)


class TaskFailure(RuntimeError):
    """A delegated command or closed-loop predicate failed."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and write the offline review plan (the default)",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="run the complete physical sequence without interactive stage pauses",
    )
    parser.add_argument(
        "--accept-arm-only-collision-model",
        action="store_true",
        help="acknowledge that the installed Hand-E is absent from the MoveIt model",
    )
    return parser.parse_args(argv)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_fields(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], *, name: str
) -> None:
    missing = sorted(set(expected) - set(value))
    unknown = sorted(set(value) - set(expected))
    if missing:
        raise ValueError(f"{name} is missing fields: {missing}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {unknown}")


def _path(value: object, *, base: Path, name: str, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    result = Path(value).expanduser()
    result = (base / result).resolve() if not result.is_absolute() else result.resolve()
    if must_exist and not result.exists():
        raise ValueError(f"{name} does not exist: {result}")
    return result


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    include_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None:
        invalid = result < minimum if include_minimum else result <= minimum
        if invalid:
            relation = ">=" if include_minimum else ">"
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _integer(value: object, *, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f"..{maximum}" if maximum is not None else f" >= {minimum}"
        raise ValueError(f"{name} must be in {minimum}{suffix}")
    return value


def _vector(value: object, *, length: int, name: str) -> list[float]:
    array = np.asarray(value, dtype=object)
    if array.shape != (length,) or any(isinstance(item, (bool, np.bool_)) for item in array):
        raise ValueError(f"{name} must contain {length} finite numbers")
    try:
        converted = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} finite numbers") from exc
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must contain {length} finite numbers")
    return [float(item) for item in converted]


def _command(value: object, *, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty string array")
    return list(value)


def _rotation_from_quaternion(values: object, *, name: str) -> tuple[np.ndarray, tuple[float, ...]]:
    quaternion = normalize_quaternion_xyzw(_vector(values, length=4, name=name))
    transform = cartesian_pose_to_transform(CartesianPose((0.0, 0.0, 0.0), quaternion))
    return transform[:3, :3], quaternion


def _load_transform(path: Path, *, name: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        payload = _read_json(path)
        if "matrix" in payload:
            value = payload["matrix"]
        elif "T_link8_configured_ee" in payload:
            value = payload["T_link8_configured_ee"]
        else:
            raise ValueError(f"{name} JSON must contain matrix or T_link8_configured_ee")
    return validate_rigid_transform(np.asarray(value, dtype=float), name=name)


def _parse_dataclass_config(
    cls: type[Any], value: object, *, name: str, excluded: set[str] | None = None
) -> Any:
    mapping = _mapping(value, name=name)
    excluded = excluded or set()
    expected = {field.name for field in fields(cls)} - excluded
    _exact_fields(mapping, expected, name=name)
    return cls(**{key: mapping[key] for key in expected})


def _load_wrench_profile(
    path: Path,
    *,
    calibration_ids: Mapping[str, str] | None = None,
) -> tuple[GF220WrenchAdapterConfig, dict[str, Any]]:
    payload = _read_json(path)
    expected = {
        "schema",
        "status",
        "profile_id",
        "task_frame_id",
        "calibration_id_source",
        "bottle_axis_task",
        "axis_point_task_m",
        "min_samples",
        "max_age_s",
        "max_pair_skew_s",
        "future_tolerance_s",
        "sensors",
    }
    _exact_fields(payload, expected, name="GF220 wrench transform profile")
    if payload["schema"] != WRENCH_PROFILE_SCHEMA:
        raise ValueError(f"GF220 wrench profile schema must be {WRENCH_PROFILE_SCHEMA!r}")
    if payload["status"] not in {"draft", "accepted"}:
        raise ValueError("GF220 wrench profile status must be draft or accepted")
    if payload["calibration_id_source"] != LIVE_CALIBRATION_SOURCE:
        raise ValueError(f"calibration_id_source must be {LIVE_CALIBRATION_SOURCE!r}")
    raw_sensors = payload["sensors"]
    if not isinstance(raw_sensors, list) or len(raw_sensors) != 2:
        raise ValueError("GF220 wrench profile must contain exactly two sensors")
    sensor_expected = {
        "serial",
        "logical_side",
        "R_task_sensor",
        "contact_position_task_m",
        "force6d_bias",
        "provenance",
    }
    sensors: list[GF220WrenchSensorCalibration] = []
    for index, raw in enumerate(raw_sensors):
        sensor = _mapping(raw, name=f"GF220 wrench sensors[{index}]")
        _exact_fields(sensor, sensor_expected, name=f"GF220 wrench sensors[{index}]")
        serial = sensor["serial"]
        if not isinstance(serial, str) or not serial:
            raise ValueError(f"GF220 wrench sensors[{index}].serial must be non-empty")
        calibration_id = (
            calibration_ids[serial]
            if calibration_ids is not None
            else f"plan-only-live-calibration:{serial}"
        )
        sensors.append(
            GF220WrenchSensorCalibration(
                serial=serial,
                logical_side=sensor["logical_side"],
                calibration_id=calibration_id,
                R_task_sensor=np.asarray(sensor["R_task_sensor"], dtype=float),
                contact_position_task_m=np.asarray(sensor["contact_position_task_m"], dtype=float),
                force6d_bias=np.asarray(sensor["force6d_bias"], dtype=float),
                provenance=sensor["provenance"],
            )
        )
    adapter = GF220WrenchAdapterConfig(
        profile_id=payload["profile_id"],
        task_frame_id=payload["task_frame_id"],
        sensors=(sensors[0], sensors[1]),
        min_samples=payload["min_samples"],
        max_age_s=payload["max_age_s"],
        max_pair_skew_s=payload["max_pair_skew_s"],
        future_tolerance_s=payload["future_tolerance_s"],
    )
    axis = np.asarray(_vector(payload["bottle_axis_task"], length=3, name="bottle_axis_task"))
    norm = float(np.linalg.norm(axis))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("GF220 bottle_axis_task must be unit length")
    point = _vector(payload["axis_point_task_m"], length=3, name="axis_point_task_m")
    return adapter, {
        "status": payload["status"],
        "bottle_axis_task": axis.tolist(),
        "axis_point_task_m": point,
    }


def _validate_config(
    payload: Mapping[str, Any], *, config_path: Path, execute: bool
) -> dict[str, Any]:
    _exact_fields(payload, TOP_LEVEL_FIELDS, name="config")
    if payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"config schema must be {CONFIG_SCHEMA!r}")
    base = config_path.parent.resolve()
    repo_root = _path(payload["repo_root"], base=base, name="repo_root")
    if not repo_root.is_dir():
        raise ValueError("repo_root must be a directory")

    inputs = _mapping(payload["inputs"], name="inputs")
    _exact_fields(inputs, {"cap_frame", "T_link8_configured_ee"}, name="inputs")
    cap_frame_path = _path(inputs["cap_frame"], base=base, name="inputs.cap_frame")
    tool_path = _path(
        inputs["T_link8_configured_ee"], base=base, name="inputs.T_link8_configured_ee"
    )
    source_frame = parse_bottle_cap_frame_payload(_read_json(cap_frame_path))
    T_link8_ee = _load_transform(tool_path, name="T_link8_configured_ee")

    commands = _mapping(payload["commands"], name="commands")
    _exact_fields(commands, {"moveit_pose", "moveit_sequence", "hande"}, name="commands")
    parsed_commands = {name: _command(commands[name], name=f"commands.{name}") for name in commands}

    grasp = _mapping(payload["grasp"], name="grasp")
    _exact_fields(
        grasp,
        {
            "orientation_xyzw",
            "bottle_axis_tcp",
            "approach_standoff_m",
            "fixture_verified",
            "orientation_reviewed",
        },
        name="grasp",
    )
    rotation, grasp_quaternion = _rotation_from_quaternion(
        grasp["orientation_xyzw"], name="grasp.orientation_xyzw"
    )
    bottle_axis_tcp = _vector(grasp["bottle_axis_tcp"], length=3, name="grasp.bottle_axis_tcp")
    cap_transform = np.eye(4)
    cap_transform[:3, :3] = rotation
    cap_transform[:3, 3] = source_frame.T_base_cap_tcp[:3, 3]
    frame = BottleCapFrame(
        base_frame=source_frame.base_frame,
        cap_frame=source_frame.cap_frame,
        tcp_frame=source_frame.tcp_frame,
        T_base_cap_tcp=cap_transform,
        bottle_axis_base=source_frame.bottle_axis_base,
        bottle_axis_tcp=tuple(bottle_axis_tcp),
    )
    approach_standoff_m = _number(
        grasp["approach_standoff_m"],
        name="grasp.approach_standoff_m",
        minimum=0.0,
        maximum=0.20,
    )
    for flag in ("fixture_verified", "orientation_reviewed"):
        if not isinstance(grasp[flag], bool):
            raise ValueError(f"grasp.{flag} must be boolean")

    press = _mapping(payload["press"], name="press")
    _exact_fields(
        press,
        {
            "initial_distance_m",
            "retry_increment_m",
            "max_retries",
            "max_confirm_windows",
        },
        name="press",
    )
    initial_press_m = _number(
        press["initial_distance_m"], name="press.initial_distance_m", minimum=0.0, maximum=0.02
    )
    retry_press_m = _number(
        press["retry_increment_m"], name="press.retry_increment_m", minimum=0.0, maximum=0.005
    )
    max_press_retries = _integer(
        press["max_retries"], name="press.max_retries", minimum=0, maximum=3
    )
    max_press_confirm_windows = _integer(
        press["max_confirm_windows"],
        name="press.max_confirm_windows",
        minimum=1,
        maximum=10,
    )

    twist = _mapping(payload["twist"], name="twist")
    _exact_fields(
        twist,
        {
            "step_angle_deg",
            "max_steps",
            "batch_steps",
            "thread_pitch_m_per_revolution",
            "downfeed_m_per_revolution",
            "downfeed_revolutions",
            "minimum_release_revolutions",
            "max_axial_upfeed_m",
            "velocity_scale",
            "acceleration_scale",
            "max_cartesian_speed_m_s",
            "tactile",
        },
        name="twist",
    )
    twist_degrees = _number(
        twist["step_angle_deg"],
        name="twist.step_angle_deg",
        minimum=-10.000001,
        maximum=10.0,
        include_minimum=True,
    )
    if abs(twist_degrees) < 5.0:
        raise ValueError("absolute twist.step_angle_deg must lie in [5, 10]")
    twist_angle_rad = math.radians(twist_degrees)
    max_twist_steps = _integer(
        twist["max_steps"], name="twist.max_steps", minimum=1, maximum=360
    )
    twist_batch_steps = _integer(
        twist["batch_steps"], name="twist.batch_steps", minimum=1, maximum=12
    )
    thread_pitch = _number(
        twist["thread_pitch_m_per_revolution"],
        name="twist.thread_pitch_m_per_revolution",
        minimum=0.0,
        maximum=0.01,
        include_minimum=True,
    )
    twist_downfeed = _number(
        twist["downfeed_m_per_revolution"],
        name="twist.downfeed_m_per_revolution",
        minimum=0.0,
        maximum=0.01,
        include_minimum=True,
    )
    twist_downfeed_revolutions = _number(
        twist["downfeed_revolutions"],
        name="twist.downfeed_revolutions",
        minimum=0.0,
        maximum=2.0,
        include_minimum=True,
    )
    release_minimum_revolutions = _number(
        twist["minimum_release_revolutions"],
        name="twist.minimum_release_revolutions",
        minimum=0.0,
        maximum=5.0,
    )
    max_axial_upfeed_m = _number(
        twist["max_axial_upfeed_m"],
        name="twist.max_axial_upfeed_m",
        minimum=0.0,
        maximum=0.008,
    )
    maximum_planned_twist_rad = max_twist_steps * abs(twist_angle_rad)
    maximum_planned_upfeed_m = thread_pitch * max(
        0.0,
        maximum_planned_twist_rad / (2.0 * math.pi)
        - twist_downfeed_revolutions,
    )
    if maximum_planned_upfeed_m > max_axial_upfeed_m + 1e-12:
        raise ValueError(
            "twist max_steps and thread pitch exceed twist.max_axial_upfeed_m"
        )
    if release_minimum_revolutions * 2.0 * math.pi >= maximum_planned_twist_rad:
        raise ValueError(
            "twist.minimum_release_revolutions must be below the maximum twist budget"
        )
    twist_motion = {
        "velocity_scale": _number(
            twist["velocity_scale"], name="twist.velocity_scale", minimum=0.0
        ),
        "acceleration_scale": _number(
            twist["acceleration_scale"],
            name="twist.acceleration_scale",
            minimum=0.0,
        ),
        "max_cartesian_speed_m_s": _number(
            twist["max_cartesian_speed_m_s"],
            name="twist.max_cartesian_speed_m_s",
            minimum=0.0,
        ),
    }
    if (
        twist_motion["velocity_scale"] > 0.08
        or twist_motion["acceleration_scale"] > 0.08
    ):
        raise ValueError("twist velocity and acceleration scales are capped at 0.08")
    if twist_motion["max_cartesian_speed_m_s"] > 0.025:
        raise ValueError("twist.max_cartesian_speed_m_s is capped at 0.025 m/s")
    tactile_config = _parse_dataclass_config(
        TactileTwistConfig, twist["tactile"], name="twist.tactile"
    )

    wrist = _mapping(payload["wrist"], name="wrist")
    _exact_fields(wrist, {"travel_limit_deg"}, name="wrist")
    wrist_limit_rad = math.radians(
        _number(
            wrist["travel_limit_deg"], name="wrist.travel_limit_deg", minimum=5.0, maximum=180.0
        )
    )

    regrasp = _mapping(payload["regrasp"], name="regrasp")
    _exact_fields(
        regrasp,
        {"max_cycles", "unload_opening_mm", "grasp_opening_mm", "max_unload_windows"},
        name="regrasp",
    )
    max_regrasps = _integer(
        regrasp["max_cycles"], name="regrasp.max_cycles", minimum=0, maximum=64
    )
    max_unload_windows = _integer(
        regrasp["max_unload_windows"], name="regrasp.max_unload_windows", minimum=1, maximum=10
    )

    release = _mapping(payload["release_verify"], name="release_verify")
    _exact_fields(
        release,
        {"axial_test_lift_m", "max_axis_force_delta_n", "max_tangential_delta_n"},
        name="release_verify",
    )

    hande = _mapping(payload["hande"], name="hande")
    _exact_fields(
        hande,
        {
            "port",
            "open_width_mm",
            "grasp_width_mm",
            "speed_register",
            "force_register",
            "timeout_s",
        },
        name="hande",
    )
    if not isinstance(hande["port"], str) or not Path(hande["port"]).is_absolute():
        raise ValueError("hande.port must be an absolute device path")

    gf220 = _mapping(payload["gf220"], name="gf220")
    _exact_fields(
        gf220,
        {
            "manual_warp_profile",
            "wrench_transform_profile",
            "live",
            "startup_timeout_s",
            "calibration_timeout_s",
            "window_timeout_s",
        },
        name="gf220",
    )
    manual_warp_path = _path(
        gf220["manual_warp_profile"], base=base, name="gf220.manual_warp_profile"
    )
    wrench_profile_path = _path(
        gf220["wrench_transform_profile"], base=base, name="gf220.wrench_transform_profile"
    )
    manual_warp = GF220ManualWarpProfile.read(manual_warp_path)
    live_config = _parse_dataclass_config(GF220LiveConfig, gf220["live"], name="gf220.live")
    adapter_config, wrench_meta = _load_wrench_profile(wrench_profile_path)
    if {sensor.serial for sensor in adapter_config.sensors} != {
        sensor.serial for sensor in manual_warp.sensors
    }:
        raise ValueError("GF220 manual-warp and wrench-transform profiles name different serials")
    if tactile_config.profile_id != adapter_config.profile_id:
        raise ValueError("twist.tactile.profile_id must match GF220 wrench profile_id")
    timing_values = {
        "max_age_s": (
            adapter_config.max_age_s,
            tactile_config.max_age_s,
            live_config.max_frame_age_s,
        ),
        "max_pair_skew_s": (
            adapter_config.max_pair_skew_s,
            tactile_config.max_pair_skew_s,
            live_config.max_pair_skew_s,
        ),
    }
    for name, values in timing_values.items():
        if not all(math.isclose(values[0], value, abs_tol=1e-12) for value in values[1:]):
            raise ValueError(
                f"GF220 seed, tactile, and live {name} must match; got {values}"
            )

    motion = _mapping(payload["motion"], name="motion")
    motion_fields = {
        "lift_distance_m",
        "velocity_scale",
        "acceleration_scale",
        "max_cartesian_speed_m_s",
        "max_step_m",
        "jump_threshold",
        "revolute_jump_threshold_rad",
        "planning_time_s",
        "planning_attempts",
        "server_timeout_s",
        "max_pose_translation_m",
        "max_pose_rotation_rad",
        "position_tolerance_m",
        "orientation_tolerance_rad",
        "min_joint_limit_margin_rad",
        "max_joint_sweep_rad",
        "max_total_joint_sweep_rad",
    }
    _exact_fields(motion, motion_fields, name="motion")

    loop_kwargs: dict[str, Any] = dict(
        twist_step_angle_rad=twist_angle_rad,
        wrist_travel_limit_rad=wrist_limit_rad,
        max_twist_steps=max_twist_steps,
        max_regrasp_cycles=max_regrasps,
        max_press_retries=max_press_retries,
        max_unload_windows=max_unload_windows,
        twist_batch_steps=twist_batch_steps,
    )
    if "max_press_confirm_windows" in {
        field.name for field in fields(MedicineBottleTwistLoopConfig)
    }:
        loop_kwargs["max_press_confirm_windows"] = max_press_confirm_windows
    loop_config = MedicineBottleTwistLoopConfig(**loop_kwargs)
    if execute:
        if not grasp["fixture_verified"]:
            raise ValueError("execution requires grasp.fixture_verified=true")
        if not grasp["orientation_reviewed"]:
            raise ValueError("execution requires grasp.orientation_reviewed=true")
        # Rotation and FORCE6D bias from this file are seed placeholders only.
        # Every physical run must qualify fresh unloaded/grasp/press windows
        # before the first formal twist command.

    normalized = {
        "repo_root": repo_root,
        "cap_frame_path": cap_frame_path,
        "tool_path": tool_path,
        "frame": frame,
        "grasp_quaternion": grasp_quaternion,
        "approach_standoff_m": approach_standoff_m,
        "fixture_verified": grasp["fixture_verified"],
        "orientation_reviewed": grasp["orientation_reviewed"],
        "initial_press_m": initial_press_m,
        "retry_press_m": retry_press_m,
        "thread_pitch_m_per_revolution": thread_pitch,
        "twist_downfeed_m_per_revolution": twist_downfeed,
        "twist_downfeed_revolutions": twist_downfeed_revolutions,
        "release_min_total_twist_rad": (
            release_minimum_revolutions * 2.0 * math.pi
        ),
        "max_axial_upfeed_m": max_axial_upfeed_m,
        "twist_motion": twist_motion,
        "loop_config": loop_config,
        "tactile_config": tactile_config,
        "commands": parsed_commands,
        "T_link8_configured_ee": T_link8_ee,
        "regrasp_unload_opening_mm": _number(
            regrasp["unload_opening_mm"],
            name="regrasp.unload_opening_mm",
            minimum=0.0,
            maximum=50.0,
            include_minimum=True,
        ),
        "regrasp_grasp_opening_mm": _number(
            regrasp["grasp_opening_mm"],
            name="regrasp.grasp_opening_mm",
            minimum=0.0,
            maximum=50.0,
            include_minimum=True,
        ),
        "release_test_lift_m": _number(
            release["axial_test_lift_m"],
            name="release_verify.axial_test_lift_m",
            minimum=0.0,
            maximum=0.01,
        ),
        "release_axis_delta_n": _number(
            release["max_axis_force_delta_n"],
            name="release_verify.max_axis_force_delta_n",
            minimum=0.0,
        ),
        "release_tangent_delta_n": _number(
            release["max_tangential_delta_n"],
            name="release_verify.max_tangential_delta_n",
            minimum=0.0,
        ),
        "hande": {
            "port": hande["port"],
            "open_width_mm": _number(
                hande["open_width_mm"],
                name="hande.open_width_mm",
                minimum=0.0,
                maximum=50.0,
                include_minimum=True,
            ),
            "grasp_width_mm": _number(
                hande["grasp_width_mm"],
                name="hande.grasp_width_mm",
                minimum=0.0,
                maximum=50.0,
                include_minimum=True,
            ),
            "speed_register": _integer(
                hande["speed_register"], name="hande.speed_register", minimum=0, maximum=255
            ),
            "force_register": _integer(
                hande["force_register"], name="hande.force_register", minimum=0, maximum=255
            ),
            "timeout_s": _number(hande["timeout_s"], name="hande.timeout_s", minimum=0.0),
        },
        "manual_warp_path": manual_warp_path,
        "wrench_profile_path": wrench_profile_path,
        "gf220_live_config": live_config,
        "adapter_plan_config": adapter_config,
        "wrench_meta": wrench_meta,
        "gf220_startup_timeout_s": _number(
            gf220["startup_timeout_s"], name="gf220.startup_timeout_s", minimum=0.0
        ),
        "gf220_calibration_timeout_s": _number(
            gf220["calibration_timeout_s"], name="gf220.calibration_timeout_s", minimum=0.0
        ),
        "gf220_window_timeout_s": _number(
            gf220["window_timeout_s"], name="gf220.window_timeout_s", minimum=0.0
        ),
        "motion": {
            key: (
                _integer(motion[key], name=f"motion.{key}", minimum=1, maximum=20)
                if key == "planning_attempts"
                else _number(motion[key], name=f"motion.{key}", minimum=0.0)
            )
            for key in motion_fields
        },
    }
    if (
        normalized["motion"]["velocity_scale"] > 0.05
        or normalized["motion"]["acceleration_scale"] > 0.05
    ):
        raise ValueError("motion velocity and acceleration scales are capped at 0.05")
    if normalized["motion"]["max_cartesian_speed_m_s"] > 0.015:
        raise ValueError("motion.max_cartesian_speed_m_s is capped at 0.015 m/s")
    return normalized


def _pose(transform: np.ndarray) -> CartesianPose:
    checked = validate_rigid_transform(transform, name="configured-EE target")
    return CartesianPose(
        tuple(float(value) for value in checked[:3, 3]),
        tuple(float(value) for value in rotation_matrix_to_quaternion_xyzw(checked[:3, :3])),
    )


def _axis_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle_rad) * skew + (1.0 - math.cos(angle_rad)) * (skew @ skew)


def _sequence_payload(base_frame: str, transforms: Sequence[np.ndarray]) -> dict[str, Any]:
    return {
        "schema": POSE_SEQUENCE_SCHEMA,
        "base_frame": base_frame,
        "waypoints": [_pose(transform).as_dict() for transform in transforms],
    }


def _translation_waypoints(
    start: np.ndarray,
    end: np.ndarray,
    *,
    max_segment_m: float = CONTACT_WAYPOINT_MAX_SEGMENT_M,
) -> list[np.ndarray]:
    """Interpolate a fixed-orientation configured-EE translation, excluding ``start``."""

    start_checked = validate_rigid_transform(start, name="translation start")
    end_checked = validate_rigid_transform(end, name="translation end")
    if not math.isfinite(max_segment_m) or not 0.0 < max_segment_m <= 0.05:
        raise ValueError("max_segment_m must be finite and lie in (0, 0.05]")
    if not np.allclose(start_checked[:3, :3], end_checked[:3, :3], atol=1e-9, rtol=0.0):
        raise ValueError("translation interpolation requires a fixed orientation")

    delta = end_checked[:3, 3] - start_checked[:3, 3]
    distance_m = float(np.linalg.norm(delta))
    segment_count = max(1, int(math.ceil(distance_m / max_segment_m - 1e-12)))
    waypoints: list[np.ndarray] = []
    for index in range(1, segment_count + 1):
        waypoint = np.array(start_checked, copy=True)
        waypoint[:3, 3] += (index / segment_count) * delta
        waypoints.append(waypoint)
    return waypoints


def _tool_options(config: Mapping[str, Any]) -> list[str]:
    transform = config["T_link8_configured_ee"]
    quaternion = rotation_matrix_to_quaternion_xyzw(transform[:3, :3])
    return [
        "--link8-to-configured-ee-position-m",
        *(str(value) for value in transform[:3, 3]),
        "--link8-to-configured-ee-orientation-xyzw",
        *(str(value) for value in quaternion),
    ]


def _audit_options(config: Mapping[str, Any]) -> list[str]:
    motion = config["motion"]
    return [
        "--min-joint-limit-margin-rad",
        str(motion["min_joint_limit_margin_rad"]),
        "--max-joint-sweep-rad",
        str(motion["max_joint_sweep_rad"]),
        "--max-total-joint-sweep-rad",
        str(motion["max_total_joint_sweep_rad"]),
    ]


def _pose_command(
    config: Mapping[str, Any], target: np.ndarray, output: Path, *, execute: bool
) -> list[str]:
    pose = _pose(target)
    motion = config["motion"]
    command = [
        *config["commands"]["moveit_pose"],
        "--target-position-m",
        *(str(value) for value in pose.position_m),
        "--target-orientation-xyzw",
        *(str(value) for value in pose.orientation_xyzw),
        "--target-frame",
        "configured-ee",
        *_tool_options(config),
        "--base-frame",
        config["frame"].base_frame,
        "--planning-time-s",
        str(motion["planning_time_s"]),
        "--planning-attempts",
        str(motion["planning_attempts"]),
        "--server-timeout-s",
        str(motion["server_timeout_s"]),
        "--velocity-scale",
        str(motion["velocity_scale"]),
        "--acceleration-scale",
        str(motion["acceleration_scale"]),
        "--max-translation-m",
        str(motion["max_pose_translation_m"]),
        "--max-rotation-rad",
        str(motion["max_pose_rotation_rad"]),
        "--position-tolerance-m",
        str(motion["position_tolerance_m"]),
        "--orientation-tolerance-rad",
        str(motion["orientation_tolerance_rad"]),
        *_audit_options(config),
        "--output",
        str(output),
    ]
    if execute:
        command.extend(("--execute", "--accept-arm-only-collision-model"))
    return command


def _sequence_command(
    config: Mapping[str, Any],
    input_path: Path,
    output: Path,
    *,
    execute: bool,
    motion_profile: Mapping[str, float] | None = None,
) -> list[str]:
    motion = config["motion"]
    speed = motion if motion_profile is None else motion_profile
    command = [
        *config["commands"]["moveit_sequence"],
        "--input",
        str(input_path),
        "--output",
        str(output),
        *_tool_options(config),
        "--base-frame",
        config["frame"].base_frame,
        "--server-timeout-s",
        str(motion["server_timeout_s"]),
        "--max-step-m",
        str(motion["max_step_m"]),
        "--jump-threshold",
        str(motion["jump_threshold"]),
        "--revolute-jump-threshold-rad",
        str(motion["revolute_jump_threshold_rad"]),
        "--velocity-scale",
        str(speed["velocity_scale"]),
        "--acceleration-scale",
        str(speed["acceleration_scale"]),
        "--max-cartesian-speed-m-s",
        str(speed["max_cartesian_speed_m_s"]),
        *_audit_options(config),
    ]
    if execute:
        command.extend(("--execute", "--accept-arm-only-collision-model"))
    return command


def _hande_command(config: Mapping[str, Any], opening_mm: float) -> list[str]:
    hande = config["hande"]
    return [
        *config["commands"]["hande"],
        "move",
        "--port",
        hande["port"],
        "--opening-mm",
        str(opening_mm),
        "--speed",
        str(hande["speed_register"]),
        "--force",
        str(hande["force_register"]),
        "--timeout-s",
        str(hande["timeout_s"]),
    ]


def _hande_status_command(config: Mapping[str, Any]) -> list[str]:
    return [
        *config["commands"]["hande"],
        "status",
        "--port",
        config["hande"]["port"],
    ]


def _require_hande_grasped(payload: object, *, operation: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        raise TaskFailure(f"{operation}: Hand-E response did not report ok=true")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise TaskFailure(f"{operation}: Hand-E response has no state object")
    object_status = state.get("object_status")
    if (
        not isinstance(object_status, Mapping)
        or object_status.get("code") != 2
        or object_status.get("grasped") is not True
    ):
        raise TaskFailure(f"{operation}: Hand-E did not confirm cap contact (gOBJ=2)")
    if state.get("fault_code") != 0:
        raise TaskFailure(f"{operation}: Hand-E reported a non-zero fault")
    return state


def _require_hande_motion_ok(payload: object, *, operation: str) -> Mapping[str, Any]:
    """Require a completed, fault-free Hand-E command.

    Object contact is intentionally decided from the GF220 tactile window.  A
    Hand-E ``gOBJ`` code is useful telemetry, but it is not an additional gate
    for this presence-only task mode.
    """

    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        raise TaskFailure(f"{operation}: Hand-E response did not report ok=true")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise TaskFailure(f"{operation}: Hand-E response has no state object")
    if state.get("fault_code") != 0:
        raise TaskFailure(f"{operation}: Hand-E reported a non-zero fault")
    return state


def _require_hande_hold_retained(
    payload: object,
    *,
    operation: str,
    commanded_grasp_opening_mm: float,
    full_opening_mm: float,
    initial_opening_mm: float | None = None,
    max_drift_mm: float = MAX_BOOTSTRAP_RETRY_OPENING_DRIFT_MM,
) -> Mapping[str, Any]:
    """Verify retained cap contact after transient ``gOBJ`` has cleared.

    The standalone Hand-E client clears ``gGTO`` when it disconnects, so a
    later read-only status reports ``gOBJ=0`` even while the fingers remain on
    the cap.  Prefer the persistent measured opening from the successful
    initial grasp; use a conservative cap-width band only when that value is
    unavailable.  The caller also has the immediately preceding bilateral
    tactile window as independent contact evidence.
    """

    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        raise TaskFailure(f"{operation}: Hand-E status did not report ok=true")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise TaskFailure(f"{operation}: Hand-E status has no state object")
    if state.get("activated") is not True:
        raise TaskFailure(f"{operation}: Hand-E is not activated")
    if state.get("fault_code") != 0:
        raise TaskFailure(f"{operation}: Hand-E reported a non-zero fault")
    opening = state.get("opening_width_mm")
    if (
        isinstance(opening, bool)
        or not isinstance(opening, (int, float))
        or not math.isfinite(float(opening))
    ):
        raise TaskFailure(f"{operation}: Hand-E opening is not finite")
    opening_mm = float(opening)
    if initial_opening_mm is not None:
        initial = float(initial_opening_mm)
        if not math.isfinite(initial):
            raise TaskFailure(f"{operation}: initial grasp opening is not finite")
        drift = abs(opening_mm - initial)
        if drift <= max_drift_mm:
            return state
        raise TaskFailure(
            f"{operation}: retained opening drift {drift:.3f} mm exceeds "
            f"{max_drift_mm:.3f} mm"
        )

    lower = float(commanded_grasp_opening_mm) + 5.0
    upper = float(full_opening_mm) - 2.0
    if not lower <= opening_mm <= upper:
        raise TaskFailure(
            f"{operation}: measured opening {opening_mm:.3f} mm does not retain "
            f"the cap in [{lower:.3f}, {upper:.3f}] mm"
        )
    return state


def _bootstrap_window_arrays(
    stages: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for stage, pair in stages.items():
        for index, (serial, window) in enumerate(
            zip(pair.serials, pair.windows, strict=True)
        ):
            force = window.force6d
            if force is None:
                raise TaskFailure(f"{stage}/{serial}: FORCE6D window is missing")
            prefix = f"{stage}_{index}"
            arrays[f"{prefix}_serial_utf8"] = np.frombuffer(
                serial.encode("utf-8"), dtype=np.uint8
            )
            arrays[f"{prefix}_wrench"] = np.asarray(force.values, dtype=float)
            arrays[f"{prefix}_sdk_time_s"] = np.asarray(force.sdk_time_s, dtype=float)
            arrays[f"{prefix}_capture_monotonic_s"] = np.asarray(
                force.capture_monotonic_s, dtype=float
            )
            arrays[f"{prefix}_valid"] = np.asarray(force.valid, dtype=bool)
    return arrays


class AtomicActionLog:
    def __init__(self, path: Path, *, mode: str) -> None:
        self.path = path
        self.payload: dict[str, Any] = {
            "schema": ACTION_LOG_SCHEMA,
            "mode": mode,
            "started_at": _now(),
            "events": [],
        }
        self._write()

    def append(self, operation: str, state: str, **details: Any) -> int:
        index = len(self.payload["events"])
        self.payload["events"].append(
            {
                "index": index,
                "operation": operation,
                "state": state,
                "wall_time": _now(),
                "monotonic_s": time.monotonic(),
                **details,
            }
        )
        self._write()
        return index

    def _write(self) -> None:
        _atomic_json(self.path, self.payload)


CommandExecutor = Callable[[Sequence[str], Path, Path], subprocess.CompletedProcess[str]]


def _run_process(
    command: Sequence[str], cwd: Path, log_prefix: Path
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(list(command), cwd=cwd, text=True, capture_output=True, check=False)
    log_prefix.with_suffix(".stdout.log").write_text(completed.stdout, encoding="utf-8")
    log_prefix.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise TaskFailure(f"command failed with exit {completed.returncode}: {list(command)!r}")
    return completed


class SubprocessMedicineBottleIO:
    """Bind the seven loop operations to audited subprocesses and live GF220 data."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        output_dir: Path,
        action_log: AtomicActionLog,
        tactile_source: GF220LiveSource,
        adapter_config: GF220WrenchAdapterConfig,
        initial_grasp_opening_mm: float | None = None,
        command_executor: CommandExecutor = _run_process,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.action_log = action_log
        self.source = tactile_source
        self.adapter = adapter_config
        self.initial_grasp_opening_mm = initial_grasp_opening_mm
        self.command_executor = command_executor
        self.current = np.array(config["frame"].T_base_cap_tcp, copy=True)
        self.axis_base = np.asarray(config["frame"].bottle_axis_base, dtype=float)
        self.axis_task = np.asarray(config["wrench_meta"]["bottle_axis_task"], dtype=float)
        self.segment_press_depth_m = 0.0
        self.total_thread_upfeed_m = 0.0
        self.upfeed_global_peak_resistance_proxy_sensor_units = 0.0
        self.not_before_s = time.monotonic()
        self.last_observation: TactileForceObservation | None = None
        self.segment_baseline_wrench_by_side: dict[str, np.ndarray] | None = None
        self.press_proxy_sign_by_side = {"left": 1.0, "right": 1.0}
        self.presence_config: Any | None = None
        self.contact_presence_threshold_sensor_units: float | None = None
        self.release_axis_delta_sensor_units: float | None = None
        self.release_tangent_delta_sensor_units: float | None = None

    def _run(
        self, operation: str, command: list[str], output_path: Path | None = None
    ) -> dict[str, Any]:
        index = self.action_log.append(operation, "start", command=command)
        prefix = self.output_dir / "commands" / f"{index:04d}_{operation}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        try:
            completed = self.command_executor(command, self.config["repo_root"], prefix)
            result: dict[str, Any] = {}
            if output_path is not None:
                result = _read_json(output_path)
                if result.get("execution_succeeded") is not True:
                    raise TaskFailure(f"{operation} did not report execution_succeeded=true")
            elif completed.stdout.strip():
                result = json.loads(completed.stdout.splitlines()[-1])
            self.not_before_s = time.monotonic()
            self.action_log.append(
                operation, "passed", artifact=str(output_path) if output_path else None
            )
            return result
        except BaseException as exc:
            self.action_log.append(operation, "failed", error=str(exc))
            raise

    def _move_sequence(
        self,
        operation: str,
        targets: Sequence[np.ndarray],
        *,
        motion_profile: Mapping[str, float] | None = None,
    ) -> None:
        event_index = len(self.action_log.payload["events"])
        input_path = self.output_dir / "poses" / f"{event_index:04d}_{operation}.json"
        output_path = self.output_dir / "motion" / f"{event_index:04d}_{operation}.json"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(input_path, _sequence_payload(self.config["frame"].base_frame, targets))
        self._run(
            operation,
            _sequence_command(
                self.config,
                input_path,
                output_path,
                execute=True,
                motion_profile=motion_profile,
            ),
            output_path,
        )
        self.current = np.array(targets[-1], copy=True)

    def _gripper(
        self, operation: str, opening_mm: float, *, require_grasped: bool = False
    ) -> dict[str, Any]:
        payload = self._run(operation, _hande_command(self.config, opening_mm))
        _require_hande_motion_ok(payload, operation=operation)
        if require_grasped:
            _require_hande_grasped(payload, operation=operation)
        return payload

    def install_tactile_bootstrap(self, result: Any) -> None:
        self.adapter = result.adapter
        self.presence_config = result.presence_config
        self.contact_presence_threshold_sensor_units = float(
            result.contact_presence_threshold_sensor_units
        )
        self.release_axis_delta_sensor_units = float(
            result.release_axis_delta_sensor_units
        )
        self.release_tangent_delta_sensor_units = float(
            result.release_tangent_delta_sensor_units
        )

    def retract_bootstrap_press(self) -> None:
        if self.segment_press_depth_m <= 0.0:
            raise TaskFailure("bootstrap press depth was not recorded")
        target = self.current.copy()
        target[:3, 3] += self.segment_press_depth_m * self.axis_base
        self._move_sequence("bootstrap_press_retract", [target])
        self.segment_press_depth_m = 0.0

    def verify_grasp_before_bootstrap_retry(self, *, attempt: int) -> None:
        operation = f"bootstrap_retry_{attempt}_hande_contact_recheck"
        payload = self._run(
            operation,
            _hande_status_command(self.config),
        )
        _require_hande_motion_ok(payload, operation=operation)

    def _verify_any_pad_contact(self, *, operation: str) -> None:
        threshold = self.contact_presence_threshold_sensor_units
        if threshold is None:
            raise TaskFailure(f"{operation}: task-session presence gate is unavailable")
        pair = self.source.collect_classifier_windows(
            self.not_before_s,
            self.adapter.min_samples,
            self.config["gf220_window_timeout_s"],
        )
        now_s = time.monotonic()
        adapted = adapt_gf220_force6d_windows(pair, self.adapter, now_s=now_s)
        values = {
            "left": abs(float(np.median(adapted.left.wrench_task[:, 2]))),
            "right": abs(float(np.median(adapted.right.wrench_task[:, 2]))),
        }
        observed = max(values.values())
        if observed < threshold:
            raise TaskFailure(
                f"{operation}: neither tactile pad reports contact "
                f"({observed:.6g} < {threshold:.6g})"
            )
        self.not_before_s = max(
            float(adapted.left.capture_monotonic_s[-1]),
            float(adapted.right.capture_monotonic_s[-1]),
        )
        self.action_log.append(
            operation,
            "passed",
            predicate="any_pad_fz_presence",
            measured_absolute_fz_sensor_units=values,
            minimum_any_pad_fz_sensor_units=threshold,
        )

    def press_engage(self, *, context: TwistLoopContext) -> None:
        distance = (
            self.config["initial_press_m"]
            if context.press_retry_count == 0
            else self.config["retry_press_m"]
        )
        target = self.current.copy()
        target[:3, 3] -= distance * self.axis_base
        self._move_sequence(f"press_engage_{context.press_retry_count}", [target])
        self.segment_press_depth_m += distance

    def _twist_targets(
        self,
        *,
        angle_rad: float,
        step_count: int,
        context: TwistLoopContext,
    ) -> tuple[list[np.ndarray], float, float]:
        target = self.current.copy()
        targets: list[np.ndarray] = []
        total_downfeed_m = 0.0
        total_upfeed_m = 0.0
        transition_rad = (
            self.config["twist_downfeed_revolutions"] * 2.0 * math.pi
        )
        initial_progress_rad = abs(
            float(
                getattr(
                    context,
                    "total_signed_twist_rad",
                    getattr(context, "twist_step_count", 0) * angle_rad,
                )
            )
        )
        for offset in range(step_count):
            before_rad = initial_progress_rad + offset * abs(angle_rad)
            after_rad = before_rad + abs(angle_rad)
            downfeed_rad = max(
                0.0,
                min(after_rad, transition_rad) - min(before_rad, transition_rad),
            )
            upfeed_rad = abs(angle_rad) - downfeed_rad
            downfeed_m = (
                self.config["twist_downfeed_m_per_revolution"]
                * downfeed_rad
                / (2.0 * math.pi)
            )
            upfeed_m = (
                self.config["thread_pitch_m_per_revolution"]
                * upfeed_rad
                / (2.0 * math.pi)
            )
            target = target.copy()
            target[:3, :3] = (
                _axis_rotation(self.axis_base, angle_rad) @ target[:3, :3]
            )
            target[:3, 3] += (upfeed_m - downfeed_m) * self.axis_base
            targets.append(target)
            total_downfeed_m += downfeed_m
            total_upfeed_m += upfeed_m
        proposed_upfeed_m = self.total_thread_upfeed_m + total_upfeed_m
        if proposed_upfeed_m > self.config["max_axial_upfeed_m"] + 1e-12:
            raise TaskFailure(
                "twist axial upfeed hard limit exceeded "
                f"({proposed_upfeed_m:.6f} > "
                f"{self.config['max_axial_upfeed_m']:.6f} m)"
            )
        return targets, total_downfeed_m, total_upfeed_m

    def execute_twist_step(self, *, angle_rad: float, context: TwistLoopContext) -> None:
        targets, downfeed_m, upfeed_m = self._twist_targets(
            angle_rad=angle_rad, step_count=1, context=context
        )
        self._move_sequence(
            f"{context.phase.value}_step_{context.twist_step_count + 1}",
            targets,
            motion_profile=self.config["twist_motion"],
        )
        self.segment_press_depth_m += downfeed_m
        self.total_thread_upfeed_m += upfeed_m

    def execute_twist_batch(
        self, *, angle_rad: float, step_count: int, context: TwistLoopContext
    ) -> None:
        targets, downfeed_m, upfeed_m = self._twist_targets(
            angle_rad=angle_rad, step_count=step_count, context=context
        )
        first = context.twist_step_count + 1
        last = context.twist_step_count + step_count
        self._move_sequence(
            f"{context.phase.value}_batch_{first}_{last}",
            targets,
            motion_profile=self.config["twist_motion"],
        )
        self.segment_press_depth_m += downfeed_m
        self.total_thread_upfeed_m += upfeed_m

    def _window_proxy_metrics(
        self,
        observation: TactileForceObservation,
        *,
        update_press_signs: bool = False,
    ) -> dict[str, Any]:
        medians = {
            "left": np.median(observation.left.wrench_task, axis=0),
            "right": np.median(observation.right.wrench_task, axis=0),
        }
        baseline = self.segment_baseline_wrench_by_side
        deltas = {
            side: (
                value - baseline[side]
                if baseline is not None
                else np.full(6, np.nan, dtype=float)
            )
            for side, value in medians.items()
        }
        if update_press_signs and baseline is not None:
            minimum = (
                float(self.presence_config.press_retain_fz)
                if self.presence_config is not None
                else 1e-12
            )
            for side, delta in deltas.items():
                if abs(float(delta[2])) >= minimum:
                    self.press_proxy_sign_by_side[side] = math.copysign(
                        1.0, float(delta[2])
                    )
        aligned = {
            side: (
                self.press_proxy_sign_by_side[side] * float(delta[2])
                if np.isfinite(delta[2])
                else math.nan
            )
            for side, delta in deltas.items()
        }
        finite_resistance = [
            abs(float(delta[2]))
            for delta in deltas.values()
            if math.isfinite(float(delta[2]))
        ]
        return {
            "median_wrench_unloaded_relative_sensor_units_by_side": {
                side: value.tolist() for side, value in medians.items()
            },
            "median_wrench_segment_baseline_delta_sensor_units_by_side": {
                side: value.tolist() for side, value in deltas.items()
            },
            "press_aligned_raw_fz_proxy_sensor_units_by_side": aligned,
            "combined_press_aligned_raw_fz_proxy_sensor_units": (
                float(sum(finite_resistance)) if finite_resistance else math.nan
            ),
            "proxy_semantics": (
                "known-downpress-aligned raw SDK Fz; not a calibrated force axis"
            ),
        }

    def collect_force_window(self, *, context: TwistLoopContext) -> TactileForceObservation:
        pair = self.source.collect_classifier_windows(
            self.not_before_s,
            self.adapter.min_samples,
            self.config["gf220_window_timeout_s"],
        )
        now_s = time.monotonic()
        adapted = adapt_gf220_force6d_windows(pair, self.adapter, now_s=now_s)
        observation = TactileForceObservation(adapted.left, adapted.right, now_s)
        self.not_before_s = max(
            float(adapted.left.capture_monotonic_s[-1]),
            float(adapted.right.capture_monotonic_s[-1]),
        )
        self.last_observation = observation
        if context.phase is TwistLoopPhase.PRE_PRESS_BASELINE:
            self.segment_baseline_wrench_by_side = {
                "left": np.median(observation.left.wrench_task, axis=0),
                "right": np.median(observation.right.wrench_task, axis=0),
            }
        proxy_metrics = self._window_proxy_metrics(
            observation,
            update_press_signs=context.phase is TwistLoopPhase.PRESS_VERIFY,
        )
        in_upfeed_release_phase = (
            context.phase is TwistLoopPhase.TWIST
            and abs(context.total_signed_twist_rad)
            > self.config["release_min_total_twist_rad"] + 1e-12
        )
        current_proxy = float(
            proxy_metrics["combined_press_aligned_raw_fz_proxy_sensor_units"]
        )
        if in_upfeed_release_phase and math.isfinite(current_proxy):
            self.upfeed_global_peak_resistance_proxy_sensor_units = max(
                self.upfeed_global_peak_resistance_proxy_sensor_units,
                current_proxy,
            )
        release_limit = None
        if in_upfeed_release_phase and self.presence_config is not None:
            release_limit = release_resistance_limit_sensor_units(
                self.presence_config,
                self.upfeed_global_peak_resistance_proxy_sensor_units,
            )
        self.action_log.append(
            "collect_force_window",
            "passed",
            phase=context.phase.value,
            samples=int(adapted.left.wrench_task.shape[0]),
            left_serial=adapted.left_serial,
            right_serial=adapted.right_serial,
            evidence_end_monotonic_s=self.not_before_s,
            in_upfeed_release_phase=in_upfeed_release_phase,
            upfeed_global_peak_resistance_proxy_sensor_units=(
                self.upfeed_global_peak_resistance_proxy_sensor_units
                if in_upfeed_release_phase
                else None
            ),
            release_resistance_limit_sensor_units=release_limit,
            **proxy_metrics,
        )
        return observation

    def unload(self, *, context: TwistLoopContext) -> None:
        self._gripper("unload_open", self.config["regrasp_unload_opening_mm"])
        if self.segment_press_depth_m > 0.0:
            target = self.current.copy()
            target[:3, 3] += self.segment_press_depth_m * self.axis_base
            self._move_sequence("unload_axial_retract", [target])
            self.segment_press_depth_m = 0.0

    def reset(self, *, angle_rad: float, context: TwistLoopContext) -> None:
        max_step = abs(self.config["loop_config"].twist_step_angle_rad)
        count = max(1, int(math.ceil(abs(angle_rad) / max_step)))
        increment = angle_rad / count
        targets: list[np.ndarray] = []
        target = self.current.copy()
        for _ in range(count):
            target = target.copy()
            target[:3, :3] = (
                _axis_rotation(self.axis_base, increment) @ target[:3, :3]
            )
            targets.append(target)
        self._move_sequence(f"wrist_reset_{context.regrasp_count + 1}", targets)

    def regrasp(self, *, context: TwistLoopContext) -> None:
        self._gripper(
            "regrasp_close",
            self.config["regrasp_grasp_opening_mm"],
        )
        self._verify_any_pad_contact(operation="regrasp_any_pad_contact")

    @staticmethod
    def _force_summary(
        observation: TactileForceObservation, axis: np.ndarray
    ) -> tuple[float, float]:
        axis_force = 0.0
        tangential = 0.0
        for window in (observation.left, observation.right):
            force = np.median(window.wrench_task[:, :3], axis=0)
            axis_force += float(np.dot(force, axis))
            position = np.asarray(window.contact_position_task_m, dtype=float)
            if position.shape == (3,):
                radius = position - float(np.dot(position, axis)) * axis
            else:
                radius = np.median(position, axis=0)
                radius -= float(np.dot(radius, axis)) * axis
            tangent = np.cross(axis, radius)
            norm = float(np.linalg.norm(tangent))
            if norm > 1e-12:
                tangential += float(np.dot(force, tangent / norm))
        return axis_force, tangential

    def release_verify(self, *, context: TwistLoopContext) -> bool:
        if self.last_observation is None:
            raise TaskFailure("release verification has no pre-lift tactile window")
        before_metrics = self._window_proxy_metrics(self.last_observation)
        before_proxy = float(
            before_metrics["combined_press_aligned_raw_fz_proxy_sensor_units"]
        )
        after_observation = self.collect_force_window(context=context)
        after_metrics = self._window_proxy_metrics(after_observation)
        after_proxy = float(
            after_metrics["combined_press_aligned_raw_fz_proxy_sensor_units"]
        )
        contact_threshold = self.contact_presence_threshold_sensor_units
        presence_config = self.presence_config
        if contact_threshold is None or presence_config is None:
            raise TaskFailure("release verification thresholds are unavailable")
        global_peak = self.upfeed_global_peak_resistance_proxy_sensor_units
        resistance_limit = release_resistance_limit_sensor_units(
            presence_config,
            global_peak,
        )
        after_presence = max(
            abs(float(np.median(after_observation.left.wrench_task[:, 2]))),
            abs(float(np.median(after_observation.right.wrench_task[:, 2]))),
        )
        verified = (
            global_peak >= presence_config.release_min_peak_fz
            and
            after_presence >= contact_threshold
            and after_proxy <= resistance_limit
        )
        if not verified:
            self.action_log.append(
                "release_verify",
                "rejected",
                predicate=(
                    "cap_contact_retained_and_fresh_stationary_window_keeps_"
                    "vertical_shear_proxy_low"
                ),
                before_proxy_sensor_units=before_proxy,
                after_proxy_sensor_units=after_proxy,
                upfeed_global_peak_resistance_proxy_sensor_units=global_peak,
                release_peak_fraction=presence_config.release_resistance_peak_fraction,
                release_noise_floor_sensor_units=presence_config.press_retain_fz,
                maximum_release_proxy_sensor_units=resistance_limit,
                after_presence_sensor_units=after_presence,
                minimum_presence_sensor_units=contact_threshold,
            )
            return False
        self.segment_press_depth_m = 0.0
        self.action_log.append(
            "release_verify",
            "passed",
            predicate=(
                "cap_contact_retained_and_fresh_stationary_window_keeps_"
                "vertical_shear_proxy_low"
            ),
            before_proxy_sensor_units=before_proxy,
            after_proxy_sensor_units=after_proxy,
            upfeed_global_peak_resistance_proxy_sensor_units=global_peak,
            release_peak_fraction=presence_config.release_resistance_peak_fraction,
            release_noise_floor_sensor_units=presence_config.press_retain_fz,
            maximum_release_proxy_sensor_units=resistance_limit,
            after_presence_sensor_units=after_presence,
            minimum_presence_sensor_units=contact_threshold,
        )
        return True


def _resolved_transforms(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    frame: BottleCapFrame = config["frame"]
    axis = np.asarray(frame.bottle_axis_base, dtype=float)
    contact = np.array(frame.T_base_cap_tcp, copy=True)
    approach = contact.copy()
    approach[:3, 3] += config["approach_standoff_m"] * axis
    first_press = contact.copy()
    first_press[:3, 3] -= config["initial_press_m"] * axis
    first_twist = first_press.copy()
    first_twist[:3, :3] = (
        _axis_rotation(axis, config["loop_config"].twist_step_angle_rad) @ first_twist[:3, :3]
    )
    revolution_fraction = (
        abs(config["loop_config"].twist_step_angle_rad) / (2.0 * math.pi)
    )
    first_twist[:3, 3] -= (
        config["twist_downfeed_m_per_revolution"]
        * revolution_fraction
        * axis
    )
    lift = first_twist.copy()
    lift[:3, 3] += config["motion"]["lift_distance_m"] * axis
    return {
        "approach": approach,
        "contact": contact,
        "first_press": first_press,
        "first_formal_twist": first_twist,
        "nominal_lift": lift,
    }


def _write_plan(config: Mapping[str, Any], output_dir: Path) -> Path:
    transforms = _resolved_transforms(config)
    contact_waypoints = _translation_waypoints(
        transforms["approach"], transforms["contact"]
    )
    artifacts = output_dir / "plan_inputs"
    artifacts.mkdir(parents=True, exist_ok=True)
    contact_sequence = artifacts / "contact_sequence.json"
    press_twist = artifacts / "press_twist_preview.json"
    _atomic_json(
        contact_sequence,
        _sequence_payload(config["frame"].base_frame, contact_waypoints),
    )
    _atomic_json(
        press_twist,
        _sequence_payload(
            config["frame"].base_frame,
            [
                transforms["first_press"],
                transforms["first_formal_twist"],
            ],
        ),
    )
    approach_output = output_dir / "planned_subprocess_outputs" / "approach.json"
    contact_output = output_dir / "planned_subprocess_outputs" / "contact.json"
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "review_only",
        "execution_authorized": False,
        "cap_frame": config["frame"].as_dict(),
        "resolved_targets": {name: value.tolist() for name, value in transforms.items()},
        "contact_approach": {
            "waypoint_count": len(contact_waypoints),
            "max_configured_ee_segment_m": CONTACT_WAYPOINT_MAX_SEGMENT_M,
        },
        "loop": {
            **asdict(config["loop_config"]),
            "thread_pitch_m_per_revolution": config[
                "thread_pitch_m_per_revolution"
            ],
            "twist_downfeed_m_per_revolution": config[
                "twist_downfeed_m_per_revolution"
            ],
            "twist_downfeed_revolutions": config[
                "twist_downfeed_revolutions"
            ],
            "minimum_release_revolutions": (
                config["release_min_total_twist_rad"] / (2.0 * math.pi)
            ),
            "max_axial_upfeed_m": config["max_axial_upfeed_m"],
            "policy": (
                "pre-press baseline -> bounded axial PRESS increments -> any-pad Fz "
                "presence -> first revolution downfeed -> later revolutions helical "
                "upfeed; press-aligned raw-Fz proxy drop gates release; loss of all "
                "tactile presence stops; reset/regrasp re-establishes the press gate"
            ),
        },
        "gf220": {
            "manual_warp_profile": str(config["manual_warp_path"]),
            "wrench_transform_profile": str(config["wrench_profile_path"]),
            "wrench_profile_status": config["wrench_meta"]["status"],
            "profile_role": "serial-side/contact-geometry seed only",
            "mandatory_execute_bootstrap": [
                "unloaded_FORCE6D",
                "any_pad_grasp_Fz_presence",
                "known_axial_press_FORCE6D",
            ],
        },
        "command_templates": {
            "approach": _pose_command(
                config, transforms["approach"], approach_output, execute=True
            ),
            "contact": _sequence_command(config, contact_sequence, contact_output, execute=True),
            "hande_open": _hande_command(config, config["hande"]["open_width_mm"]),
            "hande_grasp": _hande_command(config, config["hande"]["grasp_width_mm"]),
        },
        "artifacts": {
            "contact_sequence": str(contact_sequence),
            "press_twist_preview": str(press_twist),
        },
    }
    path = output_dir / "resolved_plan.json"
    _atomic_json(path, plan)
    return path


def _safe_recovery_log(
    log: AtomicActionLog, operation: str, state: str, **details: Any
) -> None:
    try:
        log.append(operation, state, **details)
    except BaseException:
        pass


def _attempt_failure_recovery(
    config: Mapping[str, Any],
    output_dir: Path,
    log: AtomicActionLog,
    *,
    current: np.ndarray,
    original_error: BaseException,
) -> None:
    """Best-effort open and axial retreat without ever replacing ``original_error``."""

    _safe_recovery_log(
        log,
        "failure_recovery",
        "start",
        original_error=str(original_error),
    )
    try:
        completed = _run_process(
            _hande_command(config, config["hande"]["open_width_mm"]),
            config["repo_root"],
            output_dir / "failure_recovery_hande_open",
        )
        response = (
            json.loads(completed.stdout.splitlines()[-1])
            if completed.stdout.strip()
            else None
        )
        _safe_recovery_log(
            log,
            "failure_recovery_hande_open",
            "passed",
            response=response,
        )
    except BaseException as exc:
        _safe_recovery_log(
            log, "failure_recovery_hande_open", "failed", error=str(exc)
        )

    try:
        frame: BottleCapFrame = config["frame"]
        axis = np.asarray(frame.bottle_axis_base, dtype=float)
        current_checked = validate_rigid_transform(current, name="recovery current configured EE")
        standoff_axis_coordinate = float(
            np.dot(
                frame.T_base_cap_tcp[:3, 3]
                + config["approach_standoff_m"] * axis,
                axis,
            )
        )
        current_axis_coordinate = float(np.dot(current_checked[:3, 3], axis))
        retract_distance_m = max(0.0, standoff_axis_coordinate - current_axis_coordinate)
        if retract_distance_m <= 1e-9:
            _safe_recovery_log(
                log,
                "failure_recovery_retract",
                "passed",
                skipped="already_at_or_above_standoff",
            )
        else:
            target = np.array(current_checked, copy=True)
            target[:3, 3] += retract_distance_m * axis
            waypoints = _translation_waypoints(current_checked, target)
            input_path = output_dir / "poses" / "failure_recovery_retract.json"
            output_path = output_dir / "motion" / "failure_recovery_retract.json"
            _atomic_json(
                input_path,
                _sequence_payload(frame.base_frame, waypoints),
            )
            _run_process(
                _sequence_command(config, input_path, output_path, execute=True),
                config["repo_root"],
                output_dir / "failure_recovery_retract",
            )
            payload = _read_json(output_path)
            if payload.get("execution_succeeded") is not True:
                raise TaskFailure(
                    "failure recovery retract did not report execution_succeeded=true"
                )
            _safe_recovery_log(
                log,
                "failure_recovery_retract",
                "passed",
                artifact=str(output_path),
                retract_distance_m=retract_distance_m,
                waypoint_count=len(waypoints),
            )
    except BaseException as exc:
        _safe_recovery_log(log, "failure_recovery_retract", "failed", error=str(exc))
    _safe_recovery_log(log, "failure_recovery", "finished")


def _bootstrap_pair_summary(pair: Any) -> dict[str, Any]:
    sensors: list[dict[str, Any]] = []
    for serial, window in zip(pair.serials, pair.windows, strict=True):
        force = window.force6d
        if force is None:
            raise TaskFailure(f"bootstrap/{serial}: FORCE6D window is missing")
        capture = np.asarray(force.capture_monotonic_s, dtype=float)
        sdk = np.asarray(force.sdk_time_s, dtype=float)
        sensors.append(
            {
                "serial": serial,
                "calibration_id": window.calibration_id,
                "samples": int(np.asarray(force.values).shape[0]),
                "capture_monotonic_s": [float(capture[0]), float(capture[-1])],
                "sdk_time_s": [float(sdk[0]), float(sdk[-1])],
            }
        )
    return {
        "serials": list(pair.serials),
        "endpoint_skew_s": float(pair.endpoint_skew_s),
        "collected_monotonic_s": float(pair.collected_monotonic_s),
        "sensors": sensors,
    }


def _run_task_session_tactile_bootstrap(
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    log: AtomicActionLog,
    source: GF220LiveSource,
    io: SubprocessMedicineBottleIO,
    seed_adapter: GF220WrenchAdapterConfig,
    unloaded_window: Any,
    grasp_window: Any,
    bootstrap_fn: Callable[..., Any] | None = None,
) -> Any:
    """Establish any-pad Fz presence with bounded incremental press retries."""

    bootstrap_fn = bootstrap_fn or bootstrap_medicine_bottle_tactile_presence
    raw_path = output_dir / "tactile_bootstrap_windows.npz"
    report_path = output_dir / "tactile_bootstrap.json"
    attempts_path = output_dir / "tactile_bootstrap_attempts.json"
    attempts_payload: dict[str, Any] = {
        "schema": BOOTSTRAP_ATTEMPTS_SCHEMA,
        "status": "running",
        "maximum_retries": config["loop_config"].max_press_retries,
        "initial_press_m": config["initial_press_m"],
        "retry_increment_m": config["retry_press_m"],
        "raw_windows": str(raw_path),
        "attempts": [],
    }
    press_windows: dict[str, Any] = {}
    _atomic_json(attempts_path, attempts_payload)

    for attempt in range(config["loop_config"].max_press_retries + 1):
        if attempt > 0:
            io.verify_grasp_before_bootstrap_retry(attempt=attempt)
        context = TwistLoopContext(
            phase=TwistLoopPhase.PRESS_ENGAGE,
            twist_step_count=0,
            segment_step_count=0,
            regrasp_count=0,
            press_retry_count=attempt,
            segment_wrist_travel_rad=0.0,
            total_signed_twist_rad=0.0,
        )
        io.press_engage(context=context)
        press_window = source.collect_classifier_windows(
            io.not_before_s,
            seed_adapter.min_samples,
            config["gf220_window_timeout_s"],
        )
        stage_name = f"axial_press_{attempt}"
        press_windows[stage_name] = press_window
        _atomic_npz(
            raw_path,
            **_bootstrap_window_arrays(
                {
                    "unloaded": unloaded_window,
                    "grasp": grasp_window,
                    **press_windows,
                }
            ),
        )
        attempt_record: dict[str, Any] = {
            "attempt": attempt,
            "press_increment_m": (
                config["initial_press_m"] if attempt == 0 else config["retry_press_m"]
            ),
            "cumulative_press_depth_m": io.segment_press_depth_m,
            "evidence": _bootstrap_pair_summary(press_window),
        }
        log.append(
            "bootstrap_axial_press_window",
            "passed",
            attempt=attempt,
            cumulative_press_depth_m=io.segment_press_depth_m,
            raw_windows=str(raw_path),
            evidence=attempt_record["evidence"],
        )
        try:
            bootstrap = bootstrap_fn(
                unloaded_window,
                grasp_window,
                press_window,
                seed_adapter=seed_adapter,
                tactile_config=config["tactile_config"],
                release_min_total_twist_rad=config[
                    "release_min_total_twist_rad"
                ],
            )
        except TactileBootstrapQualityError as exc:
            retry_allowed = (
                exc.reason_code in BOOTSTRAP_RETRYABLE_QUALITY_REASONS
                and attempt < config["loop_config"].max_press_retries
            )
            retry_decision_reason = (
                "bounded_extra_press_for_absent_tactile_response"
                if retry_allowed
                else "quality_reason_not_retryable"
                if exc.reason_code not in BOOTSTRAP_RETRYABLE_QUALITY_REASONS
                else "bounded_press_retries_exhausted"
            )

            attempt_record.update(
                {
                    "status": "rejected_quality",
                    "reason_code": exc.reason_code,
                    "error": str(exc),
                    "details": exc.details,
                    "retry_authorized": retry_allowed,
                    "retry_decision_reason": retry_decision_reason,
                }
            )
            attempts_payload["attempts"].append(attempt_record)
            attempts_payload["status"] = (
                "retrying"
                if retry_allowed
                else "exhausted"
                if attempt == config["loop_config"].max_press_retries
                else "stopped_quality_gate"
            )
            _atomic_json(attempts_path, attempts_payload)
            log.append(
                "task_session_tactile_bootstrap_attempt",
                "rejected",
                attempt=attempt,
                reason_code=exc.reason_code,
                error=str(exc),
                details=exc.details,
                retry_authorized=retry_allowed,
                retry_decision_reason=retry_decision_reason,
                cumulative_press_depth_m=io.segment_press_depth_m,
                attempts_artifact=str(attempts_path),
            )
            if not retry_allowed:
                raise TaskFailure(
                    "task-session tactile bootstrap stopped after bounded press "
                    f"attempt {attempt}: {retry_decision_reason}: "
                    f"{exc.reason_code}: {exc}"
                ) from exc
            continue
        except BaseException as exc:
            attempt_record.update(
                {
                    "status": "invalid_evidence",
                    "error": str(exc),
                }
            )
            attempts_payload["attempts"].append(attempt_record)
            attempts_payload["status"] = "failed_invalid_evidence"
            _atomic_json(attempts_path, attempts_payload)
            log.append(
                "task_session_tactile_bootstrap_attempt",
                "failed",
                attempt=attempt,
                error=str(exc),
                attempts_artifact=str(attempts_path),
            )
            raise

        attempt_record.update(
            {
                "status": "passed",
                "bootstrap_report": str(report_path),
            }
        )
        attempts_payload["attempts"].append(attempt_record)
        attempts_payload["status"] = "passed"
        attempts_payload["selected_attempt"] = attempt
        attempts_payload["final_cumulative_press_depth_m"] = io.segment_press_depth_m
        _atomic_json(report_path, bootstrap.report)
        _atomic_json(attempts_path, attempts_payload)
        io.install_tactile_bootstrap(bootstrap)
        log.append(
            "task_session_tactile_bootstrap",
            "passed",
            attempt=attempt,
            predicate="any_pad_grasp_and_downpress_tactile_response",
            report=str(report_path),
            attempts_artifact=str(attempts_path),
            raw_windows=str(raw_path),
        )
        # Identification presses are evidence only.  Retract their cumulative
        # depth before the twist loop records its grasp-only baseline.
        io.retract_bootstrap_press()
        return bootstrap

    raise AssertionError("bounded bootstrap loop exited without a result")


def _execute(config: Mapping[str, Any], output_dir: Path, log: AtomicActionLog) -> dict[str, Any]:
    frame: BottleCapFrame = config["frame"]
    transforms = _resolved_transforms(config)
    source = GF220LiveSource(config["manual_warp_path"], config=config["gf220_live_config"])
    cap_region_entered = False
    io: SubprocessMedicineBottleIO | None = None
    result: dict[str, Any]
    try:
        log.append("gf220_start", "start")
        source.start(timeout_s=config["gf220_startup_timeout_s"])
        log.append("gf220_start", "passed")

        open_result = _run_process(
            _hande_command(config, config["hande"]["open_width_mm"]),
            config["repo_root"],
            output_dir / "hande_initial_open",
        )
        if open_result.stdout.strip():
            log.append(
                "hande_initial_open",
                "passed",
                response=json.loads(open_result.stdout.splitlines()[-1]),
            )

        calibrations = source.calibrate_unloaded(timeout_s=config["gf220_calibration_timeout_s"])
        calibration_ids = {value.serial: value.calibration_id for value in calibrations}
        seed_adapter, _ = _load_wrench_profile(
            config["wrench_profile_path"], calibration_ids=calibration_ids
        )
        log.append("gf220_calibrate_unloaded", "passed", calibration_ids=calibration_ids)
        unloaded_not_before_s = time.monotonic()
        unloaded_window = source.collect_classifier_windows(
            unloaded_not_before_s,
            seed_adapter.min_samples,
            config["gf220_window_timeout_s"],
        )
        log.append(
            "bootstrap_unloaded_window",
            "passed",
            samples=seed_adapter.min_samples,
            serials=list(unloaded_window.serials),
        )

        approach_output = output_dir / "motion" / "approach.json"
        approach_output.parent.mkdir(parents=True, exist_ok=True)
        _run_process(
            _pose_command(config, transforms["approach"], approach_output, execute=True),
            config["repo_root"],
            output_dir / "approach",
        )
        approach_payload = _read_json(approach_output)
        if approach_payload.get("execution_succeeded") is not True:
            raise TaskFailure("approach did not report execution_succeeded=true")
        log.append("approach", "passed", artifact=str(approach_output))

        contact_input = output_dir / "poses" / "contact.json"
        contact_output = output_dir / "motion" / "contact.json"
        contact_waypoints = _translation_waypoints(
            transforms["approach"], transforms["contact"]
        )
        _atomic_json(
            contact_input,
            _sequence_payload(frame.base_frame, contact_waypoints),
        )
        # From this point onward even a partially completed descent may have
        # entered the cap region, so every exception takes the open+retract path.
        cap_region_entered = True
        _run_process(
            _sequence_command(config, contact_input, contact_output, execute=True),
            config["repo_root"],
            output_dir / "contact",
        )
        contact_payload = _read_json(contact_output)
        if contact_payload.get("execution_succeeded") is not True:
            raise TaskFailure("contact approach did not report execution_succeeded=true")
        log.append("contact", "passed", artifact=str(contact_output))

        grasp_result = _run_process(
            _hande_command(config, config["hande"]["grasp_width_mm"]),
            config["repo_root"],
            output_dir / "hande_cap_grasp",
        )
        grasp_payload = (
            json.loads(grasp_result.stdout.splitlines()[-1])
            if grasp_result.stdout.strip()
            else None
        )
        grasp_state = _require_hande_motion_ok(
            grasp_payload, operation="hande_cap_grasp"
        )
        grasp_opening = grasp_state.get("opening_width_mm")
        initial_grasp_opening_mm = (
            float(grasp_opening)
            if isinstance(grasp_opening, (int, float))
            and not isinstance(grasp_opening, bool)
            and math.isfinite(float(grasp_opening))
            else None
        )
        log.append(
            "hande_cap_grasp",
            "passed",
            predicate="command_completed_without_fault; tactile presence follows",
            response=grasp_payload,
        )
        grasp_not_before_s = time.monotonic()
        grasp_window = source.collect_classifier_windows(
            grasp_not_before_s,
            seed_adapter.min_samples,
            config["gf220_window_timeout_s"],
        )
        log.append(
            "bootstrap_grasp_window",
            "passed",
            samples=seed_adapter.min_samples,
            serials=list(grasp_window.serials),
        )

        io = SubprocessMedicineBottleIO(
            config=config,
            output_dir=output_dir,
            action_log=log,
            tactile_source=source,
            adapter_config=seed_adapter,
            initial_grasp_opening_mm=initial_grasp_opening_mm,
        )
        bootstrap = _run_task_session_tactile_bootstrap(
            config=config,
            output_dir=output_dir,
            log=log,
            source=source,
            io=io,
            seed_adapter=seed_adapter,
            unloaded_window=unloaded_window,
            grasp_window=grasp_window,
        )
        evaluator = TactilePresenceSupervisorEvaluator(
            config=bootstrap.presence_config,
            wrist_travel_limit_rad=config["loop_config"].wrist_travel_limit_rad,
        )
        loop_result = run_medicine_bottle_twist_loop(io, evaluator, config["loop_config"])
        _atomic_json(
            output_dir / "twist_loop_result.json",
            {
                "status": loop_result.status.value,
                "reason": loop_result.reason.value,
                "twist_step_count": loop_result.twist_step_count,
                "regrasp_count": loop_result.regrasp_count,
                "total_signed_twist_rad": loop_result.total_signed_twist_rad,
                "total_revolutions": abs(loop_result.total_signed_twist_rad)
                / (2.0 * math.pi),
                "total_thread_upfeed_m": io.total_thread_upfeed_m,
                "upfeed_global_peak_resistance_proxy_sensor_units": (
                    io.upfeed_global_peak_resistance_proxy_sensor_units
                ),
                "release_verify_attempts": loop_result.release_verify_attempts,
                "events": [
                    {
                        "operation": event.operation,
                        "context": {
                            **asdict(event.context),
                            "phase": event.context.phase.value,
                        },
                        "angle_rad": event.angle_rad,
                        "tactile_action": (
                            event.tactile_action.value
                            if event.tactile_action is not None
                            else None
                        ),
                        "release_verified": event.release_verified,
                    }
                    for event in loop_result.events
                ],
            },
        )
        if not loop_result.completed:
            raise TaskFailure(f"twist loop stopped: {loop_result.reason.value}")

        lift_target = io.current.copy()
        lift_target[:3, 3] += config["motion"]["lift_distance_m"] * io.axis_base
        io._move_sequence("lift_released_cap", [lift_target])
        result = {
            "twist_loop_status": loop_result.status.value,
            "twist_loop_reason": loop_result.reason.value,
            "twist_steps": loop_result.twist_step_count,
            "regrasps": loop_result.regrasp_count,
            "final_tcp": io.current.tolist(),
        }
    except BaseException as original_error:
        if cap_region_entered:
            current = io.current if io is not None else transforms["contact"]
            try:
                _attempt_failure_recovery(
                    config,
                    output_dir,
                    log,
                    current=current,
                    original_error=original_error,
                )
            except BaseException as recovery_error:
                _safe_recovery_log(
                    log,
                    "failure_recovery",
                    "failed",
                    error=f"unexpected recovery failure: {recovery_error}",
                )
        try:
            source.close()
        except BaseException as close_error:
            _safe_recovery_log(log, "gf220_close", "failed", error=str(close_error))
        raise
    else:
        source.close()
        return result


def run(args: argparse.Namespace) -> int:
    if args.execute and not args.accept_arm_only_collision_model:
        raise ValueError("--execute requires --accept-arm-only-collision-model")
    if not args.execute and args.accept_arm_only_collision_model:
        raise ValueError("collision-model acknowledgement is valid only with --execute")
    if args.execute and "realtime" not in platform.release().lower():
        raise RuntimeError("physical execution requires a realtime kernel")

    config_path = args.config.expanduser().resolve()
    config = _validate_config(
        _read_json(config_path), config_path=config_path, execute=args.execute
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    mode = "execute" if args.execute else "plan_only"
    log = AtomicActionLog(output_dir / "action_log.json", mode=mode)
    report_path = output_dir / "stage_report.json"
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "running",
        "mode": mode,
        "config": str(config_path),
        "started_at": _now(),
        "updated_at": _now(),
        "arm_only_collision_model_acknowledged": bool(args.accept_arm_only_collision_model),
        "kernel_release": platform.release(),
        "stages": [],
    }
    _atomic_json(report_path, report)
    try:
        plan_path = _write_plan(config, output_dir)
        report["stages"].append(
            {"name": "RESOLVE_AND_VALIDATE", "status": "passed", "artifact": str(plan_path)}
        )
        report["updated_at"] = _now()
        _atomic_json(report_path, report)
        if not args.execute:
            report["status"] = "review_ready"
            report["finished_at"] = _now()
            report["not_executed"] = [
                "APPROACH",
                "CONTACT",
                "GRASP",
                "PRE_PRESS_BASELINE_PRESS_CONFIRM_TWIST",
                "LIFT_CAP",
            ]
            _atomic_json(report_path, report)
            print(f"review-ready medicine-bottle plan: {plan_path}")
            return 0

        details = _execute(config, output_dir, log)
        report["stages"].append(
            {"name": "CONTINUOUS_EXECUTION", "status": "passed", "details": details}
        )
        report["status"] = "completed"
        report["finished_at"] = _now()
        report["updated_at"] = _now()
        _atomic_json(report_path, report)
        print(f"medicine-bottle task completed: {report_path}")
        return 0
    except BaseException as exc:
        report["status"] = "failed"
        report["failed_at"] = _now()
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        report["updated_at"] = _now()
        _atomic_json(report_path, report)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (ValueError, RuntimeError, TaskFailure) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
