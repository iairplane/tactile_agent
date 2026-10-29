#!/usr/bin/env python3
"""Run one fixed, fail-closed OBSERVE-to-LIFT pick state machine.

The default is a review run: it captures or consumes RGB-D, grounds the one-shot
demonstration, validates all geometric predicates, and asks MoveIt to plan only the
global pregrasp.  Physical commands require both ``--execute`` and the explicit
arm-only collision-model acknowledgement.

The runner intentionally launches the camera, core, ROS, and Hand-E tools as separate
processes.  Their Python environments remain isolated and every stage writes its own
artifact plus an atomically updated ``stage_report.json``.

Exactly one live-pose source is configured: ``inputs.object_template_dir`` registers
the current localization automatically, while ``inputs.live_object_pose`` consumes an
explicit external artifact read-only.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from demo2gripper.autonomous_pick import (
    PickStage,
    PickThresholds,
    assert_stage_prefix,
    atomic_write_json,
    cartesian_measured_end_tcp_transform,
    cartesian_result_succeeded,
    evaluate_descend_tcp_residual,
    hande_contact_verified,
    hande_open_succeeded,
    hande_width_snapshot_valid,
    measured_pose_within_tolerance,
    pose_result_succeeded,
    prepare_execution_candidate,
    rotation_matrix_to_quaternion_xyzw,
    validate_pick_geometry,
)
from demo2gripper.franka import validate_rigid_transform
from demo2gripper.supported_object import conservative_projected_width
from demo2gripper.trajectory_safety import (
    DEFAULT_MAX_JOINT_SWEEP_RAD,
    DEFAULT_MAX_TOTAL_JOINT_SWEEP_RAD,
    DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD,
    validate_fr3_trajectory_audit_thresholds,
)

SCHEMA = "demo2gripper.autonomous_pick_config.v1"
REPORT_SCHEMA = "demo2gripper.autonomous_pick_report.v1"
LIVE_OBJECT_POSE_SCHEMA = "demo2gripper.live_object_pose.v1"
LIVE_POSE_FRAME_CONVENTION = "T_parent_child, right-handed, column vectors, metres"


class StageFailure(RuntimeError):
    """A stage predicate or delegated command failed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Use an existing RGB-D capture instead of the configured live camera",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run all physical PREGRASP through LIFT stages",
    )
    parser.add_argument(
        "--accept-arm-only-collision-model",
        action="store_true",
        help="Acknowledge that MoveIt does not model the installed Hand-E",
    )
    return parser.parse_args()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _selected_points_artifact_path(localization_dir: Path, anchor: Mapping[str, Any]) -> Path:
    """Resolve the selected points produced by this run's localization stage."""

    outputs = anchor.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("live object anchor has no output provenance")
    artifact_name = outputs.get("selected_points_base")
    if not isinstance(artifact_name, str) or not artifact_name:
        raise ValueError("live object anchor has no selected base-frame point artifact")
    artifact_path = (localization_dir / artifact_name).resolve()
    if not artifact_path.is_relative_to(localization_dir.resolve()):
        raise ValueError("selected point artifact escapes the localization directory")
    if not artifact_path.is_file():
        raise ValueError(f"selected point artifact is missing: {artifact_path}")
    return artifact_path


def _projected_width_from_localization(
    localization_dir: Path,
    anchor: Mapping[str, Any],
    closing_axis_base: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Load and cross-check selected point clouds before measuring jaw-axis width."""

    frames = anchor.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("live object anchor has no selected-point-cloud provenance")
    artifact_path = _selected_points_artifact_path(localization_dir, anchor)

    points_by_frame: list[np.ndarray] = []
    with np.load(artifact_path, allow_pickle=False) as archive:
        expected_keys: list[str] = []
        for index, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                raise ValueError(f"anchor frame {index} is not an object")
            key = frame.get("selected_points_base_key")
            if not isinstance(key, str) or not key:
                raise ValueError(f"anchor frame {index} has no selected point key")
            expected_keys.append(key)
            if len(set(expected_keys)) != len(expected_keys):
                raise ValueError(f"selected point key {key!r} is duplicated")
            if key not in archive.files:
                raise ValueError(f"selected point artifact is missing array {key!r}")
            points = np.asarray(archive[key], dtype=float)
            if points.ndim != 2 or points.shape[1:] != (3,):
                raise ValueError(f"selected point array {key!r} must have shape (N, 3)")
            if len(points) != int(frame.get("point_count", -1)):
                raise ValueError(f"selected point array {key!r} count differs from anchor")
            if not np.isfinite(points).all():
                raise ValueError(f"selected point array {key!r} contains NaN or infinity")
            if not np.allclose(
                np.median(points, axis=0),
                np.asarray(frame.get("center_base_m"), dtype=float),
                atol=1e-9,
                rtol=0.0,
            ):
                raise ValueError(f"selected point array {key!r} center differs from anchor")
            bounds = frame.get("robust_bounds_p01_p99_base_m")
            if not isinstance(bounds, Mapping):
                raise ValueError(f"anchor frame {index} has no robust point bounds")
            point_bounds = np.quantile(points, [0.01, 0.99], axis=0)
            reported_bounds = np.asarray([bounds.get("min"), bounds.get("max")], dtype=float)
            if reported_bounds.shape != (2, 3) or not np.allclose(
                point_bounds, reported_bounds, atol=1e-9, rtol=0.0
            ):
                raise ValueError(f"selected point array {key!r} bounds differ from anchor")
            points_by_frame.append(points)
        if set(archive.files) != set(expected_keys):
            raise ValueError("selected point artifact arrays differ from anchor frames")

    width, frame_spans = conservative_projected_width(
        points_by_frame,
        closing_axis_base,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )
    return width, {
        "method": "maximum_per_frame_full_projection_span",
        "artifact": str(artifact_path),
        "coordinate_frame": "FR3 base",
        "closing_axis_base": np.asarray(closing_axis_base, dtype=float).tolist(),
        "per_frame_width_m": frame_spans.tolist(),
        "selected_width_m": width,
    }


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


def _three_floats(value: Any, *, name: str) -> list[float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return [float(item) for item in array]


def _positive(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _nonnegative_register(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    number = int(value)
    if number < 0 or float(value) != number:
        raise ValueError(f"{name} must be a non-negative integer")
    return number


def _registration_command(
    normalized: Mapping[str, Any],
    *,
    live_points: Path,
    initial_anchor: Path,
    output: Path,
) -> list[str]:
    """Build registration from only the current run's localization artifacts."""

    template_dir = normalized.get("object_template_dir")
    if not isinstance(template_dir, Path):
        raise ValueError("automatic registration requires inputs.object_template_dir")
    return [
        *normalized["commands"]["register"],
        "--template-dir",
        str(template_dir),
        "--live-points",
        str(live_points.resolve()),
        "--initial-anchor",
        str(initial_anchor.resolve()),
        "--output",
        str(output.resolve()),
    ]


def _validated_live_pose_summary(
    path: Path,
    *,
    expected_object_frame_id: str | None,
    source: str,
    expected_live_points: Path | None = None,
    expected_template_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate a live pose before GROUND and retain its quality evidence."""

    if (expected_live_points is None) != (expected_template_dir is None):
        raise ValueError("automatic registration validation requires both live points and template")
    path = path.resolve()
    if not path.is_file():
        raise StageFailure(f"live object pose is missing: {path}")
    if path.suffix.lower() == ".npy":
        if expected_live_points is not None or expected_template_dir is not None:
            raise StageFailure("automatic registration must produce an evidence JSON")
        if not isinstance(expected_object_frame_id, str) or not expected_object_frame_id:
            raise StageFailure("NPY live pose requires inputs.object_frame_id")
        validate_rigid_transform(np.load(path, allow_pickle=False), name="T_base_object")
        return {
            "source": source,
            "artifact": str(path),
            "format": "npy",
            "object_frame_id": expected_object_frame_id,
            "quality_contract": "explicit_external_read_only_matrix",
        }
    if path.suffix.lower() != ".json":
        raise StageFailure("live object pose must be a JSON or NPY artifact")

    payload = _read_json(path)
    if payload.get("schema") != LIVE_OBJECT_POSE_SCHEMA:
        raise StageFailure(f"live pose JSON schema must be {LIVE_OBJECT_POSE_SCHEMA!r}")
    if payload.get("status") != "accepted":
        raise StageFailure("live object pose status is not accepted")
    if payload.get("frame_convention") != LIVE_POSE_FRAME_CONVENTION:
        raise StageFailure("live object pose frame convention is invalid")
    if payload.get("base_frame_id") != "base":
        raise StageFailure("live object pose base frame must be 'base'")
    object_frame_id = payload.get("object_frame_id")
    if not isinstance(object_frame_id, str) or not object_frame_id:
        raise StageFailure("live object pose has no object_frame_id")
    if expected_object_frame_id is not None and object_frame_id != expected_object_frame_id:
        raise StageFailure("live object pose object_frame_id differs from the config")
    quality = payload.get("quality")
    if not isinstance(quality, Mapping) or quality.get("accepted") is not True:
        raise StageFailure("live object pose did not pass all registration quality gates")
    validate_rigid_transform(payload.get("T_base_object"), name="T_base_object")

    provenance = payload.get("provenance")
    if expected_live_points is not None or expected_template_dir is not None:
        if not isinstance(provenance, Mapping):
            raise StageFailure("automatic registration has no provenance")
        live_points_value = provenance.get("live_points")
        if not isinstance(live_points_value, str) or (
            Path(live_points_value).expanduser().resolve() != expected_live_points.resolve()
        ):
            raise StageFailure(
                "automatic registration did not consume this run's fresh localization points"
            )
        template_manifest_value = provenance.get("template_manifest")
        template_points_value = provenance.get("template_points")
        expected_manifest = (expected_template_dir / "template_manifest.json").resolve()
        expected_points = (expected_template_dir / "object_points_m.npy").resolve()
        if (
            not isinstance(template_manifest_value, str)
            or Path(template_manifest_value).expanduser().resolve() != expected_manifest
            or not isinstance(template_points_value, str)
            or Path(template_points_value).expanduser().resolve() != expected_points
        ):
            raise StageFailure(
                "automatic registration provenance does not match the configured template"
            )

    return {
        "source": source,
        "artifact": str(path),
        "format": "json",
        "schema": payload["schema"],
        "status": payload["status"],
        "object_frame_id": object_frame_id,
        "quality": dict(quality),
        "provenance": dict(provenance) if isinstance(provenance, Mapping) else None,
        "initialization": payload.get("initialization"),
        "orientation_prior": payload.get("orientation_prior"),
    }


def _validate_config(
    config: Mapping[str, Any], *, config_path: Path, execute: bool
) -> dict[str, Any]:
    if config.get("schema") != SCHEMA:
        raise ValueError(f"config schema must be {SCHEMA!r}")
    base = config_path.parent.resolve()
    repo_root = _resolve_path(config.get("repo_root", "."), base=base, name="repo_root")
    inputs = config.get("inputs")
    observation = config.get("observation")
    motion = config.get("motion")
    gripper = config.get("gripper")
    for name, value in (
        ("inputs", inputs),
        ("observation", observation),
        ("motion", motion),
        ("gripper", gripper),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"config.{name} must be an object")

    live_object_pose_value = inputs.get("live_object_pose")
    object_template_dir_value = inputs.get("object_template_dir")
    if (live_object_pose_value is None) == (object_template_dir_value is None):
        raise ValueError(
            "configure exactly one of inputs.live_object_pose or inputs.object_template_dir"
        )

    normalized: dict[str, Any] = {
        "repo_root": repo_root,
        "compiler_report": _resolve_path(
            inputs.get("compiler_report"), base=base, name="inputs.compiler_report"
        ),
        "live_object_pose": (
            _resolve_path(
                live_object_pose_value,
                base=base,
                name="inputs.live_object_pose",
            )
            if live_object_pose_value is not None
            else None
        ),
        "object_template_dir": (
            _resolve_path(
                object_template_dir_value,
                base=base,
                name="inputs.object_template_dir",
            )
            if object_template_dir_value is not None
            else None
        ),
        "object_frame_id": inputs.get("object_frame_id"),
        "t_base_camera": _resolve_path(
            inputs.get("T_base_camera"), base=base, name="inputs.T_base_camera"
        ),
        "t_link8_tcp": _resolve_path(
            inputs.get("T_link8_tcp"), base=base, name="inputs.T_link8_tcp"
        ),
        "trajectory": (
            _resolve_path(inputs["trajectory"], base=base, name="inputs.trajectory")
            if inputs.get("trajectory") is not None
            else None
        ),
        "camera_serial": observation.get("camera_serial"),
        "capture_frames": int(observation.get("frames", 5)),
        "warmup_frames": int(observation.get("warmup_frames", 30)),
        "workspace_min_m": _three_floats(
            observation.get("workspace_min_base_m"),
            name="observation.workspace_min_base_m",
        ),
        "workspace_max_m": _three_floats(
            observation.get("workspace_max_base_m"),
            name="observation.workspace_max_base_m",
        ),
        "support_band_m": observation.get("support_band_base_z_m"),
        "support_z_m": observation.get("support_top_base_z_m"),
        "min_height_m": _positive(
            observation.get("min_height_above_support_m"),
            name="observation.min_height_above_support_m",
        ),
        "lift_delta_base_m": _three_floats(
            motion.get("lift_delta_base_m", [0.0, 0.0, 0.04]),
            name="motion.lift_delta_base_m",
        ),
        "failure_retreat_m": _positive(
            motion.get("failure_retreat_m", 0.08),
            name="motion.failure_retreat_m",
        ),
        "velocity_scale": _positive(
            motion.get("velocity_scale", 0.03), name="motion.velocity_scale"
        ),
        "acceleration_scale": _positive(
            motion.get("acceleration_scale", 0.03),
            name="motion.acceleration_scale",
        ),
        "max_cartesian_speed_m_s": _positive(
            motion.get("max_cartesian_speed_m_s", 0.02),
            name="motion.max_cartesian_speed_m_s",
        ),
        "descend_correction_speed_m_s": _positive(
            motion.get("descend_correction_speed_m_s", 0.01),
            name="motion.descend_correction_speed_m_s",
        ),
        "max_descend_corrections": int(motion.get("max_descend_corrections", 2)),
        "pose_position_tolerance_m": _positive(
            motion.get("pose_position_tolerance_m", 0.001),
            name="motion.pose_position_tolerance_m",
        ),
        "pose_orientation_tolerance_rad": _positive(
            motion.get("pose_orientation_tolerance_rad", 0.005),
            name="motion.pose_orientation_tolerance_rad",
        ),
        "max_pregrasp_translation_m": _positive(
            motion.get("max_pregrasp_translation_m", 0.50),
            name="motion.max_pregrasp_translation_m",
        ),
        "max_pregrasp_rotation_rad": _positive(
            motion.get("max_pregrasp_rotation_rad", math.pi),
            name="motion.max_pregrasp_rotation_rad",
        ),
        "min_joint_limit_margin_rad": _positive(
            motion.get(
                "min_joint_limit_margin_rad",
                DEFAULT_MIN_JOINT_LIMIT_MARGIN_RAD,
            ),
            name="motion.min_joint_limit_margin_rad",
        ),
        "max_joint_sweep_rad": _positive(
            motion.get("max_joint_sweep_rad", DEFAULT_MAX_JOINT_SWEEP_RAD),
            name="motion.max_joint_sweep_rad",
        ),
        "max_total_joint_sweep_rad": _positive(
            motion.get(
                "max_total_joint_sweep_rad",
                DEFAULT_MAX_TOTAL_JOINT_SWEEP_RAD,
            ),
            name="motion.max_total_joint_sweep_rad",
        ),
        "gripper_port": gripper.get("port"),
        "gripper_speed": _nonnegative_register(
            gripper.get("speed_register", 20), name="gripper.speed_register"
        ),
        "gripper_force": _nonnegative_register(
            gripper.get("force_register", 20), name="gripper.force_register"
        ),
        "gripper_timeout_s": _positive(gripper.get("timeout_s", 8.0), name="gripper.timeout_s"),
        "thresholds": PickThresholds.from_mapping(config.get("thresholds")),
    }
    if normalized["capture_frames"] < 3 or normalized["warmup_frames"] < 0:
        raise ValueError("capture requires at least 3 frames and non-negative warmup")
    if np.any(
        np.asarray(normalized["workspace_min_m"]) >= np.asarray(normalized["workspace_max_m"])
    ):
        raise ValueError("workspace min must be strictly below workspace max")
    if (normalized["support_band_m"] is None) == (normalized["support_z_m"] is None):
        raise ValueError("configure exactly one support band or support top")
    if normalized["support_band_m"] is not None:
        band = np.asarray(normalized["support_band_m"], dtype=float)
        if band.shape != (2,) or not np.isfinite(band).all() or band[0] >= band[1]:
            raise ValueError("support band must contain finite min < max")
        normalized["support_band_m"] = band.tolist()
    elif not math.isfinite(float(normalized["support_z_m"])):
        raise ValueError("support top must be finite")
    if normalized["velocity_scale"] > 0.05 or normalized["acceleration_scale"] > 0.05:
        raise ValueError("velocity and acceleration scale are capped at 0.05")
    if normalized["max_cartesian_speed_m_s"] > 0.03:
        raise ValueError("Cartesian speed is capped at 0.03 m/s")
    if normalized["descend_correction_speed_m_s"] > 0.01:
        raise ValueError("descend correction speed is capped at 0.01 m/s")
    if (
        normalized["max_descend_corrections"] not in {1, 2}
        or float(motion.get("max_descend_corrections", 2)) != normalized["max_descend_corrections"]
    ):
        raise ValueError("motion.max_descend_corrections must be 1 or 2")
    if normalized["pose_position_tolerance_m"] > 0.003:
        raise ValueError("pose position tolerance is capped at 0.003 m")
    if normalized["pose_orientation_tolerance_rad"] > 0.05:
        raise ValueError("pose orientation tolerance is capped at 0.05 rad")
    if normalized["max_pregrasp_translation_m"] > 0.60:
        raise ValueError("global pregrasp translation cap cannot exceed 0.60 m")
    if normalized["max_pregrasp_rotation_rad"] > math.pi:
        raise ValueError("global pregrasp rotation cap cannot exceed pi radians")
    validate_fr3_trajectory_audit_thresholds(
        min_joint_limit_margin_rad=normalized[
            "min_joint_limit_margin_rad"
        ],
        max_joint_sweep_rad=normalized["max_joint_sweep_rad"],
        max_total_joint_sweep_rad=normalized[
            "max_total_joint_sweep_rad"
        ],
    )
    if normalized["failure_retreat_m"] > 0.10:
        raise ValueError("failure retreat is capped at 0.10 m")
    if normalized["object_frame_id"] is not None and (
        not isinstance(normalized["object_frame_id"], str)
        or not normalized["object_frame_id"].strip()
    ):
        raise ValueError("inputs.object_frame_id must be a non-empty string")
    if (
        normalized["live_object_pose"] is not None
        and normalized["live_object_pose"].suffix.lower() == ".npy"
    ):
        if (
            not isinstance(normalized["object_frame_id"], str)
            or not normalized["object_frame_id"].strip()
        ):
            raise ValueError("inputs.object_frame_id is required for an NPY live_object_pose")
    if execute and (
        not isinstance(normalized["gripper_port"], str)
        or not Path(normalized["gripper_port"]).is_absolute()
    ):
        raise ValueError("execution requires an absolute gripper.port")

    required_commands = ["localize", "ground", "pose"]
    if normalized["object_template_dir"] is not None:
        required_commands.append("register")
    if execute:
        required_commands.extend(("cartesian", "hande"))
    normalized["commands"] = {name: _command(config, name) for name in required_commands}
    capture = config.get("commands", {}).get("capture")
    if capture is not None:
        normalized["commands"]["capture"] = _command(config, "capture")

    for path_name in (
        "compiler_report",
        "t_base_camera",
        "t_link8_tcp",
    ):
        if not normalized[path_name].is_file():
            raise ValueError(f"missing {path_name}: {normalized[path_name]}")
    if normalized["live_object_pose"] is not None and not normalized["live_object_pose"].is_file():
        raise ValueError(f"missing live_object_pose: {normalized['live_object_pose']}")
    if normalized["object_template_dir"] is not None:
        template_dir = normalized["object_template_dir"]
        if not template_dir.is_dir():
            raise ValueError(f"missing object_template_dir: {template_dir}")
        for name in ("template_manifest.json", "object_points_m.npy"):
            if not (template_dir / name).is_file():
                raise ValueError(
                    f"inputs.object_template_dir is missing required {name}: {template_dir}"
                )
    if normalized["trajectory"] is not None and not normalized["trajectory"].is_file():
        raise ValueError(f"missing trajectory: {normalized['trajectory']}")
    return normalized


def _run_command(
    command: Sequence[str], *, cwd: Path, log_prefix: Path
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    log_prefix.with_suffix(".stdout.log").write_text(completed.stdout, encoding="utf-8")
    log_prefix.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise StageFailure(
            f"command failed with exit {completed.returncode}; "
            f"see {log_prefix.with_suffix('.stderr.log')}"
        )
    return completed


def _artifact_summary(path: Path, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"artifact": str(path.resolve())}
    if payload is not None:
        for key in (
            "schema",
            "status",
            "mode",
            "moveit_error_code",
            "path_fraction",
            "execution_status",
            "execution_error_code",
            "action_status",
        ):
            if key in payload:
                result[key] = payload[key]
    return result


def _append_stage(
    report: dict[str, Any],
    report_path: Path,
    stage: PickStage,
    *,
    status: str,
    details: Mapping[str, Any],
    started_at: str,
) -> None:
    report["stages"].append(
        {
            "name": stage.value,
            "status": status,
            "started_at": started_at,
            "finished_at": _now(),
            "details": dict(details),
        }
    )
    assert_stage_prefix(report["stages"])
    if status == "passed":
        report["last_completed_stage"] = stage.value
    report["updated_at"] = _now()
    atomic_write_json(report_path, report)


def _motion_options(normalized: Mapping[str, Any]) -> list[str]:
    return [
        "--velocity-scale",
        str(normalized["velocity_scale"]),
        "--acceleration-scale",
        str(normalized["acceleration_scale"]),
    ]


def _trajectory_audit_options(normalized: Mapping[str, Any]) -> list[str]:
    return [
        "--min-joint-limit-margin-rad",
        str(normalized["min_joint_limit_margin_rad"]),
        "--max-joint-sweep-rad",
        str(normalized["max_joint_sweep_rad"]),
        "--max-total-joint-sweep-rad",
        str(normalized["max_total_joint_sweep_rad"]),
    ]


def _configured_ee_pose_command(
    normalized: Mapping[str, Any],
    T_base_configured_ee: np.ndarray,
    T_link8_configured_ee: np.ndarray,
) -> list[str]:
    """Build a MoveIt pose command without relabelling the TCP as link8.

    ``moveit_pose_goal.py`` owns the configured-EE to link8 conversion.  Passing
    both rigid transforms here keeps the planner target and the downstream TCP
    feedback checks on the same frame contract.
    """

    target = np.asarray(T_base_configured_ee, dtype=float)
    link8_configured_ee = np.asarray(T_link8_configured_ee, dtype=float)
    target_quaternion = rotation_matrix_to_quaternion_xyzw(target[:3, :3])
    link8_configured_ee_quaternion = rotation_matrix_to_quaternion_xyzw(link8_configured_ee[:3, :3])
    return [
        *normalized["commands"]["pose"],
        "--target-position-m",
        *(str(value) for value in target[:3, 3]),
        "--target-orientation-xyzw",
        *(str(value) for value in target_quaternion),
        "--max-translation-m",
        str(normalized["max_pregrasp_translation_m"]),
        "--max-rotation-rad",
        str(normalized["max_pregrasp_rotation_rad"]),
        "--target-frame",
        "configured-ee",
        "--link8-to-configured-ee-position-m",
        *(str(value) for value in link8_configured_ee[:3, 3]),
        "--link8-to-configured-ee-orientation-xyzw",
        *(str(value) for value in link8_configured_ee_quaternion),
        "--position-tolerance-m",
        str(normalized["pose_position_tolerance_m"]),
        "--orientation-tolerance-rad",
        str(normalized["pose_orientation_tolerance_rad"]),
        *_trajectory_audit_options(normalized),
        *_motion_options(normalized),
    ]


def _hande_command(normalized: Mapping[str, Any], action: str) -> list[str]:
    command = [
        *normalized["commands"]["hande"],
        action,
        "--port",
        normalized["gripper_port"],
    ]
    if action in {"open", "close"}:
        command.extend(
            (
                "--speed",
                str(normalized["gripper_speed"]),
                "--force",
                str(normalized["gripper_force"]),
                "--timeout-s",
                str(normalized["gripper_timeout_s"]),
            )
        )
    return command


def _recover_before_lift(
    normalized: Mapping[str, Any], *, repo_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Best-effort OPEN then +Z retreat after a failed descend/contact stage."""

    recovery: dict[str, Any] = {
        "policy": "open_in_place_then_retreat_base_positive_z",
        "requested_retreat_m": normalized["failure_retreat_m"],
        "open_succeeded": False,
        "retreat_succeeded": False,
    }
    try:
        open_result = _run_command(
            _hande_command(normalized, "open"),
            cwd=repo_root,
            log_prefix=output_dir / "recovery_hande_open",
        )
        open_payload = json.loads(open_result.stdout)
        open_path = output_dir / "recovery_hande_open.json"
        atomic_write_json(open_path, open_payload)
        recovery["open_artifact"] = str(open_path)
        recovery["open_succeeded"] = hande_open_succeeded(open_payload, normalized["thresholds"])
    except Exception as exc:
        recovery["open_error"] = str(exc)
        return recovery
    if not recovery["open_succeeded"]:
        recovery["retreat_skipped"] = "open did not reach the safe width"
        return recovery

    retreat_path = output_dir / "recovery_retreat.json"
    try:
        _run_command(
            [
                *normalized["commands"]["cartesian"],
                "--delta-z-m",
                str(normalized["failure_retreat_m"]),
                *_motion_options(normalized),
                *_trajectory_audit_options(normalized),
                "--max-cartesian-speed-m-s",
                str(normalized["max_cartesian_speed_m_s"]),
                "--output",
                str(retreat_path),
                "--execute",
                "--accept-arm-only-collision-model",
            ],
            cwd=repo_root,
            log_prefix=output_dir / "recovery_retreat",
        )
        retreat_payload = _read_json(retreat_path)
        recovery["retreat_artifact"] = str(retreat_path)
        recovery["retreat_succeeded"] = cartesian_result_succeeded(retreat_payload, executed=True)
    except Exception as exc:
        recovery["retreat_error"] = str(exc)
    return recovery


def run(args: argparse.Namespace) -> int:
    if args.execute and not args.accept_arm_only_collision_model:
        raise ValueError("--execute requires --accept-arm-only-collision-model")
    if not args.execute and args.accept_arm_only_collision_model:
        raise ValueError("collision-model acknowledgement is only valid with --execute")
    config_path = args.config.expanduser().resolve()
    config = _read_json(config_path)
    normalized = _validate_config(config, config_path=config_path, execute=args.execute)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    report_path = output_dir / "stage_report.json"
    thresholds: PickThresholds = normalized["thresholds"]
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
        "stage_order": [stage.value for stage in PickStage],
        "transition_thresholds": thresholds.to_dict(),
        "resolved_motion_parameters": {
            "grasp_endpoint_source": "selected_object_relative_se3",
            "approach_source": "selected_gripper_tcp_local_plus_z",
            "lift_delta_base_m": normalized["lift_delta_base_m"],
            "failure_retreat_m": normalized["failure_retreat_m"],
            "velocity_scale": normalized["velocity_scale"],
            "acceleration_scale": normalized["acceleration_scale"],
            "max_cartesian_speed_m_s": normalized["max_cartesian_speed_m_s"],
            "descend_correction_speed_m_s": normalized["descend_correction_speed_m_s"],
            "max_descend_corrections": normalized["max_descend_corrections"],
            "pose_position_tolerance_m": normalized["pose_position_tolerance_m"],
            "pose_orientation_tolerance_rad": normalized["pose_orientation_tolerance_rad"],
            "max_pregrasp_translation_m": normalized["max_pregrasp_translation_m"],
            "max_pregrasp_rotation_rad": normalized["max_pregrasp_rotation_rad"],
            "min_joint_limit_margin_rad": normalized[
                "min_joint_limit_margin_rad"
            ],
            "max_joint_sweep_rad": normalized["max_joint_sweep_rad"],
            "max_total_joint_sweep_rad": normalized[
                "max_total_joint_sweep_rad"
            ],
            "gripper_speed_register": normalized["gripper_speed"],
            "gripper_force_register": normalized["gripper_force"],
        },
        "stages": [],
        "last_completed_stage": None,
    }
    atomic_write_json(report_path, report)
    active_stage: PickStage | None = None
    active_started = _now()
    repo_root: Path = normalized["repo_root"]

    try:
        active_stage = PickStage.OBSERVE
        active_started = _now()
        if args.input_dir is not None:
            input_dir = args.input_dir.expanduser().resolve()
            if not input_dir.is_dir():
                raise StageFailure(f"RGB-D input directory does not exist: {input_dir}")
            observation_details = {"source": "existing_capture", "input_dir": str(input_dir)}
        else:
            camera_serial = normalized["camera_serial"]
            capture_command = normalized["commands"].get("capture")
            if not isinstance(camera_serial, str) or not camera_serial:
                raise StageFailure("live OBSERVE requires observation.camera_serial")
            if capture_command is None:
                raise StageFailure("live OBSERVE requires commands.capture")
            input_dir = output_dir / "input"
            command = [
                *capture_command,
                "--serial",
                camera_serial,
                "--output-dir",
                str(input_dir),
                "--frames",
                str(normalized["capture_frames"]),
                "--warmup-frames",
                str(normalized["warmup_frames"]),
            ]
            _run_command(command, cwd=repo_root, log_prefix=output_dir / "observe")
            observation_details = {
                "source": "live_realsense",
                "camera_serial": camera_serial,
                "input_dir": str(input_dir),
                "frame_count": normalized["capture_frames"],
            }

        localization_dir = output_dir / "object_localization"
        localize_command = [
            *normalized["commands"]["localize"],
            "--input-dir",
            str(input_dir),
            "--t-base-camera",
            str(normalized["t_base_camera"]),
            "--output-dir",
            str(localization_dir),
            "--workspace-min-m",
            *(str(value) for value in normalized["workspace_min_m"]),
            "--workspace-max-m",
            *(str(value) for value in normalized["workspace_max_m"]),
            "--min-height-above-support-m",
            str(normalized["min_height_m"]),
            "--max-center-deviation-m",
            str(thresholds.max_anchor_center_deviation_m),
            "--max-point-count-cv",
            str(thresholds.max_anchor_point_count_cv),
        ]
        if normalized["support_band_m"] is not None:
            localize_command.extend(
                [
                    "--support-band-m",
                    *(str(value) for value in normalized["support_band_m"]),
                ]
            )
        else:
            localize_command.extend(["--support-z-m", str(normalized["support_z_m"])])
        _run_command(localize_command, cwd=repo_root, log_prefix=output_dir / "localize")
        anchor_path = localization_dir / "live_object_anchor.json"
        anchor = _read_json(anchor_path)
        if (
            anchor.get("status") != "accepted"
            or anchor.get("quality", {}).get("accepted") is not True
        ):
            raise StageFailure("live object localization did not pass all quality gates")
        observation_details["localization"] = _artifact_summary(anchor_path, anchor)
        observation_details["center_base_m"] = anchor.get("center_base_m")
        selected_points_path = _selected_points_artifact_path(localization_dir, anchor)
        if normalized["object_template_dir"] is not None:
            live_object_pose_path = output_dir / "registration" / "live_object_pose.json"
            register_command = _registration_command(
                normalized,
                live_points=selected_points_path,
                initial_anchor=anchor_path,
                output=live_object_pose_path,
            )
            _run_command(
                register_command,
                cwd=repo_root,
                log_prefix=output_dir / "register",
            )
            live_pose_details = _validated_live_pose_summary(
                live_object_pose_path,
                expected_object_frame_id=normalized["object_frame_id"],
                source="automatic_registration_from_current_observation",
                expected_live_points=selected_points_path,
                expected_template_dir=normalized["object_template_dir"],
            )
        else:
            live_object_pose_path = normalized["live_object_pose"]
            live_pose_details = _validated_live_pose_summary(
                live_object_pose_path,
                expected_object_frame_id=normalized["object_frame_id"],
                source="explicit_external_read_only_input",
            )
        observation_details["live_object_pose"] = live_pose_details
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=observation_details,
            started_at=active_started,
        )

        active_stage = PickStage.GROUND
        active_started = _now()
        candidate_path = output_dir / "grounded_candidate.json"
        ground_command = [
            *normalized["commands"]["ground"],
            "--compiler-report",
            str(normalized["compiler_report"]),
            "--live-object-pose",
            str(live_object_pose_path),
            "--live-object-anchor",
            str(anchor_path),
            "--agent-lift-delta-base-m",
            *(str(value) for value in normalized["lift_delta_base_m"]),
            "--output",
            str(candidate_path),
        ]
        if normalized["object_frame_id"] is not None:
            ground_command.extend(("--object-frame-id", str(normalized["object_frame_id"])))
        if normalized["trajectory"] is not None:
            ground_command.extend(("--trajectory", str(normalized["trajectory"])))
        _run_command(ground_command, cwd=repo_root, log_prefix=output_dir / "ground")
        grounded_candidate = _read_json(candidate_path)
        candidate = prepare_execution_candidate(
            grounded_candidate,
            anchor_z_m=None,
            grasp_tcp_vertical_offset_m=None,
            lift_delta_base_m=np.asarray(normalized["lift_delta_base_m"], dtype=float),
        )
        execution_candidate_path = output_dir / "execution_candidate.json"
        atomic_write_json(execution_candidate_path, candidate)
        T_link8_tcp = np.load(normalized["t_link8_tcp"], allow_pickle=False)
        grasp_transform = np.asarray(candidate["candidate"]["T_base_grasp_tcp"], dtype=float)
        projected_width, projected_width_evidence = _projected_width_from_localization(
            localization_dir, anchor, grasp_transform[:3, 0]
        )
        geometry = validate_pick_geometry(
            anchor,
            candidate,
            T_link8_tcp,
            thresholds,
            object_width_along_closing_axis_m=projected_width,
        )
        geometry_path = output_dir / "validated_geometry.json"
        atomic_write_json(
            geometry_path,
            {
                "schema": "demo2gripper.validated_pick_geometry.v1",
                "status": "accepted",
                "thresholds": thresholds.to_dict(),
                "geometry": geometry.to_dict(),
                "object_width_estimation": projected_width_evidence,
                "provenance": {
                    "anchor": str(anchor_path),
                    "live_object_pose": str(live_object_pose_path),
                    "grounded_candidate": str(candidate_path),
                    "execution_candidate": str(execution_candidate_path),
                    "T_link8_tcp": str(normalized["t_link8_tcp"]),
                },
            },
        )
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "grounded_candidate": _artifact_summary(candidate_path, grounded_candidate),
                "execution_candidate": _artifact_summary(execution_candidate_path, candidate),
                "validated_geometry": str(geometry_path),
                "live_object_pose": live_pose_details,
                "object_width_along_closing_axis_m": (geometry.object_width_along_closing_axis_m),
                "approach_delta_base_m": geometry.approach_delta_base_m.tolist(),
            },
            started_at=active_started,
        )

        active_stage = PickStage.PLAN
        active_started = _now()
        pregrasp_link8_target = geometry.T_base_pregrasp_link8
        pregrasp_tcp_target = geometry.T_base_pregrasp_tcp
        pregrasp_tcp_quaternion = rotation_matrix_to_quaternion_xyzw(pregrasp_tcp_target[:3, :3])
        pregrasp_link8_quaternion = rotation_matrix_to_quaternion_xyzw(
            pregrasp_link8_target[:3, :3]
        )
        plan_path = output_dir / "pregrasp_plan.json"
        pose_base_command = _configured_ee_pose_command(
            normalized, pregrasp_tcp_target, T_link8_tcp
        )
        _run_command(
            [*pose_base_command, "--output", str(plan_path)],
            cwd=repo_root,
            log_prefix=output_dir / "pregrasp_plan",
        )
        plan_payload = _read_json(plan_path)
        if not pose_result_succeeded(plan_payload, executed=False):
            raise StageFailure("MoveIt did not produce a valid global pregrasp plan")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                **_artifact_summary(plan_path, plan_payload),
                "plan_role": "reachability_precheck",
                "execution_policy": (
                    "PREGRASP replans from measured state immediately before execution; "
                    "moveit_pose_goal then audits the complete returned joint path and "
                    "sends that exact path through ExecuteTrajectory"
                ),
                "scope": "later stages passed static geometry predicates only",
                "T_base_pregrasp_link8": pregrasp_link8_target.tolist(),
                "configured_ee_target_position_base_m": (pregrasp_tcp_target[:3, 3].tolist()),
                "configured_ee_target_orientation_xyzw": (pregrasp_tcp_quaternion.tolist()),
                "link8_target_orientation_xyzw": (pregrasp_link8_quaternion.tolist()),
                "T_link8_configured_ee": T_link8_tcp.tolist(),
            },
            started_at=active_started,
        )

        if not args.execute:
            report["status"] = "review_ready"
            report["finished_at"] = _now()
            report["not_executed"] = [
                stage.value
                for stage in (
                    PickStage.PREGRASP,
                    PickStage.DESCEND,
                    PickStage.CLOSE,
                    PickStage.VERIFY_CONTACT,
                    PickStage.LIFT,
                )
            ]
            atomic_write_json(report_path, report)
            print(f"review-ready plan saved: {report_path}")
            return 0

        active_stage = PickStage.PREGRASP
        active_started = _now()
        open_log = output_dir / "hande_open"
        open_result = _run_command(
            _hande_command(normalized, "open"), cwd=repo_root, log_prefix=open_log
        )
        open_payload = json.loads(open_result.stdout)
        atomic_write_json(output_dir / "hande_open.json", open_payload)
        if not hande_open_succeeded(open_payload, thresholds):
            raise StageFailure("Hand-E did not reach the required preshape opening")

        execute_path = output_dir / "pregrasp_execution.json"
        _run_command(
            [
                *pose_base_command,
                "--output",
                str(execute_path),
                "--execute",
                "--accept-arm-only-collision-model",
            ],
            cwd=repo_root,
            log_prefix=output_dir / "pregrasp_execution",
        )
        execute_payload = _read_json(execute_path)
        if not pose_result_succeeded(execute_payload, executed=True):
            raise StageFailure("global pregrasp execution did not succeed")
        within_tolerance, pose_errors = measured_pose_within_tolerance(execute_payload, thresholds)
        if not within_tolerance:
            raise StageFailure(f"measured pregrasp pose missed tolerance: {pose_errors}")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "preshape": _artifact_summary(output_dir / "hande_open.json", open_payload),
                "motion": _artifact_summary(execute_path, execute_payload),
                "measured_pose_errors": pose_errors,
            },
            started_at=active_started,
        )

        active_stage = PickStage.DESCEND
        active_started = _now()
        approach_path = output_dir / "approach_execution.json"
        delta = geometry.approach_delta_base_m
        cartesian_options = [
            *_motion_options(normalized),
            *_trajectory_audit_options(normalized),
            "--max-cartesian-speed-m-s",
            str(normalized["max_cartesian_speed_m_s"]),
        ]
        _run_command(
            [
                *normalized["commands"]["cartesian"],
                "--delta-x-m",
                str(delta[0]),
                "--delta-y-m",
                str(delta[1]),
                "--delta-z-m",
                str(delta[2]),
                *cartesian_options,
                "--output",
                str(approach_path),
                "--execute",
                "--accept-arm-only-collision-model",
            ],
            cwd=repo_root,
            log_prefix=output_dir / "approach_execution",
        )
        approach_payload = _read_json(approach_path)
        if not cartesian_result_succeeded(approach_payload, executed=True):
            raise StageFailure("Cartesian DESCEND did not completely execute")
        measured_delta = _three_floats(
            approach_payload.get("measured_translation_base_m"),
            name="measured approach translation",
        )
        approach_error = float(np.linalg.norm(np.asarray(measured_delta) - delta))
        descend_summary_path = output_dir / "descend_closed_loop.json"
        descend_summary: dict[str, Any] = {
            "schema": "demo2gripper.descend_closed_loop.v1",
            "status": "running",
            "target_tcp_position_base_m": geometry.T_base_grasp_tcp[:3, 3].tolist(),
            "support_top_base_z_m": geometry.support_top_base_z_m,
            "max_corrections": normalized["max_descend_corrections"],
            "correction_speed_m_s": normalized["descend_correction_speed_m_s"],
            "initial_motion": {
                **_artifact_summary(approach_path, approach_payload),
                "requested_delta_base_m": delta.tolist(),
                "measured_translation_error_m": approach_error,
            },
            "corrections": [],
        }

        def evaluate_cartesian_end(
            payload: Mapping[str, Any],
        ) -> tuple[np.ndarray | None, dict[str, Any]]:
            measured_tcp = cartesian_measured_end_tcp_transform(payload, T_link8_tcp)[:3, 3]
            try:
                return evaluate_descend_tcp_residual(
                    measured_tcp,
                    geometry.T_base_grasp_tcp[:3, 3],
                    support_top_base_z_m=geometry.support_top_base_z_m,
                    thresholds=thresholds,
                )
            except ValueError as exc:
                descend_summary["status"] = "rejected"
                descend_summary["guard_error"] = str(exc)
                atomic_write_json(descend_summary_path, descend_summary)
                raise StageFailure(f"DESCEND absolute TCP guard failed: {exc}") from exc

        correction, residual_details = evaluate_cartesian_end(approach_payload)
        descend_summary["initial_measurement"] = residual_details
        atomic_write_json(descend_summary_path, descend_summary)

        correction_options = [
            *_motion_options(normalized),
            *_trajectory_audit_options(normalized),
            "--max-cartesian-speed-m-s",
            str(normalized["descend_correction_speed_m_s"]),
        ]
        for index in range(1, normalized["max_descend_corrections"] + 1):
            if correction is None:
                break
            correction_path = output_dir / f"descend_correction_{index:02d}.json"
            requested_correction = correction.copy()
            correction_record: dict[str, Any] = {
                "index": index,
                "status": "commanding",
                "artifact": str(correction_path.resolve()),
                "requested_delta_base_m": requested_correction.tolist(),
            }
            descend_summary["corrections"].append(correction_record)
            atomic_write_json(descend_summary_path, descend_summary)
            _run_command(
                [
                    *normalized["commands"]["cartesian"],
                    "--delta-x-m",
                    str(requested_correction[0]),
                    "--delta-y-m",
                    str(requested_correction[1]),
                    "--delta-z-m",
                    str(requested_correction[2]),
                    *correction_options,
                    "--output",
                    str(correction_path),
                    "--execute",
                    "--accept-arm-only-collision-model",
                ],
                cwd=repo_root,
                log_prefix=output_dir / f"descend_correction_{index:02d}",
            )
            correction_payload = _read_json(correction_path)
            if not cartesian_result_succeeded(correction_payload, executed=True):
                raise StageFailure(f"DESCEND correction {index} did not completely execute")
            correction_record.update(_artifact_summary(correction_path, correction_payload))
            correction_record["status"] = "executed"
            correction, residual_details = evaluate_cartesian_end(correction_payload)
            correction_record["measurement_after"] = residual_details
            atomic_write_json(descend_summary_path, descend_summary)

        if correction is not None:
            descend_summary["status"] = "rejected"
            descend_summary["guard_error"] = (
                "final TCP residual remains above tolerance after the correction limit"
            )
            descend_summary["final_measurement"] = residual_details
            atomic_write_json(descend_summary_path, descend_summary)
            raise StageFailure(descend_summary["guard_error"])

        descend_summary["status"] = "accepted"
        descend_summary["final_measurement"] = residual_details
        atomic_write_json(descend_summary_path, descend_summary)
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "initial_motion": descend_summary["initial_motion"],
                "closed_loop": _artifact_summary(descend_summary_path, descend_summary),
                "correction_count": len(descend_summary["corrections"]),
                "final_measurement": residual_details,
            },
            started_at=active_started,
        )

        active_stage = PickStage.CLOSE
        active_started = _now()
        close_result = _run_command(
            _hande_command(normalized, "close"),
            cwd=repo_root,
            log_prefix=output_dir / "hande_close",
        )
        close_payload = json.loads(close_result.stdout)
        close_path = output_dir / "hande_close.json"
        atomic_write_json(close_path, close_payload)
        if not hande_contact_verified(close_payload, thresholds):
            raise StageFailure("Hand-E close completed without detecting an object")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=_artifact_summary(close_path, close_payload),
            started_at=active_started,
        )

        active_stage = PickStage.VERIFY_CONTACT
        active_started = _now()
        if not hande_contact_verified(close_payload, thresholds):
            raise StageFailure("Hand-E close feedback did not retain its contact evidence")
        close_opening_m = float(close_payload["state"]["opening_width_mm"]) * 1e-3
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                **_artifact_summary(close_path, close_payload),
                "contact_evidence": "immediate_close_result",
                "opening_width_m": close_opening_m,
                "reconnect_semantics": (
                    "Hand-E gOBJ resets after reconnect; a separate status command is "
                    "not used as pre-lift contact evidence"
                ),
            },
            started_at=active_started,
        )

        active_stage = PickStage.LIFT
        active_started = _now()
        segment_summaries = []
        for index, lift_delta in enumerate(geometry.lift_deltas_base_m, start=1):
            segment_path = output_dir / f"lift_segment_{index:02d}.json"
            _run_command(
                [
                    *normalized["commands"]["cartesian"],
                    "--delta-x-m",
                    str(lift_delta[0]),
                    "--delta-y-m",
                    str(lift_delta[1]),
                    "--delta-z-m",
                    str(lift_delta[2]),
                    *cartesian_options,
                    "--output",
                    str(segment_path),
                    "--execute",
                    "--accept-arm-only-collision-model",
                ],
                cwd=repo_root,
                log_prefix=output_dir / f"lift_segment_{index:02d}",
            )
            segment_payload = _read_json(segment_path)
            if not cartesian_result_succeeded(segment_payload, executed=True):
                raise StageFailure(f"LIFT segment {index} did not completely execute")
            measured = np.asarray(
                _three_floats(
                    segment_payload.get("measured_translation_base_m"),
                    name=f"measured LIFT segment {index}",
                )
            )
            error = float(np.linalg.norm(measured - lift_delta))
            if error > thresholds.max_measured_position_error_m:
                raise StageFailure(
                    f"LIFT segment {index} measured error {error:.6f} m exceeds tolerance"
                )
            segment_summaries.append(
                {
                    **_artifact_summary(segment_path, segment_payload),
                    "requested_delta_base_m": lift_delta.tolist(),
                    "measured_translation_error_m": error,
                }
            )

        final_status_result = _run_command(
            _hande_command(normalized, "status"),
            cwd=repo_root,
            log_prefix=output_dir / "hande_post_lift",
        )
        final_status = json.loads(final_status_result.stdout)
        final_status_path = output_dir / "hande_post_lift.json"
        atomic_write_json(final_status_path, final_status)
        if not hande_width_snapshot_valid(
            final_status,
            thresholds,
            reference_opening_m=close_opening_m,
            max_drift_m=0.002,
        ):
            raise StageFailure("Hand-E post-lift width/fault snapshot is inconsistent")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "segments": segment_summaries,
                "post_lift_contact": _artifact_summary(final_status_path, final_status),
                "post_lift_verification": (
                    "fault-free width in grasp range and within 2 mm of close result; "
                    "gOBJ intentionally ignored after reconnect"
                ),
            },
            started_at=active_started,
        )
        report["status"] = "succeeded"
        report["finished_at"] = _now()
        report["object_state"] = "held_after_robot_specific_short_lift"
        atomic_write_json(report_path, report)
        print(f"autonomous pick succeeded: {report_path}")
        return 0
    except Exception as exc:
        recovery = None
        if args.execute and active_stage in {
            PickStage.DESCEND,
            PickStage.CLOSE,
            PickStage.VERIFY_CONTACT,
        }:
            recovery = _recover_before_lift(normalized, repo_root=repo_root, output_dir=output_dir)
        if active_stage is not None and len(report["stages"]) < len(PickStage):
            expected_stage = list(PickStage)[len(report["stages"])]
            if active_stage is expected_stage:
                failure_details: dict[str, Any] = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                descend_summary_path = output_dir / "descend_closed_loop.json"
                if active_stage is PickStage.DESCEND and descend_summary_path.is_file():
                    failure_details["closed_loop"] = _artifact_summary(
                        descend_summary_path, _read_json(descend_summary_path)
                    )
                _append_stage(
                    report,
                    report_path,
                    active_stage,
                    status="failed",
                    details=failure_details,
                    started_at=active_started,
                )
        report["status"] = "failed"
        report["failed_stage"] = active_stage.value if active_stage is not None else None
        report["failure"] = {"error_type": type(exc).__name__, "error": str(exc)}
        if recovery is not None:
            report["recovery"] = recovery
        report["finished_at"] = _now()
        atomic_write_json(report_path, report)
        print(f"autonomous pick failed at {report['failed_stage']}: {exc}", file=sys.stderr)
        return 2


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception:  # pragma: no cover - preserves diagnostics for integration failures
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
