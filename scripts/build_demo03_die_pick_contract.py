#!/usr/bin/env python3
"""Build demo-03's review-only die-pick SE(3) contract and config template.

The reverse-placement slice supplies grasp geometry evidence only.  This builder
does not compile or replay its hand path, localize the current die, invoke ROS,
or authorize robot execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from demo2gripper.franka import validate_rigid_transform
from demo2gripper.task_skill import (
    AnchorFrame,
    FailurePolicy,
    ObjectGraspRelation,
    PlannerMode,
    SkillStage,
    SkillStatus,
    StageName,
    StageTargetPolicy,
    TaskSkill,
)

REPORT_SCHEMA = "demo2gripper.agentic_skill_compiler_report.v1"
MANIFEST_SCHEMA = "demo2gripper.demo03_die_pick_contract_manifest.v1"
CONFIG_SCHEMA = "demo2gripper.autonomous_pick_config.v1"
SELECTED_FRAME_CONVENTION = "T_object_gripper_tcp; +X closing, +Z approach"
SELECTED_GRIPPER = "robotiq_hande"
INTENT_SCHEMA = "demo2gripper.robot_adapted_grasp_intent.v1"
INTENT_ROLE = "robot_adapted_hande"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--demo-dir", required=True, type=Path)
    parser.add_argument("--selected-grasp", type=Path)
    parser.add_argument("--selection-provenance", type=Path)
    parser.add_argument("--pickup-result", type=Path)
    parser.add_argument("--reverse-provenance", type=Path)
    parser.add_argument("--source-timestamps", type=Path)
    parser.add_argument("--object-template-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config-template", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: Path, *, repo_root: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _display(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _load_selected_grasp(path: Path) -> tuple[dict[str, Any], np.ndarray, float, float]:
    payload = _read_json(path)
    if payload.get("frame_convention") != SELECTED_FRAME_CONVENTION:
        raise ValueError(f"selected grasp must use {SELECTED_FRAME_CONVENTION!r}")
    if payload.get("gripper_name") != SELECTED_GRIPPER:
        raise ValueError(f"selected grasp gripper must be {SELECTED_GRIPPER!r}")
    object_frame_id = payload.get("object_frame_id")
    if not isinstance(object_frame_id, str) or not object_frame_id.strip():
        raise ValueError("selected grasp needs a non-empty object_frame_id")
    intent = payload.get("intent_provenance")
    if not isinstance(intent, Mapping) or (
        intent.get("schema") != INTENT_SCHEMA
        or intent.get("intent_role") != INTENT_ROLE
    ):
        raise ValueError("selected grasp lacks the robot-adapted Hand-E intent contract")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping) or selection.get("hard_gates_passed") is not True:
        raise ValueError("selected grasp did not pass its recorded hard gates")
    thresholds = selection.get("hard_gate_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("selected grasp lacks hard-gate thresholds")

    transform = validate_rigid_transform(
        payload.get("T_object_gripper_tcp"), name="T_object_gripper_tcp"
    )
    width = _finite_number(payload.get("pregrasp_width_m"), name="pregrasp_width_m")
    approach = _finite_number(
        payload.get("approach_distance_m"), name="approach_distance_m"
    )
    if not 0.0 < width <= 0.05:
        raise ValueError("Hand-E pregrasp width must lie in (0, 0.05] m")
    if approach <= 0.0:
        raise ValueError("approach distance must be positive")

    checks = (
        (width, "min_gripper_width_m", lambda value, limit: value >= limit),
        (width, "max_gripper_width_m", lambda value, limit: value <= limit),
        (
            _finite_number(payload.get("center_error_m"), name="center_error_m"),
            "max_center_distance_m",
            lambda value, limit: value <= limit,
        ),
        (
            _finite_number(
                payload.get("closing_axis_error_rad"), name="closing_axis_error_rad"
            ),
            "max_closing_axis_error_rad",
            lambda value, limit: value <= limit,
        ),
        (
            _finite_number(
                payload.get("approach_axis_error_rad"), name="approach_axis_error_rad"
            ),
            "max_approach_axis_error_rad",
            lambda value, limit: value <= limit,
        ),
    )
    for measured, threshold_name, predicate in checks:
        threshold = _finite_number(thresholds.get(threshold_name), name=threshold_name)
        if not predicate(measured, threshold):
            raise ValueError(f"selected grasp violates {threshold_name}")
    return payload, transform, width, approach


def _critical_proxy_interval(
    pickup: Mapping[str, Any],
    reverse: Mapping[str, Any],
    timestamps: np.ndarray,
) -> tuple[tuple[int, int], tuple[float, float], list[int]]:
    if pickup.get("pickup_demonstration_validated") is not True:
        raise ValueError("reverse proxy pickup relation is not validated")
    if reverse.get("not_original_pickup") is not True:
        raise ValueError("reverse provenance must explicitly mark not_original_pickup=true")
    interval = pickup.get("critical_grasp_interval")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in interval)
    ):
        raise ValueError("pickup critical_grasp_interval must contain two integers")
    start, end = interval
    source_indices = reverse.get("source_frame_indices")
    if (
        start < 0
        or end <= start
        or not isinstance(source_indices, list)
        or end >= len(source_indices)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in source_indices)
    ):
        raise ValueError("reverse proxy interval or source-frame map is invalid")
    if timestamps.ndim != 1 or len(timestamps) <= max(source_indices):
        raise ValueError("source timestamps do not cover the reverse proxy frame map")
    if not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("source timestamps must be finite and strictly increasing")

    mapped = np.asarray(source_indices, dtype=int)
    proxy_times = timestamps[mapped[0]] - timestamps[mapped]
    if not np.all(np.diff(proxy_times) > 0.0):
        raise ValueError("reverse proxy frame map does not yield increasing time")
    return (
        (start, end),
        (float(proxy_times[start]), float(proxy_times[end])),
        source_indices[start:end],
    )


def _config_template(
    *,
    repo_root: Path,
    compiler_report: Path,
    object_template_dir: Path,
    object_frame_id: str,
) -> dict[str, Any]:
    python = repo_root / ".venv" / "bin" / "python"
    return {
        "schema": CONFIG_SCHEMA,
        "_template": {
            "status": "requires_fresh_live_workspace_measurement",
            "execution_authorized": False,
            "missing_fields": [
                "observation.workspace_min_base_m",
                "observation.workspace_max_base_m",
                (
                    "exactly one of observation.support_band_base_z_m or "
                    "observation.support_top_base_z_m"
                ),
                "thresholds.min_object_height_above_support_m",
                "thresholds.max_object_height_above_support_m",
                "thresholds.min_anchor_z_m",
                "thresholds.max_anchor_z_m",
                "thresholds.tcp_xy_min_base_m",
                "thresholds.tcp_xy_max_base_m",
                "thresholds.object_extent_min_m",
                "thresholds.object_extent_max_m",
            ],
            "instruction": (
                "Fill bounds from a fresh post-lid-opening RGB-D capture, then run "
                "scripts/run_autonomous_pick.py without --execute and review registration, "
                "grounding, geometry gates, and the MoveIt plan."
            ),
            "fixed_scene_assumption": (
                "the die and camera have not rotated relative to the demonstration; "
                "the registration orientation prior encodes that assumption"
            ),
            "reverse_proxy_limit": (
                "the selected grasp was ranked from a time-reversed placement proxy, "
                "not an observed original pickup"
            ),
        },
        "repo_root": str(repo_root),
        "commands": {
            "capture": [
                "/home/vitai/miniconda3/envs/fhh/bin/python",
                "scripts/capture_realsense_rgbd.py",
            ],
            "localize": [str(python), "scripts/localize_supported_object.py"],
            "register": [
                str(python),
                "scripts/register_live_object_pose.py",
                "--orientation-prior",
                str((repo_root / "calibration" / "T_base_camera.npy").resolve()),
                "--orientation-prior-source",
                (
                    "demo03 fixed-scene assumption: template axes equal demonstration "
                    "camera axes and the die has not rotated"
                ),
                "--lock-orientation-to-prior",
            ],
            "ground": [str(python), "scripts/ground_live_grasp_candidate.py"],
            "pose": ["scripts/run_ros_python.sh", "scripts/moveit_pose_goal.py"],
            "cartesian": [
                "scripts/run_ros_python.sh",
                "scripts/moveit_cartesian_smoke.py",
            ],
            "hande": [
                "/usr/bin/env",
                "PYTHONPATH=/home/vitai/fhh/lerobot/src",
                "/home/vitai/miniconda3/envs/fhh/bin/python",
                "scripts/hande_cli.py",
            ],
        },
        "inputs": {
            "compiler_report": str(compiler_report.resolve()),
            "object_template_dir": str(object_template_dir.resolve()),
            "object_frame_id": object_frame_id,
            "T_base_camera": str(
                (repo_root / "calibration" / "T_base_camera.npy").resolve()
            ),
            "T_link8_tcp": str(
                (repo_root / "calibration" / "T_link8_canonical_tcp.npy").resolve()
            ),
        },
        "observation": {
            "camera_serial": "254622073804",
            "frames": 5,
            "warmup_frames": 30,
            "workspace_min_base_m": None,
            "workspace_max_base_m": None,
            "support_band_base_z_m": None,
            "support_top_base_z_m": None,
            "min_height_above_support_m": 0.008,
        },
        "motion": {
            "lift_delta_base_m": [0.0, 0.0, 0.04],
            "failure_retreat_m": 0.08,
            "velocity_scale": 0.03,
            "acceleration_scale": 0.03,
            "max_cartesian_speed_m_s": 0.02,
            "descend_correction_speed_m_s": 0.01,
            "max_descend_corrections": 2,
            "pose_position_tolerance_m": 0.001,
            "pose_orientation_tolerance_rad": 0.005,
            "max_pregrasp_translation_m": 0.5,
            "max_pregrasp_rotation_rad": float(np.pi),
        },
        "gripper": {
            "port": "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AQ00O97B-if00-port0",
            "speed_register": 20,
            "force_register": 20,
            "timeout_s": 8.0,
        },
        "thresholds": {
            "min_object_height_above_support_m": None,
            "max_object_height_above_support_m": None,
            "min_anchor_z_m": None,
            "max_anchor_z_m": None,
            "tcp_xy_min_base_m": None,
            "tcp_xy_max_base_m": None,
            "object_extent_min_m": None,
            "object_extent_max_m": None,
        },
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    demo_dir = _resolve(args.demo_dir, repo_root=repo_root)
    reverse_dir = demo_dir / "derived" / "reverse_pick_proxy_0650_to_0440"
    selected_path = _resolve(
        args.selected_grasp
        or reverse_dir / "graspgenx_hande" / "selected_grasp.json",
        repo_root=repo_root,
    )
    selection_provenance_path = _resolve(
        args.selection_provenance
        or reverse_dir / "graspgenx_hande" / "selection_provenance.json",
        repo_root=repo_root,
    )
    pickup_path = _resolve(
        args.pickup_result or reverse_dir / "pickup_result" / "pickup_grasp.json",
        repo_root=repo_root,
    )
    reverse_provenance_path = _resolve(
        args.reverse_provenance or reverse_dir / "provenance.json",
        repo_root=repo_root,
    )
    timestamps_path = _resolve(
        args.source_timestamps or demo_dir / "input" / "timestamps_s.npy",
        repo_root=repo_root,
    )
    template_dir = _resolve(
        args.object_template_dir or reverse_dir / "object_intent",
        repo_root=repo_root,
    )
    output_dir = _resolve(
        args.output_dir
        or demo_dir / "multicontact_plan" / "inputs" / "die_pick_contract",
        repo_root=repo_root,
    )
    config_path = _resolve(
        args.config_template
        or demo_dir / "multicontact_plan" / "inputs" / "die_pick_config.template.json",
        repo_root=repo_root,
    )
    skill_path = output_dir / "agentic_skill.json"
    report_path = output_dir / "compiler_report.json"
    manifest_path = output_dir / "manifest.json"
    outputs = (skill_path, report_path, manifest_path, config_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise ValueError(f"outputs already exist; pass --overwrite: {existing}")
    for path in (
        selected_path,
        selection_provenance_path,
        pickup_path,
        reverse_provenance_path,
        timestamps_path,
        template_dir / "template_manifest.json",
        template_dir / "object_points_m.npy",
    ):
        if not path.is_file():
            raise ValueError(f"required input is missing: {path}")

    selected, transform, width, approach = _load_selected_grasp(selected_path)
    selection_provenance = _read_json(selection_provenance_path)
    if (
        selection_provenance.get("schema")
        != "demo2gripper.reverse_proxy_grasp_selection_provenance.v1"
        or selection_provenance.get("not_original_pickup") is not True
        or selection_provenance.get("execution_authorized") is not False
    ):
        raise ValueError("selection provenance does not identify a review-only reverse proxy")
    recorded_selected = selection_provenance.get("artifacts", {}).get(
        "selected_grasp", {}
    )
    if recorded_selected.get("sha256") != _sha256(selected_path):
        raise ValueError("selection provenance hash does not match selected_grasp.json")
    template_manifest = _read_json(template_dir / "template_manifest.json")
    if template_manifest.get("frame_id") != selected["object_frame_id"]:
        raise ValueError("object template frame_id differs from the selected grasp")
    if template_manifest.get("object_points_file_sha256") != _sha256(
        template_dir / "object_points_m.npy"
    ):
        raise ValueError("object template point-cloud hash is stale")
    pickup = _read_json(pickup_path)
    reverse = _read_json(reverse_provenance_path)
    timestamps = np.load(timestamps_path, allow_pickle=False)
    frames, times, original_frames = _critical_proxy_interval(
        pickup, reverse, timestamps
    )

    stage = SkillStage(
        name=StageName.GRASP,
        frame_interval=frames,
        time_interval_s=times,
        anchor=AnchorFrame.GRASP_TRANSLATION,
        planner_mode=PlannerMode.GRIPPER_ACTION,
        target_policy=StageTargetPolicy.OBJECT_RELATIVE_GRASP,
        waypoint_artifact=None,
        preconditions=(
            "fresh_matching_rigid_T_base_object",
            "audited_pregrasp_and_descent_complete",
            "selected_grasp_hard_gates_passed",
        ),
        completion=("robotiq_hande_contact_verified",),
        failure_policy=FailurePolicy.RETREAT_REOBSERVE_RETRY,
    )
    relation = ObjectGraspRelation.se3(
        transform[:3, 3],
        transform[:3, :3],
        approach_distance_m=approach,
        object_frame_id=str(selected["object_frame_id"]),
        gripper_name=SELECTED_GRIPPER,
        jaw_opening_m=width,
    )
    skill = TaskSkill(
        skill_id=f"{demo_dir.name}_die_pick_selected_se3_contract_v1",
        stages=(stage,),
        grasp_relation=relation,
        status=SkillStatus.REVIEW_ONLY,
        execution_authorized=False,
        robot_execution_ready=False,
        blocking_conditions=(
            "fresh_live_rigid_object_pose_pending",
            "live_workspace_bounds_pending",
            "moveit_ik_collision_plan_pending",
            "hande_collision_geometry_unmodeled",
        ),
    )
    TaskSkill.from_dict(skill.to_dict())

    report = {
        "schema": REPORT_SCHEMA,
        "mode": "selected_object_se3_contract_only",
        "status": "review_only",
        "execution_authorized": False,
        "robot_execution_ready": False,
        "description": (
            "A minimal TaskSkill v3 carrier for the selected Hand-E object-relative "
            "SE(3) relation consumed by ground_live_grasp_candidate.py."
        ),
        "coordinate_contract": {
            "convention": "right-handed T_parent_child, column vectors, metres",
            "spatial_transfer_model": "object_relative_se3",
            "required_live_input": "rigid T_base_object with matching object_frame_id",
        },
        "grasp_intent": {
            "contract": "selected_robot_specific_object_relative_se3",
            "frame_convention": SELECTED_FRAME_CONVENTION,
            "object_frame_id": selected["object_frame_id"],
            "gripper_name": SELECTED_GRIPPER,
            "candidate_index": selected.get("candidate_index"),
            "T_object_gripper_tcp": transform.tolist(),
            "pregrasp_width_m": width,
            "approach_distance_m": approach,
            "selection": selected["selection"],
            "intent_provenance": selected["intent_provenance"],
        },
        "proxy_evidence": {
            "semantics": reverse.get("semantics"),
            "not_original_pickup": True,
            "trajectory_replay": False,
            "demonstration_path_consumed": False,
            "critical_proxy_frame_interval": list(frames),
            "critical_proxy_time_interval_s": list(times),
            "mapped_original_frames_in_reverse_order": original_frames,
            "role": "grasp geometry and ranking evidence only",
        },
        "phase_graph": {
            "entry": "GRASP",
            "nominal_order": ["GRASP"],
            "scope": (
                "contract carrier only; run_autonomous_pick.py owns OBSERVE through "
                "LIFT and plans motion from the live scene"
            ),
        },
        "task_skill": skill.to_dict(),
        "artifacts": {
            "task_skill": skill_path.name,
            "compiler_report": report_path.name,
            "trajectory": None,
            "trajectory_role": "absent_by_design",
            "trajectory_consumed_for_execution": False,
        },
        "source": {
            "demo_dir": _display(demo_dir, repo_root=repo_root),
            "selected_grasp": _display(selected_path, repo_root=repo_root),
            "selection_provenance": _display(
                selection_provenance_path, repo_root=repo_root
            ),
            "pickup_result": _display(pickup_path, repo_root=repo_root),
            "reverse_provenance": _display(
                reverse_provenance_path, repo_root=repo_root
            ),
            "source_timestamps": _display(timestamps_path, repo_root=repo_root),
        },
        "limitations": [
            "The evidence is a time-reversed placement proxy, not an observed pickup.",
            "No human trajectory is represented as an executable motion target.",
            "A fresh matching full T_base_object is required before grounding.",
            "IK, collision, support clearance, live width, and MoveIt planning are pending.",
        ],
    }
    config = _config_template(
        repo_root=repo_root,
        compiler_report=report_path,
        object_template_dir=template_dir,
        object_frame_id=str(selected["object_frame_id"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(skill.to_json(indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "review_only",
        "execution_authorized": False,
        "outputs": {
            "agentic_skill": _display(skill_path, repo_root=repo_root),
            "compiler_report": _display(report_path, repo_root=repo_root),
            "die_pick_config_template": _display(config_path, repo_root=repo_root),
        },
        "input_sha256": {
            _display(path, repo_root=repo_root): _sha256(path)
            for path in (
                selected_path,
                selection_provenance_path,
                pickup_path,
                reverse_provenance_path,
                timestamps_path,
                template_dir / "template_manifest.json",
                template_dir / "object_points_m.npy",
            )
        },
        "output_sha256": {
            _display(path, repo_root=repo_root): _sha256(path)
            for path in (skill_path, report_path, config_path)
        },
        "next_step": (
            "Fill the config template's live workspace/support bounds from a fresh "
            "capture and save it as die_pick_config.json; first run plan-only."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {config_path}")
    print("status: review_only; no trajectory compiled; no robot command issued")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
