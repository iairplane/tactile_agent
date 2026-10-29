"""Offline compiler and deterministic state machine for the test-tube task.

This module is deliberately below the hardware boundary.  It turns a reviewed
human-demonstration stage description into a task-specific state-machine
contract, classifies which component owns every decision, and evaluates named
predicate events.  It does not import ROS, MoveIt, libfranka, Hand-E, or a
tactile SDK and it cannot dispatch a command.

The human video supplies intent and object-relative constraints.  Free-space
goals remain online-agent decisions, live object/socket frames ground the
relative constraints, and tactile/visual predicates only decide transitions
or recoveries.  Demonstrated hand trajectories are never robot waypoints.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

TEST_TUBE_REVIEW_SCHEMA = "demo2gripper.test_tube_stage_review.v1"
TEST_TUBE_REFINED_REVIEW_SCHEMA = "demo2gripper.test_tube_stage_review.v2"
TEST_TUBE_CONFIG_SCHEMA = "demo2gripper.test_tube_preflight_config.v1"
TEST_TUBE_PLAN_SCHEMA = "demo2gripper.test_tube_preflight_plan.v1"
TEST_TUBE_TACTILE_OBSERVABILITY_SCHEMA = "demo2gripper.test_tube_tactile_observability.v1"
TEST_TUBE_FRONT_RGB_REFERENCE_SCHEMA = "demo2gripper.test_tube_front_rgb_pivot_reference.v1"
TEST_TUBE_RGB_TRACKER_VALIDATION_SCHEMA = "demo2gripper.test_tube_rgb_tracker_validation.v1"
TEST_TUBE_CONTACT_BOOTSTRAP_SCHEMA = "demo2gripper.test_tube_contact_bootstrap.v2"
TEST_TUBE_CONTACT_BOOTSTRAP_MODULE = "demo2gripper.test_tube_contact_bootstrap"
TEST_TUBE_GEOMETRY_SCHEMAS = {
    "tube_grasp_relation": "demo2gripper.test_tube_grasp_relation.v1",
    "live_tube_pose": "demo2gripper.test_tube_live_pose_observation.v1",
    "socket_frame": "demo2gripper.test_tube_socket_frame_observation.v1",
}

REVIEW_STAGE_ORDER = (
    "APPROACH",
    "GRASP_AND_EXTRACT",
    "LOOSEN_AND_PIVOT",
    "RECLAMP",
    "APPROACH_SOCKET",
    "INSERT",
    "RELEASE_AND_RETREAT",
)

REFINED_REVIEW_STAGE_ORDER = (
    "APPROACH",
    "GRASP_AND_EXTRACT",
    "LOOSEN_AND_PIVOT",
    "RECLAMP",
    "APPROACH_RACK_ENTRY",
    "ALIGN_AND_INSERT",
    "RELEASE_AND_RETREAT",
)

_ANGLE_PREDICATE_BY_SOURCE = {
    "tactile_relative_angle": "tactile_relative_angle_reached",
    "front_rgb_calibrated": "front_rgb_calibrated_angle_reached",
}


class TubeState(StrEnum):
    """Task-specific states before any runner maps actions to hardware."""

    OBSERVE = "OBSERVE"
    APPROACH_TUBE = "APPROACH_TUBE"
    SIDE_GRASP = "SIDE_GRASP"
    EXTRACT = "EXTRACT"
    LEVEL_HELD_TUBE = "LEVEL_HELD_TUBE"
    PIVOT_MICRO_LOOSEN = "PIVOT_MICRO_LOOSEN"
    PIVOT_EVALUATE = "PIVOT_EVALUATE"
    RECLAMP = "RECLAMP"
    RECLAMP_RECOVERY = "RECLAMP_RECOVERY"
    APPROACH_SOCKET = "APPROACH_SOCKET"
    ALIGN_SOCKET = "ALIGN_SOCKET"
    INSERT_MICROSTEP = "INSERT_MICROSTEP"
    INSERT_EVALUATE = "INSERT_EVALUATE"
    JAM_RECOVERY_RETRACT = "JAM_RECOVERY_RETRACT"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    COMPLETE = "COMPLETE"
    HOLD = "HOLD"


class ControlAuthority(StrEnum):
    AGENTIC = "agentic"
    RELATIVE_POSE = "relative_pose"
    TACTILE_SUPERVISION = "tactile_supervision"
    PERCEPTION_SUPERVISION = "perception_supervision"


@dataclass(frozen=True)
class TransitionDecision:
    """One declarative transition; ``action`` is not a device command."""

    state: TubeState
    next_state: TubeState
    event: str
    action: str
    reason: str
    recovery: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "next_state": self.next_state.value,
            "event": self.event,
            "action": self.action,
            "reason": self.reason,
            "recovery": self.recovery,
        }


def _object(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


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


def _integer(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name=name)


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_legacy_review(review: Mapping[str, Any]) -> dict[str, Any]:
    if review.get("status") != "review_only":
        raise ValueError("stage review must remain review_only")
    if review.get("hardware_commands_dispatched") is not False:
        raise ValueError("stage review must state hardware_commands_dispatched=false")

    raw_stages = review.get("stages")
    if not isinstance(raw_stages, list) or len(raw_stages) != len(REVIEW_STAGE_ORDER):
        raise ValueError("stage review must contain the seven reviewed stages")
    stages: list[dict[str, Any]] = []
    for index, (raw, expected_name) in enumerate(zip(raw_stages, REVIEW_STAGE_ORDER, strict=True)):
        stage = _object(raw, name=f"stages[{index}]")
        if stage.get("name") != expected_name:
            raise ValueError(f"stage {index} must be {expected_name!r}, got {stage.get('name')!r}")
        frames = stage.get("approx_frames")
        seconds = stage.get("approx_seconds")
        if (
            not isinstance(frames, list)
            or len(frames) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in frames)
            or frames[0] < 0
            or frames[1] <= frames[0]
        ):
            raise ValueError(f"{expected_name}.approx_frames is invalid")
        if not isinstance(seconds, list) or len(seconds) != 2:
            raise ValueError(f"{expected_name}.approx_seconds is invalid")
        seconds_normalized = [
            _number(item, name=f"{expected_name}.approx_seconds", minimum=0, include_minimum=True)
            for item in seconds
        ]
        if seconds_normalized[1] <= seconds_normalized[0]:
            raise ValueError(f"{expected_name}.approx_seconds is invalid")
        stages.append(
            {
                "name": expected_name,
                "approx_frames": list(frames),
                "approx_seconds": seconds_normalized,
                "human_demo_role": "intent_and_boundary_evidence_only",
                "demonstrated_path_replay": False,
            }
        )

    target = _number(
        review.get("angle_target_deg"),
        name="angle_target_deg",
        minimum=0,
        maximum=90,
    )
    return {
        "source_schema": TEST_TUBE_REVIEW_SCHEMA,
        "source_digest_sha256": _canonical_digest(review),
        "target_angle_deg": target,
        "source_angle_reference": review.get("angle_reference"),
        "stages": stages,
        "source_execution_blockers": list(review.get("execution_blockers", [])),
    }


def _validate_refined_review(review: Mapping[str, Any]) -> dict[str, Any]:
    if review.get("status") != "preexecution_analysis_only":
        raise ValueError("refined stage review must remain preexecution_analysis_only")
    if review.get("hardware_commands_dispatched") is not False:
        raise ValueError("stage review must state hardware_commands_dispatched=false")
    recording = _object(review.get("recording"), name="recording")
    action_frames = recording.get("action_interval_frames")
    if (
        not isinstance(action_frames, list)
        or len(action_frames) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in action_frames)
    ):
        raise ValueError("recording.action_interval_frames is invalid")

    raw_stages = review.get("stages")
    if not isinstance(raw_stages, list) or len(raw_stages) != len(REFINED_REVIEW_STAGE_ORDER):
        raise ValueError("refined stage review must contain the seven reviewed stages")
    stages: list[dict[str, Any]] = []
    previous_end: int | None = None
    for index, (raw, expected_name) in enumerate(
        zip(raw_stages, REFINED_REVIEW_STAGE_ORDER, strict=True)
    ):
        stage = _object(raw, name=f"stages[{index}]")
        if stage.get("name") != expected_name:
            raise ValueError(
                f"refined stage {index} must be {expected_name!r}, got {stage.get('name')!r}"
            )
        frames = stage.get("frames_inclusive")
        if (
            not isinstance(frames, list)
            or len(frames) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in frames)
            or frames[0] < 0
            or frames[1] < frames[0]
        ):
            raise ValueError(f"{expected_name}.frames_inclusive is invalid")
        if previous_end is not None and frames[0] != previous_end + 1:
            raise ValueError("refined stage intervals must be contiguous")
        previous_end = frames[1]
        times = stage.get("relative_time_s_inclusive")
        if not isinstance(times, list) or len(times) != 2:
            raise ValueError(f"{expected_name}.relative_time_s_inclusive is invalid")
        normalized_times = [
            _number(
                value,
                name=f"{expected_name}.relative_time_s_inclusive",
                minimum=0,
                include_minimum=True,
            )
            for value in times
        ]
        if normalized_times[1] < normalized_times[0]:
            raise ValueError(f"{expected_name}.relative_time_s_inclusive is invalid")
        stages.append(
            {
                "name": expected_name,
                "frames_half_open": [frames[0], frames[1] + 1],
                "relative_time_s_inclusive": normalized_times,
                "boundary_confidence": stage.get("boundary_confidence"),
                "human_demo_role": "intent_and_boundary_evidence_only",
                "demonstrated_path_replay": False,
            }
        )
    if [stages[0]["frames_half_open"][0], stages[-1]["frames_half_open"][1] - 1] != list(
        action_frames
    ):
        raise ValueError("refined stages must exactly cover recording.action_interval_frames")

    geometry = _object(review.get("observable_geometry"), name="observable_geometry")
    angle = _object(geometry.get("angle_observation"), name="angle_observation")
    if angle.get("physical_angle_available") is not False:
        raise ValueError("refined demo must not claim a physical 3D pivot angle")
    if isinstance(angle.get("apparent_45_deg_frame"), bool) or not isinstance(
        angle.get("apparent_45_deg_frame"), int
    ):
        raise ValueError("angle_observation.apparent_45_deg_frame must be an integer")
    return {
        "source_schema": TEST_TUBE_REFINED_REVIEW_SCHEMA,
        "source_digest_sha256": _canonical_digest(review),
        "target_angle_deg": 45.0,
        "target_angle_semantics": "same_fixed_camera_demonstration_reference",
        "physical_3d_angle_available": False,
        "demonstrated_target_frame": angle["apparent_45_deg_frame"],
        "demonstrated_target_frame_uncertainty": angle.get("frame_uncertainty"),
        "stages": stages,
        "observable_geometry": geometry,
        "source_execution_blockers": list(review.get("execution_blockers", [])),
    }


def _validate_review(review: Mapping[str, Any]) -> dict[str, Any]:
    schema = review.get("schema")
    if schema == TEST_TUBE_REVIEW_SCHEMA:
        return _validate_legacy_review(review)
    if schema == TEST_TUBE_REFINED_REVIEW_SCHEMA:
        return _validate_refined_review(review)
    raise ValueError(
        "stage review schema must be one of "
        f"{TEST_TUBE_REVIEW_SCHEMA!r}, {TEST_TUBE_REFINED_REVIEW_SCHEMA!r}"
    )


def validate_test_tube_geometry_artifact(
    name: str,
    payload: Mapping[str, Any],
    *,
    task_id: str,
) -> dict[str, Any]:
    """Validate one reviewed, plan-only geometry artifact and summarize its limit."""

    expected_schema = TEST_TUBE_GEOMETRY_SCHEMAS.get(name)
    if expected_schema is None:
        raise ValueError(f"unknown test-tube geometry artifact {name!r}")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} geometry artifact must be an object")
    if payload.get("schema") != expected_schema:
        raise ValueError(f"{name} schema must be {expected_schema!r}")
    if payload.get("task_id") != task_id:
        raise ValueError(f"{name} task_id does not match the preflight task")
    if payload.get("status") != "reviewed_plan_only_evidence":
        raise ValueError(f"{name} must have status='reviewed_plan_only_evidence'")
    if payload.get("execution_authorized") is not False:
        raise ValueError(f"{name} must state execution_authorized=false")
    if payload.get("hardware_commands_dispatched") is not False:
        raise ValueError(f"{name} must state hardware_commands_dispatched=false")
    convention = payload.get("transform_convention")
    if not isinstance(convention, str) or not convention.startswith("T_parent_child"):
        raise ValueError(f"{name} must use the T_parent_child transform convention")

    if name == "tube_grasp_relation":
        if payload.get("T_object_gripper_tcp") is not None:
            raise ValueError("plan-only tube grasp relation must keep T_object_gripper_tcp=null")
        _object(
            payload.get("robot_candidate_constraints"),
            name="tube_grasp_relation.robot_candidate_constraints",
        )
        metric_ready = False
        limit = payload.get("execution_start_requirement")
    elif name == "live_tube_pose":
        if (
            payload.get("metric_pose_available") is not False
            or payload.get("T_base_tube") is not None
        ):
            raise ValueError("plan-only live tube pose must keep metric pose unavailable")
        requirement = _object(
            payload.get("execution_start_requirement"),
            name="live_tube_pose.execution_start_requirement",
        )
        if requirement.get("fresh_observation_required") is not True:
            raise ValueError("live tube pose must require a fresh execution-start observation")
        metric_ready = False
        limit = dict(requirement)
    else:
        if (
            payload.get("metric_socket_pose_available") is not False
            or payload.get("T_base_socket") is not None
            or payload.get("insertion_target_depth_m") is not None
        ):
            raise ValueError(
                "plan-only socket frame must keep metric insertion geometry unavailable"
            )
        requirement = _object(
            payload.get("execution_start_requirement"),
            name="socket_frame.execution_start_requirement",
        )
        if requirement.get("fail_closed_if_missing") is not True:
            raise ValueError(
                "socket frame must fail closed until metric entry geometry is measured"
            )
        metric_ready = False
        limit = dict(requirement)

    return {
        "name": name,
        "schema": expected_schema,
        "status": payload["status"],
        "source_digest_sha256": _canonical_digest(payload),
        "plan_only_reviewed": True,
        "metric_execution_geometry_ready": metric_ready,
        "execution_start_requirement": limit,
    }


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema") != TEST_TUBE_CONFIG_SCHEMA:
        raise ValueError(f"config schema must be {TEST_TUBE_CONFIG_SCHEMA!r}")
    task_id = _string(config.get("task_id"), name="task_id")
    frames = _object(config.get("frames"), name="frames")
    normalized_frames = {
        key: _string(frames.get(key), name=f"frames.{key}")
        for key in ("base", "tcp", "tube", "socket")
    }
    if frames.get("transform_convention") != "T_parent_child":
        raise ValueError("frames.transform_convention must be 'T_parent_child'")
    normalized_frames["transform_convention"] = "T_parent_child"

    evidence = _object(config.get("evidence"), name="evidence")
    normalized_evidence: dict[str, dict[str, Any]] = {}
    for key in (
        "tube_grasp_relation",
        "live_tube_pose",
        "socket_frame",
        "tactile_observability",
        "gf220_sdk_calibrations",
        "front_rgb_pivot_reference",
        "front_rgb_tracker_validation",
        "d435_live_source",
        "offline_runtime_orchestrator",
        "offline_predicate_builder",
        "rgb_tracker_qualification_tool",
        "rgb_tracker_qualification_evaluator",
    ):
        item = _object(evidence.get(key), name=f"evidence.{key}")
        normalized_evidence[key] = {
            "path": _optional_string(item.get("path"), name=f"evidence.{key}.path"),
            "reviewed": _boolean(item.get("reviewed"), name=f"evidence.{key}.reviewed"),
        }

    tilt = _object(config.get("tilt"), name="tilt")
    sources = tilt.get("allowed_target_sources")
    allowed = {"tactile_relative_angle", "front_rgb_calibrated"}
    if (
        not isinstance(sources, list)
        or not sources
        or any(source not in allowed for source in sources)
        or len(sources) != len(set(sources))
    ):
        raise ValueError(
            "tilt.allowed_target_sources must be a unique non-empty subset of "
            "tactile_relative_angle/front_rgb_calibrated"
        )
    normalized_tilt = {
        "target_semantics": tilt.get("target_semantics"),
        "target_relative_deg": _number(
            tilt.get("target_relative_deg"),
            name="tilt.target_relative_deg",
            minimum=0,
            maximum=90,
        ),
        "tolerance_deg": _number(
            tilt.get("tolerance_deg"),
            name="tilt.tolerance_deg",
            minimum=0,
            maximum=20,
        ),
        "confirmation_windows": _integer(
            tilt.get("confirmation_windows"), name="tilt.confirmation_windows"
        ),
        "allowed_target_sources": list(sources),
        "target_image_axis_deg": (
            None
            if tilt.get("target_image_axis_deg") is None
            else _number(
                tilt.get("target_image_axis_deg"),
                name="tilt.target_image_axis_deg",
                minimum=0,
                maximum=180,
                include_minimum=True,
            )
        ),
        "tactile_angle_calibration_id": _optional_string(
            tilt.get("tactile_angle_calibration_id"),
            name="tilt.tactile_angle_calibration_id",
        ),
        "front_rgb_angle_calibration_id": _optional_string(
            tilt.get("front_rgb_angle_calibration_id"),
            name="tilt.front_rgb_angle_calibration_id",
        ),
        "front_rgb_runtime_gate_qualified": _boolean(
            tilt.get("front_rgb_runtime_gate_qualified"),
            name="tilt.front_rgb_runtime_gate_qualified",
        ),
    }
    if normalized_tilt["target_semantics"] != "same_fixed_camera_demonstration_reference":
        raise ValueError(
            "tilt.target_semantics must be 'same_fixed_camera_demonstration_reference'"
        )

    tactile = _object(config.get("tactile"), name="tactile")
    jam = _object(tactile.get("insertion_jam"), name="tactile.insertion_jam")
    modes = _object(tactile.get("predicate_modes"), name="tactile.predicate_modes")
    bootstrap = _object(tactile.get("bootstrap"), name="tactile.bootstrap")
    allowed_modes = {"runtime_qualified", "conditional_runtime_bootstrap", "shadow_only"}
    normalized_modes: dict[str, str] = {}
    for predicate in ("contact", "slip", "insertion_jam"):
        mode = modes.get(predicate)
        if mode not in allowed_modes:
            raise ValueError(
                f"tactile.predicate_modes.{predicate} must be one of {sorted(allowed_modes)}"
            )
        normalized_modes[predicate] = mode
    normalized_tactile = {
        "profile_id": _string(tactile.get("profile_id"), name="tactile.profile_id"),
        "execution_qualified": _boolean(
            tactile.get("execution_qualified"), name="tactile.execution_qualified"
        ),
        "predicate_modes": normalized_modes,
        "bootstrap": {
            "fresh_unloaded_baseline_required": _boolean(
                bootstrap.get("fresh_unloaded_baseline_required"),
                name="tactile.bootstrap.fresh_unloaded_baseline_required",
            ),
            "positive_contact_required": _boolean(
                bootstrap.get("positive_contact_required"),
                name="tactile.bootstrap.positive_contact_required",
            ),
            "unloaded_window_count": _integer(
                bootstrap.get("unloaded_window_count"),
                name="tactile.bootstrap.unloaded_window_count",
            ),
            "positive_contact_window_count": _integer(
                bootstrap.get("positive_contact_window_count"),
                name="tactile.bootstrap.positive_contact_window_count",
            ),
            "session_scoped": _boolean(
                bootstrap.get("session_scoped"),
                name="tactile.bootstrap.session_scoped",
            ),
            "min_unique_samples": _integer(
                bootstrap.get("min_unique_samples"),
                name="tactile.bootstrap.min_unique_samples",
                minimum=2,
            ),
            "min_calibration_windows_per_stage": _integer(
                bootstrap.get("min_calibration_windows_per_stage"),
                name="tactile.bootstrap.min_calibration_windows_per_stage",
                minimum=3,
            ),
            "max_age_s": _number(
                bootstrap.get("max_age_s"),
                name="tactile.bootstrap.max_age_s",
                minimum=0,
            ),
            "max_pair_skew_s": _number(
                bootstrap.get("max_pair_skew_s"),
                name="tactile.bootstrap.max_pair_skew_s",
                minimum=0,
            ),
            "minimum_fz_change_sensor_units": _number(
                bootstrap.get("minimum_fz_change_sensor_units"),
                name="tactile.bootstrap.minimum_fz_change_sensor_units",
                minimum=0,
            ),
            "minimum_threshold_floor_sensor_units": _number(
                bootstrap.get("minimum_threshold_floor_sensor_units"),
                name="tactile.bootstrap.minimum_threshold_floor_sensor_units",
                minimum=0,
            ),
            "positive_margin_noise_multiplier": _number(
                bootstrap.get("positive_margin_noise_multiplier"),
                name="tactile.bootstrap.positive_margin_noise_multiplier",
                minimum=0,
            ),
            "threshold_noise_multiplier": _number(
                bootstrap.get("threshold_noise_multiplier"),
                name="tactile.bootstrap.threshold_noise_multiplier",
                minimum=0,
            ),
            "minimum_consistent_positive_fraction": _number(
                bootstrap.get("minimum_consistent_positive_fraction"),
                name="tactile.bootstrap.minimum_consistent_positive_fraction",
                minimum=0,
                maximum=1,
            ),
            "engage_fraction_of_positive": _number(
                bootstrap.get("engage_fraction_of_positive"),
                name="tactile.bootstrap.engage_fraction_of_positive",
                minimum=0,
                maximum=1,
            ),
            "retain_fraction_of_positive": _number(
                bootstrap.get("retain_fraction_of_positive"),
                name="tactile.bootstrap.retain_fraction_of_positive",
                minimum=0,
                maximum=1,
            ),
            "contact_confirmation_windows": _integer(
                bootstrap.get("contact_confirmation_windows"),
                name="tactile.bootstrap.contact_confirmation_windows",
            ),
            "loss_confirmation_windows": _integer(
                bootstrap.get("loss_confirmation_windows"),
                name="tactile.bootstrap.loss_confirmation_windows",
            ),
        },
        "contact_policy": tactile.get("contact_policy"),
        "contact_confirmation_windows": _integer(
            tactile.get("contact_confirmation_windows"),
            name="tactile.contact_confirmation_windows",
        ),
        "slip_confirmation_windows": _integer(
            tactile.get("slip_confirmation_windows"),
            name="tactile.slip_confirmation_windows",
        ),
        "pivot_open_step_m": _number(
            tactile.get("pivot_open_step_m"),
            name="tactile.pivot_open_step_m",
            minimum=0,
            maximum=0.001,
        ),
        "max_pivot_opening_delta_m": _number(
            tactile.get("max_pivot_opening_delta_m"),
            name="tactile.max_pivot_opening_delta_m",
            minimum=0,
            maximum=0.005,
        ),
        "reclamp_step_m": _number(
            tactile.get("reclamp_step_m"),
            name="tactile.reclamp_step_m",
            minimum=0,
            maximum=0.002,
        ),
        "max_reclamp_steps": _integer(
            tactile.get("max_reclamp_steps"), name="tactile.max_reclamp_steps"
        ),
        "insertion_jam": {
            "min_progress_ratio": _number(
                jam.get("min_progress_ratio"),
                name="tactile.insertion_jam.min_progress_ratio",
                minimum=0,
                maximum=1,
            ),
            "confirmation_windows": _integer(
                jam.get("confirmation_windows"),
                name="tactile.insertion_jam.confirmation_windows",
            ),
            "resistance_delta_threshold": (
                None
                if jam.get("resistance_delta_threshold") is None
                else _number(
                    jam.get("resistance_delta_threshold"),
                    name="tactile.insertion_jam.resistance_delta_threshold",
                    minimum=0,
                )
            ),
            "resistance_units": _optional_string(
                jam.get("resistance_units"),
                name="tactile.insertion_jam.resistance_units",
            ),
        },
    }
    if normalized_tactile["contact_policy"] != "any_finger_valid_contact":
        raise ValueError(
            "tactile.contact_policy must be 'any_finger_valid_contact'; "
            "bilateral symmetry is not required for a side-held tube"
        )
    if normalized_modes["contact"] == "conditional_runtime_bootstrap" and not all(
        (
            normalized_tactile["bootstrap"]["fresh_unloaded_baseline_required"],
            normalized_tactile["bootstrap"]["positive_contact_required"],
            normalized_tactile["bootstrap"]["session_scoped"],
        )
    ):
        raise ValueError(
            "conditional contact bootstrap requires a fresh unloaded baseline, "
            "positive contact, and session-scoped thresholds"
        )
    bootstrap_config = normalized_tactile["bootstrap"]
    if (
        bootstrap_config["unloaded_window_count"]
        < bootstrap_config["min_calibration_windows_per_stage"]
    ):
        raise ValueError(
            "test-tube contact bootstrap needs at least min_calibration_windows_per_stage "
            "unloaded windows"
        )
    if (
        bootstrap_config["positive_contact_window_count"]
        < bootstrap_config["min_calibration_windows_per_stage"]
    ):
        raise ValueError(
            "test-tube contact bootstrap needs at least min_calibration_windows_per_stage "
            "positive-contact windows"
        )
    if (
        bootstrap_config["retain_fraction_of_positive"]
        >= bootstrap_config["engage_fraction_of_positive"]
    ):
        raise ValueError(
            "tactile.bootstrap.retain_fraction_of_positive must be smaller than "
            "engage_fraction_of_positive"
        )
    if normalized_tactile["max_pivot_opening_delta_m"] < normalized_tactile["pivot_open_step_m"]:
        raise ValueError("max_pivot_opening_delta_m must cover one pivot step")

    motion = _object(config.get("motion"), name="motion")
    normalized_motion = {
        "approach_standoff_m": _number(
            motion.get("approach_standoff_m"),
            name="motion.approach_standoff_m",
            minimum=0,
            maximum=0.20,
        ),
        "extract_clearance_m": _number(
            motion.get("extract_clearance_m"),
            name="motion.extract_clearance_m",
            minimum=0,
            maximum=0.20,
        ),
        "pivot_max_microsteps": _integer(
            motion.get("pivot_max_microsteps"), name="motion.pivot_max_microsteps"
        ),
        "insertion_step_m": _number(
            motion.get("insertion_step_m"),
            name="motion.insertion_step_m",
            minimum=0,
            maximum=0.003,
        ),
        "insertion_target_depth_m": (
            None
            if motion.get("insertion_target_depth_m") is None
            else _number(
                motion.get("insertion_target_depth_m"),
                name="motion.insertion_target_depth_m",
                minimum=0,
                maximum=0.10,
            )
        ),
        "jam_retract_m": _number(
            motion.get("jam_retract_m"),
            name="motion.jam_retract_m",
            minimum=0,
            maximum=0.01,
        ),
        "max_jam_recoveries": _integer(
            motion.get("max_jam_recoveries"), name="motion.max_jam_recoveries"
        ),
        "retreat_m": _number(
            motion.get("retreat_m"),
            name="motion.retreat_m",
            minimum=0,
            maximum=0.20,
        ),
        "velocity_scale": _number(
            motion.get("velocity_scale"),
            name="motion.velocity_scale",
            minimum=0,
            maximum=0.10,
        ),
        "acceleration_scale": _number(
            motion.get("acceleration_scale"),
            name="motion.acceleration_scale",
            minimum=0,
            maximum=0.10,
        ),
    }
    readiness = _object(config.get("execution_readiness", {}), name="execution_readiness")
    readiness_keys = (
        "exact_tube_tcp_instantiated",
        "gripper_width_and_close_budget_grounded",
        "collision_scene_validated",
        "motion_bounds_validated",
        "offline_runtime_orchestrator_available",
        "offline_predicate_builder_available",
        "rgb_tracker_qualification_tool_available",
        "long_lived_front_rgb_source_available",
        "live_hardware_adapter_available",
        "live_feedback_predicate_producers_available",
        "support_after_release_verifier_available",
        "robot_command_state_path_healthy",
        "front_rgb_stream_healthy",
    )
    normalized_readiness = {
        key: _boolean(readiness.get(key, False), name=f"execution_readiness.{key}")
        for key in readiness_keys
    }
    return {
        "task_id": task_id,
        "frames": normalized_frames,
        "evidence": normalized_evidence,
        "tilt": normalized_tilt,
        "tactile": normalized_tactile,
        "motion": normalized_motion,
        "execution_readiness": normalized_readiness,
        "config_digest_sha256": _canonical_digest(config),
    }


def _angle_condition(tilt: Mapping[str, Any]) -> str:
    return " OR ".join(
        _ANGLE_PREDICATE_BY_SOURCE[source] for source in tilt["allowed_target_sources"]
    )


def _contact_bootstrap_contract(tactile: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the implemented session bootstrap without importing a device stack."""

    bootstrap = tactile["bootstrap"]
    enabled = tactile["predicate_modes"]["contact"] == "conditional_runtime_bootstrap"
    fit_keys = (
        "min_unique_samples",
        "min_calibration_windows_per_stage",
        "max_age_s",
        "max_pair_skew_s",
        "minimum_fz_change_sensor_units",
        "minimum_threshold_floor_sensor_units",
        "positive_margin_noise_multiplier",
        "threshold_noise_multiplier",
        "minimum_consistent_positive_fraction",
        "engage_fraction_of_positive",
        "retain_fraction_of_positive",
        "contact_confirmation_windows",
        "loss_confirmation_windows",
    )
    return {
        "enabled": enabled,
        "implementation": {
            "module": TEST_TUBE_CONTACT_BOOTSTRAP_MODULE,
            "schema": TEST_TUBE_CONTACT_BOOTSTRAP_SCHEMA,
            "fit_function": "bootstrap_test_tube_contact",
            "gate_class": "TestTubeContactGate",
            "input_type": "GF220ClassifierWindows",
        },
        "signal": "maximum_absolute_raw_sdk_fz_change_across_either_pad",
        "logic": "any_finger_valid_contact",
        "fit_config": {key: bootstrap[key] for key in fit_keys},
        "side_grasp_phases": [
            {
                "phase": "before_bounded_close",
                "action": "capture_repeated_fresh_unloaded_dual_force6d_windows",
                "requirements": [
                    "at_least_min_calibration_windows_per_stage",
                    "windows_nonoverlapping_and_strictly_monotonic",
                    "runtime_session_id_created_and_persisted",
                    "same_live_sdk_session_as_positive_window",
                    "fresh_dual_tactile_samples",
                    "stable_sensor_serials_and_calibration_ids",
                ],
            },
            {
                "phase": "bounded_close",
                "action": "apply_reviewed_side_grasp_relation_and_one_bounded_close",
                "requirements": ["tube_between_fingers", "grasp_close_budget_remaining"],
            },
            {
                "phase": "after_bounded_close",
                "action": "capture_repeated_fresh_positive_contact_dual_force6d_windows",
                "requirements": [
                    "at_least_min_calibration_windows_per_stage",
                    "repeatable_response_above_unloaded_noise_margin",
                    "window_strictly_after_unloaded_window",
                    "same_sensor_serials_and_calibration_ids",
                ],
            },
            {
                "phase": "activate_session_gate",
                "action": "fit_profile_then_require_contact_present",
                "requirements": [
                    "at_least_one_pad_has_minimum_fz_change",
                    "profile_valid_for_current_session_only",
                ],
            },
        ],
        "runtime_outcomes": {
            "CONTACT_PRESENT": "runner_promotes_contact_present",
            "CONTACT_ABSENT": "runner_promotes_contact_lost_after_prior_contact",
            "INVALID": "runner_promotes_invalid_evidence_and_HOLD",
        },
        "required_runtime_provenance": [
            "session_id",
            "profile_id",
            "sensor_serials",
            "calibration_ids",
            "strict_sdk_and_host_time_high_water_marks",
        ],
        "threshold_reuse_across_sessions": False,
        "explicitly_not_qualified": [
            "physical_force_units",
            "tube_angle",
            "slip_detected",
            "insertion_jam",
            "logical_left_right_finger_identity",
        ],
    }


def _state_contracts(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    frames = config["frames"]
    motion = config["motion"]
    tactile = config["tactile"]
    tilt = config["tilt"]
    pivot_failures = ["contact_lost"]
    if tactile["predicate_modes"]["slip"] == "runtime_qualified":
        pivot_failures.append("slip_detected")
    bootstrap_contract = _contact_bootstrap_contract(tactile)
    return [
        {
            "state": "OBSERVE",
            "authority": ["agentic", "perception_supervision"],
            "goal": "localize_live_tube_and_socket_frames",
            "completion": ["tube_pose_valid", "socket_frame_valid"],
        },
        {
            "state": "APPROACH_TUBE",
            "authority": ["agentic"],
            "goal": "collision_free_side_grasp_standoff",
            "target_frame": frames["base"],
            "standoff_m": motion["approach_standoff_m"],
            "completion": ["approach_pose_reached"],
        },
        {
            "state": "SIDE_GRASP",
            "authority": [
                "relative_pose",
                "tactile_supervision",
                "perception_supervision",
            ],
            "goal": "transfer_T_tube_to_tcp_side_grasp_then_close",
            "target_frame": frames["tube"],
            "completion": ["grasp_established"],
            "contact_gate_mode": tactile["predicate_modes"]["contact"],
            "bootstrap_phases": (
                bootstrap_contract["side_grasp_phases"] if bootstrap_contract["enabled"] else []
            ),
            "bootstrap_completion_requirement": (
                "session_contact_profile_active AND contact_present"
                if bootstrap_contract["enabled"]
                else "contact_present"
            ),
            "extraction_requirement": (
                "contact_present AND gripper_state_valid AND grasp_width_in_range AND "
                "(bilateral_contact OR visual_tube_between_fingers)"
            ),
            "note": (
                "one valid finger contact is sufficient only for contact-loss monitoring; "
                "it never authorizes extraction by itself"
            ),
        },
        {
            "state": "EXTRACT",
            "authority": ["agentic", "tactile_supervision"],
            "goal": "clear_rack_without_replaying_human_path",
            "clearance_m": motion["extract_clearance_m"],
            "completion": ["tube_clear_of_source", "contact_present"],
        },
        {
            "state": "LEVEL_HELD_TUBE",
            "authority": [
                "relative_pose",
                "perception_supervision",
                "tactile_supervision",
            ],
            "goal": "keep_the_rigid_grasp_and_servo_the_live_tube_axis_to_horizontal",
            "completion": [
                "tube_horizontal",
                "level_target_reached",
                "contact_present",
            ],
            "failure_mode": "hold_if_the_signed_horizontal_axis_is_unobservable",
        },
        {
            "state": "PIVOT_MICRO_LOOSEN",
            "authority": ["relative_pose", "tactile_supervision"],
            "goal": "hold_wrist_pose_and_apply_one_bounded_jaw_opening_step",
            "jaw_open_step_m": tactile["pivot_open_step_m"],
            "opening_budget_m": tactile["max_pivot_opening_delta_m"],
            "completion": ["one_microstep_complete"],
        },
        {
            "state": "PIVOT_EVALUATE",
            "authority": ["perception_supervision", "tactile_supervision"],
            "goal": "keep_contact_and_choose_continue_or_reclamp",
            "angle_target_deg": tilt["target_relative_deg"],
            "target_image_axis_deg": tilt["target_image_axis_deg"],
            "angle_tolerance_deg": tilt["tolerance_deg"],
            "angle_gate": _angle_condition(tilt),
            "failure_precedence": pivot_failures,
            "shadow_observers": (
                ["slip_detected"] if tactile["predicate_modes"]["slip"] == "shadow_only" else []
            ),
        },
        {
            "state": "RECLAMP",
            "authority": ["tactile_supervision"],
            "goal": "immediately_restore_hold_then_verify_settled_contact",
            "jaw_close_step_m": tactile["reclamp_step_m"],
            "completion": ["contact_present", "reclamp_command_settled"],
        },
        {
            "state": "RECLAMP_RECOVERY",
            "authority": ["tactile_supervision"],
            "goal": "restore_hold_after_slip_loss_or_pivot_budget_exhaustion",
            "jaw_close_step_m": tactile["reclamp_step_m"],
            "completion": ["contact_present", "reclamp_command_settled"],
            "next": ("PIVOT_EVALUATE if pivot budget remains; otherwise HOLD"),
        },
        {
            "state": "APPROACH_SOCKET",
            "authority": ["agentic"],
            "goal": "collision_free_socket_standoff_from_live_scene",
            "completion": ["socket_standoff_reached"],
        },
        {
            "state": "ALIGN_SOCKET",
            "authority": [
                "relative_pose",
                "perception_supervision",
                "tactile_supervision",
            ],
            "goal": "align_live_tube_axis_to_live_socket_axis",
            "completion": ["tube_socket_axes_aligned", "tube_tip_at_socket_entry"],
        },
        {
            "state": "INSERT_MICROSTEP",
            "authority": ["relative_pose", "tactile_supervision"],
            "goal": "advance_only_along_live_socket_axis",
            "step_m": motion["insertion_step_m"],
            "completion": ["one_axial_step_complete"],
        },
        {
            "state": "INSERT_EVALUATE",
            "authority": ["perception_supervision", "tactile_supervision"],
            "goal": "choose_continue_seated_or_jam_recovery",
            "jam_definition": (
                "low_axial_progress_for_confirmation_window AND calibrated_resistance_or_shear_rise"
            ),
            "jam_gate_mode": tactile["predicate_modes"]["insertion_jam"],
            "physical_transition_enabled": tactile["predicate_modes"]["insertion_jam"]
            == "runtime_qualified",
            "completion": [
                "insertion_seated AND no_jam",
                "legacy insertion_depth_reached AND no_jam",
            ],
        },
        {
            "state": "JAM_RECOVERY_RETRACT",
            "authority": ["agentic", "relative_pose", "tactile_supervision"],
            "goal": "stop_push_retract_along_negative_socket_axis_reobserve_and_realign",
            "retract_m": motion["jam_retract_m"],
            "maximum_recoveries": motion["max_jam_recoveries"],
        },
        {
            "state": "RELEASE",
            "authority": ["perception_supervision", "tactile_supervision"],
            "goal": "open_only_after_seated_and_support_verified",
            "completion": ["gripper_open", "contact_released", "tube_stays_supported"],
        },
        {
            "state": "RETREAT",
            "authority": ["agentic"],
            "goal": "retreat_along_collision_free_exit",
            "distance_m": motion["retreat_m"],
            "completion": ["retreat_complete"],
        },
    ]


def _transition_contract(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fixed transition order, with recovery predicates taking precedence."""

    modes = config["tactile"]["predicate_modes"]
    pivot_contact_failure = "contact_lost"
    if modes["slip"] == "runtime_qualified":
        pivot_contact_failure += " OR slip_detected"
    jam_enabled = modes["insertion_jam"] == "runtime_qualified"
    bootstrap_enabled = modes["contact"] == "conditional_runtime_bootstrap"
    side_grasp_established_event = (
        "session_contact_profile_active AND grasp_established"
        if bootstrap_enabled
        else "grasp_established"
    )
    return [
        {
            "from": "ANY_ACTIVE_STATE",
            "priority": 0,
            "event": "invalid_or_stale_evidence OR overload",
            "to": "HOLD",
            "action": "stop_and_hold",
        },
        {
            "from": "OBSERVE",
            "priority": 1,
            "event": "tube_pose_valid AND socket_frame_valid",
            "to": "APPROACH_TUBE",
            "action": "request_agentic_approach_plan",
        },
        {
            "from": "APPROACH_TUBE",
            "priority": 1,
            "event": "approach_pose_reached",
            "to": "SIDE_GRASP",
            "action": (
                "capture_unloaded_force6d_window_then_ground_side_grasp_relation"
                if bootstrap_enabled
                else "ground_side_grasp_relation"
            ),
        },
        {
            "from": "SIDE_GRASP",
            "priority": 1,
            "event": side_grasp_established_event,
            "to": "EXTRACT",
            "action": "request_agentic_extraction",
        },
        {
            "from": "SIDE_GRASP",
            "priority": 2,
            "event": (
                "NOT session_contact_profile_active AND "
                "unloaded_window_ready AND grasp_close_budget_remaining"
            ),
            "to": "SIDE_GRASP",
            "action": "bounded_close_capture_positive_window_and_fit_session_contact_gate",
            "enabled": bootstrap_enabled,
            "implementation": (f"{TEST_TUBE_CONTACT_BOOTSTRAP_MODULE}.bootstrap_test_tube_contact"),
        },
        {
            "from": "SIDE_GRASP",
            "priority": 3,
            "event": "contact_present AND NOT grasp_established",
            "to": "SIDE_GRASP",
            "action": "collect_grasp_confirmation_evidence",
        },
        {
            "from": "SIDE_GRASP",
            "priority": 4,
            "event": (
                "session_contact_profile_active AND NOT contact_present AND "
                "grasp_close_budget_remaining"
                if bootstrap_enabled
                else "NOT contact_present AND grasp_close_budget_remaining"
            ),
            "to": "SIDE_GRASP",
            "action": "bounded_close_probe",
        },
        {
            "from": "SIDE_GRASP",
            "priority": 5,
            "event": (
                "contact_bootstrap_failed OR grasp_close_budget_exhausted"
                if bootstrap_enabled
                else "grasp_close_budget_exhausted"
            ),
            "to": "HOLD",
            "action": "stop_and_hold",
        },
        {
            "from": "EXTRACT",
            "priority": 1,
            "event": "tube_clear_of_source AND contact_present",
            "to": "LEVEL_HELD_TUBE",
            "action": "servo_held_tube_to_horizontal",
        },
        {
            "from": "LEVEL_HELD_TUBE",
            "priority": 1,
            "event": "contact_lost OR stationary_qualified_slip_detected",
            "to": "HOLD",
            "action": "stop_and_hold",
        },
        {
            "from": "LEVEL_HELD_TUBE",
            "priority": 2,
            "event": "tube_horizontal AND level_target_reached AND contact_present",
            "to": "PIVOT_MICRO_LOOSEN",
            "action": "one_micro_loosen",
        },
        {
            "from": "PIVOT_MICRO_LOOSEN",
            "priority": 2,
            "event": "one_microstep_complete",
            "to": "PIVOT_EVALUATE",
            "action": "collect_fresh_angle_and_contact_window",
        },
        {
            "from": "PIVOT_MICRO_LOOSEN|PIVOT_EVALUATE",
            "priority": 1,
            "event": pivot_contact_failure,
            "to": "RECLAMP_RECOVERY",
            "action": "immediate_bounded_reclamp",
        },
        {
            "from": "PIVOT_EVALUATE",
            "priority": 2,
            "event": f"contact_present AND ({_angle_condition(config['tilt'])})",
            "to": "RECLAMP",
            "action": "reclamp_at_target",
        },
        {
            "from": "PIVOT_EVALUATE",
            "priority": 3,
            "event": "contact_present AND pivot_budget_remaining",
            "to": "PIVOT_MICRO_LOOSEN",
            "action": "next_micro_loosen",
        },
        {
            "from": "PIVOT_EVALUATE",
            "priority": 4,
            "event": "contact_present AND NOT pivot_budget_remaining",
            "to": "RECLAMP_RECOVERY",
            "action": "reclamp_then_hold",
        },
        {
            "from": "RECLAMP",
            "priority": 1,
            "event": "contact_present AND reclamp_command_settled AND pivot_target_confirmed",
            "to": "APPROACH_SOCKET",
            "action": "freeze_regrasp_relation",
        },
        {
            "from": "RECLAMP_RECOVERY",
            "priority": 1,
            "event": "contact_present AND reclamp_command_settled AND pivot_budget_remaining",
            "to": "PIVOT_EVALUATE",
            "action": "collect_fresh_angle_and_contact_window",
        },
        {
            "from": "RECLAMP_RECOVERY",
            "priority": 2,
            "event": ("contact_present AND reclamp_command_settled AND NOT pivot_budget_remaining"),
            "to": "HOLD",
            "action": "stop_and_hold",
        },
        {
            "from": "APPROACH_SOCKET",
            "priority": 1,
            "event": "socket_standoff_reached",
            "to": "ALIGN_SOCKET",
            "action": "ground_tube_socket_axis_relation",
        },
        {
            "from": "ALIGN_SOCKET",
            "priority": 1,
            "event": "tube_socket_axes_aligned AND tube_tip_at_socket_entry",
            "to": "INSERT_MICROSTEP",
            "action": "one_axial_insert_step",
        },
        {
            "from": "INSERT_MICROSTEP",
            "priority": 1,
            "event": "one_axial_step_complete",
            "to": "INSERT_EVALUATE",
            "action": "collect_fresh_progress_and_resistance_window",
        },
        {
            "from": "INSERT_EVALUATE",
            "priority": 1,
            "event": "insertion_jam AND jam_recovery_budget_remaining",
            "to": "JAM_RECOVERY_RETRACT",
            "action": "stop_push_and_retract",
            "enabled": jam_enabled,
            "activation_requirement": "labelled_runtime_qualified_insertion_jam_profile",
        },
        {
            "from": "INSERT_EVALUATE",
            "priority": 1,
            "event": "insertion_jam AND NOT jam_recovery_budget_remaining",
            "to": "HOLD",
            "action": "stop_and_hold",
            "enabled": jam_enabled,
            "activation_requirement": "labelled_runtime_qualified_insertion_jam_profile",
        },
        {
            "from": "INSERT_EVALUATE",
            "priority": 2,
            "event": "insertion_seated AND no_jam",
            "to": "RELEASE",
            "action": "verify_support_before_open",
            "enabled": jam_enabled,
            "activation_requirement": "debounced_session_insertion_seat_event",
        },
        {
            "from": "INSERT_EVALUATE",
            "priority": 3,
            "event": "insertion_depth_reached AND no_jam",
            "to": "RELEASE",
            "action": "verify_support_before_open",
            "enabled": jam_enabled,
            "activation_requirement": "labelled_runtime_qualified_insertion_jam_profile",
        },
        {
            "from": "INSERT_EVALUATE",
            "priority": 4,
            "event": "axial_progress_ok AND insertion_depth_remaining",
            "to": "INSERT_MICROSTEP",
            "action": "next_axial_microstep",
            "enabled": jam_enabled,
            "activation_requirement": "labelled_runtime_qualified_insertion_jam_profile",
        },
        {
            "from": "JAM_RECOVERY_RETRACT",
            "priority": 1,
            "event": "retract_complete AND scene_reobserved",
            "to": "ALIGN_SOCKET",
            "action": "realign_socket_entry",
        },
        {
            "from": "RELEASE",
            "priority": 1,
            "event": "gripper_open AND contact_released AND tube_stays_supported",
            "to": "RETREAT",
            "action": "request_agentic_retreat",
        },
        {
            "from": "RETREAT",
            "priority": 1,
            "event": "retreat_complete",
            "to": "COMPLETE",
            "action": "none",
        },
    ]


def _authority_partition() -> dict[str, Any]:
    return {
        "agentic": {
            "owns": [
                "free_space_approach_goals",
                "collision_aware_MoveIt_path_requests",
                "source_rack_clearance",
                "socket_standoff",
                "jam_recovery_reobserve_and_realign_goal",
                "retreat_goal",
            ],
            "does_not_own": ["contact_truth", "tilt_target_truth", "jam_truth"],
        },
        "relative_pose": {
            "owns": [
                "T_tube_tcp_side_grasp",
                "tube_axis_target_at_45_deg",
                "tube_axis_to_socket_axis_alignment",
                "axial_insertion_direction",
            ],
            "requires": ["live_T_base_tube", "live_T_base_socket"],
            "demonstrated_path_replay": False,
        },
        "tactile_supervision": {
            "owns": [
                "contact_present_or_lost",
                "slip_detected",
                "contact_component_of_pivot_continue_or_reclamp",
                "tactile_component_of_reclamp_stability",
                "insertion_jam",
                "release_contact_state",
            ],
            "does_not_own": [
                "free_space_pose_generation",
                "collision_avoidance",
                "tube_angle_without_a_calibrated_angle_source",
            ],
        },
        "perception_supervision": {
            "owns": [
                "target_tube_identity_and_metric_scene_frames",
                "visual_tube_between_fingers",
                "front_rgb_tube_axis_target",
                "tube_clear_of_source",
                "tube_tip_and_socket_axis_alignment",
                "axial_progress_and_insertion_depth",
                "tube_stays_supported_after_release",
            ],
            "does_not_own": [
                "free_space_motion_generation",
                "physical_contact_truth",
                "unobserved_transparent_geometry",
            ],
        },
    }


def _read_observability_summary(
    payload: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    if payload is None:
        return (
            {
                "provided": False,
                "runtime_gate_eligible": False,
                "mode": "missing",
            },
            ["tactile_observability_artifact_missing"],
        )
    if payload.get("schema") != TEST_TUBE_TACTILE_OBSERVABILITY_SCHEMA:
        raise ValueError(
            f"tactile observability schema must be {TEST_TUBE_TACTILE_OBSERVABILITY_SCHEMA!r}"
        )
    if payload.get("hardware_commands_dispatched") is not False:
        raise ValueError("tactile observability must state hardware_commands_dispatched=false")
    eligible = payload.get("runtime_gate_eligible") is True
    summary = {
        "provided": True,
        "source_schema": TEST_TUBE_TACTILE_OBSERVABILITY_SCHEMA,
        "source_digest_sha256": _canonical_digest(payload),
        "status": payload.get("status"),
        "runtime_gate_eligible": eligible,
        "mode": "runtime_qualified" if eligible else "shadow_only",
        "channels": payload.get("channels", {}),
        "synchronization": payload.get("synchronization", {}),
        "predicates": payload.get("predicates", {}),
        "threshold_fit": payload.get("threshold_fit", {}),
        "source_blockers": payload.get("blockers", []),
        "next_calibration_capture": payload.get("next_calibration_capture", []),
    }
    return summary, []


def _read_front_rgb_reference(
    payload: Mapping[str, Any] | None,
    *,
    tilt: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if payload is None:
        return (
            {"provided": False, "runtime_gate_eligible": False},
            ["front_rgb_pivot_reference_artifact_missing"],
        )
    if payload.get("schema") != TEST_TUBE_FRONT_RGB_REFERENCE_SCHEMA:
        raise ValueError(
            f"front RGB reference schema must be {TEST_TUBE_FRONT_RGB_REFERENCE_SCHEMA!r}"
        )
    if payload.get("hardware_commands_dispatched") is not False:
        raise ValueError("front RGB reference must state hardware_commands_dispatched=false")
    if payload.get("physical_3d_angle_claimed") is not False:
        raise ValueError("front RGB reference must not claim a physical 3D angle")
    target = _object(payload.get("target"), name="front_rgb_reference.target")
    target_axis = _number(
        target.get("target_image_axis_deg"),
        name="front_rgb_reference.target.target_image_axis_deg",
        minimum=0,
        maximum=180,
        include_minimum=True,
    )
    configured_axis = tilt["target_image_axis_deg"]
    if configured_axis is None or not math.isclose(
        target_axis, configured_axis, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("front RGB reference and config target image axis disagree")
    calibration_id = _string(
        payload.get("calibration_id"), name="front_rgb_reference.calibration_id"
    )
    if calibration_id != tilt["front_rgb_angle_calibration_id"]:
        raise ValueError("front RGB reference and config calibration ID disagree")
    eligible = payload.get("runtime_gate_eligible") is True
    return (
        {
            "provided": True,
            "source_schema": TEST_TUBE_FRONT_RGB_REFERENCE_SCHEMA,
            "source_digest_sha256": _canonical_digest(payload),
            "calibration_id": calibration_id,
            "target_image_axis_deg": target_axis,
            "runtime_gate_eligible": eligible,
            "status": payload.get("status"),
            "runtime_requirements": payload.get("runtime_requirements", []),
            "remaining_blocker": payload.get("remaining_blocker"),
        },
        [] if eligible else ["front_rgb_live_axis_gate_not_qualified"],
    )


def _read_rgb_tracker_validation(
    payload: Mapping[str, Any] | None,
    *,
    tilt: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if payload is None:
        return (
            {
                "provided": False,
                "runtime_gate_eligible": False,
                "offline_replay_passed": False,
            },
            ["front_rgb_tracker_validation_artifact_missing"],
        )
    if payload.get("schema") != TEST_TUBE_RGB_TRACKER_VALIDATION_SCHEMA:
        raise ValueError(
            "front RGB tracker validation schema must be "
            f"{TEST_TUBE_RGB_TRACKER_VALIDATION_SCHEMA!r}"
        )
    if payload.get("hardware_commands_dispatched") is not False:
        raise ValueError("front RGB tracker validation must dispatch no hardware commands")
    if payload.get("depth_used_for_tube_axis") is not False:
        raise ValueError("transparent-tube axis tracker must not claim D435 depth as tube depth")
    replay = _object(payload.get("offline_replay"), name="front_rgb_tracker.offline_replay")
    if replay.get("passed") is not True:
        raise ValueError("front RGB tracker offline replay must pass before preflight inclusion")
    target_axis = _number(
        replay.get("learned_target_axis_deg"),
        name="front_rgb_tracker.offline_replay.learned_target_axis_deg",
        minimum=0,
        maximum=180,
        include_minimum=True,
    )
    if not math.isclose(target_axis, tilt["target_image_axis_deg"], rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("front RGB tracker and config target image axis disagree")
    confirmations = _integer(
        replay.get("consecutive_frames_required"),
        name="front_rgb_tracker.offline_replay.consecutive_frames_required",
    )
    if confirmations != tilt["confirmation_windows"]:
        raise ValueError("front RGB tracker and config confirmation windows disagree")
    eligible = payload.get("runtime_gate_eligible") is True
    return (
        {
            "provided": True,
            "source_schema": TEST_TUBE_RGB_TRACKER_VALIDATION_SCHEMA,
            "source_digest_sha256": _canonical_digest(payload),
            "status": payload.get("status"),
            "offline_replay_passed": True,
            "runtime_gate_eligible": eligible,
            "tracked_fraction": replay.get("tracked_fraction"),
            "reviewed_axis_mae_deg": replay.get("reviewed_axis_mae_deg"),
            "reviewed_axis_max_error_deg": replay.get("reviewed_axis_max_error_deg"),
            "first_reclamp_frame": replay.get("first_reclamp_frame"),
            "reviewed_target_frame": replay.get("reviewed_target_frame"),
            "qualification_boundary": payload.get("qualification_boundary"),
        },
        [] if eligible else ["front_rgb_live_axis_gate_not_qualified"],
    )


def _semantic_trace(
    *,
    name: str,
    initial_state: TubeState,
    steps: list[dict[str, Any]],
    allowed_angle_sources: tuple[str, ...],
    expected_final_state: TubeState,
) -> dict[str, Any]:
    """Execute a deterministic predicate trace against the pure state evaluator."""

    state = initial_state
    receipts: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        expected_state = TubeState(step["state"])
        if state is not expected_state:
            raise RuntimeError(
                f"offline trace {name!r} step {index} expected {expected_state}, got {state}"
            )
        promoted = tuple(step["promoted_predicates"])
        decision = decide_test_tube_transition(
            state,
            promoted,
            allowed_angle_sources=allowed_angle_sources,
        )
        expected_next = TubeState(step["expected_next_state"])
        if decision.next_state is not expected_next:
            raise RuntimeError(
                f"offline trace {name!r} step {index} expected next {expected_next}, "
                f"got {decision.next_state}"
            )
        receipts.append(
            {
                "step": index,
                "state": state.value,
                "raw_observations": list(step.get("raw_observations", promoted)),
                "promoted_predicates": list(promoted),
                "shadow_observations": list(step.get("shadow_observations", [])),
                "event": decision.event,
                "action": decision.action,
                "next_state": decision.next_state.value,
                "recovery": decision.recovery,
            }
        )
        state = decision.next_state
    if state is not expected_final_state:
        raise RuntimeError(
            f"offline trace {name!r} ended in {state}, expected {expected_final_state}"
        )
    return {
        "name": name,
        "kind": "deterministic_offline_semantic_trace",
        "hardware_commands_dispatched": False,
        "manual_stage_calls_required": False,
        "initial_state": initial_state.value,
        "expected_final_state": expected_final_state.value,
        "result": "pass",
        "steps": receipts,
    }


def _deterministic_offline_traces(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Prove nominal/recovery dispatch order without claiming runtime readiness."""

    angle_sources = tuple(config["tilt"]["allowed_target_sources"])
    target_predicate = _ANGLE_PREDICATE_BY_SOURCE[angle_sources[0]]
    nominal = [
        {
            "state": "OBSERVE",
            "promoted_predicates": ["tube_pose_valid", "socket_frame_valid"],
            "expected_next_state": "APPROACH_TUBE",
        },
        {
            "state": "APPROACH_TUBE",
            "promoted_predicates": ["approach_pose_reached"],
            "expected_next_state": "SIDE_GRASP",
        },
        {
            "state": "SIDE_GRASP",
            "raw_observations": [
                "session_contact_profile_active",
                "contact_present",
                "gripper_state_valid",
                "grasp_width_in_range",
                "visual_tube_between_fingers",
            ],
            "promoted_predicates": ["grasp_established"],
            "expected_next_state": "EXTRACT",
        },
        {
            "state": "EXTRACT",
            "promoted_predicates": ["tube_clear_of_source", "contact_present"],
            "expected_next_state": "LEVEL_HELD_TUBE",
        },
        {
            "state": "LEVEL_HELD_TUBE",
            "promoted_predicates": [
                "tube_horizontal",
                "level_target_reached",
                "contact_present",
            ],
            "expected_next_state": "PIVOT_MICRO_LOOSEN",
        },
        {
            "state": "PIVOT_MICRO_LOOSEN",
            "promoted_predicates": ["one_microstep_complete"],
            "expected_next_state": "PIVOT_EVALUATE",
        },
        {
            "state": "PIVOT_EVALUATE",
            "promoted_predicates": ["contact_present", target_predicate],
            "expected_next_state": "RECLAMP",
        },
        {
            "state": "RECLAMP",
            "promoted_predicates": [
                "contact_present",
                "reclamp_command_settled",
                "pivot_target_confirmed",
            ],
            "expected_next_state": "APPROACH_SOCKET",
        },
        {
            "state": "APPROACH_SOCKET",
            "promoted_predicates": ["socket_standoff_reached"],
            "expected_next_state": "ALIGN_SOCKET",
        },
        {
            "state": "ALIGN_SOCKET",
            "promoted_predicates": ["tube_socket_axes_aligned", "tube_tip_at_socket_entry"],
            "expected_next_state": "INSERT_MICROSTEP",
        },
        {
            "state": "INSERT_MICROSTEP",
            "promoted_predicates": ["one_axial_step_complete"],
            "expected_next_state": "INSERT_EVALUATE",
        },
        {
            "state": "INSERT_EVALUATE",
            "promoted_predicates": ["insertion_depth_reached", "no_jam"],
            "expected_next_state": "RELEASE",
        },
        {
            "state": "RELEASE",
            "promoted_predicates": [
                "gripper_open",
                "contact_released",
                "tube_stays_supported",
            ],
            "expected_next_state": "RETREAT",
        },
        {
            "state": "RETREAT",
            "promoted_predicates": ["retreat_complete"],
            "expected_next_state": "COMPLETE",
        },
    ]
    recovery = [
        {
            "state": "PIVOT_EVALUATE",
            "raw_observations": ["contact_lost", target_predicate],
            "promoted_predicates": ["contact_lost", target_predicate],
            "expected_next_state": "RECLAMP_RECOVERY",
        },
        {
            "state": "RECLAMP_RECOVERY",
            "promoted_predicates": [
                "contact_present",
                "reclamp_command_settled",
                "pivot_budget_remaining",
            ],
            "expected_next_state": "PIVOT_EVALUATE",
        },
        {
            "state": "PIVOT_EVALUATE",
            "promoted_predicates": ["contact_present", target_predicate],
            "expected_next_state": "RECLAMP",
        },
    ]
    jam_shadow = [
        {
            "state": "INSERT_EVALUATE",
            "raw_observations": [
                "insertion_jam",
                "axial_progress_ok",
                "insertion_depth_remaining",
            ],
            "promoted_predicates": ["axial_progress_ok", "insertion_depth_remaining"],
            "shadow_observations": ["insertion_jam"],
            "expected_next_state": "INSERT_MICROSTEP",
        }
    ]
    traces = [
        _semantic_trace(
            name="nominal_full_sequence",
            initial_state=TubeState.OBSERVE,
            steps=nominal,
            allowed_angle_sources=angle_sources,
            expected_final_state=TubeState.COMPLETE,
        ),
        _semantic_trace(
            name="pivot_contact_loss_reclamp_resume",
            initial_state=TubeState.PIVOT_EVALUATE,
            steps=recovery,
            allowed_angle_sources=angle_sources,
            expected_final_state=TubeState.RECLAMP,
        ),
    ]
    if config["tactile"]["predicate_modes"]["insertion_jam"] == "shadow_only":
        traces.append(
            _semantic_trace(
                name="shadow_jam_does_not_trigger_recovery",
                initial_state=TubeState.INSERT_EVALUATE,
                steps=jam_shadow,
                allowed_angle_sources=angle_sources,
                expected_final_state=TubeState.INSERT_MICROSTEP,
            )
        )
    return traces


def compile_test_tube_preflight(
    *,
    stage_review: Mapping[str, Any],
    config: Mapping[str, Any],
    tactile_observability: Mapping[str, Any] | None = None,
    front_rgb_reference: Mapping[str, Any] | None = None,
    front_rgb_tracker_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a deterministic offline contract without authorizing hardware."""

    reviewed = _validate_review(stage_review)
    normalized = _validate_config(config)
    if not math.isclose(
        reviewed["target_angle_deg"],
        normalized["tilt"]["target_relative_deg"],
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError("stage review and config disagree on the tilt target")

    tactile_summary, blockers = _read_observability_summary(tactile_observability)
    front_rgb_summary, front_rgb_blockers = _read_front_rgb_reference(
        front_rgb_reference, tilt=normalized["tilt"]
    )
    blockers.extend(front_rgb_blockers)
    tracker_summary, tracker_blockers = _read_rgb_tracker_validation(
        front_rgb_tracker_validation,
        tilt=normalized["tilt"],
    )
    blockers.extend(tracker_blockers)
    evidence = normalized["evidence"]
    for key in ("tube_grasp_relation", "live_tube_pose", "socket_frame"):
        item = evidence[key]
        if item["path"] is None:
            blockers.append(f"{key}_artifact_missing")
        elif not item["reviewed"]:
            blockers.append(f"{key}_not_reviewed")
    tactile_ref = evidence["tactile_observability"]
    if tactile_ref["path"] is None:
        blockers.append("tactile_observability_path_missing")
    elif not tactile_ref["reviewed"]:
        blockers.append("tactile_observability_not_reviewed")
    front_rgb_ref = evidence["front_rgb_pivot_reference"]
    if front_rgb_ref["path"] is None:
        blockers.append("front_rgb_pivot_reference_path_missing")
    elif not front_rgb_ref["reviewed"]:
        blockers.append("front_rgb_pivot_reference_not_reviewed")
    tracker_ref = evidence["front_rgb_tracker_validation"]
    if tracker_ref["path"] is None:
        blockers.append("front_rgb_tracker_validation_path_missing")
    elif not tracker_ref["reviewed"]:
        blockers.append("front_rgb_tracker_validation_not_reviewed")
    for key in (
        "d435_live_source",
        "offline_runtime_orchestrator",
        "offline_predicate_builder",
        "rgb_tracker_qualification_tool",
        "rgb_tracker_qualification_evaluator",
    ):
        item = evidence[key]
        if item["path"] is None:
            blockers.append(f"{key}_path_missing")
        elif not item["reviewed"]:
            blockers.append(f"{key}_not_reviewed")
    tactile_modes = normalized["tactile"]["predicate_modes"]
    contact_bootstrap_contract = _contact_bootstrap_contract(normalized["tactile"])
    fresh_predicate = tactile_summary.get("predicates", {}).get("fresh_dual_tactile_samples", {})
    contact_bootstrap_ready = bool(
        tactile_modes["contact"] == "conditional_runtime_bootstrap"
        and fresh_predicate.get("runtime_gate_eligible") is True
        and evidence["gf220_sdk_calibrations"]["path"] is not None
        and evidence["gf220_sdk_calibrations"]["reviewed"]
    )
    if tactile_modes["contact"] == "conditional_runtime_bootstrap":
        if not contact_bootstrap_ready:
            blockers.append("contact_gate_bootstrap_inputs_missing")
        else:
            blockers.append("contact_gate_live_session_bootstrap_pending")
    elif tactile_modes["contact"] == "runtime_qualified":
        if not normalized["tactile"]["execution_qualified"]:
            blockers.append("contact_gate_not_execution_qualified")
    else:
        blockers.append("contact_gate_shadow_only")
    if tactile_modes["slip"] != "runtime_qualified":
        blockers.append("slip_gate_shadow_only")
    if tactile_modes["insertion_jam"] != "runtime_qualified":
        blockers.append("insertion_jam_gate_shadow_only")
    calibrated_angle_sources = {
        source
        for source, calibration_id in (
            (
                "tactile_relative_angle",
                normalized["tilt"]["tactile_angle_calibration_id"],
            ),
            (
                "front_rgb_calibrated",
                (
                    normalized["tilt"]["front_rgb_angle_calibration_id"]
                    if normalized["tilt"]["front_rgb_runtime_gate_qualified"]
                    else None
                ),
            ),
        )
        if calibration_id is not None and source in normalized["tilt"]["allowed_target_sources"]
    }
    if not calibrated_angle_sources:
        blockers.append("no_calibrated_45deg_trigger")
    if (
        "front_rgb_calibrated" in normalized["tilt"]["allowed_target_sources"]
        and normalized["tilt"]["target_image_axis_deg"] is None
    ):
        blockers.append("front_rgb_target_image_axis_not_fitted")
    jam = normalized["tactile"]["insertion_jam"]
    if jam["resistance_delta_threshold"] is None or jam["resistance_units"] is None:
        blockers.append("insertion_jam_resistance_threshold_not_calibrated")
    if normalized["motion"]["insertion_target_depth_m"] is None:
        blockers.append("insertion_target_depth_not_measured")
    blockers.extend(
        [
            "fresh_metric_T_base_tube_required",
            "metric_T_base_socket_required",
        ]
    )
    readiness = normalized["execution_readiness"]
    readiness_blockers = {
        "exact_tube_tcp_instantiated": "exact_T_tube_tcp_not_instantiated",
        "gripper_width_and_close_budget_grounded": (
            "gripper_width_and_close_budget_not_grounded"
        ),
        "collision_scene_validated": "hande_tube_rack_collision_scene_not_validated",
        "motion_bounds_validated": "motion_bounds_not_validated_against_live_geometry",
        "offline_runtime_orchestrator_available": "offline_runtime_orchestrator_missing",
        "offline_predicate_builder_available": "offline_predicate_builder_missing",
        "rgb_tracker_qualification_tool_available": (
            "rgb_tracker_qualification_tool_missing"
        ),
        "long_lived_front_rgb_source_available": "long_lived_front_rgb_source_missing",
        "live_hardware_adapter_available": "live_test_tube_hardware_adapter_missing",
        "live_feedback_predicate_producers_available": (
            "live_feedback_predicate_producers_missing"
        ),
        "support_after_release_verifier_available": (
            "support_after_release_verifier_missing"
        ),
        "robot_command_state_path_healthy": "robot_command_state_path_not_confirmed",
        "front_rgb_stream_healthy": "front_rgb_stream_health_not_confirmed",
    }
    hardware_blockers = [
        blocker for key, blocker in readiness_blockers.items() if not readiness[key]
    ]
    if tactile_modes["contact"] == "conditional_runtime_bootstrap":
        hardware_blockers.append("contact_gate_live_session_bootstrap_pending")
    blockers.extend(hardware_blockers)
    blockers = sorted(set(blockers))

    geometry_contract_ready = not any(
        blocker.startswith(("tube_grasp_relation", "live_tube_pose", "socket_frame"))
        for blocker in blockers
    )
    offline_contract_blockers = {
        "tactile_observability_artifact_missing",
        "tactile_observability_path_missing",
        "tactile_observability_not_reviewed",
        "contact_gate_bootstrap_inputs_missing",
        "front_rgb_pivot_reference_artifact_missing",
        "front_rgb_pivot_reference_path_missing",
        "front_rgb_pivot_reference_not_reviewed",
        "front_rgb_tracker_validation_artifact_missing",
        "front_rgb_tracker_validation_path_missing",
        "front_rgb_tracker_validation_not_reviewed",
        "d435_live_source_path_missing",
        "d435_live_source_not_reviewed",
        "offline_runtime_orchestrator_path_missing",
        "offline_runtime_orchestrator_not_reviewed",
        "offline_predicate_builder_path_missing",
        "offline_predicate_builder_not_reviewed",
        "rgb_tracker_qualification_tool_path_missing",
        "rgb_tracker_qualification_tool_not_reviewed",
        "rgb_tracker_qualification_evaluator_path_missing",
        "rgb_tracker_qualification_evaluator_not_reviewed",
    }
    offline_contract_ready = geometry_contract_ready and not any(
        blocker in offline_contract_blockers for blocker in blockers
    )
    # The reviewed geometry artifacts intentionally contain projective/candidate
    # contracts and null metric transforms.  They are enough to review the task,
    # but never enough to send a target to MoveIt.
    moveit_plan_only_ready = False
    runtime_gates_ready = not any(
        blocker
        in {
            "tactile_observability_artifact_missing",
            "tactile_observability_path_missing",
            "tactile_observability_not_reviewed",
            "contact_gate_bootstrap_inputs_missing",
            "contact_gate_not_execution_qualified",
            "contact_gate_shadow_only",
            "slip_gate_shadow_only",
            "insertion_jam_gate_shadow_only",
            "no_calibrated_45deg_trigger",
            "front_rgb_target_image_axis_not_fitted",
            "front_rgb_pivot_reference_artifact_missing",
            "front_rgb_pivot_reference_path_missing",
            "front_rgb_pivot_reference_not_reviewed",
            "front_rgb_tracker_validation_artifact_missing",
            "front_rgb_tracker_validation_path_missing",
            "front_rgb_tracker_validation_not_reviewed",
            "front_rgb_live_axis_gate_not_qualified",
            "insertion_jam_resistance_threshold_not_calibrated",
            "insertion_target_depth_not_measured",
            "live_feedback_predicate_producers_missing",
            "support_after_release_verifier_missing",
            "front_rgb_stream_health_not_confirmed",
        }
        for blocker in blockers
    )

    return {
        "schema": TEST_TUBE_PLAN_SCHEMA,
        "task_id": normalized["task_id"],
        "status": (
            "offline_contract_ready_live_grounding_required"
            if offline_contract_ready
            else "blocked_preflight"
        ),
        "execution_authorized": False,
        "hardware_execution_enabled": False,
        "hardware_boundary": {
            "dispatch_supported": False,
            "robot_or_gripper_imports": False,
            "subprocess_commands_present": False,
            "output_is_offline_review_contract_only": True,
        },
        "source_review": reviewed,
        "frames": normalized["frames"],
        "evidence": evidence,
        "authority_partition": _authority_partition(),
        "stage_contracts": _state_contracts(normalized),
        "transition_precedence": [
            "INVALID_OR_STALE_OR_OVERLOAD",
            "CONTACT_LOST_OR_SLIP",
            "TARGET_OR_JAM_OR_COMPLETION",
            "CONTINUE_MICROSTEP",
        ],
        "state_machine": {
            "initial_state": "OBSERVE",
            "terminal_states": ["COMPLETE", "HOLD"],
            "default_unsatisfied_guard_action": "collect_fresh_evidence",
        },
        "deterministic_offline_traces": {
            "purpose": "verify autonomous stage selection without hardware or manual stage calls",
            "runtime_readiness_claimed": False,
            "all_passed": True,
            "traces": _deterministic_offline_traces(normalized),
        },
        "transitions": _transition_contract(normalized),
        "recovery_limits": {
            "pivot_max_microsteps": normalized["motion"]["pivot_max_microsteps"],
            "maximum_pivot_opening_delta_m": normalized["tactile"]["max_pivot_opening_delta_m"],
            "maximum_reclamp_steps": normalized["tactile"]["max_reclamp_steps"],
            "maximum_jam_recoveries": normalized["motion"]["max_jam_recoveries"],
            "jam_retract_m": normalized["motion"]["jam_retract_m"],
            "on_budget_exhausted": "HOLD",
        },
        "gates": {
            "grasp_established": {
                "purpose": "authorize extraction only after a retained side grasp is confirmed",
                "condition": (
                    "contact_present AND gripper_state_valid AND grasp_width_in_range AND "
                    "(bilateral_contact OR visual_tube_between_fingers)"
                ),
                "contact_present_alone_is_sufficient": False,
                "required_producers": [
                    "tactile_contact_provider",
                    "gripper_state_provider",
                    "grasp_geometry_provider",
                ],
                "on_unconfirmed_contact": "remain_in_SIDE_GRASP_and_collect_evidence",
            },
            "pivot_contact": {
                "policy": "any_finger_valid_contact",
                "mode": tactile_modes["contact"],
                "implementation": contact_bootstrap_contract["implementation"],
                "symmetric_contact_required": False,
                "runtime_reclamp_triggers": [
                    "contact_lost",
                    *(["slip_detected"] if tactile_modes["slip"] == "runtime_qualified" else []),
                ],
                "trigger_action": "immediate_bounded_reclamp",
            },
            "slip": {
                "mode": tactile_modes["slip"],
                "may_trigger_transition": tactile_modes["slip"] == "runtime_qualified",
                "shadow_action": "record_only",
            },
            "pivot_target": {
                "target_semantics": normalized["tilt"]["target_semantics"],
                "target_relative_deg": normalized["tilt"]["target_relative_deg"],
                "target_image_axis_deg": normalized["tilt"]["target_image_axis_deg"],
                "tolerance_deg": normalized["tilt"]["tolerance_deg"],
                "confirmation_windows": normalized["tilt"]["confirmation_windows"],
                "allowed_target_sources": normalized["tilt"]["allowed_target_sources"],
                "condition": _angle_condition(normalized["tilt"]),
                "tactile_angle_calibration_id": normalized["tilt"]["tactile_angle_calibration_id"],
                "front_rgb_angle_calibration_id": normalized["tilt"][
                    "front_rgb_angle_calibration_id"
                ],
                "front_rgb_runtime_gate_qualified": normalized["tilt"][
                    "front_rgb_runtime_gate_qualified"
                ],
            },
            "insertion_jam": {
                **normalized["tactile"]["insertion_jam"],
                "mode": tactile_modes["insertion_jam"],
                "may_trigger_transition": tactile_modes["insertion_jam"] == "runtime_qualified",
                "condition": (
                    "low_axial_progress_for_confirmation_window AND "
                    "calibrated_resistance_or_shear_rise"
                ),
                "single_force_spike_is_not_a_jam": True,
                "response": "stop_retract_reobserve_realign",
            },
        },
        "motion_limits": {
            **normalized["motion"],
            "validation_status": "draft_review_bounds_not_geometry_validated",
        },
        "execution_readiness": normalized["execution_readiness"],
        "first_run_contact_bootstrap": {
            **contact_bootstrap_contract,
            "contract_inputs_ready": contact_bootstrap_ready,
            "live_session_windows_ready": False,
            "preflight_inputs_ready": contact_bootstrap_ready,
            **normalized["tactile"]["bootstrap"],
        },
        "tactile_observability": tactile_summary,
        "front_rgb_pivot_reference": front_rgb_summary,
        "front_rgb_tracker_validation": tracker_summary,
        "preflight": {
            "offline_contract_valid": True,
            "offline_task_contract_ready": offline_contract_ready,
            "geometry_contract_ready": geometry_contract_ready,
            "metric_execution_geometry_ready": False,
            "moveit_plan_only_ready": moveit_plan_only_ready,
            "runtime_transition_gates_ready": runtime_gates_ready,
            "hardware_execution_ready": False,
            "blocking_conditions": blockers,
            "hardware_blocking_conditions": sorted(hardware_blockers),
        },
        "provenance": {
            "config_digest_sha256": normalized["config_digest_sha256"],
            "demonstrated_trajectory_used_as_waypoints": False,
        },
    }


def _predicates(values: Iterable[str]) -> frozenset[str]:
    result = frozenset(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError("predicates must be non-empty strings")
    return result


def decide_test_tube_transition(
    state: TubeState | str,
    predicates: Iterable[str],
    *,
    allowed_angle_sources: Iterable[str] = (
        "tactile_relative_angle",
        "front_rgb_calibrated",
    ),
) -> TransitionDecision:
    """Evaluate a named-predicate snapshot with fixed recovery precedence.

    This evaluator intentionally returns semantic actions only.  A future
    audited runner must map one action at a time to bounded MoveIt/Hand-E
    primitives and persist the corresponding completion receipt.
    """

    current = TubeState(state)
    p = _predicates(predicates)
    angle_sources = tuple(allowed_angle_sources)
    if (
        not angle_sources
        or any(source not in _ANGLE_PREDICATE_BY_SOURCE for source in angle_sources)
        or len(angle_sources) != len(set(angle_sources))
    ):
        raise ValueError("allowed_angle_sources is invalid")
    angle_predicates = {_ANGLE_PREDICATE_BY_SOURCE[source] for source in angle_sources}
    if current in {TubeState.COMPLETE, TubeState.HOLD}:
        return TransitionDecision(current, current, "TERMINAL", "none", "state is terminal")
    if p & {"invalid_evidence", "stale_evidence", "overload"}:
        return TransitionDecision(
            current,
            TubeState.HOLD,
            "INVALID_OR_STALE_OR_OVERLOAD",
            "stop_and_hold",
            "runtime evidence is not safe to consume",
            recovery=True,
        )

    if current is TubeState.OBSERVE and {"tube_pose_valid", "socket_frame_valid"} <= p:
        return TransitionDecision(
            current,
            TubeState.APPROACH_TUBE,
            "LIVE_FRAMES_READY",
            "request_agentic_approach_plan",
            "both live task frames are available",
        )
    if current is TubeState.APPROACH_TUBE and "approach_pose_reached" in p:
        return TransitionDecision(
            current,
            TubeState.SIDE_GRASP,
            "APPROACH_COMPLETE",
            "ground_side_grasp_relation",
            "collision-free standoff was reached",
        )
    if current is TubeState.SIDE_GRASP:
        if "grasp_established" in p:
            return TransitionDecision(
                current,
                TubeState.EXTRACT,
                "GRASP_ESTABLISHED",
                "request_agentic_extraction",
                "gripper state and tactile/visual geometry jointly confirm a retained grasp",
            )
        if "grasp_close_budget_exhausted" in p:
            return TransitionDecision(
                current,
                TubeState.HOLD,
                "GRASP_BUDGET_EXHAUSTED",
                "stop_and_hold",
                "no contact was acquired within the bounded close budget",
                recovery=True,
            )
        if "contact_present" in p:
            return TransitionDecision(
                current,
                current,
                "CONTACT_PRESENT_GRASP_UNCONFIRMED",
                "collect_grasp_confirmation_evidence",
                "contact alone may be a rack or one-sided touch and cannot authorize extraction",
            )
        return TransitionDecision(
            current,
            current,
            "CONTACT_NOT_YET_PRESENT",
            "bounded_close_probe",
            "continue only within the configured grasp budget",
        )
    if current is TubeState.EXTRACT and ("contact_lost" in p or "slip_detected" in p):
        return TransitionDecision(
            current,
            TubeState.HOLD,
            "EXTRACTION_CONTACT_FAILURE",
            "stop_and_hold",
            "contact was lost while clearing the source rack",
            recovery=True,
        )
    if (
        current is TubeState.EXTRACT
        and {
            "tube_clear_of_source",
            "contact_present",
        }
        <= p
    ):
        return TransitionDecision(
            current,
            TubeState.LEVEL_HELD_TUBE,
            "EXTRACTION_COMPLETE",
            "servo_held_tube_to_horizontal",
            "tube is clear and remains rigidly held; level it before loosening",
        )
    if current is TubeState.LEVEL_HELD_TUBE:
        if "contact_lost" in p or "slip_detected" in p:
            return TransitionDecision(
                current,
                TubeState.HOLD,
                "LEVELING_CONTACT_FAILURE",
                "stop_and_hold",
                "contact was lost or a stationary slip was confirmed while leveling",
                recovery=True,
            )
        if {
            "tube_horizontal",
            "level_target_reached",
            "contact_present",
        } <= p:
            return TransitionDecision(
                current,
                TubeState.PIVOT_MICRO_LOOSEN,
                "HELD_TUBE_LEVEL_STABLE",
                "one_micro_loosen",
                "the live tube axis and measured TCP target are stably level",
            )
        return TransitionDecision(
            current,
            current,
            "HELD_TUBE_LEVELING",
            "continue_closed_loop_leveling",
            "keep the grasp rigid until tube geometry and TCP convergence agree",
        )
    if current in {TubeState.PIVOT_MICRO_LOOSEN, TubeState.PIVOT_EVALUATE}:
        if "contact_lost" in p or "slip_detected" in p:
            return TransitionDecision(
                current,
                TubeState.RECLAMP_RECOVERY,
                "CONTACT_LOST_OR_SLIP",
                "immediate_bounded_reclamp",
                "pivot recovery has precedence over angle progress",
                recovery=True,
            )
        if current is TubeState.PIVOT_MICRO_LOOSEN and "one_microstep_complete" in p:
            return TransitionDecision(
                current,
                TubeState.PIVOT_EVALUATE,
                "MICROSTEP_COMPLETE",
                "collect_fresh_angle_and_contact_window",
                "evaluate only after the bounded opening step settles",
            )
        angle_reached = bool(p & angle_predicates)
        if current is TubeState.PIVOT_EVALUATE and "contact_present" in p and angle_reached:
            return TransitionDecision(
                current,
                TubeState.RECLAMP,
                "PIVOT_TARGET_REACHED",
                "reclamp_at_target",
                "one calibrated angle source reached the target while contact remained",
            )
        if (
            current is TubeState.PIVOT_EVALUATE
            and {
                "contact_present",
                "pivot_budget_remaining",
            }
            <= p
        ):
            return TransitionDecision(
                current,
                TubeState.PIVOT_MICRO_LOOSEN,
                "PIVOT_CONTINUE",
                "one_micro_loosen",
                "target is not reached and bounded pivot budget remains",
            )
        if current is TubeState.PIVOT_EVALUATE and "contact_present" in p:
            return TransitionDecision(
                current,
                TubeState.RECLAMP_RECOVERY,
                "PIVOT_BUDGET_EXHAUSTED",
                "reclamp_then_hold",
                "target was not reached within the bounded opening budget",
                recovery=True,
            )
    if (
        current is TubeState.RECLAMP
        and {
            "contact_present",
            "reclamp_command_settled",
            "pivot_target_confirmed",
        }
        <= p
    ):
        return TransitionDecision(
            current,
            TubeState.APPROACH_SOCKET,
            "RECLAMP_STABLE",
            "request_agentic_socket_approach",
            "hold was restored and relative rotation stopped",
        )
    if (
        current is TubeState.RECLAMP_RECOVERY
        and {
            "contact_present",
            "reclamp_command_settled",
            "pivot_budget_remaining",
        }
        <= p
    ):
        return TransitionDecision(
            current,
            TubeState.PIVOT_EVALUATE,
            "RECOVERY_RECLAMP_STABLE",
            "collect_fresh_angle_and_contact_window",
            "hold was restored; re-evaluate before any further opening",
            recovery=True,
        )
    if (
        current is TubeState.RECLAMP_RECOVERY
        and {
            "contact_present",
            "reclamp_command_settled",
        }
        <= p
    ):
        return TransitionDecision(
            current,
            TubeState.HOLD,
            "PIVOT_RECOVERY_BUDGET_EXHAUSTED",
            "stop_and_hold",
            "hold was restored but no pivot budget remains",
            recovery=True,
        )
    if current is TubeState.APPROACH_SOCKET and "socket_standoff_reached" in p:
        return TransitionDecision(
            current,
            TubeState.ALIGN_SOCKET,
            "SOCKET_STANDOFF_REACHED",
            "ground_tube_socket_axis_relation",
            "free-space approach is complete",
        )
    if (
        current is TubeState.ALIGN_SOCKET
        and {
            "tube_socket_axes_aligned",
            "tube_tip_at_socket_entry",
        }
        <= p
    ):
        return TransitionDecision(
            current,
            TubeState.INSERT_MICROSTEP,
            "INSERTION_ALIGNED",
            "one_axial_insert_step",
            "tube and socket are aligned at the entry",
        )
    if current is TubeState.INSERT_MICROSTEP and "one_axial_step_complete" in p:
        return TransitionDecision(
            current,
            TubeState.INSERT_EVALUATE,
            "AXIAL_STEP_COMPLETE",
            "collect_fresh_progress_and_resistance_window",
            "evaluate after each bounded axial advance",
        )
    if current is TubeState.INSERT_EVALUATE:
        if "insertion_jam" in p:
            if "jam_recovery_budget_remaining" in p:
                return TransitionDecision(
                    current,
                    TubeState.JAM_RECOVERY_RETRACT,
                    "INSERTION_JAM",
                    "stop_push_and_retract",
                    "low progress plus calibrated resistance requires realignment",
                    recovery=True,
                )
            return TransitionDecision(
                current,
                TubeState.HOLD,
                "JAM_RECOVERY_EXHAUSTED",
                "stop_and_hold",
                "the bounded jam recovery budget is exhausted",
                recovery=True,
            )
        if {"insertion_seated", "no_jam"} <= p:
            return TransitionDecision(
                current,
                TubeState.RELEASE,
                "INSERTION_SEATED",
                "verify_support_then_open",
                "debounced balanced resistance and low progress indicate a seated tube",
            )
        if {"insertion_depth_reached", "no_jam"} <= p:
            return TransitionDecision(
                current,
                TubeState.RELEASE,
                "INSERTION_COMPLETE",
                "verify_support_then_open",
                "measured insertion depth is reached without a jam",
            )
        if {"axial_progress_ok", "insertion_depth_remaining"} <= p:
            return TransitionDecision(
                current,
                TubeState.INSERT_MICROSTEP,
                "INSERTION_CONTINUE",
                "one_axial_insert_step",
                "axial progress is healthy and depth remains",
            )
    if (
        current is TubeState.JAM_RECOVERY_RETRACT
        and {
            "retract_complete",
            "scene_reobserved",
        }
        <= p
    ):
        return TransitionDecision(
            current,
            TubeState.ALIGN_SOCKET,
            "JAM_RECOVERY_COMPLETE",
            "realign_socket_entry",
            "bounded retract and live re-observation completed",
            recovery=True,
        )
    if (
        current is TubeState.RELEASE
        and {
            "gripper_open",
            "contact_released",
            "tube_stays_supported",
        }
        <= p
    ):
        return TransitionDecision(
            current,
            TubeState.RETREAT,
            "RELEASE_VERIFIED",
            "request_agentic_retreat",
            "the rack supports the tube after contact release",
        )
    if current is TubeState.RETREAT and "retreat_complete" in p:
        return TransitionDecision(
            current,
            TubeState.COMPLETE,
            "TASK_COMPLETE",
            "none",
            "safe retreat completed",
        )
    return TransitionDecision(
        current,
        current,
        "WAIT_FOR_PREDICATE",
        "collect_fresh_evidence",
        "no transition guard is currently satisfied",
    )


__all__ = [
    "ControlAuthority",
    "REFINED_REVIEW_STAGE_ORDER",
    "REVIEW_STAGE_ORDER",
    "TEST_TUBE_CONFIG_SCHEMA",
    "TEST_TUBE_CONTACT_BOOTSTRAP_MODULE",
    "TEST_TUBE_CONTACT_BOOTSTRAP_SCHEMA",
    "TEST_TUBE_FRONT_RGB_REFERENCE_SCHEMA",
    "TEST_TUBE_GEOMETRY_SCHEMAS",
    "TEST_TUBE_PLAN_SCHEMA",
    "TEST_TUBE_REVIEW_SCHEMA",
    "TEST_TUBE_REFINED_REVIEW_SCHEMA",
    "TEST_TUBE_RGB_TRACKER_VALIDATION_SCHEMA",
    "TEST_TUBE_TACTILE_OBSERVABILITY_SCHEMA",
    "TransitionDecision",
    "TubeState",
    "compile_test_tube_preflight",
    "decide_test_tube_transition",
    "validate_test_tube_geometry_artifact",
]
