"""Compile a review-only stage contract for removing one nested top cup.

This module is deliberately plan-only.  It consumes reviewed demonstration
artifacts, records which component owns each target, and never imports a robot
driver or sends a command.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .franka import validate_rigid_transform

CUP_STAGE_PLAN_SCHEMA = "demo2gripper.top_cup_stage_plan.v3"
CUP_RIM_RELATION_SCHEMA = "demo2gripper.cup_rim_relation.v1"
SEMANTIC_STAGES_SCHEMA = "demo2gripper.image_only_semantic_stages.v1"
TACTILE_POLICY_SCHEMA = "demo2gripper.tactile_supervision_config.v1"

DEMONSTRATION_STAGE_ORDER = (
    "APPROACH",
    "PRESHAPE",
    "GRASP",
    "LIFT",
    "TRANSPORT",
    "PLACE",
    "RELEASE",
    "RETREAT",
)

CUP_STAGE_ORDER = (
    "OBSERVE",
    "GROUND_TOP_RIM",
    "APPROACH",
    "RIM_TOUCH_ALIGN",
    "GENTLE_PINCH",
    "SEPARATE",
    "VERIFY_ONE",
    "CLEAR_LIFT",
    "TRANSPORT",
    "PLACE",
    "RELEASE",
    "RETREAT",
)

REQUIRED_TACTILE_PREDICATES = (
    "rim_capture_verified",
    "cup_remains_coupled",
    "release_confirmed",
)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_relation(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != CUP_RIM_RELATION_SCHEMA:
        raise ValueError(f"cup rim relation schema must be {CUP_RIM_RELATION_SCHEMA!r}")
    if payload.get("status") != "review_only" or payload.get("execution_authorized") is not False:
        raise ValueError("cup rim relation must be review_only and execution_authorized=false")
    if payload.get("relation_role") != "human_demo_geometry_only":
        raise ValueError("cup rim relation must be labelled human_demo_geometry_only")
    quality = _mapping(payload.get("quality"), name="cup rim relation quality")
    if quality.get("review_only") is not True or quality.get("execution_authorized") is not False:
        raise ValueError("cup rim relation quality must preserve its review-only boundary")
    validate_rigid_transform(
        np.asarray(payload.get("T_camera_rim"), dtype=float), name="T_camera_rim"
    )
    validate_rigid_transform(
        np.asarray(payload.get("T_rim_human_pinch_tcp"), dtype=float),
        name="T_rim_human_pinch_tcp",
    )
    width = payload.get("demonstrated_pinch_width_m")
    if isinstance(width, bool) or not isinstance(width, (int, float)) or not np.isfinite(width):
        raise ValueError("demonstrated_pinch_width_m must be finite")
    if width <= 0.0:
        raise ValueError("demonstrated_pinch_width_m must be positive")


def _semantic_stage_index(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if payload.get("schema") != SEMANTIC_STAGES_SCHEMA:
        raise ValueError(f"semantic stages schema must be {SEMANTIC_STAGES_SCHEMA!r}")
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list):
        raise ValueError("semantic stages must be an array")
    indexed: dict[str, dict[str, Any]] = {}
    previous_end = -1
    for raw in raw_stages:
        item = _mapping(raw, name="semantic stage")
        name = item.get("name")
        start = item.get("start_frame")
        end_inclusive = item.get("end_frame")
        if not isinstance(name, str) or not name:
            raise ValueError("each semantic stage needs a non-empty name")
        if name in indexed:
            raise ValueError(f"duplicate semantic stage {name!r}")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (start, end_inclusive)
        ):
            raise ValueError(f"semantic stage {name} frame bounds must be integers")
        assert isinstance(start, int) and isinstance(end_inclusive, int)
        if start < 0 or end_inclusive < start:
            raise ValueError(f"semantic stage {name} has an invalid frame interval")
        if start <= previous_end:
            raise ValueError("semantic stages must be ordered and non-overlapping")
        previous_end = end_inclusive
        indexed[name] = {
            "frames_half_open": [start, end_inclusive + 1],
            "confidence": item.get("confidence"),
            "evidence": item.get("evidence"),
        }
    if tuple(indexed) != DEMONSTRATION_STAGE_ORDER:
        raise ValueError(
            "semantic stages must exactly match the reviewed top-cup sequence: "
            f"{DEMONSTRATION_STAGE_ORDER!r}"
        )
    return indexed


def _validate_relation_intervals(
    relation: Mapping[str, Any], semantic: Mapping[str, Mapping[str, Any]]
) -> None:
    raw = _mapping(
        relation.get("stage_intervals_half_open"),
        name="cup rim relation stage_intervals_half_open",
    )
    if tuple(raw) != DEMONSTRATION_STAGE_ORDER:
        raise ValueError("cup rim relation stage intervals do not cover the reviewed sequence")
    for name in DEMONSTRATION_STAGE_ORDER:
        values = raw[name]
        if not isinstance(values, list) or values != semantic[name]["frames_half_open"]:
            raise ValueError(f"cup rim relation and semantic stages disagree for {name}")


def _validate_tactile_policy(
    payload: Mapping[str, Any],
) -> tuple[bool, dict[str, str]]:
    if payload.get("schema") != TACTILE_POLICY_SCHEMA:
        raise ValueError(f"tactile policy schema must be {TACTILE_POLICY_SCHEMA!r}")
    profile_id = payload.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("tactile policy needs a non-empty profile_id")
    status = payload.get("status")
    if status not in {"review_only", "ready"}:
        raise ValueError("tactile policy status must be review_only or ready")
    predicates = _mapping(payload.get("predicates"), name="tactile predicates")
    missing = [name for name in REQUIRED_TACTILE_PREDICATES if name not in predicates]
    if missing:
        raise ValueError(f"tactile policy is missing predicates: {missing}")
    capture = _mapping(
        predicates.get("rim_capture_verified"),
        name="rim_capture_verified predicate",
    )
    capture_terms = capture.get("all")
    if not isinstance(capture_terms, list) or not {
        "inner_rim_match",
        "outer_rim_match",
    }.issubset(capture_terms):
        raise ValueError(
            "rim_capture_verified must require independent inner_rim_match and "
            "outer_rim_match predicates"
        )
    sensors = _mapping(payload.get("sensors"), name="tactile sensors")
    contact_roles: dict[str, str] = {}
    for side in ("left", "right"):
        sensor = _mapping(sensors.get(side), name=f"tactile sensor {side}")
        if sensor.get("required") is not True:
            raise ValueError(f"tactile sensor {side} must be required")
        role = sensor.get("contact_role")
        if role not in {"inner", "outer"}:
            raise ValueError(
                f"tactile sensor {side} contact_role must be inner or outer"
            )
        contact_roles[side] = role
    if set(contact_roles.values()) != {"inner", "outer"}:
        raise ValueError(
            "tactile policy must assign exactly one inner-rim and one outer-rim role"
        )
    qualified = status == "ready" and payload.get("execution_qualified") is True
    if status == "ready" and not qualified:
        raise ValueError("a ready tactile policy must set execution_qualified=true")
    return qualified, contact_roles


def _evidence(
    semantic: Mapping[str, Mapping[str, Any]], *names: str
) -> dict[str, Any]:
    return {
        "role": "semantic_intent_only",
        "demonstrated_stages": list(names),
        "frame_intervals_half_open": [semantic[name]["frames_half_open"] for name in names],
        "confidence": [semantic[name]["confidence"] for name in names],
    }


def _stage(
    name: str,
    *,
    authority: str,
    planner_or_action: str,
    evidence: dict[str, Any],
    inputs: tuple[str, ...],
    preconditions: tuple[str, ...],
    completion: tuple[str, ...],
    failure_transition: str,
    next_stage: str | None,
    target_contract: Mapping[str, Any] | None = None,
    tactile_gate_mode: str = "not_applicable",
) -> dict[str, Any]:
    return {
        "name": name,
        "authority": authority,
        "planner_or_action": planner_or_action,
        "demonstration_evidence": evidence,
        "artifact_inputs": list(inputs),
        "target_contract": None if target_contract is None else dict(target_contract),
        "preconditions": list(preconditions),
        "completion": list(completion),
        "failure_transition": failure_transition,
        "next_on_success": next_stage,
        "tactile_gate_mode": tactile_gate_mode,
        "hardware_execution_enabled": False,
    }


def compile_top_cup_stage_plan(
    *,
    cup_rim_relation: Mapping[str, Any],
    semantic_stages: Mapping[str, Any],
    tactile_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the deterministic, non-executable top-cup stage contract."""

    relation = _mapping(cup_rim_relation, name="cup_rim_relation")
    semantics = _mapping(semantic_stages, name="semantic_stages")
    tactile = _mapping(tactile_policy, name="tactile_policy")
    _validate_relation(relation)
    semantic = _semantic_stage_index(semantics)
    _validate_relation_intervals(relation, semantic)
    tactile_qualified, contact_roles = _validate_tactile_policy(tactile)
    tactile_mode = "qualified_profile_review_only" if tactile_qualified else "shadow"

    relation_target = {
        "frame": "live_top_rim",
        "transform": "T_rim_human_pinch_tcp",
        "source": "cup_rim_relation",
        "allowed_use": "single interaction endpoint only",
        "demonstrated_path_replay": False,
    }
    agentic_target = {
        "source": "online_agent_from_live_scene",
        "demonstrated_path_replay": False,
    }
    stages = (
        _stage(
            "OBSERVE",
            authority="online_perception",
            planner_or_action="capture_and_validate_external_rgbd",
            evidence=_evidence(semantic, "APPROACH"),
            inputs=("live_external_rgbd",),
            preconditions=("camera_calibration_valid", "workspace_clear"),
            completion=("fresh_scene_observation", "cup_stack_candidate_visible"),
            failure_transition="HOLD_REOBSERVE",
            next_stage="GROUND_TOP_RIM",
        ),
        _stage(
            "GROUND_TOP_RIM",
            authority="online_perception",
            planner_or_action="fit_live_top_rim_center_and_axis",
            evidence=_evidence(semantic, "PRESHAPE"),
            inputs=("live_external_rgbd", "cup_rim_relation"),
            preconditions=("fresh_scene_observation",),
            completion=(
                "top_rim_pose_stable",
                "rim_contact_candidates_grounded",
            ),
            failure_transition="HOLD_CHANGE_VIEW_OR_TOOL",
            next_stage="APPROACH",
        ),
        _stage(
            "APPROACH",
            authority="online_agent_moveit",
            planner_or_action="moveit_free_space_plan",
            evidence=_evidence(semantic, "APPROACH"),
            inputs=("live_top_rim_pose", "current_robot_state", "planning_scene"),
            target_contract={**agentic_target, "goal": "collision_free_rim_standoff"},
            preconditions=("live_rim_anchor_fresh", "ik_and_collision_plan_valid"),
            completion=("rim_standoff_reached", "unexpected_contact_false"),
            failure_transition="CANCEL_RETREAT_REOBSERVE",
            next_stage="RIM_TOUCH_ALIGN",
            tactile_gate_mode=tactile_mode,
        ),
        _stage(
            "RIM_TOUCH_ALIGN",
            authority="demonstration_relative_pose",
            planner_or_action="ground_rim_relative_endpoint_then_bounded_cartesian_align",
            evidence=_evidence(semantic, "PRESHAPE", "GRASP"),
            inputs=("live_top_rim_pose", "cup_rim_relation", "current_robot_state"),
            target_contract=relation_target,
            preconditions=("live_rim_anchor_fresh", "rim_relative_target_reviewed"),
            completion=("rim_touch_alignment_pose_reached", "unexpected_contact_false"),
            failure_transition="UNLOAD_RETREAT_REGROUND",
            next_stage="GENTLE_PINCH",
            tactile_gate_mode=tactile_mode,
        ),
        _stage(
            "GENTLE_PINCH",
            authority="tactile_probe_policy",
            planner_or_action="cup_tactile_pre_lift_gate_state_machine",
            evidence=_evidence(semantic, "GRASP"),
            inputs=("tactile_policy", "hande_state", "fixed_rim_task_frame"),
            target_contract={
                "tcp_motion_during_loaded_contact": "forbidden",
                "demonstrated_path_replay": False,
                "decision_map": {
                    "NO_CONTACT": "PROBE_DOWN",
                    "MULTI_RIM_OR_TOO_DEEP": "UNLOAD_THEN_RAISE_AND_REPROBE",
                    "INNER_OUTER_RIM_MATCH": "ALLOW_LIFT",
                    "ROLE_MISMATCH": "UNLOAD_THEN_REALIGN_AND_REPROBE",
                    "UNKNOWN": "STOP",
                },
                "contact_role_by_finger": contact_roles,
                "contact_role_assignment_scope": (
                    "current_probe_hypothesis; may reverse after replanning"
                ),
                "image_symmetry_required": False,
                "normal_load_relation": (
                    "optional_calibrated_inner_outer_ratio_independent_of_images"
                ),
                "lift_gate_predicates": [
                    "rim_capture_verified",
                    "inner_rim_match",
                    "outer_rim_match",
                    "tactile_lift_permission",
                ],
                "front_rgbd_cup_counting": False,
                "policy_profile_id": tactile["profile_id"],
            },
            preconditions=("rim_touch_alignment_pose_reached", "sensor_healthy"),
            completion=(
                "rim_capture_verified",
                "inner_rim_match",
                "outer_rim_match",
                "tactile_lift_permission",
            ),
            failure_transition="UNLOAD_HOLD_OR_BOUNDED_REALIGN",
            next_stage="SEPARATE",
            tactile_gate_mode=tactile_mode,
        ),
        _stage(
            "SEPARATE",
            authority="online_agent_task_geometry",
            planner_or_action="bounded_cartesian_micro_lift_along_live_cup_axis",
            evidence=_evidence(semantic, "LIFT"),
            inputs=("live_top_rim_pose", "tactile_policy", "tactile_gate_journal"),
            target_contract={**agentic_target, "goal": "small_single-cup_separation"},
            preconditions=(
                "rim_capture_verified",
                "inner_rim_match",
                "outer_rim_match",
                "tactile_lift_permission",
            ),
            completion=("bounded_separation_complete", "cup_remains_coupled"),
            failure_transition="STOP_RETURN_UNLOAD_REOBSERVE",
            next_stage="VERIFY_ONE",
            tactile_gate_mode=tactile_mode,
        ),
        _stage(
            "VERIFY_ONE",
            authority="tactile_hold_verification",
            planner_or_action="hold_and_verify_inner_outer_rim_coupling",
            evidence=_evidence(semantic, "LIFT"),
            inputs=("tactile_policy", "hande_state", "tactile_gate_journal"),
            preconditions=(
                "bounded_separation_complete",
                "inner_rim_match",
                "outer_rim_match",
                "robot_stationary",
            ),
            completion=("prelift_role_match_gate_preserved", "cup_remains_coupled"),
            failure_transition="STOP_HOLD_OR_SLOW_RETURN",
            next_stage="CLEAR_LIFT",
            tactile_gate_mode=tactile_mode,
        ),
        _stage(
            "CLEAR_LIFT",
            authority="online_agent_moveit",
            planner_or_action="moveit_clearance_lift",
            evidence=_evidence(semantic, "LIFT"),
            inputs=("verified_held_cup", "current_robot_state", "planning_scene"),
            target_contract={**agentic_target, "goal": "safe_stack_clearance"},
            preconditions=(
                "prelift_role_match_gate_preserved",
                "cup_remains_coupled",
            ),
            completion=("stack_clearance_reached", "cup_remains_coupled"),
            failure_transition="CANCEL_HOLD_OR_SLOW_LOWER",
            next_stage="TRANSPORT",
            tactile_gate_mode=tactile_mode,
        ),
        _stage(
            "TRANSPORT",
            authority="online_agent_moveit",
            planner_or_action="moveit_free_space_plan",
            evidence=_evidence(semantic, "TRANSPORT"),
            inputs=("live_destination", "current_robot_state", "planning_scene"),
            target_contract={**agentic_target, "goal": "destination_preplace"},
            preconditions=("stack_clearance_reached", "cup_remains_coupled"),
            completion=("destination_preplace_reached", "cup_remains_coupled"),
            failure_transition="CANCEL_AND_HOLD",
            next_stage="PLACE",
            tactile_gate_mode=tactile_mode,
        ),
        _stage(
            "PLACE",
            authority="online_agent_moveit",
            planner_or_action="moveit_bounded_placement_descent",
            evidence=_evidence(semantic, "PLACE"),
            inputs=("live_destination", "current_robot_state", "planning_scene"),
            target_contract={**agentic_target, "goal": "supported_cup_pose"},
            preconditions=("destination_preplace_reached", "cup_remains_coupled"),
            completion=("placement_support_verified",),
            failure_transition="CANCEL_HOLD_REOBSERVE_DESTINATION",
            next_stage="RELEASE",
            tactile_gate_mode=tactile_mode,
        ),
        _stage(
            "RELEASE",
            authority="task_intent",
            planner_or_action="bounded_hande_open",
            evidence=_evidence(semantic, "RELEASE"),
            inputs=("hande_state", "tactile_policy", "external_rgbd_tracking"),
            preconditions=("placement_support_verified",),
            completion=("release_confirmed", "object_stays_placed"),
            failure_transition="HOLD_REOBSERVE",
            next_stage="RETREAT",
            tactile_gate_mode=tactile_mode,
        ),
        _stage(
            "RETREAT",
            authority="online_agent_moveit",
            planner_or_action="moveit_free_space_plan",
            evidence=_evidence(semantic, "RETREAT"),
            inputs=("current_robot_state", "planning_scene", "placed_cup_pose"),
            target_contract={**agentic_target, "goal": "safe_retreat"},
            preconditions=("release_confirmed", "object_stays_placed"),
            completion=("retreat_clearance_reached", "unexpected_contact_false"),
            failure_transition="CANCEL_AND_HOLD",
            next_stage=None,
            tactile_gate_mode=tactile_mode,
        ),
    )
    assert tuple(stage["name"] for stage in stages) == CUP_STAGE_ORDER

    blockers = [
        "review_only_compiler",
        "operator_review_required",
        "live_top_rim_grounding_required",
        "hande_embodiment_ik_collision_and_trajectory_validation_required",
        "live_inner_outer_role_assignment_and_templates_required",
        "cup_specific_runtime_executor_not_implemented",
    ]
    if not tactile_qualified:
        blockers.append("tactile_supervision_shadow_only")
    return {
        "schema": CUP_STAGE_PLAN_SCHEMA,
        "status": "review_only",
        "execution_authorized": False,
        "hardware_execution_enabled": False,
        "task": "remove_exactly_one_top_cup_from_nested_stack",
        "stage_order": list(CUP_STAGE_ORDER),
        "policy_boundary": {
            "demonstration_relative_pose_stages": ["RIM_TOUCH_ALIGN"],
            "agentic_moveit_stages": [
                "APPROACH",
                "CLEAR_LIFT",
                "TRANSPORT",
                "PLACE",
                "RETREAT",
            ],
            "demonstrated_trajectory_replay": False,
            "tactile_probe_stage": "GENTLE_PINCH",
            "pre_lift_authority": "role_aware_inner_outer_rim_gate",
            "image_symmetry_required": False,
            "front_rgbd_cup_counting_used": False,
        },
        "tactile_supervision": {
            "profile_id": tactile["profile_id"],
            "mode": tactile_mode,
            "may_command_hardware": False,
            "required_predicates": list(REQUIRED_TACTILE_PREDICATES),
        },
        "source_digests_sha256": {
            "cup_rim_relation": _canonical_digest(relation),
            "semantic_stages": _canonical_digest(semantics),
            "tactile_policy": _canonical_digest(tactile),
        },
        "stages": list(stages),
        "blocking_conditions": blockers,
    }


__all__ = [
    "CUP_STAGE_ORDER",
    "CUP_STAGE_PLAN_SCHEMA",
    "compile_top_cup_stage_plan",
]
