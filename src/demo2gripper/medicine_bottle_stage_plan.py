"""Compile reviewed medicine-bottle demo analysis into a non-executable contract.

The analysis artifact records what happened in one human demonstration.  This
module validates that evidence and states which runtime component would own
each target.  It deliberately does not generate robot poses, replay the human
path, import a robot driver, or authorize hardware execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

MEDICINE_BOTTLE_ANALYSIS_SCHEMA = "demo2gripper.medicine_bottle_demo_analysis.v1"
MEDICINE_BOTTLE_STAGE_CONTRACT_SCHEMA = (
    "demo2gripper.medicine_bottle_stage_contract.v1"
)

DEMONSTRATED_STAGE_ORDER = (
    "APPROACH_CAP",
    "CAP_GRASP_PRESS_ENGAGE",
    "TWIST_1",
    "REGRASP_1",
    "TWIST_2",
    "REGRASP_2",
    "TWIST_3",
    "REGRASP_3",
    "FINAL_TWIST_AND_RELEASE_CHECK",
    "LIFT_CAP",
    "PLACE_CAP_ASIDE",
)

EXPECTED_SOURCE_AUTHORITIES = {
    "APPROACH_CAP": "online_agent_plan",
    "CAP_GRASP_PRESS_ENGAGE": "object_relative_pose_plus_tactile_supervision",
    "TWIST_1": "task_frame_rotation_about_bottle_axis",
    "REGRASP_1": "agentic_wrist_reset_keep_cap_contact",
    "TWIST_2": "task_frame_rotation_about_bottle_axis",
    "REGRASP_2": "agentic_wrist_reset_keep_cap_contact",
    "TWIST_3": "task_frame_rotation_about_bottle_axis",
    "REGRASP_3": "agentic_wrist_reset_keep_cap_contact",
    "FINAL_TWIST_AND_RELEASE_CHECK": (
        "closed_loop_axial_force_torque_and_relative_motion"
    ),
    "LIFT_CAP": "online_agent_plan_after_thread_release",
    "PLACE_CAP_ASIDE": "online_agent_plan",
}

AGENTIC_DEMONSTRATED_STAGES = (
    "APPROACH_CAP",
    "PLACE_CAP_ASIDE",
)

OBJECT_RELATIVE_STAGES = (
    "CAP_GRASP_PRESS_ENGAGE",
    "TWIST_1",
    "REGRASP_1",
    "TWIST_2",
    "REGRASP_2",
    "TWIST_3",
    "REGRASP_3",
    "FINAL_TWIST_AND_RELEASE_CHECK",
    "LIFT_CAP",
)

# The named TWIST/REGRASP stages above are evidence from the human video.  They
# are not a command script.  At runtime they collapse into one tactile-driven
# loop; the demonstrated three re-grasps are only a finite retry budget.
TWIST_REGRASP_DEMO_STAGES = (
    "TWIST_1",
    "REGRASP_1",
    "TWIST_2",
    "REGRASP_2",
    "TWIST_3",
    "REGRASP_3",
    "FINAL_TWIST_AND_RELEASE_CHECK",
)

RUNTIME_STAGE_ORDER = (
    "APPROACH_CAP",
    "CAP_GRASP_PRESS_ENGAGE",
    "TACTILE_TWIST_REGRASP_LOOP",
    "LIFT_CAP",
    "PLACE_CAP_ASIDE",
)

MISSING_STRAIGHTEN_EPISODE = "STRAIGHTEN_BOTTLE_AND_FIXTURE_PLACEMENT"

REQUIRED_SOURCE_BLOCKING_CONDITIONS = (
    "cap grasp SE3 relation not yet extracted",
    "cap and bottle task frames not yet localized in the live scene",
    "GF220 FORCE6D axes/units and left-right mapping are not execution-calibrated",
    "single-arm bottle anti-rotation strategy not yet validated",
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "capture",
        "fps",
        "recorded_frames",
        "effective_interval_frames",
        "effective_interval_s",
        "task_result",
        "observed_constraints",
        "stages",
        "missing_episode",
        "execution_status",
        "blocking_conditions",
    }
)

_OBSERVED_CONSTRAINT_FIELDS = frozenset(
    {
        "bottle_initially_upright",
        "bottle_straightening_demonstrated",
        "second_hand_stabilized_bottle",
        "bottle_visibly_moved_during_twist",
        "cap_visual_rotation_observable",
        "reason_cap_rotation_unobservable",
    }
)

_STAGE_FIELDS = frozenset({"name", "frames", "time_s", "authority"})
_MISSING_EPISODE_FIELDS = frozenset({"name", "reason"})
_TIME_TOLERANCE_S = 1e-6


def _object(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_exact_fields(
    value: Mapping[str, Any], *, expected: frozenset[str], name: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"{name} is missing fields: {missing}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {unknown}")


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _frame_interval(value: object, *, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element half-open interval")
    start, end = value
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{name} bounds must be integers")
    if start < 0 or end <= start:
        raise ValueError(f"{name} must be a non-empty half-open interval")
    return start, end


def _time_interval(value: object, *, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element half-open interval")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} bounds must be finite numbers")
        converted = float(item)
        if not math.isfinite(converted):
            raise ValueError(f"{name} bounds must be finite numbers")
        result.append(converted)
    if result[0] < 0.0 or result[1] <= result[0]:
        raise ValueError(f"{name} must be a non-empty half-open interval")
    return result[0], result[1]


def _require_time_matches_frames(
    *, frames: tuple[int, int], times_s: tuple[float, float], fps: float, name: str
) -> None:
    expected = (frames[0] / fps, frames[1] / fps)
    if any(
        not math.isclose(actual, target, rel_tol=0.0, abs_tol=_TIME_TOLERANCE_S)
        for actual, target in zip(times_s, expected, strict=True)
    ):
        raise ValueError(f"{name} time_s does not match frames/fps")


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_observed_constraints(value: object) -> dict[str, Any]:
    constraints = _object(value, name="observed_constraints")
    _require_exact_fields(
        constraints,
        expected=_OBSERVED_CONSTRAINT_FIELDS,
        name="observed_constraints",
    )
    bool_fields = _OBSERVED_CONSTRAINT_FIELDS - {"reason_cap_rotation_unobservable"}
    for field in bool_fields:
        if not isinstance(constraints[field], bool):
            raise ValueError(f"observed_constraints.{field} must be boolean")
    if constraints["bottle_initially_upright"] is not True:
        raise ValueError("this analysis contract requires an initially upright bottle")
    if constraints["bottle_straightening_demonstrated"] is not False:
        raise ValueError(
            "bottle_straightening_demonstrated must be false when the straighten "
            "episode is missing"
        )
    _nonempty_string(
        constraints["reason_cap_rotation_unobservable"],
        name="observed_constraints.reason_cap_rotation_unobservable",
    )
    return dict(constraints)


def _validate_missing_episode(value: object) -> dict[str, str]:
    missing = _object(value, name="missing_episode")
    _require_exact_fields(
        missing,
        expected=_MISSING_EPISODE_FIELDS,
        name="missing_episode",
    )
    if missing["name"] != MISSING_STRAIGHTEN_EPISODE:
        raise ValueError(
            f"missing_episode.name must be {MISSING_STRAIGHTEN_EPISODE!r}"
        )
    reason = _nonempty_string(missing["reason"], name="missing_episode.reason")
    return {"name": MISSING_STRAIGHTEN_EPISODE, "reason": reason}


def _validate_blocking_conditions(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("blocking_conditions must be an array")
    blockers = [
        _nonempty_string(item, name="blocking condition")
        for item in value
    ]
    if len(blockers) != len(set(blockers)):
        raise ValueError("blocking_conditions must not contain duplicates")
    missing = [
        item for item in REQUIRED_SOURCE_BLOCKING_CONDITIONS if item not in blockers
    ]
    if missing:
        raise ValueError(f"blocking_conditions is missing required conditions: {missing}")
    return blockers


def _validate_stages(
    value: object,
    *,
    fps: float,
    recorded_frames: int,
    effective_frames: tuple[int, int],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("stages must be an array")
    if len(value) != len(DEMONSTRATED_STAGE_ORDER):
        raise ValueError(
            "stages must exactly match the reviewed medicine-bottle sequence"
        )

    validated: list[dict[str, Any]] = []
    previous_end: int | None = None
    for index, raw in enumerate(value):
        stage = _object(raw, name=f"stages[{index}]")
        _require_exact_fields(
            stage,
            expected=_STAGE_FIELDS,
            name=f"stages[{index}]",
        )
        expected_name = DEMONSTRATED_STAGE_ORDER[index]
        if stage["name"] != expected_name:
            raise ValueError(
                "stage order must exactly match the reviewed medicine-bottle "
                f"sequence; index {index} must be {expected_name!r}"
            )
        expected_authority = EXPECTED_SOURCE_AUTHORITIES[expected_name]
        if stage["authority"] != expected_authority:
            raise ValueError(
                f"stage {expected_name} authority must be {expected_authority!r}"
            )
        frames = _frame_interval(
            stage["frames"], name=f"stage {expected_name} frames"
        )
        times_s = _time_interval(
            stage["time_s"], name=f"stage {expected_name} time_s"
        )
        _require_time_matches_frames(
            frames=frames,
            times_s=times_s,
            fps=fps,
            name=f"stage {expected_name}",
        )
        if previous_end is not None and frames[0] != previous_end:
            raise ValueError("stage frame intervals must be contiguous and ordered")
        if frames[1] > recorded_frames:
            raise ValueError(f"stage {expected_name} exceeds recorded_frames")
        previous_end = frames[1]
        validated.append(
            {
                "name": expected_name,
                "frames_half_open": list(frames),
                "time_s_half_open": list(times_s),
                "source_authority": expected_authority,
            }
        )

    aggregate = (validated[0]["frames_half_open"][0], validated[-1]["frames_half_open"][1])
    if aggregate != effective_frames:
        raise ValueError("stage frame intervals must exactly cover effective_interval_frames")
    return validated


def _target_contract(name: str) -> dict[str, Any]:
    common = {
        "demonstrated_path_replay": False,
        "requires_live_localization": True,
        "robot_pose_present": False,
    }
    if name == "APPROACH_CAP":
        return {
            **common,
            "source": "online_agent_from_live_scene",
            "goal": "collision_free_cap_standoff",
        }
    if name == "PLACE_CAP_ASIDE":
        return {
            **common,
            "source": "online_agent_from_live_scene",
            "goal": "supported_cap_placement_then_safe_retreat",
        }
    if name == "CAP_GRASP_PRESS_ENGAGE":
        return {
            **common,
            "frame": "live_bottle_cap_task_frame",
            "constraint": "press_along_negative_live_bottle_axis_at_cap_grasp_relation",
        }
    if name.startswith("TWIST_"):
        return {
            **common,
            "frame": "live_bottle_cap_task_frame",
            "constraint": "bounded_rotation_about_live_bottle_axis",
        }
    if name.startswith("REGRASP_"):
        return {
            **common,
            "frame": "live_bottle_cap_task_frame",
            "constraint": "agent_selects_wrist_reset_while_cap_contact_relation_is_maintained",
        }
    if name == "FINAL_TWIST_AND_RELEASE_CHECK":
        return {
            **common,
            "frame": "live_bottle_cap_task_frame",
            "constraint": "coaxial_press_twist_until_thread_release_predicate",
            "force_torque_predicate_usable": False,
        }
    if name == "LIFT_CAP":
        return {
            **common,
            "frame": "live_bottle_cap_task_frame",
            "constraint": "translate_along_positive_live_bottle_axis_after_thread_release",
        }
    raise AssertionError(f"unhandled stage {name}")


def _compile_stage(source: Mapping[str, Any]) -> dict[str, Any]:
    name = source["name"]
    if name in AGENTIC_DEMONSTRATED_STAGES:
        control_family = "agentic"
    elif name in OBJECT_RELATIVE_STAGES:
        control_family = "object_relative"
    else:
        raise AssertionError(f"stage {name} has no control-family assignment")
    return {
        **dict(source),
        "control_family": control_family,
        "target_contract": _target_contract(name),
        "human_demo_role": "stage_timing_and_intent_evidence_only",
        "hardware_execution_enabled": False,
    }


def _twist_regrasp_loop_contract() -> dict[str, Any]:
    """Describe how demo micro-stages become one runtime closed loop.

    Absolute force thresholds deliberately live in a versioned tactile profile,
    not in the demonstration contract.  This contract fixes the state machine,
    signal definition, and transition precedence so a runner cannot silently
    fall back to replaying three fixed wrist arcs.
    """

    return {
        "name": "TACTILE_TWIST_REGRASP_LOOP",
        "source_demo_stages": list(TWIST_REGRASP_DEMO_STAGES),
        "demonstrated_regrasp_count": 3,
        "maximum_regrasp_count": 3,
        "fixed_regrasp_count_required": False,
        "twist_extent_source": "bounded_task_geometry_and_wrist_travel",
        "tactile_controls_twist_extent": False,
        "motion_primitive": "bounded_twist_microstep_about_live_bottle_axis",
        "transition_primitive": {
            "name": "pre_press_axial_reaction_gate",
            "force_frame": "live_bottle_cap_task_frame",
            "bottle_axis": "a_hat",
            "pre_press_baseline": "b_i = median(dot(F_i, a_hat))",
            "press_reaction_sign": "p in {-1, +1} from the tactile profile",
            "per_finger_scalar": "f_axis_i = p * (dot(F_i, a_hat) - b_i)",
            "combined_scalar": "f_axis_sum = f_axis_left + f_axis_right",
            "confirmation": (
                "two_or_more_fresh_post_press_windows_with_bilateral_balanced_"
                "stable_axial_reaction_in_profile_band"
            ),
            "fresh_synchronized_force_windows_required": True,
            "threshold_source": "versioned_twist_or_lid_tactile_profile",
        },
        "twist_supervision_primitive": {
            "name": "two_finger_signed_tangential_drive",
            "force_frame": "live_bottle_cap_task_frame",
            "bottle_axis": "a_hat",
            "per_finger_direction": (
                "t_i = twist_sign * normalize(a_hat cross r_i)"
            ),
            "per_finger_scalar": "f_tan_i = dot(F_i, t_i)",
            "combined_scalar": "f_tan_drive = f_tan_left + f_tan_right",
            "cartesian_force_vector_sum_is_not_used": True,
            "reason": (
                "opposing fingers have opposite Cartesian tangent directions, "
                "so useful cap-driving forces can cancel in a raw vector sum"
            ),
            "fresh_synchronized_force_windows_required": True,
            "threshold_source": "versioned_twist_or_lid_tactile_profile",
        },
        "states": [
            "PRE_PRESS_BASELINE",
            "PRESS_INCREMENT",
            "PRESS_VERIFY",
            "TWIST_MICROSTEP",
            "TACTILE_EVALUATE",
            "WRIST_UNLOAD_RESET",
            "REGRASP_VERIFY",
            "SLIP_RECOVERY",
            "RELEASE_VERIFY",
            "COMPLETE",
            "HOLD",
        ],
        "transition_precedence": [
            "INVALID_OR_OVERLOAD",
            "SLIP_OR_FORCE_IMBALANCE",
            "THREAD_RELEASE_CANDIDATE",
            "PRESS_ENGAGEMENT_CONFIRMED",
            "REGRASP_REQUIRED",
            "TWIST_CONTINUE",
        ],
        "transitions": [
            {
                "event": "INVALID_OR_OVERLOAD",
                "condition": (
                    "stale_or_desynchronized_force OR lost_bilateral_contact "
                    "OR calibrated_overload"
                ),
                "next": "HOLD",
            },
            {
                "event": "SLIP_OR_FORCE_IMBALANCE",
                "condition": (
                    "one_finger_tangential_drive_collapses OR tangential_imbalance "
                    "exceeds_profile_band OR slip_predicate_persists"
                ),
                "next": "SLIP_RECOVERY",
            },
            {
                "event": "THREAD_RELEASE_CANDIDATE",
                "condition": (
                    "previously_engaged AND combined_tangential_drive_falls_below_"
                    "release_band_for_profile_window AND bilateral_contact_remains"
                ),
                "next": "RELEASE_VERIFY",
                "note": "do not perform another wrist reset before release verification",
            },
            {
                "event": "PRESS_ENGAGEMENT_CONFIRMED",
                "condition": (
                    "after_bounded_axial_press_both_fingers_show_stable_balanced_"
                    "axis_reaction_above_the_relative_pre_press_band_for_two_fresh_"
                    "windows"
                ),
                "next": "TWIST_MICROSTEP",
                "note": (
                    "axial tactile reaction gates the press-to-twist transition; "
                    "no probe twist is required"
                ),
            },
            {
                "event": "REGRASP_REQUIRED",
                "condition": (
                    "wrist_rotation_margin_is_exhausted AND axial_press_and_"
                    "tangential_contact_remain_stable"
                ),
                "next": "WRIST_UNLOAD_RESET",
                "note": (
                    "wrist geometry chooses the reset point; tactile drive only "
                    "verifies that reset can begin from stable cap contact"
                ),
            },
            {
                "event": "TWIST_CONTINUE",
                "condition": (
                    "planned_geometric_twist_extent_remains AND axial_press_is_"
                    "retained AND bilateral_tangential_contact_is_stable"
                ),
                "next": "TWIST_MICROSTEP",
            },
            {
                "event": "REGRASP_VERIFIED",
                "condition": (
                    "after_wrist_reset_bilateral_cap_contact_is_reestablished"
                ),
                "next": "PRE_PRESS_BASELINE",
                "note": (
                    "each regrasp captures a new pre-press baseline, then passes "
                    "the axial press gate again"
                ),
            },
            {
                "event": "THREAD_RELEASE_VERIFIED",
                "condition": (
                    "small_axial_test_has_low_resistance AND tangential_drive_"
                    "remains_in_release_band"
                ),
                "next": "COMPLETE",
            },
        ],
        "regrasp_sequence": [
            "open_only_to_unload_tangential_contact",
            "verify_axial_and_tangential_load_return_to_unload_band",
            "reset_wrist_opposite_twist_direction",
            "close_to_bilateral_contact_band",
            "capture_new_pre_press_baseline",
            "apply_bounded_axial_press_increment",
            "verify_stable_bilateral_axial_reaction_before_twist",
        ],
        "logging": [
            "raw_per_finger_force6d",
            "sensor_and_task_frame_transforms",
            "pre_press_axial_baseline",
            "per_finger_axial_reaction_delta",
            "combined_axial_press_reaction",
            "per_finger_signed_tangential_drive",
            "combined_tangential_drive",
            "force_imbalance",
            "decision_and_transition",
        ],
        "hardware_execution_enabled": False,
    }


def compile_medicine_bottle_stage_contract(
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate ``medicine_bottle_demo_analysis.v1`` and compile a closed gate.

    The returned contract is intentionally review-only.  In particular, the
    human video did not provide tactile force labels and the present GF220
    FORCE6D signal has not been execution-calibrated, so no force/torque term
    in this contract may be used as a runtime predicate.
    """

    source = _object(analysis, name="analysis")
    _require_exact_fields(source, expected=_TOP_LEVEL_FIELDS, name="analysis")
    if source["schema"] != MEDICINE_BOTTLE_ANALYSIS_SCHEMA:
        raise ValueError(
            "analysis schema must be " f"{MEDICINE_BOTTLE_ANALYSIS_SCHEMA!r}"
        )
    if source["execution_status"] != "review_only":
        raise ValueError("analysis execution_status must be 'review_only'")
    if source["task_result"] != "cap_removed_successfully":
        raise ValueError("analysis task_result must be 'cap_removed_successfully'")

    capture = _nonempty_string(source["capture"], name="capture")
    fps = _positive_float(source["fps"], name="fps")
    recorded_frames = _positive_int(source["recorded_frames"], name="recorded_frames")
    effective_frames = _frame_interval(
        source["effective_interval_frames"], name="effective_interval_frames"
    )
    effective_times = _time_interval(
        source["effective_interval_s"], name="effective_interval_s"
    )
    _require_time_matches_frames(
        frames=effective_frames,
        times_s=effective_times,
        fps=fps,
        name="effective_interval",
    )
    if effective_frames[1] > recorded_frames:
        raise ValueError("effective_interval_frames exceeds recorded_frames")

    constraints = _validate_observed_constraints(source["observed_constraints"])
    missing_episode = _validate_missing_episode(source["missing_episode"])
    source_blockers = _validate_blocking_conditions(source["blocking_conditions"])
    source_stages = _validate_stages(
        source["stages"],
        fps=fps,
        recorded_frames=recorded_frames,
        effective_frames=effective_frames,
    )
    stages = [_compile_stage(stage) for stage in source_stages]

    straighten_obligation = {
        "name": MISSING_STRAIGHTEN_EPISODE,
        "control_family": "agentic",
        "demonstration_status": "missing_from_human_demo",
        "reason": missing_episode["reason"],
        "required_when": "live_bottle_axis_is_not_upright_or_fixture_support_is_invalid",
        "target_source": "online_agent_from_live_rgbd_and_live_bottle_axis",
        "demonstrated_path_replay": False,
        "hardware_execution_enabled": False,
    }

    blockers = [
        *source_blockers,
        "human demonstration provides no tactile force labels",
        "FORCE6D is not calibrated for execution",
        "missing straighten-bottle episode has no demonstrated relative pose",
        "review-only stage contract cannot command hardware",
    ]
    return {
        "schema": MEDICINE_BOTTLE_STAGE_CONTRACT_SCHEMA,
        "status": "review_only",
        "execution_authorized": False,
        "hardware_execution_enabled": False,
        "task": "straighten_medicine_bottle_then_remove_press_and_turn_cap",
        "source": {
            "schema": MEDICINE_BOTTLE_ANALYSIS_SCHEMA,
            "capture": capture,
            "sha256": _canonical_digest(source),
            "fps": fps,
            "recorded_frames": recorded_frames,
            "effective_interval_frames": list(effective_frames),
            "effective_interval_s": list(effective_times),
            "task_result": source["task_result"],
        },
        "authority_boundary": {
            "agentic_demonstrated_stages": list(AGENTIC_DEMONSTRATED_STAGES),
            "agentic_missing_episodes": [MISSING_STRAIGHTEN_EPISODE],
            "object_relative_stages": list(OBJECT_RELATIVE_STAGES),
            "demonstrated_trajectory_replay": False,
            "object_relative_targets_require_live_cap_and_bottle_axis": True,
        },
        "missing_agentic_episodes": [straighten_obligation],
        "human_demo_evidence": {
            "bottle_straightening_demonstrated": False,
            "tactile_force_labels_provided": False,
            "force6d_labels_provided": False,
            "stage_frames_are_timing_and_intent_evidence_only": True,
            "observed_constraints": constraints,
        },
        "tactile_boundary": {
            "force6d_execution_calibrated": False,
            "force6d_axes_units_qualified": False,
            "force6d_left_right_mapping_qualified": False,
            "may_gate_execution": False,
            "may_command_hardware": False,
            "mode": "unqualified_shadow_evidence_only",
        },
        "stage_order": list(DEMONSTRATED_STAGE_ORDER),
        "runtime_stage_order": list(RUNTIME_STAGE_ORDER),
        "twist_regrasp_runtime_loop": _twist_regrasp_loop_contract(),
        "stages": stages,
        "execution_gate": {
            "state": "closed",
            "arm_commands_allowed": False,
            "gripper_commands_allowed": False,
            "reason": "review_only_input_and_unqualified_tactile_force_evidence",
        },
        "blocking_conditions": blockers,
    }


__all__ = [
    "AGENTIC_DEMONSTRATED_STAGES",
    "DEMONSTRATED_STAGE_ORDER",
    "EXPECTED_SOURCE_AUTHORITIES",
    "MEDICINE_BOTTLE_ANALYSIS_SCHEMA",
    "MEDICINE_BOTTLE_STAGE_CONTRACT_SCHEMA",
    "MISSING_STRAIGHTEN_EPISODE",
    "OBJECT_RELATIVE_STAGES",
    "RUNTIME_STAGE_ORDER",
    "TWIST_REGRASP_DEMO_STAGES",
    "compile_medicine_bottle_stage_contract",
]
