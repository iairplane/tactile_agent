from __future__ import annotations

import copy

import pytest

from demo2gripper.test_tube_preflight import (
    REFINED_REVIEW_STAGE_ORDER,
    REVIEW_STAGE_ORDER,
    TEST_TUBE_CONFIG_SCHEMA,
    TEST_TUBE_FRONT_RGB_REFERENCE_SCHEMA,
    TEST_TUBE_GEOMETRY_SCHEMAS,
    TEST_TUBE_REFINED_REVIEW_SCHEMA,
    TEST_TUBE_REVIEW_SCHEMA,
    TEST_TUBE_TACTILE_OBSERVABILITY_SCHEMA,
    TubeState,
    compile_test_tube_preflight,
    decide_test_tube_transition,
    validate_test_tube_geometry_artifact,
)


def _review() -> dict[str, object]:
    stages = []
    for index, name in enumerate(REVIEW_STAGE_ORDER):
        start = 210 + index * 25
        stages.append(
            {
                "name": name,
                "approx_frames": [start, start + 30],
                "approx_seconds": [start / 15.0, (start + 30) / 15.0],
                "policy": f"reviewed {name}",
                "exit_condition": "reviewed only",
            }
        )
    return {
        "schema": TEST_TUBE_REVIEW_SCHEMA,
        "status": "review_only",
        "source": "../input_manifest.json",
        "segmentation_method": "manual_review",
        "frame_boundaries": "approximate",
        "angle_target_deg": 45,
        "angle_reference": "relative",
        "stages": stages,
        "execution_blockers": ["calibration pending"],
        "hardware_commands_dispatched": False,
    }


def _refined_review() -> dict[str, object]:
    stages = []
    start = 210
    for name in REFINED_REVIEW_STAGE_ORDER:
        end = start + 9
        stages.append(
            {
                "name": name,
                "frames_inclusive": [start, end],
                "relative_time_s_inclusive": [start / 15.0, end / 15.0],
                "boundary_confidence": "high",
            }
        )
        start = end + 1
    return {
        "schema": TEST_TUBE_REFINED_REVIEW_SCHEMA,
        "status": "preexecution_analysis_only",
        "hardware_commands_dispatched": False,
        "recording": {"action_interval_frames": [210, start - 1]},
        "stages": stages,
        "observable_geometry": {
            "angle_observation": {
                "apparent_45_deg_frame": 235,
                "frame_uncertainty": 5,
                "physical_angle_available": False,
            }
        },
        "execution_blockers": ["metric geometry missing"],
    }


def _config(*, ready: bool = False) -> dict[str, object]:
    evidence_path = "reviewed.json" if ready else None
    return {
        "schema": TEST_TUBE_CONFIG_SCHEMA,
        "task_id": "tube-test",
        "frames": {
            "base": "fr3_link0",
            "tcp": "hande_tcp",
            "tube": "test_tube_axis",
            "socket": "rack_socket",
            "transform_convention": "T_parent_child",
        },
        "evidence": {
            name: {"path": evidence_path, "reviewed": ready}
            for name in (
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
            )
        },
        "tilt": {
            "target_semantics": "same_fixed_camera_demonstration_reference",
            "target_relative_deg": 45.0,
            "target_image_axis_deg": 135.5 if ready else None,
            "tolerance_deg": 5.0,
            "confirmation_windows": 2,
            "allowed_target_sources": [
                "front_rgb_calibrated",
            ],
            "tactile_angle_calibration_id": None,
            "front_rgb_angle_calibration_id": "front-angle-cal" if ready else None,
            "front_rgb_runtime_gate_qualified": ready,
        },
        "tactile": {
            "profile_id": "tube-profile.v1",
            "execution_qualified": ready,
            "predicate_modes": {
                "contact": ("runtime_qualified" if ready else "conditional_runtime_bootstrap"),
                "slip": "runtime_qualified" if ready else "shadow_only",
                "insertion_jam": "runtime_qualified" if ready else "shadow_only",
            },
            "bootstrap": {
                "fresh_unloaded_baseline_required": True,
                "positive_contact_required": True,
                "unloaded_window_count": 3,
                "positive_contact_window_count": 3,
                "session_scoped": True,
                "min_unique_samples": 5,
                "min_calibration_windows_per_stage": 3,
                "max_age_s": 0.20,
                "max_pair_skew_s": 0.08,
                "minimum_fz_change_sensor_units": 0.001,
                "minimum_threshold_floor_sensor_units": 0.001,
                "positive_margin_noise_multiplier": 8.0,
                "threshold_noise_multiplier": 6.0,
                "minimum_consistent_positive_fraction": 2.0 / 3.0,
                "engage_fraction_of_positive": 0.25,
                "retain_fraction_of_positive": 0.10,
                "contact_confirmation_windows": 2,
                "loss_confirmation_windows": 2,
            },
            "contact_policy": "any_finger_valid_contact",
            "contact_confirmation_windows": 2,
            "slip_confirmation_windows": 1,
            "pivot_open_step_m": 0.00025,
            "max_pivot_opening_delta_m": 0.002,
            "reclamp_step_m": 0.0005,
            "max_reclamp_steps": 4,
            "insertion_jam": {
                "min_progress_ratio": 0.25,
                "confirmation_windows": 2,
                "resistance_delta_threshold": 2.0 if ready else None,
                "resistance_units": "calibrated_sensor_units" if ready else None,
            },
        },
        "motion": {
            "approach_standoff_m": 0.06,
            "extract_clearance_m": 0.04,
            "pivot_max_microsteps": 8,
            "insertion_step_m": 0.0015,
            "insertion_target_depth_m": 0.03 if ready else None,
            "jam_retract_m": 0.003,
            "max_jam_recoveries": 2,
            "retreat_m": 0.05,
            "velocity_scale": 0.03,
            "acceleration_scale": 0.03,
        },
    }


def _tactile(*, eligible: bool) -> dict[str, object]:
    return {
        "schema": TEST_TUBE_TACTILE_OBSERVABILITY_SCHEMA,
        "status": "qualified" if eligible else "shadow_only_no_robot_contact_positive",
        "runtime_gate_eligible": eligible,
        "predicates": {
            "fresh_dual_tactile_samples": {
                "runtime_gate_eligible": True,
                "scope": "acquisition_integrity_only",
            },
            "contact_present": {"runtime_gate_eligible": eligible},
            "insertion_jam": {"runtime_gate_eligible": eligible},
        },
        "threshold_fit": {"status": "qualified" if eligible else "not_observable"},
        "hardware_commands_dispatched": False,
    }


def _front_reference(*, eligible: bool) -> dict[str, object]:
    return {
        "schema": TEST_TUBE_FRONT_RGB_REFERENCE_SCHEMA,
        "status": "qualified" if eligible else "reviewed_reference_only",
        "runtime_gate_eligible": eligible,
        "physical_3d_angle_claimed": False,
        "target": {"target_image_axis_deg": 135.5},
        "calibration_id": "front-angle-cal",
        "hardware_commands_dispatched": False,
    }


def _tracker_validation(*, eligible: bool) -> dict[str, object]:
    return {
        "schema": "demo2gripper.test_tube_rgb_tracker_validation.v1",
        "status": (
            "runtime_qualified"
            if eligible
            else "offline_human_demo_replay_pass_robot_held_unqualified"
        ),
        "runtime_gate_eligible": eligible,
        "hardware_commands_dispatched": False,
        "depth_used_for_tube_axis": False,
        "offline_replay": {
            "passed": True,
            "learned_target_axis_deg": 135.5,
            "consecutive_frames_required": 2,
            "tracked_fraction": 1.0,
            "reviewed_axis_mae_deg": 0.5,
            "reviewed_axis_max_error_deg": 1.2,
            "first_reclamp_frame": 230,
            "reviewed_target_frame": 235,
        },
        "qualification_boundary": {},
    }


def _geometry_artifact(name: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": TEST_TUBE_GEOMETRY_SCHEMAS[name],
        "task_id": "tube-test",
        "status": "reviewed_plan_only_evidence",
        "execution_authorized": False,
        "hardware_commands_dispatched": False,
        "transform_convention": "T_parent_child; metres; right-handed",
    }
    if name == "tube_grasp_relation":
        payload.update(
            {
                "T_object_gripper_tcp": None,
                "robot_candidate_constraints": {"closing_axis": "perpendicular"},
                "execution_start_requirement": "instantiate from a fresh metric observation",
            }
        )
    elif name == "live_tube_pose":
        payload.update(
            {
                "metric_pose_available": False,
                "T_base_tube": None,
                "execution_start_requirement": {
                    "fresh_observation_required": True,
                    "fail_closed_if_missing": True,
                },
            }
        )
    else:
        payload.update(
            {
                "metric_socket_pose_available": False,
                "T_base_socket": None,
                "insertion_target_depth_m": None,
                "execution_start_requirement": {"fail_closed_if_missing": True},
            }
        )
    return payload


def test_compiler_partitions_authority_and_never_authorizes_hardware() -> None:
    plan = compile_test_tube_preflight(
        stage_review=_review(), config=_config(), tactile_observability=_tactile(eligible=False)
    )

    assert plan["status"] == "blocked_preflight"
    assert plan["execution_authorized"] is False
    assert plan["hardware_execution_enabled"] is False
    assert plan["hardware_boundary"] == {
        "dispatch_supported": False,
        "robot_or_gripper_imports": False,
        "subprocess_commands_present": False,
        "output_is_offline_review_contract_only": True,
    }
    assert set(plan["authority_partition"]) == {
        "agentic",
        "relative_pose",
        "tactile_supervision",
        "perception_supervision",
    }
    assert plan["source_review"]["stages"][0]["demonstrated_path_replay"] is False
    assert plan["gates"]["pivot_target"]["condition"] == ("front_rgb_calibrated_angle_reached")
    pivot_transition = next(
        transition
        for transition in plan["transitions"]
        if transition["action"] == "reclamp_at_target"
    )
    assert "front_rgb_calibrated_angle_reached" in pivot_transition["event"]
    assert "tactile_relative_angle_reached" not in pivot_transition["event"]
    assert plan["first_run_contact_bootstrap"]["preflight_inputs_ready"] is False


def test_ready_review_evidence_opens_only_offline_contract_boundary() -> None:
    plan = compile_test_tube_preflight(
        stage_review=_review(),
        config=_config(ready=True),
        tactile_observability=_tactile(eligible=True),
        front_rgb_reference=_front_reference(eligible=True),
        front_rgb_tracker_validation=_tracker_validation(eligible=True),
    )

    assert plan["status"] == "offline_contract_ready_live_grounding_required"
    assert plan["preflight"]["offline_task_contract_ready"] is True
    assert plan["preflight"]["moveit_plan_only_ready"] is False
    assert plan["preflight"]["metric_execution_geometry_ready"] is False
    assert plan["preflight"]["runtime_transition_gates_ready"] is False
    assert plan["preflight"]["hardware_execution_ready"] is False
    assert plan["execution_authorized"] is False
    assert {
        "exact_T_tube_tcp_not_instantiated",
        "gripper_width_and_close_budget_not_grounded",
        "hande_tube_rack_collision_scene_not_validated",
        "motion_bounds_not_validated_against_live_geometry",
        "offline_runtime_orchestrator_missing",
        "live_test_tube_hardware_adapter_missing",
        "live_feedback_predicate_producers_missing",
        "support_after_release_verifier_missing",
        "robot_command_state_path_not_confirmed",
        "front_rgb_stream_health_not_confirmed",
    } <= set(plan["preflight"]["hardware_blocking_conditions"])


def test_conditional_contact_bootstrap_does_not_promote_shadow_slip_or_jam() -> None:
    config = _config(ready=True)
    config["tactile"]["execution_qualified"] = False
    config["tactile"]["predicate_modes"] = {
        "contact": "conditional_runtime_bootstrap",
        "slip": "shadow_only",
        "insertion_jam": "shadow_only",
    }
    plan = compile_test_tube_preflight(
        stage_review=_review(),
        config=config,
        tactile_observability=_tactile(eligible=False),
        front_rgb_reference=_front_reference(eligible=True),
        front_rgb_tracker_validation=_tracker_validation(eligible=False),
    )

    assert plan["preflight"]["offline_task_contract_ready"] is True
    assert plan["preflight"]["moveit_plan_only_ready"] is False
    assert plan["preflight"]["runtime_transition_gates_ready"] is False
    assert plan["first_run_contact_bootstrap"]["preflight_inputs_ready"] is True
    bootstrap = plan["first_run_contact_bootstrap"]
    assert bootstrap["contract_inputs_ready"] is True
    assert bootstrap["live_session_windows_ready"] is False
    assert "contact_gate_live_session_bootstrap_pending" in plan["preflight"][
        "hardware_blocking_conditions"
    ]
    assert bootstrap["implementation"] == {
        "module": "demo2gripper.test_tube_contact_bootstrap",
        "schema": "demo2gripper.test_tube_contact_bootstrap.v2",
        "fit_function": "bootstrap_test_tube_contact",
        "gate_class": "TestTubeContactGate",
        "input_type": "GF220ClassifierWindows",
    }
    assert [phase["phase"] for phase in bootstrap["side_grasp_phases"]] == [
        "before_bounded_close",
        "bounded_close",
        "after_bounded_close",
        "activate_session_gate",
    ]
    assert bootstrap["runtime_outcomes"]["CONTACT_ABSENT"] == (
        "runner_promotes_contact_lost_after_prior_contact"
    )
    side_grasp = next(
        contract for contract in plan["stage_contracts"] if contract["state"] == "SIDE_GRASP"
    )
    assert side_grasp["bootstrap_completion_requirement"] == (
        "session_contact_profile_active AND contact_present"
    )
    assert "grasp_established" in side_grasp["completion"]
    assert plan["gates"]["grasp_established"]["contact_present_alone_is_sufficient"] is False
    assert side_grasp["bootstrap_phases"] == bootstrap["side_grasp_phases"]
    assert plan["gates"]["slip"]["may_trigger_transition"] is False
    assert plan["gates"]["insertion_jam"]["may_trigger_transition"] is False
    pivot_recovery = next(
        transition
        for transition in plan["transitions"]
        if transition["action"] == "immediate_bounded_reclamp"
    )
    assert pivot_recovery["event"] == "contact_lost"
    jam_transitions = [
        transition for transition in plan["transitions"] if "insertion_jam" in transition["event"]
    ]
    assert jam_transitions
    assert all(transition["enabled"] is False for transition in jam_transitions)

    traces = {trace["name"]: trace for trace in plan["deterministic_offline_traces"]["traces"]}
    nominal = traces["nominal_full_sequence"]
    assert nominal["result"] == "pass"
    assert nominal["manual_stage_calls_required"] is False
    assert nominal["initial_state"] == "OBSERVE"
    assert nominal["expected_final_state"] == "COMPLETE"
    assert [step["next_state"] for step in nominal["steps"]][-1] == "COMPLETE"

    recovery = traces["pivot_contact_loss_reclamp_resume"]
    assert [step["next_state"] for step in recovery["steps"]] == [
        "RECLAMP_RECOVERY",
        "PIVOT_EVALUATE",
        "RECLAMP",
    ]
    assert recovery["steps"][0]["recovery"] is True

    jam_shadow = traces["shadow_jam_does_not_trigger_recovery"]
    assert jam_shadow["steps"][0]["shadow_observations"] == ["insertion_jam"]
    assert "insertion_jam" not in jam_shadow["steps"][0]["promoted_predicates"]
    assert jam_shadow["steps"][0]["next_state"] == "INSERT_MICROSTEP"


def test_contact_bootstrap_config_matches_implemented_window_contract() -> None:
    config = _config()
    config["tactile"]["bootstrap"]["unloaded_window_count"] = 2
    with pytest.raises(ValueError, match="at least min_calibration_windows_per_stage"):
        compile_test_tube_preflight(stage_review=_review(), config=config)

    config = _config()
    config["tactile"]["bootstrap"]["retain_fraction_of_positive"] = 0.25
    with pytest.raises(ValueError, match="retain_fraction_of_positive"):
        compile_test_tube_preflight(stage_review=_review(), config=config)


def test_side_grasp_requires_grasp_confirmation_before_extraction() -> None:
    contact_only = decide_test_tube_transition(TubeState.SIDE_GRASP, {"contact_present"})
    assert contact_only.next_state is TubeState.SIDE_GRASP
    assert contact_only.event == "CONTACT_PRESENT_GRASP_UNCONFIRMED"

    established = decide_test_tube_transition(
        TubeState.SIDE_GRASP,
        {
            "contact_present",
            "gripper_state_valid",
            "grasp_width_in_range",
            "visual_tube_between_fingers",
            "grasp_established",
        },
    )
    assert established.next_state is TubeState.EXTRACT
    assert established.event == "GRASP_ESTABLISHED"


def test_extraction_levels_rigid_hold_before_first_pivot_opening() -> None:
    extracted = decide_test_tube_transition(
        TubeState.EXTRACT,
        {"tube_clear_of_source", "contact_present"},
    )
    assert extracted.next_state is TubeState.LEVEL_HELD_TUBE
    assert extracted.action == "servo_held_tube_to_horizontal"

    moving = decide_test_tube_transition(
        TubeState.LEVEL_HELD_TUBE,
        {"contact_present", "tube_horizontal"},
    )
    assert moving.next_state is TubeState.LEVEL_HELD_TUBE
    assert moving.action == "continue_closed_loop_leveling"

    stable = decide_test_tube_transition(
        TubeState.LEVEL_HELD_TUBE,
        {"contact_present", "tube_horizontal", "level_target_reached"},
    )
    assert stable.next_state is TubeState.PIVOT_MICRO_LOOSEN
    assert stable.action == "one_micro_loosen"

    stationary_slip = decide_test_tube_transition(
        TubeState.LEVEL_HELD_TUBE,
        {"contact_present", "slip_detected"},
    )
    assert stationary_slip.next_state is TubeState.HOLD
    assert stationary_slip.event == "LEVELING_CONTACT_FAILURE"
    assert "stationary slip" in stationary_slip.reason


@pytest.mark.parametrize("name", tuple(TEST_TUBE_GEOMETRY_SCHEMAS))
def test_geometry_artifacts_are_plan_only_and_fail_closed(name: str) -> None:
    summary = validate_test_tube_geometry_artifact(
        name,
        _geometry_artifact(name),
        task_id="tube-test",
    )
    assert summary["plan_only_reviewed"] is True
    assert summary["metric_execution_geometry_ready"] is False


def test_geometry_artifact_rejects_an_execution_claim() -> None:
    artifact = _geometry_artifact("live_tube_pose")
    artifact["execution_authorized"] = True
    with pytest.raises(ValueError, match="execution_authorized=false"):
        validate_test_tube_geometry_artifact(
            "live_tube_pose",
            artifact,
            task_id="tube-test",
        )


def test_refined_review_is_normalized_without_claiming_a_3d_angle() -> None:
    plan = compile_test_tube_preflight(
        stage_review=_refined_review(),
        config=_config(),
        tactile_observability=_tactile(eligible=False),
    )
    review = plan["source_review"]
    assert review["source_schema"] == TEST_TUBE_REFINED_REVIEW_SCHEMA
    assert review["physical_3d_angle_available"] is False
    assert review["stages"][0]["frames_half_open"] == [210, 220]


def test_pivot_loss_or_slip_immediately_reclamps_before_angle_progress() -> None:
    decision = decide_test_tube_transition(
        TubeState.PIVOT_EVALUATE,
        {
            "contact_lost",
            "slip_detected",
            "front_rgb_calibrated_angle_reached",
        },
    )

    assert decision.next_state is TubeState.RECLAMP_RECOVERY
    assert decision.event == "CONTACT_LOST_OR_SLIP"
    assert decision.recovery is True


@pytest.mark.parametrize(
    "predicate",
    ["tactile_relative_angle_reached", "front_rgb_calibrated_angle_reached"],
)
def test_either_calibrated_angle_source_can_trigger_reclamp(predicate: str) -> None:
    decision = decide_test_tube_transition(TubeState.PIVOT_EVALUATE, {"contact_present", predicate})

    assert decision.next_state is TubeState.RECLAMP
    assert decision.event == "PIVOT_TARGET_REACHED"


def test_front_only_state_machine_ignores_tactile_angle_predicate() -> None:
    tactile_only = decide_test_tube_transition(
        TubeState.PIVOT_EVALUATE,
        {"contact_present", "tactile_relative_angle_reached", "pivot_budget_remaining"},
        allowed_angle_sources=("front_rgb_calibrated",),
    )
    assert tactile_only.next_state is TubeState.PIVOT_MICRO_LOOSEN

    front = decide_test_tube_transition(
        TubeState.PIVOT_EVALUATE,
        {"contact_present", "front_rgb_calibrated_angle_reached"},
        allowed_angle_sources=("front_rgb_calibrated",),
    )
    assert front.next_state is TubeState.RECLAMP


def test_pivot_continues_only_with_contact_and_budget() -> None:
    decision = decide_test_tube_transition(
        TubeState.PIVOT_EVALUATE, {"contact_present", "pivot_budget_remaining"}
    )
    assert decision.next_state is TubeState.PIVOT_MICRO_LOOSEN

    waiting = decide_test_tube_transition(TubeState.PIVOT_EVALUATE, {"pivot_budget_remaining"})
    assert waiting.next_state is TubeState.PIVOT_EVALUATE
    assert waiting.action == "collect_fresh_evidence"


def test_insertion_jam_retracts_and_realigns_with_bounded_recovery() -> None:
    jam = decide_test_tube_transition(
        TubeState.INSERT_EVALUATE,
        {"insertion_jam", "jam_recovery_budget_remaining"},
    )
    assert jam.next_state is TubeState.JAM_RECOVERY_RETRACT
    assert jam.action == "stop_push_and_retract"

    realign = decide_test_tube_transition(
        TubeState.JAM_RECOVERY_RETRACT, {"retract_complete", "scene_reobserved"}
    )
    assert realign.next_state is TubeState.ALIGN_SOCKET

    exhausted = decide_test_tube_transition(TubeState.INSERT_EVALUATE, {"insertion_jam"})
    assert exhausted.next_state is TubeState.HOLD


def test_single_force_spike_is_not_encoded_as_a_jam() -> None:
    plan = compile_test_tube_preflight(
        stage_review=_review(), config=_config(), tactile_observability=None
    )
    jam = plan["gates"]["insertion_jam"]
    assert jam["single_force_spike_is_not_a_jam"] is True
    assert "low_axial_progress" in jam["condition"]


def test_stage_order_and_target_mismatch_fail_closed() -> None:
    review = _review()
    review["stages"][0]["name"] = "INSERT"
    with pytest.raises(ValueError, match="stage 0 must be"):
        compile_test_tube_preflight(stage_review=review, config=_config())

    config = copy.deepcopy(_config())
    config["tilt"]["target_relative_deg"] = 40.0
    with pytest.raises(ValueError, match="disagree on the tilt target"):
        compile_test_tube_preflight(stage_review=_review(), config=config)
