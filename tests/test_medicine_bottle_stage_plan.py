from __future__ import annotations

import copy

import pytest

from demo2gripper.medicine_bottle_stage_plan import (
    AGENTIC_DEMONSTRATED_STAGES,
    DEMONSTRATED_STAGE_ORDER,
    EXPECTED_SOURCE_AUTHORITIES,
    MEDICINE_BOTTLE_ANALYSIS_SCHEMA,
    MEDICINE_BOTTLE_STAGE_CONTRACT_SCHEMA,
    MISSING_STRAIGHTEN_EPISODE,
    OBJECT_RELATIVE_STAGES,
    RUNTIME_STAGE_ORDER,
    TWIST_REGRASP_DEMO_STAGES,
    compile_medicine_bottle_stage_contract,
)


def _analysis() -> dict[str, object]:
    bounds = (
        (1119, 1142),
        (1142, 1152),
        (1152, 1183),
        (1183, 1189),
        (1189, 1222),
        (1222, 1228),
        (1228, 1236),
        (1236, 1240),
        (1240, 1248),
        (1248, 1253),
        (1253, 1265),
    )
    stages = [
        {
            "name": name,
            "frames": [start, end],
            "time_s": [start / 15.0, end / 15.0],
            "authority": EXPECTED_SOURCE_AUTHORITIES[name],
        }
        for name, (start, end) in zip(DEMONSTRATED_STAGE_ORDER, bounds, strict=True)
    ]
    return {
        "schema": MEDICINE_BOTTLE_ANALYSIS_SCHEMA,
        "capture": "runs/live/medicine_bottle_demo_20261023_01/raw_capture",
        "fps": 15.0,
        "recorded_frames": 1800,
        "effective_interval_frames": [1119, 1265],
        "effective_interval_s": [1119 / 15.0, 1265 / 15.0],
        "task_result": "cap_removed_successfully",
        "observed_constraints": {
            "bottle_initially_upright": True,
            "bottle_straightening_demonstrated": False,
            "second_hand_stabilized_bottle": False,
            "bottle_visibly_moved_during_twist": False,
            "cap_visual_rotation_observable": False,
            "reason_cap_rotation_unobservable": (
                "black cap lacks an asymmetric visual marker"
            ),
        },
        "stages": stages,
        "missing_episode": {
            "name": MISSING_STRAIGHTEN_EPISODE,
            "reason": "bottle was already upright throughout this demonstration",
        },
        "execution_status": "review_only",
        "blocking_conditions": [
            "cap grasp SE3 relation not yet extracted",
            "cap and bottle task frames not yet localized in the live scene",
            (
                "GF220 FORCE6D axes/units and left-right mapping are not "
                "execution-calibrated"
            ),
            "single-arm bottle anti-rotation strategy not yet validated",
        ],
    }


def test_contract_partitions_agentic_and_object_relative_authority() -> None:
    contract = compile_medicine_bottle_stage_contract(_analysis())

    assert contract["schema"] == MEDICINE_BOTTLE_STAGE_CONTRACT_SCHEMA
    assert contract["status"] == "review_only"
    assert contract["execution_authorized"] is False
    assert contract["hardware_execution_enabled"] is False
    assert contract["stage_order"] == list(DEMONSTRATED_STAGE_ORDER)
    assert contract["authority_boundary"] == {
        "agentic_demonstrated_stages": list(AGENTIC_DEMONSTRATED_STAGES),
        "agentic_missing_episodes": [MISSING_STRAIGHTEN_EPISODE],
        "object_relative_stages": list(OBJECT_RELATIVE_STAGES),
        "demonstrated_trajectory_replay": False,
        "object_relative_targets_require_live_cap_and_bottle_axis": True,
    }

    stages = {stage["name"]: stage for stage in contract["stages"]}
    assert stages["APPROACH_CAP"]["control_family"] == "agentic"
    assert stages["PLACE_CAP_ASIDE"]["control_family"] == "agentic"
    assert stages["CAP_GRASP_PRESS_ENGAGE"]["control_family"] == "object_relative"
    assert stages["TWIST_2"]["target_contract"]["constraint"] == (
        "bounded_rotation_about_live_bottle_axis"
    )
    assert stages["LIFT_CAP"]["target_contract"]["constraint"] == (
        "translate_along_positive_live_bottle_axis_after_thread_release"
    )
    assert all(stage["hardware_execution_enabled"] is False for stage in stages.values())


def test_runtime_twist_regrasp_is_one_tactile_driven_loop() -> None:
    contract = compile_medicine_bottle_stage_contract(_analysis())

    assert contract["runtime_stage_order"] == list(RUNTIME_STAGE_ORDER)
    loop = contract["twist_regrasp_runtime_loop"]
    assert loop["source_demo_stages"] == list(TWIST_REGRASP_DEMO_STAGES)
    assert loop["demonstrated_regrasp_count"] == 3
    assert loop["maximum_regrasp_count"] == 3
    assert loop["fixed_regrasp_count_required"] is False
    assert loop["tactile_controls_twist_extent"] is False
    assert loop["transition_primitive"]["name"] == "pre_press_axial_reaction_gate"
    assert loop["twist_supervision_primitive"]["name"] == (
        "two_finger_signed_tangential_drive"
    )
    assert loop["twist_supervision_primitive"][
        "cartesian_force_vector_sum_is_not_used"
    ]

    transitions = {item["event"]: item for item in loop["transitions"]}
    assert transitions["PRESS_ENGAGEMENT_CONFIRMED"]["next"] == "TWIST_MICROSTEP"
    assert transitions["TWIST_CONTINUE"]["next"] == "TWIST_MICROSTEP"
    assert transitions["REGRASP_REQUIRED"]["next"] == "WRIST_UNLOAD_RESET"
    assert transitions["REGRASP_VERIFIED"]["next"] == "PRE_PRESS_BASELINE"
    assert transitions["THREAD_RELEASE_CANDIDATE"]["next"] == "RELEASE_VERIFY"
    assert transitions["SLIP_OR_FORCE_IMBALANCE"]["next"] == "SLIP_RECOVERY"
    assert transitions["INVALID_OR_OVERLOAD"]["next"] == "HOLD"


def test_missing_straighten_episode_is_agentic_and_cannot_be_fabricated() -> None:
    contract = compile_medicine_bottle_stage_contract(_analysis())

    assert contract["missing_agentic_episodes"] == [
        {
            "name": MISSING_STRAIGHTEN_EPISODE,
            "control_family": "agentic",
            "demonstration_status": "missing_from_human_demo",
            "reason": "bottle was already upright throughout this demonstration",
            "required_when": (
                "live_bottle_axis_is_not_upright_or_fixture_support_is_invalid"
            ),
            "target_source": "online_agent_from_live_rgbd_and_live_bottle_axis",
            "demonstrated_path_replay": False,
            "hardware_execution_enabled": False,
        }
    ]


def test_human_demo_and_force6d_boundaries_fail_closed() -> None:
    contract = compile_medicine_bottle_stage_contract(_analysis())

    assert contract["human_demo_evidence"]["tactile_force_labels_provided"] is False
    assert contract["human_demo_evidence"]["force6d_labels_provided"] is False
    assert contract["tactile_boundary"] == {
        "force6d_execution_calibrated": False,
        "force6d_axes_units_qualified": False,
        "force6d_left_right_mapping_qualified": False,
        "may_gate_execution": False,
        "may_command_hardware": False,
        "mode": "unqualified_shadow_evidence_only",
    }
    assert contract["execution_gate"]["state"] == "closed"
    assert contract["execution_gate"]["arm_commands_allowed"] is False
    assert contract["execution_gate"]["gripper_commands_allowed"] is False
    assert "FORCE6D is not calibrated for execution" in contract["blocking_conditions"]


def test_stage_order_is_exact() -> None:
    analysis = _analysis()
    analysis["stages"][2], analysis["stages"][3] = (
        analysis["stages"][3],
        analysis["stages"][2],
    )

    with pytest.raises(ValueError, match="stage order must exactly match"):
        compile_medicine_bottle_stage_contract(analysis)


def test_stage_intervals_must_be_contiguous_and_cover_effective_interval() -> None:
    analysis = _analysis()
    analysis["stages"][1]["frames"] = [1143, 1152]
    analysis["stages"][1]["time_s"] = [1143 / 15.0, 1152 / 15.0]

    with pytest.raises(ValueError, match="contiguous and ordered"):
        compile_medicine_bottle_stage_contract(analysis)

    analysis = _analysis()
    analysis["effective_interval_frames"] = [1118, 1265]
    analysis["effective_interval_s"] = [1118 / 15.0, 1265 / 15.0]
    with pytest.raises(ValueError, match="exactly cover effective_interval_frames"):
        compile_medicine_bottle_stage_contract(analysis)


def test_stage_authority_is_exact() -> None:
    analysis = _analysis()
    analysis["stages"][1]["authority"] = "online_agent_plan"

    with pytest.raises(ValueError, match="CAP_GRASP_PRESS_ENGAGE authority"):
        compile_medicine_bottle_stage_contract(analysis)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update(execution_status="ready"),
            "execution_status must be 'review_only'",
        ),
        (
            lambda value: value["missing_episode"].update(name="NONE"),
            "missing_episode.name",
        ),
        (
            lambda value: value["observed_constraints"].update(
                bottle_straightening_demonstrated=True
            ),
            "bottle_straightening_demonstrated must be false",
        ),
        (
            lambda value: value["blocking_conditions"].remove(
                "GF220 FORCE6D axes/units and left-right mapping are not "
                "execution-calibrated"
            ),
            "missing required conditions",
        ),
    ],
)
def test_review_only_missing_episode_and_force_boundary_are_required(
    mutation, message: str
) -> None:
    analysis = _analysis()
    mutation(analysis)

    with pytest.raises(ValueError, match=message):
        compile_medicine_bottle_stage_contract(analysis)


def test_time_bounds_must_match_frames_and_fps() -> None:
    analysis = _analysis()
    analysis["stages"][0]["time_s"][0] += 0.01

    with pytest.raises(ValueError, match="does not match frames/fps"):
        compile_medicine_bottle_stage_contract(analysis)


def test_compilation_is_deterministic() -> None:
    analysis = _analysis()
    first = compile_medicine_bottle_stage_contract(analysis)
    second = compile_medicine_bottle_stage_contract(copy.deepcopy(analysis))

    assert first == second

