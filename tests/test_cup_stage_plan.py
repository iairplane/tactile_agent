from __future__ import annotations

import copy

import numpy as np
import pytest

from demo2gripper.cup_stage_plan import (
    CUP_STAGE_ORDER,
    CUP_STAGE_PLAN_SCHEMA,
    compile_top_cup_stage_plan,
)


def _semantic_stages() -> dict[str, object]:
    names = (
        "APPROACH",
        "PRESHAPE",
        "GRASP",
        "LIFT",
        "TRANSPORT",
        "PLACE",
        "RELEASE",
        "RETREAT",
    )
    stages = []
    start = 10
    for index, name in enumerate(names):
        stages.append(
            {
                "name": name,
                "start_frame": start,
                "end_frame": start + 4,
                "confidence": 0.8 + 0.01 * index,
                "evidence": f"reviewed {name}",
            }
        )
        start += 5
    return {
        "schema": "demo2gripper.image_only_semantic_stages.v1",
        "stages": stages,
    }


def _relation() -> dict[str, object]:
    semantic = _semantic_stages()
    intervals = {
        item["name"]: [item["start_frame"], item["end_frame"] + 1]
        for item in semantic["stages"]
    }
    return {
        "schema": "demo2gripper.cup_rim_relation.v1",
        "status": "review_only",
        "execution_authorized": False,
        "relation_role": "human_demo_geometry_only",
        "T_camera_rim": np.eye(4).tolist(),
        "T_rim_human_pinch_tcp": np.eye(4).tolist(),
        "demonstrated_pinch_width_m": 0.03,
        "stage_intervals_half_open": intervals,
        "quality": {"review_only": True, "execution_authorized": False},
    }


def _tactile_policy(*, ready: bool = False) -> dict[str, object]:
    return {
        "schema": "demo2gripper.tactile_supervision_config.v1",
        "profile_id": "nested_top_cup.v1",
        "status": "ready" if ready else "review_only",
        "execution_qualified": ready,
        "sensors": {
            "left": {"required": True, "contact_role": "inner"},
            "right": {"required": True, "contact_role": "outer"},
        },
        "predicates": {
            "rim_capture_verified": {
                "all": ["inner_rim_match", "outer_rim_match"]
            },
            "cup_remains_coupled": {"all": ["stable_hold"]},
            "release_confirmed": {"all": ["gripper_open_verified"]},
        },
    }


def test_compiler_has_fixed_order_and_preserves_authority_boundary() -> None:
    plan = compile_top_cup_stage_plan(
        cup_rim_relation=_relation(),
        semantic_stages=_semantic_stages(),
        tactile_policy=_tactile_policy(),
    )

    assert plan["schema"] == CUP_STAGE_PLAN_SCHEMA
    assert plan["stage_order"] == list(CUP_STAGE_ORDER)
    assert [stage["name"] for stage in plan["stages"]] == list(CUP_STAGE_ORDER)
    assert plan["status"] == "review_only"
    assert plan["execution_authorized"] is False
    assert plan["hardware_execution_enabled"] is False
    assert all(stage["hardware_execution_enabled"] is False for stage in plan["stages"])

    relative = [
        stage["name"]
        for stage in plan["stages"]
        if stage["authority"] == "demonstration_relative_pose"
    ]
    assert relative == ["RIM_TOUCH_ALIGN"]
    moveit = [
        stage["name"]
        for stage in plan["stages"]
        if stage["authority"] == "online_agent_moveit"
    ]
    assert moveit == ["APPROACH", "CLEAR_LIFT", "TRANSPORT", "PLACE", "RETREAT"]
    assert all(
        stage.get("target_contract") is None
        or stage["target_contract"].get("demonstrated_path_replay") is False
        for stage in plan["stages"]
    )


def test_shadow_tactile_policy_cannot_become_a_runtime_gate() -> None:
    plan = compile_top_cup_stage_plan(
        cup_rim_relation=_relation(),
        semantic_stages=_semantic_stages(),
        tactile_policy=_tactile_policy(),
    )

    pinch = next(stage for stage in plan["stages"] if stage["name"] == "GENTLE_PINCH")
    assert pinch["planner_or_action"] == "cup_tactile_pre_lift_gate_state_machine"
    assert pinch["completion"] == [
        "rim_capture_verified",
        "inner_rim_match",
        "outer_rim_match",
        "tactile_lift_permission",
    ]
    assert pinch["target_contract"]["decision_map"] == {
        "NO_CONTACT": "PROBE_DOWN",
        "MULTI_RIM_OR_TOO_DEEP": "UNLOAD_THEN_RAISE_AND_REPROBE",
        "INNER_OUTER_RIM_MATCH": "ALLOW_LIFT",
        "ROLE_MISMATCH": "UNLOAD_THEN_REALIGN_AND_REPROBE",
        "UNKNOWN": "STOP",
    }
    assert pinch["target_contract"]["contact_role_by_finger"] == {
        "left": "inner",
        "right": "outer",
    }
    assert "may reverse" in pinch["target_contract"]["contact_role_assignment_scope"]
    assert pinch["target_contract"]["image_symmetry_required"] is False
    assert pinch["target_contract"]["normal_load_relation"] == (
        "optional_calibrated_inner_outer_ratio_independent_of_images"
    )
    assert pinch["tactile_gate_mode"] == "shadow"
    assert plan["tactile_supervision"]["may_command_hardware"] is False
    assert "tactile_supervision_shadow_only" in plan["blocking_conditions"]


def test_pre_lift_gate_does_not_use_front_camera_to_count_cups() -> None:
    plan = compile_top_cup_stage_plan(
        cup_rim_relation=_relation(),
        semantic_stages=_semantic_stages(),
        tactile_policy=_tactile_policy(),
    )

    ground = next(stage for stage in plan["stages"] if stage["name"] == "GROUND_TOP_RIM")
    separate = next(stage for stage in plan["stages"] if stage["name"] == "SEPARATE")
    verify = next(stage for stage in plan["stages"] if stage["name"] == "VERIFY_ONE")
    assert ground["planner_or_action"] == "fit_live_top_rim_center_and_axis"
    assert "second_rim_resolved" not in ground["completion"]
    assert "external_rgbd_tracking" not in separate["artifact_inputs"]
    assert verify["authority"] == "tactile_hold_verification"
    assert "external_rgbd_tracking" not in verify["artifact_inputs"]
    assert plan["policy_boundary"]["front_rgbd_cup_counting_used"] is False
    serialized = str(plan).lower()
    assert "count_removed_cups" not in serialized
    assert "exactly_one_cup_verified" not in serialized


def test_qualified_tactile_input_still_cannot_authorize_review_only_plan() -> None:
    plan = compile_top_cup_stage_plan(
        cup_rim_relation=_relation(),
        semantic_stages=_semantic_stages(),
        tactile_policy=_tactile_policy(ready=True),
    )

    assert plan["tactile_supervision"]["mode"] == "qualified_profile_review_only"
    assert "tactile_supervision_shadow_only" not in plan["blocking_conditions"]
    assert plan["status"] == "review_only"
    assert plan["execution_authorized"] is False
    assert plan["hardware_execution_enabled"] is False


def test_compilation_is_deterministic_and_converts_inclusive_frames() -> None:
    inputs = (_relation(), _semantic_stages(), _tactile_policy())
    first = compile_top_cup_stage_plan(
        cup_rim_relation=inputs[0], semantic_stages=inputs[1], tactile_policy=inputs[2]
    )
    second = compile_top_cup_stage_plan(
        cup_rim_relation=copy.deepcopy(inputs[0]),
        semantic_stages=copy.deepcopy(inputs[1]),
        tactile_policy=copy.deepcopy(inputs[2]),
    )

    assert first == second
    approach = next(stage for stage in first["stages"] if stage["name"] == "APPROACH")
    assert approach["demonstration_evidence"]["frame_intervals_half_open"] == [[10, 15]]


def test_relation_and_semantic_intervals_must_agree() -> None:
    relation = _relation()
    relation["stage_intervals_half_open"]["GRASP"] = [21, 26]

    with pytest.raises(ValueError, match="disagree for GRASP"):
        compile_top_cup_stage_plan(
            cup_rim_relation=relation,
            semantic_stages=_semantic_stages(),
            tactile_policy=_tactile_policy(),
        )


def test_missing_tactile_predicate_fails_closed() -> None:
    policy = _tactile_policy()
    del policy["predicates"]["rim_capture_verified"]

    with pytest.raises(ValueError, match="rim_capture_verified"):
        compile_top_cup_stage_plan(
            cup_rim_relation=_relation(),
            semantic_stages=_semantic_stages(),
            tactile_policy=policy,
        )


def test_tactile_policy_requires_one_inner_and_one_outer_role() -> None:
    policy = _tactile_policy()
    policy["sensors"]["right"]["contact_role"] = "inner"

    with pytest.raises(ValueError, match="exactly one inner-rim"):
        compile_top_cup_stage_plan(
            cup_rim_relation=_relation(),
            semantic_stages=_semantic_stages(),
            tactile_policy=policy,
        )


def test_bilateral_contact_alone_cannot_define_rim_capture() -> None:
    policy = _tactile_policy()
    policy["predicates"]["rim_capture_verified"] = {
        "all": ["bilateral_rim_contact"]
    }

    with pytest.raises(ValueError, match="independent inner_rim_match"):
        compile_top_cup_stage_plan(
            cup_rim_relation=_relation(),
            semantic_stages=_semantic_stages(),
            tactile_policy=policy,
        )


def test_ready_tactile_label_requires_explicit_qualification() -> None:
    policy = _tactile_policy(ready=True)
    policy["execution_qualified"] = False

    with pytest.raises(ValueError, match="execution_qualified=true"):
        compile_top_cup_stage_plan(
            cup_rim_relation=_relation(),
            semantic_stages=_semantic_stages(),
            tactile_policy=policy,
        )
