from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from demo2gripper.task_skill import AnchorFrame, StageName, StageTargetPolicy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compile_agentic_skill import (  # noqa: E402
    AnchoredPath,
    SimplifiedPath,
    StageInterval,
    _first_reliable_window,
    _load_selected_grasp,
    _stage_contract,
    _task_skill_stage,
    _validate_compiled_paths,
)


def selected_grasp_payload() -> dict:
    return {
        "frame_convention": "T_object_gripper_tcp; +X closing, +Z approach",
        "object_frame_id": "die_mesh_sha256:abc123",
        "gripper_name": "robotiq_hande",
        "T_object_gripper_tcp": np.eye(4).tolist(),
        "pregrasp_width_m": 0.05,
        "approach_distance_m": 0.08,
        "center_error_m": 0.004,
        "closing_axis_error_rad": 0.1,
        "approach_axis_error_rad": 0.2,
        "intent_provenance": {
            "schema": "demo2gripper.robot_adapted_grasp_intent.v1",
            "intent_role": "robot_adapted_hande",
        },
        "selection": {
            "hard_gate_thresholds": {
                "min_gripper_width_m": 0.0,
                "max_gripper_width_m": 0.05,
                "max_center_distance_m": 0.01,
                "max_closing_axis_error_rad": 0.25,
                "max_approach_axis_error_rad": 0.25,
            },
            "hard_gates_passed": True,
        },
    }


def test_selected_grasp_contract_accepts_rigid_hande_object_pose(tmp_path: Path) -> None:
    path = tmp_path / "selected_grasp.json"
    path.write_text(json.dumps(selected_grasp_payload()), encoding="utf-8")

    payload, transform, width, approach = _load_selected_grasp(path)

    assert payload["object_frame_id"] == "die_mesh_sha256:abc123"
    np.testing.assert_allclose(transform, np.eye(4))
    assert width == pytest.approx(0.05)
    assert approach == pytest.approx(0.08)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("frame_convention", "T_gripper_object", "frame_convention"),
        ("object_frame_id", "", "object_frame_id"),
        ("gripper_name", "franka_panda", "robotiq_hande"),
        ("pregrasp_width_m", 0.0501, "0.05"),
    ],
)
def test_selected_grasp_contract_rejects_wrong_identity_or_hande_width(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    payload = selected_grasp_payload()
    payload[key] = value
    path = tmp_path / "selected_grasp.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_selected_grasp(path)


def test_selected_grasp_contract_rejects_nonrigid_pose(tmp_path: Path) -> None:
    payload = selected_grasp_payload()
    payload["T_object_gripper_tcp"][0][0] = 2.0
    path = tmp_path / "selected_grasp.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="orthonormal"):
        _load_selected_grasp(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.pop("intent_provenance"),
            "robot-adapted intent_provenance",
        ),
        (
            lambda payload: payload["intent_provenance"].update(
                {"schema": "demo2gripper.raw_hand_object_intent.v1"}
            ),
            "intent_provenance.schema",
        ),
        (
            lambda payload: payload["intent_provenance"].update(
                {"intent_role": "raw_human_intent"}
            ),
            "intent_provenance.intent_role",
        ),
        (
            lambda payload: payload["selection"].update(
                {"hard_gates_passed": False}
            ),
            "hard_gates_passed",
        ),
        (
            lambda payload: payload["selection"]["hard_gate_thresholds"].pop(
                "max_center_distance_m"
            ),
            "max_center_distance_m must be finite",
        ),
        (
            lambda payload: payload.update({"center_error_m": 0.02}),
            "violate recorded hard gates",
        ),
    ],
)
def test_selected_grasp_contract_requires_robot_adaptation_and_hard_gate_evidence(
    tmp_path: Path, mutation: Callable[[dict], object], message: str
) -> None:
    payload = selected_grasp_payload()
    mutation(payload)
    path = tmp_path / "selected_grasp.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_selected_grasp(path)


def anchored(
    stage: str,
    positions: list[list[float]],
    anchor_position: list[float],
    anchor: AnchorFrame,
) -> AnchoredPath:
    values = np.asarray(positions, dtype=float)
    count = len(values)
    path = SimplifiedPath(
        frames=np.arange(count),
        timestamps_s=np.arange(count, dtype=float),
        positions_base_m=values,
        path_length_m=float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum()),
        source_valid_frame_count=count,
        rdp_vertex_count=count,
    )
    return AnchoredPath(
        stage=stage,
        path=path,
        anchor=anchor,
        demonstrated_anchor_base_m=np.asarray(anchor_position, dtype=float),
        source_interval=(0, count),
    )


def continuous_paths() -> dict[str, AnchoredPath]:
    return {
        "APPROACH": anchored(
            "APPROACH",
            [[-0.10, 0.0, 0.12], [0.0, 0.0, 0.03]],
            [0.0, 0.0, 0.03],
            AnchorFrame.GRASP_TRANSLATION,
        ),
        "PRESHAPE": anchored(
            "PRESHAPE",
            [[0.0, 0.0, 0.03], [0.0, 0.0, 0.025]],
            [0.0, 0.0, 0.025],
            AnchorFrame.GRASP_TRANSLATION,
        ),
        "LIFT": anchored(
            "LIFT",
            [[0.0, 0.0, 0.025], [0.0, 0.0, 0.12]],
            [0.0, 0.0, 0.025],
            AnchorFrame.GRASP_TRANSLATION,
        ),
        "PLACE": anchored(
            "PLACE",
            [[0.0, 0.0, 0.12], [0.02, 0.0, 0.025]],
            [0.0, 0.0, 0.12],
            AnchorFrame.RUNTIME_STAGE_START,
        ),
        "RETREAT": anchored(
            "RETREAT",
            [[0.02, 0.0, 0.025], [0.02, 0.0, 0.07], [0.10, 0.0, 0.10]],
            [0.02, 0.0, 0.025],
            AnchorFrame.RELEASE_POSE,
        ),
    }


def test_compiled_paths_are_continuous_and_retreat_above_release_pose() -> None:
    result = _validate_compiled_paths(
        continuous_paths(),
        support_height_base_m=0.0,
        min_support_clearance_m=0.003,
        max_continuity_gap_m=1e-8,
        min_retreat_lift_m=0.02,
        max_prelift_horizontal_m=0.005,
    )

    assert result["maximum_continuity_gap_m"] == pytest.approx(0.0)
    assert result["retreat_min_vertical_offset_m"] == pytest.approx(0.0)
    assert result["retreat_peak_vertical_offset_m"] == pytest.approx(0.075)


def test_compiler_rejects_retreat_that_moves_below_release_pose() -> None:
    paths = continuous_paths()
    paths["RETREAT"] = anchored(
        "RETREAT",
        [[0.02, 0.0, 0.025], [0.03, 0.0, 0.015], [0.10, 0.0, 0.08]],
        [0.02, 0.0, 0.025],
        AnchorFrame.RELEASE_POSE,
    )

    with pytest.raises(ValueError, match="drops below its release-pose anchor"):
        _validate_compiled_paths(
            paths,
            support_height_base_m=0.0,
            min_support_clearance_m=0.003,
            max_continuity_gap_m=1e-8,
            min_retreat_lift_m=0.02,
            max_prelift_horizontal_m=0.005,
        )


@pytest.mark.parametrize(
    "retreat_positions",
    [
        [[0.02, 0.0, 0.025], [0.22, 0.0, 0.025], [0.22, 0.0, 0.055]],
        [[0.02, 0.0, 0.025], [0.22, 0.0, 0.055]],
    ],
)
def test_compiler_rejects_horizontal_retreat_before_lift_clearance(
    retreat_positions: list[list[float]],
) -> None:
    paths = continuous_paths()
    paths["RETREAT"] = anchored(
        "RETREAT",
        retreat_positions,
        [0.02, 0.0, 0.025],
        AnchorFrame.RELEASE_POSE,
    )

    with pytest.raises(ValueError, match="moves horizontally before reaching lift clearance"):
        _validate_compiled_paths(
            paths,
            support_height_base_m=0.0,
            min_support_clearance_m=0.003,
            max_continuity_gap_m=1e-8,
            min_retreat_lift_m=0.02,
            max_prelift_horizontal_m=0.005,
        )


@pytest.mark.parametrize(
    "limit_name",
    [
        "support_height_base_m",
        "min_support_clearance_m",
        "max_continuity_gap_m",
        "min_retreat_lift_m",
        "max_prelift_horizontal_m",
    ],
)
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_compiler_rejects_nonfinite_safety_limits(
    limit_name: str, invalid: float
) -> None:
    kwargs = {
        "support_height_base_m": 0.0,
        "min_support_clearance_m": 0.003,
        "max_continuity_gap_m": 1e-8,
        "min_retreat_lift_m": 0.02,
        "max_prelift_horizontal_m": 0.005,
    }
    kwargs[limit_name] = invalid

    with pytest.raises(ValueError, match=f"{limit_name} must be finite"):
        _validate_compiled_paths(continuous_paths(), **kwargs)


@pytest.mark.parametrize(
    ("stage_name", "anchor", "grounding"),
    [
        (
            "LIFT",
            AnchorFrame.RUNTIME_STAGE_START,
            "agent_generated_from_measured_post_grasp_state",
        ),
        (
            "RETREAT",
            AnchorFrame.RELEASE_POSE,
            "agent_generated_from_measured_release_state",
        ),
    ],
)
def test_report_stage_contract_uses_the_canonical_grounding_anchor(
    stage_name: str, anchor: AnchorFrame, grounding: str
) -> None:
    contract = _stage_contract(
        StageInterval(stage_name, 0, 2),
        has_path=True,
        grounding_anchor=anchor,
    )

    assert contract["grounding_anchor"] == anchor.value
    assert contract["grounding"] == grounding
    assert contract["demonstration_path_role"] == "diagnostic_only"
    assert not contract["demonstration_path_consumed_for_execution"]


@pytest.mark.parametrize(
    ("name", "anchor", "policy"),
    [
        (StageName.APPROACH, AnchorFrame.GRASP_TRANSLATION, StageTargetPolicy.ONLINE_AGENT_PLAN),
        (StageName.PRESHAPE, AnchorFrame.GRASP_TRANSLATION, StageTargetPolicy.ONLINE_AGENT_PLAN),
        (StageName.GRASP, AnchorFrame.GRASP_TRANSLATION, StageTargetPolicy.OBJECT_RELATIVE_GRASP),
        (StageName.LIFT, AnchorFrame.RUNTIME_STAGE_START, StageTargetPolicy.ONLINE_AGENT_PLAN),
        (StageName.TRANSPORT, AnchorFrame.RUNTIME_STAGE_START, StageTargetPolicy.ONLINE_AGENT_PLAN),
        (StageName.PLACE, AnchorFrame.PLACEMENT, StageTargetPolicy.ONLINE_AGENT_PLAN),
        (StageName.RELEASE, AnchorFrame.RELEASE_POSE, StageTargetPolicy.TASK_INTENT),
        (StageName.RETREAT, AnchorFrame.RELEASE_POSE, StageTargetPolicy.ONLINE_AGENT_PLAN),
    ],
)
def test_v3_stage_contract_never_references_human_waypoints(
    name: StageName, anchor: AnchorFrame, policy: StageTargetPolicy
) -> None:
    timestamps = np.arange(10, dtype=float) / 30.0
    diagnostic_path = anchored(
        name.value,
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.01]],
        [0.0, 0.0, 0.0],
        AnchorFrame.BASE,
    )

    stage = _task_skill_stage(
        StageInterval(name.value, 0, 2), timestamps, diagnostic_path
    )

    assert stage.anchor is anchor
    assert stage.target_policy is policy
    assert stage.waypoint_artifact is None


def test_full_approach_can_start_before_the_near_contact_fit() -> None:
    reliable = np.zeros(30, dtype=bool)
    reliable[5:9] = True
    reliable[10:20] = True

    assert _first_reliable_window(reliable, start=3, window=10, fraction=0.5) == 5
