from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from demo2gripper.test_tube_pregrasp import (
    CANDIDATE_SET_SCHEMA,
    assess_moveit_plan,
    build_moveit_plan_only_command,
    generate_pregrasp_candidates,
    rank_plan_assessments,
    validate_current_relation,
    verify_relation_provenance,
)


def relation() -> dict:
    start = np.array([100.0, 200.0])
    end = np.array([140.0, 80.0])
    fraction = 0.65
    grasp = start + fraction * (end - start)
    return {
        "schema": "demo2gripper.test_tube_current_image_relation.v1",
        "execution_authorized": False,
        "hardware_commands_dispatched": False,
        "provenance": {},
        "current_source_tube": {
            "axis_closed_or_lower_visible_to_open_uv_px": [start.tolist(), end.tolist()],
            "depth_observation": {
                "axis_camera_unit_closed_to_open": [0.0, -0.8, 0.6],
                "orthogonal_line_fit_rms_m": 0.0008,
            },
        },
        "transferred_side_grasp": {
            "demo_axial_fraction_closed_to_open": fraction,
            "demo_axial_fraction_uncertainty": 0.08,
            "current_grasp_centre_uv_px": grasp.tolist(),
            "depth_at_grasp": {"xyz_camera_m": [0.05, 0.02, 0.40]},
        },
        "bottom_line": {"source_axis_and_grasp_observable": True},
    }


def test_generates_side_grasp_roll_family_in_base_frame() -> None:
    T_base_camera = np.eye(4)
    T_base_camera[:3, 3] = [0.50, 0.10, -0.20]

    result = generate_pregrasp_candidates(
        relation(),
        T_base_camera,
        approach_distance_m=0.06,
        roll_step_deg=90.0,
        workspace_min_m=[0.0, -1.0, -1.0],
        workspace_max_m=[2.0, 2.0, 2.0],
    )

    assert result["schema"] == CANDIDATE_SET_SCHEMA
    assert result["execution_authorized"] is False
    assert len(result["candidates"]) == 8
    np.testing.assert_allclose(
        result["metric_geometry"]["grasp_position_base_m"], [0.55, 0.12, 0.20]
    )
    for candidate in result["candidates"]:
        grasp = np.asarray(candidate["T_base_tcp_grasp"])
        pregrasp = np.asarray(candidate["T_base_tcp_pregrasp"])
        axis = np.asarray(result["metric_geometry"]["tube_axis_base_closed_to_open"])
        assert np.linalg.det(grasp[:3, :3]) == pytest.approx(1.0)
        assert abs(float(grasp[:3, 0] @ axis)) < 1e-12
        assert abs(float(grasp[:3, 2] @ axis)) < 1e-12
        np.testing.assert_allclose(grasp[:3, 3] - pregrasp[:3, 3], 0.06 * grasp[:3, 2], atol=1e-12)
        assert candidate["plan_only"] is True
        assert candidate["may_execute"] is False


def test_rejects_inconsistent_axial_transfer() -> None:
    value = relation()
    value["transferred_side_grasp"]["current_grasp_centre_uv_px"][0] += 2.0

    with pytest.raises(ValueError, match="inconsistent"):
        validate_current_relation(value)


def test_verifies_all_declared_relation_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"reviewed pixels")
    value = relation()
    value["provenance"] = {
        "current_rgb": {
            "path": source.name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    }
    verified = verify_relation_provenance(value, repository_root=tmp_path)
    assert verified[0]["path"] == str(source)

    value["provenance"]["current_rgb"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_relation_provenance(value, repository_root=tmp_path)


def test_moveit_command_is_irreducibly_plan_only() -> None:
    candidates = generate_pregrasp_candidates(
        relation(),
        np.eye(4),
        roll_step_deg=90.0,
        workspace_min_m=[-1.0, -1.0, -1.0],
        workspace_max_m=[2.0, 2.0, 2.0],
    )["candidates"]
    T_link8_tcp = np.eye(4)
    T_link8_tcp[2, 3] = 0.19

    command = build_moveit_plan_only_command(
        candidates[0],
        planner_script="scripts/moveit_pose_goal.py",
        ros_wrapper="scripts/run_ros_python.sh",
        output_path="plan.json",
        T_link8_tcp=T_link8_tcp,
    )

    assert "--execute" not in command
    assert command[command.index("--target-frame") + 1] == "configured-ee"
    assert command[command.index("--output") + 1] == "plan.json"


def plan_for(candidate: dict, *, accepted: bool, total_sweep: float) -> dict:
    per_joint = total_sweep / 7.0
    return {
        "schema": "demo2gripper.moveit_pose_goal.v2",
        "mode": "plan_only",
        "input_target_frame": "configured-ee",
        "arm_only_collision_model": True,
        "execution_requested": False,
        "execution_dispatched": False,
        "input_target_position_base_m": candidate["pregrasp_position_base_m"],
        "input_target_orientation_xyzw": candidate["pregrasp_orientation_xyzw"],
        "moveit_planning_succeeded": True,
        "planning_succeeded": accepted,
        "planning_time_s": 0.2,
        "trajectory_audit": {
            "accepted": accepted,
            "total_joint_sweep_rad": total_sweep,
            "joint_metrics": {
                f"fr3_joint{index}": {
                    "sweep_rad": per_joint,
                    "minimum_hardware_limit_margin_rad": 0.5,
                }
                for index in range(1, 8)
            },
        },
    }


def test_ranks_audited_moveit_results_and_rejects_wrong_target() -> None:
    candidates = generate_pregrasp_candidates(
        relation(),
        np.eye(4),
        roll_step_deg=90.0,
        workspace_min_m=[-1.0, -1.0, -1.0],
        workspace_max_m=[2.0, 2.0, 2.0],
    )["candidates"]
    slow = assess_moveit_plan(
        candidates[0], plan_for(candidates[0], accepted=True, total_sweep=4.0)
    )
    fast = assess_moveit_plan(
        candidates[1], plan_for(candidates[1], accepted=True, total_sweep=2.0)
    )
    rejected = assess_moveit_plan(
        candidates[2], plan_for(candidates[2], accepted=False, total_sweep=1.0)
    )
    ranked = rank_plan_assessments([slow, rejected, fast])
    assert [item["candidate_id"] for item in ranked] == [
        candidates[1]["id"],
        candidates[0]["id"],
        candidates[2]["id"],
    ]

    mismatch = plan_for(candidates[0], accepted=True, total_sweep=1.0)
    mismatch["input_target_position_base_m"] = [0.0, 0.0, 0.0]
    assessment = assess_moveit_plan(candidates[0], mismatch)
    assert assessment["accepted"] is False
    assert any("does not match" in reason for reason in assessment["rejection_reasons"])
