import numpy as np
import pytest

from demo2gripper.runtime_grasp import (
    ApproachSource,
    make_agent_planned_pick_phases,
    make_base_aligned_grasp_candidate,
    make_grounded_pick_phases,
    make_object_relative_se3_grasp_candidate,
)
from demo2gripper.task_skill import ObjectGraspRelation


def test_unstable_approach_uses_vertical_fallback_and_demo_relative_translation() -> None:
    relation = ObjectGraspRelation.translation_only(
        np.array([-0.002, 0.001, -0.011]), jaw_opening_m=0.05
    )

    candidate = make_base_aligned_grasp_candidate(
        live_object_anchor_base_m=np.array([0.658, 0.033, 0.047]),
        grasp_relation=relation,
        closing_axis_base=np.array([1.0, 0.2, -0.04]),
        closing_axis_stable=True,
        demonstrated_approach_axis_base=np.array([0.8, 0.0, -0.6]),
        demonstrated_approach_stable=False,
        approach_distance_m=0.08,
    )

    np.testing.assert_allclose(
        candidate.T_base_grasp_tcp[:3, 3], [0.656, 0.034, 0.036]
    )
    np.testing.assert_allclose(candidate.approach_axis_base, [0.0, 0.0, -1.0])
    np.testing.assert_allclose(
        candidate.T_base_pregrasp_tcp[:3, 3], [0.656, 0.034, 0.116]
    )
    np.testing.assert_allclose(candidate.closing_axis_base[2], 0.0, atol=1e-12)
    assert candidate.approach_source is ApproachSource.BASE_VERTICAL_FALLBACK
    assert candidate.jaw_opening_m == pytest.approx(0.05)


def test_stable_demonstrated_approach_sets_pose_and_local_pregrasp_direction() -> None:
    relation = ObjectGraspRelation.translation_only(np.array([0.01, -0.02, 0.03]))
    approach = np.array([0.0, -1.0, -1.0])

    candidate = make_base_aligned_grasp_candidate(
        live_object_anchor_base_m=np.array([0.4, 0.1, 0.2]),
        grasp_relation=relation,
        closing_axis_base=np.array([1.0, 0.1, 0.0]),
        closing_axis_stable=True,
        demonstrated_approach_axis_base=approach,
        demonstrated_approach_stable=True,
        approach_distance_m=0.1,
    )

    expected_approach = approach / np.linalg.norm(approach)
    np.testing.assert_allclose(candidate.approach_axis_base, expected_approach)
    np.testing.assert_allclose(
        candidate.T_base_grasp_tcp[:3, 3]
        - candidate.T_base_pregrasp_tcp[:3, 3],
        expected_approach * 0.1,
    )
    np.testing.assert_allclose(
        candidate.T_base_grasp_tcp[:3, :3].T
        @ candidate.T_base_grasp_tcp[:3, :3],
        np.eye(3),
        atol=1e-12,
    )
    assert candidate.approach_source is ApproachSource.DEMONSTRATED_STABLE


def test_stable_but_tilted_approach_uses_vertical_safety_fallback() -> None:
    candidate = make_base_aligned_grasp_candidate(
        live_object_anchor_base_m=np.array([0.65, 0.08, 0.048]),
        grasp_relation=ObjectGraspRelation.translation_only(np.zeros(3)),
        closing_axis_base=np.array([0.92, 0.39, 0.10]),
        closing_axis_stable=True,
        demonstrated_approach_axis_base=np.array([-0.07, 0.41, -0.91]),
        demonstrated_approach_stable=True,
        approach_distance_m=0.08,
        min_downward_approach_alignment=0.95,
    )

    np.testing.assert_allclose(candidate.approach_axis_base, [0.0, 0.0, -1.0])
    np.testing.assert_allclose(
        candidate.T_base_pregrasp_tcp[:3, 3]
        - candidate.T_base_grasp_tcp[:3, 3],
        [0.0, 0.0, 0.08],
    )
    assert candidate.approach_source is ApproachSource.BASE_VERTICAL_FALLBACK


def test_pick_phases_anchor_demo_lift_at_live_relative_grasp() -> None:
    candidate = make_base_aligned_grasp_candidate(
        live_object_anchor_base_m=np.array([0.6, 0.03, 0.05]),
        grasp_relation=ObjectGraspRelation.translation_only(
            np.array([-0.002, 0.001, -0.01])
        ),
        closing_axis_base=np.array([1.0, 0.0, 0.0]),
        closing_axis_stable=True,
        demonstrated_approach_axis_base=None,
        demonstrated_approach_stable=False,
        approach_distance_m=0.08,
    )
    offsets = np.array(
        [[0.0, 0.0, 0.0], [0.005, 0.01, 0.04], [0.01, 0.02, 0.08]]
    )

    phases = make_grounded_pick_phases(
        candidate, demonstrated_lift_offsets_base_m=offsets
    )

    np.testing.assert_allclose(
        phases.approach_poses_base[0], candidate.T_base_pregrasp_tcp
    )
    np.testing.assert_allclose(
        phases.approach_poses_base[-1], candidate.T_base_grasp_tcp
    )
    np.testing.assert_allclose(
        phases.lift_positions_base_m,
        candidate.T_base_grasp_tcp[:3, 3] + offsets,
    )


def test_candidate_rejects_unstable_closing_axis_and_parallel_axes() -> None:
    kwargs = {
        "live_object_anchor_base_m": np.zeros(3),
        "grasp_relation": ObjectGraspRelation.translation_only(np.zeros(3)),
        "closing_axis_base": np.array([1.0, 0.0, 0.0]),
        "closing_axis_stable": False,
        "demonstrated_approach_axis_base": None,
        "demonstrated_approach_stable": False,
        "approach_distance_m": 0.08,
    }
    with pytest.raises(ValueError, match="stable closing axis"):
        make_base_aligned_grasp_candidate(**kwargs)

    kwargs.update(
        closing_axis_base=np.array([0.0, 0.0, -1.0]),
        closing_axis_stable=True,
    )
    with pytest.raises(ValueError, match="parallel"):
        make_base_aligned_grasp_candidate(**kwargs)


def test_lift_must_start_at_grasp_anchor() -> None:
    candidate = make_base_aligned_grasp_candidate(
        live_object_anchor_base_m=np.zeros(3),
        grasp_relation=ObjectGraspRelation.translation_only(np.zeros(3)),
        closing_axis_base=np.array([1.0, 0.0, 0.0]),
        closing_axis_stable=True,
        demonstrated_approach_axis_base=None,
        demonstrated_approach_stable=False,
        approach_distance_m=0.08,
    )
    with pytest.raises(ValueError, match="must start at the grasp anchor"):
        make_grounded_pick_phases(
            candidate,
            demonstrated_lift_offsets_base_m=np.array(
                [[0.0, 0.0, 0.01], [0.0, 0.0, 0.05]]
            ),
        )


def test_planar_side_pinch_uses_relation_axis_and_agent_motion_goals() -> None:
    relation = ObjectGraspRelation.base_aligned_planar(
        np.array([-0.002, 0.001, -0.011]),
        np.array([0.92, 0.39, 0.0]),
        jaw_opening_m=0.05,
    )

    candidate = make_base_aligned_grasp_candidate(
        live_object_anchor_base_m=np.array([0.658, 0.033, 0.047]),
        grasp_relation=relation,
        approach_distance_m=0.08,
    )
    phases = make_agent_planned_pick_phases(candidate)

    np.testing.assert_allclose(
        candidate.T_base_grasp_tcp[:3, 3], [0.656, 0.034, 0.036]
    )
    np.testing.assert_allclose(candidate.closing_axis_base, relation.closing_axis)
    np.testing.assert_allclose(candidate.approach_axis_base, [0.0, 0.0, -1.0])
    assert candidate.approach_source is ApproachSource.AGENT_SELECTED
    np.testing.assert_allclose(
        phases.lift_positions_base_m,
        [[0.656, 0.034, 0.036], [0.656, 0.034, 0.076]],
    )


def test_agent_lift_goal_must_move_upward_in_base_frame() -> None:
    candidate = make_base_aligned_grasp_candidate(
        live_object_anchor_base_m=np.zeros(3),
        grasp_relation=ObjectGraspRelation.base_aligned_planar(
            np.zeros(3), np.array([1.0, 0.0, 0.0])
        ),
        agent_approach_axis_base=np.array([0.0, 0.0, -1.0]),
        approach_distance_m=0.08,
    )

    with pytest.raises(ValueError, match="positive base-Z"):
        make_agent_planned_pick_phases(
            candidate, agent_lift_delta_base_m=np.array([0.02, 0.0, 0.0])
        )


def test_planar_relation_keeps_jaw_axis_when_agent_approach_needs_projection() -> None:
    candidate = make_base_aligned_grasp_candidate(
        live_object_anchor_base_m=np.zeros(3),
        grasp_relation=ObjectGraspRelation.base_aligned_planar(
            np.zeros(3), np.array([1.0, 0.0, 0.0])
        ),
        agent_approach_axis_base=np.array([0.2, 0.0, -1.0]),
        approach_distance_m=0.08,
    )

    np.testing.assert_allclose(candidate.closing_axis_base, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(candidate.approach_axis_base, [0.0, 0.0, -1.0])


def test_se3_candidate_preserves_selected_endpoint_and_retreats_along_local_minus_z() -> None:
    T_base_object = np.eye(4)
    T_base_object[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    T_base_object[:3, 3] = [0.6, 0.03, 0.05]
    R_object_tcp = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    relation = ObjectGraspRelation.se3(
        np.array([0.01, -0.02, 0.03]),
        R_object_tcp,
        approach_distance_m=0.08,
        object_frame_id="die_v1",
        gripper_name="robotiq_hande",
        jaw_opening_m=0.05,
    )

    candidate = make_object_relative_se3_grasp_candidate(
        T_base_object=T_base_object,
        grasp_relation=relation,
    )

    T_object_tcp = np.eye(4)
    T_object_tcp[:3, :3] = R_object_tcp
    T_object_tcp[:3, 3] = relation.offset_from_object_center_m
    expected_grasp = T_base_object @ T_object_tcp
    np.testing.assert_allclose(candidate.T_base_grasp_tcp, expected_grasp)
    np.testing.assert_allclose(
        candidate.T_base_pregrasp_tcp[:3, 3],
        expected_grasp[:3, 3] - expected_grasp[:3, 2] * 0.08,
    )
    np.testing.assert_allclose(
        candidate.T_base_pregrasp_tcp[:3, :3], expected_grasp[:3, :3]
    )
    assert candidate.approach_source is ApproachSource.SELECTED_OBJECT_RELATIVE_SE3
