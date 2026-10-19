import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from demo2gripper.models import GraspCandidates, GraspIntent
from demo2gripper.selection import (
    SelectionConfig,
    rank_grasp_candidates,
    select_grasp_candidate,
)


def _pose(rotation: np.ndarray, translation: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def _intent() -> GraspIntent:
    return GraspIntent(
        T_object_gripper_tcp=np.eye(4),
        width_m=0.05,
        contact_points_object_m=np.array(
            [[-0.025, 0.0, 0.0], [0.025, 0.0, 0.0]]
        ),
        stable_interval=(2, 8),
        support_frames=6,
    )


def test_ranking_uses_geometry_and_sign_independent_closing_axis() -> None:
    identity = np.eye(3)
    flipped_closing = np.diag([-1.0, -1.0, 1.0])
    reversed_approach = np.diag([1.0, -1.0, -1.0])
    candidates = GraspCandidates(
        poses=np.stack(
            [
                _pose(identity, (0.0, 0.0, 0.0)),
                _pose(flipped_closing, (0.0, 0.0, 0.0)),
                _pose(reversed_approach, (0.0, 0.0, 0.0)),
                _pose(identity, (0.0, 0.0, 0.0)),
                _pose(identity, (0.04, 0.0, 0.0)),
            ]
        ),
        scores=np.array([0.2, 0.8, 1.0, 10.0, 1.2]),
        widths_m=np.array([0.05, 0.05, 0.05, 0.12, 0.05]),
    )

    ranked = rank_grasp_candidates(
        _intent(), candidates, SelectionConfig(max_gripper_width_m=0.08)
    )

    assert [candidate.index for candidate in ranked] == [1, 0, 2, 4]
    assert ranked[0].closing_axis_error_rad == 0.0
    assert ranked[1].closing_axis_error_rad == 0.0
    assert np.isclose(ranked[2].approach_axis_error_rad, np.pi)
    assert all(candidate.index != 3 for candidate in ranked)
    assert select_grasp_candidate(_intent(), candidates).index == 1


def test_width_limits_are_hard_constraints() -> None:
    candidates = GraspCandidates(
        poses=np.stack([np.eye(4), np.eye(4)]),
        scores=np.array([100.0, 0.0]),
        widths_m=np.array([0.09, 0.04]),
    )

    selected = select_grasp_candidate(
        _intent(), candidates, SelectionConfig(max_gripper_width_m=0.08)
    )

    assert selected.index == 1


def test_pose_error_limits_are_hard_constraints() -> None:
    reversed_approach = np.diag([1.0, -1.0, -1.0])
    candidates = GraspCandidates(
        poses=np.stack(
            [
                _pose(np.eye(3), (0.04, 0.0, 0.0)),
                _pose(reversed_approach, (0.0, 0.0, 0.0)),
                _pose(np.eye(3), (0.01, 0.0, 0.0)),
            ]
        ),
        scores=np.array([100.0, 100.0, 0.0]),
        widths_m=np.full(3, 0.05),
    )
    config = SelectionConfig(
        max_center_distance_m=0.02,
        max_approach_axis_error_rad=np.deg2rad(30.0),
    )

    selected = select_grasp_candidate(_intent(), candidates, config)

    assert selected.index == 2

    impossible = SelectionConfig(
        max_center_distance_m=0.001,
        max_approach_axis_error_rad=np.deg2rad(30.0),
    )
    with pytest.raises(ValueError, match="width and pose-error limits"):
        select_grasp_candidate(_intent(), candidates, impossible)


def test_axially_equivalent_candidates_have_equal_geometry_cost() -> None:
    intent_pose = np.eye(4)
    intent_pose[:3, 3] = [0.03, 0.0, 0.0]
    intent = GraspIntent(
        T_object_gripper_tcp=intent_pose,
        width_m=0.05,
        contact_points_object_m=np.array(
            [[0.005, 0.0, 0.0], [0.055, 0.0, 0.0]]
        ),
        stable_interval=(0, 5),
        support_frames=5,
        continuous_symmetry_axis=2,
    )
    gauge = np.eye(4)
    gauge[:3, :3] = Rotation.from_euler("z", 1.37).as_matrix()
    yaw_equivalent_pose = gauge @ intent_pose
    candidates = GraspCandidates(
        poses=np.stack([intent_pose, yaw_equivalent_pose]),
        scores=np.ones(2),
        widths_m=np.full(2, 0.05),
    )

    ranked = rank_grasp_candidates(intent, candidates)

    assert [candidate.index for candidate in ranked] == [0, 1]
    assert np.isclose(ranked[0].center_error_m, ranked[1].center_error_m)
    assert np.isclose(
        ranked[0].closing_axis_error_rad, ranked[1].closing_axis_error_rad
    )
    assert np.isclose(
        ranked[0].approach_axis_error_rad, ranked[1].approach_axis_error_rad
    )
    assert np.isclose(ranked[0].ranking_cost, ranked[1].ranking_cost)
    np.testing.assert_allclose(
        ranked[1].T_object_gripper_tcp, yaw_equivalent_pose
    )
