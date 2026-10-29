import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from demo2gripper.grounding import (
    GroundedPositionPath,
    ground_position_offsets_base,
    ground_trajectory_by_gravity,
    ground_trajectory_by_se3,
    ground_trajectory_by_translation,
    ground_trajectory_to_pose_anchor,
    ground_trajectory_to_position_anchor,
)


def pose(rotation_deg: float, position_m: list[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("z", rotation_deg, degrees=True).as_matrix()
    transform[:3, 3] = position_m
    return transform


def demo_trajectory() -> np.ndarray:
    return np.stack(
        (
            pose(10.0, [0.1, 0.0, 0.2]),
            pose(20.0, [0.2, 0.0, 0.3]),
            pose(30.0, [0.3, 0.1, 0.4]),
        )
    )


def test_position_offsets_have_a_distinct_orientation_unresolved_result() -> None:
    offsets = np.array(
        [[-0.10, 0.0, 0.05], [-0.05, 0.0, 0.03], [0.0, 0.0, 0.0]]
    )
    grounded = ground_position_offsets_base(
        offsets,
        live_anchor_position_m=np.array([0.6, -0.1, 0.2]),
    )

    assert isinstance(grounded, GroundedPositionPath)
    np.testing.assert_allclose(
        grounded.positions_base_m,
        [[0.5, -0.1, 0.25], [0.55, -0.1, 0.23], [0.6, -0.1, 0.2]],
    )
    assert grounded.representation == "position_offsets_base_m"
    assert grounded.position_relocalized
    assert not grounded.orientation_resolved
    assert not hasattr(grounded, "poses")
    with pytest.raises(ValueError, match="unresolved orientation"):
        grounded.require_full_pose()


def test_position_offset_grounding_rejects_pose_arrays() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        ground_position_offsets_base(
            demo_trajectory(),
            live_anchor_position_m=np.zeros(3),
        )


def test_translation_only_moves_positions_without_claiming_rotation() -> None:
    trajectory = demo_trajectory()
    grounded = ground_trajectory_by_translation(
        trajectory,
        demo_anchor_position_m=np.array([0.2, 0.0, 0.3]),
        live_anchor_position_m=np.array([0.5, -0.1, 0.4]),
    )

    np.testing.assert_allclose(
        grounded.poses[:, :3, 3],
        trajectory[:, :3, 3] + [0.3, -0.1, 0.1],
    )
    np.testing.assert_array_equal(grounded.poses[:, :3, :3], trajectory[:, :3, :3])
    assert grounded.position_relocalized
    assert not grounded.rotation_relocalized
    assert grounded.method == "translation_only"
    with pytest.raises(ValueError, match="not fully pose-relocated"):
        grounded.require_full_pose()


def test_se3_relocation_transforms_position_and_rotation() -> None:
    trajectory = demo_trajectory()
    demo_anchor = pose(0.0, [0.2, 0.0, 0.3])
    live_anchor = pose(90.0, [0.5, -0.1, 0.4])

    grounded = ground_trajectory_by_se3(
        trajectory,
        T_demo_anchor=demo_anchor,
        T_live_anchor=live_anchor,
    )

    expected_delta = live_anchor @ np.linalg.inv(demo_anchor)
    np.testing.assert_allclose(grounded.poses, expected_delta[None, ...] @ trajectory, atol=1e-12)
    assert grounded.rotation_relocalized
    np.testing.assert_allclose(grounded.require_full_pose(), grounded.poses)


@pytest.mark.parametrize(
    "invalid_anchor",
    [
        np.diag([1.0, 1.0, -1.0, 1.0]),
        np.array(
            [
                [1.0, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
)
def test_se3_relocation_rejects_non_rigid_anchor(invalid_anchor: np.ndarray) -> None:
    with pytest.raises(ValueError, match="rotation"):
        ground_trajectory_by_se3(
            demo_trajectory(),
            T_demo_anchor=np.eye(4),
            T_live_anchor=invalid_anchor,
        )


def test_gravity_grounding_rotates_path_geometry_but_not_tool_rotation() -> None:
    trajectory = np.stack(
        (
            pose(15.0, [0.0, 0.0, 0.0]),
            pose(15.0, [0.0, 0.0, 0.1]),
            pose(15.0, [0.0, 0.0, 0.2]),
        )
    )
    grounded = ground_trajectory_by_gravity(
        trajectory,
        demo_anchor_position_m=np.zeros(3),
        live_anchor_position_m=np.array([0.4, 0.1, 0.2]),
        demo_gravity=np.array([0.0, 0.0, -1.0]),
        live_gravity=np.array([0.0, -1.0, 0.0]),
    )

    np.testing.assert_allclose(
        grounded.poses[:, :3, 3],
        [[0.4, 0.1, 0.2], [0.4, 0.2, 0.2], [0.4, 0.3, 0.2]],
        atol=1e-12,
    )
    np.testing.assert_array_equal(grounded.poses[:, :3, :3], trajectory[:, :3, :3])
    assert not grounded.rotation_relocalized


def test_gravity_grounding_rejects_ambiguous_or_missing_direction() -> None:
    kwargs = {
        "poses": demo_trajectory(),
        "demo_anchor_position_m": np.zeros(3),
        "live_anchor_position_m": np.zeros(3),
        "demo_gravity": np.array([0.0, 0.0, -1.0]),
    }
    with pytest.raises(ValueError, match="non-zero"):
        ground_trajectory_by_gravity(**kwargs, live_gravity=np.zeros(3))
    with pytest.raises(ValueError, match="do not define a unique"):
        ground_trajectory_by_gravity(**kwargs, live_gravity=np.array([0.0, 0.0, 1.0]))


def test_position_anchor_places_selected_release_waypoint_exactly() -> None:
    trajectory = demo_trajectory()
    live_release = np.array([0.8, -0.2, 0.1])
    grounded = ground_trajectory_to_position_anchor(
        trajectory,
        live_anchor_position_m=live_release,
        waypoint_index=1,
        anchor_kind="release",
    )

    np.testing.assert_allclose(grounded.poses[1, :3, 3], live_release)
    np.testing.assert_array_equal(grounded.poses[:, :3, :3], trajectory[:, :3, :3])
    assert grounded.anchor_kind == "release"
    assert not grounded.rotation_relocalized


def test_pose_anchor_places_selected_placement_waypoint_exactly() -> None:
    trajectory = demo_trajectory()
    live_placement = pose(-45.0, [0.7, 0.2, 0.15])
    grounded = ground_trajectory_to_pose_anchor(
        trajectory,
        T_live_anchor=live_placement,
        anchor_kind="placement",
    )

    np.testing.assert_allclose(grounded.poses[-1], live_placement, atol=1e-12)
    assert grounded.anchor_kind == "placement"
    assert grounded.rotation_relocalized


def test_grounding_rejects_invalid_pose_trajectory_and_waypoint_index() -> None:
    invalid_trajectory = demo_trajectory()
    invalid_trajectory[1, 3, 3] = 0.0
    with pytest.raises(ValueError, match=r"poses\[1\]"):
        ground_trajectory_by_translation(
            invalid_trajectory,
            demo_anchor_position_m=np.zeros(3),
            live_anchor_position_m=np.ones(3),
        )

    with pytest.raises(ValueError, match="outside a trajectory"):
        ground_trajectory_to_position_anchor(
            demo_trajectory(),
            live_anchor_position_m=np.zeros(3),
            waypoint_index=5,
        )
