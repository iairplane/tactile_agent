import json

import numpy as np
from scipy.spatial.transform import Rotation

from demo2gripper.geometry import transform_points
from demo2gripper.intent import (
    INDEX_TIP,
    THUMB_TIP,
    WRIST,
    IntentConfig,
    infer_grasp_intent,
)
from demo2gripper.io import load_observation_npz
from demo2gripper.models import GraspIntent, ObservationSequence


def _make_observation() -> ObservationSequence:
    frame_count = 12
    joints_object = np.zeros((frame_count, 21, 3))
    joints_object[:, WRIST] = [0.0, 0.0, -0.10]
    joints_object[:, THUMB_TIP] = [-0.20, 0.0, 0.0]
    joints_object[:, INDEX_TIP] = [0.20, 0.0, 0.0]

    # The longest valid contact is [2, 8); [9, 11) must not win.
    joints_object[2:8, THUMB_TIP] = [-0.025, 0.0, 0.0]
    joints_object[2:8, INDEX_TIP] = [0.025, 0.0, 0.0]
    joints_object[9:11, THUMB_TIP] = [-0.025, 0.0, 0.0]
    joints_object[9:11, INDEX_TIP] = [0.025, 0.0, 0.0]

    T_camera_object = np.repeat(np.eye(4)[None], frame_count, axis=0)
    for frame in range(frame_count):
        angle = 0.03 * frame
        T_camera_object[frame, :3, :3] = Rotation.from_euler(
            "xyz", [angle, -0.5 * angle, 0.2 * angle]
        ).as_matrix()
        T_camera_object[frame, :3, 3] = [0.3, -0.1 + 0.002 * frame, 0.7]

    return ObservationSequence(
        T_camera_object=T_camera_object,
        hand_joints_camera_m=transform_points(T_camera_object, joints_object),
        hand_joint_confidence=np.ones((frame_count, 21)),
        timestamps_s=np.arange(frame_count) / 30.0,
        object_points_m=np.array(
            [
                [-0.025, 0.0, 0.0],
                [0.025, 0.0, 0.0],
                [0.0, 0.03, 0.0],
                [0.0, -0.03, 0.0],
            ]
        ),
    )


def _change_camera_coordinates(
    observation: ObservationSequence,
) -> ObservationSequence:
    T_new_camera_old_camera = np.eye(4)
    T_new_camera_old_camera[:3, :3] = Rotation.from_euler(
        "xyz", [0.4, -0.2, 0.7]
    ).as_matrix()
    T_new_camera_old_camera[:3, 3] = [-0.4, 0.2, 0.1]
    return ObservationSequence(
        T_camera_object=T_new_camera_old_camera @ observation.T_camera_object,
        hand_joints_camera_m=transform_points(
            T_new_camera_old_camera, observation.hand_joints_camera_m
        ),
        hand_joint_confidence=observation.hand_joint_confidence,
        timestamps_s=observation.timestamps_s,
        object_points_m=observation.object_points_m,
    )


def test_infers_object_frame_intent_from_longest_stable_contact() -> None:
    observation = _make_observation()
    intent = infer_grasp_intent(
        observation,
        IntentConfig(max_tip_object_distance_m=0.01, min_stable_frames=3),
    )

    assert intent.stable_interval == (2, 8)
    assert intent.support_frames == 6
    np.testing.assert_allclose(intent.contact_center_object_m, np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(intent.closing_axis_object, [1.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(intent.approach_axis_object, [0.0, 0.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(intent.T_object_gripper_tcp[:3, :3], np.eye(3), atol=1e-10)
    assert np.isclose(intent.width_m, 0.05)
    json.dumps(intent.to_dict())


def test_intent_is_invariant_to_camera_coordinate_change() -> None:
    observation = _make_observation()
    transformed_observation = _change_camera_coordinates(observation)
    config = IntentConfig(max_tip_object_distance_m=0.01, min_stable_frames=3)

    original = infer_grasp_intent(observation, config)
    transformed = infer_grasp_intent(transformed_observation, config)

    np.testing.assert_allclose(
        transformed.T_object_gripper_tcp,
        original.T_object_gripper_tcp,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        transformed.contact_points_object_m,
        original.contact_points_object_m,
        atol=1e-10,
    )
    assert np.isclose(transformed.width_m, original.width_m)


def test_loads_observations_and_separate_object_points(tmp_path) -> None:
    observation = _make_observation()
    observations_path = tmp_path / "observations.npz"
    object_points_path = tmp_path / "object_points_m.npy"
    np.savez(
        observations_path,
        T_camera_object=observation.T_camera_object,
        hand_joints_camera_m=observation.hand_joints_camera_m,
        hand_joint_confidence=observation.hand_joint_confidence,
        timestamps_s=observation.timestamps_s,
    )
    np.save(object_points_path, observation.object_points_m)

    loaded = load_observation_npz(observations_path, object_points_path)

    np.testing.assert_allclose(loaded.T_camera_object, observation.T_camera_object)
    np.testing.assert_allclose(
        loaded.hand_joints_camera_m, observation.hand_joints_camera_m
    )
    np.testing.assert_allclose(
        loaded.hand_joint_confidence, observation.hand_joint_confidence
    )
    np.testing.assert_allclose(loaded.timestamps_s, observation.timestamps_s)
    np.testing.assert_allclose(loaded.object_points_m, observation.object_points_m)


def test_axial_symmetry_removes_per_frame_tracker_yaw() -> None:
    frame_count = 9
    joints_canonical = np.zeros((frame_count, 21, 3))
    joints_canonical[:, WRIST] = [0.0, 0.0, -0.10]
    joints_canonical[:, THUMB_TIP] = [-0.025, 0.0, 0.0]
    joints_canonical[:, INDEX_TIP] = [0.025, 0.0, 0.0]

    T_camera_canonical = np.eye(4)
    T_camera_canonical[:3, :3] = Rotation.from_euler(
        "xyz", [0.3, -0.4, 0.2]
    ).as_matrix()
    T_camera_canonical[:3, 3] = [0.2, -0.1, 0.6]
    hand_camera = transform_points(T_camera_canonical, joints_canonical)

    tracker_yaws = np.array([0.0, 0.41, -1.2, 2.3, -2.7, 1.1, -0.8, 2.9, -1.9])
    T_camera_object = np.empty((frame_count, 4, 4))
    for frame, yaw in enumerate(tracker_yaws):
        T_object_gauge = np.eye(4)
        T_object_gauge[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        T_camera_object[frame] = T_camera_canonical @ T_object_gauge

    angles = np.linspace(0.0, 2.0 * np.pi, 1440, endpoint=False)
    ring = np.column_stack(
        (0.025 * np.cos(angles), 0.025 * np.sin(angles), np.zeros_like(angles))
    )
    observation = ObservationSequence(
        T_camera_object=T_camera_object,
        hand_joints_camera_m=hand_camera,
        hand_joint_confidence=np.ones((frame_count, 21)),
        timestamps_s=np.arange(frame_count) / 30.0,
        object_points_m=ring,
    )

    intent = infer_grasp_intent(
        observation,
        IntentConfig(
            max_tip_object_distance_m=0.002,
            min_stable_frames=3,
            continuous_symmetry_axis=2,
        ),
    )

    assert intent.stable_interval == (0, frame_count)
    assert intent.continuous_symmetry_axis == 2
    np.testing.assert_allclose(intent.contact_center_object_m, np.zeros(3), atol=1e-5)
    np.testing.assert_allclose(intent.closing_axis_object, [1.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(intent.approach_axis_object, [0.0, 0.0, 1.0], atol=1e-5)
    restored = GraspIntent.from_dict(intent.to_dict())
    assert restored.continuous_symmetry_axis == 2
    np.testing.assert_allclose(
        restored.T_object_gripper_tcp, intent.T_object_gripper_tcp
    )
