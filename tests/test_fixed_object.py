import numpy as np

from demo2gripper.fixed_object import (
    backproject_masked_depth,
    cloud_motion_metrics,
    infer_fixed_camera_intent,
    voxel_downsample,
)
from demo2gripper.intent import INDEX_TIP, THUMB_TIP, WRIST, IntentConfig


def test_backprojects_masked_depth_and_excludes_hand() -> None:
    depth = np.array([[1.0, 1.0], [2.0, 2.0]])
    intrinsics = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    mask = np.ones((2, 2), dtype=bool)
    exclude = np.array([[False, True], [False, False]])

    points = backproject_masked_depth(
        depth, intrinsics, mask, exclude_mask=exclude, min_depth_m=0.5, max_depth_m=3.0
    )

    np.testing.assert_allclose(points, [[0.0, 0.0, 1.0], [0.0, 1.0, 2.0], [1.0, 1.0, 2.0]])


def test_voxel_downsample_averages_points() -> None:
    points = np.array([[0.001, 0.0, 0.0], [0.002, 0.0, 0.0], [0.011, 0.0, 0.0]])
    downsampled = voxel_downsample(points, 0.01)

    np.testing.assert_allclose(downsampled, [[0.0015, 0.0, 0.0], [0.011, 0.0, 0.0]])


def test_cloud_motion_metrics_detect_translation() -> None:
    first = np.array([[0.0, 0.0, 1.0], [0.02, 0.0, 1.0], [0.0, 0.02, 1.0]])
    second = first + [0.003, 0.0, 0.0]

    metrics = cloud_motion_metrics(first, second)

    assert np.isclose(metrics.centroid_shift_m, 0.003)
    assert np.isclose(metrics.median_nearest_distance_m, 0.003)
    assert np.isclose(metrics.p95_nearest_distance_m, 0.003)


def test_fixed_camera_intent_returns_camera_frame_tcp() -> None:
    frame_count = 8
    joints = np.zeros((frame_count, 21, 3), dtype=float)
    joints[:, WRIST] = [0.0, 0.0, 0.6]
    joints[:, THUMB_TIP] = [-0.02, 0.0, 0.7]
    joints[:, INDEX_TIP] = [0.02, 0.0, 0.7]
    confidence = np.ones((frame_count, 21), dtype=float)
    timestamps = np.arange(frame_count, dtype=float) / 30.0
    object_points = np.array(
        [[-0.02, 0.0, 0.7], [0.02, 0.0, 0.7], [0.0, 0.02, 0.7], [0.0, -0.02, 0.7]]
    )

    intent = infer_fixed_camera_intent(
        joints,
        confidence,
        timestamps,
        object_points,
        IntentConfig(max_tip_object_distance_m=0.005, min_stable_frames=3),
    )

    np.testing.assert_allclose(intent.contact_center_object_m, [0.0, 0.0, 0.7])
    np.testing.assert_allclose(intent.closing_axis_object, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(intent.approach_axis_object, [0.0, 0.0, 1.0])
    assert np.isclose(intent.width_m, 0.04)
