from __future__ import annotations

import numpy as np
import pytest

from demo2gripper.cup_rim_relation import compile_cup_rim_relation
from demo2gripper.intent import INDEX_TIP, THUMB_TIP, WRIST


def _synthetic_cup() -> dict[str, object]:
    frame_count, height, width = 12, 160, 200
    K = np.array([[300.0, 0.0, 100.0], [0.0, 300.0, 70.0], [0.0, 0.0, 1.0]])
    rim_center = np.array([0.0, 0.0, 0.75])
    rim_radius_m = 0.05
    masks = np.zeros((4, height, width), dtype=bool)
    depths = np.zeros((4, height, width), dtype=float)
    for frame in range(4):
        for x in range(80, 121):
            unit_x = (x - 100) / 20.0
            top = int(round(70 - 20 * np.sqrt(max(0.0, 1.0 - unit_x**2))))
            masks[frame, top:131, x] = True
            depths[frame, top:131, x] = rim_center[2]

    joints = np.zeros((frame_count, 21, 3), dtype=float)
    confidence = np.ones((frame_count, 21), dtype=float)
    pinch = rim_center + [rim_radius_m, 0.0, 0.0]
    joints[:, THUMB_TIP] = pinch + [-0.015, 0.0, 0.0]
    joints[:, INDEX_TIP] = pinch + [0.015, 0.0, 0.0]
    joints[:, WRIST] = pinch + [0.0, 0.0, -0.10]
    return {
        "precontact_depth_m": depths,
        "precontact_masks": masks,
        "precontact_frame_indices": np.arange(1, 5),
        "K": K,
        "hand_joints_camera_m": joints,
        "hand_joint_confidence": confidence,
        "stage_intervals": {"PRESHAPE": (1, 5), "GRASP": (5, 10)},
    }


def test_mask_depth_rim_and_human_relation_recover_synthetic_geometry() -> None:
    artifacts = compile_cup_rim_relation(**_synthetic_cup())

    np.testing.assert_allclose(
        artifacts.T_camera_rim[:3, 3], [0.0, 0.0, 0.75], atol=8e-4
    )
    np.testing.assert_allclose(artifacts.T_camera_rim[:3, 2], [0.0, 0.0, -1.0])
    assert artifacts.rim_radius_m == pytest.approx(0.05, abs=8e-4)
    assert artifacts.T_rim_human_pinch_tcp[0, 3] == pytest.approx(0.05, abs=8e-4)
    assert artifacts.T_rim_human_pinch_tcp[1, 3] == pytest.approx(0.0, abs=8e-4)
    assert artifacts.T_rim_human_pinch_tcp[2, 3] == pytest.approx(0.0, abs=8e-4)
    assert artifacts.pinch_width_m == pytest.approx(0.03)
    assert artifacts.quality["grasp_time_mask_centroids_used"] is False
    assert artifacts.quality["execution_authorized"] is False
    assert artifacts.quality["rim_arc_coverage_min_deg"] > 170.0
    assert artifacts.quality["closing_to_local_radial_angle_deg"] < 1.0


def test_supplied_image_ellipse_uses_depth_without_mask_centroid_tracking() -> None:
    inputs = _synthetic_cup()
    rows, columns = np.indices((160, 200))
    ellipse_mask = ((columns - 100) / 20.0) ** 2 + ((rows - 70) / 10.0) ** 2 <= 1.0
    masks = np.repeat(ellipse_mask[None], 4, axis=0)
    ellipse_depth = 0.75 + 0.0043 * (rows - 70)
    depths = np.repeat(np.where(ellipse_mask, ellipse_depth, 0.0)[None], 4, axis=0)
    mixed_background = ellipse_mask & (columns > 112)
    depths[:, mixed_background] = 1.7
    inputs["precontact_masks"] = masks
    inputs["precontact_depth_m"] = depths
    ellipses = np.repeat(
        np.array([[100.0, 70.0, 20.0, 10.0, 0.0]]), 4, axis=0
    )

    artifacts = compile_cup_rim_relation(**inputs, rim_ellipses_px=ellipses)

    assert artifacts.rim_radius_m == pytest.approx(0.05, abs=1.5e-3)
    assert artifacts.quality["rim_extraction_methods"] == [
        "calibrated_image_ellipse_near_depth_approximation"
    ]
    assert 0.5 < artifacts.quality["rim_depth_support_fraction_min"] < 1.0
    assert artifacts.quality["ellipse_depth_gradient_relative_error_max"] < 0.1
    assert artifacts.quality["grasp_time_mask_centroids_used"] is False


def test_one_occluded_precontact_frame_is_recorded_but_not_silently_used() -> None:
    inputs = _synthetic_cup()
    masks = np.asarray(inputs["precontact_masks"]).copy()
    masks[0] = False
    inputs["precontact_masks"] = masks

    artifacts = compile_cup_rim_relation(**inputs)

    np.testing.assert_array_equal(artifacts.valid_precontact_frame_indices, [2, 3, 4])
    assert artifacts.quality["valid_precontact_fraction"] == pytest.approx(0.75)
    assert artifacts.quality["precontact_failures"] == {1: "mask has too few pixels"}


def test_occluded_rim_sequence_fails_closed() -> None:
    inputs = _synthetic_cup()
    masks = np.asarray(inputs["precontact_masks"]).copy()
    masks[:2] = False
    inputs["precontact_masks"] = masks

    with pytest.raises(ValueError, match="precontact rim frames passed"):
        compile_cup_rim_relation(**inputs)


def test_border_truncated_masks_fail_closed() -> None:
    inputs = _synthetic_cup()
    masks = np.asarray(inputs["precontact_masks"]).copy()
    masks[:, :, 0] = True
    inputs["precontact_masks"] = masks

    with pytest.raises(ValueError, match="mask touches the image border"):
        compile_cup_rim_relation(**inputs)


def test_inconsistent_thumb_index_lines_fail_closed() -> None:
    inputs = _synthetic_cup()
    joints = np.asarray(inputs["hand_joints_camera_m"]).copy()
    pinch = np.array([0.05, 0.0, 0.75])
    for frame in (6, 8):
        joints[frame, THUMB_TIP] = pinch + [0.0, -0.015, 0.0]
        joints[frame, INDEX_TIP] = pinch + [0.0, 0.015, 0.0]
    inputs["hand_joints_camera_m"] = joints

    with pytest.raises(ValueError, match="closing line is ambiguous"):
        compile_cup_rim_relation(**inputs)


def test_centered_pinch_cannot_define_local_radial_direction() -> None:
    inputs = _synthetic_cup()
    joints = np.asarray(inputs["hand_joints_camera_m"]).copy()
    center = np.array([0.0, 0.0, 0.75])
    joints[5:10, THUMB_TIP] = center + [-0.015, 0.0, 0.0]
    joints[5:10, INDEX_TIP] = center + [0.015, 0.0, 0.0]
    joints[5:10, WRIST] = center + [0.0, 0.0, -0.10]
    inputs["hand_joints_camera_m"] = joints

    with pytest.raises(ValueError, match="too close to the rim center"):
        compile_cup_rim_relation(**inputs)


def test_low_confidence_grasp_occlusion_fails_closed() -> None:
    inputs = _synthetic_cup()
    confidence = np.asarray(inputs["hand_joint_confidence"]).copy()
    confidence[5:9, INDEX_TIP] = 0.0
    inputs["hand_joint_confidence"] = confidence

    with pytest.raises(ValueError, match="too few confident"):
        compile_cup_rim_relation(**inputs)


def test_precontact_samples_must_be_inside_preshape() -> None:
    inputs = _synthetic_cup()
    inputs["precontact_frame_indices"] = np.arange(0, 4)

    with pytest.raises(ValueError, match="inside APPROACH or PRESHAPE"):
        compile_cup_rim_relation(**inputs)
