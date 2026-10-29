from __future__ import annotations

import numpy as np
import pytest

from demo2gripper.test_tube_demo_relation import (
    DEMO_RELATION_RECEIPT_SCHEMA,
    DEMO_RELATION_REVIEW_SCHEMA,
    DEMO_TUBE_POSE_RECEIPT_SCHEMA,
    estimate_pinch_3d,
    fit_tube_from_pinch_and_length,
    instantiate_top_down_relation,
    interpolate_pinch_keypoints,
    make_tube_frame,
    parse_human_review,
    parse_qualified_pose_receipt,
    runner_geometry_fragment,
    sample_depth_patch,
)


def test_interpolate_pinch_keypoints_without_extrapolation() -> None:
    records = [
        {
            "frame": 10,
            "thumb_tip_uv": [20.0, 30.0],
            "index_tip_uv": [10.0, 30.0],
            "pinch_min_score": 0.8,
        },
        {
            "frame": 20,
            "thumb_tip_uv": [30.0, 40.0],
            "index_tip_uv": [20.0, 40.0],
            "pinch_min_score": 0.6,
        },
    ]
    result = interpolate_pinch_keypoints(records, frame=15, max_frame_gap=5)
    np.testing.assert_allclose(result.thumb_uv, [25.0, 35.0])
    np.testing.assert_allclose(result.index_uv, [15.0, 35.0])
    assert result.confidence == pytest.approx(0.6)
    assert result.source_frames == (10, 20)
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_pinch_keypoints(records, frame=5)


def test_depth_patch_and_pinch_lift_preserve_provenance() -> None:
    K = np.array([[100.0, 0.0, 5.0], [0.0, 100.0, 5.0], [0.0, 0.0, 1.0]])
    depth = np.full((12, 12), 0.4)
    depth[4, 4] = np.nan
    sample = sample_depth_patch(depth, [4.0, 4.0], radius_px=1)
    assert sample.valid_count == 8
    assert sample.depth_m == pytest.approx(0.4)
    pinch = interpolate_pinch_keypoints(
        [
            {
                "frame": 2,
                "thumb_tip_uv": [7.0, 5.0],
                "index_tip_uv": [3.0, 5.0],
                "pinch_min_score": 0.9,
            }
        ],
        frame=2,
    )
    lifted = estimate_pinch_3d(K=K, depth_m=depth, pinch=pinch, depth_radius_px=0)
    np.testing.assert_allclose(lifted.midpoint_camera_m, [0.0, 0.0, 0.4])
    assert lifted.contact_separation_m == pytest.approx(0.016)
    np.testing.assert_allclose(lifted.closing_axis_camera_unit, [1.0, 0.0, 0.0])


def test_metric_candidate_recovers_synthetic_endpoint_rays() -> None:
    K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
    closed = np.array([-0.03, 0.02, 0.50])
    opened = np.array([0.04, -0.01, 0.42])
    length = float(np.linalg.norm(opened - closed))

    def project(point: np.ndarray) -> np.ndarray:
        homogeneous = K @ point
        return homogeneous[:2] / homogeneous[2]

    fraction = 0.65
    pinch = (1.0 - fraction) * closed + fraction * opened
    result = fit_tube_from_pinch_and_length(
        K=K,
        closed_uv=project(closed),
        open_uv=project(opened),
        pinch_camera_m=pinch,
        axial_fraction_closed_to_open=fraction,
        tube_length_m=length,
        pinch_position_sigma_m=0.001,
        tube_length_sigma_m=0.001,
    )
    np.testing.assert_allclose(result.closed_camera_m, closed, atol=1e-6)
    np.testing.assert_allclose(result.open_camera_m, opened, atol=1e-6)
    assert result.centreline_residual_m < 1e-7
    assert result.length_residual_m < 1e-7


def test_top_down_relation_resolves_roll_and_preserves_axial_fraction() -> None:
    axis = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    T_camera_tube = make_tube_frame(
        centre_parent_m=[0.1, -0.2, 0.5],
        axis_closed_to_open_parent=axis,
        transverse_reference_parent=[0.0, 1.0, 0.0],
    )
    fraction = 0.7
    length = 0.1
    relation, detail = instantiate_top_down_relation(
        T_base_camera=np.eye(4),
        T_camera_tube=T_camera_tube,
        tube_length_m=length,
        axial_fraction_closed_to_open=fraction,
        human_closing_axis_camera=[0.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(relation[:3, 3], [0.0, 0.0, 0.02], atol=1e-9)
    base_tcp = np.asarray(detail["T_base_tcp_demo_target"])
    assert abs(float(base_tcp[:3, 0] @ axis)) < 1e-9
    np.testing.assert_allclose(base_tcp[:3, 2], [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(
        base_tcp[:3, :3].T @ base_tcp[:3, :3], np.eye(3), atol=1e-12
    )


def _qualified_pose_receipt() -> dict[str, object]:
    return {
        "schema": DEMO_TUBE_POSE_RECEIPT_SCHEMA,
        "frame": 246,
        "runtime_qualified": True,
        "T_camera_tube": np.eye(4).tolist(),
        "tube_frame": {
            "origin": "geometric_centre",
            "z_axis": "closed_to_open",
            "roll": "canonical_gauge",
        },
        "tube_length_m": 0.1,
        "tube_length_uncertainty_m": 0.002,
        "position_std_m": 0.004,
        "orientation_std_deg": 3.0,
        "provenance": {"method": "fiducial fixture"},
    }


def test_pose_and_human_review_must_both_be_explicitly_qualified() -> None:
    transform, length, blockers, _ = parse_qualified_pose_receipt(
        _qualified_pose_receipt(), expected_frame=246
    )
    np.testing.assert_allclose(transform, np.eye(4))
    assert length == pytest.approx(0.1)
    assert blockers == []
    review = {
        "schema": DEMO_RELATION_REVIEW_SCHEMA,
        "frame": 246,
        "accepted": True,
        "closed_open_labels_accepted": True,
        "pinch_keypoints_accepted": True,
        "top_approach_accepted": True,
        "reviewer": "operator",
    }
    review_blockers, _ = parse_human_review(review, expected_frame=246)
    assert review_blockers == []
    bad = _qualified_pose_receipt()
    bad["tube_frame"] = {"origin": "geometric_centre", "z_axis": "unsigned"}
    _, _, bad_blockers, _ = parse_qualified_pose_receipt(bad, expected_frame=246)
    assert "metric_demo_tube_end_sign_not_qualified" in bad_blockers
    _, _, template_blockers, _ = parse_qualified_pose_receipt(
        {
            "schema": DEMO_TUBE_POSE_RECEIPT_SCHEMA,
            "frame": None,
            "runtime_qualified": False,
            "T_camera_tube": None,
            "tube_length_m": None,
            "position_std_m": None,
            "orientation_std_deg": None,
        },
        expected_frame=246,
    )
    assert "metric_demo_tube_pose_frame_mismatch" in template_blockers
    assert "metric_demo_tube_position_uncertainty_too_high" in template_blockers


def test_runner_fragment_fails_closed_for_partial_relation() -> None:
    partial = {
        "schema": DEMO_RELATION_RECEIPT_SCHEMA,
        "relation_id": "relation-1",
        "runtime_qualified": False,
        "T_tube_tcp_demo_grasp": None,
    }
    with pytest.raises(ValueError, match="not runtime qualified"):
        runner_geometry_fragment(partial)
    qualified = {
        **partial,
        "runtime_qualified": True,
        "T_tube_tcp_demo_grasp": np.eye(4).tolist(),
    }
    result = runner_geometry_fragment(qualified)
    assert result["relation_runtime_qualified"] is True
    np.testing.assert_allclose(result["T_tube_tcp_demo_grasp"], np.eye(4))
