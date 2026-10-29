import numpy as np
import pytest

from demo2gripper.demo_object_intent import OBJECT_FRAME_CONVENTION
from demo2gripper.hande_embodiment import (
    ADAPTED_INTENT_SCHEMA,
    HandEEmbodimentConfig,
    adapt_human_intent_to_hande,
    adapted_intent_payload,
)
from demo2gripper.intent import INDEX_TIP, THUMB_TIP, WRIST
from demo2gripper.models import GraspIntent


def _inputs() -> dict[str, object]:
    frame_count = 10
    joints = np.zeros((frame_count, 21, 3), dtype=float)
    # Human opposition is vertical, while wrist-to-pinch has a stable +X yaw.
    joints[:, THUMB_TIP] = [0.01, 0.02, -0.015]
    joints[:, INDEX_TIP] = [0.01, 0.02, 0.015]
    joints[:, WRIST] = [-0.09, 0.02, 0.10]
    raw_transform = np.eye(4)
    raw_transform[:3, :3] = [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
    raw_transform[:3, 3] = [0.02, -0.01, 0.0]
    raw = {
        "schema": "demo2gripper.raw_hand_object_intent.v1",
        "object_frame_id": "die-template-1",
        "object_frame_convention": OBJECT_FRAME_CONVENTION,
        "object_template_sha256": "abc123",
        "T_object_gripper_tcp": raw_transform.tolist(),
        "width_m": 0.03,
        "contact_points_object_m": [[0.0, 0.0, -0.015], [0.0, 0.0, 0.015]],
        "stable_interval": [4, 10],
        "support_frames": 6,
        "continuous_symmetry_axis": None,
    }
    T_base_camera = np.eye(4)
    pickup = {
        "pickup_demonstration_validated": True,
        "stable_contact_interval": [2, 10],
        "grasp_origin_offset_from_contact_object_center_base_m": [0.012, -0.003, 0.001],
        "T_base_camera": T_base_camera.tolist(),
    }
    return {
        "raw_intent": raw,
        "pickup": pickup,
        "hand_joints_camera_m": joints,
        "T_base_camera": T_base_camera,
    }


def test_vertical_human_opposition_maps_to_planar_hande_jaws_and_top_entry() -> None:
    inputs = _inputs()
    result = adapt_human_intent_to_hande(**inputs)

    np.testing.assert_allclose(result.raw_closing_axis_base, [0.0, 0.0, -1.0])
    np.testing.assert_allclose(
        result.raw_palm_axis_base,
        np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(result.palm_planar_axis_base, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.adapted_closing_axis_base, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(result.adapted_approach_axis_base, [0.0, 0.0, -1.0])
    np.testing.assert_allclose(
        result.T_object_gripper_tcp[:3, 3], [0.012, -0.003, 0.001]
    )
    rotation = result.T_object_gripper_tcp[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert result.critical_grasp_interval == (4, 10)


def test_payload_separates_raw_human_evidence_from_robot_adapted_reference() -> None:
    inputs = _inputs()
    config = HandEEmbodimentConfig()
    result = adapt_human_intent_to_hande(**inputs, config=config)
    payload = adapted_intent_payload(
        result,
        raw_intent=inputs["raw_intent"],
        provenance={
            "raw_intent": {"path": "/tmp/raw.json", "sha256": "012345"},
        },
        config=config,
    )

    assert payload["schema"] == ADAPTED_INTENT_SCHEMA
    assert payload["intent_role"] == "robot_adapted_hande"
    assert payload["execution_consumable"] is False
    assert payload["robot_execution_ready"] is False
    assert payload["consumption_role"] == "graspgenx_candidate_ranking_reference_only"
    assert payload["position_mapping"]["source"] == (
        "pickup.grasp_origin_offset_from_contact_object_center_base_m"
    )
    assert payload["position_mapping"]["tracked_occluded_object_center_used"] is False
    assert payload["embodiment_mapping"][
        "raw_thumb_index_axis_used_as_robot_closing_axis"
    ] is False
    assert payload["provenance"]["raw_intent"]["sha256"] == "012345"
    assert payload["width_m"] == pytest.approx(0.03)
    assert payload["human_pinch_width_is_not_robot_opening"] is True
    # The existing candidate selector loads the common GraspIntent fields and
    # safely ignores the additional audit metadata.
    restored = GraspIntent.from_dict(payload)
    np.testing.assert_allclose(
        restored.T_object_gripper_tcp, result.T_object_gripper_tcp
    )


def test_adapter_rejects_unobservable_or_unstable_palm_yaw() -> None:
    inputs = _inputs()
    joints = np.asarray(inputs["hand_joints_camera_m"]).copy()
    # Wrist directly above the pinch midpoint: no support-plane direction.
    joints[4:10, WRIST] = [0.01, 0.02, 0.10]
    inputs["hand_joints_camera_m"] = joints
    with pytest.raises(ValueError, match="insufficient support-plane component"):
        adapt_human_intent_to_hande(**inputs)

    inputs = _inputs()
    joints = np.asarray(inputs["hand_joints_camera_m"]).copy()
    # Alternate the palm yaw by 90 degrees while retaining a strong projection.
    joints[5::2, WRIST] = [0.01, -0.08, 0.10]
    inputs["hand_joints_camera_m"] = joints
    with pytest.raises(ValueError, match="planar direction is not stable enough"):
        adapt_human_intent_to_hande(
            **inputs,
            config=HandEEmbodimentConfig(max_palm_planar_angle_p95_deg=10.0),
        )
