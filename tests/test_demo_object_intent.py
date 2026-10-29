import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from demo2gripper.demo_object_intent import (
    DemoObjectIntentConfig,
    build_demo_object_intent,
    write_demo_object_intent,
)
from demo2gripper.intent import INDEX_TIP, THUMB_TIP, WRIST
from demo2gripper.io import load_grasp_intent_json, load_observation_npz


def _synthetic_inputs() -> dict[str, object]:
    frame_count = 12
    reference = np.array([0.3, -0.1, 0.7])
    centers = reference + np.column_stack(
        (
            np.linspace(0.0, 0.03, frame_count),
            np.linspace(0.0, -0.01, frame_count),
            np.zeros(frame_count),
        )
    )
    joints_object = np.zeros((frame_count, 21, 3), dtype=float)
    joints_object[:, WRIST] = [0.0, 0.0, -0.10]
    joints_object[:, THUMB_TIP] = [-0.02, 0.0, 0.0]
    joints_object[:, INDEX_TIP] = [0.02, 0.0, 0.0]
    # One raw-joint outlier remains a valid frame and exercises robust aggregation.
    joints_object[6, THUMB_TIP] += [-0.01, 0.02, 0.01]
    joints_object[6, INDEX_TIP] += [0.01, 0.02, 0.01]
    joints = joints_object + centers[:, None, :]
    confidence = np.ones((frame_count, 21), dtype=float)
    timestamps = np.arange(frame_count, dtype=float) / 30.0
    cloud_object = np.array(
        [
            [-0.02, -0.02, -0.02],
            [0.02, -0.02, -0.02],
            [0.02, 0.02, 0.02],
            [-0.02, 0.02, 0.02],
        ]
    )
    return {
        "pickup": {
            "pickup_demonstration_validated": True,
            "stable_contact_interval": [2, 10],
            "reference_object_center_camera_m": reference.tolist(),
        },
        "object_centers_camera_m": centers,
        "track_timestamps_s": timestamps.copy(),
        "hand_joints_camera_m": joints,
        "hand_joint_confidence": confidence,
        "timestamps_s": timestamps,
        "precontact_object_points_camera_m": cloud_object + reference,
    }


def test_translation_tracking_recovers_raw_object_relative_hand_frame() -> None:
    inputs = _synthetic_inputs()
    artifacts = build_demo_object_intent(**inputs)

    np.testing.assert_allclose(
        artifacts.object_points_m,
        np.asarray(inputs["precontact_object_points_camera_m"])
        - np.asarray(inputs["pickup"]["reference_object_center_camera_m"]),
        atol=1e-12,
    )
    transforms = np.repeat(np.eye(4)[None], 12, axis=0)
    transforms[:, :3, 3] = np.asarray(inputs["object_centers_camera_m"])
    reconstructed = (
        np.einsum(
            "fij,fkj->fki",
            transforms[:, :3, :3],
            artifacts.hand_joints_object_m,
        )
        + transforms[:, None, :3, 3]
    )
    np.testing.assert_allclose(reconstructed, inputs["hand_joints_camera_m"])

    intent = artifacts.intent
    np.testing.assert_allclose(
        intent.contact_center_object_m, [0.015, -0.005, 0.0], atol=3e-4
    )
    np.testing.assert_allclose(intent.closing_axis_object, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(intent.approach_axis_object, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(intent.T_object_gripper_tcp[:3, :3], np.eye(3))
    assert np.isclose(intent.width_m, 0.04)
    assert intent.stable_interval == (2, 10)
    assert intent.support_frames == 8
    assert artifacts.validated_contact_interval == (2, 10)
    assert artifacts.critical_grasp_interval == (2, 10)
    assert np.count_nonzero(artifacts.validated_contact_frame_mask) == 8
    assert np.count_nonzero(artifacts.valid_contact_frame_mask) == 8


def test_pickup_critical_interval_drives_pose_but_preserves_contact_evidence() -> None:
    inputs = _synthetic_inputs()
    pickup = dict(inputs["pickup"])
    pickup["critical_grasp_interval"] = [7, 10]
    inputs["pickup"] = pickup
    joints = np.asarray(inputs["hand_joints_camera_m"]).copy()
    centers = np.asarray(inputs["object_centers_camera_m"])

    # Early validated-contact frames contain a different pose.  They remain
    # evidence of contact, but must not influence the terminal grasp SE(3).
    joints[2:7, THUMB_TIP] = centers[2:7] + [0.0, -0.02, 0.0]
    joints[2:7, INDEX_TIP] = centers[2:7] + [0.0, 0.02, 0.0]
    joints[2:7, WRIST] = centers[2:7] + [0.0, 0.0, -0.10]
    inputs["hand_joints_camera_m"] = joints

    artifacts = build_demo_object_intent(
        **inputs,
        config=DemoObjectIntentConfig(
            min_support_frames=3,
            critical_grasp_tail_frames=4,
        ),
    )

    assert artifacts.validated_contact_interval == (2, 10)
    assert artifacts.critical_grasp_interval == (7, 10)
    assert artifacts.intent.stable_interval == (7, 10)
    assert artifacts.intent.support_frames == 3
    assert np.count_nonzero(artifacts.validated_contact_frame_mask) == 8
    assert np.count_nonzero(artifacts.valid_contact_frame_mask) == 3
    np.testing.assert_allclose(
        artifacts.intent.closing_axis_object, [1.0, 0.0, 0.0], atol=1e-12
    )
    assert artifacts.quality["validated_contact_interval_frames"] == 8
    assert artifacts.quality["critical_grasp_interval_frames"] == 3
    assert (
        artifacts.quality["critical_grasp_interval_source"]
        == "pickup_grasp.critical_grasp_interval"
    )


def test_configured_terminal_tail_supports_legacy_pickup_json() -> None:
    inputs = _synthetic_inputs()
    artifacts = build_demo_object_intent(
        **inputs,
        config=DemoObjectIntentConfig(
            min_support_frames=3,
            critical_grasp_tail_frames=3,
        ),
    )

    assert artifacts.validated_contact_interval == (2, 10)
    assert artifacts.critical_grasp_interval == (7, 10)
    assert artifacts.intent.stable_interval == (7, 10)
    assert (
        artifacts.quality["critical_grasp_interval_source"]
        == "terminal_tail_of_pickup_grasp.stable_contact_interval"
    )


def test_critical_pose_ignores_occluded_tracked_center_jump() -> None:
    inputs = _synthetic_inputs()
    pickup = dict(inputs["pickup"])
    pickup["critical_grasp_interval"] = [7, 10]
    inputs["pickup"] = pickup
    baseline = build_demo_object_intent(
        **inputs,
        config=DemoObjectIntentConfig(min_support_frames=3),
    )

    jumped = dict(inputs)
    jumped_centers = np.asarray(inputs["object_centers_camera_m"]).copy()
    jumped_centers[7:10] += [0.0104, -0.006, 0.004]
    jumped["object_centers_camera_m"] = jumped_centers
    changed_diagnostic = build_demo_object_intent(
        **jumped,
        config=DemoObjectIntentConfig(min_support_frames=3),
    )

    np.testing.assert_allclose(
        changed_diagnostic.intent.T_object_gripper_tcp,
        baseline.intent.T_object_gripper_tcp,
        atol=1e-12,
    )
    assert not np.allclose(
        changed_diagnostic.hand_joints_object_m[7:10],
        baseline.hand_joints_object_m[7:10],
    )
    assert changed_diagnostic.quality["tracked_object_centers_used_for_grasp_se3"] is False


def test_critical_interval_must_be_inside_validated_contact() -> None:
    inputs = _synthetic_inputs()
    pickup = dict(inputs["pickup"])
    pickup["critical_grasp_interval"] = [1, 5]
    inputs["pickup"] = pickup

    with pytest.raises(ValueError, match="contained in stable_contact_interval"):
        build_demo_object_intent(**inputs)


def test_camera_rigid_reexpression_rotates_template_and_intent_axes() -> None:
    inputs = _synthetic_inputs()
    original = build_demo_object_intent(**inputs)
    rotation = Rotation.from_euler("xyz", [0.3, -0.4, 0.7]).as_matrix()
    translation = np.array([-0.2, 0.5, 0.1])

    transformed = dict(inputs)
    pickup = dict(inputs["pickup"])
    reference = np.asarray(pickup["reference_object_center_camera_m"])
    pickup["reference_object_center_camera_m"] = (
        rotation @ reference + translation
    ).tolist()
    transformed["pickup"] = pickup
    transformed["object_centers_camera_m"] = (
        np.asarray(inputs["object_centers_camera_m"]) @ rotation.T + translation
    )
    transformed["hand_joints_camera_m"] = (
        np.asarray(inputs["hand_joints_camera_m"]) @ rotation.T + translation
    )
    transformed["precontact_object_points_camera_m"] = (
        np.asarray(inputs["precontact_object_points_camera_m"]) @ rotation.T
        + translation
    )
    changed = build_demo_object_intent(**transformed)

    np.testing.assert_allclose(changed.object_points_m, original.object_points_m @ rotation.T)
    np.testing.assert_allclose(
        changed.intent.T_object_gripper_tcp[:3, :3],
        rotation @ original.intent.T_object_gripper_tcp[:3, :3],
    )
    np.testing.assert_allclose(
        changed.intent.contact_center_object_m,
        rotation @ original.intent.contact_center_object_m,
        atol=1e-12,
    )


def test_writes_loadable_intent_observations_and_content_hash(tmp_path: Path) -> None:
    inputs = _synthetic_inputs()
    artifacts = build_demo_object_intent(**inputs)
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = write_demo_object_intent(artifacts, first, write_observations=True)
    repeated = write_demo_object_intent(artifacts, second, write_observations=True)

    points_bytes = (first / "object_points_m.npy").read_bytes()
    file_digest = hashlib.sha256(points_bytes).hexdigest()
    assert manifest["object_points_file_sha256"] == file_digest
    assert manifest["frame_id"].startswith("object_template_camera_axes:sha256:")
    assert repeated["object_template_sha256"] == manifest["object_template_sha256"]
    assert repeated["frame_id"] == manifest["frame_id"]

    intent = load_grasp_intent_json(first / "grasp_intent.json")
    np.testing.assert_allclose(
        intent.T_object_gripper_tcp, artifacts.intent.T_object_gripper_tcp
    )
    observation = load_observation_npz(
        first / "observations.npz", first / "object_points_m.npy"
    )
    np.testing.assert_allclose(observation.object_points_m, artifacts.object_points_m)
    reconstructed = observation.hand_joints_camera_m - observation.T_camera_object[
        :, None, :3, 3
    ]
    np.testing.assert_allclose(reconstructed, artifacts.hand_joints_object_m)

    payload = json.loads((first / "grasp_intent.json").read_text())
    assert payload["inference"]["contact_geometry"].endswith(
        "without_cloud_projection"
    )
    assert payload["closing_axis_equivalence"] == "+X~-X"
    assert payload["object_frame_id"] == manifest["frame_id"]
    assert payload["template_manifest"] == "template_manifest.json"
    assert payload["object_frame_convention"] == manifest["frame_convention"]
    assert payload["inference"]["validated_contact_interval"] == [2, 10]
    assert payload["inference"]["critical_grasp_interval"] == [2, 10]
    assert (
        payload["inference"]["grasp_se3_object_anchor"]
        == "frozen_reference_object_center_camera_m"
    )
    assert (
        payload["inference"]["tracked_object_centers_usage"]
        == "diagnostic_only_not_grasp_se3"
    )


LATEST_DEMO = Path(
    "runs/live/quick_chain_validation_20261021/demo_01"
)


@pytest.mark.skipif(not LATEST_DEMO.exists(), reason="latest live demo is not present")
def test_latest_demo_builds_without_fingertip_cloud_projection(tmp_path: Path) -> None:
    pickup_dir = LATEST_DEMO / "pickup_result_final"
    pickup = json.loads((pickup_dir / "pickup_grasp.json").read_text())
    with np.load(pickup_dir / "pickup_track.npz", allow_pickle=False) as track:
        centers = track["object_centers_camera_m"]
        track_timestamps = track["timestamps_s"]
    hand_dir = LATEST_DEMO / "hand_pose"
    artifacts = build_demo_object_intent(
        pickup=pickup,
        object_centers_camera_m=centers,
        track_timestamps_s=track_timestamps,
        hand_joints_camera_m=np.load(hand_dir / "hand_joints_camera_m.npy"),
        hand_joint_confidence=np.load(hand_dir / "hand_joint_confidence.npy"),
        timestamps_s=np.load(LATEST_DEMO / "input" / "timestamps_s.npy"),
        precontact_object_points_camera_m=np.load(
            pickup_dir / "precontact_object_points_camera_m.npy"
        ),
        config=DemoObjectIntentConfig(),
    )
    assert artifacts.intent.stable_interval == tuple(pickup["stable_contact_interval"])
    assert artifacts.intent.support_frames >= 3
    assert np.isclose(
        np.linalg.det(artifacts.intent.T_object_gripper_tcp[:3, :3]), 1.0
    )
    manifest = write_demo_object_intent(artifacts, tmp_path / "latest")
    assert manifest["tracking_model"] == "translation_only"
