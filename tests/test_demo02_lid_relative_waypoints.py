from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from demo2gripper.cartesian_pose_sequence import load_pose_sequence
from demo2gripper.intent import INDEX_TIP, THUMB_TIP, WRIST
from scripts.build_demo02_lid_relative_waypoints import (
    HANDE_MAX_OPENING_M,
    build_lid_relative_waypoints,
    lid_contact_pose_at_angle,
    write_lid_relative_waypoints,
)


def test_lid_path_preserves_demonstrated_relative_pose_and_reverses_exactly(
    tmp_path,
) -> None:
    frame_count = 8
    T_base_camera = np.eye(4)
    T_base_camera[:3, :3] = Rotation.from_euler("z", 30.0, degrees=True).as_matrix()
    T_base_camera[:3, 3] = [0.2, -0.1, 0.3]

    pinch_base = np.array([0.55, 0.04, 0.05])
    closing_base = np.array([1.0, 0.0, 0.0])
    approach_base = np.array([0.0, 0.0, -1.0])
    width_m = 0.080
    joints_base = np.zeros((frame_count, 21, 3))
    joints_base[:, THUMB_TIP] = pinch_base - 0.5 * width_m * closing_base
    joints_base[:, INDEX_TIP] = pinch_base + 0.5 * width_m * closing_base
    joints_base[:, WRIST] = pinch_base - 0.10 * approach_base
    # One high-confidence outlier verifies that the pose uses robust aggregation.
    joints_base[4, [THUMB_TIP, INDEX_TIP, WRIST]] += [0.0, 0.02, 0.0]

    camera_base_rotation = T_base_camera[:3, :3].T
    joints_camera = np.einsum(
        "ij,fkj->fki",
        camera_base_rotation,
        joints_base - T_base_camera[:3, 3],
    )
    confidence = np.ones((frame_count, 21))
    confidence[2, WRIST] = 0.1

    lid_contact = np.array([0.50, 0.04, 0.04])
    hinge = np.array([0.40, 0.04, 0.04])
    artifacts = build_lid_relative_waypoints(
        hand_joints_camera_m=joints_camera,
        hand_joint_confidence=confidence,
        T_base_camera=T_base_camera,
        contact_frame_interval=(0, frame_count),
        closed_lid_contact_base_m=lid_contact,
        lid_hinge_base_m=hinge,
        lid_positive_y_base=np.array([0.0, 1.0, 0.0]),
        open_angle_deg=30.0,
        angle_step_deg=10.0,
        precontact_retreat_m=0.07,
        min_joint_confidence=0.25,
        min_support_frames=5,
    )

    np.testing.assert_allclose(
        artifacts.T_base_contact_tcp,
        artifacts.T_base_demonstrated_tcp,
        atol=1e-12,
    )
    np.testing.assert_allclose(artifacts.T_base_demonstrated_tcp[:3, 0], closing_base, atol=1e-12)
    np.testing.assert_allclose(artifacts.T_base_demonstrated_tcp[:3, 2], approach_base, atol=1e-12)
    expected_precontact = pinch_base - 0.07 * approach_base
    np.testing.assert_allclose(
        artifacts.T_base_precontact_tcp[:3, 3], expected_precontact, atol=1e-12
    )
    assert artifacts.hande_opening_m == pytest.approx(HANDE_MAX_OPENING_M)
    np.testing.assert_allclose(
        artifacts.T_base_close_tcp,
        tuple(reversed(artifacts.T_base_open_tcp)),
        atol=0.0,
    )

    final_lid = lid_contact_pose_at_angle(
        artifacts.T_base_lid_contact_closed,
        hinge,
        artifacts.T_base_lid_contact_closed[:3, 1],
        artifacts.signed_open_angles_rad[-1],
    )
    np.testing.assert_allclose(
        artifacts.T_base_open_tcp[-1],
        final_lid @ artifacts.T_lid_contact_tcp,
        atol=1e-12,
    )
    assert artifacts.T_base_open_tcp[-1][2, 3] > artifacts.T_base_contact_tcp[2, 3]
    assert artifacts.quality["valid_support_frames"] == 7

    write_lid_relative_waypoints(
        artifacts,
        tmp_path,
        base_frame="fr3_link0",
        contact_frame_interval=(0, frame_count),
        precontact_retreat_m=0.07,
    )
    open_payload = json.loads((tmp_path / "lid_open_waypoints.json").read_text())
    close_payload = json.loads((tmp_path / "lid_close_waypoints.json").read_text())
    assert close_payload["waypoints"] == list(reversed(open_payload["waypoints"]))
    precontact_payload = json.loads((tmp_path / "lid_precontact_tcp.json").read_text())
    retreat_payload = json.loads((tmp_path / "lid_final_retreat_tcp.json").read_text())
    assert retreat_payload["waypoints"] == precontact_payload["waypoints"]
    parsed_open = load_pose_sequence(
        tmp_path / "lid_open_waypoints.json", expected_base_frame="fr3_link0"
    )
    assert len(parsed_open) == 4
    relation = json.loads((tmp_path / "lid_interaction_relation.json").read_text())
    assert relation["hande"]["command_opening_m"] == pytest.approx(0.05)
    assert relation["hande"]["opening_changes_tcp_transform"] is False
    assert relation["signed_open_angles_deg"][-1] == pytest.approx(-30.0)
    assert math.isclose(
        json.loads((tmp_path / "manifest.json").read_text())["quality"][
            "closed_contact_reconstruction_error_m"
        ],
        0.0,
        abs_tol=1e-12,
    )


def test_flip_closing_axis_is_exact_right_multiply_rz_pi(tmp_path) -> None:
    frame_count = 5
    joints = np.zeros((frame_count, 21, 3))
    joints[:, THUMB_TIP] = [-0.02, 0.0, 0.0]
    joints[:, INDEX_TIP] = [0.02, 0.0, 0.0]
    joints[:, WRIST] = [0.0, 0.0, -0.10]
    common = {
        "hand_joints_camera_m": joints,
        "hand_joint_confidence": np.ones((frame_count, 21)),
        "T_base_camera": np.eye(4),
        "contact_frame_interval": (0, frame_count),
        "closed_lid_contact_base_m": np.array([0.10, 0.0, 0.0]),
        "lid_hinge_base_m": np.zeros(3),
        "lid_positive_y_base": np.array([0.0, 1.0, 0.0]),
        "open_angle_deg": 20.0,
        "angle_step_deg": 10.0,
        "precontact_retreat_m": 0.05,
        "min_support_frames": 5,
    }
    demonstrated = build_lid_relative_waypoints(**common)
    flipped = build_lid_relative_waypoints(**common, flip_closing_axis=True)

    rz_pi = np.diag([-1.0, -1.0, 1.0, 1.0])
    np.testing.assert_allclose(
        flipped.T_base_demonstrated_tcp,
        demonstrated.T_base_demonstrated_tcp @ rz_pi,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        flipped.T_base_contact_tcp[:3, 3], demonstrated.T_base_contact_tcp[:3, 3]
    )
    np.testing.assert_allclose(
        flipped.T_base_contact_tcp[:3, 2], demonstrated.T_base_contact_tcp[:3, 2]
    )

    manifest = write_lid_relative_waypoints(
        flipped,
        tmp_path,
        base_frame="fr3_link0",
        contact_frame_interval=(0, frame_count),
        precontact_retreat_m=0.05,
    )
    relation = json.loads((tmp_path / "lid_interaction_relation.json").read_text())
    assert relation["hande_embodiment_candidate"]["closing_axis_flipped"] is True
    assert manifest["hande_embodiment_candidate"] == {
        "closing_axis_flipped": True,
        "right_multiply": "Rz(pi)",
    }
    assert "--flip-closing-axis" in " ".join(manifest["assumptions"])
