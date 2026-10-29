import json
from pathlib import Path

import numpy as np
import pytest

from demo2gripper.graspgenx_adapter import (
    R_CANONICAL_HANDE_BASE,
    R_HANDE_BASE_CANONICAL_TCP,
    canonical_grasp_poses_to_object_tcp,
    center_object_points,
    hande_base_to_canonical_tcp,
    load_hande_dimensions,
    save_candidates_npz,
)
from demo2gripper.io import load_candidates_npz


def test_center_and_restore_canonical_grasp_then_apply_local_tcp_offset() -> None:
    points = np.array(
        [
            [0.10, -0.02, 0.30],
            [0.14, -0.02, 0.30],
            [0.10, 0.04, 0.36],
            [0.14, 0.04, 0.36],
        ]
    )
    centered, T_object_centered = center_object_points(points)
    np.testing.assert_allclose(centered.mean(axis=0), np.zeros(3), atol=1e-8)
    np.testing.assert_allclose(T_object_centered[:3, 3], points.mean(axis=0))
    np.testing.assert_allclose(
        centered + T_object_centered[:3, 3],
        points,
        atol=1e-8,
    )

    # Local +Z points along object +X.  A 0.133 m TCP offset must therefore
    # change object X, which catches accidental world-Z translation.
    T_centered_canonical = np.eye(4)
    T_centered_canonical[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    T_centered_canonical[:3, 3] = [0.01, 0.02, -0.03]
    result = canonical_grasp_poses_to_object_tcp(
        T_centered_canonical[None], T_object_centered, 0.133
    )[0]

    expected = T_object_centered @ T_centered_canonical
    expected[:3, 3] += expected[:3, 2] * 0.133
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_hande_native_base_axes_map_to_graspgenx_canonical_axes() -> None:
    # Shipped URDF world_joint rpy=(0, -pi/2, +pi/2).  Its native base +Y is
    # the finger-joint/closing axis and native +X runs toward the fingertips.
    np.testing.assert_allclose(
        R_CANONICAL_HANDE_BASE,
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        R_HANDE_BASE_CANONICAL_TCP[:, 0],
        [0.0, -1.0, 0.0],
        atol=1e-12,
    )  # canonical +X = native -Y (closing, sign-symmetric)
    np.testing.assert_allclose(
        R_HANDE_BASE_CANONICAL_TCP[:, 2],
        [1.0, 0.0, 0.0],
        atol=1e-12,
    )  # canonical +Z = native +X (approach)

    T_base_tcp = hande_base_to_canonical_tcp(0.133)
    np.testing.assert_allclose(T_base_tcp[:3, :3], R_HANDE_BASE_CANONICAL_TCP)
    np.testing.assert_allclose(T_base_tcp[:3, 3], [0.133, 0.0, 0.0])

    # Composing the URDF base pose with the native-base -> canonical-TCP
    # transform leaves canonical orientation unchanged and puts the TCP at
    # [0, 0, 0.133].  This is exactly the right-multiplied transform used on
    # sampler output above; adding another rotation would be a double rotate.
    T_canonical_base = np.eye(4)
    T_canonical_base[:3, :3] = R_CANONICAL_HANDE_BASE
    T_canonical_tcp = T_canonical_base @ T_base_tcp
    np.testing.assert_allclose(T_canonical_tcp[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T_canonical_tcp[:3, 3], [0.0, 0.0, 0.133], atol=1e-12)


def test_hande_defaults_come_from_fingertip_and_symmetric_open_joint(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "fingertip": [0.0, 0.0, 0.133],
                "open": {"hande_base_to_left_finger": 0.025},
            }
        ),
        encoding="utf-8",
    )
    offset, width, sources = load_hande_dimensions(config)
    assert offset == 0.133
    assert width == 0.05
    assert sources == {
        "tcp_offset": "config.fingertip[2]",
        "pregrasp_width": "2 * config.open symmetric finger joint",
    }

    offset, width, sources = load_hande_dimensions(
        config,
        tcp_offset_override_m=0.140,
        pregrasp_width_override_m=0.045,
    )
    assert offset == 0.140
    assert width == 0.045
    assert sources == {"tcp_offset": "cli_override", "pregrasp_width": "cli_override"}

    with pytest.raises(ValueError, match="exceeds the configured full opening"):
        load_hande_dimensions(config, pregrasp_width_override_m=0.051)


def test_candidate_npz_has_canonical_keys_and_constant_width(tmp_path: Path) -> None:
    output = tmp_path / "candidates.npz"
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, 3] = [0.1, 0.2, 0.3]
    scores = np.array([0.9, 0.7])
    save_candidates_npz(
        output,
        poses,
        scores,
        0.05,
        object_frame_id="die_demo_v1",
        base_to_tcp_offset_m=0.133,
    )

    with np.load(output, allow_pickle=False) as artifact:
        assert set(artifact.files) == {
            "poses",
            "scores",
            "widths_m",
            "gripper_name",
            "object_frame_id",
            "base_to_tcp_offset_m",
            "frame_convention",
        }
        np.testing.assert_allclose(artifact["poses"], poses)
        np.testing.assert_allclose(artifact["scores"], scores)
        np.testing.assert_allclose(artifact["widths_m"], [0.05, 0.05])
        assert artifact["gripper_name"].item() == "robotiq_hande"
        assert artifact["object_frame_id"].item() == "die_demo_v1"
        assert artifact["base_to_tcp_offset_m"].item() == 0.133
        assert artifact["frame_convention"].item() == (
            "T_object_gripper_tcp; +X closing, +Z approach"
        )

    loaded = load_candidates_npz(output)
    np.testing.assert_allclose(loaded.poses, poses)
    np.testing.assert_allclose(loaded.scores, scores)
    np.testing.assert_allclose(loaded.widths_m, [0.05, 0.05])
