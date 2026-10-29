from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ground_live_grasp_candidate import (  # noqa: E402
    LIVE_OBJECT_POSE_SCHEMA,
    LIVE_POSE_FRAME_CONVENTION,
    _load_live_object_pose,
    _require_matching_object_frame,
)


def live_anchor_payload(center: list[float] | None = None) -> dict:
    return {
        "schema": "demo2gripper.live_supported_object_anchor.v1",
        "status": "accepted",
        "quality": {"accepted": True},
        "center_base_m": center or [0.6, 0.03, 0.05],
    }


def live_pose_payload(anchor_path: Path | None = None) -> dict:
    transform = np.eye(4)
    transform[:3, 3] = [0.6, 0.03, 0.05]
    payload = {
        "schema": LIVE_OBJECT_POSE_SCHEMA,
        "status": "accepted",
        "quality": {"accepted": True},
        "frame_convention": LIVE_POSE_FRAME_CONVENTION,
        "base_frame_id": "base",
        "object_frame_id": "die_mesh_sha256:abc123",
        "T_base_object": transform.tolist(),
    }
    if anchor_path is not None:
        payload["initialization"] = {
            "path": str(anchor_path.resolve()),
            "anchor_center_base_m": [0.6, 0.03, 0.05],
        }
    return payload


def test_load_live_pose_json_requires_accepted_full_rigid_pose(tmp_path: Path) -> None:
    anchor_path = tmp_path / "anchor.json"
    anchor_path.write_text(json.dumps(live_anchor_payload()), encoding="utf-8")
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(live_pose_payload(anchor_path)), encoding="utf-8")

    transform, frame_id, anchor_center, evidence = _load_live_object_pose(
        path, npy_object_frame_id=None, live_object_anchor=anchor_path
    )

    np.testing.assert_allclose(transform[:3, 3], [0.6, 0.03, 0.05])
    assert frame_id == "die_mesh_sha256:abc123"
    np.testing.assert_allclose(anchor_center, [0.6, 0.03, 0.05])
    assert evidence["format"] == "json"


def test_json_pose_keeps_object_origin_distinct_from_exact_live_anchor(
    tmp_path: Path,
) -> None:
    anchor_center = [0.6, 0.03, 0.05]
    anchor_path = tmp_path / "anchor.json"
    anchor_path.write_text(
        json.dumps(live_anchor_payload(anchor_center)), encoding="utf-8"
    )
    payload = live_pose_payload(anchor_path)
    payload["T_base_object"][0][3] += 0.004
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    transform, _, loaded_anchor_center, _ = _load_live_object_pose(
        path, npy_object_frame_id=None, live_object_anchor=anchor_path
    )

    assert transform[0, 3] - loaded_anchor_center[0] == pytest.approx(0.004)
    np.testing.assert_allclose(loaded_anchor_center, anchor_center)


def test_json_pose_requires_live_anchor(tmp_path: Path) -> None:
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(live_pose_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="--live-object-anchor is required"):
        _load_live_object_pose(path, npy_object_frame_id=None)


def test_json_pose_rejects_different_anchor_path_or_center(tmp_path: Path) -> None:
    recorded_anchor = tmp_path / "recorded_anchor.json"
    recorded_anchor.write_text(json.dumps(live_anchor_payload()), encoding="utf-8")
    current_anchor = tmp_path / "current_anchor.json"
    current_anchor.write_text(json.dumps(live_anchor_payload()), encoding="utf-8")
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(live_pose_payload(recorded_anchor)), encoding="utf-8")

    with pytest.raises(ValueError, match="not initialized from this"):
        _load_live_object_pose(
            path, npy_object_frame_id=None, live_object_anchor=current_anchor
        )

    payload = live_pose_payload(current_anchor)
    payload["initialization"]["anchor_center_base_m"] = [0.604, 0.03, 0.05]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="anchor center does not match"):
        _load_live_object_pose(
            path, npy_object_frame_id=None, live_object_anchor=current_anchor
        )


def test_load_live_pose_npy_requires_explicit_object_frame_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "T_base_object.npy"
    np.save(path, np.eye(4))

    with pytest.raises(ValueError, match="--object-frame-id"):
        _load_live_object_pose(path, npy_object_frame_id=None)

    transform, frame_id, anchor_center, evidence = _load_live_object_pose(
        path, npy_object_frame_id="die_mesh_sha256:abc123"
    )
    np.testing.assert_allclose(transform, np.eye(4))
    np.testing.assert_allclose(anchor_center, np.zeros(3))
    assert frame_id == "die_mesh_sha256:abc123"
    assert evidence["format"] == "npy"


def test_live_pose_object_frame_must_match_selected_relation() -> None:
    _require_matching_object_frame("die_v1", "die_v1")
    with pytest.raises(ValueError, match="does not match"):
        _require_matching_object_frame("die_v2", "die_v1")


def test_live_pose_json_rejects_bad_quality_and_nonrigid_transform(
    tmp_path: Path,
) -> None:
    anchor_path = tmp_path / "anchor.json"
    anchor_path.write_text(json.dumps(live_anchor_payload()), encoding="utf-8")
    payload = live_pose_payload(anchor_path)
    payload["quality"]["accepted"] = False
    path = tmp_path / "rejected.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="quality gates"):
        _load_live_object_pose(
            path, npy_object_frame_id=None, live_object_anchor=anchor_path
        )

    payload = live_pose_payload(anchor_path)
    payload["T_base_object"][0][0] = 2.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="orthonormal"):
        _load_live_object_pose(
            path, npy_object_frame_id=None, live_object_anchor=anchor_path
        )
