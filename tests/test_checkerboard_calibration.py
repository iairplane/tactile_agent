from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.checkerboard_calibration import (  # noqa: E402
    CHECKERBOARD_CORNER_ORDER,
    checkerboard_metadata,
    detect_checkerboard,
    normalize_checkerboard_metadata,
    solve_checkerboard_pose,
)
from scripts.solve_eye_to_hand import discover_samples  # noqa: E402


def _checkerboard_image(
    squares_x: int = 9,
    squares_y: int = 12,
    *,
    square_px: int = 60,
    border_px: int = 60,
) -> np.ndarray:
    image = np.full(
        (squares_y * square_px + 2 * border_px, squares_x * square_px + 2 * border_px),
        255,
        dtype=np.uint8,
    )
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                image[
                    border_px + y * square_px : border_px + (y + 1) * square_px,
                    border_px + x * square_px : border_px + (x + 1) * square_px,
                ] = 0
    return image


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def test_checkerboard_metadata_rejects_ambiguous_or_inconsistent_geometry() -> None:
    metadata = checkerboard_metadata(squares_x=9, squares_y=12, square_length_m=0.02)
    board = normalize_checkerboard_metadata(metadata)
    assert board["inner_corners_x"] == 8
    assert board["inner_corners_y"] == 11
    assert board["corner_order"] == CHECKERBOARD_CORNER_ORDER

    same_parity = json.loads(json.dumps(metadata))
    same_parity["board"]["squares_y"] = 11
    same_parity["board"]["inner_corners_y"] = 10
    with pytest.raises(ValueError, match="opposite parity"):
        normalize_checkerboard_metadata(same_parity)

    inconsistent = json.loads(json.dumps(metadata))
    inconsistent["board"]["inner_corners_x"] = 7
    with pytest.raises(ValueError, match="minus one"):
        normalize_checkerboard_metadata(inconsistent)

    unknown = json.loads(json.dumps(metadata))
    unknown["board"]["guessed_size"] = True
    with pytest.raises(ValueError, match="unknown checkerboard board fields"):
        normalize_checkerboard_metadata(unknown)


def test_sector_detector_preserves_physical_corner_order_at_180_degrees() -> None:
    metadata = checkerboard_metadata(squares_x=9, squares_y=12, square_length_m=0.02)
    board = normalize_checkerboard_metadata(metadata)
    image = _checkerboard_image()
    rotated = cv2.rotate(image, cv2.ROTATE_180)
    first = detect_checkerboard(image, board, min_area_fraction=0.03)
    second = detect_checkerboard(rotated, board, min_area_fraction=0.03)

    expected_rotated_points = np.column_stack(
        (
            image.shape[1] - 1 - first.image_points[:, 0],
            image.shape[0] - 1 - first.image_points[:, 1],
        )
    )
    np.testing.assert_allclose(second.image_points, expected_rotated_points, atol=0.08)
    np.testing.assert_array_equal(first.object_points, second.object_points)
    assert int(first.meta[0, 0]) == 2
    assert int(second.meta[0, 0]) == 2

    camera_matrix = np.array(
        [
            [700.0, 0.0, (image.shape[1] - 1) / 2.0],
            [0.0, 700.0, (image.shape[0] - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    first_pose = solve_checkerboard_pose(
        first.object_points, first.image_points, camera_matrix
    ).transform
    second_pose = solve_checkerboard_pose(
        second.object_points, second.image_points, camera_matrix
    ).transform
    camera_half_turn = np.diag([-1.0, -1.0, 1.0])
    np.testing.assert_allclose(
        second_pose[:3, :3], camera_half_turn @ first_pose[:3, :3], atol=5e-4
    )
    np.testing.assert_allclose(second_pose[:3, 3], camera_half_turn @ first_pose[:3, 3], atol=5e-4)


def test_tilted_synthetic_checkerboard_recovers_pose_and_separates_ippe_solutions() -> None:
    squares_x, squares_y = 9, 12
    square_length_m = 0.02
    square_px = 100
    border_px = 100
    metadata = checkerboard_metadata(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_m=square_length_m,
    )
    board = normalize_checkerboard_metadata(metadata)
    source = _checkerboard_image(
        squares_x,
        squares_y,
        square_px=square_px,
        border_px=border_px,
    )
    output_shape = (1000, 1000)
    camera_matrix = np.array([[900.0, 0.0, 499.5], [0.0, 900.0, 499.5], [0.0, 0.0, 1.0]])
    rotation_vector = np.array([0.35, -0.25, 0.12])
    rotation, _ = cv2.Rodrigues(rotation_vector)
    translation = np.array([-0.08, -0.13, 0.58])

    source_corners = np.float32(
        [
            [0, 0],
            [source.shape[1] - 1, 0],
            [source.shape[1] - 1, source.shape[0] - 1],
            [0, source.shape[0] - 1],
        ]
    )
    margin_m = square_length_m
    physical_corners = np.array(
        [
            [-margin_m, -margin_m, 0.0],
            [squares_x * square_length_m + margin_m, -margin_m, 0.0],
            [
                squares_x * square_length_m + margin_m,
                squares_y * square_length_m + margin_m,
                0.0,
            ],
            [-margin_m, squares_y * square_length_m + margin_m, 0.0],
        ]
    )
    projected_corners, _ = cv2.projectPoints(
        physical_corners,
        rotation_vector,
        translation,
        camera_matrix,
        np.zeros(5),
    )
    homography = cv2.getPerspectiveTransform(
        source_corners, projected_corners.reshape(4, 2).astype(np.float32)
    )
    rendered = cv2.warpPerspective(source, homography, output_shape, borderValue=220)
    detection = detect_checkerboard(rendered, board, min_area_fraction=0.03)
    estimate = solve_checkerboard_pose(
        detection.object_points, detection.image_points, camera_matrix
    )

    expected_rotation = rotation @ np.diag([-1.0, -1.0, 1.0])
    expected_translation = (
        rotation
        @ np.array([(squares_x - 1) * square_length_m, (squares_y - 1) * square_length_m, 0.0])
        + translation
    )
    assert np.linalg.norm(estimate.transform[:3, 3] - expected_translation) < 0.001
    assert _rotation_error_deg(estimate.transform[:3, :3], expected_rotation) < 0.1
    assert estimate.reprojection_rms_px < 0.2
    assert estimate.second_best_reprojection_rms_px is not None
    assert estimate.second_best_reprojection_rms_px - estimate.best_ippe_reprojection_rms_px > 1.0


def _write_verified_sample(
    directory: Path, metadata: dict[str, object], *, session_id: str = "checker-session"
) -> None:
    directory.mkdir()
    identity = np.eye(4)
    np.save(directory / "T_base_flange.npy", identity, allow_pickle=False)
    np.save(directory / "T_camera_board.npy", identity, allow_pickle=False)
    np.save(
        directory / "K.npy",
        np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]),
        allow_pickle=False,
    )
    np.save(directory / "distortion.npy", np.zeros(5), allow_pickle=False)
    (directory / "board_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    board = normalize_checkerboard_metadata(metadata)
    manifest = {
        "schema": "demo2gripper.eye_to_hand_sample.v1",
        "session_id": session_id,
        "valid_for_handeye_calibration": True,
        "files": {
            "camera_matrix": "K.npy",
            "distortion_coefficients": "distortion.npy",
            "T_camera_board": "T_camera_board.npy",
            "T_base_flange": "T_base_flange.npy",
            "board_metadata_copy": "board_metadata.json",
        },
        "transforms": {
            "T_camera_board": "T_camera_board.npy",
            "T_base_flange": "T_base_flange.npy",
        },
        "camera": {
            "device": {"serial": "camera-1"},
            "distortion_model": "brown_conrady",
        },
        "board": board,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_solver_provenance_accepts_matching_checkerboards_and_rejects_geometry_change(
    tmp_path: Path,
) -> None:
    metadata = checkerboard_metadata(squares_x=9, squares_y=12, square_length_m=0.02)
    _write_verified_sample(tmp_path / "000", metadata)
    _write_verified_sample(tmp_path / "001", metadata)
    assert len(discover_samples(tmp_path, 3.0)) == 2

    changed = checkerboard_metadata(squares_x=9, squares_y=12, square_length_m=0.021)
    (tmp_path / "001" / "board_metadata.json").write_text(json.dumps(changed), encoding="utf-8")
    manifest_path = tmp_path / "001" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["board"] = normalize_checkerboard_metadata(changed)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="board schema/geometry"):
        discover_samples(tmp_path, 3.0)
