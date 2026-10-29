#!/usr/bin/env python3
"""Strict metadata and canonical corner detection for hand-eye checkerboards.

The coordinate frame is defined by OpenCV's sector-based detector rather than by
the image-space top-left corner.  Requiring opposite parity for the two square
counts lets the black/white ``meta`` signature disambiguate a 180 degree board
rotation without adding a fiducial marker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CHECKERBOARD_SCHEMA = "demo2gripper.checkerboard_board.v1"
CHECKERBOARD_SCHEMA_VERSION = 1
CHECKERBOARD_TYPE = "opencv_checkerboard"
CHECKERBOARD_CORNER_ORDER = "findChessboardCornersSBWithMeta:meta2-origin:row-major"
CHECKERBOARD_ORIGIN_META = 2


@dataclass(frozen=True)
class CheckerboardDetection:
    """A complete checkerboard detection in canonical physical-board order."""

    image_points: np.ndarray
    object_points: np.ndarray
    meta: np.ndarray
    area_fraction: float


@dataclass(frozen=True)
class CheckerboardPoseEstimate:
    """Planar PnP result using already-undistorted image points."""

    transform: np.ndarray
    reprojection_rms_px: float
    reprojection_max_px: float
    positive_ippe_candidates: int
    best_ippe_reprojection_rms_px: float
    second_best_reprojection_rms_px: float | None
    second_best_rotation_difference_deg: float | None
    second_best_translation_difference_m: float | None


def _integer(value: Any, name: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"checkerboard field {name!r} must be an integer >= {minimum}")
    return value


def _positive_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"checkerboard field {name!r} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"checkerboard field {name!r} must be finite and positive")
    return result


def normalize_checkerboard_metadata(
    metadata: dict[str, Any], *, source: str | Path = "checkerboard metadata"
) -> dict[str, Any]:
    """Validate v1 metadata and return its canonical geometry dictionary."""
    label = str(source)
    if not isinstance(metadata, dict):
        raise ValueError(f"checkerboard metadata root must be an object: {label}")
    allowed_root = {"schema", "schema_version", "type", "board"}
    unknown_root = sorted(set(metadata) - allowed_root)
    if unknown_root:
        raise ValueError(f"unknown checkerboard metadata fields in {label}: {unknown_root}")
    if metadata.get("schema") != CHECKERBOARD_SCHEMA:
        raise ValueError(f"checkerboard schema must be {CHECKERBOARD_SCHEMA!r} in {label}")
    if metadata.get("schema_version") != CHECKERBOARD_SCHEMA_VERSION:
        raise ValueError(
            f"checkerboard schema_version must be {CHECKERBOARD_SCHEMA_VERSION} in {label}"
        )
    if metadata.get("type") != CHECKERBOARD_TYPE:
        raise ValueError(f"checkerboard type must be {CHECKERBOARD_TYPE!r} in {label}")
    board = metadata.get("board")
    if not isinstance(board, dict):
        raise ValueError(f"checkerboard metadata must contain a board object: {label}")
    required_board = {
        "squares_x",
        "squares_y",
        "inner_corners_x",
        "inner_corners_y",
        "square_length_m",
        "corner_order",
        "origin_corner_meta",
    }
    missing = sorted(required_board - set(board))
    unknown = sorted(set(board) - required_board)
    if missing:
        raise ValueError(f"missing checkerboard board fields in {label}: {missing}")
    if unknown:
        raise ValueError(f"unknown checkerboard board fields in {label}: {unknown}")

    squares_x = _integer(board["squares_x"], "squares_x", 3)
    squares_y = _integer(board["squares_y"], "squares_y", 3)
    inner_x = _integer(board["inner_corners_x"], "inner_corners_x", 2)
    inner_y = _integer(board["inner_corners_y"], "inner_corners_y", 2)
    if inner_x != squares_x - 1 or inner_y != squares_y - 1:
        raise ValueError(
            f"checkerboard inner corner counts must equal square counts minus one in {label}"
        )
    if squares_x % 2 == squares_y % 2:
        raise ValueError(
            "checkerboard squares_x and squares_y must have opposite parity so the "
            f"black/white meta signature resolves 180-degree corner-order ambiguity: {label}"
        )
    if board["corner_order"] != CHECKERBOARD_CORNER_ORDER:
        raise ValueError(
            f"checkerboard corner_order must be {CHECKERBOARD_CORNER_ORDER!r} in {label}"
        )
    if board["origin_corner_meta"] != CHECKERBOARD_ORIGIN_META:
        raise ValueError(
            f"checkerboard origin_corner_meta must be {CHECKERBOARD_ORIGIN_META} in {label}"
        )
    return {
        "schema": CHECKERBOARD_SCHEMA,
        "schema_version": CHECKERBOARD_SCHEMA_VERSION,
        "type": CHECKERBOARD_TYPE,
        "squares_x": squares_x,
        "squares_y": squares_y,
        "inner_corners_x": inner_x,
        "inner_corners_y": inner_y,
        "square_length_m": _positive_float(board["square_length_m"], "square_length_m"),
        "corner_order": CHECKERBOARD_CORNER_ORDER,
        "origin_corner_meta": CHECKERBOARD_ORIGIN_META,
    }


def checkerboard_metadata(
    *, squares_x: int, squares_y: int, square_length_m: float
) -> dict[str, Any]:
    """Build and self-validate a v1 checkerboard metadata document."""
    metadata: dict[str, Any] = {
        "schema": CHECKERBOARD_SCHEMA,
        "schema_version": CHECKERBOARD_SCHEMA_VERSION,
        "type": CHECKERBOARD_TYPE,
        "board": {
            "squares_x": squares_x,
            "squares_y": squares_y,
            "inner_corners_x": squares_x - 1,
            "inner_corners_y": squares_y - 1,
            "square_length_m": square_length_m,
            "corner_order": CHECKERBOARD_CORNER_ORDER,
            "origin_corner_meta": CHECKERBOARD_ORIGIN_META,
        },
    }
    normalize_checkerboard_metadata(metadata)
    return metadata


def checkerboard_object_points(board: dict[str, Any]) -> np.ndarray:
    """Return row-major points matching the canonical SB detector output."""
    inner_x = int(board["inner_corners_x"])
    inner_y = int(board["inner_corners_y"])
    square_length_m = float(board["square_length_m"])
    grid = np.zeros((inner_y, inner_x, 3), dtype=np.float64)
    grid[:, :, :2] = np.mgrid[0:inner_x, 0:inner_y].T
    return grid.reshape(-1, 3) * square_length_m


def _validate_meta_signature(meta: np.ndarray, board: dict[str, Any]) -> np.ndarray:
    inner_x = int(board["inner_corners_x"])
    inner_y = int(board["inner_corners_y"])
    values = np.asarray(meta, dtype=np.uint8)
    if values.shape != (inner_y, inner_x):
        raise RuntimeError(
            f"checkerboard detector meta shape {values.shape} does not match ({inner_y}, {inner_x})"
        )
    expected_core = np.fromfunction(
        lambda y, x: np.where((x + y) % 2 == 0, 2, 1),
        (inner_y - 1, inner_x - 1),
        dtype=int,
    ).astype(np.uint8)
    if not np.array_equal(values[:-1, :-1], expected_core):
        raise RuntimeError(
            "checkerboard detector returned a non-canonical black/white meta signature"
        )
    if np.any(values[-1, :] != 0) or np.any(values[:, -1] != 0):
        raise RuntimeError("checkerboard detector returned an unexpected meta boundary")
    if int(values[0, 0]) != int(board["origin_corner_meta"]):
        raise RuntimeError("checkerboard detector did not resolve the configured physical origin")
    return values.copy()


def detect_checkerboard(
    image: np.ndarray,
    board: dict[str, Any],
    *,
    min_area_fraction: float,
) -> CheckerboardDetection:
    """Detect every inner corner and preserve physical order across 180-degree turns."""
    if not math.isfinite(min_area_fraction) or not 0.0 < min_area_fraction <= 1.0:
        raise ValueError("min_area_fraction must be finite and in (0, 1]")
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    elif array.ndim == 2:
        gray = array
    else:
        raise ValueError(f"checkerboard image must be grayscale or RGB, got {array.shape}")
    if gray.dtype != np.uint8:
        raise ValueError(f"checkerboard image must have dtype uint8, got {gray.dtype}")

    pattern_size = (int(board["inner_corners_x"]), int(board["inner_corners_y"]))
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners, meta = cv2.findChessboardCornersSBWithMeta(gray, pattern_size, flags)
    expected_count = pattern_size[0] * pattern_size[1]
    detected_count = 0 if corners is None else int(np.asarray(corners).reshape(-1, 2).shape[0])
    if not found or corners is None or meta is None or detected_count != expected_count:
        raise RuntimeError(
            "complete checkerboard not detected: "
            f"found={bool(found)}, corners={detected_count}, require={expected_count}"
        )
    image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(image_points).all():
        raise RuntimeError("checkerboard detector returned non-finite corners")
    canonical_meta = _validate_meta_signature(meta, board)

    image_area = float(gray.shape[0] * gray.shape[1])
    hull_area = float(cv2.contourArea(cv2.convexHull(image_points.astype(np.float32))))
    area_fraction = hull_area / image_area
    if not math.isfinite(area_fraction) or area_fraction < min_area_fraction:
        raise RuntimeError(
            f"checkerboard inner-corner area fraction {area_fraction:.4f} is below "
            f"{min_area_fraction:.4f}"
        )
    return CheckerboardDetection(
        image_points=image_points,
        object_points=checkerboard_object_points(board),
        meta=canonical_meta,
        area_fraction=area_fraction,
    )


def solve_checkerboard_pose(
    object_points: np.ndarray,
    undistorted_image_points: np.ndarray,
    camera_matrix: np.ndarray,
) -> CheckerboardPoseEstimate:
    """Solve ``T_camera_board`` and reprojection error for canonical corners."""
    object_array = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_array = np.asarray(undistorted_image_points, dtype=np.float64).reshape(-1, 2)
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if len(object_array) < 4 or len(object_array) != len(image_array):
        raise ValueError("checkerboard PnP needs matching object/image arrays with >= 4 points")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("checkerboard camera matrix must be a finite 3x3 array")
    if not np.isfinite(object_array).all() or not np.isfinite(image_array).all():
        raise ValueError("checkerboard PnP points must be finite")
    zero_distortion = np.zeros(5, dtype=np.float64)
    result = cv2.solvePnPGeneric(
        object_array,
        image_array,
        matrix,
        zero_distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    success, rotation_vectors, translation_vectors = result[:3]
    if not success or not rotation_vectors:
        raise RuntimeError("cv2.solvePnPGeneric IPPE failed for the complete checkerboard")

    candidates: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]] = []
    for rotation_vector, translation_vector in zip(
        rotation_vectors, translation_vectors, strict=True
    ):
        rotation, _ = cv2.Rodrigues(rotation_vector)
        translation = np.asarray(translation_vector, dtype=np.float64).reshape(3)
        points_camera = (rotation @ object_array.T).T + translation
        if np.min(points_camera[:, 2]) <= 0.0:
            continue
        projected, _ = cv2.projectPoints(
            object_array,
            rotation_vector,
            translation_vector,
            matrix,
            zero_distortion,
        )
        errors = np.linalg.norm(projected.reshape(-1, 2) - image_array, axis=1)
        rms_error = float(np.sqrt(np.mean(np.square(errors))))
        max_error = float(np.max(errors))
        if math.isfinite(rms_error) and math.isfinite(max_error):
            candidates.append(
                (
                    rms_error,
                    max_error,
                    np.asarray(rotation_vector, dtype=np.float64).reshape(3, 1),
                    np.asarray(translation_vector, dtype=np.float64).reshape(3, 1),
                    rotation,
                )
            )
    if not candidates:
        raise RuntimeError("checkerboard IPPE produced no all-points-in-front pose")
    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
    initial_rms, _initial_max, rotation_vector, translation_vector, initial_rotation = candidates[0]

    second_rms: float | None = None
    second_rotation_difference_deg: float | None = None
    second_translation_difference_m: float | None = None
    if len(candidates) >= 2:
        second_rms = candidates[1][0]
        relative_rotation = initial_rotation.T @ candidates[1][4]
        cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
        second_rotation_difference_deg = math.degrees(math.acos(cosine))
        second_translation_difference_m = float(
            np.linalg.norm(candidates[0][3].reshape(3) - candidates[1][3].reshape(3))
        )

    rotation_vector, translation_vector = cv2.solvePnPRefineLM(
        object_array,
        image_array,
        matrix,
        zero_distortion,
        rotation_vector,
        translation_vector,
    )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation_vector, dtype=np.float64).reshape(3)
    points_camera = (rotation @ object_array.T).T + transform[:3, 3]
    if np.min(points_camera[:, 2]) <= 0.0:
        raise RuntimeError("estimated checkerboard is not fully in front of the camera")
    projected, _ = cv2.projectPoints(
        object_array,
        rotation_vector,
        translation_vector,
        matrix,
        zero_distortion,
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_array, axis=1)
    rms_error = float(np.sqrt(np.mean(np.square(errors))))
    max_error = float(np.max(errors))
    if not math.isfinite(rms_error) or not math.isfinite(max_error):
        raise RuntimeError("checkerboard PnP produced non-finite reprojection error")
    return CheckerboardPoseEstimate(
        transform=transform,
        reprojection_rms_px=rms_error,
        reprojection_max_px=max_error,
        positive_ippe_candidates=len(candidates),
        best_ippe_reprojection_rms_px=initial_rms,
        second_best_reprojection_rms_px=second_rms,
        second_best_rotation_difference_deg=second_rotation_difference_deg,
        second_best_translation_difference_m=second_translation_difference_m,
    )
