#!/usr/bin/env python3
"""Solve fixed-camera eye-to-hand calibration from paired robot and board poses.

Each immediate child directory of ``SAMPLES_DIR`` must contain
``T_base_flange.npy`` and ``T_camera_board.npy``.  All transforms follow the
repository convention ``T_parent_child`` and use metres.  The solved equation is

    T_base_flange @ T_flange_board = T_base_camera @ T_camera_board.

Outputs are only published when every quality check passes.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from checkerboard_calibration import (
        CHECKERBOARD_SCHEMA,
        CHECKERBOARD_SCHEMA_VERSION,
        normalize_checkerboard_metadata,
    )
except ModuleNotFoundError:  # Imported as scripts.solve_eye_to_hand in CPU tests.
    from scripts.checkerboard_calibration import (
        CHECKERBOARD_SCHEMA,
        CHECKERBOARD_SCHEMA_VERSION,
        normalize_checkerboard_metadata,
    )

ROBOT_WORLD_METHOD_CONSTANTS = {
    "shah": "CALIB_ROBOT_WORLD_HAND_EYE_SHAH",
    "li": "CALIB_ROBOT_WORLD_HAND_EYE_LI",
}
SAMPLE_MANIFEST_SCHEMA = "demo2gripper.eye_to_hand_sample.v1"
SUPPORTED_BOARD_SCHEMAS = {
    "demo2gripper.charuco_board.v1": 1,
    "opencv_charuco_board.v1": 1,
    CHECKERBOARD_SCHEMA: CHECKERBOARD_SCHEMA_VERSION,
}


@dataclass(frozen=True)
class SampleProvenance:
    manifest_path: Path
    manifest_schema: str
    session_id: str
    camera_serial: str
    camera_matrix: np.ndarray
    distortion_model: str
    distortion_coefficients: np.ndarray
    board_metadata_path: Path
    board: dict[str, Any]


@dataclass(frozen=True)
class Sample:
    index: int
    name: str
    directory: Path
    T_base_flange: np.ndarray
    T_camera_board: np.ndarray
    provenance: SampleProvenance | None = None


@dataclass(frozen=True)
class Solution:
    method: str
    T_base_camera: np.ndarray
    T_flange_board: np.ndarray
    raw_T_base_camera: np.ndarray
    raw_T_flange_board: np.ndarray
    translation_refit: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "samples_dir",
        type=Path,
        help="Directory whose immediate child directories contain the paired transforms",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("calibration"))
    parser.add_argument(
        "--validation-indices",
        nargs="+",
        default=[],
        metavar="INDEX",
        help=(
            "Zero-based indices in the naturally sorted sample list to hold out. "
            "Tokens may also be comma-separated, for example: 1 4,7"
        ),
    )
    parser.add_argument(
        "--primary-method",
        choices=tuple(ROBOT_WORLD_METHOD_CONSTANTS),
        default="shah",
        help="Robot-world/hand-eye solution written to the two .npy outputs",
    )
    parser.add_argument(
        "--allow-unverified-samples",
        action="store_true",
        help=(
            "Allow sample directories without manifest.json for synthetic/offline tests. "
            "A manifest that is present is always validated."
        ),
    )
    parser.add_argument(
        "--no-park-cross-check",
        action="store_true",
        help="Disable the independent Park eye-to-hand cross-check",
    )
    parser.add_argument("--min-training-samples", type=int, default=6)
    parser.add_argument(
        "--min-validation-samples",
        type=int,
        default=3,
        help="Minimum number of held-out samples required for acceptance; set 0 explicitly to skip",
    )
    parser.add_argument(
        "--min-relative-rotation-deg",
        type=float,
        default=10.0,
        help="Minimum relative angle for a motion to count toward axis diversity",
    )
    parser.add_argument(
        "--min-axis-separation-deg",
        type=float,
        default=20.0,
        help="Required separation between two significant relative rotation axes",
    )
    parser.add_argument(
        "--min-translation-span-mm",
        type=float,
        default=50.0,
        help="Required maximum pairwise translation span in each input pose stream",
    )
    parser.add_argument(
        "--max-input-translation-m",
        type=float,
        default=3.0,
        help="Reject inputs beyond this norm, commonly indicating millimetres were supplied",
    )
    parser.add_argument(
        "--max-train-translation-mm",
        type=float,
        default=3.0,
        help="Maximum allowed per-sample training closure translation error",
    )
    parser.add_argument(
        "--max-train-rotation-deg",
        type=float,
        default=0.5,
        help="Maximum allowed per-sample training closure rotation error",
    )
    parser.add_argument(
        "--max-validation-translation-mm",
        type=float,
        default=5.0,
        help="Maximum allowed per-sample held-out closure translation error",
    )
    parser.add_argument(
        "--max-validation-rotation-deg",
        type=float,
        default=1.0,
        help="Maximum allowed per-sample held-out closure rotation error",
    )
    parser.add_argument(
        "--max-method-translation-mm",
        type=float,
        default=5.0,
        help="Maximum SHAH-versus-LI disagreement for either solved translation",
    )
    parser.add_argument(
        "--max-method-rotation-deg",
        type=float,
        default=0.5,
        help="Maximum SHAH-versus-LI disagreement for either solved rotation",
    )
    parser.add_argument(
        "--max-park-translation-mm",
        type=float,
        default=5.0,
        help="Maximum primary-versus-Park T_base_camera translation disagreement",
    )
    parser.add_argument(
        "--max-park-rotation-deg",
        type=float,
        default=0.5,
        help="Maximum primary-versus-Park T_base_camera rotation disagreement",
    )
    return parser.parse_args()


def check_opencv_calibration_capabilities() -> None:
    required_callables = ("calibrateRobotWorldHandEye", "calibrateHandEye")
    required_constants = (*ROBOT_WORLD_METHOD_CONSTANTS.values(), "CALIB_HAND_EYE_PARK")
    missing = [
        f"cv2.{name}" for name in required_callables if not callable(getattr(cv2, name, None))
    ]
    missing.extend(f"cv2.{name}" for name in required_constants if not hasattr(cv2, name))
    if missing:
        raise RuntimeError(
            "this OpenCV build cannot solve eye-to-hand calibration; missing "
            + ", ".join(missing)
            + f" (OpenCV {getattr(cv2, '__version__', 'unknown')}). Use an environment "
            "whose OpenCV build provides both hand-eye calibration APIs."
        )


def _natural_key(path: Path) -> list[tuple[int, int | str]]:
    parts = re.split(r"(\d+)", path.name.casefold())
    return [(0, int(part)) if part.isdigit() else (1, part) for part in parts]


def _validate_transform(
    value: np.ndarray,
    *,
    name: str,
    max_translation_m: float | None,
    atol: float = 1e-6,
) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}")
    if not np.isfinite(transform).all():
        raise ValueError(f"{name} contains NaN or infinity")
    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(transform[3], expected_last_row, rtol=0.0, atol=atol):
        raise ValueError(f"{name} has an invalid homogeneous last row: {transform[3].tolist()}")
    rotation = transform[:3, :3]
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > atol or abs(determinant - 1.0) > atol:
        raise ValueError(
            f"{name} rotation is not in SO(3): max orthogonality error "
            f"{orthogonality_error:.3g}, determinant {determinant:.12g}"
        )
    translation_norm = float(np.linalg.norm(transform[:3, 3]))
    if max_translation_m is not None and translation_norm > max_translation_m:
        raise ValueError(
            f"{name} translation norm {translation_norm:.6g} m exceeds "
            f"{max_translation_m:.6g} m; verify that translations are in metres"
        )
    return transform.copy()


def _load_transform(path: Path, *, name: str, max_translation_m: float) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"could not load {name} from {path}: {exc}") from exc
    return _validate_transform(value, name=f"{name} ({path})", max_translation_m=max_translation_m)


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read {description} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} root must be an object: {path}")
    return value


def _required_object(value: Any, *, name: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object in {path}")
    return value


def _required_nonempty_string(value: Any, *, name: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string in {path}")
    return value


def _required_filename(value: Any, *, field: str, expected: str, manifest_path: Path) -> str:
    filename = _required_nonempty_string(value, name=field, path=manifest_path)
    if filename != expected:
        raise ValueError(f"{field} must be {expected!r} in {manifest_path}, got {filename!r}")
    return filename


def _load_numeric_array(path: Path, *, description: str) -> np.ndarray:
    try:
        value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"could not load {description} from {path}: {exc}") from exc
    if not np.isfinite(value).all():
        raise ValueError(f"{description} contains NaN or infinity: {path}")
    return value


def _board_integer(board: dict[str, Any], name: str, path: Path, minimum: int) -> int:
    value = board.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"board field {name!r} must be an integer >= {minimum} in {path}")
    return value


def _board_positive_float(board: dict[str, Any], name: str, path: Path) -> float:
    value = board.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"board field {name!r} must be numeric in {path}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"board field {name!r} must be finite and positive in {path}")
    return result


def _normalize_board_metadata(metadata: dict[str, Any], path: Path) -> dict[str, Any]:
    if metadata.get("schema") == CHECKERBOARD_SCHEMA:
        return normalize_checkerboard_metadata(metadata, source=path)
    if metadata.get("schema") == "demo2gripper.charuco_board.v1":
        schema = "demo2gripper.charuco_board.v1"
        schema_version = metadata.get("schema_version")
        board = _required_object(metadata.get("board"), name="board", path=path)
    elif metadata.get("type") == "opencv_charuco_board":
        schema = "opencv_charuco_board.v1"
        schema_version = metadata.get("schema_version")
        board = _required_object(metadata.get("board"), name="board", path=path)
    else:
        raise ValueError(
            f"unsupported board metadata schema in {path}; expected one of "
            + ", ".join(sorted(SUPPORTED_BOARD_SCHEMAS))
        )
    if schema_version != SUPPORTED_BOARD_SCHEMAS[schema]:
        raise ValueError(
            f"unsupported board metadata schema_version {schema_version!r} in {path}; "
            f"expected {SUPPORTED_BOARD_SCHEMAS[schema]}"
        )

    square_length_m = _board_positive_float(board, "square_length_m", path)
    marker_length_m = _board_positive_float(board, "marker_length_m", path)
    if marker_length_m >= square_length_m:
        raise ValueError(f"board marker_length_m must be smaller than square_length_m in {path}")
    dictionary = _required_nonempty_string(
        board.get("dictionary"), name="board.dictionary", path=path
    )
    legacy_pattern = board.get("legacy_pattern", False)
    if not isinstance(legacy_pattern, bool):
        raise ValueError(f"board field 'legacy_pattern' must be boolean in {path}")

    normalized: dict[str, Any] = {
        "schema": schema,
        "schema_version": int(schema_version),
        "squares_x": _board_integer(board, "squares_x", path, 2),
        "squares_y": _board_integer(board, "squares_y", path, 2),
        "square_length_m": square_length_m,
        "marker_length_m": marker_length_m,
        "dictionary": dictionary,
        "legacy_pattern": legacy_pattern,
    }
    if "dictionary_id" in board:
        normalized["dictionary_id"] = _board_integer(board, "dictionary_id", path, 0)
    if "charuco_corner_count" in board:
        normalized["charuco_corner_count"] = _board_integer(board, "charuco_corner_count", path, 1)
    if "marker_ids" in board:
        marker_ids = board["marker_ids"]
        if (
            not isinstance(marker_ids, list)
            or not marker_ids
            or any(
                not isinstance(marker_id, int) or isinstance(marker_id, bool) or marker_id < 0
                for marker_id in marker_ids
            )
            or len(set(marker_ids)) != len(marker_ids)
        ):
            raise ValueError(
                f"board field 'marker_ids' must be a non-empty list of unique "
                f"non-negative integers in {path}"
            )
        normalized["marker_ids"] = marker_ids.copy()
    for name in ("width_m", "height_m", "outer_width_m", "outer_height_m"):
        if name in board:
            normalized[name] = _board_positive_float(board, name, path)
    return normalized


def _values_match(first: Any, second: Any) -> bool:
    if isinstance(first, float) and isinstance(second, (int, float)):
        return math.isclose(first, float(second), rel_tol=1e-12, abs_tol=1e-12)
    return first == second


def _validate_manifest_board_copy(
    manifest_board: dict[str, Any], board: dict[str, Any], manifest_path: Path
) -> None:
    if board.get("schema") == CHECKERBOARD_SCHEMA:
        required_names = (
            "schema",
            "schema_version",
            "type",
            "squares_x",
            "squares_y",
            "inner_corners_x",
            "inner_corners_y",
            "square_length_m",
            "corner_order",
            "origin_corner_meta",
        )
    else:
        required_names = (
            "squares_x",
            "squares_y",
            "square_length_m",
            "marker_length_m",
            "dictionary",
            "legacy_pattern",
        )
    for name in required_names:
        if name not in manifest_board:
            raise ValueError(f"manifest board.{name} is missing in {manifest_path}")
        if not _values_match(board[name], manifest_board[name]):
            raise ValueError(
                f"manifest board.{name} does not match board_metadata.json in {manifest_path}"
            )
    if board.get("schema") != CHECKERBOARD_SCHEMA and "dictionary_id" in board:
        if "dictionary_id" not in manifest_board or not _values_match(
            board["dictionary_id"], manifest_board["dictionary_id"]
        ):
            raise ValueError(
                f"manifest board.dictionary_id does not match board_metadata.json "
                f"in {manifest_path}"
            )


def _load_sample_provenance(directory: Path) -> SampleProvenance:
    manifest_path = directory / "manifest.json"
    manifest = _load_json_object(manifest_path, description="sample manifest")
    if manifest.get("schema") != SAMPLE_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported sample manifest schema {manifest.get('schema')!r} in {manifest_path}; "
            f"expected {SAMPLE_MANIFEST_SCHEMA!r}"
        )
    if manifest.get("valid_for_handeye_calibration") is not True:
        raise ValueError(
            f"sample manifest valid_for_handeye_calibration must be true in {manifest_path}"
        )
    session_id = _required_nonempty_string(
        manifest.get("session_id"), name="session_id", path=manifest_path
    )

    files = _required_object(manifest.get("files"), name="files", path=manifest_path)
    required_files = {
        "camera_matrix": "K.npy",
        "distortion_coefficients": "distortion.npy",
        "T_camera_board": "T_camera_board.npy",
        "T_base_flange": "T_base_flange.npy",
        "board_metadata_copy": "board_metadata.json",
    }
    for field, expected in required_files.items():
        _required_filename(
            files.get(field),
            field=f"files.{field}",
            expected=expected,
            manifest_path=manifest_path,
        )

    transforms = _required_object(manifest.get("transforms"), name="transforms", path=manifest_path)
    for field in ("T_camera_board", "T_base_flange"):
        expected = required_files[field]
        _required_filename(
            transforms.get(field),
            field=f"transforms.{field}",
            expected=expected,
            manifest_path=manifest_path,
        )

    camera = _required_object(manifest.get("camera"), name="camera", path=manifest_path)
    device = _required_object(camera.get("device"), name="camera.device", path=manifest_path)
    camera_serial = _required_nonempty_string(
        device.get("serial"), name="camera.device.serial", path=manifest_path
    )
    distortion_model = _required_nonempty_string(
        camera.get("distortion_model"), name="camera.distortion_model", path=manifest_path
    )

    camera_matrix = _load_numeric_array(directory / "K.npy", description="camera matrix")
    if camera_matrix.shape != (3, 3):
        matrix_path = directory / "K.npy"
        raise ValueError(
            f"camera matrix must have shape (3, 3), got {camera_matrix.shape}: {matrix_path}"
        )
    if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
        raise ValueError(f"camera matrix focal lengths must be positive: {directory / 'K.npy'}")
    if not np.allclose(camera_matrix[2], np.array([0.0, 0.0, 1.0]), rtol=0.0, atol=1e-12):
        raise ValueError(f"camera matrix has an invalid last row: {directory / 'K.npy'}")
    distortion = _load_numeric_array(
        directory / "distortion.npy", description="distortion coefficients"
    ).reshape(-1)
    if distortion.size < 4:
        distortion_path = directory / "distortion.npy"
        raise ValueError(
            f"distortion coefficients must contain at least 4 values: {distortion_path}"
        )

    board_metadata_path = directory / "board_metadata.json"
    board_metadata = _load_json_object(board_metadata_path, description="board metadata")
    board = _normalize_board_metadata(board_metadata, board_metadata_path)
    manifest_board = _required_object(manifest.get("board"), name="board", path=manifest_path)
    _validate_manifest_board_copy(manifest_board, board, manifest_path)

    return SampleProvenance(
        manifest_path=manifest_path,
        manifest_schema=SAMPLE_MANIFEST_SCHEMA,
        session_id=session_id,
        camera_serial=camera_serial,
        camera_matrix=camera_matrix,
        distortion_model=distortion_model,
        distortion_coefficients=distortion,
        board_metadata_path=board_metadata_path,
        board=board,
    )


def _compare_verified_sample_provenance(
    reference: SampleProvenance, candidate: SampleProvenance
) -> None:
    comparisons = (
        ("session_id", reference.session_id, candidate.session_id),
        ("camera serial", reference.camera_serial, candidate.camera_serial),
        ("distortion model", reference.distortion_model, candidate.distortion_model),
    )
    for label, expected, actual in comparisons:
        if actual != expected:
            raise ValueError(
                f"sample provenance mismatch for {label}: {candidate.manifest_path} has "
                f"{actual!r}, expected {expected!r} from {reference.manifest_path}"
            )
    for label, expected, actual in (
        ("camera matrix", reference.camera_matrix, candidate.camera_matrix),
        (
            "distortion coefficients",
            reference.distortion_coefficients,
            candidate.distortion_coefficients,
        ),
    ):
        if expected.shape != actual.shape or not np.allclose(
            expected, actual, rtol=1e-9, atol=1e-12
        ):
            raise ValueError(
                f"sample provenance mismatch for {label}: {candidate.manifest_path} "
                f"differs from {reference.manifest_path}"
            )
    if candidate.board.keys() != reference.board.keys() or any(
        not _values_match(reference.board[name], candidate.board[name]) for name in reference.board
    ):
        raise ValueError(
            f"sample provenance mismatch for board schema/geometry: "
            f"{candidate.board_metadata_path} differs from {reference.board_metadata_path}"
        )


def discover_samples(
    samples_dir: Path,
    max_translation_m: float,
    *,
    allow_unverified_samples: bool = False,
) -> list[Sample]:
    if not samples_dir.is_dir():
        raise ValueError(f"samples directory does not exist or is not a directory: {samples_dir}")
    pairs: list[tuple[Path, Path, Path]] = []
    for directory in sorted(
        (entry for entry in samples_dir.iterdir() if entry.is_dir()), key=_natural_key
    ):
        base_path = directory / "T_base_flange.npy"
        camera_path = directory / "T_camera_board.npy"
        if not base_path.exists() and not camera_path.exists():
            continue
        if not base_path.is_file() or not camera_path.is_file():
            missing = [path.name for path in (base_path, camera_path) if not path.is_file()]
            raise ValueError(f"sample {directory.name!r} is incomplete; missing {missing}")
        pairs.append((directory, base_path, camera_path))
    if not pairs:
        raise ValueError(
            f"no paired T_base_flange.npy and T_camera_board.npy files found under {samples_dir}"
        )

    samples: list[Sample] = []
    for index, (directory, base_path, camera_path) in enumerate(pairs):
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            provenance = _load_sample_provenance(directory)
        elif allow_unverified_samples:
            provenance = None
        else:
            raise ValueError(
                f"sample {directory.name!r} has no manifest.json; real calibration samples "
                "must be verified. Use --allow-unverified-samples only for synthetic/offline tests"
            )
        samples.append(
            Sample(
                index=index,
                name=directory.name,
                directory=directory,
                T_base_flange=_load_transform(
                    base_path,
                    name="T_base_flange",
                    max_translation_m=max_translation_m,
                ),
                T_camera_board=_load_transform(
                    camera_path,
                    name="T_camera_board",
                    max_translation_m=max_translation_m,
                ),
                provenance=provenance,
            )
        )
    verified = [sample.provenance for sample in samples if sample.provenance is not None]
    if verified and len(verified) != len(samples):
        raise ValueError(
            "verified manifest samples and unverified samples cannot be mixed, even with "
            "--allow-unverified-samples"
        )
    if verified:
        reference = verified[0]
        for candidate in verified[1:]:
            _compare_verified_sample_provenance(reference, candidate)
    return samples


def parse_validation_indices(tokens: Sequence[str], sample_count: int) -> set[int]:
    result: set[int] = set()
    for token in tokens:
        for part in token.split(","):
            part = part.strip()
            if not part:
                raise ValueError("--validation-indices contains an empty index")
            try:
                index = int(part)
            except ValueError as exc:
                raise ValueError(f"invalid validation index {part!r}") from exc
            if index < 0 or index >= sample_count:
                raise ValueError(f"validation index {index} is outside [0, {sample_count - 1}]")
            if index in result:
                raise ValueError(f"validation index {index} was specified more than once")
            result.add(index)
    return result


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def transform_difference(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    delta = invert_transform(first) @ second
    return {
        "translation_mm": float(np.linalg.norm(delta[:3, 3]) * 1e3),
        "rotation_deg": rotation_angle_deg(delta[:3, :3]),
    }


def motion_diversity(
    transforms: Sequence[np.ndarray], min_relative_rotation_deg: float
) -> dict[str, float | int]:
    translation_span_m = 0.0
    significant_axes: list[np.ndarray] = []
    max_relative_rotation_deg = 0.0
    for first_index in range(len(transforms)):
        for second_index in range(first_index + 1, len(transforms)):
            first = transforms[first_index]
            second = transforms[second_index]
            translation_span_m = max(
                translation_span_m,
                float(np.linalg.norm(first[:3, 3] - second[:3, 3])),
            )
            relative_rotation = first[:3, :3].T @ second[:3, :3]
            rotation_vector = np.asarray(cv2.Rodrigues(relative_rotation)[0]).reshape(3)
            angle_rad = float(np.linalg.norm(rotation_vector))
            angle_deg = math.degrees(angle_rad)
            max_relative_rotation_deg = max(max_relative_rotation_deg, angle_deg)
            if angle_deg >= min_relative_rotation_deg and angle_rad > 1e-12:
                significant_axes.append(rotation_vector / angle_rad)

    max_axis_separation_deg = 0.0
    for first_index in range(len(significant_axes)):
        for second_index in range(first_index + 1, len(significant_axes)):
            # A rotation axis has an arbitrary sign for this diversity check.
            cosine = abs(
                float(np.dot(significant_axes[first_index], significant_axes[second_index]))
            )
            separation = math.degrees(math.acos(float(np.clip(cosine, 0.0, 1.0))))
            max_axis_separation_deg = max(max_axis_separation_deg, separation)

    return {
        "translation_span_mm": translation_span_m * 1e3,
        "max_relative_rotation_deg": max_relative_rotation_deg,
        "significant_relative_rotation_count": len(significant_axes),
        "max_axis_separation_deg": max_axis_separation_deg,
    }


def validate_motion_diversity(
    training_samples: Sequence[Sample],
    *,
    min_relative_rotation_deg: float,
    min_axis_separation_deg: float,
    min_translation_span_mm: float,
) -> dict[str, dict[str, float | int]]:
    result = {
        "T_base_flange": motion_diversity(
            [sample.T_base_flange for sample in training_samples],
            min_relative_rotation_deg,
        ),
        "T_camera_board": motion_diversity(
            [sample.T_camera_board for sample in training_samples],
            min_relative_rotation_deg,
        ),
    }
    failures: list[str] = []
    for stream_name, metrics in result.items():
        if int(metrics["significant_relative_rotation_count"]) < 2:
            failures.append(
                f"{stream_name} has fewer than two relative rotations of at least "
                f"{min_relative_rotation_deg:g} deg"
            )
        if float(metrics["max_axis_separation_deg"]) < min_axis_separation_deg:
            failures.append(
                f"{stream_name} rotation-axis separation "
                f"{float(metrics['max_axis_separation_deg']):.3f} deg is below "
                f"{min_axis_separation_deg:g} deg"
            )
        if float(metrics["translation_span_mm"]) < min_translation_span_mm:
            failures.append(
                f"{stream_name} translation span "
                f"{float(metrics['translation_span_mm']):.3f} mm is below "
                f"{min_translation_span_mm:g} mm"
            )
    if failures:
        raise ValueError("insufficient training motion diversity: " + "; ".join(failures))
    return result


def _rotation_translation_lists(
    transforms: Iterable[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for transform in transforms:
        rotations.append(np.ascontiguousarray(transform[:3, :3], dtype=np.float64))
        translations.append(np.ascontiguousarray(transform[:3, 3].reshape(3, 1), dtype=np.float64))
    return rotations, translations


def _make_transform(rotation: np.ndarray, translation: np.ndarray, name: str) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return _validate_transform(transform, name=name, max_translation_m=None, atol=1e-5)


def refit_robot_world_translations(
    training_samples: Sequence[Sample],
    flange_board_rotation: np.ndarray,
    base_camera_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve the AX=ZB translations after rotations are projected onto SO(3).

    OpenCV's LI implementation solves rotations and translations together, then
    projects its two raw 3x3 blocks onto SO(3) without updating the translations.
    Re-solving the linear translation equation removes that scale-dependent
    inconsistency.  Applying the same step to SHAH is effectively a no-op.
    """

    R_z = np.asarray(base_camera_rotation, dtype=np.float64).reshape(3, 3)
    blocks: list[np.ndarray] = []
    right_hand_sides: list[np.ndarray] = []
    for sample in training_samples:
        R_a = sample.T_base_flange[:3, :3]
        t_a = sample.T_base_flange[:3, 3]
        t_b = sample.T_camera_board[:3, 3]
        blocks.append(np.hstack((R_a, -np.eye(3, dtype=np.float64))))
        right_hand_sides.append(R_z @ t_b - t_a)

    coefficient_matrix = np.vstack(blocks)
    right_hand_side = np.concatenate(right_hand_sides)
    translation_pair, _, rank, singular_values = np.linalg.lstsq(
        coefficient_matrix,
        right_hand_side,
        rcond=None,
    )
    if rank != 6 or singular_values.size != 6 or singular_values[-1] <= 0.0:
        raise RuntimeError(
            "robot-world translation refit is rank deficient: "
            f"rank={rank}, singular_values={singular_values.tolist()}"
        )
    residual_rows = (
        coefficient_matrix @ translation_pair - right_hand_side
    ).reshape(-1, 3)
    residual_norms = np.linalg.norm(residual_rows, axis=1)
    diagnostics: dict[str, Any] = {
        "equation": "R_base_flange*t_flange_board - t_base_camera = "
        "R_base_camera*t_camera_board - t_base_flange",
        "sample_count": len(training_samples),
        "rank": int(rank),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "singular_values": singular_values.tolist(),
        "equation_residual_translation_mm": {
            "rms": float(np.sqrt(np.mean(np.square(residual_norms))) * 1e3),
            "max": float(np.max(residual_norms) * 1e3),
        },
    }
    return translation_pair[:3], translation_pair[3:], diagnostics


def solve_robot_world(training_samples: Sequence[Sample], method: str) -> Solution:
    base_flange_rotations, base_flange_translations = _rotation_translation_lists(
        sample.T_base_flange for sample in training_samples
    )
    camera_board_rotations, camera_board_translations = _rotation_translation_lists(
        sample.T_camera_board for sample in training_samples
    )
    try:
        (
            flange_board_rotation,
            flange_board_translation,
            base_camera_rotation,
            base_camera_translation,
        ) = cv2.calibrateRobotWorldHandEye(
            base_flange_rotations,
            base_flange_translations,
            camera_board_rotations,
            camera_board_translations,
            method=getattr(cv2, ROBOT_WORLD_METHOD_CONSTANTS[method]),
        )
    except cv2.error as exc:
        raise RuntimeError(f"OpenCV {method.upper()} calibration failed: {exc}") from exc

    # OpenCV solves A X = Z B.  Passing A=T_base_flange and B=T_camera_board
    # makes X=T_flange_board and Z=T_base_camera exactly, despite the generic
    # robot-world argument names in the OpenCV API.  Refit translation after
    # OpenCV's rotation projection so SHAH and LI satisfy the same rigid model.
    raw_T_base_camera = _make_transform(
        base_camera_rotation,
        base_camera_translation,
        f"raw {method} T_base_camera",
    )
    raw_T_flange_board = _make_transform(
        flange_board_rotation,
        flange_board_translation,
        f"raw {method} T_flange_board",
    )
    (
        flange_board_translation,
        base_camera_translation,
        translation_refit,
    ) = refit_robot_world_translations(
        training_samples,
        flange_board_rotation,
        base_camera_rotation,
    )
    T_base_camera = _make_transform(
        base_camera_rotation,
        base_camera_translation,
        f"{method} T_base_camera",
    )
    T_flange_board = _make_transform(
        flange_board_rotation,
        flange_board_translation,
        f"{method} T_flange_board",
    )
    translation_refit["raw_to_refit"] = {
        "T_base_camera": transform_difference(raw_T_base_camera, T_base_camera),
        "T_flange_board": transform_difference(raw_T_flange_board, T_flange_board),
    }
    return Solution(
        method=method,
        T_base_camera=T_base_camera,
        T_flange_board=T_flange_board,
        raw_T_base_camera=raw_T_base_camera,
        raw_T_flange_board=raw_T_flange_board,
        translation_refit=translation_refit,
    )


def solve_park_base_camera(training_samples: Sequence[Sample]) -> np.ndarray:
    # OpenCV's eye-to-hand PARK form expects T_flange_base (the inverse robot
    # poses) together with T_camera_board and returns T_base_camera.
    flange_base = [invert_transform(sample.T_base_flange) for sample in training_samples]
    flange_base_rotations, flange_base_translations = _rotation_translation_lists(flange_base)
    camera_board_rotations, camera_board_translations = _rotation_translation_lists(
        sample.T_camera_board for sample in training_samples
    )
    try:
        rotation, translation = cv2.calibrateHandEye(
            flange_base_rotations,
            flange_base_translations,
            camera_board_rotations,
            camera_board_translations,
            method=cv2.CALIB_HAND_EYE_PARK,
        )
    except cv2.error as exc:
        raise RuntimeError(f"OpenCV PARK calibration failed: {exc}") from exc
    return _make_transform(rotation, translation, "park T_base_camera")


def residual_summary(samples: Sequence[Sample], solution: Solution) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    translations: list[float] = []
    rotations: list[float] = []
    for sample in samples:
        left = sample.T_base_flange @ solution.T_flange_board
        right = solution.T_base_camera @ sample.T_camera_board
        error = transform_difference(left, right)
        translations.append(error["translation_mm"])
        rotations.append(error["rotation_deg"])
        per_sample.append(
            {
                "index": sample.index,
                "name": sample.name,
                **error,
            }
        )

    def stats(values: Sequence[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "median": None, "rms": None, "max": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "rms": float(np.sqrt(np.mean(np.square(array)))),
            "max": float(np.max(array)),
        }

    return {
        "count": len(samples),
        "translation_mm": stats(translations),
        "rotation_deg": stats(rotations),
        "per_sample": per_sample,
    }


def _max_metric(summary: dict[str, Any], metric: str) -> float:
    value = summary[metric]["max"]
    if value is None:
        return 0.0
    return float(value)


def quality_failures(
    solutions: dict[str, Solution],
    solution_reports: dict[str, dict[str, Any]],
    primary_method: str,
    method_consistency: dict[str, dict[str, float]],
    park_consistency: dict[str, float] | None,
    args: argparse.Namespace,
) -> list[str]:
    failures: list[str] = []
    for method, solution in solutions.items():
        for transform_name, transform in (
            ("T_base_camera", solution.T_base_camera),
            ("T_flange_board", solution.T_flange_board),
        ):
            translation_norm = float(np.linalg.norm(transform[:3, 3]))
            if translation_norm > args.max_input_translation_m:
                failures.append(
                    f"{method.upper()} {transform_name} translation norm "
                    f"{translation_norm:.3f} m > {args.max_input_translation_m:g} m"
                )
        train = solution_reports[method]["training_residual"]
        train_translation = _max_metric(train, "translation_mm")
        train_rotation = _max_metric(train, "rotation_deg")
        if train_translation > args.max_train_translation_mm:
            failures.append(
                f"{method.upper()} training translation max {train_translation:.3f} mm "
                f"> {args.max_train_translation_mm:g} mm"
            )
        if train_rotation > args.max_train_rotation_deg:
            failures.append(
                f"{method.upper()} training rotation max {train_rotation:.3f} deg "
                f"> {args.max_train_rotation_deg:g} deg"
            )
        validation = solution_reports[method]["validation_residual"]
        if int(validation["count"]) > 0:
            validation_translation = _max_metric(validation, "translation_mm")
            validation_rotation = _max_metric(validation, "rotation_deg")
            if validation_translation > args.max_validation_translation_mm:
                failures.append(
                    f"{method.upper()} validation translation max "
                    f"{validation_translation:.3f} mm > "
                    f"{args.max_validation_translation_mm:g} mm"
                )
            if validation_rotation > args.max_validation_rotation_deg:
                failures.append(
                    f"{method.upper()} validation rotation max {validation_rotation:.3f} deg "
                    f"> {args.max_validation_rotation_deg:g} deg"
                )

    for transform_name, disagreement in method_consistency.items():
        if disagreement["translation_mm"] > args.max_method_translation_mm:
            failures.append(
                f"SHAH/LI {transform_name} translation disagreement "
                f"{disagreement['translation_mm']:.3f} mm > "
                f"{args.max_method_translation_mm:g} mm"
            )
        if disagreement["rotation_deg"] > args.max_method_rotation_deg:
            failures.append(
                f"SHAH/LI {transform_name} rotation disagreement "
                f"{disagreement['rotation_deg']:.3f} deg > "
                f"{args.max_method_rotation_deg:g} deg"
            )

    if park_consistency is not None:
        if park_consistency["translation_mm"] > args.max_park_translation_mm:
            failures.append(
                f"{primary_method.upper()}/PARK T_base_camera translation disagreement "
                f"{park_consistency['translation_mm']:.3f} mm > "
                f"{args.max_park_translation_mm:g} mm"
            )
        if park_consistency["rotation_deg"] > args.max_park_rotation_deg:
            failures.append(
                f"{primary_method.upper()}/PARK T_base_camera rotation disagreement "
                f"{park_consistency['rotation_deg']:.3f} deg > "
                f"{args.max_park_rotation_deg:g} deg"
            )
    return failures


def _write_npy_temp(directory: Path, stem: str, value: np.ndarray) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{stem}.", suffix=".tmp", dir=directory)
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _write_json_temp(directory: Path, stem: str, value: dict[str, Any]) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{stem}.", suffix=".tmp", dir=directory)
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_exchange(first: Path, second: Path) -> None:
    """Atomically exchange two directory entries on Linux."""
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "libc does not provide renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_exchange = 2
    result = renameat2(
        at_fdcwd,
        os.fsencode(first),
        at_fdcwd,
        os.fsencode(second),
        rename_exchange,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            f"atomic directory exchange failed: {os.strerror(error_number)}",
            f"{first} <-> {second}",
        )


def _copy_existing_generation(output_dir: Path, staging_dir: Path) -> None:
    if output_dir.is_symlink():
        raise ValueError(f"output directory must not be a symbolic link: {output_dir}")
    if not output_dir.is_dir():
        raise ValueError(f"output path exists but is not a directory: {output_dir}")
    # Preserve unrelated calibration artifacts while replacing this solver's
    # three files as one directory generation.
    shutil.copytree(output_dir, staging_dir, dirs_exist_ok=True, symlinks=True)


def _publish_staged_directory(staging_dir: Path, output_dir: Path) -> bool:
    """Publish staging_dir atomically; return whether it now holds the old generation."""
    if os.path.lexists(output_dir):
        _rename_exchange(staging_dir, output_dir)
        return True
    os.replace(staging_dir, output_dir)
    return False


def publish_outputs(output_dir: Path, solution: Solution, report: dict[str, Any]) -> None:
    output_dir = output_dir.expanduser()
    if output_dir.name in {"", ".", ".."}:
        raise ValueError(f"invalid output directory: {output_dir}")
    # Use one normalized absolute spelling for the lock, staging directory and
    # exchange while deliberately not following an existing output symlink.
    output_dir = Path(os.path.abspath(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.publish.lock"
    lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    staging_dir: Path | None = None
    staging_holds_old_generation = False
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.generation-", dir=output_dir.parent)
        )
        if os.path.lexists(output_dir):
            _copy_existing_generation(output_dir, staging_dir)

        targets = {
            "T_base_camera.npy": solution.T_base_camera,
            "T_flange_board.npy": solution.T_flange_board,
        }
        for filename, value in targets.items():
            temporary = _write_npy_temp(staging_dir, Path(filename).stem, value)
            os.replace(temporary, staging_dir / filename)
        temporary_report = _write_json_temp(staging_dir, "report", report)
        os.replace(temporary_report, staging_dir / "report.json")
        _fsync_directory(staging_dir)

        staging_holds_old_generation = _publish_staged_directory(staging_dir, output_dir)
        if not staging_holds_old_generation:
            staging_dir = None
        _fsync_directory(output_dir.parent)
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)
        if staging_dir is not None:
            try:
                shutil.rmtree(staging_dir)
            except OSError as exc:
                if staging_holds_old_generation:
                    print(
                        f"warning: calibration was published but old generation cleanup "
                        f"failed at {staging_dir}: {exc}",
                        file=sys.stderr,
                    )
                else:
                    raise


def _check_positive_arguments(args: argparse.Namespace) -> None:
    positive_names = (
        "min_relative_rotation_deg",
        "min_axis_separation_deg",
        "min_translation_span_mm",
        "max_input_translation_m",
        "max_train_translation_mm",
        "max_train_rotation_deg",
        "max_validation_translation_mm",
        "max_validation_rotation_deg",
        "max_method_translation_mm",
        "max_method_rotation_deg",
        "max_park_translation_mm",
        "max_park_rotation_deg",
    )
    if args.min_training_samples < 3:
        raise ValueError("--min-training-samples must be at least 3")
    if args.min_validation_samples < 0:
        raise ValueError("--min-validation-samples must be non-negative")
    for name in positive_names:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.min_axis_separation_deg > 90.0:
        raise ValueError("--min-axis-separation-deg must not exceed 90")
    if args.min_relative_rotation_deg >= 180.0:
        raise ValueError("--min-relative-rotation-deg must be below 180")


def _print_residual(label: str, summary: dict[str, Any]) -> None:
    if not summary["count"]:
        print(f"{label}: no samples")
        return
    translation = summary["translation_mm"]
    rotation = summary["rotation_deg"]
    print(
        f"{label}: n={summary['count']}, translation mm "
        f"rms/max={translation['rms']:.3f}/{translation['max']:.3f}, "
        f"rotation deg rms/max={rotation['rms']:.3f}/{rotation['max']:.3f}"
    )


def summarize_sample_provenance(
    samples: Sequence[Sample], *, allow_unverified_samples: bool
) -> dict[str, Any]:
    verified = [sample for sample in samples if sample.provenance is not None]
    unverified = [sample for sample in samples if sample.provenance is None]
    reference = verified[0].provenance if verified else None
    return {
        "verification_policy": (
            "allow_missing_manifests" if allow_unverified_samples else "strict_manifests"
        ),
        "all_samples_verified": not unverified,
        "verified_sample_count": len(verified),
        "unverified_sample_count": len(unverified),
        "unverified_samples": [
            {"index": sample.index, "name": sample.name} for sample in unverified
        ],
        "manifest_schema": reference.manifest_schema if reference is not None else None,
        "session_id": reference.session_id if reference is not None else None,
        "camera": (
            {
                "serial": reference.camera_serial,
                "K": reference.camera_matrix.tolist(),
                "distortion_model": reference.distortion_model,
                "distortion_coefficients": reference.distortion_coefficients.tolist(),
            }
            if reference is not None
            else None
        ),
        "board": reference.board if reference is not None else None,
        "verified_sources": [
            {
                "index": sample.index,
                "name": sample.name,
                "manifest": str(sample.provenance.manifest_path.resolve()),
                "board_metadata": str(sample.provenance.board_metadata_path.resolve()),
            }
            for sample in verified
            if sample.provenance is not None
        ],
    }


def main() -> int:
    args = parse_args()
    check_opencv_calibration_capabilities()
    _check_positive_arguments(args)
    samples = discover_samples(
        args.samples_dir,
        args.max_input_translation_m,
        allow_unverified_samples=args.allow_unverified_samples,
    )
    validation_indices = parse_validation_indices(args.validation_indices, len(samples))
    training_samples = [sample for sample in samples if sample.index not in validation_indices]
    validation_samples = [sample for sample in samples if sample.index in validation_indices]
    if len(training_samples) < args.min_training_samples:
        raise ValueError(
            f"need at least {args.min_training_samples} training samples after holdout, "
            f"got {len(training_samples)}"
        )
    if len(validation_samples) < args.min_validation_samples:
        raise ValueError(
            f"need at least {args.min_validation_samples} held-out validation samples; "
            f"got {len(validation_samples)}. Pass --validation-indices, or explicitly set "
            "--min-validation-samples 0 for a preliminary solve"
        )

    diversity = validate_motion_diversity(
        training_samples,
        min_relative_rotation_deg=args.min_relative_rotation_deg,
        min_axis_separation_deg=args.min_axis_separation_deg,
        min_translation_span_mm=args.min_translation_span_mm,
    )
    solutions = {
        method: solve_robot_world(training_samples, method)
        for method in ROBOT_WORLD_METHOD_CONSTANTS
    }
    solution_reports: dict[str, dict[str, Any]] = {}
    for method, solution in solutions.items():
        solution_reports[method] = {
            "T_base_camera": solution.T_base_camera.tolist(),
            "T_flange_board": solution.T_flange_board.tolist(),
            "opencv_raw_T_base_camera": solution.raw_T_base_camera.tolist(),
            "opencv_raw_T_flange_board": solution.raw_T_flange_board.tolist(),
            "translation_refit": solution.translation_refit,
            "training_residual": residual_summary(training_samples, solution),
            "validation_residual": residual_summary(validation_samples, solution),
        }

    method_consistency = {
        "T_base_camera": transform_difference(
            solutions["shah"].T_base_camera, solutions["li"].T_base_camera
        ),
        "T_flange_board": transform_difference(
            solutions["shah"].T_flange_board, solutions["li"].T_flange_board
        ),
    }
    primary = solutions[args.primary_method]
    park_base_camera: np.ndarray | None = None
    park_consistency: dict[str, float] | None = None
    if not args.no_park_cross_check:
        park_base_camera = solve_park_base_camera(training_samples)
        park_consistency = transform_difference(primary.T_base_camera, park_base_camera)

    failures = quality_failures(
        solutions,
        solution_reports,
        args.primary_method,
        method_consistency,
        park_consistency,
        args,
    )
    thresholds = {
        key: value
        for key, value in vars(args).items()
        if key.startswith("min_") or key.startswith("max_")
    }
    report: dict[str, Any] = {
        "schema": "demo2gripper.eye_to_hand_calibration.v1",
        "schema_version": 1,
        "generation_id": str(uuid.uuid4()),
        "created_utc": datetime.now(UTC).isoformat(),
        "opencv_version": cv2.__version__,
        "frame_convention": "T_parent_child, right-handed, column vectors, metres",
        "calibration_equation": ("T_base_flange @ T_flange_board = T_base_camera @ T_camera_board"),
        "samples_dir": str(args.samples_dir.resolve()),
        "sample_order": [
            {"index": sample.index, "name": sample.name, "path": str(sample.directory.resolve())}
            for sample in samples
        ],
        "sample_provenance": summarize_sample_provenance(
            samples, allow_unverified_samples=args.allow_unverified_samples
        ),
        "training_indices": [sample.index for sample in training_samples],
        "validation_indices": [sample.index for sample in validation_samples],
        "selected_method": args.primary_method,
        "thresholds": thresholds,
        "motion_diversity": diversity,
        "solutions": solution_reports,
        "shah_li_consistency": method_consistency,
        "park_cross_check": (
            {
                "enabled": True,
                "T_base_camera": park_base_camera.tolist(),
                "primary_method": args.primary_method,
                "primary_consistency": park_consistency,
            }
            if park_base_camera is not None
            else {"enabled": False}
        ),
        "acceptance": {"accepted": not failures, "failures": failures},
    }

    print(
        f"samples: {len(samples)} total, {len(training_samples)} training, "
        f"{len(validation_samples)} validation"
    )
    for method in ROBOT_WORLD_METHOD_CONSTANTS:
        _print_residual(f"{method.upper()} train", solution_reports[method]["training_residual"])
        _print_residual(
            f"{method.upper()} validation", solution_reports[method]["validation_residual"]
        )
    for transform_name, disagreement in method_consistency.items():
        print(
            f"SHAH/LI {transform_name}: {disagreement['translation_mm']:.3f} mm, "
            f"{disagreement['rotation_deg']:.3f} deg"
        )
    if park_consistency is not None:
        print(
            f"{args.primary_method.upper()}/PARK T_base_camera: "
            f"{park_consistency['translation_mm']:.3f} mm, "
            f"{park_consistency['rotation_deg']:.3f} deg"
        )
    if failures:
        detail = "\n  - ".join(failures)
        raise RuntimeError("calibration rejected; no output files were changed:\n  - " + detail)

    publish_outputs(args.output_dir, primary, report)
    print(f"accepted {args.primary_method.upper()} calibration")
    for filename in ("T_base_camera.npy", "T_flange_board.npy", "report.json"):
        print(f"wrote {args.output_dir / filename}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"error: {error}") from None
