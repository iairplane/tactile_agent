#!/usr/bin/env python3
"""Register a no-CAD demo template to multi-frame live object point clouds.

This command writes geometry evidence only.  It never connects to ROS, Franka, or the
gripper, and the resulting pose remains review-only for downstream planning gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from demo2gripper.franka import validate_rigid_transform
from demo2gripper.live_registration import (
    RegistrationConfig,
    register_partial_template,
)

SCHEMA = "demo2gripper.live_object_pose.v1"
FRAME_CONVENTION = "T_parent_child, right-handed, column vectors, metres"
TEMPLATE_SCHEMA = "demo2gripper.no_cad_object_template.v1"
ANCHOR_SCHEMA = "demo2gripper.live_supported_object_anchor.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--template-dir",
        required=True,
        type=Path,
        help="Directory containing object_points_m.npy and template_manifest.json",
    )
    parser.add_argument(
        "--live-points",
        required=True,
        type=Path,
        help="selected_object_points_base_m.npz from localize_supported_object.py",
    )
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument(
        "--t-base-camera",
        type=Path,
        help=(
            "Calibrated T_base_camera NPY; combined with the demo manifest rotation "
            "to provide the pose gauge"
        ),
    )
    initialization.add_argument(
        "--initial-anchor",
        type=Path,
        help=(
            "live_object_anchor.json; its center and embedded calibrated "
            "T_base_camera are used only as initialization"
        ),
    )
    initialization.add_argument(
        "--initial-t-base-object",
        type=Path,
        help="Explicit reviewed 4x4 T_base_object NPY initialization",
    )
    parser.add_argument(
        "--orientation-prior",
        type=Path,
        help=(
            "Optional explicit 3x3 R_base_object or 4x4 T_base_object NPY. It is "
            "never inferred from the initialization and requires source/error gates."
        ),
    )
    parser.add_argument(
        "--orientation-prior-source",
        help="Required traceable reason that the supplied orientation is valid",
    )
    parser.add_argument(
        "--lock-orientation-to-prior",
        action="store_true",
        help=(
            "Keep the output rotation exactly equal to --orientation-prior and "
            "estimate only translation from live points; free-rotation ICP remains "
            "a diagnostic consistency gate"
        ),
    )
    parser.add_argument(
        "--max-orientation-prior-error-deg",
        type=float,
        help="Required maximum selected-pose angular error when a prior is supplied",
    )
    parser.add_argument(
        "--orientation-prior-uniqueness-margin-deg",
        type=float,
        default=5.0,
        help="Required angular separation between the best and second prior matches",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--symmetry", choices=("cube", "none"), default="cube")
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--min-points-per-frame", type=int, default=100)
    parser.add_argument("--voxel-size-m", type=float, default=0.001)
    parser.add_argument("--max-correspondence-distance-m", type=float, default=0.012)
    parser.add_argument("--inlier-distance-m", type=float, default=0.005)
    parser.add_argument("--max-fit-rmse-m", type=float, default=0.0045)
    parser.add_argument("--max-held-out-rmse-m", type=float, default=0.0055)
    parser.add_argument("--max-frame-translation-deviation-m", type=float, default=0.006)
    parser.add_argument("--max-frame-rotation-deviation-deg", type=float, default=12.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_frame_sha256(points: np.ndarray, frame_convention: str) -> str:
    canonical_points = np.ascontiguousarray(np.asarray(points, dtype="<f8"))
    shape = np.ascontiguousarray(np.asarray(canonical_points.shape, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(b"demo2gripper.no_cad_object_template.v1\0")
    digest.update(frame_convention.encode("utf-8"))
    digest.update(b"\0")
    digest.update(shape.tobytes())
    digest.update(canonical_points.tobytes())
    return digest.hexdigest()


def _load_template(template_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = template_dir / "template_manifest.json"
    points_path = template_dir / "object_points_m.npy"
    if not manifest_path.is_file() or not points_path.is_file():
        raise ValueError(
            "template directory must contain template_manifest.json and object_points_m.npy"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != TEMPLATE_SCHEMA:
        raise ValueError(f"template manifest schema must be {TEMPLATE_SCHEMA!r}")
    if manifest.get("units") != "metres":
        raise ValueError("template points must use metres")
    frame_id = manifest.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ValueError("template frame_id must be a non-empty string")
    if manifest.get("object_points_file") != points_path.name:
        raise ValueError("template manifest points filename does not match input")
    expected_sha256 = manifest.get("object_points_file_sha256")
    if not isinstance(expected_sha256, str) or _sha256_file(points_path) != expected_sha256:
        raise ValueError("template point-cloud file SHA-256 does not match manifest")
    points = np.load(points_path, allow_pickle=False)
    if manifest.get("point_count") != len(points):
        raise ValueError("template point count does not match manifest")
    frame_convention = manifest.get("frame_convention")
    if not isinstance(frame_convention, str) or not frame_convention:
        raise ValueError("template frame_convention must be a non-empty string")
    semantic_sha256 = _object_frame_sha256(points, frame_convention)
    if manifest.get("object_template_sha256") != semantic_sha256:
        raise ValueError("template semantic SHA-256 does not match its points and convention")
    if frame_id != f"object_template_camera_axes:sha256:{semantic_sha256}":
        raise ValueError("template frame_id does not match its semantic SHA-256")
    validate_rigid_transform(
        manifest.get("T_demo_camera_object_template"),
        name="T_demo_camera_object_template",
    )
    return points, manifest


def _frame_sort_key(name: str) -> tuple[int, str]:
    prefix = "frame_"
    if not name.startswith(prefix):
        raise ValueError(f"unexpected live point-cloud key: {name!r}")
    try:
        return int(name[len(prefix) :]), name
    except ValueError as exc:
        raise ValueError(f"live point-cloud key has non-numeric suffix: {name!r}") from exc


def _load_live_points(path: Path) -> tuple[list[np.ndarray], list[str]]:
    if path.suffix.lower() != ".npz" or not path.is_file():
        raise ValueError("--live-points must be an existing NPZ archive")
    with np.load(path, allow_pickle=False) as archive:
        keys = sorted(archive.files, key=_frame_sort_key)
        if not keys:
            raise ValueError("live point-cloud archive is empty")
        frames = [np.asarray(archive[key], dtype=float) for key in keys]
    return frames, keys


def _reference_from_args(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    demo_camera_object = validate_rigid_transform(
        manifest["T_demo_camera_object_template"],
        name="T_demo_camera_object_template",
    )
    if args.initial_t_base_object is not None:
        reference = validate_rigid_transform(
            np.load(args.initial_t_base_object, allow_pickle=False),
            name="initial_T_base_object",
        )
        return reference, {
            "mode": "reviewed_full_pose",
            "path": str(args.initial_t_base_object.resolve()),
        }

    if args.t_base_camera is not None:
        base_camera = validate_rigid_transform(
            np.load(args.t_base_camera, allow_pickle=False), name="T_base_camera"
        )
        reference = base_camera @ demo_camera_object
        return reference, {
            "mode": "calibrated_camera_plus_demo_template_orientation",
            "path": str(args.t_base_camera.resolve()),
            "T_base_camera": base_camera.tolist(),
        }

    anchor = _read_json(args.initial_anchor)
    if anchor.get("schema") != ANCHOR_SCHEMA or anchor.get("status") != "accepted":
        raise ValueError("initial anchor is not an accepted supported-object artifact")
    center = np.asarray(anchor.get("center_base_m"), dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("initial anchor center_base_m must be a finite three-vector")
    inputs = anchor.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("initial anchor has no input calibration evidence")
    base_camera = validate_rigid_transform(
        inputs.get("T_base_camera_value"), name="anchor T_base_camera_value"
    )
    reference = base_camera @ demo_camera_object
    reference[:3, 3] = center
    return reference, {
        "mode": "accepted_anchor_plus_embedded_calibrated_camera",
        "path": str(args.initial_anchor.resolve()),
        "T_base_camera": base_camera.tolist(),
        "anchor_center_base_m": center.tolist(),
    }


def _quality_payload(result: Any, config: RegistrationConfig) -> dict[str, Any]:
    frame_metrics = [metrics.to_dict() for metrics in result.frame_metrics]
    held_out_rmse = max(
        frame_metrics[index]["rmse_m"] for index in result.held_out_frame_indices
    )
    locked = bool(result.orientation_locked_to_prior)
    unlocked_fit = (
        None
        if result.unlocked_fit_metrics is None
        else result.unlocked_fit_metrics.to_dict()
    )
    return {
        "accepted": True,
        "method": (
            "explicit_prior_rotation_locked_bidirectional_trimmed_translation_icp"
            if locked
            else "multi_start_bidirectional_trimmed_point_to_point_icp_svd"
        ),
        "template_observation_model": "partial_single_view",
        "validation_model": "fit_even_frames_hold_out_odd_frames",
        "symmetry_model": (
            "proper_cube_rotation_group_24" if config.symmetry == "cube" else "none"
        ),
        "pose_identifiability": (
            "rotation_fixed_by_explicit_prior"
            if locked
            else (
                "modulo_cube_rotation_group"
                if config.symmetry == "cube"
                else "unique"
            )
        ),
        "orientation_gauge": (
            "explicit_prior_rotation_locked_translation_fitted"
            if locked
            else "equivalent_hypothesis_nearest_supplied_reference_rotation"
        ),
        "cube_axes_object": (
            None
            if result.cube_axes_object is None
            else result.cube_axes_object.tolist()
        ),
        "cube_axes_source": (
            None
            if result.cube_axes_object is None
            else "two_orthogonal_template_face_normals_ransac"
        ),
        "fit_frame_indices": list(result.fit_frame_indices),
        "held_out_frame_indices": list(result.held_out_frame_indices),
        "fit": result.fit_metrics.to_dict(),
        "held_out_max_rmse_m": held_out_rmse,
        "per_frame": [
            {
                "frame_index": index,
                **metrics,
                "refined_translation_deviation_m": float(
                    result.frame_translation_deviation_m[index]
                ),
                "refined_rotation_deviation_deg_modulo_symmetry": float(
                    result.frame_rotation_deviation_deg[index]
                ),
                "role": (
                    "held_out"
                    if index in result.held_out_frame_indices
                    else "fit"
                ),
            }
            for index, metrics in enumerate(frame_metrics)
        ],
        "hypotheses": {
            "initial_count": result.initial_hypothesis_count,
            "near_optimal_count": result.near_optimal_hypothesis_count,
            "symmetry_equivalent_near_optimal_count": (
                result.symmetry_equivalent_hypothesis_count
            ),
            "non_symmetry_ambiguity_check": (
                "free_rotation_diagnostic_resolved_by_explicit_prior_before_lock"
                if locked
                else (
                    "resolved_by_explicit_orientation_prior"
                    if result.orientation_prior_used
                    else "passed_geometry_only"
                )
            ),
        },
        "orientation_prior": {
            "used": result.orientation_prior_used,
            "rotation_locked": locked,
            "error_deg": result.orientation_prior_error_deg,
            "error_semantics": (
                "free_rotation_diagnostic_to_prior"
                if locked
                else "selected_output_rotation_to_prior"
            ),
            "second_best_error_deg": result.orientation_prior_second_best_error_deg,
            "max_error_deg": result.orientation_prior_max_error_deg,
            "uniqueness_margin_deg": (
                result.orientation_prior_uniqueness_margin_deg
            ),
            "check": (
                "free_rotation_branch_validated_then_prior_locked_exactly"
                if locked
                else (
                    "passed_unique_near_optimal_branch"
                    if result.orientation_prior_used
                    else "not_requested"
                )
            ),
        },
        "rotation_lock": {
            "enabled": locked,
            "output_rotation_estimated_from_live_points": not locked,
            "final_error_to_prior_deg": result.orientation_lock_error_deg,
            "free_rotation_fit": unlocked_fit,
            "locked_rotation_fit": result.fit_metrics.to_dict() if locked else None,
            "fit_rmse_increase_m": (
                None
                if unlocked_fit is None
                else result.fit_metrics.rmse_m - unlocked_fit["rmse_m"]
            ),
            "free_rotation_role": (
                "diagnostic_prior_consistency_and_per_frame_rotation_gate"
                if locked
                else None
            ),
        },
        "thresholds": {
            "max_correspondence_distance_m": config.max_correspondence_distance_m,
            "inlier_distance_m": config.inlier_distance_m,
            "min_template_inlier_fraction": config.min_template_inlier_fraction,
            "min_live_inlier_fraction": config.min_live_inlier_fraction,
            "max_fit_rmse_m": config.max_fit_rmse_m,
            "max_held_out_rmse_m": config.max_held_out_rmse_m,
            "max_frame_translation_deviation_m": (
                config.max_frame_translation_deviation_m
            ),
            "max_frame_rotation_deviation_deg": (
                config.max_frame_rotation_deviation_deg
            ),
        },
    }


def run(args: argparse.Namespace) -> Path:
    template_dir = args.template_dir.resolve()
    points, manifest = _load_template(template_dir)
    live_frames, frame_keys = _load_live_points(args.live_points.resolve())
    reference, initialization = _reference_from_args(args, manifest)
    prior_path = getattr(args, "orientation_prior", None)
    prior_source = getattr(args, "orientation_prior_source", None)
    prior_max_error = getattr(args, "max_orientation_prior_error_deg", None)
    prior_uniqueness = getattr(
        args, "orientation_prior_uniqueness_margin_deg", 5.0
    )
    lock_orientation = bool(getattr(args, "lock_orientation_to_prior", False))
    if prior_path is None:
        if lock_orientation:
            raise ValueError(
                "--lock-orientation-to-prior requires --orientation-prior"
            )
        if prior_source is not None or prior_max_error is not None:
            raise ValueError(
                "orientation prior source/error options require --orientation-prior"
            )
        orientation_prior = None
        prior_record: dict[str, Any] = {"used": False}
    else:
        if not isinstance(prior_source, str) or not prior_source.strip():
            raise ValueError(
                "--orientation-prior-source is required with --orientation-prior"
            )
        if prior_max_error is None:
            raise ValueError(
                "--max-orientation-prior-error-deg is required with --orientation-prior"
            )
        orientation_prior = np.load(prior_path, allow_pickle=False)
        if orientation_prior.shape not in {(3, 3), (4, 4)}:
            raise ValueError("orientation prior NPY must contain a 3x3 or 4x4 matrix")
        prior_record = {
            "used": True,
            "path": str(prior_path.resolve()),
            "file_sha256": _sha256_file(prior_path),
            "source": prior_source.strip(),
            "input_shape": list(orientation_prior.shape),
            "translation_ignored": orientation_prior.shape == (4, 4),
            "rotation_locked": lock_orientation,
            "output_rotation_source": (
                "exact_explicit_prior"
                if lock_orientation
                else "live_geometry_branch_selected_by_prior"
            ),
        }
    config = RegistrationConfig(
        symmetry=args.symmetry,
        min_frames=args.min_frames,
        min_points_per_frame=args.min_points_per_frame,
        voxel_size_m=args.voxel_size_m,
        max_correspondence_distance_m=args.max_correspondence_distance_m,
        inlier_distance_m=args.inlier_distance_m,
        max_fit_rmse_m=args.max_fit_rmse_m,
        max_held_out_rmse_m=args.max_held_out_rmse_m,
        max_frame_translation_deviation_m=args.max_frame_translation_deviation_m,
        max_frame_rotation_deviation_deg=args.max_frame_rotation_deviation_deg,
    )
    result = register_partial_template(
        points,
        live_frames,
        reference,
        config=config,
        orientation_prior_base_object=orientation_prior,
        max_orientation_prior_error_deg=(
            None if orientation_prior is None else prior_max_error
        ),
        orientation_prior_uniqueness_margin_deg=(
            None if orientation_prior is None else prior_uniqueness
        ),
        lock_orientation_to_prior=lock_orientation,
    )
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise ValueError(f"output already exists; use --overwrite: {output}")
    payload = {
        "schema": SCHEMA,
        "status": "accepted",
        "authorized_for_robot_execution": False,
        "frame_convention": FRAME_CONVENTION,
        "base_frame_id": "base",
        "object_frame_id": manifest["frame_id"],
        "T_base_object": result.T_base_object.tolist(),
        "quality": _quality_payload(result, config),
        "initialization": initialization,
        "orientation_prior": prior_record,
        "provenance": {
            "registration_mode": (
                "translation_only_with_locked_explicit_orientation_prior"
                if lock_orientation
                else "full_se3"
            ),
            "template_manifest": str(
                (template_dir / "template_manifest.json").resolve()
            ),
            "template_points": str((template_dir / "object_points_m.npy").resolve()),
            "template_points_sha256": manifest["object_points_file_sha256"],
            "live_points": str(args.live_points.resolve()),
            "live_frame_keys": frame_keys,
        },
        "limitations": [
            (
                "The absolute output rotation comes from the explicit prior; live depth "
                "geometry only validates its fit and cannot identify printed die faces."
                if lock_orientation
                else "Depth geometry cannot identify printed die face labels; the returned "
                "rotation is a canonical representative modulo the 24 cube rotations."
                if config.symmetry == "cube"
                else "No object symmetry was declared."
            ),
            (
                "This artifact is review-only; IK, collision, support-clearance, and "
                "trajectory checks remain mandatory."
            ),
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, output)
    return output


def main() -> int:
    args = parse_args()
    try:
        output = run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"live object registration rejected: {exc}", file=sys.stderr)
        return 2
    print(f"accepted review-only live object pose: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

