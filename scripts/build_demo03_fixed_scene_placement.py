#!/usr/bin/env python3
"""Compile demo-03 final die pixels into a fixed-scene placement relation.

The output is deliberately translation-only.  It estimates a robust base-frame
anchor for the die after the demonstrated placement, but does not estimate or
invent a box pose.  When a live grounded pick candidate is supplied, the same
TCP-to-die translation is carried to that fixed target and review-only waypoint
JSON files are emitted.  This script never imports ROS or sends robot commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from demo2gripper.autonomous_pick import rotation_matrix_to_quaternion_xyzw
from demo2gripper.cartesian_pose_sequence import (
    POSE_SEQUENCE_SCHEMA,
    CartesianPose,
    validate_pose_sequence_motion,
)
from demo2gripper.fixed_object import validate_intrinsics
from demo2gripper.franka import validate_rigid_transform

RELATION_SCHEMA = "demo2gripper.demo03_fixed_scene_placement_relation.v1"
GROUNDING_SCHEMA = "demo2gripper.demo03_fixed_scene_placement_grounding.v1"
GROUNDED_PICK_SCHEMA = "demo2gripper.grounded_live_grasp_candidate.v1"
GROUNDED_PICK_SCHEMAS = {
    GROUNDED_PICK_SCHEMA,
    "demo2gripper.grounded_live_grasp_candidate.v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--demo-dir", required=True, type=Path)
    parser.add_argument(
        "--mask-dir",
        type=Path,
        help="Reverse-proxy die masks (defaults to demo SAM2 object 2 masks)",
    )
    parser.add_argument(
        "--reverse-provenance",
        type=Path,
        help="Mapping from reverse-proxy mask index to original demo frame",
    )
    parser.add_argument(
        "--t-base-camera",
        type=Path,
        default=Path("calibration/T_base_camera.npy"),
    )
    parser.add_argument(
        "--original-final-frames",
        type=int,
        nargs=2,
        default=(552, 651),
        metavar=("START", "END"),
        help="Half-open original-frame interval used for the stationary final die",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=1.5)
    parser.add_argument("--min-valid-points", type=int, default=250)
    parser.add_argument("--min-valid-fraction", type=float, default=0.35)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--max-center-rms-m", type=float, default=0.003)
    parser.add_argument("--max-center-deviation-m", type=float, default=0.006)
    parser.add_argument("--max-point-count-cv", type=float, default=0.10)
    parser.add_argument(
        "--grounded-pick",
        type=Path,
        help="Live execution_candidate.json; enables placement waypoint generation",
    )
    parser.add_argument("--transport-height-m", type=float, default=0.080)
    parser.add_argument("--retreat-height-m", type=float, default=0.080)
    parser.add_argument("--base-frame", default="fr3_link0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <demo-dir>/multicontact_plan/inputs",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: Path, *, repo_root: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (repo_root / expanded).resolve()


def _display(path: Path, *, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError(f"output already exists; pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_positive(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _validate_quality_args(args: argparse.Namespace) -> None:
    min_depth = _finite_positive(args.min_depth_m, name="min_depth_m")
    max_depth = _finite_positive(args.max_depth_m, name="max_depth_m")
    if min_depth >= max_depth:
        raise ValueError("depth range must satisfy min_depth_m < max_depth_m")
    if args.min_valid_points < 1:
        raise ValueError("min_valid_points must be positive")
    if not 0.0 < args.min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must lie in (0, 1]")
    if args.min_frames < 3:
        raise ValueError("min_frames must be at least three")
    _finite_positive(args.max_center_rms_m, name="max_center_rms_m")
    _finite_positive(args.max_center_deviation_m, name="max_center_deviation_m")
    if not math.isfinite(args.max_point_count_cv) or args.max_point_count_cv < 0.0:
        raise ValueError("max_point_count_cv must be finite and non-negative")
    _finite_positive(args.transport_height_m, name="transport_height_m")
    _finite_positive(args.retreat_height_m, name="retreat_height_m")


def _base_points(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    T_base_camera: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    valid = mask & np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    rows, columns = np.nonzero(valid)
    z = depth_m[rows, columns].astype(np.float64, copy=False)
    camera = np.column_stack(
        (
            (columns - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (rows - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    )
    return camera @ T_base_camera[:3, :3].T + T_base_camera[:3, 3]


def _mask_source_mapping(provenance: Mapping[str, Any]) -> dict[int, int]:
    if provenance.get("schema") != "demo2gripper.time_reversed_proxy.v1":
        raise ValueError("reverse provenance has the wrong schema")
    if provenance.get("not_original_pickup") is not True:
        raise ValueError("reverse provenance must explicitly state not_original_pickup=true")
    source = provenance.get("source_frame_indices")
    if (
        not isinstance(source, list)
        or not source
        or any(isinstance(value, bool) or not isinstance(value, int) for value in source)
    ):
        raise ValueError("reverse provenance source_frame_indices must be an integer array")
    if len(set(source)) != len(source):
        raise ValueError("reverse provenance source_frame_indices must be unique")
    return {int(original): proxy for proxy, original in enumerate(source)}


def infer_final_anchor(
    *,
    depth_dir: Path,
    mask_dir: Path,
    source_to_proxy: Mapping[int, int],
    original_frame_interval: tuple[int, int],
    intrinsics: np.ndarray,
    T_base_camera: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    min_valid_points: int,
    min_valid_fraction: float,
    min_frames: int,
    max_center_rms_m: float,
    max_center_deviation_m: float,
    max_point_count_cv: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    start, end = original_frame_interval
    if start < 0 or end <= start:
        raise ValueError("original final-frame interval must be non-empty and half-open")
    selected_original = [index for index in range(start, end) if index in source_to_proxy]
    if len(selected_original) < min_frames:
        raise ValueError(
            f"only {len(selected_original)} mask-mapped final frames; need {min_frames}"
        )

    frame_records: list[dict[str, Any]] = []
    points_by_frame: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    robust_mins: list[np.ndarray] = []
    robust_maxs: list[np.ndarray] = []
    mask_indices: list[int] = []
    original_indices: list[int] = []
    for original_index in selected_original:
        proxy_index = int(source_to_proxy[original_index])
        depth_path = depth_dir / f"{original_index:06d}.npy"
        mask_path = mask_dir / f"{proxy_index:06d}.png"
        if not depth_path.is_file() or not mask_path.is_file():
            raise ValueError(f"missing paired depth/mask for original frame {original_index}")
        depth = np.load(depth_path, allow_pickle=False)
        if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.floating):
            raise ValueError(f"depth frame {depth_path} must be a 2D floating metre array")
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None or raw_mask.shape != depth.shape:
            raise ValueError(f"mask {mask_path} is unreadable or does not match depth shape")
        mask = raw_mask > 0
        mask_pixels = int(np.count_nonzero(mask))
        if mask_pixels == 0:
            raise ValueError(f"mask {mask_path} is empty")
        points = _base_points(
            depth,
            mask,
            intrinsics,
            T_base_camera,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )
        valid_fraction = len(points) / mask_pixels
        if len(points) < min_valid_points or valid_fraction < min_valid_fraction:
            raise ValueError(
                f"original frame {original_index} has {len(points)} valid die points "
                f"({valid_fraction:.3f} of mask)"
            )
        center = np.median(points, axis=0)
        robust_min, robust_max = np.quantile(points, [0.01, 0.99], axis=0)
        points_by_frame.append(points)
        centers.append(center)
        robust_mins.append(robust_min)
        robust_maxs.append(robust_max)
        mask_indices.append(proxy_index)
        original_indices.append(original_index)
        frame_records.append(
            {
                "original_frame_index": original_index,
                "reverse_proxy_frame_index": proxy_index,
                "mask_pixel_count": mask_pixels,
                "valid_depth_point_count": len(points),
                "valid_depth_fraction_in_mask": valid_fraction,
                "center_base_m": center.tolist(),
                "robust_bounds_p01_p99_base_m": {
                    "min": robust_min.tolist(),
                    "max": robust_max.tolist(),
                },
            }
        )

    center_array = np.stack(centers)
    anchor = np.median(center_array, axis=0)
    residuals = np.linalg.norm(center_array - anchor, axis=1)
    center_rms = float(np.sqrt(np.mean(residuals**2)))
    center_max = float(np.max(residuals))
    counts = np.asarray([len(points) for points in points_by_frame], dtype=float)
    point_count_cv = float(np.std(counts) / np.mean(counts))
    if center_rms > max_center_rms_m:
        raise ValueError(
            f"final die center RMS {center_rms:.6f} m exceeds {max_center_rms_m:.6f} m"
        )
    if center_max > max_center_deviation_m:
        raise ValueError(
            f"final die max deviation {center_max:.6f} m exceeds {max_center_deviation_m:.6f} m"
        )
    if point_count_cv > max_point_count_cv:
        raise ValueError(
            f"final die point-count CV {point_count_cv:.6f} exceeds {max_point_count_cv:.6f}"
        )

    robust_min = np.median(np.stack(robust_mins), axis=0)
    robust_max = np.median(np.stack(robust_maxs), axis=0)
    offsets = np.zeros(len(points_by_frame) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(points) for points in points_by_frame])
    evidence = {
        "points_base_m": np.concatenate(points_by_frame),
        "frame_offsets": offsets,
        "original_frame_indices": np.asarray(original_indices, dtype=np.int64),
        "reverse_proxy_frame_indices": np.asarray(mask_indices, dtype=np.int64),
        "frame_centers_base_m": center_array,
        "frame_center_residuals_m": residuals,
    }
    result = {
        "final_die_anchor_base_m": anchor.tolist(),
        "anchor_semantics": (
            "coordinate-wise median of per-frame medians of valid aligned-depth points "
            "inside the final stationary die mask; visible-surface anchor, not volume center"
        ),
        "robust_bounds_p01_p99_base_m": {
            "min": robust_min.tolist(),
            "max": robust_max.tolist(),
            "extent": (robust_max - robust_min).tolist(),
        },
        "quality": {
            "accepted": True,
            "frame_count": len(frame_records),
            "center_rms_deviation_m": center_rms,
            "center_max_deviation_m": center_max,
            "center_residual_p95_m": float(np.quantile(residuals, 0.95)),
            "point_count_min": int(np.min(counts)),
            "point_count_median": float(np.median(counts)),
            "point_count_max": int(np.max(counts)),
            "point_count_cv": point_count_cv,
            "limits": {
                "min_frames": min_frames,
                "min_valid_points_per_frame": min_valid_points,
                "min_valid_fraction_in_mask": min_valid_fraction,
                "max_center_rms_m": max_center_rms_m,
                "max_center_deviation_m": max_center_deviation_m,
                "max_point_count_cv": max_point_count_cv,
            },
        },
        "frames": frame_records,
    }
    return result, evidence


def _pose_payload(T_base_tcp: np.ndarray, *, base_frame: str) -> dict[str, Any]:
    transform = validate_rigid_transform(T_base_tcp, name="T_base_tcp")
    quaternion = rotation_matrix_to_quaternion_xyzw(transform[:3, :3])
    return {
        "schema": POSE_SEQUENCE_SCHEMA,
        "base_frame": base_frame,
        "waypoints": [
            {
                "position_m": transform[:3, 3].tolist(),
                "orientation_xyzw": quaternion.tolist(),
            }
        ],
    }


def _linear_sequence_payload(
    start: np.ndarray,
    end: np.ndarray,
    *,
    base_frame: str,
    max_segment_m: float = 0.05,
) -> dict[str, Any]:
    first = validate_rigid_transform(start, name="T_base_tcp_start")
    last = validate_rigid_transform(end, name="T_base_tcp_end")
    if not np.allclose(first[:3, :3], last[:3, :3], atol=1e-8):
        raise ValueError("linear placement/retreat sequence must preserve orientation")
    distance = float(np.linalg.norm(last[:3, 3] - first[:3, 3]))
    segment_count = max(1, math.ceil(distance / max_segment_m))
    quaternion = rotation_matrix_to_quaternion_xyzw(first[:3, :3])
    waypoints = []
    poses = []
    for alpha in np.linspace(0.0, 1.0, segment_count + 1):
        position = (1.0 - alpha) * first[:3, 3] + alpha * last[:3, 3]
        waypoints.append({"position_m": position.tolist(), "orientation_xyzw": quaternion.tolist()})
        poses.append(CartesianPose(tuple(position), tuple(quaternion)))
    validate_pose_sequence_motion(poses[0], poses)
    return {
        "schema": POSE_SEQUENCE_SCHEMA,
        "base_frame": base_frame,
        "waypoints": waypoints,
    }


def ground_waypoints(
    *,
    grounded_pick: Mapping[str, Any],
    final_die_anchor_base_m: np.ndarray,
    transport_height_m: float,
    retreat_height_m: float,
    base_frame: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if grounded_pick.get("schema") not in GROUNDED_PICK_SCHEMAS:
        raise ValueError(f"grounded pick schema must be one of {sorted(GROUNDED_PICK_SCHEMAS)!r}")
    pick_anchor = np.asarray(grounded_pick.get("live_object_anchor_base_m"), dtype=float)
    if pick_anchor.shape != (3,) or not np.isfinite(pick_anchor).all():
        raise ValueError("grounded pick live_object_anchor_base_m must contain three values")
    candidate = grounded_pick.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("grounded pick has no candidate object")
    grasp = validate_rigid_transform(
        candidate.get("T_base_grasp_tcp"), name="grounded pick T_base_grasp_tcp"
    )
    target_anchor = np.asarray(final_die_anchor_base_m, dtype=float)
    if target_anchor.shape != (3,) or not np.isfinite(target_anchor).all():
        raise ValueError("final die anchor must contain three finite values")

    anchor_delta = target_anchor - pick_anchor
    placement = grasp.copy()
    placement[:3, 3] += anchor_delta
    transport = placement.copy()
    transport[2, 3] += transport_height_m
    retreat = placement.copy()
    retreat[2, 3] += retreat_height_m
    transforms = {
        "transport_tcp": transport,
        "placement_tcp": placement,
        "retreat_tcp": retreat,
    }
    payloads = {
        name: _pose_payload(transform, base_frame=base_frame)
        for name, transform in transforms.items()
    }
    payloads["place_waypoints"] = _linear_sequence_payload(
        transport, placement, base_frame=base_frame
    )
    payloads["retreat_waypoints"] = _linear_sequence_payload(
        placement, retreat, base_frame=base_frame
    )
    grounding = {
        "schema": GROUNDING_SCHEMA,
        "status": "review_only_plan_inputs",
        "execution_authorized": False,
        "robot_execution_ready": False,
        "base_frame": base_frame,
        "policy": "translate the actual grounded grasp TCP with the object anchor",
        "pick_object_anchor_base_m": pick_anchor.tolist(),
        "fixed_scene_target_die_anchor_base_m": target_anchor.tolist(),
        "anchor_translation_base_m": anchor_delta.tolist(),
        "T_base_grasp_tcp": grasp.tolist(),
        "T_base_placement_tcp": placement.tolist(),
        "T_base_transport_tcp": transport.tolist(),
        "T_base_retreat_tcp": retreat.tolist(),
        "transport_height_m": transport_height_m,
        "retreat_height_m": retreat_height_m,
        "orientation_policy": "preserve actual grounded grasp TCP orientation",
        "release_policy": "open only after inside/on-bottom predicate passes",
        "limitations": [
            "No live box pose or box orientation was estimated.",
            "Targets are valid only while the camera calibration and physical box remain fixed.",
            "The files are MoveIt plan-only inputs until reachability and collision checks pass.",
        ],
    }
    return grounding, payloads


def main() -> int:
    args = parse_args()
    _validate_quality_args(args)
    repo_root = Path(__file__).resolve().parents[1]
    demo_dir = _resolve(args.demo_dir, repo_root=repo_root)
    input_dir = demo_dir / "input"
    mask_dir = _resolve(
        args.mask_dir or demo_dir / "sam2_die_reverse_proxy" / "masks" / "2",
        repo_root=repo_root,
    )
    reverse_provenance = _resolve(
        args.reverse_provenance
        or demo_dir / "derived" / "reverse_pick_proxy_0650_to_0440" / "provenance.json",
        repo_root=repo_root,
    )
    t_base_camera_path = _resolve(args.t_base_camera, repo_root=repo_root)
    output_dir = _resolve(
        args.output_dir or demo_dir / "multicontact_plan" / "inputs",
        repo_root=repo_root,
    )
    required = (
        input_dir / "K.npy",
        input_dir / "timestamps_s.npy",
        reverse_provenance,
        t_base_camera_path,
    )
    for path in required:
        if not path.is_file():
            raise ValueError(f"required input does not exist: {path}")
    if not mask_dir.is_dir():
        raise ValueError(f"mask directory does not exist: {mask_dir}")

    intrinsics = validate_intrinsics(np.load(input_dir / "K.npy", allow_pickle=False))
    T_base_camera = validate_rigid_transform(
        np.load(t_base_camera_path, allow_pickle=False), name="T_base_camera"
    )
    provenance = _read_json(reverse_provenance)
    source_to_proxy = _mask_source_mapping(provenance)
    input_manifest_path = input_dir / "input_manifest.json"
    mask_manifest_path = mask_dir.parent.parent / "manifest.json"
    for path in (input_manifest_path, mask_manifest_path):
        if not path.is_file():
            raise ValueError(f"required provenance manifest does not exist: {path}")
    start, end = (int(value) for value in args.original_final_frames)
    result, evidence = infer_final_anchor(
        depth_dir=input_dir / "depth",
        mask_dir=mask_dir,
        source_to_proxy=source_to_proxy,
        original_frame_interval=(start, end),
        intrinsics=intrinsics,
        T_base_camera=T_base_camera,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        min_valid_points=args.min_valid_points,
        min_valid_fraction=args.min_valid_fraction,
        min_frames=args.min_frames,
        max_center_rms_m=args.max_center_rms_m,
        max_center_deviation_m=args.max_center_deviation_m,
        max_point_count_cv=args.max_point_count_cv,
    )

    relation_path = output_dir / "box_placement_relation.json"
    evidence_path = output_dir / "final_die_evidence.npz"
    relation = {
        "schema": RELATION_SCHEMA,
        "status": "accepted_fixed_scene_translation_only",
        "artifact_role": "review_only_fixed_scene_placement_anchor",
        "execution_authorized": False,
        "robot_execution_ready": False,
        "base_frame": args.base_frame,
        "relation_mode": "fixed_scene_translation_only",
        **result,
        "placement_contract": {
            "formula": (
                "p_base_tcp_place = p_base_tcp_grasp + "
                "(p_base_die_final_demo - p_base_die_pick_live)"
            ),
            "orientation": "preserve actual grounded grasp TCP orientation",
            "pick_anchor_requirement": (
                "live_object_anchor_base_m must use the same visible-surface "
                "translation-anchor semantics"
            ),
            "transport_offset_base_m": [0.0, 0.0, args.transport_height_m],
            "retreat_offset_base_m": [0.0, 0.0, args.retreat_height_m],
            "release_guard": ["die_inside_box", "die_supported_by_box_bottom"],
        },
        "fixed_scene_assumptions": [
            "The external camera and T_base_camera have not moved since the demonstration.",
            "The black box has not moved since the final demonstrated placement.",
            "Only the final die translation anchor is transferred; no T_base_box is claimed.",
        ],
        "provenance": {
            "demo_dir": _display(demo_dir, repo_root=repo_root),
            "depth_dir": _display(input_dir / "depth", repo_root=repo_root),
            "mask_dir": _display(mask_dir, repo_root=repo_root),
            "reverse_proxy_provenance": _display(reverse_provenance, repo_root=repo_root),
            "reverse_proxy_provenance_sha256": _sha256(reverse_provenance),
            "reverse_proxy_not_original_pickup": True,
            "original_frame_interval": [start, end],
            "input_manifest": _display(input_manifest_path, repo_root=repo_root),
            "input_manifest_sha256": _sha256(input_manifest_path),
            "mask_manifest": _display(mask_manifest_path, repo_root=repo_root),
            "mask_manifest_sha256": _sha256(mask_manifest_path),
            "intrinsics": _display(input_dir / "K.npy", repo_root=repo_root),
            "intrinsics_sha256": _sha256(input_dir / "K.npy"),
            "timestamps": _display(input_dir / "timestamps_s.npy", repo_root=repo_root),
            "timestamps_sha256": _sha256(input_dir / "timestamps_s.npy"),
            "T_base_camera": _display(t_base_camera_path, repo_root=repo_root),
            "T_base_camera_sha256": _sha256(t_base_camera_path),
            "evidence_npz": _display(evidence_path, repo_root=repo_root),
        },
        "limitations": [
            "The anchor is a visible-surface statistic, not the die geometric center.",
            "The box pose and orientation are unresolved.",
            "Moving the camera or box invalidates this relation.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if evidence_path.exists() and not args.overwrite:
        raise ValueError(f"output already exists; pass --overwrite: {evidence_path}")
    np.savez_compressed(evidence_path, **evidence)
    relation["provenance"]["evidence_npz_sha256"] = _sha256(evidence_path)
    _write_json(relation_path, relation, overwrite=args.overwrite)

    generated = [relation_path, evidence_path]
    grounding_path: Path | None = None
    if args.grounded_pick is not None:
        grounded_pick_path = _resolve(args.grounded_pick, repo_root=repo_root)
        grounded_pick = _read_json(grounded_pick_path)
        grounding, payloads = ground_waypoints(
            grounded_pick=grounded_pick,
            final_die_anchor_base_m=np.asarray(relation["final_die_anchor_base_m"]),
            transport_height_m=args.transport_height_m,
            retreat_height_m=args.retreat_height_m,
            base_frame=args.base_frame,
        )
        grounding["provenance"] = {
            "placement_relation": _display(relation_path, repo_root=repo_root),
            "placement_relation_sha256": _sha256(relation_path),
            "grounded_pick": _display(grounded_pick_path, repo_root=repo_root),
            "grounded_pick_sha256": _sha256(grounded_pick_path),
        }
        placement_dir = output_dir / "placement"
        for name, payload in payloads.items():
            path = placement_dir / f"{name}.json"
            _write_json(path, payload, overwrite=args.overwrite)
            generated.append(path)
        grounding_path = placement_dir / "placement_grounding.json"
        _write_json(grounding_path, grounding, overwrite=args.overwrite)
        generated.append(grounding_path)

    manifest = {
        "schema": "demo2gripper.demo03_fixed_scene_placement_manifest.v1",
        "status": "review_only_plan_inputs" if grounding_path else "relation_ready",
        "execution_authorized": False,
        "robot_execution_ready": False,
        "relation": _display(relation_path, repo_root=repo_root),
        "grounded_pick_supplied": grounding_path is not None,
        "grounding": (
            None if grounding_path is None else _display(grounding_path, repo_root=repo_root)
        ),
        "runtime_grounding_command_template": [
            ".venv/bin/python",
            "scripts/build_demo03_fixed_scene_placement.py",
            "--demo-dir",
            _display(demo_dir, repo_root=repo_root),
            "--grounded-pick",
            "<fresh-live-pick-run>/execution_candidate.json",
            "--overwrite",
        ],
        "deferred_waypoint_outputs": [
            "placement/transport_tcp.json",
            "placement/place_waypoints.json",
            "placement/placement_tcp.json",
            "placement/retreat_waypoints.json",
            "placement/retreat_tcp.json",
        ],
        "generated_artifacts": [_display(path, repo_root=repo_root) for path in generated],
    }
    manifest_path = output_dir / "box_placement_manifest.json"
    _write_json(manifest_path, manifest, overwrite=args.overwrite)
    anchor = np.asarray(relation["final_die_anchor_base_m"])
    print(f"wrote {relation_path}")
    print(f"final die anchor base [m]: {anchor.tolist()}")
    print(
        "quality: "
        f"frames={relation['quality']['frame_count']}, "
        f"rms={relation['quality']['center_rms_deviation_m'] * 1000.0:.3f} mm, "
        f"max={relation['quality']['center_max_deviation_m'] * 1000.0:.3f} mm"
    )
    if grounding_path is None:
        print("waypoints deferred: pass --grounded-pick after the live pick is grounded")
    else:
        print(f"wrote plan-only placement inputs under {grounding_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
