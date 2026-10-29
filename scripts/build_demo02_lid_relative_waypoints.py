#!/usr/bin/env python3
"""Compile demo-02 hand/lid geometry into reviewable TCP waypoint files.

This is an offline geometry compiler.  It never imports ROS or talks to the FR3.
The contact pose comes from the demonstrated hand frame: canonical TCP ``+X`` is
thumb-to-index and ``+Z`` is wrist-to-pinch.  The pose is stored relative to a
closed-lid contact frame and is then carried around the supplied lid hinge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from demo2gripper.cartesian_pose_sequence import (
    MAX_SEGMENT_ROTATION_RAD,
    MAX_WAYPOINTS,
    POSE_SEQUENCE_SCHEMA,
    CartesianPose,
    validate_pose_sequence_motion,
)
from demo2gripper.franka import validate_rigid_transform
from demo2gripper.geometry import normalize, robust_direction_mean, robust_mean
from demo2gripper.intent import INDEX_TIP, THUMB_TIP, WRIST

RELATION_SCHEMA = "demo2gripper.demo02_lid_interaction_relation.v1"
MANIFEST_SCHEMA = "demo2gripper.demo02_lid_relative_waypoints_manifest.v1"
HANDE_MAX_OPENING_M = 0.050


@dataclass(frozen=True)
class LidRelativeArtifacts:
    """In-memory result of the offline demo-02 lid geometry compilation."""

    T_base_lid_contact_closed: np.ndarray
    T_base_demonstrated_tcp: np.ndarray
    T_lid_contact_tcp: np.ndarray
    T_base_precontact_tcp: np.ndarray
    T_base_contact_tcp: np.ndarray
    T_base_open_tcp: tuple[np.ndarray, ...]
    T_base_close_tcp: tuple[np.ndarray, ...]
    T_base_retreat_tcp: np.ndarray
    signed_open_angles_rad: tuple[float, ...]
    demonstrated_pinch_width_m: float
    hande_opening_m: float
    closing_axis_flipped: bool
    quality: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--hand-joints", required=True, type=Path)
    parser.add_argument("--hand-confidence", required=True, type=Path)
    parser.add_argument("--t-base-camera", required=True, type=Path)
    parser.add_argument(
        "--contact-frames",
        required=True,
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Half-open demonstrated closed-lid contact interval [START, END)",
    )
    parser.add_argument(
        "--closed-lid-contact-base-m",
        required=True,
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Origin of the rigid lid-contact frame while the lid is closed",
    )
    parser.add_argument(
        "--lid-hinge-base-m",
        required=True,
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="A point on the lid hinge axis",
    )
    parser.add_argument(
        "--lid-positive-y-base",
        type=float,
        nargs=3,
        default=(0.0, 1.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Signed +Y direction of the lid-contact frame and hinge",
    )
    parser.add_argument("--open-angle-deg", required=True, type=float)
    parser.add_argument("--angle-step-deg", type=float, default=10.0)
    parser.add_argument("--precontact-retreat-m", type=float, default=0.070)
    parser.add_argument("--min-joint-confidence", type=float, default=0.25)
    parser.add_argument("--min-support-frames", type=int, default=5)
    parser.add_argument(
        "--flip-closing-axis",
        action="store_true",
        help="Use the equivalent Hand-E candidate T_tcp@Rz(pi), flipping TCP +X/+Y",
    )
    parser.add_argument("--base-frame", default="fr3_link0")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return vector


def _angle_errors_deg(directions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip(unit @ reference, -1.0, 1.0)))


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis, name="hinge axis")
    return Rotation.from_rotvec(axis * angle_rad).as_matrix()


def _closed_lid_contact_frame(
    contact_base_m: np.ndarray,
    hinge_base_m: np.ndarray,
    positive_y_base: np.ndarray,
) -> np.ndarray:
    """Make a right-handed closed-lid frame with +X pointing away from the hinge."""

    contact = _finite_vector(contact_base_m, name="closed lid contact")
    hinge = _finite_vector(hinge_base_m, name="lid hinge")
    y_axis = normalize(
        _finite_vector(positive_y_base, name="lid positive Y"),
        name="lid positive Y",
    )
    radial = contact - hinge
    x_axis = normalize(
        radial - float(np.dot(radial, y_axis)) * y_axis,
        name="hinge-to-contact radial direction",
    )
    z_axis = normalize(np.cross(x_axis, y_axis), name="closed lid normal")
    y_axis = normalize(np.cross(z_axis, x_axis), name="lid positive Y")

    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = contact
    return validate_rigid_transform(transform, name="T_base_lid_contact_closed")


def lid_contact_pose_at_angle(
    T_base_lid_contact_closed: np.ndarray,
    hinge_base_m: np.ndarray,
    hinge_axis_base: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """Rotate the rigid lid-contact frame around its fixed base-frame hinge."""

    closed = validate_rigid_transform(T_base_lid_contact_closed, name="T_base_lid_contact_closed")
    hinge = _finite_vector(hinge_base_m, name="lid hinge")
    delta = _axis_angle_rotation(hinge_axis_base, float(angle_rad))
    result = np.eye(4)
    result[:3, :3] = delta @ closed[:3, :3]
    result[:3, 3] = hinge + delta @ (closed[:3, 3] - hinge)
    return validate_rigid_transform(result, name="T_base_lid_contact(theta)")


def _open_angles(
    *,
    T_base_lid_contact_closed: np.ndarray,
    hinge_base_m: np.ndarray,
    hinge_axis_base: np.ndarray,
    open_angle_deg: float,
    angle_step_deg: float,
) -> tuple[float, ...]:
    if not math.isfinite(open_angle_deg) or not 0.0 < open_angle_deg < 180.0:
        raise ValueError("open_angle_deg must lie in (0, 180)")
    max_step_deg = math.degrees(MAX_SEGMENT_ROTATION_RAD)
    if not math.isfinite(angle_step_deg) or not 0.0 < angle_step_deg <= max_step_deg:
        raise ValueError(f"angle_step_deg must lie in (0, {max_step_deg:.1f}]")

    magnitude_rad = math.radians(open_angle_deg)
    plus = lid_contact_pose_at_angle(
        T_base_lid_contact_closed,
        hinge_base_m,
        hinge_axis_base,
        magnitude_rad,
    )
    minus = lid_contact_pose_at_angle(
        T_base_lid_contact_closed,
        hinge_base_m,
        hinge_axis_base,
        -magnitude_rad,
    )
    closed_z = float(T_base_lid_contact_closed[2, 3])
    plus_rise = float(plus[2, 3] - closed_z)
    minus_rise = float(minus[2, 3] - closed_z)
    if max(plus_rise, minus_rise) <= 1e-6:
        raise ValueError("neither hinge rotation direction raises the lid contact along base +Z")
    sign = 1.0 if plus_rise > minus_rise else -1.0

    values_deg = [0.0]
    next_angle = angle_step_deg
    while next_angle < open_angle_deg - 1e-9:
        values_deg.append(next_angle)
        next_angle += angle_step_deg
    values_deg.append(open_angle_deg)
    if len(values_deg) > MAX_WAYPOINTS:
        raise ValueError(f"open sequence exceeds the {MAX_WAYPOINTS}-waypoint cap")
    return tuple(sign * math.radians(value) for value in values_deg)


def build_lid_relative_waypoints(
    *,
    hand_joints_camera_m: np.ndarray,
    hand_joint_confidence: np.ndarray,
    T_base_camera: np.ndarray,
    contact_frame_interval: tuple[int, int],
    closed_lid_contact_base_m: np.ndarray,
    lid_hinge_base_m: np.ndarray,
    lid_positive_y_base: np.ndarray | tuple[float, float, float] = (0.0, 1.0, 0.0),
    open_angle_deg: float,
    angle_step_deg: float,
    precontact_retreat_m: float,
    min_joint_confidence: float = 0.25,
    min_support_frames: int = 5,
    flip_closing_axis: bool = False,
) -> LidRelativeArtifacts:
    """Infer the demonstrated lid-relative TCP and ground the hinged path."""

    joints = np.asarray(hand_joints_camera_m, dtype=float)
    confidence = np.asarray(hand_joint_confidence, dtype=float)
    if joints.ndim != 3 or joints.shape[1:] != (21, 3):
        raise ValueError(f"hand joints must have shape (F, 21, 3), got {joints.shape}")
    if confidence.shape != joints.shape[:2]:
        raise ValueError("hand confidence must have shape (F, 21) matching the hand joints")
    frame_count = len(joints)
    start, end = (int(value) for value in contact_frame_interval)
    if not 0 <= start < end <= frame_count:
        raise ValueError(
            "contact_frame_interval must be a non-empty half-open interval "
            f"inside [0, {frame_count}]"
        )
    if not 0.0 <= min_joint_confidence <= 1.0:
        raise ValueError("min_joint_confidence must lie in [0, 1]")
    if min_support_frames < 3:
        raise ValueError("min_support_frames must be at least 3")
    if not math.isfinite(precontact_retreat_m) or precontact_retreat_m <= 0.0:
        raise ValueError("precontact_retreat_m must be positive and finite")

    base_camera = validate_rigid_transform(T_base_camera, name="T_base_camera")
    critical_indices = np.arange(start, end)
    critical_joints = joints[start:end]
    critical_confidence = confidence[start:end]
    key_joints = np.array([WRIST, THUMB_TIP, INDEX_TIP])
    eligible = np.isfinite(critical_joints[:, key_joints]).all(axis=(1, 2))
    eligible &= np.isfinite(critical_confidence[:, key_joints]).all(axis=1)
    eligible &= np.all(critical_confidence[:, key_joints] >= min_joint_confidence, axis=1)
    selected_indices = critical_indices[eligible]
    selected = critical_joints[eligible]
    if len(selected) < min_support_frames:
        raise ValueError(
            f"contact interval has only {len(selected)} confidence-valid frames; "
            f"need {min_support_frames}"
        )

    rotation = base_camera[:3, :3]
    translation = base_camera[:3, 3]
    selected_base = np.einsum("ij,fkj->fki", rotation, selected) + translation
    wrist = selected_base[:, WRIST]
    thumb = selected_base[:, THUMB_TIP]
    index = selected_base[:, INDEX_TIP]
    pinch = 0.5 * (thumb + index)
    closing_raw = index - thumb
    approach_raw = pinch - wrist

    closing_norm = np.linalg.norm(closing_raw, axis=1)
    nondegenerate = closing_norm > 1e-6
    closing_unit = np.zeros_like(closing_raw)
    closing_unit[nondegenerate] = closing_raw[nondegenerate] / closing_norm[nondegenerate, None]
    approach_projected = approach_raw - (
        np.sum(approach_raw * closing_unit, axis=1, keepdims=True) * closing_unit
    )
    nondegenerate &= np.linalg.norm(approach_projected, axis=1) > 1e-6
    if np.count_nonzero(nondegenerate) < min_support_frames:
        raise ValueError(
            "too few contact frames define non-degenerate, orthogonalized "
            "thumb/index and wrist/pinch axes"
        )

    selected_indices = selected_indices[nondegenerate]
    selected_confidence = critical_confidence[eligible][nondegenerate][:, key_joints]
    thumb = thumb[nondegenerate]
    index = index[nondegenerate]
    pinch = pinch[nondegenerate]
    closing_raw = closing_raw[nondegenerate]
    approach_raw = approach_raw[nondegenerate]
    closing_unit = closing_unit[nondegenerate]
    approach_projected = approach_projected[nondegenerate]
    approach_unit = approach_projected / np.linalg.norm(approach_projected, axis=1, keepdims=True)

    # Aggregate only after every source frame defines a proper orthogonal pair.
    x_reference = robust_direction_mean(closing_unit, name="thumb-to-index +X")
    z_axis = robust_direction_mean(approach_unit, name="wrist-to-pinch +Z")
    x_axis = normalize(
        x_reference - float(np.dot(x_reference, z_axis)) * z_axis,
        name="canonical TCP +X",
    )
    y_axis = normalize(np.cross(z_axis, x_axis), name="canonical TCP +Y")
    x_axis = normalize(np.cross(y_axis, z_axis), name="canonical TCP +X")
    if float(np.dot(x_axis, x_reference)) < 0.0:
        x_axis *= -1.0
        y_axis *= -1.0

    T_base_demonstrated_tcp = np.eye(4)
    T_base_demonstrated_tcp[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    T_base_demonstrated_tcp[:3, 3] = robust_mean(pinch)
    if flip_closing_axis:
        # A parallel-jaw closing line is unsigned.  Right multiplication keeps
        # the demonstrated TCP origin and +Z approach while reversing +X/+Y.
        T_tcp_flipped_tcp = np.diag([-1.0, -1.0, 1.0, 1.0])
        T_base_demonstrated_tcp = T_base_demonstrated_tcp @ T_tcp_flipped_tcp
    T_base_demonstrated_tcp = validate_rigid_transform(
        T_base_demonstrated_tcp, name="T_base_demonstrated_tcp"
    )

    T_base_lid_contact_closed = _closed_lid_contact_frame(
        closed_lid_contact_base_m,
        lid_hinge_base_m,
        lid_positive_y_base,
    )
    T_lid_contact_tcp = np.linalg.inv(T_base_lid_contact_closed) @ T_base_demonstrated_tcp
    T_lid_contact_tcp = validate_rigid_transform(T_lid_contact_tcp, name="T_lid_contact_tcp")

    angles = _open_angles(
        T_base_lid_contact_closed=T_base_lid_contact_closed,
        hinge_base_m=lid_hinge_base_m,
        hinge_axis_base=T_base_lid_contact_closed[:3, 1],
        open_angle_deg=open_angle_deg,
        angle_step_deg=angle_step_deg,
    )
    open_poses = tuple(
        lid_contact_pose_at_angle(
            T_base_lid_contact_closed,
            lid_hinge_base_m,
            T_base_lid_contact_closed[:3, 1],
            theta,
        )
        @ T_lid_contact_tcp
        for theta in angles
    )
    for index_value, pose in enumerate(open_poses):
        validate_rigid_transform(pose, name=f"T_base_tcp(open[{index_value}])")
    close_poses = tuple(pose.copy() for pose in reversed(open_poses))
    contact_pose = open_poses[0].copy()
    retreat_local = np.eye(4)
    retreat_local[2, 3] = -precontact_retreat_m
    precontact_pose = contact_pose @ retreat_local
    retreat_pose = precontact_pose.copy()

    demonstrated_widths = np.linalg.norm(index - thumb, axis=1)
    demonstrated_width = float(np.median(demonstrated_widths))
    hande_opening = min(max(demonstrated_width, 0.0), HANDE_MAX_OPENING_M)

    center = T_base_demonstrated_tcp[:3, 3]
    center_residuals = np.linalg.norm(pinch - center, axis=1)
    x_errors = _angle_errors_deg(closing_unit, x_axis)
    z_errors = _angle_errors_deg(approach_unit, z_axis)
    raw_axis_cosine = np.abs(
        np.sum(
            closing_unit * (approach_raw / np.linalg.norm(approach_raw, axis=1, keepdims=True)),
            axis=1,
        )
    )
    quality = {
        "contact_interval_frames": end - start,
        "valid_support_frames": len(selected_indices),
        "valid_support_fraction": float(len(selected_indices) / (end - start)),
        "valid_frame_indices": selected_indices.tolist(),
        "min_selected_joint_confidence": float(np.min(selected_confidence)),
        "median_selected_joint_confidence": float(np.median(selected_confidence)),
        "pinch_center_residual_median_m": float(np.median(center_residuals)),
        "pinch_center_residual_p95_m": float(np.quantile(center_residuals, 0.95)),
        "closing_axis_error_median_deg": float(np.median(x_errors)),
        "closing_axis_error_p95_deg": float(np.quantile(x_errors, 0.95)),
        "approach_axis_error_median_deg": float(np.median(z_errors)),
        "approach_axis_error_p95_deg": float(np.quantile(z_errors, 0.95)),
        "raw_closing_approach_abs_cosine_p95": float(np.quantile(raw_axis_cosine, 0.95)),
        "orthogonalized_closing_approach_abs_dot_max": float(
            np.max(np.abs(np.sum(closing_unit * approach_unit, axis=1)))
        ),
        "pinch_width_median_m": demonstrated_width,
        "pinch_width_mad_m": float(np.median(np.abs(demonstrated_widths - demonstrated_width))),
        "hande_opening_clamped": bool(demonstrated_width > HANDE_MAX_OPENING_M),
        "closed_contact_reconstruction_error_m": float(
            np.linalg.norm(contact_pose[:3, 3] - T_base_demonstrated_tcp[:3, 3])
        ),
        "closed_contact_reconstruction_rotation_error_deg": float(
            np.rad2deg(
                Rotation.from_matrix(
                    contact_pose[:3, :3].T @ T_base_demonstrated_tcp[:3, :3]
                ).magnitude()
            )
        ),
    }

    return LidRelativeArtifacts(
        T_base_lid_contact_closed=T_base_lid_contact_closed,
        T_base_demonstrated_tcp=T_base_demonstrated_tcp,
        T_lid_contact_tcp=T_lid_contact_tcp,
        T_base_precontact_tcp=precontact_pose,
        T_base_contact_tcp=contact_pose,
        T_base_open_tcp=open_poses,
        T_base_close_tcp=close_poses,
        T_base_retreat_tcp=retreat_pose,
        signed_open_angles_rad=angles,
        demonstrated_pinch_width_m=demonstrated_width,
        hande_opening_m=hande_opening,
        closing_axis_flipped=bool(flip_closing_axis),
        quality=quality,
    )


def _pose(transform: np.ndarray) -> CartesianPose:
    checked = validate_rigid_transform(transform, name="T_base_tcp")
    quaternion = Rotation.from_matrix(checked[:3, :3]).as_quat()
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return CartesianPose(
        tuple(float(value) for value in checked[:3, 3]),
        tuple(float(value) for value in quaternion),
    )


def _pose_payload(transforms: tuple[np.ndarray, ...], *, base_frame: str) -> dict[str, Any]:
    return {
        "schema": POSE_SEQUENCE_SCHEMA,
        "base_frame": base_frame,
        "waypoints": [_pose(transform).as_dict() for transform in transforms],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_lid_relative_waypoints(
    artifacts: LidRelativeArtifacts,
    output_dir: Path,
    *,
    base_frame: str,
    contact_frame_interval: tuple[int, int],
    precontact_retreat_m: float,
    input_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Write the relation, strict pose sequences, and review manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pose_transforms = {
        "lid_precontact_tcp.json": (artifacts.T_base_precontact_tcp,),
        "lid_contact_tcp.json": (artifacts.T_base_contact_tcp,),
        "lid_open_waypoints.json": artifacts.T_base_open_tcp,
        "lid_close_waypoints.json": artifacts.T_base_close_tcp,
        "lid_final_retreat_tcp.json": (artifacts.T_base_retreat_tcp,),
    }
    for filename, transforms in pose_transforms.items():
        _write_json(
            output_dir / filename,
            _pose_payload(transforms, base_frame=base_frame),
        )

    # Exercise the same bounded sequence validator used by the ROS-side tool.
    contact = _pose(artifacts.T_base_contact_tcp)
    open_motion = validate_pose_sequence_motion(
        contact, [_pose(value) for value in artifacts.T_base_open_tcp]
    )
    close_motion = validate_pose_sequence_motion(
        _pose(artifacts.T_base_open_tcp[-1]),
        [_pose(value) for value in artifacts.T_base_close_tcp],
    )

    relation = {
        "schema": RELATION_SCHEMA,
        "base_frame": base_frame,
        "convention": {
            "transforms": "T_parent_child; metres; column vectors",
            "canonical_tcp": "+X thumb-to-index closing; +Z wrist-to-pinch approach",
            "lid_contact_frame": "+X hinge-to-contact; +Y supplied hinge direction; +Z=+X cross +Y",
        },
        "contact_frame_interval": list(contact_frame_interval),
        "T_base_lid_contact_closed": artifacts.T_base_lid_contact_closed.tolist(),
        "T_base_demonstrated_tcp": artifacts.T_base_demonstrated_tcp.tolist(),
        "T_lid_contact_tcp": artifacts.T_lid_contact_tcp.tolist(),
        "kinematic_rule": "T_base_tcp(theta)=T_base_lid_contact(theta)@T_lid_contact_tcp",
        "hande_embodiment_candidate": {
            "closing_axis_flipped": artifacts.closing_axis_flipped,
            "rule": "optional T_base_tcp=T_base_tcp_demonstrated@Rz(pi)",
            "effect": "TCP +X/+Y reverse; TCP +Z and position remain unchanged",
        },
        "signed_open_angles_deg": [
            math.degrees(value) for value in artifacts.signed_open_angles_rad
        ],
        "precontact": {
            "retreat_distance_m": float(precontact_retreat_m),
            "rule": "T_base_precontact_tcp=T_base_contact_tcp@Trans(0,0,-distance)",
        },
        "hande": {
            "demonstrated_pinch_width_m": artifacts.demonstrated_pinch_width_m,
            "command_opening_m": artifacts.hande_opening_m,
            "maximum_opening_m": HANDE_MAX_OPENING_M,
            "opening_changes_tcp_transform": False,
        },
    }
    relation_path = output_dir / "lid_interaction_relation.json"
    _write_json(relation_path, relation)

    resolved_inputs: dict[str, Any] = {}
    if input_paths is not None:
        for name, raw_path in input_paths.items():
            path = raw_path.expanduser().resolve()
            resolved_inputs[name] = {
                "path": str(path),
                "sha256": _sha256(path),
            }
    output_names = ["lid_interaction_relation.json", *pose_transforms]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "offline_geometry_only_no_ros_no_robot_commands",
        "base_frame": base_frame,
        "inputs": resolved_inputs,
        "hande_embodiment_candidate": {
            "closing_axis_flipped": artifacts.closing_axis_flipped,
            "right_multiply": "Rz(pi)" if artifacts.closing_axis_flipped else "identity",
        },
        "outputs": {
            name: {"path": str((output_dir / name).resolve()), "sha256": _sha256(output_dir / name)}
            for name in output_names
        },
        "quality": artifacts.quality,
        "motion_validation": {"open": open_motion, "close": close_motion},
        "assumptions": [
            "T_base_camera is the fixed-camera calibration valid for this demonstration.",
            "The supplied interval shows contact with the lid still in its closed pose.",
            "The lid is rigid and rotates about the supplied fixed hinge axis.",
            "The opening sign is selected so the lid-contact origin rises along base +Z.",
            "The human pinch center is used as the canonical Hand-E TCP proxy.",
            (
                "The parallel-jaw closing line is unsigned; --flip-closing-axis "
                "selects the equivalent T_tcp@Rz(pi) Hand-E embodiment candidate."
            ),
            "The Hand-E opening is capped at 0.050 m without translating or rotating the TCP.",
            "The close sequence is the exact reverse of the generated open sequence.",
        ],
        "limitations": [
            "No IK, collision, force, contact, or robot execution check is performed.",
            "The hinge and lid-contact coordinates are operator-supplied geometry.",
            (
                "Hand tracking errors and eye-to-hand calibration errors propagate "
                "into T_lid_contact_tcp."
            ),
        ],
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    joints = np.load(args.hand_joints)
    confidence = np.load(args.hand_confidence)
    T_base_camera = np.load(args.t_base_camera)
    interval = tuple(args.contact_frames)
    artifacts = build_lid_relative_waypoints(
        hand_joints_camera_m=joints,
        hand_joint_confidence=confidence,
        T_base_camera=T_base_camera,
        contact_frame_interval=(int(interval[0]), int(interval[1])),
        closed_lid_contact_base_m=np.asarray(args.closed_lid_contact_base_m),
        lid_hinge_base_m=np.asarray(args.lid_hinge_base_m),
        lid_positive_y_base=np.asarray(args.lid_positive_y_base),
        open_angle_deg=args.open_angle_deg,
        angle_step_deg=args.angle_step_deg,
        precontact_retreat_m=args.precontact_retreat_m,
        min_joint_confidence=args.min_joint_confidence,
        min_support_frames=args.min_support_frames,
        flip_closing_axis=args.flip_closing_axis,
    )
    manifest = write_lid_relative_waypoints(
        artifacts,
        args.output_dir,
        base_frame=args.base_frame,
        contact_frame_interval=(int(interval[0]), int(interval[1])),
        precontact_retreat_m=args.precontact_retreat_m,
        input_paths={
            "hand_joints": args.hand_joints,
            "hand_confidence": args.hand_confidence,
            "T_base_camera": args.t_base_camera,
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
