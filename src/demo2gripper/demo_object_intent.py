"""Build a translation-tracked no-CAD object template and hand grasp intent.

This module is the bridge between the validated pickup detector and the
object-frame GraspGenX candidate-selection contract.  It deliberately does
not project fingertips onto the single-view object cloud: the raw tracked
thumb, index finger, and wrist geometry carries the demonstrated relation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import normalize, robust_direction_mean, robust_mean
from .intent import INDEX_TIP, THUMB_TIP, WRIST
from .models import GraspIntent, JsonSerializable

OBJECT_FRAME_CONVENTION = (
    "origin=reference_object_center; axes=demonstration_opencv_color_camera; "
    "right_handed; metres; translation_tracking_only"
)


@dataclass(frozen=True)
class DemoObjectIntentConfig(JsonSerializable):
    """Confidence and support gates for the explicit contact interval."""

    min_tip_confidence: float = 0.4
    min_wrist_confidence: float = 0.25
    min_support_frames: int = 3
    critical_grasp_tail_frames: int | None = None

    def __post_init__(self) -> None:
        for name in ("min_tip_confidence", "min_wrist_confidence"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.min_support_frames < 1:
            raise ValueError("min_support_frames must be positive")
        if (
            self.critical_grasp_tail_frames is not None
            and self.critical_grasp_tail_frames < 1
        ):
            raise ValueError("critical_grasp_tail_frames must be positive when set")


@dataclass(frozen=True)
class DemoObjectIntentArtifacts:
    """In-memory outputs and diagnostics for one demonstration."""

    object_points_m: np.ndarray
    intent: GraspIntent
    T_camera_object_template: np.ndarray
    object_centers_camera_m: np.ndarray
    hand_joints_camera_m: np.ndarray
    hand_joints_object_m: np.ndarray
    hand_joint_confidence: np.ndarray
    validated_contact_frame_mask: np.ndarray
    valid_contact_frame_mask: np.ndarray
    timestamps_s: np.ndarray
    validated_contact_interval: tuple[int, int]
    critical_grasp_interval: tuple[int, int]
    quality: dict[str, Any]


def _resolve_contact_intervals(
    pickup: Mapping[str, Any],
    *,
    frame_count: int,
    critical_grasp_tail_frames: int | None,
) -> tuple[tuple[int, int], tuple[int, int], str]:
    try:
        validated = tuple(int(value) for value in pickup["stable_contact_interval"])
    except (KeyError, TypeError) as error:
        raise ValueError("pickup must contain stable_contact_interval") from error
    if len(validated) != 2:
        raise ValueError("stable_contact_interval must contain exactly two frame indices")
    start, end = validated
    if not 0 <= start < end <= frame_count:
        raise ValueError(
            "stable_contact_interval must be a non-empty half-open interval "
            f"inside [0, {frame_count}]"
        )

    supplied = pickup.get("critical_grasp_interval")
    if supplied is not None:
        try:
            critical = tuple(int(value) for value in supplied)
        except TypeError as error:
            raise ValueError(
                "critical_grasp_interval must contain exactly two frame indices"
            ) from error
        if len(critical) != 2:
            raise ValueError(
                "critical_grasp_interval must contain exactly two frame indices"
            )
        source = "pickup_grasp.critical_grasp_interval"
    elif critical_grasp_tail_frames is not None:
        critical = (max(start, end - critical_grasp_tail_frames), end)
        source = "terminal_tail_of_pickup_grasp.stable_contact_interval"
    else:
        critical = validated
        source = "pickup_grasp.stable_contact_interval_legacy_fallback"

    critical_start, critical_end = critical
    if not start <= critical_start < critical_end <= end:
        raise ValueError(
            "critical_grasp_interval must be a non-empty half-open interval "
            "contained in stable_contact_interval"
        )
    return (start, end), (critical_start, critical_end), source


def _require_array(
    value: np.ndarray,
    *,
    name: str,
    shape: tuple[int | None, ...],
    finite: bool = True,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        rendered = tuple("N" if value is None else value for value in shape)
        raise ValueError(f"{name} must have shape {rendered}, got {array.shape}")
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _angle_quantile_deg(directions: np.ndarray, reference: np.ndarray, q: float) -> float:
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    angles = np.rad2deg(np.arccos(np.clip(unit @ reference, -1.0, 1.0)))
    return float(np.quantile(angles, q))


def build_demo_object_intent(
    *,
    pickup: Mapping[str, Any],
    object_centers_camera_m: np.ndarray,
    track_timestamps_s: np.ndarray,
    hand_joints_camera_m: np.ndarray,
    hand_joint_confidence: np.ndarray,
    timestamps_s: np.ndarray,
    precontact_object_points_camera_m: np.ndarray,
    config: DemoObjectIntentConfig = DemoObjectIntentConfig(),
) -> DemoObjectIntentArtifacts:
    """Build a camera-axis object template and raw hand-relative grasp intent.

    The template origin is the pickup detector's reference object centre.  Its
    axes are parallel to the demonstration camera axes.  The grasp SE(3) uses
    the frozen clean reference centre as its object anchor.  Per-frame tracked
    centres and translation-compensated joints remain diagnostic outputs; they
    do not perturb the key-pose estimate when an occluded mask centroid jumps.

    ``pickup["stable_contact_interval"]`` remains the validated contact evidence.
    When present, ``pickup["critical_grasp_interval"]`` is the terminal key-pose
    interval used for grasp SE(3); otherwise the optional configured terminal
    tail is used, with the full stable interval retained as a legacy fallback.
    Contact geometry is aggregated from raw joints; the single-view template
    cloud is not used to snap or project fingertip positions.
    """

    if pickup.get("pickup_demonstration_validated") is not True:
        raise ValueError("pickup demonstration must be validated")
    try:
        reference_center = _require_array(
            np.asarray(pickup["reference_object_center_camera_m"]),
            name="reference object center",
            shape=(3,),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(
            "pickup must contain reference_object_center_camera_m"
        ) from error

    joints = _require_array(
        hand_joints_camera_m,
        name="hand joints",
        shape=(None, 21, 3),
        finite=False,
    )
    frame_count = len(joints)
    confidence = _require_array(
        hand_joint_confidence,
        name="hand joint confidence",
        shape=(frame_count, 21),
        finite=False,
    )
    timestamps = _require_array(
        timestamps_s, name="timestamps", shape=(frame_count,)
    )
    track_timestamps = _require_array(
        track_timestamps_s, name="track timestamps", shape=(frame_count,)
    )
    centers = _require_array(
        object_centers_camera_m,
        name="object centers",
        shape=(frame_count, 3),
        finite=False,
    )
    cloud_camera = _require_array(
        precontact_object_points_camera_m,
        name="precontact object points",
        shape=(None, 3),
    )
    if len(cloud_camera) < 3:
        raise ValueError("precontact object points must contain at least three points")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    timestamp_offset = float(track_timestamps[0] - timestamps[0])
    relative_timestamp_error = (track_timestamps - track_timestamps[0]) - (
        timestamps - timestamps[0]
    )
    relative_timestamp_drift = float(np.max(np.abs(relative_timestamp_error)))
    median_period = float(np.median(np.diff(track_timestamps)))
    if relative_timestamp_drift > 0.5 * median_period:
        raise ValueError(
            "hand and pickup-track timestamps drift by more than half a frame"
        )
    validated_interval, critical_interval, interval_source = _resolve_contact_intervals(
        pickup,
        frame_count=frame_count,
        critical_grasp_tail_frames=config.critical_grasp_tail_frames,
    )
    validated_start, validated_end = validated_interval
    critical_start, critical_end = critical_interval

    object_points = cloud_camera - reference_center
    joints_object = joints - centers[:, None, :]
    joints_frozen_anchor = joints - reference_center
    eligible = np.isfinite(joints[:, [WRIST, THUMB_TIP, INDEX_TIP]]).all(axis=(1, 2))
    eligible &= np.isfinite(confidence[:, [WRIST, THUMB_TIP, INDEX_TIP]]).all(axis=1)
    eligible &= confidence[:, THUMB_TIP] >= config.min_tip_confidence
    eligible &= confidence[:, INDEX_TIP] >= config.min_tip_confidence
    eligible &= confidence[:, WRIST] >= config.min_wrist_confidence
    validated_selected = np.zeros(frame_count, dtype=bool)
    validated_selected[validated_start:validated_end] = True
    validated_selected &= eligible
    selected = np.zeros(frame_count, dtype=bool)
    selected[critical_start:critical_end] = True
    selected &= eligible
    frame_indices = np.flatnonzero(selected)
    if len(frame_indices) < config.min_support_frames:
        raise ValueError(
            "critical grasp interval has only "
            f"{len(frame_indices)} valid frames; need {config.min_support_frames}"
        )

    thumb = joints_frozen_anchor[frame_indices, THUMB_TIP]
    index = joints_frozen_anchor[frame_indices, INDEX_TIP]
    wrist = joints_frozen_anchor[frame_indices, WRIST]
    midpoints = 0.5 * (thumb + index)
    closing = index - thumb
    closing_norm = np.linalg.norm(closing, axis=1)
    closing_valid = closing_norm > 1e-6
    if np.count_nonzero(closing_valid) < config.min_support_frames:
        raise ValueError("too few contact frames define a thumb-to-index closing axis")

    thumb = thumb[closing_valid]
    index = index[closing_valid]
    wrist = wrist[closing_valid]
    midpoints = midpoints[closing_valid]
    closing = closing[closing_valid]
    frame_indices = frame_indices[closing_valid]
    closing_unit = closing / np.linalg.norm(closing, axis=1, keepdims=True)

    # Orthogonalize wrist-to-pinch approach per frame before robust direction
    # aggregation.  This prevents aperture jitter from leaking into +Z.
    approach_raw = midpoints - wrist
    approach_orthogonal = approach_raw - (
        np.sum(approach_raw * closing_unit, axis=1, keepdims=True) * closing_unit
    )
    approach_valid = np.linalg.norm(approach_orthogonal, axis=1) > 1e-6
    if np.count_nonzero(approach_valid) < config.min_support_frames:
        raise ValueError(
            "too few contact frames define a wrist-to-pinch approach orthogonal "
            "to the closing axis"
        )

    thumb = thumb[approach_valid]
    index = index[approach_valid]
    midpoints = midpoints[approach_valid]
    closing = closing[approach_valid]
    approach_orthogonal = approach_orthogonal[approach_valid]
    frame_indices = frame_indices[approach_valid]

    x_reference = robust_direction_mean(closing, name="thumb-to-index closing axis")
    z_axis = robust_direction_mean(
        approach_orthogonal, name="wrist-to-pinch approach axis"
    )
    x_axis = normalize(
        x_reference - np.dot(x_reference, z_axis) * z_axis,
        name="closing axis",
    )
    y_axis = normalize(np.cross(z_axis, x_axis), name="lateral axis")
    x_axis = normalize(np.cross(y_axis, z_axis), name="closing axis")
    if np.dot(x_axis, x_reference) < 0.0:
        # Preserve +X as the demonstrated thumb-to-index representative.  The
        # jaw line is still unsigned for downstream candidate comparison.
        x_axis = -x_axis
        y_axis = -y_axis

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = robust_mean(midpoints)
    contact_points = np.stack((robust_mean(thumb), robust_mean(index)))
    widths = np.abs(closing @ x_axis)
    intent = GraspIntent(
        T_object_gripper_tcp=transform,
        width_m=float(np.median(widths)),
        contact_points_object_m=contact_points,
        stable_interval=critical_interval,
        support_frames=len(frame_indices),
    )

    T_camera_object_template = np.eye(4, dtype=float)
    T_camera_object_template[:3, 3] = reference_center
    valid_contact_mask = np.zeros(frame_count, dtype=bool)
    valid_contact_mask[frame_indices] = True
    critical_centers = centers[critical_start:critical_end]
    finite_critical_centers = critical_centers[
        np.isfinite(critical_centers).all(axis=1)
    ]
    tracked_center_offset_p95_m = (
        float(
            np.quantile(
                np.linalg.norm(finite_critical_centers - reference_center, axis=1),
                0.95,
            )
        )
        if len(finite_critical_centers)
        else None
    )
    quality = {
        "validated_contact_interval": list(validated_interval),
        "validated_contact_interval_frames": validated_end - validated_start,
        "critical_grasp_interval": list(critical_interval),
        "critical_grasp_interval_frames": critical_end - critical_start,
        "critical_grasp_interval_source": interval_source,
        "grasp_se3_anchor": "frozen_reference_object_center_camera_m",
        "grasp_se3_anchor_camera_m": reference_center.tolist(),
        "tracked_object_centers_used_for_grasp_se3": False,
        "critical_tracked_center_offset_from_anchor_p95_m": (
            tracked_center_offset_p95_m
        ),
        "explicit_interval_frames": critical_end - critical_start,
        "valid_support_frames": len(frame_indices),
        "valid_frame_indices": frame_indices.tolist(),
        "closing_axis_angle_p95_deg": _angle_quantile_deg(closing, x_axis, 0.95),
        "approach_axis_angle_p95_deg": _angle_quantile_deg(
            approach_orthogonal, z_axis, 0.95
        ),
        "pinch_width_median_m": float(np.median(np.linalg.norm(closing, axis=1))),
        "pinch_width_mad_m": float(
            np.median(
                np.abs(
                    np.linalg.norm(closing, axis=1)
                    - np.median(np.linalg.norm(closing, axis=1))
                )
            )
        ),
        "min_tip_confidence": float(
            np.min(
                confidence[
                    frame_indices[:, None], np.array([THUMB_TIP, INDEX_TIP])[None, :]
                ]
            )
        ),
        "min_wrist_confidence": float(np.min(confidence[frame_indices, WRIST])),
        "timestamp_timebase_offset_s": timestamp_offset,
        "relative_timestamp_drift_max_s": relative_timestamp_drift,
    }
    return DemoObjectIntentArtifacts(
        object_points_m=object_points,
        intent=intent,
        T_camera_object_template=T_camera_object_template,
        object_centers_camera_m=centers,
        hand_joints_camera_m=joints,
        hand_joints_object_m=joints_object,
        hand_joint_confidence=confidence,
        validated_contact_frame_mask=validated_selected,
        valid_contact_frame_mask=valid_contact_mask,
        timestamps_s=timestamps,
        validated_contact_interval=validated_interval,
        critical_grasp_interval=critical_interval,
        quality=quality,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_frame_sha256(object_points_m: np.ndarray) -> str:
    """Hash canonical template content together with its coordinate convention."""

    points = np.ascontiguousarray(np.asarray(object_points_m, dtype="<f8"))
    shape = np.ascontiguousarray(np.asarray(points.shape, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(b"demo2gripper.no_cad_object_template.v1\0")
    digest.update(OBJECT_FRAME_CONVENTION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(shape.tobytes())
    digest.update(points.tobytes())
    return digest.hexdigest()


def write_demo_object_intent(
    artifacts: DemoObjectIntentArtifacts,
    output_dir: str | Path,
    *,
    provenance: Mapping[str, str] | None = None,
    write_observations: bool = False,
) -> dict[str, Any]:
    """Write the no-CAD template, direct hand intent, and frame manifest."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = [
        output / "object_points_m.npy",
        output / "grasp_intent.json",
        output / "template_manifest.json",
    ]
    if write_observations:
        targets.append(output / "observations.npz")
    existing = [path.name for path in targets if path.exists()]
    if existing:
        raise ValueError(f"refusing to overwrite existing outputs: {', '.join(existing)}")

    points_path = output / "object_points_m.npy"
    np.save(points_path, artifacts.object_points_m, allow_pickle=False)
    points_file_sha256 = _sha256_file(points_path)
    object_frame_sha256 = _object_frame_sha256(artifacts.object_points_m)
    frame_id = f"object_template_camera_axes:sha256:{object_frame_sha256}"

    intent_payload = {
        **artifacts.intent.to_dict(),
        "schema": "demo2gripper.raw_hand_object_intent.v1",
        "object_frame_id": frame_id,
        "object_frame_convention": OBJECT_FRAME_CONVENTION,
        "object_template_sha256": object_frame_sha256,
        "template_manifest": "template_manifest.json",
        "closing_axis_equivalence": "+X~-X",
        "inference": {
            "contact_geometry": "raw_thumb_index_wrist_without_cloud_projection",
            "validated_contact_interval": list(artifacts.validated_contact_interval),
            "validated_contact_interval_source": "pickup_grasp.stable_contact_interval",
            "critical_grasp_interval": list(artifacts.critical_grasp_interval),
            "contact_interval_source": artifacts.quality[
                "critical_grasp_interval_source"
            ],
            "grasp_se3_object_anchor": "frozen_reference_object_center_camera_m",
            "grasp_se3_object_anchor_camera_m": artifacts.T_camera_object_template[
                :3, 3
            ].tolist(),
            "tracked_object_centers_usage": "diagnostic_only_not_grasp_se3",
            "diagnostic_joint_reexpression": "per_frame_center_translation_only",
            "rotation_tracking": False,
        },
        "quality": artifacts.quality,
    }
    (output / "grasp_intent.json").write_text(
        json.dumps(intent_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    reference_center = artifacts.T_camera_object_template[:3, 3]
    manifest: dict[str, Any] = {
        "schema": "demo2gripper.no_cad_object_template.v1",
        "frame_id": frame_id,
        "object_points_file": points_path.name,
        "object_points_file_sha256": points_file_sha256,
        "object_template_sha256": object_frame_sha256,
        "point_count": len(artifacts.object_points_m),
        "units": "metres",
        "frame_convention": OBJECT_FRAME_CONVENTION,
        "T_demo_camera_object_template": (
            artifacts.T_camera_object_template.tolist()
        ),
        "origin": "reference_object_center_camera_m",
        "reference_object_center_camera_m": reference_center.tolist(),
        "axes": "parallel to the demonstration OpenCV color optical camera axes",
        "tracking_model": "translation_only",
        "rotation_observed": False,
        "intent_file": "grasp_intent.json",
        "closing_axis": "+X is the thumb-to-index jaw line; +X and -X are equivalent",
        "approach_axis": "+Z is wrist-to-pinch, orthogonalized against +X",
    }
    if provenance:
        manifest["provenance"] = dict(provenance)
    if write_observations:
        transforms = np.repeat(
            np.eye(4, dtype=float)[None], len(artifacts.timestamps_s), axis=0
        )
        transforms[:, :3, 3] = artifacts.object_centers_camera_m
        np.savez_compressed(
            output / "observations.npz",
            T_camera_object=transforms,
            hand_joints_camera_m=artifacts.hand_joints_camera_m,
            hand_joint_confidence=artifacts.hand_joint_confidence,
            timestamps_s=artifacts.timestamps_s,
            hand_joints_object_m=artifacts.hand_joints_object_m,
            validated_contact_frame_mask=artifacts.validated_contact_frame_mask,
            valid_contact_frame_mask=artifacts.valid_contact_frame_mask,
        )
        manifest["observations_file"] = "observations.npz"
    (output / "template_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
