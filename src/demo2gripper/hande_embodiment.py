"""Adapt raw human pinch evidence to a supported-tabletop Hand-E intent.

The raw thumb-to-index line describes human anatomy, not necessarily a feasible
parallel-jaw orientation.  This adapter keeps the validated object-relative
grasp position, uses the stable wrist-to-pinch direction to recover planar
grasp yaw, and assigns the robot approach from the support normal.

The result is a candidate-ranking reference.  It is deliberately not a robot
command and does not bypass GraspGenX, IK, collision, or runtime safety gates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .demo_object_intent import OBJECT_FRAME_CONVENTION
from .franka import validate_rigid_transform
from .geometry import normalize, robust_direction_mean
from .intent import INDEX_TIP, THUMB_TIP, WRIST
from .models import JsonSerializable

RAW_INTENT_SCHEMA = "demo2gripper.raw_hand_object_intent.v1"
ADAPTED_INTENT_SCHEMA = "demo2gripper.robot_adapted_grasp_intent.v1"


@dataclass(frozen=True)
class HandEEmbodimentConfig(JsonSerializable):
    """Fail-closed evidence thresholds for tabletop embodiment mapping."""

    min_support_frames: int = 5
    min_palm_planar_component: float = 0.25
    max_palm_planar_angle_p95_deg: float = 10.0
    max_grasp_offset_m: float = 0.05

    def __post_init__(self) -> None:
        if self.min_support_frames < 1:
            raise ValueError("min_support_frames must be positive")
        if not 0.0 < self.min_palm_planar_component <= 1.0:
            raise ValueError("min_palm_planar_component must lie in (0, 1]")
        if not 0.0 < self.max_palm_planar_angle_p95_deg < 90.0:
            raise ValueError("max_palm_planar_angle_p95_deg must lie in (0, 90)")
        if not np.isfinite(self.max_grasp_offset_m) or self.max_grasp_offset_m <= 0.0:
            raise ValueError("max_grasp_offset_m must be positive and finite")


@dataclass(frozen=True)
class HandEEmbodimentResult:
    """Pure geometry result plus the evidence used to construct it."""

    T_object_gripper_tcp: np.ndarray
    critical_grasp_interval: tuple[int, int]
    support_frames: int
    position_offset_base_m: np.ndarray
    position_offset_object_m: np.ndarray
    support_normal_base: np.ndarray
    raw_closing_axis_object: np.ndarray
    raw_closing_axis_base: np.ndarray
    raw_approach_axis_object: np.ndarray
    raw_approach_axis_base: np.ndarray
    observed_closing_axis_base: np.ndarray
    raw_palm_axis_camera: np.ndarray
    raw_palm_axis_base: np.ndarray
    palm_planar_axis_base: np.ndarray
    adapted_closing_axis_base: np.ndarray
    adapted_approach_axis_base: np.ndarray
    palm_planar_component: float
    palm_planar_angle_p95_deg: float
    observed_closing_angle_p95_deg: float


def _vector3(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _interval(value: Any, *, name: str, frame_count: int) -> tuple[int, int]:
    try:
        interval = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain two integer frame indices") from error
    if len(interval) != 2:
        raise ValueError(f"{name} must contain two integer frame indices")
    start, end = interval
    if not 0 <= start < end <= frame_count:
        raise ValueError(f"{name} must be a non-empty half-open interval")
    return start, end


def _angle_p95_deg(directions: np.ndarray, reference: np.ndarray, *, unsigned: bool) -> float:
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    cosine = unit @ reference
    if unsigned:
        cosine = np.abs(cosine)
    angles = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(np.quantile(angles, 0.95))


def _canonicalize_unsigned_planar_axis(axis: np.ndarray) -> np.ndarray:
    """Choose a deterministic representative for an undirected XY line."""

    result = normalize(axis, name="adapted closing axis").copy()
    planar = result[:2]
    first = int(np.argmax(np.abs(planar)))
    if planar[first] < 0.0:
        result = -result
    result[np.abs(result) < 1e-15] = 0.0
    return result


def adapt_human_intent_to_hande(
    *,
    raw_intent: Mapping[str, Any],
    pickup: Mapping[str, Any],
    hand_joints_camera_m: np.ndarray,
    T_base_camera: np.ndarray,
    support_normal_base: np.ndarray = np.array([0.0, 0.0, 1.0]),
    config: HandEEmbodimentConfig = HandEEmbodimentConfig(),
) -> HandEEmbodimentResult:
    """Construct a top-entry Hand-E reference from raw human grasp evidence.

    The object template axes are the demonstration camera axes, so the rotation
    of ``T_base_camera`` is also ``R_base_object``.  The adapted local ``+X`` is
    the undirected horizontal jaw line and local ``+Z`` points toward the
    object along the negative support normal.
    """

    if raw_intent.get("schema") != RAW_INTENT_SCHEMA:
        raise ValueError(f"raw intent must use schema {RAW_INTENT_SCHEMA!r}")
    if raw_intent.get("object_frame_convention") != OBJECT_FRAME_CONVENTION:
        raise ValueError("raw intent does not use the camera-axis object frame")
    if pickup.get("pickup_demonstration_validated") is not True:
        raise ValueError("pickup demonstration must be validated")

    base_camera = validate_rigid_transform(T_base_camera, name="T_base_camera")
    if "T_base_camera" in pickup:
        pickup_base_camera = validate_rigid_transform(
            np.asarray(pickup["T_base_camera"], dtype=float),
            name="pickup.T_base_camera",
        )
        if not np.allclose(base_camera, pickup_base_camera, atol=1e-9, rtol=0.0):
            raise ValueError("T_base_camera differs from the pickup calibration")
    R_base_object = base_camera[:3, :3]

    raw_transform = validate_rigid_transform(
        np.asarray(raw_intent.get("T_object_gripper_tcp"), dtype=float),
        name="raw T_object_gripper_tcp",
    )
    joints = np.asarray(hand_joints_camera_m, dtype=float)
    if joints.ndim != 3 or joints.shape[1:] != (21, 3):
        raise ValueError("hand_joints_camera_m must have shape (F, 21, 3)")
    critical = _interval(
        raw_intent.get("stable_interval"),
        name="raw intent stable_interval",
        frame_count=len(joints),
    )
    validated = _interval(
        pickup.get("stable_contact_interval"),
        name="pickup stable_contact_interval",
        frame_count=len(joints),
    )
    if not validated[0] <= critical[0] < critical[1] <= validated[1]:
        raise ValueError("critical grasp interval is outside validated contact")

    start, end = critical
    selected = joints[start:end, [WRIST, THUMB_TIP, INDEX_TIP]]
    finite = np.isfinite(selected).all(axis=(1, 2))
    selected = selected[finite]
    if len(selected) < config.min_support_frames:
        raise ValueError(
            f"critical grasp interval has {len(selected)} finite frames; "
            f"need {config.min_support_frames}"
        )
    wrist = selected[:, 0]
    thumb = selected[:, 1]
    index = selected[:, 2]
    midpoint = 0.5 * (thumb + index)
    palm_camera = midpoint - wrist
    closing_camera = index - thumb
    palm_norm = np.linalg.norm(palm_camera, axis=1)
    closing_norm = np.linalg.norm(closing_camera, axis=1)
    valid_directions = (palm_norm > 1e-6) & (closing_norm > 1e-6)
    palm_camera = palm_camera[valid_directions]
    closing_camera = closing_camera[valid_directions]
    if len(palm_camera) < config.min_support_frames:
        raise ValueError("too few critical frames define palm and closing directions")

    raw_palm_camera = robust_direction_mean(
        palm_camera, name="critical wrist-to-pinch direction"
    )
    observed_closing_camera = robust_direction_mean(
        closing_camera, name="critical thumb-to-index direction"
    )
    raw_palm_base = R_base_object @ raw_palm_camera
    observed_closing_base = R_base_object @ observed_closing_camera
    raw_closing_object = raw_transform[:3, 0]
    raw_closing_base = R_base_object @ raw_closing_object
    raw_approach_object = raw_transform[:3, 2]
    raw_approach_base = R_base_object @ raw_approach_object

    normal = normalize(
        _vector3(support_normal_base, name="support_normal_base"),
        name="support normal",
    )
    palm_planar = raw_palm_base - float(np.dot(raw_palm_base, normal)) * normal
    planar_component = float(np.linalg.norm(palm_planar))
    if planar_component < config.min_palm_planar_component:
        raise ValueError(
            "wrist-to-pinch direction has insufficient support-plane component"
        )
    palm_planar = palm_planar / planar_component

    palm_samples_unit_camera = palm_camera / np.linalg.norm(
        palm_camera, axis=1, keepdims=True
    )
    palm_samples_base = palm_samples_unit_camera @ R_base_object.T
    palm_samples_planar = palm_samples_base - (
        (palm_samples_base @ normal)[:, None] * normal[None, :]
    )
    planar_norms = np.linalg.norm(palm_samples_planar, axis=1)
    usable_planar = planar_norms >= config.min_palm_planar_component
    if np.count_nonzero(usable_planar) < config.min_support_frames:
        raise ValueError("too few critical frames have an observable palm yaw")
    palm_planar_p95 = _angle_p95_deg(
        palm_samples_planar[usable_planar], palm_planar, unsigned=False
    )
    if palm_planar_p95 > config.max_palm_planar_angle_p95_deg:
        raise ValueError(
            "critical wrist-to-pinch planar direction is not stable enough: "
            f"p95={palm_planar_p95:.3f} deg"
        )

    adapted_closing_base = _canonicalize_unsigned_planar_axis(
        np.cross(normal, palm_planar)
    )
    adapted_approach_base = -normal
    lateral_base = normalize(
        np.cross(adapted_approach_base, adapted_closing_base),
        name="adapted lateral axis",
    )
    adapted_closing_base = normalize(
        np.cross(lateral_base, adapted_approach_base),
        name="adapted closing axis",
    )
    R_base_tcp = np.column_stack(
        (adapted_closing_base, lateral_base, adapted_approach_base)
    )
    R_object_tcp = R_base_object.T @ R_base_tcp

    offset_base = _vector3(
        pickup.get("grasp_origin_offset_from_contact_object_center_base_m"),
        name="validated grasp origin offset",
    )
    if float(np.linalg.norm(offset_base)) > config.max_grasp_offset_m:
        raise ValueError("validated grasp origin offset exceeds its configured limit")
    offset_object = R_base_object.T @ offset_base
    adapted_transform = np.eye(4, dtype=float)
    adapted_transform[:3, :3] = R_object_tcp
    adapted_transform[:3, 3] = offset_object
    validate_rigid_transform(adapted_transform, name="adapted T_object_gripper_tcp")

    closing_samples_base = closing_camera @ R_base_object.T
    return HandEEmbodimentResult(
        T_object_gripper_tcp=adapted_transform,
        critical_grasp_interval=critical,
        support_frames=len(palm_camera),
        position_offset_base_m=offset_base,
        position_offset_object_m=offset_object,
        support_normal_base=normal,
        raw_closing_axis_object=raw_closing_object,
        raw_closing_axis_base=raw_closing_base,
        raw_approach_axis_object=raw_approach_object,
        raw_approach_axis_base=raw_approach_base,
        observed_closing_axis_base=observed_closing_base,
        raw_palm_axis_camera=raw_palm_camera,
        raw_palm_axis_base=raw_palm_base,
        palm_planar_axis_base=palm_planar,
        adapted_closing_axis_base=adapted_closing_base,
        adapted_approach_axis_base=adapted_approach_base,
        palm_planar_component=planar_component,
        palm_planar_angle_p95_deg=palm_planar_p95,
        observed_closing_angle_p95_deg=_angle_p95_deg(
            closing_samples_base, observed_closing_base, unsigned=False
        ),
    )


def adapted_intent_payload(
    result: HandEEmbodimentResult,
    *,
    raw_intent: Mapping[str, Any],
    provenance: Mapping[str, Any],
    config: HandEEmbodimentConfig,
) -> dict[str, Any]:
    """Serialize an auditable Hand-E candidate-ranking reference."""

    payload: dict[str, Any] = {
        "schema": ADAPTED_INTENT_SCHEMA,
        "intent_role": "robot_adapted_hande",
        "execution_consumable": False,
        "robot_execution_ready": False,
        "consumption_role": "graspgenx_candidate_ranking_reference_only",
        "gripper_name": "robotiq_hande",
        "frame_convention": "T_object_gripper_tcp; +X closing, +Z approach",
        "object_frame_id": raw_intent["object_frame_id"],
        "object_frame_convention": raw_intent["object_frame_convention"],
        "object_template_sha256": raw_intent.get("object_template_sha256"),
        "T_object_gripper_tcp": result.T_object_gripper_tcp.tolist(),
        "width_m": float(raw_intent["width_m"]),
        "contact_points_object_m": raw_intent["contact_points_object_m"],
        "stable_interval": list(result.critical_grasp_interval),
        "support_frames": result.support_frames,
        "continuous_symmetry_axis": raw_intent.get("continuous_symmetry_axis"),
        "closing_axis_equivalence": "+X~-X",
        "human_pinch_width_is_not_robot_opening": True,
        "position_mapping": {
            "source": "pickup.grasp_origin_offset_from_contact_object_center_base_m",
            "offset_base_m": result.position_offset_base_m.tolist(),
            "offset_object_m": result.position_offset_object_m.tolist(),
            "tracked_occluded_object_center_used": False,
            "frozen_reference_object_center_used": False,
        },
        "raw_human_evidence": {
            "critical_grasp_interval": list(result.critical_grasp_interval),
            "support_frames": result.support_frames,
            "raw_intent_closing_axis_object": result.raw_closing_axis_object.tolist(),
            "raw_intent_closing_axis_base": result.raw_closing_axis_base.tolist(),
            "raw_intent_approach_axis_object": (
                result.raw_approach_axis_object.tolist()
            ),
            "raw_intent_approach_axis_base": result.raw_approach_axis_base.tolist(),
            "observed_thumb_to_index_axis_base": (
                result.observed_closing_axis_base.tolist()
            ),
            "wrist_to_pinch_axis_camera": result.raw_palm_axis_camera.tolist(),
            "wrist_to_pinch_axis_base": result.raw_palm_axis_base.tolist(),
            "wrist_to_pinch_planar_axis_base": (
                result.palm_planar_axis_base.tolist()
            ),
            "wrist_to_pinch_planar_component": result.palm_planar_component,
            "wrist_to_pinch_planar_angle_p95_deg": (
                result.palm_planar_angle_p95_deg
            ),
            "observed_closing_angle_p95_deg": (
                result.observed_closing_angle_p95_deg
            ),
        },
        "embodiment_mapping": {
            "method": "supported_tabletop_parallel_jaw_v1",
            "semantic": "horizontal_side-face_closure_with_top_entry",
            "support_normal_base": result.support_normal_base.tolist(),
            "adapted_closing_axis_base": (
                result.adapted_closing_axis_base.tolist()
            ),
            "adapted_approach_axis_base": (
                result.adapted_approach_axis_base.tolist()
            ),
            "raw_thumb_index_axis_used_as_robot_closing_axis": False,
            "palm_planar_yaw_used": True,
            "approach_source": "negative_support_normal",
            "config": config.to_dict(),
        },
        "provenance": dict(provenance),
        "limitations": [
            "This artifact is a ranking reference, not a robot command.",
            "Human fingertip contact points and aperture are diagnostic only.",
            (
                "The mapping assumes the supported tabletop orientation represented "
                "by the supplied calibration."
            ),
            (
                "GraspGenX selection, IK, collision, support-clearance, and runtime "
                "safety checks remain required."
            ),
        ],
    }
    return payload


__all__ = [
    "ADAPTED_INTENT_SCHEMA",
    "RAW_INTENT_SCHEMA",
    "HandEEmbodimentConfig",
    "HandEEmbodimentResult",
    "adapt_human_intent_to_hande",
    "adapted_intent_payload",
]
