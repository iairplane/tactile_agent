"""Fail-closed geometry and reporting helpers for the staged live pick runner.

This module contains no ROS, camera, or serial imports.  The command-line runner
uses it to validate every target before it starts a process that can move hardware.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from .franka import validate_rigid_transform


class PickStage(StrEnum):
    """Fixed stage order for one autonomous pick attempt."""

    OBSERVE = "OBSERVE"
    GROUND = "GROUND"
    PLAN = "PLAN"
    PREGRASP = "PREGRASP"
    DESCEND = "DESCEND"
    CLOSE = "CLOSE"
    VERIFY_CONTACT = "VERIFY_CONTACT"
    LIFT = "LIFT"


STAGE_ORDER = tuple(PickStage)


@dataclass(frozen=True)
class PickThresholds:
    """Geometric and feedback gates applied before each stage transition."""

    min_pregrasp_clearance_m: float = 0.040
    min_tcp_support_clearance_m: float = 0.003
    min_object_width_m: float = 0.005
    max_gripper_opening_m: float = 0.050
    gripper_width_margin_m: float = 0.003
    max_grasp_offset_m: float = 0.050
    max_approach_distance_m: float = 0.120
    max_lift_segment_m: float = 0.050
    max_lift_distance_m: float = 0.120
    min_downward_approach_alignment: float = 0.950
    max_closing_axis_vertical_component: float = 0.250
    max_measured_position_error_m: float = 0.008
    max_measured_tcp_position_error_m: float = 0.005
    max_measured_orientation_error_deg: float = 5.0
    max_final_grasp_tcp_error_m: float = 0.001
    max_descend_correction_m: float = 0.012
    max_descend_correction_xy_m: float = 0.003
    min_opening_before_descent_m: float = 0.048
    min_grasped_opening_m: float = 0.020
    max_grasped_opening_m: float = 0.035
    max_anchor_center_deviation_m: float = 0.002
    max_anchor_rms_deviation_m: float = 0.001
    max_anchor_point_count_cv: float = 0.10
    min_object_height_above_support_m: float = 0.028
    max_object_height_above_support_m: float = 0.040
    min_anchor_z_m: float = 0.042
    max_anchor_z_m: float = 0.052
    tcp_xy_min_base_m: tuple[float, float] = (0.625, 0.020)
    tcp_xy_max_base_m: tuple[float, float] = (0.685, 0.060)
    object_extent_min_m: tuple[float, float, float] = (0.025, 0.014, 0.014)
    object_extent_max_m: tuple[float, float, float] = (0.045, 0.035, 0.024)

    def __post_init__(self) -> None:
        positive = (
            "min_pregrasp_clearance_m",
            "min_tcp_support_clearance_m",
            "min_object_width_m",
            "max_gripper_opening_m",
            "gripper_width_margin_m",
            "max_grasp_offset_m",
            "max_approach_distance_m",
            "max_lift_segment_m",
            "max_lift_distance_m",
            "max_measured_position_error_m",
            "max_measured_tcp_position_error_m",
            "max_measured_orientation_error_deg",
            "max_final_grasp_tcp_error_m",
            "max_descend_correction_m",
            "max_descend_correction_xy_m",
            "min_opening_before_descent_m",
            "min_grasped_opening_m",
            "max_grasped_opening_m",
            "max_anchor_center_deviation_m",
            "max_anchor_rms_deviation_m",
            "max_anchor_point_count_cv",
            "min_object_height_above_support_m",
            "max_object_height_above_support_m",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < self.min_downward_approach_alignment <= 1.0:
            raise ValueError("min_downward_approach_alignment must lie in (0, 1]")
        if not 0.0 <= self.max_closing_axis_vertical_component < 1.0:
            raise ValueError("max_closing_axis_vertical_component must lie in [0, 1)")
        usable_opening = self.max_gripper_opening_m - self.gripper_width_margin_m
        if usable_opening <= self.min_object_width_m:
            raise ValueError("gripper width margin leaves no usable opening")
        if self.min_opening_before_descent_m > self.max_gripper_opening_m:
            raise ValueError("minimum preshape opening exceeds maximum gripper opening")
        if self.max_final_grasp_tcp_error_m > 0.002:
            raise ValueError("final grasp TCP tolerance is capped at 0.002 m")
        if self.max_final_grasp_tcp_error_m >= self.max_descend_correction_m:
            raise ValueError(
                "final grasp TCP tolerance must be below the descend correction cap"
            )
        if self.max_descend_correction_xy_m > self.max_descend_correction_m:
            raise ValueError("descend XY correction cap cannot exceed the 3D cap")
        if self.min_grasped_opening_m >= self.max_grasped_opening_m:
            raise ValueError("grasped opening range must satisfy min < max")
        if self.max_grasped_opening_m >= self.min_opening_before_descent_m:
            raise ValueError("grasped opening range must lie below the preshape opening")
        if self.min_object_height_above_support_m >= self.max_object_height_above_support_m:
            raise ValueError("object height range must satisfy min < max")
        if not math.isfinite(self.min_anchor_z_m) or not math.isfinite(self.max_anchor_z_m):
            raise ValueError("anchor Z bounds must be finite")
        if self.min_anchor_z_m >= self.max_anchor_z_m:
            raise ValueError("anchor Z range must satisfy min < max")
        for name in ("tcp_xy_min_base_m", "tcp_xy_max_base_m"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain two finite values")
            object.__setattr__(self, name, tuple(float(item) for item in value))
        if np.any(np.asarray(self.tcp_xy_min_base_m) >= self.tcp_xy_max_base_m):
            raise ValueError("TCP XY min must be strictly below TCP XY max")
        for name in ("object_extent_min_m", "object_extent_max_m"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, tuple(float(item) for item in value))
        if np.any(np.asarray(self.object_extent_min_m) >= self.object_extent_max_m):
            raise ValueError("object extent min must be strictly below object extent max")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> PickThresholds:
        if value is None:
            return cls()
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown pick thresholds: {sorted(unknown)}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidatedPickGeometry:
    """Command-frame targets that passed all static transition predicates."""

    T_base_pregrasp_tcp: np.ndarray
    T_base_grasp_tcp: np.ndarray
    T_base_pregrasp_link8: np.ndarray
    approach_delta_base_m: np.ndarray
    lift_deltas_base_m: np.ndarray
    object_width_along_closing_axis_m: float
    support_top_base_z_m: float
    object_top_base_z_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "T_base_pregrasp_tcp": self.T_base_pregrasp_tcp.tolist(),
            "T_base_grasp_tcp": self.T_base_grasp_tcp.tolist(),
            "T_base_pregrasp_link8": self.T_base_pregrasp_link8.tolist(),
            "approach_delta_base_m": self.approach_delta_base_m.tolist(),
            "lift_deltas_base_m": self.lift_deltas_base_m.tolist(),
            "object_width_along_closing_axis_m": (
                self.object_width_along_closing_axis_m
            ),
            "support_top_base_z_m": self.support_top_base_z_m,
            "object_top_base_z_m": self.object_top_base_z_m,
        }


def _vector3(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _accepted_anchor(anchor: Mapping[str, Any]) -> None:
    if anchor.get("schema") != "demo2gripper.live_supported_object_anchor.v1":
        raise ValueError("unsupported live object anchor schema")
    if anchor.get("status") != "accepted":
        raise ValueError("live object anchor was not accepted")
    quality = anchor.get("quality")
    if not isinstance(quality, Mapping) or quality.get("accepted") is not True:
        raise ValueError("live object anchor did not pass its quality gates")


def _support_top(anchor: Mapping[str, Any]) -> float:
    support = anchor.get("support")
    if not isinstance(support, Mapping):
        raise ValueError("live object anchor has no support specification")
    if support.get("mode") == "known_plane":
        value = support.get("top_base_z_m")
    elif support.get("mode") == "band_upper_edge":
        value = support.get("effective_top_base_z_m")
    else:
        raise ValueError("unsupported live object support mode")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("support top must be finite")
    return float(value)


def prepare_execution_candidate(
    candidate_payload: Mapping[str, Any],
    *,
    anchor_z_m: float | None,
    grasp_tcp_vertical_offset_m: float | None,
    lift_delta_base_m: np.ndarray,
) -> dict[str, Any]:
    """Prepare a grounded endpoint while preserving selected SE(3) grasps exactly.

    The v2 contract already contains a robot-specific object-relative TCP pose, so only
    the online lift target may be replaced.  The v1 path is explicitly legacy and may
    apply its historical anchor-relative Z correction.
    """

    lift_delta = _vector3(lift_delta_base_m, name="lift_delta_base_m")
    if lift_delta[2] <= 0.0:
        raise ValueError("robot lift must have positive base Z")
    payload = deepcopy(dict(candidate_payload))
    candidate = payload.get("candidate")
    phases = payload.get("phase_targets")
    if not isinstance(candidate, dict) or not isinstance(phases, dict):
        raise ValueError("grounded candidate is missing candidate or phase targets")
    pregrasp = validate_rigid_transform(
        np.asarray(candidate.get("T_base_pregrasp_tcp"), dtype=float),
        name="T_base_pregrasp_tcp",
    ).copy()
    grasp = validate_rigid_transform(
        np.asarray(candidate.get("T_base_grasp_tcp"), dtype=float),
        name="T_base_grasp_tcp",
    ).copy()

    approach_phase = phases.get("APPROACH")
    grasp_phase = phases.get("GRASP")
    lift_phase = phases.get("LIFT")
    if not all(isinstance(value, dict) for value in (approach_phase, grasp_phase, lift_phase)):
        raise ValueError("grounded candidate is missing required phase targets")
    schema = payload.get("schema")
    if schema == "demo2gripper.grounded_live_grasp_candidate.v2":
        evidence = payload.get("evidence")
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("grasp_relation_mode") != "se3"
            or evidence.get("selected_grasp_frame_validated") is not True
            or evidence.get("object_frame_id_matched") is not True
            or evidence.get("agent_approach_override_allowed") is not False
        ):
            raise ValueError("v2 grounded candidate lacks validated SE(3) evidence")
        if (
            grasp_phase.get("source") != "selected_object_relative_se3"
            or grasp_phase.get("endpoint_immutable") is not True
        ):
            raise ValueError("v2 GRASP must be an immutable selected SE(3) endpoint")
        phase_grasp = validate_rigid_transform(
            np.asarray(grasp_phase.get("T_base_gripper_tcp"), dtype=float),
            name="phase GRASP T_base_gripper_tcp",
        )
        if not np.allclose(phase_grasp, grasp, atol=1e-9, rtol=0.0):
            raise ValueError("v2 GRASP phase differs from the selected candidate endpoint")
        payload["robot_execution_overrides"] = {
            "grasp_endpoint_modified": False,
            "lift_delta_base_m": lift_delta.tolist(),
        }
    elif schema == "demo2gripper.grounded_live_grasp_candidate.v1":
        if grasp_phase.get("source") != "legacy_base_aligned_planar_demonstration":
            raise ValueError(
                "v1 candidate must explicitly identify its legacy planar source"
            )
        if (
            anchor_z_m is None
            or grasp_tcp_vertical_offset_m is None
            or not math.isfinite(anchor_z_m)
            or not math.isfinite(grasp_tcp_vertical_offset_m)
        ):
            raise ValueError("legacy anchor and robot grasp-height offset must be finite")
        approach = grasp[:3, 3] - pregrasp[:3, 3]
        grasp[2, 3] = anchor_z_m + grasp_tcp_vertical_offset_m
        pregrasp[:3, 3] = grasp[:3, 3] - approach
        candidate["T_base_pregrasp_tcp"] = pregrasp.tolist()
        candidate["T_base_grasp_tcp"] = grasp.tolist()
        payload["robot_execution_overrides"] = {
            "legacy_transfer_contract": "base_aligned_planar",
            "grasp_tcp_vertical_offset_from_live_anchor_m": (
                grasp_tcp_vertical_offset_m
            ),
            "reason": "legacy human pinch Z is not the configured Hand-E TCP height",
            "lift_delta_base_m": lift_delta.tolist(),
        }
    else:
        raise ValueError("unsupported grounded grasp candidate schema")

    approach_phase["poses_base"] = [pregrasp.tolist(), grasp.tolist()]
    grasp_phase["T_base_gripper_tcp"] = grasp.tolist()
    lift_phase["positions_base_m"] = [
        grasp[:3, 3].tolist(),
        (grasp[:3, 3] + lift_delta).tolist(),
    ]
    lift_phase["source"] = "robot_specific_short_lift"
    return payload


def validate_live_anchor_safety(
    anchor: Mapping[str, Any], thresholds: PickThresholds
) -> None:
    """Apply setup-specific localization and support-object gates."""

    _accepted_anchor(anchor)
    center = _vector3(anchor.get("center_base_m"), name="anchor center")
    if not thresholds.min_anchor_z_m <= center[2] <= thresholds.max_anchor_z_m:
        raise ValueError("live object anchor Z is outside the configured range")
    stability = anchor.get("stability")
    if not isinstance(stability, Mapping):
        raise ValueError("live object anchor has no stability evidence")
    checks = (
        (
            "center_max_deviation_m",
            thresholds.max_anchor_center_deviation_m,
        ),
        ("center_rms_deviation_m", thresholds.max_anchor_rms_deviation_m),
        ("point_count_cv", thresholds.max_anchor_point_count_cv),
    )
    for name, maximum in checks:
        value = stability.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"live object stability {name} is missing or non-finite")
        if float(value) > maximum:
            raise ValueError(f"live object stability {name} exceeds its threshold")
    bounds = anchor.get("robust_bounds_p01_p99_base_m")
    if not isinstance(bounds, Mapping):
        raise ValueError("live object anchor has no robust bounds")
    object_top = _vector3(bounds.get("max"), name="object robust bounds max")[2]
    height = float(object_top - _support_top(anchor))
    if not (
        thresholds.min_object_height_above_support_m
        <= height
        <= thresholds.max_object_height_above_support_m
    ):
        raise ValueError("observed object top-to-support height is outside the configured range")


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized XYZW quaternion."""

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    transform = np.eye(4)
    transform[:3, :3] = matrix
    validate_rigid_transform(transform, name="quaternion rotation")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def quaternion_xyzw_to_rotation_matrix(quaternion: Any) -> np.ndarray:
    """Convert a finite, non-zero XYZW quaternion to a proper rotation matrix."""

    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = value / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def cartesian_measured_end_tcp_transform(
    payload: Mapping[str, Any], T_link8_tcp: np.ndarray
) -> np.ndarray:
    """Compose the measured Cartesian link8 pose with the configured TCP transform."""

    position = _vector3(
        payload.get("measured_end_position_base_m"),
        name="measured Cartesian link8 end position",
    )
    orientation = payload.get("measured_end_orientation_xyzw")
    if orientation is None:
        # Older v1 artifacts contain the held orientation only.  The Cartesian
        # command constrains that orientation throughout the segment.
        orientation = payload.get("held_orientation_xyzw")
    rotation = quaternion_xyzw_to_rotation_matrix(orientation)
    T_base_link8 = np.eye(4)
    T_base_link8[:3, :3] = rotation
    T_base_link8[:3, 3] = position
    return T_base_link8 @ validate_rigid_transform(T_link8_tcp, name="T_link8_tcp")


def evaluate_descend_tcp_residual(
    measured_tcp_position_base_m: Any,
    target_tcp_position_base_m: Any,
    *,
    support_top_base_z_m: float,
    thresholds: PickThresholds,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return a bounded downward correction, or ``None`` when grasp-ready.

    The correction is expressed in the base frame.  A correction can only continue
    downward, must stay within the tight XY and 3D caps, and is never accepted when
    the measured TCP violates the configured support clearance.
    """

    measured = _vector3(
        measured_tcp_position_base_m, name="measured descend TCP position"
    )
    target = _vector3(target_tcp_position_base_m, name="target grasp TCP position")
    if not math.isfinite(support_top_base_z_m):
        raise ValueError("support top must be finite")
    residual = target - measured
    norm = float(np.linalg.norm(residual))
    xy_norm = float(np.linalg.norm(residual[:2]))
    support_clearance = float(measured[2] - support_top_base_z_m)
    details: dict[str, Any] = {
        "measured_tcp_position_base_m": measured.tolist(),
        "target_tcp_position_base_m": target.tolist(),
        "residual_base_m": residual.tolist(),
        "residual_norm_m": norm,
        "residual_xy_norm_m": xy_norm,
        "measured_tcp_support_clearance_m": support_clearance,
        "tolerance_m": thresholds.max_final_grasp_tcp_error_m,
    }
    if support_clearance < thresholds.min_tcp_support_clearance_m:
        raise ValueError(
            "measured descend TCP violates the support-surface clearance guard"
        )
    if norm <= thresholds.max_final_grasp_tcp_error_m:
        details["status"] = "within_tolerance"
        return None, details
    if norm > thresholds.max_descend_correction_m:
        raise ValueError(
            f"descend residual {norm:.6f} m exceeds the correction cap"
        )
    if xy_norm > thresholds.max_descend_correction_xy_m:
        raise ValueError(
            f"descend XY residual {xy_norm:.6f} m exceeds the correction cap"
        )
    if residual[2] > 1e-9:
        raise ValueError("descend correction would move upward")
    details["status"] = "correction_required"
    return residual, details


def validate_pick_geometry(
    anchor: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    T_link8_tcp: np.ndarray,
    thresholds: PickThresholds,
    *,
    object_width_along_closing_axis_m: float | None = None,
) -> ValidatedPickGeometry:
    """Validate perception, demonstrated relation, tool mapping, and phase distances."""

    validate_live_anchor_safety(anchor, thresholds)
    candidate_schema = candidate_payload.get("schema")
    if candidate_schema not in {
        "demo2gripper.grounded_live_grasp_candidate.v1",
        "demo2gripper.grounded_live_grasp_candidate.v2",
    }:
        raise ValueError("unsupported grounded grasp candidate schema")
    evidence = candidate_payload.get("evidence")
    if candidate_schema.endswith(".v2"):
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("grasp_relation_mode") != "se3"
            or evidence.get("selected_grasp_frame_validated") is not True
            or evidence.get("object_frame_id_matched") is not True
        ):
            raise ValueError("grounded candidate has no validated selected SE(3) grasp")
    elif (
        not isinstance(evidence, Mapping)
        or evidence.get("closing_axis_stable") is not True
    ):
        raise ValueError("legacy grounded candidate has no stable closing axis")

    candidate = candidate_payload.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("grounded candidate payload has no candidate")
    pregrasp = validate_rigid_transform(
        np.asarray(candidate.get("T_base_pregrasp_tcp"), dtype=float),
        name="T_base_pregrasp_tcp",
    )
    grasp = validate_rigid_transform(
        np.asarray(candidate.get("T_base_grasp_tcp"), dtype=float),
        name="T_base_grasp_tcp",
    )
    link8_tcp = validate_rigid_transform(T_link8_tcp, name="T_link8_tcp")

    anchor_center = _vector3(anchor.get("center_base_m"), name="anchor center")
    source_anchor = _vector3(
        candidate_payload.get("live_object_anchor_base_m"),
        name="candidate live object anchor",
    )
    anchor_tolerance = (
        thresholds.max_anchor_center_deviation_m
        if candidate_schema.endswith(".v2")
        else 1e-9
    )
    if not np.allclose(anchor_center, source_anchor, atol=anchor_tolerance, rtol=0.0):
        raise ValueError("grounded candidate does not use this live object anchor")

    bounds = anchor.get("robust_bounds_p01_p99_base_m")
    if not isinstance(bounds, Mapping):
        raise ValueError("live object anchor has no robust bounds")
    bounds_min = _vector3(bounds.get("min"), name="object robust bounds min")
    bounds_max = _vector3(bounds.get("max"), name="object robust bounds max")
    extents = bounds_max - bounds_min
    if np.any(extents <= 0.0):
        raise ValueError("live object robust bounds are empty")
    if np.any(extents < thresholds.object_extent_min_m) or np.any(
        extents > thresholds.object_extent_max_m
    ):
        raise ValueError("live object robust extents do not match the configured object")

    support_top = _support_top(anchor)
    object_top = float(bounds_max[2])
    closing_axis = grasp[:3, 0]
    approach_axis = grasp[:3, 2]
    if object_width_along_closing_axis_m is None:
        object_width = float(np.dot(np.abs(closing_axis), extents))
    else:
        object_width = float(object_width_along_closing_axis_m)
        if not np.isfinite(object_width) or object_width <= 0.0:
            raise ValueError("projected object width must be positive and finite")
    usable_opening = (
        thresholds.max_gripper_opening_m - thresholds.gripper_width_margin_m
    )
    if object_width < thresholds.min_object_width_m:
        raise ValueError(
            f"estimated object width {object_width:.6f} m is below the minimum"
        )
    if object_width > usable_opening:
        raise ValueError(
            f"estimated object width {object_width:.6f} m exceeds usable gripper "
            f"opening {usable_opening:.6f} m"
        )
    if abs(float(closing_axis[2])) > thresholds.max_closing_axis_vertical_component:
        raise ValueError("demonstrated closing axis is too vertical for this supported pick")
    downward_alignment = float(np.dot(approach_axis, [0.0, 0.0, -1.0]))
    if downward_alignment < thresholds.min_downward_approach_alignment:
        raise ValueError("approach axis is not sufficiently downward")

    approach_delta = grasp[:3, 3] - pregrasp[:3, 3]
    approach_distance = float(np.linalg.norm(approach_delta))
    if approach_distance <= 0.0 or approach_distance > thresholds.max_approach_distance_m:
        raise ValueError("approach distance is outside the configured range")
    if not np.allclose(
        approach_delta / approach_distance, approach_axis, atol=1e-6
    ):
        raise ValueError("pregrasp-to-grasp displacement does not follow the approach axis")
    if pregrasp[2, 3] - object_top < thresholds.min_pregrasp_clearance_m:
        raise ValueError("pregrasp does not clear the observed object top")
    if grasp[2, 3] - support_top < thresholds.min_tcp_support_clearance_m:
        raise ValueError("grasp TCP does not clear the support surface")
    if np.any(grasp[:2, 3] < thresholds.tcp_xy_min_base_m) or np.any(
        grasp[:2, 3] > thresholds.tcp_xy_max_base_m
    ):
        raise ValueError("grasp TCP XY is outside the configured inner workspace")
    grasp_offset = float(np.linalg.norm(grasp[:3, 3] - anchor_center))
    if grasp_offset > thresholds.max_grasp_offset_m:
        raise ValueError("demonstrated grasp offset is outside the configured range")

    phase_targets = candidate_payload.get("phase_targets")
    if not isinstance(phase_targets, Mapping):
        raise ValueError("grounded candidate has no phase targets")
    lift = phase_targets.get("LIFT")
    if not isinstance(lift, Mapping):
        raise ValueError("grounded candidate has no LIFT phase")
    lift_positions = np.asarray(lift.get("positions_base_m"), dtype=float)
    if (
        lift_positions.ndim != 2
        or lift_positions.shape[1:] != (3,)
        or len(lift_positions) < 2
        or not np.isfinite(lift_positions).all()
    ):
        raise ValueError("LIFT positions must be a finite (N, 3) array, N >= 2")
    if not np.allclose(lift_positions[0], grasp[:3, 3], atol=1e-8):
        raise ValueError("LIFT path does not start at the grounded grasp")
    lift_deltas = np.diff(lift_positions, axis=0)
    segment_lengths = np.linalg.norm(lift_deltas, axis=1)
    if np.any(segment_lengths <= 0.0):
        raise ValueError("LIFT path contains a zero-length segment")
    if float(np.max(segment_lengths)) > thresholds.max_lift_segment_m:
        raise ValueError("LIFT path contains an overlong segment")
    if float(np.linalg.norm(lift_positions[-1] - lift_positions[0])) > (
        thresholds.max_lift_distance_m
    ):
        raise ValueError("LIFT endpoint is too far from the grasp")
    if np.any(lift_positions[1:, 2] <= lift_positions[:-1, 2]):
        raise ValueError("LIFT path must rise monotonically")

    return ValidatedPickGeometry(
        T_base_pregrasp_tcp=pregrasp,
        T_base_grasp_tcp=grasp,
        T_base_pregrasp_link8=pregrasp @ np.linalg.inv(link8_tcp),
        approach_delta_base_m=approach_delta,
        lift_deltas_base_m=lift_deltas,
        object_width_along_closing_axis_m=object_width,
        support_top_base_z_m=support_top,
        object_top_base_z_m=object_top,
    )


def pose_result_succeeded(payload: Mapping[str, Any], *, executed: bool) -> bool:
    """Interpret the stable subset of the MoveIt pose-goal report contract."""

    trajectory = payload.get("trajectory")
    audit = payload.get("trajectory_audit")
    planned = (
        payload.get("moveit_error_code") == 1
        and isinstance(trajectory, Mapping)
        and int(trajectory.get("point_count", 0)) > 1
        and isinstance(audit, Mapping)
        and audit.get("accepted") is True
        and payload.get("planning_succeeded") is True
    )
    if not executed:
        return planned and payload.get("mode") == "plan_only"
    return (
        planned
        and payload.get("mode") == "execute"
        and payload.get("execution_policy")
        == "plan_then_audit_then_execute_exact_trajectory"
        and payload.get("execution_dispatched") is True
        and payload.get("execution_status") == 4
        and payload.get("execution_error_code") == 1
        and payload.get("execution_succeeded") is True
    )


def cartesian_result_succeeded(payload: Mapping[str, Any], *, executed: bool) -> bool:
    """Interpret a bounded Cartesian tool report without importing ROS messages."""

    trajectory = payload.get("trajectory")
    audit = payload.get("trajectory_audit")
    planned = (
        payload.get("moveit_error_code") == 1
        and float(payload.get("path_fraction", 0.0)) >= 1.0 - 1e-9
        and isinstance(trajectory, Mapping)
        and int(trajectory.get("point_count", 0)) > 1
        and isinstance(audit, Mapping)
        and audit.get("accepted") is True
        and payload.get("planning_succeeded") is True
    )
    if not executed:
        return planned and payload.get("mode") == "plan_only"
    return (
        planned
        and payload.get("mode") == "execute"
        and payload.get("execution_policy")
        == "plan_then_audit_then_execute_exact_trajectory"
        and payload.get("execution_dispatched") is True
        and payload.get("execution_status") == 4
        and payload.get("execution_error_code") == 1
        and payload.get("execution_succeeded") is True
    )


def measured_pose_within_tolerance(
    payload: Mapping[str, Any], thresholds: PickThresholds
) -> tuple[bool, dict[str, float]]:
    """Check the measured configured EE (or link8 input frame) against its target."""

    configured_ee = payload.get("input_target_frame") == "configured-ee"
    if configured_ee:
        target_position_key = "input_target_position_base_m"
        measured_position_key = "measured_end_configured_ee_position_base_m"
        target_orientation_key = "input_target_orientation_xyzw"
        measured_orientation_key = "measured_end_configured_ee_orientation_xyzw"
    else:
        target_position_key = "target_position_base_m"
        measured_position_key = "measured_end_position_base_m"
        target_orientation_key = "target_orientation_xyzw"
        measured_orientation_key = "measured_end_orientation_xyzw"
    target_position = _vector3(payload.get(target_position_key), name="target position")
    measured_position = _vector3(
        payload.get(measured_position_key), name="measured end position"
    )
    target_quaternion = np.asarray(payload.get(target_orientation_key), dtype=float)
    measured_quaternion = np.asarray(payload.get(measured_orientation_key), dtype=float)
    if target_quaternion.shape != (4,) or measured_quaternion.shape != (4,):
        raise ValueError("pose result quaternions must have four values")
    if not np.isfinite(target_quaternion).all() or not np.isfinite(measured_quaternion).all():
        raise ValueError("pose result quaternions must be finite")
    target_norm = float(np.linalg.norm(target_quaternion))
    measured_norm = float(np.linalg.norm(measured_quaternion))
    if target_norm <= 1e-9 or measured_norm <= 1e-9:
        raise ValueError("pose result quaternions must be non-zero")
    target_quaternion /= target_norm
    measured_quaternion /= measured_norm
    dot = float(np.clip(abs(np.dot(target_quaternion, measured_quaternion)), 0.0, 1.0))
    orientation_error = math.degrees(2.0 * math.acos(dot))
    position_error = float(np.linalg.norm(measured_position - target_position))
    errors = {
        "position_error_m": position_error,
        "orientation_error_deg": orientation_error,
    }
    return (
        position_error
        <= (
            thresholds.max_measured_tcp_position_error_m
            if configured_ee
            else thresholds.max_measured_position_error_m
        )
        and orientation_error <= thresholds.max_measured_orientation_error_deg,
        errors,
    )


def hande_open_succeeded(payload: Mapping[str, Any], thresholds: PickThresholds) -> bool:
    state = payload.get("state")
    if payload.get("ok") is not True or not isinstance(state, Mapping):
        return False
    width_mm = state.get("opening_width_mm")
    return isinstance(width_mm, (int, float)) and float(width_mm) * 1e-3 >= (
        thresholds.min_opening_before_descent_m
    )


def hande_contact_verified(
    payload: Mapping[str, Any], thresholds: PickThresholds
) -> bool:
    state = payload.get("state")
    object_status = state.get("object_status") if isinstance(state, Mapping) else None
    opening_mm = state.get("opening_width_mm") if isinstance(state, Mapping) else None
    return (
        payload.get("ok") is True
        and isinstance(state, Mapping)
        and isinstance(object_status, Mapping)
        and object_status.get("code") == 2
        and object_status.get("grasped") is True
        and state.get("fault_code") == 0
        and isinstance(opening_mm, (int, float))
        and thresholds.min_grasped_opening_m
        <= float(opening_mm) * 1e-3
        <= thresholds.max_grasped_opening_m
    )


def hande_width_snapshot_valid(
    payload: Mapping[str, Any],
    thresholds: PickThresholds,
    *,
    reference_opening_m: float | None = None,
    max_drift_m: float = 0.002,
) -> bool:
    """Validate retained width after reconnecting, where Hand-E resets gOBJ to zero."""

    state = payload.get("state")
    if payload.get("ok") is not True or not isinstance(state, Mapping):
        return False
    opening_mm = state.get("opening_width_mm")
    if not isinstance(opening_mm, (int, float)) or state.get("fault_code") != 0:
        return False
    opening_m = float(opening_mm) * 1e-3
    if not thresholds.min_grasped_opening_m <= opening_m <= thresholds.max_grasped_opening_m:
        return False
    if reference_opening_m is None:
        return True
    return (
        math.isfinite(reference_opening_m)
        and abs(opening_m - reference_opening_m) <= max_drift_m
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist a valid stage report after every transition."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def assert_stage_prefix(stages: Sequence[Mapping[str, Any]]) -> None:
    """Reject reports whose completed/failed stages skipped a transition."""

    observed = [PickStage(item["name"]) for item in stages]
    if observed != list(STAGE_ORDER[: len(observed)]):
        raise ValueError("pick stages do not form the required ordered prefix")


__all__ = [
    "PickStage",
    "PickThresholds",
    "STAGE_ORDER",
    "ValidatedPickGeometry",
    "assert_stage_prefix",
    "atomic_write_json",
    "cartesian_measured_end_tcp_transform",
    "cartesian_result_succeeded",
    "evaluate_descend_tcp_residual",
    "hande_contact_verified",
    "hande_open_succeeded",
    "hande_width_snapshot_valid",
    "measured_pose_within_tolerance",
    "pose_result_succeeded",
    "prepare_execution_candidate",
    "quaternion_xyzw_to_rotation_matrix",
    "rotation_matrix_to_quaternion_xyzw",
    "validate_live_anchor_safety",
    "validate_pick_geometry",
]
