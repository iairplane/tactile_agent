"""Recover auditable human-to-tube grasp constraints from an RGB-D demo.

This module is deliberately free of camera, robot, ROS, and gripper I/O.  It
separates what the demonstration actually observes (image-space tube ends,
RGB-D lifted fingertips, and an axial grasp fraction) from the extra metric
pose evidence required to instantiate a full ``T_tube_tcp`` relation.

All transforms use metres, column vectors, and ``T_parent_child``.  The signed
tube frame has its origin at the geometric centre and local ``+Z`` from the
closed end to the open end.  The canonical TCP has local ``+X`` along the jaw
closing axis and local ``+Z`` pointing toward the object.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .franka import validate_rigid_transform

DEMO_RELATION_RECEIPT_SCHEMA = "demo2gripper.test_tube_demo_relation_receipt.v1"
DEMO_TUBE_POSE_RECEIPT_SCHEMA = "demo2gripper.test_tube_demo_pose_receipt.v1"
DEMO_RELATION_REVIEW_SCHEMA = "demo2gripper.test_tube_demo_relation_review.v1"


def _finite_vector(value: object, *, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _unit(value: object, *, name: str) -> np.ndarray:
    result = _finite_vector(value, size=3, name=name)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be non-zero")
    return result / norm


def _positive(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _intrinsics(value: object) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError("K must be a finite 3x3 matrix")
    if result[0, 0] <= 0.0 or result[1, 1] <= 0.0:
        raise ValueError("K focal lengths must be positive")
    return result


def backproject_uv(K: np.ndarray, uv: object, depth_m: float) -> np.ndarray:
    """Backproject one aligned RGB-D point in the OpenCV optical frame."""

    intrinsics = _intrinsics(K)
    point = _finite_vector(uv, size=2, name="uv")
    depth = _positive(depth_m, name="depth_m")
    return depth * np.linalg.solve(intrinsics, np.array([*point, 1.0]))


@dataclass(frozen=True)
class DepthPatchSample:
    """Robust aligned-depth sample and the evidence used to produce it."""

    uv: np.ndarray
    depth_m: float
    valid_count: int
    patch_count: int
    p10_m: float
    p90_m: float
    radius_px: int

    @property
    def spread_m(self) -> float:
        return self.p90_m - self.p10_m

    def as_dict(self) -> dict[str, Any]:
        return {
            "uv_px": self.uv.tolist(),
            "depth_m": self.depth_m,
            "valid_count": self.valid_count,
            "patch_count": self.patch_count,
            "p10_m": self.p10_m,
            "p90_m": self.p90_m,
            "spread_m": self.spread_m,
            "radius_px": self.radius_px,
        }


def sample_depth_patch(
    depth_m: np.ndarray,
    uv: object,
    *,
    radius_px: int = 3,
    accepted_range_m: tuple[float, float] = (0.15, 1.0),
) -> DepthPatchSample:
    """Sample a small aligned-depth patch without assigning object identity."""

    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2:
        raise ValueError("depth_m must be two-dimensional")
    point = _finite_vector(uv, size=2, name="uv")
    if isinstance(radius_px, bool) or not isinstance(radius_px, int) or radius_px < 0:
        raise ValueError("radius_px must be a non-negative integer")
    lower, upper = map(float, accepted_range_m)
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0.0 < lower < upper:
        raise ValueError("accepted_range_m must be a positive increasing pair")
    x, y = np.rint(point).astype(int)
    x0, x1 = max(0, x - radius_px), min(depth.shape[1], x + radius_px + 1)
    y0, y1 = max(0, y - radius_px), min(depth.shape[0], y + radius_px + 1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("uv lies outside the depth image")
    patch = depth[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch >= lower) & (patch <= upper)]
    if valid.size == 0:
        raise ValueError("depth patch has no accepted sample")
    return DepthPatchSample(
        uv=point.copy(),
        depth_m=float(np.median(valid)),
        valid_count=int(valid.size),
        patch_count=int(patch.size),
        p10_m=float(np.percentile(valid, 10)),
        p90_m=float(np.percentile(valid, 90)),
        radius_px=radius_px,
    )


@dataclass(frozen=True)
class InterpolatedPinch2D:
    frame: int
    thumb_uv: np.ndarray
    index_uv: np.ndarray
    confidence: float
    source_frames: tuple[int, int]
    interpolation_fraction: float

    @property
    def midpoint_uv(self) -> np.ndarray:
        return 0.5 * (self.thumb_uv + self.index_uv)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "thumb_uv_px": self.thumb_uv.tolist(),
            "index_uv_px": self.index_uv.tolist(),
            "midpoint_uv_px": self.midpoint_uv.tolist(),
            "confidence": self.confidence,
            "source_frames": list(self.source_frames),
            "interpolation_fraction": self.interpolation_fraction,
        }


def interpolate_pinch_keypoints(
    records: Sequence[Mapping[str, Any]],
    *,
    frame: int,
    max_frame_gap: int = 10,
) -> InterpolatedPinch2D:
    """Interpolate thumb/index tips to a reviewed grasp frame.

    Exact-frame evidence is preferred.  Interpolation is allowed only between
    two records no farther than ``max_frame_gap`` from the requested frame.
    Extrapolation is never performed.
    """

    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError("frame must be a non-negative integer")
    if isinstance(max_frame_gap, bool) or not isinstance(max_frame_gap, int) or max_frame_gap < 0:
        raise ValueError("max_frame_gap must be a non-negative integer")
    parsed: list[tuple[int, np.ndarray, np.ndarray, float]] = []
    for record in records:
        record_frame = int(record.get("frame", -1))
        thumb = _finite_vector(record.get("thumb_tip_uv"), size=2, name="thumb_tip_uv")
        index = _finite_vector(record.get("index_tip_uv"), size=2, name="index_tip_uv")
        confidence = float(record.get("pinch_min_score", 0.0))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("pinch_min_score must be in [0, 1]")
        parsed.append((record_frame, thumb, index, confidence))
    if not parsed:
        raise ValueError("no RTMPose pinch records were supplied")
    parsed.sort(key=lambda item: item[0])
    exact = [item for item in parsed if item[0] == frame]
    if exact:
        source = exact[0]
        return InterpolatedPinch2D(
            frame=frame,
            thumb_uv=source[1],
            index_uv=source[2],
            confidence=source[3],
            source_frames=(frame, frame),
            interpolation_fraction=0.0,
        )
    before = [item for item in parsed if item[0] < frame]
    after = [item for item in parsed if item[0] > frame]
    if not before or not after:
        raise ValueError("pinch frame is not bracketed; extrapolation is forbidden")
    left, right = before[-1], after[0]
    if frame - left[0] > max_frame_gap or right[0] - frame > max_frame_gap:
        raise ValueError("pinch interpolation gap exceeds max_frame_gap")
    fraction = (frame - left[0]) / (right[0] - left[0])
    return InterpolatedPinch2D(
        frame=frame,
        thumb_uv=(1.0 - fraction) * left[1] + fraction * right[1],
        index_uv=(1.0 - fraction) * left[2] + fraction * right[2],
        confidence=min(left[3], right[3]),
        source_frames=(left[0], right[0]),
        interpolation_fraction=float(fraction),
    )


@dataclass(frozen=True)
class Pinch3DEstimate:
    frame: int
    thumb_camera_m: np.ndarray
    index_camera_m: np.ndarray
    midpoint_camera_m: np.ndarray
    closing_axis_camera_unit: np.ndarray
    contact_separation_m: float
    thumb_depth: DepthPatchSample
    index_depth: DepthPatchSample
    keypoint_confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "thumb_camera_m": self.thumb_camera_m.tolist(),
            "index_camera_m": self.index_camera_m.tolist(),
            "midpoint_camera_m": self.midpoint_camera_m.tolist(),
            "closing_axis_camera_unit_index_to_thumb": self.closing_axis_camera_unit.tolist(),
            "closing_axis_sign_is_a_parallel_jaw_gauge": True,
            "contact_separation_m": self.contact_separation_m,
            "keypoint_confidence": self.keypoint_confidence,
            "thumb_depth": self.thumb_depth.as_dict(),
            "index_depth": self.index_depth.as_dict(),
        }


def estimate_pinch_3d(
    *,
    K: np.ndarray,
    depth_m: np.ndarray,
    pinch: InterpolatedPinch2D,
    depth_radius_px: int = 3,
    accepted_range_m: tuple[float, float] = (0.15, 0.8),
) -> Pinch3DEstimate:
    """Lift the two visible fingertip surfaces and form a midpoint candidate."""

    thumb_depth = sample_depth_patch(
        depth_m,
        pinch.thumb_uv,
        radius_px=depth_radius_px,
        accepted_range_m=accepted_range_m,
    )
    index_depth = sample_depth_patch(
        depth_m,
        pinch.index_uv,
        radius_px=depth_radius_px,
        accepted_range_m=accepted_range_m,
    )
    thumb = backproject_uv(K, pinch.thumb_uv, thumb_depth.depth_m)
    index = backproject_uv(K, pinch.index_uv, index_depth.depth_m)
    delta = thumb - index
    separation = float(np.linalg.norm(delta))
    if separation <= 1e-6:
        raise ValueError("lifted thumb and index points are degenerate")
    return Pinch3DEstimate(
        frame=pinch.frame,
        thumb_camera_m=thumb,
        index_camera_m=index,
        midpoint_camera_m=0.5 * (thumb + index),
        closing_axis_camera_unit=delta / separation,
        contact_separation_m=separation,
        thumb_depth=thumb_depth,
        index_depth=index_depth,
        keypoint_confidence=pinch.confidence,
    )


@dataclass(frozen=True)
class TubeMetricCandidate:
    closed_camera_m: np.ndarray
    open_camera_m: np.ndarray
    pinch_projection_camera_m: np.ndarray
    axis_camera_unit: np.ndarray
    centre_camera_m: np.ndarray
    fitted_length_m: float
    centreline_residual_m: float
    length_residual_m: float
    closed_depth_z_m: float
    open_depth_z_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "closed_camera_m": self.closed_camera_m.tolist(),
            "open_camera_m": self.open_camera_m.tolist(),
            "pinch_projection_camera_m": self.pinch_projection_camera_m.tolist(),
            "axis_camera_unit_closed_to_open": self.axis_camera_unit.tolist(),
            "centre_camera_m": self.centre_camera_m.tolist(),
            "fitted_length_m": self.fitted_length_m,
            "centreline_residual_m": self.centreline_residual_m,
            "length_residual_m": self.length_residual_m,
            "closed_depth_z_m": self.closed_depth_z_m,
            "open_depth_z_m": self.open_depth_z_m,
        }


def fit_tube_from_pinch_and_length(
    *,
    K: np.ndarray,
    closed_uv: object,
    open_uv: object,
    pinch_camera_m: object,
    axial_fraction_closed_to_open: float,
    tube_length_m: float,
    pinch_position_sigma_m: float = 0.008,
    tube_length_sigma_m: float = 0.020,
) -> TubeMetricCandidate:
    """Fit endpoint depths using a fingertip midpoint and a tube-length prior.

    This is a diagnostic candidate, not self-qualifying geometry.  It assumes
    the midpoint of the two visible fingertip keypoints lies on the tube
    centreline, an assumption that RGB-D alone cannot verify.
    """

    intrinsics = _intrinsics(K)
    closed = _finite_vector(closed_uv, size=2, name="closed_uv")
    opened = _finite_vector(open_uv, size=2, name="open_uv")
    pinch = _finite_vector(pinch_camera_m, size=3, name="pinch_camera_m")
    fraction = float(axial_fraction_closed_to_open)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("axial_fraction_closed_to_open must be in (0, 1)")
    length = _positive(tube_length_m, name="tube_length_m")
    pinch_sigma = _positive(pinch_position_sigma_m, name="pinch_position_sigma_m")
    length_sigma = _positive(tube_length_sigma_m, name="tube_length_sigma_m")
    closed_ray = np.linalg.solve(intrinsics, np.array([*closed, 1.0]))
    open_ray = np.linalg.solve(intrinsics, np.array([*opened, 1.0]))
    design = np.column_stack(((1.0 - fraction) * closed_ray, fraction * open_ray))
    initial = np.linalg.lstsq(design, pinch, rcond=None)[0]
    initial = np.clip(initial, 0.05, 5.0)

    def residual(depths: np.ndarray) -> np.ndarray:
        closed_point = depths[0] * closed_ray
        open_point = depths[1] * open_ray
        midpoint = (1.0 - fraction) * closed_point + fraction * open_point
        return np.r_[
            (midpoint - pinch) / pinch_sigma,
            (np.linalg.norm(open_point - closed_point) - length) / length_sigma,
        ]

    solution = least_squares(residual, initial, bounds=(0.05, 5.0), method="trf")
    if not solution.success:
        raise ValueError(f"tube metric fit failed: {solution.message}")
    closed_point = solution.x[0] * closed_ray
    open_point = solution.x[1] * open_ray
    delta = open_point - closed_point
    fitted_length = float(np.linalg.norm(delta))
    if fitted_length <= 1e-6:
        raise ValueError("tube metric fit produced a degenerate axis")
    midpoint = (1.0 - fraction) * closed_point + fraction * open_point
    return TubeMetricCandidate(
        closed_camera_m=closed_point,
        open_camera_m=open_point,
        pinch_projection_camera_m=midpoint,
        axis_camera_unit=delta / fitted_length,
        centre_camera_m=0.5 * (closed_point + open_point),
        fitted_length_m=fitted_length,
        centreline_residual_m=float(np.linalg.norm(midpoint - pinch)),
        length_residual_m=abs(fitted_length - length),
        closed_depth_z_m=float(solution.x[0]),
        open_depth_z_m=float(solution.x[1]),
    )


def make_tube_frame(
    *,
    centre_parent_m: object,
    axis_closed_to_open_parent: object,
    transverse_reference_parent: object,
) -> np.ndarray:
    """Build a signed, roll-canonical tube-centre frame."""

    centre = _finite_vector(centre_parent_m, size=3, name="centre_parent_m")
    z_axis = _unit(axis_closed_to_open_parent, name="axis_closed_to_open_parent")
    reference = _unit(transverse_reference_parent, name="transverse_reference_parent")
    x_axis = reference - float(reference @ z_axis) * z_axis
    if np.linalg.norm(x_axis) <= 1e-6:
        raise ValueError("transverse reference is parallel to the tube axis")
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    result[:3, 3] = centre
    return validate_rigid_transform(result, name="T_parent_tube")


def instantiate_top_down_relation(
    *,
    T_base_camera: np.ndarray,
    T_camera_tube: np.ndarray,
    tube_length_m: float,
    axial_fraction_closed_to_open: float,
    human_closing_axis_camera: object,
    top_approach_direction_base: object = (0.0, 0.0, -1.0),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Instantiate the reviewed grasp constraints as a full ``T_tube_tcp``.

    The demonstrated axial location is preserved.  User-requested top approach
    is a hard constraint: TCP ``+Z`` is exactly the requested base-frame
    direction.  TCP ``+X`` is perpendicular to both that tool direction and
    the tube axis; the human thumb/index direction only chooses its otherwise
    symmetric sign.  If the tube and tool axes are parallel, the demonstrated
    thumb/index direction supplies the otherwise free jaw yaw.
    """

    base_camera = validate_rigid_transform(T_base_camera, name="T_base_camera")
    camera_tube = validate_rigid_transform(T_camera_tube, name="T_camera_tube")
    length = _positive(tube_length_m, name="tube_length_m")
    fraction = float(axial_fraction_closed_to_open)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("axial_fraction_closed_to_open must be in (0, 1)")
    requested_approach = _unit(
        top_approach_direction_base, name="top_approach_direction_base"
    )
    T_base_tube = base_camera @ camera_tube
    tube_axis = _unit(T_base_tube[:3, 2], name="tube axis in base")
    z_tcp = requested_approach
    human_axis_base = base_camera[:3, :3] @ _unit(
        human_closing_axis_camera, name="human_closing_axis_camera"
    )
    human_horizontal = human_axis_base - float(human_axis_base @ z_tcp) * z_tcp
    x_tcp = np.cross(z_tcp, tube_axis)
    if np.linalg.norm(x_tcp) <= 1e-6:
        x_tcp = human_horizontal
    if np.linalg.norm(x_tcp) <= 1e-6:
        fallback = np.asarray([1.0, 0.0, 0.0])
        x_tcp = fallback - float(fallback @ z_tcp) * z_tcp
    if np.linalg.norm(x_tcp) <= 1e-6:
        fallback = np.asarray([0.0, 1.0, 0.0])
        x_tcp = fallback - float(fallback @ z_tcp) * z_tcp
    x_tcp /= np.linalg.norm(x_tcp)
    if np.linalg.norm(human_horizontal) > 1e-6 and float(x_tcp @ human_horizontal) < 0.0:
        x_tcp *= -1.0
    y_tcp = np.cross(z_tcp, x_tcp)
    y_tcp /= np.linalg.norm(y_tcp)
    T_base_tcp = np.eye(4)
    T_base_tcp[:3, :3] = np.column_stack((x_tcp, y_tcp, z_tcp))
    axial_offset = (fraction - 0.5) * length
    T_base_tcp[:3, 3] = (T_base_tube @ np.array([0.0, 0.0, axial_offset, 1.0]))[:3]
    T_tube_tcp = np.linalg.inv(T_base_tube) @ T_base_tcp
    validate_rigid_transform(T_tube_tcp, name="T_tube_tcp")
    detail = {
        "T_base_tube_demo": T_base_tube.tolist(),
        "T_base_tcp_demo_target": T_base_tcp.tolist(),
        "tube_axis_base_unit": tube_axis.tolist(),
        "tcp_closing_axis_base_unit": x_tcp.tolist(),
        "tcp_approach_axis_base_unit": z_tcp.tolist(),
        "requested_top_direction_base_unit": requested_approach.tolist(),
        "tcp_tool_axis_is_requested_top_direction": True,
        "jaw_axis_is_perpendicular_to_tube_and_tool_axis": True,
        "axial_offset_from_tube_centre_m": axial_offset,
        "human_closing_axis_used_for_sign_only": True,
    }
    return T_tube_tcp, detail


def parse_qualified_pose_receipt(
    value: object,
    *,
    expected_frame: int,
    max_position_std_m: float = 0.010,
    max_orientation_std_deg: float = 8.0,
) -> tuple[np.ndarray | None, float | None, list[str], dict[str, Any]]:
    """Validate the independent metric evidence needed for a full relation."""

    blockers: list[str] = []
    detail: dict[str, Any] = {}
    if value is None:
        return None, None, ["metric_demo_tube_pose_receipt_missing"], detail
    if not isinstance(value, Mapping):
        raise ValueError("tube pose receipt must be an object")
    if value.get("schema") != DEMO_TUBE_POSE_RECEIPT_SCHEMA:
        blockers.append("metric_demo_tube_pose_receipt_schema_invalid")
    try:
        receipt_frame = int(value.get("frame", -1))
    except (TypeError, ValueError):
        receipt_frame = -1
    if receipt_frame != expected_frame:
        blockers.append("metric_demo_tube_pose_frame_mismatch")
    if value.get("runtime_qualified") is not True:
        blockers.append("metric_demo_tube_pose_not_runtime_qualified")
    frame_contract = value.get("tube_frame")
    if not isinstance(frame_contract, Mapping):
        blockers.append("metric_demo_tube_frame_contract_missing")
    else:
        if frame_contract.get("origin") != "geometric_centre":
            blockers.append("metric_demo_tube_origin_not_geometric_centre")
        if frame_contract.get("z_axis") != "closed_to_open":
            blockers.append("metric_demo_tube_end_sign_not_qualified")
    try:
        position_std = float(value.get("position_std_m", math.inf))
    except (TypeError, ValueError):
        position_std = math.inf
    try:
        orientation_std = float(value.get("orientation_std_deg", math.inf))
    except (TypeError, ValueError):
        orientation_std = math.inf
    if not math.isfinite(position_std) or position_std > max_position_std_m:
        blockers.append("metric_demo_tube_position_uncertainty_too_high")
    if not math.isfinite(orientation_std) or orientation_std > max_orientation_std_deg:
        blockers.append("metric_demo_tube_orientation_uncertainty_too_high")
    length = value.get("tube_length_m")
    length_uncertainty = value.get("tube_length_uncertainty_m")
    try:
        tube_length = _positive(length, name="tube_length_m")
        length_std = _positive(length_uncertainty, name="tube_length_uncertainty_m")
    except (TypeError, ValueError):
        tube_length, length_std = None, math.inf
        blockers.append("measured_tube_length_missing")
    else:
        if length_std > 0.005:
            blockers.append("measured_tube_length_uncertainty_too_high")
    try:
        transform = validate_rigid_transform(
            np.asarray(value.get("T_camera_tube"), dtype=float), name="T_camera_tube"
        )
    except (TypeError, ValueError):
        transform = None
        blockers.append("metric_demo_T_camera_tube_missing_or_invalid")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        blockers.append("metric_demo_tube_pose_provenance_missing")
    detail.update(
        {
            "position_std_m": position_std,
            "orientation_std_deg": orientation_std,
            "tube_length_uncertainty_m": None if not math.isfinite(length_std) else length_std,
            "provenance": provenance,
        }
    )
    return transform, tube_length, sorted(set(blockers)), detail


def parse_human_review(value: object, *, expected_frame: int) -> tuple[list[str], dict[str, Any]]:
    """Validate the one-time visual review of labels and keypoints."""

    blockers: list[str] = []
    if value is None:
        return ["human_relation_review_missing"], {}
    if not isinstance(value, Mapping):
        raise ValueError("human relation review must be an object")
    if value.get("schema") != DEMO_RELATION_REVIEW_SCHEMA:
        blockers.append("human_relation_review_schema_invalid")
    try:
        review_frame = int(value.get("frame", -1))
    except (TypeError, ValueError):
        review_frame = -1
    if review_frame != expected_frame:
        blockers.append("human_relation_review_frame_mismatch")
    for key, blocker in (
        ("accepted", "human_relation_review_not_accepted"),
        ("closed_open_labels_accepted", "tube_endpoint_sign_not_reviewed"),
        ("pinch_keypoints_accepted", "pinch_keypoints_not_reviewed"),
        ("top_approach_accepted", "top_approach_not_reviewed"),
    ):
        if value.get(key) is not True:
            blockers.append(blocker)
    reviewer = value.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        blockers.append("human_relation_reviewer_missing")
    return sorted(set(blockers)), dict(value)


def relation_uncertainty(
    *,
    tube_length_m: float,
    tube_length_uncertainty_m: float,
    axial_fraction: float,
    axial_fraction_uncertainty: float,
    pose_orientation_std_deg: float,
) -> dict[str, float]:
    """Propagate the scalar axial-position uncertainty of the relation."""

    length = _positive(tube_length_m, name="tube_length_m")
    length_std = _positive(tube_length_uncertainty_m, name="tube_length_uncertainty_m")
    fraction = float(axial_fraction)
    fraction_std = _positive(axial_fraction_uncertainty, name="axial_fraction_uncertainty")
    orientation_std = _positive(pose_orientation_std_deg, name="pose_orientation_std_deg")
    axial_std = math.sqrt((length * fraction_std) ** 2 + ((fraction - 0.5) * length_std) ** 2)
    return {
        "axial_translation_std_m": axial_std,
        "orientation_std_deg_lower_bound": orientation_std,
    }


def runner_geometry_fragment(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the relation fields accepted by the closed-loop runner.

    Partial evidence is rejected here so callers cannot accidentally replace a
    missing transform with identity.
    """

    if receipt.get("schema") != DEMO_RELATION_RECEIPT_SCHEMA:
        raise ValueError("unexpected demo relation receipt schema")
    if receipt.get("runtime_qualified") is not True:
        raise ValueError("demo relation receipt is not runtime qualified")
    transform = validate_rigid_transform(
        np.asarray(receipt.get("T_tube_tcp_demo_grasp"), dtype=float),
        name="T_tube_tcp_demo_grasp",
    )
    relation_id = receipt.get("relation_id")
    if not isinstance(relation_id, str) or not relation_id:
        raise ValueError("relation_id must be a non-empty string")
    return {
        "relation_id": relation_id,
        "T_tube_tcp_demo_grasp": transform.tolist(),
        "relation_source": DEMO_RELATION_RECEIPT_SCHEMA,
        "relation_runtime_qualified": True,
        "tube_axis_tube": [0.0, 0.0, 1.0],
    }
