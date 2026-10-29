"""Review-only human-to-cup-rim relation compiler.

The cup rim is frozen from clean pre-contact RGB-D frames.  Grasp-time mask
centroids are deliberately absent from this contract because the hand occludes
the cup and can move a visible-mask centroid without moving the cup.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import normalize, robust_direction_mean, robust_mean
from .intent import INDEX_TIP, THUMB_TIP, WRIST
from .models import JsonSerializable


@dataclass(frozen=True)
class CupRimRelationConfig(JsonSerializable):
    """Quality gates for one circular top-rim and one demonstrated pinch."""

    min_precontact_frames: int = 3
    min_valid_precontact_fraction: float = 0.6
    min_rim_points: int = 24
    depth_search_px: int = 3
    min_arc_coverage_deg: float = 100.0
    min_rim_radius_m: float = 0.02
    max_rim_radius_m: float = 0.09
    max_plane_residual_m: float = 0.004
    max_radial_residual_m: float = 0.006
    max_ellipse_depth_residual_m: float = 0.015
    max_depth_cluster_gap_m: float = 0.08
    min_ellipse_depth_support_fraction: float = 0.5
    min_ellipse_axis_ratio: float = 0.15
    max_ellipse_axis_ratio: float = 0.98
    min_ellipse_depth_gradient_m_per_px: float = 5e-5
    max_ellipse_gradient_relative_error: float = 0.75
    max_rim_center_spread_m: float = 0.008
    max_cup_axis_spread_deg: float = 12.0
    max_mask_area_cv: float = 0.18
    min_joint_confidence: float = 0.4
    min_grasp_frames: int = 3
    max_pinch_center_spread_m: float = 0.015
    max_closing_axis_spread_deg: float = 25.0
    min_radial_offset_m: float = 0.01
    max_rim_radial_error_m: float = 0.025
    max_rim_axial_error_m: float = 0.035
    max_closing_to_radial_angle_deg: float = 50.0

    def __post_init__(self) -> None:
        positive = (
            "min_precontact_frames",
            "min_rim_points",
            "depth_search_px",
            "min_grasp_frames",
        )
        if any(int(getattr(self, name)) < 1 for name in positive):
            raise ValueError("frame, point, search, and support counts must be positive")
        if not 0.0 < self.min_valid_precontact_fraction <= 1.0:
            raise ValueError("min_valid_precontact_fraction must lie in (0, 1]")
        if not 0.0 < self.min_joint_confidence <= 1.0:
            raise ValueError("min_joint_confidence must lie in (0, 1]")
        if not 0.0 < self.min_ellipse_depth_support_fraction <= 1.0:
            raise ValueError("min_ellipse_depth_support_fraction must lie in (0, 1]")
        if not 0.0 < self.min_ellipse_axis_ratio < self.max_ellipse_axis_ratio <= 1.0:
            raise ValueError("ellipse axis-ratio limits must lie in (0, 1] and be ordered")
        if not 0.0 < self.min_rim_radius_m < self.max_rim_radius_m:
            raise ValueError("rim radius limits must be positive and ordered")


@dataclass(frozen=True)
class RimFrameEstimate:
    frame_index: int
    center_camera_m: np.ndarray
    cup_axis_camera: np.ndarray
    radius_m: float
    rim_points_camera_m: np.ndarray
    plane_residual_m: float
    radial_residual_m: float
    arc_coverage_deg: float
    mask_area_px: int
    extraction: str
    depth_support_fraction: float
    depth_gradient_relative_error: float | None


@dataclass(frozen=True)
class CupRimRelationArtifacts:
    """Geometry proposal.  It never authorizes or commands robot motion."""

    T_camera_rim: np.ndarray
    T_rim_human_pinch_tcp: np.ndarray
    T_camera_human_pinch_tcp: np.ndarray
    rim_radius_m: float
    pinch_width_m: float
    precontact_frame_indices: np.ndarray
    valid_precontact_frame_indices: np.ndarray
    grasp_frame_indices: np.ndarray
    representative_precontact_frame: int
    representative_grasp_frame: int
    rim_points_camera_m: np.ndarray
    quality: dict[str, Any]


def _require_interval(
    stages: Mapping[str, tuple[int, int] | list[int]], name: str, frame_count: int
) -> tuple[int, int]:
    try:
        values = tuple(int(value) for value in stages[name])
    except (KeyError, TypeError) as error:
        raise ValueError(f"stage_intervals must contain {name}") from error
    if len(values) != 2 or not 0 <= values[0] < values[1] <= frame_count:
        raise ValueError(f"{name} must be a non-empty half-open frame interval")
    return values


def _deproject(pixels_xy: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    x = pixels_xy[:, 0]
    y = pixels_xy[:, 1]
    z = depth_m
    return np.column_stack(
        (
            (x - K[0, 2]) * z / K[0, 0],
            (y - K[1, 2]) * z / K[1, 1],
            z,
        )
    )


def _top_contour_pixels(
    mask: np.ndarray, depth: np.ndarray, search_px: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return one valid depth sample near the upper silhouette in each column."""

    height, width = mask.shape
    pixels: list[tuple[float, float]] = []
    depths: list[float] = []
    for x in range(width):
        rows = np.flatnonzero(mask[:, x])
        if len(rows) == 0:
            continue
        top = int(rows[0])
        for y in range(top, min(height, top + search_px)):
            value = float(depth[y, x])
            if mask[y, x] and np.isfinite(value) and value > 0.0:
                pixels.append((float(x), float(y)))
                depths.append(value)
                break
    return np.asarray(pixels, dtype=float), np.asarray(depths, dtype=float)


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    interior = np.zeros_like(mask, dtype=bool)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    rows, columns = np.nonzero(mask & ~interior)
    return np.column_stack((columns, rows)).astype(float)


def _near_depth_cluster(
    values: np.ndarray, config: CupRimRelationConfig
) -> tuple[float, float]:
    """Return the closest substantial depth interval, excluding far background."""

    ordered = np.sort(np.asarray(values, dtype=float))
    minimum_count = max(
        config.min_rim_points,
        int(np.ceil(config.min_ellipse_depth_support_fraction * len(ordered))),
    )
    boundaries = np.concatenate(
        (
            [0],
            np.flatnonzero(np.diff(ordered) > config.max_depth_cluster_gap_m) + 1,
            [len(ordered)],
        )
    )
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end - start >= minimum_count:
            return float(ordered[start]), float(ordered[end - 1])
    raise ValueError("no substantial near-depth cluster exists inside the rim mask")


def _ray(K: np.ndarray, pixel_xy: np.ndarray) -> np.ndarray:
    return np.array(
        [
            (pixel_xy[0] - K[0, 2]) / K[0, 0],
            (pixel_xy[1] - K[1, 2]) / K[1, 1],
            1.0,
        ]
    )


def _fit_rim_from_image_ellipse(
    *,
    frame_index: int,
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    config: CupRimRelationConfig,
    ellipse: np.ndarray,
    area: int,
) -> RimFrameEstimate:
    """Recover a circle approximately from a calibrated image ellipse.

    Thin RealSense rim edges frequently mix cup and background depth.  This
    path therefore obtains orientation from the full image ellipse and uses a
    robust near-depth model only for plane offset and tilt-sign disambiguation.
    The approximation and its consistency residuals remain explicit evidence.
    """

    cx, cy, radius_major_px, radius_minor_px, angle = np.asarray(ellipse, dtype=float)
    if not np.isfinite([cx, cy, radius_major_px, radius_minor_px, angle]).all():
        raise ValueError("rim ellipse contains NaN or infinity")
    if radius_minor_px <= 2.0 or radius_major_px <= radius_minor_px:
        raise ValueError("rim ellipse radii must satisfy major > minor > 2 pixels")
    axis_ratio = float(radius_minor_px / radius_major_px)
    if not config.min_ellipse_axis_ratio <= axis_ratio <= config.max_ellipse_axis_ratio:
        raise ValueError("rim ellipse axis ratio is geometrically ambiguous")

    boundary = _mask_boundary(mask)
    columns = np.rint(boundary[:, 0]).astype(int)
    rows = np.rint(boundary[:, 1]).astype(int)
    boundary_depth = depth[rows, columns]
    valid_depth = np.isfinite(boundary_depth) & (boundary_depth > 0.0)
    rows = rows[valid_depth]
    columns = columns[valid_depth]
    boundary_depth = boundary_depth[valid_depth]
    if len(rows) < config.min_rim_points:
        raise ValueError("rim mask has too few valid boundary depth samples")
    lower_depth, upper_depth = _near_depth_cluster(boundary_depth, config)
    selected = (boundary_depth >= lower_depth) & (boundary_depth <= upper_depth)
    rows = rows[selected]
    columns = columns[selected]
    values = depth[rows, columns]
    design = np.column_stack((columns - cx, rows - cy, np.ones(len(rows))))
    coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank < 3:
        raise ValueError("near-depth support cannot define an image depth plane")
    residuals = values - design @ coefficients
    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual)))
    inliers = np.abs(residuals - median_residual) <= max(0.003, 3.0 * mad)
    if np.count_nonzero(inliers) < config.min_rim_points:
        raise ValueError("too few robust near-depth inliers")
    coefficients, _, _, _ = np.linalg.lstsq(
        design[inliers], values[inliers], rcond=None
    )
    residuals = values[inliers] - design[inliers] @ coefficients
    depth_residual = float(np.sqrt(np.mean(residuals**2)))
    depth_support_fraction = float(np.count_nonzero(inliers) / np.count_nonzero(valid_depth))
    if depth_support_fraction < config.min_ellipse_depth_support_fraction:
        raise ValueError("near-depth support fraction is too small")
    if depth_residual > config.max_ellipse_depth_residual_m:
        raise ValueError("ellipse near-depth residual exceeds the quality gate")

    center_depth = float(coefficients[2])
    if not np.isfinite(center_depth) or center_depth <= 0.0:
        raise ValueError("ellipse center depth is invalid")
    center_pixel = np.array([cx, cy])
    center_camera = center_depth * _ray(K, center_pixel)
    major_image = np.array([np.cos(angle), np.sin(angle)])
    minor_image = np.array([-major_image[1], major_image[0]])
    depth_gradient = float(np.dot(coefficients[:2], minor_image))
    transverse_tilt = float(np.sqrt(max(0.0, 1.0 - axis_ratio**2)))
    if (
        transverse_tilt > 0.2
        and abs(depth_gradient) < config.min_ellipse_depth_gradient_m_per_px
    ):
        raise ValueError("ellipse tilt sign is not resolved by the depth gradient")
    metric_minor = np.array(
        [minor_image[0] / K[0, 0], minor_image[1] / K[1, 1]]
    )
    metric_minor /= np.linalg.norm(metric_minor)
    sign = 1.0 if depth_gradient >= 0.0 else -1.0
    normal = normalize(
        np.array(
            [
                sign * transverse_tilt * metric_minor[0],
                sign * transverse_tilt * metric_minor[1],
                -axis_ratio,
            ]
        ),
        name="ellipse-derived cup axis",
    )
    center_ray = _ray(K, center_pixel)
    minor_ray_step = np.array(
        [minor_image[0] / K[0, 0], minor_image[1] / K[1, 1], 0.0]
    )
    denominator = float(np.dot(normal, center_ray))
    if abs(denominator) < 1e-6:
        raise ValueError("ellipse plane is nearly tangent to the camera ray")
    predicted_gradient = float(
        -center_depth * np.dot(normal, minor_ray_step) / denominator
    )
    gradient_relative_error = float(
        abs(predicted_gradient - depth_gradient)
        / max(abs(depth_gradient), config.min_ellipse_depth_gradient_m_per_px)
    )
    if gradient_relative_error > config.max_ellipse_gradient_relative_error:
        raise ValueError("ellipse tilt and observed depth gradient disagree")

    plane_offset = float(np.dot(normal, center_camera))

    def intersect(pixel_xy: np.ndarray) -> np.ndarray:
        ray = _ray(K, pixel_xy)
        divisor = float(np.dot(normal, ray))
        if abs(divisor) < 1e-6:
            raise ValueError("ellipse endpoint ray is parallel to its inferred plane")
        return plane_offset * ray / divisor

    major_a = intersect(center_pixel + radius_major_px * major_image)
    major_b = intersect(center_pixel - radius_major_px * major_image)
    radius = float(0.5 * np.linalg.norm(major_a - major_b))
    if not config.min_rim_radius_m <= radius <= config.max_rim_radius_m:
        raise ValueError("rim radius lies outside configured cup limits")

    offset = boundary - center_pixel
    c, s = np.cos(angle), np.sin(angle)
    ellipse_x = c * offset[:, 0] + s * offset[:, 1]
    ellipse_y = -s * offset[:, 0] + c * offset[:, 1]
    normalized_radius = np.sqrt(
        (ellipse_x / radius_major_px) ** 2
        + (ellipse_y / radius_minor_px) ** 2
    )
    normalized_error = np.abs(normalized_radius - 1.0)
    boundary_residual = float(np.quantile(normalized_error, 0.95) * radius)
    if boundary_residual > config.max_radial_residual_m:
        raise ValueError("mask boundary does not support the supplied rim ellipse")
    boundary_inliers = normalized_error <= config.max_radial_residual_m / radius
    if np.count_nonzero(boundary_inliers) < config.min_rim_points:
        raise ValueError("too little mask boundary supports the supplied rim ellipse")
    boundary_angles = np.sort(
        np.mod(
            np.arctan2(
                ellipse_y[boundary_inliers] / radius_minor_px,
                ellipse_x[boundary_inliers] / radius_major_px,
            ),
            2.0 * np.pi,
        )
    )
    gaps = np.diff(
        np.concatenate((boundary_angles, boundary_angles[:1] + 2.0 * np.pi))
    )
    coverage_deg = float(np.rad2deg(2.0 * np.pi - np.max(gaps)))
    if coverage_deg < config.min_arc_coverage_deg:
        raise ValueError("mask boundary has insufficient ellipse coverage")

    major_camera = normalize(major_a - major_b, name="rim major direction")
    major_camera = normalize(
        major_camera - np.dot(major_camera, normal) * normal,
        name="rim in-plane direction",
    )
    minor_camera = normalize(np.cross(normal, major_camera), name="rim in-plane direction")
    theta = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    rim_points = center_camera + radius * (
        np.cos(theta)[:, None] * major_camera
        + np.sin(theta)[:, None] * minor_camera
    )
    return RimFrameEstimate(
        frame_index=frame_index,
        center_camera_m=center_camera,
        cup_axis_camera=normal,
        radius_m=radius,
        rim_points_camera_m=rim_points,
        plane_residual_m=depth_residual,
        radial_residual_m=boundary_residual,
        arc_coverage_deg=coverage_deg,
        mask_area_px=area,
        extraction="calibrated_image_ellipse_near_depth_approximation",
        depth_support_fraction=depth_support_fraction,
        depth_gradient_relative_error=gradient_relative_error,
    )


def _fit_rim(
    *,
    frame_index: int,
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    config: CupRimRelationConfig,
    ellipse: np.ndarray | None,
) -> RimFrameEstimate:
    if mask.ndim != 2 or depth.shape != mask.shape:
        raise ValueError("each depth and mask frame must share shape (H, W)")
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    area = int(np.count_nonzero(mask))
    if area < config.min_rim_points:
        raise ValueError("mask has too few pixels")
    if np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
        raise ValueError("mask touches the image border")

    if ellipse is not None:
        return _fit_rim_from_image_ellipse(
            frame_index=frame_index,
            depth=depth,
            mask=mask,
            K=K,
            config=config,
            ellipse=ellipse,
            area=area,
        )

    pixels, depths = _top_contour_pixels(mask, depth, config.depth_search_px)
    extraction = "mask_upper_silhouette_circle_fit"
    if len(pixels) < config.min_rim_points:
        raise ValueError("too few valid rim depth samples")
    points = _deproject(pixels, depths, K)

    # Two robust SVD passes remove isolated background depths without using a
    # moving object centroid.
    inliers = np.ones(len(points), dtype=bool)
    for _ in range(2):
        selected = points[inliers]
        center = np.median(selected, axis=0)
        _, singular, vh = np.linalg.svd(selected - center, full_matrices=False)
        if len(singular) < 3 or singular[1] <= 1e-8:
            raise ValueError("rim samples do not span a plane")
        normal = vh[-1]
        residuals = np.abs((points - center) @ normal)
        median = float(np.median(residuals[inliers]))
        mad = float(np.median(np.abs(residuals[inliers] - median)))
        cutoff = max(config.max_plane_residual_m, median + 3.0 * max(mad, 1e-6))
        inliers = residuals <= cutoff
        if np.count_nonzero(inliers) < config.min_rim_points:
            raise ValueError("too few planar rim inliers")

    selected = points[inliers]
    plane_origin = np.mean(selected, axis=0)
    _, singular, vh = np.linalg.svd(selected - plane_origin, full_matrices=False)
    basis = vh[:2].T
    normal = normalize(vh[-1], name="cup axis")
    plane_residual = float(np.sqrt(np.mean(((selected - plane_origin) @ normal) ** 2)))
    if plane_residual > config.max_plane_residual_m:
        raise ValueError("rim plane residual exceeds the quality gate")

    xy = (selected - plane_origin) @ basis
    scale = float(np.sqrt(np.mean(np.sum(xy**2, axis=1))))
    if scale <= 1e-8:
        raise ValueError("rim samples collapse to one point")
    normalized_xy = xy / scale
    design = np.column_stack((2.0 * normalized_xy, np.ones(len(normalized_xy))))
    rhs = np.sum(normalized_xy**2, axis=1)
    solution, _, rank, singular_design = np.linalg.lstsq(design, rhs, rcond=None)
    if rank < 3 or singular_design[-1] <= 1e-8:
        raise ValueError("rim circle fit is ill-conditioned")
    center_xy = solution[:2] * scale
    radius_sq = float(solution[2] + np.dot(solution[:2], solution[:2]))
    if radius_sq <= 0.0:
        raise ValueError("rim circle fit has non-positive radius")
    radius = float(np.sqrt(radius_sq) * scale)
    center_camera = plane_origin + basis @ center_xy
    radial_distances = np.linalg.norm((selected - center_camera) @ basis, axis=1)
    radial_residual = float(np.sqrt(np.mean((radial_distances - radius) ** 2)))
    if not config.min_rim_radius_m <= radius <= config.max_rim_radius_m:
        raise ValueError("rim radius lies outside configured cup limits")
    if radial_residual > config.max_radial_residual_m:
        raise ValueError("rim radial residual exceeds the quality gate")

    centered_xy = (selected - center_camera) @ basis
    angles = np.sort(np.mod(np.arctan2(centered_xy[:, 1], centered_xy[:, 0]), 2.0 * np.pi))
    gaps = np.diff(np.concatenate((angles, angles[:1] + 2.0 * np.pi)))
    coverage_deg = float(np.rad2deg(2.0 * np.pi - np.max(gaps)))
    if coverage_deg < config.min_arc_coverage_deg:
        raise ValueError("visible rim arc is too short for an unambiguous circle")

    # The open end is assumed to face the camera.  This sign convention is
    # reported for review and is never sufficient to execute a robot motion.
    if np.dot(normal, -center_camera) < 0.0:
        normal = -normal
    return RimFrameEstimate(
        frame_index=frame_index,
        center_camera_m=center_camera,
        cup_axis_camera=normal,
        radius_m=radius,
        rim_points_camera_m=selected,
        plane_residual_m=plane_residual,
        radial_residual_m=radial_residual,
        arc_coverage_deg=coverage_deg,
        mask_area_px=area,
        extraction=extraction,
        depth_support_fraction=float(len(selected) / area),
        depth_gradient_relative_error=None,
    )


def _unsigned_direction_mean(directions: np.ndarray) -> np.ndarray:
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    scatter = np.einsum("ni,nj->ij", unit, unit)
    values, vectors = np.linalg.eigh(scatter)
    return normalize(vectors[:, int(np.argmax(values))], name="unsigned closing axis")


def _unsigned_angles_deg(directions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip(np.abs(unit @ reference), 0.0, 1.0)))


def compile_cup_rim_relation(
    *,
    precontact_depth_m: np.ndarray,
    precontact_masks: np.ndarray,
    precontact_frame_indices: np.ndarray,
    K: np.ndarray,
    hand_joints_camera_m: np.ndarray,
    hand_joint_confidence: np.ndarray,
    stage_intervals: Mapping[str, tuple[int, int] | list[int]],
    rim_ellipses_px: np.ndarray | None = None,
    config: CupRimRelationConfig = CupRimRelationConfig(),
) -> CupRimRelationArtifacts:
    """Compile a frozen pre-contact rim frame and demonstrated pinch relation.

    Stage intervals are half-open.  The output is a human-geometry proposal for
    review; robot embodiment, grounding, IK, collision and execution are absent.
    """

    depths = np.asarray(precontact_depth_m, dtype=float)
    masks = np.asarray(precontact_masks, dtype=bool)
    indices = np.asarray(precontact_frame_indices, dtype=int)
    intrinsics = np.asarray(K, dtype=float)
    joints = np.asarray(hand_joints_camera_m, dtype=float)
    confidence = np.asarray(hand_joint_confidence, dtype=float)
    frame_count = len(joints)
    if depths.ndim != 3 or masks.shape != depths.shape:
        raise ValueError("precontact depth and masks must share shape (P, H, W)")
    if indices.shape != (len(depths),) or len(np.unique(indices)) != len(indices):
        raise ValueError("precontact_frame_indices must be unique with shape (P,)")
    if len(indices) < config.min_precontact_frames:
        raise ValueError("too few requested precontact frames")
    if np.any(indices < 0) or np.any(indices >= frame_count):
        raise ValueError("precontact frame index lies outside the hand sequence")
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("K must be a finite 3x3 matrix")
    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        raise ValueError("K focal lengths must be positive")
    if joints.shape != (frame_count, 21, 3):
        raise ValueError("hand_joints_camera_m must have shape (F, 21, 3)")
    if confidence.shape != (frame_count, 21):
        raise ValueError("hand_joint_confidence must have shape (F, 21)")
    ellipses = None if rim_ellipses_px is None else np.asarray(rim_ellipses_px, dtype=float)
    if ellipses is not None and ellipses.shape != (len(depths), 5):
        raise ValueError("rim_ellipses_px must have shape (P, 5)")

    preshape_start, preshape_end = _require_interval(
        stage_intervals, "PRESHAPE", frame_count
    )
    grasp_start, grasp_end = _require_interval(stage_intervals, "GRASP", frame_count)
    if preshape_end > grasp_start:
        raise ValueError("PRESHAPE must end no later than GRASP starts")
    precontact_start = preshape_start
    if "APPROACH" in stage_intervals:
        approach_start, approach_end = _require_interval(
            stage_intervals, "APPROACH", frame_count
        )
        if approach_end > preshape_start:
            raise ValueError("APPROACH must end no later than PRESHAPE starts")
        precontact_start = approach_start
    if np.any(indices < precontact_start) or np.any(indices >= preshape_end):
        raise ValueError("all precontact frames must lie inside APPROACH or PRESHAPE")

    estimates: list[RimFrameEstimate] = []
    failures: dict[int, str] = {}
    for local_index, frame_index in enumerate(indices):
        try:
            estimates.append(
                _fit_rim(
                    frame_index=int(frame_index),
                    depth=depths[local_index],
                    mask=masks[local_index],
                    K=intrinsics,
                    config=config,
                    ellipse=None if ellipses is None else ellipses[local_index],
                )
            )
        except ValueError as error:
            failures[int(frame_index)] = str(error)
    required = max(
        config.min_precontact_frames,
        int(np.ceil(config.min_valid_precontact_fraction * len(indices))),
    )
    if len(estimates) < required:
        raise ValueError(
            f"only {len(estimates)}/{len(indices)} precontact rim frames passed; "
            f"need {required}; failures={failures}"
        )

    areas = np.asarray([estimate.mask_area_px for estimate in estimates], dtype=float)
    area_cv = float(np.std(areas) / np.mean(areas))
    if area_cv > config.max_mask_area_cv:
        raise ValueError("precontact mask area is unstable")
    centers = np.stack([estimate.center_camera_m for estimate in estimates])
    center = robust_mean(centers)
    center_spread = float(np.quantile(np.linalg.norm(centers - center, axis=1), 0.95))
    if center_spread > config.max_rim_center_spread_m:
        raise ValueError("precontact rim center is unstable")
    axes = np.stack([estimate.cup_axis_camera for estimate in estimates])
    cup_axis = robust_direction_mean(axes, name="cup axis")
    axis_angles = np.rad2deg(np.arccos(np.clip(axes @ cup_axis, -1.0, 1.0)))
    axis_spread = float(np.quantile(axis_angles, 0.95))
    if axis_spread > config.max_cup_axis_spread_deg:
        raise ValueError("precontact cup axis is unstable")
    radius = float(np.median([estimate.radius_m for estimate in estimates]))

    grasp_indices = np.arange(grasp_start, grasp_end)
    valid = np.isfinite(joints[grasp_indices][:, [WRIST, THUMB_TIP, INDEX_TIP]]).all(
        axis=(1, 2)
    )
    valid &= np.isfinite(
        confidence[grasp_indices][:, [WRIST, THUMB_TIP, INDEX_TIP]]
    ).all(axis=1)
    valid &= np.all(
        confidence[grasp_indices][:, [WRIST, THUMB_TIP, INDEX_TIP]]
        >= config.min_joint_confidence,
        axis=1,
    )
    selected_frames = grasp_indices[valid]
    if len(selected_frames) < config.min_grasp_frames:
        raise ValueError("too few confident thumb/index/wrist frames in GRASP")
    thumb = joints[selected_frames, THUMB_TIP]
    index = joints[selected_frames, INDEX_TIP]
    wrist = joints[selected_frames, WRIST]
    closing = index - thumb
    nonzero = np.linalg.norm(closing, axis=1) > 1e-6
    selected_frames = selected_frames[nonzero]
    thumb, index, wrist, closing = (
        values[nonzero] for values in (thumb, index, wrist, closing)
    )
    if len(selected_frames) < config.min_grasp_frames:
        raise ValueError("too few GRASP frames define a closing axis")
    pinch_centers = 0.5 * (thumb + index)
    pinch_center = robust_mean(pinch_centers)
    pinch_spread = float(
        np.quantile(np.linalg.norm(pinch_centers - pinch_center, axis=1), 0.95)
    )
    if pinch_spread > config.max_pinch_center_spread_m:
        raise ValueError("pinch center is unstable during GRASP")
    closing_axis = _unsigned_direction_mean(closing)
    closing_spread = float(np.quantile(_unsigned_angles_deg(closing, closing_axis), 0.95))
    if closing_spread > config.max_closing_axis_spread_deg:
        raise ValueError("closing line is ambiguous during GRASP")

    rim_to_pinch = pinch_center - center
    axial_offset = float(np.dot(rim_to_pinch, cup_axis))
    radial_vector = rim_to_pinch - axial_offset * cup_axis
    radial_distance = float(np.linalg.norm(radial_vector))
    if radial_distance < config.min_radial_offset_m:
        raise ValueError("pinch is too close to the rim center to define a radial direction")
    if abs(radial_distance - radius) > config.max_rim_radial_error_m:
        raise ValueError("pinch center does not lie near the fitted rim")
    if abs(axial_offset) > config.max_rim_axial_error_m:
        raise ValueError("pinch center is too far above or below the fitted rim plane")
    radial_axis = radial_vector / radial_distance
    tangential_axis = normalize(np.cross(cup_axis, radial_axis), name="rim tangent")
    radial_axis = normalize(np.cross(tangential_axis, cup_axis), name="rim radial")
    closing_radial_angle = float(
        np.rad2deg(np.arccos(np.clip(abs(np.dot(closing_axis, radial_axis)), 0.0, 1.0)))
    )
    if closing_radial_angle > config.max_closing_to_radial_angle_deg:
        raise ValueError("demonstrated closing line is not radial to the cup rim")

    # Pick the undirected closing representative closest to +radial for stable JSON.
    if np.dot(closing_axis, radial_axis) < 0.0:
        closing_axis = -closing_axis
    approach = pinch_centers - wrist
    approach -= np.sum(approach * closing_axis, axis=1, keepdims=True) * closing_axis
    approach_valid = np.linalg.norm(approach, axis=1) > 1e-6
    if np.count_nonzero(approach_valid) < config.min_grasp_frames:
        raise ValueError("wrist-to-pinch direction cannot define a TCP approach axis")
    approach_axis = robust_direction_mean(
        approach[approach_valid], name="human wrist-to-pinch approach"
    )
    tcp_x = normalize(
        closing_axis - np.dot(closing_axis, approach_axis) * approach_axis,
        name="TCP closing axis",
    )
    tcp_y = normalize(np.cross(approach_axis, tcp_x), name="TCP lateral axis")
    tcp_z = normalize(np.cross(tcp_x, tcp_y), name="TCP approach axis")

    T_camera_rim = np.eye(4)
    T_camera_rim[:3, :3] = np.column_stack(
        (radial_axis, tangential_axis, cup_axis)
    )
    T_camera_rim[:3, 3] = center
    T_camera_tcp = np.eye(4)
    T_camera_tcp[:3, :3] = np.column_stack((tcp_x, tcp_y, tcp_z))
    T_camera_tcp[:3, 3] = pinch_center
    T_rim_camera = np.eye(4)
    T_rim_camera[:3, :3] = T_camera_rim[:3, :3].T
    T_rim_camera[:3, 3] = -T_camera_rim[:3, :3].T @ center
    T_rim_tcp = T_rim_camera @ T_camera_tcp

    representative_rim = min(
        estimates, key=lambda item: np.linalg.norm(item.center_camera_m - center)
    )
    representative_grasp = int(
        selected_frames[np.argmin(np.linalg.norm(pinch_centers - pinch_center, axis=1))]
    )
    quality: dict[str, Any] = {
        "review_only": True,
        "execution_authorized": False,
        "object_anchor": "frozen_precontact_top_rim_frame",
        "grasp_time_mask_centroids_used": False,
        "requested_precontact_frames": indices.tolist(),
        "valid_precontact_frames": [estimate.frame_index for estimate in estimates],
        "precontact_failures": failures,
        "valid_precontact_fraction": len(estimates) / len(indices),
        "rim_extraction_methods": sorted({estimate.extraction for estimate in estimates}),
        "mask_area_cv": area_cv,
        "rim_center_spread_p95_m": center_spread,
        "cup_axis_spread_p95_deg": axis_spread,
        "rim_plane_residual_max_m": max(item.plane_residual_m for item in estimates),
        "rim_radial_residual_max_m": max(item.radial_residual_m for item in estimates),
        "rim_arc_coverage_min_deg": min(item.arc_coverage_deg for item in estimates),
        "rim_depth_support_fraction_min": min(
            item.depth_support_fraction for item in estimates
        ),
        "ellipse_depth_gradient_relative_error_max": (
            max(
                item.depth_gradient_relative_error
                for item in estimates
                if item.depth_gradient_relative_error is not None
            )
            if any(item.depth_gradient_relative_error is not None for item in estimates)
            else None
        ),
        "rim_radius_median_m": radius,
        "valid_grasp_frames": selected_frames.tolist(),
        "pinch_center_spread_p95_m": pinch_spread,
        "closing_axis_spread_p95_deg": closing_spread,
        "closing_to_local_radial_angle_deg": closing_radial_angle,
        "pinch_radial_distance_m": radial_distance,
        "pinch_radial_error_m": abs(radial_distance - radius),
        "pinch_axial_offset_m": axial_offset,
        "min_selected_joint_confidence": float(
            np.min(confidence[selected_frames][:, [WRIST, THUMB_TIP, INDEX_TIP]])
        ),
        "cup_axis_sign_convention": "top opening assumed to face camera",
    }
    return CupRimRelationArtifacts(
        T_camera_rim=T_camera_rim,
        T_rim_human_pinch_tcp=T_rim_tcp,
        T_camera_human_pinch_tcp=T_camera_tcp,
        rim_radius_m=radius,
        pinch_width_m=float(np.median(np.linalg.norm(closing, axis=1))),
        precontact_frame_indices=indices,
        valid_precontact_frame_indices=np.asarray(
            [estimate.frame_index for estimate in estimates], dtype=int
        ),
        grasp_frame_indices=selected_frames,
        representative_precontact_frame=representative_rim.frame_index,
        representative_grasp_frame=representative_grasp,
        rim_points_camera_m=representative_rim.rim_points_camera_m,
        quality=quality,
    )


__all__ = [
    "CupRimRelationArtifacts",
    "CupRimRelationConfig",
    "compile_cup_rim_relation",
]
