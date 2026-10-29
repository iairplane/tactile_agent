"""Fail-closed RGB-D localization for one object resting on a support surface.

This module deliberately estimates only a translation anchor from observed surface
points.  It does not infer object orientation, a grasp pose, or a robot command.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .fixed_object import validate_intrinsics
from .franka import validate_rigid_transform


@dataclass(frozen=True)
class SupportedObjectConfig:
    """Geometric and quality limits for supported-object localization."""

    workspace_min_base_m: np.ndarray
    workspace_max_base_m: np.ndarray
    support_top_base_z_m: float
    min_height_above_support_m: float
    min_depth_m: float = 0.15
    max_depth_m: float = 1.5
    min_component_points: int = 150
    min_dominance_ratio: float = 1.8
    morphology_radius_px: int = 1
    min_frames: int = 3
    max_center_deviation_m: float = 0.008
    max_point_count_cv: float = 0.25


@dataclass(frozen=True)
class ComponentEvidence:
    """Geometry measured for one above-support image component."""

    label: int
    point_count: int
    center_base_m: np.ndarray
    bounds_min_base_m: np.ndarray
    bounds_max_base_m: np.ndarray
    robust_bounds_min_base_m: np.ndarray
    robust_bounds_max_base_m: np.ndarray
    bbox_xywh: tuple[int, int, int, int]
    center_uv_px: np.ndarray
    points_base_m: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class FrameEvidence:
    """Selected component and ambiguity evidence for one depth frame."""

    selected: ComponentEvidence
    eligible_component_count: int
    second_largest_point_count: int
    dominance_ratio: float | None


@dataclass(frozen=True)
class LocalizationResult:
    """Stable translation anchor and the selected mask from the last frame."""

    frames: tuple[FrameEvidence, ...]
    center_base_m: np.ndarray
    bounds_min_base_m: np.ndarray
    bounds_max_base_m: np.ndarray
    center_rms_deviation_m: float
    center_max_deviation_m: float
    point_count_cv: float
    final_mask: np.ndarray


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return vector


def conservative_projected_width(
    points_by_frame: list[np.ndarray] | tuple[np.ndarray, ...],
    axis_base: np.ndarray,
    *,
    lower_quantile: float = 0.0,
    upper_quantile: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Measure object width along an arbitrary base-frame axis.

    Each frame contributes a projection span.  The default includes every selected
    point.  The returned width is the maximum span across frames so that a partially
    occluded frame cannot shrink a better-observed width from another accepted frame.
    """

    axis = _finite_vector(axis_base, name="projection axis")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("projection axis must be non-zero")
    if not (
        np.isfinite([lower_quantile, upper_quantile]).all()
        and 0.0 <= lower_quantile < upper_quantile <= 1.0
    ):
        raise ValueError("projection quantiles must satisfy 0 <= lower < upper <= 1")
    if not points_by_frame:
        raise ValueError("at least one point-cloud frame is required")

    unit_axis = axis / norm
    spans: list[float] = []
    for frame_index, raw_points in enumerate(points_by_frame):
        points = np.asarray(raw_points, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
            raise ValueError(
                f"point-cloud frame {frame_index} must have shape (N, 3), N >= 2"
            )
        if not np.isfinite(points).all():
            raise ValueError(f"point-cloud frame {frame_index} contains NaN or infinity")
        projections = points @ unit_axis
        lower, upper = np.quantile(
            projections, [lower_quantile, upper_quantile]
        )
        span = float(upper - lower)
        if not np.isfinite(span) or span <= 0.0:
            raise ValueError(f"point-cloud frame {frame_index} has no projected width")
        spans.append(span)
    frame_spans = np.asarray(spans, dtype=float)
    return float(np.max(frame_spans)), frame_spans


def validate_config(config: SupportedObjectConfig) -> tuple[np.ndarray, np.ndarray]:
    """Validate all limits before any frame is accepted."""

    workspace_min = _finite_vector(
        config.workspace_min_base_m, name="workspace_min_base_m"
    )
    workspace_max = _finite_vector(
        config.workspace_max_base_m, name="workspace_max_base_m"
    )
    if np.any(workspace_min >= workspace_max):
        raise ValueError("workspace bounds must satisfy min < max on every axis")
    finite_scalars = {
        "support_top_base_z_m": config.support_top_base_z_m,
        "min_height_above_support_m": config.min_height_above_support_m,
        "min_depth_m": config.min_depth_m,
        "max_depth_m": config.max_depth_m,
        "min_dominance_ratio": config.min_dominance_ratio,
        "max_center_deviation_m": config.max_center_deviation_m,
        "max_point_count_cv": config.max_point_count_cv,
    }
    for name, value in finite_scalars.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 < config.min_depth_m < config.max_depth_m:
        raise ValueError("depth limits must satisfy 0 < min_depth_m < max_depth_m")
    if config.min_height_above_support_m <= 0.0:
        raise ValueError("min_height_above_support_m must be positive")
    object_threshold = config.support_top_base_z_m + config.min_height_above_support_m
    if not workspace_min[2] <= object_threshold < workspace_max[2]:
        raise ValueError("above-support threshold must lie inside the workspace z bounds")
    if config.min_component_points < 1:
        raise ValueError("min_component_points must be positive")
    if config.min_dominance_ratio <= 1.0:
        raise ValueError("min_dominance_ratio must be greater than one")
    if config.morphology_radius_px < 0:
        raise ValueError("morphology_radius_px must be non-negative")
    if config.min_frames < 2:
        raise ValueError("min_frames must be at least two")
    if config.max_center_deviation_m <= 0.0:
        raise ValueError("max_center_deviation_m must be positive")
    if config.max_point_count_cv < 0.0:
        raise ValueError("max_point_count_cv must be non-negative")
    return workspace_min, workspace_max


def _base_points_for_depth(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    T_base_camera: np.ndarray,
) -> np.ndarray:
    """Return an ``(H,W,3)`` array of base-frame points."""

    rows, columns = np.indices(depth_m.shape, dtype=np.float64)
    z = depth_m.astype(np.float64, copy=False)
    camera_points = np.stack(
        (
            (columns - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (rows - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        ),
        axis=-1,
    )
    return camera_points @ T_base_camera[:3, :3].T + T_base_camera[:3, 3]


def _connected_components(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    T_base_camera: np.ndarray,
    config: SupportedObjectConfig,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
) -> tuple[list[ComponentEvidence], np.ndarray, np.ndarray]:
    if depth_m.ndim != 2:
        raise ValueError(f"depth frame must be two-dimensional, got {depth_m.shape}")
    points_base = _base_points_for_depth(depth_m, intrinsics, T_base_camera)
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= config.min_depth_m)
        & (depth_m <= config.max_depth_m)
        & np.isfinite(points_base).all(axis=2)
        & np.all(points_base >= workspace_min, axis=2)
        & np.all(points_base <= workspace_max, axis=2)
        & (
            points_base[:, :, 2]
            >= config.support_top_base_z_m + config.min_height_above_support_m
        )
    )
    connected_mask = valid.astype(np.uint8)
    radius = config.morphology_radius_px
    if radius:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        connected_mask = cv2.morphologyEx(connected_mask, cv2.MORPH_CLOSE, kernel)
        connected_mask = cv2.morphologyEx(connected_mask, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        connected_mask, connectivity=8
    )
    components: list[ComponentEvidence] = []
    for label in range(1, count):
        source_pixels = valid & (labels == label)
        rows, columns = np.nonzero(source_pixels)
        point_count = len(rows)
        if point_count < config.min_component_points:
            continue
        points = points_base[rows, columns]
        components.append(
            ComponentEvidence(
                label=label,
                point_count=point_count,
                center_base_m=np.median(points, axis=0),
                bounds_min_base_m=np.min(points, axis=0),
                bounds_max_base_m=np.max(points, axis=0),
                robust_bounds_min_base_m=np.quantile(points, 0.01, axis=0),
                robust_bounds_max_base_m=np.quantile(points, 0.99, axis=0),
                bbox_xywh=(
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_HEIGHT]),
                ),
                center_uv_px=np.asarray(
                    [float(np.median(columns)), float(np.median(rows))]
                ),
                points_base_m=points,
            )
        )
    components.sort(key=lambda component: component.point_count, reverse=True)
    return components, labels, valid


def localize_supported_object(
    depth_frames_m: list[np.ndarray] | tuple[np.ndarray, ...],
    intrinsics: np.ndarray,
    T_base_camera: np.ndarray,
    config: SupportedObjectConfig,
) -> LocalizationResult:
    """Locate one dominant, stationary above-support component across all frames.

    Raises ``ValueError`` when the evidence is incomplete, ambiguous, or unstable.
    No partial localization result is returned on rejection.
    """

    workspace_min, workspace_max = validate_config(config)
    intrinsics = validate_intrinsics(intrinsics)
    T_base_camera = validate_rigid_transform(T_base_camera, name="T_base_camera")
    if len(depth_frames_m) < config.min_frames:
        raise ValueError(
            f"need at least {config.min_frames} frames, got {len(depth_frames_m)}"
        )

    expected_shape: tuple[int, int] | None = None
    frame_evidence: list[FrameEvidence] = []
    final_mask: np.ndarray | None = None
    for frame_index, raw_depth in enumerate(depth_frames_m):
        depth = np.asarray(raw_depth)
        if not np.issubdtype(depth.dtype, np.floating):
            raise ValueError(f"depth frame {frame_index} must use floating-point metres")
        if expected_shape is None:
            expected_shape = depth.shape
        if depth.shape != expected_shape:
            raise ValueError(
                f"depth frame {frame_index} shape {depth.shape} differs from {expected_shape}"
            )
        components, labels, source_mask = _connected_components(
            depth,
            intrinsics,
            T_base_camera,
            config,
            workspace_min,
            workspace_max,
        )
        if not components:
            raise ValueError(
                f"frame {frame_index} has no above-support component with at least "
                f"{config.min_component_points} points"
            )
        selected = components[0]
        second_count = components[1].point_count if len(components) > 1 else 0
        dominance = selected.point_count / second_count if second_count else None
        if dominance is not None and dominance < config.min_dominance_ratio:
            raise ValueError(
                f"frame {frame_index} is ambiguous: largest/second component ratio "
                f"{dominance:.3f} is below {config.min_dominance_ratio:.3f}"
            )
        frame_evidence.append(
            FrameEvidence(
                selected=selected,
                eligible_component_count=len(components),
                second_largest_point_count=second_count,
                dominance_ratio=dominance,
            )
        )
        final_mask = (labels == selected.label) & source_mask

    centers = np.stack([frame.selected.center_base_m for frame in frame_evidence])
    center = np.median(centers, axis=0)
    center_residuals = np.linalg.norm(centers - center, axis=1)
    center_rms = float(np.sqrt(np.mean(center_residuals**2)))
    center_max = float(np.max(center_residuals))
    if center_max > config.max_center_deviation_m:
        raise ValueError(
            f"component is not stable: maximum center deviation {center_max:.6f} m "
            f"exceeds {config.max_center_deviation_m:.6f} m"
        )

    point_counts = np.asarray(
        [frame.selected.point_count for frame in frame_evidence], dtype=float
    )
    point_count_cv = float(np.std(point_counts) / np.mean(point_counts))
    if point_count_cv > config.max_point_count_cv:
        raise ValueError(
            f"component point count is not stable: CV {point_count_cv:.3f} exceeds "
            f"{config.max_point_count_cv:.3f}"
        )

    robust_min = np.stack(
        [frame.selected.robust_bounds_min_base_m for frame in frame_evidence]
    )
    robust_max = np.stack(
        [frame.selected.robust_bounds_max_base_m for frame in frame_evidence]
    )
    assert final_mask is not None
    return LocalizationResult(
        frames=tuple(frame_evidence),
        center_base_m=center,
        bounds_min_base_m=np.median(robust_min, axis=0),
        bounds_max_base_m=np.median(robust_max, axis=0),
        center_rms_deviation_m=center_rms,
        center_max_deviation_m=center_max,
        point_count_cv=point_count_cv,
        final_mask=final_mask,
    )
