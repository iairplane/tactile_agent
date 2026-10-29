"""Deterministic RGB-D localization for the current upright black-cap bottle scene.

This is deliberately a narrow, review-only localizer.  It assumes that the medicine
bottle is already upright on the table, that its cap is the dominant dark connected
component inside an operator supplied image ROI, and that aligned metric depth is
available.  It does not recognize arbitrary bottles, estimate bottle tilt, infer cap
yaw, or authorize robot motion.

The returned cap frame uses the repository's strict
``medicine_bottle_cap_frame.v1`` contract.  Its origin is the robust centre of the
observed cap points, its local axes are aligned with the FR3 base frame, and the
bottle axis is fixed to base ``+Z`` by the stated upright-bottle prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .fixed_object import validate_intrinsics
from .franka import validate_rigid_transform
from .medicine_bottle import BottleCapFrame

LOCALIZATION_SCHEMA = "demo2gripper.current_scene_medicine_bottle_localization.v1"


@dataclass(frozen=True)
class CurrentBottleCapConfig:
    """Scene-specific colour, depth, size, and ambiguity gates.

    ``roi_xyxy_px`` is ``(x_min, y_min, x_max_exclusive, y_max_exclusive)``.
    Requiring an explicit ROI prevents dark tripods and cables elsewhere in the
    current workspace from being mistaken for the black cap.
    """

    roi_xyxy_px: tuple[int, int, int, int]
    max_dark_value: int = 95
    min_depth_m: float = 0.15
    max_depth_m: float = 1.50
    min_component_pixels: int = 120
    min_valid_depth_ratio: float = 0.50
    min_dominance_ratio: float = 1.35
    morphology_radius_px: int = 2
    component_margin_px: int = 1
    min_cap_diameter_m: float = 0.012
    max_cap_diameter_m: float = 0.080
    robust_quantile: float = 0.05


@dataclass(frozen=True)
class BottleCapLocalization:
    """Review evidence for one selected black-cap component."""

    frame: BottleCapFrame
    selected_mask: np.ndarray
    roi_xyxy_px: tuple[int, int, int, int]
    bbox_xywh_px: tuple[int, int, int, int]
    center_uv_px: tuple[float, float]
    cap_center_base_m: np.ndarray
    robust_bounds_base_m: np.ndarray
    point_count: int
    component_pixel_count: int
    valid_depth_ratio: float
    eligible_component_count: int
    dominance_ratio: float | None
    estimated_cap_diameter_m: float
    depth_median_m: float
    depth_mad_m: float

    def report_dict(self) -> dict[str, object]:
        return {
            "schema": LOCALIZATION_SCHEMA,
            "status": "review_only",
            "execution_authorized": False,
            "hardware_execution_enabled": False,
            "scope": "current_upright_black_cap_medicine_bottle_scene_only",
            "assumptions": {
                "bottle_is_upright_on_table": True,
                "bottle_axis_base_is_positive_z": True,
                "cap_is_dominant_dark_component_inside_explicit_roi": True,
                "aligned_metric_depth": True,
                "cap_yaw_observed": False,
                "tcp_yaw_policy": "base_aligned_due_to_rotational_symmetry",
            },
            "method": {
                "segmentation": "dark_connected_component_inside_roi",
                "centre": "midpoint_of_robust_base_frame_point_bounds",
                "robust_bounds_quantiles": "configured symmetric lower/upper quantiles",
            },
            "roi_xyxy_px": list(self.roi_xyxy_px),
            "bbox_xywh_px": list(self.bbox_xywh_px),
            "center_uv_px": list(self.center_uv_px),
            "cap_center_base_m": self.cap_center_base_m.tolist(),
            "robust_bounds_base_m": {
                "min": self.robust_bounds_base_m[0].tolist(),
                "max": self.robust_bounds_base_m[1].tolist(),
            },
            "quality": {
                "point_count": self.point_count,
                "component_pixel_count": self.component_pixel_count,
                "valid_depth_ratio": self.valid_depth_ratio,
                "eligible_component_count": self.eligible_component_count,
                "dominance_ratio": self.dominance_ratio,
                "estimated_cap_diameter_m": self.estimated_cap_diameter_m,
                "depth_median_m": self.depth_median_m,
                "depth_mad_m": self.depth_mad_m,
            },
            "output_contract": "medicine_bottle_cap_frame.json",
        }


@dataclass(frozen=True)
class _Candidate:
    label: int
    mask: np.ndarray
    bbox_xywh_px: tuple[int, int, int, int]
    component_pixel_count: int
    rows: np.ndarray
    columns: np.ndarray
    depths_m: np.ndarray
    points_base_m: np.ndarray
    valid_depth_ratio: float
    robust_bounds_base_m: np.ndarray
    estimated_cap_diameter_m: float


def _validate_config(
    config: CurrentBottleCapConfig, image_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    height, width = image_shape
    roi = config.roi_xyxy_px
    if (
        not isinstance(roi, tuple)
        or len(roi) != 4
        or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in roi)
    ):
        raise ValueError("roi_xyxy_px must contain exactly four integer coordinates")
    x_min, y_min, x_max, y_max = (int(value) for value in roi)
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        raise ValueError(
            f"roi_xyxy_px must lie inside the {width}x{height} image with min < max"
        )
    if isinstance(config.max_dark_value, bool) or not 0 <= config.max_dark_value <= 255:
        raise ValueError("max_dark_value must be an integer in [0, 255]")
    finite = {
        "min_depth_m": config.min_depth_m,
        "max_depth_m": config.max_depth_m,
        "min_valid_depth_ratio": config.min_valid_depth_ratio,
        "min_dominance_ratio": config.min_dominance_ratio,
        "min_cap_diameter_m": config.min_cap_diameter_m,
        "max_cap_diameter_m": config.max_cap_diameter_m,
        "robust_quantile": config.robust_quantile,
    }
    for name, value in finite.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 < config.min_depth_m < config.max_depth_m:
        raise ValueError("depth bounds must satisfy 0 < min_depth_m < max_depth_m")
    if config.min_component_pixels < 3:
        raise ValueError("min_component_pixels must be at least three")
    if not 0.0 < config.min_valid_depth_ratio <= 1.0:
        raise ValueError("min_valid_depth_ratio must lie in (0, 1]")
    if config.min_dominance_ratio <= 1.0:
        raise ValueError("min_dominance_ratio must be greater than one")
    if config.morphology_radius_px < 0:
        raise ValueError("morphology_radius_px must be non-negative")
    if config.component_margin_px < 0:
        raise ValueError("component_margin_px must be non-negative")
    if not 0.0 < config.min_cap_diameter_m < config.max_cap_diameter_m:
        raise ValueError("cap diameter bounds must satisfy 0 < min < max")
    if not 0.0 <= config.robust_quantile < 0.25:
        raise ValueError("robust_quantile must lie in [0, 0.25)")
    return x_min, y_min, x_max, y_max


def _backproject_selected(
    rows: np.ndarray,
    columns: np.ndarray,
    depths_m: np.ndarray,
    intrinsics: np.ndarray,
    T_base_camera: np.ndarray,
) -> np.ndarray:
    points_camera = np.column_stack(
        (
            (columns.astype(float) - intrinsics[0, 2]) * depths_m / intrinsics[0, 0],
            (rows.astype(float) - intrinsics[1, 2]) * depths_m / intrinsics[1, 1],
            depths_m,
        )
    )
    return points_camera @ T_base_camera[:3, :3].T + T_base_camera[:3, 3]


def localize_current_bottle_cap(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    T_base_camera: np.ndarray,
    config: CurrentBottleCapConfig,
    *,
    base_frame: str = "fr3_link0",
    cap_frame: str = "medicine_cap_axis",
    tcp_frame: str = "hande_tcp",
) -> BottleCapLocalization:
    """Locate the black cap under explicit current-scene assumptions.

    Dark components are found from RGB alone, while component acceptance and the
    resulting centre use aligned metric depth.  More than one plausible component
    must pass a dominance test; otherwise localization fails closed.
    """

    image = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=float)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError("rgb must use uint8 values")
    if depth.shape != image.shape[:2]:
        raise ValueError(
            f"depth shape {depth.shape} does not match RGB shape {image.shape[:2]}"
        )
    K = validate_intrinsics(intrinsics)
    base_camera = validate_rigid_transform(T_base_camera, name="T_base_camera")
    x_min, y_min, x_max, y_max = _validate_config(config, depth.shape)

    # Channel maximum keeps this independent of whether the caller labels the byte
    # order RGB or BGR.  The current cap is black; colour identity is not inferred.
    dark = np.max(image, axis=2) <= config.max_dark_value
    roi_mask = np.zeros(depth.shape, dtype=bool)
    roi_mask[y_min:y_max, x_min:x_max] = True
    dark &= roi_mask
    connected = dark.astype(np.uint8)
    radius = config.morphology_radius_px
    if radius:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel)
        connected = cv2.morphologyEx(connected, cv2.MORPH_OPEN, kernel)
        connected[~roi_mask] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    candidates: list[_Candidate] = []
    for label in range(1, count):
        component_pixel_count = int(stats[label, cv2.CC_STAT_AREA])
        if component_pixel_count < config.min_component_pixels:
            continue
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        right = left + width
        bottom = top + height
        margin = config.component_margin_px
        if (
            left < x_min + margin
            or top < y_min + margin
            or right > x_max - margin
            or bottom > y_max - margin
        ):
            continue

        component_mask = labels == label
        valid = (
            component_mask
            & np.isfinite(depth)
            & (depth >= config.min_depth_m)
            & (depth <= config.max_depth_m)
        )
        rows, columns = np.nonzero(valid)
        valid_ratio = float(len(rows) / component_pixel_count)
        if len(rows) < config.min_component_pixels or valid_ratio < config.min_valid_depth_ratio:
            continue
        depths = depth[rows, columns]
        points_base = _backproject_selected(
            rows, columns, depths, K, base_camera
        )
        lower = config.robust_quantile
        robust_bounds = np.quantile(points_base, [lower, 1.0 - lower], axis=0)
        xy_span = robust_bounds[1, :2] - robust_bounds[0, :2]
        diameter = float(np.max(xy_span))
        if not config.min_cap_diameter_m <= diameter <= config.max_cap_diameter_m:
            continue
        candidates.append(
            _Candidate(
                label=label,
                mask=component_mask,
                bbox_xywh_px=(left, top, width, height),
                component_pixel_count=component_pixel_count,
                rows=rows,
                columns=columns,
                depths_m=depths,
                points_base_m=points_base,
                valid_depth_ratio=valid_ratio,
                robust_bounds_base_m=robust_bounds,
                estimated_cap_diameter_m=diameter,
            )
        )

    if not candidates:
        raise ValueError(
            "no black-cap component passed the ROI, depth, size, and boundary gates"
        )
    candidates.sort(key=lambda item: item.component_pixel_count, reverse=True)
    selected = candidates[0]
    dominance: float | None = None
    if len(candidates) > 1:
        dominance = selected.component_pixel_count / candidates[1].component_pixel_count
        if dominance < config.min_dominance_ratio:
            raise ValueError(
                "black-cap localization is ambiguous: largest/second component ratio "
                f"{dominance:.3f} is below {config.min_dominance_ratio:.3f}"
            )

    centre = np.mean(selected.robust_bounds_base_m, axis=0)
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = centre
    frame = BottleCapFrame(
        base_frame=base_frame,
        cap_frame=cap_frame,
        tcp_frame=tcp_frame,
        T_base_cap_tcp=transform,
        bottle_axis_base=(0.0, 0.0, 1.0),
        bottle_axis_tcp=(0.0, 0.0, 1.0),
    )
    depth_median = float(np.median(selected.depths_m))
    depth_mad = float(np.median(np.abs(selected.depths_m - depth_median)))
    return BottleCapLocalization(
        frame=frame,
        selected_mask=selected.mask.copy(),
        roi_xyxy_px=(x_min, y_min, x_max, y_max),
        bbox_xywh_px=selected.bbox_xywh_px,
        center_uv_px=(
            float(np.median(selected.columns)),
            float(np.median(selected.rows)),
        ),
        cap_center_base_m=centre,
        robust_bounds_base_m=selected.robust_bounds_base_m.copy(),
        point_count=len(selected.rows),
        component_pixel_count=selected.component_pixel_count,
        valid_depth_ratio=selected.valid_depth_ratio,
        eligible_component_count=len(candidates),
        dominance_ratio=dominance,
        estimated_cap_diameter_m=selected.estimated_cap_diameter_m,
        depth_median_m=depth_median,
        depth_mad_m=depth_mad,
    )


def draw_bottle_cap_overlay(
    rgb: np.ndarray, localization: BottleCapLocalization
) -> np.ndarray:
    """Draw the selected pixels and explicit scene assumptions for review."""

    overlay = np.asarray(rgb).copy()
    if overlay.dtype != np.uint8 or overlay.ndim != 3 or overlay.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3) and dtype uint8")
    selected = localization.selected_mask.astype(bool)
    tint = np.zeros_like(overlay)
    tint[:, :, 1] = 255
    overlay[selected] = cv2.addWeighted(
        overlay[selected], 0.35, tint[selected], 0.65, 0.0
    )
    contours, _ = cv2.findContours(
        selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    x_min, y_min, x_max, y_max = localization.roi_xyxy_px
    cv2.rectangle(overlay, (x_min, y_min), (x_max - 1, y_max - 1), (255, 255, 0), 2)
    u, v = np.rint(localization.center_uv_px).astype(int)
    cv2.drawMarker(overlay, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    x, y, _, _ = localization.bbox_xywh_px
    cv2.putText(
        overlay,
        "CURRENT SCENE: upright bottle, axis=base +Z",
        (max(4, x_min), max(20, y_min - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"black cap: {localization.point_count} depth pts, d~{localization.estimated_cap_diameter_m * 1000:.1f} mm",
        (max(4, x), min(overlay.shape[0] - 8, y + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


__all__ = [
    "LOCALIZATION_SCHEMA",
    "BottleCapLocalization",
    "CurrentBottleCapConfig",
    "draw_bottle_cap_overlay",
    "localize_current_bottle_cap",
]
