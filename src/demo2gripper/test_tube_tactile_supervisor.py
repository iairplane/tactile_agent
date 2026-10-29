"""Image evidence for test-tube tactile supervision, without robot control.

The GF220 recordings made by :mod:`scripts.record_front_rgbd_tactile` are raw
UVC images.  They are not ViTai SDK ``WARPED_IMG``/``DIFF_IMG`` products and
do not contain ``FORCE6D``.  This module therefore reports measurements in raw
image pixels only.  A caller must not interpret an image angle as the tube
angle or promote a negative-only noise fit to a physical runtime gate.

The small evaluator is useful for two purposes:

* qualifying stream freshness and duplicate-frame handling; and
* running contact-area, contact-orientation and marker-shear features in
  shadow mode until a robot-contact calibration supplies positive examples.

It deliberately owns no cameras, SDK handles, ROS nodes, grippers, or robot
command path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class RawTactileNoiseModel:
    """Negative-only image model fitted from unloaded raw UVC frames."""

    reference_bgr: np.ndarray
    roi_mask: np.ndarray
    residual_threshold: float
    min_contact_area_px: int
    marker_points_uv: np.ndarray
    marker_shear_p90_threshold_px: float
    fit_frame_count: int

    def __post_init__(self) -> None:
        reference = np.asarray(self.reference_bgr)
        if reference.ndim != 3 or reference.shape[-1] != 3:
            raise ValueError("reference_bgr must have shape HxWx3")
        if reference.dtype != np.uint8:
            raise ValueError("reference_bgr must be uint8")
        roi = np.asarray(self.roi_mask, dtype=bool)
        if roi.shape != reference.shape[:2] or not roi.any():
            raise ValueError("roi_mask must be a non-empty mask matching reference_bgr")
        points = np.asarray(self.marker_points_uv, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (2,):
            raise ValueError("marker_points_uv must have shape Nx2")
        if not np.isfinite(points).all():
            raise ValueError("marker_points_uv must be finite")
        if not math.isfinite(self.residual_threshold) or self.residual_threshold <= 0.0:
            raise ValueError("residual_threshold must be positive and finite")
        if self.min_contact_area_px <= 0:
            raise ValueError("min_contact_area_px must be positive")
        if (
            not math.isfinite(self.marker_shear_p90_threshold_px)
            or self.marker_shear_p90_threshold_px <= 0.0
        ):
            raise ValueError("marker_shear_p90_threshold_px must be positive and finite")
        if self.fit_frame_count < 2:
            raise ValueError("fit_frame_count must be at least two")
        reference = np.array(reference, copy=True)
        roi = np.array(roi, copy=True)
        points = np.array(points, copy=True)
        reference.setflags(write=False)
        roi.setflags(write=False)
        points.setflags(write=False)
        object.__setattr__(self, "reference_bgr", reference)
        object.__setattr__(self, "roi_mask", roi)
        object.__setattr__(self, "marker_points_uv", points)


@dataclass(frozen=True)
class ContactGeometry:
    """Largest raw-image residual patch in one tactile frame."""

    area_px: int
    area_fraction: float
    center_uv_px: tuple[float, float] | None
    orientation_deg: float | None
    elongation: float | None
    orientation_concentration: float | None
    contact_candidate: bool
    orientation_valid: bool


@dataclass(frozen=True)
class MarkerMotion:
    """Global image motion and non-rigid marker displacement in pixels."""

    feature_count: int
    tracked_count: int
    global_flow_uv_px: tuple[float, float] | None
    shear_median_px: float | None
    shear_p90_px: float | None
    valid: bool


@dataclass(frozen=True)
class RawTactileFrameEvidence:
    residual_p99: float
    residual_max: float
    geometry: ContactGeometry
    marker_motion: MarkerMotion
    slip_shadow_candidate: bool


def default_pad_roi(image_shape: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
    """Return a conservative central ROI for the GF220's directly viewed pad.

    The raw camera also sees a reflected side wall.  Keeping the ROI on the
    directly viewed central pad avoids treating that reflection as a second
    contact.  This normalized ROI is only a shadow-analysis default; a warped
    pad calibration should replace it for runtime use.
    """

    if len(image_shape) not in (2, 3):
        raise ValueError("image_shape must be HxW or HxWxC")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height < 16 or width < 16:
        raise ValueError("tactile image is too small")
    y0, y1 = round(0.21 * height), round(0.81 * height)
    x0, x1 = round(0.16 * width), round(0.72 * width)
    roi = np.zeros((height, width), dtype=bool)
    roi[y0:y1, x0:x1] = True
    return roi


def compensated_residual(
    frame_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    roi_mask: np.ndarray,
) -> np.ndarray:
    """Return colour residual magnitude after removing global channel offsets."""

    frame = _uint8_image(frame_bgr, "frame_bgr").astype(np.float32)
    reference = _uint8_image(reference_bgr, "reference_bgr").astype(np.float32)
    if frame.shape != reference.shape:
        raise ValueError("frame_bgr and reference_bgr must have the same shape")
    roi = _roi(roi_mask, frame.shape[:2])
    delta = frame - reference
    channel_offset = np.median(delta[roi], axis=0)
    return np.linalg.norm(delta - channel_offset, axis=2) / math.sqrt(3.0)


def residual_contact_geometry(
    residual: np.ndarray,
    roi_mask: np.ndarray,
    *,
    residual_threshold: float,
    min_contact_area_px: int,
    min_orientation_elongation: float = 1.6,
    min_orientation_concentration: float = 0.35,
) -> ContactGeometry:
    """Measure the largest spatially coherent residual patch.

    The reported angle is modulo 180 degrees in raw image coordinates, where
    +u points right and +v points down.  It is never a tube/world angle.
    """

    values = np.asarray(residual, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("residual must be a finite 2D array")
    roi = _roi(roi_mask, values.shape)
    if not math.isfinite(residual_threshold) or residual_threshold <= 0.0:
        raise ValueError("residual_threshold must be positive and finite")
    if min_contact_area_px <= 0:
        raise ValueError("min_contact_area_px must be positive")
    if min_orientation_elongation <= 1.0:
        raise ValueError("min_orientation_elongation must exceed one")
    if not 0.0 < min_orientation_concentration < 1.0:
        raise ValueError("min_orientation_concentration must lie in (0, 1)")

    mask = values >= residual_threshold
    mask &= roi
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndimage.label(mask)
    if count == 0:
        return ContactGeometry(0, 0.0, None, None, None, None, False, False)
    areas = np.bincount(labels.ravel())[1:]
    component_label = int(np.argmax(areas) + 1)
    component = labels == component_label
    area = int(component.sum())
    rows, cols = np.nonzero(component)
    center = (float(np.mean(cols)), float(np.mean(rows)))
    contact = area >= min_contact_area_px
    if area < 3:
        return ContactGeometry(
            area,
            area / float(roi.sum()),
            center,
            None,
            None,
            None,
            contact,
            False,
        )

    coordinates = np.column_stack((cols - center[0], rows - center[1]))
    covariance = coordinates.T @ coordinates / float(area)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    small, large = (float(eigenvalues[0]), float(eigenvalues[1]))
    elongation = math.sqrt(max(large, 0.0) / max(small, 1e-12))
    concentration = (large - small) / max(large + small, 1e-12)
    major = eigenvectors[:, 1]
    angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
    angle = ((angle + 90.0) % 180.0) - 90.0
    orientation_valid = bool(
        contact
        and elongation >= min_orientation_elongation
        and concentration >= min_orientation_concentration
    )
    return ContactGeometry(
        area_px=area,
        area_fraction=area / float(roi.sum()),
        center_uv_px=center,
        orientation_deg=angle if orientation_valid else None,
        elongation=elongation,
        orientation_concentration=concentration,
        contact_candidate=contact,
        orientation_valid=orientation_valid,
    )


def summarize_marker_displacements(
    reference_points_uv: np.ndarray,
    observed_points_uv: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    min_tracked_points: int = 12,
) -> MarkerMotion:
    """Separate common camera motion from local marker shear."""

    reference = _points(reference_points_uv, "reference_points_uv")
    observed = _points(observed_points_uv, "observed_points_uv")
    if reference.shape != observed.shape:
        raise ValueError("reference and observed points must have the same shape")
    if min_tracked_points < 3:
        raise ValueError("min_tracked_points must be at least three")
    keep = np.ones(len(reference), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if keep.shape != (len(reference),):
        raise ValueError("valid must have shape N")
    keep &= np.isfinite(reference).all(axis=1) & np.isfinite(observed).all(axis=1)
    tracked = int(keep.sum())
    if tracked < min_tracked_points:
        return MarkerMotion(len(reference), tracked, None, None, None, False)
    displacement = observed[keep] - reference[keep]
    global_flow = np.median(displacement, axis=0)
    shear = np.linalg.norm(displacement - global_flow, axis=1)
    return MarkerMotion(
        feature_count=len(reference),
        tracked_count=tracked,
        global_flow_uv_px=(float(global_flow[0]), float(global_flow[1])),
        shear_median_px=float(np.median(shear)),
        shear_p90_px=float(np.quantile(shear, 0.90)),
        valid=True,
    )


def detect_marker_points(
    reference_bgr: np.ndarray,
    roi_mask: np.ndarray,
    *,
    max_points: int = 120,
) -> np.ndarray:
    """Detect raw-image marker corners with OpenCV when available."""

    if max_points < 12:
        raise ValueError("max_points must be at least twelve")
    cv2 = _cv2()
    reference = _uint8_image(reference_bgr, "reference_bgr")
    roi = _roi(roi_mask, reference.shape[:2])
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=8,
        mask=(roi.astype(np.uint8) * 255),
        blockSize=5,
    )
    if points is None:
        return np.empty((0, 2), dtype=float)
    return points.reshape(-1, 2).astype(float)


def track_marker_motion(
    frame_bgr: np.ndarray,
    model: RawTactileNoiseModel,
    *,
    min_tracked_points: int = 12,
) -> MarkerMotion:
    """Track reference marker corners into one raw UVC frame."""

    if len(model.marker_points_uv) == 0:
        return MarkerMotion(0, 0, None, None, None, False)
    cv2 = _cv2()
    frame = _uint8_image(frame_bgr, "frame_bgr")
    if frame.shape != model.reference_bgr.shape:
        raise ValueError("frame_bgr shape does not match the fitted model")
    reference_gray = cv2.cvtColor(model.reference_bgr, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    reference_points = model.marker_points_uv.astype(np.float32).reshape(-1, 1, 2)
    observed, status, _ = cv2.calcOpticalFlowPyrLK(
        reference_gray,
        frame_gray,
        reference_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if observed is None or status is None:
        return MarkerMotion(len(reference_points), 0, None, None, None, False)
    return summarize_marker_displacements(
        model.marker_points_uv,
        observed.reshape(-1, 2),
        status.reshape(-1).astype(bool),
        min_tracked_points=min_tracked_points,
    )


def fit_unloaded_raw_model(
    frames_bgr: np.ndarray,
    *,
    roi_mask: np.ndarray | None = None,
    residual_quantile: float = 0.999,
    residual_floor: float = 12.0,
    min_contact_area_floor_px: int = 64,
    marker_shear_floor_px: float = 0.5,
) -> RawTactileNoiseModel:
    """Fit an unloaded noise ceiling, not a physical contact classifier."""

    frames = np.asarray(frames_bgr)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("frames_bgr must be uint8 with shape NxHxWx3")
    if len(frames) < 2:
        raise ValueError("at least two unloaded frames are required")
    if not 0.9 <= residual_quantile < 1.0:
        raise ValueError("residual_quantile must lie in [0.9, 1)")
    if residual_floor <= 0.0 or min_contact_area_floor_px <= 0:
        raise ValueError("noise floors must be positive")
    reference = np.median(frames, axis=0).astype(np.uint8)
    roi = (
        default_pad_roi(reference.shape)
        if roi_mask is None
        else _roi(roi_mask, reference.shape[:2])
    )
    sampled_residuals: list[np.ndarray] = []
    for frame in frames:
        residual = compensated_residual(frame, reference, roi)
        sampled_residuals.append(residual[roi][::4])
    residual_threshold = max(
        float(residual_floor),
        float(np.quantile(np.concatenate(sampled_residuals), residual_quantile)),
    )

    maximum_noise_component = 0
    for frame in frames:
        geometry = residual_contact_geometry(
            compensated_residual(frame, reference, roi),
            roi,
            residual_threshold=residual_threshold,
            min_contact_area_px=1,
        )
        maximum_noise_component = max(maximum_noise_component, geometry.area_px)
    min_contact_area = max(
        int(min_contact_area_floor_px),
        4 * (int(maximum_noise_component) + 1),
    )
    marker_points = detect_marker_points(reference, roi)

    provisional = RawTactileNoiseModel(
        reference_bgr=reference,
        roi_mask=roi,
        residual_threshold=residual_threshold,
        min_contact_area_px=min_contact_area,
        marker_points_uv=marker_points,
        marker_shear_p90_threshold_px=max(float(marker_shear_floor_px), 1e-6),
        fit_frame_count=len(frames),
    )
    baseline_shear = []
    for frame in frames:
        motion = track_marker_motion(frame, provisional)
        if motion.valid and motion.shear_p90_px is not None:
            baseline_shear.append(motion.shear_p90_px)
    shear_threshold = float(marker_shear_floor_px)
    if baseline_shear:
        shear_threshold = max(
            shear_threshold,
            3.0 * float(np.quantile(baseline_shear, residual_quantile)),
        )
    return RawTactileNoiseModel(
        reference_bgr=reference,
        roi_mask=roi,
        residual_threshold=residual_threshold,
        min_contact_area_px=min_contact_area,
        marker_points_uv=marker_points,
        marker_shear_p90_threshold_px=shear_threshold,
        fit_frame_count=len(frames),
    )


def evaluate_raw_tactile_frame(
    frame_bgr: np.ndarray,
    model: RawTactileNoiseModel,
) -> RawTactileFrameEvidence:
    """Extract one frame's shadow-only contact and marker-motion evidence."""

    residual = compensated_residual(frame_bgr, model.reference_bgr, model.roi_mask)
    geometry = residual_contact_geometry(
        residual,
        model.roi_mask,
        residual_threshold=model.residual_threshold,
        min_contact_area_px=model.min_contact_area_px,
    )
    motion = track_marker_motion(frame_bgr, model)
    slip_candidate = bool(
        geometry.contact_candidate
        and motion.valid
        and motion.shear_p90_px is not None
        and motion.shear_p90_px >= model.marker_shear_p90_threshold_px
    )
    roi_values = residual[model.roi_mask]
    return RawTactileFrameEvidence(
        residual_p99=float(np.quantile(roi_values, 0.99)),
        residual_max=float(np.max(roi_values)),
        geometry=geometry,
        marker_motion=motion,
        slip_shadow_candidate=slip_candidate,
    )


def _uint8_image(value: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 with shape HxWx3")
    return image


def _roi(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    roi = np.asarray(value, dtype=bool)
    if roi.shape != shape or not roi.any():
        raise ValueError("roi_mask must be a non-empty mask matching the image")
    return roi


def _points(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError(f"{name} must have shape Nx2")
    return points


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on visualization extra
        raise RuntimeError(
            "OpenCV is required for raw tactile marker tracking; "
            "install demo2gripper[visualization]"
        ) from exc
    return cv2
