"""Fail-closed temporal qualification for a source test tube.

F2G-Pose supplies a single-frame pose proposal.  This module supplies the
pieces that a robot control loop needs around that proposal: a persistent mask
identity, foreground-depth rejection, cylindrical pose canonicalisation,
temporal filtering, measured window dispersion, and explicit HOLD results.

The source tube is assumed to remain still until contact.  A qualified reviewed
RGB/mask/endpoint seed is registered into the first live frame and then
propagated with optical flow and ECC; SAM is therefore an initializer or an
explicit reseed source, not a claimed per-control-frame tracker.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from .f2g_pose_service import (
    ForegroundDepthGateConfig,
    PoseRequest,
    evaluate_foreground_depth,
)
from .franka import validate_rigid_transform
from .pose_observation import PoseCapture, PoseObservation
from .test_tube_closed_loop import ObjectPoseEstimate, PoseBackendDescriptor

SOURCE_SEED_QUALIFICATION_SCHEMA = "demo2gripper.test_tube_source_seed_qualification.v1"


class TubePoseTrackingError(RuntimeError):
    """A frame cannot safely update the live tube pose."""


class PoseProvider(Protocol):
    """Subset of :class:`RemotePoseProvider` used by this module."""

    def estimate(
        self,
        *,
        frame_id: str,
        capture_timestamp_s: float,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray,
        mask_source: str,
        mask_source_score: float | None = None,
    ) -> PoseObservation: ...


def _fraction(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _binary_mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if result.dtype == np.bool_:
        mask = result.copy()
    elif np.issubdtype(result.dtype, np.integer) and np.isin(result, (0, 1, 255)).all():
        mask = result > 0
    else:
        raise ValueError(f"{name} must be a binary mask")
    if not mask.any():
        raise ValueError(f"{name} must not be empty")
    return np.ascontiguousarray(mask)


def _rgb(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 3 or result.shape[2] != 3 or result.dtype != np.uint8:
        raise ValueError("rgb must be HxWx3 uint8 in RGB order")
    return np.ascontiguousarray(result)


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    return 0.0 if union == 0 else float(np.count_nonzero(first & second) / union)


def _warp_mask(mask: np.ndarray, affine: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - visualization extra is expected live
        raise TubePoseTrackingError("OpenCV is required for source-mask tracking") from error
    height, width = shape
    warped = cv2.warpAffine(
        mask.astype(np.uint8),
        np.asarray(affine, dtype=np.float32),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 0


def _gray(rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - visualization extra is expected live
        raise TubePoseTrackingError("OpenCV is required for source-mask tracking") from error
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


@dataclass(frozen=True)
class StaticSourceMaskTrackerConfig:
    """Quality limits for propagating one reviewed, pre-contact tube mask."""

    max_features: int = 200
    min_tracked_features: int = 12
    feature_quality_level: float = 0.01
    feature_min_distance_px: float = 4.0
    max_forward_backward_error_px: float = 1.5
    ransac_reprojection_threshold_px: float = 2.0
    min_flow_inlier_fraction: float = 0.60
    min_ecc_correlation: float = 0.65
    min_internal_iou: float = 0.65
    min_external_update_iou: float = 0.55
    max_area_ratio_change: float = 0.30
    max_border_loss_fraction: float = 0.18
    max_occluder_overlap_fraction: float = 0.20
    max_tracking_gap_s: float = 0.30
    ecc_iterations: int = 40
    ecc_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        _positive_int(self.max_features, "max_features")
        _positive_int(self.min_tracked_features, "min_tracked_features")
        if self.min_tracked_features > self.max_features:
            raise ValueError("min_tracked_features cannot exceed max_features")
        _positive_int(self.ecc_iterations, "ecc_iterations")
        for name in (
            "feature_quality_level",
            "min_flow_inlier_fraction",
            "min_ecc_correlation",
            "min_internal_iou",
            "min_external_update_iou",
            "max_area_ratio_change",
            "max_border_loss_fraction",
            "max_occluder_overlap_fraction",
        ):
            _fraction(getattr(self, name), name)
        for name in (
            "feature_min_distance_px",
            "max_forward_backward_error_px",
            "ransac_reprojection_threshold_px",
            "max_tracking_gap_s",
            "ecc_epsilon",
        ):
            _positive(getattr(self, name), name)


@dataclass(frozen=True)
class StaticMaskTrackResult:
    """One accepted mask propagation or an explicit HOLD result."""

    decision: str
    tracking_id: str
    mask_id: str | None
    mask: np.ndarray | None
    consistency_iou: float | None
    occluded: bool
    reasons: tuple[str, ...]
    seed_to_current: np.ndarray | None
    tracked_feature_count: int
    flow_inlier_fraction: float | None
    ecc_correlation: float | None
    score_provenance: str | None

    @property
    def accepted(self) -> bool:
        return self.decision in {"SEEDED", "TRACKED"} and self.mask is not None


class StaticSourceMaskTracker:
    """Track a reviewed source-tube mask without rerunning SAM every frame.

    ``consistency_iou`` is an observed overlap: either flow-vs-ECC agreement or
    flow-vs-external-mask overlap.  It is not model confidence and is not IoU
    against hidden ground truth.
    """

    def __init__(self, config: StaticSourceMaskTrackerConfig | None = None) -> None:
        self.config = config or StaticSourceMaskTrackerConfig()
        self._tracking_id: str | None = None
        self._sequence = 0
        self._last_timestamp_s: float | None = None
        self._last_rgb: np.ndarray | None = None
        self._last_mask: np.ndarray | None = None
        self._seed_area: int | None = None
        self._seed_to_last = np.eye(3, dtype=np.float64)

    @property
    def seeded(self) -> bool:
        return self._last_mask is not None

    @property
    def tracking_id(self) -> str | None:
        return self._tracking_id

    def reset(self) -> None:
        self._tracking_id = None
        self._sequence = 0
        self._last_timestamp_s = None
        self._last_rgb = None
        self._last_mask = None
        self._seed_area = None
        self._seed_to_last = np.eye(3, dtype=np.float64)

    def seed(
        self,
        *,
        rgb: np.ndarray,
        reviewed_mask: np.ndarray,
        timestamp_s: float,
        tracking_id: str | None = None,
    ) -> StaticMaskTrackResult:
        image = _rgb(rgb)
        mask = _binary_mask(reviewed_mask, image.shape[:2], "reviewed_mask")
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if tracking_id is None:
            digest = hashlib.sha256(mask.view(np.uint8).tobytes()).hexdigest()[:12]
            tracking_id = f"tube-{digest}"
        if not isinstance(tracking_id, str) or not tracking_id.strip():
            raise ValueError("tracking_id must be a non-empty string")
        self._tracking_id = tracking_id
        self._sequence = 1
        self._last_timestamp_s = timestamp
        self._last_rgb = image.copy()
        self._last_mask = mask.copy()
        self._seed_area = int(mask.sum())
        self._seed_to_last = np.eye(3, dtype=np.float64)
        return StaticMaskTrackResult(
            decision="SEEDED",
            tracking_id=tracking_id,
            mask_id=f"reviewed-seed:{tracking_id}:1",
            mask=mask,
            consistency_iou=1.0,
            occluded=False,
            reasons=(),
            seed_to_current=self._seed_to_last.copy(),
            tracked_feature_count=0,
            flow_inlier_fraction=None,
            ecc_correlation=None,
            score_provenance="reviewed_seed_identity",
        )

    def update(
        self,
        *,
        rgb: np.ndarray,
        timestamp_s: float,
        external_mask: np.ndarray | None = None,
        occluder_mask: np.ndarray | None = None,
    ) -> StaticMaskTrackResult:
        if (
            self._tracking_id is None
            or self._last_timestamp_s is None
            or self._last_rgb is None
            or self._last_mask is None
            or self._seed_area is None
        ):
            raise TubePoseTrackingError("a reviewed seed mask is required before update")
        image = _rgb(rgb)
        if image.shape != self._last_rgb.shape:
            return self._hold("frame_shape_changed")
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp) or timestamp <= self._last_timestamp_s:
            raise ValueError("mask tracker timestamps must be finite and strictly increasing")
        if timestamp - self._last_timestamp_s > self.config.max_tracking_gap_s:
            return self._hold("tracking_gap_exceeded")

        previous_gray = _gray(self._last_rgb)
        current_gray = _gray(image)
        try:
            import cv2
        except ImportError as error:  # pragma: no cover
            raise TubePoseTrackingError("OpenCV is required for source-mask tracking") from error

        points = cv2.goodFeaturesToTrack(
            previous_gray,
            mask=self._last_mask.astype(np.uint8) * 255,
            maxCorners=self.config.max_features,
            qualityLevel=self.config.feature_quality_level,
            minDistance=self.config.feature_min_distance_px,
        )
        if points is None or len(points) < self.config.min_tracked_features:
            return self._hold("insufficient_mask_features")
        current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points, None
        )
        if current_points is None or forward_status is None:
            return self._hold("forward_optical_flow_failed")
        back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, previous_gray, current_points, None
        )
        if back_points is None or backward_status is None:
            return self._hold("backward_optical_flow_failed")
        forward_ok = forward_status.reshape(-1) > 0
        backward_ok = backward_status.reshape(-1) > 0
        fb_error = np.linalg.norm(back_points.reshape(-1, 2) - points.reshape(-1, 2), axis=1)
        good = forward_ok & backward_ok & (fb_error <= self.config.max_forward_backward_error_px)
        if int(good.sum()) < self.config.min_tracked_features:
            return self._hold("insufficient_forward_backward_consistent_features")
        previous_good = points.reshape(-1, 2)[good]
        current_good = current_points.reshape(-1, 2)[good]
        affine, ransac = cv2.estimateAffinePartial2D(
            previous_good,
            current_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_reprojection_threshold_px,
        )
        if affine is None or ransac is None:
            return self._hold("flow_affine_estimation_failed")
        inlier_fraction = float(np.mean(ransac.reshape(-1) > 0))
        if inlier_fraction < self.config.min_flow_inlier_fraction:
            return self._hold(
                "flow_affine_inlier_fraction_too_low",
                tracked_feature_count=int(good.sum()),
                flow_inlier_fraction=inlier_fraction,
            )
        flow_mask = _warp_mask(self._last_mask, affine, image.shape[:2])

        ecc_warp = np.asarray(affine, dtype=np.float32).copy()
        try:
            correlation, ecc_warp = cv2.findTransformECC(
                current_gray,
                previous_gray,
                ecc_warp,
                cv2.MOTION_AFFINE,
                (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    self.config.ecc_iterations,
                    self.config.ecc_epsilon,
                ),
                inputMask=self._last_mask.astype(np.uint8) * 255,
                gaussFiltSize=5,
            )
            ecc_mask_u8 = cv2.warpAffine(
                self._last_mask.astype(np.uint8),
                ecc_warp,
                (image.shape[1], image.shape[0]),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            ecc_mask = ecc_mask_u8 > 0
        except cv2.error:
            return self._hold(
                "ecc_registration_failed",
                tracked_feature_count=int(good.sum()),
                flow_inlier_fraction=inlier_fraction,
            )
        correlation = float(correlation)
        if not math.isfinite(correlation) or correlation < self.config.min_ecc_correlation:
            return self._hold(
                "ecc_correlation_too_low",
                tracked_feature_count=int(good.sum()),
                flow_inlier_fraction=inlier_fraction,
                ecc_correlation=correlation,
            )

        candidate = ecc_mask
        consistency = _mask_iou(flow_mask, ecc_mask)
        provenance = "optical_flow_vs_ecc_iou"
        if external_mask is not None:
            external = _binary_mask(external_mask, image.shape[:2], "external_mask")
            consistency = _mask_iou(flow_mask, external)
            provenance = "optical_flow_vs_external_mask_iou"
            if consistency < self.config.min_external_update_iou:
                return self._hold(
                    "external_mask_disagrees_with_flow",
                    consistency_iou=consistency,
                    tracked_feature_count=int(good.sum()),
                    flow_inlier_fraction=inlier_fraction,
                    ecc_correlation=correlation,
                    score_provenance=provenance,
                )
            candidate = external
        elif consistency < self.config.min_internal_iou:
            return self._hold(
                "flow_ecc_mask_iou_too_low",
                consistency_iou=consistency,
                tracked_feature_count=int(good.sum()),
                flow_inlier_fraction=inlier_fraction,
                ecc_correlation=correlation,
                score_provenance=provenance,
            )

        area_ratio = float(candidate.sum() / self._seed_area)
        if abs(area_ratio - 1.0) > self.config.max_area_ratio_change:
            return self._hold(
                "tracked_mask_area_changed_too_much",
                consistency_iou=consistency,
                tracked_feature_count=int(good.sum()),
                flow_inlier_fraction=inlier_fraction,
                ecc_correlation=correlation,
                score_provenance=provenance,
            )
        full_warp = np.ones_like(self._last_mask, dtype=np.uint8)
        visible = cv2.warpAffine(
            full_warp,
            affine,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        border_loss = 1.0 - float(visible.sum() / full_warp.size)
        if border_loss > self.config.max_border_loss_fraction:
            return self._hold("tracked_region_left_image")

        occluded = False
        if occluder_mask is not None:
            occluder = _binary_mask_allow_empty(occluder_mask, image.shape[:2], "occluder_mask")
            overlap = float(np.count_nonzero(candidate & occluder) / candidate.sum())
            if overlap > self.config.max_occluder_overlap_fraction:
                occluded = True
                return self._hold(
                    "tube_mask_occluded",
                    occluded=True,
                    consistency_iou=consistency,
                    tracked_feature_count=int(good.sum()),
                    flow_inlier_fraction=inlier_fraction,
                    ecc_correlation=correlation,
                    score_provenance=provenance,
                )

        step = np.eye(3, dtype=np.float64)
        step[:2] = affine
        self._seed_to_last = step @ self._seed_to_last
        self._last_rgb = image.copy()
        self._last_mask = candidate.copy()
        self._last_timestamp_s = timestamp
        self._sequence += 1
        return StaticMaskTrackResult(
            decision="TRACKED",
            tracking_id=self._tracking_id,
            mask_id=f"static-source:{self._tracking_id}:{self._sequence}",
            mask=candidate,
            consistency_iou=consistency,
            occluded=occluded,
            reasons=(),
            seed_to_current=self._seed_to_last.copy(),
            tracked_feature_count=int(good.sum()),
            flow_inlier_fraction=inlier_fraction,
            ecc_correlation=correlation,
            score_provenance=provenance,
        )

    def _hold(
        self,
        reason: str,
        *,
        occluded: bool = False,
        consistency_iou: float | None = None,
        tracked_feature_count: int = 0,
        flow_inlier_fraction: float | None = None,
        ecc_correlation: float | None = None,
        score_provenance: str | None = None,
    ) -> StaticMaskTrackResult:
        assert self._tracking_id is not None
        return StaticMaskTrackResult(
            decision="HOLD",
            tracking_id=self._tracking_id,
            mask_id=None,
            mask=None,
            consistency_iou=consistency_iou,
            occluded=occluded,
            reasons=(reason,),
            seed_to_current=None,
            tracked_feature_count=tracked_feature_count,
            flow_inlier_fraction=flow_inlier_fraction,
            ecc_correlation=ecc_correlation,
            score_provenance=score_provenance,
        )


def _binary_mask_allow_empty(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if result.dtype == np.bool_:
        return np.ascontiguousarray(result)
    if np.issubdtype(result.dtype, np.integer) and np.isin(result, (0, 1, 255)).all():
        return np.ascontiguousarray(result > 0)
    raise ValueError(f"{name} must be a binary mask")


@dataclass(frozen=True)
class ReviewedTubeEnds:
    """Human-reviewed tube endpoints in the reviewed seed RGB frame."""

    open_uv: tuple[float, float]
    closed_uv: tuple[float, float]
    evidence_id: str
    axis_sign_source: str = "open_end_detector"

    def __post_init__(self) -> None:
        for name in ("open_uv", "closed_uv"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain two finite pixels")
        if np.linalg.norm(np.asarray(self.open_uv) - np.asarray(self.closed_uv)) < 3.0:
            raise ValueError("reviewed tube endpoints must be distinct")
        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip():
            raise ValueError("evidence_id must be a non-empty string")
        if self.axis_sign_source not in {
            "open_end_detector",
            "source_support_geometry",
            "source_support_anchor",
            "qualified_base_z",
        }:
            raise ValueError("unsupported reviewed endpoint axis_sign_source")

    def warped(self, seed_to_current: np.ndarray) -> ReviewedTubeEnds:
        transform = np.asarray(seed_to_current, dtype=float)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError("seed_to_current must be finite 3x3")

        def apply(point: tuple[float, float]) -> tuple[float, float]:
            homogeneous = transform @ np.array([point[0], point[1], 1.0])
            if abs(float(homogeneous[2])) < 1e-9:
                raise ValueError("endpoint warp maps a point to infinity")
            return (
                float(homogeneous[0] / homogeneous[2]),
                float(homogeneous[1] / homogeneous[2]),
            )

        return ReviewedTubeEnds(
            open_uv=apply(self.open_uv),
            closed_uv=apply(self.closed_uv),
            evidence_id=self.evidence_id,
            axis_sign_source=self.axis_sign_source,
        )


@dataclass(frozen=True)
class TubePoseQualifierConfig:
    """Temporal and geometric limits around single-frame F2G proposals."""

    window_size: int = 7
    min_window_frames: int = 3
    long_axis_index: int | None = None
    max_translation_jump_m: float = 0.030
    max_orientation_jump_deg: float = 25.0
    max_window_position_std_m: float = 0.012
    max_window_orientation_std_deg: float = 12.0
    max_endpoint_mean_error_px: float = 80.0
    min_endpoint_assignment_margin_px: float = 4.0
    max_projected_axis_error_deg: float = 18.0
    max_projected_center_error_px: float = 35.0
    max_foreground_axis_error_deg: float = 22.0
    max_foreground_center_line_error_m: float = 0.025
    min_foreground_line_eigenvalue_ratio: float = 2.0
    min_foreground_line_points: int = 80
    canonicalize_cylinder_from_scene_geometry: bool = False

    def __post_init__(self) -> None:
        _positive_int(self.window_size, "window_size")
        _positive_int(self.min_window_frames, "min_window_frames")
        if self.min_window_frames > self.window_size:
            raise ValueError("min_window_frames cannot exceed window_size")
        if self.long_axis_index is not None and self.long_axis_index not in {0, 1, 2}:
            raise ValueError("long_axis_index must be 0, 1, 2, or null")
        for name in (
            "max_translation_jump_m",
            "max_orientation_jump_deg",
            "max_window_position_std_m",
            "max_window_orientation_std_deg",
            "max_endpoint_mean_error_px",
            "min_endpoint_assignment_margin_px",
            "max_projected_axis_error_deg",
            "max_projected_center_error_px",
            "max_foreground_axis_error_deg",
            "max_foreground_center_line_error_m",
            "min_foreground_line_eigenvalue_ratio",
        ):
            _positive(getattr(self, name), name)
        _positive_int(self.min_foreground_line_points, "min_foreground_line_points")
        if not isinstance(self.canonicalize_cylinder_from_scene_geometry, bool):
            raise TypeError("canonicalize_cylinder_from_scene_geometry must be boolean")


@dataclass(frozen=True)
class QualifiedTubePose:
    T_camera_tube: np.ndarray
    position_std_m: float
    orientation_std_deg: float
    axis_sign_source: str
    cylindrical_roll_canonicalized: bool
    sample_count: int
    size_m: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class TubePoseQualificationResult:
    decision: str
    pose: QualifiedTubePose | None
    reasons: tuple[str, ...]
    provisional_T_camera_tube: np.ndarray | None = None


class TubePoseQualifier:
    """Canonicalize and temporally qualify single-frame cylindrical poses."""

    def __init__(
        self,
        reviewed_ends: ReviewedTubeEnds,
        config: TubePoseQualifierConfig | None = None,
    ) -> None:
        self.reviewed_ends = reviewed_ends
        self.config = config or TubePoseQualifierConfig()
        self._poses: deque[np.ndarray] = deque(maxlen=self.config.window_size)
        self._last_axis: np.ndarray | None = None
        self._last_x: np.ndarray | None = None
        self._long_axis_index: int | None = self.config.long_axis_index
        self._axis_sign_resolved = False

    def reset(self) -> None:
        self._poses.clear()
        self._last_axis = None
        self._last_x = None
        self._long_axis_index = self.config.long_axis_index
        self._axis_sign_resolved = False

    def update(
        self,
        *,
        proposal: PoseObservation,
        K: np.ndarray,
        current_reviewed_ends: ReviewedTubeEnds | None = None,
        foreground_depth_m: np.ndarray | None = None,
    ) -> TubePoseQualificationResult:
        transform, size = proposal.require_pose()
        raw = validate_rigid_transform(transform, name="F2G T_camera_object")
        size_array = np.asarray(size, dtype=float)
        if size_array.shape != (3,) or not np.isfinite(size_array).all():
            return TubePoseQualificationResult("HOLD", None, ("invalid_f2g_size",))
        if self._long_axis_index is None:
            order = np.argsort(size_array)
            if (
                not self.config.canonicalize_cylinder_from_scene_geometry
                and size_array[order[-1]] < 1.5 * size_array[order[-2]]
            ):
                return TubePoseQualificationResult(
                    "HOLD", None, ("tube_long_axis_not_distinct_in_f2g_size",)
                )
            self._long_axis_index = int(order[-1])
        ends = current_reviewed_ends or self.reviewed_ends
        center = raw[:3, 3].copy()
        axis = raw[:3, self._long_axis_index].copy()
        axis /= np.linalg.norm(axis)
        qualified_size = size_array.copy()
        geometry_long_axis_index = self._long_axis_index
        sign_source = "temporal_continuity"
        scene_geometry = None
        if (
            foreground_depth_m is not None
            and self.config.canonicalize_cylinder_from_scene_geometry
        ):
            scene_geometry = _scene_cylinder_geometry(
                K=K,
                reviewed_ends=ends,
                foreground_depth_m=foreground_depth_m,
                config=self.config,
            )
            if scene_geometry is None:
                return TubePoseQualificationResult(
                    "HOLD", None, ("scene_cylinder_geometry_unresolved",)
                )
            center, axis, observed_length_m, observed_diameter_m = scene_geometry
            qualified_size = np.asarray(
                [observed_diameter_m, observed_diameter_m, observed_length_m], dtype=float
            )
            geometry_long_axis_index = 2
            sign_source = ends.axis_sign_source
            self._axis_sign_resolved = True
            geometry_transform = self._canonical_transform(center, axis)
        else:
            geometry_transform = raw
        if foreground_depth_m is not None:
            crosscheck_reasons = _source_geometry_crosscheck(
                T_camera_tube=geometry_transform,
                long_axis_index=geometry_long_axis_index,
                size_m=qualified_size,
                K=K,
                reviewed_ends=ends,
                foreground_depth_m=foreground_depth_m,
                config=self.config,
            )
            if crosscheck_reasons:
                return TubePoseQualificationResult(
                    "HOLD",
                    None,
                    crosscheck_reasons,
                    provisional_T_camera_tube=geometry_transform,
                )
        if scene_geometry is not None:
            if self._last_axis is not None and float(np.dot(axis, self._last_axis)) < 0.0:
                axis *= -1.0
        elif self._last_axis is None:
            sign = self._resolve_initial_axis_sign(
                center=center,
                unsigned_axis=axis,
                half_length_m=0.5 * float(size_array[self._long_axis_index]),
                K=K,
                ends=ends,
            )
            if sign is None:
                return TubePoseQualificationResult(
                    "HOLD", None, ("reviewed_endpoints_do_not_resolve_axis_sign",)
                )
            axis *= sign
            sign_source = ends.axis_sign_source
            self._axis_sign_resolved = True
        elif float(np.dot(axis, self._last_axis)) < 0.0:
            axis *= -1.0
        canonical = self._canonical_transform(center, axis)
        if self._poses:
            translation_jump = float(np.linalg.norm(canonical[:3, 3] - self._poses[-1][:3, 3]))
            rotation_jump = math.degrees(
                Rotation.from_matrix(canonical[:3, :3] @ self._poses[-1][:3, :3].T).magnitude()
            )
            reasons: list[str] = []
            if translation_jump > self.config.max_translation_jump_m:
                reasons.append("pose_translation_jump")
            if rotation_jump > self.config.max_orientation_jump_deg:
                reasons.append("pose_orientation_jump")
            if reasons:
                return TubePoseQualificationResult(
                    "HOLD", None, tuple(reasons), provisional_T_camera_tube=canonical
                )

        self._poses.append(canonical)
        self._last_axis = axis.copy()
        self._last_x = canonical[:3, 0].copy()
        if len(self._poses) < self.config.min_window_frames:
            return TubePoseQualificationResult(
                "WARMUP",
                None,
                ("temporal_window_not_full_enough",),
                provisional_T_camera_tube=canonical,
            )
        filtered, position_std, orientation_std = _mean_and_dispersion(tuple(self._poses))
        reasons = []
        if position_std > self.config.max_window_position_std_m:
            reasons.append("temporal_position_dispersion_too_high")
        if orientation_std > self.config.max_window_orientation_std_deg:
            reasons.append("temporal_orientation_dispersion_too_high")
        if reasons:
            return TubePoseQualificationResult(
                "HOLD", None, tuple(reasons), provisional_T_camera_tube=filtered
            )
        return TubePoseQualificationResult(
            "TRACKED",
            QualifiedTubePose(
                T_camera_tube=filtered,
                position_std_m=position_std,
                orientation_std_deg=orientation_std,
                axis_sign_source=sign_source,
                cylindrical_roll_canonicalized=True,
                sample_count=len(self._poses),
                size_m=tuple(float(value) for value in qualified_size),
            ),
            (),
        )

    def _resolve_initial_axis_sign(
        self,
        *,
        center: np.ndarray,
        unsigned_axis: np.ndarray,
        half_length_m: float,
        K: np.ndarray,
        ends: ReviewedTubeEnds,
    ) -> float | None:
        intrinsics = np.asarray(K, dtype=float)
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError("K must be finite 3x3")
        positive = center + unsigned_axis * half_length_m
        negative = center - unsigned_axis * half_length_m
        if positive[2] <= 0.0 or negative[2] <= 0.0:
            return None
        positive_uv = _project(positive, intrinsics)
        negative_uv = _project(negative, intrinsics)
        open_uv = np.asarray(ends.open_uv)
        closed_uv = np.asarray(ends.closed_uv)
        positive_open_cost = float(
            np.linalg.norm(positive_uv - open_uv) + np.linalg.norm(negative_uv - closed_uv)
        )
        negative_open_cost = float(
            np.linalg.norm(negative_uv - open_uv) + np.linalg.norm(positive_uv - closed_uv)
        )
        best = min(positive_open_cost, negative_open_cost)
        margin = abs(positive_open_cost - negative_open_cost)
        if (
            0.5 * best > self.config.max_endpoint_mean_error_px
            or margin < self.config.min_endpoint_assignment_margin_px
        ):
            return None
        return 1.0 if positive_open_cost < negative_open_cost else -1.0

    def _canonical_transform(self, center: np.ndarray, signed_axis: np.ndarray) -> np.ndarray:
        z_axis = signed_axis / np.linalg.norm(signed_axis)
        references = (
            self._last_x,
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        x_axis: np.ndarray | None = None
        for reference in references:
            if reference is None:
                continue
            projected = reference - z_axis * float(np.dot(reference, z_axis))
            norm = float(np.linalg.norm(projected))
            if norm > 1e-5:
                x_axis = projected / norm
                break
        if x_axis is None:  # pragma: no cover - three orthogonal fallbacks make this unreachable
            raise TubePoseTrackingError("cannot canonicalize cylinder roll")
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        result[:3, 3] = center
        return validate_rigid_transform(result, name="canonical T_camera_tube")


def _project(point_camera: np.ndarray, K: np.ndarray) -> np.ndarray:
    homogeneous = K @ point_camera
    return homogeneous[:2] / homogeneous[2]


def _scene_cylinder_geometry(
    *,
    K: np.ndarray,
    reviewed_ends: ReviewedTubeEnds,
    foreground_depth_m: np.ndarray,
    config: TubePoseQualifierConfig,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Recover a signed cylinder centre/axis from qualified RGB-D evidence.

    Transparent objects commonly return depth on only one visible strip.  The
    raw point-cloud centroid is therefore not the tube centre.  We fit the 3-D
    line to the selected foreground cluster, intersect it with the ray through
    the SAM endpoint midpoint, and use the SAM endpoints to determine signed
    extent.  This keeps the result metric without copying a model's ambiguous
    cylinder rotation or a biased partial-surface centroid.
    """

    intrinsics = np.asarray(K, dtype=float)
    depth = np.asarray(foreground_depth_m, dtype=float)
    if intrinsics.shape != (3, 3) or depth.ndim != 2:
        return None
    valid = np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.nonzero(valid)
    if len(rows) < config.min_foreground_line_points:
        return None
    z = depth[rows, columns]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    points = np.column_stack(((columns - cx) * z / fx, (rows - cy) * z / fy, z))
    centroid = points.mean(axis=0)
    covariance = (points - centroid).T @ (points - centroid)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    ordered = np.sort(np.maximum(eigenvalues, 0.0))
    if float(ordered[-1] / max(ordered[-2], 1e-12)) < config.min_foreground_line_eigenvalue_ratio:
        return None
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    axis /= np.linalg.norm(axis)
    axial_coordinates = (points - centroid) @ axis
    radial_vectors = points - centroid - np.outer(axial_coordinates, axis)
    diameter_m = 2.0 * float(np.quantile(np.linalg.norm(radial_vectors, axis=1), 0.95))
    if not math.isfinite(diameter_m) or diameter_m <= 0.0:
        return None

    open_uv = np.asarray(reviewed_ends.open_uv, dtype=float)
    closed_uv = np.asarray(reviewed_ends.closed_uv, dtype=float)
    midpoint_uv = 0.5 * (open_uv + closed_uv)
    midpoint_ray = np.linalg.solve(intrinsics, np.r_[midpoint_uv, 1.0])
    # Closest points between the observed 3-D line and the SAM midpoint ray.
    system = np.column_stack((axis, -midpoint_ray))
    line_t, ray_depth = np.linalg.lstsq(system, -centroid, rcond=None)[0]
    if not math.isfinite(float(ray_depth)) or ray_depth <= 0.0:
        return None
    line_midpoint = centroid + float(line_t) * axis
    center = float(ray_depth) * midpoint_ray
    if float(np.linalg.norm(center - line_midpoint)) > config.max_foreground_center_line_error_m:
        return None

    def line_parameter(uv: np.ndarray) -> tuple[float, float]:
        ray = np.linalg.solve(intrinsics, np.r_[uv, 1.0])
        matrix = np.column_stack((axis, -ray))
        parameter, ray_scale = np.linalg.lstsq(matrix, -center, rcond=None)[0]
        point_line = center + float(parameter) * axis
        point_ray = float(ray_scale) * ray
        return float(parameter), float(np.linalg.norm(point_line - point_ray))

    open_parameter, open_error = line_parameter(open_uv)
    closed_parameter, closed_error = line_parameter(closed_uv)
    if max(open_error, closed_error) > config.max_foreground_center_line_error_m:
        return None
    if open_parameter < closed_parameter:
        axis *= -1.0
        open_parameter *= -1.0
        closed_parameter *= -1.0
    length_m = float(open_parameter - closed_parameter)
    if not math.isfinite(length_m) or length_m <= 0.0:
        return None
    center += 0.5 * (open_parameter + closed_parameter) * axis
    return center, axis, length_m, diameter_m


def _source_geometry_crosscheck(
    *,
    T_camera_tube: np.ndarray,
    long_axis_index: int,
    size_m: np.ndarray,
    K: np.ndarray,
    reviewed_ends: ReviewedTubeEnds,
    foreground_depth_m: np.ndarray,
    config: TubePoseQualifierConfig,
) -> tuple[str, ...]:
    """Reject a stable F2G proposal that disagrees with measured scene geometry.

    This gate never edits the F2G pose. It compares the proposal with two
    independent observations: the warped reviewed image endpoints and a PCA
    line fitted to the selected foreground depth cluster.
    """

    transform = validate_rigid_transform(T_camera_tube, name="F2G T_camera_tube")
    intrinsics = np.asarray(K, dtype=float)
    depth = np.asarray(foreground_depth_m, dtype=float)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        return ("source_crosscheck_intrinsics_invalid",)
    if depth.ndim != 2:
        return ("source_crosscheck_foreground_depth_invalid",)
    center = transform[:3, 3]
    axis = transform[:3, long_axis_index]
    half_length = 0.5 * float(size_m[long_axis_index])
    endpoints_3d = (center - half_length * axis, center + half_length * axis)
    if center[2] <= 0.0 or min(item[2] for item in endpoints_3d) <= 0.0:
        return ("f2g_pose_projects_behind_camera",)
    predicted = np.stack([_project(item, intrinsics) for item in endpoints_3d])
    predicted_axis = predicted[1] - predicted[0]
    reviewed_axis = np.asarray(reviewed_ends.open_uv) - np.asarray(reviewed_ends.closed_uv)
    predicted_norm = float(np.linalg.norm(predicted_axis))
    reviewed_norm = float(np.linalg.norm(reviewed_axis))
    reasons: list[str] = []
    if predicted_norm <= 1e-6 or reviewed_norm <= 1e-6:
        reasons.append("projected_tube_axis_degenerate")
    else:
        cosine = float(
            np.clip(
                abs(np.dot(predicted_axis, reviewed_axis)) / (predicted_norm * reviewed_norm),
                0.0,
                1.0,
            )
        )
        error_deg = math.degrees(math.acos(cosine))
        if error_deg > config.max_projected_axis_error_deg:
            reasons.append("f2g_projected_axis_disagrees_with_reviewed_endpoints")
    predicted_center_uv = _project(center, intrinsics)
    reviewed_center_uv = 0.5 * (
        np.asarray(reviewed_ends.open_uv) + np.asarray(reviewed_ends.closed_uv)
    )
    if (
        float(np.linalg.norm(predicted_center_uv - reviewed_center_uv))
        > config.max_projected_center_error_px
    ):
        reasons.append("f2g_projected_center_disagrees_with_reviewed_tube")

    valid = np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.nonzero(valid)
    if len(rows) < config.min_foreground_line_points:
        reasons.append("foreground_line_has_insufficient_points")
        return tuple(reasons)
    z = depth[rows, columns]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    points = np.column_stack(((columns - cx) * z / fx, (rows - cy) * z / fy, z))
    centroid = points.mean(axis=0)
    covariance = (points - centroid).T @ (points - centroid)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    ordered_eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))
    line_ratio = float(ordered_eigenvalues[-1] / max(ordered_eigenvalues[-2], 1e-12))
    if line_ratio < config.min_foreground_line_eigenvalue_ratio:
        reasons.append("foreground_depth_cluster_is_not_line_like")
    line_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    axis_cosine = float(np.clip(abs(np.dot(axis, line_axis)), 0.0, 1.0))
    if math.degrees(math.acos(axis_cosine)) > config.max_foreground_axis_error_deg:
        reasons.append("f2g_axis_disagrees_with_foreground_depth_line")
    center_delta = center - centroid
    center_line_error = float(
        np.linalg.norm(center_delta - np.dot(center_delta, line_axis) * line_axis)
    )
    if center_line_error > config.max_foreground_center_line_error_m:
        reasons.append("f2g_center_disagrees_with_foreground_depth_line")
    return tuple(reasons)


def _mean_and_dispersion(
    transforms: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, float, float]:
    positions = np.stack([item[:3, 3] for item in transforms])
    mean_position = positions.mean(axis=0)
    rotations = Rotation.from_matrix(np.stack([item[:3, :3] for item in transforms]))
    mean_rotation = rotations.mean()
    position_std = float(np.sqrt(np.mean(np.sum(np.square(positions - mean_position), axis=1))))
    residual = mean_rotation.inv() * rotations
    orientation_std = float(math.degrees(np.sqrt(np.mean(np.square(residual.magnitude())))))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = mean_rotation.as_matrix()
    result[:3, 3] = mean_position
    return result, position_std, orientation_std


@dataclass(frozen=True)
class RealTimeTubePoseTrackerConfig:
    max_capture_age_s: float = 0.25
    future_timestamp_tolerance_s: float = 0.02
    max_provider_latency_s: float = 0.25
    camera_position_std_m: float = 0.0
    camera_orientation_std_deg: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "max_capture_age_s",
            "future_timestamp_tolerance_s",
            "max_provider_latency_s",
        ):
            _positive(getattr(self, name), name)
        for name in ("camera_position_std_m", "camera_orientation_std_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class TubePoseTrackResult:
    decision: str
    estimate: ObjectPoseEstimate | None
    mask_track: StaticMaskTrackResult
    reasons: tuple[str, ...]
    provider_latency_s: float | None


class RealTimeTubePoseTracker:
    """Direct adapter from aligned RGB-D frames to ``ObjectPoseEstimate``.

    The class owns no camera and issues no robot command.  An execution loop can
    call :meth:`update` on each camera frame; any failed gate yields ``HOLD`` and
    no stale pose estimate.
    """

    def __init__(
        self,
        *,
        provider: PoseProvider,
        backend: PoseBackendDescriptor,
        T_base_camera: np.ndarray,
        reviewed_initial_mask: np.ndarray,
        reviewed_ends: ReviewedTubeEnds,
        mask_config: StaticSourceMaskTrackerConfig | None = None,
        qualifier_config: TubePoseQualifierConfig | None = None,
        depth_config: ForegroundDepthGateConfig | None = None,
        config: RealTimeTubePoseTrackerConfig | None = None,
        tracking_id: str | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider = provider
        self.backend = backend
        self.T_base_camera = validate_rigid_transform(T_base_camera, name="T_base_camera").copy()
        initial = np.asarray(reviewed_initial_mask)
        if initial.ndim != 2:
            raise ValueError("reviewed_initial_mask must be HxW")
        self._reviewed_initial_mask = initial.copy()
        self._reviewed_ends = reviewed_ends
        self.mask_tracker = StaticSourceMaskTracker(mask_config)
        self.qualifier = TubePoseQualifier(reviewed_ends, qualifier_config)
        self.depth_config = depth_config or ForegroundDepthGateConfig()
        self.config = config or RealTimeTubePoseTrackerConfig()
        self._tracking_id_override = tracking_id
        self._monotonic = monotonic
        self._last_timestamp_s: float | None = None
        self._frame_sequence = 0

    def update(
        self,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
        timestamp_s: float,
        frame_id: str,
        external_mask: np.ndarray | None = None,
        occluder_mask: np.ndarray | None = None,
    ) -> TubePoseTrackResult:
        image = _rgb(rgb)
        timestamp = float(timestamp_s)
        if self._last_timestamp_s is not None and timestamp <= self._last_timestamp_s:
            raise ValueError("tube pose timestamps must be strictly increasing")
        self._last_timestamp_s = timestamp
        self._frame_sequence += 1
        now = float(self._monotonic())
        age = now - timestamp
        if age < -self.config.future_timestamp_tolerance_s:
            return self._result_without_mask("HOLD", "capture_timestamp_in_future")
        if age > self.config.max_capture_age_s:
            return self._result_without_mask("HOLD", "capture_frame_stale")

        if not self.mask_tracker.seeded:
            mask_track = self.mask_tracker.seed(
                rgb=image,
                reviewed_mask=self._reviewed_initial_mask,
                timestamp_s=timestamp,
                tracking_id=self._tracking_id_override,
            )
        else:
            mask_track = self.mask_tracker.update(
                rgb=image,
                timestamp_s=timestamp,
                external_mask=external_mask,
                occluder_mask=occluder_mask,
            )
        if not mask_track.accepted or mask_track.mask is None or mask_track.mask_id is None:
            return TubePoseTrackResult("HOLD", None, mask_track, mask_track.reasons, None)

        request = PoseRequest(
            request_id=f"local-depth-gate-{self._frame_sequence}",
            capture=PoseCapture(
                sequence=self._frame_sequence,
                frame_id=frame_id,
                timestamp_s=timestamp,
                clock_domain="client_monotonic",
            ),
            rgb=image,
            depth_m=depth_m,
            K=K,
            mask=mask_track.mask,
            mask_source=mask_track.mask_id,
            mask_source_score=mask_track.consistency_iou,
        )
        depth_result = evaluate_foreground_depth(request, self.depth_config)
        depth = depth_result.diagnostics
        if not depth.gate_passed:
            return TubePoseTrackResult("HOLD", None, mask_track, depth.reject_reasons, None)

        started = float(self._monotonic())
        try:
            proposal = self.provider.estimate(
                frame_id=frame_id,
                capture_timestamp_s=timestamp,
                rgb=image,
                depth_m=request.depth_m,
                K=request.K,
                mask=mask_track.mask,
                mask_source=mask_track.mask_id,
                mask_source_score=mask_track.consistency_iou,
            )
        except Exception as error:
            return TubePoseTrackResult(
                "HOLD",
                None,
                mask_track,
                (f"pose_provider_rejected:{type(error).__name__}",),
                float(self._monotonic()) - started,
            )
        provider_latency = float(self._monotonic()) - started
        if provider_latency > self.config.max_provider_latency_s:
            return TubePoseTrackResult(
                "HOLD",
                None,
                mask_track,
                ("pose_provider_latency_exceeded",),
                provider_latency,
            )
        current_ends = (
            self._reviewed_ends.warped(mask_track.seed_to_current)
            if mask_track.seed_to_current is not None
            else self._reviewed_ends
        )
        qualified = self.qualifier.update(
            proposal=proposal,
            K=request.K,
            current_reviewed_ends=current_ends,
            foreground_depth_m=depth_result.sanitized_depth_m,
        )
        if qualified.pose is None:
            return TubePoseTrackResult(
                qualified.decision,
                None,
                mask_track,
                qualified.reasons,
                provider_latency,
            )
        pose = qualified.pose
        background_fraction = depth.mask_near_background_fraction
        assert mask_track.consistency_iou is not None
        estimate = ObjectPoseEstimate(
            object_id="tube",
            T_base_object=self.T_base_camera @ pose.T_camera_tube,
            observed_at_s=timestamp,
            position_std_m=math.hypot(pose.position_std_m, self.config.camera_position_std_m),
            orientation_std_deg=math.hypot(
                pose.orientation_std_deg, self.config.camera_orientation_std_deg
            ),
            backend=self.backend,
            tracking_id=mask_track.tracking_id,
            inference_latency_s=provider_latency,
            mask_id=mask_track.mask_id,
            mask_iou=mask_track.consistency_iou,
            valid_depth_count=depth.valid_mask_pixels,
            foreground_depth_support_count=depth.selected_cluster_pixels,
            foreground_depth_support_fraction=depth.selected_cluster_fraction,
            background_depth_fraction=(1.0 if background_fraction is None else background_fraction),
            masked_depth_cluster_qualified=depth.gate_passed,
            requires_rgbd_quality_gates=True,
            axis_sign_source=pose.axis_sign_source,
            cylindrical_roll_canonicalized=pose.cylindrical_roll_canonicalized,
            occluded_by_gripper=False,
        )
        return TubePoseTrackResult("TRACKED", estimate, mask_track, (), provider_latency)

    def _result_without_mask(self, decision: str, reason: str) -> TubePoseTrackResult:
        tracking_id = self.mask_tracker.tracking_id or "unseeded-tube"
        empty = StaticMaskTrackResult(
            decision="HOLD",
            tracking_id=tracking_id,
            mask_id=None,
            mask=None,
            consistency_iou=None,
            occluded=False,
            reasons=(reason,),
            seed_to_current=None,
            tracked_feature_count=0,
            flow_inlier_fraction=None,
            ecc_correlation=None,
            score_provenance=None,
        )
        return TubePoseTrackResult(decision, None, empty, (reason,), None)


class StaticSourceTubeSceneInterpreter:
    """``test_tube_live_io.SceneInterpreter`` adapter for the source stages.

    A local foreground-depth gate runs before the live adapter invokes the
    remote provider.  Mask/depth failure is represented by ``tube=None``, so
    the controller observes no pose and holds.  Proposal qualification remains
    separate in :meth:`qualify_pose` and never converts mask agreement into
    F2G confidence.

    Later task predicates may be supplied by a separate scene interpreter via
    ``downstream``.  Without one, they stay false and the later stages remain
    fail-closed.
    """

    _SOURCE_STATE_NAMES = {"observe", "approach_tube", "side_grasp"}

    def __init__(
        self,
        *,
        reviewed_seed_rgb: np.ndarray,
        reviewed_initial_mask: np.ndarray,
        reviewed_ends: ReviewedTubeEnds,
        expected_camera_serial: str,
        expected_K: np.ndarray,
        mask_config: StaticSourceMaskTrackerConfig | None = None,
        qualifier_config: TubePoseQualifierConfig | None = None,
        depth_config: ForegroundDepthGateConfig | None = None,
        tracking_id: str | None = None,
        external_mask_pattern: str | None = None,
        hold_position_std_m: float = 1.0,
        hold_orientation_std_deg: float = 180.0,
        downstream: object | None = None,
    ) -> None:
        seed_rgb = _rgb(reviewed_seed_rgb)
        self._initial_mask = _binary_mask(
            reviewed_initial_mask,
            seed_rgb.shape[:2],
            "reviewed_initial_mask",
        )
        self._reviewed_seed_rgb = seed_rgb.copy()
        self._reviewed_ends = reviewed_ends
        if not isinstance(expected_camera_serial, str) or not expected_camera_serial:
            raise ValueError("expected_camera_serial must be non-empty")
        self._expected_camera_serial = expected_camera_serial
        self._expected_K = np.asarray(expected_K, dtype=float)
        if self._expected_K.shape != (3, 3) or not np.isfinite(self._expected_K).all():
            raise ValueError("expected_K must be finite 3x3")
        self._mask_tracker = StaticSourceMaskTracker(mask_config)
        self._qualifier = TubePoseQualifier(reviewed_ends, qualifier_config)
        self._depth_config = depth_config or ForegroundDepthGateConfig()
        self._tracking_id = tracking_id
        self._external_mask_pattern = external_mask_pattern
        self._hold_position_std_m = _positive(hold_position_std_m, "hold_position_std_m")
        self._hold_orientation_std_deg = _positive(
            hold_orientation_std_deg, "hold_orientation_std_deg"
        )
        self._downstream = downstream
        self._sequence = 0
        self._last_track: StaticMaskTrackResult | None = None
        self._last_hold_reasons: tuple[str, ...] = ()
        self._last_foreground_depth_m: np.ndarray | None = None

    @property
    def last_hold_reasons(self) -> tuple[str, ...]:
        return self._last_hold_reasons

    def observe(
        self,
        *,
        frame: object,
        state: object,
        T_base_tcp: np.ndarray,
    ) -> object:
        from .test_tube_live_io import SceneEvidence, SceneObjectMask

        if getattr(frame, "serial", None) != self._expected_camera_serial:
            raise TubePoseTrackingError("live camera serial differs from reviewed seed")
        live_K = np.asarray(getattr(getattr(frame, "intrinsics"), "matrix"), dtype=float)
        if live_K.shape != (3, 3) or not np.allclose(
            live_K,
            self._expected_K,
            atol=1e-9,
            rtol=0.0,
        ):
            raise TubePoseTrackingError("live camera intrinsics differ from reviewed seed")
        downstream_evidence = None
        if self._downstream is not None:
            downstream_evidence = self._downstream.observe(
                frame=frame,
                state=state,
                T_base_tcp=T_base_tcp,
            )
        state_name = getattr(state, "value", str(state)).lower()
        if state_name not in self._SOURCE_STATE_NAMES:
            if downstream_evidence is not None:
                return downstream_evidence
            return SceneEvidence(tube=None, socket=None)

        rgb = _rgb(np.asarray(getattr(frame, "rgb")))
        timestamp = float(getattr(frame, "host_received_monotonic_s"))
        if not self._mask_tracker.seeded:
            # The mask/endpoints were reviewed in ``_reviewed_seed_rgb``.  Seed
            # that exact image, then register it into the first live frame;
            # treating a same-sized live frame as the reviewed frame would
            # silently apply stale pixels after any camera/object motion.
            seed_timestamp = timestamp - min(
                1e-3,
                0.5 * self._mask_tracker.config.max_tracking_gap_s,
            )
            self._mask_tracker.seed(
                rgb=self._reviewed_seed_rgb,
                reviewed_mask=self._initial_mask,
                timestamp_s=seed_timestamp,
                tracking_id=self._tracking_id,
            )
            track = self._mask_tracker.update(
                rgb=rgb,
                timestamp_s=timestamp,
                external_mask=self._external_mask(frame),
            )
        else:
            track = self._mask_tracker.update(
                rgb=rgb,
                timestamp_s=timestamp,
                external_mask=self._external_mask(frame),
            )
        self._last_track = track
        if not track.accepted or track.mask is None or track.mask_id is None:
            self._last_hold_reasons = track.reasons
            return _replace_scene_tube(downstream_evidence, None)

        self._sequence += 1
        request = PoseRequest(
            request_id=f"scene-local-depth-{self._sequence}",
            capture=PoseCapture(
                sequence=self._sequence,
                frame_id=f"scene:{getattr(frame, 'sequence', self._sequence)}",
                timestamp_s=timestamp,
                clock_domain="client_monotonic",
            ),
            rgb=rgb,
            depth_m=np.asarray(getattr(frame, "depth_m")),
            K=np.asarray(getattr(getattr(frame, "intrinsics"), "matrix")),
            mask=track.mask,
            mask_source=track.mask_id,
            mask_source_score=track.consistency_iou,
        )
        depth_result = evaluate_foreground_depth(request, self._depth_config)
        depth = depth_result.diagnostics
        if not depth.gate_passed:
            self._last_foreground_depth_m = None
            self._last_hold_reasons = depth.reject_reasons
            return _replace_scene_tube(downstream_evidence, None)
        self._last_foreground_depth_m = depth_result.sanitized_depth_m
        self._last_hold_reasons = ()
        assert track.consistency_iou is not None
        tube = SceneObjectMask(
            object_id="tube",
            mask=track.mask,
            mask_id=track.mask_id,
            mask_iou=track.consistency_iou,
            tracking_id=track.tracking_id,
            occluded_by_gripper=False,
        )
        return _replace_scene_tube(downstream_evidence, tube)

    def qualify_pose(
        self,
        *,
        object_id: str,
        proposal: PoseObservation,
        mask: object,
        frame: object,
    ) -> object:
        from .test_tube_live_io import QualifiedCameraPose

        if object_id != "tube":
            if self._downstream is None:
                raise TubePoseTrackingError(f"unsupported scene object: {object_id}")
            return self._downstream.qualify_pose(
                object_id=object_id, proposal=proposal, mask=mask, frame=frame
            )
        if self._last_track is None or self._last_track.seed_to_current is None:
            raise TubePoseTrackingError("tube mask track is unavailable for pose qualification")
        tracking_id = str(getattr(mask, "tracking_id"))
        if tracking_id != self._last_track.tracking_id:
            raise TubePoseTrackingError("pose mask changed tracking identity")
        ends = self._reviewed_ends.warped(self._last_track.seed_to_current)
        result = self._qualifier.update(
            proposal=proposal,
            K=np.asarray(getattr(getattr(frame, "intrinsics"), "matrix")),
            current_reviewed_ends=ends,
            foreground_depth_m=self._last_foreground_depth_m,
        )
        if result.pose is not None and result.decision == "TRACKED":
            self._last_hold_reasons = ()
            return QualifiedCameraPose(
                T_camera_object=result.pose.T_camera_tube,
                position_std_m=result.pose.position_std_m,
                orientation_std_deg=result.pose.orientation_std_deg,
                axis_sign_source=result.pose.axis_sign_source,
                cylindrical_roll_canonicalized=True,
            )
        self._last_hold_reasons = result.reasons
        if result.provisional_T_camera_tube is None:
            raise TubePoseTrackingError("tube pose is unqualified: " + ", ".join(result.reasons))
        # The current proposal is exposed only with deliberately blocking
        # uncertainty.  This lets the resident controller keep observing while
        # preventing a motion command during warm-up or a rejected jump.
        return QualifiedCameraPose(
            T_camera_object=result.provisional_T_camera_tube,
            position_std_m=self._hold_position_std_m,
            orientation_std_deg=self._hold_orientation_std_deg,
            axis_sign_source=(
                "open_end_detector" if result.decision == "WARMUP" else "temporal_continuity"
            ),
            cylindrical_roll_canonicalized=True,
        )

    def held_tube_pose(self, **kwargs: object) -> object | None:
        if self._downstream is None:
            return None
        return self._downstream.held_tube_pose(**kwargs)

    def set_grasp_relation(
        self,
        T_tube_tcp_actual: np.ndarray,
        *,
        T_base_tube_at_contact: np.ndarray,
    ) -> None:
        """Forward the realized contact geometry to the downstream producer."""

        if self._downstream is None:
            raise TubePoseTrackingError("downstream scene interpreter is required for extraction")
        setter = getattr(self._downstream, "set_grasp_relation", None)
        if not callable(setter):
            raise TubePoseTrackingError(
                "downstream scene interpreter cannot accept a grasp relation"
            )
        setter(
            T_tube_tcp_actual,
            T_base_tube_at_contact=T_base_tube_at_contact,
        )

    def reset_support_history_for_release(self) -> None:
        """Forward the release boundary to the downstream RGB qualifier."""

        if self._downstream is None:
            raise TubePoseTrackingError("downstream scene interpreter is required for release")
        resetter = getattr(self._downstream, "reset_support_history_for_release", None)
        if not callable(resetter):
            raise TubePoseTrackingError(
                "downstream scene interpreter cannot reset release support history"
            )
        resetter()

    def observe_tactile(self, **kwargs: object) -> None:
        """Forward insertion events to the downstream scene producer."""

        if self._downstream is None:
            return
        observer = getattr(self._downstream, "observe_tactile", None)
        if callable(observer):
            observer(**kwargs)

    def close(self) -> None:
        if self._downstream is not None:
            self._downstream.close()

    def _external_mask(self, frame: object) -> np.ndarray | None:
        if self._external_mask_pattern is None:
            return None
        sequence = int(getattr(frame, "sequence"))
        path = self._external_mask_pattern.format(sequence=sequence)
        try:
            import cv2
        except ImportError as error:  # pragma: no cover
            raise TubePoseTrackingError("OpenCV is required to load external masks") from error
        value = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return None if value is None else value > 0


def _replace_scene_tube(evidence: object | None, tube: object | None) -> object:
    from .test_tube_live_io import SceneEvidence

    if evidence is None:
        return SceneEvidence(tube=tube, socket=None)
    return SceneEvidence(
        tube=tube,
        socket=getattr(evidence, "socket"),
        front_rgb_angle_target_reached=bool(getattr(evidence, "front_rgb_angle_target_reached")),
        tube_clear_of_source=bool(getattr(evidence, "tube_clear_of_source")),
        tube_tip_at_socket_entry=bool(getattr(evidence, "tube_tip_at_socket_entry")),
        tube_socket_axes_aligned=bool(getattr(evidence, "tube_socket_axes_aligned")),
        tube_stays_supported=bool(getattr(evidence, "tube_stays_supported")),
    )


def _load_mask(path_value: object) -> np.ndarray:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("reviewed_initial_mask must be a non-empty path")
    path = Path(path_value).expanduser().resolve()
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover
            raise TubePoseTrackingError("OpenCV is required to load the seed mask") from error
        value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if value is None:
            raise ValueError(f"cannot read reviewed seed mask: {path}")
    return np.asarray(value) > 0


def _load_rgb(path_value: object) -> np.ndarray:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("reviewed_seed_rgb must be a non-empty path")
    path = Path(path_value).expanduser().resolve()
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover
            raise TubePoseTrackingError("OpenCV is required to load the seed RGB") from error
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot read reviewed seed RGB: {path}")
        value = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _rgb(np.asarray(value))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_qualified_source_seed(
    config: Mapping[str, object],
    *,
    reviewed_ends: ReviewedTubeEnds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load a human-reviewed seed pair only when its receipt binds every input."""

    def artifact(name: str) -> Path:
        value = config.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty path")
        result = Path(value).expanduser().resolve()
        if not result.is_file():
            raise ValueError(f"{name} does not exist: {result}")
        return result

    rgb_path = artifact("reviewed_seed_rgb")
    mask_path = artifact("reviewed_initial_mask")
    receipt_path = artifact("reviewed_seed_qualification_receipt")
    camera_receipt_path = artifact("camera_extrinsic_qualification_receipt")
    intrinsics_path = artifact("intrinsics")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("reviewed seed qualification receipt must be an object")
    camera_receipt = json.loads(camera_receipt_path.read_text(encoding="utf-8"))
    if not isinstance(camera_receipt, Mapping):
        raise ValueError("camera extrinsic qualification receipt must be an object")
    camera_serial = config.get("camera_serial")
    valid = bool(
        payload.get("schema") == SOURCE_SEED_QUALIFICATION_SCHEMA
        and payload.get("status") == "accepted"
        and payload.get("runtime_qualified") is True
        and isinstance(payload.get("qualification_id"), str)
        and payload["qualification_id"]
        and isinstance(payload.get("workspace_calibration_id"), str)
        and payload["workspace_calibration_id"]
        and isinstance(payload.get("camera_extrinsic_qualification_id"), str)
        and payload["camera_extrinsic_qualification_id"]
        and isinstance(camera_serial, str)
        and camera_serial
        and payload.get("camera_serial") == camera_serial
        and payload.get("intrinsics_sha256") == _file_sha256(intrinsics_path)
        and camera_receipt.get("schema") == "demo2gripper.test_tube_live_io_qualification.v1"
        and camera_receipt.get("kind") == "fixed_camera_extrinsic"
        and camera_receipt.get("runtime_qualified") is True
        and payload.get("workspace_calibration_id")
        == camera_receipt.get("workspace_calibration_id")
        and payload.get("camera_extrinsic_qualification_id")
        == camera_receipt.get("qualification_id")
        and payload.get("seed_rgb_sha256") == _file_sha256(rgb_path)
        and payload.get("reviewed_mask_sha256") == _file_sha256(mask_path)
        and payload.get("reviewed_endpoints_evidence_id") == reviewed_ends.evidence_id
    )
    try:
        receipt_open = np.asarray(payload.get("reviewed_open_uv"), dtype=float)
        receipt_closed = np.asarray(payload.get("reviewed_closed_uv"), dtype=float)
    except (TypeError, ValueError):
        valid = False
    else:
        valid = bool(
            valid
            and receipt_open.shape == (2,)
            and receipt_closed.shape == (2,)
            and np.isfinite(receipt_open).all()
            and np.isfinite(receipt_closed).all()
            and np.allclose(receipt_open, reviewed_ends.open_uv, atol=1e-6, rtol=0.0)
            and np.allclose(
                receipt_closed,
                reviewed_ends.closed_uv,
                atol=1e-6,
                rtol=0.0,
            )
        )
    if not valid:
        raise ValueError("reviewed source seed is not runtime qualified")
    rgb = _load_rgb(str(rgb_path))
    mask = _load_mask(str(mask_path))
    _binary_mask(mask, rgb.shape[:2], "reviewed_initial_mask")
    K = np.load(intrinsics_path, allow_pickle=False)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("reviewed seed intrinsics must be finite 3x3")
    assert isinstance(camera_serial, str)
    return rgb, mask, np.asarray(K, dtype=float), camera_serial


def _load_optional_factory(spec: object, config: object, output_dir: object) -> object | None:
    if spec is None:
        return None
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError("downstream_factory must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(factory):
        raise ValueError(f"downstream factory is not callable: {spec}")
    return factory(config=config or {}, output_dir=output_dir)


def create_static_source_tube_scene_interpreter(
    *, config: Mapping[str, object], output_dir: object
) -> StaticSourceTubeSceneInterpreter:
    """Factory directly loadable by ``test_tube_live_io.create_live_io``."""

    if not isinstance(config, Mapping):
        raise ValueError("scene interpreter config must be an object")
    open_uv = config.get("reviewed_open_uv")
    closed_uv = config.get("reviewed_closed_uv")
    if (
        not isinstance(open_uv, (list, tuple))
        or len(open_uv) != 2
        or not isinstance(closed_uv, (list, tuple))
        or len(closed_uv) != 2
    ):
        raise ValueError("reviewed_open_uv and reviewed_closed_uv must each have two values")
    evidence_id = config.get("reviewed_endpoints_evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("reviewed_endpoints_evidence_id must be non-empty")

    def section(name: str) -> dict[str, object]:
        value = config.get(name, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be an object")
        return dict(value)

    reviewed_ends = ReviewedTubeEnds(
        open_uv=(float(open_uv[0]), float(open_uv[1])),
        closed_uv=(float(closed_uv[0]), float(closed_uv[1])),
        evidence_id=evidence_id,
    )
    seed_rgb, seed_mask, expected_K, expected_serial = _load_qualified_source_seed(
        config,
        reviewed_ends=reviewed_ends,
    )
    downstream = _load_optional_factory(
        config.get("downstream_factory"),
        config.get("downstream_config", {}),
        output_dir,
    )
    return StaticSourceTubeSceneInterpreter(
        reviewed_seed_rgb=seed_rgb,
        reviewed_initial_mask=seed_mask,
        reviewed_ends=reviewed_ends,
        expected_camera_serial=expected_serial,
        expected_K=expected_K,
        mask_config=StaticSourceMaskTrackerConfig(**section("mask_tracker")),
        qualifier_config=TubePoseQualifierConfig(**section("pose_qualifier")),
        depth_config=ForegroundDepthGateConfig(**section("foreground_depth")),
        tracking_id=(None if config.get("tracking_id") is None else str(config["tracking_id"])),
        external_mask_pattern=(
            None
            if config.get("external_mask_pattern") is None
            else str(config["external_mask_pattern"])
        ),
        hold_position_std_m=float(config.get("hold_position_std_m", 1.0)),
        hold_orientation_std_deg=float(config.get("hold_orientation_std_deg", 180.0)),
        downstream=downstream,
    )


__all__ = [
    "PoseProvider",
    "QualifiedTubePose",
    "RealTimeTubePoseTracker",
    "RealTimeTubePoseTrackerConfig",
    "ReviewedTubeEnds",
    "SOURCE_SEED_QUALIFICATION_SCHEMA",
    "StaticSourceTubeSceneInterpreter",
    "StaticMaskTrackResult",
    "StaticSourceMaskTracker",
    "StaticSourceMaskTrackerConfig",
    "TubePoseQualificationResult",
    "TubePoseQualifier",
    "TubePoseQualifierConfig",
    "TubePoseTrackResult",
    "TubePoseTrackingError",
    "create_static_source_tube_scene_interpreter",
]
