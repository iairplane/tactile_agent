"""Online scene bootstrap for the test-tube task.

This path replaces the old reviewed source RGB/mask/endpoints with observations
from the current camera session:

* SAM 3 proposes the tube and the source support/cup from text prompts;
* the visible free endpoint must be unambiguously farther from that support;
* foreground depth must pass the same local background-rejection gate used by
  the static path;
* F2G-Pose is still only a pose/size proposal and must pass image/depth and
  temporal cross-checks before motion can be authorized.

The module deliberately returns no source object while bootstrapping.  A model
  timeout, ambiguous support geometry, unstable identity, or failed depth check therefore
causes the resident controller to hold rather than use a stale seed.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from .f2g_pose_service import ForegroundDepthGateConfig, PoseRequest, evaluate_foreground_depth
from .franka import validate_rigid_transform
from .pose_observation import PoseCapture, PoseObservation
from .sam3_mask_service import Sam3MaskClient, Sam3MaskObservation
from .test_tube_pose_tracker import (
    ReviewedTubeEnds,
    StaticSourceMaskTracker,
    StaticSourceMaskTrackerConfig,
    TubePoseQualifier,
    TubePoseQualifierConfig,
    TubePoseTrackingError,
    _binary_mask,
    _mask_iou,
    _replace_scene_tube,
    _rgb,
)


class OnlineMaskProvider(Protocol):
    def segment(
        self,
        *,
        frame_id: str,
        capture_timestamp_s: float,
        rgb: np.ndarray,
        prompt: str,
    ) -> Sam3MaskObservation: ...


@dataclass(frozen=True)
class OnlineTubeBootstrapConfig:
    min_tube_score: float = 0.55
    min_support_score: float = 0.35
    min_stable_frames: int = 3
    min_consecutive_mask_iou: float = 0.72
    max_endpoint_jitter_px: float = 8.0
    min_axis_length_px: float = 35.0
    min_support_assignment_margin_px: float = 18.0
    max_near_support_distance_px: float = 45.0
    max_tube_support_overlap_fraction: float = 0.45
    min_mask_pixels: int = 120
    min_support_mask_pixels: int = 200
    source_roi_uvxy: tuple[int, int, int, int] | None = None
    support_roi_uvxy: tuple[int, int, int, int] | None = None
    support_anchor_uv: tuple[float, float] | None = None
    allow_qualified_base_z_fallback: bool = False
    endpoint_depth_radius_px: int = 7
    min_base_z_separation_m: float = 0.025

    def __post_init__(self) -> None:
        for name in (
            "min_tube_score",
            "min_support_score",
            "min_consecutive_mask_iou",
            "max_tube_support_overlap_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        for name in (
            "max_endpoint_jitter_px",
            "min_axis_length_px",
            "min_support_assignment_margin_px",
            "max_near_support_distance_px",
            "min_base_z_separation_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "min_stable_frames",
            "min_mask_pixels",
            "min_support_mask_pixels",
            "endpoint_depth_radius_px",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("source_roi_uvxy", "support_roi_uvxy"):
            value = getattr(self, name)
            if value is None:
                continue
            roi = tuple(value)
            if len(roi) != 4 or not (0 <= roi[0] < roi[2] and 0 <= roi[1] < roi[3]):
                raise ValueError(f"{name} must be a valid u0,v0,u1,v1 box")
        if self.support_anchor_uv is not None:
            anchor = np.asarray(self.support_anchor_uv, dtype=float)
            if anchor.shape != (2,) or not np.isfinite(anchor).all():
                raise ValueError("support_anchor_uv must contain two finite pixels")
        if not isinstance(self.allow_qualified_base_z_fallback, bool):
            raise TypeError("allow_qualified_base_z_fallback must be boolean")


@dataclass(frozen=True)
class OnlineTubeSizeConfig:
    """Shape and temporal gates for F2G's runtime metric dimensions."""

    min_stable_frames: int = 3
    min_length_m: float = 0.030
    max_length_m: float = 0.300
    min_aspect_ratio: float = 1.5
    max_cross_section_ratio: float = 1.8
    max_relative_window_deviation: float = 0.15

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_stable_frames, bool)
            or not isinstance(self.min_stable_frames, int)
            or self.min_stable_frames < 2
        ):
            raise ValueError("min_stable_frames must be an integer of at least two")
        for name in (
            "min_length_m",
            "max_length_m",
            "min_aspect_ratio",
            "max_cross_section_ratio",
            "max_relative_window_deviation",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_length_m <= self.min_length_m:
            raise ValueError("max_length_m must exceed min_length_m")


class OnlineTubeSizeFilter:
    """Accept only an elongated, cylinder-like, temporally stable F2G size."""

    def __init__(self, config: OnlineTubeSizeConfig | None = None) -> None:
        self.config = config or OnlineTubeSizeConfig()
        self._window: deque[np.ndarray] = deque(maxlen=self.config.min_stable_frames)
        self.last_reasons: tuple[str, ...] = ("runtime_f2g_tube_size_not_observed",)

    def reset(self, reason: str = "runtime_f2g_tube_size_not_observed") -> None:
        self._window.clear()
        self.last_reasons = (reason,)

    def update(self, size_m: object) -> tuple[float, float, float] | None:
        try:
            size = np.sort(np.asarray(size_m, dtype=float))
        except (TypeError, ValueError):
            size = np.asarray([], dtype=float)
        reasons: list[str] = []
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0.0):
            reasons.append("runtime_f2g_tube_size_invalid")
        else:
            if not self.config.min_length_m <= size[2] <= self.config.max_length_m:
                reasons.append("runtime_f2g_tube_length_out_of_bounds")
            if size[2] / size[1] < self.config.min_aspect_ratio:
                reasons.append("runtime_f2g_shape_not_tube_like")
            if size[1] / size[0] > self.config.max_cross_section_ratio:
                reasons.append("runtime_f2g_cross_section_not_tube_like")
        if reasons:
            self._window.clear()
            self.last_reasons = tuple(reasons)
            return None
        self._window.append(size)
        if len(self._window) < self.config.min_stable_frames:
            self.last_reasons = ("runtime_f2g_tube_size_window_not_full",)
            return None
        values = np.stack(self._window)
        median = np.median(values, axis=0)
        relative = float(np.max(np.abs(values - median) / median))
        if relative > self.config.max_relative_window_deviation:
            self.last_reasons = ("runtime_f2g_tube_size_unstable",)
            return None
        self.last_reasons = ()
        return tuple(float(value) for value in median)


@dataclass(frozen=True)
class OnlineTubeSeed:
    mask: np.ndarray
    support_mask: np.ndarray | None
    open_uv: tuple[float, float]
    closed_uv: tuple[float, float]
    tube_score: float
    support_score: float | None
    axis_sign_source: str
    stable_frames: int
    evidence_id: str


class OnlineTubeSeedBuilder:
    """Require a stable tube identity and a task-grounded free endpoint.

    The source tube is supported by a cup.  Its open/free endpoint is therefore
    the PCA endpoint farther from the support, while the closed endpoint is the
    one next to the support.  This deliberately does not use an ``open rim``
    text prompt: SAM 3 was observed to return the entire transparent tube for
    that prompt, which cannot disambiguate the signed axis.

    A base-height fallback is available only when the caller explicitly binds
    a qualified ``T_base_camera`` and both endpoint depth estimates have enough
    vertical separation.  Any ambiguous observation clears the temporal
    window and returns ``None``.
    """

    def __init__(self, config: OnlineTubeBootstrapConfig | None = None) -> None:
        self.config = config or OnlineTubeBootstrapConfig()
        self._history: deque[OnlineTubeSeed] = deque(maxlen=self.config.min_stable_frames)
        self.last_reasons: tuple[str, ...] = ("online_seed_not_observed",)

    def reset(self, reason: str) -> None:
        self._history.clear()
        self.last_reasons = (reason,)

    def update(
        self,
        *,
        tube: Sam3MaskObservation,
        support: Sam3MaskObservation | None,
        image_shape: tuple[int, int],
        depth_m: np.ndarray | None = None,
        K: np.ndarray | None = None,
        T_base_camera: np.ndarray | None = None,
        camera_extrinsic_qualified: bool = False,
    ) -> OnlineTubeSeed | None:
        reasons: list[str] = []
        if tube.mask.shape != image_shape or (
            support is not None and support.mask.shape != image_shape
        ):
            reasons.append("sam_mask_shape_mismatch")
            self.reset(reasons[0])
            self.last_reasons = tuple(reasons)
            return None
        if tube.score < self.config.min_tube_score:
            reasons.append("tube_sam_score_too_low")
        try:
            mask = _binary_mask(tube.mask, image_shape, "online tube mask")
        except ValueError:
            reasons.append("tube_sam_mask_invalid")
            self.reset(reasons[0])
            self.last_reasons = tuple(reasons)
            return None
        if int(mask.sum()) < self.config.min_mask_pixels:
            reasons.append("tube_sam_mask_too_small")
        if self.config.source_roi_uvxy is not None:
            u0, v0, u1, v1 = self.config.source_roi_uvxy
            inside = np.zeros(image_shape, dtype=bool)
            inside[v0:v1, u0:u1] = True
            if float(np.count_nonzero(mask & inside) / np.count_nonzero(mask)) < 0.90:
                reasons.append("tube_sam_mask_outside_source_roi")
        unsigned_ends = None if reasons else self._unsigned_endpoints(mask)
        if unsigned_ends is None and not reasons:
            reasons.append("tube_mask_does_not_resolve_axis_endpoints")
        signed = None
        support_score: float | None = None
        support_reasons: tuple[str, ...] = ()
        if unsigned_ends is not None:
            signed, support_score, support_reasons = self._sign_from_support(
                tube_mask=mask,
                support=support,
                endpoints=unsigned_ends,
                image_shape=image_shape,
            )
            if signed is None and self.config.allow_qualified_base_z_fallback:
                signed = self._sign_from_qualified_base_z(
                    tube_mask=mask,
                    endpoints=unsigned_ends,
                    depth_m=depth_m,
                    K=K,
                    T_base_camera=T_base_camera,
                    camera_extrinsic_qualified=camera_extrinsic_qualified,
                )
                if signed is None:
                    reasons.extend(support_reasons)
                    reasons.append("qualified_base_z_does_not_resolve_tube_axis_sign")
            elif signed is None:
                reasons.extend(support_reasons)
        if reasons:
            self.reset(reasons[0])
            self.last_reasons = tuple(reasons)
            return None
        assert signed is not None
        open_uv, closed_uv, axis_sign_source = signed
        if self._history:
            previous = self._history[-1]
            if _mask_iou(mask, previous.mask) < self.config.min_consecutive_mask_iou:
                self.reset("online_tube_mask_identity_unstable")
                return None
            jitter = max(
                float(np.linalg.norm(np.asarray(open_uv) - previous.open_uv)),
                float(np.linalg.norm(np.asarray(closed_uv) - previous.closed_uv)),
            )
            if jitter > self.config.max_endpoint_jitter_px:
                self.reset("online_tube_endpoints_unstable")
                return None
            if axis_sign_source != previous.axis_sign_source:
                self.reset("online_tube_axis_sign_source_changed")
                return None
        current_support_mask = (
            None
            if support is None or axis_sign_source != "source_support_geometry"
            else _binary_mask(support.mask, image_shape, "online source-support mask")
        )
        if (
            self._history
            and current_support_mask is not None
            and self._history[-1].support_mask is not None
            and _mask_iou(current_support_mask, self._history[-1].support_mask)
            < self.config.min_consecutive_mask_iou
        ):
            self.reset("online_source_support_identity_unstable")
            return None
        digest = hashlib.sha256()
        digest.update(mask.view(np.uint8).tobytes())
        if current_support_mask is not None:
            digest.update(current_support_mask.view(np.uint8).tobytes())
        digest.update(np.asarray([*open_uv, *closed_uv], dtype=np.float64).tobytes())
        digest.update(axis_sign_source.encode("utf-8"))
        current = OnlineTubeSeed(
            mask=mask.copy(),
            support_mask=(None if current_support_mask is None else current_support_mask.copy()),
            open_uv=open_uv,
            closed_uv=closed_uv,
            tube_score=float(tube.score),
            support_score=support_score,
            axis_sign_source=axis_sign_source,
            stable_frames=len(self._history) + 1,
            evidence_id=f"online-sam3-{digest.hexdigest()[:16]}",
        )
        self._history.append(current)
        if len(self._history) < self.config.min_stable_frames:
            self.last_reasons = ("online_seed_temporal_window_not_full",)
            return None
        self.last_reasons = ()
        return current

    def _unsigned_endpoints(self, tube_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        rows, columns = np.nonzero(tube_mask)
        points = np.column_stack((columns, rows)).astype(float)
        centre = points.mean(axis=0)
        covariance = (points - centre).T @ (points - centre)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        projection = (points - centre) @ axis
        low = centre + np.quantile(projection, 0.01) * axis
        high = centre + np.quantile(projection, 0.99) * axis
        axis_length = float(np.linalg.norm(high - low))
        if axis_length < self.config.min_axis_length_px:
            return None
        return low, high

    def _sign_from_support(
        self,
        *,
        tube_mask: np.ndarray,
        support: Sam3MaskObservation | None,
        endpoints: tuple[np.ndarray, np.ndarray],
        image_shape: tuple[int, int],
    ) -> tuple[
        tuple[tuple[float, float], tuple[float, float], str] | None,
        float | None,
        tuple[str, ...],
    ]:
        low, high = endpoints
        support_points: np.ndarray | None = None
        score: float | None = None
        reasons: list[str] = []
        if support is not None:
            score = float(support.score)
            if score < self.config.min_support_score:
                reasons.append("source_support_sam_score_too_low")
            try:
                support_mask = _binary_mask(support.mask, image_shape, "online source-support mask")
            except ValueError:
                support_mask = None
                reasons.append("source_support_sam_mask_invalid")
            if support_mask is not None:
                if int(support_mask.sum()) < self.config.min_support_mask_pixels:
                    reasons.append("source_support_sam_mask_too_small")
                overlap = float(np.count_nonzero(tube_mask & support_mask) / tube_mask.sum())
                if overlap > self.config.max_tube_support_overlap_fraction:
                    reasons.append("source_support_mask_contains_too_much_tube")
                if self.config.support_roi_uvxy is not None:
                    u0, v0, u1, v1 = self.config.support_roi_uvxy
                    inside = np.zeros(image_shape, dtype=bool)
                    inside[v0:v1, u0:u1] = True
                    fraction_inside = float(
                        np.count_nonzero(support_mask & inside) / support_mask.sum()
                    )
                    if fraction_inside < 0.90:
                        reasons.append("source_support_mask_outside_support_roi")
                rows, columns = np.nonzero(support_mask)
                support_points = np.column_stack((columns, rows)).astype(float)
        elif self.config.support_anchor_uv is not None:
            support_points = np.asarray(self.config.support_anchor_uv, dtype=float)[None, :]
        else:
            reasons.append("source_support_observation_missing")
        if reasons or support_points is None:
            return None, score, tuple(reasons)
        low_distance = float(np.min(np.linalg.norm(support_points - low, axis=1)))
        high_distance = float(np.min(np.linalg.norm(support_points - high, axis=1)))
        near, far = sorted((low_distance, high_distance))
        if near > self.config.max_near_support_distance_px:
            reasons.append("source_support_not_adjacent_to_tube_endpoint")
        if far - near < self.config.min_support_assignment_margin_px:
            reasons.append("source_support_does_not_resolve_tube_axis_sign")
        if reasons:
            return None, score, tuple(reasons)
        free, supported = (low, high) if low_distance > high_distance else (high, low)
        sign_source = "source_support_geometry" if support is not None else "source_support_anchor"
        return (
            (tuple(free.tolist()), tuple(supported.tolist()), sign_source),
            score,
            (),
        )

    def _sign_from_qualified_base_z(
        self,
        *,
        tube_mask: np.ndarray,
        endpoints: tuple[np.ndarray, np.ndarray],
        depth_m: np.ndarray | None,
        K: np.ndarray | None,
        T_base_camera: np.ndarray | None,
        camera_extrinsic_qualified: bool,
    ) -> tuple[tuple[float, float], tuple[float, float], str] | None:
        if not camera_extrinsic_qualified:
            return None
        try:
            depth = np.asarray(depth_m, dtype=float)
            intrinsics = np.asarray(K, dtype=float)
            transform = validate_rigid_transform(T_base_camera, name="T_base_camera")
        except (TypeError, ValueError):
            return None
        if (
            depth.shape != tube_mask.shape
            or intrinsics.shape != (3, 3)
            or transform.shape != (4, 4)
            or not np.isfinite(intrinsics).all()
            or not np.isfinite(transform).all()
        ):
            return None
        points_base: list[np.ndarray] = []
        for endpoint in endpoints:
            point_camera = self._endpoint_camera_point(
                endpoint=endpoint,
                tube_mask=tube_mask,
                depth_m=depth,
                K=intrinsics,
            )
            if point_camera is None:
                return None
            points_base.append((transform @ np.r_[point_camera, 1.0])[:3])
        base_z = np.asarray([point[2] for point in points_base])
        if not np.isfinite(base_z).all():
            return None
        if abs(float(base_z[0] - base_z[1])) < self.config.min_base_z_separation_m:
            return None
        high_index = int(np.argmax(base_z))
        free = endpoints[high_index]
        supported = endpoints[1 - high_index]
        return tuple(free.tolist()), tuple(supported.tolist()), "qualified_base_z"

    def _endpoint_camera_point(
        self,
        *,
        endpoint: np.ndarray,
        tube_mask: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
    ) -> np.ndarray | None:
        rows, columns = np.nonzero(tube_mask)
        distances = np.hypot(columns - endpoint[0], rows - endpoint[1])
        selected = distances <= self.config.endpoint_depth_radius_px
        if int(np.count_nonzero(selected)) < 3:
            return None
        selected_rows = rows[selected]
        selected_columns = columns[selected]
        values = depth_m[selected_rows, selected_columns]
        valid = np.isfinite(values) & (values > 0.0)
        if int(np.count_nonzero(valid)) < 3:
            return None
        z = float(np.median(values[valid]))
        u = float(np.median(selected_columns[valid]))
        v = float(np.median(selected_rows[valid]))
        try:
            ray = np.linalg.solve(K, np.array([u, v, 1.0]))
        except np.linalg.LinAlgError:
            return None
        point = ray * z
        return point if np.isfinite(point).all() else None


class OnlineSourceTubeSceneInterpreter:
    """Scene adapter using runtime SAM 3 bootstrap plus qualified F2G poses."""

    _SOURCE_STATE_NAMES = {"observe", "approach_tube", "side_grasp"}

    def __init__(
        self,
        *,
        mask_provider: OnlineMaskProvider,
        expected_camera_serial: str,
        tube_prompt: str,
        source_support_prompt: str,
        T_base_camera: np.ndarray | None = None,
        camera_extrinsic_qualified: bool = False,
        bootstrap_config: OnlineTubeBootstrapConfig | None = None,
        mask_config: StaticSourceMaskTrackerConfig | None = None,
        qualifier_config: TubePoseQualifierConfig | None = None,
        depth_config: ForegroundDepthGateConfig | None = None,
        size_config: OnlineTubeSizeConfig | None = None,
        tracking_id: str = "online-test-tube",
        expected_sam_backend: Mapping[str, object] | None = None,
        downstream: object | None = None,
    ) -> None:
        if not expected_camera_serial:
            raise ValueError("expected_camera_serial must be non-empty")
        if not tube_prompt.strip():
            raise ValueError("tube prompt must be non-empty")
        if not source_support_prompt.strip() and (
            bootstrap_config is None or bootstrap_config.support_anchor_uv is None
        ):
            raise ValueError("source-support prompt or reviewed support anchor is required")
        self._provider = mask_provider
        self._serial = expected_camera_serial
        self._tube_prompt = tube_prompt
        self._support_prompt = source_support_prompt
        self._seed_builder = OnlineTubeSeedBuilder(bootstrap_config)
        self._T_base_camera = (
            None
            if T_base_camera is None
            else validate_rigid_transform(T_base_camera, name="T_base_camera").copy()
        )
        if not isinstance(camera_extrinsic_qualified, bool):
            raise TypeError("camera_extrinsic_qualified must be boolean")
        self._camera_extrinsic_qualified = camera_extrinsic_qualified
        if self._seed_builder.config.allow_qualified_base_z_fallback and (
            self._T_base_camera is None or not self._camera_extrinsic_qualified
        ):
            raise ValueError("base-Z axis fallback requires an explicitly qualified T_base_camera")
        self._mask_tracker = StaticSourceMaskTracker(mask_config)
        self._qualifier_config = qualifier_config or TubePoseQualifierConfig()
        self._source_position_std_limit_m = self._qualifier_config.max_window_position_std_m
        self._source_orientation_std_limit_deg = (
            self._qualifier_config.max_window_orientation_std_deg
        )
        self._qualifier: TubePoseQualifier | None = None
        self._seed_ends: ReviewedTubeEnds | None = None
        self._depth_config = depth_config or ForegroundDepthGateConfig()
        self._size_filter = OnlineTubeSizeFilter(size_config)
        self._tracking_id = tracking_id
        self._expected_sam_backend = dict(expected_sam_backend or {"name": "sam3"})
        self._downstream = downstream
        self._expected_K: np.ndarray | None = None
        self._sequence = 0
        self._last_track = None
        self._last_foreground_depth_m: np.ndarray | None = None
        self._last_hold_reasons: tuple[str, ...] = ("online_seed_not_observed",)
        self._qualified_size_m: tuple[float, float, float] | None = None
        self._qualified_tracking_id: str | None = None
        self._last_qualified_transform: np.ndarray | None = None
        self._qualified_image_mask: np.ndarray | None = None
        self._qualified_image_ends: ReviewedTubeEnds | None = None
        self._source_depth_only_loss = False
        self._source_rgb_identity_retained = False

    @property
    def last_hold_reasons(self) -> tuple[str, ...]:
        return self._last_hold_reasons

    @property
    def source_depth_only_loss(self) -> bool:
        """Current RGB still agrees with the last qualified tube; only depth failed.

        This is visibility evidence for an already committed finite approach,
        never a replacement metric pose or a renewed pose timestamp.
        """
        return self._source_depth_only_loss

    @property
    def source_rgb_identity_retained(self) -> bool:
        """The current SAM image still matches the metric-qualified source tube.

        Unlike :attr:`source_depth_only_loss`, this certificate also survives
        a frame whose support cue cannot re-sign the cylinder axis.  It carries
        no metric pose and is usable only by an already committed finite
        approach whose target came from an earlier qualified frame.
        """

        return self._source_rgb_identity_retained

    @property
    def estimated_tube_length_m(self) -> float | None:
        return None if self._qualified_size_m is None else max(self._qualified_size_m)

    def limit_source_pose_uncertainty(
        self, *, position_std_m: float, orientation_std_deg: float
    ) -> None:
        """Apply the consumer's receipt limits before committing source geometry."""
        if not all(math.isfinite(v) and v > 0 for v in (position_std_m, orientation_std_deg)):
            raise ValueError("source pose uncertainty limits must be positive and finite")
        self._source_position_std_limit_m = min(
            self._source_position_std_limit_m, position_std_m
        )
        self._source_orientation_std_limit_deg = min(
            self._source_orientation_std_limit_deg, orientation_std_deg
        )

    def observe(self, *, frame: object, state: object, T_base_tcp: np.ndarray) -> object:
        downstream_evidence = None
        if self._downstream is not None:
            downstream_evidence = self._downstream.observe(
                frame=frame, state=state, T_base_tcp=T_base_tcp
            )
        return self._observe_source(
            frame=frame,
            state=state,
            downstream_evidence=downstream_evidence,
        )

    def observe_burst(
        self,
        *,
        first_frame: object,
        read_frame: Callable[[], object],
        state: object,
        T_base_tcp: np.ndarray,
    ) -> tuple[object, object]:
        """Forward a camera-rate burst through the nested socket tracker.

        The socket layer returns both its final frame and already-produced
        downstream evidence.  Source SAM/F2G preparation is then performed
        exactly once on that final frame; calling ``observe`` again here would
        feed a duplicate timestamp to the socket tracker and clear its window.
        """

        frame = first_frame
        downstream_evidence = None
        if self._downstream is not None:
            downstream_burst = getattr(self._downstream, "observe_burst", None)
            if callable(downstream_burst):
                frame, downstream_evidence = downstream_burst(
                    first_frame=first_frame,
                    read_frame=read_frame,
                    state=state,
                    T_base_tcp=T_base_tcp,
                )
            else:
                downstream_evidence = self._downstream.observe(
                    frame=frame,
                    state=state,
                    T_base_tcp=T_base_tcp,
                )
        evidence = self._observe_source(
            frame=frame,
            state=state,
            downstream_evidence=downstream_evidence,
        )
        return frame, evidence

    def _observe_source(
        self,
        *,
        frame: object,
        state: object,
        downstream_evidence: object | None,
    ) -> object:
        from .test_tube_live_io import SceneEvidence, SceneObjectMask

        self._source_depth_only_loss = False
        self._source_rgb_identity_retained = False
        if str(getattr(frame, "serial")) != self._serial:
            raise TubePoseTrackingError("live camera serial differs from online bootstrap")
        K = np.asarray(getattr(getattr(frame, "intrinsics"), "matrix"), dtype=float)
        if self._expected_K is None:
            self._expected_K = K.copy()
        elif not np.allclose(K, self._expected_K, atol=1e-9, rtol=0.0):
            raise TubePoseTrackingError("camera intrinsics changed during online bootstrap")
        state_name = getattr(state, "value", str(state)).lower()
        if state_name not in self._SOURCE_STATE_NAMES:
            return downstream_evidence or SceneEvidence(tube=None, socket=None)
        rgb = _rgb(np.asarray(getattr(frame, "rgb")))
        timestamp = float(getattr(frame, "host_received_monotonic_s"))
        frame_id = f"d435-{self._serial}-{getattr(frame, 'sequence')}"
        # A transparent tube has too little stable texture for LK optical flow.
        # Re-run SAM on each pre-contact observation and let the seed builder's
        # mask-IoU/end-point window enforce identity and temporal consistency.
        try:
            tube = self._provider.segment(
                frame_id=frame_id,
                capture_timestamp_s=timestamp,
                rgb=rgb,
                prompt=self._tube_prompt,
            )
        except Exception as error:  # noqa: BLE001 - model boundary becomes HOLD
            reason = f"online_sam3_failed:{type(error).__name__}"
            self._reset_source_bootstrap(reason)
            self._last_hold_reasons = (reason,)
            return _replace_scene_tube(downstream_evidence, None)
        support = None
        if self._support_prompt:
            try:
                support = self._provider.segment(
                    frame_id=frame_id,
                    capture_timestamp_s=timestamp,
                    rgb=rgb,
                    prompt=self._support_prompt,
                )
            except Exception:  # noqa: BLE001 - qualified base-Z may be the fallback
                support = None
        if not self._sam_backend_matches(tube) or (
            support is not None and not self._sam_backend_matches(support)
        ):
            reason = "online_sam3_backend_identity_changed"
            self._reset_source_bootstrap(reason)
            self._last_hold_reasons = (reason,)
            return _replace_scene_tube(downstream_evidence, None)
        # This is deliberately evaluated from the current SAM response before
        # support-based axis signing.  A weak/occluded cup may make the current
        # metric axis unavailable even though the tube mask, PCA endpoints and
        # fixed session identity still match the last metric-qualified frame.
        # The boolean never creates a mask/pose or refreshes the reference.
        self._source_rgb_identity_retained = (
            self._rgb_observation_matches_qualified_source(
                tube=tube,
                image_shape=rgb.shape[:2],
            )
        )
        seed = self._seed_builder.update(
            tube=tube,
            support=support,
            image_shape=rgb.shape[:2],
            depth_m=np.asarray(getattr(frame, "depth_m")),
            K=K,
            T_base_camera=self._T_base_camera,
            camera_extrinsic_qualified=self._camera_extrinsic_qualified,
        )
        if seed is None:
            self._last_hold_reasons = self._seed_builder.last_reasons
            return _replace_scene_tube(downstream_evidence, None)
        previous_mask = None if self._last_track is None else self._last_track.mask
        self._seed_ends = ReviewedTubeEnds(
            open_uv=seed.open_uv,
            closed_uv=seed.closed_uv,
            evidence_id=seed.evidence_id,
            axis_sign_source=seed.axis_sign_source,
        )
        if self._qualifier is None:
            self._qualifier = TubePoseQualifier(self._seed_ends, self._qualifier_config)
        track = self._mask_tracker.seed(
            rgb=rgb,
            reviewed_mask=seed.mask,
            timestamp_s=timestamp,
            tracking_id=self._tracking_id,
        )
        consistency = 1.0 if previous_mask is None else _mask_iou(previous_mask, seed.mask)
        track = replace(
            track,
            mask_id=f"sam3:{tube.request_id}",
            consistency_iou=consistency,
            score_provenance="consecutive_sam3_mask_iou",
        )
        self._last_track = track
        if not track.accepted or track.mask is None or track.mask_id is None:
            reason = track.reasons[0] if track.reasons else "online_mask_tracking_failed"
            self._reset_source_bootstrap(reason)
            self._last_hold_reasons = track.reasons or (reason,)
            return _replace_scene_tube(downstream_evidence, None)
        self._sequence += 1
        request = PoseRequest(
            request_id=f"online-scene-depth-{self._sequence}",
            capture=PoseCapture(
                sequence=self._sequence,
                frame_id=frame_id,
                timestamp_s=timestamp,
                clock_domain="client_monotonic",
            ),
            rgb=rgb,
            depth_m=np.asarray(getattr(frame, "depth_m")),
            K=K,
            mask=track.mask,
            mask_source=track.mask_id,
            mask_source_score=track.consistency_iou,
        )
        depth = evaluate_foreground_depth(request, self._depth_config)
        if not depth.diagnostics.gate_passed:
            self._last_foreground_depth_m = None
            self._last_hold_reasons = depth.diagnostics.reject_reasons
            self._source_rgb_identity_retained = self._image_matches_qualified_source()
            self._source_depth_only_loss = self._source_rgb_identity_retained
            return _replace_scene_tube(downstream_evidence, None)
        self._last_foreground_depth_m = depth.sanitized_depth_m
        self._last_hold_reasons = ()
        return _replace_scene_tube(
            downstream_evidence,
            SceneObjectMask(
                object_id="tube",
                mask=track.mask,
                mask_id=track.mask_id,
                mask_iou=1.0 if track.consistency_iou is None else track.consistency_iou,
                tracking_id=track.tracking_id,
                occluded_by_gripper=False,
            ),
        )

    def qualify_pose(
        self, *, object_id: str, proposal: PoseObservation, mask: object, frame: object
    ) -> object:
        from .test_tube_live_io import QualifiedCameraPose

        if object_id != "tube":
            if self._downstream is None:
                raise TubePoseTrackingError(f"unsupported scene object: {object_id}")
            return self._downstream.qualify_pose(
                object_id=object_id, proposal=proposal, mask=mask, frame=frame
            )
        if (
            self._qualifier is None
            or self._seed_ends is None
            or self._last_track is None
            or self._last_track.seed_to_current is None
        ):
            raise TubePoseTrackingError("online tube seed is not qualified")
        if str(getattr(mask, "tracking_id")) != self._last_track.tracking_id:
            raise TubePoseTrackingError("online tube tracking identity changed")
        if self._qualified_tracking_id not in (None, self._last_track.tracking_id):
            raise TubePoseTrackingError("qualified tube tracking identity changed")
        if not proposal.accepted or proposal.T_camera_object is None or proposal.size_m is None:
            self._last_hold_reasons = tuple(proposal.reject_reasons) or (
                "online_f2g_proposal_rejected",
            )
            self._qualifier.reset()
            self._size_filter.reset("runtime_f2g_pose_sequence_interrupted")
            return None
        ends = self._seed_ends.warped(self._last_track.seed_to_current)
        result = self._qualifier.update(
            proposal=proposal,
            K=np.asarray(getattr(getattr(frame, "intrinsics"), "matrix")),
            current_reviewed_ends=ends,
            foreground_depth_m=self._last_foreground_depth_m,
        )
        if result.pose is not None and result.decision == "TRACKED":
            observed_size = np.sort(np.asarray(
                proposal.size_m if result.pose.size_m is None else result.pose.size_m,
                dtype=float,
            ))
            size = self._size_filter.update(observed_size)
            if size is None:
                self._last_hold_reasons = self._size_filter.last_reasons
                return None
            if self._qualified_size_m is not None:
                frozen = np.asarray(self._qualified_size_m)
                # A stable but substantially clipped/new cylinder is not the
                # original tube.  Check the current geometry, not just its median.
                if np.max(np.abs(observed_size - frozen) / frozen) > (
                    self._size_filter.config.max_relative_window_deviation
                ):
                    self._last_hold_reasons = ("runtime_tube_geometry_changed",)
                    self._qualifier.reset()
                    self._size_filter.reset("runtime_tube_geometry_changed")
                    return None
                size = self._qualified_size_m
            transform = np.asarray(result.pose.T_camera_tube).copy()
            # Canonical +Z points to the open end.  Replacing the observed
            # length must preserve that measured endpoint, not shift the
            # normalized grasp target by silently changing the metric origin.
            center_correction_m = 0.5 * (float(observed_size[-1]) - max(size))
            transform[:3, 3] += center_correction_m * transform[:3, 2]
            position_std_m = math.hypot(result.pose.position_std_m, center_correction_m)
            if (position_std_m > self._source_position_std_limit_m
                    or result.pose.orientation_std_deg > self._source_orientation_std_limit_deg):
                self._last_hold_reasons = ("runtime_tube_pose_uncertainty_exceeded",)
                self._qualifier.reset()
                self._size_filter.reset("runtime_tube_pose_uncertainty_exceeded")
                return None
            if self._last_qualified_transform is not None:
                previous = self._last_qualified_transform
                translation_jump = float(np.linalg.norm(transform[:3, 3] - previous[:3, 3]))
                axis_angle_deg = math.degrees(math.acos(float(np.clip(
                    np.dot(transform[:3, 2], previous[:3, 2]), -1.0, 1.0
                ))))
                if (translation_jump > self._qualifier_config.max_translation_jump_m
                        or axis_angle_deg > self._qualifier_config.max_orientation_jump_deg):
                    self._last_hold_reasons = ("qualified_tube_pose_continuity_lost",)
                    self._qualifier.reset()
                    self._size_filter.reset("qualified_tube_pose_continuity_lost")
                    return None
            if self._qualified_size_m is None:
                setter = getattr(self._downstream, "set_tube_length_m", None)
                if callable(setter):
                    setter(max(size))
                self._qualified_size_m = size
                self._qualified_tracking_id = self._last_track.tracking_id
            self._last_qualified_transform = transform.copy()
            current_mask = getattr(self._last_track, "mask", None)
            if current_mask is not None and isinstance(ends, ReviewedTubeEnds):
                self._qualified_image_mask = np.asarray(current_mask, dtype=bool).copy()
                self._qualified_image_ends = ends
            self._last_hold_reasons = ()
            return QualifiedCameraPose(
                T_camera_object=transform,
                position_std_m=position_std_m,
                orientation_std_deg=result.pose.orientation_std_deg,
                axis_sign_source=result.pose.axis_sign_source,
                cylindrical_roll_canonicalized=True,
                size_m=size,
            )
        self._last_hold_reasons = result.reasons
        if result.decision == "HOLD":
            self._qualifier.reset()
            self._size_filter.reset("runtime_f2g_pose_sequence_interrupted")
        # Warm-up/rejected estimates are ordinary HOLD conditions.  Returning
        # no pose lets the resident policy reobserve without promoting an
        # uncertain transform or aborting the process.
        return None

    def _image_matches_qualified_source(self) -> bool:
        track = self._last_track
        reference = self._qualified_image_ends
        current = self._seed_ends
        if (
            track is None
            or track.mask is None
            or track.tracking_id != self._qualified_tracking_id
            or self._qualified_image_mask is None
            or reference is None
            or current is None
        ):
            return False
        config = self._seed_builder.config
        if _mask_iou(self._qualified_image_mask, track.mask) < config.min_consecutive_mask_iou:
            return False
        # Compare with the last metric-qualified frame, not just the previous
        # RGB frame: gradual image drift must not refresh its own reference.
        return bool(all(
            np.linalg.norm(np.asarray(getattr(current, end)) - np.asarray(getattr(reference, end)))
            <= config.max_endpoint_jitter_px
            for end in ("open_uv", "closed_uv")
        ))

    def _rgb_observation_matches_qualified_source(
        self,
        *,
        tube: Sam3MaskObservation,
        image_shape: tuple[int, int],
    ) -> bool:
        """Match one current SAM mask to the frozen metric-qualified image identity.

        This check intentionally repeats the tube-only bootstrap gates that are
        independent of support/depth, then compares the unsigned PCA endpoints
        in both assignments.  Thus a support-sign failure can retain identity,
        while a low score, out-of-ROI mask, changed shape or endpoint drift
        cannot.
        """

        reference = self._qualified_image_ends
        reference_mask = self._qualified_image_mask
        if (
            self._qualified_tracking_id is None
            or self._qualified_tracking_id != self._tracking_id
            or reference is None
            or reference_mask is None
            or tube.score < self._seed_builder.config.min_tube_score
        ):
            return False
        try:
            mask = _binary_mask(tube.mask, image_shape, "current online tube mask")
        except ValueError:
            return False
        config = self._seed_builder.config
        if int(mask.sum()) < config.min_mask_pixels:
            return False
        if config.source_roi_uvxy is not None:
            u0, v0, u1, v1 = config.source_roi_uvxy
            inside = np.zeros(image_shape, dtype=bool)
            inside[v0:v1, u0:u1] = True
            if float(np.count_nonzero(mask & inside) / np.count_nonzero(mask)) < 0.90:
                return False
        if _mask_iou(reference_mask, mask) < config.min_consecutive_mask_iou:
            return False
        endpoints = self._seed_builder._unsigned_endpoints(mask)
        if endpoints is None:
            return False
        ref_open = np.asarray(reference.open_uv, dtype=float)
        ref_closed = np.asarray(reference.closed_uv, dtype=float)
        first, second = (np.asarray(value, dtype=float) for value in endpoints)
        return bool(any(
            np.linalg.norm(open_uv - ref_open) <= config.max_endpoint_jitter_px
            and np.linalg.norm(closed_uv - ref_closed) <= config.max_endpoint_jitter_px
            for open_uv, closed_uv in ((first, second), (second, first))
        ))

    def _sam_backend_matches(self, observation: Sam3MaskObservation) -> bool:
        for key in ("name", "implementation", "checkpoint"):
            expected = self._expected_sam_backend.get(key)
            if expected is not None and observation.backend.get(key) != expected:
                return False
        return True

    def _reset_source_bootstrap(self, reason: str) -> None:
        self._source_depth_only_loss = False
        self._source_rgb_identity_retained = False
        self._qualified_image_mask = None
        self._qualified_image_ends = None
        self._mask_tracker.reset()
        self._seed_builder.reset(reason)
        if self._qualifier is not None:
            self._qualifier.reset()
        self._qualifier = None
        self._seed_ends = None
        self._last_track = None
        self._last_foreground_depth_m = None
        self._size_filter.reset(reason)
        # Dropout invalidates pose evidence, not a physical object's dimensions.
        # Reacquisition still passes all gates and the frozen-identity checks.

    def held_tube_pose(self, **kwargs: object) -> object | None:
        if self._downstream is None:
            return None
        return self._downstream.held_tube_pose(**kwargs)

    def set_grasp_relation(
        self, T_tube_tcp_actual: np.ndarray, *, T_base_tube_at_contact: np.ndarray
    ) -> None:
        if self._downstream is None:
            raise TubePoseTrackingError("downstream scene interpreter is required")
        self._downstream.set_grasp_relation(
            T_tube_tcp_actual, T_base_tube_at_contact=T_base_tube_at_contact
        )

    def reset_support_history_for_release(self) -> None:
        if self._downstream is None:
            raise TubePoseTrackingError("downstream scene interpreter is required")
        self._downstream.reset_support_history_for_release()

    def socket_pose(self, **kwargs: object) -> object | None:
        if self._downstream is None:
            return None
        getter = getattr(self._downstream, "socket_pose", None)
        return None if not callable(getter) else getter(**kwargs)

    def observe_tactile(self, **kwargs: object) -> None:
        if self._downstream is None:
            return
        observer = getattr(self._downstream, "observe_tactile", None)
        if callable(observer):
            observer(**kwargs)

    def close(self) -> None:
        if self._downstream is not None:
            self._downstream.close()


def _factory(spec: object, config: object, output_dir: object) -> object | None:
    if spec is None:
        return None
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError("downstream_factory must use module:function syntax")
    module, function = spec.split(":", 1)
    value = getattr(importlib.import_module(module), function, None)
    if not callable(value):
        raise ValueError(f"downstream factory is not callable: {spec}")
    return value(config=config or {}, output_dir=output_dir)


def create_online_test_tube_scene_interpreter(
    *, config: Mapping[str, object], output_dir: object
) -> OnlineSourceTubeSceneInterpreter:
    """Factory loadable by ``test_tube_live_io.create_live_io``."""

    if not isinstance(config, Mapping):
        raise ValueError("online scene config must be an object")
    endpoint = config.get("sam3_service_url")
    serial = config.get("camera_serial")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("sam3_service_url must be non-empty")
    if not isinstance(serial, str) or not serial:
        raise ValueError("camera_serial must be non-empty")

    def section(name: str) -> dict[str, object]:
        value = config.get(name, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be an object")
        return dict(value)

    client = Sam3MaskClient(
        endpoint,
        timeout_s=float(config.get("sam3_timeout_s", 4.0)),
        auth_token=(
            None if config.get("sam3_auth_token") is None else str(config["sam3_auth_token"])
        ),
    )
    health = client.health()
    backend = health.get("backend")
    if (
        health.get("ready") is not True
        or not isinstance(backend, Mapping)
        or backend.get("name") != "sam3"
    ):
        raise TubePoseTrackingError("SAM 3 bootstrap service is not ready")
    downstream = _factory(
        config.get("downstream_factory"), config.get("downstream_config", {}), output_dir
    )
    bootstrap = section("tube_bootstrap")
    for name in ("source_roi_uvxy", "support_roi_uvxy", "support_anchor_uv"):
        if name in bootstrap and bootstrap[name] is not None:
            bootstrap[name] = tuple(bootstrap[name])
    raw_transform = config.get("T_base_camera")
    return OnlineSourceTubeSceneInterpreter(
        mask_provider=client,
        expected_camera_serial=serial,
        tube_prompt=str(config.get("tube_prompt", "transparent test tube")),
        source_support_prompt=str(config.get("source_support_prompt", "white paper cup")),
        T_base_camera=None if raw_transform is None else np.asarray(raw_transform, dtype=float),
        camera_extrinsic_qualified=config.get("camera_extrinsic_qualified") is True,
        bootstrap_config=OnlineTubeBootstrapConfig(**bootstrap),
        mask_config=StaticSourceMaskTrackerConfig(**section("mask_tracker")),
        qualifier_config=TubePoseQualifierConfig(**section("pose_qualifier")),
        depth_config=ForegroundDepthGateConfig(**section("foreground_depth")),
        size_config=OnlineTubeSizeConfig(**section("size_qualifier")),
        tracking_id=str(config.get("tracking_id", "online-test-tube")),
        expected_sam_backend=dict(backend),
        downstream=downstream,
    )


__all__ = [
    "OnlineMaskProvider",
    "OnlineSourceTubeSceneInterpreter",
    "OnlineTubeBootstrapConfig",
    "OnlineTubeSeed",
    "OnlineTubeSeedBuilder",
    "OnlineTubeSizeConfig",
    "OnlineTubeSizeFilter",
    "create_online_test_tube_scene_interpreter",
]
