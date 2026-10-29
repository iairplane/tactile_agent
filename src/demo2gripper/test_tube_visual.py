"""Image-axis observations and a visual gate for the test-tube pivot stage.

The GF220 sensors supervise contact and slip during the gravity pivot.  They do
not, before task-specific calibration, provide an absolute tube angle.  The
fixed front camera therefore supplies the pivot progress variable.  This
module deliberately works in the image plane: a target learned from the same
fixed camera is compared modulo 180 degrees, so no unsupported conversion to a
world-frame 45 degree angle is made.

This file contains no camera or robot driver code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def normalize_axis_angle_deg(value: float) -> float:
    """Return an undirected line angle in ``[0, 180)`` degrees."""

    angle = float(value)
    if not math.isfinite(angle):
        raise ValueError("axis angle must be finite")
    return angle % 180.0


def axis_angle_error_deg(observed: float, target: float) -> float:
    """Signed shortest error between two undirected image axes."""

    observed_n = normalize_axis_angle_deg(observed)
    target_n = normalize_axis_angle_deg(target)
    return (observed_n - target_n + 90.0) % 180.0 - 90.0


@dataclass(frozen=True)
class TubeAxisObservation:
    """PCA estimate of a tube-like image region.

    ``axis_angle_deg`` uses image coordinates: zero points right and positive
    angles rotate toward image +Y (clockwise on a displayed image).  The axis
    is undirected, hence angles differing by 180 degrees are equivalent.
    """

    axis_angle_deg: float | None
    centroid_uv: tuple[float, float] | None
    span_px: float
    orthogonal_rms_px: float
    anisotropy: float
    point_count: int
    valid: bool


def estimate_tube_axis_from_points(
    points_uv: np.ndarray,
    *,
    min_points: int = 20,
    min_span_px: float = 30.0,
    min_anisotropy: float = 0.65,
) -> TubeAxisObservation:
    """Estimate an image axis from mask or edge pixels.

    The function accepts an ``(N, 2)`` array ordered as ``(u, v)``.  It is
    suitable for a reviewed segmentation mask or paired tube edges.  It is not
    itself a transparent-object segmenter.
    """

    points = np.asarray(points_uv, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError("points_uv must have shape (N, 2)")
    if not np.isfinite(points).all():
        raise ValueError("points_uv must be finite")
    if isinstance(min_points, bool) or not isinstance(min_points, int) or min_points < 2:
        raise ValueError("min_points must be an integer >= 2")
    if not math.isfinite(min_span_px) or min_span_px <= 0:
        raise ValueError("min_span_px must be positive")
    if not math.isfinite(min_anisotropy) or not 0.0 <= min_anisotropy < 1.0:
        raise ValueError("min_anisotropy must be in [0, 1)")

    count = int(points.shape[0])
    if count < 2:
        return TubeAxisObservation(None, None, 0.0, math.inf, 0.0, count, False)

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / max(count - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major = max(float(eigenvalues[order[0]]), 0.0)
    minor = max(float(eigenvalues[order[1]]), 0.0)
    direction = eigenvectors[:, order[0]]
    angle = normalize_axis_angle_deg(math.degrees(math.atan2(direction[1], direction[0])))
    # Four standard deviations captures about 95% of a line-like region and is
    # stable for either a filled mask or its two visible edges.
    span = 4.0 * math.sqrt(major)
    orthogonal_rms = math.sqrt(minor)
    total = major + minor
    anisotropy = (major - minor) / total if total > 1e-12 else 0.0
    valid = count >= min_points and span >= min_span_px and anisotropy >= min_anisotropy
    return TubeAxisObservation(
        angle if valid else None,
        (float(centroid[0]), float(centroid[1])),
        span,
        orthogonal_rms,
        anisotropy,
        count,
        valid,
    )


def estimate_tube_axis_from_mask(
    mask: np.ndarray,
    *,
    min_points: int = 20,
    min_span_px: float = 30.0,
    min_anisotropy: float = 0.65,
) -> TubeAxisObservation:
    """Estimate the tube axis from a two-dimensional boolean mask."""

    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    rows, columns = np.nonzero(values.astype(bool))
    points = np.column_stack((columns, rows))
    return estimate_tube_axis_from_points(
        points,
        min_points=min_points,
        min_span_px=min_span_px,
        min_anisotropy=min_anisotropy,
    )


@dataclass(frozen=True)
class VisualAngleGateConfig:
    target_image_axis_deg: float
    tolerance_deg: float = 5.0
    min_consecutive_frames: int = 4
    max_rate_deg_s: float = 120.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_image_axis_deg",
            normalize_axis_angle_deg(self.target_image_axis_deg),
        )
        if not math.isfinite(self.tolerance_deg) or not 0 < self.tolerance_deg < 45:
            raise ValueError("tolerance_deg must be in (0, 45)")
        if (
            isinstance(self.min_consecutive_frames, bool)
            or not isinstance(self.min_consecutive_frames, int)
            or self.min_consecutive_frames < 1
        ):
            raise ValueError("min_consecutive_frames must be a positive integer")
        if not math.isfinite(self.max_rate_deg_s) or self.max_rate_deg_s <= 0:
            raise ValueError("max_rate_deg_s must be positive")


@dataclass(frozen=True)
class VisualAngleGateResult:
    decision: str
    angle_error_deg: float | None
    consecutive_in_band: int
    accepted: bool
    reason: str


class VisualAngleGate:
    """Debounce the front-camera pivot target before requesting reclamping."""

    def __init__(self, config: VisualAngleGateConfig) -> None:
        self.config = config
        self._last_timestamp_s: float | None = None
        self._last_angle_deg: float | None = None
        self._consecutive = 0

    def reset(self) -> None:
        self._last_timestamp_s = None
        self._last_angle_deg = None
        self._consecutive = 0

    def update(
        self,
        *,
        timestamp_s: float,
        observation: TubeAxisObservation,
    ) -> VisualAngleGateResult:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if self._last_timestamp_s is not None and timestamp <= self._last_timestamp_s:
            raise ValueError("timestamps must be strictly increasing")

        if not observation.valid or observation.axis_angle_deg is None:
            self._last_timestamp_s = timestamp
            self._last_angle_deg = None
            self._consecutive = 0
            return VisualAngleGateResult(
                "HOLD_AND_REOBSERVE",
                None,
                0,
                False,
                "tube axis is not observable in the front image",
            )

        angle = normalize_axis_angle_deg(observation.axis_angle_deg)
        if self._last_timestamp_s is not None and self._last_angle_deg is not None:
            elapsed = timestamp - self._last_timestamp_s
            rate = abs(axis_angle_error_deg(angle, self._last_angle_deg)) / elapsed
            if rate > self.config.max_rate_deg_s:
                self._last_timestamp_s = timestamp
                self._last_angle_deg = None
                self._consecutive = 0
                return VisualAngleGateResult(
                    "HOLD_AND_REOBSERVE",
                    None,
                    0,
                    False,
                    "image-axis jump exceeds the configured tracking rate",
                )

        error = axis_angle_error_deg(angle, self.config.target_image_axis_deg)
        self._last_timestamp_s = timestamp
        self._last_angle_deg = angle
        if abs(error) <= self.config.tolerance_deg:
            self._consecutive += 1
        else:
            self._consecutive = 0

        ready = self._consecutive >= self.config.min_consecutive_frames
        return VisualAngleGateResult(
            "RECLAMP" if ready else "CONTINUE_BOUNDED_PIVOT",
            error,
            self._consecutive,
            ready,
            (
                "front-camera tube axis is stably inside the learned target band"
                if ready
                else "tube axis has not yet remained inside the target band"
            ),
        )


__all__ = [
    "TubeAxisObservation",
    "VisualAngleGate",
    "VisualAngleGateConfig",
    "VisualAngleGateResult",
    "axis_angle_error_deg",
    "estimate_tube_axis_from_mask",
    "estimate_tube_axis_from_points",
    "normalize_axis_angle_deg",
]
