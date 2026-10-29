"""Fail-closed front-RGB axis tracking for the test-tube pivot stage.

The transparent tube does not provide trustworthy D435 depth.  This module
therefore tracks only a reviewed image-plane axis.  It accepts ordinary line
segments (for example, paired transparent-tube edges), checks them against a
region of interest and the previous accepted observation, and refuses to
update on a missing, implausible, or ambiguous candidate set.

The tracker needs an explicit seed.  This prevents a background edge from
silently becoming the tube when the pivot stage starts.  It contains no
camera, ROS, gripper, or robot command code.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from demo2gripper.test_tube_visual import (
    TubeAxisObservation,
    axis_angle_error_deg,
    normalize_axis_angle_deg,
)


@dataclass(frozen=True)
class ImageLineCandidate:
    """One finite image-line segment in ``(u, v)`` pixel coordinates."""

    start_uv: tuple[float, float]
    end_uv: tuple[float, float]
    edge_support: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray((*self.start_uv, *self.end_uv, self.edge_support), dtype=float)
        if values.shape != (5,) or not np.isfinite(values).all():
            raise ValueError("line coordinates and edge_support must be finite")
        if not 0.0 <= self.edge_support <= 1.0:
            raise ValueError("edge_support must lie in [0, 1]")
        if self.length_px <= 0.0:
            raise ValueError("line endpoints must be distinct")

    @property
    def length_px(self) -> float:
        return float(np.linalg.norm(np.asarray(self.end_uv) - np.asarray(self.start_uv)))

    @property
    def midpoint_uv(self) -> tuple[float, float]:
        midpoint = 0.5 * (np.asarray(self.start_uv) + np.asarray(self.end_uv))
        return float(midpoint[0]), float(midpoint[1])

    @property
    def axis_angle_deg(self) -> float:
        delta = np.asarray(self.end_uv) - np.asarray(self.start_uv)
        return normalize_axis_angle_deg(math.degrees(math.atan2(delta[1], delta[0])))


@dataclass(frozen=True)
class TubeAxisTrackerConfig:
    """Conservative continuity and ambiguity limits for image-axis tracking."""

    min_segment_length_px: float = 28.0
    nominal_segment_length_px: float = 120.0
    max_angle_rate_deg_s: float = 120.0
    angle_slack_deg: float = 2.0
    max_perpendicular_speed_px_s: float = 500.0
    perpendicular_slack_px: float = 10.0
    max_midpoint_speed_px_s: float = 1400.0
    midpoint_slack_px: float = 20.0
    equivalent_angle_deg: float = 4.0
    equivalent_line_distance_px: float = 32.0
    min_tracking_score: float = 0.42
    min_ambiguity_margin: float = 0.035
    max_tracking_gap_s: float = 0.25

    def __post_init__(self) -> None:
        positive = {
            "min_segment_length_px": self.min_segment_length_px,
            "nominal_segment_length_px": self.nominal_segment_length_px,
            "max_angle_rate_deg_s": self.max_angle_rate_deg_s,
            "angle_slack_deg": self.angle_slack_deg,
            "max_perpendicular_speed_px_s": self.max_perpendicular_speed_px_s,
            "perpendicular_slack_px": self.perpendicular_slack_px,
            "max_midpoint_speed_px_s": self.max_midpoint_speed_px_s,
            "midpoint_slack_px": self.midpoint_slack_px,
            "equivalent_angle_deg": self.equivalent_angle_deg,
            "equivalent_line_distance_px": self.equivalent_line_distance_px,
            "min_ambiguity_margin": self.min_ambiguity_margin,
            "max_tracking_gap_s": self.max_tracking_gap_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < self.min_tracking_score < 1.0:
            raise ValueError("min_tracking_score must lie in (0, 1)")


@dataclass(frozen=True)
class TubeAxisTrackResult:
    """Result of one tracker update.

    ``decision`` is one of ``TRACKED``, ``HOLD_AND_REOBSERVE``, or
    ``RESEED_REQUIRED``.  Only ``TRACKED`` contains a valid observation.
    """

    decision: str
    observation: TubeAxisObservation
    selected_line: ImageLineCandidate | None
    considered_candidate_count: int
    equivalent_candidate_count: int
    tracking_score: float | None
    ambiguity_margin: float | None
    reason: str


@dataclass(frozen=True)
class _ScoredCandidate:
    line: ImageLineCandidate
    score: float
    angle_error_deg: float
    perpendicular_distance_px: float
    midpoint_distance_px: float


def _invalid_observation() -> TubeAxisObservation:
    return TubeAxisObservation(None, None, 0.0, math.inf, 0.0, 0, False)


def _point_to_line_distance_px(
    point_uv: np.ndarray,
    line: ImageLineCandidate,
) -> float:
    start = np.asarray(line.start_uv, dtype=float)
    delta = np.asarray(line.end_uv, dtype=float) - start
    direction = delta / np.linalg.norm(delta)
    relative = np.asarray(point_uv, dtype=float) - start
    return float(abs(direction[0] * relative[1] - direction[1] * relative[0]))


def _inside_roi(
    point_uv: tuple[float, float],
    roi_uvxy: tuple[float, float, float, float] | None,
) -> bool:
    if roi_uvxy is None:
        return True
    u0, v0, u1, v1 = (float(value) for value in roi_uvxy)
    if not all(math.isfinite(value) for value in (u0, v0, u1, v1)) or u1 <= u0 or v1 <= v0:
        raise ValueError("roi_uvxy must be finite with positive width and height")
    u, v = point_uv
    return u0 <= u < u1 and v0 <= v < v1


def _line_observation(line: ImageLineCandidate) -> TubeAxisObservation:
    return TubeAxisObservation(
        axis_angle_deg=line.axis_angle_deg,
        centroid_uv=line.midpoint_uv,
        span_px=line.length_px,
        orthogonal_rms_px=0.0,
        anisotropy=1.0,
        point_count=max(2, round(line.length_px)),
        valid=True,
    )


def _same_axis_hypothesis(
    first: ImageLineCandidate,
    second: ImageLineCandidate,
    config: TubeAxisTrackerConfig,
) -> bool:
    if abs(axis_angle_error_deg(first.axis_angle_deg, second.axis_angle_deg)) > (
        config.equivalent_angle_deg
    ):
        return False
    first_to_second = _point_to_line_distance_px(np.asarray(first.midpoint_uv), second)
    second_to_first = _point_to_line_distance_px(np.asarray(second.midpoint_uv), first)
    return max(first_to_second, second_to_first) <= config.equivalent_line_distance_px


class TubeAxisCandidateTracker:
    """Track a reviewed tube edge while rejecting jumps and ambiguity."""

    def __init__(self, config: TubeAxisTrackerConfig | None = None) -> None:
        self.config = config or TubeAxisTrackerConfig()
        self._last_input_timestamp_s: float | None = None
        self._last_accepted_timestamp_s: float | None = None
        self._last_line: ImageLineCandidate | None = None

    @property
    def seeded(self) -> bool:
        return self._last_line is not None

    def reset(self) -> None:
        self._last_input_timestamp_s = None
        self._last_accepted_timestamp_s = None
        self._last_line = None

    def seed(self, *, timestamp_s: float, line: ImageLineCandidate) -> TubeAxisObservation:
        """Install a reviewed initial line before automatic tracking starts."""

        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if line.length_px < self.config.min_segment_length_px:
            raise ValueError("seed line is shorter than min_segment_length_px")
        self._last_input_timestamp_s = timestamp
        self._last_accepted_timestamp_s = timestamp
        self._last_line = line
        return _line_observation(line)

    def update(
        self,
        *,
        timestamp_s: float,
        candidates: Sequence[ImageLineCandidate],
        roi_uvxy: tuple[float, float, float, float] | None = None,
    ) -> TubeAxisTrackResult:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if self._last_input_timestamp_s is not None and timestamp <= self._last_input_timestamp_s:
            raise ValueError("timestamps must be strictly increasing")
        self._last_input_timestamp_s = timestamp

        if self._last_line is None or self._last_accepted_timestamp_s is None:
            return self._failure(
                "RESEED_REQUIRED",
                len(candidates),
                "tracker has no reviewed initial tube-axis seed",
            )

        elapsed = timestamp - self._last_accepted_timestamp_s
        if elapsed > self.config.max_tracking_gap_s:
            self._last_line = None
            self._last_accepted_timestamp_s = None
            return self._failure(
                "RESEED_REQUIRED",
                len(candidates),
                "tracking gap exceeded; automatic reacquisition is disabled",
            )

        scored = self._score_candidates(candidates, elapsed=elapsed, roi_uvxy=roi_uvxy)
        if not scored:
            return self._failure(
                "HOLD_AND_REOBSERVE",
                0,
                "no candidate satisfies ROI, length, and temporal continuity limits",
            )

        best = scored[0]
        equivalent_count = sum(
            _same_axis_hypothesis(best.line, item.line, self.config) for item in scored
        )
        runner_up = next(
            (
                item
                for item in scored[1:]
                if not _same_axis_hypothesis(best.line, item.line, self.config)
            ),
            None,
        )
        margin = best.score - runner_up.score if runner_up is not None else 1.0
        if best.score < self.config.min_tracking_score:
            return self._failure(
                "HOLD_AND_REOBSERVE",
                len(scored),
                "best candidate does not meet the minimum tracking score",
                best=best,
                equivalent_count=equivalent_count,
                margin=margin,
            )
        if margin < self.config.min_ambiguity_margin:
            return self._failure(
                "HOLD_AND_REOBSERVE",
                len(scored),
                "two distinct axis hypotheses have indistinguishable scores",
                best=best,
                equivalent_count=equivalent_count,
                margin=margin,
            )

        self._last_line = best.line
        self._last_accepted_timestamp_s = timestamp
        return TubeAxisTrackResult(
            decision="TRACKED",
            observation=_line_observation(best.line),
            selected_line=best.line,
            considered_candidate_count=len(scored),
            equivalent_candidate_count=equivalent_count,
            tracking_score=best.score,
            ambiguity_margin=margin,
            reason="unique temporally continuous tube-axis hypothesis",
        )

    def _score_candidates(
        self,
        candidates: Sequence[ImageLineCandidate],
        *,
        elapsed: float,
        roi_uvxy: tuple[float, float, float, float] | None,
    ) -> list[_ScoredCandidate]:
        assert self._last_line is not None
        angle_limit = self.config.max_angle_rate_deg_s * elapsed + self.config.angle_slack_deg
        perpendicular_limit = (
            self.config.max_perpendicular_speed_px_s * elapsed + self.config.perpendicular_slack_px
        )
        midpoint_limit = (
            self.config.max_midpoint_speed_px_s * elapsed + self.config.midpoint_slack_px
        )
        previous_midpoint = np.asarray(self._last_line.midpoint_uv)
        accepted: list[_ScoredCandidate] = []
        for line in candidates:
            if line.length_px < self.config.min_segment_length_px:
                continue
            if not _inside_roi(line.midpoint_uv, roi_uvxy):
                continue
            angle_error = abs(
                axis_angle_error_deg(line.axis_angle_deg, self._last_line.axis_angle_deg)
            )
            perpendicular = _point_to_line_distance_px(
                np.asarray(line.midpoint_uv), self._last_line
            )
            midpoint = float(np.linalg.norm(np.asarray(line.midpoint_uv) - previous_midpoint))
            if (
                angle_error > angle_limit
                or perpendicular > perpendicular_limit
                or midpoint > midpoint_limit
            ):
                continue
            score = (
                0.45 * (1.0 - angle_error / angle_limit)
                + 0.20 * (1.0 - perpendicular / perpendicular_limit)
                + 0.10 * (1.0 - midpoint / midpoint_limit)
                + 0.20 * min(line.length_px / self.config.nominal_segment_length_px, 1.0)
                + 0.05 * line.edge_support
            )
            accepted.append(_ScoredCandidate(line, score, angle_error, perpendicular, midpoint))
        accepted.sort(key=lambda item: item.score, reverse=True)
        return accepted

    @staticmethod
    def _failure(
        decision: str,
        candidate_count: int,
        reason: str,
        *,
        best: _ScoredCandidate | None = None,
        equivalent_count: int = 0,
        margin: float | None = None,
    ) -> TubeAxisTrackResult:
        return TubeAxisTrackResult(
            decision=decision,
            observation=_invalid_observation(),
            selected_line=best.line if best is not None else None,
            considered_candidate_count=candidate_count,
            equivalent_candidate_count=equivalent_count,
            tracking_score=best.score if best is not None else None,
            ambiguity_margin=margin,
            reason=reason,
        )


def extract_hough_line_candidates(
    image_bgr: np.ndarray,
    *,
    roi_uvxy: tuple[int, int, int, int],
    canny_low: int = 35,
    canny_high: int = 100,
    hough_threshold: int = 22,
    min_line_length_px: int = 28,
    max_line_gap_px: int = 14,
) -> list[ImageLineCandidate]:
    """Extract CPU Hough segments from a bounded front-RGB ROI.

    OpenCV is imported lazily because it belongs to the repository's
    ``visualization`` extra.  This function finds candidates only; temporal
    association and ambiguity rejection remain the tracker's responsibility.
    """

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise RuntimeError("OpenCV is required for Hough line extraction") from error

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_bgr must be a uint8 HxWx3 array")
    height, width = image.shape[:2]
    u0, v0, u1, v1 = roi_uvxy
    if not (0 <= u0 < u1 <= width and 0 <= v0 < v1 <= height):
        raise ValueError("roi_uvxy must lie inside the image")
    if not 0 < canny_low < canny_high:
        raise ValueError("Canny thresholds must satisfy 0 < low < high")
    if hough_threshold <= 0 or min_line_length_px <= 0 or max_line_gap_px < 0:
        raise ValueError("Hough parameters are invalid")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0.0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    roi_mask = np.zeros_like(edges)
    roi_mask[v0:v1, u0:u1] = 255
    edges = cv2.bitwise_and(edges, roi_mask)
    raw_lines = cv2.HoughLinesP(
        edges,
        1.0,
        np.pi / 360.0,
        threshold=hough_threshold,
        minLineLength=min_line_length_px,
        maxLineGap=max_line_gap_px,
    )
    if raw_lines is None:
        return []

    result: list[ImageLineCandidate] = []
    # OpenCV wheels return either (N, 1, 4) or (N, 4) here.
    for x0, y0, x1, y1 in np.asarray(raw_lines).reshape(-1, 4):
        result.append(
            ImageLineCandidate(
                (float(x0), float(y0)),
                (float(x1), float(y1)),
                edge_support=1.0,
            )
        )
    return result


__all__ = [
    "ImageLineCandidate",
    "TubeAxisCandidateTracker",
    "TubeAxisTrackResult",
    "TubeAxisTrackerConfig",
    "extract_hough_line_candidates",
]
