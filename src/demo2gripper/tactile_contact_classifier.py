"""Image-first, dual-finger classification of a locally contacted cup rim.

This module deliberately answers a narrower question than
``tactile_rim_probe``: given fresh ViTai ``WARPED_IMG`` and ``DIFF_IMG``
windows, what *local contact pattern* is visible on the two finger pads?
It does not identify which physical cup was touched and it does not command a
gripper or robot.

The two contacts have different jobs: one finger touches the cup rim from the
inside and the other from the outside.  Success therefore means that both
fingers match their independently calibrated role templates; it does not mean
that their images, areas, or loads are equal.  ``MULTI_RIM`` is emitted only
when a role's observed band count reaches its calibrated multi-rim threshold.
This still is not, by itself, proof that two cups were grasped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy import ndimage

from .tactile_rim_probe import RimContactRole, TimedArrayWindow


class TactileRimClass(StrEnum):
    NO_CONTACT = "no_contact"
    INNER_OUTER_RIM_MATCH = "inner_outer_rim_match"
    MULTI_RIM = "multi_rim"
    ROLE_MISMATCH = "role_mismatch"
    UNKNOWN = "unknown"


# Public alias kept local to this classifier while sharing the same role type
# as the downstream probe/gate state machine.
FingerRimRole = RimContactRole


@dataclass(frozen=True)
class ForceContactProfile:
    """Optional, calibrated force veto/corroboration for one finger."""

    inward_normal_sensor: np.ndarray
    no_contact_max_n: float
    contact_min_n: float
    overload_n: float
    max_load_range_n: float

    def __post_init__(self) -> None:
        normal = _unit_vector(self.inward_normal_sensor, 3, "inward_normal_sensor")
        normal.setflags(write=False)
        object.__setattr__(self, "inward_normal_sensor", normal)
        values = (
            float(self.no_contact_max_n),
            float(self.contact_min_n),
            float(self.overload_n),
            float(self.max_load_range_n),
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("force thresholds must be finite and non-negative")
        if not values[0] < values[1] < values[2]:
            raise ValueError(
                "force thresholds must satisfy no_contact_max < contact_min < overload"
            )
        if values[3] <= 0.0:
            raise ValueError("max_load_range_n must be positive")


@dataclass(frozen=True)
class FingerRimClassifierProfile:
    """Sensor-specific INNER or OUTER template in the SDK's warped pad frame."""

    calibration_id: str
    role: FingerRimRole
    unloaded_warped: np.ndarray
    rim_roi_mask: np.ndarray
    target_center_uv_px: np.ndarray
    target_tangent_uv: np.ndarray
    diff_neutral_value: float
    diff_threshold: float
    warped_delta_threshold: float
    min_contact_area_px: int
    max_contact_area_px: int
    min_contact_frame_fraction: float
    min_warped_support_fraction: float
    min_roi_fraction: float
    min_band_area_px: int
    expected_band_count: int
    multi_rim_min_band_count: int
    min_band_elongation: float
    max_tangent_error_deg: float
    min_distinct_band_separation_px: float
    max_center_error_px: float
    max_center_jitter_px: float
    force: ForceContactProfile | None = None

    def __post_init__(self) -> None:
        if not self.calibration_id:
            raise ValueError("calibration_id must be non-empty")
        try:
            role = FingerRimRole(self.role)
        except ValueError as exc:
            raise ValueError(
                "role must be FingerRimRole.INNER or FingerRimRole.OUTER"
            ) from exc
        object.__setattr__(self, "role", role)
        baseline = np.array(self.unloaded_warped, copy=True)
        if (
            baseline.ndim != 3
            or baseline.shape[-1] != 3
            or not np.issubdtype(baseline.dtype, np.number)
            or not np.isfinite(baseline).all()
        ):
            raise ValueError("unloaded_warped must be a finite HxWx3 numeric image")
        roi = np.array(self.rim_roi_mask, dtype=bool, copy=True)
        if roi.shape != baseline.shape[:2] or not roi.any():
            raise ValueError("rim_roi_mask must be a non-empty mask matching the image")
        center = _finite_vector(self.target_center_uv_px, 2, "target_center_uv_px")
        height, width = roi.shape
        if not (0.0 <= center[0] < width and 0.0 <= center[1] < height):
            raise ValueError("target_center_uv_px must lie inside the image")
        tangent = _unit_vector(self.target_tangent_uv, 2, "target_tangent_uv")
        for value in (baseline, roi, center, tangent):
            value.setflags(write=False)
        object.__setattr__(self, "unloaded_warped", baseline)
        object.__setattr__(self, "rim_roi_mask", roi)
        object.__setattr__(self, "target_center_uv_px", center)
        object.__setattr__(self, "target_tangent_uv", tangent)

        neutral = float(self.diff_neutral_value)
        if not math.isfinite(neutral) or not 0.0 <= neutral <= 255.0:
            raise ValueError("diff_neutral_value must lie in [0, 255]")

        for name in (
            "diff_threshold",
            "warped_delta_threshold",
            "min_band_elongation",
            "max_tangent_error_deg",
            "min_distinct_band_separation_px",
            "max_center_error_px",
            "max_center_jitter_px",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_tangent_error_deg > 90.0:
            raise ValueError("max_tangent_error_deg cannot exceed 90 degrees")
        if self.min_contact_area_px <= 0 or self.min_band_area_px <= 0:
            raise ValueError("contact and band areas must be positive")
        if self.max_contact_area_px < self.min_contact_area_px:
            raise ValueError("max_contact_area_px must be >= min_contact_area_px")
        if self.expected_band_count < 1:
            raise ValueError("expected_band_count must be positive")
        if self.multi_rim_min_band_count <= self.expected_band_count:
            raise ValueError(
                "multi_rim_min_band_count must exceed expected_band_count"
            )
        for name in (
            "min_contact_frame_fraction",
            "min_warped_support_fraction",
            "min_roi_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")


@dataclass(frozen=True)
class DualRimClassifierConfig:
    profile_id: str
    left: FingerRimClassifierProfile
    right: FingerRimClassifierProfile
    min_unique_image_samples: int = 4
    min_unique_force_samples: int = 3
    max_age_s: float = 0.15
    max_pair_skew_s: float = 0.08

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        if self.left.rim_roi_mask.shape != self.right.rim_roi_mask.shape:
            raise ValueError("left and right warped image shapes must match")
        if {self.left.role, self.right.role} != {
            FingerRimRole.INNER,
            FingerRimRole.OUTER,
        }:
            raise ValueError("profiles must assign exactly one INNER and one OUTER role")
        if (self.left.force is None) != (self.right.force is None):
            raise ValueError("FORCE6D must be configured for both fingers or neither")
        if self.min_unique_image_samples < 2 or self.min_unique_force_samples < 1:
            raise ValueError("minimum unique sample counts are invalid")
        for name in ("max_age_s", "max_pair_skew_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
@dataclass(frozen=True)
class FingerRimClassifierWindow:
    calibration_id: str
    warped: TimedArrayWindow
    diff: TimedArrayWindow
    force6d: TimedArrayWindow | None = None

    def __post_init__(self) -> None:
        if not self.calibration_id:
            raise ValueError("calibration_id must be non-empty")


@dataclass(frozen=True)
class FingerRimEvidence:
    expected_role: FingerRimRole | None
    expected_band_count: int | None
    valid: bool
    contact: bool
    resolved: bool
    band_count: int
    patch_area_px: int
    roi_fraction: float
    warped_support_fraction: float
    center_uv_px: np.ndarray | None
    center_error_px: float | None
    tangent_error_deg: float | None
    elongation: float | None
    center_jitter_px: float | None
    contact_frame_fraction: float
    normal_load_n: float | None
    normal_load_peak_n: float | None
    force_state: str | None
    role_matched: bool


@dataclass(frozen=True)
class TactileRimClassification:
    label: TactileRimClass
    score: float
    left: FingerRimEvidence
    right: FingerRimEvidence
    reasons: tuple[str, ...]
    force_used: bool
    evidence_key: tuple[str | float, ...] | None


@dataclass(frozen=True)
class _FreshWindow:
    values: np.ndarray
    sdk_time_s: np.ndarray
    capture_time_s: np.ndarray

    @property
    def endpoint(self) -> tuple[float, float]:
        return float(self.sdk_time_s[-1]), float(self.capture_time_s[-1])


def _finite_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _unit_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    result = _finite_vector(value, size, name)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return result / norm


def _fresh_window(
    window: TimedArrayWindow,
    *,
    minimum_samples: int,
    now_s: float,
    max_age_s: float,
) -> tuple[_FreshWindow | None, str | None]:
    indices = np.flatnonzero(window.valid)
    if indices.size == 0:
        return None, "no valid samples"
    if np.any(np.diff(window.sdk_time_s[indices]) < 0.0):
        return None, "SDK timestamps moved backwards"
    if np.any(np.diff(window.capture_monotonic_s[indices]) < 0.0):
        return None, "host timestamps moved backwards"
    finite = np.asarray(
        [np.isfinite(np.asarray(window.values[index], dtype=float)).all() for index in indices]
    )
    indices = indices[finite]
    if indices.size == 0:
        return None, "no finite samples"
    _, first = np.unique(window.sdk_time_s[indices], return_index=True)
    indices = indices[np.sort(first)]
    captures = window.capture_monotonic_s[indices]
    fresh = (captures >= now_s - max_age_s) & (captures <= now_s + 1e-6)
    indices = indices[fresh]
    if indices.size < minimum_samples:
        return None, (
            f"only {indices.size} fresh unique SDK samples; requires {minimum_samples}"
        )
    return (
        _FreshWindow(
            values=window.values[indices],
            sdk_time_s=window.sdk_time_s[indices],
            capture_time_s=window.capture_monotonic_s[indices],
        ),
        None,
    )


def _magnitude(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(frame, dtype=float)
    if value.shape == shape:
        return np.abs(value)
    if value.shape == (*shape, 3):
        return np.sqrt(np.mean(np.square(value), axis=-1))
    raise ValueError(f"expected {shape} or {(*shape, 3)}, got {value.shape}")


def _clean_mask(magnitude: np.ndarray, threshold: float, noise_area: int) -> np.ndarray:
    mask = magnitude >= threshold
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    areas = np.bincount(labels.ravel())
    keep = areas >= max(2, noise_area)
    keep[0] = False
    return keep[labels]


def _geometry(points_vu: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    points_uv = points_vu[:, ::-1].astype(float)
    center = np.mean(points_uv, axis=0)
    centered = points_uv - center
    covariance = centered.T @ centered / max(1, len(points_uv))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = eigenvectors[:, -1]
    elongation = math.sqrt(
        (max(float(eigenvalues[-1]), 0.0) + 1e-9)
        / (max(float(eigenvalues[0]), 0.0) + 1e-9)
    )
    return center, tangent, elongation


def _angle_deg(measured: np.ndarray, target: np.ndarray) -> float:
    direction = measured.copy()
    if float(np.dot(direction, target)) < 0.0:
        direction *= -1.0
    cross = target[0] * direction[1] - target[1] * direction[0]
    dot = float(np.clip(np.dot(target, direction), -1.0, 1.0))
    return math.degrees(math.atan2(cross, dot))


def _rim_bands(mask: np.ndarray, profile: FingerRimClassifierProfile) -> int:
    labels, count = ndimage.label(mask & profile.rim_roi_mask)
    normal = np.array([-profile.target_tangent_uv[1], profile.target_tangent_uv[0]])
    candidates: list[tuple[float, np.ndarray]] = []
    component_floor = max(2, profile.min_band_area_px // 4)
    for component_id in range(1, count + 1):
        points = np.argwhere(labels == component_id)
        if len(points) < component_floor:
            continue
        center, tangent, elongation = _geometry(points)
        if (
            elongation < profile.min_band_elongation
            or abs(_angle_deg(tangent, profile.target_tangent_uv))
            > profile.max_tangent_error_deg
        ):
            continue
        candidates.append((float(center @ normal), points))
    candidates.sort(key=lambda item: item[0])
    clusters: list[list[np.ndarray]] = []
    cluster_centers: list[float] = []
    for normal_center, points in candidates:
        if (
            not clusters
            or normal_center - cluster_centers[-1]
            >= profile.min_distinct_band_separation_px
        ):
            clusters.append([points])
            cluster_centers.append(normal_center)
        else:
            clusters[-1].append(points)
            weights = [len(value) for value in clusters[-1]]
            centers = [
                float(_geometry(value)[0] @ normal) for value in clusters[-1]
            ]
            cluster_centers[-1] = float(np.average(centers, weights=weights))
    return sum(
        sum(len(points) for points in cluster) >= profile.min_band_area_px
        for cluster in clusters
    )


def _empty_evidence() -> FingerRimEvidence:
    return FingerRimEvidence(
        expected_role=None,
        expected_band_count=None,
        valid=False,
        contact=False,
        resolved=False,
        band_count=0,
        patch_area_px=0,
        roi_fraction=0.0,
        warped_support_fraction=0.0,
        center_uv_px=None,
        center_error_px=None,
        tangent_error_deg=None,
        elongation=None,
        center_jitter_px=None,
        contact_frame_fraction=0.0,
        normal_load_n=None,
        normal_load_peak_n=None,
        force_state=None,
        role_matched=False,
    )


def _force_evidence(
    force: _FreshWindow, profile: ForceContactProfile
) -> tuple[float, float, str]:
    values = np.asarray(force.values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"FORCE6D must have shape (N, 6), got {values.shape}")
    loads = values[:, :3] @ profile.inward_normal_sensor
    median = float(np.median(loads))
    peak = float(np.max(loads))
    load_range = float(np.ptp(loads))
    if peak > profile.overload_n:
        state = "overload"
    elif load_range > profile.max_load_range_n:
        state = "unstable"
    elif median < -profile.no_contact_max_n:
        state = "sign_conflict"
    elif median >= profile.contact_min_n:
        state = "contact"
    elif peak <= profile.no_contact_max_n:
        state = "no_contact"
    else:
        state = "ambiguous"
    return median, peak, state


def _finger_evidence(
    warped: _FreshWindow,
    diff: _FreshWindow,
    force: _FreshWindow | None,
    profile: FingerRimClassifierProfile,
) -> FingerRimEvidence:
    shape = profile.rim_roi_mask.shape
    warped_delta = np.stack(
        [
            _magnitude(np.asarray(frame, dtype=float) - profile.unloaded_warped, shape)
            for frame in warped.values
        ]
    )
    # pyvitaisdk encodes zero change at the centre of its uint8 DIFF_IMG
    # range (127 in v1.0.11), rather than at zero.  Subtract in floating point
    # before taking a magnitude so both positive and negative changes survive.
    diff_magnitude = np.stack(
        [
            _magnitude(
                np.asarray(frame, dtype=float) - profile.diff_neutral_value,
                shape,
            )
            for frame in diff.values
        ]
    )
    noise_area = max(2, profile.min_band_area_px // 4)
    diff_masks = [
        _clean_mask(value, profile.diff_threshold, noise_area)
        for value in diff_magnitude
    ]
    warped_masks = [
        _clean_mask(value, profile.warped_delta_threshold, noise_area)
        for value in warped_delta
    ]
    contact_fraction = float(
        np.mean([np.count_nonzero(mask) >= profile.min_contact_area_px for mask in diff_masks])
    )
    warped_fraction = float(
        np.mean(
            [np.count_nonzero(mask) >= profile.min_contact_area_px for mask in warped_masks]
        )
    )
    diff_mask = _clean_mask(
        np.median(diff_magnitude, axis=0), profile.diff_threshold, noise_area
    )
    warped_mask = _clean_mask(
        np.median(warped_delta, axis=0), profile.warped_delta_threshold, noise_area
    )
    area = int(np.count_nonzero(diff_mask))
    contact = bool(
        area >= profile.min_contact_area_px
        and contact_fraction >= profile.min_contact_frame_fraction
    )
    warped_contact = bool(
        np.count_nonzero(warped_mask) >= profile.min_contact_area_px
        and warped_fraction >= profile.min_contact_frame_fraction
    )
    if area:
        supported = ndimage.binary_dilation(warped_mask, iterations=1)
        warped_support = float(np.count_nonzero(diff_mask & supported) / area)
        roi_fraction = float(np.count_nonzero(diff_mask & profile.rim_roi_mask) / area)
    else:
        warped_support = 0.0
        roi_fraction = 0.0

    intermittent = 0.0 < contact_fraction < profile.min_contact_frame_fraction
    channel_disagreement = contact != warped_contact
    low_support = contact and warped_support < profile.min_warped_support_fraction
    resolved = not (intermittent or channel_disagreement or low_support)
    band_count = _rim_bands(diff_mask, profile) if contact else 0

    center: np.ndarray | None = None
    center_error: float | None = None
    tangent_error: float | None = None
    elongation: float | None = None
    jitter: float | None = None
    if contact:
        points = np.argwhere(diff_mask)
        center, tangent, elongation = _geometry(points)
        center_error = float(np.linalg.norm(center - profile.target_center_uv_px))
        tangent_error = _angle_deg(tangent, profile.target_tangent_uv)
        frame_centers = [
            _geometry(np.argwhere(mask))[0]
            for mask in diff_masks
            if np.count_nonzero(mask) >= profile.min_contact_area_px
        ]
        if len(frame_centers) >= 2:
            stacked = np.stack(frame_centers)
            median_center = np.median(stacked, axis=0)
            jitter = float(np.max(np.linalg.norm(stacked - median_center, axis=1)))
        else:
            jitter = math.inf

    normal_load = peak_load = None
    force_state = None
    if force is not None:
        assert profile.force is not None
        normal_load, peak_load, force_state = _force_evidence(force, profile.force)

    role_matched = bool(
        resolved
        and contact
        and band_count == profile.expected_band_count
        and area <= profile.max_contact_area_px
        and roi_fraction >= profile.min_roi_fraction
        and center_error is not None
        and center_error <= profile.max_center_error_px
        and tangent_error is not None
        and abs(tangent_error) <= profile.max_tangent_error_deg
        and elongation is not None
        and elongation >= profile.min_band_elongation
        and jitter is not None
        and jitter <= profile.max_center_jitter_px
    )
    return FingerRimEvidence(
        expected_role=profile.role,
        expected_band_count=profile.expected_band_count,
        valid=True,
        contact=contact,
        resolved=resolved,
        band_count=band_count,
        patch_area_px=area,
        roi_fraction=roi_fraction,
        warped_support_fraction=warped_support,
        center_uv_px=center,
        center_error_px=center_error,
        tangent_error_deg=tangent_error,
        elongation=elongation,
        center_jitter_px=jitter,
        contact_frame_fraction=contact_fraction,
        normal_load_n=normal_load,
        normal_load_peak_n=peak_load,
        force_state=force_state,
        role_matched=role_matched,
    )


def _classification(
    label: TactileRimClass,
    score: float,
    left: FingerRimEvidence,
    right: FingerRimEvidence,
    reasons: tuple[str, ...],
    force_used: bool,
    evidence_key: tuple[str | float, ...] | None,
) -> TactileRimClassification:
    return TactileRimClassification(
        label=label,
        score=float(np.clip(score, 0.0, 1.0)),
        left=left,
        right=right,
        reasons=reasons,
        force_used=force_used,
        evidence_key=evidence_key,
    )


def classify_tactile_rim(
    left_window: FingerRimClassifierWindow,
    right_window: FingerRimClassifierWindow,
    config: DualRimClassifierConfig,
    *,
    now_s: float,
) -> TactileRimClassification:
    """Classify one fresh, settled dual-finger tactile window.

    FORCE6D is optional at the profile level. If a profile contains a force
    calibration, a force window becomes required and acts as a consistency and
    overload veto. Without that profile the result is explicitly image-only.
    """

    now_s = float(now_s)
    if not math.isfinite(now_s):
        raise ValueError("now_s must be finite")
    empty = _empty_evidence()
    if left_window.calibration_id != config.left.calibration_id:
        return _classification(
            TactileRimClass.UNKNOWN,
            0.0,
            empty,
            empty,
            ("left calibration_id does not match the classifier profile",),
            False,
            None,
        )
    if right_window.calibration_id != config.right.calibration_id:
        return _classification(
            TactileRimClass.UNKNOWN,
            0.0,
            empty,
            empty,
            ("right calibration_id does not match the classifier profile",),
            False,
            None,
        )

    fresh: dict[str, _FreshWindow] = {}
    errors: list[str] = []
    for side, window in (("left", left_window), ("right", right_window)):
        for channel, value in (("warped", window.warped), ("diff", window.diff)):
            result, error = _fresh_window(
                value,
                minimum_samples=config.min_unique_image_samples,
                now_s=now_s,
                max_age_s=config.max_age_s,
            )
            if result is None:
                errors.append(f"{side}.{channel}: {error}")
            else:
                fresh[f"{side}.{channel}"] = result
        profile = config.left if side == "left" else config.right
        if profile.force is not None:
            if window.force6d is None:
                errors.append(f"{side}.force6d: required by the calibrated profile")
            else:
                result, error = _fresh_window(
                    window.force6d,
                    minimum_samples=config.min_unique_force_samples,
                    now_s=now_s,
                    max_age_s=config.max_age_s,
                )
                if result is None:
                    errors.append(f"{side}.force6d: {error}")
                else:
                    fresh[f"{side}.force6d"] = result
    force_used = config.left.force is not None or config.right.force is not None
    if errors:
        return _classification(
            TactileRimClass.UNKNOWN,
            0.0,
            empty,
            empty,
            tuple(errors),
            force_used,
            None,
        )
    endpoints = [window.endpoint[1] for window in fresh.values()]
    if max(endpoints) - min(endpoints) > config.max_pair_skew_s:
        return _classification(
            TactileRimClass.UNKNOWN,
            0.0,
            empty,
            empty,
            ("required tactile channels are not synchronized",),
            force_used,
            None,
        )
    ordered_names = ["left.warped", "left.diff", "right.warped", "right.diff"]
    ordered_names.extend(
        name for name in ("left.force6d", "right.force6d") if name in fresh
    )
    evidence_key = (
        config.profile_id,
        config.left.calibration_id,
        config.left.role.value,
        config.right.calibration_id,
        config.right.role.value,
        *(value for name in ordered_names for value in fresh[name].endpoint),
    )
    try:
        left = _finger_evidence(
            fresh["left.warped"],
            fresh["left.diff"],
            fresh.get("left.force6d"),
            config.left,
        )
        right = _finger_evidence(
            fresh["right.warped"],
            fresh["right.diff"],
            fresh.get("right.force6d"),
            config.right,
        )
    except ValueError as exc:
        return _classification(
            TactileRimClass.UNKNOWN,
            0.0,
            empty,
            empty,
            (f"tactile channel shape is invalid: {exc}",),
            force_used,
            evidence_key,
        )

    force_failures = [
        f"{side} force is {evidence.force_state}"
        for side, evidence in (("left", left), ("right", right))
        if evidence.force_state in {"overload", "unstable", "sign_conflict", "ambiguous"}
    ]
    if force_failures:
        return _classification(
            TactileRimClass.UNKNOWN,
            0.0,
            left,
            right,
            tuple(force_failures),
            force_used,
            evidence_key,
        )
    if not left.resolved or not right.resolved:
        return _classification(
            TactileRimClass.UNKNOWN,
            0.0,
            left,
            right,
            ("WARPED_IMG and DIFF_IMG contact evidence is transient or inconsistent",),
            force_used,
            evidence_key,
        )
    force_image_disagreement = [
        side
        for side, evidence in (("left", left), ("right", right))
        if evidence.force_state is not None
        and (
            (evidence.contact and evidence.force_state != "contact")
            or (not evidence.contact and evidence.force_state != "no_contact")
        )
    ]
    if force_image_disagreement:
        return _classification(
            TactileRimClass.UNKNOWN,
            0.0,
            left,
            right,
            (f"image and force disagree on {', '.join(force_image_disagreement)}",),
            force_used,
            evidence_key,
        )
    if not left.contact and not right.contact:
        return _classification(
            TactileRimClass.NO_CONTACT,
            1.0,
            left,
            right,
            ("neither finger has a persistent, cross-checked contact patch",),
            force_used,
            evidence_key,
        )
    multi_role = [
        profile.role.value
        for evidence, profile in (
            (left, config.left),
            (right, config.right),
        )
        if evidence.contact
        and evidence.band_count >= profile.multi_rim_min_band_count
    ]
    if multi_role:
        score = min(
            evidence.warped_support_fraction
            for evidence in (left, right)
            if evidence.contact
        )
        return _classification(
            TactileRimClass.MULTI_RIM,
            score,
            left,
            right,
            (
                "rim-band count exceeds the calibrated role template on "
                f"{', '.join(multi_role)}",
            ),
            force_used,
            evidence_key,
        )
    if left.contact != right.contact:
        return _classification(
            TactileRimClass.ROLE_MISMATCH,
            1.0,
            left,
            right,
            ("only one of the required INNER/OUTER roles has persistent contact",),
            force_used,
            evidence_key,
        )
    role_failures = [
        evidence.expected_role.value
        for evidence in (left, right)
        if not evidence.role_matched and evidence.expected_role is not None
    ]
    if role_failures:
        return _classification(
            TactileRimClass.ROLE_MISMATCH,
            0.0,
            left,
            right,
            (
                "contact does not match its independently calibrated role template on "
                f"{', '.join(role_failures)}",
            ),
            force_used,
            evidence_key,
        )
    score = min(
        left.warped_support_fraction,
        right.warped_support_fraction,
    )
    return _classification(
        TactileRimClass.INNER_OUTER_RIM_MATCH,
        score,
        left,
        right,
        (
            "the INNER and OUTER fingers each match their own calibrated contact template",
        ),
        force_used,
        evidence_key,
    )


__all__ = [
    "DualRimClassifierConfig",
    "FingerRimClassifierProfile",
    "FingerRimClassifierWindow",
    "FingerRimEvidence",
    "FingerRimRole",
    "ForceContactProfile",
    "TactileRimClass",
    "TactileRimClassification",
    "classify_tactile_rim",
]
