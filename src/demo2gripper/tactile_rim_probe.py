"""Pure tactile evaluator for active probing of a disposable-cup rim.

The module intentionally owns no camera, USB, serial, ROS, or robot handles.  A
sensor sidecar supplies synchronized windows and a task runner executes the
bounded recommendation.  In particular, TCP corrections must only be applied
after the runner has opened the gripper enough to unload the cup wall.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
from scipy import ndimage

MAX_PROBE_CLOSE_STEP_M = 0.002
MAX_PROBE_RELEASE_STEP_M = 0.005
MAX_PROBE_TCP_STEP_M = 0.003
MAX_PROBE_YAW_STEP_RAD = math.radians(3.0)
MAX_ADJUSTMENT_AUTHORIZATION_TTL_S = 5.0


class RimProbeState(StrEnum):
    SENSOR_INVALID = "sensor_invalid"
    NO_CONTACT = "no_contact"
    LEFT_ONLY = "left_only"
    RIGHT_ONLY = "right_only"
    CONTACT_UNRESOLVED = "contact_unresolved"
    ROLE_MISMATCH = "role_mismatch"
    ROLE_MATCH_LOW_LOAD = "role_match_low_load"
    ROLE_LOAD_OUT_OF_TOLERANCE = "role_load_out_of_tolerance"
    UNSTABLE = "unstable"
    OVERLOAD = "overload"
    EXCESS_DEFORMATION = "excess_deformation"
    CAPTURE_CANDIDATE = "capture_candidate"
    CAPTURED = "captured"


class ProbeAction(StrEnum):
    HOLD = "hold"
    WAIT = "wait"
    CLOSE_STEP = "close_step"
    RELEASE = "release"
    RELEASE_THEN_ADJUST = "release_then_adjust"
    APPLY_TCP_ADJUSTMENT = "apply_tcp_adjustment"
    ACCEPT = "accept"


class RimContactRole(StrEnum):
    """Physical side of the same cup wall assigned to one tactile finger."""

    INNER = "inner"
    OUTER = "outer"


@dataclass(frozen=True)
class TimedArrayWindow:
    """One channel window with device and common-host acquisition times.

    ``sdk_time_s`` is used to reject cached frames. ``capture_monotonic_s`` is
    in a common host clock and is used for age and cross-device skew checks.
    """

    values: np.ndarray
    sdk_time_s: np.ndarray
    capture_monotonic_s: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        sdk_time_s = np.array(self.sdk_time_s, dtype=float, copy=True)
        capture_time_s = np.array(
            self.capture_monotonic_s, dtype=float, copy=True
        )
        valid = np.array(self.valid, dtype=bool, copy=True)
        if values.ndim < 2:
            raise ValueError("window values must have a sample axis and a value axis")
        if not (
            np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.bool_)
        ):
            raise ValueError("window values must have a numeric or boolean dtype")
        sample_count = values.shape[0]
        for name, value in (
            ("sdk_time_s", sdk_time_s),
            ("capture_monotonic_s", capture_time_s),
            ("valid", valid),
        ):
            if value.shape != (sample_count,):
                raise ValueError(f"{name} must have shape ({sample_count},)")
        if not np.isfinite(sdk_time_s).all() or not np.isfinite(capture_time_s).all():
            raise ValueError("window timestamps must be finite")
        values.setflags(write=False)
        sdk_time_s.setflags(write=False)
        capture_time_s.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sdk_time_s", sdk_time_s)
        object.__setattr__(self, "capture_monotonic_s", capture_time_s)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True)
class FingerTactileWindow:
    calibration_id: str
    diff: TimedArrayWindow
    force6d: TimedArrayWindow
    depth: TimedArrayWindow | None = None

    def __post_init__(self) -> None:
        if not self.calibration_id:
            raise ValueError("calibration_id must be non-empty")


@dataclass(frozen=True)
class FingerRimTarget:
    """Calibrated role-specific image and force target for one tactile finger.

    An inner-wall contact and an outer-wall contact are different templates.
    Nothing in this type requires their image area or geometry to be equal.
    The role belongs to the current probe hypothesis, not permanently to a
    physical left/right sensor serial.
    """

    calibration_id: str
    contact_role: RimContactRole
    roi_mask: np.ndarray
    target_center_uv_px: np.ndarray
    target_tangent_uv: np.ndarray
    diff_threshold: float
    min_patch_area_px: int
    max_patch_area_px: int
    min_roi_fraction: float
    max_center_error_px: float
    max_tangent_error_deg: float
    min_elongation: float
    max_center_jitter_px: float
    min_contact_frame_fraction: float
    max_significant_components: int
    inward_normal_sensor: np.ndarray
    normal_load_band_n: tuple[float, float]
    max_normal_load_range_n: float
    max_indentation_m: float | None = None
    image_error_to_task_m_per_px: np.ndarray | None = None
    tangent_error_to_tcp_yaw: float | None = None
    toward_sensor_task: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.calibration_id:
            raise ValueError("calibration_id must be non-empty")
        object.__setattr__(self, "contact_role", RimContactRole(self.contact_role))
        roi = np.array(self.roi_mask, dtype=bool, copy=True)
        if roi.ndim != 2 or not roi.any():
            raise ValueError("roi_mask must be a non-empty 2D mask")
        roi.setflags(write=False)
        object.__setattr__(self, "roi_mask", roi)

        center = _finite_vector(self.target_center_uv_px, 2, "target_center_uv_px")
        height, width = roi.shape
        if not (0.0 <= center[0] < width and 0.0 <= center[1] < height):
            raise ValueError("target_center_uv_px must lie inside the tactile image")
        center.setflags(write=False)
        object.__setattr__(self, "target_center_uv_px", center)

        tangent = _unit_vector(self.target_tangent_uv, 2, "target_tangent_uv")
        tangent.setflags(write=False)
        object.__setattr__(self, "target_tangent_uv", tangent)
        normal = _unit_vector(self.inward_normal_sensor, 3, "inward_normal_sensor")
        normal.setflags(write=False)
        object.__setattr__(self, "inward_normal_sensor", normal)

        if not math.isfinite(self.diff_threshold) or self.diff_threshold <= 0.0:
            raise ValueError("diff_threshold must be positive and finite")
        if self.min_patch_area_px <= 0:
            raise ValueError("min_patch_area_px must be positive")
        if self.max_patch_area_px < self.min_patch_area_px:
            raise ValueError("max_patch_area_px must be >= min_patch_area_px")
        if not 0.0 < self.min_roi_fraction <= 1.0:
            raise ValueError("min_roi_fraction must lie in (0, 1]")
        for name in (
            "max_center_error_px",
            "max_tangent_error_deg",
            "min_elongation",
            "max_center_jitter_px",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_tangent_error_deg > 90.0:
            raise ValueError("max_tangent_error_deg cannot exceed 90 degrees")
        if self.max_significant_components <= 0:
            raise ValueError("max_significant_components must be positive")
        if not 0.0 < self.min_contact_frame_fraction <= 1.0:
            raise ValueError("min_contact_frame_fraction must lie in (0, 1]")

        load_min, load_max = (float(value) for value in self.normal_load_band_n)
        if not (math.isfinite(load_min) and math.isfinite(load_max)):
            raise ValueError("normal_load_band_n must be finite")
        if load_min < 0.0 or load_max <= load_min:
            raise ValueError("normal_load_band_n must satisfy 0 <= min < max")
        object.__setattr__(self, "normal_load_band_n", (load_min, load_max))
        load_range = float(self.max_normal_load_range_n)
        if not math.isfinite(load_range) or load_range <= 0.0:
            raise ValueError("max_normal_load_range_n must be positive and finite")
        object.__setattr__(self, "max_normal_load_range_n", load_range)

        if self.max_indentation_m is not None:
            indentation = float(self.max_indentation_m)
            if not math.isfinite(indentation) or indentation <= 0.0:
                raise ValueError("max_indentation_m must be positive and finite")
            object.__setattr__(self, "max_indentation_m", indentation)

        if self.image_error_to_task_m_per_px is not None:
            jacobian = np.array(
                self.image_error_to_task_m_per_px, dtype=float, copy=True
            )
            if jacobian.shape != (3, 2) or not np.isfinite(jacobian).all():
                raise ValueError("image_error_to_task_m_per_px must be finite shape (3, 2)")
            jacobian.setflags(write=False)
            object.__setattr__(self, "image_error_to_task_m_per_px", jacobian)
        if self.tangent_error_to_tcp_yaw is not None:
            gain = float(self.tangent_error_to_tcp_yaw)
            if not math.isfinite(gain):
                raise ValueError("tangent_error_to_tcp_yaw must be finite")
            object.__setattr__(self, "tangent_error_to_tcp_yaw", gain)
        if self.toward_sensor_task is not None:
            direction = _unit_vector(self.toward_sensor_task, 3, "toward_sensor_task")
            direction.setflags(write=False)
            object.__setattr__(self, "toward_sensor_task", direction)


@dataclass(frozen=True)
class RimProbeConfig:
    profile_id: str
    left: FingerRimTarget
    right: FingerRimTarget
    min_unique_image_samples: int = 4
    min_unique_force_samples: int = 3
    max_age_s: float = 0.15
    max_pair_skew_s: float = 0.08
    # Optional calibrated INNER/OUTER force-ratio gate.  This is independent of
    # image geometry and does not assume equal loads on the two different roles.
    inner_to_outer_normal_load_ratio_band: tuple[float, float] | None = None
    close_step_m: float = 0.0005
    release_step_m: float = 0.0010
    recenter_step_m: float = 0.0005
    max_translation_step_m: float = 0.0010
    max_yaw_step_rad: float = math.radians(1.0)
    max_translation_disagreement_m: float = 0.0010
    max_yaw_disagreement_rad: float = math.radians(1.0)
    min_translation_agreement_cosine: float = 0.25
    adjustment_authorization_ttl_s: float = 1.0
    # Optional task prior: translation from an ambiguous multi-band contact to a
    # shallower rim-height hypothesis, expressed in the calibrated task frame.
    too_deep_recovery_translation_task_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        if {self.left.contact_role, self.right.contact_role} != {
            RimContactRole.INNER,
            RimContactRole.OUTER,
        }:
            raise ValueError(
                "cup-rim profile must assign exactly one INNER and one OUTER role"
            )
        if self.left.roi_mask.shape != self.right.roi_mask.shape:
            raise ValueError("left and right tactile image shapes must match")
        if self.min_unique_image_samples < 2 or self.min_unique_force_samples < 1:
            raise ValueError("minimum unique sample counts are invalid")
        for name in (
            "max_age_s",
            "max_pair_skew_s",
            "close_step_m",
            "release_step_m",
            "recenter_step_m",
            "max_translation_step_m",
            "max_yaw_step_rad",
            "max_translation_disagreement_m",
            "max_yaw_disagreement_rad",
            "adjustment_authorization_ttl_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.inner_to_outer_normal_load_ratio_band is not None:
            ratio_min, ratio_max = (
                float(value)
                for value in self.inner_to_outer_normal_load_ratio_band
            )
            if (
                not math.isfinite(ratio_min)
                or not math.isfinite(ratio_max)
                or ratio_min <= 0.0
                or ratio_max < ratio_min
            ):
                raise ValueError(
                    "inner_to_outer_normal_load_ratio_band must satisfy "
                    "0 < min <= max"
                )
            object.__setattr__(
                self,
                "inner_to_outer_normal_load_ratio_band",
                (ratio_min, ratio_max),
            )
        if not -1.0 <= self.min_translation_agreement_cosine <= 1.0:
            raise ValueError("min_translation_agreement_cosine must lie in [-1, 1]")
        caps = {
            "close_step_m": MAX_PROBE_CLOSE_STEP_M,
            "release_step_m": MAX_PROBE_RELEASE_STEP_M,
            "recenter_step_m": MAX_PROBE_TCP_STEP_M,
            "max_translation_step_m": MAX_PROBE_TCP_STEP_M,
            "max_translation_disagreement_m": 2.0 * MAX_PROBE_TCP_STEP_M,
            "max_yaw_step_rad": MAX_PROBE_YAW_STEP_RAD,
            "max_yaw_disagreement_rad": 2.0 * MAX_PROBE_YAW_STEP_RAD,
            "adjustment_authorization_ttl_s": MAX_ADJUSTMENT_AUTHORIZATION_TTL_S,
        }
        for name, cap in caps.items():
            if float(getattr(self, name)) > cap:
                raise ValueError(f"{name} exceeds the reviewed cap {cap}")
        if self.too_deep_recovery_translation_task_m is not None:
            correction = _finite_vector(
                self.too_deep_recovery_translation_task_m,
                3,
                "too_deep_recovery_translation_task_m",
            )
            norm = float(np.linalg.norm(correction))
            if norm <= 1e-12 or norm > self.max_translation_step_m:
                raise ValueError(
                    "too_deep recovery must be non-zero and no larger than "
                    "max_translation_step_m"
                )
            correction.setflags(write=False)
            object.__setattr__(
                self, "too_deep_recovery_translation_task_m", correction
            )


@dataclass(frozen=True)
class FingerRimMetrics:
    valid: bool
    contact: bool
    patch_area_px: int
    significant_components: int
    roi_fraction: float
    center_uv_px: np.ndarray | None
    center_error_uv_px: np.ndarray | None
    tangent_error_deg: float | None
    elongation: float | None
    center_jitter_px: float | None
    contact_frame_fraction: float
    normal_load_n: float | None
    normal_load_mad_n: float | None
    normal_load_peak_n: float | None
    normal_load_range_n: float | None
    indentation_p95_m: float | None
    geometry_score: float


@dataclass(frozen=True)
class ProbeAdjustment:
    action: ProbeAction
    # Positive opens the Hand-E; negative closes it.
    jaw_opening_delta_m: float
    # These are proposals, not authorized commands. The runner must first unload
    # and confirm release with a later fresh tactile window.
    proposed_tcp_translation_task_m: np.ndarray
    proposed_tcp_yaw_delta_rad: float
    release_confirmation_required: bool

    def __post_init__(self) -> None:
        jaw_delta = float(self.jaw_opening_delta_m)
        yaw_delta = float(self.proposed_tcp_yaw_delta_rad)
        translation = np.asarray(
            self.proposed_tcp_translation_task_m, dtype=float
        ).copy()
        if not math.isfinite(jaw_delta):
            raise ValueError("jaw_opening_delta_m must be finite")
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("proposed TCP translation must contain three finite values")
        if not math.isfinite(yaw_delta):
            raise ValueError("proposed TCP yaw must be finite")
        translation.setflags(write=False)
        object.__setattr__(self, "jaw_opening_delta_m", jaw_delta)
        object.__setattr__(self, "proposed_tcp_translation_task_m", translation)
        object.__setattr__(self, "proposed_tcp_yaw_delta_rad", yaw_delta)


@dataclass(frozen=True)
class ReleaseCommandReceipt:
    """Evidence that the runner completed the requested unload command."""

    command_id: str
    probe_context_id: str
    completed_at_monotonic_s: float
    applied_jaw_opening_delta_m: float

    def __post_init__(self) -> None:
        completed_at = float(self.completed_at_monotonic_s)
        applied_delta = float(self.applied_jaw_opening_delta_m)
        if not self.command_id:
            raise ValueError("release command_id must be non-empty")
        if not self.probe_context_id:
            raise ValueError("release probe_context_id must be non-empty")
        if not math.isfinite(completed_at):
            raise ValueError("release completion time must be finite")
        if not math.isfinite(applied_delta) or applied_delta <= 0.0:
            raise ValueError("applied jaw-opening delta must be positive and finite")
        if applied_delta > MAX_PROBE_RELEASE_STEP_M:
            raise ValueError("applied jaw-opening delta exceeds the reviewed cap")
        object.__setattr__(self, "completed_at_monotonic_s", completed_at)
        object.__setattr__(self, "applied_jaw_opening_delta_m", applied_delta)


@dataclass(frozen=True)
class TcpAdjustmentAuthorization:
    """Short-lived one-shot token which a runner must consume before TCP motion."""

    authorization_id: str
    probe_context_id: str
    profile_id: str
    calibration_ids: tuple[str, str]
    contact_roles: tuple[RimContactRole, RimContactRole]
    issued_at_monotonic_s: float
    expires_at_monotonic_s: float
    release_command_id: str
    proposal_evidence_key: tuple[float, ...]
    release_evidence_key: tuple[float, ...]
    adjustment: ProbeAdjustment

    def __post_init__(self) -> None:
        issued_at = float(self.issued_at_monotonic_s)
        expires_at = float(self.expires_at_monotonic_s)
        if not self.authorization_id or not self.release_command_id:
            raise ValueError("authorization and release command ids must be non-empty")
        if not self.probe_context_id or not self.profile_id:
            raise ValueError("authorization provenance must be non-empty")
        if len(self.calibration_ids) != 2 or not all(self.calibration_ids):
            raise ValueError("authorization calibration ids must be non-empty")
        roles = tuple(RimContactRole(value) for value in self.contact_roles)
        if len(roles) != 2 or set(roles) != {
            RimContactRole.INNER,
            RimContactRole.OUTER,
        }:
            raise ValueError(
                "authorization must bind exactly one INNER and one OUTER role"
            )
        if not math.isfinite(issued_at) or not math.isfinite(expires_at):
            raise ValueError("authorization times must be finite")
        if expires_at <= issued_at:
            raise ValueError("authorization must expire after it is issued")
        if self.adjustment.action is not ProbeAction.APPLY_TCP_ADJUSTMENT:
            raise ValueError("authorization must contain an APPLY_TCP_ADJUSTMENT")
        object.__setattr__(self, "issued_at_monotonic_s", issued_at)
        object.__setattr__(self, "expires_at_monotonic_s", expires_at)
        object.__setattr__(self, "contact_roles", roles)


@dataclass(frozen=True)
class RimProbeEvaluation:
    state: RimProbeState
    confidence: float
    left: FingerRimMetrics
    right: FingerRimMetrics
    predicates: frozenset[str]
    reasons: tuple[str, ...]
    recommendation: ProbeAdjustment
    evidence_key: tuple[float, ...] | None
    consecutive_good_windows: int = 0
    probe_context_id: str | None = None
    profile_id: str | None = None
    calibration_ids: tuple[str, str] | None = None
    contact_roles: tuple[RimContactRole, RimContactRole] | None = None
    evaluated_at_monotonic_s: float | None = None
    adjustment_authorization_ttl_s: float | None = None
    evidence_not_before_monotonic_s: float | None = None


def _finite_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.array(value, dtype=float, copy=True)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return vector


def _unit_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = _finite_vector(value, size, name)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def _empty_metrics() -> FingerRimMetrics:
    return FingerRimMetrics(
        valid=False,
        contact=False,
        patch_area_px=0,
        significant_components=0,
        roi_fraction=0.0,
        center_uv_px=None,
        center_error_uv_px=None,
        tangent_error_deg=None,
        elongation=None,
        center_jitter_px=None,
        contact_frame_fraction=0.0,
        normal_load_n=None,
        normal_load_mad_n=None,
        normal_load_peak_n=None,
        normal_load_range_n=None,
        indentation_p95_m=None,
        geometry_score=0.0,
    )


@dataclass(frozen=True)
class _FreshWindow:
    values: np.ndarray
    sdk_time_s: np.ndarray
    capture_time_s: np.ndarray

    @property
    def last_sdk_time_s(self) -> float:
        return float(self.sdk_time_s[-1])

    @property
    def last_capture_time_s(self) -> float:
        return float(self.capture_time_s[-1])


def _evidence_key(*windows: _FreshWindow) -> tuple[float, ...]:
    values: list[float] = []
    for window in windows:
        values.extend((window.last_sdk_time_s, window.last_capture_time_s))
    return tuple(values)


def _fresh_unique_window(
    window: TimedArrayWindow,
    *,
    minimum_samples: int,
    now_s: float,
    max_age_s: float,
    not_before_s: float | None,
    channel_name: str,
) -> tuple[_FreshWindow | None, str | None]:
    valid_indices = np.flatnonzero(window.valid)
    if valid_indices.size == 0:
        return None, f"{channel_name} has no valid samples"
    sdk_times = window.sdk_time_s[valid_indices]
    if np.any(np.diff(sdk_times) < 0.0):
        return None, f"{channel_name} SDK timestamps moved backwards"
    values = window.values[valid_indices]
    finite_rows = np.asarray(
        [np.isfinite(np.asarray(value, dtype=float)).all() for value in values],
        dtype=bool,
    )
    valid_indices = valid_indices[finite_rows]
    if valid_indices.size == 0:
        return None, f"{channel_name} has no finite valid samples"

    # Keep the first receipt of each SDK frame. A cached frame repeatedly returned
    # with a newer host timestamp must not become fresh evidence.
    _, first_positions = np.unique(window.sdk_time_s[valid_indices], return_index=True)
    selected = valid_indices[np.sort(first_positions)]
    capture_times = window.capture_monotonic_s[selected]
    if np.any(np.diff(capture_times) < 0.0):
        return None, f"{channel_name} host capture timestamps moved backwards"
    fresh_mask = (capture_times >= now_s - max_age_s) & (capture_times <= now_s + 1e-6)
    if not_before_s is not None:
        fresh_mask &= capture_times > not_before_s
    selected = selected[fresh_mask]
    if selected.size < minimum_samples:
        return None, (
            f"{channel_name} has {selected.size} fresh unique SDK samples; "
            f"requires {minimum_samples}"
        )
    capture_times = window.capture_monotonic_s[selected]
    return (
        _FreshWindow(
            values=window.values[selected],
            sdk_time_s=window.sdk_time_s[selected],
            capture_time_s=capture_times,
        ),
        None,
    )


def _diff_magnitude(frame: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(frame, dtype=float)
    if value.shape == expected_shape:
        return np.abs(value)
    if value.shape == (*expected_shape, 3):
        return np.sqrt(np.mean(np.square(value), axis=-1))
    raise ValueError(
        f"diff frame must have shape {expected_shape} or {(*expected_shape, 3)}, "
        f"got {value.shape}"
    )


def _clean_contact_mask(magnitude: np.ndarray, target: FingerRimTarget) -> np.ndarray:
    mask = magnitude >= target.diff_threshold
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    component_areas = np.bincount(labels.ravel())
    noise_floor = max(2, target.min_patch_area_px // 4)
    keep = component_areas >= noise_floor
    keep[0] = False
    return keep[labels]


def _mask_geometry(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    rows, cols = np.nonzero(mask)
    center = np.array([float(np.mean(cols)), float(np.mean(rows))])
    points = np.column_stack((cols, rows)).astype(float)
    centered = points - center
    covariance = centered.T @ centered / max(points.shape[0], 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = max(float(eigenvalues[-1]), 0.0)
    minor = max(float(eigenvalues[0]), 0.0)
    tangent = eigenvectors[:, -1]
    elongation = math.sqrt((major + 1e-9) / (minor + 1e-9))
    return center, tangent, elongation


def _signed_undirected_angle_deg(measured: np.ndarray, target: np.ndarray) -> float:
    direction = measured.copy()
    if float(np.dot(direction, target)) < 0.0:
        direction *= -1.0
    cross = target[0] * direction[1] - target[1] * direction[0]
    dot = float(np.clip(np.dot(target, direction), -1.0, 1.0))
    return math.degrees(math.atan2(cross, dot))


def _image_metrics(diff: _FreshWindow, target: FingerRimTarget) -> dict[str, object]:
    expected_shape = target.roi_mask.shape
    magnitudes = np.stack(
        [_diff_magnitude(frame, expected_shape) for frame in diff.values], axis=0
    )
    frame_masks = [_clean_contact_mask(magnitude, target) for magnitude in magnitudes]
    contact_frame_fraction = float(
        np.mean(
            [
                np.count_nonzero(frame_mask) >= target.min_patch_area_px
                for frame_mask in frame_masks
            ]
        )
    )
    aggregate = np.median(magnitudes, axis=0)
    mask = _clean_contact_mask(aggregate, target)
    area = int(np.count_nonzero(mask))
    if (
        area < target.min_patch_area_px
        or contact_frame_fraction < target.min_contact_frame_fraction
    ):
        return {
            "contact": False,
            "area": area,
            "components": 0,
            "roi_fraction": 0.0,
            "center": None,
            "center_error": None,
            "tangent_error_deg": None,
            "elongation": None,
            "jitter": None,
            "contact_frame_fraction": contact_frame_fraction,
            "geometry_score": 0.0,
            "mask": mask,
        }

    labels, component_count = ndimage.label(mask)
    areas = np.bincount(labels.ravel())[1:]
    largest = int(np.max(areas)) if areas.size else 0
    significant_floor = max(target.min_patch_area_px // 2, int(math.ceil(0.20 * largest)))
    significant_components = int(np.count_nonzero(areas >= significant_floor))
    center, tangent, elongation = _mask_geometry(mask)
    center_error = target.target_center_uv_px - center
    tangent_error_deg = _signed_undirected_angle_deg(tangent, target.target_tangent_uv)
    roi_fraction = float(np.count_nonzero(mask & target.roi_mask) / area)

    frame_centers: list[np.ndarray] = []
    for frame_mask in frame_masks:
        if np.count_nonzero(frame_mask) >= target.min_patch_area_px:
            frame_centers.append(_mask_geometry(frame_mask)[0])
    if len(frame_centers) >= 2:
        centers = np.stack(frame_centers)
        median_center = np.median(centers, axis=0)
        jitter = float(np.max(np.linalg.norm(centers - median_center, axis=1)))
    else:
        jitter = math.inf

    area_score = min(1.0, area / target.min_patch_area_px)
    roi_score = min(1.0, roi_fraction / target.min_roi_fraction)
    center_score = max(
        0.0,
        1.0 - float(np.linalg.norm(center_error)) / target.max_center_error_px,
    )
    tangent_score = max(
        0.0, 1.0 - abs(tangent_error_deg) / target.max_tangent_error_deg
    )
    elongation_score = min(1.0, elongation / target.min_elongation)
    geometry_score = float(
        np.mean([area_score, roi_score, center_score, tangent_score, elongation_score])
    )
    return {
        "contact": True,
        "area": area,
        "components": significant_components,
        "roi_fraction": roi_fraction,
        "center": center,
        "center_error": center_error,
        "tangent_error_deg": tangent_error_deg,
        "elongation": elongation,
        "jitter": jitter,
        "contact_frame_fraction": contact_frame_fraction,
        "geometry_score": geometry_score,
        "mask": mask,
    }


def _normal_load_metrics(
    force: _FreshWindow, target: FingerRimTarget
) -> tuple[float, float, float, float]:
    values = np.asarray(force.values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"force6d values must have shape (N, 6), got {values.shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        loads = values[:, :3] @ target.inward_normal_sensor
    if not np.isfinite(loads).all():
        raise ValueError("projected inward normal loads are not finite")
    median = float(np.median(loads))
    mad = float(np.median(np.abs(loads - median)))
    peak = float(np.max(loads))
    load_range = float(np.max(loads) - np.min(loads))
    if not all(math.isfinite(value) for value in (median, mad, peak, load_range)):
        raise ValueError("derived inward normal-load metrics are not finite")
    return median, mad, peak, load_range


def _indentation_p95(
    depth: _FreshWindow | None,
    contact_mask: np.ndarray,
    expected_shape: tuple[int, int],
) -> float | None:
    if depth is None or not np.any(contact_mask):
        return None
    values = np.asarray(depth.values, dtype=float)
    if values.ndim != 3 or values.shape[1:] != expected_shape:
        raise ValueError(
            f"depth values must have shape (N, {expected_shape[0]}, {expected_shape[1]})"
        )
    # A short crush event must not disappear in a temporal median. The profile
    # only enables this hard gate after the depth scale and sign are calibrated.
    frame_p95 = [
        float(np.percentile(frame[contact_mask], 95.0)) for frame in values
    ]
    return max(frame_p95)


def _finger_metrics(
    diff: _FreshWindow,
    force: _FreshWindow,
    depth: _FreshWindow | None,
    target: FingerRimTarget,
) -> FingerRimMetrics:
    image = _image_metrics(diff, target)
    normal_load, load_mad, load_peak, load_range = _normal_load_metrics(force, target)
    indentation = _indentation_p95(depth, image["mask"], target.roi_mask.shape)
    return FingerRimMetrics(
        valid=True,
        contact=bool(image["contact"]),
        patch_area_px=int(image["area"]),
        significant_components=int(image["components"]),
        roi_fraction=float(image["roi_fraction"]),
        center_uv_px=image["center"],  # type: ignore[arg-type]
        center_error_uv_px=image["center_error"],  # type: ignore[arg-type]
        tangent_error_deg=image["tangent_error_deg"],  # type: ignore[arg-type]
        elongation=image["elongation"],  # type: ignore[arg-type]
        center_jitter_px=image["jitter"],  # type: ignore[arg-type]
        contact_frame_fraction=float(image["contact_frame_fraction"]),
        normal_load_n=normal_load,
        normal_load_mad_n=load_mad,
        normal_load_peak_n=load_peak,
        normal_load_range_n=load_range,
        indentation_p95_m=indentation,
        geometry_score=float(image["geometry_score"]),
    )


def _force_only_metrics(force: _FreshWindow, target: FingerRimTarget) -> FingerRimMetrics:
    normal_load, load_mad, load_peak, load_range = _normal_load_metrics(force, target)
    return replace(
        _empty_metrics(),
        normal_load_n=normal_load,
        normal_load_mad_n=load_mad,
        normal_load_peak_n=load_peak,
        normal_load_range_n=load_range,
    )


def _zero_adjustment(action: ProbeAction, jaw_delta_m: float = 0.0) -> ProbeAdjustment:
    return ProbeAdjustment(
        action=action,
        jaw_opening_delta_m=float(jaw_delta_m),
        proposed_tcp_translation_task_m=np.zeros(3, dtype=float),
        proposed_tcp_yaw_delta_rad=0.0,
        release_confirmation_required=False,
    )


def _invalid_evaluation(reason: str) -> RimProbeEvaluation:
    return RimProbeEvaluation(
        state=RimProbeState.SENSOR_INVALID,
        confidence=0.0,
        left=_empty_metrics(),
        right=_empty_metrics(),
        predicates=frozenset(),
        reasons=(reason,),
        recommendation=_zero_adjustment(ProbeAction.HOLD),
        evidence_key=None,
    )


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if not np.isfinite(vector).all():
        return np.zeros_like(vector)
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        return vector
    norm = scale * float(np.linalg.norm(vector / scale))
    if norm <= limit:
        return vector
    return vector * (limit / norm)


def _realignment(
    left: FingerRimMetrics,
    right: FingerRimMetrics,
    config: RimProbeConfig,
) -> tuple[np.ndarray, float, bool]:
    translations: list[np.ndarray] = []
    yaw_deltas: list[float] = []
    for metrics, target in ((left, config.left), (right, config.right)):
        if (
            metrics.contact
            and metrics.center_error_uv_px is not None
            and target.image_error_to_task_m_per_px is not None
        ):
            with np.errstate(over="ignore", invalid="ignore"):
                correction = (
                    target.image_error_to_task_m_per_px
                    @ metrics.center_error_uv_px
                )
            if np.isfinite(correction).all():
                translations.append(correction)
        if (
            metrics.contact
            and metrics.tangent_error_deg is not None
            and target.tangent_error_to_tcp_yaw is not None
        ):
            with np.errstate(over="ignore", invalid="ignore"):
                yaw_correction = target.tangent_error_to_tcp_yaw * math.radians(
                    metrics.tangent_error_deg
                )
            if math.isfinite(yaw_correction):
                yaw_deltas.append(yaw_correction)
    translations = [
        _clip_norm(correction, config.max_translation_step_m)
        for correction in translations
    ]
    if len(translations) >= 2:
        # Compare only corrections after each role's independent image-to-task
        # calibration.  Raw patch area, center, and shape are never compared
        # across the inner and outer fingers.
        left_norm = float(np.linalg.norm(translations[0]))
        right_norm = float(np.linalg.norm(translations[1]))
        directions_disagree = False
        if left_norm > 1e-12 and right_norm > 1e-12:
            cosine = float(
                np.dot(translations[0], translations[1]) / (left_norm * right_norm)
            )
            directions_disagree = cosine < config.min_translation_agreement_cosine
        if (
            directions_disagree
            or np.linalg.norm(translations[0] - translations[1])
            > config.max_translation_disagreement_m
        ):
            translations = []
    yaw_deltas = [
        float(np.clip(value, -config.max_yaw_step_rad, config.max_yaw_step_rad))
        for value in yaw_deltas
    ]
    if len(yaw_deltas) >= 2:
        opposite_sign = yaw_deltas[0] * yaw_deltas[1] < 0.0
        if (
            opposite_sign
            or abs(yaw_deltas[0] - yaw_deltas[1])
            > config.max_yaw_disagreement_rad
        ):
            yaw_deltas = []
    if translations:
        translation = np.median(np.stack(translations), axis=0)
    else:
        translation = np.zeros(3, dtype=float)
    yaw = float(np.median(yaw_deltas)) if yaw_deltas else 0.0
    mapped = float(np.linalg.norm(translation)) > 1e-12 or abs(yaw) > 1e-12
    return translation, yaw, mapped


def _single_contact_adjustment(
    target: FingerRimTarget, config: RimProbeConfig
) -> tuple[np.ndarray, bool]:
    if target.toward_sensor_task is None:
        return np.zeros(3, dtype=float), False
    translation = target.toward_sensor_task * config.recenter_step_m
    return _clip_norm(translation, config.max_translation_step_m), True


def _matches_role_template(
    metrics: FingerRimMetrics, target: FingerRimTarget
) -> bool:
    """Match one finger only against its configured inner/outer template."""

    return bool(
        metrics.contact
        and metrics.patch_area_px <= target.max_patch_area_px
        and metrics.significant_components >= 1
        and metrics.significant_components <= target.max_significant_components
        and metrics.roi_fraction >= target.min_roi_fraction
        and metrics.center_error_uv_px is not None
        and np.linalg.norm(metrics.center_error_uv_px) <= target.max_center_error_px
        and metrics.tangent_error_deg is not None
        and abs(metrics.tangent_error_deg) <= target.max_tangent_error_deg
        and metrics.elongation is not None
        and metrics.elongation >= target.min_elongation
    )


def _role_match_predicates(
    config: RimProbeConfig,
    *,
    left_matches: bool,
    right_matches: bool,
) -> set[str]:
    """Return side-bound and role-bound predicates without cross-side comparison."""

    matches: set[str] = set()
    for side, target, matched in (
        ("left", config.left, left_matches),
        ("right", config.right, right_matches),
    ):
        if matched:
            role = target.contact_role.value
            matches.add(f"{side}_{role}_rim_match")
            matches.add(f"{role}_rim_match")
    return matches


def _evaluate_rim_probe_impl(
    left_window: FingerTactileWindow,
    right_window: FingerTactileWindow,
    config: RimProbeConfig,
    *,
    now_s: float,
    not_before_s: float | None = None,
) -> RimProbeEvaluation:
    """Evaluate one settled tactile window and recommend one bounded next action.

    This is a windowed task-level assessment, not a force servo. The caller
    must use low Hand-E speed/force limits and wait for a command to stop.
    Passing that command's completion time as ``not_before_s`` prevents older
    samples from entering the decision window.
    """

    now_s = float(now_s)
    if not math.isfinite(now_s):
        raise ValueError("now_s must be finite")
    if not_before_s is not None:
        not_before_s = float(not_before_s)
        if not math.isfinite(not_before_s) or not_before_s > now_s:
            raise ValueError("not_before_s must be finite and no later than now_s")
    if left_window.calibration_id != config.left.calibration_id:
        return _invalid_evaluation("left tactile calibration_id does not match profile")
    if right_window.calibration_id != config.right.calibration_id:
        return _invalid_evaluation("right tactile calibration_id does not match profile")

    fresh: dict[str, _FreshWindow] = {}
    force_metrics: dict[str, FingerRimMetrics] = {}
    force_errors: list[str] = []
    for side, window, target in (
        ("left", left_window.force6d, config.left),
        ("right", right_window.force6d, config.right),
    ):
        name = f"{side}.force6d"
        result, error = _fresh_unique_window(
            window,
            minimum_samples=config.min_unique_force_samples,
            now_s=now_s,
            max_age_s=config.max_age_s,
            not_before_s=not_before_s,
            channel_name=name,
        )
        if error is not None or result is None:
            force_errors.append(error or f"{name} is invalid")
            continue
        fresh[name] = result
        try:
            force_metrics[side] = _force_only_metrics(result, target)
        except ValueError as exc:
            force_errors.append(f"{name}: {exc}")

    overloaded_sides = [
        side
        for side, target in (("left", config.left), ("right", config.right))
        if side in force_metrics
        and float(force_metrics[side].normal_load_peak_n) > target.normal_load_band_n[1]
    ]
    if overloaded_sides:
        evidence_key = _evidence_key(
            *(
                fresh[f"{side}.force6d"]
                for side in ("left", "right")
                if f"{side}.force6d" in fresh
            )
        )
        return RimProbeEvaluation(
            state=RimProbeState.OVERLOAD,
            confidence=1.0,
            left=force_metrics.get("left", _empty_metrics()),
            right=force_metrics.get("right", _empty_metrics()),
            predicates=frozenset({"overload"}),
            reasons=(
                f"fresh force peak exceeded the fragile ceiling on "
                f"{', '.join(overloaded_sides)}",
            ),
            recommendation=_zero_adjustment(ProbeAction.RELEASE, config.release_step_m),
            evidence_key=evidence_key,
        )
    unstable_force_sides = [
        side
        for side, target in (("left", config.left), ("right", config.right))
        if side in force_metrics
        and float(force_metrics[side].normal_load_range_n)
        > target.max_normal_load_range_n
    ]
    if unstable_force_sides:
        evidence_key = _evidence_key(
            *(
                fresh[f"{side}.force6d"]
                for side in ("left", "right")
                if f"{side}.force6d" in fresh
            )
        )
        return RimProbeEvaluation(
            state=RimProbeState.UNSTABLE,
            confidence=0.0,
            left=force_metrics.get("left", _empty_metrics()),
            right=force_metrics.get("right", _empty_metrics()),
            predicates=frozenset(),
            reasons=(
                f"normal load did not settle on {', '.join(unstable_force_sides)}",
            ),
            recommendation=_zero_adjustment(ProbeAction.RELEASE, config.release_step_m),
            evidence_key=evidence_key,
        )
    if force_errors:
        return _invalid_evaluation("; ".join(force_errors))

    for side, window in (("left", left_window.diff), ("right", right_window.diff)):
        name = f"{side}.diff"
        result, error = _fresh_unique_window(
            window,
            minimum_samples=config.min_unique_image_samples,
            now_s=now_s,
            max_age_s=config.max_age_s,
            not_before_s=not_before_s,
            channel_name=name,
        )
        if error is not None or result is None:
            return _invalid_evaluation(error or f"{name} is invalid")
        fresh[name] = result

    depths: dict[str, _FreshWindow | None] = {"left": None, "right": None}
    for side, window, target in (
        ("left", left_window.depth, config.left),
        ("right", right_window.depth, config.right),
    ):
        if target.max_indentation_m is None:
            continue
        if window is None:
            return _invalid_evaluation(f"{side}.depth is required by the profile")
        result, error = _fresh_unique_window(
            window,
            minimum_samples=config.min_unique_image_samples,
            now_s=now_s,
            max_age_s=config.max_age_s,
            not_before_s=not_before_s,
            channel_name=f"{side}.depth",
        )
        if error is not None or result is None:
            return _invalid_evaluation(error or f"{side}.depth is invalid")
        depths[side] = result

    endpoints = [window.last_capture_time_s for window in fresh.values()]
    endpoints.extend(
        depth.last_capture_time_s for depth in depths.values() if depth is not None
    )
    if max(endpoints) - min(endpoints) > config.max_pair_skew_s:
        return _invalid_evaluation(
            f"required tactile channels are desynchronized by "
            f"{max(endpoints) - min(endpoints):.3f}s"
        )

    try:
        left = _finger_metrics(
            fresh["left.diff"], fresh["left.force6d"], depths["left"], config.left
        )
        right = _finger_metrics(
            fresh["right.diff"], fresh["right.force6d"], depths["right"], config.right
        )
    except ValueError as exc:
        return _invalid_evaluation(f"tactile channel shape is invalid: {exc}")
    evidence_windows = [
        fresh[name]
        for name in ("left.diff", "right.diff", "left.force6d", "right.force6d")
    ]
    evidence_windows.extend(
        depths[side] for side in ("left", "right") if depths[side] is not None
    )
    evidence_key = _evidence_key(*evidence_windows)
    predicates = {"sensor_healthy"}

    left_load = float(left.normal_load_n)
    right_load = float(right.normal_load_n)
    if (
        float(left.normal_load_peak_n) > config.left.normal_load_band_n[1]
        or float(right.normal_load_peak_n) > config.right.normal_load_band_n[1]
    ):
        return RimProbeEvaluation(
            state=RimProbeState.OVERLOAD,
            confidence=1.0,
            left=left,
            right=right,
            predicates=frozenset(predicates | {"overload"}),
            reasons=("at least one finger exceeded its fragile normal-load ceiling",),
            recommendation=_zero_adjustment(ProbeAction.RELEASE, config.release_step_m),
            evidence_key=evidence_key,
        )

    for side, metrics, target in (
        ("left", left, config.left),
        ("right", right, config.right),
    ):
        if (
            target.max_indentation_m is not None
            and metrics.indentation_p95_m is not None
            and metrics.indentation_p95_m > target.max_indentation_m
        ):
            return RimProbeEvaluation(
                state=RimProbeState.EXCESS_DEFORMATION,
                confidence=1.0,
                left=left,
                right=right,
                predicates=frozenset(predicates | {"cup_deformation_budget_exceeded"}),
                reasons=(f"{side} indentation exceeded the calibrated budget",),
                recommendation=_zero_adjustment(
                    ProbeAction.RELEASE, config.release_step_m
                ),
                evidence_key=evidence_key,
            )

    inconsistent_force_sides = [
        side
        for side, metrics, target in (
            ("left", left, config.left),
            ("right", right, config.right),
        )
        if float(metrics.normal_load_n) < -target.normal_load_band_n[0]
    ]
    if inconsistent_force_sides:
        return RimProbeEvaluation(
            state=RimProbeState.CONTACT_UNRESOLVED,
            confidence=0.0,
            left=left,
            right=right,
            predicates=frozenset(predicates),
            reasons=(
                f"inward normal-load sign conflicts with the calibration on "
                f"{', '.join(inconsistent_force_sides)}",
            ),
            recommendation=_zero_adjustment(ProbeAction.RELEASE, config.release_step_m),
            evidence_key=evidence_key,
        )

    if not left.contact and not right.contact:
        force_contact_sides = [
            side
            for side, metrics, target in (
                ("left", left, config.left),
                ("right", right, config.right),
            )
            if float(metrics.normal_load_peak_n) >= target.normal_load_band_n[0]
        ]
        if force_contact_sides:
            return RimProbeEvaluation(
                state=RimProbeState.CONTACT_UNRESOLVED,
                confidence=0.0,
                left=left,
                right=right,
                predicates=frozenset(predicates),
                reasons=(
                    "force indicates contact but no persistent tactile image patch exists",
                ),
                recommendation=_zero_adjustment(
                    ProbeAction.RELEASE, config.release_step_m
                ),
                evidence_key=evidence_key,
            )
        if left.contact_frame_fraction > 0.0 or right.contact_frame_fraction > 0.0:
            return RimProbeEvaluation(
                state=RimProbeState.UNSTABLE,
                confidence=0.0,
                left=left,
                right=right,
                predicates=frozenset(predicates),
                reasons=("intermittent image contact did not persist across the window",),
                recommendation=_zero_adjustment(ProbeAction.WAIT),
                evidence_key=evidence_key,
            )
        return RimProbeEvaluation(
            state=RimProbeState.NO_CONTACT,
            confidence=1.0,
            left=left,
            right=right,
            predicates=frozenset(predicates),
            reasons=("neither tactile pad has a persistent contact patch",),
            recommendation=_zero_adjustment(ProbeAction.CLOSE_STEP, -config.close_step_m),
            evidence_key=evidence_key,
        )

    if left.contact != right.contact:
        contacting_left = left.contact
        target = config.left if contacting_left else config.right
        noncontact_metrics = right if contacting_left else left
        noncontact_target = config.right if contacting_left else config.left
        if (
            noncontact_metrics.normal_load_peak_n is not None
            and noncontact_metrics.normal_load_peak_n
            >= noncontact_target.normal_load_band_n[0]
        ):
            return RimProbeEvaluation(
                state=RimProbeState.CONTACT_UNRESOLVED,
                confidence=0.0,
                left=left,
                right=right,
                predicates=frozenset(predicates),
                reasons=(
                    "image and force disagree on the nominally non-contacting finger",
                ),
                recommendation=_zero_adjustment(
                    ProbeAction.RELEASE, config.release_step_m
                ),
                evidence_key=evidence_key,
            )
        translation, mapped = _single_contact_adjustment(target, config)
        side = "left" if contacting_left else "right"
        missing = "right" if contacting_left else "left"
        action = ProbeAction.RELEASE_THEN_ADJUST if mapped else ProbeAction.RELEASE
        reasons = [f"{side} contacted before {missing}; unload before TCP recentering"]
        if not mapped:
            reasons.append("no calibrated one-sided TCP correction is available")
        return RimProbeEvaluation(
            state=RimProbeState.LEFT_ONLY if contacting_left else RimProbeState.RIGHT_ONLY,
            confidence=1.0,
            left=left,
            right=right,
            predicates=frozenset(predicates | {f"contact_present_{side}"}),
            reasons=tuple(reasons),
            recommendation=ProbeAdjustment(
                action=action,
                jaw_opening_delta_m=config.release_step_m,
                proposed_tcp_translation_task_m=translation,
                proposed_tcp_yaw_delta_rad=0.0,
                release_confirmation_required=mapped,
            ),
            evidence_key=evidence_key,
        )

    unresolved = []
    multi_band_contact = False
    for side, metrics, target in (
        ("left", left, config.left),
        ("right", right, config.right),
    ):
        if metrics.significant_components < 1:
            unresolved.append(f"{side} contact is fragmented without a dominant patch")
        if metrics.patch_area_px > target.max_patch_area_px:
            unresolved.append(f"{side} contact patch is too large")
        if metrics.significant_components > target.max_significant_components:
            unresolved.append(f"{side} contact has ambiguous separated bands")
            multi_band_contact = True
    if unresolved:
        prior_correction = config.too_deep_recovery_translation_task_m
        use_too_deep_prior = multi_band_contact and prior_correction is not None
        unresolved_predicates = predicates | {"bilateral_contact"}
        if multi_band_contact:
            # This remains a task-level hypothesis.  Two optical bands can come
            # from one rolled rim, so downstream code may recover to a shallower
            # height but must not report a measured cup count.
            unresolved_predicates.add("multi_rim_or_too_deep_hypothesis")
        if use_too_deep_prior:
            unresolved_predicates.add("calibrated_shallower_recovery_available")
        return RimProbeEvaluation(
            state=RimProbeState.CONTACT_UNRESOLVED,
            confidence=0.0,
            left=left,
            right=right,
            predicates=frozenset(unresolved_predicates),
            reasons=tuple(
                unresolved
                + (
                    [
                        "task profile treats separated bands as a too-deep hypothesis; "
                        "unload before moving to the shallower rim-height hypothesis"
                    ]
                    if use_too_deep_prior
                    else []
                )
            ),
            recommendation=(
                ProbeAdjustment(
                    action=ProbeAction.RELEASE_THEN_ADJUST,
                    jaw_opening_delta_m=config.release_step_m,
                    proposed_tcp_translation_task_m=prior_correction,
                    proposed_tcp_yaw_delta_rad=0.0,
                    release_confirmation_required=True,
                )
                if use_too_deep_prior
                else _zero_adjustment(ProbeAction.RELEASE, config.release_step_m)
            ),
            evidence_key=evidence_key,
        )

    unstable = []
    for side, metrics, target in (
        ("left", left, config.left),
        ("right", right, config.right),
    ):
        if (
            metrics.center_jitter_px is None
            or metrics.center_jitter_px > target.max_center_jitter_px
        ):
            unstable.append(f"{side} contact center is not stable")
    if unstable:
        return RimProbeEvaluation(
            state=RimProbeState.UNSTABLE,
            confidence=0.0,
            left=left,
            right=right,
            predicates=frozenset(predicates | {"bilateral_contact"}),
            reasons=tuple(unstable),
            recommendation=_zero_adjustment(ProbeAction.WAIT),
            evidence_key=evidence_key,
        )

    left_role_match = _matches_role_template(left, config.left)
    right_role_match = _matches_role_template(right, config.right)
    role_match_predicates = _role_match_predicates(
        config,
        left_matches=left_role_match,
        right_matches=right_role_match,
    )
    if not (left_role_match and right_role_match):
        translation, yaw, mapped = _realignment(left, right, config)
        action = ProbeAction.RELEASE_THEN_ADJUST if mapped else ProbeAction.RELEASE
        mismatches = [
            f"{side} contact does not match its calibrated {target.contact_role.value}-rim template"
            for side, target, matched in (
                ("left", config.left, left_role_match),
                ("right", config.right, right_role_match),
            )
            if not matched
        ]
        reasons = mismatches
        if not mapped:
            reasons.append("no calibrated image-to-task correction is available")
        return RimProbeEvaluation(
            state=RimProbeState.ROLE_MISMATCH,
            confidence=min(left.geometry_score, right.geometry_score),
            left=left,
            right=right,
            predicates=frozenset(
                predicates | {"bilateral_contact"} | role_match_predicates
            ),
            reasons=tuple(reasons),
            recommendation=ProbeAdjustment(
                action=action,
                jaw_opening_delta_m=config.release_step_m,
                proposed_tcp_translation_task_m=translation,
                proposed_tcp_yaw_delta_rad=yaw,
                release_confirmation_required=mapped,
            ),
            evidence_key=evidence_key,
        )

    if (
        left_load < config.left.normal_load_band_n[0]
        or right_load < config.right.normal_load_band_n[0]
    ):
        return RimProbeEvaluation(
            state=RimProbeState.ROLE_MATCH_LOW_LOAD,
            confidence=min(left.geometry_score, right.geometry_score),
            left=left,
            right=right,
            predicates=frozenset(
                predicates | {"bilateral_contact"} | role_match_predicates
            ),
            reasons=(
                "inner- and outer-rim templates match independently, but at least "
                "one normal load is too low",
            ),
            recommendation=_zero_adjustment(ProbeAction.CLOSE_STEP, -config.close_step_m),
            evidence_key=evidence_key,
        )

    load_by_role = {
        config.left.contact_role: left_load,
        config.right.contact_role: right_load,
    }
    role_ratio_band = config.inner_to_outer_normal_load_ratio_band
    if role_ratio_band is not None:
        inner_to_outer_ratio = (
            load_by_role[RimContactRole.INNER]
            / max(load_by_role[RimContactRole.OUTER], 1e-9)
        )
        if not role_ratio_band[0] <= inner_to_outer_ratio <= role_ratio_band[1]:
            return RimProbeEvaluation(
                state=RimProbeState.ROLE_LOAD_OUT_OF_TOLERANCE,
                confidence=0.0,
                left=left,
                right=right,
                predicates=frozenset(
                    predicates | {"bilateral_contact"} | role_match_predicates
                ),
                reasons=(
                    "calibrated inner/outer normal-load ratio is outside its "
                    "independent force tolerance",
                ),
                recommendation=_zero_adjustment(
                    ProbeAction.RELEASE, config.release_step_m
                ),
                evidence_key=evidence_key,
            )
        predicates.add("inner_outer_normal_load_ratio_valid")

    predicates.update(
        {
            "bilateral_contact",
            "normal_load_in_fragile_band",
            "cup_deformation_budget_ok",
        }
    )
    predicates.update(role_match_predicates)
    confidence = min(left.geometry_score, right.geometry_score)
    return RimProbeEvaluation(
        state=RimProbeState.CAPTURE_CANDIDATE,
        confidence=max(0.0, confidence),
        left=left,
        right=right,
        predicates=frozenset(predicates),
        reasons=(
            "one fresh window independently matches the inner- and outer-rim profiles",
        ),
        recommendation=_zero_adjustment(ProbeAction.WAIT),
        evidence_key=evidence_key,
        consecutive_good_windows=1,
    )


def evaluate_rim_probe(
    left_window: FingerTactileWindow,
    right_window: FingerTactileWindow,
    config: RimProbeConfig,
    *,
    now_s: float,
    not_before_s: float | None = None,
    probe_context_id: str | None = None,
) -> RimProbeEvaluation:
    """Evaluate a settled window and attach authorization provenance.

    ``probe_context_id`` should identify the profile version, probe episode, and
    held TCP pose. It is optional for shadow evaluation, but a missing value can
    never pass the stability or TCP-adjustment authorization gates.
    """

    if probe_context_id is not None and not probe_context_id:
        raise ValueError("probe_context_id must be non-empty when provided")
    result = _evaluate_rim_probe_impl(
        left_window,
        right_window,
        config,
        now_s=now_s,
        not_before_s=not_before_s,
    )
    return replace(
        result,
        probe_context_id=probe_context_id,
        profile_id=config.profile_id,
        calibration_ids=(
            left_window.calibration_id,
            right_window.calibration_id,
        ),
        contact_roles=(config.left.contact_role, config.right.contact_role),
        evaluated_at_monotonic_s=float(now_s),
        adjustment_authorization_ttl_s=config.adjustment_authorization_ttl_s,
        evidence_not_before_monotonic_s=not_before_s,
    )


class RimProbeStabilityGate:
    """Require distinct, contiguous windows from one probe context."""

    def __init__(
        self,
        required_good_windows: int = 3,
        *,
        max_window_gap_s: float = 0.5,
    ) -> None:
        if required_good_windows <= 0:
            raise ValueError("required_good_windows must be positive")
        if not math.isfinite(max_window_gap_s) or max_window_gap_s <= 0.0:
            raise ValueError("max_window_gap_s must be positive and finite")
        self.required_good_windows = int(required_good_windows)
        self.max_window_gap_s = float(max_window_gap_s)
        self._last_evidence_key: tuple[float, ...] | None = None
        self._last_identity: (
            tuple[
                str,
                str,
                tuple[str, str],
                tuple[RimContactRole, RimContactRole],
            ]
            | None
        ) = None
        self._last_capture_time_s: float | None = None
        self._good_windows = 0

    def reset(self) -> None:
        self._last_evidence_key = None
        self._last_identity = None
        self._last_capture_time_s = None
        self._good_windows = 0

    @staticmethod
    def _identity(
        evaluation: RimProbeEvaluation,
    ) -> (
        tuple[
            str,
            str,
            tuple[str, str],
            tuple[RimContactRole, RimContactRole],
        ]
        | None
    ):
        if (
            not evaluation.probe_context_id
            or not evaluation.profile_id
            or evaluation.calibration_ids is None
            or len(evaluation.calibration_ids) != 2
            or not all(evaluation.calibration_ids)
            or evaluation.contact_roles is None
        ):
            return None
        try:
            roles = tuple(
                RimContactRole(value) for value in evaluation.contact_roles
            )
        except (TypeError, ValueError):
            return None
        if len(roles) != 2 or set(roles) != {
            RimContactRole.INNER,
            RimContactRole.OUTER,
        }:
            return None
        return (
            evaluation.probe_context_id,
            evaluation.profile_id,
            evaluation.calibration_ids,
            roles,
        )

    @staticmethod
    def _latest_capture_time(key: tuple[float, ...] | None) -> float | None:
        if key is None or len(key) == 0 or len(key) % 2 != 0:
            return None
        try:
            values = np.asarray(key, dtype=float)
        except (TypeError, ValueError):
            return None
        if (
            values.ndim != 1
            or values.size not in (8, 10, 12)
            or not np.isfinite(values).all()
        ):
            return None
        return float(np.max(values[1::2]))

    def update(self, evaluation: RimProbeEvaluation) -> RimProbeEvaluation:
        identity = self._identity(evaluation)
        if identity is None:
            self.reset()
            return replace(
                evaluation,
                reasons=("probe provenance is incomplete; stability cannot advance",),
                consecutive_good_windows=0,
            )

        if identity != self._last_identity:
            self.reset()
            self._last_identity = identity

        key = evaluation.evidence_key
        capture_time_s = self._latest_capture_time(key)
        evidence_valid = key is not None and capture_time_s is not None
        evidence_advanced = evidence_valid and self._last_evidence_key is None
        evidence_reversed = False
        evidence_layout_changed = False
        if evidence_valid and self._last_evidence_key is not None:
            if len(key) != len(self._last_evidence_key):
                evidence_layout_changed = True
            else:
                evidence_advanced = all(
                    current > previous
                    for current, previous in zip(
                        key, self._last_evidence_key, strict=True
                    )
                )
                evidence_reversed = any(
                    current < previous
                    for current, previous in zip(
                        key, self._last_evidence_key, strict=True
                    )
                )

        gap_exceeded = bool(
            capture_time_s is not None
            and self._last_capture_time_s is not None
            and capture_time_s - self._last_capture_time_s > self.max_window_gap_s
        )
        if gap_exceeded:
            self._last_evidence_key = None
            self._last_capture_time_s = None
            self._good_windows = 0
            evidence_advanced = True
            evidence_reversed = False
            evidence_layout_changed = False

        if evaluation.state is RimProbeState.CAPTURE_CANDIDATE:
            required_predicates = {
                "sensor_healthy",
                "bilateral_contact",
                "inner_rim_match",
                "outer_rim_match",
                "normal_load_in_fragile_band",
                "cup_deformation_budget_ok",
            }
            assert evaluation.contact_roles is not None
            roles = tuple(
                RimContactRole(value) for value in evaluation.contact_roles
            )
            required_predicates.update(
                {
                    f"left_{roles[0].value}_rim_match",
                    f"right_{roles[1].value}_rim_match",
                }
            )
            if (
                not required_predicates.issubset(evaluation.predicates)
                or evaluation.recommendation.action is not ProbeAction.WAIT
            ):
                self._last_evidence_key = None
                self._last_capture_time_s = None
                self._good_windows = 0
                return replace(
                    evaluation,
                    reasons=("capture candidate is internally inconsistent",),
                    consecutive_good_windows=0,
                )
            if not evidence_valid or evidence_reversed or evidence_layout_changed:
                self._last_evidence_key = key if evidence_valid else None
                self._last_capture_time_s = capture_time_s
                self._good_windows = 0
                reason = (
                    "evidence layout changed during stability evaluation"
                    if evidence_layout_changed
                    else "evidence timestamps are invalid or moved backwards"
                )
                return replace(
                    evaluation,
                    reasons=(reason,),
                    consecutive_good_windows=0,
                )
            if evidence_advanced:
                self._good_windows += 1
                self._last_evidence_key = key
                self._last_capture_time_s = capture_time_s
            if self._good_windows >= self.required_good_windows:
                return replace(
                    evaluation,
                    state=RimProbeState.CAPTURED,
                    predicates=evaluation.predicates | {"rim_capture_verified"},
                    reasons=(
                        f"{self._good_windows} distinct fresh windows satisfy the profile",
                    ),
                    recommendation=_zero_adjustment(ProbeAction.ACCEPT),
                    consecutive_good_windows=self._good_windows,
                )
            if gap_exceeded:
                reason = "window gap exceeded the continuity limit; stability restarted"
            elif not evidence_advanced:
                reason = (
                    "not every required SDK/host evidence channel advanced; "
                    "stability did not advance"
                )
            else:
                reason = (
                    f"waiting for {self.required_good_windows} consecutive good windows"
                )
            return replace(
                evaluation,
                reasons=(reason,),
                consecutive_good_windows=self._good_windows,
            )

        self._good_windows = 0
        if evidence_valid:
            self._last_evidence_key = key
            self._last_capture_time_s = capture_time_s
        else:
            self._last_evidence_key = None
            self._last_capture_time_s = None
        return replace(evaluation, consecutive_good_windows=0)


def authorize_tcp_adjustment(
    proposal: RimProbeEvaluation,
    release_evaluation: RimProbeEvaluation,
    release_receipt: ReleaseCommandReceipt,
    config: RimProbeConfig,
    *,
    now_s: float,
) -> TcpAdjustmentAuthorization | None:
    """Authorize a proposed TCP correction only after a fresh release window.

    The task runner must execute the positive jaw-opening delta from ``proposal``
    first. It then evaluates samples acquired strictly after ``release_receipt``
    and calls this function immediately. A ``None`` result means that TCP motion
    remains unauthorized.
    """

    pending = proposal.recommendation
    expected_calibration_ids = (
        config.left.calibration_id,
        config.right.calibration_id,
    )
    expected_contact_roles = (
        config.left.contact_role,
        config.right.contact_role,
    )
    expected_evidence_length = 2 * (
        4
        + int(config.left.max_indentation_m is not None)
        + int(config.right.max_indentation_m is not None)
    )
    now_s = float(now_s)
    if (
        not math.isfinite(now_s)
        or pending.action is not ProbeAction.RELEASE_THEN_ADJUST
        or not pending.release_confirmation_required
        or not proposal.probe_context_id
        or proposal.probe_context_id != release_evaluation.probe_context_id
        or proposal.probe_context_id != release_receipt.probe_context_id
        or proposal.profile_id != config.profile_id
        or release_evaluation.profile_id != config.profile_id
        or proposal.calibration_ids != expected_calibration_ids
        or release_evaluation.calibration_ids != expected_calibration_ids
        or proposal.contact_roles != expected_contact_roles
        or release_evaluation.contact_roles != expected_contact_roles
        or proposal.evidence_key is None
        or release_evaluation.evidence_key is None
        or release_evaluation.state is not RimProbeState.NO_CONTACT
        or len(proposal.evidence_key) != expected_evidence_length
        or len(release_evaluation.evidence_key) != expected_evidence_length
    ):
        return None
    try:
        proposal_evidence = np.asarray(proposal.evidence_key, dtype=float)
        release_evidence = np.asarray(release_evaluation.evidence_key, dtype=float)
    except (TypeError, ValueError):
        return None
    if (
        proposal_evidence.ndim != 1
        or release_evidence.ndim != 1
        or not np.isfinite(proposal_evidence).all()
        or not np.isfinite(release_evidence).all()
    ):
        return None
    proposal_time = proposal.evaluated_at_monotonic_s
    release_time = release_evaluation.evaluated_at_monotonic_s
    proposal_ttl = proposal.adjustment_authorization_ttl_s
    release_ttl = release_evaluation.adjustment_authorization_ttl_s
    release_not_before = release_evaluation.evidence_not_before_monotonic_s
    receipt_time = release_receipt.completed_at_monotonic_s
    if (
        proposal_time is None
        or release_time is None
        or proposal_ttl is None
        or release_ttl is None
        or release_not_before is None
        or not math.isfinite(proposal_time)
        or not math.isfinite(release_time)
        or not math.isfinite(proposal_ttl)
        or not math.isfinite(release_ttl)
        or not math.isfinite(release_not_before)
        or receipt_time <= proposal_time
        or release_not_before < receipt_time
        or release_time <= release_not_before
        or now_s < release_time
        or proposal_ttl <= 0.0
        or release_ttl <= 0.0
        or not math.isclose(
            proposal_ttl,
            config.adjustment_authorization_ttl_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            release_ttl,
            config.adjustment_authorization_ttl_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or now_s >= proposal_time + proposal_ttl
    ):
        return None
    if not np.all(release_evidence > proposal_evidence):
        return None
    if not np.all(release_evidence[1::2] > receipt_time):
        return None
    translation = pending.proposed_tcp_translation_task_m
    yaw_delta = pending.proposed_tcp_yaw_delta_rad
    if (
        pending.jaw_opening_delta_m <= 0.0
        or pending.jaw_opening_delta_m > config.release_step_m + 1e-12
        or release_receipt.applied_jaw_opening_delta_m
        < pending.jaw_opening_delta_m - 1e-12
        or release_receipt.applied_jaw_opening_delta_m
        > config.release_step_m + 1e-12
        or not np.isfinite(translation).all()
        or float(np.linalg.norm(translation)) > config.max_translation_step_m + 1e-12
        or not math.isfinite(yaw_delta)
        or abs(yaw_delta) > config.max_yaw_step_rad + 1e-12
    ):
        return None
    adjustment = ProbeAdjustment(
        action=ProbeAction.APPLY_TCP_ADJUSTMENT,
        jaw_opening_delta_m=0.0,
        proposed_tcp_translation_task_m=translation.copy(),
        proposed_tcp_yaw_delta_rad=yaw_delta,
        release_confirmation_required=False,
    )
    token_material = repr(
        (
            release_receipt.command_id,
            proposal.probe_context_id,
            config.profile_id,
            expected_calibration_ids,
            expected_contact_roles,
            tuple(proposal_evidence),
            tuple(release_evidence),
            tuple(translation),
            yaw_delta,
        )
    ).encode("utf-8")
    authorization_id = hashlib.sha256(token_material).hexdigest()
    return TcpAdjustmentAuthorization(
        authorization_id=authorization_id,
        probe_context_id=proposal.probe_context_id,
        profile_id=config.profile_id,
        calibration_ids=expected_calibration_ids,
        contact_roles=expected_contact_roles,
        issued_at_monotonic_s=now_s,
        expires_at_monotonic_s=proposal_time + proposal_ttl,
        release_command_id=release_receipt.command_id,
        proposal_evidence_key=tuple(float(value) for value in proposal_evidence),
        release_evidence_key=tuple(float(value) for value in release_evidence),
        adjustment=adjustment,
    )


class TcpAdjustmentAuthorizationLedger:
    """Consume each short-lived TCP-adjustment authorization at most once.

    A real runner must persist consumed ids in its atomic stage journal so a
    process restart cannot replay a physical command.
    """

    def __init__(self, consumed_authorization_ids: Iterable[str] = ()) -> None:
        self._consumed = {str(value) for value in consumed_authorization_ids}

    def consume(
        self,
        authorization: TcpAdjustmentAuthorization,
        *,
        now_s: float,
        probe_context_id: str,
        profile_id: str,
        calibration_ids: tuple[str, str],
        contact_roles: tuple[RimContactRole, RimContactRole],
    ) -> ProbeAdjustment | None:
        now_s = float(now_s)
        if (
            not math.isfinite(now_s)
            or authorization.authorization_id in self._consumed
            or now_s < authorization.issued_at_monotonic_s
            or now_s >= authorization.expires_at_monotonic_s
            or probe_context_id != authorization.probe_context_id
            or profile_id != authorization.profile_id
            or calibration_ids != authorization.calibration_ids
            or contact_roles != authorization.contact_roles
        ):
            return None
        self._consumed.add(authorization.authorization_id)
        return authorization.adjustment


__all__ = [
    "FingerRimMetrics",
    "FingerRimTarget",
    "FingerTactileWindow",
    "ProbeAction",
    "ProbeAdjustment",
    "ReleaseCommandReceipt",
    "RimContactRole",
    "RimProbeConfig",
    "RimProbeEvaluation",
    "RimProbeStabilityGate",
    "RimProbeState",
    "TimedArrayWindow",
    "TcpAdjustmentAuthorization",
    "TcpAdjustmentAuthorizationLedger",
    "authorize_tcp_adjustment",
    "evaluate_rim_probe",
]
