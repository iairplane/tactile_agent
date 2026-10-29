"""Pure tactile primitives for the staged test-tube task.

The module consumes already acquired, synchronized GF220 windows and the
session-bound output of :class:`TestTubeContactGate`.  It owns no SDK, USB,
ROS, gripper, or robot handle and never emits a device command.  Its action is
only a semantic recommendation for a runner that applies its own motion and
budget limits.

Image-axis, slip, and insertion-jam observations are kept separate from their
runtime qualification.  A candidate is always available for logging, but it
is promoted to a predicate only when the corresponding immutable profile is
explicitly marked as qualified.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from .gf220_live import GF220ClassifierWindows
from .gf220_wrench_adapter import AdaptedGF220Wrenches
from .test_tube_contact_bootstrap import (
    TestTubeContactDecision,
    TestTubeContactProfile,
)
from .test_tube_tactile_supervisor import (
    compensated_residual,
    residual_contact_geometry,
)


class TestTubeTactileMode(StrEnum):
    GRASP = "grasp"
    HOLD = "hold"
    PIVOT = "pivot"
    RECLAMP = "reclamp"
    INSERT = "insert"
    RELEASE = "release"


class TestTubeTactileAction(StrEnum):
    HOLD_INVALID = "hold_invalid"
    WAIT = "wait"
    CLOSE_STEP = "close_step"
    GRASP_READY = "grasp_ready"
    PIVOT_OPEN_STEP = "pivot_open_step"
    RECLAMP_STEP = "reclamp_step"
    PIVOT_READY = "pivot_ready"
    RECLAMP_READY = "reclamp_ready"
    INSERT_STEP = "insert_step"
    RETRACT_REALIGN = "retract_realign"
    RELEASE_READY = "release_ready"


@dataclass(frozen=True)
class TestTubeFingerImageProfile:
    """One pad's calibrated image geometry and pad-to-TCP orientation."""

    serial: str
    calibration_id: str
    unloaded_warped: np.ndarray
    roi_mask: np.ndarray
    R_tcp_pad: np.ndarray
    residual_threshold: float
    min_contact_area_px: int
    min_orientation_elongation: float = 1.6
    min_orientation_concentration: float = 0.35
    min_contact_frame_fraction: float = 0.60

    def __post_init__(self) -> None:
        _identifier(self.serial, "serial")
        _identifier(self.calibration_id, "calibration_id")
        image = np.asarray(self.unloaded_warped)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("unloaded_warped must be a uint8 HxWx3 image")
        roi = np.asarray(self.roi_mask, dtype=bool)
        if roi.shape != image.shape[:2] or not roi.any():
            raise ValueError("roi_mask must be non-empty and match unloaded_warped")
        rotation = _rotation(self.R_tcp_pad, "R_tcp_pad")
        threshold = _positive(self.residual_threshold, "residual_threshold")
        if (
            isinstance(self.min_contact_area_px, bool)
            or not isinstance(self.min_contact_area_px, int)
            or self.min_contact_area_px < 1
        ):
            raise ValueError("min_contact_area_px must be a positive integer")
        elongation = _positive(
            self.min_orientation_elongation, "min_orientation_elongation"
        )
        if elongation <= 1.0:
            raise ValueError("min_orientation_elongation must exceed one")
        concentration = _fraction(
            self.min_orientation_concentration,
            "min_orientation_concentration",
        )
        contact_fraction = _fraction(
            self.min_contact_frame_fraction,
            "min_contact_frame_fraction",
        )
        image = np.array(image, copy=True)
        roi = np.array(roi, copy=True)
        image.setflags(write=False)
        roi.setflags(write=False)
        object.__setattr__(self, "unloaded_warped", image)
        object.__setattr__(self, "roi_mask", roi)
        object.__setattr__(self, "R_tcp_pad", rotation)
        object.__setattr__(self, "residual_threshold", threshold)
        object.__setattr__(self, "min_orientation_elongation", elongation)
        object.__setattr__(self, "min_orientation_concentration", concentration)
        object.__setattr__(self, "min_contact_frame_fraction", contact_fraction)


@dataclass(frozen=True)
class TestTubeInsertionJamProfile:
    """Qualified relative resistance gate in arbitrary calibrated signal units."""

    profile_id: str
    task_frame_id: str
    insertion_axis_task: np.ndarray
    reaction_sign: int
    resistance_baseline_signal: float
    resistance_noise_signal: float
    minimum_resistance_delta_signal: float
    runtime_qualified: bool = False
    resistance_noise_multiplier: float = 8.0
    minimum_progress_ratio: float = 0.25
    confirmation_windows: int = 2

    def __post_init__(self) -> None:
        _identifier(self.profile_id, "profile_id")
        _identifier(self.task_frame_id, "task_frame_id")
        axis = _unit_vector(self.insertion_axis_task, "insertion_axis_task")
        if self.reaction_sign not in {-1, 1}:
            raise ValueError("reaction_sign must be -1 or +1")
        baseline = _finite(self.resistance_baseline_signal, "resistance_baseline_signal")
        noise = _nonnegative(self.resistance_noise_signal, "resistance_noise_signal")
        minimum = _nonnegative(
            self.minimum_resistance_delta_signal,
            "minimum_resistance_delta_signal",
        )
        multiplier = _positive(
            self.resistance_noise_multiplier,
            "resistance_noise_multiplier",
        )
        ratio = _fraction(self.minimum_progress_ratio, "minimum_progress_ratio")
        if (
            isinstance(self.confirmation_windows, bool)
            or not isinstance(self.confirmation_windows, int)
            or self.confirmation_windows < 1
        ):
            raise ValueError("confirmation_windows must be a positive integer")
        if not isinstance(self.runtime_qualified, bool):
            raise TypeError("runtime_qualified must be boolean")
        object.__setattr__(self, "insertion_axis_task", axis)
        object.__setattr__(self, "resistance_baseline_signal", baseline)
        object.__setattr__(self, "resistance_noise_signal", noise)
        object.__setattr__(self, "minimum_resistance_delta_signal", minimum)
        object.__setattr__(self, "resistance_noise_multiplier", multiplier)
        object.__setattr__(self, "minimum_progress_ratio", ratio)

    @property
    def resistance_delta_threshold_signal(self) -> float:
        return max(
            self.minimum_resistance_delta_signal,
            self.resistance_noise_multiplier * self.resistance_noise_signal,
        )


@dataclass(frozen=True)
class TestTubeTactileControllerProfile:
    """Immutable controller policy and qualification boundary."""

    profile_id: str
    session_id: str
    contact_profile: TestTubeContactProfile
    fingers: tuple[TestTubeFingerImageProfile, TestTubeFingerImageProfile]
    retained_width_min_m: float
    retained_width_max_m: float
    axis_runtime_qualified: bool = False
    slip_runtime_qualified: bool = False
    jam: TestTubeInsertionJamProfile | None = None
    min_unique_image_samples: int = 4
    max_age_s: float = 0.25
    max_pair_skew_s: float = 0.08
    future_tolerance_s: float = 1e-6
    bilateral_confirmation_windows: int = 2
    target_axis_tolerance_deg: float = 5.0
    axis_confirmation_windows: int = 4
    max_pad_axis_disagreement_deg: float = 10.0
    slip_center_shift_threshold_px: float = 4.0
    slip_confirmation_windows: int = 2

    def __post_init__(self) -> None:
        _identifier(self.profile_id, "profile_id")
        _identifier(self.session_id, "session_id")
        if not isinstance(self.contact_profile, TestTubeContactProfile):
            raise TypeError("contact_profile must be TestTubeContactProfile")
        if self.contact_profile.session_id != self.session_id:
            raise ValueError("contact profile belongs to a different session")
        if not isinstance(self.fingers, tuple) or len(self.fingers) != 2:
            raise ValueError("fingers must contain exactly two image profiles")
        if any(not isinstance(item, TestTubeFingerImageProfile) for item in self.fingers):
            raise TypeError("fingers must contain TestTubeFingerImageProfile values")
        serials = {item.serial for item in self.fingers}
        contact_serials = {item.serial for item in self.contact_profile.fingers}
        if len(serials) != 2 or serials != contact_serials:
            raise ValueError("image and contact profiles must identify the same two serials")
        image_calibrations = {item.serial: item.calibration_id for item in self.fingers}
        contact_calibrations = {
            item.serial: item.calibration_id for item in self.contact_profile.fingers
        }
        if image_calibrations != contact_calibrations:
            raise ValueError("image and contact calibration IDs must match by serial")
        minimum_width = _nonnegative(self.retained_width_min_m, "retained_width_min_m")
        maximum_width = _positive(self.retained_width_max_m, "retained_width_max_m")
        if minimum_width >= maximum_width:
            raise ValueError("retained width range must be increasing")
        for name in ("axis_runtime_qualified", "slip_runtime_qualified"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.jam is not None and not isinstance(self.jam, TestTubeInsertionJamProfile):
            raise TypeError("jam must be TestTubeInsertionJamProfile or None")
        for name in (
            "min_unique_image_samples",
            "bilateral_confirmation_windows",
            "axis_confirmation_windows",
            "slip_confirmation_windows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "max_age_s",
            "max_pair_skew_s",
            "target_axis_tolerance_deg",
            "max_pad_axis_disagreement_deg",
            "slip_center_shift_threshold_px",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        future = _nonnegative(self.future_tolerance_s, "future_tolerance_s")
        object.__setattr__(self, "future_tolerance_s", future)
        object.__setattr__(self, "retained_width_min_m", minimum_width)
        object.__setattr__(self, "retained_width_max_m", maximum_width)


@dataclass(frozen=True)
class TestTubeFingerImagePrimitive:
    serial: str
    valid: bool
    contact_candidate: bool
    contact_frame_fraction: float
    patch_area_px: float | None
    patch_center_uv_px: tuple[float, float] | None
    patch_axis_deg: float | None
    elongation: float | None
    orientation_concentration: float | None
    residual_p99: float | None
    reason: str


@dataclass(frozen=True)
class TestTubeTactilePrimitives:
    contact_present: bool
    bilateral_contact: bool
    active_serials: tuple[str, ...]
    fingers: tuple[TestTubeFingerImagePrimitive, TestTubeFingerImagePrimitive]
    axis_valid: bool
    axis_tcp: tuple[float, float, float] | None
    axis_base: tuple[float, float, float] | None
    pad_axis_disagreement_deg: float | None
    target_axis_error_deg: float | None
    target_axis_candidate: bool
    maximum_patch_center_shift_px: float | None
    slip_candidate: bool
    insertion_progress_ratio: float | None
    insertion_resistance_signal: float | None
    insertion_resistance_delta_signal: float | None
    insertion_resistance_threshold_signal: float | None
    insertion_lateral_signal: float | None
    insertion_jam_candidate: bool


@dataclass(frozen=True)
class TestTubeTactileMemory:
    sequence: int = 0
    last_evidence_id: str | None = None
    last_sdk_time_s_by_serial: tuple[tuple[str, float], ...] = ()
    last_capture_time_s_by_serial: tuple[tuple[str, float], ...] = ()
    last_patch_center_by_serial: tuple[tuple[str, tuple[float, float]], ...] = ()
    last_axis_base: tuple[float, float, float] | None = None
    last_contact_present: bool = False
    last_bilateral_contact: bool = False
    bilateral_windows: int = 0
    target_axis_windows: int = 0
    slip_windows: int = 0
    jam_windows: int = 0

    def __post_init__(self) -> None:
        for name in (
            "sequence",
            "bilateral_windows",
            "target_axis_windows",
            "slip_windows",
            "jam_windows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.last_axis_base is not None:
            axis = _unit_vector(self.last_axis_base, "last_axis_base")
            object.__setattr__(self, "last_axis_base", tuple(float(v) for v in axis))
        for name in (
            "last_sdk_time_s_by_serial",
            "last_capture_time_s_by_serial",
        ):
            values = getattr(self, name)
            serials = [serial for serial, _ in values]
            if len(serials) != len(set(serials)):
                raise ValueError(f"{name} contains duplicate serials")
            for serial, timestamp in values:
                _identifier(serial, f"{name} serial")
                _nonnegative(timestamp, f"{name} timestamp")
        center_serials = [serial for serial, _ in self.last_patch_center_by_serial]
        if len(center_serials) != len(set(center_serials)):
            raise ValueError("last_patch_center_by_serial contains duplicate serials")
        for serial, center in self.last_patch_center_by_serial:
            _identifier(serial, "last_patch_center_by_serial serial")
            point = np.asarray(center, dtype=float)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError("last patch centers must be finite 2-vectors")


@dataclass(frozen=True)
class TestTubeTactileEvaluation:
    valid: bool
    mode: TestTubeTactileMode
    action: TestTubeTactileAction
    primitives: TestTubeTactilePrimitives
    promoted_predicates: tuple[str, ...]
    shadow_predicates: tuple[str, ...]
    reason: str
    evidence_id: str
    next_memory: TestTubeTactileMemory


def evaluate_test_tube_tactile(
    *,
    mode: TestTubeTactileMode | str,
    windows: GF220ClassifierWindows,
    contact: TestTubeContactDecision,
    profile: TestTubeTactileControllerProfile,
    memory: TestTubeTactileMemory = TestTubeTactileMemory(),
    now_s: float,
    T_base_tcp: np.ndarray | None = None,
    target_axis_base: np.ndarray | None = None,
    gripper_width_m: float | None = None,
    gripper_open_confirmed: bool = False,
    adapted_wrenches: AdaptedGF220Wrenches | None = None,
    commanded_axial_step_m: float | None = None,
    measured_axial_progress_m: float | None = None,
) -> TestTubeTactileEvaluation:
    """Evaluate one fresh tactile window without dispatching hardware commands."""

    mode = TestTubeTactileMode(mode)
    if not isinstance(windows, GF220ClassifierWindows):
        raise TypeError("windows must be GF220ClassifierWindows")
    if not isinstance(contact, TestTubeContactDecision):
        raise TypeError("contact must be TestTubeContactDecision")
    if not isinstance(profile, TestTubeTactileControllerProfile):
        raise TypeError("profile must be TestTubeTactileControllerProfile")
    if not isinstance(memory, TestTubeTactileMemory):
        raise TypeError("memory must be TestTubeTactileMemory")
    now = _finite(now_s, "now_s")
    if not isinstance(gripper_open_confirmed, bool):
        raise TypeError("gripper_open_confirmed must be boolean")

    invalid_reasons: list[str] = []
    width_value: float | None = None
    if gripper_width_m is not None:
        try:
            width_value = _nonnegative(gripper_width_m, "gripper_width_m")
        except ValueError as error:
            invalid_reasons.append(str(error))
    profile_by_serial = {item.serial: item for item in profile.fingers}
    if len(windows.serials) != 2 or set(windows.serials) != set(profile_by_serial):
        invalid_reasons.append("sensor serials do not match the controller profile")
    if windows.endpoint_skew_s > profile.max_pair_skew_s:
        invalid_reasons.append("left/right tactile window skew exceeds the profile")

    previous_sdk = dict(memory.last_sdk_time_s_by_serial)
    previous_host = dict(memory.last_capture_time_s_by_serial)
    image_primitives: list[TestTubeFingerImagePrimitive] = []
    endpoints_sdk: list[tuple[str, float]] = []
    endpoints_host: list[tuple[str, float]] = []
    for serial in sorted(profile_by_serial):
        if serial not in windows.serials:
            image_primitives.append(_invalid_finger(serial, "sensor window is missing"))
            continue
        source = windows.for_serial(serial)
        finger, sdk_endpoint, host_endpoint, reason = _evaluate_finger_image(
            source,
            profile_by_serial[serial],
            min_samples=profile.min_unique_image_samples,
            max_age_s=profile.max_age_s,
            future_tolerance_s=profile.future_tolerance_s,
            now_s=now,
            previous_sdk_time_s=previous_sdk.get(serial),
            previous_capture_time_s=previous_host.get(serial),
        )
        image_primitives.append(finger)
        if reason is not None:
            invalid_reasons.append(f"{serial}: {reason}")
        else:
            assert sdk_endpoint is not None and host_endpoint is not None
            endpoints_sdk.append((serial, sdk_endpoint))
            endpoints_host.append((serial, host_endpoint))

    contact_reason = _validate_contact_receipt(contact, profile, windows)
    if contact_reason is not None:
        invalid_reasons.append(contact_reason)
    active_serials = tuple(sorted(contact.active_serials)) if contact.valid else ()
    qualified_contact_serials = {
        item.serial
        for item in profile.contact_profile.fingers
        if item.contact_response_qualified
    }
    bilateral_contact = bool(
        contact.valid
        and contact.contact_present
        and set(active_serials) == set(profile_by_serial)
        and qualified_contact_serials == set(profile_by_serial)
    )

    rotation_base_tcp: np.ndarray | None = None
    target_axis: np.ndarray | None = None
    if T_base_tcp is not None:
        try:
            rotation_base_tcp = _transform_rotation(T_base_tcp, "T_base_tcp")
        except ValueError as error:
            invalid_reasons.append(str(error))
    if target_axis_base is not None:
        try:
            target_axis = _unit_vector(target_axis_base, "target_axis_base")
        except ValueError as error:
            invalid_reasons.append(str(error))
    if (
        mode in {TestTubeTactileMode.PIVOT, TestTubeTactileMode.RECLAMP}
        and profile.axis_runtime_qualified
    ):
        if rotation_base_tcp is None:
            invalid_reasons.append("qualified tactile axis requires T_base_tcp")
        if target_axis is None:
            invalid_reasons.append("qualified tactile axis requires target_axis_base")

    axis_tcp, axis_base, disagreement = _fuse_axis(
        tuple(image_primitives),
        profile_by_serial,
        rotation_base_tcp,
        memory.last_axis_base,
        bilateral_contact=bilateral_contact,
        max_disagreement_deg=profile.max_pad_axis_disagreement_deg,
    )
    axis_error = (
        _undirected_angle_deg(axis_base, target_axis)
        if axis_base is not None and target_axis is not None
        else None
    )
    axis_candidate = bool(
        axis_error is not None and axis_error <= profile.target_axis_tolerance_deg
    )

    previous_centers = dict(memory.last_patch_center_by_serial)
    shifts = []
    for finger in image_primitives:
        if finger.patch_center_uv_px is None or finger.serial not in previous_centers:
            continue
        shifts.append(
            float(
                np.linalg.norm(
                    np.asarray(finger.patch_center_uv_px)
                    - np.asarray(previous_centers[finger.serial])
                )
            )
        )
    maximum_shift = max(shifts) if shifts else None
    lost_one_pad = memory.last_bilateral_contact and not bilateral_contact
    incoherent_axis = (
        disagreement is not None
        and disagreement > profile.max_pad_axis_disagreement_deg
    )
    slip_candidate = bool(
        contact.contact_present
        and maximum_shift is not None
        and maximum_shift >= profile.slip_center_shift_threshold_px
        and (lost_one_pad or incoherent_axis)
    )

    progress_ratio: float | None = None
    resistance: float | None = None
    resistance_delta: float | None = None
    resistance_threshold: float | None = None
    lateral_signal: float | None = None
    jam_candidate = False
    if mode is TestTubeTactileMode.INSERT:
        (
            progress_ratio,
            resistance,
            resistance_delta,
            resistance_threshold,
            lateral_signal,
            jam_candidate,
            insertion_error,
        ) = _evaluate_insertion(
            profile.jam,
            adapted_wrenches,
            windows,
            commanded_axial_step_m,
            measured_axial_progress_m,
            now_s=now,
            max_age_s=profile.max_age_s,
            future_tolerance_s=profile.future_tolerance_s,
        )
        if insertion_error is not None:
            invalid_reasons.append(insertion_error)

    primitives = TestTubeTactilePrimitives(
        contact_present=bool(contact.valid and contact.contact_present),
        bilateral_contact=bilateral_contact,
        active_serials=active_serials,
        fingers=(image_primitives[0], image_primitives[1]),
        axis_valid=axis_base is not None,
        axis_tcp=_tuple3(axis_tcp),
        axis_base=_tuple3(axis_base),
        pad_axis_disagreement_deg=disagreement,
        target_axis_error_deg=axis_error,
        target_axis_candidate=axis_candidate,
        maximum_patch_center_shift_px=maximum_shift,
        slip_candidate=slip_candidate,
        insertion_progress_ratio=progress_ratio,
        insertion_resistance_signal=resistance,
        insertion_resistance_delta_signal=resistance_delta,
        insertion_resistance_threshold_signal=resistance_threshold,
        insertion_lateral_signal=lateral_signal,
        insertion_jam_candidate=jam_candidate,
    )

    if invalid_reasons:
        return _invalid_evaluation(
            mode,
            primitives,
            memory,
            "; ".join(sorted(set(invalid_reasons))),
        )

    bilateral_windows = memory.bilateral_windows + 1 if bilateral_contact else 0
    target_windows = memory.target_axis_windows + 1 if axis_candidate else 0
    slip_windows = memory.slip_windows + 1 if slip_candidate else 0
    jam_windows = memory.jam_windows + 1 if jam_candidate else 0
    if mode not in {TestTubeTactileMode.PIVOT, TestTubeTactileMode.RECLAMP}:
        target_windows = 0
    if mode is not TestTubeTactileMode.INSERT:
        jam_windows = 0

    grasp_ready = bool(
        bilateral_windows >= profile.bilateral_confirmation_windows
        and width_value is not None
        and profile.retained_width_min_m
        <= width_value
        <= profile.retained_width_max_m
    )
    axis_ready = bool(
        profile.axis_runtime_qualified
        and target_windows >= profile.axis_confirmation_windows
    )
    slip_confirmed = bool(
        profile.slip_runtime_qualified
        and slip_windows >= profile.slip_confirmation_windows
    )
    jam_confirmed = bool(
        profile.jam is not None
        and profile.jam.runtime_qualified
        and jam_windows >= profile.jam.confirmation_windows
    )

    promoted: set[str] = set()
    shadow: set[str] = set()
    if primitives.contact_present:
        promoted.add("contact_present")
    if bilateral_contact:
        promoted.add("bilateral_contact")
    if grasp_ready:
        promoted.add("grasp_established")
    if axis_candidate:
        (promoted if axis_ready else shadow).add("tactile_relative_angle_reached")
    if slip_candidate:
        (promoted if slip_confirmed else shadow).add("slip_detected")
    if jam_candidate:
        (promoted if jam_confirmed else shadow).add("insertion_jam")
    elif (
        mode is TestTubeTactileMode.INSERT
        and profile.jam is not None
        and profile.jam.runtime_qualified
    ):
        promoted.add("no_jam")

    action, reason = _semantic_action(
        mode=mode,
        primitives=primitives,
        profile=profile,
        grasp_ready=grasp_ready,
        axis_ready=axis_ready,
        slip_confirmed=slip_confirmed,
        jam_confirmed=jam_confirmed,
        gripper_open_confirmed=gripper_open_confirmed,
    )
    evidence_id = _evidence_digest(
        profile=profile,
        mode=mode,
        memory=memory,
        endpoints_sdk=endpoints_sdk,
        endpoints_host=endpoints_host,
        primitives=primitives,
        promoted=promoted,
        shadow=shadow,
    )
    centers = tuple(
        (finger.serial, finger.patch_center_uv_px)
        for finger in image_primitives
        if finger.patch_center_uv_px is not None
    )
    next_memory = TestTubeTactileMemory(
        sequence=memory.sequence + 1,
        last_evidence_id=evidence_id,
        last_sdk_time_s_by_serial=tuple(sorted(endpoints_sdk)),
        last_capture_time_s_by_serial=tuple(sorted(endpoints_host)),
        last_patch_center_by_serial=tuple(sorted(centers)),
        last_axis_base=_tuple3(axis_base) or memory.last_axis_base,
        last_contact_present=primitives.contact_present,
        last_bilateral_contact=bilateral_contact,
        bilateral_windows=bilateral_windows,
        target_axis_windows=target_windows,
        slip_windows=slip_windows,
        jam_windows=jam_windows,
    )
    return TestTubeTactileEvaluation(
        valid=True,
        mode=mode,
        action=action,
        primitives=primitives,
        promoted_predicates=tuple(sorted(promoted)),
        shadow_predicates=tuple(sorted(shadow)),
        reason=reason,
        evidence_id=evidence_id,
        next_memory=next_memory,
    )


def _evaluate_finger_image(
    source: Any,
    profile: TestTubeFingerImageProfile,
    *,
    min_samples: int,
    max_age_s: float,
    future_tolerance_s: float,
    now_s: float,
    previous_sdk_time_s: float | None,
    previous_capture_time_s: float | None,
) -> tuple[TestTubeFingerImagePrimitive, float | None, float | None, str | None]:
    if source.calibration_id != profile.calibration_id:
        reason = "calibration_id changed"
        return _invalid_finger(profile.serial, reason), None, None, reason
    warped = source.warped
    diff = source.diff
    if not (
        np.array_equal(warped.sdk_time_s, diff.sdk_time_s)
        and np.array_equal(warped.capture_monotonic_s, diff.capture_monotonic_s)
        and np.array_equal(warped.valid, diff.valid)
    ):
        reason = "WARPED_IMG and DIFF_IMG frame identities differ"
        return _invalid_finger(profile.serial, reason), None, None, reason
    images = np.asarray(warped.values)
    if images.ndim != 4 or images.shape[1:] != profile.unloaded_warped.shape:
        reason = "WARPED_IMG shape does not match the image profile"
        return _invalid_finger(profile.serial, reason), None, None, reason
    if images.dtype != np.uint8:
        reason = "WARPED_IMG must be uint8"
        return _invalid_finger(profile.serial, reason), None, None, reason
    valid = np.asarray(warped.valid, dtype=bool)
    indices = np.flatnonzero(valid)
    if indices.size:
        _, unique = np.unique(warped.sdk_time_s[indices], return_index=True)
        indices = indices[np.sort(unique)]
    if indices.size < min_samples:
        reason = f"only {indices.size} fresh unique image samples"
        return _invalid_finger(profile.serial, reason), None, None, reason
    sdk = warped.sdk_time_s[indices]
    host = warped.capture_monotonic_s[indices]
    if np.any(np.diff(sdk) <= 0.0) or np.any(np.diff(host) <= 0.0):
        reason = "image timestamps must strictly increase"
        return _invalid_finger(profile.serial, reason), None, None, reason
    if host[0] < now_s - max_age_s:
        reason = "image window is stale"
        return _invalid_finger(profile.serial, reason), None, None, reason
    if host[-1] > now_s + future_tolerance_s:
        reason = "image window contains future samples"
        return _invalid_finger(profile.serial, reason), None, None, reason
    if previous_sdk_time_s is not None and sdk[0] <= previous_sdk_time_s:
        reason = "image window overlaps or replays SDK evidence"
        return _invalid_finger(profile.serial, reason), None, None, reason
    if previous_capture_time_s is not None and host[0] <= previous_capture_time_s:
        reason = "image window overlaps or replays host evidence"
        return _invalid_finger(profile.serial, reason), None, None, reason

    geometries = []
    residual_p99 = []
    for index in indices:
        residual = compensated_residual(
            images[index], profile.unloaded_warped, profile.roi_mask
        )
        residual_p99.append(float(np.quantile(residual[profile.roi_mask], 0.99)))
        geometries.append(
            residual_contact_geometry(
                residual,
                profile.roi_mask,
                residual_threshold=profile.residual_threshold,
                min_contact_area_px=profile.min_contact_area_px,
                min_orientation_elongation=profile.min_orientation_elongation,
                min_orientation_concentration=profile.min_orientation_concentration,
            )
        )
    contacts = [item for item in geometries if item.contact_candidate]
    oriented = [item for item in geometries if item.orientation_valid]
    contact_fraction = len(contacts) / len(geometries)
    contact_candidate = contact_fraction >= profile.min_contact_frame_fraction
    orientation_valid = (
        contact_candidate
        and len(oriented) / len(geometries) >= profile.min_contact_frame_fraction
    )
    center = None
    area = None
    elongation = None
    concentration = None
    if contacts:
        centers = [item.center_uv_px for item in contacts if item.center_uv_px is not None]
        if centers:
            median = np.median(np.asarray(centers, dtype=float), axis=0)
            center = (float(median[0]), float(median[1]))
        area = float(np.median([item.area_px for item in contacts]))
        elongations = [item.elongation for item in contacts if item.elongation is not None]
        concentrations = [
            item.orientation_concentration
            for item in contacts
            if item.orientation_concentration is not None
        ]
        elongation = float(np.median(elongations)) if elongations else None
        concentration = float(np.median(concentrations)) if concentrations else None
    angle = None
    if orientation_valid:
        try:
            angle = _undirected_mean_deg(
                [float(item.orientation_deg) for item in oriented]
            )
        except ValueError:
            orientation_valid = False
    primitive = TestTubeFingerImagePrimitive(
        serial=profile.serial,
        valid=True,
        contact_candidate=contact_candidate,
        contact_frame_fraction=contact_fraction,
        patch_area_px=area,
        patch_center_uv_px=center,
        patch_axis_deg=angle,
        elongation=elongation,
        orientation_concentration=concentration,
        residual_p99=float(np.median(residual_p99)),
        reason=(
            "elongated contact patch observed"
            if angle is not None
            else "no qualified elongated contact patch"
        ),
    )
    return primitive, float(sdk[-1]), float(host[-1]), None


def _validate_contact_receipt(
    decision: TestTubeContactDecision,
    profile: TestTubeTactileControllerProfile,
    windows: GF220ClassifierWindows,
) -> str | None:
    if not decision.valid:
        return f"contact gate invalid: {decision.reason}"
    state = decision.next_state
    if state.session_id != profile.session_id:
        return "contact receipt session_id mismatch"
    if state.profile_id != profile.contact_profile.profile_id:
        return "contact receipt profile_id mismatch"
    if state.contact_present != decision.contact_present:
        return "contact receipt state disagrees with the decision"
    if len(decision.active_serials) != len(set(decision.active_serials)):
        return "contact receipt contains duplicate active serials"
    if not decision.contact_present and decision.active_serials:
        return "absent contact receipt cannot name active serials"
    if not set(decision.active_serials).issubset(windows.serials):
        return "contact receipt names an unknown active serial"
    sdk_endpoints = dict(state.last_sdk_time_s_by_serial)
    host_endpoints = dict(state.last_capture_time_s_by_serial)
    if set(sdk_endpoints) != set(windows.serials) or set(host_endpoints) != set(
        windows.serials
    ):
        return "contact receipt endpoint serials mismatch"
    contact_calibrations = {
        item.serial: item for item in profile.contact_profile.fingers
    }
    for serial in windows.serials:
        force = windows.for_serial(serial).force6d
        if force is None:
            return f"{serial}: FORCE6D is missing"
        valid = np.flatnonzero(force.valid)
        if valid.size == 0:
            return f"{serial}: FORCE6D has no valid sample"
        endpoint = int(valid[-1])
        if not math.isclose(
            sdk_endpoints[serial], float(force.sdk_time_s[endpoint]), abs_tol=1e-9
        ) or not math.isclose(
            host_endpoints[serial],
            float(force.capture_monotonic_s[endpoint]),
            abs_tol=1e-9,
        ):
            return "contact receipt is not bound to the supplied tactile window"
        calibration = contact_calibrations[serial]
        if serial in decision.active_serials and not calibration.contact_response_qualified:
            return "contact receipt activates an unqualified tactile pad"
        if calibration.contact_response_qualified:
            values = np.asarray(force.values[force.valid, 2], dtype=float)
            delta = abs(float(np.median(values)) - calibration.unloaded_fz_sensor_units)
            if (
                serial in decision.active_serials
                and delta < profile.contact_profile.retain_threshold_sensor_units
            ):
                return "contact receipt active serial is below the retain threshold"
            if (
                serial not in decision.active_serials
                and delta >= profile.contact_profile.engage_threshold_sensor_units
            ):
                return "contact receipt omits a serial above the engage threshold"
    return None


def _fuse_axis(
    fingers: tuple[TestTubeFingerImagePrimitive, ...],
    profiles: dict[str, TestTubeFingerImageProfile],
    R_base_tcp: np.ndarray | None,
    previous_axis_base: tuple[float, float, float] | None,
    *,
    bilateral_contact: bool,
    max_disagreement_deg: float,
) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    if not bilateral_contact or R_base_tcp is None:
        return None, None, None
    vectors = []
    weights = []
    for finger in fingers:
        if not finger.valid or finger.patch_axis_deg is None:
            continue
        angle = math.radians(finger.patch_axis_deg)
        pad_axis = np.array([math.cos(angle), math.sin(angle), 0.0])
        vector = profiles[finger.serial].R_tcp_pad @ pad_axis
        vectors.append(vector / np.linalg.norm(vector))
        weights.append(
            max(float(finger.orientation_concentration or 0.0), 1e-6)
            * max(float(finger.contact_frame_fraction), 1e-6)
        )
    if len(vectors) != 2:
        return None, None, None
    disagreement = _undirected_angle_deg(vectors[0], vectors[1])
    if disagreement > max_disagreement_deg:
        return None, None, disagreement
    if float(np.dot(vectors[0], vectors[1])) < 0.0:
        vectors[1] = -vectors[1]
    fused = weights[0] * vectors[0] + weights[1] * vectors[1]
    if np.linalg.norm(fused) <= 1e-9:
        return None, None, disagreement
    fused /= np.linalg.norm(fused)
    base = R_base_tcp @ fused
    base /= np.linalg.norm(base)
    if previous_axis_base is not None and float(
        np.dot(base, np.asarray(previous_axis_base, dtype=float))
    ) < 0.0:
        fused = -fused
        base = -base
    return fused, base, disagreement


def _evaluate_insertion(
    profile: TestTubeInsertionJamProfile | None,
    wrenches: AdaptedGF220Wrenches | None,
    source_windows: GF220ClassifierWindows,
    commanded_step_m: float | None,
    measured_progress_m: float | None,
    *,
    now_s: float,
    max_age_s: float,
    future_tolerance_s: float,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    bool,
    str | None,
]:
    if profile is None:
        return None, None, None, None, None, False, "insertion jam profile is missing"
    if wrenches is None:
        return None, None, None, None, None, False, "adapted insertion wrenches are missing"
    if commanded_step_m is None or measured_progress_m is None:
        return None, None, None, None, None, False, "insertion progress is missing"
    try:
        command = _positive(commanded_step_m, "commanded_axial_step_m")
        progress = _nonnegative(measured_progress_m, "measured_axial_progress_m")
    except ValueError as error:
        return None, None, None, None, None, False, str(error)
    if wrenches.profile_id != profile.profile_id:
        return None, None, None, None, None, False, "insertion wrench profile_id mismatch"
    if wrenches.task_frame_id != profile.task_frame_id:
        return None, None, None, None, None, False, "insertion wrench task frame mismatch"
    adapted = (
        ("left", wrenches.left_serial, wrenches.left),
        ("right", wrenches.right_serial, wrenches.right),
    )
    if {wrenches.left_serial, wrenches.right_serial} != set(source_windows.serials):
        return None, None, None, None, None, False, "insertion wrench serials mismatch"
    for name, serial, window in adapted:
        valid = np.flatnonzero(window.valid)
        if valid.size == 0:
            return None, None, None, None, None, False, f"{name} wrench has no valid samples"
        source_force = source_windows.for_serial(serial).force6d
        if source_force is None:
            return None, None, None, None, None, False, f"{serial}: FORCE6D is missing"
        if not (
            np.array_equal(window.sdk_time_s, source_force.sdk_time_s)
            and np.array_equal(
                window.capture_monotonic_s,
                source_force.capture_monotonic_s,
            )
            and np.array_equal(window.valid, source_force.valid)
        ):
            return (
                None,
                None,
                None,
                None,
                None,
                False,
                "adapted insertion wrenches are not bound to the supplied window",
            )
        times = window.capture_monotonic_s[valid]
        if times[0] < now_s - max_age_s:
            return None, None, None, None, None, False, f"{name} wrench window is stale"
        if times[-1] > now_s + future_tolerance_s:
            return None, None, None, None, None, False, f"{name} wrench is from the future"
    left_force = np.median(wrenches.left.wrench_task[wrenches.left.valid, :3], axis=0)
    right_force = np.median(wrenches.right.wrench_task[wrenches.right.valid, :3], axis=0)
    net_force = left_force + right_force
    axial = profile.reaction_sign * float(np.dot(net_force, profile.insertion_axis_task))
    lateral = float(
        np.linalg.norm(
            net_force - np.dot(net_force, profile.insertion_axis_task) * profile.insertion_axis_task
        )
    )
    delta = axial - profile.resistance_baseline_signal
    threshold = profile.resistance_delta_threshold_signal
    ratio = progress / command
    candidate = bool(ratio < profile.minimum_progress_ratio and delta >= threshold)
    return ratio, axial, delta, threshold, lateral, candidate, None


def _semantic_action(
    *,
    mode: TestTubeTactileMode,
    primitives: TestTubeTactilePrimitives,
    profile: TestTubeTactileControllerProfile,
    grasp_ready: bool,
    axis_ready: bool,
    slip_confirmed: bool,
    jam_confirmed: bool,
    gripper_open_confirmed: bool,
) -> tuple[TestTubeTactileAction, str]:
    if slip_confirmed:
        return TestTubeTactileAction.RECLAMP_STEP, "qualified slip requires bounded reclamp"
    if mode is TestTubeTactileMode.GRASP:
        if grasp_ready:
            return (
                TestTubeTactileAction.GRASP_READY,
                "bilateral contact and retained width are stable",
            )
        if not primitives.contact_present:
            return TestTubeTactileAction.CLOSE_STEP, "no contact is present"
        return TestTubeTactileAction.WAIT, "contact is not yet a stable bilateral grasp"
    if mode is TestTubeTactileMode.HOLD:
        if not primitives.contact_present:
            return TestTubeTactileAction.HOLD_INVALID, "held-tube contact is absent"
        return TestTubeTactileAction.WAIT, "continue observing the held tube"
    if mode is TestTubeTactileMode.PIVOT:
        if not primitives.contact_present:
            return TestTubeTactileAction.RECLAMP_STEP, "contact was lost during pivot"
        if not profile.axis_runtime_qualified:
            return (
                TestTubeTactileAction.HOLD_INVALID,
                "tactile axis profile is not runtime qualified",
            )
        if not primitives.axis_valid:
            return TestTubeTactileAction.WAIT, "tube axis is not observable in both pads"
        if axis_ready:
            return TestTubeTactileAction.PIVOT_READY, "qualified tactile axis reached the target"
        return (
            TestTubeTactileAction.PIVOT_OPEN_STEP,
            "target axis has not reached its debounce count",
        )
    if mode is TestTubeTactileMode.RECLAMP:
        if not primitives.contact_present:
            return TestTubeTactileAction.RECLAMP_STEP, "contact is not restored"
        if axis_ready:
            return (
                TestTubeTactileAction.RECLAMP_READY,
                "target axis remains stable after reclamp",
            )
        return TestTubeTactileAction.WAIT, "waiting for contact and target-axis confirmation"
    if mode is TestTubeTactileMode.INSERT:
        if not primitives.contact_present:
            return TestTubeTactileAction.HOLD_INVALID, "tube contact was lost before insertion"
        if profile.jam is None or not profile.jam.runtime_qualified:
            return (
                TestTubeTactileAction.HOLD_INVALID,
                "insertion jam profile is not runtime qualified",
            )
        if jam_confirmed:
            return (
                TestTubeTactileAction.RETRACT_REALIGN,
                "qualified low-progress resistance persisted",
            )
        if primitives.insertion_jam_candidate:
            return TestTubeTactileAction.WAIT, "waiting for a second independent jam window"
        if (
            primitives.insertion_progress_ratio is not None
            and primitives.insertion_progress_ratio >= profile.jam.minimum_progress_ratio
        ):
            return TestTubeTactileAction.INSERT_STEP, "axial progress is healthy without a jam"
        return TestTubeTactileAction.WAIT, "low progress without qualified resistance is unresolved"
    if gripper_open_confirmed and not primitives.contact_present:
        return TestTubeTactileAction.RELEASE_READY, "gripper is open and tactile contact is absent"
    return TestTubeTactileAction.WAIT, "release is not yet confirmed"


def _invalid_evaluation(
    mode: TestTubeTactileMode,
    primitives: TestTubeTactilePrimitives,
    memory: TestTubeTactileMemory,
    reason: str,
) -> TestTubeTactileEvaluation:
    payload = {
        "mode": mode.value,
        "sequence": memory.sequence,
        "reason": reason,
        "invalid": True,
    }
    evidence_id = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return TestTubeTactileEvaluation(
        valid=False,
        mode=mode,
        action=TestTubeTactileAction.HOLD_INVALID,
        primitives=primitives,
        promoted_predicates=(),
        shadow_predicates=(),
        reason=reason,
        evidence_id=evidence_id,
        next_memory=memory,
    )


def _evidence_digest(
    *,
    profile: TestTubeTactileControllerProfile,
    mode: TestTubeTactileMode,
    memory: TestTubeTactileMemory,
    endpoints_sdk: list[tuple[str, float]],
    endpoints_host: list[tuple[str, float]],
    primitives: TestTubeTactilePrimitives,
    promoted: set[str],
    shadow: set[str],
) -> str:
    payload = {
        "profile_id": profile.profile_id,
        "session_id": profile.session_id,
        "mode": mode.value,
        "sequence": memory.sequence,
        "sdk_endpoints": endpoints_sdk,
        "host_endpoints": endpoints_host,
        "contact_present": primitives.contact_present,
        "bilateral_contact": primitives.bilateral_contact,
        "axis_base": primitives.axis_base,
        "axis_error_deg": primitives.target_axis_error_deg,
        "finger_images": [
            {
                "serial": finger.serial,
                "contact": finger.contact_candidate,
                "contact_fraction": finger.contact_frame_fraction,
                "area_px": finger.patch_area_px,
                "center_uv_px": finger.patch_center_uv_px,
                "axis_deg": finger.patch_axis_deg,
                "residual_p99": finger.residual_p99,
            }
            for finger in primitives.fingers
        ],
        "slip_candidate": primitives.slip_candidate,
        "progress_ratio": primitives.insertion_progress_ratio,
        "resistance_delta": primitives.insertion_resistance_delta_signal,
        "jam_candidate": primitives.insertion_jam_candidate,
        "promoted": sorted(promoted),
        "shadow": sorted(shadow),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _invalid_finger(serial: str, reason: str) -> TestTubeFingerImagePrimitive:
    return TestTubeFingerImagePrimitive(
        serial=serial,
        valid=False,
        contact_candidate=False,
        contact_frame_fraction=0.0,
        patch_area_px=None,
        patch_center_uv_px=None,
        patch_axis_deg=None,
        elongation=None,
        orientation_concentration=None,
        residual_p99=None,
        reason=reason,
    )


def _undirected_mean_deg(values: list[float]) -> float:
    angles = np.radians(np.asarray(values, dtype=float) * 2.0)
    vector = np.mean(np.exp(1j * angles))
    if abs(vector) <= 1e-9:
        raise ValueError("contact-patch axes have no coherent orientation")
    angle = 0.5 * math.degrees(math.atan2(vector.imag, vector.real))
    return ((angle + 90.0) % 180.0) - 90.0


def _undirected_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    left = _unit_vector(first, "first axis")
    right = _unit_vector(second, "second axis")
    return math.degrees(math.acos(float(np.clip(abs(np.dot(left, right)), -1.0, 1.0))))


def _transform_rotation(value: Any, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} must have homogeneous final row")
    return _rotation(transform[:3, :3], f"{name} rotation")


def _rotation(value: Any, name: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"{name} must be a finite 3x3 rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6
    ):
        raise ValueError(f"{name} must be a proper rotation")
    result = np.array(rotation, copy=True)
    result.setflags(write=False)
    return result


def _unit_vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    result = np.array(vector / norm, copy=True)
    result.setflags(write=False)
    return result


def _tuple3(value: np.ndarray | None) -> tuple[float, float, float] | None:
    return None if value is None else tuple(float(item) for item in value)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without outer whitespace")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _fraction(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1]")
    return result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


__all__ = [
    "TestTubeFingerImagePrimitive",
    "TestTubeFingerImageProfile",
    "TestTubeInsertionJamProfile",
    "TestTubeTactileAction",
    "TestTubeTactileControllerProfile",
    "TestTubeTactileEvaluation",
    "TestTubeTactileMemory",
    "TestTubeTactileMode",
    "TestTubeTactilePrimitives",
    "evaluate_test_tube_tactile",
]
