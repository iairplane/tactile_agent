"""Image-only GF220 contact gate for bounded test-tube approach steps.

This module is deliberately pure.  It fits a negative (unloaded) image model
from synchronized SDK ``WARPED_IMG`` and ``DIFF_IMG`` windows, then classifies
new windows.  ``FORCE6D`` is neither read nor interpreted.  Robot motion and
sensor ownership live in :mod:`scripts.run_test_tube_contact_approach`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .gf220_live import GF220Calibration, GF220ClassifierWindows
from .test_tube_tactile_supervisor import compensated_residual

CONTACT_APPROACH_SCHEMA = "demo2gripper.test_tube_contact_approach.v1"
MAX_APPROACH_STEP_M = 0.004
MAX_APPROACH_DISTANCE_M = 0.040
MAX_APPROACH_STEPS = 10


@dataclass(frozen=True)
class FingerContactImageProfile:
    serial: str
    calibration_id: str
    warped_reference: np.ndarray
    diff_reference: np.ndarray
    roi_mask: np.ndarray
    warped_residual_threshold: float
    diff_residual_threshold: float
    min_component_area_px: int
    unloaded_frame_count: int

    def __post_init__(self) -> None:
        for name in ("serial", "calibration_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        warped = _image(self.warped_reference, "warped_reference")
        diff = _image(self.diff_reference, "diff_reference")
        if warped.shape != diff.shape:
            raise ValueError("warped and diff references must have the same shape")
        roi = np.asarray(self.roi_mask, dtype=bool)
        if roi.shape != warped.shape[:2] or not roi.any():
            raise ValueError("roi_mask must be non-empty and match the references")
        for name in ("warped_residual_threshold", "diff_residual_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.min_component_area_px < 1:
            raise ValueError("min_component_area_px must be positive")
        if self.unloaded_frame_count < 10:
            raise ValueError("unloaded_frame_count must be at least ten")
        for name, value in (
            ("warped_reference", warped),
            ("diff_reference", diff),
            ("roi_mask", roi),
        ):
            copied = np.array(value, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


@dataclass(frozen=True)
class ContactApproachProfile:
    session_id: str
    fingers: tuple[FingerContactImageProfile, FingerContactImageProfile]
    min_unique_frames: int = 5
    contact_frame_fraction: float = 0.60
    max_pair_skew_s: float = 0.08

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        if len(self.fingers) != 2 or len({item.serial for item in self.fingers}) != 2:
            raise ValueError("fingers must contain two distinct sensors")
        if self.min_unique_frames < 2:
            raise ValueError("min_unique_frames must be at least two")
        if not 0.0 < self.contact_frame_fraction <= 1.0:
            raise ValueError("contact_frame_fraction must lie in (0, 1]")
        if not math.isfinite(self.max_pair_skew_s) or self.max_pair_skew_s <= 0.0:
            raise ValueError("max_pair_skew_s must be positive and finite")


@dataclass(frozen=True)
class FingerContactObservation:
    serial: str
    contact: bool
    contact_frame_fraction: float
    median_component_area_px: float
    maximum_component_area_px: int
    median_warped_p99: float
    median_diff_p99: float


@dataclass(frozen=True)
class ContactApproachMemory:
    forward_stopped: bool = False
    bilateral_windows: int = 0
    last_sdk_time_s_by_serial: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ContactWindowDecision:
    valid: bool
    action: str
    stop_forward: bool
    grasp_ready: bool
    active_serials: tuple[str, ...]
    bilateral_contact: bool
    observations: tuple[FingerContactObservation, FingerContactObservation]
    reason: str
    next_memory: ContactApproachMemory


def make_approach_schedule(
    target_grasp_position_m: np.ndarray,
    approach_axis_base: np.ndarray,
    remaining_distance_m: float,
    *,
    step_m: float = MAX_APPROACH_STEP_M,
    maximum_steps: int = MAX_APPROACH_STEPS,
) -> tuple[np.ndarray, ...]:
    """Return absolute TCP targets from the present standoff to the grasp.

    ``approach_axis_base`` points from the standoff toward the object.  The
    caller is responsible for placing the robot at
    ``target_grasp_position_m - remaining_distance_m * approach_axis_base``
    before running the first microstep.
    """

    target = _vector3(target_grasp_position_m, "target_grasp_position_m")
    axis = _vector3(approach_axis_base, "approach_axis_base")
    axis /= np.linalg.norm(axis)
    remaining = float(remaining_distance_m)
    step = float(step_m)
    if not math.isfinite(remaining) or not 0.0 < remaining <= MAX_APPROACH_DISTANCE_M:
        raise ValueError("remaining_distance_m must lie in (0, 0.040]")
    if not math.isfinite(step) or not 0.0 < step <= MAX_APPROACH_STEP_M:
        raise ValueError("step_m must lie in (0, 0.004]")
    if (
        isinstance(maximum_steps, bool)
        or not isinstance(maximum_steps, int)
        or not 1 <= maximum_steps <= MAX_APPROACH_STEPS
    ):
        raise ValueError("maximum_steps must be an integer in [1, 10]")
    count = min(int(math.ceil(remaining / step - 1e-12)), maximum_steps)
    travelled = [min((index + 1) * step, remaining) for index in range(count)]
    return tuple(
        np.asarray(target - axis * (remaining - distance), dtype=float)
        for distance in travelled
    )


def fit_unloaded_contact_profile(
    *,
    session_id: str,
    calibrations: tuple[GF220Calibration, GF220Calibration],
    windows: tuple[GF220ClassifierWindows, ...],
    min_unique_frames: int = 5,
    contact_frame_fraction: float = 0.60,
    max_pair_skew_s: float = 0.08,
) -> ContactApproachProfile:
    """Fit per-pad residual and coherent-area thresholds from empty pads."""

    if not session_id:
        raise ValueError("session_id must be non-empty")
    if len(calibrations) != 2 or len({item.serial for item in calibrations}) != 2:
        raise ValueError("calibrations must contain two distinct sensors")
    if len(windows) < 2:
        raise ValueError("at least two unloaded windows are required")
    calibration_by_id = {item.calibration_id: item for item in calibrations}
    serial_by_calibration = {item.calibration_id: item.serial for item in calibrations}
    grouped: dict[str, dict[str, list[np.ndarray]]] = {
        item.serial: {"warped": [], "diff": []} for item in calibrations
    }
    last_sdk: dict[str, float] = {}
    for pair in windows:
        if pair.endpoint_skew_s > max_pair_skew_s:
            raise ValueError("unloaded window exceeds pair-skew limit")
        if set(pair.serials) != set(grouped):
            raise ValueError("unloaded sensor serials changed")
        for serial, source in zip(pair.serials, pair.windows, strict=True):
            calibration = calibration_by_id.get(source.calibration_id)
            if calibration is None or serial_by_calibration[source.calibration_id] != serial:
                raise ValueError("unloaded window calibration identity changed")
            indices, end_sdk = _fresh_indices(
                source,
                min_unique_frames=min_unique_frames,
                previous_sdk_time_s=last_sdk.get(serial),
            )
            last_sdk[serial] = end_sdk
            grouped[serial]["warped"].extend(np.asarray(source.warped.values)[indices])
            grouped[serial]["diff"].extend(np.asarray(source.diff.values)[indices])

    profiles: list[FingerContactImageProfile] = []
    calibration_by_serial = {item.serial: item for item in calibrations}
    for serial in sorted(grouped):
        warped = np.asarray(grouped[serial]["warped"], dtype=np.uint8)
        diff = np.asarray(grouped[serial]["diff"], dtype=np.uint8)
        if warped.shape != diff.shape or warped.ndim != 4:
            raise ValueError("unloaded WARPED_IMG/DIFF_IMG shapes do not match")
        reference_warped = np.median(warped, axis=0).astype(np.uint8)
        reference_diff = np.median(diff, axis=0).astype(np.uint8)
        roi = _central_roi(reference_warped.shape[:2])
        warped_residuals = np.stack(
            [compensated_residual(frame, reference_warped, roi) for frame in warped]
        )
        diff_residuals = np.stack(
            [compensated_residual(frame, reference_diff, roi) for frame in diff]
        )
        warped_threshold = _negative_threshold(warped_residuals[:, roi])
        diff_threshold = _negative_threshold(diff_residuals[:, roi])
        unloaded_areas = [
            _largest_component_area(
                np.maximum(
                    warped_residuals[index] / warped_threshold,
                    diff_residuals[index] / diff_threshold,
                )
                >= 1.0,
                roi,
            )
            for index in range(warped.shape[0])
        ]
        roi_area = int(roi.sum())
        area_floor = max(32, int(math.ceil(0.0015 * roi_area)))
        area_margin = max(16, int(math.ceil(0.0005 * roi_area)))
        minimum_area = max(area_floor, max(unloaded_areas, default=0) + area_margin)
        calibration = calibration_by_serial[serial]
        profiles.append(
            FingerContactImageProfile(
                serial=serial,
                calibration_id=calibration.calibration_id,
                warped_reference=reference_warped,
                diff_reference=reference_diff,
                roi_mask=roi,
                warped_residual_threshold=warped_threshold,
                diff_residual_threshold=diff_threshold,
                min_component_area_px=minimum_area,
                unloaded_frame_count=int(warped.shape[0]),
            )
        )
    return ContactApproachProfile(
        session_id=session_id,
        fingers=(profiles[0], profiles[1]),
        min_unique_frames=min_unique_frames,
        contact_frame_fraction=contact_frame_fraction,
        max_pair_skew_s=max_pair_skew_s,
    )


def evaluate_contact_window(
    windows: GF220ClassifierWindows,
    profile: ContactApproachProfile,
    memory: ContactApproachMemory = ContactApproachMemory(),
) -> ContactWindowDecision:
    """Classify one non-overlapping dual-pad window and update stop state."""

    empty = tuple(
        FingerContactObservation(item.serial, False, 0.0, 0.0, 0, 0.0, 0.0)
        for item in profile.fingers
    )
    if windows.endpoint_skew_s > profile.max_pair_skew_s:
        return ContactWindowDecision(
            False,
            "hold_invalid",
            True,
            False,
            (),
            False,
            empty,  # type: ignore[arg-type]
            "window exceeds pair-skew limit",
            ContactApproachMemory(True, 0, memory.last_sdk_time_s_by_serial),
        )
    source_by_serial = dict(zip(windows.serials, windows.windows, strict=True))
    profile_serials = {item.serial for item in profile.fingers}
    if set(source_by_serial) != profile_serials:
        raise ValueError("window sensor identities do not match profile")
    previous = dict(memory.last_sdk_time_s_by_serial)
    observations: list[FingerContactObservation] = []
    endpoints: list[tuple[str, float]] = []
    try:
        for finger in profile.fingers:
            source = source_by_serial[finger.serial]
            if source.calibration_id != finger.calibration_id:
                raise ValueError(f"{finger.serial}: calibration identity changed")
            indices, endpoint = _fresh_indices(
                source,
                min_unique_frames=profile.min_unique_frames,
                previous_sdk_time_s=previous.get(finger.serial),
            )
            observations.append(_observe_finger(source, indices, finger, profile))
            endpoints.append((finger.serial, endpoint))
    except ValueError as exc:
        return ContactWindowDecision(
            False,
            "hold_invalid",
            True,
            False,
            (),
            False,
            empty,  # type: ignore[arg-type]
            str(exc),
            ContactApproachMemory(True, 0, memory.last_sdk_time_s_by_serial),
        )

    active = tuple(item.serial for item in observations if item.contact)
    bilateral = len(active) == 2
    bilateral_windows = memory.bilateral_windows + 1 if bilateral else 0
    stopped = memory.forward_stopped or bool(active)
    ready = stopped and bilateral_windows >= 2
    if ready:
        action, reason = "grasp_ready", "bilateral image contact persisted for two windows"
    elif stopped:
        action = "one_pad_contact_hold"
        reason = (
            "contact stopped forward motion; bilateral confirmation is incomplete"
            if bilateral
            else "contact stopped forward motion; both pads are not in contact"
        )
    else:
        action, reason = "advance_clear", "no pad contact observed"
    next_memory = ContactApproachMemory(
        forward_stopped=stopped,
        bilateral_windows=bilateral_windows,
        last_sdk_time_s_by_serial=tuple(sorted(endpoints)),
    )
    return ContactWindowDecision(
        valid=True,
        action=action,
        stop_forward=stopped,
        grasp_ready=ready,
        active_serials=active,
        bilateral_contact=bilateral,
        observations=(observations[0], observations[1]),
        reason=reason,
        next_memory=next_memory,
    )


def _observe_finger(
    source: object,
    indices: np.ndarray,
    profile: FingerContactImageProfile,
    controller: ContactApproachProfile,
) -> FingerContactObservation:
    warped = np.asarray(source.warped.values)[indices]
    diff = np.asarray(source.diff.values)[indices]
    contacts: list[bool] = []
    areas: list[int] = []
    warped_p99: list[float] = []
    diff_p99: list[float] = []
    for warped_frame, diff_frame in zip(warped, diff, strict=True):
        warped_residual = compensated_residual(
            warped_frame, profile.warped_reference, profile.roi_mask
        )
        diff_residual = compensated_residual(
            diff_frame, profile.diff_reference, profile.roi_mask
        )
        fused = np.maximum(
            warped_residual / profile.warped_residual_threshold,
            diff_residual / profile.diff_residual_threshold,
        )
        area = _largest_component_area(fused >= 1.0, profile.roi_mask)
        areas.append(area)
        contacts.append(area >= profile.min_component_area_px)
        warped_p99.append(float(np.quantile(warped_residual[profile.roi_mask], 0.99)))
        diff_p99.append(float(np.quantile(diff_residual[profile.roi_mask], 0.99)))
    fraction = sum(contacts) / len(contacts)
    return FingerContactObservation(
        serial=profile.serial,
        contact=fraction >= controller.contact_frame_fraction,
        contact_frame_fraction=float(fraction),
        median_component_area_px=float(np.median(areas)),
        maximum_component_area_px=int(max(areas)),
        median_warped_p99=float(np.median(warped_p99)),
        median_diff_p99=float(np.median(diff_p99)),
    )


def _fresh_indices(
    source: object,
    *,
    min_unique_frames: int,
    previous_sdk_time_s: float | None,
) -> tuple[np.ndarray, float]:
    warped, diff = source.warped, source.diff
    if not (
        np.array_equal(warped.sdk_time_s, diff.sdk_time_s)
        and np.array_equal(warped.capture_monotonic_s, diff.capture_monotonic_s)
        and np.array_equal(warped.valid, diff.valid)
    ):
        raise ValueError("WARPED_IMG and DIFF_IMG frame identities differ")
    valid = np.asarray(warped.valid, dtype=bool)
    indices = np.flatnonzero(valid)
    if indices.size:
        _, unique = np.unique(warped.sdk_time_s[indices], return_index=True)
        indices = indices[np.sort(unique)]
    if indices.size < min_unique_frames:
        raise ValueError(f"only {indices.size} unique image frames")
    sdk = np.asarray(warped.sdk_time_s, dtype=float)[indices]
    host = np.asarray(warped.capture_monotonic_s, dtype=float)[indices]
    if np.any(np.diff(sdk) <= 0.0) or np.any(np.diff(host) <= 0.0):
        raise ValueError("image timestamps must strictly increase")
    if previous_sdk_time_s is not None and sdk[0] <= previous_sdk_time_s:
        raise ValueError("image window overlaps or replays prior evidence")
    warped_values = np.asarray(warped.values)
    diff_values = np.asarray(diff.values)
    if warped_values.dtype != np.uint8 or diff_values.dtype != np.uint8:
        raise ValueError("WARPED_IMG and DIFF_IMG must be uint8")
    if warped_values.shape != diff_values.shape or warped_values.ndim != 4:
        raise ValueError("WARPED_IMG and DIFF_IMG shapes must match as NxHxWx3")
    return indices, float(sdk[-1])


def _negative_threshold(values: np.ndarray) -> float:
    frames = np.asarray(values, dtype=float)
    if frames.ndim != 2 or not np.isfinite(frames).all():
        raise ValueError("negative residual evidence must be finite frames by pixels")
    high = np.quantile(frames, 0.995, axis=1)
    median = float(np.median(high))
    robust_sigma = 1.4826 * float(np.median(np.abs(high - median)))
    return max(2.0, float(np.quantile(frames, 0.9995)), median + 8.0 * robust_sigma)


def _largest_component_area(mask: np.ndarray, roi: np.ndarray) -> int:
    candidate = np.asarray(mask, dtype=bool) & roi
    candidate = ndimage.binary_opening(candidate, structure=np.ones((3, 3), dtype=bool))
    candidate = ndimage.binary_closing(candidate, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndimage.label(candidate)
    if count == 0:
        return 0
    return int(np.max(np.bincount(labels.ravel())[1:]))


def _central_roi(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if height < 16 or width < 16:
        raise ValueError("tactile images are too small")
    margin_y = max(2, int(round(0.05 * height)))
    margin_x = max(2, int(round(0.05 * width)))
    roi = np.zeros((height, width), dtype=bool)
    roi[margin_y : height - margin_y, margin_x : width - margin_x] = True
    return roi


def _image(value: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 HxWx3")
    return image


def _vector3(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    if float(np.linalg.norm(vector)) <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return vector


__all__ = [
    "CONTACT_APPROACH_SCHEMA",
    "MAX_APPROACH_DISTANCE_M",
    "MAX_APPROACH_STEP_M",
    "MAX_APPROACH_STEPS",
    "ContactApproachMemory",
    "ContactApproachProfile",
    "ContactWindowDecision",
    "FingerContactImageProfile",
    "FingerContactObservation",
    "evaluate_contact_window",
    "fit_unloaded_contact_profile",
    "make_approach_schedule",
]
