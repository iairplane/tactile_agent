"""Pure adapter from synchronized GF220 FORCE6D data to task-frame wrenches.

This module owns no SDK, USB, ROS, gripper, or robot handles.  It only accepts
windows already acquired by :mod:`demo2gripper.gf220_live`, verifies their
identity, timing, and calibration provenance, and transforms each sensor's
FORCE6D estimate into a common task frame.

The returned torque is the bias-corrected sensor-origin torque rotated into the
task frame.  It is deliberately *not* shifted by ``r x F`` here:
``FingerWrenchWindow.contact_position_task_m`` carries that lever arm, and the
twist supervisor applies it exactly once when computing task-axis moment.

The adapter deliberately fails closed.  In particular, serial-to-side mapping,
the FORCE6D axes/units qualification, the task-to-sensor rotation, the sensor
origin/contact position, and the unloaded FORCE6D bias must all have explicit
non-empty provenance identifiers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .gf220_live import GF220ClassifierWindows
from .tactile_contact_classifier import FingerRimClassifierWindow
from .tactile_rim_probe import TimedArrayWindow
from .tactile_twist_supervisor import FingerWrenchWindow

REQUIRED_PROVENANCE_FIELDS = frozenset(
    {
        "serial_side_mapping_id",
        "task_sensor_rotation_id",
        "contact_position_id",
        "force6d_bias_id",
        "force6d_axes_units_id",
    }
)


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _readonly_vector(value: object, *, length: int, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (length,) or any(isinstance(item, (bool, np.bool_)) for item in raw):
        raise ValueError(f"{name} must contain exactly {length} finite numbers")
    try:
        vector = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly {length} finite numbers") from exc
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain exactly {length} finite numbers")
    vector = np.ascontiguousarray(vector)
    vector.setflags(write=False)
    return vector


def _readonly_rotation(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (3, 3) or any(isinstance(item, (bool, np.bool_)) for item in raw.flat):
        raise ValueError(f"{name} must be a finite 3x3 rotation matrix")
    try:
        rotation = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 3x3 rotation matrix") from exc
    if not np.isfinite(rotation).all():
        raise ValueError(f"{name} must be a finite 3x3 rotation matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} must have determinant +1")
    rotation = np.ascontiguousarray(rotation)
    rotation.setflags(write=False)
    return rotation


def _validated_provenance(value: object, *, name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    missing = sorted(REQUIRED_PROVENANCE_FIELDS - set(value))
    if missing:
        raise ValueError(f"{name} is missing required fields: {missing}")
    copied: dict[str, str] = {}
    for key, item in value.items():
        copied[key] = _nonempty_string(item, name=f"{name}.{key}")
    return MappingProxyType(copied)


@dataclass(frozen=True, eq=False)
class GF220WrenchSensorCalibration:
    """All reviewed information needed to adapt one physical GF220 sensor."""

    serial: str
    logical_side: str
    calibration_id: str
    R_task_sensor: np.ndarray
    contact_position_task_m: np.ndarray
    force6d_bias: np.ndarray
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "serial", _nonempty_string(self.serial, name="serial"))
        side = _nonempty_string(self.logical_side, name="logical_side")
        if side not in {"left", "right"}:
            raise ValueError("logical_side must be 'left' or 'right'")
        object.__setattr__(self, "logical_side", side)
        object.__setattr__(
            self,
            "calibration_id",
            _nonempty_string(self.calibration_id, name="calibration_id"),
        )
        object.__setattr__(
            self,
            "R_task_sensor",
            _readonly_rotation(self.R_task_sensor, name="R_task_sensor"),
        )
        object.__setattr__(
            self,
            "contact_position_task_m",
            _readonly_vector(
                self.contact_position_task_m,
                length=3,
                name="contact_position_task_m",
            ),
        )
        object.__setattr__(
            self,
            "force6d_bias",
            _readonly_vector(self.force6d_bias, length=6, name="force6d_bias"),
        )
        object.__setattr__(
            self,
            "provenance",
            _validated_provenance(self.provenance, name="provenance"),
        )


@dataclass(frozen=True)
class GF220WrenchAdapterConfig:
    """Strict two-finger mapping and freshness limits for one task frame."""

    profile_id: str
    task_frame_id: str
    sensors: tuple[GF220WrenchSensorCalibration, GF220WrenchSensorCalibration]
    min_samples: int = 3
    max_age_s: float = 0.20
    max_pair_skew_s: float = 0.08
    future_tolerance_s: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "profile_id", _nonempty_string(self.profile_id, name="profile_id")
        )
        object.__setattr__(
            self,
            "task_frame_id",
            _nonempty_string(self.task_frame_id, name="task_frame_id"),
        )
        if not isinstance(self.sensors, tuple) or len(self.sensors) != 2:
            raise ValueError("sensors must contain exactly two calibrations")
        if any(not isinstance(item, GF220WrenchSensorCalibration) for item in self.sensors):
            raise TypeError("sensors must contain GF220WrenchSensorCalibration values")
        serials = [item.serial for item in self.sensors]
        if len(set(serials)) != 2:
            raise ValueError("sensor serials must be unique")
        sides = {item.logical_side for item in self.sensors}
        if sides != {"left", "right"}:
            raise ValueError("sensor mapping must contain exactly one left and one right side")
        if (
            isinstance(self.min_samples, bool)
            or not isinstance(self.min_samples, int)
            or self.min_samples < 2
        ):
            raise ValueError("min_samples must be an integer >= 2")
        for name in ("max_age_s", "max_pair_skew_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        tolerance = float(self.future_tolerance_s)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("future_tolerance_s must be finite and non-negative")
        object.__setattr__(self, "future_tolerance_s", tolerance)


@dataclass(frozen=True)
class AdaptedGF220Wrenches:
    """Left/right task-frame windows plus immutable adapter provenance."""

    profile_id: str
    task_frame_id: str
    left_serial: str
    right_serial: str
    left: FingerWrenchWindow
    right: FingerWrenchWindow
    provenance_by_serial: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        if not isinstance(self.left, FingerWrenchWindow) or not isinstance(
            self.right, FingerWrenchWindow
        ):
            raise TypeError("left and right must be FingerWrenchWindow values")
        copied = {
            serial: MappingProxyType(dict(provenance))
            for serial, provenance in self.provenance_by_serial.items()
        }
        object.__setattr__(self, "provenance_by_serial", MappingProxyType(copied))


def _validate_force_window(
    source: FingerRimClassifierWindow,
    *,
    expected_calibration_id: str,
    min_samples: int,
    max_age_s: float,
    future_tolerance_s: float,
    now_s: float,
    serial: str,
) -> TimedArrayWindow:
    if source.calibration_id != expected_calibration_id:
        raise ValueError(
            f"{serial}: calibration_id {source.calibration_id!r} does not match "
            f"configured {expected_calibration_id!r}"
        )
    force = source.force6d
    if force is None:
        raise ValueError(f"{serial}: FORCE6D window is missing")
    values = np.asarray(force.values)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"{serial}: FORCE6D values must have shape (N, 6)")
    if values.shape[0] < min_samples:
        raise ValueError(
            f"{serial}: FORCE6D window has {values.shape[0]} samples; "
            f"requires at least {min_samples}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{serial}: FORCE6D values must be finite")
    if force.valid.shape != (values.shape[0],) or not np.all(force.valid):
        raise ValueError(f"{serial}: FORCE6D window contains invalid samples")
    sdk_time = np.asarray(force.sdk_time_s, dtype=float)
    capture_time = np.asarray(force.capture_monotonic_s, dtype=float)
    if not np.isfinite(sdk_time).all() or not np.isfinite(capture_time).all():
        raise ValueError(f"{serial}: FORCE6D timestamps must be finite")
    if np.any(np.diff(sdk_time) <= 0.0):
        raise ValueError(f"{serial}: FORCE6D SDK timestamps must strictly increase")
    if np.any(np.diff(capture_time) <= 0.0):
        raise ValueError(f"{serial}: FORCE6D host timestamps must strictly increase")
    if capture_time[-1] > now_s + future_tolerance_s:
        raise ValueError(f"{serial}: FORCE6D window contains future samples")
    if capture_time[0] < now_s - max_age_s:
        raise ValueError(f"{serial}: FORCE6D window is stale")

    # gf220_live requests these channels in one SDK call.  Refuse a window if
    # another producer supplies force evidence with different frame identity.
    for channel_name, channel in (("warped", source.warped), ("diff", source.diff)):
        if not np.array_equal(force.sdk_time_s, channel.sdk_time_s):
            raise ValueError(f"{serial}: FORCE6D and {channel_name} SDK timestamps differ")
        if not np.array_equal(force.capture_monotonic_s, channel.capture_monotonic_s):
            raise ValueError(f"{serial}: FORCE6D and {channel_name} host timestamps differ")
        if not np.array_equal(force.valid, channel.valid):
            raise ValueError(f"{serial}: FORCE6D and {channel_name} validity differ")
    return force


def _transform_window(
    source: FingerRimClassifierWindow,
    calibration: GF220WrenchSensorCalibration,
    config: GF220WrenchAdapterConfig,
    *,
    now_s: float,
) -> FingerWrenchWindow:
    force = _validate_force_window(
        source,
        expected_calibration_id=calibration.calibration_id,
        min_samples=config.min_samples,
        max_age_s=config.max_age_s,
        future_tolerance_s=config.future_tolerance_s,
        now_s=now_s,
        serial=calibration.serial,
    )
    corrected = np.asarray(force.values, dtype=float) - calibration.force6d_bias[None, :]
    force_task = corrected[:, :3] @ calibration.R_task_sensor.T
    torque_task = corrected[:, 3:] @ calibration.R_task_sensor.T
    wrench_task = np.concatenate((force_task, torque_task), axis=1)
    if not np.isfinite(wrench_task).all():
        raise ValueError(f"{calibration.serial}: transformed wrench is not finite")
    return FingerWrenchWindow(
        calibration_id=calibration.calibration_id,
        task_frame_id=config.task_frame_id,
        wrench_task=wrench_task,
        contact_position_task_m=calibration.contact_position_task_m,
        sdk_time_s=force.sdk_time_s,
        capture_monotonic_s=force.capture_monotonic_s,
        valid=force.valid,
    )


def adapt_gf220_force6d_windows(
    source: GF220ClassifierWindows,
    config: GF220WrenchAdapterConfig,
    *,
    now_s: float,
) -> AdaptedGF220Wrenches:
    """Validate and transform synchronized GF220 windows into left/right wrenches."""

    if not isinstance(source, GF220ClassifierWindows):
        raise TypeError("source must be a GF220ClassifierWindows instance")
    if not isinstance(config, GF220WrenchAdapterConfig):
        raise TypeError("config must be a GF220WrenchAdapterConfig instance")
    now_s = float(now_s)
    if not math.isfinite(now_s):
        raise ValueError("now_s must be finite")
    if len(source.serials) != 2 or len(set(source.serials)) != 2:
        raise ValueError("source must contain exactly two unique sensor serials")
    if len(source.windows) != 2:
        raise ValueError("source must contain exactly two sensor windows")

    configured = {item.serial: item for item in config.sensors}
    if set(source.serials) != set(configured):
        raise ValueError(
            "source serials do not exactly match the configured serial-to-side mapping"
        )
    windows_by_serial = dict(zip(source.serials, source.windows, strict=True))
    transformed: dict[str, FingerWrenchWindow] = {}
    for serial, calibration in configured.items():
        transformed[calibration.logical_side] = _transform_window(
            windows_by_serial[serial], calibration, config, now_s=now_s
        )

    left = transformed["left"]
    right = transformed["right"]
    start_skew = abs(left.capture_monotonic_s[0] - right.capture_monotonic_s[0])
    end_skew = abs(left.capture_monotonic_s[-1] - right.capture_monotonic_s[-1])
    if max(start_skew, end_skew) > config.max_pair_skew_s:
        raise ValueError(
            "left/right FORCE6D windows exceed max_pair_skew_s "
            f"({max(start_skew, end_skew):.6f} > {config.max_pair_skew_s:.6f})"
        )

    side_to_calibration = {item.logical_side: item for item in config.sensors}
    left_calibration = side_to_calibration["left"]
    right_calibration = side_to_calibration["right"]
    return AdaptedGF220Wrenches(
        profile_id=config.profile_id,
        task_frame_id=config.task_frame_id,
        left_serial=left_calibration.serial,
        right_serial=right_calibration.serial,
        left=left,
        right=right,
        provenance_by_serial={
            item.serial: item.provenance for item in config.sensors
        },
    )


__all__ = [
    "AdaptedGF220Wrenches",
    "GF220WrenchAdapterConfig",
    "GF220WrenchSensorCalibration",
    "REQUIRED_PROVENANCE_FIELDS",
    "adapt_gf220_force6d_windows",
]
