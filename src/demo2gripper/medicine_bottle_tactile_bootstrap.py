"""Task-session FORCE6D bootstrap for a press-and-turn bottle cap.

The bootstrap consumes three synchronized raw GF220 windows acquired in one
device session:

``UNLOADED -> GRASPED_WITHOUT_DOWNPRESS -> KNOWN_AXIAL_PRESS``.

It estimates the full unloaded six-axis bias and a sensor-to-task rotation for
each finger.  The grasp response identifies the radial direction and the known
axial press identifies the bottle-axis direction; two non-collinear vectors
are sufficient to construct a proper rotation.  Serial-to-side assignment and
contact positions remain geometric inputs from the seed profile.

The result is deliberately session-scoped.  It never writes an accepted
machine calibration and it cannot authorize motion.  The hardware runner
records the returned report and uses the adapter only for the remainder of the
same execution.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from .gf220_live import GF220ClassifierWindows
from .gf220_wrench_adapter import (
    GF220WrenchAdapterConfig,
    GF220WrenchSensorCalibration,
    adapt_gf220_force6d_windows,
)
from .medicine_bottle_tactile_presence import MedicineBottlePresenceConfig
from .tactile_twist_supervisor import (
    TactileTwistConfig,
    build_tactile_twist_baseline,
)

BOOTSTRAP_SCHEMA = "demo2gripper.medicine_bottle_tactile_bootstrap.v1"


class TactileBootstrapQualityError(ValueError):
    """Valid evidence that does not yet identify a usable task-frame signal."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


@dataclass(frozen=True)
class MedicineBottleTactileBootstrapConfig:
    """Quality gates for one task-session tactile bootstrap."""

    min_samples: int = 5
    min_signal_to_noise: float = 3.0
    max_direction_cosine: float = 0.90
    min_opposed_radial_cosine: float = 0.80
    engage_fraction_of_observed_press: float = 0.60
    retain_fraction_of_observed_press: float = 0.30
    threshold_headroom_fraction: float = 0.10
    release_axis_noise_multiplier: float = 6.0
    release_tangent_noise_multiplier: float = 6.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_samples, bool)
            or not isinstance(self.min_samples, int)
            or self.min_samples < 3
        ):
            raise ValueError("min_samples must be an integer >= 3")
        if not math.isfinite(self.min_signal_to_noise) or self.min_signal_to_noise <= 1.0:
            raise ValueError("min_signal_to_noise must be finite and > 1")
        for name in (
            "release_axis_noise_multiplier",
            "release_tangent_noise_multiplier",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 1.0:
                raise ValueError(f"{name} must be finite and > 1")
        for name in (
            "max_direction_cosine",
            "min_opposed_radial_cosine",
            "engage_fraction_of_observed_press",
            "retain_fraction_of_observed_press",
            "threshold_headroom_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
        if self.retain_fraction_of_observed_press >= self.engage_fraction_of_observed_press:
            raise ValueError(
                "retain_fraction_of_observed_press must be smaller than "
                "engage_fraction_of_observed_press"
            )


@dataclass(frozen=True)
class MedicineBottleTactileBootstrapResult:
    adapter: GF220WrenchAdapterConfig
    tactile_config: TactileTwistConfig
    contact_normal_threshold_by_side: dict[str, float]
    release_axis_delta_sensor_units: float
    release_tangent_delta_sensor_units: float
    report: dict[str, Any]


@dataclass(frozen=True)
class MedicineBottlePresenceBootstrapConfig:
    """Threshold fractions for scalar any-pad sensor-Fz presence."""

    min_samples: int = 5
    minimum_fz_change_sensor_units: float = 1e-6
    contact_retain_fraction: float = 0.10
    press_engage_fraction: float = 0.10
    press_retain_fraction: float = 0.05
    unload_return_fraction: float = 0.25
    release_delta_fraction: float = 0.50

    def __post_init__(self) -> None:
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least two")
        if (
            not math.isfinite(self.minimum_fz_change_sensor_units)
            or self.minimum_fz_change_sensor_units <= 0.0
        ):
            raise ValueError("minimum_fz_change_sensor_units must be positive")
        for name in (
            "contact_retain_fraction",
            "press_engage_fraction",
            "press_retain_fraction",
            "unload_return_fraction",
            "release_delta_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")


@dataclass(frozen=True)
class MedicineBottlePresenceBootstrapResult:
    adapter: GF220WrenchAdapterConfig
    presence_config: MedicineBottlePresenceConfig
    contact_presence_threshold_sensor_units: float
    release_axis_delta_sensor_units: float
    release_tangent_delta_sensor_units: float
    report: dict[str, Any]


@dataclass(frozen=True)
class _RawWindow:
    calibration_id: str
    wrench: np.ndarray
    sdk_time_s: np.ndarray
    capture_time_s: np.ndarray


def _unit(value: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def _robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    if values.shape[0] > 1:
        standard = np.std(values, axis=0, ddof=1)
        scale = np.where(scale > 1e-12, scale, standard)
    # A perfectly constant fake or quantized channel carries no measured
    # noise.  The epsilon prevents division by zero without manufacturing a
    # useful signal: the supervisor's later baseline still requires actual
    # variation in every channel it uses.
    return np.maximum(scale, np.finfo(float).eps)


def _extract_stage(
    pair: GF220ClassifierWindows,
    *,
    expected_serials: set[str],
    min_samples: int,
    max_pair_skew_s: float,
    stage: str,
) -> dict[str, _RawWindow]:
    if not isinstance(pair, GF220ClassifierWindows):
        raise TypeError(f"{stage} must be GF220ClassifierWindows")
    if len(pair.serials) != 2 or len(set(pair.serials)) != 2:
        raise ValueError(f"{stage} must contain exactly two unique serials")
    if set(pair.serials) != expected_serials:
        raise ValueError(f"{stage} serials do not match the seed adapter")
    if pair.endpoint_skew_s > max_pair_skew_s:
        raise ValueError(f"{stage} exceeds the configured pair skew")

    result: dict[str, _RawWindow] = {}
    for serial, window in zip(pair.serials, pair.windows, strict=True):
        force = window.force6d
        if force is None:
            raise ValueError(f"{stage}/{serial}: FORCE6D window is missing")
        wrench = np.asarray(force.values, dtype=float)
        sdk_time = np.asarray(force.sdk_time_s, dtype=float)
        capture_time = np.asarray(force.capture_monotonic_s, dtype=float)
        valid = np.asarray(force.valid, dtype=bool)
        if wrench.ndim != 2 or wrench.shape[1] != 6:
            raise ValueError(f"{stage}/{serial}: FORCE6D must have shape (N, 6)")
        if wrench.shape[0] < min_samples:
            raise ValueError(
                f"{stage}/{serial}: requires at least {min_samples} samples"
            )
        if (
            sdk_time.shape != (wrench.shape[0],)
            or capture_time.shape != (wrench.shape[0],)
            or valid.shape != (wrench.shape[0],)
        ):
            raise ValueError(f"{stage}/{serial}: FORCE6D metadata shape mismatch")
        if (
            not np.isfinite(wrench).all()
            or not np.isfinite(sdk_time).all()
            or not np.isfinite(capture_time).all()
        ):
            raise ValueError(f"{stage}/{serial}: FORCE6D values and times must be finite")
        if not np.all(valid):
            raise ValueError(f"{stage}/{serial}: FORCE6D contains invalid samples")
        if np.any(np.diff(sdk_time) <= 0.0) or np.any(np.diff(capture_time) <= 0.0):
            raise ValueError(f"{stage}/{serial}: timestamps must strictly increase")
        result[serial] = _RawWindow(
            calibration_id=window.calibration_id,
            wrench=wrench,
            sdk_time_s=sdk_time,
            capture_time_s=capture_time,
        )
    return result


def _rotation_from_two_responses(
    *,
    normal_sensor: np.ndarray,
    axial_sensor: np.ndarray,
    radial_task: np.ndarray,
    axial_task: np.ndarray,
    max_direction_cosine: float,
) -> tuple[np.ndarray, float]:
    axial_sensor_unit = _unit(axial_sensor, name="axial sensor response")
    normal_sensor_unit = _unit(normal_sensor, name="grasp sensor response")
    direction_cosine = abs(float(np.dot(axial_sensor_unit, normal_sensor_unit)))
    if direction_cosine > max_direction_cosine:
        raise TactileBootstrapQualityError(
            "response_directions_collinear",
            "grasp and axial-press FORCE6D directions are too nearly collinear "
            f"({direction_cosine:.6f} > {max_direction_cosine:.6f})",
            details={
                "absolute_direction_cosine": direction_cosine,
                "maximum": max_direction_cosine,
            },
        )
    sensor_radial = normal_sensor_unit - float(
        np.dot(normal_sensor_unit, axial_sensor_unit)
    ) * axial_sensor_unit
    sensor_radial = _unit(sensor_radial, name="orthogonalized grasp response")
    sensor_tangent = _unit(
        np.cross(axial_sensor_unit, sensor_radial), name="sensor tangent"
    )

    task_axis = _unit(axial_task, name="task axial direction")
    task_radial = np.asarray(radial_task, dtype=float)
    task_radial -= float(np.dot(task_radial, task_axis)) * task_axis
    task_radial = _unit(task_radial, name="task radial direction")
    task_tangent = _unit(np.cross(task_axis, task_radial), name="task tangent")

    sensor_basis = np.column_stack((sensor_radial, sensor_tangent, axial_sensor_unit))
    task_basis = np.column_stack((task_radial, task_tangent, task_axis))
    rotation = task_basis @ sensor_basis.T
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0):
        raise ValueError("estimated task-sensor rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8):
        raise ValueError("estimated task-sensor rotation is not proper")
    return rotation, direction_cosine


def _window_digest(stages: dict[str, dict[str, _RawWindow]]) -> str:
    digest = hashlib.sha256(b"demo2gripper.medicine_bottle_tactile_bootstrap.v1\0")
    for stage in ("unloaded", "grasp", "axial_press"):
        for serial in sorted(stages[stage]):
            window = stages[stage][serial]
            digest.update(stage.encode("utf-8"))
            digest.update(serial.encode("utf-8"))
            digest.update(window.calibration_id.encode("utf-8"))
            for value in (window.wrench, window.sdk_time_s, window.capture_time_s):
                contiguous = np.ascontiguousarray(value)
                digest.update(str(contiguous.dtype).encode("ascii"))
                digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
                digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _adaptive_press_thresholds(
    base: TactileTwistConfig,
    *,
    left_z: float,
    right_z: float,
    sum_z: float,
    settings: MedicineBottleTactileBootstrapConfig,
) -> TactileTwistConfig:
    smallest_finger_z = min(left_z, right_z)
    headroom = 1.0 - settings.threshold_headroom_fraction
    if smallest_finger_z <= 0.0 or sum_z <= 0.0:
        raise TactileBootstrapQualityError(
            "nonpositive_bilateral_axial_reaction",
            "known axial press did not produce positive bilateral reaction",
            details={
                "left_z": left_z,
                "right_z": right_z,
                "sum_z": sum_z,
            },
        )

    engage_finger = max(
        base.press_engage_finger_z,
        settings.engage_fraction_of_observed_press * smallest_finger_z,
    )
    engage_sum = max(
        base.press_engage_sum_z,
        settings.engage_fraction_of_observed_press * sum_z,
    )
    if engage_finger >= headroom * smallest_finger_z or engage_sum >= headroom * sum_z:
        raise TactileBootstrapQualityError(
            "axial_press_gate_headroom_insufficient",
            "known axial press lacks headroom above the configured/adaptive engagement gate",
            details={
                "smallest_finger_z": smallest_finger_z,
                "sum_z": sum_z,
                "engage_finger_z": engage_finger,
                "engage_sum_z": engage_sum,
            },
        )
    retain_finger = max(
        base.press_retain_finger_z,
        settings.retain_fraction_of_observed_press * smallest_finger_z,
    )
    retain_sum = max(
        base.press_retain_sum_z,
        settings.retain_fraction_of_observed_press * sum_z,
    )
    retain_finger = min(retain_finger, 0.8 * engage_finger)
    retain_sum = min(retain_sum, 0.8 * engage_sum)
    return replace(
        base,
        press_engage_finger_z=engage_finger,
        press_engage_sum_z=engage_sum,
        press_retain_finger_z=retain_finger,
        press_retain_sum_z=retain_sum,
    )


def bootstrap_medicine_bottle_tactile(
    unloaded: GF220ClassifierWindows,
    grasp: GF220ClassifierWindows,
    axial_press: GF220ClassifierWindows,
    *,
    seed_adapter: GF220WrenchAdapterConfig,
    tactile_config: TactileTwistConfig,
    bottle_axis_task: np.ndarray,
    axis_point_task_m: np.ndarray,
    config: MedicineBottleTactileBootstrapConfig = MedicineBottleTactileBootstrapConfig(),
) -> MedicineBottleTactileBootstrapResult:
    """Estimate a session adapter and noise-normalized press gates.

    ``seed_adapter`` contributes only identity-independent geometry and timing
    limits: serial/side mapping, task frame, contact positions, and declared
    FORCE6D units provenance.  Its rotations and biases are replaced.
    """

    if not isinstance(seed_adapter, GF220WrenchAdapterConfig):
        raise TypeError("seed_adapter must be GF220WrenchAdapterConfig")
    if not isinstance(tactile_config, TactileTwistConfig):
        raise TypeError("tactile_config must be TactileTwistConfig")
    if tactile_config.profile_id != seed_adapter.profile_id:
        raise ValueError("tactile and seed-adapter profile ids differ")
    axis = _unit(np.asarray(bottle_axis_task, dtype=float), name="bottle_axis_task")
    axis_point = np.asarray(axis_point_task_m, dtype=float)
    if axis_point.shape != (3,) or not np.isfinite(axis_point).all():
        raise ValueError("axis_point_task_m must contain three finite values")

    serials = {sensor.serial for sensor in seed_adapter.sensors}
    min_samples = max(config.min_samples, seed_adapter.min_samples)
    stages = {
        "unloaded": _extract_stage(
            unloaded,
            expected_serials=serials,
            min_samples=min_samples,
            max_pair_skew_s=seed_adapter.max_pair_skew_s,
            stage="unloaded",
        ),
        "grasp": _extract_stage(
            grasp,
            expected_serials=serials,
            min_samples=min_samples,
            max_pair_skew_s=seed_adapter.max_pair_skew_s,
            stage="grasp",
        ),
        "axial_press": _extract_stage(
            axial_press,
            expected_serials=serials,
            min_samples=min_samples,
            max_pair_skew_s=seed_adapter.max_pair_skew_s,
            stage="axial_press",
        ),
    }
    for serial in serials:
        calibration_ids = {stages[name][serial].calibration_id for name in stages}
        seed = next(
            sensor for sensor in seed_adapter.sensors if sensor.serial == serial
        )
        if calibration_ids != {seed.calibration_id}:
            raise ValueError(f"{serial}: calibration id changed across bootstrap stages")
        if not (
            stages["unloaded"][serial].capture_time_s[-1]
            < stages["grasp"][serial].capture_time_s[0]
            and stages["grasp"][serial].capture_time_s[-1]
            < stages["axial_press"][serial].capture_time_s[0]
        ):
            raise ValueError(f"{serial}: bootstrap windows overlap or are out of order")

    session_hash = _window_digest(stages)
    estimated: list[GF220WrenchSensorCalibration] = []
    sensor_reports: dict[str, Any] = {}
    radial_units: list[np.ndarray] = []
    task_press_axis = tactile_config.press_reaction_sign * axis
    for seed in seed_adapter.sensors:
        serial = seed.serial
        raw_unloaded = stages["unloaded"][serial].wrench
        raw_grasp = stages["grasp"][serial].wrench
        raw_press = stages["axial_press"][serial].wrench
        bias = np.median(raw_unloaded, axis=0)
        grasp_delta = np.median(raw_grasp[:, :3], axis=0) - bias[:3]
        press_delta = np.median(raw_press[:, :3], axis=0) - np.median(
            raw_grasp[:, :3], axis=0
        )
        unloaded_noise = float(np.linalg.norm(_robust_scale(raw_unloaded[:, :3])))
        grasp_noise = float(np.linalg.norm(_robust_scale(raw_grasp[:, :3])))
        press_noise = float(np.linalg.norm(_robust_scale(raw_press[:, :3])))
        normal_snr = float(np.linalg.norm(grasp_delta)) / math.hypot(
            unloaded_noise, grasp_noise
        )
        axial_snr = float(np.linalg.norm(press_delta)) / math.hypot(
            grasp_noise, press_noise
        )
        if normal_snr < config.min_signal_to_noise:
            raise TactileBootstrapQualityError(
                "grasp_response_snr_below_minimum",
                f"{serial}: grasp response SNR {normal_snr:.3f} is below "
                f"{config.min_signal_to_noise:.3f}",
                details={
                    "serial": serial,
                    "observed_snr": normal_snr,
                    "minimum_snr": config.min_signal_to_noise,
                },
            )
        if axial_snr < config.min_signal_to_noise:
            raise TactileBootstrapQualityError(
                "axial_press_response_snr_below_minimum",
                f"{serial}: axial-press response SNR {axial_snr:.3f} is below "
                f"{config.min_signal_to_noise:.3f}",
                details={
                    "serial": serial,
                    "observed_snr": axial_snr,
                    "minimum_snr": config.min_signal_to_noise,
                },
            )
        radial = np.asarray(seed.contact_position_task_m, dtype=float) - axis_point
        radial -= float(np.dot(radial, axis)) * axis
        radial = _unit(radial, name=f"{serial} contact radial")
        rotation, direction_cosine = _rotation_from_two_responses(
            normal_sensor=grasp_delta,
            axial_sensor=press_delta,
            radial_task=radial,
            axial_task=task_press_axis,
            max_direction_cosine=config.max_direction_cosine,
        )
        provenance = dict(seed.provenance)
        provenance["task_sensor_rotation_id"] = (
            f"task-session-bootstrap:{session_hash}:rotation:{serial}"
        )
        provenance["force6d_bias_id"] = (
            f"task-session-bootstrap:{session_hash}:unloaded-median:{serial}"
        )
        estimated.append(
            GF220WrenchSensorCalibration(
                serial=serial,
                logical_side=seed.logical_side,
                calibration_id=seed.calibration_id,
                R_task_sensor=rotation,
                contact_position_task_m=seed.contact_position_task_m,
                force6d_bias=bias,
                provenance=provenance,
            )
        )
        radial_units.append(radial)
        sensor_reports[serial] = {
            "logical_side": seed.logical_side,
            "calibration_id": seed.calibration_id,
            "unloaded_force6d_bias": bias.tolist(),
            "grasp_force_delta_sensor": grasp_delta.tolist(),
            "axial_press_force_delta_sensor": press_delta.tolist(),
            "grasp_signal_to_noise": normal_snr,
            "axial_press_signal_to_noise": axial_snr,
            "absolute_direction_cosine": direction_cosine,
            "R_task_sensor": rotation.tolist(),
            "contact_position_task_m": seed.contact_position_task_m.tolist(),
            "evidence": {
                name: {
                    "samples": int(stages[name][serial].wrench.shape[0]),
                    "sdk_time_s": [
                        float(stages[name][serial].sdk_time_s[0]),
                        float(stages[name][serial].sdk_time_s[-1]),
                    ],
                    "capture_monotonic_s": [
                        float(stages[name][serial].capture_time_s[0]),
                        float(stages[name][serial].capture_time_s[-1]),
                    ],
                }
                for name in stages
            },
        }

    if float(np.dot(radial_units[0], radial_units[1])) > -config.min_opposed_radial_cosine:
        raise ValueError("left/right configured contact positions are not opposed")

    adapter = GF220WrenchAdapterConfig(
        profile_id=seed_adapter.profile_id,
        task_frame_id=seed_adapter.task_frame_id,
        sensors=(estimated[0], estimated[1]),
        min_samples=seed_adapter.min_samples,
        max_age_s=seed_adapter.max_age_s,
        max_pair_skew_s=seed_adapter.max_pair_skew_s,
        future_tolerance_s=seed_adapter.future_tolerance_s,
    )
    unloaded_task = adapt_gf220_force6d_windows(
        unloaded, adapter, now_s=unloaded.collected_monotonic_s
    )
    grasp_task = adapt_gf220_force6d_windows(
        grasp, adapter, now_s=grasp.collected_monotonic_s
    )
    press_task = adapt_gf220_force6d_windows(
        axial_press, adapter, now_s=axial_press.collected_monotonic_s
    )
    baseline = build_tactile_twist_baseline(
        grasp_task.left,
        grasp_task.right,
        bottle_axis_task=axis,
        axis_point_task_m=axis_point,
        config=tactile_config,
        now_s=grasp.collected_monotonic_s,
    )
    reaction_sign = tactile_config.press_reaction_sign
    left_delta = reaction_sign * (
        float(np.median(press_task.left.wrench_task[:, :3] @ axis))
        - baseline.left_axis_center
    )
    right_delta = reaction_sign * (
        float(np.median(press_task.right.wrench_task[:, :3] @ axis))
        - baseline.right_axis_center
    )
    left_z = left_delta / baseline.left_axis_scale
    right_z = right_delta / baseline.right_axis_scale
    sum_z = (left_delta + right_delta) / baseline.press_sum_scale
    absolute_sum = abs(left_delta) + abs(right_delta)
    imbalance = (
        abs(abs(left_delta) - abs(right_delta)) / absolute_sum
        if absolute_sum > 1e-15
        else math.inf
    )
    if imbalance > tactile_config.max_press_imbalance_enter:
        raise TactileBootstrapQualityError(
            "axial_press_imbalance_exceeded",
            "known axial press is too imbalanced for bootstrap "
            f"({imbalance:.6f} > {tactile_config.max_press_imbalance_enter:.6f})",
            details={
                "observed_imbalance": imbalance,
                "maximum_imbalance": tactile_config.max_press_imbalance_enter,
            },
        )
    adaptive = _adaptive_press_thresholds(
        tactile_config,
        left_z=left_z,
        right_z=right_z,
        sum_z=sum_z,
        settings=config,
    )
    side_to_sensor = {sensor.logical_side: sensor for sensor in adapter.sensors}
    side_to_window = {"left": grasp_task.left, "right": grasp_task.right}
    side_to_unloaded = {"left": unloaded_task.left, "right": unloaded_task.right}
    contact_thresholds: dict[str, float] = {}
    contact_report: dict[str, Any] = {}
    for side in ("left", "right"):
        sensor = side_to_sensor[side]
        radial = np.asarray(sensor.contact_position_task_m, dtype=float) - axis_point
        radial -= float(np.dot(radial, axis)) * axis
        radial = _unit(radial, name=f"{side} contact radial")
        grasp_normal = side_to_window[side].wrench_task[:, :3] @ radial
        unloaded_normal = side_to_unloaded[side].wrench_task[:, :3] @ radial
        observed = float(np.median(grasp_normal) - np.median(unloaded_normal))
        noise = float(_robust_scale(unloaded_normal[:, None])[0])
        threshold = max(config.min_signal_to_noise * noise, 0.30 * observed)
        if observed <= 0.0 or threshold >= 0.90 * observed:
            raise TactileBootstrapQualityError(
                "bilateral_contact_gate_headroom_insufficient",
                f"{side}: bilateral contact response lacks threshold headroom",
                details={
                    "side": side,
                    "observed_normal": observed,
                    "candidate_threshold": threshold,
                },
            )
        contact_thresholds[side] = threshold
        contact_report[side] = {
            "observed_normal_sensor_units": observed,
            "unloaded_normal_noise_sensor_units": noise,
            "minimum_normal_sensor_units": threshold,
        }
    release_axis_delta = (
        config.release_axis_noise_multiplier * baseline.press_sum_scale
    )
    release_tangent_delta = (
        config.release_tangent_noise_multiplier * baseline.drive_scale
    )
    report = {
        "schema": BOOTSTRAP_SCHEMA,
        "status": "qualified_for_this_task_session",
        "session_evidence_sha256": session_hash,
        "profile_id": adapter.profile_id,
        "task_frame_id": adapter.task_frame_id,
        "policy": {
            "force6d_bias": "per-sensor median of unloaded window",
            "radial_direction": "grasp response mapped to configured contact radial",
            "axial_direction": "known downpress response mapped to signed bottle axis",
            "formal_twist": (
                "fixed geometric microsteps; tactile supervises retention, "
                "slip, imbalance, and release"
            ),
        },
        "quality_gates": asdict(config),
        "sensors": sensor_reports,
        "known_press": {
            "left_delta_sensor_units": left_delta,
            "right_delta_sensor_units": right_delta,
            "left_z": left_z,
            "right_z": right_z,
            "sum_z": sum_z,
            "imbalance": imbalance,
        },
        "bilateral_contact_gate": contact_report,
        "adaptive_tactile_config": asdict(adaptive),
        "effective_sensor_unit_thresholds": {
            "press_engage_left": adaptive.press_engage_finger_z
            * baseline.left_axis_scale,
            "press_engage_right": adaptive.press_engage_finger_z
            * baseline.right_axis_scale,
            "press_engage_sum": adaptive.press_engage_sum_z
            * baseline.press_sum_scale,
            "press_retain_left": adaptive.press_retain_finger_z
            * baseline.left_axis_scale,
            "press_retain_right": adaptive.press_retain_finger_z
            * baseline.right_axis_scale,
            "press_retain_sum": adaptive.press_retain_sum_z
            * baseline.press_sum_scale,
            "twist_engage_drive": adaptive.engage_drive_z * baseline.drive_scale,
            "twist_retain_drive": adaptive.retain_drive_z * baseline.drive_scale,
            "release_min_peak_drive": adaptive.release_min_peak_z
            * baseline.drive_scale,
            "release_verify_axis_delta": release_axis_delta,
            "release_verify_tangential_delta": release_tangent_delta,
        },
    }
    return MedicineBottleTactileBootstrapResult(
        adapter=adapter,
        tactile_config=adaptive,
        contact_normal_threshold_by_side=contact_thresholds,
        release_axis_delta_sensor_units=release_axis_delta,
        release_tangent_delta_sensor_units=release_tangent_delta,
        report=report,
    )


def bootstrap_medicine_bottle_tactile_presence(
    unloaded: GF220ClassifierWindows,
    grasp: GF220ClassifierWindows,
    axial_press: GF220ClassifierWindows,
    *,
    seed_adapter: GF220WrenchAdapterConfig,
    tactile_config: TactileTwistConfig,
    release_min_total_twist_rad: float = 2.0 * math.pi,
    config: MedicineBottlePresenceBootstrapConfig = (
        MedicineBottlePresenceBootstrapConfig()
    ),
) -> MedicineBottlePresenceBootstrapResult:
    """Bootstrap the evidence-supported scalar any-pad presence mode.

    Only raw SDK ``Fz`` changes are interpreted.  The unloaded median is
    removed, while rotation, direction, SNR, bilateral balance, and moment are
    deliberately left unclaimed.
    """

    if not isinstance(seed_adapter, GF220WrenchAdapterConfig):
        raise TypeError("seed_adapter must be GF220WrenchAdapterConfig")
    if tactile_config.profile_id != seed_adapter.profile_id:
        raise ValueError("tactile and seed-adapter profile ids differ")
    if (
        not math.isfinite(release_min_total_twist_rad)
        or release_min_total_twist_rad <= 0.0
    ):
        raise ValueError("release_min_total_twist_rad must be positive and finite")
    serials = {sensor.serial for sensor in seed_adapter.sensors}
    min_samples = max(config.min_samples, seed_adapter.min_samples)
    stages = {
        "unloaded": _extract_stage(
            unloaded,
            expected_serials=serials,
            min_samples=min_samples,
            max_pair_skew_s=seed_adapter.max_pair_skew_s,
            stage="unloaded",
        ),
        "grasp": _extract_stage(
            grasp,
            expected_serials=serials,
            min_samples=min_samples,
            max_pair_skew_s=seed_adapter.max_pair_skew_s,
            stage="grasp",
        ),
        "axial_press": _extract_stage(
            axial_press,
            expected_serials=serials,
            min_samples=min_samples,
            max_pair_skew_s=seed_adapter.max_pair_skew_s,
            stage="axial_press",
        ),
    }
    for seed in seed_adapter.sensors:
        windows = [stages[name][seed.serial] for name in stages]
        if {window.calibration_id for window in windows} != {seed.calibration_id}:
            raise ValueError(
                f"{seed.serial}: calibration id changed across bootstrap stages"
            )
        if not (
            windows[0].capture_time_s[-1] < windows[1].capture_time_s[0]
            and windows[1].capture_time_s[-1] < windows[2].capture_time_s[0]
        ):
            raise ValueError(
                f"{seed.serial}: bootstrap windows overlap or are out of order"
            )

    session_hash = _window_digest(stages)
    estimated: list[GF220WrenchSensorCalibration] = []
    sensor_reports: dict[str, Any] = {}
    grasp_changes: list[float] = []
    press_changes: list[float] = []
    for seed in seed_adapter.sensors:
        serial = seed.serial
        raw_unloaded = stages["unloaded"][serial].wrench
        raw_grasp = stages["grasp"][serial].wrench
        raw_press = stages["axial_press"][serial].wrench
        bias = np.median(raw_unloaded, axis=0)
        unloaded_fz = float(bias[2])
        grasp_fz = float(np.median(raw_grasp[:, 2]))
        press_fz = float(np.median(raw_press[:, 2]))
        grasp_change = abs(grasp_fz - unloaded_fz)
        press_change = abs(press_fz - grasp_fz)
        grasp_changes.append(grasp_change)
        press_changes.append(press_change)
        provenance = dict(seed.provenance)
        provenance["task_sensor_rotation_id"] = (
            f"presence-only:{session_hash}:identity-preserves-raw-fz:{serial}"
        )
        provenance["force6d_bias_id"] = (
            f"task-session-presence:{session_hash}:unloaded-median:{serial}"
        )
        estimated.append(
            GF220WrenchSensorCalibration(
                serial=serial,
                logical_side=seed.logical_side,
                calibration_id=seed.calibration_id,
                R_task_sensor=np.eye(3),
                contact_position_task_m=seed.contact_position_task_m,
                force6d_bias=bias,
                provenance=provenance,
            )
        )
        sensor_reports[serial] = {
            "logical_side": seed.logical_side,
            "calibration_id": seed.calibration_id,
            "unloaded_force6d_bias": bias.tolist(),
            "unloaded_fz_sensor_units": unloaded_fz,
            "grasp_fz_sensor_units": grasp_fz,
            "axial_press_fz_sensor_units": press_fz,
            "absolute_grasp_fz_change": grasp_change,
            "absolute_press_fz_change": press_change,
        }

    grasp_response = max(grasp_changes)
    press_response = max(press_changes)
    minimum = config.minimum_fz_change_sensor_units
    if grasp_response < minimum:
        raise TactileBootstrapQualityError(
            "grasp_tactile_response_absent",
            "neither GF220 pad changed Fz after the Hand-E grasp",
            details={
                "observed_any_pad_fz_change": grasp_response,
                "minimum_change": minimum,
            },
        )
    if press_response < minimum:
        raise TactileBootstrapQualityError(
            "axial_press_tactile_response_absent",
            "neither GF220 pad changed Fz after the bounded axial press",
            details={
                "observed_any_pad_fz_change": press_response,
                "minimum_change": minimum,
            },
        )

    contact_threshold = max(minimum, config.contact_retain_fraction * grasp_response)
    press_engage = max(minimum, config.press_engage_fraction * press_response)
    press_retain = max(minimum, config.press_retain_fraction * press_response)
    unload_return = max(minimum, config.unload_return_fraction * grasp_response)
    release_delta = max(
        minimum,
        config.release_delta_fraction * max(grasp_response, press_response),
    )
    adapter = GF220WrenchAdapterConfig(
        profile_id=seed_adapter.profile_id,
        task_frame_id=seed_adapter.task_frame_id,
        sensors=(estimated[0], estimated[1]),
        min_samples=seed_adapter.min_samples,
        max_age_s=seed_adapter.max_age_s,
        max_pair_skew_s=seed_adapter.max_pair_skew_s,
        future_tolerance_s=seed_adapter.future_tolerance_s,
    )
    presence_config = MedicineBottlePresenceConfig(
        profile_id=seed_adapter.profile_id,
        min_unique_samples=tactile_config.min_unique_samples,
        max_age_s=tactile_config.max_age_s,
        max_pair_skew_s=tactile_config.max_pair_skew_s,
        min_window_span_s=tactile_config.min_window_span_s,
        contact_retain_fz=contact_threshold,
        press_engage_fz=press_engage,
        press_retain_fz=press_retain,
        unload_return_fz=unload_return,
        press_required_consecutive_windows=1,
        wrist_reset_fraction=tactile_config.wrist_reset_fraction,
        release_min_total_twist_rad=release_min_total_twist_rad,
        release_resistance_peak_fraction=tactile_config.release_drive_peak_fraction,
        release_min_peak_fz=press_engage,
        release_required_consecutive_windows=tactile_config.release_consecutive_windows,
    )
    report = {
        "schema": BOOTSTRAP_SCHEMA,
        "status": "qualified_presence_and_release_proxy_for_this_task_session",
        "mode": "any_pad_raw_fz_presence_plus_press_aligned_release_proxy",
        "session_evidence_sha256": session_hash,
        "profile_id": adapter.profile_id,
        "task_frame_id": adapter.task_frame_id,
        "policy": {
            "force6d_bias": "per-sensor median of unloaded window",
            "presence": "maximum absolute raw SDK Fz change across either pad",
            "release_proxy": (
                "sum of each pad's absolute raw SDK Fz change from the segment "
                "baseline; the known downpress records per-pad sign for diagnostics"
            ),
            "sensor_rotation": "not estimated; identity preserves raw SDK Fz",
            "formal_twist": (
                "bounded geometric rotation; after the minimum turn, release is "
                "requested only when the press-aligned raw-Fz resistance proxy "
                "falls from its measured peak for consecutive fresh windows"
            ),
            "explicitly_not_used": [
                "signal_to_noise_gate",
                "bilateral_balance_gate",
                "response_direction_gate",
                "3d_rotation_identification",
                "twist_moment_gate",
            ],
        },
        "quality_gates": asdict(config),
        "sensors": sensor_reports,
        "observed_any_pad": {
            "grasp_fz_change": grasp_response,
            "axial_press_fz_change": press_response,
        },
        "presence_config": asdict(presence_config),
        "effective_sensor_unit_thresholds": {
            "contact_retain_fz": contact_threshold,
            "press_engage_fz": press_engage,
            "press_retain_fz": press_retain,
            "unload_return_fz": unload_return,
            "release_verify_delta": release_delta,
        },
    }
    return MedicineBottlePresenceBootstrapResult(
        adapter=adapter,
        presence_config=presence_config,
        contact_presence_threshold_sensor_units=contact_threshold,
        release_axis_delta_sensor_units=release_delta,
        release_tangent_delta_sensor_units=release_delta,
        report=report,
    )


__all__ = [
    "BOOTSTRAP_SCHEMA",
    "MedicineBottleTactileBootstrapConfig",
    "MedicineBottleTactileBootstrapResult",
    "MedicineBottlePresenceBootstrapConfig",
    "MedicineBottlePresenceBootstrapResult",
    "TactileBootstrapQualityError",
    "bootstrap_medicine_bottle_tactile",
    "bootstrap_medicine_bottle_tactile_presence",
]
