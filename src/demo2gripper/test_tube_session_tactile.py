"""Session-scoped tactile bootstrap for the resident test-tube task.

This interpreter deliberately separates *installation geometry* from values that
must be learned again every time the sensors start.  It derives an unloaded
FORCE6D/image-noise envelope from fresh windows while the Hand-E is open, then
uses only deltas from that envelope for first contact, bilateral contact, slip,
pivot and insertion events.  No pre-recorded force threshold or labelled jam
receipt is consumed.

The module owns no hardware and emits no commands.  A caller still bounds every
motion and the outer task controller remains responsible for stopping/holding.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .gf220_live import GF220Calibration, GF220ClassifierWindows
from .test_tube_closed_loop import GripperFeedback, TactileFeedback
from .test_tube_preflight import TubeState
from .test_tube_tactile_supervisor import compensated_residual, residual_contact_geometry

SESSION_BOOTSTRAP_SCHEMA = "demo2gripper.test_tube_session_tactile_bootstrap.v1"
SESSION_AUDIT_SCHEMA = "demo2gripper.test_tube_session_tactile_audit.v1"

# Standard opposed-pad Hand-E installation assumption.  Pad image coordinates
# are ``(+u right, +v down, +n out of the image)``.  The canonical TCP is
# ``(+X jaw closing, +Z approach)``.  These proper rotations make the two
# mirrored pad images describe the same physical line in the TCP Y-Z plane.
# This is an explicit approximation for the current rig, not a measured pad
# extrinsic calibration.
_STANDARD_R_TCP_PAD_BY_SIDE = {
    "left": np.asarray(
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    ),
    "right": np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    ),
}
_JAW_AXIS_TCP = np.asarray([1.0, 0.0, 0.0], dtype=float)
_GRAVITY_DIRECTION_BASE = np.asarray([0.0, 0.0, -1.0], dtype=float)


class SessionTactileError(ValueError):
    """The live window or bounded bootstrap policy is invalid."""


@dataclass(frozen=True)
class SessionTactileConfig:
    """Dimensionless/session-derived bounds; no sensor-unit threshold lives here."""

    unloaded_windows: int = 3
    min_unique_samples: int = 4
    max_age_s: float = 0.25
    max_pair_skew_s: float = 0.08
    unloaded_min_width_m: float = 0.040
    force_noise_multiplier: float = 8.0
    image_noise_multiplier: float = 8.0
    retain_threshold_fraction: float = 0.50
    min_contact_area_fraction: float = 0.002
    contact_frame_fraction: float = 0.50
    contact_confirmation_windows: int = 1
    contact_loss_confirmation_windows: int = 2
    bilateral_confirmation_windows: int = 2
    interior_contact_confirmation_windows: int = 2
    contact_center_edge_margin_fraction: float = 0.10
    slip_center_fraction_of_patch_radius: float = 0.35
    pivot_slip_center_fraction_of_patch_radius: float = 1.0
    slip_confirmation_windows: int = 2
    pivot_target_delta_deg: float = 45.0
    pivot_target_tolerance_deg: float = 8.0
    pivot_pad_disagreement_deg: float = 12.0
    pivot_confirmation_windows: int = 2
    gravity_pivot_lever_tcp: tuple[float, float, float] | None = None
    pivot_direction_min_delta_deg: float = 3.0
    pivot_direction_min_moment: float = 0.05
    rotation_stop_delta_deg: float = 2.0
    rotation_stop_confirmation_windows: int = 2
    insertion_resistance_noise_multiplier: float = 10.0
    insertion_reference_windows: int = 3
    insertion_progress_ratio_min: float = 0.25
    insertion_balance_max_fraction: float = 0.45
    insertion_event_confirmation_windows: int = 2
    minimum_successful_steps_before_seat: int = 2

    def __post_init__(self) -> None:
        for name in (
            "unloaded_windows",
            "min_unique_samples",
            "contact_confirmation_windows",
            "contact_loss_confirmation_windows",
            "bilateral_confirmation_windows",
            "interior_contact_confirmation_windows",
            "slip_confirmation_windows",
            "pivot_confirmation_windows",
            "rotation_stop_confirmation_windows",
            "insertion_event_confirmation_windows",
            "minimum_successful_steps_before_seat",
            "insertion_reference_windows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_unique_samples < 2:
            raise ValueError("min_unique_samples must be at least two")
        for name in (
            "max_age_s",
            "max_pair_skew_s",
            "unloaded_min_width_m",
            "force_noise_multiplier",
            "image_noise_multiplier",
            "slip_center_fraction_of_patch_radius",
            "pivot_slip_center_fraction_of_patch_radius",
            "pivot_target_delta_deg",
            "pivot_target_tolerance_deg",
            "pivot_pad_disagreement_deg",
            "pivot_direction_min_delta_deg",
            "pivot_direction_min_moment",
            "rotation_stop_delta_deg",
            "insertion_resistance_noise_multiplier",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "retain_threshold_fraction",
            "min_contact_area_fraction",
            "contact_frame_fraction",
            "contact_center_edge_margin_fraction",
            "insertion_progress_ratio_min",
            "insertion_balance_max_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
        if self.contact_center_edge_margin_fraction >= 0.5:
            raise ValueError("contact_center_edge_margin_fraction must be below 0.5")
        if self.pivot_target_delta_deg + self.pivot_target_tolerance_deg >= 90.0:
            raise ValueError("pivot target plus tolerance must be below 90 degrees")
        if self.pivot_direction_min_delta_deg >= (
            self.pivot_target_delta_deg - self.pivot_target_tolerance_deg
        ):
            raise ValueError(
                "pivot_direction_min_delta_deg must be below the target-angle band"
            )
        if self.pivot_direction_min_moment >= 1.0:
            raise ValueError("pivot_direction_min_moment must be below one")
        if self.gravity_pivot_lever_tcp is not None:
            lever = np.asarray(self.gravity_pivot_lever_tcp, dtype=float)
            if lever.shape != (3,) or not np.isfinite(lever).all():
                raise ValueError("gravity_pivot_lever_tcp must be a finite 3-vector")
            norm = float(np.linalg.norm(lever))
            if norm <= 1e-9:
                raise ValueError("gravity_pivot_lever_tcp must be nonzero")
            object.__setattr__(
                self,
                "gravity_pivot_lever_tcp",
                tuple(float(value) for value in lever / norm),
            )


@dataclass(frozen=True)
class _WindowSummary:
    serial: str
    calibration_id: str
    force_samples: np.ndarray
    force_median: np.ndarray
    warped: np.ndarray
    sdk_end_s: float
    host_end_s: float


@dataclass(frozen=True)
class _PadBaseline:
    serial: str
    calibration_id: str
    force_vector: np.ndarray
    force_noise: float
    image_reference: np.ndarray
    image_residual_threshold: float
    min_contact_area_px: int


@dataclass(frozen=True)
class _PadObservation:
    serial: str
    force_delta: float
    force_active: bool
    image_active: bool
    center_interior: bool
    active: bool
    center_uv: tuple[float, float] | None
    axis_deg: float | None
    area_px: float | None


class SessionBootstrapTestTubeTactileInterpreter:
    """Learn session thresholds online and emit debounced task events."""

    _BASELINE_STATES = {TubeState.OBSERVE}
    _PIVOT_STATES = {
        TubeState.PIVOT_MICRO_LOOSEN,
        TubeState.PIVOT_EVALUATE,
        TubeState.RECLAMP,
        TubeState.RECLAMP_RECOVERY,
    }
    _INSERT_STATES = {
        TubeState.INSERT_MICROSTEP,
        TubeState.INSERT_EVALUATE,
        TubeState.JAM_RECOVERY_RETRACT,
    }

    def __init__(
        self,
        *,
        calibrations: Sequence[GF220Calibration],
        output_dir: Path,
        config: SessionTactileConfig = SessionTactileConfig(),
    ) -> None:
        if isinstance(calibrations, (str, bytes)) or len(calibrations) != 2:
            raise SessionTactileError("calibrations must contain exactly two pads")
        if not all(isinstance(item, GF220Calibration) for item in calibrations):
            raise SessionTactileError("calibrations must contain GF220Calibration values")
        by_serial = {item.serial: item for item in calibrations}
        if len(by_serial) != 2:
            raise SessionTactileError("GF220 serials must be unique")
        self.config = config
        self._calibrations = by_serial
        # Physical left/right sign is intentionally not assumed.  The outer
        # controller probes/reverses the jaw correction sign.  These labels only
        # keep the two serials distinct in TactileFeedback/audit evidence.
        ordered = sorted(by_serial)
        self._side = {ordered[0]: "left", ordered[1]: "right"}
        self._baseline_windows: list[dict[str, _WindowSummary]] = []
        self._baselines: dict[str, _PadBaseline] | None = None
        self._session_id: str | None = None
        self._last_sdk: dict[str, float] = {}
        self._last_host: dict[str, float] = {}
        self._contact_present = False
        self._contact_windows = 0
        self._loss_windows = 0
        self._bilateral_windows = 0
        self._interior_bilateral_windows = 0
        self._previous_centers: dict[str, tuple[float, float]] = {}
        self._previous_axes: dict[str, float] = {}
        self._hold_axes: dict[str, float] = {}
        self._pivot_reference_axes: dict[str, float] | None = None
        self._pivot_expected_direction_sign: int | None = None
        self._pivot_expected_moment: float | None = None
        self._pivot_target_windows = 0
        self._pivot_target_latched = False
        self._pivot_wrong_direction_windows = 0
        self._rotation_stop_windows = 0
        self._slip_windows = 0
        self._steady_contact_serials: set[str] = set()
        self._insertion_reference: dict[str, np.ndarray] | None = None
        self._insertion_reference_windows: list[dict[str, np.ndarray]] = []
        self._insertion_noise = 0.0
        self._successful_insert_steps = 0
        self._insert_cycle_counted = False
        self._jam_windows = 0
        self._seat_windows = 0
        self._last_state: TubeState | None = None
        self._sequence = 0
        self._directory = Path(output_dir) / "tactile_interpreter"
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def session_ready(self) -> bool:
        return self._baselines is not None

    def evaluate(
        self,
        *,
        windows: GF220ClassifierWindows,
        state: TubeState,
        now_s: float,
        T_base_tcp: np.ndarray,
        gripper: GripperFeedback,
        commanded_insertion_step_m: float | None,
        measured_insertion_progress_m: float | None,
    ) -> TactileFeedback:
        state = TubeState(state)
        observed_at = _observed_at(windows, now_s)
        blockers: list[str] = []
        summary: dict[str, _WindowSummary] | None = None
        try:
            summary = self._summarize(windows, now_s=now_s)
        except (TypeError, ValueError) as error:
            blockers.append(str(error))

        if summary is not None and self._baselines is None:
            if state not in self._BASELINE_STATES:
                blockers.append("unloaded_session_baseline_not_ready_before_motion")
            elif gripper.moving or gripper.object_detected:
                blockers.append("unloaded_baseline_requires_stationary_empty_gripper")
            elif gripper.width_m < self.config.unloaded_min_width_m:
                blockers.append("unloaded_baseline_requires_open_gripper")
            else:
                self._baseline_windows.append(summary)
                if len(self._baseline_windows) >= self.config.unloaded_windows:
                    self._fit_baseline()
                else:
                    blockers.append("collecting_unloaded_session_baseline")

        if summary is None or self._baselines is None:
            if state is TubeState.ALIGN_SOCKET:
                self._clear_insertion_reference()
            feedback = TactileFeedback(valid=False, observed_at_s=observed_at)
            self._write_audit(
                state=state,
                feedback=feedback,
                blockers=blockers or ["session_tactile_unavailable"],
                pads=(),
                signals={},
            )
            self._last_state = state
            return feedback

        pads = self._observe_pads(summary)
        # Optical deformation is the direct finger-pad contact observation.
        # FORCE6D is deliberately not required here: the unloaded force noise
        # envelope can be larger than the light first contact on a thin tube.
        # Force-qualified ``active`` pads are still used below for load/slip,
        # pivot direction and insertion resistance.
        optical = tuple(item for item in pads if item.image_active)
        seen = bool(optical)
        self._contact_windows = self._contact_windows + 1 if seen else 0
        self._loss_windows = self._loss_windows + 1 if not seen else 0
        if not self._contact_present and (
            self._contact_windows >= self.config.contact_confirmation_windows
        ):
            self._contact_present = True
            self._loss_windows = 0
        elif self._contact_present and (
            self._loss_windows >= self.config.contact_loss_confirmation_windows
        ):
            self._contact_present = False
            self._contact_windows = 0

        bilateral_raw = len(optical) == 2
        self._bilateral_windows = self._bilateral_windows + 1 if bilateral_raw else 0
        bilateral = bool(
            self._contact_present
            and self._bilateral_windows >= self.config.bilateral_confirmation_windows
        )
        interior_raw = bool(
            bilateral_raw and all(item.center_interior for item in optical)
        )
        self._interior_bilateral_windows = (
            self._interior_bilateral_windows + 1 if interior_raw else 0
        )
        bilateral_interior = bool(
            bilateral
            and self._interior_bilateral_windows
            >= self.config.interior_contact_confirmation_windows
        )
        single_side = None
        if self._contact_present and len(optical) == 1:
            single_side = self._side[optical[0].serial]

        slip, slip_signals = self._slip_event(pads, state=state)
        (
            pivot_reached,
            rotation_stopped,
            pivot_direction_fault,
            pivot_signals,
        ) = self._pivot_events(
            pads,
            state=state,
            bilateral=bilateral,
            T_base_tcp=T_base_tcp,
        )
        if pivot_direction_fault:
            # The outer state machine already gives ``slip_detected`` immediate
            # precedence over angle progress and issues a bounded RECLAMP.  A
            # confirmed opposite gravity-pivot direction uses that same safe
            # recovery path rather than adding a second motion command type.
            slip = True
            slip_signals = {
                **slip_signals,
                "gravity_pivot_direction_fault_reclamp": True,
            }
        resistance_high, jam, seated, insertion_signals = self._insertion_events(
            summary,
            state=state,
            bilateral=bilateral,
            slip_detected=slip,
            commanded_step_m=commanded_insertion_step_m,
            measured_progress_m=measured_insertion_progress_m,
        )

        feedback = TactileFeedback(
            valid=True,
            observed_at_s=observed_at,
            contact_present=self._contact_present,
            bilateral_contact=bilateral,
            bilateral_contact_interior=bilateral_interior,
            single_contact_side=single_side,
            slip_detected=slip,
            relative_angle_target_reached=pivot_reached,
            rotation_stopped=rotation_stopped,
            insertion_jam=jam,
            insertion_resistance_high=resistance_high,
            insertion_seated=seated,
            insertion_reference_ready=self._insertion_reference is not None,
        )
        self._write_audit(
            state=state,
            feedback=feedback,
            blockers=(),
            pads=pads,
            signals={
                "optical_contact_serials": [item.serial for item in optical],
                "force_qualified_contact_serials": [
                    item.serial for item in pads if item.active
                ],
                "bilateral_optical_contact_raw": bilateral_raw,
                "bilateral_interior_contact_raw": interior_raw,
                "bilateral_interior_confirmation_windows": (
                    self._interior_bilateral_windows
                ),
                **slip_signals,
                **pivot_signals,
                **insertion_signals,
            },
        )
        self._last_state = state
        return feedback

    def close(self) -> None:
        return None

    def _summarize(
        self, windows: GF220ClassifierWindows, *, now_s: float
    ) -> dict[str, _WindowSummary]:
        if not isinstance(windows, GF220ClassifierWindows):
            raise TypeError("windows must be GF220ClassifierWindows")
        if set(windows.serials) != set(self._calibrations) or len(windows.serials) != 2:
            raise SessionTactileError("live GF220 serials differ from startup calibration")
        if windows.endpoint_skew_s > self.config.max_pair_skew_s:
            raise SessionTactileError("GF220 pair endpoint skew exceeds session policy")
        result: dict[str, _WindowSummary] = {}
        for serial in sorted(self._calibrations):
            source = windows.for_serial(serial)
            calibration = self._calibrations[serial]
            if source.calibration_id != calibration.calibration_id:
                raise SessionTactileError(f"{serial}: calibration_id changed")
            force = source.force6d
            warped = source.warped
            if force is None or warped is None:
                raise SessionTactileError(f"{serial}: FORCE6D/WARPED_IMG is missing")
            f_indices = _fresh_indices(force, self.config.min_unique_samples, serial, "force")
            i_indices = _fresh_indices(warped, self.config.min_unique_samples, serial, "image")
            f_host = np.asarray(force.capture_monotonic_s[f_indices], dtype=float)
            i_host = np.asarray(warped.capture_monotonic_s[i_indices], dtype=float)
            newest = min(float(f_host[-1]), float(i_host[-1]))
            # Freshness describes when the observation became available, so it
            # is bounded by the latest common FORCE6D/image endpoint.  The
            # beginning of a multi-frame classifier window may legitimately be
            # older than max_age_s (four GF220 frames span roughly 0.23--0.31 s
            # on the live 11--13 Hz streams) even when its endpoint is fresh.
            # Window integrity is independently enforced by increasing unique
            # timestamps; replay is rejected against the previous endpoint.
            if newest < now_s - self.config.max_age_s:
                raise SessionTactileError(f"{serial}: tactile window is stale")
            if max(float(f_host[-1]), float(i_host[-1])) > now_s + 1e-6:
                raise SessionTactileError(f"{serial}: tactile window is from the future")
            sdk_end = min(
                float(force.sdk_time_s[f_indices[-1]]),
                float(warped.sdk_time_s[i_indices[-1]]),
            )
            if serial in self._last_sdk and (
                sdk_end <= self._last_sdk[serial] or newest <= self._last_host[serial]
            ):
                raise SessionTactileError(f"{serial}: tactile window overlaps or replays")
            force_values = np.asarray(force.values[f_indices], dtype=float)
            images = np.asarray(warped.values[i_indices])
            if force_values.ndim != 2 or force_values.shape[1] != 6:
                raise SessionTactileError(f"{serial}: FORCE6D must be Nx6")
            if images.ndim != 4 or images.shape[1:] != calibration.unloaded_warped.shape:
                raise SessionTactileError(f"{serial}: image shape changed")
            if not np.isfinite(force_values).all() or images.dtype != np.uint8:
                raise SessionTactileError(f"{serial}: tactile values are invalid")
            result[serial] = _WindowSummary(
                serial=serial,
                calibration_id=calibration.calibration_id,
                force_samples=force_values,
                force_median=np.median(force_values, axis=0),
                warped=images,
                sdk_end_s=sdk_end,
                host_end_s=newest,
            )
        self._last_sdk = {serial: item.sdk_end_s for serial, item in result.items()}
        self._last_host = {serial: item.host_end_s for serial, item in result.items()}
        return result

    def _fit_baseline(self) -> None:
        baselines: dict[str, _PadBaseline] = {}
        digest = hashlib.sha256(SESSION_BOOTSTRAP_SCHEMA.encode())
        for serial in sorted(self._calibrations):
            calibration = self._calibrations[serial]
            windows = [item[serial] for item in self._baseline_windows]
            force = np.concatenate([item.force_samples for item in windows], axis=0)
            force_center = np.median(force, axis=0)
            force_deviation = np.linalg.norm(force - force_center, axis=1)
            force_noise = _robust_upper_noise(force_deviation)
            images = np.concatenate([item.warped for item in windows], axis=0)
            image_reference = np.clip(
                np.rint(np.median(images.astype(np.float32), axis=0)), 0, 255
            ).astype(np.uint8)
            image_reference.setflags(write=False)
            image_scores: list[float] = []
            for image in images:
                residual = compensated_residual(
                    image,
                    image_reference,
                    np.ones(image.shape[:2], dtype=bool),
                )
                image_scores.append(float(np.quantile(residual, 0.99)))
            image_threshold = _strict_upper_bound(
                np.asarray(image_scores, dtype=float),
                self.config.image_noise_multiplier,
            )
            force_threshold = max(
                _strict_positive(force_noise * self.config.force_noise_multiplier),
                _numeric_resolution(force_center),
            )
            min_area = max(
                1,
                int(
                    math.ceil(
                        self.config.min_contact_area_fraction
                        * calibration.unloaded_warped.shape[0]
                        * calibration.unloaded_warped.shape[1]
                    )
                ),
            )
            baselines[serial] = _PadBaseline(
                serial=serial,
                calibration_id=calibration.calibration_id,
                force_vector=force_center,
                force_noise=force_threshold / self.config.force_noise_multiplier,
                image_reference=image_reference,
                image_residual_threshold=image_threshold,
                min_contact_area_px=min_area,
            )
            digest.update(serial.encode())
            digest.update(calibration.calibration_id.encode())
            digest.update(np.ascontiguousarray(force).tobytes())
            digest.update(image_reference.tobytes())
            digest.update(np.asarray(image_scores, dtype=float).tobytes())
        self._baselines = baselines
        self._session_id = f"test-tube-session-{digest.hexdigest()}"
        payload = {
            "schema": SESSION_BOOTSTRAP_SCHEMA,
            "status": "runtime_session_ready",
            "session_id": self._session_id,
            "serial_side_labels": self._side,
            "side_label_contract": (
                "deterministic serial order mapped to standard opposed Hand-E "
                "left/right pads; approximate, not measured"
                if self.config.gravity_pivot_lever_tcp is not None
                else "deterministic_serial_order; physical sign is probed by the "
                "outer controller"
            ),
            "config": asdict(self.config),
            "pads": [
                {
                    "serial": item.serial,
                    "calibration_id": item.calibration_id,
                    "force_vector": item.force_vector.tolist(),
                    "force_noise": item.force_noise,
                    "force_engage_threshold": max(
                        _strict_positive(item.force_noise * self.config.force_noise_multiplier),
                        _numeric_resolution(item.force_vector),
                    ),
                    "image_reference_sha256": hashlib.sha256(
                        item.image_reference.tobytes()
                    ).hexdigest(),
                    "image_residual_threshold": item.image_residual_threshold,
                    "min_contact_area_px": item.min_contact_area_px,
                }
                for item in baselines.values()
            ],
            "runtime_derived": [
                "unloaded_force_baseline",
                "first_contact_threshold",
                "bilateral_contact",
                "slip_event",
                "pivot_delta_event",
                *(
                    ["gravity_pivot_direction_from_tcp_pose_and_pad_axis"]
                    if self.config.gravity_pivot_lever_tcp is not None
                    else []
                ),
                "insertion_resistance_and_seat_or_jam_event",
            ],
            "hardware_commands_dispatched": False,
        }
        _atomic_json(self._directory / "session_bootstrap.json", payload)

    def _observe_pads(self, summary: Mapping[str, _WindowSummary]) -> tuple[_PadObservation, ...]:
        assert self._baselines is not None
        values: list[_PadObservation] = []
        for serial in sorted(summary):
            source = summary[serial]
            baseline = self._baselines[serial]
            force_delta = float(np.linalg.norm(source.force_median - baseline.force_vector))
            engage = max(
                _strict_positive(baseline.force_noise * self.config.force_noise_multiplier),
                _numeric_resolution(baseline.force_vector),
            )
            force_threshold = (
                engage * self.config.retain_threshold_fraction
                if self._contact_present
                else engage
            )
            geometries = []
            roi = np.ones(source.warped.shape[1:3], dtype=bool)
            for image in source.warped:
                residual = compensated_residual(
                    image, baseline.image_reference, roi
                )
                geometries.append(
                    residual_contact_geometry(
                        residual,
                        roi,
                        residual_threshold=baseline.image_residual_threshold,
                        min_contact_area_px=baseline.min_contact_area_px,
                        min_orientation_elongation=1.35,
                        min_orientation_concentration=0.25,
                    )
                )
            contacts = [item for item in geometries if item.contact_candidate]
            image_active = len(contacts) / len(geometries) >= self.config.contact_frame_fraction
            centers = [item.center_uv_px for item in contacts if item.center_uv_px is not None]
            oriented = [
                item.orientation_deg
                for item in contacts
                if item.orientation_deg is not None
            ]
            center = None
            if centers:
                center_array = np.median(np.asarray(centers, dtype=float), axis=0)
                center = (float(center_array[0]), float(center_array[1]))
            axis = _undirected_mean_deg([float(item) for item in oriented]) if oriented else None
            area = float(np.median([item.area_px for item in contacts])) if contacts else None
            force_active = force_delta >= force_threshold
            height, width = source.warped.shape[1:3]
            margin_u = self.config.contact_center_edge_margin_fraction * width
            margin_v = self.config.contact_center_edge_margin_fraction * height
            center_interior = bool(
                image_active
                and center is not None
                and margin_u <= center[0] <= (width - 1) - margin_u
                and margin_v <= center[1] <= (height - 1) - margin_v
            )
            values.append(
                _PadObservation(
                    serial=serial,
                    force_delta=force_delta,
                    force_active=force_active,
                    image_active=image_active,
                    center_interior=center_interior,
                    active=force_active and image_active,
                    center_uv=center,
                    axis_deg=axis,
                    area_px=area,
                )
            )
        return tuple(values)

    def _slip_event(
        self, pads: Sequence[_PadObservation], *, state: TubeState
    ) -> tuple[bool, dict[str, Any]]:
        # EXTRACT and LEVEL_HELD_TUBE share the HOLD tactile mode, but they do
        # not share one slip displacement reference.  The extraction trace can
        # legitimately walk a contact patch across the pad as the arm clears
        # the source.  Carrying that accumulated streak into the first
        # deliberate wrist-leveling sample makes the new phase fail before it
        # has moved.  Re-anchor the per-frame displacement test on the first
        # LEVEL sample.  Contact-loss state is intentionally retained, so a
        # missing pad is still promoted by the normal two-window debounce.
        level_reference_reset = bool(
            state is TubeState.LEVEL_HELD_TUBE
            and self._last_state is TubeState.EXTRACT
        )
        if level_reference_reset:
            self._slip_windows = 0
            self._previous_centers.clear()

        shifts: list[float] = []
        thresholds: list[float] = []
        threshold_crossings: list[bool] = []
        active_now = {item.serial for item in pads if item.active}
        pivot_contact_state = state in self._PIVOT_STATES
        center_fraction = (
            self.config.pivot_slip_center_fraction_of_patch_radius
            if pivot_contact_state
            else self.config.slip_center_fraction_of_patch_radius
        )
        for item in pads:
            if item.center_uv is None or item.serial not in self._previous_centers:
                continue
            shifts.append(
                float(
                    np.linalg.norm(
                        np.asarray(item.center_uv) - np.asarray(self._previous_centers[item.serial])
                    )
                )
            )
            radius = math.sqrt(max(float(item.area_px or 1.0), 1.0) / math.pi)
            thresholds.append(
                max(1.0, center_fraction * radius)
            )
            threshold_crossings.append(shifts[-1] >= thresholds[-1])
        steady_hold_state = state in {
            TubeState.EXTRACT,
            TubeState.LEVEL_HELD_TUBE,
            TubeState.APPROACH_SOCKET,
            TubeState.ALIGN_SOCKET,
            TubeState.INSERT_MICROSTEP,
            TubeState.INSERT_EVALUATE,
            *self._PIVOT_STATES,
        }
        if not steady_hold_state:
            self._steady_contact_serials.clear()
        elif len(active_now) == 2:
            self._steady_contact_serials = set(active_now)
        pad_loss = bool(self._steady_contact_serials - active_now)
        # Each finger has its own contact-patch scale.  A large patch on one
        # pad must not hide a displacement that is significant for the smaller
        # patch on the other pad.
        patch_motion = any(threshold_crossings)
        # During transport/insertion, losing a previously tracked contact patch
        # is itself a slip symptom: requiring a displacement from the missing
        # patch would make this branch impossible to trigger.  The same debounce
        # as patch motion keeps a single dropped image window from becoming an
        # event.
        candidate = bool(steady_hold_state and (pad_loss or patch_motion))
        self._slip_windows = self._slip_windows + 1 if candidate else 0
        confirmed = self._slip_windows >= self.config.slip_confirmation_windows
        self._previous_centers = {
            item.serial: item.center_uv
            for item in pads
            if item.active and item.center_uv is not None
        }
        return confirmed, {
            "slip_candidate": candidate,
            "contact_patch_lost": pad_loss,
            "slip_windows": self._slip_windows,
            "maximum_patch_center_shift_px": max(shifts) if shifts else None,
            "slip_threshold_px": max(thresholds) if thresholds else None,
            "pivot_contact_monitoring": pivot_contact_state,
            "slip_reference_reset_on_level_entry": level_reference_reset,
        }

    def _pivot_events(
        self,
        pads: Sequence[_PadObservation],
        *,
        state: TubeState,
        bilateral: bool,
        T_base_tcp: np.ndarray,
    ) -> tuple[bool, bool, bool, dict[str, Any]]:
        axes = {
            item.serial: item.axis_deg
            for item in pads
            if item.active and item.axis_deg is not None
        }
        if bilateral and state not in self._PIVOT_STATES and len(axes) == 2:
            self._hold_axes = dict(axes)
        if state in self._PIVOT_STATES and self._pivot_reference_axes is None:
            if len(self._hold_axes) == 2:
                self._pivot_reference_axes = dict(self._hold_axes)
                if self.config.gravity_pivot_lever_tcp is not None:
                    moment = _gravity_pivot_moment_about_jaw(
                        T_base_tcp,
                        self.config.gravity_pivot_lever_tcp,
                    )
                    self._pivot_expected_moment = moment
                    if abs(moment) >= self.config.pivot_direction_min_moment:
                        self._pivot_expected_direction_sign = 1 if moment > 0.0 else -1

        deltas: list[float] = []
        signed_deltas: dict[str, float] = {}
        if self._pivot_reference_axes is not None and len(axes) == 2:
            for serial in sorted(axes):
                if serial in self._pivot_reference_axes:
                    deltas.append(
                        _undirected_delta_deg(axes[serial], self._pivot_reference_axes[serial])
                    )
                    signed_deltas[serial] = _standard_pad_signed_axis_delta_deg(
                        reference_deg=self._pivot_reference_axes[serial],
                        observed_deg=axes[serial],
                        side=self._side[serial],
                    )

        significant_signs = [
            1 if value > 0.0 else -1
            for value in signed_deltas.values()
            if abs(value) >= self.config.pivot_direction_min_delta_deg
        ]
        observed_sign: int | None = None
        if len(significant_signs) == 2:
            observed_sign = (
                significant_signs[0]
                if significant_signs[0] == significant_signs[1]
                else 0
            )
        direction_enabled = self.config.gravity_pivot_lever_tcp is not None
        direction_matches = bool(
            not direction_enabled
            or (
                self._pivot_expected_direction_sign is not None
                and observed_sign == self._pivot_expected_direction_sign
            )
        )
        direction_fault_candidate = bool(
            direction_enabled
            and state in {TubeState.PIVOT_MICRO_LOOSEN, TubeState.PIVOT_EVALUATE}
            and (
                self._pivot_expected_direction_sign is None
                or (observed_sign is not None and not direction_matches)
            )
        )
        self._pivot_wrong_direction_windows = (
            self._pivot_wrong_direction_windows + 1
            if direction_fault_candidate
            else 0
        )
        direction_fault = self._pivot_wrong_direction_windows >= 1
        pivot_candidate = bool(
            state in self._PIVOT_STATES
            and bilateral
            and len(deltas) == 2
            and min(deltas)
            >= self.config.pivot_target_delta_deg - self.config.pivot_target_tolerance_deg
            and max(deltas)
            <= self.config.pivot_target_delta_deg + self.config.pivot_target_tolerance_deg
            and max(deltas) - min(deltas) <= self.config.pivot_pad_disagreement_deg
            and direction_matches
        )
        self._pivot_target_windows = self._pivot_target_windows + 1 if pivot_candidate else 0
        if self._pivot_target_windows >= self.config.pivot_confirmation_windows:
            self._pivot_target_latched = True

        step_deltas = [
            _undirected_delta_deg(axes[serial], self._previous_axes[serial])
            for serial in sorted(axes)
            if serial in self._previous_axes
        ]
        stop_candidate = bool(
            state in {TubeState.RECLAMP, TubeState.RECLAMP_RECOVERY}
            and bilateral
            and len(step_deltas) == 2
            and max(step_deltas) <= self.config.rotation_stop_delta_deg
        )
        self._rotation_stop_windows = self._rotation_stop_windows + 1 if stop_candidate else 0
        stopped = (
            self._rotation_stop_windows >= self.config.rotation_stop_confirmation_windows
        )
        if len(axes) == 2:
            self._previous_axes = dict(axes)
        return self._pivot_target_latched, stopped, direction_fault, {
            "pivot_reference_ready": self._pivot_reference_axes is not None,
            "pivot_pad_delta_deg": deltas,
            "pivot_pad_signed_delta_deg": signed_deltas,
            "gravity_pivot_standard_mount_assumption": direction_enabled,
            "gravity_pivot_lever_tcp": self.config.gravity_pivot_lever_tcp,
            "gravity_pivot_expected_moment": self._pivot_expected_moment,
            "gravity_pivot_expected_sign": self._pivot_expected_direction_sign,
            "gravity_pivot_observed_sign": observed_sign,
            "gravity_pivot_direction_matches": direction_matches,
            "gravity_pivot_direction_fault_candidate": direction_fault_candidate,
            "gravity_pivot_direction_fault_windows": self._pivot_wrong_direction_windows,
            "gravity_pivot_direction_fault": direction_fault,
            "pivot_candidate": pivot_candidate,
            "pivot_confirmation_windows": self._pivot_target_windows,
            "rotation_step_delta_deg": step_deltas,
            "rotation_stop_candidate": stop_candidate,
            "rotation_stop_windows": self._rotation_stop_windows,
        }

    def _insertion_events(
        self,
        summary: Mapping[str, _WindowSummary],
        *,
        state: TubeState,
        bilateral: bool,
        slip_detected: bool,
        commanded_step_m: float | None,
        measured_progress_m: float | None,
    ) -> tuple[bool, bool, bool, dict[str, Any]]:
        if state is TubeState.ALIGN_SOCKET and self._last_state is not TubeState.ALIGN_SOCKET:
            self._clear_insertion_reference()
        if state is TubeState.ALIGN_SOCKET and not bilateral:
            self._clear_insertion_reference()
        if state is TubeState.ALIGN_SOCKET and bilateral:
            sample = {serial: item.force_median.copy() for serial, item in summary.items()}
            self._insertion_reference_windows.append(sample)
            self._insertion_reference_windows = self._insertion_reference_windows[
                -self.config.insertion_reference_windows :
            ]
            if (
                len(self._insertion_reference_windows)
                >= self.config.insertion_reference_windows
            ):
                self._insertion_reference = {
                    serial: np.median(
                        np.stack(
                            [item[serial] for item in self._insertion_reference_windows],
                            axis=0,
                        ),
                        axis=0,
                    )
                    for serial in summary
                }
                reference_deviations = [
                    sum(
                        float(
                            np.linalg.norm(
                                item[serial] - self._insertion_reference[serial]
                            )
                        )
                        for serial in summary
                    )
                    for item in self._insertion_reference_windows
                ]
                self._insertion_noise = _robust_upper_noise(
                    np.asarray(reference_deviations, dtype=float)
                )
            self._insert_cycle_counted = False
        if state is TubeState.INSERT_MICROSTEP:
            self._insert_cycle_counted = False
        # Once a held contact patch has slipped or disappeared for the
        # configured consecutive-window debounce, the pre-insertion load is no
        # longer comparable to the current grasp.  Invalidate it and force a
        # bounded realignment/re-bootstrap instead of interpreting the old
        # reference as either a seat or a jam.
        if state in self._INSERT_STATES and slip_detected:
            self._clear_insertion_reference()
            return False, False, False, {
                "insertion_reference_ready": False,
                "insertion_reference_invalidated_by_slip": True,
            }
        if state not in self._INSERT_STATES or self._insertion_reference is None:
            self._jam_windows = 0
            self._seat_windows = 0
            return False, False, False, {
                "insertion_reference_ready": self._insertion_reference is not None,
                "insertion_reference_windows_collected": len(
                    self._insertion_reference_windows
                ),
                "insertion_reference_windows_required": (
                    self.config.insertion_reference_windows
                ),
            }
        if commanded_step_m is None or measured_progress_m is None:
            return False, False, False, {"insertion_reference_ready": True}
        command = float(commanded_step_m)
        progress = float(measured_progress_m)
        if (
            not math.isfinite(command)
            or command <= 0.0
            or not math.isfinite(progress)
            or progress < 0.0
        ):
            return False, False, False, {"insertion_measurement_invalid": True}
        deltas = {
            serial: float(
                np.linalg.norm(item.force_median - self._insertion_reference[serial])
            )
            for serial, item in summary.items()
        }
        total = sum(deltas.values())
        assert self._baselines is not None
        noise = max(
            self._insertion_noise,
            sum(item.force_noise for item in self._baselines.values()),
        )
        threshold = max(
            _strict_positive(noise * self.config.insertion_resistance_noise_multiplier),
            sum(_numeric_resolution(item.force_vector) for item in self._baselines.values()),
        )
        ratio = progress / command
        resistance_high = total >= threshold
        asymmetry = abs(
            deltas[sorted(deltas)[0]] - deltas[sorted(deltas)[1]]
        ) / max(total, _strict_positive(total))
        low_progress = ratio < self.config.insertion_progress_ratio_min
        if (
            state is TubeState.INSERT_EVALUATE
            and not self._insert_cycle_counted
            and ratio >= self.config.insertion_progress_ratio_min
        ):
            self._successful_insert_steps += 1
            self._insert_cycle_counted = True
        seat_candidate = bool(
            state is TubeState.INSERT_EVALUATE
            and bilateral
            and low_progress
            and resistance_high
            and asymmetry <= self.config.insertion_balance_max_fraction
            and self._successful_insert_steps
            >= self.config.minimum_successful_steps_before_seat
        )
        jam_candidate = bool(
            state is TubeState.INSERT_EVALUATE
            and low_progress
            and resistance_high
            and not seat_candidate
        )
        self._seat_windows = self._seat_windows + 1 if seat_candidate else 0
        self._jam_windows = self._jam_windows + 1 if jam_candidate else 0
        seated = self._seat_windows >= self.config.insertion_event_confirmation_windows
        jam = self._jam_windows >= self.config.insertion_event_confirmation_windows
        return resistance_high, jam, seated, {
            "insertion_reference_ready": True,
            "insertion_force_delta_by_serial": deltas,
            "insertion_resistance_signal": total,
            "insertion_resistance_threshold": threshold,
            "insertion_progress_ratio": ratio,
            "insertion_load_asymmetry": asymmetry,
            "successful_insert_steps": self._successful_insert_steps,
            "insertion_seat_candidate": seat_candidate,
            "insertion_jam_candidate": jam_candidate,
            "insertion_seat_windows": self._seat_windows,
            "insertion_jam_windows": self._jam_windows,
        }

    def _clear_insertion_reference(self) -> None:
        self._insertion_reference = None
        self._insertion_reference_windows.clear()
        self._insertion_noise = 0.0
        self._successful_insert_steps = 0
        self._insert_cycle_counted = False
        self._jam_windows = 0
        self._seat_windows = 0

    def _write_audit(
        self,
        *,
        state: TubeState,
        feedback: TactileFeedback,
        blockers: Sequence[str],
        pads: Sequence[_PadObservation],
        signals: Mapping[str, Any],
    ) -> None:
        payload = {
            "schema": SESSION_AUDIT_SCHEMA,
            "sequence": self._sequence,
            "session_id": self._session_id,
            "state": state.value,
            "bootstrap_windows_collected": len(self._baseline_windows),
            "bootstrap_windows_required": self.config.unloaded_windows,
            "session_ready": self.session_ready,
            "feedback": asdict(feedback),
            "blockers": list(blockers),
            "pads": [asdict(item) for item in pads],
            "signals": dict(signals),
            "hardware_commands_dispatched": False,
        }
        _atomic_json(self._directory / f"{self._sequence:06d}.json", payload)
        _atomic_json(self._directory / "latest.json", payload)
        self._sequence += 1


def create_session_bootstrap_tactile_interpreter(
    *,
    config: Mapping[str, Any],
    calibrations: Sequence[GF220Calibration],
    output_dir: Path | str,
) -> SessionBootstrapTestTubeTactileInterpreter:
    """Live-IO factory for receipt-free, session-derived tactile events."""

    if not isinstance(config, Mapping):
        raise TypeError("session tactile config must be an object")
    allowed = {item.name for item in SessionTactileConfig.__dataclass_fields__.values()}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise SessionTactileError(f"unknown session tactile config keys: {unknown}")
    return SessionBootstrapTestTubeTactileInterpreter(
        calibrations=calibrations,
        output_dir=Path(output_dir),
        config=SessionTactileConfig(**dict(config)),
    )


def _fresh_indices(channel: Any, minimum: int, serial: str, label: str) -> np.ndarray:
    values = np.asarray(channel.values)
    valid = np.asarray(channel.valid, dtype=bool)
    indices = np.flatnonzero(valid)
    if indices.size:
        sdk = np.asarray(channel.sdk_time_s[indices], dtype=float)
        _, unique = np.unique(sdk, return_index=True)
        indices = indices[np.sort(unique)]
    if indices.size < minimum:
        raise SessionTactileError(f"{serial}: only {indices.size} unique {label} samples")
    sdk = np.asarray(channel.sdk_time_s[indices], dtype=float)
    host = np.asarray(channel.capture_monotonic_s[indices], dtype=float)
    if np.any(np.diff(sdk) <= 0.0) or np.any(np.diff(host) <= 0.0):
        raise SessionTactileError(f"{serial}: {label} timestamps are not increasing")
    if values.shape[0] != len(valid):
        raise SessionTactileError(f"{serial}: {label} values/timestamps differ")
    return indices


def _robust_upper_noise(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - center)))
    return _strict_positive(center + mad)


def _strict_upper_bound(values: np.ndarray, multiplier: float) -> float:
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - center)))
    return _strict_positive(center + multiplier * max(mad, _numeric_resolution(values)))


def _strict_positive(value: float) -> float:
    value = float(value)
    if math.isfinite(value) and value > 0.0:
        return value
    return float(np.nextafter(0.0, 1.0))


def _numeric_resolution(values: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(np.asarray(values, dtype=float)))))
    return max(np.finfo(float).eps * scale * 64.0, float(np.nextafter(0.0, 1.0)))


def _gravity_pivot_moment_about_jaw(
    T_base_tcp: np.ndarray,
    gravity_pivot_lever_tcp: Sequence[float],
) -> float:
    """Return the signed gravity moment about canonical TCP +X.

    Only the sign is used.  The lever is the configured vector from the grasp
    point toward the held object's centre of mass.  Gravity is rotated from
    ``fr3_link0`` into the current TCP, so the expected direction follows the
    measured wrist pose rather than a hard-coded clockwise/counter-clockwise
    label.
    """

    transform = np.asarray(T_base_tcp, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise SessionTactileError("T_base_tcp must be a finite 4x4 transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise SessionTactileError("T_base_tcp rotation must be right-handed orthonormal")
    lever = np.asarray(gravity_pivot_lever_tcp, dtype=float)
    lever /= np.linalg.norm(lever)
    gravity_tcp = rotation.T @ _GRAVITY_DIRECTION_BASE
    gravity_tcp /= np.linalg.norm(gravity_tcp)
    return float(_JAW_AXIS_TCP @ np.cross(lever, gravity_tcp))


def _standard_pad_signed_axis_delta_deg(
    *,
    reference_deg: float,
    observed_deg: float,
    side: str,
) -> float:
    """Resolve one pad's undirected image line into a signed TCP rotation.

    The line is still invariant to a 180-degree sign flip.  Choosing the
    observed vector with positive dot product to the reference provides the
    unique local continuation for the bounded (<90 degree) gravity pivot.
    """

    try:
        rotation = _STANDARD_R_TCP_PAD_BY_SIDE[side]
    except KeyError as exc:
        raise SessionTactileError(f"unsupported standard pad side {side!r}") from exc

    def axis(angle_deg: float) -> np.ndarray:
        angle = math.radians(float(angle_deg))
        pad = np.asarray([math.cos(angle), math.sin(angle), 0.0], dtype=float)
        value = rotation @ pad
        return value / np.linalg.norm(value)

    reference = axis(reference_deg)
    observed = axis(observed_deg)
    if float(reference @ observed) < 0.0:
        observed = -observed
    sine = float(_JAW_AXIS_TCP @ np.cross(reference, observed))
    cosine = float(np.clip(reference @ observed, -1.0, 1.0))
    return math.degrees(math.atan2(sine, cosine))


def _undirected_delta_deg(left: float, right: float) -> float:
    delta = abs((float(left) - float(right)) % 180.0)
    return min(delta, 180.0 - delta)


def _undirected_mean_deg(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("undirected mean needs values")
    doubled = np.deg2rad(2.0 * np.asarray(values, dtype=float))
    vector = np.asarray([np.cos(doubled).sum(), np.sin(doubled).sum()])
    if np.linalg.norm(vector) <= 1e-12:
        return float(np.median(values) % 180.0)
    return float((0.5 * math.degrees(math.atan2(vector[1], vector[0]))) % 180.0)


def _observed_at(windows: GF220ClassifierWindows, fallback: float) -> float:
    endpoints: list[float] = []
    for source in windows.windows:
        for channel in (source.force6d, source.warped):
            if channel is None:
                continue
            valid = np.flatnonzero(channel.valid)
            if valid.size:
                endpoints.append(float(channel.capture_monotonic_s[valid[-1]]))
    return min(endpoints) if endpoints else float(fallback)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "SESSION_AUDIT_SCHEMA",
    "SESSION_BOOTSTRAP_SCHEMA",
    "SessionBootstrapTestTubeTactileInterpreter",
    "SessionTactileConfig",
    "SessionTactileError",
    "create_session_bootstrap_tactile_interpreter",
]
