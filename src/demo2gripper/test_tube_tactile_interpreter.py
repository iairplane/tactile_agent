"""Concrete, fail-closed GF220 interpreter for the live test-tube policy.

The live adapter injects this module through
``demo2gripper.test_tube_tactile_interpreter:create_test_tube_tactile_interpreter``.
It owns no device handles and emits no robot commands.  It maps the current
``TubeState`` to the pure tactile controller mode, preserves that controller's
session memory, and writes one audit receipt for every evaluation.

Nothing in this module invents force units or thresholds.  Contact requires a
session-specific positive-contact profile.  Pivot/reclamp requires an accepted
pad-axis qualification.  Insertion requires both an accepted wrench transform
and an accepted jam profile.  A missing or mismatched qualification returns a
``TactileFeedback(valid=False)`` so the closed-loop policy holds.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .gf220_live import GF220Calibration, GF220ClassifierWindows
from .gf220_wrench_adapter import (
    GF220WrenchAdapterConfig,
    GF220WrenchSensorCalibration,
    adapt_gf220_force6d_windows,
)
from .test_tube_closed_loop import GripperFeedback, TactileFeedback
from .test_tube_contact_bootstrap import (
    TestTubeContactBootstrapConfig,
    TestTubeContactGate,
    TestTubeContactProfile,
    TestTubeFingerContactCalibration,
)
from .test_tube_preflight import TubeState
from .test_tube_tactile_controller import (
    TestTubeFingerImageProfile,
    TestTubeInsertionJamProfile,
    TestTubeTactileControllerProfile,
    TestTubeTactileEvaluation,
    TestTubeTactileMemory,
    TestTubeTactileMode,
    evaluate_test_tube_tactile,
)

QUALIFICATION_SCHEMA = "demo2gripper.test_tube_tactile_interpreter_qualification.v1"
AUDIT_SCHEMA = "demo2gripper.test_tube_tactile_interpreter_audit.v1"


_MODE_BY_STATE: dict[TubeState, TestTubeTactileMode] = {
    TubeState.OBSERVE: TestTubeTactileMode.GRASP,
    TubeState.APPROACH_TUBE: TestTubeTactileMode.GRASP,
    TubeState.SIDE_GRASP: TestTubeTactileMode.GRASP,
    TubeState.EXTRACT: TestTubeTactileMode.HOLD,
    TubeState.LEVEL_HELD_TUBE: TestTubeTactileMode.HOLD,
    TubeState.PIVOT_MICRO_LOOSEN: TestTubeTactileMode.PIVOT,
    TubeState.PIVOT_EVALUATE: TestTubeTactileMode.PIVOT,
    TubeState.RECLAMP: TestTubeTactileMode.RECLAMP,
    TubeState.RECLAMP_RECOVERY: TestTubeTactileMode.RECLAMP,
    TubeState.APPROACH_SOCKET: TestTubeTactileMode.HOLD,
    TubeState.ALIGN_SOCKET: TestTubeTactileMode.HOLD,
    TubeState.INSERT_MICROSTEP: TestTubeTactileMode.INSERT,
    TubeState.INSERT_EVALUATE: TestTubeTactileMode.INSERT,
    TubeState.JAM_RECOVERY_RETRACT: TestTubeTactileMode.INSERT,
    TubeState.RELEASE: TestTubeTactileMode.RELEASE,
    TubeState.RETREAT: TestTubeTactileMode.RELEASE,
    TubeState.COMPLETE: TestTubeTactileMode.RELEASE,
    TubeState.HOLD: TestTubeTactileMode.HOLD,
}


class TestTubeTactileInterpreterError(ValueError):
    """The supplied qualification artifact violates the interpreter contract."""


def mode_for_tube_state(state: TubeState | str) -> TestTubeTactileMode:
    """Return the one tactile semantic mode owned by a policy state."""

    return _MODE_BY_STATE[TubeState(state)]


class FailClosedTestTubeTactileInterpreter:
    """Interpreter returned when execution qualifications are unavailable."""

    def __init__(self, *, blockers: Sequence[str], output_dir: Path) -> None:
        normalized = tuple(sorted({str(item) for item in blockers if str(item)}))
        self.blockers = normalized or ("tactile_interpreter_unqualified",)
        self._audit = _AuditWriter(Path(output_dir), qualification_id=None)

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
        del T_base_tcp, gripper, commanded_insertion_step_m, measured_insertion_progress_m
        mode = mode_for_tube_state(state)
        observed_at_s = _observed_at(windows, fallback=now_s)
        self._audit.write(
            state=TubeState(state),
            mode=mode,
            observed_at_s=observed_at_s,
            feedback=TactileFeedback(valid=False, observed_at_s=observed_at_s),
            evaluation=None,
            blockers=self.blockers,
            rotation_delta_deg=None,
            rotation_stable_windows=0,
        )
        return TactileFeedback(valid=False, observed_at_s=observed_at_s)

    def close(self) -> None:
        return None


class QualifiedTestTubeTactileInterpreter:
    """Stateful bridge from synchronized GF220 windows to closed-loop feedback."""

    def __init__(
        self,
        *,
        profile: TestTubeTactileControllerProfile,
        serial_side_mapping: Mapping[str, str],
        target_axis_base: np.ndarray,
        rotation_stop_runtime_qualified: bool,
        rotation_stop_threshold_deg: float,
        rotation_stop_confirmation_windows: int,
        output_dir: Path,
        qualification_id: str,
        wrench_adapter: GF220WrenchAdapterConfig | None = None,
    ) -> None:
        if set(serial_side_mapping) != {item.serial for item in profile.fingers}:
            raise TestTubeTactileInterpreterError(
                "serial_side_mapping must name the exact two controller serials"
            )
        if set(serial_side_mapping.values()) != {"left", "right"}:
            raise TestTubeTactileInterpreterError(
                "serial_side_mapping must contain exactly one left and one right"
            )
        axis = _unit_vector(target_axis_base, "target_axis_base")
        threshold = _positive(rotation_stop_threshold_deg, "rotation_stop_threshold_deg")
        if (
            isinstance(rotation_stop_confirmation_windows, bool)
            or not isinstance(rotation_stop_confirmation_windows, int)
            or rotation_stop_confirmation_windows < 1
        ):
            raise TestTubeTactileInterpreterError(
                "rotation_stop_confirmation_windows must be a positive integer"
            )
        if not isinstance(rotation_stop_runtime_qualified, bool):
            raise TestTubeTactileInterpreterError(
                "rotation_stop_runtime_qualified must be boolean"
            )
        self.profile = profile
        self._mapping = dict(serial_side_mapping)
        self._target_axis_base = axis
        self._rotation_stop_runtime_qualified = rotation_stop_runtime_qualified
        self._rotation_stop_threshold_deg = threshold
        self._rotation_stop_confirmation_windows = rotation_stop_confirmation_windows
        self._wrench_adapter = wrench_adapter
        self._contact_gate = TestTubeContactGate(profile.contact_profile)
        self._memory = TestTubeTactileMemory()
        self._rotation_stable_windows = 0
        self._audit = _AuditWriter(Path(output_dir), qualification_id=qualification_id)

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
        mode = mode_for_tube_state(state)
        blockers = self._qualification_blockers(mode)
        observed_at_s = _observed_at(windows, fallback=now_s)
        if blockers:
            feedback = TactileFeedback(valid=False, observed_at_s=observed_at_s)
            self._rotation_stable_windows = 0
            self._audit.write(
                state=state,
                mode=mode,
                observed_at_s=observed_at_s,
                feedback=feedback,
                evaluation=None,
                blockers=blockers,
                rotation_delta_deg=None,
                rotation_stable_windows=0,
            )
            return feedback

        contact = self._contact_gate.update(
            windows,
            session_id=self.profile.session_id,
            now_monotonic_s=now_s,
        )
        adapted = None
        adapter_error: str | None = None
        if mode is TestTubeTactileMode.INSERT and self._wrench_adapter is not None:
            try:
                adapted = adapt_gf220_force6d_windows(
                    windows, self._wrench_adapter, now_s=now_s
                )
            except (TypeError, ValueError) as error:
                adapter_error = f"wrench_adapter_invalid: {error}"

        if adapter_error is not None:
            feedback = TactileFeedback(valid=False, observed_at_s=observed_at_s)
            self._audit.write(
                state=state,
                mode=mode,
                observed_at_s=observed_at_s,
                feedback=feedback,
                evaluation=None,
                blockers=(adapter_error,),
                rotation_delta_deg=None,
                rotation_stable_windows=0,
            )
            return feedback

        previous_axis = self._memory.last_axis_base
        evaluation = evaluate_test_tube_tactile(
            mode=mode,
            windows=windows,
            contact=contact,
            profile=self.profile,
            memory=self._memory,
            now_s=now_s,
            T_base_tcp=T_base_tcp,
            target_axis_base=self._target_axis_base,
            gripper_width_m=gripper.width_m,
            gripper_open_confirmed=gripper.open_confirmed,
            adapted_wrenches=adapted,
            commanded_axial_step_m=commanded_insertion_step_m,
            measured_axial_progress_m=measured_insertion_progress_m,
        )
        if evaluation.valid:
            self._memory = evaluation.next_memory

        rotation_delta = _axis_delta_deg(previous_axis, evaluation.primitives.axis_base)
        rotation_stopped = self._update_rotation_stopped(
            mode=mode,
            evaluation=evaluation,
            rotation_delta_deg=rotation_delta,
        )
        promoted = set(evaluation.promoted_predicates)
        primitives = evaluation.primitives
        single_side = None
        if evaluation.valid and len(primitives.active_serials) == 1:
            single_side = self._mapping[primitives.active_serials[0]]
        resistance_high = bool(
            evaluation.valid
            and self.profile.jam is not None
            and self.profile.jam.runtime_qualified
            and primitives.insertion_resistance_delta_signal is not None
            and primitives.insertion_resistance_threshold_signal is not None
            and primitives.insertion_resistance_delta_signal
            >= primitives.insertion_resistance_threshold_signal
        )
        feedback = TactileFeedback(
            valid=evaluation.valid,
            observed_at_s=observed_at_s,
            contact_present=evaluation.valid and primitives.contact_present,
            bilateral_contact=evaluation.valid and primitives.bilateral_contact,
            single_contact_side=single_side,
            slip_detected=evaluation.valid and "slip_detected" in promoted,
            relative_angle_target_reached=(
                evaluation.valid and "tactile_relative_angle_reached" in promoted
            ),
            rotation_stopped=evaluation.valid and rotation_stopped,
            insertion_jam=evaluation.valid and "insertion_jam" in promoted,
            insertion_resistance_high=resistance_high,
        )
        blockers = () if evaluation.valid else (evaluation.reason,)
        self._audit.write(
            state=state,
            mode=mode,
            observed_at_s=observed_at_s,
            feedback=feedback,
            evaluation=evaluation,
            blockers=blockers,
            rotation_delta_deg=rotation_delta,
            rotation_stable_windows=self._rotation_stable_windows,
        )
        return feedback

    def close(self) -> None:
        return None

    def _qualification_blockers(self, mode: TestTubeTactileMode) -> tuple[str, ...]:
        blockers: list[str] = []
        if not all(
            item.contact_response_qualified
            for item in self.profile.contact_profile.fingers
        ):
            blockers.append("positive_contact_response_not_qualified_for_both_pads")
        if mode in {TestTubeTactileMode.PIVOT, TestTubeTactileMode.RECLAMP}:
            if not self.profile.axis_runtime_qualified:
                blockers.append("tactile_axis_not_runtime_qualified")
            if not self._rotation_stop_runtime_qualified:
                blockers.append("rotation_stop_not_runtime_qualified")
        if mode is TestTubeTactileMode.INSERT:
            if self.profile.jam is None or not self.profile.jam.runtime_qualified:
                blockers.append("insertion_jam_not_runtime_qualified")
            if self._wrench_adapter is None:
                blockers.append("insertion_wrench_adapter_missing")
        return tuple(blockers)

    def _update_rotation_stopped(
        self,
        *,
        mode: TestTubeTactileMode,
        evaluation: TestTubeTactileEvaluation,
        rotation_delta_deg: float | None,
    ) -> bool:
        if (
            mode is not TestTubeTactileMode.RECLAMP
            or not evaluation.valid
            or not evaluation.primitives.bilateral_contact
            or not evaluation.primitives.axis_valid
            or rotation_delta_deg is None
            or rotation_delta_deg > self._rotation_stop_threshold_deg
            or evaluation.primitives.slip_candidate
        ):
            self._rotation_stable_windows = 0
            return False
        self._rotation_stable_windows += 1
        return (
            self._rotation_stop_runtime_qualified
            and self._rotation_stable_windows
            >= self._rotation_stop_confirmation_windows
        )


def create_test_tube_tactile_interpreter(
    *,
    config: Mapping[str, Any],
    calibrations: Sequence[GF220Calibration],
    output_dir: Path | str,
) -> QualifiedTestTubeTactileInterpreter | FailClosedTestTubeTactileInterpreter:
    """Factory used directly by :func:`test_tube_live_io.create_live_io`.

    A missing, draft, rejected, or calibration-mismatched bundle creates a
    fail-closed interpreter.  This lets static preflight and CPU tests inspect
    the exact blocker without substituting guessed thresholds.
    """

    destination = Path(output_dir)
    try:
        calibration_by_serial = _validate_calibrations(calibrations)
        bundle_value = config.get("qualification_bundle")
        if not isinstance(bundle_value, (str, os.PathLike)) or not str(bundle_value):
            raise TestTubeTactileInterpreterError("qualification_bundle_missing")
        bundle_path = Path(bundle_value).expanduser().resolve()
        payload = _read_json(bundle_path)
        if payload.get("schema") != QUALIFICATION_SCHEMA:
            raise TestTubeTactileInterpreterError("qualification_bundle_schema_invalid")
        if payload.get("status") != "accepted" or payload.get("runtime_qualified") is not True:
            raise TestTubeTactileInterpreterError(
                "qualification_bundle_not_accepted_for_runtime"
            )
        return _build_qualified_interpreter(
            payload=payload,
            bundle_path=bundle_path,
            calibration_by_serial=calibration_by_serial,
            output_dir=destination,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return FailClosedTestTubeTactileInterpreter(
            blockers=(str(error) or error.__class__.__name__,),
            output_dir=destination,
        )


def create_session_bootstrap_tactile_interpreter(
    *,
    config: Mapping[str, Any],
    calibrations: Sequence[GF220Calibration],
    output_dir: Path | str,
) -> Any:
    """Create the receipt-free, session-derived runtime interpreter.

    The lazy import keeps the legacy immutable-qualification path independent
    while exposing one stable factory location to the live adapter/preflight.
    """

    from .test_tube_session_tactile import (
        create_session_bootstrap_tactile_interpreter as create,
    )

    return create(
        config=config,
        calibrations=calibrations,
        output_dir=output_dir,
    )


def _build_qualified_interpreter(
    *,
    payload: Mapping[str, Any],
    bundle_path: Path,
    calibration_by_serial: Mapping[str, GF220Calibration],
    output_dir: Path,
) -> QualifiedTestTubeTactileInterpreter:
    qualification_id = _identifier(payload.get("qualification_id"), "qualification_id")
    session_id = _identifier(payload.get("session_id"), "session_id")
    contact_payload = _mapping(payload.get("contact_profile"), "contact_profile")
    bootstrap = TestTubeContactBootstrapConfig(
        **dict(_mapping(contact_payload.get("config"), "contact_profile.config"))
    )
    raw_contacts = _pair(contact_payload.get("fingers"), "contact_profile.fingers")
    contact_fingers = tuple(
        TestTubeFingerContactCalibration(**dict(_mapping(item, "contact finger")))
        for item in raw_contacts
    )
    contact_profile = TestTubeContactProfile(
        profile_id=_identifier(contact_payload.get("profile_id"), "contact profile_id"),
        session_id=session_id,
        session_evidence_sha256=_sha256_string(
            contact_payload.get("session_evidence_sha256"),
            "contact session_evidence_sha256",
        ),
        fingers=(contact_fingers[0], contact_fingers[1]),
        engage_threshold_sensor_units=_positive(
            contact_payload.get("engage_threshold_sensor_units"),
            "contact engage threshold",
        ),
        retain_threshold_sensor_units=_positive(
            contact_payload.get("retain_threshold_sensor_units"),
            "contact retain threshold",
        ),
        config=bootstrap,
    )
    if (
        contact_profile.retain_threshold_sensor_units
        >= contact_profile.engage_threshold_sensor_units
    ):
        raise TestTubeTactileInterpreterError(
            "contact retain threshold must be smaller than engage threshold"
        )
    _require_current_calibrations(contact_fingers, calibration_by_serial)

    controller_payload = _mapping(payload.get("controller"), "controller")
    raw_images = _pair(controller_payload.get("fingers"), "controller.fingers")
    image_profiles: list[TestTubeFingerImageProfile] = []
    for raw in raw_images:
        item = _mapping(raw, "controller finger")
        serial = _identifier(item.get("serial"), "controller finger serial")
        calibration = calibration_by_serial.get(serial)
        if calibration is None:
            raise TestTubeTactileInterpreterError(
                f"controller finger serial {serial!r} is not live"
            )
        if item.get("calibration_id") != calibration.calibration_id:
            raise TestTubeTactileInterpreterError(
                f"{serial}: controller calibration_id differs from live baseline"
            )
        roi_path = _artifact_path(item.get("roi_mask"), bundle_path.parent, "roi_mask")
        roi = np.load(roi_path, allow_pickle=False)
        image_profiles.append(
            TestTubeFingerImageProfile(
                serial=serial,
                calibration_id=calibration.calibration_id,
                unloaded_warped=calibration.unloaded_warped,
                roi_mask=roi,
                R_tcp_pad=np.asarray(item.get("R_tcp_pad"), dtype=float),
                residual_threshold=item.get("residual_threshold"),
                min_contact_area_px=item.get("min_contact_area_px"),
                min_orientation_elongation=item.get(
                    "min_orientation_elongation", 1.6
                ),
                min_orientation_concentration=item.get(
                    "min_orientation_concentration", 0.35
                ),
                min_contact_frame_fraction=item.get(
                    "min_contact_frame_fraction", 0.60
                ),
            )
        )

    jam_payload = controller_payload.get("jam")
    jam = None
    if jam_payload is not None:
        raw_jam = _mapping(jam_payload, "controller.jam")
        jam = TestTubeInsertionJamProfile(
            profile_id=_identifier(raw_jam.get("profile_id"), "jam profile_id"),
            task_frame_id=_identifier(raw_jam.get("task_frame_id"), "jam task_frame_id"),
            insertion_axis_task=np.asarray(raw_jam.get("insertion_axis_task"), dtype=float),
            reaction_sign=raw_jam.get("reaction_sign"),
            resistance_baseline_signal=raw_jam.get("resistance_baseline_signal"),
            resistance_noise_signal=raw_jam.get("resistance_noise_signal"),
            minimum_resistance_delta_signal=raw_jam.get(
                "minimum_resistance_delta_signal"
            ),
            runtime_qualified=raw_jam.get("runtime_qualified", False),
            resistance_noise_multiplier=raw_jam.get(
                "resistance_noise_multiplier", 8.0
            ),
            minimum_progress_ratio=raw_jam.get("minimum_progress_ratio", 0.25),
            confirmation_windows=raw_jam.get("confirmation_windows", 2),
        )
    profile = TestTubeTactileControllerProfile(
        profile_id=_identifier(controller_payload.get("profile_id"), "controller profile_id"),
        session_id=session_id,
        contact_profile=contact_profile,
        fingers=(image_profiles[0], image_profiles[1]),
        retained_width_min_m=controller_payload.get("retained_width_min_m"),
        retained_width_max_m=controller_payload.get("retained_width_max_m"),
        axis_runtime_qualified=controller_payload.get("axis_runtime_qualified", False),
        slip_runtime_qualified=controller_payload.get("slip_runtime_qualified", False),
        jam=jam,
        min_unique_image_samples=controller_payload.get("min_unique_image_samples", 4),
        max_age_s=controller_payload.get("max_age_s", 0.25),
        max_pair_skew_s=controller_payload.get("max_pair_skew_s", 0.08),
        future_tolerance_s=controller_payload.get("future_tolerance_s", 1e-6),
        bilateral_confirmation_windows=controller_payload.get(
            "bilateral_confirmation_windows", 2
        ),
        target_axis_tolerance_deg=controller_payload.get(
            "target_axis_tolerance_deg", 5.0
        ),
        axis_confirmation_windows=controller_payload.get("axis_confirmation_windows", 4),
        max_pad_axis_disagreement_deg=controller_payload.get(
            "max_pad_axis_disagreement_deg", 10.0
        ),
        slip_center_shift_threshold_px=controller_payload.get(
            "slip_center_shift_threshold_px", 4.0
        ),
        slip_confirmation_windows=controller_payload.get("slip_confirmation_windows", 2),
    )
    mapping_payload = _mapping(payload.get("serial_side_mapping"), "serial_side_mapping")
    serial_side_mapping = {
        _identifier(serial, "serial_side_mapping serial"): _identifier(side, "logical side")
        for serial, side in mapping_payload.items()
    }
    rotation = _mapping(payload.get("rotation_stop"), "rotation_stop")
    wrench_adapter = _parse_wrench_adapter(
        payload.get("wrench_adapter"), calibration_by_serial
    )
    if jam is not None and wrench_adapter is not None:
        if (
            jam.profile_id != wrench_adapter.profile_id
            or jam.task_frame_id != wrench_adapter.task_frame_id
        ):
            raise TestTubeTactileInterpreterError(
                "jam profile and wrench adapter identity differ"
            )
    return QualifiedTestTubeTactileInterpreter(
        profile=profile,
        serial_side_mapping=serial_side_mapping,
        target_axis_base=np.asarray(payload.get("target_axis_base"), dtype=float),
        rotation_stop_runtime_qualified=rotation.get("runtime_qualified", False),
        rotation_stop_threshold_deg=rotation.get("maximum_axis_delta_deg"),
        rotation_stop_confirmation_windows=rotation.get("confirmation_windows"),
        output_dir=output_dir,
        qualification_id=qualification_id,
        wrench_adapter=wrench_adapter,
    )


def _parse_wrench_adapter(
    value: object,
    calibration_by_serial: Mapping[str, GF220Calibration],
) -> GF220WrenchAdapterConfig | None:
    if value is None:
        return None
    payload = _mapping(value, "wrench_adapter")
    raw_sensors = _pair(payload.get("sensors"), "wrench_adapter.sensors")
    sensors: list[GF220WrenchSensorCalibration] = []
    for raw in raw_sensors:
        item = _mapping(raw, "wrench adapter sensor")
        serial = _identifier(item.get("serial"), "wrench sensor serial")
        calibration = calibration_by_serial.get(serial)
        if calibration is None:
            raise TestTubeTactileInterpreterError(
                f"wrench serial {serial!r} is not live"
            )
        if item.get("calibration_id") != calibration.calibration_id:
            raise TestTubeTactileInterpreterError(
                f"{serial}: wrench calibration_id differs from live baseline"
            )
        sensors.append(
            GF220WrenchSensorCalibration(
                serial=serial,
                logical_side=item.get("logical_side"),
                calibration_id=calibration.calibration_id,
                R_task_sensor=np.asarray(item.get("R_task_sensor"), dtype=float),
                contact_position_task_m=np.asarray(
                    item.get("contact_position_task_m"), dtype=float
                ),
                force6d_bias=np.asarray(item.get("force6d_bias"), dtype=float),
                provenance=_mapping(item.get("provenance"), "wrench provenance"),
            )
        )
    return GF220WrenchAdapterConfig(
        profile_id=_identifier(payload.get("profile_id"), "wrench profile_id"),
        task_frame_id=_identifier(payload.get("task_frame_id"), "wrench task_frame_id"),
        sensors=(sensors[0], sensors[1]),
        min_samples=payload.get("min_samples", 3),
        max_age_s=payload.get("max_age_s", 0.20),
        max_pair_skew_s=payload.get("max_pair_skew_s", 0.08),
        future_tolerance_s=payload.get("future_tolerance_s", 1e-6),
    )


class _AuditWriter:
    def __init__(self, output_dir: Path, *, qualification_id: str | None) -> None:
        self.directory = output_dir / "tactile_interpreter"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.qualification_id = qualification_id
        self.sequence = 0

    def write(
        self,
        *,
        state: TubeState,
        mode: TestTubeTactileMode,
        observed_at_s: float,
        feedback: TactileFeedback,
        evaluation: TestTubeTactileEvaluation | None,
        blockers: Sequence[str],
        rotation_delta_deg: float | None,
        rotation_stable_windows: int,
    ) -> None:
        payload: dict[str, Any] = {
            "schema": AUDIT_SCHEMA,
            "sequence": self.sequence,
            "qualification_id": self.qualification_id,
            "state": state.value,
            "mode": mode.value,
            "observed_at_s": observed_at_s,
            "valid": feedback.valid,
            "feedback": asdict(feedback),
            "blockers": list(blockers),
            "rotation_delta_deg": rotation_delta_deg,
            "rotation_stable_windows": rotation_stable_windows,
            "hardware_commands_dispatched": False,
        }
        if evaluation is not None:
            primitives = evaluation.primitives
            payload["controller"] = {
                "evidence_id": evaluation.evidence_id,
                "valid": evaluation.valid,
                "action": evaluation.action.value,
                "reason": evaluation.reason,
                "promoted_predicates": list(evaluation.promoted_predicates),
                "shadow_predicates": list(evaluation.shadow_predicates),
                "contact_present": primitives.contact_present,
                "bilateral_contact": primitives.bilateral_contact,
                "active_serials": list(primitives.active_serials),
                "axis_valid": primitives.axis_valid,
                "axis_base": primitives.axis_base,
                "target_axis_error_deg": primitives.target_axis_error_deg,
                "slip_candidate": primitives.slip_candidate,
                "insertion_progress_ratio": primitives.insertion_progress_ratio,
                "insertion_resistance_signal": primitives.insertion_resistance_signal,
                "insertion_resistance_delta_signal": (
                    primitives.insertion_resistance_delta_signal
                ),
                "insertion_resistance_threshold_signal": (
                    primitives.insertion_resistance_threshold_signal
                ),
                "insertion_jam_candidate": primitives.insertion_jam_candidate,
                "fingers": [asdict(item) for item in primitives.fingers],
            }
        name = f"{self.sequence:06d}.json"
        _atomic_json(self.directory / name, payload)
        _atomic_json(self.directory / "latest.json", payload)
        self.sequence += 1


def _observed_at(windows: GF220ClassifierWindows, *, fallback: float) -> float:
    endpoints: list[float] = []
    for window in windows.windows:
        for channel in (window.force6d, window.warped, window.diff):
            if channel is None:
                continue
            valid = np.flatnonzero(channel.valid)
            if valid.size:
                endpoints.append(float(channel.capture_monotonic_s[valid[-1]]))
                break
    return min(endpoints) if endpoints else float(fallback)


def _axis_delta_deg(
    previous: tuple[float, float, float] | None,
    current: tuple[float, float, float] | None,
) -> float | None:
    if previous is None or current is None:
        return None
    left = _unit_vector(previous, "previous tactile axis")
    right = _unit_vector(current, "current tactile axis")
    return math.degrees(math.acos(float(np.clip(abs(np.dot(left, right)), -1.0, 1.0))))


def _validate_calibrations(
    calibrations: Sequence[GF220Calibration],
) -> dict[str, GF220Calibration]:
    if isinstance(calibrations, (str, bytes)) or len(calibrations) != 2:
        raise TestTubeTactileInterpreterError(
            "calibrations must contain exactly two GF220Calibration values"
        )
    if not all(isinstance(item, GF220Calibration) for item in calibrations):
        raise TestTubeTactileInterpreterError(
            "calibrations must contain GF220Calibration values"
        )
    result = {item.serial: item for item in calibrations}
    if len(result) != 2:
        raise TestTubeTactileInterpreterError("live GF220 serials must be unique")
    return result


def _require_current_calibrations(
    fingers: Sequence[TestTubeFingerContactCalibration],
    calibrations: Mapping[str, GF220Calibration],
) -> None:
    if {item.serial for item in fingers} != set(calibrations):
        raise TestTubeTactileInterpreterError(
            "contact profile serials differ from the live GF220 pair"
        )
    for item in fingers:
        if item.calibration_id != calibrations[item.serial].calibration_id:
            raise TestTubeTactileInterpreterError(
                f"{item.serial}: contact calibration_id differs from live baseline"
            )


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise TestTubeTactileInterpreterError(
            f"qualification_bundle_not_found: {path}"
        )
    return _mapping(json.loads(path.read_text(encoding="utf-8")), "qualification bundle")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TestTubeTactileInterpreterError(f"{name} must be an object")
    return value


def _pair(value: object, name: str) -> tuple[object, object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise TestTubeTactileInterpreterError(f"{name} must contain exactly two entries")
    return value[0], value[1]


def _artifact_path(value: object, base: Path, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise TestTubeTactileInterpreterError(f"{name} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise TestTubeTactileInterpreterError(f"{name} artifact not found: {path}")
    return path


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TestTubeTactileInterpreterError(f"{name} must be a non-empty string")
    return value


def _sha256_string(value: object, name: str) -> str:
    result = _identifier(value, name)
    if len(result) != 64:
        raise TestTubeTactileInterpreterError(f"{name} must be a 64-character digest")
    try:
        bytes.fromhex(result)
    except ValueError as error:
        raise TestTubeTactileInterpreterError(f"{name} must be hexadecimal") from error
    return result


def _positive(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TestTubeTactileInterpreterError(f"{name} must be positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TestTubeTactileInterpreterError(f"{name} must be positive") from error
    if not math.isfinite(result) or result <= 0.0:
        raise TestTubeTactileInterpreterError(f"{name} must be positive")
    return result


def _unit_vector(value: object, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise TestTubeTactileInterpreterError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise TestTubeTactileInterpreterError(f"{name} must be non-zero")
    result = np.array(vector / norm, copy=True)
    result.setflags(write=False)
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "AUDIT_SCHEMA",
    "QUALIFICATION_SCHEMA",
    "FailClosedTestTubeTactileInterpreter",
    "QualifiedTestTubeTactileInterpreter",
    "TestTubeTactileInterpreterError",
    "create_test_tube_tactile_interpreter",
    "mode_for_tube_state",
]
