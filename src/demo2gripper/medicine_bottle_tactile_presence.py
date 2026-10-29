"""Session-relative tactile supervisor for the medicine-bottle task.

The GF220 evidence available on the real setup changes only the SDK's sensor
``Fz`` channel.  That signal cannot identify a 3-D sensor-to-task rotation, a
bilateral load split, or a twisting moment.  A known axial press is therefore
used to learn the sign of each pad's session-local ``Fz`` response.  The sum of
the positive press-aligned responses is a *vertical-shear proxy* in raw sensor
units; it is not claimed to be a calibrated force axis.

Motion remains deterministic: axial motion is bounded by the runner and twist
is executed in fixed geometric microsteps.  This evaluator can permit a press,
stop after tactile contact disappears, request the configured wrist regrasp,
and request release only after that proxy drops from a measured peak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .medicine_bottle_twist_loop import (
    TactileForceObservation,
    TwistLoopContext,
    TwistLoopPhase,
)
from .tactile_twist_supervisor import (
    TactileTwistAction,
    TactileTwistEvaluation,
    TactileTwistMemory,
    TactileTwistResult,
    TactileTwistState,
)


@dataclass(frozen=True)
class MedicineBottlePresenceConfig:
    """Session-derived scalar thresholds, in raw GF220 FORCE6D units."""

    profile_id: str
    min_unique_samples: int
    max_age_s: float
    max_pair_skew_s: float
    min_window_span_s: float
    contact_retain_fz: float
    press_engage_fz: float
    press_retain_fz: float
    unload_return_fz: float
    press_required_consecutive_windows: int = 1
    wrist_reset_fraction: float = 0.9
    release_min_total_twist_rad: float = 2.0 * math.pi
    release_resistance_peak_fraction: float = 0.35
    release_min_peak_fz: float = 0.0
    release_required_consecutive_windows: int = 2

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        if self.min_unique_samples < 2:
            raise ValueError("min_unique_samples must be at least two")
        if self.press_required_consecutive_windows < 1:
            raise ValueError("press_required_consecutive_windows must be positive")
        if self.release_required_consecutive_windows < 1:
            raise ValueError("release_required_consecutive_windows must be positive")
        for name in (
            "max_age_s",
            "max_pair_skew_s",
            "min_window_span_s",
            "contact_retain_fz",
            "press_engage_fz",
            "press_retain_fz",
            "unload_return_fz",
            "release_min_total_twist_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.min_window_span_s > self.max_age_s:
            raise ValueError("min_window_span_s cannot exceed max_age_s")
        if self.press_retain_fz > self.press_engage_fz:
            raise ValueError("press_retain_fz cannot exceed press_engage_fz")
        if (
            not math.isfinite(self.release_resistance_peak_fraction)
            or not 0.0 < self.release_resistance_peak_fraction < 1.0
        ):
            raise ValueError("release_resistance_peak_fraction must lie in (0, 1)")
        if not math.isfinite(self.release_min_peak_fz) or self.release_min_peak_fz < 0.0:
            raise ValueError("release_min_peak_fz must be finite and non-negative")
        if not 0.0 < self.wrist_reset_fraction <= 1.0:
            raise ValueError("wrist_reset_fraction must lie in (0, 1]")


def release_resistance_limit_sensor_units(
    config: MedicineBottlePresenceConfig,
    peak_resistance_sensor_units: float,
) -> float:
    """Return the shared raw-Fz release limit for twist and stationary verify."""

    peak = float(peak_resistance_sensor_units)
    if not math.isfinite(peak) or peak < 0.0:
        raise ValueError("peak_resistance_sensor_units must be finite and non-negative")
    return max(
        config.press_retain_fz,
        config.release_resistance_peak_fraction * peak,
    )


@dataclass(frozen=True)
class _PresenceBaseline:
    task_frame_id: str
    calibration_ids: tuple[str, str]
    left_fz: float
    right_fz: float
    evidence_key: tuple[float, ...]


def _window_fz(
    observation: TactileForceObservation,
    *,
    config: MedicineBottlePresenceConfig,
) -> tuple[float, float, tuple[float, ...], tuple[str, ...]]:
    reasons: list[str] = []
    windows = (observation.left, observation.right)
    if windows[0].task_frame_id != windows[1].task_frame_id:
        reasons.append("left and right tactile windows use different frames")
    medians: list[float] = []
    evidence: list[float] = []
    starts: list[float] = []
    ends: list[float] = []
    for side, window in zip(("left", "right"), windows, strict=True):
        valid = np.flatnonzero(window.valid)
        if valid.size:
            finite = np.isfinite(window.wrench_task[valid, 2])
            finite &= np.isfinite(window.sdk_time_s[valid])
            finite &= np.isfinite(window.capture_monotonic_s[valid])
            valid = valid[finite]
        if valid.size:
            _, positions = np.unique(window.sdk_time_s[valid], return_index=True)
            valid = valid[np.sort(positions)]
            capture = window.capture_monotonic_s[valid]
            fresh = (capture >= observation.now_monotonic_s - config.max_age_s) & (
                capture <= observation.now_monotonic_s + 1e-6
            )
            valid = valid[fresh]
        if valid.size < config.min_unique_samples:
            reasons.append(
                f"{side} has {valid.size} fresh unique samples; "
                f"requires {config.min_unique_samples}"
            )
            medians.append(math.nan)
            evidence.extend((math.nan, math.nan))
            continue
        capture = window.capture_monotonic_s[valid]
        if np.any(np.diff(capture) < 0.0):
            reasons.append(f"{side} host timestamps moved backwards")
        if float(capture[-1] - capture[0]) < config.min_window_span_s:
            reasons.append(f"{side} samples do not span the required window")
        medians.append(float(np.median(window.wrench_task[valid, 2])))
        starts.append(float(capture[0]))
        ends.append(float(capture[-1]))
        evidence.extend(
            (
                float(window.sdk_time_s[valid][-1]),
                float(capture[-1]),
            )
        )
    if len(ends) == 2 and abs(ends[0] - ends[1]) > config.max_pair_skew_s:
        reasons.append("left/right tactile endpoints exceed pair skew")
    return medians[0], medians[1], tuple(evidence), tuple(reasons)


def _evaluation(
    *,
    state: TactileTwistState,
    action: TactileTwistAction,
    valid: bool,
    presence: float | None,
    press_delta: float | None,
    signed_proxy: float | None,
    resistance_proxy: float | None,
    wrist_fraction: float,
    predicates: set[str],
    reasons: tuple[str, ...],
    evidence_key: tuple[float, ...] | None,
) -> TactileTwistEvaluation:
    return TactileTwistEvaluation(
        state=state,
        action=action,
        valid=valid,
        left=None,
        right=None,
        axial_press_reaction=presence,
        axial_press_delta=press_delta,
        axial_press_z=None,
        axial_press_imbalance=None,
        axial_press_range_z=None,
        signed_tangential_drive=signed_proxy,
        drive_delta_from_baseline=resistance_proxy,
        drive_z=None,
        force_axis_moment=None,
        force_axis_moment_delta=None,
        force_axis_moment_z=None,
        wrench_axis_torque=None,
        load_imbalance=None,
        wrist_travel_fraction=wrist_fraction,
        predicates=frozenset(predicates),
        reasons=reasons,
        evidence_key=evidence_key,
    )


class TactilePresenceSupervisorEvaluator:
    """Evaluate contact and thread resistance from a press-aligned Fz proxy."""

    def __init__(
        self,
        *,
        config: MedicineBottlePresenceConfig,
        wrist_travel_limit_rad: float,
    ) -> None:
        if not isinstance(config, MedicineBottlePresenceConfig):
            raise TypeError("config must be MedicineBottlePresenceConfig")
        limit = float(wrist_travel_limit_rad)
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("wrist_travel_limit_rad must be positive and finite")
        self._config = config
        self._wrist_limit = limit
        self._baseline: _PresenceBaseline | None = None
        self._memory = TactileTwistMemory()
        self._press_signs = (1.0, 1.0)
        self._global_peak_resistance = 0.0
        self._release_low_windows = 0

    @property
    def memory(self) -> TactileTwistMemory:
        return self._memory

    def rebaseline(
        self, observation: TactileForceObservation, *, context: TwistLoopContext
    ) -> None:
        if context.phase is not TwistLoopPhase.PRE_PRESS_BASELINE:
            raise ValueError("presence baseline is allowed only before axial press")
        left, right, evidence, reasons = _window_fz(
            observation, config=self._config
        )
        if reasons:
            raise ValueError("; ".join(reasons))
        self._baseline = _PresenceBaseline(
            task_frame_id=observation.left.task_frame_id,
            calibration_ids=(
                observation.left.calibration_id,
                observation.right.calibration_id,
            ),
            left_fz=left,
            right_fz=right,
            evidence_key=evidence,
        )
        self._memory = TactileTwistMemory()

    def evaluate(
        self, observation: TactileForceObservation, *, context: TwistLoopContext
    ) -> TactileTwistResult:
        if self._baseline is None:
            raise RuntimeError("rebaseline() must run before tactile evaluation")
        wrist_fraction = context.segment_wrist_travel_rad / self._wrist_limit
        left, right, evidence, reasons = _window_fz(
            observation, config=self._config
        )
        baseline = self._baseline
        if observation.left.task_frame_id != baseline.task_frame_id or (
            observation.left.calibration_id,
            observation.right.calibration_id,
        ) != baseline.calibration_ids:
            reasons = (*reasons, "tactile provenance changed after baseline")
        if evidence == baseline.evidence_key or evidence == self._memory.last_evidence_key:
            reasons = (*reasons, "tactile evidence window was replayed")
        if reasons:
            result = TactileTwistResult(
                evaluation=_evaluation(
                    state=TactileTwistState.INVALID,
                    action=TactileTwistAction.HOLD_INVALID,
                    valid=False,
                    presence=None,
                    press_delta=None,
                    signed_proxy=None,
                    resistance_proxy=None,
                    wrist_fraction=wrist_fraction,
                    predicates=set(),
                    reasons=tuple(reasons),
                    evidence_key=evidence,
                ),
                memory=self._memory,
            )
            return result

        # The task-session adapter subtracts the unloaded median and preserves
        # raw SDK Fz as task Z.  Presence is intentionally an OR across pads.
        presence = max(abs(left), abs(right))
        press_delta = max(
            abs(left - baseline.left_fz), abs(right - baseline.right_fz)
        )
        left_delta = left - baseline.left_fz
        right_delta = right - baseline.right_fz
        aligned_left = self._press_signs[0] * left_delta
        aligned_right = self._press_signs[1] * right_delta
        signed_proxy = aligned_left + aligned_right
        resistance = abs(left_delta) + abs(right_delta)
        predicates: set[str] = set()
        state = TactileTwistState.NO_ENGAGEMENT
        action = TactileTwistAction.HOLD_FOR_ENGAGEMENT
        reason = "tactile presence has not reached the session gate"
        stable_windows = self._memory.press_stable_windows
        press_confirmed = self._memory.press_confirmed
        release_requests = self._release_low_windows

        if context.phase is TwistLoopPhase.PRESS_VERIFY:
            if press_delta >= self._config.press_engage_fz:
                old_left_sign, old_right_sign = self._press_signs
                self._press_signs = (
                    math.copysign(1.0, left_delta)
                    if abs(left_delta) >= self._config.press_retain_fz
                    else old_left_sign,
                    math.copysign(1.0, right_delta)
                    if abs(right_delta) >= self._config.press_retain_fz
                    else old_right_sign,
                )
                aligned_left = self._press_signs[0] * left_delta
                aligned_right = self._press_signs[1] * right_delta
                signed_proxy = aligned_left + aligned_right
                resistance = abs(left_delta) + abs(right_delta)
                stable_windows += 1
                predicates.add("any_pad_press_response_present")
                if stable_windows >= self._config.press_required_consecutive_windows:
                    state = TactileTwistState.PRESS_ENGAGED
                    action = TactileTwistAction.START_TWIST
                    press_confirmed = True
                    reason = "at least one pad changed after the bounded downpress"
                else:
                    state = TactileTwistState.PRESS_NOT_ENGAGED
                    action = TactileTwistAction.HOLD_FOR_ENGAGEMENT
                    reason = "press response seen once; waiting for confirmation"
            else:
                state = TactileTwistState.PRESS_NOT_ENGAGED
                action = TactileTwistAction.PRESS_MORE
                stable_windows = 0
                reason = "no pad changed after the bounded downpress"
        elif context.phase is TwistLoopPhase.UNLOAD_CHECK:
            # The Hand-E open command and fresh sensor timestamps establish
            # this transition.  Gel recovery can lag the finger motion, so a
            # residual Fz value must not block the planned wrist return.
            state = TactileTwistState.NO_ENGAGEMENT
            action = TactileTwistAction.UNLOAD_CONFIRMED
            predicates.add("fresh_window_after_successful_gripper_open")
            reason = "gripper opened and a fresh tactile window was received"
        elif not press_confirmed:
            state = TactileTwistState.PRESS_NOT_ENGAGED
            action = TactileTwistAction.PRESS_MORE
            reason = "fixed twist is blocked until downpress changes a pad"
        elif presence < self._config.contact_retain_fz:
            state = TactileTwistState.SLIP_OR_IMBALANCED
            action = TactileTwistAction.SLIP_RECOVERY
            reason = "both pads lost tactile presence during fixed-step twist"
        else:
            state = TactileTwistState.TWIST_ENGAGED
            predicates.add("any_pad_tactile_presence_retained")
            predicates.add("press_aligned_fz_proxy_available")
            in_upfeed_phase = (
                abs(context.total_signed_twist_rad)
                > self._config.release_min_total_twist_rad + 1e-12
            )
            if in_upfeed_phase:
                self._global_peak_resistance = max(
                    self._global_peak_resistance, resistance
                )
            release_eligible = (
                in_upfeed_phase
                and self._global_peak_resistance >= self._config.release_min_peak_fz
            )
            release_threshold = release_resistance_limit_sensor_units(
                self._config,
                self._global_peak_resistance,
            )
            if release_eligible and resistance <= release_threshold:
                release_requests += 1
                predicates.add("vertical_shear_proxy_below_peak_fraction")
            else:
                release_requests = 0
            self._release_low_windows = release_requests
            if (
                release_eligible
                and release_requests
                >= self._config.release_required_consecutive_windows
            ):
                state = TactileTwistState.RELEASE_CANDIDATE
                action = TactileTwistAction.RELEASE_CANDIDATE
                predicates.add("thread_release_proxy_drop_confirmed")
                reason = (
                    "press-aligned Fz resistance proxy stayed below its peak "
                    "fraction for the required fresh windows"
                )
            elif release_eligible and release_requests > 0:
                action = TactileTwistAction.HOLD_FOR_ENGAGEMENT
                reason = "possible thread release; waiting for another fresh window"
            elif wrist_fraction >= self._config.wrist_reset_fraction:
                action = TactileTwistAction.WRIST_RESET_REGRASP
                reason = "wrist boundary reached while tactile presence remains"
            else:
                action = TactileTwistAction.TWIST_CONTINUE
                reason = "tactile presence remains during fixed 5-degree twist"

        memory = TactileTwistMemory(
            previous_state=state,
            peak_abs_drive_delta=self._global_peak_resistance,
            peak_abs_force_axis_moment_delta=0.0,
            peak_abs_left_delta=max(self._memory.peak_abs_left_delta, abs(left)),
            peak_abs_right_delta=max(self._memory.peak_abs_right_delta, abs(right)),
            engaged_windows=(
                self._memory.engaged_windows + 1
                if state is TactileTwistState.TWIST_ENGAGED
                else self._memory.engaged_windows
            ),
            release_low_windows=release_requests,
            press_stable_windows=stable_windows,
            press_confirmed=press_confirmed,
            last_evidence_key=evidence,
        )
        self._memory = memory
        return TactileTwistResult(
            evaluation=_evaluation(
                state=state,
                action=action,
                valid=True,
                presence=presence,
                press_delta=press_delta,
                signed_proxy=signed_proxy,
                resistance_proxy=resistance,
                wrist_fraction=wrist_fraction,
                predicates=predicates,
                reasons=(reason,),
                evidence_key=evidence,
            ),
            memory=memory,
        )


__all__ = [
    "MedicineBottlePresenceConfig",
    "TactilePresenceSupervisorEvaluator",
    "release_resistance_limit_sensor_units",
]
