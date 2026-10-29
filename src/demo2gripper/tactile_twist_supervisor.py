"""Pure tactile stage supervisor for press-and-turn bottle caps.

The module consumes synchronized wrench windows that have already been
transformed into one cap task frame.  It has no SDK, USB, ROS, gripper, or arm
handles and never emits a hardware command.  Its action is a recommendation to
the deterministic task runner.

All force thresholds are expressed relative to a per-segment stationary
baseline and the peak observed during that segment.  Until the GF220 axes and
units are physically qualified, the resulting values are deliberately labelled
as sensor units rather than N or N.m.

The evaluator does not choose a rotation angle.  A runner first records a
``PRE_PRESS_BASELINE`` while the fingers hold the cap without axial downforce,
then applies bounded press increments.  Stable, balanced reaction force along
the bottle axis is the only gate into formal twist.  During formal twist the
axial signal verifies that downforce remains established and tangential signals
only supervise bilateral contact, slip, imbalance, and thread-release evidence.
``WRIST_RESET_REGRASP`` is produced solely at the configured wrist-travel
boundary while both axial press and tangential contact remain stable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class TactileTwistState(StrEnum):
    INVALID = "invalid"
    PRESS_NOT_ENGAGED = "press_not_engaged"
    PRESS_ENGAGED = "press_engaged"
    NO_ENGAGEMENT = "no_engagement"
    TWIST_ENGAGED = "twist_engaged"
    SLIP_OR_IMBALANCED = "slip_or_imbalanced"
    RELEASE_CANDIDATE = "release_candidate"


class TactileTwistAction(StrEnum):
    HOLD_INVALID = "hold_invalid"
    HOLD_FOR_ENGAGEMENT = "hold_for_engagement"
    PRESS_MORE = "press_more"
    START_TWIST = "start_twist"
    TWIST_CONTINUE = "twist_continue"
    WRIST_RESET_REGRASP = "wrist_reset_regrasp"
    SLIP_RECOVERY = "slip_recovery"
    RELEASE_CANDIDATE = "release_candidate"
    UNLOAD_CONFIRMED = "unload_confirmed"


class TactileEvaluationMode(StrEnum):
    PRESS = "press"
    TWIST = "twist"
    UNLOAD = "unload"


@dataclass(frozen=True)
class FingerWrenchWindow:
    """One finger's task-frame wrench, contact point, and source timestamps.

    ``wrench_task`` stores ``[fx, fy, fz, tx, ty, tz]``.  The force and torque
    must already be rotated into ``task_frame_id``.  ``contact_position_task_m``
    is the point at which the force acts, relative to the same task-frame
    origin; it can be one fixed ``(3,)`` position or one ``(N, 3)`` position per
    sample.
    """

    calibration_id: str
    task_frame_id: str
    wrench_task: np.ndarray
    contact_position_task_m: np.ndarray
    sdk_time_s: np.ndarray
    capture_monotonic_s: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        if not self.calibration_id:
            raise ValueError("calibration_id must be non-empty")
        if not self.task_frame_id:
            raise ValueError("task_frame_id must be non-empty")
        wrench = np.array(self.wrench_task, dtype=float, copy=True)
        if wrench.ndim != 2 or wrench.shape[1] != 6:
            raise ValueError("wrench_task must have shape (N, 6)")
        sample_count = wrench.shape[0]
        positions = np.array(self.contact_position_task_m, dtype=float, copy=True)
        if positions.shape == (3,):
            positions = np.broadcast_to(positions, (sample_count, 3)).copy()
        if positions.shape != (sample_count, 3):
            raise ValueError("contact_position_task_m must have shape (3,) or (N, 3)")
        sdk_time = np.array(self.sdk_time_s, dtype=float, copy=True)
        capture_time = np.array(self.capture_monotonic_s, dtype=float, copy=True)
        valid = np.array(self.valid, dtype=bool, copy=True)
        for name, value in (
            ("sdk_time_s", sdk_time),
            ("capture_monotonic_s", capture_time),
            ("valid", valid),
        ):
            if value.shape != (sample_count,):
                raise ValueError(f"{name} must have shape ({sample_count},)")
        if not np.isfinite(sdk_time).all() or not np.isfinite(capture_time).all():
            raise ValueError("timestamps must be finite")
        wrench.setflags(write=False)
        positions.setflags(write=False)
        sdk_time.setflags(write=False)
        capture_time.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "wrench_task", wrench)
        object.__setattr__(self, "contact_position_task_m", positions)
        object.__setattr__(self, "sdk_time_s", sdk_time)
        object.__setattr__(self, "capture_monotonic_s", capture_time)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True)
class TactileTwistConfig:
    profile_id: str
    min_unique_samples: int = 4
    max_age_s: float = 0.20
    max_pair_skew_s: float = 0.05
    min_window_span_s: float = 0.06
    baseline_axis_alignment_cosine: float = 0.999
    press_reaction_sign: int = 1
    press_engage_sum_z: float = 5.0
    press_retain_sum_z: float = 2.5
    press_engage_finger_z: float = 2.5
    press_retain_finger_z: float = 1.25
    max_press_imbalance_enter: float = 0.45
    max_press_imbalance_retain: float = 0.65
    max_press_range_z: float = 8.0
    press_required_consecutive_windows: int = 2
    engage_drive_z: float = 5.0
    retain_drive_z: float = 2.5
    engage_moment_z: float = 5.0
    retain_moment_z: float = 2.5
    engage_finger_z: float = 2.5
    retain_finger_z: float = 1.25
    max_imbalance_enter: float = 0.45
    max_imbalance_retain: float = 0.65
    release_drive_peak_fraction: float = 0.25
    release_moment_peak_fraction: float = 0.25
    release_min_peak_z: float = 6.0
    release_consecutive_windows: int = 2
    wrist_reset_fraction: float = 0.90
    unload_return_z: float = 2.5

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        if self.min_unique_samples < 2:
            raise ValueError("min_unique_samples must be at least two")
        for name in ("max_age_s", "max_pair_skew_s", "min_window_span_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.min_window_span_s > self.max_age_s:
            raise ValueError("min_window_span_s cannot exceed max_age_s")
        if not 0.0 < self.baseline_axis_alignment_cosine <= 1.0:
            raise ValueError("baseline_axis_alignment_cosine must lie in (0, 1]")
        if self.press_reaction_sign not in (-1, 1):
            raise ValueError("press_reaction_sign must be -1 or +1")
        for name in (
            "press_engage_sum_z",
            "press_retain_sum_z",
            "press_engage_finger_z",
            "press_retain_finger_z",
            "max_press_range_z",
            "engage_drive_z",
            "retain_drive_z",
            "engage_moment_z",
            "retain_moment_z",
            "engage_finger_z",
            "retain_finger_z",
            "release_min_peak_z",
            "unload_return_z",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.press_retain_sum_z >= self.press_engage_sum_z:
            raise ValueError(
                "press_retain_sum_z must be smaller than press_engage_sum_z"
            )
        if self.press_retain_finger_z >= self.press_engage_finger_z:
            raise ValueError(
                "press_retain_finger_z must be smaller than press_engage_finger_z"
            )
        if self.retain_drive_z >= self.engage_drive_z:
            raise ValueError("retain_drive_z must be smaller than engage_drive_z")
        if self.retain_moment_z >= self.engage_moment_z:
            raise ValueError("retain_moment_z must be smaller than engage_moment_z")
        if self.retain_finger_z >= self.engage_finger_z:
            raise ValueError("retain_finger_z must be smaller than engage_finger_z")
        for name in (
            "max_imbalance_enter",
            "max_imbalance_retain",
            "max_press_imbalance_enter",
            "max_press_imbalance_retain",
            "release_drive_peak_fraction",
            "release_moment_peak_fraction",
            "wrist_reset_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
        if self.max_imbalance_enter >= self.max_imbalance_retain:
            raise ValueError(
                "max_imbalance_enter must be smaller than max_imbalance_retain"
            )
        if self.max_press_imbalance_enter >= self.max_press_imbalance_retain:
            raise ValueError(
                "max_press_imbalance_enter must be smaller than "
                "max_press_imbalance_retain"
            )
        if self.press_required_consecutive_windows < 1:
            raise ValueError("press_required_consecutive_windows must be positive")
        if self.release_consecutive_windows < 1:
            raise ValueError("release_consecutive_windows must be positive")


@dataclass(frozen=True)
class TactileTwistBaseline:
    profile_id: str
    task_frame_id: str
    calibration_ids: tuple[str, str]
    bottle_axis_task: np.ndarray
    left_axis_center: float
    right_axis_center: float
    left_axis_scale: float
    right_axis_scale: float
    press_sum_center: float
    press_sum_scale: float
    left_tangent_center: float
    right_tangent_center: float
    left_tangent_scale: float
    right_tangent_scale: float
    drive_center: float
    drive_scale: float
    force_axis_moment_center: float
    force_axis_moment_scale: float
    wrench_axis_torque_center: float
    wrench_axis_torque_scale: float
    evidence_key: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.profile_id or not self.task_frame_id:
            raise ValueError("baseline profile and task frame must be non-empty")
        if len(self.calibration_ids) != 2 or not all(self.calibration_ids):
            raise ValueError("baseline calibration_ids must contain two non-empty ids")
        axis = _unit_vector(self.bottle_axis_task, "bottle_axis_task")
        axis.setflags(write=False)
        object.__setattr__(self, "bottle_axis_task", axis)
        for name in (
            "left_axis_center",
            "right_axis_center",
            "press_sum_center",
            "left_tangent_center",
            "right_tangent_center",
            "drive_center",
            "force_axis_moment_center",
            "wrench_axis_torque_center",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        for name in (
            "left_axis_scale",
            "right_axis_scale",
            "press_sum_scale",
            "left_tangent_scale",
            "right_tangent_scale",
            "drive_scale",
            "force_axis_moment_scale",
            "wrench_axis_torque_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if len(self.evidence_key) != 4 or not all(
            math.isfinite(float(value)) for value in self.evidence_key
        ):
            raise ValueError("baseline evidence_key must contain four finite values")


@dataclass(frozen=True)
class TactileTwistMemory:
    previous_state: TactileTwistState = TactileTwistState.NO_ENGAGEMENT
    peak_abs_drive_delta: float = 0.0
    peak_abs_force_axis_moment_delta: float = 0.0
    peak_abs_left_delta: float = 0.0
    peak_abs_right_delta: float = 0.0
    engaged_windows: int = 0
    release_low_windows: int = 0
    press_stable_windows: int = 0
    press_confirmed: bool = False
    last_evidence_key: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "previous_state", TactileTwistState(self.previous_state))
        for name in (
            "peak_abs_drive_delta",
            "peak_abs_force_axis_moment_delta",
            "peak_abs_left_delta",
            "peak_abs_right_delta",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            self.engaged_windows < 0
            or self.release_low_windows < 0
            or self.press_stable_windows < 0
        ):
            raise ValueError("memory counters must be non-negative")
        if not isinstance(self.press_confirmed, bool):
            raise ValueError("press_confirmed must be boolean")


@dataclass(frozen=True)
class FingerTwistMetrics:
    axial_reaction_force: float
    axial_reaction_delta: float
    axial_reaction_z: float
    axial_range_z: float
    signed_tangential_force: float
    tangential_delta_from_baseline: float
    tangential_z: float
    radial_lever_m: float
    force_axis_moment: float
    sensor_axis_torque: float
    wrench_axis_torque: float


@dataclass(frozen=True)
class TactileTwistEvaluation:
    state: TactileTwistState
    action: TactileTwistAction
    valid: bool
    left: FingerTwistMetrics | None
    right: FingerTwistMetrics | None
    axial_press_reaction: float | None
    axial_press_delta: float | None
    axial_press_z: float | None
    axial_press_imbalance: float | None
    axial_press_range_z: float | None
    signed_tangential_drive: float | None
    drive_delta_from_baseline: float | None
    drive_z: float | None
    force_axis_moment: float | None
    force_axis_moment_delta: float | None
    force_axis_moment_z: float | None
    wrench_axis_torque: float | None
    load_imbalance: float | None
    wrist_travel_fraction: float
    predicates: frozenset[str]
    reasons: tuple[str, ...]
    evidence_key: tuple[float, ...] | None


@dataclass(frozen=True)
class TactileTwistResult:
    evaluation: TactileTwistEvaluation
    memory: TactileTwistMemory


@dataclass(frozen=True)
class _FreshFinger:
    force: np.ndarray
    torque: np.ndarray
    position: np.ndarray
    sdk_time_s: np.ndarray
    capture_time_s: np.ndarray

    @property
    def evidence_pair(self) -> tuple[float, float]:
        return float(self.sdk_time_s[-1]), float(self.capture_time_s[-1])


@dataclass(frozen=True)
class _FingerSeries:
    axial: np.ndarray
    tangential: np.ndarray
    radial_lever_m: np.ndarray
    force_axis_moment: np.ndarray
    sensor_axis_torque: np.ndarray
    wrench_axis_torque: np.ndarray


def _unit_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.array(value, dtype=float, copy=True)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def _finite_point(value: np.ndarray, name: str) -> np.ndarray:
    point = np.array(value, dtype=float, copy=True)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must contain three finite values")
    return point


def _fresh_finger(
    window: FingerWrenchWindow,
    *,
    config: TactileTwistConfig,
    now_s: float,
) -> tuple[_FreshFinger | None, str | None]:
    valid_indices = np.flatnonzero(window.valid)
    if valid_indices.size == 0:
        return None, "has no valid samples"
    sdk_times = window.sdk_time_s[valid_indices]
    capture_times = window.capture_monotonic_s[valid_indices]
    if np.any(np.diff(sdk_times) < 0.0):
        return None, "SDK timestamps moved backwards"
    if np.any(np.diff(capture_times) < 0.0):
        return None, "host timestamps moved backwards"
    finite = np.isfinite(window.wrench_task[valid_indices]).all(axis=1)
    finite &= np.isfinite(window.contact_position_task_m[valid_indices]).all(axis=1)
    valid_indices = valid_indices[finite]
    if valid_indices.size == 0:
        return None, "has no finite valid samples"

    # Keep the first host receipt of each SDK sample.  A cached force value
    # repeatedly published with a new host time is one sample, not new evidence.
    _, first_positions = np.unique(window.sdk_time_s[valid_indices], return_index=True)
    selected = valid_indices[np.sort(first_positions)]
    capture_times = window.capture_monotonic_s[selected]
    fresh = (capture_times >= now_s - config.max_age_s) & (
        capture_times <= now_s + 1e-6
    )
    selected = selected[fresh]
    if selected.size < config.min_unique_samples:
        return None, (
            f"has {selected.size} fresh unique SDK samples; "
            f"requires {config.min_unique_samples}"
        )
    capture_times = window.capture_monotonic_s[selected]
    if float(capture_times[-1] - capture_times[0]) < config.min_window_span_s:
        return None, "fresh samples do not span the required time window"
    wrench = window.wrench_task[selected]
    return (
        _FreshFinger(
            force=wrench[:, :3],
            torque=wrench[:, 3:],
            position=window.contact_position_task_m[selected],
            sdk_time_s=window.sdk_time_s[selected],
            capture_time_s=capture_times,
        ),
        None,
    )


def _series(
    finger: _FreshFinger,
    *,
    bottle_axis_task: np.ndarray,
    axis_point_task_m: np.ndarray,
) -> _FingerSeries:
    offset = finger.position - axis_point_task_m[None, :]
    radial = offset - np.outer(offset @ bottle_axis_task, bottle_axis_task)
    lever = np.linalg.norm(radial, axis=1)
    if np.any(lever <= 1e-6):
        raise ValueError("contact positions must be at least 1 um from the bottle axis")
    tangent = np.cross(np.broadcast_to(bottle_axis_task, radial.shape), radial)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    tangential = np.einsum("ij,ij->i", finger.force, tangent)
    force_moment_vector = np.cross(offset, finger.force)
    force_axis_moment = force_moment_vector @ bottle_axis_task
    sensor_axis_torque = finger.torque @ bottle_axis_task
    return _FingerSeries(
        axial=finger.force @ bottle_axis_task,
        tangential=tangential,
        radial_lever_m=lever,
        force_axis_moment=force_axis_moment,
        sensor_axis_torque=sensor_axis_torque,
        wrench_axis_torque=force_axis_moment + sensor_axis_torque,
    )


def _median(value: np.ndarray) -> float:
    return float(np.median(value))


def _robust_scale(value: np.ndarray, *, name: str) -> float:
    center = np.median(value)
    scale = float(1.4826 * np.median(np.abs(value - center)))
    if scale <= 1e-12:
        scale = float(np.std(value, ddof=1)) if value.size > 1 else 0.0
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError(
            f"{name} baseline has no measurable variation; relative thresholds "
            "cannot be qualified"
        )
    return scale


def _prepare_pair(
    left: FingerWrenchWindow,
    right: FingerWrenchWindow,
    *,
    config: TactileTwistConfig,
    now_s: float,
) -> tuple[_FreshFinger | None, _FreshFinger | None, tuple[str, ...]]:
    if not math.isfinite(now_s):
        raise ValueError("now_s must be finite")
    reasons: list[str] = []
    if left.task_frame_id != right.task_frame_id:
        reasons.append("left and right wrench windows use different task frames")
    fresh_left, left_error = _fresh_finger(left, config=config, now_s=now_s)
    fresh_right, right_error = _fresh_finger(right, config=config, now_s=now_s)
    if left_error:
        reasons.append(f"left wrench window {left_error}")
    if right_error:
        reasons.append(f"right wrench window {right_error}")
    if fresh_left is not None and fresh_right is not None:
        skew = abs(fresh_left.evidence_pair[1] - fresh_right.evidence_pair[1])
        if skew > config.max_pair_skew_s:
            reasons.append(
                f"left/right latest-sample skew {skew:.6f}s exceeds "
                f"{config.max_pair_skew_s:.6f}s"
            )
    return fresh_left, fresh_right, tuple(reasons)


def build_tactile_twist_baseline(
    left: FingerWrenchWindow,
    right: FingerWrenchWindow,
    *,
    bottle_axis_task: np.ndarray,
    axis_point_task_m: np.ndarray,
    config: TactileTwistConfig,
    now_s: float,
) -> TactileTwistBaseline:
    """Build a stationary pre-press baseline for one twist segment.

    The fingers already hold the cap, but no axial press increment has run.  A
    new baseline must be captured after each wrist reset/regrasp and before its
    first press increment.  Runtime invalidity raises ``ValueError`` because a
    runner must not start a segment without a qualified baseline.
    """

    axis = _unit_vector(bottle_axis_task, "bottle_axis_task")
    axis_point = _finite_point(axis_point_task_m, "axis_point_task_m")
    fresh_left, fresh_right, reasons = _prepare_pair(
        left, right, config=config, now_s=now_s
    )
    if reasons:
        raise ValueError("invalid baseline window: " + "; ".join(reasons))
    assert fresh_left is not None and fresh_right is not None
    left_series = _series(
        fresh_left, bottle_axis_task=axis, axis_point_task_m=axis_point
    )
    right_series = _series(
        fresh_right, bottle_axis_task=axis, axis_point_task_m=axis_point
    )
    left_axis_scale = _robust_scale(left_series.axial, name="left axial reaction")
    right_axis_scale = _robust_scale(right_series.axial, name="right axial reaction")
    left_scale = _robust_scale(left_series.tangential, name="left tangential force")
    right_scale = _robust_scale(right_series.tangential, name="right tangential force")
    left_moment_scale = _robust_scale(
        left_series.force_axis_moment, name="left force-axis moment"
    )
    right_moment_scale = _robust_scale(
        right_series.force_axis_moment, name="right force-axis moment"
    )
    left_wrench_scale = _robust_scale(
        left_series.wrench_axis_torque, name="left wrench-axis torque"
    )
    right_wrench_scale = _robust_scale(
        right_series.wrench_axis_torque, name="right wrench-axis torque"
    )
    return TactileTwistBaseline(
        profile_id=config.profile_id,
        task_frame_id=left.task_frame_id,
        calibration_ids=(left.calibration_id, right.calibration_id),
        bottle_axis_task=axis,
        left_axis_center=_median(left_series.axial),
        right_axis_center=_median(right_series.axial),
        left_axis_scale=left_axis_scale,
        right_axis_scale=right_axis_scale,
        press_sum_center=_median(left_series.axial) + _median(right_series.axial),
        press_sum_scale=math.hypot(left_axis_scale, right_axis_scale),
        left_tangent_center=_median(left_series.tangential),
        right_tangent_center=_median(right_series.tangential),
        left_tangent_scale=left_scale,
        right_tangent_scale=right_scale,
        drive_center=(
            _median(left_series.tangential) + _median(right_series.tangential)
        ),
        drive_scale=math.hypot(left_scale, right_scale),
        force_axis_moment_center=(
            _median(left_series.force_axis_moment)
            + _median(right_series.force_axis_moment)
        ),
        force_axis_moment_scale=math.hypot(left_moment_scale, right_moment_scale),
        wrench_axis_torque_center=(
            _median(left_series.wrench_axis_torque)
            + _median(right_series.wrench_axis_torque)
        ),
        wrench_axis_torque_scale=math.hypot(left_wrench_scale, right_wrench_scale),
        evidence_key=(*fresh_left.evidence_pair, *fresh_right.evidence_pair),
    )


def _invalid_result(
    *,
    reasons: tuple[str, ...],
    memory: TactileTwistMemory,
    wrist_fraction: float,
    evidence_key: tuple[float, ...] | None = None,
) -> TactileTwistResult:
    evaluation = TactileTwistEvaluation(
        state=TactileTwistState.INVALID,
        action=TactileTwistAction.HOLD_INVALID,
        valid=False,
        left=None,
        right=None,
        axial_press_reaction=None,
        axial_press_delta=None,
        axial_press_z=None,
        axial_press_imbalance=None,
        axial_press_range_z=None,
        signed_tangential_drive=None,
        drive_delta_from_baseline=None,
        drive_z=None,
        force_axis_moment=None,
        force_axis_moment_delta=None,
        force_axis_moment_z=None,
        wrench_axis_torque=None,
        load_imbalance=None,
        wrist_travel_fraction=wrist_fraction,
        predicates=frozenset(),
        reasons=reasons,
        evidence_key=evidence_key,
    )
    return TactileTwistResult(evaluation=evaluation, memory=memory)


def evaluate_tactile_twist(
    left: FingerWrenchWindow,
    right: FingerWrenchWindow,
    *,
    bottle_axis_task: np.ndarray,
    axis_point_task_m: np.ndarray,
    commanded_twist_sign: int,
    wrist_travel_rad: float,
    wrist_travel_limit_rad: float,
    baseline: TactileTwistBaseline,
    config: TactileTwistConfig,
    mode: TactileEvaluationMode,
    now_s: float,
    memory: TactileTwistMemory | None = None,
) -> TactileTwistResult:
    """Evaluate one fresh tactile window and recommend a stage transition.

    ``PRESS`` evaluates the reaction to a completed bounded axial increment;
    ``TWIST`` supervises a completed geometric rotation microstep; ``UNLOAD``
    verifies return toward the pre-press baseline before wrist reset.  The
    runner owns every motion and this function never selects a distance or
    angle.
    """

    memory = memory or TactileTwistMemory()
    mode = TactileEvaluationMode(mode)
    if commanded_twist_sign not in (-1, 1):
        raise ValueError("commanded_twist_sign must be -1 or +1")
    wrist_travel = float(wrist_travel_rad)
    wrist_limit = float(wrist_travel_limit_rad)
    if (
        not math.isfinite(wrist_travel)
        or not math.isfinite(wrist_limit)
        or wrist_travel < 0.0
        or wrist_limit <= 0.0
    ):
        raise ValueError("wrist travel must be non-negative with a positive limit")
    wrist_fraction = wrist_travel / wrist_limit
    axis = _unit_vector(bottle_axis_task, "bottle_axis_task")
    axis_point = _finite_point(axis_point_task_m, "axis_point_task_m")

    provenance_reasons: list[str] = []
    if baseline.profile_id != config.profile_id:
        provenance_reasons.append("baseline profile does not match evaluator profile")
    if (
        left.task_frame_id != baseline.task_frame_id
        or right.task_frame_id != baseline.task_frame_id
    ):
        provenance_reasons.append("wrench task frame does not match baseline")
    if (left.calibration_id, right.calibration_id) != baseline.calibration_ids:
        provenance_reasons.append("sensor calibration ids changed after baseline")
    if float(np.dot(axis, baseline.bottle_axis_task)) < config.baseline_axis_alignment_cosine:
        provenance_reasons.append("bottle axis changed relative to the segment baseline")
    fresh_left, fresh_right, fresh_reasons = _prepare_pair(
        left, right, config=config, now_s=now_s
    )
    provenance_reasons.extend(fresh_reasons)
    if provenance_reasons:
        return _invalid_result(
            reasons=tuple(provenance_reasons),
            memory=memory,
            wrist_fraction=wrist_fraction,
        )
    assert fresh_left is not None and fresh_right is not None
    evidence_key = (*fresh_left.evidence_pair, *fresh_right.evidence_pair)
    if memory.last_evidence_key is not None and evidence_key == memory.last_evidence_key:
        return _invalid_result(
            reasons=("tactile evidence window was replayed",),
            memory=memory,
            wrist_fraction=wrist_fraction,
            evidence_key=evidence_key,
        )
    if (
        float(fresh_left.capture_time_s[0]) <= baseline.evidence_key[1]
        or float(fresh_right.capture_time_s[0]) <= baseline.evidence_key[3]
    ):
        return _invalid_result(
            reasons=("tactile evidence window overlaps the pre-press baseline",),
            memory=memory,
            wrist_fraction=wrist_fraction,
            evidence_key=evidence_key,
        )
    if memory.last_evidence_key is not None and (
        float(fresh_left.capture_time_s[0]) <= memory.last_evidence_key[1]
        or float(fresh_right.capture_time_s[0]) <= memory.last_evidence_key[3]
    ):
        return _invalid_result(
            reasons=("tactile evidence window overlaps previously consumed evidence",),
            memory=memory,
            wrist_fraction=wrist_fraction,
            evidence_key=evidence_key,
        )

    try:
        left_series = _series(
            fresh_left, bottle_axis_task=axis, axis_point_task_m=axis_point
        )
        right_series = _series(
            fresh_right, bottle_axis_task=axis, axis_point_task_m=axis_point
        )
    except ValueError as exc:
        return _invalid_result(
            reasons=(str(exc),),
            memory=memory,
            wrist_fraction=wrist_fraction,
            evidence_key=evidence_key,
        )

    left_tangent = _median(left_series.tangential)
    right_tangent = _median(right_series.tangential)
    reaction_sign = config.press_reaction_sign
    left_axis = reaction_sign * _median(left_series.axial)
    right_axis = reaction_sign * _median(right_series.axial)
    left_axis_center = reaction_sign * baseline.left_axis_center
    right_axis_center = reaction_sign * baseline.right_axis_center
    left_axis_delta = left_axis - left_axis_center
    right_axis_delta = right_axis - right_axis_center
    left_axis_z = left_axis_delta / baseline.left_axis_scale
    right_axis_z = right_axis_delta / baseline.right_axis_scale
    axial_press = left_axis + right_axis
    axial_press_delta = axial_press - reaction_sign * baseline.press_sum_center
    axial_press_z = axial_press_delta / baseline.press_sum_scale
    axial_absolute_sum = abs(left_axis_delta) + abs(right_axis_delta)
    axial_imbalance = (
        abs(abs(left_axis_delta) - abs(right_axis_delta)) / axial_absolute_sum
        if axial_absolute_sum > 1e-15
        else 0.0
    )
    left_axis_range_z = (
        float(np.percentile(reaction_sign * left_series.axial, 95.0))
        - float(np.percentile(reaction_sign * left_series.axial, 5.0))
    ) / baseline.left_axis_scale
    right_axis_range_z = (
        float(np.percentile(reaction_sign * right_series.axial, 95.0))
        - float(np.percentile(reaction_sign * right_series.axial, 5.0))
    ) / baseline.right_axis_scale
    axial_range_z = max(left_axis_range_z, right_axis_range_z)
    left_delta = left_tangent - baseline.left_tangent_center
    right_delta = right_tangent - baseline.right_tangent_center
    left_z = abs(left_delta) / baseline.left_tangent_scale
    right_z = abs(right_delta) / baseline.right_tangent_scale
    signed_drive = left_tangent + right_tangent
    drive_delta = signed_drive - baseline.drive_center
    drive_z = abs(drive_delta) / baseline.drive_scale
    force_axis_moment = (
        _median(left_series.force_axis_moment)
        + _median(right_series.force_axis_moment)
    )
    force_axis_moment_delta = (
        force_axis_moment - baseline.force_axis_moment_center
    )
    moment_z = abs(force_axis_moment_delta) / baseline.force_axis_moment_scale
    wrench_axis_torque = (
        _median(left_series.wrench_axis_torque)
        + _median(right_series.wrench_axis_torque)
    )
    absolute_finger_sum = abs(left_delta) + abs(right_delta)
    imbalance = (
        abs(abs(left_delta) - abs(right_delta)) / absolute_finger_sum
        if absolute_finger_sum > 1e-15
        else 0.0
    )
    left_direction_ok = commanded_twist_sign * left_delta > 0.0
    right_direction_ok = commanded_twist_sign * right_delta > 0.0
    drive_direction_ok = commanded_twist_sign * drive_delta > 0.0
    moment_direction_ok = commanded_twist_sign * force_axis_moment_delta > 0.0
    directions_coherent = (
        left_direction_ok
        and right_direction_ok
        and drive_direction_ok
        and moment_direction_ok
    )

    press_enter = (
        left_axis_z >= config.press_engage_finger_z
        and right_axis_z >= config.press_engage_finger_z
        and axial_press_z >= config.press_engage_sum_z
        and axial_imbalance <= config.max_press_imbalance_enter
        and axial_range_z <= config.max_press_range_z
    )
    press_retain = (
        left_axis_z >= config.press_retain_finger_z
        and right_axis_z >= config.press_retain_finger_z
        and axial_press_z >= config.press_retain_sum_z
        and axial_imbalance <= config.max_press_imbalance_retain
        and axial_range_z <= config.max_press_range_z
    )

    was_engaged = memory.engaged_windows > 0
    enter_engaged = (
        directions_coherent
        and left_z >= config.engage_finger_z
        and right_z >= config.engage_finger_z
        and drive_z >= config.engage_drive_z
        and moment_z >= config.engage_moment_z
        and imbalance <= config.max_imbalance_enter
    )
    retain_engaged = (
        directions_coherent
        and left_z >= config.retain_finger_z
        and right_z >= config.retain_finger_z
        and drive_z >= config.retain_drive_z
        and moment_z >= config.retain_moment_z
        and imbalance <= config.max_imbalance_retain
    )

    tangentially_engaged = enter_engaged or (was_engaged and retain_engaged)
    peak_drive = (
        max(memory.peak_abs_drive_delta, abs(drive_delta))
        if mode is TactileEvaluationMode.TWIST and tangentially_engaged and press_retain
        else memory.peak_abs_drive_delta
    )
    peak_moment = (
        max(memory.peak_abs_force_axis_moment_delta, abs(force_axis_moment_delta))
        if mode is TactileEvaluationMode.TWIST and tangentially_engaged and press_retain
        else memory.peak_abs_force_axis_moment_delta
    )
    peak_left = (
        max(memory.peak_abs_left_delta, abs(left_delta))
        if mode is TactileEvaluationMode.TWIST and tangentially_engaged and press_retain
        else memory.peak_abs_left_delta
    )
    peak_right = (
        max(memory.peak_abs_right_delta, abs(right_delta))
        if mode is TactileEvaluationMode.TWIST and tangentially_engaged and press_retain
        else memory.peak_abs_right_delta
    )
    peak_qualified = (
        memory.engaged_windows > 0
        and memory.peak_abs_drive_delta / baseline.drive_scale >= config.release_min_peak_z
        and memory.peak_abs_force_axis_moment_delta
        / baseline.force_axis_moment_scale
        >= config.release_min_peak_z
    )
    release_low = (
        peak_qualified
        and abs(drive_delta)
        <= config.release_drive_peak_fraction * memory.peak_abs_drive_delta
        and abs(force_axis_moment_delta)
        <= config.release_moment_peak_fraction
        * memory.peak_abs_force_axis_moment_delta
    )
    release_windows = memory.release_low_windows + 1 if release_low else 0

    strong_signal = (
        drive_z >= config.retain_drive_z
        or moment_z >= config.retain_moment_z
        or left_z >= config.retain_finger_z
        or right_z >= config.retain_finger_z
    )
    slip_or_imbalanced = strong_signal and (
        not directions_coherent
        or imbalance > config.max_imbalance_retain
    )

    predicates: set[str] = set()
    reasons: list[str] = []
    press_stable_windows = memory.press_stable_windows
    press_confirmed = memory.press_confirmed
    if directions_coherent:
        predicates.add("tangential_direction_matches_command")
    if imbalance <= config.max_imbalance_enter:
        predicates.add("bilateral_tangential_load_balanced")
    if tangentially_engaged:
        predicates.add("tangential_engagement_stable")
    if press_retain:
        predicates.add("axial_press_retained")

    if mode is TactileEvaluationMode.PRESS:
        one_sided_press = (
            max(left_axis_z, right_axis_z) >= config.press_engage_finger_z
            and axial_imbalance > config.max_press_imbalance_retain
        )
        if one_sided_press:
            state = TactileTwistState.SLIP_OR_IMBALANCED
            action = TactileTwistAction.SLIP_RECOVERY
            press_stable_windows = 0
            reasons.append("left/right axial press reaction is severely imbalanced")
        elif press_enter:
            press_stable_windows += 1
            predicates.add("bilateral_axial_press_stable")
            if press_stable_windows >= config.press_required_consecutive_windows:
                state = TactileTwistState.PRESS_ENGAGED
                action = TactileTwistAction.START_TWIST
                press_confirmed = True
                predicates.add("axial_press_engaged")
                reasons.append("stable bilateral axial reaction confirms downpress")
            else:
                state = TactileTwistState.PRESS_NOT_ENGAGED
                action = TactileTwistAction.HOLD_FOR_ENGAGEMENT
                reasons.append("axial press passed once; waiting for stability window")
        else:
            state = TactileTwistState.PRESS_NOT_ENGAGED
            action = TactileTwistAction.PRESS_MORE
            press_stable_windows = 0
            reasons.append("axial press reaction has not reached the engagement band")
    elif mode is TactileEvaluationMode.UNLOAD:
        unloaded = (
            abs(left_axis_z) <= config.unload_return_z
            and abs(right_axis_z) <= config.unload_return_z
            and drive_z <= config.unload_return_z
        )
        state = TactileTwistState.NO_ENGAGEMENT
        if unloaded:
            action = TactileTwistAction.UNLOAD_CONFIRMED
            predicates.add("returned_to_pre_press_baseline")
            reasons.append("axial and tangential loads returned to the unload band")
        else:
            action = TactileTwistAction.HOLD_FOR_ENGAGEMENT
            reasons.append("loads have not returned to the pre-press unload band")
    elif not press_confirmed:
        state = TactileTwistState.PRESS_NOT_ENGAGED
        action = TactileTwistAction.PRESS_MORE
        reasons.append("formal twist is blocked until axial press is confirmed")
    elif not press_retain:
        state = TactileTwistState.SLIP_OR_IMBALANCED
        action = TactileTwistAction.SLIP_RECOVERY
        reasons.append("axial downpress was lost or became unstable during twist")
    elif release_windows >= config.release_consecutive_windows:
        state = TactileTwistState.RELEASE_CANDIDATE
        action = TactileTwistAction.RELEASE_CANDIDATE
        predicates.add("drive_and_axis_moment_dropped_from_peak")
        reasons.append(
            "tangential drive and axis moment dropped from peak while axial press held"
        )
    elif slip_or_imbalanced:
        state = TactileTwistState.SLIP_OR_IMBALANCED
        action = TactileTwistAction.SLIP_RECOVERY
        reasons.append("tangential contact slipped, reversed, or became imbalanced")
    elif tangentially_engaged and not release_low:
        state = TactileTwistState.TWIST_ENGAGED
        if wrist_fraction >= config.wrist_reset_fraction:
            action = TactileTwistAction.WRIST_RESET_REGRASP
            predicates.add("wrist_regrasp_boundary_reached")
            reasons.append("wrist boundary reached with stable tactile contact")
        else:
            action = TactileTwistAction.TWIST_CONTINUE
            reasons.append("axial press and bilateral tangential contact remain stable")
    else:
        state = TactileTwistState.NO_ENGAGEMENT
        action = TactileTwistAction.HOLD_FOR_ENGAGEMENT
        reasons.append("tangential supervision has not settled")

    engaged_windows = (
        memory.engaged_windows + 1
        if state is TactileTwistState.TWIST_ENGAGED
        else memory.engaged_windows
    )
    next_memory = TactileTwistMemory(
        previous_state=state,
        peak_abs_drive_delta=peak_drive,
        peak_abs_force_axis_moment_delta=peak_moment,
        peak_abs_left_delta=peak_left,
        peak_abs_right_delta=peak_right,
        engaged_windows=engaged_windows,
        release_low_windows=release_windows,
        press_stable_windows=press_stable_windows,
        press_confirmed=press_confirmed,
        last_evidence_key=evidence_key,
    )

    left_metrics = FingerTwistMetrics(
        axial_reaction_force=left_axis,
        axial_reaction_delta=left_axis_delta,
        axial_reaction_z=left_axis_z,
        axial_range_z=left_axis_range_z,
        signed_tangential_force=left_tangent,
        tangential_delta_from_baseline=left_delta,
        tangential_z=left_z,
        radial_lever_m=_median(left_series.radial_lever_m),
        force_axis_moment=_median(left_series.force_axis_moment),
        sensor_axis_torque=_median(left_series.sensor_axis_torque),
        wrench_axis_torque=_median(left_series.wrench_axis_torque),
    )
    right_metrics = FingerTwistMetrics(
        axial_reaction_force=right_axis,
        axial_reaction_delta=right_axis_delta,
        axial_reaction_z=right_axis_z,
        axial_range_z=right_axis_range_z,
        signed_tangential_force=right_tangent,
        tangential_delta_from_baseline=right_delta,
        tangential_z=right_z,
        radial_lever_m=_median(right_series.radial_lever_m),
        force_axis_moment=_median(right_series.force_axis_moment),
        sensor_axis_torque=_median(right_series.sensor_axis_torque),
        wrench_axis_torque=_median(right_series.wrench_axis_torque),
    )
    evaluation = TactileTwistEvaluation(
        state=state,
        action=action,
        valid=True,
        left=left_metrics,
        right=right_metrics,
        axial_press_reaction=axial_press,
        axial_press_delta=axial_press_delta,
        axial_press_z=axial_press_z,
        axial_press_imbalance=axial_imbalance,
        axial_press_range_z=axial_range_z,
        signed_tangential_drive=signed_drive,
        drive_delta_from_baseline=drive_delta,
        drive_z=drive_z,
        force_axis_moment=force_axis_moment,
        force_axis_moment_delta=force_axis_moment_delta,
        force_axis_moment_z=moment_z,
        wrench_axis_torque=wrench_axis_torque,
        load_imbalance=imbalance,
        wrist_travel_fraction=wrist_fraction,
        predicates=frozenset(predicates),
        reasons=tuple(reasons),
        evidence_key=evidence_key,
    )
    return TactileTwistResult(evaluation=evaluation, memory=next_memory)


__all__ = [
    "FingerTwistMetrics",
    "FingerWrenchWindow",
    "TactileEvaluationMode",
    "TactileTwistAction",
    "TactileTwistBaseline",
    "TactileTwistConfig",
    "TactileTwistEvaluation",
    "TactileTwistMemory",
    "TactileTwistResult",
    "TactileTwistState",
    "build_tactile_twist_baseline",
    "evaluate_tactile_twist",
]
