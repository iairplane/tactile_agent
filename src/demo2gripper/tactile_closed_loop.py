"""Pure closed-loop policy for role-aware tactile cup-rim probing.

The policy owns no hardware. Each call consumes one settled tactile window
and returns one atomic action. A runner executes that action, records its
receipt, and calls the policy again with new evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

from .cup_tactile_gate import CupRimObservation
from .tactile_rim_probe import (
    MAX_PROBE_CLOSE_STEP_M,
    MAX_PROBE_RELEASE_STEP_M,
    MAX_PROBE_TCP_STEP_M,
)


class TactileClosedLoopAction(StrEnum):
    CLOSE_STEP = "CLOSE_STEP"
    OPEN_RELIEF = "OPEN_RELIEF"
    MOVE_Z = "MOVE_Z"
    MOVE_LATERAL = "MOVE_LATERAL"
    VERIFY = "VERIFY"
    ALLOW_LIFT = "ALLOW_LIFT"
    STOP = "STOP"


class PendingRecovery(StrEnum):
    NONE = "NONE"
    PROBE_DOWN = "PROBE_DOWN"
    RAISE = "RAISE"
    REALIGN = "REALIGN"


@dataclass(frozen=True)
class TactileClosedLoopConfig:
    close_step_m: float = 0.00025
    open_relief_step_m: float = 0.0005
    probe_down_step_m: float = 0.0005
    raise_step_m: float = 0.0005
    lateral_gain: float = 1.0
    max_lateral_step_m: float = 0.0005
    required_stable_matches: int = 3
    max_iterations: int = 16
    max_close_steps: int = 4
    max_probe_down_steps: int = 6
    max_relief_steps: int = 6
    max_adjustment_cycles: int = 6

    def __post_init__(self) -> None:
        positive = (
            "close_step_m",
            "open_relief_step_m",
            "probe_down_step_m",
            "raise_step_m",
            "lateral_gain",
            "max_lateral_step_m",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        caps = {
            "close_step_m": MAX_PROBE_CLOSE_STEP_M,
            "open_relief_step_m": MAX_PROBE_RELEASE_STEP_M,
            "probe_down_step_m": MAX_PROBE_TCP_STEP_M,
            "raise_step_m": MAX_PROBE_TCP_STEP_M,
            "max_lateral_step_m": MAX_PROBE_TCP_STEP_M,
        }
        for name, cap in caps.items():
            if getattr(self, name) > cap:
                raise ValueError(f"{name} exceeds the reviewed cap {cap}")
        counts = (
            "required_stable_matches",
            "max_iterations",
            "max_close_steps",
            "max_probe_down_steps",
            "max_relief_steps",
            "max_adjustment_cycles",
        )
        for name in counts:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class TactileMotionAuthorization:
    """One-shot, journal-backed permission for exactly one TCP correction."""

    authorization_id: str
    probe_context_id: str
    proposal_evidence_id: str
    release_evidence_id: str
    release_command_id: str
    release_completed_at_monotonic_s: float
    applied_jaw_opening_delta_m: float
    issued_at_monotonic_s: float
    expires_at_monotonic_s: float
    action: TactileClosedLoopAction
    z_delta_m: float = 0.0
    lateral_delta_m: float = 0.0
    persisted: bool = False

    def __post_init__(self) -> None:
        for name in (
            "authorization_id",
            "probe_context_id",
            "proposal_evidence_id",
            "release_evidence_id",
            "release_command_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        action = TactileClosedLoopAction(self.action)
        object.__setattr__(self, "action", action)
        if action not in {
            TactileClosedLoopAction.MOVE_Z,
            TactileClosedLoopAction.MOVE_LATERAL,
        }:
            raise ValueError("authorization may only permit MOVE_Z or MOVE_LATERAL")
        completed = float(self.release_completed_at_monotonic_s)
        applied_opening = float(self.applied_jaw_opening_delta_m)
        issued = float(self.issued_at_monotonic_s)
        expires = float(self.expires_at_monotonic_s)
        z_delta = float(self.z_delta_m)
        lateral_delta = float(self.lateral_delta_m)
        if not all(
            math.isfinite(value)
            for value in (
                completed,
                applied_opening,
                issued,
                expires,
                z_delta,
                lateral_delta,
            )
        ):
            raise ValueError("authorization times and deltas must be finite")
        if not 0.0 < applied_opening <= MAX_PROBE_RELEASE_STEP_M:
            raise ValueError("authorization must prove a bounded positive jaw opening")
        if not completed < issued < expires:
            raise ValueError("authorization times must satisfy completed < issued < expires")
        if action is TactileClosedLoopAction.MOVE_Z:
            if abs(z_delta) <= 0.0 or abs(lateral_delta) > 0.0:
                raise ValueError("MOVE_Z authorization must carry only z_delta_m")
        elif abs(lateral_delta) <= 0.0 or abs(z_delta) > 0.0:
            raise ValueError(
                "MOVE_LATERAL authorization must carry only lateral_delta_m"
            )
        if not isinstance(self.persisted, bool):
            raise TypeError("persisted must be a bool")
        object.__setattr__(self, "release_completed_at_monotonic_s", completed)
        object.__setattr__(self, "applied_jaw_opening_delta_m", applied_opening)
        object.__setattr__(self, "issued_at_monotonic_s", issued)
        object.__setattr__(self, "expires_at_monotonic_s", expires)
        object.__setattr__(self, "z_delta_m", z_delta)
        object.__setattr__(self, "lateral_delta_m", lateral_delta)


@dataclass(frozen=True)
class TactileClosedLoopEvidence:
    """One fresh role-aware result at a fixed TCP probe context.

    ``evidence_id`` must be derived from the gate's profile, calibration IDs,
    frozen INNER/OUTER role assignment, and complete SDK/host evidence key.
    ``side_bound_error_m`` is measured-minus-target in task-frame metres and is
    usable only when ``side_bound_error_calibrated`` is true.
    """

    observation: CupRimObservation
    evidence_id: str
    probe_context_id: str
    captured_at_monotonic_s: float
    fresh: bool = True
    valid: bool = True
    overload: bool = False
    can_close_further: bool = False
    side_bound_error_m: float | None = None
    side_bound_error_calibrated: bool = False
    motion_authorization: TactileMotionAuthorization | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation", CupRimObservation(self.observation))
        for name in ("evidence_id", "probe_context_id"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
        captured = float(self.captured_at_monotonic_s)
        if not math.isfinite(captured):
            raise ValueError("captured_at_monotonic_s must be finite")
        object.__setattr__(self, "captured_at_monotonic_s", captured)
        for name in (
            "fresh",
            "valid",
            "overload",
            "can_close_further",
            "side_bound_error_calibrated",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if self.side_bound_error_m is not None:
            error = float(self.side_bound_error_m)
            if not math.isfinite(error):
                raise ValueError("side_bound_error_m must be finite")
            object.__setattr__(self, "side_bound_error_m", error)
        if self.motion_authorization is not None and not isinstance(
            self.motion_authorization, TactileMotionAuthorization
        ):
            raise TypeError("motion_authorization has the wrong type")


@dataclass(frozen=True)
class TactileProbeState:
    iterations: int = 0
    consecutive_matches: int = 0
    close_steps: int = 0
    probe_down_steps: int = 0
    relief_steps: int = 0
    adjustment_cycles: int = 0
    last_evidence_id: str | None = None
    last_capture_monotonic_s: float | None = None
    probe_context_id: str | None = None
    require_new_probe_context: bool = False
    pending_recovery: PendingRecovery = PendingRecovery.NONE
    pending_lateral_delta_m: float = 0.0
    pending_proposal_evidence_id: str | None = None
    consumed_authorization_ids: tuple[str, ...] = ()
    terminal: bool = False
    lift_allowed: bool = False

    def __post_init__(self) -> None:
        counts = (
            "iterations",
            "consecutive_matches",
            "close_steps",
            "probe_down_steps",
            "relief_steps",
            "adjustment_cycles",
        )
        for name in counts:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "pending_recovery", PendingRecovery(self.pending_recovery))
        lateral = float(self.pending_lateral_delta_m)
        if not math.isfinite(lateral):
            raise ValueError("pending_lateral_delta_m must be finite")
        object.__setattr__(self, "pending_lateral_delta_m", lateral)
        if self.pending_recovery is not PendingRecovery.REALIGN and abs(lateral) > 0.0:
            raise ValueError("only REALIGN may carry a lateral correction")
        if self.lift_allowed and not self.terminal:
            raise ValueError("a lift-allowed state must be terminal")


@dataclass(frozen=True)
class TactileClosedLoopDecision:
    action: TactileClosedLoopAction
    next_state: TactileProbeState
    reason: str
    jaw_delta_m: float = 0.0
    z_delta_m: float = 0.0
    lateral_delta_m: float = 0.0
    authorization_id: str | None = None

    def __post_init__(self) -> None:
        action = TactileClosedLoopAction(self.action)
        object.__setattr__(self, "action", action)
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        deltas = {
            "jaw": float(self.jaw_delta_m),
            "z": float(self.z_delta_m),
            "lateral": float(self.lateral_delta_m),
        }
        if not all(math.isfinite(value) for value in deltas.values()):
            raise ValueError("action deltas must be finite")
        expected_axis = {
            TactileClosedLoopAction.CLOSE_STEP: "jaw",
            TactileClosedLoopAction.OPEN_RELIEF: "jaw",
            TactileClosedLoopAction.MOVE_Z: "z",
            TactileClosedLoopAction.MOVE_LATERAL: "lateral",
        }.get(action)
        nonzero = {name for name, value in deltas.items() if abs(value) > 0.0}
        if expected_axis is None and nonzero:
            raise ValueError(f"{action.value} cannot carry a motion delta")
        if expected_axis is not None and nonzero != {expected_axis}:
            raise ValueError(f"{action.value} must carry exactly one motion delta")
        tcp_move = action in {
            TactileClosedLoopAction.MOVE_Z,
            TactileClosedLoopAction.MOVE_LATERAL,
        }
        if tcp_move != bool(self.authorization_id):
            raise ValueError("every TCP move, and only a TCP move, needs authorization")


def _decision(
    action: TactileClosedLoopAction,
    state: TactileProbeState,
    reason: str,
    *,
    jaw: float = 0.0,
    z: float = 0.0,
    lateral: float = 0.0,
    authorization_id: str | None = None,
) -> TactileClosedLoopDecision:
    return TactileClosedLoopDecision(
        action, state, reason, jaw, z, lateral, authorization_id
    )


def _stop(state: TactileProbeState, reason: str) -> TactileClosedLoopDecision:
    stopped = replace(
        state,
        pending_recovery=PendingRecovery.NONE,
        pending_lateral_delta_m=0.0,
        pending_proposal_evidence_id=None,
        terminal=True,
        lift_allowed=False,
    )
    return _decision(TactileClosedLoopAction.STOP, stopped, reason)


def _open_relief(
    state: TactileProbeState,
    config: TactileClosedLoopConfig,
    reason: str,
) -> TactileClosedLoopDecision:
    if state.relief_steps >= config.max_relief_steps:
        return _stop(state, "open-relief step budget is exhausted")
    relieved = replace(
        state,
        consecutive_matches=0,
        relief_steps=state.relief_steps + 1,
    )
    return _decision(
        TactileClosedLoopAction.OPEN_RELIEF,
        relieved,
        reason,
        jaw=config.open_relief_step_m,
    )


def _authorized_tcp_move(
    state: TactileProbeState,
    evidence: TactileClosedLoopEvidence,
    config: TactileClosedLoopConfig,
    *,
    now_monotonic_s: float | None,
) -> TactileClosedLoopDecision:
    authorization = evidence.motion_authorization
    if authorization is None or not authorization.persisted:
        return _stop(state, "TCP correction lacks a persisted one-shot authorization")
    if now_monotonic_s is None or not math.isfinite(now_monotonic_s):
        return _stop(state, "TCP correction lacks a current monotonic time")
    if (
        authorization.authorization_id in state.consumed_authorization_ids
        or authorization.probe_context_id != state.probe_context_id
        or authorization.proposal_evidence_id
        != state.pending_proposal_evidence_id
        or authorization.release_evidence_id != evidence.evidence_id
        or authorization.applied_jaw_opening_delta_m
        < config.open_relief_step_m - 1e-12
        or evidence.captured_at_monotonic_s
        <= authorization.release_completed_at_monotonic_s
        or authorization.issued_at_monotonic_s < evidence.captured_at_monotonic_s
        or now_monotonic_s < authorization.issued_at_monotonic_s
        or now_monotonic_s >= authorization.expires_at_monotonic_s
    ):
        return _stop(state, "TCP authorization is stale, replayed, or context-mismatched")

    recovery = state.pending_recovery
    expected_action = (
        TactileClosedLoopAction.MOVE_LATERAL
        if recovery is PendingRecovery.REALIGN
        else TactileClosedLoopAction.MOVE_Z
    )
    expected_z = {
        PendingRecovery.PROBE_DOWN: -config.probe_down_step_m,
        PendingRecovery.RAISE: config.raise_step_m,
    }.get(recovery, 0.0)
    expected_lateral = (
        state.pending_lateral_delta_m
        if recovery is PendingRecovery.REALIGN
        else 0.0
    )
    if (
        authorization.action is not expected_action
        or not math.isclose(authorization.z_delta_m, expected_z, abs_tol=1e-12)
        or not math.isclose(
            authorization.lateral_delta_m, expected_lateral, abs_tol=1e-12
        )
    ):
        return _stop(state, "TCP authorization does not match the pending correction")

    moved = replace(
        state,
        consecutive_matches=0,
        close_steps=0,
        probe_down_steps=(
            state.probe_down_steps + 1
            if recovery is PendingRecovery.PROBE_DOWN
            else state.probe_down_steps
        ),
        pending_recovery=PendingRecovery.NONE,
        pending_lateral_delta_m=0.0,
        pending_proposal_evidence_id=None,
        consumed_authorization_ids=(
            *state.consumed_authorization_ids,
            authorization.authorization_id,
        ),
        require_new_probe_context=True,
    )
    reason = {
        PendingRecovery.PROBE_DOWN: "released; probe one authorized step along rim -Z",
        PendingRecovery.RAISE: "released; move one authorized step toward rim +Z",
        PendingRecovery.REALIGN: "released; apply the authorized side-bound correction",
    }[recovery]
    return _decision(
        expected_action,
        moved,
        reason,
        z=expected_z,
        lateral=expected_lateral,
        authorization_id=authorization.authorization_id,
    )


def decide_tactile_action(
    state: TactileProbeState,
    evidence: TactileClosedLoopEvidence | None,
    config: TactileClosedLoopConfig = TactileClosedLoopConfig(),
    *,
    now_monotonic_s: float | None = None,
) -> TactileClosedLoopDecision:
    """Consume one role-aware window and return one bounded atomic action."""

    if not isinstance(state, TactileProbeState):
        raise TypeError("state must be a TactileProbeState")
    if state.terminal:
        return _stop(state, "probe episode is already terminal")
    if evidence is None:
        return _stop(state, "fresh tactile evidence is required")
    if not isinstance(evidence, TactileClosedLoopEvidence):
        raise TypeError("evidence has the wrong type")
    if (
        not evidence.fresh
        or not evidence.valid
        or not evidence.evidence_id.strip()
        or not evidence.probe_context_id.strip()
        or evidence.evidence_id == state.last_evidence_id
        or (
            state.last_capture_monotonic_s is not None
            and evidence.captured_at_monotonic_s <= state.last_capture_monotonic_s
        )
    ):
        return _stop(state, "tactile evidence is missing, invalid, stale, or replayed")
    if evidence.overload:
        return _stop(state, "tactile overload blocks all further probing and lift")
    if state.iterations >= config.max_iterations:
        return _stop(state, "tactile probe iteration budget is exhausted")
    if state.probe_context_id is not None:
        context_changed = evidence.probe_context_id != state.probe_context_id
        if state.require_new_probe_context != context_changed:
            return _stop(state, "probe context did not track the preceding TCP motion")

    current = replace(
        state,
        iterations=state.iterations + 1,
        last_evidence_id=evidence.evidence_id,
        last_capture_monotonic_s=evidence.captured_at_monotonic_s,
        probe_context_id=evidence.probe_context_id,
        require_new_probe_context=False,
    )
    observation = evidence.observation
    if observation is CupRimObservation.UNKNOWN:
        return _stop(current, "unknown tactile state blocks lift")

    if current.pending_recovery is not PendingRecovery.NONE:
        if observation is not CupRimObservation.NO_CONTACT:
            return _open_relief(
                current,
                config,
                "contact remains after unloading; apply one more relief step",
            )
        return _authorized_tcp_move(
            current,
            evidence,
            config,
            now_monotonic_s=now_monotonic_s,
        )

    if observation is CupRimObservation.NO_CONTACT:
        current = replace(current, consecutive_matches=0)
        if evidence.can_close_further and current.close_steps < config.max_close_steps:
            closed = replace(current, close_steps=current.close_steps + 1)
            return _decision(
                TactileClosedLoopAction.CLOSE_STEP,
                closed,
                "no contact and safe jaw margin remains; close one bounded step",
                jaw=-config.close_step_m,
            )
        if current.probe_down_steps >= config.max_probe_down_steps:
            return _stop(current, "no-contact probe-down budget is exhausted")
        pending = replace(
            current,
            pending_recovery=PendingRecovery.PROBE_DOWN,
            pending_proposal_evidence_id=evidence.evidence_id,
        )
        return _open_relief(
            pending,
            config,
            "close budget or opening limit reached; unload before probing lower",
        )

    if observation is CupRimObservation.MULTI_RIM_OR_TOO_DEEP:
        if current.adjustment_cycles >= config.max_adjustment_cycles:
            return _stop(current, "tactile adjustment budget is exhausted")
        pending = replace(
            current,
            pending_recovery=PendingRecovery.RAISE,
            pending_proposal_evidence_id=evidence.evidence_id,
            adjustment_cycles=current.adjustment_cycles + 1,
            consecutive_matches=0,
        )
        return _open_relief(
            pending,
            config,
            "multi-rim/too-deep evidence requires unload before moving upward",
        )

    if observation is CupRimObservation.ROLE_MISMATCH:
        error = evidence.side_bound_error_m
        if (
            error is None
            or abs(error) <= 0.0
            or not evidence.side_bound_error_calibrated
        ):
            return _stop(current, "role mismatch lacks a calibrated side-bound error")
        if current.adjustment_cycles >= config.max_adjustment_cycles:
            return _stop(current, "tactile adjustment budget is exhausted")
        correction = max(
            -config.max_lateral_step_m,
            min(config.max_lateral_step_m, -config.lateral_gain * error),
        )
        pending = replace(
            current,
            pending_recovery=PendingRecovery.REALIGN,
            pending_lateral_delta_m=correction,
            pending_proposal_evidence_id=evidence.evidence_id,
            adjustment_cycles=current.adjustment_cycles + 1,
            consecutive_matches=0,
        )
        return _open_relief(
            pending,
            config,
            "role mismatch requires unload before the side-bound correction",
        )

    if observation is CupRimObservation.INNER_OUTER_RIM_MATCH:
        matched = replace(
            current,
            consecutive_matches=current.consecutive_matches + 1,
        )
        if matched.consecutive_matches < config.required_stable_matches:
            return _decision(
                TactileClosedLoopAction.VERIFY,
                matched,
                "role-aware match is not stable yet; capture a fresh window",
            )
        allowed = replace(matched, terminal=True, lift_allowed=True)
        return _decision(
            TactileClosedLoopAction.ALLOW_LIFT,
            allowed,
            "consecutive fresh inner/outer role matches are stable",
        )

    return _stop(current, f"unsupported tactile observation {observation.value}")


__all__ = [
    "PendingRecovery",
    "TactileClosedLoopAction",
    "TactileClosedLoopConfig",
    "TactileClosedLoopDecision",
    "TactileClosedLoopEvidence",
    "TactileMotionAuthorization",
    "TactileProbeState",
    "decide_tactile_action",
]
