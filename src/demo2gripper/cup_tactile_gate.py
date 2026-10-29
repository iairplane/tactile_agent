"""Pre-lift tactile state machine for the nested top-cup task.

The gate consumes only post-evaluator tactile evidence.  It deliberately has no
front-camera input and never counts cups from RGB-D.  All returned actions are
declarative: a runner still owns command execution, completion receipts, and its
atomic journal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from .tactile_rim_probe import (
    MAX_PROBE_TCP_STEP_M,
    ProbeAction,
    RimContactRole,
    RimProbeEvaluation,
    RimProbeState,
    TcpAdjustmentAuthorization,
    TcpAdjustmentAuthorizationLedger,
)


class CupRimObservation(StrEnum):
    NO_CONTACT = "NO_CONTACT"
    MULTI_RIM_OR_TOO_DEEP = "MULTI_RIM_OR_TOO_DEEP"
    INNER_OUTER_RIM_MATCH = "INNER_OUTER_RIM_MATCH"
    ROLE_MISMATCH = "ROLE_MISMATCH"
    UNKNOWN = "UNKNOWN"


class CupTactileGateState(StrEnum):
    AWAIT_TACTILE = "AWAIT_TACTILE"
    AWAIT_RELEASE_AUTHORIZATION = "AWAIT_RELEASE_AUTHORIZATION"
    AWAIT_MOTION_COMPLETION = "AWAIT_MOTION_COMPLETION"
    LIFT_ALLOWED = "LIFT_ALLOWED"
    STOPPED = "STOPPED"


class CupTactileGateAction(StrEnum):
    REQUEST_FRESH_TACTILE = "REQUEST_FRESH_TACTILE"
    PROBE_DOWN = "PROBE_DOWN"
    GENTLE_CLOSE = "GENTLE_CLOSE"
    UNLOAD = "UNLOAD"
    RAISE_AND_REPROBE = "RAISE_AND_REPROBE"
    REALIGN_AND_REPROBE = "REALIGN_AND_REPROBE"
    ALLOW_LIFT = "ALLOW_LIFT"
    STOP = "STOP"


@dataclass(frozen=True)
class CupTactileGateConfig:
    """Task-frame motion caps for one probe episode.

    Rim-task ``+Z`` points out of the cup opening.  Thus probing down is ``-Z``
    and recovery from a too-deep hypothesis must be ``+Z``.
    """

    probe_down_step_m: float = 0.0005
    max_probe_down_steps: int = 4
    max_adjustment_cycles: int = 4
    required_stable_windows: int = 3

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.probe_down_step_m)
            or self.probe_down_step_m <= 0.0
            or self.probe_down_step_m > MAX_PROBE_TCP_STEP_M
        ):
            raise ValueError(
                "probe_down_step_m must be positive and within the reviewed TCP cap"
            )
        for name in (
            "max_probe_down_steps",
            "max_adjustment_cycles",
            "required_stable_windows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class CupTactileGateDecision:
    state: CupTactileGateState
    action: CupTactileGateAction
    observation: CupRimObservation | None
    reason: str
    lift_allowed: bool = False
    jaw_opening_delta_m: float = 0.0
    tcp_translation_rim_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    tcp_yaw_delta_rad: float = 0.0
    probe_context_id: str | None = None
    source_evidence_key: tuple[float, ...] | None = None
    satisfied_predicates: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", CupTactileGateState(self.state))
        object.__setattr__(self, "action", CupTactileGateAction(self.action))
        if self.observation is not None:
            object.__setattr__(
                self, "observation", CupRimObservation(self.observation)
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        jaw_delta = float(self.jaw_opening_delta_m)
        yaw_delta = float(self.tcp_yaw_delta_rad)
        translation = np.array(self.tcp_translation_rim_m, dtype=float, copy=True)
        if not math.isfinite(jaw_delta) or not math.isfinite(yaw_delta):
            raise ValueError("gate command deltas must be finite")
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("tcp_translation_rim_m must contain three finite values")
        if float(np.linalg.norm(translation)) > MAX_PROBE_TCP_STEP_M + 1e-12:
            raise ValueError("gate TCP translation exceeds the reviewed cap")
        if self.lift_allowed != (
            self.state is CupTactileGateState.LIFT_ALLOWED
            and self.action is CupTactileGateAction.ALLOW_LIFT
        ):
            raise ValueError("lift_allowed is only valid for the ALLOW_LIFT decision")
        if not isinstance(self.satisfied_predicates, frozenset) or any(
            not isinstance(value, str) or not value
            for value in self.satisfied_predicates
        ):
            raise ValueError("satisfied_predicates must be a frozenset of names")
        translation.setflags(write=False)
        object.__setattr__(self, "jaw_opening_delta_m", jaw_delta)
        object.__setattr__(self, "tcp_yaw_delta_rad", yaw_delta)
        object.__setattr__(self, "tcp_translation_rim_m", translation)


_ROLE_MISMATCH_STATES = frozenset(
    {
        RimProbeState.LEFT_ONLY,
        RimProbeState.RIGHT_ONLY,
        RimProbeState.ROLE_MISMATCH,
    }
)


def classify_cup_rim_observation(
    evaluation: RimProbeEvaluation,
) -> CupRimObservation:
    """Map a settled tactile evaluation into the cup task's five outcomes."""

    if not isinstance(evaluation, RimProbeEvaluation):
        raise TypeError("evaluation must be a RimProbeEvaluation")
    if evaluation.state is RimProbeState.NO_CONTACT:
        return CupRimObservation.NO_CONTACT
    if "multi_rim_or_too_deep_hypothesis" in evaluation.predicates:
        return CupRimObservation.MULTI_RIM_OR_TOO_DEEP
    if evaluation.state in _ROLE_MISMATCH_STATES:
        return CupRimObservation.ROLE_MISMATCH
    if (
        evaluation.state is RimProbeState.CAPTURED
        and evaluation.recommendation.action is ProbeAction.ACCEPT
        and "rim_capture_verified" in evaluation.predicates
        and "inner_rim_match" in evaluation.predicates
        and "outer_rim_match" in evaluation.predicates
    ):
        return CupRimObservation.INNER_OUTER_RIM_MATCH
    return CupRimObservation.UNKNOWN


class CupTactileLiftGate:
    """Stateful pre-lift gate with bounded probe and one-shot adjustments."""

    _CAPTURE_PREDICATES = frozenset(
        {
            "sensor_healthy",
            "bilateral_contact",
            "inner_rim_match",
            "outer_rim_match",
            "normal_load_in_fragile_band",
            "cup_deformation_budget_ok",
            "rim_capture_verified",
        }
    )

    def __init__(
        self,
        config: CupTactileGateConfig = CupTactileGateConfig(),
        *,
        consumed_authorization_ids: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(config, CupTactileGateConfig):
            raise TypeError("config must be a CupTactileGateConfig")
        self.config = config
        self.state = CupTactileGateState.AWAIT_TACTILE
        self._ledger = TcpAdjustmentAuthorizationLedger(consumed_authorization_ids)
        self._pending: RimProbeEvaluation | None = None
        self._pending_observation: CupRimObservation | None = None
        self._motion_action: CupTactileGateAction | None = None
        self._last_evidence_key: tuple[float, ...] | None = None
        self._profile_id: str | None = None
        self._calibration_ids: tuple[str, str] | None = None
        self._contact_roles: tuple[RimContactRole, RimContactRole] | None = None
        self._expected_probe_context_id: str | None = None
        self._probe_down_steps = 0
        self._adjustment_cycles = 0

    @property
    def lift_allowed(self) -> bool:
        return self.state is CupTactileGateState.LIFT_ALLOWED

    def _decision(
        self,
        action: CupTactileGateAction,
        reason: str,
        *,
        observation: CupRimObservation | None = None,
        jaw_delta: float = 0.0,
        translation: np.ndarray | None = None,
        yaw_delta: float = 0.0,
        evaluation: RimProbeEvaluation | None = None,
    ) -> CupTactileGateDecision:
        return CupTactileGateDecision(
            state=self.state,
            action=action,
            observation=observation,
            reason=reason,
            lift_allowed=self.lift_allowed,
            jaw_opening_delta_m=jaw_delta,
            tcp_translation_rim_m=(
                np.zeros(3, dtype=float) if translation is None else translation
            ),
            tcp_yaw_delta_rad=yaw_delta,
            probe_context_id=(
                self._expected_probe_context_id
                if evaluation is None
                else evaluation.probe_context_id
            ),
            source_evidence_key=(
                None if evaluation is None else evaluation.evidence_key
            ),
            satisfied_predicates=(
                frozenset(
                    {
                        "rim_capture_verified",
                        "inner_rim_match",
                        "outer_rim_match",
                        "tactile_lift_permission",
                    }
                )
                if self.lift_allowed
                else frozenset()
            ),
        )

    def _stop(
        self, reason: str, evaluation: RimProbeEvaluation | None = None
    ) -> CupTactileGateDecision:
        self.state = CupTactileGateState.STOPPED
        self._pending = None
        self._pending_observation = None
        self._motion_action = None
        return self._decision(
            CupTactileGateAction.STOP,
            reason,
            observation=CupRimObservation.UNKNOWN,
            evaluation=evaluation,
        )

    def _accept_provenance(self, evaluation: RimProbeEvaluation) -> str | None:
        if (
            not evaluation.probe_context_id
            or not evaluation.profile_id
            or evaluation.calibration_ids is None
            or len(evaluation.calibration_ids) != 2
            or not all(evaluation.calibration_ids)
            or evaluation.contact_roles is None
            or evaluation.evidence_key is None
        ):
            return (
                "tactile evidence lacks context, complementary inner/outer roles, "
                "profile, calibration, or timestamps"
            )
        try:
            contact_roles = tuple(
                RimContactRole(value) for value in evaluation.contact_roles
            )
        except (TypeError, ValueError):
            return "tactile evidence has an invalid inner/outer role assignment"
        if len(contact_roles) != 2 or set(contact_roles) != {
            RimContactRole.INNER,
            RimContactRole.OUTER,
        }:
            return "tactile evidence has an invalid inner/outer role assignment"
        try:
            evidence = np.asarray(evaluation.evidence_key, dtype=float)
        except (TypeError, ValueError):
            return "tactile evidence timestamps are malformed"
        if (
            evidence.ndim != 1
            or evidence.size == 0
            or evidence.size % 2 != 0
            or not np.isfinite(evidence).all()
        ):
            return "tactile evidence timestamps are malformed"
        if self._profile_id is None:
            self._profile_id = evaluation.profile_id
            self._calibration_ids = evaluation.calibration_ids
            self._contact_roles = contact_roles
        elif (
            evaluation.profile_id != self._profile_id
            or evaluation.calibration_ids != self._calibration_ids
            or contact_roles != self._contact_roles
        ):
            return (
                "tactile profile, role assignment, or calibration identity changed "
                "during the gate"
            )
        if (
            self._expected_probe_context_id is not None
            and evaluation.probe_context_id != self._expected_probe_context_id
        ):
            return "probe context does not match the completed motion"
        if self._last_evidence_key is not None:
            previous = np.asarray(self._last_evidence_key, dtype=float)
            if evidence.shape != previous.shape or not np.all(evidence > previous):
                return "every tactile SDK and host timestamp must advance"
        self._last_evidence_key = tuple(float(value) for value in evidence)
        self._expected_probe_context_id = evaluation.probe_context_id
        return None

    @staticmethod
    def _side_role_predicates(evaluation: RimProbeEvaluation) -> frozenset[str]:
        assert evaluation.contact_roles is not None
        roles = tuple(RimContactRole(value) for value in evaluation.contact_roles)
        return frozenset(
            {
                f"left_{roles[0].value}_rim_match",
                f"right_{roles[1].value}_rim_match",
            }
        )

    def observe(
        self, evaluation: RimProbeEvaluation
    ) -> CupTactileGateDecision:
        """Consume one post-stability tactile evaluation at a stationary pose."""

        if self.state is not CupTactileGateState.AWAIT_TACTILE:
            raise RuntimeError(f"cannot observe tactile evidence while {self.state.value}")
        if not isinstance(evaluation, RimProbeEvaluation):
            raise TypeError("evaluation must be a RimProbeEvaluation")
        provenance_error = self._accept_provenance(evaluation)
        if provenance_error is not None:
            return self._stop(provenance_error, evaluation)

        if evaluation.state is RimProbeState.CAPTURE_CANDIDATE:
            if evaluation.recommendation.action is not ProbeAction.WAIT:
                return self._stop("capture candidate requested an unsafe action", evaluation)
            return self._decision(
                CupTactileGateAction.REQUEST_FRESH_TACTILE,
                "inner/outer role templates match, but stability evidence is incomplete",
                evaluation=evaluation,
            )

        if evaluation.state is RimProbeState.ROLE_MATCH_LOW_LOAD:
            recommendation = evaluation.recommendation
            if (
                recommendation.action is not ProbeAction.CLOSE_STEP
                or recommendation.jaw_opening_delta_m >= 0.0
            ):
                return self._stop("low-load contact lacks a bounded close step", evaluation)
            self.state = CupTactileGateState.AWAIT_MOTION_COMPLETION
            self._motion_action = CupTactileGateAction.GENTLE_CLOSE
            return self._decision(
                CupTactileGateAction.GENTLE_CLOSE,
                "the inner and outer role templates match independently; apply one "
                "bounded low-force close step",
                jaw_delta=recommendation.jaw_opening_delta_m,
                evaluation=evaluation,
            )

        observation = classify_cup_rim_observation(evaluation)
        if observation is CupRimObservation.NO_CONTACT:
            if self._probe_down_steps >= self.config.max_probe_down_steps:
                return self._stop("probe-down step budget is exhausted", evaluation)
            self._probe_down_steps += 1
            self.state = CupTactileGateState.AWAIT_MOTION_COMPLETION
            self._motion_action = CupTactileGateAction.PROBE_DOWN
            return self._decision(
                CupTactileGateAction.PROBE_DOWN,
                "no tactile contact; probe one bounded step along rim -Z",
                observation=observation,
                translation=np.array([0.0, 0.0, -self.config.probe_down_step_m]),
                evaluation=evaluation,
            )

        if observation is CupRimObservation.INNER_OUTER_RIM_MATCH:
            if (
                not self._CAPTURE_PREDICATES.issubset(evaluation.predicates)
                or not self._side_role_predicates(evaluation).issubset(
                    evaluation.predicates
                )
                or evaluation.consecutive_good_windows
                < self.config.required_stable_windows
            ):
                return self._stop(
                    "inner/outer role-aware lift evidence is internally incomplete",
                    evaluation,
                )
            self.state = CupTactileGateState.LIFT_ALLOWED
            return self._decision(
                CupTactileGateAction.ALLOW_LIFT,
                "stable inner- and outer-rim role matches are the sole pre-lift gate",
                observation=observation,
                evaluation=evaluation,
            )

        if observation in {
            CupRimObservation.MULTI_RIM_OR_TOO_DEEP,
            CupRimObservation.ROLE_MISMATCH,
        }:
            recommendation = evaluation.recommendation
            translation = recommendation.proposed_tcp_translation_task_m
            has_lateral_or_yaw = bool(
                np.linalg.norm(translation[:2]) > 1e-12
                or abs(recommendation.proposed_tcp_yaw_delta_rad) > 1e-12
            )
            valid_adjustment = bool(
                recommendation.action is ProbeAction.RELEASE_THEN_ADJUST
                and recommendation.release_confirmation_required
                and recommendation.jaw_opening_delta_m > 0.0
            )
            if observation is CupRimObservation.MULTI_RIM_OR_TOO_DEEP:
                valid_adjustment = bool(
                    valid_adjustment
                    and translation[2] > 0.0
                    and np.linalg.norm(translation[:2]) <= 1e-12
                    and abs(recommendation.proposed_tcp_yaw_delta_rad) <= 1e-12
                )
            else:
                valid_adjustment = valid_adjustment and has_lateral_or_yaw
            if not valid_adjustment:
                return self._stop(
                    "tactile class has no calibrated unload-then-adjust recovery",
                    evaluation,
                )
            if self._adjustment_cycles >= self.config.max_adjustment_cycles:
                return self._stop("tactile adjustment budget is exhausted", evaluation)
            self._adjustment_cycles += 1
            self._pending = evaluation
            self._pending_observation = observation
            self.state = CupTactileGateState.AWAIT_RELEASE_AUTHORIZATION
            return self._decision(
                CupTactileGateAction.UNLOAD,
                "unload and prove fresh no-contact before any TCP correction",
                observation=observation,
                jaw_delta=recommendation.jaw_opening_delta_m,
                evaluation=evaluation,
            )

        return self._stop(
            f"unresolved tactile state {evaluation.state.value}; lift remains blocked",
            evaluation,
        )

    def apply_adjustment_authorization(
        self,
        authorization: TcpAdjustmentAuthorization,
        *,
        now_s: float,
    ) -> CupTactileGateDecision:
        """Consume one release-backed token and request its bounded TCP correction."""

        if self.state is not CupTactileGateState.AWAIT_RELEASE_AUTHORIZATION:
            raise RuntimeError(
                f"cannot apply an adjustment authorization while {self.state.value}"
            )
        pending = self._pending
        observation = self._pending_observation
        assert pending is not None and observation is not None
        if (
            pending.evidence_key is None
            or authorization.proposal_evidence_key != pending.evidence_key
            or authorization.probe_context_id != pending.probe_context_id
            or authorization.profile_id != pending.profile_id
            or authorization.calibration_ids != pending.calibration_ids
            or authorization.contact_roles != self._contact_roles
        ):
            return self._stop("adjustment authorization does not match the pending probe")
        adjustment = self._ledger.consume(
            authorization,
            now_s=now_s,
            probe_context_id=authorization.probe_context_id,
            profile_id=authorization.profile_id,
            calibration_ids=authorization.calibration_ids,
            contact_roles=authorization.contact_roles,
        )
        if adjustment is None:
            return self._stop("adjustment authorization is invalid, expired, or replayed")
        proposed = pending.recommendation
        if (
            not np.allclose(
                adjustment.proposed_tcp_translation_task_m,
                proposed.proposed_tcp_translation_task_m,
                atol=1e-12,
                rtol=0.0,
            )
            or not math.isclose(
                adjustment.proposed_tcp_yaw_delta_rad,
                proposed.proposed_tcp_yaw_delta_rad,
                abs_tol=1e-12,
                rel_tol=0.0,
            )
        ):
            return self._stop("authorization changed the calibrated correction")
        translation = adjustment.proposed_tcp_translation_task_m
        yaw = adjustment.proposed_tcp_yaw_delta_rad
        if observation is CupRimObservation.MULTI_RIM_OR_TOO_DEEP:
            if (
                translation[2] <= 0.0
                or np.linalg.norm(translation[:2]) > 1e-12
                or abs(yaw) > 1e-12
            ):
                return self._stop("too-deep recovery is not a pure rim +Z adjustment")
            action = CupTactileGateAction.RAISE_AND_REPROBE
            reason = "release confirmed; move one bounded step toward shallower rim +Z"
        else:
            if np.linalg.norm(translation[:2]) <= 1e-12 and abs(yaw) <= 1e-12:
                return self._stop(
                    "role-mismatch recovery lacks lateral or angular correction"
                )
            action = CupTactileGateAction.REALIGN_AND_REPROBE
            reason = (
                "release confirmed; apply the calibrated role-aware "
                "lateral/angular correction"
            )
        self.state = CupTactileGateState.AWAIT_MOTION_COMPLETION
        self._motion_action = action
        return self._decision(
            action,
            reason,
            observation=observation,
            translation=translation,
            yaw_delta=yaw,
            evaluation=pending,
        )

    def motion_completed(
        self,
        *,
        succeeded: bool,
        next_probe_context_id: str,
    ) -> CupTactileGateDecision:
        """Acknowledge a runner command before accepting another tactile window."""

        if self.state is not CupTactileGateState.AWAIT_MOTION_COMPLETION:
            raise RuntimeError(f"no gate motion is pending while {self.state.value}")
        if not isinstance(succeeded, bool):
            raise ValueError("succeeded must be a bool")
        if not succeeded:
            return self._stop("the requested tactile-gate motion did not complete")
        if not isinstance(next_probe_context_id, str) or not next_probe_context_id.strip():
            return self._stop("completed motion lacks a next probe_context_id")
        action = self._motion_action
        assert action is not None
        tcp_moved = action in {
            CupTactileGateAction.PROBE_DOWN,
            CupTactileGateAction.RAISE_AND_REPROBE,
            CupTactileGateAction.REALIGN_AND_REPROBE,
        }
        if tcp_moved and next_probe_context_id == self._expected_probe_context_id:
            return self._stop("TCP motion must create a new probe_context_id")
        if not tcp_moved and next_probe_context_id != self._expected_probe_context_id:
            return self._stop("jaw-only motion must keep the fixed TCP probe context")
        self._expected_probe_context_id = next_probe_context_id
        self._pending = None
        self._pending_observation = None
        self._motion_action = None
        self.state = CupTactileGateState.AWAIT_TACTILE
        return self._decision(
            CupTactileGateAction.REQUEST_FRESH_TACTILE,
            "motion completed; evaluate only fresh tactile evidence at the settled pose",
        )


__all__ = [
    "CupRimObservation",
    "CupTactileGateAction",
    "CupTactileGateConfig",
    "CupTactileGateDecision",
    "CupTactileGateState",
    "CupTactileLiftGate",
    "classify_cup_rim_observation",
]
