from __future__ import annotations

import pytest

from demo2gripper.cup_tactile_gate import CupRimObservation
from demo2gripper.tactile_closed_loop import (
    PendingRecovery,
    TactileClosedLoopAction,
    TactileClosedLoopConfig,
    TactileClosedLoopEvidence,
    TactileMotionAuthorization,
    TactileProbeState,
    decide_tactile_action,
)

CONTEXT_1 = "cup-probe:pose-1"
CONTEXT_2 = "cup-probe:pose-2"


def _authorization(
    *,
    proposal_sequence: int,
    release_sequence: int,
    action: TactileClosedLoopAction,
    z_delta_m: float = 0.0,
    lateral_delta_m: float = 0.0,
    context: str = CONTEXT_1,
    authorization_id: str = "authorization-1",
    persisted: bool = True,
) -> TactileMotionAuthorization:
    return TactileMotionAuthorization(
        authorization_id=authorization_id,
        probe_context_id=context,
        proposal_evidence_id=f"window-{proposal_sequence}",
        release_evidence_id=f"window-{release_sequence}",
        release_command_id=f"release-{release_sequence}",
        release_completed_at_monotonic_s=10.0 + release_sequence - 0.25,
        applied_jaw_opening_delta_m=0.001,
        issued_at_monotonic_s=10.0 + release_sequence + 0.1,
        expires_at_monotonic_s=10.0 + release_sequence + 1.0,
        action=action,
        z_delta_m=z_delta_m,
        lateral_delta_m=lateral_delta_m,
        persisted=persisted,
    )


def _evidence(
    observation: CupRimObservation,
    sequence: int,
    *,
    context: str = CONTEXT_1,
    **kwargs: object,
) -> TactileClosedLoopEvidence:
    return TactileClosedLoopEvidence(
        observation=observation,
        evidence_id=f"window-{sequence}",
        probe_context_id=context,
        captured_at_monotonic_s=10.0 + sequence,
        **kwargs,
    )


def test_closed_jaw_no_contact_unloads_before_authorized_down_probe() -> None:
    config = TactileClosedLoopConfig(
        close_step_m=0.0002,
        probe_down_step_m=0.0004,
        max_close_steps=1,
    )
    close = decide_tactile_action(
        TactileProbeState(),
        _evidence(CupRimObservation.NO_CONTACT, 1, can_close_further=True),
        config,
    )
    assert close.action is TactileClosedLoopAction.CLOSE_STEP
    assert close.jaw_delta_m == pytest.approx(-0.0002)

    unload = decide_tactile_action(
        close.next_state,
        _evidence(CupRimObservation.NO_CONTACT, 2, can_close_further=False),
        config,
    )
    assert unload.action is TactileClosedLoopAction.OPEN_RELIEF
    assert unload.next_state.pending_recovery is PendingRecovery.PROBE_DOWN

    down = decide_tactile_action(
        unload.next_state,
        _evidence(
            CupRimObservation.NO_CONTACT,
            3,
            motion_authorization=_authorization(
                proposal_sequence=2,
                release_sequence=3,
                action=TactileClosedLoopAction.MOVE_Z,
                z_delta_m=-0.0004,
            ),
        ),
        config,
        now_monotonic_s=13.2,
    )
    assert down.action is TactileClosedLoopAction.MOVE_Z
    assert down.z_delta_m == pytest.approx(-0.0004)
    assert down.authorization_id == "authorization-1"
    assert down.next_state.close_steps == 0
    assert down.next_state.require_new_probe_context is True

    close_at_new_height = decide_tactile_action(
        down.next_state,
        _evidence(
            CupRimObservation.NO_CONTACT,
            4,
            context=CONTEXT_2,
            can_close_further=True,
        ),
        config,
    )
    assert close_at_new_height.action is TactileClosedLoopAction.CLOSE_STEP


def test_tcp_move_without_persisted_authorization_stops() -> None:
    unload = decide_tactile_action(
        TactileProbeState(),
        _evidence(CupRimObservation.NO_CONTACT, 1),
    )
    stopped = decide_tactile_action(
        unload.next_state,
        _evidence(
            CupRimObservation.NO_CONTACT,
            2,
            motion_authorization=_authorization(
                proposal_sequence=1,
                release_sequence=2,
                action=TactileClosedLoopAction.MOVE_Z,
                z_delta_m=-0.0005,
                persisted=False,
            ),
        ),
        now_monotonic_s=12.2,
    )
    assert stopped.action is TactileClosedLoopAction.STOP
    assert "persisted" in stopped.reason


def test_too_deep_unloads_then_raises_with_context_bound_authorization() -> None:
    config = TactileClosedLoopConfig(open_relief_step_m=0.0007, raise_step_m=0.0003)
    unload = decide_tactile_action(
        TactileProbeState(),
        _evidence(CupRimObservation.MULTI_RIM_OR_TOO_DEEP, 1),
        config,
    )
    assert unload.action is TactileClosedLoopAction.OPEN_RELIEF
    assert unload.jaw_delta_m == pytest.approx(0.0007)
    assert unload.next_state.pending_recovery is PendingRecovery.RAISE

    raised = decide_tactile_action(
        unload.next_state,
        _evidence(
            CupRimObservation.NO_CONTACT,
            2,
            motion_authorization=_authorization(
                proposal_sequence=1,
                release_sequence=2,
                action=TactileClosedLoopAction.MOVE_Z,
                z_delta_m=0.0003,
            ),
        ),
        config,
        now_monotonic_s=12.2,
    )
    assert raised.action is TactileClosedLoopAction.MOVE_Z
    assert raised.z_delta_m == pytest.approx(0.0003)


def test_contact_remaining_during_unload_repeats_only_bounded_relief() -> None:
    config = TactileClosedLoopConfig(max_relief_steps=2)
    first = decide_tactile_action(
        TactileProbeState(),
        _evidence(CupRimObservation.MULTI_RIM_OR_TOO_DEEP, 1),
        config,
    )
    second = decide_tactile_action(
        first.next_state,
        _evidence(CupRimObservation.MULTI_RIM_OR_TOO_DEEP, 2),
        config,
    )
    assert second.action is TactileClosedLoopAction.OPEN_RELIEF
    stopped = decide_tactile_action(
        second.next_state,
        _evidence(CupRimObservation.MULTI_RIM_OR_TOO_DEEP, 3),
        config,
    )
    assert stopped.action is TactileClosedLoopAction.STOP


def test_role_mismatch_uses_only_calibrated_bounded_side_error() -> None:
    config = TactileClosedLoopConfig(lateral_gain=0.5, max_lateral_step_m=0.0006)
    unload = decide_tactile_action(
        TactileProbeState(),
        _evidence(
            CupRimObservation.ROLE_MISMATCH,
            1,
            side_bound_error_m=0.004,
            side_bound_error_calibrated=True,
        ),
        config,
    )
    assert unload.action is TactileClosedLoopAction.OPEN_RELIEF

    realign = decide_tactile_action(
        unload.next_state,
        _evidence(
            CupRimObservation.NO_CONTACT,
            2,
            motion_authorization=_authorization(
                proposal_sequence=1,
                release_sequence=2,
                action=TactileClosedLoopAction.MOVE_LATERAL,
                lateral_delta_m=-0.0006,
            ),
        ),
        config,
        now_monotonic_s=12.2,
    )
    assert realign.action is TactileClosedLoopAction.MOVE_LATERAL
    assert realign.lateral_delta_m == pytest.approx(-0.0006)


def test_uncalibrated_side_error_stops() -> None:
    stopped = decide_tactile_action(
        TactileProbeState(),
        _evidence(
            CupRimObservation.ROLE_MISMATCH,
            1,
            side_bound_error_m=0.001,
        ),
    )
    assert stopped.action is TactileClosedLoopAction.STOP
    assert "calibrated" in stopped.reason


def test_only_consecutive_fresh_role_matches_allow_lift() -> None:
    config = TactileClosedLoopConfig(required_stable_matches=3)
    state = TactileProbeState()
    for sequence in (1, 2):
        decision = decide_tactile_action(
            state,
            _evidence(CupRimObservation.INNER_OUTER_RIM_MATCH, sequence),
            config,
        )
        assert decision.action is TactileClosedLoopAction.VERIFY
        state = decision.next_state
    allowed = decide_tactile_action(
        state,
        _evidence(CupRimObservation.INNER_OUTER_RIM_MATCH, 3),
        config,
    )
    assert allowed.action is TactileClosedLoopAction.ALLOW_LIFT
    assert allowed.next_state.lift_allowed is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fresh": False},
        {"valid": False},
        {"overload": True},
    ],
)
def test_bad_or_overload_evidence_fails_closed(kwargs: dict[str, bool]) -> None:
    decision = decide_tactile_action(
        TactileProbeState(),
        _evidence(CupRimObservation.NO_CONTACT, 1, **kwargs),
    )
    assert decision.action is TactileClosedLoopAction.STOP
    assert decision.next_state.lift_allowed is False


def test_missing_unknown_replayed_and_exhausted_inputs_stop() -> None:
    assert (
        decide_tactile_action(TactileProbeState(), None).action
        is TactileClosedLoopAction.STOP
    )
    unknown = decide_tactile_action(
        TactileProbeState(),
        _evidence(CupRimObservation.UNKNOWN, 1),
    )
    assert unknown.action is TactileClosedLoopAction.STOP
    replayed = decide_tactile_action(
        TactileProbeState(
            last_evidence_id="window-1", last_capture_monotonic_s=11.0
        ),
        _evidence(CupRimObservation.NO_CONTACT, 1),
    )
    assert replayed.action is TactileClosedLoopAction.STOP
    exhausted = decide_tactile_action(
        TactileProbeState(iterations=1),
        _evidence(CupRimObservation.NO_CONTACT, 2),
        TactileClosedLoopConfig(max_iterations=1),
    )
    assert exhausted.action is TactileClosedLoopAction.STOP


def test_jaw_motion_keeps_context_and_tcp_motion_requires_new_context() -> None:
    close = decide_tactile_action(
        TactileProbeState(),
        _evidence(CupRimObservation.NO_CONTACT, 1, can_close_further=True),
    )
    wrong_context = decide_tactile_action(
        close.next_state,
        _evidence(CupRimObservation.NO_CONTACT, 2, context=CONTEXT_2),
    )
    assert wrong_context.action is TactileClosedLoopAction.STOP


def test_motion_configuration_cannot_exceed_reviewed_probe_caps() -> None:
    with pytest.raises(ValueError, match="reviewed cap"):
        TactileClosedLoopConfig(max_lateral_step_m=0.004)
