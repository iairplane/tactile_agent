from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from demo2gripper.cup_tactile_gate import (
    CupRimObservation,
    CupTactileGateAction,
    CupTactileGateConfig,
    CupTactileGateState,
    CupTactileLiftGate,
    classify_cup_rim_observation,
)
from demo2gripper.tactile_rim_probe import (
    FingerRimMetrics,
    ProbeAction,
    ProbeAdjustment,
    RimContactRole,
    RimProbeEvaluation,
    RimProbeState,
    TcpAdjustmentAuthorization,
)

CONTEXT_1 = "cup-probe:episode-1:tcp-pose-1"
CONTEXT_2 = "cup-probe:episode-1:tcp-pose-2"
PROFILE = "nested-cup-test.v1"
CALIBRATIONS = ("left-cal", "right-cal")


def _metrics() -> FingerRimMetrics:
    return FingerRimMetrics(
        valid=True,
        contact=True,
        patch_area_px=100,
        significant_components=1,
        roi_fraction=1.0,
        center_uv_px=np.array([32.0, 32.0]),
        center_error_uv_px=np.zeros(2),
        tangent_error_deg=0.0,
        elongation=4.0,
        center_jitter_px=0.2,
        contact_frame_fraction=1.0,
        normal_load_n=0.2,
        normal_load_mad_n=0.01,
        normal_load_peak_n=0.22,
        normal_load_range_n=0.03,
        indentation_p95_m=None,
        geometry_score=0.95,
    )


def _evaluation(
    state: RimProbeState,
    *,
    action: ProbeAction,
    predicates: frozenset[str] = frozenset({"sensor_healthy"}),
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw: float = 0.0,
    jaw_delta: float = 0.0,
    release_required: bool = False,
    evidence_offset: float = 0.0,
    context: str = CONTEXT_1,
    consecutive: int = 0,
) -> RimProbeEvaluation:
    evidence = tuple(
        value + evidence_offset
        for value in (1.0, 9.90, 2.0, 9.91, 3.0, 9.92, 4.0, 9.93)
    )
    return RimProbeEvaluation(
        state=state,
        confidence=0.9,
        left=_metrics(),
        right=_metrics(),
        predicates=predicates,
        reasons=("test evidence",),
        recommendation=ProbeAdjustment(
            action=action,
            jaw_opening_delta_m=jaw_delta,
            proposed_tcp_translation_task_m=np.asarray(translation, dtype=float),
            proposed_tcp_yaw_delta_rad=yaw,
            release_confirmation_required=release_required,
        ),
        evidence_key=evidence,
        consecutive_good_windows=consecutive,
        probe_context_id=context,
        profile_id=PROFILE,
        calibration_ids=CALIBRATIONS,
        contact_roles=(RimContactRole.INNER, RimContactRole.OUTER),
        evaluated_at_monotonic_s=10.0 + evidence_offset,
        adjustment_authorization_ttl_s=1.0,
    )


def _authorization(
    proposal: RimProbeEvaluation,
    *,
    translation: tuple[float, float, float],
    yaw: float = 0.0,
    authorization_id: str = "auth-001",
) -> TcpAdjustmentAuthorization:
    assert proposal.evidence_key is not None
    return TcpAdjustmentAuthorization(
        authorization_id=authorization_id,
        probe_context_id=proposal.probe_context_id or "",
        profile_id=proposal.profile_id or "",
        calibration_ids=proposal.calibration_ids or ("", ""),
        contact_roles=proposal.contact_roles
        or (RimContactRole.INNER, RimContactRole.OUTER),
        issued_at_monotonic_s=10.20,
        expires_at_monotonic_s=10.90,
        release_command_id="release-001",
        proposal_evidence_key=proposal.evidence_key,
        release_evidence_key=tuple(value + 1.0 for value in proposal.evidence_key),
        adjustment=ProbeAdjustment(
            action=ProbeAction.APPLY_TCP_ADJUSTMENT,
            jaw_opening_delta_m=0.0,
            proposed_tcp_translation_task_m=np.asarray(translation, dtype=float),
            proposed_tcp_yaw_delta_rad=yaw,
            release_confirmation_required=False,
        ),
    )


def test_no_contact_requests_bounded_down_probe_then_fresh_tactile() -> None:
    gate = CupTactileLiftGate(CupTactileGateConfig(probe_down_step_m=0.0004))
    evaluation = _evaluation(
        RimProbeState.NO_CONTACT,
        action=ProbeAction.CLOSE_STEP,
        jaw_delta=-0.0004,
    )

    decision = gate.observe(evaluation)

    assert decision.observation is CupRimObservation.NO_CONTACT
    assert decision.action is CupTactileGateAction.PROBE_DOWN
    assert decision.state is CupTactileGateState.AWAIT_MOTION_COMPLETION
    assert decision.lift_allowed is False
    np.testing.assert_allclose(decision.tcp_translation_rim_m, [0.0, 0.0, -0.0004])
    followup = gate.motion_completed(
        succeeded=True, next_probe_context_id=CONTEXT_2
    )
    assert followup.action is CupTactileGateAction.REQUEST_FRESH_TACTILE
    assert followup.state is CupTactileGateState.AWAIT_TACTILE


def test_multi_rim_unloads_then_requires_authorized_upward_reprobe() -> None:
    gate = CupTactileLiftGate()
    proposal = _evaluation(
        RimProbeState.CONTACT_UNRESOLVED,
        action=ProbeAction.RELEASE_THEN_ADJUST,
        predicates=frozenset(
            {
                "sensor_healthy",
                "bilateral_contact",
                "multi_rim_or_too_deep_hypothesis",
                "calibrated_shallower_recovery_available",
            }
        ),
        translation=(0.0, 0.0, 0.0005),
        jaw_delta=0.001,
        release_required=True,
    )

    unload = gate.observe(proposal)

    assert unload.observation is CupRimObservation.MULTI_RIM_OR_TOO_DEEP
    assert unload.action is CupTactileGateAction.UNLOAD
    assert unload.jaw_opening_delta_m == pytest.approx(0.001)
    assert gate.state is CupTactileGateState.AWAIT_RELEASE_AUTHORIZATION

    move = gate.apply_adjustment_authorization(
        _authorization(proposal, translation=(0.0, 0.0, 0.0005)),
        now_s=10.30,
    )
    assert move.action is CupTactileGateAction.RAISE_AND_REPROBE
    np.testing.assert_allclose(move.tcp_translation_rim_m, [0.0, 0.0, 0.0005])
    assert move.lift_allowed is False


def test_role_mismatch_uses_only_release_backed_lateral_or_yaw_adjustment() -> None:
    gate = CupTactileLiftGate()
    proposal = _evaluation(
        RimProbeState.ROLE_MISMATCH,
        action=ProbeAction.RELEASE_THEN_ADJUST,
        predicates=frozenset({"sensor_healthy", "bilateral_contact"}),
        translation=(0.0004, 0.0, 0.0),
        yaw=0.01,
        jaw_delta=0.001,
        release_required=True,
    )

    assert (
        classify_cup_rim_observation(proposal)
        is CupRimObservation.ROLE_MISMATCH
    )
    assert gate.observe(proposal).action is CupTactileGateAction.UNLOAD
    move = gate.apply_adjustment_authorization(
        _authorization(proposal, translation=(0.0004, 0.0, 0.0), yaw=0.01),
        now_s=10.30,
    )

    assert move.action is CupTactileGateAction.REALIGN_AND_REPROBE
    np.testing.assert_allclose(move.tcp_translation_rim_m, [0.0004, 0.0, 0.0])
    assert move.tcp_yaw_delta_rad == pytest.approx(0.01)


def test_only_stable_inner_and_outer_role_matches_allow_lift() -> None:
    gate = CupTactileLiftGate()
    candidate = _evaluation(
        RimProbeState.CAPTURE_CANDIDATE,
        action=ProbeAction.WAIT,
        predicates=frozenset(
            {
                "sensor_healthy",
                "bilateral_contact",
                "inner_rim_match",
                "outer_rim_match",
                "left_inner_rim_match",
                "right_outer_rim_match",
                "normal_load_in_fragile_band",
                "cup_deformation_budget_ok",
            }
        ),
        consecutive=2,
    )
    waiting = gate.observe(candidate)
    assert waiting.action is CupTactileGateAction.REQUEST_FRESH_TACTILE
    assert waiting.lift_allowed is False

    captured = _evaluation(
        RimProbeState.CAPTURED,
        action=ProbeAction.ACCEPT,
        predicates=candidate.predicates | {"rim_capture_verified"},
        consecutive=3,
        evidence_offset=1.0,
    )
    allowed = gate.observe(captured)

    assert (
        allowed.observation is CupRimObservation.INNER_OUTER_RIM_MATCH
    )
    assert allowed.action is CupTactileGateAction.ALLOW_LIFT
    assert allowed.state is CupTactileGateState.LIFT_ALLOWED
    assert allowed.lift_allowed is True
    assert allowed.satisfied_predicates == frozenset(
        {
            "rim_capture_verified",
            "inner_rim_match",
            "outer_rim_match",
            "tactile_lift_permission",
        }
    )


def test_unknown_or_uncalibrated_contact_stops_and_blocks_lift() -> None:
    unknown = _evaluation(
        RimProbeState.CONTACT_UNRESOLVED,
        action=ProbeAction.RELEASE,
        jaw_delta=0.001,
    )
    gate = CupTactileLiftGate()
    stopped = gate.observe(unknown)
    assert stopped.observation is CupRimObservation.UNKNOWN
    assert stopped.action is CupTactileGateAction.STOP
    assert stopped.state is CupTactileGateState.STOPPED
    assert gate.lift_allowed is False

    multi_without_mapping = _evaluation(
        RimProbeState.CONTACT_UNRESOLVED,
        action=ProbeAction.RELEASE,
        predicates=frozenset(
            {"sensor_healthy", "multi_rim_or_too_deep_hypothesis"}
        ),
        jaw_delta=0.001,
    )
    stopped = CupTactileLiftGate().observe(multi_without_mapping)
    assert stopped.action is CupTactileGateAction.STOP
    assert "no calibrated" in stopped.reason


def test_bilateral_contact_without_both_role_matches_never_allows_lift() -> None:
    captured = _evaluation(
        RimProbeState.CAPTURED,
        action=ProbeAction.ACCEPT,
        predicates=frozenset(
            {
                "sensor_healthy",
                "bilateral_contact",
                "normal_load_in_fragile_band",
                "cup_deformation_budget_ok",
                "rim_capture_verified",
            }
        ),
        consecutive=3,
    )

    stopped = CupTactileLiftGate().observe(captured)

    assert stopped.action is CupTactileGateAction.STOP
    assert stopped.lift_allowed is False


def test_role_assignment_cannot_change_within_one_gate_episode() -> None:
    predicates = frozenset(
        {
            "sensor_healthy",
            "bilateral_contact",
            "inner_rim_match",
            "outer_rim_match",
            "left_inner_rim_match",
            "right_outer_rim_match",
            "normal_load_in_fragile_band",
            "cup_deformation_budget_ok",
        }
    )
    gate = CupTactileLiftGate()
    first = _evaluation(
        RimProbeState.CAPTURE_CANDIDATE,
        action=ProbeAction.WAIT,
        predicates=predicates,
        consecutive=1,
    )
    assert gate.observe(first).action is CupTactileGateAction.REQUEST_FRESH_TACTILE

    swapped = replace(
        _evaluation(
            RimProbeState.CAPTURED,
            action=ProbeAction.ACCEPT,
            predicates=(
                predicates
                - {"left_inner_rim_match", "right_outer_rim_match"}
                | {"left_outer_rim_match", "right_inner_rim_match", "rim_capture_verified"}
            ),
            evidence_offset=1.0,
            consecutive=3,
        ),
        contact_roles=(RimContactRole.OUTER, RimContactRole.INNER),
    )
    stopped = gate.observe(swapped)
    assert stopped.action is CupTactileGateAction.STOP
    assert "role assignment" in stopped.reason


def test_replayed_or_wrong_context_evidence_fails_closed() -> None:
    gate = CupTactileLiftGate()
    first = _evaluation(
        RimProbeState.CAPTURE_CANDIDATE,
        action=ProbeAction.WAIT,
    )
    assert gate.observe(first).action is CupTactileGateAction.REQUEST_FRESH_TACTILE

    replayed = gate.observe(first)
    assert replayed.action is CupTactileGateAction.STOP
    assert "timestamp" in replayed.reason


def test_adjustment_authorization_is_one_shot_and_bound_to_pending_evidence() -> None:
    proposal = _evaluation(
        RimProbeState.LEFT_ONLY,
        action=ProbeAction.RELEASE_THEN_ADJUST,
        translation=(0.0005, 0.0, 0.0),
        jaw_delta=0.001,
        release_required=True,
    )
    authorization = _authorization(
        proposal,
        translation=(0.0005, 0.0, 0.0),
        authorization_id="already-consumed",
    )
    gate = CupTactileLiftGate(
        consumed_authorization_ids=(authorization.authorization_id,)
    )
    assert gate.observe(proposal).action is CupTactileGateAction.UNLOAD

    stopped = gate.apply_adjustment_authorization(authorization, now_s=10.30)
    assert stopped.action is CupTactileGateAction.STOP
    assert "replayed" in stopped.reason


def test_gate_config_caps_probe_step_and_attempt_budget() -> None:
    with pytest.raises(ValueError, match="reviewed TCP cap"):
        CupTactileGateConfig(probe_down_step_m=0.004)

    gate = CupTactileLiftGate(CupTactileGateConfig(max_probe_down_steps=1))
    first = _evaluation(
        RimProbeState.NO_CONTACT,
        action=ProbeAction.CLOSE_STEP,
        jaw_delta=-0.0004,
    )
    assert gate.observe(first).action is CupTactileGateAction.PROBE_DOWN
    gate.motion_completed(succeeded=True, next_probe_context_id=CONTEXT_2)
    second = _evaluation(
        RimProbeState.NO_CONTACT,
        action=ProbeAction.CLOSE_STEP,
        jaw_delta=-0.0004,
        evidence_offset=1.0,
        context=CONTEXT_2,
    )
    stopped = gate.observe(second)
    assert stopped.action is CupTactileGateAction.STOP
    assert "budget" in stopped.reason
