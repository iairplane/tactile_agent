from __future__ import annotations

from dataclasses import replace

import pytest

from demo2gripper.test_tube_predicates import (
    PREDICATE_TO_PRODUCER_MANIFEST,
    ContactStatus,
    GripperReceipt,
    InsertionProgressReceipt,
    JamAssessment,
    MotionEvent,
    MotionReceipt,
    PredicateBuilderConfig,
    PredicateReceiptBundle,
    ReceiptHeader,
    RgbAxisReceipt,
    SceneReceipt,
    SupportAfterReleaseReceipt,
    TactileAxisReceipt,
    TactileContactReceipt,
    TrackerStatus,
    predicate_producer_manifest,
)
from demo2gripper.test_tube_predicates import (
    TestTubePredicateBuilder as TubePredicateBuilder,
)
from demo2gripper.test_tube_predicates import (
    TestTubePredicateProvider as TubePredicateProvider,
)
from demo2gripper.test_tube_preflight import TubeState, decide_test_tube_transition


def _config(**updates: object) -> PredicateBuilderConfig:
    values = {
        "task_id": "tube-task",
        "session_id": "session-1",
        "provenance_id": "geometry-1",
        "contact_profile_id": "contact-1",
        "rgb_qualification_id": "rgb-1",
        "insertion_profile_id": "jam-1",
        "retained_width_min_m": 0.006,
        "retained_width_max_m": 0.020,
        "rgb_runtime_qualified": True,
        "jam_runtime_qualified": True,
    }
    values.update(updates)
    return PredicateBuilderConfig(**values)  # type: ignore[arg-type]


def _header(
    producer: str,
    *,
    timestamp: float = 10.0,
    receipt_id: str | None = None,
    **updates: object,
) -> ReceiptHeader:
    values = {
        "task_id": "tube-task",
        "session_id": "session-1",
        "provenance_id": "geometry-1",
        "receipt_id": receipt_id or f"{producer}-1",
        "producer_id": producer,
        "observed_at_s": timestamp,
        "valid": True,
    }
    values.update(updates)
    return ReceiptHeader(**values)  # type: ignore[arg-type]


def _tactile(
    status: ContactStatus = ContactStatus.PRESENT,
    *,
    bilateral: bool = False,
    shadow_slip: bool = False,
    **updates: object,
) -> TactileContactReceipt:
    values = {
        "header": _header("tactile"),
        "contact_profile_id": "contact-1",
        "status": status,
        "runtime_contact_qualified": True,
        "bilateral_contact": bilateral,
        "slip_shadow_candidate": shadow_slip,
    }
    values.update(updates)
    return TactileContactReceipt(**values)  # type: ignore[arg-type]


def _tactile_axis(
    *,
    target_confirmed: bool = False,
    slip_candidate: bool = False,
    slip_confirmed: bool = False,
    axis_qualified: bool = True,
    slip_qualified: bool = True,
    target_count: int = 4,
    slip_count: int = 2,
    **updates: object,
) -> TactileAxisReceipt:
    values = {
        "profile_id": "tactile-controller-1",
        "session_id": "session-1",
        "provenance_id": "geometry-1",
        "source_evidence_id": "tactile-evaluation-1",
        "evaluation_valid": True,
        "axis_tcp": (1.0, 0.0, 0.0),
        "axis_base": (0.0, 1.0, 0.0),
        "target_error_deg": 2.0,
        "target_tolerance_deg": 5.0,
        "target_candidate": True,
        "target_confirmed": target_confirmed,
        "target_confirmation_count": target_count,
        "target_confirmation_required": 4,
        "runtime_axis_qualified": axis_qualified,
        "slip_candidate": slip_candidate,
        "slip_confirmed": slip_confirmed,
        "slip_confirmation_count": slip_count,
        "slip_confirmation_required": 2,
        "runtime_slip_qualified": slip_qualified,
    }
    values.update(updates)
    return TactileAxisReceipt(**values)  # type: ignore[arg-type]


def _gripper(
    width_m: float = 0.012,
    *,
    opened: bool = False,
    **updates: object,
) -> GripperReceipt:
    values = {
        "header": _header("gripper"),
        "measured_width_m": width_m,
        "command_settled": True,
        "fault_free": True,
        "open_confirmed": opened,
    }
    values.update(updates)
    return GripperReceipt(**values)  # type: ignore[arg-type]


def _rgb(
    *,
    target: bool = False,
    between: bool = False,
    confirmed: bool = False,
    **updates: object,
) -> RgbAxisReceipt:
    values = {
        "header": _header("rgb"),
        "qualification_id": "rgb-1",
        "tracker_status": TrackerStatus.TRACKED,
        "runtime_qualified": True,
        "target_angle_reached": target,
        "pivot_target_confirmed": confirmed,
        "tube_between_fingers": between,
    }
    values.update(updates)
    return RgbAxisReceipt(**values)  # type: ignore[arg-type]


def _build(state: TubeState, bundle: PredicateReceiptBundle, **config: object):
    return TubePredicateBuilder(_config(**config)).build(
        state=state,
        step_index=0,
        bundle=bundle,
    )


def test_scene_receipt_promotes_only_observe_predicates() -> None:
    scene = SceneReceipt(
        _header("scene"),
        tube_pose_valid=True,
        socket_frame_valid=True,
        tube_clear_of_source=True,
    )
    snapshot = _build(TubeState.OBSERVE, PredicateReceiptBundle(10.05, scene=scene))

    assert snapshot.predicates == ("socket_frame_valid", "tube_pose_valid")
    assert "tube_clear_of_source" not in snapshot.predicates
    assert set(snapshot.predicates) <= PREDICATE_TO_PRODUCER_MANIFEST.keys()


def test_optional_receipt_never_leaks_a_predicate_into_the_wrong_state() -> None:
    scene = SceneReceipt(
        _header("scene"),
        tube_pose_valid=True,
        socket_frame_valid=True,
    )
    snapshot = _build(
        TubeState.OBSERVE,
        PredicateReceiptBundle(10.05, scene=scene, tactile=_tactile()),
    )
    assert snapshot.predicates == ("socket_frame_valid", "tube_pose_valid")


def test_contact_alone_cannot_establish_grasp() -> None:
    snapshot = _build(
        TubeState.SIDE_GRASP,
        PredicateReceiptBundle(10.05, gripper=_gripper(), tactile=_tactile()),
    )

    assert snapshot.predicates == ("contact_present",)
    decision = decide_test_tube_transition(TubeState.SIDE_GRASP, snapshot.predicates)
    assert decision.event == "CONTACT_PRESENT_GRASP_UNCONFIRMED"


def test_grasp_requires_contact_retained_width_and_bilateral_or_visual_geometry() -> None:
    bilateral = _build(
        TubeState.SIDE_GRASP,
        PredicateReceiptBundle(
            10.05,
            gripper=_gripper(),
            tactile=_tactile(bilateral=True),
        ),
    )
    visual = _build(
        TubeState.SIDE_GRASP,
        PredicateReceiptBundle(
            10.05,
            gripper=_gripper(),
            tactile=_tactile(),
            rgb_axis=_rgb(between=True),
        ),
    )

    assert "grasp_established" in bilateral.predicates
    assert "grasp_established" in visual.predicates
    assert (
        decide_test_tube_transition(TubeState.SIDE_GRASP, bilateral.predicates).next_state
        is TubeState.EXTRACT
    )


@pytest.mark.parametrize(
    "gripper",
    [
        _gripper(0.003),
        _gripper(0.025),
        _gripper(command_settled=False),
        _gripper(fault_free=False),
    ],
)
def test_invalid_retained_width_blocks_grasp(gripper: GripperReceipt) -> None:
    snapshot = _build(
        TubeState.SIDE_GRASP,
        PredicateReceiptBundle(
            10.05,
            gripper=gripper,
            tactile=_tactile(bilateral=True),
        ),
    )
    assert "grasp_established" not in snapshot.predicates


def test_unqualified_rgb_never_promotes_visual_grasp_or_angle() -> None:
    unqualified = _rgb(runtime_qualified=False, target=True, between=True)
    grasp = _build(
        TubeState.SIDE_GRASP,
        PredicateReceiptBundle(
            10.05,
            gripper=_gripper(),
            tactile=_tactile(),
            rgb_axis=unqualified,
        ),
    )
    pivot = _build(
        TubeState.PIVOT_EVALUATE,
        PredicateReceiptBundle(10.05, tactile=_tactile(), rgb_axis=unqualified),
    )

    assert "grasp_established" not in grasp.predicates
    assert pivot.predicates == ("invalid_evidence",)
    assert "front_rgb_calibrated_angle_reached" not in pivot.predicates


def test_rgb_hold_status_cannot_authorize_another_pivot_step() -> None:
    snapshot = _build(
        TubeState.PIVOT_EVALUATE,
        PredicateReceiptBundle(
            10.05,
            tactile=_tactile(),
            rgb_axis=_rgb(target=True, tracker_status=TrackerStatus.HOLD),
        ),
    )
    assert snapshot.predicates == ("invalid_evidence",)


def test_shadow_slip_is_logged_but_never_promoted() -> None:
    snapshot = _build(
        TubeState.EXTRACT,
        PredicateReceiptBundle(
            10.05,
            scene=SceneReceipt(_header("scene"), tube_clear_of_source=True),
            tactile=_tactile(shadow_slip=True),
        ),
    )
    assert set(snapshot.predicates) == {"contact_present", "tube_clear_of_source"}
    assert "slip_detected" not in snapshot.predicates
    assert snapshot.evidence_receipt["shadow_observations"]["slip_shadow_candidate"] is True


def test_level_state_requires_held_geometry_motion_and_contact_receipts() -> None:
    snapshot = _build(
        TubeState.LEVEL_HELD_TUBE,
        PredicateReceiptBundle(
            10.05,
            scene=SceneReceipt(_header("scene"), tube_horizontal=True),
            motion=MotionReceipt(
                _header("motion"),
                MotionEvent.LEVEL_TARGET_REACHED,
                command_terminal=True,
                measured_target_within_tolerance=True,
                success=True,
            ),
            tactile=_tactile(),
        ),
    )

    assert set(snapshot.predicates) == {
        "contact_present",
        "tube_horizontal",
        "level_target_reached",
    }
    assert (
        decide_test_tube_transition(
            TubeState.LEVEL_HELD_TUBE,
            snapshot.predicates,
        ).next_state
        is TubeState.PIVOT_MICRO_LOOSEN
    )


def test_qualified_debounced_tactile_axis_can_replace_rgb_for_pivot() -> None:
    tactile = _tactile(
        axis_evaluation=_tactile_axis(target_confirmed=True),
    )
    snapshot = _build(
        TubeState.PIVOT_EVALUATE,
        PredicateReceiptBundle(10.05, tactile=tactile),
        rgb_runtime_qualified=False,
        tactile_profile_id="tactile-controller-1",
        tactile_axis_runtime_qualified=True,
    )

    assert set(snapshot.predicates) == {
        "contact_present",
        "tactile_relative_angle_reached",
    }
    assert (
        decide_test_tube_transition(
            TubeState.PIVOT_EVALUATE,
            (*snapshot.predicates, "pivot_budget_remaining"),
        ).next_state
        is TubeState.RECLAMP
    )


def test_tactile_axis_needs_builder_profile_qualification_and_confirmation() -> None:
    tactile = _tactile(axis_evaluation=_tactile_axis(target_confirmed=True))
    unqualified = _build(
        TubeState.PIVOT_EVALUATE,
        PredicateReceiptBundle(10.05, tactile=tactile),
        rgb_runtime_qualified=False,
    )
    not_confirmed = _build(
        TubeState.PIVOT_EVALUATE,
        PredicateReceiptBundle(
            10.05,
            tactile=_tactile(axis_evaluation=_tactile_axis(target_confirmed=False)),
        ),
        rgb_runtime_qualified=False,
        tactile_profile_id="tactile-controller-1",
        tactile_axis_runtime_qualified=True,
    )

    assert unqualified.predicates == ("invalid_evidence",)
    assert not_confirmed.predicates == ("contact_present",)


def test_only_qualified_debounced_slip_is_promoted() -> None:
    scene = SceneReceipt(_header("scene"), tube_clear_of_source=True)
    qualified = _build(
        TubeState.EXTRACT,
        PredicateReceiptBundle(
            10.05,
            scene=scene,
            tactile=_tactile(
                axis_evaluation=_tactile_axis(
                    slip_candidate=True,
                    slip_confirmed=True,
                )
            ),
        ),
        tactile_profile_id="tactile-controller-1",
        slip_runtime_qualified=True,
    )
    shadow = _build(
        TubeState.EXTRACT,
        PredicateReceiptBundle(
            10.05,
            scene=scene,
            tactile=_tactile(
                header=_header("tactile", receipt_id="tactile-shadow"),
                axis_evaluation=_tactile_axis(
                    slip_candidate=True,
                    slip_confirmed=False,
                    slip_qualified=False,
                ),
            ),
        ),
    )

    assert "slip_detected" in qualified.predicates
    assert "slip_detected" not in shadow.predicates
    assert shadow.evidence_receipt["shadow_observations"]["slip_shadow_candidate"] is True


def test_confirmed_tactile_receipt_rejects_insufficient_window_count() -> None:
    with pytest.raises(ValueError, match="consecutively confirmed axis"):
        _tactile_axis(target_confirmed=True, target_count=3)


def _insertion(**updates: object) -> InsertionProgressReceipt:
    values = {
        "header": _header("insertion"),
        "profile_id": "jam-1",
        "measured_depth_m": 0.002,
        "target_depth_m": 0.010,
        "depth_tolerance_m": 0.001,
        "axial_progress_m": 0.002,
        "minimum_progress_m": 0.001,
        "jam_assessment": JamAssessment.CLEAR,
        "runtime_jam_qualified": True,
        "jam_shadow_candidate": False,
    }
    values.update(updates)
    return InsertionProgressReceipt(**values)  # type: ignore[arg-type]


def test_shadow_only_jam_never_promotes_jam_or_no_jam() -> None:
    snapshot = _build(
        TubeState.INSERT_EVALUATE,
        PredicateReceiptBundle(
            10.05,
            insertion=_insertion(
                jam_assessment=JamAssessment.JAM,
                runtime_jam_qualified=False,
                jam_shadow_candidate=True,
            ),
        ),
        jam_runtime_qualified=False,
    )
    assert snapshot.predicates == ("invalid_evidence",)
    assert "insertion_jam" not in snapshot.predicates
    assert snapshot.evidence_receipt["shadow_observations"]["jam_shadow_candidate"] is True


def test_qualified_insertion_receipt_derives_depth_and_progress_predicates() -> None:
    progressing = _build(
        TubeState.INSERT_EVALUATE,
        PredicateReceiptBundle(10.05, insertion=_insertion()),
    )
    reached = _build(
        TubeState.INSERT_EVALUATE,
        PredicateReceiptBundle(
            10.05,
            insertion=_insertion(measured_depth_m=0.0095, axial_progress_m=0.0),
        ),
    )
    assert set(progressing.predicates) == {
        "axial_progress_ok",
        "insertion_depth_remaining",
        "no_jam",
    }
    assert set(reached.predicates) == {"insertion_depth_reached", "no_jam"}


def test_missing_state_specific_receipt_is_invalid() -> None:
    snapshot = _build(
        TubeState.ALIGN_SOCKET,
        PredicateReceiptBundle(10.05),
    )
    assert snapshot.predicates == ("invalid_evidence",)
    assert snapshot.evidence_receipt["rejection_reasons"] == [
        "missing_required_producers:scene"
    ]


def test_stale_or_cross_session_receipts_fail_closed() -> None:
    stale = SceneReceipt(_header("scene", timestamp=8.0), tube_pose_valid=True)
    stale_snapshot = _build(TubeState.OBSERVE, PredicateReceiptBundle(10.0, scene=stale))
    wrong_session = SceneReceipt(
        _header("scene", session_id="session-other"),
        tube_pose_valid=True,
    )
    mismatch_snapshot = _build(
        TubeState.OBSERVE,
        PredicateReceiptBundle(10.05, scene=wrong_session),
    )

    assert stale_snapshot.predicates == ("stale_evidence",)
    assert mismatch_snapshot.predicates == ("invalid_evidence",)


def test_support_requires_linked_visual_receipt_after_open_and_release() -> None:
    scene = SceneReceipt(_header("scene", timestamp=10.01, receipt_id="scene-release"))
    gripper = _gripper(
        opened=True,
        header=_header("gripper", timestamp=10.00, receipt_id="gripper-open"),
    )
    tactile = _tactile(
        ContactStatus.RELEASED_AFTER_OPEN,
        header=_header("tactile", timestamp=10.00, receipt_id="tactile-release"),
    )
    support = SupportAfterReleaseReceipt(
        header=_header("support", timestamp=10.04),
        verification_method="front_rgb",
        scene_receipt_id="scene-release",
        gripper_receipt_id="gripper-open",
        tactile_receipt_id="tactile-release",
        support_confirmed=True,
    )
    snapshot = _build(
        TubeState.RELEASE,
        PredicateReceiptBundle(
            10.05,
            scene=scene,
            gripper=gripper,
            tactile=tactile,
            support=support,
        ),
    )

    assert set(snapshot.predicates) == {
        "contact_released",
        "gripper_open",
        "tube_stays_supported",
    }

    bad_support = replace(support, scene_receipt_id="unlinked-scene")
    rejected = _build(
        TubeState.RELEASE,
        PredicateReceiptBundle(
            10.05,
            scene=scene,
            gripper=gripper,
            tactile=tactile,
            support=bad_support,
        ),
    )
    assert rejected.predicates == ("invalid_evidence",)


def test_motion_receipt_is_state_and_event_specific() -> None:
    motion = MotionReceipt(
        _header("motion"),
        event=MotionEvent.APPROACH_POSE_REACHED,
        command_terminal=True,
        measured_target_within_tolerance=True,
        success=True,
    )
    approach = _build(
        TubeState.APPROACH_TUBE,
        PredicateReceiptBundle(10.05, motion=motion),
    )
    wrong_state = _build(
        TubeState.RETREAT,
        PredicateReceiptBundle(10.05, motion=motion),
    )
    assert approach.predicates == ("approach_pose_reached",)
    assert wrong_state.predicates == ()


def test_provider_implements_runtime_protocol_and_requires_monotonic_bundles() -> None:
    bundle = PredicateReceiptBundle(
        10.05,
        scene=SceneReceipt(
            _header("scene"),
            tube_pose_valid=True,
            socket_frame_valid=True,
        ),
    )

    class Source:
        def collect(self, **_: object) -> PredicateReceiptBundle:
            return bundle

    provider = TubePredicateProvider(TubePredicateBuilder(_config()), Source())
    snapshot = provider.observe(state=TubeState.OBSERVE, step_index=0, budgets={})
    assert snapshot.state is TubeState.OBSERVE
    assert snapshot.sequence == 0
    with pytest.raises(ValueError, match="strictly increase"):
        provider.observe(state=TubeState.OBSERVE, step_index=1, budgets={})


def test_manifest_is_json_safe_and_labels_qualified_tactile_channels() -> None:
    manifest = predicate_producer_manifest()
    assert manifest["grasp_established"]["producer"].startswith("tactile+gripper")
    assert manifest["slip_detected"]["producer"] == "qualified_tactile_axis_evaluation"
    assert (
        manifest["tactile_relative_angle_reached"]["producer"]
        == "qualified_tactile_axis_evaluation"
    )
    assert set(PREDICATE_TO_PRODUCER_MANIFEST) == set(manifest)

