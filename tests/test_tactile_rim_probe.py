from dataclasses import replace

import numpy as np

from demo2gripper.tactile_rim_probe import (
    FingerRimTarget,
    FingerTactileWindow,
    ProbeAction,
    ReleaseCommandReceipt,
    RimContactRole,
    RimProbeConfig,
    RimProbeStabilityGate,
    RimProbeState,
    TcpAdjustmentAuthorizationLedger,
    TimedArrayWindow,
    authorize_tcp_adjustment,
    evaluate_rim_probe,
)

IMAGE_SHAPE = (64, 64)
PROBE_CONTEXT = "nested-cup-rim-test:episode-1:tcp-pose-1"


def make_target(
    calibration_id: str,
    *,
    contact_role: RimContactRole,
    toward_sensor_task: tuple[float, float, float],
    max_components: int = 1,
) -> FingerRimTarget:
    roi = np.zeros(IMAGE_SHAPE, dtype=bool)
    roi[25:40, 15:50] = True
    return FingerRimTarget(
        calibration_id=calibration_id,
        contact_role=contact_role,
        roi_mask=roi,
        target_center_uv_px=np.array([32.0, 32.0]),
        target_tangent_uv=np.array([1.0, 0.0]),
        diff_threshold=10.0,
        min_patch_area_px=40,
        max_patch_area_px=250,
        min_roi_fraction=0.95,
        max_center_error_px=3.0,
        max_tangent_error_deg=10.0,
        min_elongation=3.0,
        max_center_jitter_px=1.5,
        min_contact_frame_fraction=0.75,
        max_significant_components=max_components,
        inward_normal_sensor=np.array([1.0, 0.0, 0.0]),
        normal_load_band_n=(0.10, 0.50),
        max_normal_load_range_n=0.15,
        image_error_to_task_m_per_px=np.array(
            [[0.0001, 0.0], [0.0, 0.0], [0.0, 0.0001]]
        ),
        tangent_error_to_tcp_yaw=-1.0,
        toward_sensor_task=np.asarray(toward_sensor_task),
    )


def make_config(*, max_components: int = 1) -> RimProbeConfig:
    return RimProbeConfig(
        profile_id="nested-cup-rim-test.v1",
        left=make_target(
            "left-cal",
            contact_role=RimContactRole.INNER,
            toward_sensor_task=(1.0, 0.0, 0.0),
            max_components=max_components,
        ),
        right=make_target(
            "right-cal",
            contact_role=RimContactRole.OUTER,
            toward_sensor_task=(-1.0, 0.0, 0.0),
            max_components=max_components,
        ),
        min_unique_image_samples=4,
        min_unique_force_samples=3,
        max_age_s=0.20,
        max_pair_skew_s=0.05,
        close_step_m=0.0004,
        release_step_m=0.0010,
        recenter_step_m=0.0005,
        max_translation_step_m=0.0008,
        max_yaw_step_rad=np.deg2rad(1.0),
    )


def stripe(*, center_u: int = 32, center_v: int = 32) -> np.ndarray:
    image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    image[center_v - 2 : center_v + 3, center_u - 12 : center_u + 13] = 30
    return image


def two_stripes() -> np.ndarray:
    image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    image[25:28, 20:45] = 30
    image[36:39, 20:45] = 30
    return image


def fragmented_line() -> np.ndarray:
    image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    for start_u in (16, 27, 38, 49):
        image[31:33, start_u : start_u + 5] = 30
    return image


def timed_images(
    image: np.ndarray,
    *,
    sdk_offset: float = 0.0,
    capture_offset: float = 0.0,
    repeated_sdk_time: bool = False,
) -> TimedArrayWindow:
    count = 5
    sdk = np.full(count, 1.0 + sdk_offset) if repeated_sdk_time else sdk_offset + np.arange(count)
    capture = capture_offset + 0.0005 * sdk_offset + np.linspace(9.80, 9.98, count)
    return TimedArrayWindow(
        values=np.stack([image] * count),
        sdk_time_s=sdk,
        capture_monotonic_s=capture,
        valid=np.ones(count, dtype=bool),
    )


def timed_force(
    load_n: float | np.ndarray,
    *,
    sdk_offset: float = 0.0,
    capture_offset: float = 0.0,
) -> TimedArrayWindow:
    loads = np.asarray(load_n, dtype=float)
    if loads.ndim == 0:
        loads = np.full(4, float(loads))
    count = len(loads)
    force = np.zeros((count, 6), dtype=float)
    force[:, 0] = loads
    return TimedArrayWindow(
        values=force,
        sdk_time_s=sdk_offset + np.arange(count),
        capture_monotonic_s=(
            capture_offset + 0.0005 * sdk_offset + np.linspace(9.82, 9.98, count)
        ),
        valid=np.ones(count, dtype=bool),
    )


def timed_depth(*, sdk_offset: float = 0.0) -> TimedArrayWindow:
    count = 5
    return TimedArrayWindow(
        values=np.zeros((count, *IMAGE_SHAPE), dtype=float),
        sdk_time_s=sdk_offset + np.arange(count),
        capture_monotonic_s=0.0005 * sdk_offset + np.linspace(9.80, 9.98, count),
        valid=np.ones(count, dtype=bool),
    )


def finger(
    side: str,
    image: np.ndarray,
    load_n: float,
    *,
    sdk_offset: float = 0.0,
    capture_offset: float = 0.0,
    repeated_sdk_time: bool = False,
) -> FingerTactileWindow:
    return FingerTactileWindow(
        calibration_id=f"{side}-cal",
        diff=timed_images(
            image,
            sdk_offset=sdk_offset,
            capture_offset=capture_offset,
            repeated_sdk_time=repeated_sdk_time,
        ),
        force6d=timed_force(
            load_n, sdk_offset=sdk_offset, capture_offset=capture_offset
        ),
    )


def evaluate(
    left_image: np.ndarray,
    right_image: np.ndarray,
    *,
    left_load: float = 0.20,
    right_load: float = 0.20,
    sdk_offset: float = 0.0,
    capture_offset: float = 0.0,
    config: RimProbeConfig | None = None,
    now_s: float = 10.0,
    probe_context_id: str | None = PROBE_CONTEXT,
    not_before_s: float | None = None,
):
    return evaluate_rim_probe(
        finger(
            "left",
            left_image,
            left_load,
            sdk_offset=sdk_offset,
            capture_offset=capture_offset,
        ),
        finger(
            "right",
            right_image,
            right_load,
            sdk_offset=sdk_offset,
            capture_offset=capture_offset,
        ),
        config or make_config(),
        now_s=now_s,
        probe_context_id=probe_context_id,
        not_before_s=not_before_s,
    )


def test_independent_inner_outer_templates_accept_different_patch_areas() -> None:
    outer_patch = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    outer_patch[30:34, 22:43] = 30
    result = evaluate(stripe(), outer_patch)

    assert result.state is RimProbeState.CAPTURE_CANDIDATE
    assert result.recommendation.action is ProbeAction.WAIT
    assert "inner_rim_match" in result.predicates
    assert "outer_rim_match" in result.predicates
    assert "left_inner_rim_match" in result.predicates
    assert "right_outer_rim_match" in result.predicates
    assert result.left.patch_area_px == 125
    assert result.right.patch_area_px == 84
    assert result.left.patch_area_px != result.right.patch_area_px
    assert result.left.roi_fraction == 1.0
    assert result.left.elongation is not None and result.left.elongation > 3.0


def test_no_contact_requests_one_bounded_close_step() -> None:
    result = evaluate(np.zeros(IMAGE_SHAPE), np.zeros(IMAGE_SHAPE), left_load=0.0, right_load=0.0)

    assert result.state is RimProbeState.NO_CONTACT
    assert result.recommendation.action is ProbeAction.CLOSE_STEP
    assert result.recommendation.jaw_opening_delta_m == -make_config().close_step_m


def test_one_sided_contact_unloads_before_recentring() -> None:
    result = evaluate(stripe(), np.zeros(IMAGE_SHAPE), right_load=0.0)

    assert result.state is RimProbeState.LEFT_ONLY
    assert result.recommendation.action is ProbeAction.RELEASE_THEN_ADJUST
    assert result.recommendation.release_confirmation_required
    assert result.recommendation.jaw_opening_delta_m > 0.0
    np.testing.assert_allclose(
        result.recommendation.proposed_tcp_translation_task_m,
        [0.0005, 0.0, 0.0],
    )


def test_one_sided_image_with_force_on_both_fingers_is_not_recentered() -> None:
    result = evaluate(stripe(), np.zeros(IMAGE_SHAPE), right_load=0.20)

    assert result.state is RimProbeState.CONTACT_UNRESOLVED
    assert result.recommendation.action is ProbeAction.RELEASE
    assert "image and force disagree" in result.reasons[0]


def test_off_center_bilateral_contact_returns_clipped_calibrated_correction() -> None:
    result = evaluate(stripe(center_u=38), stripe(center_u=38))

    assert result.state is RimProbeState.ROLE_MISMATCH
    assert result.recommendation.action is ProbeAction.RELEASE_THEN_ADJUST
    assert result.recommendation.release_confirmation_required
    np.testing.assert_allclose(
        result.recommendation.proposed_tcp_translation_task_m,
        [-0.0006, 0.0, 0.0],
    )


def test_disagreeing_left_right_corrections_are_not_averaged_into_motion() -> None:
    base = make_config()
    config = replace(
        base,
        right=replace(
            base.right,
            image_error_to_task_m_per_px=-base.right.image_error_to_task_m_per_px,
        ),
        # The vectors have similar magnitude but point in opposite directions;
        # the direction-agreement guard must reject them independently of distance.
        max_translation_disagreement_m=0.002,
    )
    result = evaluate(stripe(center_u=38), stripe(center_u=38), config=config)

    assert result.state is RimProbeState.ROLE_MISMATCH
    assert result.recommendation.action is ProbeAction.RELEASE
    np.testing.assert_allclose(
        result.recommendation.proposed_tcp_translation_task_m,
        np.zeros(3),
    )


def test_separated_contact_bands_are_ambiguous_not_declared_two_cups() -> None:
    result = evaluate(two_stripes(), two_stripes())

    assert result.state is RimProbeState.CONTACT_UNRESOLVED
    assert result.recommendation.action is ProbeAction.RELEASE
    assert any("ambiguous separated bands" in reason for reason in result.reasons)


def test_task_prior_can_recover_from_multi_band_as_too_deep_hypothesis() -> None:
    config = replace(
        make_config(),
        too_deep_recovery_translation_task_m=np.array([0.0, 0.0, 0.0005]),
    )
    result = evaluate(two_stripes(), two_stripes(), config=config)

    assert result.state is RimProbeState.CONTACT_UNRESOLVED
    assert result.recommendation.action is ProbeAction.RELEASE_THEN_ADJUST
    assert result.recommendation.release_confirmation_required
    np.testing.assert_allclose(
        result.recommendation.proposed_tcp_translation_task_m,
        [0.0, 0.0, 0.0005],
    )
    assert "multi_rim_or_too_deep_hypothesis" in result.predicates
    assert "calibrated_shallower_recovery_available" in result.predicates
    assert any("too-deep hypothesis" in reason for reason in result.reasons)


def test_role_specific_outer_template_may_allow_two_bands() -> None:
    base = make_config()
    config = replace(
        base,
        right=replace(
            base.right,
            max_significant_components=2,
            min_elongation=1.0,
        ),
    )

    result = evaluate(stripe(), two_stripes(), config=config)

    assert result.state is RimProbeState.CAPTURE_CANDIDATE
    assert "inner_rim_match" in result.predicates
    assert "outer_rim_match" in result.predicates
    assert "multi_rim_or_too_deep_hypothesis" not in result.predicates


def test_inner_and_outer_images_are_checked_against_different_templates() -> None:
    base = make_config()
    config = replace(
        base,
        left=replace(
            base.left,
            target_center_uv_px=np.array([26.0, 32.0]),
            max_center_error_px=2.0,
        ),
        right=replace(
            base.right,
            target_center_uv_px=np.array([38.0, 32.0]),
            max_center_error_px=2.0,
        ),
    )
    accepted = evaluate(
        stripe(center_u=26), stripe(center_u=38), config=config
    )
    mismatched = evaluate(
        stripe(center_u=38), stripe(center_u=26), config=config
    )

    assert accepted.state is RimProbeState.CAPTURE_CANDIDATE
    assert mismatched.state is RimProbeState.ROLE_MISMATCH
    assert "inner_rim_match" not in mismatched.predicates
    assert "outer_rim_match" not in mismatched.predicates


def test_fragmented_union_cannot_pose_as_one_elongated_rim_patch() -> None:
    result = evaluate(fragmented_line(), fragmented_line())

    assert result.state is RimProbeState.CONTACT_UNRESOLVED
    assert result.recommendation.action is ProbeAction.RELEASE
    assert any("without a dominant patch" in reason for reason in result.reasons)


def test_force_overload_releases_even_when_image_geometry_is_good() -> None:
    result = evaluate(stripe(), stripe(), left_load=0.60)

    assert result.state is RimProbeState.OVERLOAD
    assert result.recommendation.action is ProbeAction.RELEASE
    assert result.recommendation.jaw_opening_delta_m > 0.0


def test_single_force_spike_cannot_be_hidden_by_window_median() -> None:
    left = FingerTactileWindow(
        calibration_id="left-cal",
        diff=timed_images(stripe()),
        force6d=timed_force(np.array([0.20, 0.20, 0.20, 2.0])),
    )
    result = evaluate_rim_probe(left, finger("right", stripe(), 0.20), make_config(), now_s=10.0)

    assert result.state is RimProbeState.OVERLOAD
    assert result.recommendation.action is ProbeAction.RELEASE


def test_force_chatter_and_negative_spike_request_release() -> None:
    for loads in (
        np.array([0.0, 0.40, 0.0, 0.40]),
        np.array([0.20, 0.20, 0.20, -2.0]),
    ):
        left = FingerTactileWindow(
            calibration_id="left-cal",
            diff=timed_images(stripe()),
            force6d=timed_force(loads),
        )
        result = evaluate_rim_probe(
            left,
            finger("right", stripe(), 0.20),
            make_config(),
            now_s=10.0,
        )

        assert result.state is RimProbeState.UNSTABLE
        assert result.recommendation.action is ProbeAction.RELEASE


def test_force_contact_without_image_patch_releases_instead_of_closing() -> None:
    result = evaluate(np.zeros(IMAGE_SHAPE), np.zeros(IMAGE_SHAPE))

    assert result.state is RimProbeState.CONTACT_UNRESOLVED
    assert result.recommendation.action is ProbeAction.RELEASE
    assert "force indicates contact" in result.reasons[0]


def test_unequal_role_loads_pass_without_a_symmetric_force_gate() -> None:
    result = evaluate(stripe(), stripe(), left_load=0.40, right_load=0.15)

    assert result.state is RimProbeState.CAPTURE_CANDIDATE
    assert "inner_rim_match" in result.predicates
    assert "outer_rim_match" in result.predicates


def test_optional_role_specific_force_ratio_is_an_independent_tolerance() -> None:
    config = replace(
        make_config(),
        inner_to_outer_normal_load_ratio_band=(1.5, 3.0),
    )
    accepted = evaluate(
        stripe(), stripe(), left_load=0.30, right_load=0.15, config=config
    )
    assert accepted.state is RimProbeState.CAPTURE_CANDIDATE
    assert "inner_outer_normal_load_ratio_valid" in accepted.predicates

    rejected = evaluate(
        stripe(), stripe(), left_load=0.15, right_load=0.30, config=config
    )
    assert rejected.state is RimProbeState.ROLE_LOAD_OUT_OF_TOLERANCE
    assert rejected.recommendation.action is ProbeAction.RELEASE


def test_cached_sdk_frames_fail_closed_instead_of_meaning_no_contact() -> None:
    left = finger("left", stripe(), 0.20, repeated_sdk_time=True)
    result = evaluate_rim_probe(left, finger("right", stripe(), 0.20), make_config(), now_s=10.0)

    assert result.state is RimProbeState.SENSOR_INVALID
    assert result.recommendation.action is ProbeAction.HOLD
    assert "unique SDK samples" in result.reasons[0]


def test_old_precommand_frames_are_removed_before_window_evaluation() -> None:
    capture_times = np.array([1.0, 2.0, 3.0, 9.96, 9.98])
    stale_mixed = TimedArrayWindow(
        values=np.stack([stripe()] * 5),
        sdk_time_s=np.arange(5, dtype=float),
        capture_monotonic_s=capture_times,
        valid=np.ones(5, dtype=bool),
    )
    left = FingerTactileWindow(
        calibration_id="left-cal",
        diff=stale_mixed,
        force6d=timed_force(0.20),
    )
    result = evaluate_rim_probe(left, finger("right", stripe(), 0.20), make_config(), now_s=10.0)

    assert result.state is RimProbeState.SENSOR_INVALID
    assert "fresh unique SDK samples" in result.reasons[0]


def test_command_completion_epoch_excludes_recent_precommand_frames() -> None:
    result = evaluate_rim_probe(
        finger("left", stripe(), 0.20),
        finger("right", stripe(), 0.20),
        make_config(),
        now_s=10.0,
        probe_context_id=PROBE_CONTEXT,
        not_before_s=9.90,
    )

    assert result.state is RimProbeState.SENSOR_INVALID
    assert "fresh unique SDK samples" in result.reasons[0]


def test_intermittent_image_contact_waits_instead_of_becoming_contact() -> None:
    intermittent = timed_images(stripe())
    values = intermittent.values.copy()
    values[[0, 2, 4]] = 0
    left = FingerTactileWindow(
        calibration_id="left-cal",
        diff=TimedArrayWindow(
            values=values,
            sdk_time_s=intermittent.sdk_time_s,
            capture_monotonic_s=intermittent.capture_monotonic_s,
            valid=intermittent.valid,
        ),
        force6d=timed_force(0.0),
    )
    right = FingerTactileWindow(
        calibration_id="right-cal",
        diff=TimedArrayWindow(
            values=values,
            sdk_time_s=intermittent.sdk_time_s,
            capture_monotonic_s=intermittent.capture_monotonic_s,
            valid=intermittent.valid,
        ),
        force6d=timed_force(0.0),
    )
    result = evaluate_rim_probe(left, right, make_config(), now_s=10.0)

    assert result.state is RimProbeState.UNSTABLE
    assert result.recommendation.action is ProbeAction.WAIT


def test_malformed_live_image_shape_returns_invalid_instead_of_raising() -> None:
    wrong_shape = timed_images(np.zeros((8, 8), dtype=np.uint8))
    left = FingerTactileWindow(
        calibration_id="left-cal",
        diff=wrong_shape,
        force6d=timed_force(0.20),
    )
    result = evaluate_rim_probe(left, finger("right", stripe(), 0.20), make_config(), now_s=10.0)

    assert result.state is RimProbeState.SENSOR_INVALID
    assert result.recommendation.action is ProbeAction.HOLD


def test_stability_gate_counts_only_distinct_good_windows() -> None:
    gate = RimProbeStabilityGate(required_good_windows=3)
    first = evaluate(stripe(), stripe(), sdk_offset=0.0)

    assert gate.update(first).consecutive_good_windows == 1
    duplicate = gate.update(first)
    assert duplicate.state is RimProbeState.CAPTURE_CANDIDATE
    assert duplicate.consecutive_good_windows == 1
    assert "not every required SDK/host evidence channel advanced" in duplicate.reasons[0]

    image_only_update = evaluate_rim_probe(
        FingerTactileWindow(
            calibration_id="left-cal",
            diff=timed_images(stripe(), sdk_offset=10.0),
            force6d=timed_force(0.20, sdk_offset=0.0),
        ),
        FingerTactileWindow(
            calibration_id="right-cal",
            diff=timed_images(stripe(), sdk_offset=10.0),
            force6d=timed_force(0.20, sdk_offset=0.0),
        ),
        make_config(),
        now_s=10.0,
        probe_context_id=PROBE_CONTEXT,
    )
    assert gate.update(image_only_update).consecutive_good_windows == 1

    second = gate.update(evaluate(stripe(), stripe(), sdk_offset=10.0))
    assert second.consecutive_good_windows == 2
    accepted = gate.update(evaluate(stripe(), stripe(), sdk_offset=20.0))
    assert accepted.state is RimProbeState.CAPTURED
    assert accepted.recommendation.action is ProbeAction.ACCEPT
    assert "rim_capture_verified" in accepted.predicates


def test_stability_gate_requires_depth_to_advance_when_depth_is_required() -> None:
    base = make_config()
    config = replace(
        base,
        left=replace(base.left, max_indentation_m=0.001),
        right=replace(base.right, max_indentation_m=0.001),
    )

    def depth_evaluation(main_offset: float, depth_offset: float):
        left = replace(
            finger("left", stripe(), 0.20, sdk_offset=main_offset),
            depth=timed_depth(sdk_offset=depth_offset),
        )
        right = replace(
            finger("right", stripe(), 0.20, sdk_offset=main_offset),
            depth=timed_depth(sdk_offset=depth_offset),
        )
        return evaluate_rim_probe(
            left,
            right,
            config,
            now_s=10.0,
            probe_context_id=PROBE_CONTEXT,
        )

    gate = RimProbeStabilityGate(required_good_windows=3)
    assert gate.update(depth_evaluation(0.0, 0.0)).consecutive_good_windows == 1
    depth_replay = gate.update(depth_evaluation(10.0, 0.0))
    assert depth_replay.consecutive_good_windows == 1
    assert "not every required SDK/host evidence channel advanced" in depth_replay.reasons[0]


def test_stability_gate_never_combines_different_or_missing_probe_contexts() -> None:
    gate = RimProbeStabilityGate(required_good_windows=2)
    first = evaluate(stripe(), stripe(), sdk_offset=0.0)
    assert gate.update(first).consecutive_good_windows == 1

    changed = evaluate(
        stripe(),
        stripe(),
        sdk_offset=10.0,
        probe_context_id="nested-cup-rim-test:episode-1:tcp-pose-2",
    )
    reset = gate.update(changed)
    assert reset.state is RimProbeState.CAPTURE_CANDIDATE
    assert reset.consecutive_good_windows == 1

    missing = evaluate(
        stripe(), stripe(), sdk_offset=20.0, probe_context_id=None
    )
    refused = gate.update(missing)
    assert refused.state is RimProbeState.CAPTURE_CANDIDATE
    assert refused.consecutive_good_windows == 0
    assert "provenance is incomplete" in refused.reasons[0]


def test_stability_gate_never_combines_different_role_assignments() -> None:
    gate = RimProbeStabilityGate(required_good_windows=2)
    first = evaluate(stripe(), stripe(), sdk_offset=0.0)
    assert gate.update(first).consecutive_good_windows == 1

    next_evaluation = evaluate(stripe(), stripe(), sdk_offset=10.0)
    swapped = replace(
        next_evaluation,
        contact_roles=(RimContactRole.OUTER, RimContactRole.INNER),
        predicates=(
            next_evaluation.predicates
            - {"left_inner_rim_match", "right_outer_rim_match"}
            | {"left_outer_rim_match", "right_inner_rim_match"}
        ),
    )
    reset = gate.update(swapped)

    assert reset.state is RimProbeState.CAPTURE_CANDIDATE
    assert reset.consecutive_good_windows == 1


def test_stability_gate_requires_host_capture_time_to_advance() -> None:
    gate = RimProbeStabilityGate(required_good_windows=2)
    first = evaluate(stripe(), stripe())
    assert gate.update(first).consecutive_good_windows == 1

    # SDK timestamps move, while the common-host capture endpoints are replayed.
    replayed_host = evaluate_rim_probe(
        FingerTactileWindow(
            calibration_id="left-cal",
            diff=timed_images(stripe(), sdk_offset=10.0, capture_offset=-0.005),
            force6d=timed_force(0.20, sdk_offset=10.0, capture_offset=-0.005),
        ),
        FingerTactileWindow(
            calibration_id="right-cal",
            diff=timed_images(stripe(), sdk_offset=10.0, capture_offset=-0.005),
            force6d=timed_force(0.20, sdk_offset=10.0, capture_offset=-0.005),
        ),
        make_config(),
        now_s=10.0,
        probe_context_id=PROBE_CONTEXT,
    )
    refused = gate.update(replayed_host)
    assert refused.state is RimProbeState.CAPTURE_CANDIDATE
    assert refused.consecutive_good_windows == 1


def test_stability_gate_restarts_after_a_long_window_gap() -> None:
    gate = RimProbeStabilityGate(required_good_windows=2, max_window_gap_s=0.5)
    first = evaluate(stripe(), stripe())
    assert gate.update(first).consecutive_good_windows == 1
    assert first.evidence_key is not None

    later_key = tuple(
        value + (1.0 if index % 2 else 10.0)
        for index, value in enumerate(first.evidence_key)
    )
    restarted = gate.update(
        replace(first, evidence_key=later_key, evaluated_at_monotonic_s=11.0)
    )
    assert restarted.state is RimProbeState.CAPTURE_CANDIDATE
    assert restarted.consecutive_good_windows == 1
    assert "continuity limit" in restarted.reasons[0]


def test_tcp_adjustment_requires_a_later_confirmed_release_window() -> None:
    config = make_config()
    proposal = evaluate(stripe(), np.zeros(IMAGE_SHAPE), right_load=0.0)
    assert proposal.recommendation.action is ProbeAction.RELEASE_THEN_ADJUST
    receipt = ReleaseCommandReceipt(
        command_id="release-001",
        probe_context_id=PROBE_CONTEXT,
        completed_at_monotonic_s=10.02,
        applied_jaw_opening_delta_m=config.release_step_m,
    )

    assert (
        authorize_tcp_adjustment(
            proposal, proposal, receipt, config, now_s=10.03
        )
        is None
    )
    released = evaluate(
        np.zeros(IMAGE_SHAPE),
        np.zeros(IMAGE_SHAPE),
        left_load=0.0,
        right_load=0.0,
        sdk_offset=10.0,
        capture_offset=0.25,
        now_s=10.25,
        not_before_s=receipt.completed_at_monotonic_s,
    )
    authorized = authorize_tcp_adjustment(
        proposal, released, receipt, config, now_s=10.26
    )

    assert authorized is not None
    assert authorized.adjustment.action is ProbeAction.APPLY_TCP_ADJUSTMENT
    assert authorized.contact_roles == (
        RimContactRole.INNER,
        RimContactRole.OUTER,
    )
    assert not authorized.adjustment.release_confirmation_required
    np.testing.assert_allclose(
        authorized.adjustment.proposed_tcp_translation_task_m,
        proposal.recommendation.proposed_tcp_translation_task_m,
    )
    ledger = TcpAdjustmentAuthorizationLedger()
    dispatched = ledger.consume(
        authorized,
        now_s=10.27,
        probe_context_id=PROBE_CONTEXT,
        profile_id=config.profile_id,
        calibration_ids=(config.left.calibration_id, config.right.calibration_id),
        contact_roles=(config.left.contact_role, config.right.contact_role),
    )
    assert dispatched is authorized.adjustment
    assert (
        ledger.consume(
            authorized,
            now_s=10.28,
            probe_context_id=PROBE_CONTEXT,
            profile_id=config.profile_id,
            calibration_ids=(config.left.calibration_id, config.right.calibration_id),
            contact_roles=(config.left.contact_role, config.right.contact_role),
        )
        is None
    )


def test_tcp_adjustment_is_bound_to_context_profile_calibration_and_ttl() -> None:
    config = make_config()
    proposal = evaluate(stripe(), np.zeros(IMAGE_SHAPE), right_load=0.0)
    receipt = ReleaseCommandReceipt(
        command_id="release-002",
        probe_context_id=PROBE_CONTEXT,
        completed_at_monotonic_s=10.02,
        applied_jaw_opening_delta_m=config.release_step_m,
    )
    released = evaluate(
        np.zeros(IMAGE_SHAPE),
        np.zeros(IMAGE_SHAPE),
        left_load=0.0,
        right_load=0.0,
        sdk_offset=10.0,
        capture_offset=0.25,
        now_s=10.25,
        not_before_s=receipt.completed_at_monotonic_s,
    )

    invalid_releases = (
        replace(released, probe_context_id="different-probe-context"),
        replace(released, profile_id="different-profile.v1"),
        replace(released, calibration_ids=("other-left", "other-right")),
        replace(
            released,
            contact_roles=(RimContactRole.OUTER, RimContactRole.INNER),
        ),
        replace(released, adjustment_authorization_ttl_s=0.5),
        replace(released, evaluated_at_monotonic_s=11.01),
    )
    for invalid in invalid_releases:
        assert (
            authorize_tcp_adjustment(
                proposal, invalid, receipt, config, now_s=10.26
            )
            is None
        )
    assert (
        authorize_tcp_adjustment(
            proposal, released, receipt, config, now_s=11.01
        )
        is None
    )


def test_tcp_adjustment_requires_every_sdk_and_host_channel_to_advance() -> None:
    config = make_config()
    proposal = evaluate(stripe(), np.zeros(IMAGE_SHAPE), right_load=0.0)
    receipt = ReleaseCommandReceipt(
        command_id="release-003",
        probe_context_id=PROBE_CONTEXT,
        completed_at_monotonic_s=10.02,
        applied_jaw_opening_delta_m=config.release_step_m,
    )
    released = evaluate(
        np.zeros(IMAGE_SHAPE),
        np.zeros(IMAGE_SHAPE),
        left_load=0.0,
        right_load=0.0,
        sdk_offset=10.0,
        capture_offset=0.25,
        now_s=10.25,
        not_before_s=receipt.completed_at_monotonic_s,
    )
    assert proposal.evidence_key is not None
    assert released.evidence_key is not None
    replayed_host_key = tuple(
        proposal.evidence_key[index] if index % 2 else value
        for index, value in enumerate(released.evidence_key)
    )
    replayed_host = replace(released, evidence_key=replayed_host_key)

    assert (
        authorize_tcp_adjustment(
            proposal, replayed_host, receipt, config, now_s=10.26
        )
        is None
    )


def test_tcp_adjustment_revalidates_bounds_and_cannot_be_mutated_in_place() -> None:
    config = make_config()
    proposal = evaluate(stripe(), np.zeros(IMAGE_SHAPE), right_load=0.0)
    receipt = ReleaseCommandReceipt(
        command_id="release-004",
        probe_context_id=PROBE_CONTEXT,
        completed_at_monotonic_s=10.02,
        applied_jaw_opening_delta_m=config.release_step_m,
    )
    released = evaluate(
        np.zeros(IMAGE_SHAPE),
        np.zeros(IMAGE_SHAPE),
        left_load=0.0,
        right_load=0.0,
        sdk_offset=10.0,
        capture_offset=0.25,
        now_s=10.25,
        not_before_s=receipt.completed_at_monotonic_s,
    )

    with np.testing.assert_raises(ValueError):
        proposal.recommendation.proposed_tcp_translation_task_m[0] = 0.1

    oversized = replace(
        proposal,
        recommendation=replace(
            proposal.recommendation,
            proposed_tcp_translation_task_m=np.array([0.01, 0.0, 0.0]),
        ),
    )
    assert (
        authorize_tcp_adjustment(
            oversized, released, receipt, config, now_s=10.26
        )
        is None
    )


def test_safety_critical_profile_arrays_are_copied_and_read_only() -> None:
    source_normal = np.array([1.0, 0.0, 0.0])
    source_roi = np.ones(IMAGE_SHAPE, dtype=bool)
    target = replace(
        make_config().left,
        inward_normal_sensor=source_normal,
        roi_mask=source_roi,
    )
    source_normal[:] = 0.0
    source_roi[:] = False

    np.testing.assert_allclose(target.inward_normal_sensor, [1.0, 0.0, 0.0])
    assert target.roi_mask.all()
    for array in (
        target.roi_mask,
        target.target_center_uv_px,
        target.target_tangent_uv,
        target.inward_normal_sensor,
        target.image_error_to_task_m_per_px,
        target.toward_sensor_task,
    ):
        assert array is not None
        assert not array.flags.writeable


def test_transient_depth_crush_is_not_hidden_by_temporal_aggregation() -> None:
    base = make_config()
    config = replace(
        base,
        left=replace(base.left, max_indentation_m=0.001),
        right=replace(base.right, max_indentation_m=0.001),
    )
    depth = timed_depth()
    depth_values = depth.values.copy()
    depth_values[2, 30:35, 20:45] = 0.002
    transient_depth = replace(depth, values=depth_values)
    left = replace(finger("left", stripe(), 0.20), depth=transient_depth)
    right = replace(finger("right", stripe(), 0.20), depth=timed_depth())

    result = evaluate_rim_probe(
        left,
        right,
        config,
        now_s=10.0,
        probe_context_id=PROBE_CONTEXT,
    )
    assert result.state is RimProbeState.EXCESS_DEFORMATION
    assert result.recommendation.action is ProbeAction.RELEASE


def test_probe_configuration_rejects_unreviewed_command_sizes() -> None:
    with np.testing.assert_raises(ValueError):
        replace(make_config(), close_step_m=0.003)
    with np.testing.assert_raises(ValueError):
        replace(make_config(), max_translation_step_m=0.004)
    with np.testing.assert_raises(ValueError):
        replace(make_config(), adjustment_authorization_ttl_s=6.0)


def test_probe_configuration_requires_complementary_inner_outer_roles() -> None:
    config = make_config()
    with np.testing.assert_raises(ValueError):
        replace(
            config,
            right=replace(config.right, contact_role=RimContactRole.INNER),
        )


def test_channel_desynchronization_fails_closed() -> None:
    left = FingerTactileWindow(
        calibration_id="left-cal",
        diff=timed_images(stripe()),
        force6d=timed_force(0.20, capture_offset=-0.06),
    )
    result = evaluate_rim_probe(left, finger("right", stripe(), 0.20), make_config(), now_s=10.0)

    assert result.state is RimProbeState.SENSOR_INVALID
    assert "desynchronized" in result.reasons[0]
