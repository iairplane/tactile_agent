import numpy as np
import pytest

from demo2gripper.intent import INDEX_TIP, THUMB_TIP, WRIST, IntentConfig
from demo2gripper.pickup import (
    align_hand_joints_to_reference_translation,
    assess_approach_for_fixed_object_replay,
    detect_pickup_events,
    infer_pre_lift_pickup_intent,
    measure_hand_object_coupling,
    measure_prelift_object_motion,
    validate_contact_before_lift,
)


def synthetic_pickup_track() -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.arange(80, dtype=float) / 30.0
    centers = np.zeros((80, 3), dtype=float)
    centers[:, :2] = [0.4, -0.1]
    centers[:20, 2] = 0.05
    centers[20:30, 2] = np.linspace(0.05, 0.09, 10)
    centers[30:60, 2] = 0.09
    centers[60:, 2] = np.linspace(0.09, 0.05, 20)
    centers[:20, 2] += 0.0002 * np.sin(np.arange(20))
    return centers, timestamps


def test_detects_sustained_pickup_events() -> None:
    centers, timestamps = synthetic_pickup_track()

    events = detect_pickup_events(
        centers,
        timestamps,
        reference_frame_count=15,
        min_onset_frames=3,
        min_airborne_height_m=0.03,
        min_airborne_frames=10,
        min_airborne_duration_s=0.25,
    )

    assert 21 <= events.lift_onset_frame <= 24
    assert events.airborne_interval[0] >= events.lift_onset_frame
    assert events.peak_lift_m > 0.039
    assert events.airborne_duration_s > 0.9
    np.testing.assert_allclose(events.reference_center_base_m, [0.4, -0.1, 0.05], atol=3e-4)


def test_rejects_motion_without_an_airborne_hold() -> None:
    centers, timestamps = synthetic_pickup_track()
    centers[30:, 2] = 0.05

    with pytest.raises(ValueError, match="no sustained airborne interval"):
        detect_pickup_events(
            centers,
            timestamps,
            reference_frame_count=15,
            min_onset_frames=3,
            min_airborne_height_m=0.03,
            min_airborne_frames=10,
            min_airborne_duration_s=0.25,
        )


def test_measures_rigid_hand_object_coupling() -> None:
    centers, _ = synthetic_pickup_track()
    offset = np.array([0.002, -0.001, 0.011])
    pinch = centers - offset
    confidence = np.ones(len(centers))

    coupling = measure_hand_object_coupling(
        centers, pinch, confidence, (20, 40), min_frames=20
    )

    np.testing.assert_allclose(coupling.median_offset_base_m, offset)
    assert coupling.relative_motion_p95_m < 1e-12
    assert np.isclose(coupling.median_center_distance_m, np.linalg.norm(offset))
    assert coupling.moving_step_pairs > 0
    assert np.isclose(coupling.velocity_cosine_median, 1.0)
    assert coupling.step_displacement_mismatch_p95_m < 1e-12


def test_hand_object_coupling_rejects_sparse_pose_jumps() -> None:
    centers, _ = synthetic_pickup_track()
    offset = np.array([0.002, -0.001, 0.011])
    pinch = centers - offset
    pinch[[24, 37]] += np.array([0.080, -0.050, 0.030])
    confidence = np.ones(len(centers))

    coupling = measure_hand_object_coupling(
        centers,
        pinch,
        confidence,
        (20, 50),
        min_frames=28,
        max_offset_residual_m=0.010,
    )

    assert coupling.valid_frames == 28
    np.testing.assert_allclose(coupling.median_offset_base_m, offset)
    assert coupling.relative_motion_p95_m < 1e-12
    assert coupling.moving_step_pairs > 0
    assert np.isclose(coupling.velocity_cosine_median, 1.0)
    assert coupling.step_displacement_mismatch_p95_m < 1e-12

    with pytest.raises(ValueError, match="only 28 inlier.*need 29"):
        measure_hand_object_coupling(
            centers,
            pinch,
            confidence,
            (20, 50),
            min_frames=29,
            max_offset_residual_m=0.010,
        )


def test_contact_must_lead_directly_into_lift() -> None:
    validate_contact_before_lift((10, 28), 30, max_gap_frames=2)
    with pytest.raises(ValueError, match="stable contact ends 3 frames"):
        validate_contact_before_lift((10, 27), 30, max_gap_frames=2)


def test_prelift_object_slide_invalidates_direct_replay_only() -> None:
    centers = np.full((30, 3), [0.40, -0.10, 0.05], dtype=float)
    centers[20:30, 0] += np.linspace(0.009, 0.011, 10)

    motion = measure_prelift_object_motion(
        centers,
        np.array([0.40, -0.10, 0.05]),
        (20, 30),
        reference_motion_p95_m=0.0008,
        max_shift_m=0.005,
        noise_multiplier=6.0,
    )

    assert motion.valid_frames == 10
    assert motion.contact_center_shift_m > 0.009
    assert motion.contact_shift_p95_m > motion.effective_shift_limit_m
    assert not motion.replay_target_valid


def test_translation_alignment_tracks_contact_while_preserving_shift_gate() -> None:
    frame_count = 30
    contact_start = 10
    lift_onset = 32
    reference_center = np.array([0.0, 0.0, 0.70])
    translations = np.zeros((frame_count, 3), dtype=float)
    translations[16:, 0] = np.linspace(0.0, 0.050, frame_count - 16)
    object_centers = reference_center + translations

    midpoint = np.repeat(reference_center[None], frame_count, axis=0)
    midpoint[:contact_start, 1] = np.linspace(-0.080, -0.010, contact_start)
    midpoint += translations
    joints = np.zeros((frame_count, 21, 3), dtype=float)
    joints[:, WRIST] = midpoint + [0.0, -0.10, 0.0]
    joints[:, THUMB_TIP] = midpoint + [-0.020, 0.0, 0.0]
    joints[:, INDEX_TIP] = midpoint + [0.020, 0.0, 0.0]
    confidence = np.ones((frame_count, 21), dtype=float)
    timestamps = np.arange(frame_count, dtype=float) / 30.0
    reference_cloud = np.array(
        [
            [-0.020, 0.0, 0.70],
            [0.020, 0.0, 0.70],
            [0.0, 0.020, 0.70],
            [0.0, -0.020, 0.70],
        ]
    )
    intent_config = IntentConfig(max_tip_object_distance_m=0.004, min_stable_frames=5)

    raw_intent = infer_pre_lift_pickup_intent(
        joints,
        confidence,
        timestamps,
        reference_cloud,
        intent_config,
        approach_lookback_frames=10,
        approach_guard_frames=0,
        min_approach_frames=8,
    )
    assert raw_intent.stable_interval[1] < frame_count

    aligned_joints, track_valid, measured_translations = (
        align_hand_joints_to_reference_translation(
            joints, object_centers, reference_center
        )
    )
    aligned_confidence = confidence.copy()
    aligned_confidence[~track_valid] = 0.0
    aligned_intent = infer_pre_lift_pickup_intent(
        aligned_joints,
        aligned_confidence,
        timestamps,
        reference_cloud,
        intent_config,
        approach_lookback_frames=10,
        approach_guard_frames=0,
        min_approach_frames=8,
    )

    assert aligned_intent.stable_interval == (contact_start, frame_count)
    validate_contact_before_lift(
        aligned_intent.stable_interval, lift_onset, max_gap_frames=2
    )
    np.testing.assert_allclose(measured_translations, translations)
    np.testing.assert_allclose(
        aligned_intent.T_camera_gripper_tcp[:3, 3], reference_center, atol=1e-12
    )
    np.testing.assert_allclose(
        aligned_intent.T_camera_gripper_tcp[:3, 0], [1.0, 0.0, 0.0], atol=1e-12
    )
    assert np.isclose(aligned_intent.demonstrated_pinch_width_m, 0.040)

    motion = measure_prelift_object_motion(
        object_centers,
        reference_center,
        aligned_intent.stable_interval,
        reference_motion_p95_m=0.0001,
        max_shift_m=0.005,
        noise_multiplier=6.0,
    )
    assert motion.contact_shift_p95_m > 0.045
    assert not motion.replay_target_valid

    restored_grasp_center = (
        aligned_intent.T_camera_gripper_tcp[:3, 3] + motion.shift_vector_base_m
    )
    np.testing.assert_allclose(
        restored_grasp_center - motion.contact_center_base_m,
        aligned_intent.T_camera_gripper_tcp[:3, 3] - reference_center,
        atol=1e-12,
    )


def test_translation_alignment_marks_missing_track_without_inventing_motion() -> None:
    joints = np.arange(3 * 21 * 3, dtype=float).reshape(3, 21, 3) / 1000.0
    centers = np.array(
        [[0.30, -0.10, 0.70], [np.nan, np.nan, np.nan], [0.35, -0.08, 0.70]]
    )
    reference = np.array([0.30, -0.10, 0.70])

    aligned, valid, translations = align_hand_joints_to_reference_translation(
        joints, centers, reference
    )

    np.testing.assert_array_equal(valid, [True, False, True])
    np.testing.assert_allclose(aligned[0], joints[0])
    np.testing.assert_allclose(aligned[2], joints[2] - [0.05, 0.02, 0.0])
    np.testing.assert_allclose(aligned[1], joints[1])
    assert np.isnan(translations[1]).all()
    np.testing.assert_allclose(
        translations[[0, 2]], [[0.0, 0.0, 0.0], [0.05, 0.02, 0.0]]
    )


def test_pre_lift_intent_preserves_raw_pinch_axis() -> None:
    frame_count = 18
    joints = np.zeros((frame_count, 21, 3), dtype=float)
    joints[:, WRIST] = [0.0, -0.10, 0.60]
    approach_y = np.r_[np.linspace(-0.08, -0.02, 6), np.zeros(frame_count - 6)]
    joints[:, THUMB_TIP] = np.column_stack(
        (np.full(frame_count, -0.02), approach_y, np.full(frame_count, 0.70))
    )
    joints[:, INDEX_TIP] = np.column_stack(
        (np.full(frame_count, 0.02), approach_y, np.full(frame_count, 0.70))
    )
    confidence = np.ones((frame_count, 21), dtype=float)
    timestamps = np.arange(frame_count, dtype=float) / 30.0
    object_points = np.array(
        [[-0.02, 0.0, 0.70], [0.02, 0.0, 0.70], [0.0, 0.02, 0.70]]
    )

    intent = infer_pre_lift_pickup_intent(
        joints,
        confidence,
        timestamps,
        object_points,
        IntentConfig(max_tip_object_distance_m=0.005, min_stable_frames=5),
        approach_lookback_frames=6,
        approach_guard_frames=0,
        min_approach_frames=5,
    )

    np.testing.assert_allclose(intent.T_camera_gripper_tcp[:3, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(intent.T_camera_gripper_tcp[:3, 3], [0.0, 0.0, 0.70])
    np.testing.assert_allclose(
        intent.T_camera_gripper_tcp[:3, :3].T
        @ intent.T_camera_gripper_tcp[:3, :3],
        np.eye(3),
        atol=1e-12,
    )
    assert np.isclose(intent.demonstrated_pinch_width_m, 0.04)
    assert intent.stable_interval == (6, frame_count)
    assert intent.approach_interval == (0, 6)
    assert intent.approach_trajectory_displacement_m > 0.04
    assert intent.approach_window_angle_p95_deg < 1e-6


def test_pre_lift_intent_reports_window_sensitive_approach() -> None:
    frame_count = 50
    midpoint = np.zeros((frame_count, 3), dtype=float)
    midpoint[:, 2] = 0.70
    midpoint[:15, 1] = np.linspace(-0.10, -0.04, 15)
    midpoint[15:30, 1] = np.linspace(-0.04, -0.004, 15)
    midpoint[15:30, 2] = np.linspace(0.66, 0.696, 15)
    joints = np.zeros((frame_count, 21, 3), dtype=float)
    joints[:, WRIST] = midpoint + [0.0, -0.10, 0.0]
    joints[:, THUMB_TIP] = midpoint + [-0.02, 0.0, 0.0]
    joints[:, INDEX_TIP] = midpoint + [0.02, 0.0, 0.0]
    confidence = np.ones((frame_count, 21), dtype=float)
    timestamps = np.arange(frame_count, dtype=float) / 30.0
    object_points = np.array(
        [[-0.02, 0.0, 0.70], [0.02, 0.0, 0.70], [0.0, 0.02, 0.70]]
    )

    intent = infer_pre_lift_pickup_intent(
        joints,
        confidence,
        timestamps,
        object_points,
        IntentConfig(max_tip_object_distance_m=0.005, min_stable_frames=10),
        approach_lookback_frames=30,
        approach_guard_frames=0,
        min_approach_frames=15,
    )

    assert intent.stable_interval == (30, frame_count)
    assert intent.approach_window_count == 4
    assert intent.approach_window_angle_p95_deg > 15.0

    replay = assess_approach_for_fixed_object_replay(
        intent,
        min_trajectory_displacement_m=0.010,
        max_trajectory_residual_p95_m=0.015,
        min_signal_to_residual=3.0,
        min_window_count=3,
        max_window_angle_p95_deg=15.0,
    )

    # Contact inference remains valid; curvature only withholds direct replay.
    assert intent.stable_interval == (30, frame_count)
    assert "approach_window_instability" in replay.gate_failures
    assert not replay.window_stable
    assert replay.gate_failures


def test_contact_pose_survives_unavailable_free_space_trajectory() -> None:
    frame_count = 24
    midpoint = np.zeros((frame_count, 3), dtype=float)
    midpoint[:, 2] = 0.70
    midpoint[:8, 1] = -0.04
    joints = np.zeros((frame_count, 21, 3), dtype=float)
    joints[:, WRIST] = midpoint + [0.0, -0.10, 0.0]
    joints[:, THUMB_TIP] = midpoint + [-0.02, 0.0, 0.0]
    joints[:, INDEX_TIP] = midpoint + [0.02, 0.0, 0.0]
    confidence = np.ones((frame_count, 21), dtype=float)
    timestamps = np.arange(frame_count, dtype=float) / 30.0
    object_points = np.array(
        [[-0.02, 0.0, 0.70], [0.02, 0.0, 0.70], [0.0, 0.02, 0.70]]
    )

    intent = infer_pre_lift_pickup_intent(
        joints,
        confidence,
        timestamps,
        object_points,
        IntentConfig(max_tip_object_distance_m=0.005, min_stable_frames=10),
        approach_lookback_frames=8,
        approach_guard_frames=0,
        min_approach_frames=8,
    )
    replay = assess_approach_for_fixed_object_replay(
        intent,
        min_trajectory_displacement_m=0.010,
        max_trajectory_residual_p95_m=0.015,
        min_signal_to_residual=3.0,
        min_window_count=1,
        max_window_angle_p95_deg=15.0,
    )

    assert intent.stable_interval == (8, frame_count)
    assert intent.approach_interval is None
    assert intent.approach_trajectory_displacement_m is None
    assert replay.gate_failures == ("approach_trajectory_unavailable",)
