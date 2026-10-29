from __future__ import annotations

import math

import pytest

from demo2gripper.pose_motion import (
    classify_measured_pose_outcome,
    compose_pose,
    link8_pose_from_configured_ee,
    normalize_quaternion_xyzw,
    pose_delta,
    quaternion_angle_rad,
)


def test_configured_ee_to_link8_round_trip_with_measured_offset() -> None:
    ee_position = [0.658, 0.038, 0.117]
    ee_orientation = normalize_quaternion_xyzw([0.70, 0.70, 0.10, -0.10])
    offset_position = [0.0, 0.0, 0.190]
    offset_orientation = (0.0, 0.0, 0.0, 1.0)

    link8_position, link8_orientation = link8_pose_from_configured_ee(
        ee_position,
        ee_orientation,
        offset_position,
        offset_orientation,
    )
    reconstructed_position, reconstructed_orientation = compose_pose(
        link8_position,
        link8_orientation,
        offset_position,
        offset_orientation,
    )

    assert reconstructed_position == pytest.approx(ee_position, abs=1e-12)
    assert quaternion_angle_rad(reconstructed_orientation, ee_orientation) == pytest.approx(
        0.0, abs=1e-12
    )


def test_downward_configured_ee_places_link8_above_tcp() -> None:
    half_turn_x = (1.0, 0.0, 0.0, 0.0)

    link8_position, _ = link8_pose_from_configured_ee(
        [0.6, 0.0, 0.04],
        half_turn_x,
        [0.0, 0.0, 0.190],
        (0.0, 0.0, 0.0, 1.0),
    )

    assert link8_position == pytest.approx([0.6, 0.0, 0.23], abs=1e-12)


def test_pose_delta_uses_shortest_quaternion_rotation() -> None:
    identity = (0.0, 0.0, 0.0, 1.0)
    quarter_turn_z = normalize_quaternion_xyzw(
        [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]
    )

    translation_m, rotation_rad = pose_delta(
        [0.0, 0.0, 0.0], identity, [0.3, 0.4, 0.0], quarter_turn_z
    )

    assert translation_m == pytest.approx(0.5)
    assert rotation_rad == pytest.approx(math.pi / 2.0)
    assert quaternion_angle_rad(identity, (0.0, 0.0, 0.0, -1.0)) == 0.0


@pytest.mark.parametrize(
    "quaternion",
    ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, math.inf, 1.0]),
)
def test_normalize_quaternion_rejects_invalid_values(quaternion: list[float]) -> None:
    with pytest.raises(ValueError):
        normalize_quaternion_xyzw(quaternion)


def test_pose_outcome_distinguishes_action_success_without_physical_motion() -> None:
    outcome, translation_m, rotation_rad = classify_measured_pose_outcome(
        start_position=[0.5, 0.4, 0.5],
        start_orientation=(0.0, 0.0, 0.0, 1.0),
        measured_position=[0.500002, 0.4, 0.5],
        measured_orientation=(0.0, 0.0, 0.0, 1.0),
        target_position_error_m=0.249,
        target_rotation_error_rad=1.96,
        position_tolerance_m=0.003,
        orientation_tolerance_rad=0.05,
    )

    assert outcome == "controller_reported_success_without_measured_motion"
    assert translation_m == pytest.approx(0.000002)
    assert rotation_rad == pytest.approx(0.0)


def test_pose_outcome_distinguishes_endpoint_mismatch_after_real_motion() -> None:
    outcome, translation_m, _ = classify_measured_pose_outcome(
        start_position=[0.5, 0.4, 0.5],
        start_orientation=(0.0, 0.0, 0.0, 1.0),
        measured_position=[0.55, 0.4, 0.5],
        measured_orientation=(0.0, 0.0, 0.0, 1.0),
        target_position_error_m=0.020,
        target_rotation_error_rad=0.0,
        position_tolerance_m=0.003,
        orientation_tolerance_rad=0.05,
    )

    assert outcome == "measured_endpoint_mismatch"
    assert translation_m == pytest.approx(0.05)


def test_pose_outcome_accepts_measured_target_before_stationary_diagnostic() -> None:
    outcome, _, _ = classify_measured_pose_outcome(
        start_position=[0.5, 0.4, 0.5],
        start_orientation=(0.0, 0.0, 0.0, 1.0),
        measured_position=[0.5001, 0.4, 0.5],
        measured_orientation=(0.0, 0.0, 0.0, 1.0),
        target_position_error_m=0.0002,
        target_rotation_error_rad=0.001,
        position_tolerance_m=0.003,
        orientation_tolerance_rad=0.05,
    )

    assert outcome == "measured_target_reached"
