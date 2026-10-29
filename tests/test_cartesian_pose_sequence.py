from __future__ import annotations

import math

import pytest

from demo2gripper.cartesian_pose_sequence import (
    MAX_WAYPOINTS,
    POSE_SEQUENCE_SCHEMA,
    CartesianPose,
    configured_ee_to_link8_waypoints,
    parse_pose_sequence_payload,
    validate_cartesian_speed_m_s,
    validate_pose_sequence_motion,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def pose(x: float, *, orientation: tuple[float, float, float, float] = IDENTITY) -> CartesianPose:
    return CartesianPose((x, 0.0, 0.0), orientation)


def payload(waypoints: list[dict]) -> dict:
    return {
        "schema": POSE_SEQUENCE_SCHEMA,
        "base_frame": "fr3_link0",
        "waypoints": waypoints,
    }


def test_parses_absolute_configured_ee_poses_and_normalizes_quaternions() -> None:
    waypoints = parse_pose_sequence_payload(
        payload(
            [
                {
                    "position_m": [0.6, -0.1, 0.2],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 2.0],
                }
            ]
        ),
        expected_base_frame="fr3_link0",
    )

    assert waypoints == [CartesianPose((0.6, -0.1, 0.2), IDENTITY)]


@pytest.mark.parametrize(
    ("bad_payload", "message"),
    [
        ({"waypoints": []}, "schema"),
        (
            {
                "schema": POSE_SEQUENCE_SCHEMA,
                "base_frame": "camera",
                "waypoints": [],
            },
            "does not match",
        ),
        (
            payload(
                [
                    {
                        "position_m": [0.0, 0.0, math.inf],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ]
            ),
            "finite",
        ),
        (
            payload(
                [
                    {
                        "position_m": [0.0, 0.0, 0.0],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 0.0],
                    }
                ]
            ),
            "non-zero",
        ),
        (
            payload(
                [
                    {"position_m": [0.0, 0.0, 0.0], "orientation_xyzw": list(IDENTITY)}
                ]
                * (MAX_WAYPOINTS + 1)
            ),
            "1..32",
        ),
    ],
)
def test_malformed_pose_sequence_fails_closed(bad_payload: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_pose_sequence_payload(bad_payload, expected_base_frame="fr3_link0")


def test_motion_validation_includes_current_pose_and_reports_all_segments() -> None:
    metrics = validate_pose_sequence_motion(
        pose(0.0), [pose(0.03), pose(0.07), pose(0.10)]
    )

    assert metrics["waypoint_count"] == 3
    assert metrics["total_cartesian_length_m"] == pytest.approx(0.10)
    assert [item["translation_m"] for item in metrics["segment_metrics"]] == pytest.approx(
        [0.03, 0.04, 0.03]
    )


def test_rejects_segment_translation_over_50_mm_from_current_pose() -> None:
    with pytest.raises(ValueError, match="segment 0 translation"):
        validate_pose_sequence_motion(pose(0.0), [pose(0.050001)])


def test_rejects_segment_rotation_over_20_degrees() -> None:
    angle = math.radians(20.01)
    rotated = (
        0.0,
        0.0,
        math.sin(angle / 2.0),
        math.cos(angle / 2.0),
    )

    with pytest.raises(ValueError, match="segment 0 rotation"):
        validate_pose_sequence_motion(pose(0.0), [pose(0.0, orientation=rotated)])


def test_rejects_total_cartesian_length_over_half_metre() -> None:
    waypoints = [pose(0.05 if index % 2 == 0 else 0.0) for index in range(11)]

    with pytest.raises(ValueError, match="total Cartesian length"):
        validate_pose_sequence_motion(pose(0.0), waypoints)


def test_converts_each_configured_ee_pose_with_explicit_transform() -> None:
    converted = configured_ee_to_link8_waypoints(
        [CartesianPose((0.6, 0.0, 0.04), (1.0, 0.0, 0.0, 0.0))],
        link8_to_configured_ee_position_m=(0.0, 0.0, 0.190),
        link8_to_configured_ee_orientation_xyzw=IDENTITY,
    )

    assert converted[0].position_m == pytest.approx((0.6, 0.0, 0.23))
    assert converted[0].orientation_xyzw == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_cartesian_speed_has_non_weakenable_25_mm_s_cap() -> None:
    assert validate_cartesian_speed_m_s(0.025) == pytest.approx(0.025)
    with pytest.raises(ValueError, match="capped at 0.025"):
        validate_cartesian_speed_m_s(0.025001)
    with pytest.raises(ValueError, match="positive and finite"):
        validate_cartesian_speed_m_s(0.0)
