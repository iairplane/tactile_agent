from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from demo2gripper.cartesian_pose_sequence import (
    MAX_SEGMENT_ROTATION_RAD,
    POSE_SEQUENCE_SCHEMA,
)
from demo2gripper.medicine_bottle import (
    AXIAL_LIFT_GEOMETRY_SCHEMA,
    AXIS_DIRECTION,
    BOTTLE_CAP_FRAME_SCHEMA,
    FRAME_CONVENTION,
    PRESS_TWIST_GEOMETRY_SCHEMA,
    BottleCapFrame,
    build_axial_lift_geometry,
    build_press_twist_geometry,
    cartesian_pose_to_transform,
    parse_bottle_cap_frame_payload,
)


def _frame_payload() -> dict[str, object]:
    transform = np.eye(4)
    transform[:3, 3] = [0.5, -0.1, 0.2]
    return {
        "schema": BOTTLE_CAP_FRAME_SCHEMA,
        "transform_convention": FRAME_CONVENTION,
        "units": "metres",
        "axis_direction": AXIS_DIRECTION,
        "base_frame": "fr3_link0",
        "cap_frame": "medicine_cap",
        "tcp_frame": "hande_tcp",
        "T_base_cap_tcp": transform.tolist(),
        "bottle_axis_base": [0.0, 0.0, 1.0],
        "bottle_axis_tcp": [0.0, 0.0, 1.0],
    }


def _frame() -> BottleCapFrame:
    return parse_bottle_cap_frame_payload(
        _frame_payload(), expected_base_frame="fr3_link0"
    )


def test_strict_frame_contract_round_trips_and_is_immutable() -> None:
    frame = _frame()

    assert frame.as_dict() == _frame_payload()
    assert frame.T_base_cap_tcp.flags.writeable is False
    with pytest.raises(ValueError):
        frame.T_base_cap_tcp[0, 3] = 123.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(extra=True), "unknown fields"),
        (lambda value: value.pop("tcp_frame"), "missing fields"),
        (lambda value: value.update(units="millimetres"), "units"),
        (
            lambda value: value.update(bottle_axis_base=[0.0, 0.0, 2.0]),
            "unit vector",
        ),
        (
            lambda value: value.update(bottle_axis_base=[False, 0.0, 1.0]),
            "finite numbers",
        ),
        (
            lambda value: value.update(bottle_axis_tcp=[0.0, 1.0, 0.0]),
            "does not map",
        ),
        (
            lambda value: value.update(T_base_cap_tcp=np.zeros((4, 4)).tolist()),
            "bottom row",
        ),
        (
            lambda value: value.update(
                T_base_cap_tcp=[
                    [True, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ),
            "finite numbers",
        ),
        (lambda value: value.update(cap_frame="fr3_link0"), "must be distinct"),
    ],
)
def test_frame_contract_fails_closed(mutation, message: str) -> None:
    payload = copy.deepcopy(_frame_payload())
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        parse_bottle_cap_frame_payload(payload, expected_base_frame="fr3_link0")


def test_frame_contract_rejects_unexpected_base_frame() -> None:
    with pytest.raises(ValueError, match="does not match expected"):
        parse_bottle_cap_frame_payload(
            _frame_payload(), expected_base_frame="panda_link0"
        )


def test_press_twist_is_coaxial_bounded_and_supports_thread_pitch() -> None:
    geometry = build_press_twist_geometry(
        _frame(),
        press_distance_m=0.010,
        twist_step_angle_rad=math.radians(15.0),
        twist_steps=4,
        thread_pitch_m_per_revolution=0.006,
    )

    assert len(geometry.press_waypoints) == 1
    assert len(geometry.twist_waypoints) == 4
    assert geometry.press_waypoints[0].position_m == pytest.approx((0.5, -0.1, 0.19))
    pitch_step = 0.006 * 15.0 / 360.0
    assert [pose.position_m[2] for pose in geometry.twist_waypoints] == pytest.approx(
        [0.19 + pitch_step * index for index in range(1, 5)]
    )

    initial_rotation = _frame().T_base_cap_tcp[:3, :3]
    for index, pose in enumerate(geometry.twist_waypoints, start=1):
        transform = cartesian_pose_to_transform(pose)
        expected_angle = math.radians(15.0 * index)
        expected = np.array(
            [
                [math.cos(expected_angle), -math.sin(expected_angle), 0.0],
                [math.sin(expected_angle), math.cos(expected_angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ) @ initial_rotation
        np.testing.assert_allclose(transform[:3, :3], expected, atol=1e-10)

    metrics = geometry.motion_validation
    assert metrics["waypoint_count"] == 5
    assert max(item["rotation_rad"] for item in metrics["segment_metrics"]) <= (
        MAX_SEGMENT_ROTATION_RAD + 1e-12
    )
    payload = geometry.as_dict()
    assert payload["schema"] == PRESS_TWIST_GEOMETRY_SCHEMA
    assert payload["status"] == "geometry_only"
    assert payload["execution_authorized"] is False
    assert payload["hardware_execution_enabled"] is False
    assert payload["pose_sequences"]["PRESS"]["schema"] == POSE_SEQUENCE_SCHEMA
    assert payload["geometric_criteria"]["TWIST"]["coaxial_lateral_drift_m"] == 0.0


def test_negative_twist_angle_retains_positive_outward_pitch() -> None:
    geometry = build_press_twist_geometry(
        _frame(),
        press_distance_m=0.005,
        twist_step_angle_rad=-math.radians(20.0),
        twist_steps=2,
        thread_pitch_m_per_revolution=0.009,
    )

    assert geometry.twist_waypoints[-1].position_m[2] > geometry.press_waypoints[0].position_m[2]
    final_rotation = cartesian_pose_to_transform(geometry.twist_waypoints[-1])[:3, :3]
    angle = -math.radians(40.0)
    np.testing.assert_allclose(
        final_rotation,
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        atol=1e-10,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"press_distance_m": 0.0}, "must be positive"),
        ({"press_distance_m": 0.050001}, "segment cap"),
        ({"twist_step_angle_rad": 0.0}, "must be non-zero"),
        (
            {"twist_step_angle_rad": math.radians(20.001)},
            "at most",
        ),
        ({"twist_steps": 32}, "at most 32"),
        ({"twist_steps": True}, "positive integer"),
        ({"thread_pitch_m_per_revolution": -0.001}, "non-negative"),
    ],
)
def test_press_twist_rejects_unbounded_or_ambiguous_inputs(kwargs, message: str) -> None:
    values = {
        "press_distance_m": 0.010,
        "twist_step_angle_rad": math.radians(10.0),
        "twist_steps": 3,
        "thread_pitch_m_per_revolution": 0.0,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_press_twist_geometry(_frame(), **values)


def test_axial_lift_uses_fixed_orientation_and_twenty_mm_or_smaller_steps() -> None:
    press_twist = build_press_twist_geometry(
        _frame(),
        press_distance_m=0.010,
        twist_step_angle_rad=math.radians(20.0),
        twist_steps=3,
    )
    start = cartesian_pose_to_transform(press_twist.twist_waypoints[-1])
    lift = build_axial_lift_geometry(
        _frame(), T_base_start_tcp=start, lift_distance_m=0.055
    )

    assert len(lift.lift_waypoints) == 3
    assert [pose.position_m[2] for pose in lift.lift_waypoints] == pytest.approx(
        [start[2, 3] + 0.055 / 3.0 * index for index in range(1, 4)]
    )
    for pose in lift.lift_waypoints:
        np.testing.assert_allclose(
            cartesian_pose_to_transform(pose)[:3, :3], start[:3, :3], atol=1e-10
        )
    assert max(
        item["translation_m"] for item in lift.motion_validation["segment_metrics"]
    ) <= 0.020 + 1e-12
    payload = lift.as_dict()
    assert payload["schema"] == AXIAL_LIFT_GEOMETRY_SCHEMA
    assert payload["execution_authorized"] is False
    assert payload["geometric_criteria"]["LIFT"]["strictly_monotonic_axial_motion"] is True
    assert payload["pose_sequences"]["LIFT"]["schema"] == POSE_SEQUENCE_SCHEMA


@pytest.mark.parametrize(
    ("distance", "step", "message"),
    [
        (0.0, 0.020, "must be positive"),
        (0.020, 0.0, "must be positive"),
        (0.020, 0.020001, "capped at 0.020"),
        (0.33, 0.010, "exceeds the 32-waypoint cap"),
    ],
)
def test_axial_lift_rejects_unsafe_sampling(
    distance: float, step: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_axial_lift_geometry(
            _frame(),
            T_base_start_tcp=_frame().T_base_cap_tcp,
            lift_distance_m=distance,
            max_step_m=step,
        )


def test_axial_lift_rejects_start_pose_that_is_not_coaxial() -> None:
    start = np.eye(4)
    angle = math.radians(10.0)
    start[:3, :3] = [
        [1.0, 0.0, 0.0],
        [0.0, math.cos(angle), -math.sin(angle)],
        [0.0, math.sin(angle), math.cos(angle)],
    ]

    with pytest.raises(ValueError, match="does not map"):
        build_axial_lift_geometry(
            _frame(), T_base_start_tcp=start, lift_distance_m=0.02
        )
