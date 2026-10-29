import numpy as np
import pytest

from demo2gripper.test_tube_visual import (
    TubeAxisObservation,
    VisualAngleGate,
    VisualAngleGateConfig,
    axis_angle_error_deg,
    estimate_tube_axis_from_points,
)


def _line_points(angle_deg: float, *, count: int = 101) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-direction[1], direction[0]])
    along = np.linspace(-60.0, 60.0, count)
    across = 1.2 * np.sin(np.linspace(0.0, 6.0 * np.pi, count))
    return np.array([320.0, 240.0]) + along[:, None] * direction + across[:, None] * normal


def _observation(angle_deg: float) -> TubeAxisObservation:
    return estimate_tube_axis_from_points(_line_points(angle_deg))


def test_axis_estimate_is_undirected_and_accurate() -> None:
    first = _observation(126.0)
    reversed_points = _line_points(126.0)[::-1]
    second = estimate_tube_axis_from_points(reversed_points)

    assert first.valid and second.valid
    assert abs(axis_angle_error_deg(first.axis_angle_deg, 126.0)) < 0.5
    assert abs(axis_angle_error_deg(second.axis_angle_deg, 126.0)) < 0.5


def test_short_or_round_region_is_not_a_tube_axis() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False)
    circle = np.column_stack((np.cos(theta), np.sin(theta))) * 20.0
    observation = estimate_tube_axis_from_points(circle)
    assert not observation.valid
    assert observation.axis_angle_deg is None


def test_visual_gate_requires_consecutive_target_observations() -> None:
    gate = VisualAngleGate(
        VisualAngleGateConfig(
            target_image_axis_deg=126.0,
            tolerance_deg=3.0,
            min_consecutive_frames=3,
        )
    )
    results = [
        gate.update(timestamp_s=index / 15.0, observation=_observation(angle))
        for index, angle in enumerate((135.0, 128.0, 126.5, 125.0))
    ]
    assert [item.decision for item in results] == [
        "CONTINUE_BOUNDED_PIVOT",
        "CONTINUE_BOUNDED_PIVOT",
        "CONTINUE_BOUNDED_PIVOT",
        "RECLAMP",
    ]
    assert results[-1].accepted


def test_missing_axis_resets_target_debounce() -> None:
    gate = VisualAngleGate(
        VisualAngleGateConfig(target_image_axis_deg=126.0, min_consecutive_frames=2)
    )
    first = gate.update(timestamp_s=0.0, observation=_observation(126.0))
    missing = TubeAxisObservation(None, None, 0.0, np.inf, 0.0, 0, False)
    second = gate.update(timestamp_s=0.1, observation=missing)
    third = gate.update(timestamp_s=0.2, observation=_observation(126.0))
    assert first.consecutive_in_band == 1
    assert second.decision == "HOLD_AND_REOBSERVE"
    assert third.consecutive_in_band == 1


def test_implausible_image_axis_jump_is_rejected() -> None:
    gate = VisualAngleGate(
        VisualAngleGateConfig(target_image_axis_deg=126.0, max_rate_deg_s=50.0)
    )
    gate.update(timestamp_s=0.0, observation=_observation(126.0))
    result = gate.update(timestamp_s=0.01, observation=_observation(100.0))
    assert result.decision == "HOLD_AND_REOBSERVE"
    assert not result.accepted


def test_timestamp_must_increase() -> None:
    gate = VisualAngleGate(VisualAngleGateConfig(target_image_axis_deg=126.0))
    gate.update(timestamp_s=1.0, observation=_observation(126.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        gate.update(timestamp_s=1.0, observation=_observation(126.0))
