import numpy as np
import pytest

from demo2gripper.test_tube_tangential import tangential_direction


def observe(forces, rotations=None, wrist=None):
    return tangential_direction(
        forces, [[1, 0, 0], [1, 0, 0]],
        rotations if rotations is not None else [np.eye(3), np.eye(3)],
        np.eye(3) if wrist is None else wrist,
        min_magnitude=0.01, min_cancellation_ratio=0.1,
    )


def test_normal_squeeze_does_not_change_tangent_direction():
    result = observe([[100, 1, 1], [-80, 1, 1]])
    assert result.force_base == pytest.approx([0, 2, 2])
    assert result.elevation_deg == pytest.approx(45)


def test_opposed_mounts_are_rotated_before_sum():
    result = observe([[2, 0, 1], [2, 0, -1]],
                     [np.eye(3), np.diag([-1, 1, -1])])
    assert result.force_base == pytest.approx([0, 0, 2])


def test_wrist_rotation_changes_base_direction():
    result = observe([[1, 1, 0], [1, 1, 0]],
                     wrist=[[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    assert result.elevation_deg == pytest.approx(90)


def test_cancelled_or_absent_signal_has_no_angle():
    for forces in ([[0, 1, 0], [0, -1, 0]], [[5, 0, 0], [5, 0, 0]]):
        result = observe(forces)
        assert not result.valid
        assert result.elevation_deg is None


def test_reflection_is_not_a_sensor_rotation():
    with pytest.raises(ValueError):
        observe([[0, 1, 1], [0, 1, 1]], [np.eye(3), np.diag([1, 1, -1])])
