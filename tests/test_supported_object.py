from __future__ import annotations

import numpy as np
import pytest

from demo2gripper.supported_object import (
    SupportedObjectConfig,
    conservative_projected_width,
    localize_supported_object,
)


def intrinsics() -> np.ndarray:
    return np.asarray(
        [
            [100.0, 0.0, 31.5],
            [0.0, 100.0, 31.5],
            [0.0, 0.0, 1.0],
        ]
    )


def config(**changes: object) -> SupportedObjectConfig:
    values: dict[str, object] = {
        "workspace_min_base_m": np.asarray([0.0, -0.5, 0.7]),
        "workspace_max_base_m": np.asarray([1.0, 0.5, 1.2]),
        "support_top_base_z_m": 0.8,
        "min_height_above_support_m": 0.05,
        "min_component_points": 50,
        "min_dominance_ratio": 1.8,
        "morphology_radius_px": 0,
        "min_frames": 3,
        "max_center_deviation_m": 0.01,
        "max_point_count_cv": 0.1,
    }
    values.update(changes)
    return SupportedObjectConfig(**values)  # type: ignore[arg-type]


def depth_with_square(
    *, left: int = 22, top: int = 20, width: int = 20, object_depth: float = 1.0
) -> np.ndarray:
    depth = np.full((64, 64), 0.8, dtype=np.float32)
    depth[top : top + width, left : left + width] = object_depth
    return depth


def test_localizer_backprojects_to_base_and_aggregates_surface_center() -> None:
    T_base_camera = np.eye(4)
    T_base_camera[:3, 3] = [0.4, 0.0, 0.0]
    depths = [depth_with_square(object_depth=value) for value in (1.0, 1.001, 0.999)]

    result = localize_supported_object(depths, intrinsics(), T_base_camera, config())

    assert result.center_base_m == pytest.approx([0.4, -0.02, 1.0], abs=1e-6)
    assert [frame.selected.point_count for frame in result.frames] == [400, 400, 400]
    assert np.count_nonzero(result.final_mask) == 400
    assert result.bounds_min_base_m[0] < result.center_base_m[0]
    assert result.bounds_max_base_m[0] > result.center_base_m[0]
    assert result.center_max_deviation_m == pytest.approx(0.001, abs=1e-6)
    assert result.frames[0].selected.points_base_m.shape == (400, 3)


def test_projected_width_uses_axis_projection_and_largest_robust_frame() -> None:
    angle = np.deg2rad(30.0)
    axis = np.asarray([np.cos(angle), np.sin(angle), 0.0])
    across = np.asarray([-np.sin(angle), np.cos(angle), 0.0])
    grid_x, grid_y = np.meshgrid(np.linspace(-0.02, 0.02, 101), np.linspace(-0.01, 0.01, 51))
    complete = grid_x.reshape(-1, 1) * axis + grid_y.reshape(-1, 1) * across
    occluded = complete[grid_x.ravel() <= 0.015]

    width, spans = conservative_projected_width([occluded, complete], axis)

    assert spans[0] < spans[1]
    assert width == pytest.approx(0.04, abs=1e-6)
    aabb_extent = np.ptp(complete, axis=0)
    assert float(np.dot(np.abs(axis), aabb_extent)) > width


def test_localizer_rejects_two_similarly_sized_components() -> None:
    depth = np.full((64, 64), 0.8, dtype=np.float32)
    depth[10:22, 8:20] = 1.0
    depth[38:50, 42:54] = 1.0
    T_base_camera = np.eye(4)
    T_base_camera[:3, 3] = [0.4, 0.0, 0.0]

    with pytest.raises(ValueError, match="ambiguous"):
        localize_supported_object([depth] * 3, intrinsics(), T_base_camera, config())


def test_localizer_rejects_component_that_moves_between_frames() -> None:
    T_base_camera = np.eye(4)
    T_base_camera[:3, 3] = [0.4, 0.0, 0.0]
    depths = [
        depth_with_square(left=10),
        depth_with_square(left=22),
        depth_with_square(left=34),
    ]

    with pytest.raises(ValueError, match="maximum center deviation"):
        localize_supported_object(depths, intrinsics(), T_base_camera, config())
