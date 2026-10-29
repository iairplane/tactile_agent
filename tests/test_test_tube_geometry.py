from __future__ import annotations

import numpy as np
import pytest

from demo2gripper.test_tube_geometry import (
    GEOMETRY_CONFIG_SCHEMA,
    axial_fraction,
    compile_geometry_contracts,
    depth_proxy_summary,
    fixed_length_endpoint_solutions,
    image_axis_angle_deg,
    projective_axis_plane,
)


def _config() -> dict:
    return {
        "schema": GEOMETRY_CONFIG_SCHEMA,
        "task_id": "test",
        "review": {
            "source_grasp": {
                "frame": 12,
                "closed_end_uv": [10, 90],
                "open_end_uv": [10, 10],
                "pinch_centre_uv": [12, 30],
                "pinch_centre_uncertainty_px": 4,
            },
            "live_source": {
                "live_review_frame": 1,
                "closed_end_uv": [10, 90],
                "open_end_uv": [10, 10],
                "axis_endpoint_uncertainty_px": 5,
                "distractor_axis_uv": [[20, 80], [80, 40]],
                "depth_proxy": {
                    "window_radius_px": 1,
                    "accepted_range_m": [0.3, 0.8],
                },
                "open_hand_depth_proxy": {
                    "demo_frame": 12,
                    "uv": [10, 10],
                    "window_radius_px": 1,
                    "accepted_range_m": [0.3, 0.8],
                },
            },
            "socket_reference": {
                "demo_terminal_frame": 30,
                "closed_end_uv": [20, 80],
                "open_end_uv": [80, 40],
                "socket_entry_uv": [20, 80],
                "socket_entry_uncertainty_px": 5,
                "axis_endpoint_uncertainty_px": 5,
                "depth_proxy": {
                    "window_radius_px": 1,
                    "accepted_range_m": [0.3, 0.8],
                },
                "open_hand_depth_proxy": {
                    "demo_frame": 30,
                    "uv": [80, 40],
                    "window_radius_px": 1,
                    "accepted_range_m": [0.3, 0.8],
                },
            },
        },
        "assumptions": {
            "tube_length_nominal_m": 0.11,
            "tube_length_uncertainty_m": 0.02,
            "tube_diameter_nominal_m": 0.015,
            "tube_diameter_uncertainty_m": 0.003,
        },
    }


def test_axial_fraction_preserves_dimensionless_grasp_relation() -> None:
    fraction, offset = axial_fraction([0, 10], [0, 0], [2, 2.5])

    assert fraction == pytest.approx(0.75)
    assert abs(offset) == pytest.approx(2.0)
    assert image_axis_angle_deg([0, 10], [0, 0]) == pytest.approx(90.0)


def test_projective_plane_contains_both_camera_rays() -> None:
    K = np.array([[500.0, 0.0, 50.0], [0.0, 500.0, 50.0], [0.0, 0.0, 1.0]])
    result = projective_axis_plane(K, [20, 80], [80, 40])
    normal = np.asarray(result["normal_camera"])

    assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert normal @ np.asarray(result["start_ray_camera_unit"]) == pytest.approx(0.0)
    assert normal @ np.asarray(result["end_ray_camera_unit"]) == pytest.approx(0.0)


def test_depth_proxy_does_not_assign_object_identity() -> None:
    depth = np.zeros((100, 100), dtype=float)
    depth[79:82, 19:22] = 0.6
    summary = depth_proxy_summary(
        [depth, depth.copy()],
        uv=[20, 80],
        radius_px=1,
        accepted_range_m=(0.3, 0.8),
    )

    assert summary["aggregate"]["accepted_pixels"] == 18
    assert summary["aggregate"]["median_m"] == pytest.approx(0.6)


def test_fixed_length_solver_retains_both_ray_branches() -> None:
    K = np.eye(3)
    solutions = fixed_length_endpoint_solutions(
        K,
        known_uv=[0.0, 0.0],
        known_depth_m=1.0,
        unknown_uv=[0.1, 0.0],
        length_m=0.2,
    )

    assert len(solutions) == 2
    assert all(item["endpoint_distance_m"] == pytest.approx(0.2) for item in solutions)


def test_contracts_keep_metric_transforms_null() -> None:
    config = _config()
    K = np.array([[500.0, 0.0, 50.0], [0.0, 500.0, 50.0], [0.0, 0.0, 1.0]])
    depth = np.zeros((100, 100), dtype=float)
    depth[79:82, 19:22] = 0.6
    contracts = compile_geometry_contracts(
        config=config,
        K=K,
        T_base_camera=np.eye(4),
        live_depth_frames=[depth, depth.copy()],
        socket_depth_frames=[depth],
        source_hand_depth_frames=[depth],
        socket_hand_depth_frames=[depth],
        provenance={"test": True},
    )

    grasp = contracts["tube_grasp_relation"]
    live = contracts["live_tube_pose"]
    socket = contracts["socket_frame"]
    assert grasp["T_object_gripper_tcp"] is None
    assert grasp["robot_candidate_constraints"]["cylindrical_roll"].startswith("free S1")
    assert live["T_base_tube"] is None
    assert live["adjacent_surface_depth_proxy"]["may_be_used_as_T_base_tube_translation"] is False
    assert socket["T_base_socket"] is None
    assert socket["insertion_target_depth_m"] is None
    assert all(contract["execution_authorized"] is False for contract in contracts.values())
