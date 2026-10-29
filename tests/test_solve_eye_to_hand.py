from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import solve_eye_to_hand as solver


def _transform(rotation_vector: list[float], translation: list[float]) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _exact_robot_world_samples() -> tuple[list[solver.Sample], np.ndarray, np.ndarray]:
    T_flange_board = _transform([0.18, -0.11, 0.07], [0.025, -0.018, 0.145])
    T_base_camera = _transform([-0.23, 0.16, 0.09], [0.64, 0.51, 0.28])
    robot_poses = [
        _transform([0.00, 0.00, 0.00], [0.46, 0.08, 0.39]),
        _transform([0.20, 0.00, 0.00], [0.51, 0.09, 0.41]),
        _transform([0.00, -0.24, 0.00], [0.55, 0.13, 0.37]),
        _transform([0.00, 0.00, 0.27], [0.49, 0.18, 0.43]),
        _transform([0.16, -0.19, 0.05], [0.58, 0.16, 0.36]),
        _transform([-0.13, 0.12, -0.21], [0.53, 0.05, 0.45]),
    ]
    samples = []
    for index, T_base_flange in enumerate(robot_poses):
        T_camera_board = (
            np.linalg.inv(T_base_camera) @ T_base_flange @ T_flange_board
        )
        samples.append(
            solver.Sample(
                index=index,
                name=f"{index:03d}",
                directory=Path(f"/{index:03d}"),
                T_base_flange=T_base_flange,
                T_camera_board=T_camera_board,
            )
        )
    return samples, T_flange_board, T_base_camera


def test_robot_world_translation_is_refit_after_rotation_projection(monkeypatch) -> None:
    samples, expected_flange_board, expected_base_camera = _exact_robot_world_samples()
    raw_flange_board_translation = expected_flange_board[:3, 3] + [0.012, -0.009, 0.021]
    raw_base_camera_translation = expected_base_camera[:3, 3] + [-0.018, 0.014, -0.011]

    def fake_calibrate(*args, **kwargs):
        del args, kwargs
        return (
            expected_flange_board[:3, :3],
            raw_flange_board_translation.reshape(3, 1),
            expected_base_camera[:3, :3],
            raw_base_camera_translation.reshape(3, 1),
        )

    monkeypatch.setattr(cv2, "calibrateRobotWorldHandEye", fake_calibrate)
    result = solver.solve_robot_world(samples, "li")

    np.testing.assert_allclose(result.T_flange_board, expected_flange_board, atol=1e-12)
    np.testing.assert_allclose(result.T_base_camera, expected_base_camera, atol=1e-12)
    np.testing.assert_allclose(
        result.raw_T_flange_board[:3, 3], raw_flange_board_translation
    )
    np.testing.assert_allclose(
        result.raw_T_base_camera[:3, 3], raw_base_camera_translation
    )
    assert result.translation_refit["rank"] == 6
    assert np.isfinite(result.translation_refit["condition_number"])
    assert (
        result.translation_refit["equation_residual_translation_mm"]["max"]
        < 1e-9
    )
