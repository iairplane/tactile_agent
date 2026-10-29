from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from demo2gripper.gf220_live import GF220ClassifierWindows
from demo2gripper.gf220_wrench_adapter import (
    GF220WrenchAdapterConfig,
    GF220WrenchSensorCalibration,
)
from demo2gripper.medicine_bottle_tactile_bootstrap import (
    MedicineBottleTactileBootstrapConfig,
    bootstrap_medicine_bottle_tactile,
    bootstrap_medicine_bottle_tactile_presence,
)
from demo2gripper.tactile_contact_classifier import FingerRimClassifierWindow
from demo2gripper.tactile_rim_probe import TimedArrayWindow
from demo2gripper.tactile_twist_supervisor import TactileTwistConfig

SERIAL_LEFT = "sensor-left"
SERIAL_RIGHT = "sensor-right"
COUNT = 8
NOISE = np.array(
    [
        [-0.05, 0.03, -0.04, 0.02, -0.01, 0.03],
        [-0.03, -0.02, 0.01, -0.01, 0.03, -0.02],
        [-0.01, 0.01, 0.03, 0.01, -0.02, 0.01],
        [0.00, -0.03, -0.02, -0.02, 0.01, -0.01],
        [0.01, 0.02, 0.04, 0.03, 0.02, 0.02],
        [0.02, -0.01, -0.03, -0.03, -0.03, -0.03],
        [0.04, 0.04, 0.02, 0.02, 0.04, 0.04],
        [0.06, -0.04, -0.01, -0.01, -0.04, -0.04],
    ]
)


def _provenance(serial: str) -> dict[str, str]:
    return {
        "serial_side_mapping_id": f"{serial}:mapping",
        "task_sensor_rotation_id": "seed-will-be-replaced",
        "contact_position_id": "measured-cap-radius",
        "force6d_bias_id": "seed-will-be-replaced",
        "force6d_axes_units_id": "vendor-force6d-vector-v1",
    }


def _seed() -> GF220WrenchAdapterConfig:
    return GF220WrenchAdapterConfig(
        profile_id="medicine-cap-session-v1",
        task_frame_id="cap-task",
        sensors=(
            GF220WrenchSensorCalibration(
                serial=SERIAL_LEFT,
                logical_side="left",
                calibration_id="left-cal",
                R_task_sensor=np.eye(3),
                contact_position_task_m=np.array([0.02, 0.0, 0.0]),
                force6d_bias=np.zeros(6),
                provenance=_provenance(SERIAL_LEFT),
            ),
            GF220WrenchSensorCalibration(
                serial=SERIAL_RIGHT,
                logical_side="right",
                calibration_id="right-cal",
                R_task_sensor=np.eye(3),
                contact_position_task_m=np.array([-0.02, 0.0, 0.0]),
                force6d_bias=np.zeros(6),
                provenance=_provenance(SERIAL_RIGHT),
            ),
        ),
        min_samples=5,
        max_age_s=0.20,
        max_pair_skew_s=0.02,
    )


def _timed(values: np.ndarray, start_s: float) -> TimedArrayWindow:
    count = len(values)
    host = start_s + 0.01 * np.arange(count)
    sdk = 1000.0 * host
    return TimedArrayWindow(
        values=values,
        sdk_time_s=sdk,
        capture_monotonic_s=host,
        valid=np.ones(count, dtype=bool),
    )


def _finger(calibration_id: str, wrench: np.ndarray, start_s: float) -> FingerRimClassifierWindow:
    images = np.zeros((len(wrench), 2, 2, 3), dtype=np.uint8)
    return FingerRimClassifierWindow(
        calibration_id=calibration_id,
        warped=_timed(images, start_s),
        diff=_timed(images, start_s),
        force6d=_timed(wrench, start_s),
    )


def _pair(
    *,
    left: np.ndarray,
    right: np.ndarray,
    start_s: float,
) -> GF220ClassifierWindows:
    # Reverse physical order to prove the bootstrap keys by serial.
    windows = {
        SERIAL_LEFT: _finger("left-cal", left, start_s),
        SERIAL_RIGHT: _finger("right-cal", right, start_s + 0.001),
    }
    serials = (SERIAL_RIGHT, SERIAL_LEFT)
    end_s = start_s + 0.071
    return GF220ClassifierWindows(
        sensor_names=("right-pad", "left-pad"),
        serials=serials,
        windows=(windows[serials[0]], windows[serials[1]]),
        start_skew_s=0.001,
        end_skew_s=0.001,
        collected_monotonic_s=end_s + 0.005,
    )


def _windows(axial_magnitude: float = 4.0):
    left_bias = np.array([1.0, -2.0, 0.5, 0.2, -0.3, 0.4])
    right_bias = np.array([-0.5, 1.5, -1.0, -0.4, 0.2, -0.1])
    unloaded = _pair(
        left=left_bias + NOISE,
        right=right_bias + 1.1 * NOISE,
        start_s=1.0,
    )
    # Left raw radial is +X.  Right raw radial is +Y and must map to task -X.
    left_grasp = left_bias + np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]) + 1.2 * NOISE
    right_grasp = right_bias + np.array([0.0, 3.5, 0.0, 0.0, 0.0, 0.0]) + 1.3 * NOISE
    grasp = _pair(left=left_grasp, right=right_grasp, start_s=2.0)
    press = _pair(
        left=left_grasp
        + np.array([0.0, 0.0, axial_magnitude, 0.0, 0.0, 0.0])
        + 0.4 * NOISE,
        right=right_grasp
        + np.array([0.0, 0.0, 1.1 * axial_magnitude, 0.0, 0.0, 0.0])
        + 0.5 * NOISE,
        start_s=3.0,
    )
    return unloaded, grasp, press, left_bias, right_bias


def _tactile() -> TactileTwistConfig:
    return TactileTwistConfig(profile_id="medicine-cap-session-v1")


def test_bootstrap_estimates_bias_force_directions_and_dynamic_gates() -> None:
    unloaded, grasp, press, left_bias, right_bias = _windows()

    result = bootstrap_medicine_bottle_tactile(
        unloaded,
        grasp,
        press,
        seed_adapter=_seed(),
        tactile_config=_tactile(),
        bottle_axis_task=np.array([0.0, 0.0, 1.0]),
        axis_point_task_m=np.zeros(3),
    )

    by_side = {sensor.logical_side: sensor for sensor in result.adapter.sensors}
    np.testing.assert_allclose(by_side["left"].force6d_bias, left_bias, atol=0.02)
    np.testing.assert_allclose(by_side["right"].force6d_bias, right_bias, atol=0.03)
    np.testing.assert_allclose(
        by_side["left"].R_task_sensor @ np.array([1.0, 0.0, 0.0]),
        [1.0, 0.0, 0.0],
        atol=0.03,
    )
    np.testing.assert_allclose(
        by_side["right"].R_task_sensor @ np.array([0.0, 1.0, 0.0]),
        [-1.0, 0.0, 0.0],
        atol=0.03,
    )
    for sensor in result.adapter.sensors:
        np.testing.assert_allclose(
            sensor.R_task_sensor @ np.array([0.0, 0.0, 1.0]),
            [0.0, 0.0, 1.0],
            atol=0.03,
        )
        assert "task-session-bootstrap:" in sensor.provenance["force6d_bias_id"]
    assert set(result.contact_normal_threshold_by_side) == {"left", "right"}
    assert all(value > 0.0 for value in result.contact_normal_threshold_by_side.values())
    assert result.release_axis_delta_sensor_units > 0.0
    assert result.release_tangent_delta_sensor_units > 0.0
    assert result.tactile_config.press_engage_finger_z >= _tactile().press_engage_finger_z
    assert result.report["status"] == "qualified_for_this_task_session"
    assert result.report["policy"]["formal_twist"].startswith("fixed geometric")


def test_bootstrap_rejects_weak_axial_identification_signal() -> None:
    unloaded, grasp, press, _, _ = _windows(axial_magnitude=0.01)

    with pytest.raises(ValueError, match="axial-press response SNR"):
        bootstrap_medicine_bottle_tactile(
            unloaded,
            grasp,
            press,
            seed_adapter=_seed(),
            tactile_config=_tactile(),
            bottle_axis_task=np.array([0.0, 0.0, 1.0]),
            axis_point_task_m=np.zeros(3),
        )


def test_bootstrap_rejects_overlapping_stage_evidence() -> None:
    unloaded, grasp, press, _, _ = _windows()
    overlapping = replace(grasp, windows=unloaded.windows, collected_monotonic_s=1.08)

    with pytest.raises(ValueError, match="overlap or are out of order"):
        bootstrap_medicine_bottle_tactile(
            unloaded,
            overlapping,
            press,
            seed_adapter=_seed(),
            tactile_config=_tactile(),
            bottle_axis_task=np.array([0.0, 0.0, 1.0]),
            axis_point_task_m=np.zeros(3),
        )


def test_bootstrap_quality_config_rejects_non_strict_noise_multiplier() -> None:
    with pytest.raises(ValueError, match="release_axis_noise_multiplier"):
        MedicineBottleTactileBootstrapConfig(release_axis_noise_multiplier=1.0)


def test_presence_bootstrap_accepts_collinear_one_pad_fz_changes() -> None:
    left_bias = np.array([1.0, -2.0, 0.5, 0.2, -0.3, 0.4])
    right_bias = np.array([-0.5, 1.5, -1.0, -0.4, 0.2, -0.1])
    unloaded = _pair(
        left=left_bias + NOISE,
        right=right_bias + NOISE,
        start_s=1.0,
    )
    grasp = _pair(
        left=left_bias + np.array([0, 0, 0.30, 0, 0, 0]) + NOISE,
        right=right_bias + NOISE,
        start_s=2.0,
    )
    press = _pair(
        left=left_bias + np.array([0, 0, 0.42, 0, 0, 0]) + NOISE,
        right=right_bias + NOISE,
        start_s=3.0,
    )

    result = bootstrap_medicine_bottle_tactile_presence(
        unloaded,
        grasp,
        press,
        seed_adapter=_seed(),
        tactile_config=_tactile(),
    )

    assert result.report["mode"] == (
        "any_pad_raw_fz_presence_plus_press_aligned_release_proxy"
    )
    assert result.report["observed_any_pad"]["grasp_fz_change"] == pytest.approx(
        0.30, abs=0.02
    )
    assert result.report["observed_any_pad"]["axial_press_fz_change"] == pytest.approx(
        0.12, abs=0.02
    )
    assert result.presence_config.press_required_consecutive_windows == 1
    assert result.contact_presence_threshold_sensor_units > 0.0
    for sensor in result.adapter.sensors:
        np.testing.assert_allclose(sensor.R_task_sensor, np.eye(3))


def test_presence_bootstrap_retries_only_when_no_pad_changes_after_press() -> None:
    left_bias = np.array([1.0, -2.0, 0.5, 0.2, -0.3, 0.4])
    right_bias = np.array([-0.5, 1.5, -1.0, -0.4, 0.2, -0.1])
    unloaded = _pair(
        left=left_bias + NOISE,
        right=right_bias + NOISE,
        start_s=1.0,
    )
    grasp_left = left_bias + np.array([0, 0, 0.30, 0, 0, 0]) + NOISE
    grasp = _pair(left=grasp_left, right=right_bias + NOISE, start_s=2.0)
    press = _pair(left=grasp_left, right=right_bias + NOISE, start_s=3.0)

    with pytest.raises(ValueError) as raised:
        bootstrap_medicine_bottle_tactile_presence(
            unloaded,
            grasp,
            press,
            seed_adapter=_seed(),
            tactile_config=_tactile(),
        )
    assert raised.value.reason_code == "axial_press_tactile_response_absent"
