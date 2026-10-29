from __future__ import annotations

import numpy as np
import pytest

from demo2gripper.test_tube_tactile_supervisor import (
    RawTactileNoiseModel,
    compensated_residual,
    evaluate_raw_tactile_frame,
    fit_unloaded_raw_model,
    residual_contact_geometry,
    summarize_marker_displacements,
)


def _noise_frames(count: int = 8) -> np.ndarray:
    generator = np.random.default_rng(4)
    baseline = np.full((96, 128, 3), 100, dtype=np.int16)
    noise = generator.integers(-2, 3, size=(count, *baseline.shape), dtype=np.int16)
    return np.clip(baseline + noise, 0, 255).astype(np.uint8)


def test_negative_only_model_rejects_unloaded_noise() -> None:
    frames = _noise_frames()
    model = fit_unloaded_raw_model(frames, min_contact_area_floor_px=32)

    evidence = evaluate_raw_tactile_frame(frames[-1], model)

    assert model.residual_threshold >= 12.0
    assert model.min_contact_area_px >= 32
    assert not evidence.geometry.contact_candidate
    assert not evidence.geometry.orientation_valid
    assert not evidence.slip_shadow_candidate


def test_elongated_residual_patch_reports_raw_image_orientation() -> None:
    residual = np.zeros((80, 100), dtype=float)
    residual[34:42, 20:80] = 30.0
    roi = np.ones_like(residual, dtype=bool)

    result = residual_contact_geometry(
        residual,
        roi,
        residual_threshold=10.0,
        min_contact_area_px=100,
    )

    assert result.contact_candidate
    assert result.orientation_valid
    assert result.area_px == 480
    assert result.orientation_deg == pytest.approx(0.0, abs=0.2)
    assert result.elongation is not None and result.elongation > 5.0


def test_round_patch_cannot_supply_a_direction() -> None:
    yy, xx = np.ogrid[:80, :100]
    residual = np.zeros((80, 100), dtype=float)
    residual[(xx - 50) ** 2 + (yy - 40) ** 2 <= 12**2] = 30.0

    result = residual_contact_geometry(
        residual,
        np.ones_like(residual, dtype=bool),
        residual_threshold=10.0,
        min_contact_area_px=100,
    )

    assert result.contact_candidate
    assert not result.orientation_valid
    assert result.orientation_deg is None


def test_marker_motion_removes_common_camera_translation() -> None:
    xx, yy = np.meshgrid(np.arange(5), np.arange(4))
    reference = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
    observed = reference + np.array([2.0, -3.0])

    result = summarize_marker_displacements(reference, observed)

    assert result.valid
    assert result.global_flow_uv_px == pytest.approx((2.0, -3.0))
    assert result.shear_median_px == pytest.approx(0.0)
    assert result.shear_p90_px == pytest.approx(0.0)


def test_local_marker_deformation_is_separate_from_global_motion() -> None:
    xx, yy = np.meshgrid(np.arange(6), np.arange(5))
    reference = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
    observed = reference + np.array([1.0, 0.0])
    observed[-10:, 1] += 2.0

    result = summarize_marker_displacements(reference, observed)

    assert result.valid
    assert result.global_flow_uv_px == pytest.approx((1.0, 0.0))
    assert result.shear_p90_px == pytest.approx(2.0)


def test_model_and_residual_reject_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="uint8"):
        fit_unloaded_raw_model(_noise_frames().astype(float))
    with pytest.raises(ValueError, match="same shape"):
        compensated_residual(
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.zeros((9, 8, 3), dtype=np.uint8),
            np.ones((8, 8), dtype=bool),
        )
    with pytest.raises(ValueError, match="positive"):
        RawTactileNoiseModel(
            reference_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
            roi_mask=np.ones((8, 8), dtype=bool),
            residual_threshold=0.0,
            min_contact_area_px=2,
            marker_points_uv=np.empty((0, 2)),
            marker_shear_p90_threshold_px=0.5,
            fit_frame_count=2,
        )
