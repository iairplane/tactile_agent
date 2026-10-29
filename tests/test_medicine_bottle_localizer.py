from __future__ import annotations

import numpy as np
import pytest

from demo2gripper.medicine_bottle import (
    BOTTLE_CAP_FRAME_SCHEMA,
    parse_bottle_cap_frame_payload,
)
from demo2gripper.medicine_bottle_localizer import (
    LOCALIZATION_SCHEMA,
    CurrentBottleCapConfig,
    draw_bottle_cap_overlay,
    localize_current_bottle_cap,
)


def _scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.full((100, 120, 3), 230, dtype=np.uint8)
    rgb[30:55, 50:70] = 20
    depth = np.zeros((100, 120), dtype=np.float32)
    depth[30:55, 50:70] = 1.0
    intrinsics = np.array(
        [[500.0, 0.0, 60.0], [0.0, 500.0, 50.0], [0.0, 0.0, 1.0]]
    )
    T_base_camera = np.eye(4)
    T_base_camera[:3, 3] = [0.5, 0.1, -0.8]
    return rgb, depth, intrinsics, T_base_camera


def _config(**changes: object) -> CurrentBottleCapConfig:
    values: dict[str, object] = {
        "roi_xyxy_px": (40, 20, 80, 70),
        "max_dark_value": 80,
        "min_component_pixels": 100,
        "morphology_radius_px": 0,
        "min_cap_diameter_m": 0.010,
        "max_cap_diameter_m": 0.070,
        "robust_quantile": 0.0,
    }
    values.update(changes)
    return CurrentBottleCapConfig(**values)  # type: ignore[arg-type]


def test_localizer_emits_strict_review_only_cap_frame() -> None:
    rgb, depth, intrinsics, T_base_camera = _scene()

    result = localize_current_bottle_cap(
        rgb, depth, intrinsics, T_base_camera, _config()
    )

    assert result.bbox_xywh_px == (50, 30, 20, 25)
    assert result.point_count == 500
    assert result.valid_depth_ratio == pytest.approx(1.0)
    assert result.estimated_cap_diameter_m == pytest.approx(0.048)
    np.testing.assert_allclose(result.cap_center_base_m, [0.499, 0.084, 0.2])
    assert result.frame.bottle_axis_base == (0.0, 0.0, 1.0)
    assert result.frame.bottle_axis_tcp == (0.0, 0.0, 1.0)
    assert result.frame.as_dict()["schema"] == BOTTLE_CAP_FRAME_SCHEMA
    parsed = parse_bottle_cap_frame_payload(
        result.frame.as_dict(), expected_base_frame="fr3_link0"
    )
    np.testing.assert_allclose(parsed.T_base_cap_tcp[:3, 3], result.cap_center_base_m)

    report = result.report_dict()
    assert report["schema"] == LOCALIZATION_SCHEMA
    assert report["status"] == "review_only"
    assert report["execution_authorized"] is False
    assert report["hardware_execution_enabled"] is False
    assumptions = report["assumptions"]
    assert assumptions["bottle_axis_base_is_positive_z"] is True
    assert assumptions["cap_yaw_observed"] is False


def test_overlay_marks_result_without_changing_input() -> None:
    rgb, depth, intrinsics, T_base_camera = _scene()
    original = rgb.copy()
    result = localize_current_bottle_cap(
        rgb, depth, intrinsics, T_base_camera, _config()
    )

    overlay = draw_bottle_cap_overlay(rgb, result)

    assert overlay.shape == rgb.shape
    assert overlay.dtype == np.uint8
    assert not np.array_equal(overlay, rgb)
    np.testing.assert_array_equal(rgb, original)


def test_roi_excludes_a_larger_dark_distractor() -> None:
    rgb, depth, intrinsics, T_base_camera = _scene()
    rgb[5:28, 2:35] = 0
    depth[5:28, 2:35] = 0.9

    result = localize_current_bottle_cap(
        rgb, depth, intrinsics, T_base_camera, _config()
    )

    assert result.bbox_xywh_px == (50, 30, 20, 25)
    assert result.eligible_component_count == 1


def test_equal_plausible_components_fail_ambiguity_gate() -> None:
    rgb, depth, intrinsics, T_base_camera = _scene()
    rgb[30:55, 85:105] = 20
    depth[30:55, 85:105] = 1.0
    config = _config(roi_xyxy_px=(35, 20, 110, 70))

    with pytest.raises(ValueError, match="ambiguous"):
        localize_current_bottle_cap(
            rgb, depth, intrinsics, T_base_camera, config
        )


def test_invalid_depth_and_clipped_roi_fail_closed() -> None:
    rgb, depth, intrinsics, T_base_camera = _scene()
    depth[30:55, 50:70] = 0.0
    with pytest.raises(ValueError, match="no black-cap component"):
        localize_current_bottle_cap(
            rgb, depth, intrinsics, T_base_camera, _config()
        )

    rgb, depth, intrinsics, T_base_camera = _scene()
    with pytest.raises(ValueError, match="no black-cap component"):
        localize_current_bottle_cap(
            rgb,
            depth,
            intrinsics,
            T_base_camera,
            _config(roi_xyxy_px=(50, 30, 75, 60)),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"roi_xyxy_px": (0, 0, 121, 100)}, "inside"),
        ({"min_valid_depth_ratio": 0.0}, "min_valid_depth_ratio"),
        ({"min_dominance_ratio": 1.0}, "greater than one"),
        ({"min_cap_diameter_m": 0.1}, "diameter bounds"),
        ({"robust_quantile": 0.25}, "robust_quantile"),
    ],
)
def test_config_validation_fails_closed(changes: dict[str, object], message: str) -> None:
    rgb, depth, intrinsics, T_base_camera = _scene()

    with pytest.raises(ValueError, match=message):
        localize_current_bottle_cap(
            rgb, depth, intrinsics, T_base_camera, _config(**changes)
        )

