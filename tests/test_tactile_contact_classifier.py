from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from demo2gripper.tactile_contact_classifier import (
    DualRimClassifierConfig,
    FingerRimClassifierProfile,
    FingerRimClassifierWindow,
    FingerRimRole,
    ForceContactProfile,
    TactileRimClass,
    classify_tactile_rim,
)
from demo2gripper.tactile_rim_probe import TimedArrayWindow

NOW_S = 10.0
SHAPE = (64, 64)
SAMPLE_TIMES = np.array([9.86, 9.90, 9.94, 9.98])


def _timed(values: np.ndarray, *, times: np.ndarray = SAMPLE_TIMES) -> TimedArrayWindow:
    return TimedArrayWindow(
        values=values,
        sdk_time_s=np.arange(len(values), dtype=float) + 100.0,
        capture_monotonic_s=times,
        valid=np.ones(len(values), dtype=bool),
    )


def _band(*rows: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    for start, stop in rows:
        mask[start:stop, 14:50] = True
    return mask


def _images(
    mask: np.ndarray,
    *,
    sample_count: int = 4,
    diff_contact_value: int = 157,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.full((*SHAPE, 3), 100, dtype=np.uint8)
    warped = np.repeat(baseline[None], sample_count, axis=0)
    diff = np.full((sample_count, *SHAPE, 3), 127, dtype=np.uint8)
    warped[:, mask] = 130
    diff[:, mask] = diff_contact_value
    return warped, diff


def _force(loads: list[float]) -> TimedArrayWindow:
    values = np.zeros((len(loads), 6), dtype=float)
    values[:, 0] = loads
    return _timed(values, times=SAMPLE_TIMES[-len(loads) :])


def _profile(
    calibration_id: str,
    role: FingerRimRole,
    *,
    with_force: bool = False,
) -> FingerRimClassifierProfile:
    roi = np.zeros(SHAPE, dtype=bool)
    if role is FingerRimRole.INNER:
        roi[14:30, 8:56] = True
        target_center = np.array([31.5, 21.0])
        max_contact_area_px = 160
        expected_band_count = 1
        multi_rim_min_band_count = 2
        min_band_elongation = 3.0
        min_distinct_band_separation_px = 8.0
    else:
        roi[34:51, 8:56] = True
        target_center = np.array([31.5, 42.0])
        max_contact_area_px = 340
        expected_band_count = 2
        multi_rim_min_band_count = 3
        min_band_elongation = 2.5
        min_distinct_band_separation_px = 4.0
    force = (
        ForceContactProfile(
            inward_normal_sensor=np.array([1.0, 0.0, 0.0]),
            no_contact_max_n=0.2,
            contact_min_n=0.5,
            overload_n=2.0,
            max_load_range_n=0.4,
        )
        if with_force
        else None
    )
    return FingerRimClassifierProfile(
        calibration_id=calibration_id,
        role=role,
        unloaded_warped=np.full((*SHAPE, 3), 100, dtype=np.uint8),
        rim_roi_mask=roi,
        target_center_uv_px=target_center,
        target_tangent_uv=np.array([1.0, 0.0]),
        diff_neutral_value=127.0,
        diff_threshold=10.0,
        warped_delta_threshold=10.0,
        min_contact_area_px=50,
        max_contact_area_px=max_contact_area_px,
        min_contact_frame_fraction=0.75,
        min_warped_support_fraction=0.80,
        min_roi_fraction=0.85,
        min_band_area_px=50,
        expected_band_count=expected_band_count,
        multi_rim_min_band_count=multi_rim_min_band_count,
        min_band_elongation=min_band_elongation,
        max_tangent_error_deg=10.0,
        min_distinct_band_separation_px=min_distinct_band_separation_px,
        max_center_error_px=4.0,
        max_center_jitter_px=2.0,
        force=force,
    )


def _config(*, with_force: bool = False) -> DualRimClassifierConfig:
    return DualRimClassifierConfig(
        profile_id="synthetic-rim-v1",
        left=_profile("left-v1", FingerRimRole.INNER, with_force=with_force),
        right=_profile("right-v1", FingerRimRole.OUTER, with_force=with_force),
    )


def _window(
    calibration_id: str,
    mask: np.ndarray,
    *,
    force: TimedArrayWindow | None = None,
    warped_override: np.ndarray | None = None,
    diff_contact_value: int = 157,
    times: np.ndarray = SAMPLE_TIMES,
) -> FingerRimClassifierWindow:
    warped, diff = _images(mask, diff_contact_value=diff_contact_value)
    if warped_override is not None:
        warped = warped_override
    return FingerRimClassifierWindow(
        calibration_id=calibration_id,
        warped=_timed(warped, times=times),
        diff=_timed(diff, times=times),
        force6d=force,
    )


def test_sdk_neutral_127_is_image_only_no_contact() -> None:
    empty = np.zeros(SHAPE, dtype=bool)
    result = classify_tactile_rim(
        _window("left-v1", empty),
        _window("right-v1", empty),
        _config(),
        now_s=NOW_S,
    )

    assert result.label is TactileRimClass.NO_CONTACT
    assert not result.force_used


def test_negative_direction_diff_is_also_detected() -> None:
    inner = _band((20, 23))
    outer = _band((37, 41), (44, 48))
    result = classify_tactile_rim(
        _window("left-v1", inner, diff_contact_value=97),
        _window("right-v1", outer, diff_contact_value=97),
        _config(),
        now_s=NOW_S,
    )

    assert result.label is TactileRimClass.INNER_OUTER_RIM_MATCH


def test_independent_inner_outer_templates_match_despite_unequal_area() -> None:
    inner = _band((20, 23))
    outer = _band((37, 41), (44, 48))
    result = classify_tactile_rim(
        _window("left-v1", inner),
        _window("right-v1", outer),
        _config(),
        now_s=NOW_S,
    )

    assert result.label is TactileRimClass.INNER_OUTER_RIM_MATCH
    assert result.left.expected_role is FingerRimRole.INNER
    assert result.right.expected_role is FingerRimRole.OUTER
    assert result.left.band_count == 1
    assert result.right.band_count == 2
    assert result.left.role_matched and result.right.role_matched
    assert result.right.patch_area_px > 2 * result.left.patch_area_px


def test_probe_hypothesis_can_swap_roles_between_physical_fingers() -> None:
    inner = _band((20, 23))
    outer = _band((37, 41), (44, 48))
    config = DualRimClassifierConfig(
        profile_id="synthetic-rim-swapped-v1",
        left=_profile("left-v1", FingerRimRole.OUTER),
        right=_profile("right-v1", FingerRimRole.INNER),
    )
    result = classify_tactile_rim(
        _window("left-v1", outer),
        _window("right-v1", inner),
        config,
        now_s=NOW_S,
    )

    assert result.label is TactileRimClass.INNER_OUTER_RIM_MATCH
    assert result.left.expected_role is FingerRimRole.OUTER
    assert result.right.expected_role is FingerRimRole.INNER
    assert result.evidence_key is not None
    assert result.evidence_key[:5] == (
        "synthetic-rim-swapped-v1",
        "left-v1",
        "outer",
        "right-v1",
        "inner",
    )


def test_config_requires_one_inner_and_one_outer_role() -> None:
    inner = _profile("left-v1", FingerRimRole.INNER)
    with pytest.raises(ValueError, match="exactly one INNER and one OUTER"):
        DualRimClassifierConfig(
            profile_id="invalid-duplicate-role",
            left=inner,
            right=_profile("right-v1", FingerRimRole.INNER),
        )


def test_separated_parallel_bands_are_multi_rim_pattern() -> None:
    outer = _band((37, 41), (44, 48))
    double = _band((16, 19), (25, 28))
    result = classify_tactile_rim(
        _window("left-v1", double),
        _window("right-v1", outer),
        _config(),
        now_s=NOW_S,
    )

    assert result.label is TactileRimClass.MULTI_RIM
    assert result.left.band_count == 2


def test_one_sided_contact_is_role_mismatch() -> None:
    inner = _band((20, 23))
    empty = np.zeros(SHAPE, dtype=bool)
    result = classify_tactile_rim(
        _window("left-v1", inner),
        _window("right-v1", empty),
        _config(),
        now_s=NOW_S,
    )

    assert result.label is TactileRimClass.ROLE_MISMATCH


def test_contact_on_wrong_role_template_is_role_mismatch() -> None:
    inner = _band((20, 23))
    result = classify_tactile_rim(
        _window("left-v1", inner),
        _window("right-v1", inner),
        _config(),
        now_s=NOW_S,
    )

    assert result.label is TactileRimClass.ROLE_MISMATCH
    assert not result.right.role_matched


def test_warped_and_diff_disagreement_is_unknown() -> None:
    inner = _band((20, 23))
    outer = _band((37, 41), (44, 48))
    unchanged_warped = np.full((4, *SHAPE, 3), 100, dtype=np.uint8)
    result = classify_tactile_rim(
        _window("left-v1", inner, warped_override=unchanged_warped),
        _window("right-v1", outer),
        _config(),
        now_s=NOW_S,
    )

    assert result.label is TactileRimClass.UNKNOWN


def test_optional_force_disagreement_is_unknown() -> None:
    inner = _band((20, 23))
    outer = _band((37, 41), (44, 48))
    contact_force = _force([0.75, 0.80, 0.78])
    no_force = _force([0.02, 0.03, 0.01])
    result = classify_tactile_rim(
        _window("left-v1", inner, force=contact_force),
        _window("right-v1", outer, force=no_force),
        _config(with_force=True),
        now_s=NOW_S,
    )

    assert result.label is TactileRimClass.UNKNOWN
    assert result.force_used


def test_unequal_role_forces_do_not_fail_without_a_calibrated_ratio_gate() -> None:
    inner = _band((20, 23))
    outer = _band((37, 41), (44, 48))
    result = classify_tactile_rim(
        _window("left-v1", inner, force=_force([1.48, 1.50, 1.49])),
        _window("right-v1", outer, force=_force([0.50, 0.52, 0.51])),
        _config(with_force=True),
        now_s=NOW_S,
    )

    assert result.label is TactileRimClass.INNER_OUTER_RIM_MATCH


def test_stale_or_duplicate_windows_fail_to_unknown() -> None:
    inner = _band((20, 23))
    outer = _band((37, 41), (44, 48))
    stale_times = SAMPLE_TIMES - 1.0
    result = classify_tactile_rim(
        _window("left-v1", inner, times=stale_times),
        _window("right-v1", outer),
        _config(),
        now_s=NOW_S,
    )
    assert result.label is TactileRimClass.UNKNOWN

    warped, diff = _images(inner)
    duplicate = TimedArrayWindow(
        values=warped,
        sdk_time_s=np.ones(4),
        capture_monotonic_s=SAMPLE_TIMES,
        valid=np.ones(4, dtype=bool),
    )
    left = replace(
        _window("left-v1", inner),
        warped=duplicate,
        diff=replace(duplicate, values=diff),
    )
    duplicate_result = classify_tactile_rim(
        left,
        _window("right-v1", outer),
        _config(),
        now_s=NOW_S,
    )
    assert duplicate_result.label is TactileRimClass.UNKNOWN
