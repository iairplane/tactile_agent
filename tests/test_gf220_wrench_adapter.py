from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from demo2gripper.gf220_live import GF220ClassifierWindows
from demo2gripper.gf220_wrench_adapter import (
    GF220WrenchAdapterConfig,
    GF220WrenchSensorCalibration,
    adapt_gf220_force6d_windows,
)
from demo2gripper.tactile_contact_classifier import FingerRimClassifierWindow
from demo2gripper.tactile_rim_probe import TimedArrayWindow

NOW_S = 10.0
SDK_TIME = np.array([100.0, 101.0, 102.0, 103.0])
HOST_TIME = np.array([9.84, 9.89, 9.94, 9.99])


def _provenance(prefix: str) -> dict[str, str]:
    return {
        "serial_side_mapping_id": f"{prefix}-side-v1",
        "task_sensor_rotation_id": f"{prefix}-rotation-v1",
        "contact_position_id": f"{prefix}-position-v1",
        "force6d_bias_id": f"{prefix}-bias-v1",
        "force6d_axes_units_id": f"{prefix}-axes-units-v1",
    }


def _calibration(
    serial: str,
    side: str,
    *,
    rotation: np.ndarray | None = None,
    position: np.ndarray | None = None,
    bias: np.ndarray | None = None,
) -> GF220WrenchSensorCalibration:
    return GF220WrenchSensorCalibration(
        serial=serial,
        logical_side=side,
        calibration_id=f"{serial}-cal",
        R_task_sensor=np.eye(3) if rotation is None else rotation,
        contact_position_task_m=(
            np.array([0.02, 0.0, 0.0]) if position is None else position
        ),
        force6d_bias=np.zeros(6) if bias is None else bias,
        provenance=_provenance(serial),
    )


def _config() -> GF220WrenchAdapterConfig:
    left_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return GF220WrenchAdapterConfig(
        profile_id="medicine-cap-force-v1",
        task_frame_id="cap_task",
        sensors=(
            _calibration(
                "serial-left",
                "left",
                rotation=left_rotation,
                position=np.array([0.10, 0.0, 0.0]),
                bias=np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5]),
            ),
            _calibration(
                "serial-right",
                "right",
                position=np.array([-0.10, 0.0, 0.0]),
            ),
        ),
        min_samples=3,
        max_age_s=0.20,
        max_pair_skew_s=0.02,
    )


def _timed(values: np.ndarray, *, sdk=SDK_TIME, host=HOST_TIME) -> TimedArrayWindow:
    return TimedArrayWindow(
        values=values,
        sdk_time_s=np.asarray(sdk, dtype=float),
        capture_monotonic_s=np.asarray(host, dtype=float),
        valid=np.ones(len(values), dtype=bool),
    )


def _finger(calibration_id: str, force6d: np.ndarray) -> FingerRimClassifierWindow:
    count = len(force6d)
    images = np.zeros((count, 2, 2, 3), dtype=np.uint8)
    return FingerRimClassifierWindow(
        calibration_id=calibration_id,
        warped=_timed(images),
        diff=_timed(images),
        force6d=_timed(force6d),
    )


def _source(
    *,
    left_force: np.ndarray | None = None,
    right_force: np.ndarray | None = None,
    serials: tuple[str, str] = ("serial-right", "serial-left"),
) -> GF220ClassifierWindows:
    left = np.tile(np.array([2.0, 2.0, 3.0, 0.5, 0.5, 2.5]), (4, 1))
    right = np.tile(np.array([0.0, -1.0, 0.0, 0.0, 0.0, 2.0]), (4, 1))
    windows_by_serial = {
        "serial-left": _finger(
            "serial-left-cal", left if left_force is None else left_force
        ),
        "serial-right": _finger(
            "serial-right-cal", right if right_force is None else right_force
        ),
    }
    return GF220ClassifierWindows(
        sensor_names=("sensor-0", "sensor-1"),
        serials=serials,
        windows=(windows_by_serial[serials[0]], windows_by_serial[serials[1]]),
        start_skew_s=0.0,
        end_skew_s=0.0,
        collected_monotonic_s=NOW_S,
    )


def test_adapts_by_serial_and_leaves_force_moment_for_supervisor() -> None:
    result = adapt_gf220_force6d_windows(_source(), _config(), now_s=NOW_S)

    assert result.left_serial == "serial-left"
    assert result.right_serial == "serial-right"
    assert result.left.calibration_id == "serial-left-cal"
    assert result.right.calibration_id == "serial-right-cal"
    assert result.left.task_frame_id == result.right.task_frame_id == "cap_task"
    # Left raw-bias force [1,0,0] rotates to [0,1,0].  The adapter rotates the
    # intrinsic torque [0,0,2] but deliberately leaves r x F to the supervisor.
    np.testing.assert_allclose(
        result.left.wrench_task[:, :3], np.tile([0.0, 1.0, 0.0], (4, 1))
    )
    np.testing.assert_allclose(
        result.left.wrench_task[:, 3:], np.tile([0.0, 0.0, 2.0], (4, 1))
    )
    # The contact positions remain available for the supervisor's one r x F.
    np.testing.assert_allclose(
        result.right.wrench_task[:, :3], np.tile([0.0, -1.0, 0.0], (4, 1))
    )
    np.testing.assert_allclose(
        result.right.wrench_task[:, 3:], np.tile([0.0, 0.0, 2.0], (4, 1))
    )
    np.testing.assert_allclose(
        result.left.contact_position_task_m,
        np.tile([0.10, 0.0, 0.0], (4, 1)),
    )
    np.testing.assert_allclose(
        result.right.contact_position_task_m,
        np.tile([-0.10, 0.0, 0.0], (4, 1)),
    )
    np.testing.assert_array_equal(result.left.sdk_time_s, SDK_TIME)
    np.testing.assert_array_equal(result.left.capture_monotonic_s, HOST_TIME)
    assert not result.left.wrench_task.flags.writeable
    assert result.provenance_by_serial["serial-left"]["force6d_bias_id"] == (
        "serial-left-bias-v1"
    )
    with pytest.raises(TypeError):
        result.provenance_by_serial["serial-left"]["force6d_bias_id"] = "changed"


@pytest.mark.parametrize(
    ("config_factory", "message"),
    [
        (
            lambda: GF220WrenchAdapterConfig(
                profile_id="p",
                task_frame_id="task",
                sensors=(
                    _calibration("a", "left"),
                    _calibration("b", "left"),
                ),
            ),
            "exactly one left and one right",
        ),
        (
            lambda: GF220WrenchSensorCalibration(
                serial="a",
                logical_side="left",
                calibration_id="cal",
                R_task_sensor=np.diag([1.0, 1.0, 2.0]),
                contact_position_task_m=np.zeros(3),
                force6d_bias=np.zeros(6),
                provenance=_provenance("a"),
            ),
            "orthonormal",
        ),
        (
            lambda: GF220WrenchSensorCalibration(
                serial="a",
                logical_side="left",
                calibration_id="cal",
                R_task_sensor=np.eye(3),
                contact_position_task_m=np.zeros(3),
                force6d_bias=np.zeros(6),
                provenance={"serial_side_mapping_id": "mapping"},
            ),
            "missing required fields",
        ),
    ],
)
def test_config_requires_complete_mapping_rotation_and_provenance(
    config_factory, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config_factory()


def test_source_serials_must_exactly_match_configured_mapping() -> None:
    source = GF220ClassifierWindows(
        sensor_names=("a", "b"),
        serials=("serial-left", "unexpected"),
        windows=_source().windows,
        start_skew_s=0.0,
        end_skew_s=0.0,
        collected_monotonic_s=NOW_S,
    )

    with pytest.raises(ValueError, match="serial-to-side mapping"):
        adapt_gf220_force6d_windows(source, _config(), now_s=NOW_S)


def test_calibration_id_mismatch_fails_closed() -> None:
    source = _source()
    wrong_window = replace(source.windows[1], calibration_id="wrong")
    changed = replace(source, windows=(source.windows[0], wrong_window))

    with pytest.raises(ValueError, match="calibration_id"):
        adapt_gf220_force6d_windows(changed, _config(), now_s=NOW_S)


def test_missing_or_invalid_force_window_fails_closed() -> None:
    source = _source()
    missing = replace(source.windows[1], force6d=None)
    with pytest.raises(ValueError, match="FORCE6D window is missing"):
        adapt_gf220_force6d_windows(
            replace(source, windows=(source.windows[0], missing)),
            _config(),
            now_s=NOW_S,
        )

    invalid_force = replace(
        source.windows[1].force6d,
        valid=np.array([True, True, False, True]),
    )
    invalid = replace(source.windows[1], force6d=invalid_force)
    with pytest.raises(ValueError, match="invalid samples"):
        adapt_gf220_force6d_windows(
            replace(source, windows=(source.windows[0], invalid)),
            _config(),
            now_s=NOW_S,
        )


def test_stale_repeated_or_desynchronized_force_fails_closed() -> None:
    source = _source()
    left_force = source.windows[1].force6d
    assert left_force is not None

    stale = replace(
        left_force,
        capture_monotonic_s=np.array([9.60, 9.65, 9.70, 9.75]),
    )
    with pytest.raises(ValueError, match="stale"):
        adapt_gf220_force6d_windows(
            replace(source, windows=(source.windows[0], replace(source.windows[1], force6d=stale))),
            _config(),
            now_s=NOW_S,
        )

    repeated = replace(left_force, sdk_time_s=np.array([1.0, 2.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="strictly increase"):
        adapt_gf220_force6d_windows(
            replace(
                source,
                windows=(source.windows[0], replace(source.windows[1], force6d=repeated)),
            ),
            _config(),
            now_s=NOW_S,
        )

    shifted = replace(
        left_force,
        capture_monotonic_s=left_force.capture_monotonic_s + 0.04,
    )
    shifted_source = replace(
        source,
        windows=(
            source.windows[0],
            replace(
                source.windows[1],
                force6d=shifted,
                warped=replace(
                    source.windows[1].warped,
                    capture_monotonic_s=shifted.capture_monotonic_s,
                ),
                diff=replace(
                    source.windows[1].diff,
                    capture_monotonic_s=shifted.capture_monotonic_s,
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="max_pair_skew_s"):
        adapt_gf220_force6d_windows(shifted_source, _config(), now_s=NOW_S + 0.04)


def test_force_shape_and_finiteness_are_strict() -> None:
    source = _source()
    malformed = np.zeros((4, 5), dtype=float)
    malformed_window = replace(source.windows[1].force6d, values=malformed)
    with pytest.raises(ValueError, match=r"shape \(N, 6\)"):
        adapt_gf220_force6d_windows(
            replace(
                source,
                windows=(
                    source.windows[0],
                    replace(source.windows[1], force6d=malformed_window),
                ),
            ),
            _config(),
            now_s=NOW_S,
        )

    nonfinite = np.zeros((4, 6), dtype=float)
    nonfinite[2, 1] = np.nan
    nonfinite_window = replace(source.windows[1].force6d, values=nonfinite)
    with pytest.raises(ValueError, match="must be finite"):
        adapt_gf220_force6d_windows(
            replace(
                source,
                windows=(
                    source.windows[0],
                    replace(source.windows[1], force6d=nonfinite_window),
                ),
            ),
            _config(),
            now_s=NOW_S,
        )
