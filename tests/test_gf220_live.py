from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from demo2gripper.gf220_live import (
    GF220Calibration,
    GF220Frame,
    GF220LiveConfig,
    GF220LiveSource,
    GF220SDKTypes,
)

TIME_STAMP = "timestamp"
RAW_IMG = "raw"
WARPED_IMG = "warped"
DIFF_IMG = "diff"
FORCE6D = "force6d"


@dataclass
class _FakeHandle:
    serial: str
    seed: int
    delay_s: float = 0.001
    collect_attempts: int = 0
    collect_calls: int = 0
    transient_collect_failures: int = 0
    persistent_collect_failure: bool = False
    calibrate_calls: int = 0
    warp_calls: int = 0
    release_calls: int = 0
    owner_thread: int | None = None
    access_threads: set[int] = field(default_factory=set)
    requested_types: list[tuple[Any, ...]] = field(default_factory=list)

    def touch(self) -> None:
        thread_id = threading.get_ident()
        self.access_threads.add(thread_id)
        if self.owner_thread is None:
            self.owner_thread = thread_id
        assert thread_id == self.owner_thread


class _FakeImageProcessor:
    def __init__(self, handle: _FakeHandle) -> None:
        self._handle = handle

    def set_warp_params(self, *, corner_points: Any, mode: str) -> None:
        self._handle.touch()
        assert np.asarray(corner_points).shape == (4, 2)
        assert mode == "manual"
        self._handle.warp_calls += 1


class _FakeDevice:
    def __init__(
        self,
        *,
        config: _FakeHandle,
        stream_format: Any,
        output_format: Any,
    ) -> None:
        self._handle = config
        self._handle.touch()
        assert stream_format == {
            "width": 8,
            "height": 6,
            "fps": 30,
            "fourcc": "MJPG",
        }
        assert output_format == "W4_H4"
        self._image_processor = _FakeImageProcessor(config)

    def collect_sensor_data(self, *data_types: Any) -> dict[Any, Any]:
        self._handle.touch()
        self._handle.requested_types.append(data_types)
        time.sleep(self._handle.delay_s)
        self._handle.collect_attempts += 1
        if self._handle.transient_collect_failures > 0:
            self._handle.transient_collect_failures -= 1
            raise OSError("injected transient USB read failure")
        if self._handle.persistent_collect_failure:
            raise OSError("injected persistent USB read failure")
        self._handle.collect_calls += 1
        call = self._handle.collect_calls
        # Every even SDK timestamp deliberately repeats the previous RAW frame.
        # A timestamp-only filter would therefore publish twice as many frames.
        image_version = (call + 1) // 2
        raw_value = (self._handle.seed + image_version) % 251
        warped_value = (self._handle.seed + 2 * image_version) % 251
        sample = {
            TIME_STAMP: np.asarray(call, dtype=np.int64),
            RAW_IMG: np.full((6, 8, 3), raw_value, dtype=np.uint8),
            WARPED_IMG: np.full((4, 4, 3), warped_value, dtype=np.uint8),
            DIFF_IMG: np.full((4, 4, 3), 127 + self._handle.seed, dtype=np.uint8),
            FORCE6D: np.asarray(
                [
                    self._handle.seed,
                    call,
                    -call,
                    call / 10.0,
                    -call / 10.0,
                    self._handle.seed + call / 100.0,
                ],
                dtype=np.float64,
            ),
        }
        return {data_type: sample[data_type] for data_type in data_types}

    def calibrate(self) -> np.ndarray:
        self._handle.touch()
        self._handle.calibrate_calls += 1
        return np.full((4, 4, 3), 80 + self._handle.seed, dtype=np.uint8)

    def release(self) -> None:
        self._handle.touch()
        self._handle.release_calls += 1


class _FakeFinder:
    def __init__(self, handles: dict[str, _FakeHandle]) -> None:
        self._handles = handles

    def get_sns(self) -> list[str]:
        return list(self._handles)

    def get_device_by_sn(self, serial: str) -> _FakeHandle | None:
        return self._handles.get(serial)


def _write_profile(path: Path, *, second_x_offset: int = 0) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "demo2gripper.gf220_manual_warp_profile.v1",
                "status": "test",
                "stream": {
                    "width": 8,
                    "height": 6,
                    "fps": 30,
                    "fourcc": "MJPG",
                },
                "output": {"width": 4, "height": 4},
                "sensors": [
                    {
                        "name": "finger_a",
                        "sdk_serial": "serial-a",
                        "logical_side": "left",
                        "corner_order": ("top_left_top_right_bottom_right_bottom_left"),
                        "corners_xy": [[0, 0], [6, 0], [6, 5], [0, 5]],
                    },
                    {
                        "name": "finger_b",
                        "sdk_serial": "serial-b",
                        "logical_side": "right",
                        "corner_order": ("top_left_top_right_bottom_right_bottom_left"),
                        "corners_xy": [
                            [1 + second_x_offset, 0],
                            [7, 0],
                            [7, 5],
                            [1 + second_x_offset, 5],
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _fake_sdk(handles: dict[str, _FakeHandle]) -> GF220SDKTypes:
    return GF220SDKTypes(
        finder_type=lambda: _FakeFinder(handles),
        device_type=_FakeDevice,
        stream_format_type=lambda **values: values,
        output_profile="W4_H4",
        time_stamp=TIME_STAMP,
        raw_img=RAW_IMG,
        warped_img=WARPED_IMG,
        diff_img=DIFF_IMG,
        force6d=FORCE6D,
    )


def _config(*, max_pair_skew_s: float = 0.05) -> GF220LiveConfig:
    return GF220LiveConfig(
        buffer_frames=8,
        warmup_frames=2,
        warp_validation_frames=1,
        discovery_retries=1,
        cached_poll_delay_s=0.0,
        max_consecutive_sdk_errors=3,
        sdk_error_retry_delay_s=0.0,
        sdk_timestamp_scale_s=0.001,
        max_frame_age_s=0.5,
        max_pair_skew_s=max_pair_skew_s,
        close_timeout_s=1.0,
    )


def test_live_source_rejects_cached_frames_and_honors_not_before(
    tmp_path: Path,
) -> None:
    handles = {
        "serial-a": _FakeHandle("serial-a", seed=1),
        "serial-b": _FakeHandle("serial-b", seed=2),
    }
    source = GF220LiveSource(
        _write_profile(tmp_path / "profile.json"),
        config=_config(),
        sdk_types=_fake_sdk(handles),
    )
    try:
        source.start(timeout_s=1.0)
        calibrations = source.calibrate_unloaded(timeout_s=1.0)
        cutoff = time.monotonic()
        result = source.collect_classifier_windows(cutoff, samples=4, timeout=1.0)

        assert result.serials == ("serial-a", "serial-b")
        assert result.endpoint_skew_s <= source.config.max_pair_skew_s
        for calibration, window in zip(calibrations, result.windows, strict=True):
            assert window.calibration_id == calibration.calibration_id
            assert np.all(window.warped.capture_monotonic_s > cutoff)
            assert np.all(window.diff.capture_monotonic_s > cutoff)
            assert window.force6d is not None
            assert np.all(window.force6d.capture_monotonic_s > cutoff)
            assert window.force6d.values.shape == (4, 6)
            assert window.force6d.values.dtype == np.float64
            assert not window.force6d.values.flags.writeable
            assert np.all(np.diff(window.warped.sdk_time_s) >= 0.002 - 1e-12)
            assert np.array_equal(window.warped.sdk_time_s, window.diff.sdk_time_s)
            assert np.array_equal(window.warped.sdk_time_s, window.force6d.sdk_time_s)

        later_cutoff = time.monotonic()
        later = source.collect_classifier_windows(later_cutoff, samples=3, timeout=1.0)
        assert all(
            np.all(window.warped.capture_monotonic_s > later_cutoff) for window in later.windows
        )
        with pytest.raises(RuntimeError, match="only once"):
            source.calibrate_unloaded(timeout_s=1.0)
    finally:
        source.close()

    for handle in handles.values():
        assert handle.calibrate_calls == 1
        assert handle.warp_calls == 1
        assert handle.release_calls == 1
        assert handle.access_threads == {handle.owner_thread}
        assert handle.owner_thread is not None
        assert any(
            requested == (TIME_STAMP, RAW_IMG, WARPED_IMG, DIFF_IMG, FORCE6D)
            for requested in handle.requested_types
        )


def test_live_source_recovers_from_bounded_consecutive_sdk_errors(
    tmp_path: Path,
) -> None:
    handles = {
        "serial-a": _FakeHandle("serial-a", seed=1),
        "serial-b": _FakeHandle("serial-b", seed=2),
    }
    source = GF220LiveSource(
        _write_profile(tmp_path / "profile.json"),
        config=_config(),
        sdk_types=_fake_sdk(handles),
    )
    try:
        source.start(timeout_s=1.0)
        source.calibrate_unloaded(timeout_s=1.0)
        before_attempts = handles["serial-a"].collect_attempts
        handles["serial-a"].transient_collect_failures = 2
        deadline = time.monotonic() + 1.0
        while (
            handles["serial-a"].transient_collect_failures > 0
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert handles["serial-a"].transient_collect_failures == 0
        cutoff = time.monotonic()
        result = source.collect_classifier_windows(cutoff, samples=3, timeout=1.0)
        assert result.serials == ("serial-a", "serial-b")
        assert handles["serial-a"].collect_attempts >= before_attempts + 3
        assert handles["serial-a"].collect_attempts > handles["serial-a"].collect_calls
    finally:
        source.close()


def test_live_source_fails_closed_after_persistent_sdk_errors(
    tmp_path: Path,
) -> None:
    handles = {
        "serial-a": _FakeHandle("serial-a", seed=1),
        "serial-b": _FakeHandle("serial-b", seed=2),
    }
    source = GF220LiveSource(
        _write_profile(tmp_path / "profile.json"),
        config=_config(),
        sdk_types=_fake_sdk(handles),
    )
    try:
        source.start(timeout_s=1.0)
        source.calibrate_unloaded(timeout_s=1.0)
        handles["serial-a"].persistent_collect_failure = True
        with pytest.raises(
            RuntimeError,
            match=(
                r"GF220 acquisition failed for serial-a: RuntimeError: "
                r"serial-a: collect_sensor_data failed 3 consecutive times; "
                r"last error: OSError: injected persistent USB read failure"
            ),
        ):
            source.collect_classifier_windows(
                time.monotonic(), samples=3, timeout=1.0
            )
    finally:
        source.close()


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"max_consecutive_sdk_errors": 0}, "positive integer"),
        ({"sdk_error_retry_delay_s": -0.001}, "finite and non-negative"),
    ],
)
def test_live_config_rejects_invalid_sdk_retry_bounds(
    values: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GF220LiveConfig(**values)


def test_frame_copies_force6d_as_strict_readonly_vector() -> None:
    source_force = np.arange(6, dtype=np.float32)
    frame = GF220Frame(
        sdk_time_s=1.0,
        capture_monotonic_s=2.0,
        raw_crc32=3,
        raw=np.zeros((6, 8, 3), dtype=np.uint8),
        warped=np.zeros((4, 4, 3), dtype=np.uint8),
        diff=np.zeros((4, 4, 3), dtype=np.uint8),
        force6d=source_force,
    )

    source_force[:] = -1
    assert frame.force6d.shape == (6,)
    assert frame.force6d.dtype == np.float64
    assert np.array_equal(frame.force6d, np.arange(6, dtype=np.float64))
    assert not frame.force6d.flags.writeable

    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        GF220Frame(
            sdk_time_s=1.0,
            capture_monotonic_s=2.0,
            raw_crc32=3,
            raw=np.zeros((6, 8, 3), dtype=np.uint8),
            warped=np.zeros((4, 4, 3), dtype=np.uint8),
            diff=np.zeros((4, 4, 3), dtype=np.uint8),
            force6d=np.zeros((1, 6), dtype=np.float64),
        )


def test_collect_rejects_pairs_outside_skew_bound(tmp_path: Path) -> None:
    handles = {
        "serial-a": _FakeHandle("serial-a", seed=1),
        "serial-b": _FakeHandle("serial-b", seed=2),
    }
    source = GF220LiveSource(
        _write_profile(tmp_path / "profile.json"),
        config=_config(max_pair_skew_s=1e-12),
        sdk_types=_fake_sdk(handles),
    )
    try:
        source.start(timeout_s=1.0)
        source.calibrate_unloaded(timeout_s=1.0)
        with pytest.raises(TimeoutError, match="host skew"):
            source.collect_classifier_windows(time.monotonic(), samples=3, timeout=0.08)
    finally:
        source.close()


def test_calibration_identity_binds_serial_warp_and_baseline() -> None:
    baseline = np.full((4, 4, 3), 90, dtype=np.uint8)
    original = GF220Calibration.create(
        serial="serial-a", warp_sha256="01" * 32, unloaded_warped=baseline
    )
    same = GF220Calibration.create(
        serial="serial-a", warp_sha256="01" * 32, unloaded_warped=baseline
    )
    other_serial = GF220Calibration.create(
        serial="serial-b", warp_sha256="01" * 32, unloaded_warped=baseline
    )
    other_warp = GF220Calibration.create(
        serial="serial-a", warp_sha256="02" * 32, unloaded_warped=baseline
    )
    changed = baseline.copy()
    changed[0, 0, 0] += 1
    other_baseline = GF220Calibration.create(
        serial="serial-a", warp_sha256="01" * 32, unloaded_warped=changed
    )

    assert original.calibration_id == same.calibration_id
    assert (
        len(
            {
                original.calibration_id,
                other_serial.calibration_id,
                other_warp.calibration_id,
                other_baseline.calibration_id,
            }
        )
        == 4
    )
    assert not original.unloaded_warped.flags.writeable


def test_close_before_calibration_releases_each_device_on_its_owner_thread(
    tmp_path: Path,
) -> None:
    handles = {
        "serial-a": _FakeHandle("serial-a", seed=1),
        "serial-b": _FakeHandle("serial-b", seed=2),
    }
    source = GF220LiveSource(
        _write_profile(tmp_path / "profile.json"),
        config=_config(),
        sdk_types=_fake_sdk(handles),
    )
    source.start(timeout_s=1.0)
    source.close()
    source.close()

    for handle in handles.values():
        assert handle.calibrate_calls == 0
        assert handle.release_calls == 1
        assert handle.access_threads == {handle.owner_thread}
