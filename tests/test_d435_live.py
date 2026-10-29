from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np
import pytest

from demo2gripper.d435_live import (
    D435BackendFrame,
    D435DeviceInfo,
    D435FrameTimeout,
    D435Intrinsics,
    D435LiveConfig,
    D435LiveSource,
)

INTRINSICS = D435Intrinsics(
    width=4,
    height=3,
    fx=600.0,
    fy=601.0,
    ppx=2.0,
    ppy=1.5,
    distortion_model="brown_conrady",
    distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
)


def _frame(number: int, *, sensor_s: float | None = None) -> D435BackendFrame:
    stamp = number / 30.0 if sensor_s is None else sensor_s
    return D435BackendFrame(
        rgb=np.full((3, 4, 3), number, dtype=np.uint8),
        depth_raw=np.full((3, 4), 1000 + number, dtype=np.uint16),
        intrinsics=INTRINSICS,
        color_frame_number=number,
        depth_frame_number=number + 100,
        color_sensor_timestamp_s=stamp,
        depth_sensor_timestamp_s=stamp + 0.001,
        color_sdk_timestamp_s=stamp + 10.0,
        depth_sdk_timestamp_s=stamp + 10.001,
        sdk_timestamp_domain="hardware_clock",
    )


class _Session:
    def __init__(
        self,
        frames: list[D435BackendFrame | BaseException],
        *,
        serial: str = "front-123",
    ) -> None:
        self.frames = deque(frames)
        self._device_info = D435DeviceInfo(serial=serial, name="Intel RealSense D435")
        self.requested_timeouts_ms: list[int] = []
        self.stop_calls = 0

    @property
    def device_info(self) -> D435DeviceInfo:
        return self._device_info

    @property
    def depth_scale_m(self) -> float:
        return 0.001

    def wait_for_aligned_pair(self, timeout_ms: int) -> D435BackendFrame:
        self.requested_timeouts_ms.append(timeout_ms)
        if not self.frames:
            raise TimeoutError("no queued frame")
        value = self.frames.popleft()
        if isinstance(value, BaseException):
            raise value
        return value

    def stop(self) -> None:
        self.stop_calls += 1


class _Backend:
    def __init__(
        self,
        session: _Session,
        *,
        serials: tuple[str, ...] = ("front-123",),
    ) -> None:
        self.session = session
        self.serials = serials
        self.open_calls = 0

    def list_devices(self) -> tuple[D435DeviceInfo, ...]:
        return tuple(D435DeviceInfo(serial=serial) for serial in self.serials)

    def open(self, config: D435LiveConfig) -> _Session:
        self.open_calls += 1
        return self.session


def _config(*, warmup_frames: int = 0) -> D435LiveConfig:
    return D435LiveConfig(
        serial="front-123",
        width=4,
        height=3,
        fps=30,
        warmup_frames=warmup_frames,
        frame_timeout_ms=25,
        max_pair_skew_ms=5.0,
    )


def test_returns_fresh_aligned_metric_frames_with_all_timestamps() -> None:
    session = _Session([_frame(10), _frame(11)])
    source = D435LiveSource(_config(warmup_frames=1), backend=_Backend(session))

    with source as running:
        result = running.read()
        assert result.sequence == 1
        assert result.serial == "front-123"
        assert result.color_frame_number == 11
        assert result.depth_frame_number == 111
        assert result.color_sensor_timestamp_s == pytest.approx(11 / 30)
        assert result.color_sdk_timestamp_s == pytest.approx(10 + 11 / 30)
        assert result.sdk_timestamp_domain == "hardware_clock"
        assert result.host_received_monotonic_s > 0.0
        assert result.sensor_pair_skew_ms == pytest.approx(1.0)
        assert np.array_equal(result.depth_m, result.depth_raw.astype(np.float32) * 0.001)
        assert np.array_equal(result.intrinsics.matrix, INTRINSICS.matrix)
        assert not result.rgb.flags.writeable
        assert not result.depth_raw.flags.writeable
        assert not result.depth_m.flags.writeable

    assert session.stop_calls == 1


def test_backend_timeout_is_bounded_and_reported() -> None:
    session = _Session([TimeoutError("SDK wait expired")])
    source = D435LiveSource(_config(), backend=_Backend(session)).start()
    with pytest.raises(D435FrameTimeout, match="SDK wait expired"):
        source.read(timeout_ms=7)
    assert session.requested_timeouts_ms == [pytest.approx(7, abs=1)]
    source.stop()


@pytest.mark.parametrize(
    ("candidate", "problem"),
    [
        (_frame(20), "color frame number"),
        (_frame(21, sensor_s=0.5), "color hardware timestamp"),
    ],
)
def test_replayed_or_stale_pair_is_never_published(
    candidate: D435BackendFrame, problem: str
) -> None:
    first = _frame(20, sensor_s=1.0)
    session = _Session([first, candidate, TimeoutError("end of fake stream")])
    source = D435LiveSource(_config(), backend=_Backend(session)).start()
    assert source.read().color_frame_number == 20

    with pytest.raises(D435FrameTimeout, match=problem):
        source.read()
    source.stop()


def test_rejected_replay_can_be_followed_by_a_fresh_pair_in_same_read() -> None:
    first = _frame(30, sensor_s=1.0)
    replay = replace(first, rgb=np.full((3, 4, 3), 99, dtype=np.uint8))
    fresh = _frame(31, sensor_s=1.1)
    session = _Session([first, replay, fresh])
    source = D435LiveSource(_config(), backend=_Backend(session)).start()
    assert source.read().color_frame_number == 30
    assert source.read().color_frame_number == 31
    source.stop()


def test_repeated_hardware_timestamps_are_accepted_when_freshness_clocks_advance() -> None:
    first = _frame(30, sensor_s=1.0)
    fresh = replace(
        _frame(31, sensor_s=1.0),
        color_sdk_timestamp_s=first.color_sdk_timestamp_s + 0.1,
        depth_sdk_timestamp_s=first.depth_sdk_timestamp_s + 0.1,
    )
    session = _Session([first, fresh])
    source = D435LiveSource(_config(), backend=_Backend(session)).start()

    assert source.read().color_frame_number == 30
    assert source.read().color_frame_number == 31
    source.stop()


def test_regressed_hardware_timestamp_is_rejected_even_if_other_clocks_advance() -> None:
    first = _frame(30, sensor_s=1.0)
    regressed = replace(
        _frame(31, sensor_s=0.9),
        color_sdk_timestamp_s=first.color_sdk_timestamp_s + 0.1,
        depth_sdk_timestamp_s=first.depth_sdk_timestamp_s + 0.1,
    )
    session = _Session([first, regressed, TimeoutError("end of fake stream")])
    source = D435LiveSource(_config(), backend=_Backend(session)).start()

    assert source.read().color_frame_number == 30
    with pytest.raises(D435FrameTimeout, match="hardware timestamp"):
        source.read()
    source.stop()


def test_rejected_skew_pair_can_be_followed_by_a_fresh_pair_in_same_read() -> None:
    bad_skew = replace(
        _frame(40, sensor_s=1.0),
        depth_sensor_timestamp_s=1.0325,
    )
    fresh = _frame(41, sensor_s=1.1)
    config = replace(_config(), max_pair_skew_ms=20.0)
    session = _Session([bad_skew, fresh])
    source = D435LiveSource(config, backend=_Backend(session)).start()

    result = source.read()

    assert result.color_frame_number == 41
    assert result.sensor_pair_skew_ms == pytest.approx(1.0)
    assert config.max_pair_skew_ms == 20.0
    assert len(session.requested_timeouts_ms) == 2
    source.stop()


def test_missing_or_wrong_active_serial_fails_closed() -> None:
    missing_session = _Session([])
    missing_backend = _Backend(missing_session, serials=("other",))
    with pytest.raises(RuntimeError, match="not found"):
        D435LiveSource(_config(), backend=missing_backend).start()
    assert missing_backend.open_calls == 0

    wrong_session = _Session([], serial="other")
    wrong_backend = _Backend(wrong_session)
    with pytest.raises(RuntimeError, match="opened RealSense serial"):
        D435LiveSource(_config(), backend=wrong_backend).start()
    assert wrong_session.stop_calls == 1


def test_context_manager_stops_once_when_body_raises() -> None:
    session = _Session([_frame(1)])
    source = D435LiveSource(_config(), backend=_Backend(session))
    with pytest.raises(LookupError, match="operator test"):
        with source:
            raise LookupError("operator test")
    source.stop()
    assert session.stop_calls == 1


def test_intrinsics_change_is_a_fatal_contract_error() -> None:
    changed = replace(INTRINSICS, fx=INTRINSICS.fx + 1.0)
    session = _Session([_frame(1), replace(_frame(2), intrinsics=changed)])
    source = D435LiveSource(_config(), backend=_Backend(session)).start()
    source.read()
    with pytest.raises(RuntimeError, match="intrinsics changed"):
        source.read()
    source.stop()
