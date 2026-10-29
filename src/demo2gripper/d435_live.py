"""Fail-closed, long-lived aligned RGB-D acquisition from one pinned D435.

Importing this module never imports ``pyrealsense2`` and never opens a device.
The production backend is loaded only by :meth:`D435LiveSource.start`; tests can
inject the small backend protocol below.  The production path only discovers a
device, starts color/depth streams, aligns depth to color, and reads frames.  It
does not change exposure, emitter, advanced-mode, or any other persistent sensor
setting.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class D435FrameTimeout(TimeoutError):
    """No contract-valid fresh RGB-D pair arrived within the requested bound."""


@dataclass(frozen=True)
class D435DeviceInfo:
    """Read-only identity reported by a RealSense device."""

    serial: str
    name: str = ""
    firmware_version: str = ""
    usb_type_descriptor: str = ""
    physical_port: str = ""
    product_id: str = ""
    product_line: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.serial, str) or not self.serial.strip():
            raise ValueError("D435 serial must be a non-empty string")


@dataclass(frozen=True)
class D435Intrinsics:
    """Color optical intrinsics used by both RGB and aligned depth."""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    distortion_model: str
    distortion_coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width < 1:
            raise ValueError("intrinsics width must be a positive integer")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height < 1:
            raise ValueError("intrinsics height must be a positive integer")
        numeric = (self.fx, self.fy, self.ppx, self.ppy, *self.distortion_coefficients)
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("intrinsics values must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("intrinsics focal lengths must be positive")
        if not isinstance(self.distortion_model, str) or not self.distortion_model:
            raise ValueError("distortion_model must be a non-empty string")

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.asarray(
            [
                [self.fx, 0.0, self.ppx],
                [0.0, self.fy, self.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        matrix.setflags(write=False)
        return matrix


@dataclass(frozen=True)
class D435BackendFrame:
    """One depth-to-color aligned pair returned by an injected backend."""

    rgb: np.ndarray
    depth_raw: np.ndarray
    intrinsics: D435Intrinsics
    color_frame_number: int
    depth_frame_number: int
    color_sensor_timestamp_s: float
    depth_sensor_timestamp_s: float
    color_sdk_timestamp_s: float
    depth_sdk_timestamp_s: float
    sdk_timestamp_domain: str


class D435BackendSession(Protocol):
    """One opened SDK pipeline, hidden behind a mockable boundary."""

    @property
    def device_info(self) -> D435DeviceInfo: ...

    @property
    def depth_scale_m(self) -> float: ...

    def wait_for_aligned_pair(self, timeout_ms: int) -> D435BackendFrame: ...

    def stop(self) -> None: ...


class D435Backend(Protocol):
    """Minimal read-only backend used by :class:`D435LiveSource`."""

    def list_devices(self) -> Sequence[D435DeviceInfo]: ...

    def open(self, config: D435LiveConfig) -> D435BackendSession: ...


@dataclass(frozen=True)
class D435LiveConfig:
    """Pinned stream configuration and frame-validity bounds."""

    serial: str
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 15
    frame_timeout_ms: int = 1000
    max_pair_skew_ms: float = 20.0

    def __post_init__(self) -> None:
        if not isinstance(self.serial, str) or not self.serial.strip():
            raise ValueError("serial must be a non-empty string")
        for name in ("width", "height", "fps", "frame_timeout_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.warmup_frames, bool)
            or not isinstance(self.warmup_frames, int)
            or self.warmup_frames < 0
        ):
            raise ValueError("warmup_frames must be a non-negative integer")
        if not math.isfinite(self.max_pair_skew_ms) or self.max_pair_skew_ms <= 0.0:
            raise ValueError("max_pair_skew_ms must be positive and finite")


@dataclass(frozen=True)
class D435Frame:
    """One immutable RGB/aligned-depth observation from the pinned stream."""

    sequence: int
    serial: str
    rgb: np.ndarray
    depth_raw: np.ndarray
    depth_m: np.ndarray
    intrinsics: D435Intrinsics
    color_frame_number: int
    depth_frame_number: int
    color_sensor_timestamp_s: float
    depth_sensor_timestamp_s: float
    color_sdk_timestamp_s: float
    depth_sdk_timestamp_s: float
    sdk_timestamp_domain: str
    host_received_monotonic_s: float

    @property
    def sensor_pair_skew_ms(self) -> float:
        return abs(self.color_sensor_timestamp_s - self.depth_sensor_timestamp_s) * 1e3

    @property
    def sdk_pair_skew_ms(self) -> float:
        return abs(self.color_sdk_timestamp_s - self.depth_sdk_timestamp_s) * 1e3


@dataclass(frozen=True)
class _PairState:
    color_frame_number: int
    depth_frame_number: int
    color_sensor_timestamp_s: float
    depth_sensor_timestamp_s: float
    color_sdk_timestamp_s: float
    depth_sdk_timestamp_s: float
    sdk_timestamp_domain: str
    host_received_monotonic_s: float


class D435LiveSource:
    """Own a RealSense pipeline and return only new, aligned RGB-D pairs.

    A rejected repeated/regressed pair is never exposed to the caller.  The
    source keeps polling within the same per-call timeout and raises
    :class:`D435FrameTimeout` if a valid successor does not arrive.
    """

    def __init__(
        self,
        config: D435LiveConfig,
        *,
        backend: D435Backend | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.config = config
        self._backend = backend
        self._monotonic_ns = monotonic_ns
        self._session: D435BackendSession | None = None
        self._device_info: D435DeviceInfo | None = None
        self._depth_scale_m: float | None = None
        self._intrinsics: D435Intrinsics | None = None
        self._previous: _PairState | None = None
        self._sequence = 0
        self._started = False
        self._closed = False

    @property
    def device_info(self) -> D435DeviceInfo:
        if self._device_info is None:
            raise RuntimeError("D435 source has not started")
        return self._device_info

    @property
    def depth_scale_m(self) -> float:
        if self._depth_scale_m is None:
            raise RuntimeError("D435 source has not started")
        return self._depth_scale_m

    @property
    def intrinsics(self) -> D435Intrinsics:
        if self._intrinsics is None:
            raise RuntimeError("no D435 frame has established intrinsics")
        return self._intrinsics

    def start(self) -> D435LiveSource:
        if self._closed:
            raise RuntimeError("D435 source is closed")
        if self._started:
            return self
        backend = self._backend or PyRealSenseBackend.load_vendor()
        available = tuple(backend.list_devices())
        serials = [device.serial for device in available]
        if self.config.serial not in serials:
            raise RuntimeError(
                f"RealSense serial {self.config.serial!r} not found; available serials: {serials}"
            )

        session: D435BackendSession | None = None
        try:
            session = backend.open(self.config)
            if session.device_info.serial != self.config.serial:
                raise RuntimeError(
                    f"opened RealSense serial {session.device_info.serial!r}; "
                    f"expected {self.config.serial!r}"
                )
            depth_scale_m = float(session.depth_scale_m)
            if not math.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
                raise RuntimeError("D435 depth scale must be positive and finite")
            self._session = session
            self._device_info = session.device_info
            self._depth_scale_m = depth_scale_m
            self._started = True
            for _ in range(self.config.warmup_frames):
                self._read_valid_pair(self.config.frame_timeout_ms, publish=False)
        except BaseException:
            if session is not None:
                session.stop()
            self._session = None
            self._started = False
            raise
        return self

    def read(self, *, timeout_ms: int | None = None) -> D435Frame:
        """Return the next contract-valid frame, bounded by ``timeout_ms``."""

        if not self._started or self._session is None:
            raise RuntimeError("start() must be called before read()")
        if self._closed:
            raise RuntimeError("D435 source is closed")
        effective_timeout = self.config.frame_timeout_ms if timeout_ms is None else timeout_ms
        if (
            isinstance(effective_timeout, bool)
            or not isinstance(effective_timeout, int)
            or effective_timeout < 1
        ):
            raise ValueError("timeout_ms must be a positive integer")
        result = self._read_valid_pair(effective_timeout, publish=True)
        assert result is not None
        return result

    def stop(self) -> None:
        """Stop the SDK pipeline once; safe to call repeatedly."""

        if self._closed:
            return
        self._closed = True
        session = self._session
        self._session = None
        self._started = False
        if session is not None:
            session.stop()

    close = stop

    def __enter__(self) -> D435LiveSource:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def _read_valid_pair(self, timeout_ms: int, *, publish: bool) -> D435Frame | None:
        session = self._session
        if session is None:
            raise RuntimeError("D435 session is unavailable")
        deadline_ns = self._monotonic_ns() + timeout_ms * 1_000_000
        last_problem = "no RGB-D pair arrived"
        last_rejection: str | None = None
        while True:
            remaining_ns = deadline_ns - self._monotonic_ns()
            if remaining_ns <= 0:
                break
            remaining_ms = max(1, math.ceil(remaining_ns / 1_000_000))
            try:
                candidate = session.wait_for_aligned_pair(remaining_ms)
            except TimeoutError as error:
                backend_problem = str(error) or "backend frame timeout"
                last_problem = (
                    f"{backend_problem}; most recent rejected pair: {last_rejection}"
                    if last_rejection is not None
                    else backend_problem
                )
                break

            host_s = self._monotonic_ns() * 1e-9
            try:
                state = self._validate_candidate(candidate, host_s)
            except ValueError as error:
                last_problem = str(error)
                last_rejection = last_problem
                continue

            self._previous = state
            if self._intrinsics is None:
                self._intrinsics = candidate.intrinsics
            if not publish:
                return None

            rgb = _immutable_array(candidate.rgb, dtype=np.uint8)
            depth_raw = _immutable_array(candidate.depth_raw, dtype=np.uint16)
            assert self._depth_scale_m is not None
            depth_m = np.ascontiguousarray(depth_raw, dtype=np.float32) * self._depth_scale_m
            depth_m.setflags(write=False)
            self._sequence += 1
            return D435Frame(
                sequence=self._sequence,
                serial=self.config.serial,
                rgb=rgb,
                depth_raw=depth_raw,
                depth_m=depth_m,
                intrinsics=candidate.intrinsics,
                color_frame_number=state.color_frame_number,
                depth_frame_number=state.depth_frame_number,
                color_sensor_timestamp_s=state.color_sensor_timestamp_s,
                depth_sensor_timestamp_s=state.depth_sensor_timestamp_s,
                color_sdk_timestamp_s=state.color_sdk_timestamp_s,
                depth_sdk_timestamp_s=state.depth_sdk_timestamp_s,
                sdk_timestamp_domain=state.sdk_timestamp_domain,
                host_received_monotonic_s=state.host_received_monotonic_s,
            )

        raise D435FrameTimeout(
            f"timed out after {timeout_ms} ms waiting for a fresh aligned RGB-D pair; "
            f"last observation: {last_problem}"
        )

    def _validate_candidate(self, frame: D435BackendFrame, host_s: float) -> _PairState:
        expected_rgb_shape = (self.config.height, self.config.width, 3)
        expected_depth_shape = (self.config.height, self.config.width)
        if frame.rgb.shape != expected_rgb_shape or frame.rgb.dtype != np.uint8:
            raise RuntimeError(
                f"D435 RGB is {frame.rgb.shape}/{frame.rgb.dtype}; "
                f"expected {expected_rgb_shape}/uint8"
            )
        if frame.depth_raw.shape != expected_depth_shape or frame.depth_raw.dtype != np.uint16:
            raise RuntimeError(
                f"D435 aligned depth is {frame.depth_raw.shape}/{frame.depth_raw.dtype}; "
                f"expected {expected_depth_shape}/uint16"
            )
        if (frame.intrinsics.width, frame.intrinsics.height) != (
            self.config.width,
            self.config.height,
        ):
            raise RuntimeError("D435 intrinsics resolution does not match the configured stream")
        if self._intrinsics is not None and frame.intrinsics != self._intrinsics:
            raise RuntimeError("D435 color intrinsics changed during the live session")
        if not isinstance(frame.sdk_timestamp_domain, str) or not frame.sdk_timestamp_domain:
            raise RuntimeError("D435 SDK timestamp domain is missing")

        integers = (frame.color_frame_number, frame.depth_frame_number)
        invalid_number = any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in integers
        )
        if invalid_number:
            raise RuntimeError("D435 frame numbers must be non-negative integers")
        timestamps = (
            frame.color_sensor_timestamp_s,
            frame.depth_sensor_timestamp_s,
            frame.color_sdk_timestamp_s,
            frame.depth_sdk_timestamp_s,
            host_s,
        )
        if not all(math.isfinite(float(value)) for value in timestamps):
            raise RuntimeError("D435 timestamps must be finite")

        state = _PairState(
            color_frame_number=frame.color_frame_number,
            depth_frame_number=frame.depth_frame_number,
            color_sensor_timestamp_s=float(frame.color_sensor_timestamp_s),
            depth_sensor_timestamp_s=float(frame.depth_sensor_timestamp_s),
            color_sdk_timestamp_s=float(frame.color_sdk_timestamp_s),
            depth_sdk_timestamp_s=float(frame.depth_sdk_timestamp_s),
            sdk_timestamp_domain=frame.sdk_timestamp_domain,
            host_received_monotonic_s=host_s,
        )
        pair_skew_ms = abs(
            state.color_sensor_timestamp_s - state.depth_sensor_timestamp_s
        ) * 1e3
        if pair_skew_ms > self.config.max_pair_skew_ms:
            raise ValueError(
                f"sensor timestamp skew {pair_skew_ms:.3f} ms exceeds "
                f"{self.config.max_pair_skew_ms:.3f} ms"
            )

        previous = self._previous
        if previous is not None:
            # RealSense ``sensor_timestamp`` metadata may legitimately repeat
            # on adjacent color/depth frames (observed on a bandwidth-limited
            # USB 2 link), even while the SDK timestamps and frame counters
            # advance.  Freshness therefore requires strict progress from the
            # per-stream frame counter, SDK timestamp and host receipt clock;
            # hardware timestamps remain monotonic but equality alone is not a
            # replay.  An actual hardware timestamp regression still fails
            # closed, and the color/depth hardware skew gate above is retained.
            strict_checks = (
                (state.color_frame_number, previous.color_frame_number, "color frame number"),
                (state.depth_frame_number, previous.depth_frame_number, "depth frame number"),
                (
                    state.color_sdk_timestamp_s,
                    previous.color_sdk_timestamp_s,
                    "color SDK timestamp",
                ),
                (
                    state.depth_sdk_timestamp_s,
                    previous.depth_sdk_timestamp_s,
                    "depth SDK timestamp",
                ),
                (
                    state.host_received_monotonic_s,
                    previous.host_received_monotonic_s,
                    "host monotonic timestamp",
                ),
            )
            nondecreasing_checks = (
                (
                    state.color_sensor_timestamp_s,
                    previous.color_sensor_timestamp_s,
                    "color hardware timestamp",
                ),
                (
                    state.depth_sensor_timestamp_s,
                    previous.depth_sensor_timestamp_s,
                    "depth hardware timestamp",
                ),
            )
            regressed = [name for current, old, name in strict_checks if current <= old]
            regressed.extend(
                name for current, old, name in nondecreasing_checks if current < old
            )
            if state.sdk_timestamp_domain != previous.sdk_timestamp_domain:
                regressed.append("SDK timestamp domain changed")
            if regressed:
                raise ValueError("replayed or stale pair: " + "; ".join(regressed))
        return state


def _immutable_array(value: np.ndarray, *, dtype: np.dtype[Any] | type[np.generic]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


class PyRealSenseBackend:
    """Thin lazy-loaded adapter for ``pyrealsense2``."""

    def __init__(self, rs_module: Any) -> None:
        self._rs = rs_module

    @classmethod
    def load_vendor(cls) -> PyRealSenseBackend:
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError(
                "pyrealsense2 is unavailable; run D435 acquisition in the RealSense environment"
            ) from error
        return cls(rs)

    def list_devices(self) -> tuple[D435DeviceInfo, ...]:
        return tuple(self._device_info(device) for device in self._rs.context().query_devices())

    def open(self, config: D435LiveConfig) -> D435BackendSession:
        rs = self._rs
        pipeline = rs.pipeline()
        stream_config = rs.config()
        stream_config.enable_device(config.serial)
        stream_config.enable_stream(
            rs.stream.color, config.width, config.height, rs.format.rgb8, config.fps
        )
        stream_config.enable_stream(
            rs.stream.depth, config.width, config.height, rs.format.z16, config.fps
        )
        profile = pipeline.start(stream_config)
        try:
            device = profile.get_device()
            info = self._device_info(device)
            depth_scale_m = float(device.first_depth_sensor().get_depth_scale())
            return _PyRealSenseSession(
                rs_module=rs,
                pipeline=pipeline,
                align_to_color=rs.align(rs.stream.color),
                device_info=info,
                depth_scale_m=depth_scale_m,
            )
        except BaseException:
            pipeline.stop()
            raise

    def _device_info(self, device: Any) -> D435DeviceInfo:
        rs = self._rs

        def read(name: str) -> str:
            key = getattr(rs.camera_info, name)
            return str(device.get_info(key)) if device.supports(key) else ""

        return D435DeviceInfo(
            serial=read("serial_number"),
            name=read("name"),
            firmware_version=read("firmware_version"),
            usb_type_descriptor=read("usb_type_descriptor"),
            physical_port=read("physical_port"),
            product_id=read("product_id"),
            product_line=read("product_line"),
        )


class _PyRealSenseSession:
    def __init__(
        self,
        *,
        rs_module: Any,
        pipeline: Any,
        align_to_color: Any,
        device_info: D435DeviceInfo,
        depth_scale_m: float,
    ) -> None:
        self._rs = rs_module
        self._pipeline = pipeline
        self._align_to_color = align_to_color
        self._device_info = device_info
        self._depth_scale_m = depth_scale_m
        self._stopped = False

    @property
    def device_info(self) -> D435DeviceInfo:
        return self._device_info

    @property
    def depth_scale_m(self) -> float:
        return self._depth_scale_m

    def wait_for_aligned_pair(self, timeout_ms: int) -> D435BackendFrame:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as error:
            raise TimeoutError(str(error)) from error
        aligned = self._align_to_color.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            raise TimeoutError("frameset is missing color or aligned depth")

        color_domain = _timestamp_domain_name(color)
        depth_domain = _timestamp_domain_name(depth)
        if color_domain != depth_domain:
            raise RuntimeError(
                "color/depth SDK timestamp domains differ: "
                f"color={color_domain}, depth={depth_domain}"
            )
        intrinsics = color.profile.as_video_stream_profile().intrinsics
        return D435BackendFrame(
            rgb=np.asanyarray(color.get_data()),
            depth_raw=np.asanyarray(depth.get_data()),
            intrinsics=D435Intrinsics(
                width=int(intrinsics.width),
                height=int(intrinsics.height),
                fx=float(intrinsics.fx),
                fy=float(intrinsics.fy),
                ppx=float(intrinsics.ppx),
                ppy=float(intrinsics.ppy),
                distortion_model=str(intrinsics.model).removeprefix("distortion."),
                distortion_coefficients=tuple(float(value) for value in intrinsics.coeffs),
            ),
            color_frame_number=int(color.get_frame_number()),
            depth_frame_number=int(depth.get_frame_number()),
            color_sensor_timestamp_s=_sensor_timestamp_s(self._rs, color),
            depth_sensor_timestamp_s=_sensor_timestamp_s(self._rs, depth),
            color_sdk_timestamp_s=float(color.get_timestamp()) * 1e-3,
            depth_sdk_timestamp_s=float(depth.get_timestamp()) * 1e-3,
            sdk_timestamp_domain=color_domain,
        )

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._pipeline.stop()


def _timestamp_domain_name(frame: Any) -> str:
    return str(frame.get_frame_timestamp_domain()).removeprefix("timestamp_domain.")


def _sensor_timestamp_s(rs_module: Any, frame: Any) -> float:
    metadata = rs_module.frame_metadata_value.sensor_timestamp
    if not frame.supports_frame_metadata(metadata):
        raise RuntimeError("frame does not expose RealSense sensor timestamp metadata")
    return float(frame.get_frame_metadata(metadata)) * 1e-6


__all__ = [
    "D435Backend",
    "D435BackendFrame",
    "D435BackendSession",
    "D435DeviceInfo",
    "D435Frame",
    "D435FrameTimeout",
    "D435Intrinsics",
    "D435LiveConfig",
    "D435LiveSource",
    "PyRealSenseBackend",
]
