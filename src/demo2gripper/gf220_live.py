"""Long-lived image and FORCE6D acquisition for two ViTai GF220 sensors.

The vendor SDK is imported only when :meth:`GF220LiveSource.start` is called.
Each SDK device is constructed, calibrated, sampled, and released by one
dedicated thread; callers only read copied frames from bounded deques.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .tactile_contact_classifier import FingerRimClassifierWindow
from .tactile_rim_probe import TimedArrayWindow

PROFILE_SCHEMA = "demo2gripper.gf220_manual_warp_profile.v1"
DEFAULT_PROFILE = Path(__file__).resolve().parents[2] / "configs/gf220_manual_warp_draft.json"


@dataclass(frozen=True)
class GF220SDKTypes:
    """Injected vendor types used by :class:`GF220LiveSource`.

    Tests provide fakes through this type, so importing this module never
    imports ``pyvitaisdk`` or opens a camera.
    """

    finder_type: Callable[[], Any]
    device_type: Callable[..., Any]
    stream_format_type: Callable[..., Any]
    output_profile: Any
    time_stamp: Any
    raw_img: Any
    warped_img: Any
    diff_img: Any
    force6d: Any

    @classmethod
    def load_vendor(cls) -> GF220SDKTypes:
        try:
            from pyvitaisdk import (
                GF225,
                GF225OutputProfile,
                GFDataType,
                VTSDeviceFinder,
            )
            from pyvitaisdk.vts.core.base.profile import VideoStreamFmt
        except ImportError as exc:
            raise RuntimeError(
                "pyvitaisdk is unavailable; run GF220 acquisition in the fhh environment"
            ) from exc
        return cls(
            finder_type=VTSDeviceFinder,
            device_type=GF225,
            stream_format_type=VideoStreamFmt,
            output_profile=GF225OutputProfile.W240_H240,
            time_stamp=GFDataType.TIME_STAMP,
            raw_img=GFDataType.RAW_IMG,
            warped_img=GFDataType.WARPED_IMG,
            diff_img=GFDataType.DIFF_IMG,
            force6d=GFDataType.FORCE6D_VECTOR,
        )


@dataclass(frozen=True)
class GF220LiveConfig:
    buffer_frames: int = 24
    warmup_frames: int = 8
    warp_validation_frames: int = 2
    discovery_retries: int = 10
    discovery_retry_delay_s: float = 0.5
    cached_poll_delay_s: float = 0.002
    max_consecutive_sdk_errors: int = 3
    sdk_error_retry_delay_s: float = 0.01
    sdk_timestamp_scale_s: float = 0.001
    max_frame_age_s: float = 0.15
    max_pair_skew_s: float = 0.08
    close_timeout_s: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "buffer_frames",
            "warmup_frames",
            "warp_validation_frames",
            "discovery_retries",
            "max_consecutive_sdk_errors",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "sdk_timestamp_scale_s",
            "max_frame_age_s",
            "max_pair_skew_s",
            "close_timeout_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "discovery_retry_delay_s",
            "cached_poll_delay_s",
            "sdk_error_retry_delay_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class GF220SensorSpec:
    name: str
    serial: str
    logical_side: str | None
    corners_xy: tuple[tuple[float, float], ...]
    warp_sha256: str


@dataclass(frozen=True)
class GF220ManualWarpProfile:
    path: Path
    stream_width: int
    stream_height: int
    stream_fps: int
    stream_fourcc: str
    output_width: int
    output_height: int
    sensors: tuple[GF220SensorSpec, GF220SensorSpec]

    @classmethod
    def read(cls, path: str | Path) -> GF220ManualWarpProfile:
        profile_path = Path(path)
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if payload.get("schema") != PROFILE_SCHEMA:
            raise ValueError(f"expected profile schema {PROFILE_SCHEMA!r}")

        stream = payload.get("stream")
        output = payload.get("output")
        if not isinstance(stream, dict) or not isinstance(output, dict):
            raise ValueError("profile stream and output must be objects")
        width = _positive_int(stream.get("width"), "stream.width")
        height = _positive_int(stream.get("height"), "stream.height")
        fps = _positive_int(stream.get("fps"), "stream.fps")
        output_width = _positive_int(output.get("width"), "output.width")
        output_height = _positive_int(output.get("height"), "output.height")
        fourcc = stream.get("fourcc")
        if not isinstance(fourcc, str) or not fourcc:
            raise ValueError("stream.fourcc must be a non-empty string")

        sensor_payloads = payload.get("sensors")
        if not isinstance(sensor_payloads, list) or len(sensor_payloads) != 2:
            raise ValueError("profile must contain exactly two sensors")
        names: set[str] = set()
        serials: set[str] = set()
        sensors: list[GF220SensorSpec] = []
        for sensor in sensor_payloads:
            if not isinstance(sensor, dict):
                raise ValueError("each sensor profile must be an object")
            name = sensor.get("name")
            serial = sensor.get("sdk_serial")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("sensor names must be non-empty and unique")
            if not isinstance(serial, str) or not serial or serial in serials:
                raise ValueError("sensor serials must be non-empty and unique")
            if sensor.get("corner_order") != ("top_left_top_right_bottom_right_bottom_left"):
                raise ValueError(f"{name}: unsupported corner order")
            corners = np.asarray(sensor.get("corners_xy"), dtype=float)
            if corners.shape != (4, 2) or not np.isfinite(corners).all():
                raise ValueError(f"{name}: corners_xy must be a finite 4x2 array")
            if (
                np.any(corners[:, 0] < 0)
                or np.any(corners[:, 0] >= width)
                or np.any(corners[:, 1] < 0)
                or np.any(corners[:, 1] >= height)
            ):
                raise ValueError(f"{name}: corners_xy lie outside the stream")
            area = 0.5 * abs(
                float(
                    np.dot(corners[:, 0], np.roll(corners[:, 1], -1))
                    - np.dot(corners[:, 1], np.roll(corners[:, 0], -1))
                )
            )
            if area <= 0.0:
                raise ValueError(f"{name}: manual warp quadrilateral has zero area")
            logical_side = sensor.get("logical_side")
            if logical_side not in (None, "left", "right"):
                raise ValueError(f"{name}: logical_side must be left, right, or null")

            warp_payload = {
                "schema": PROFILE_SCHEMA,
                "stream": {
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "fourcc": fourcc,
                },
                "output": {"width": output_width, "height": output_height},
                "serial": serial,
                "corners_xy": corners.tolist(),
            }
            warp_sha256 = hashlib.sha256(
                json.dumps(warp_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            sensors.append(
                GF220SensorSpec(
                    name=name,
                    serial=serial,
                    logical_side=logical_side,
                    corners_xy=tuple(tuple(float(value) for value in row) for row in corners),
                    warp_sha256=warp_sha256,
                )
            )
            names.add(name)
            serials.add(serial)

        return cls(
            path=profile_path.resolve(),
            stream_width=width,
            stream_height=height,
            stream_fps=fps,
            stream_fourcc=fourcc,
            output_width=output_width,
            output_height=output_height,
            sensors=(sensors[0], sensors[1]),
        )


@dataclass(frozen=True)
class GF220Calibration:
    serial: str
    warp_sha256: str
    baseline_sha256: str
    calibration_id: str
    unloaded_warped: np.ndarray

    @classmethod
    def create(
        cls,
        *,
        serial: str,
        warp_sha256: str,
        unloaded_warped: np.ndarray,
    ) -> GF220Calibration:
        baseline = np.ascontiguousarray(unloaded_warped).copy()
        if baseline.ndim != 3 or baseline.shape[-1] != 3:
            raise ValueError("unloaded calibration must be an HxWx3 image")
        baseline_hash = hashlib.sha256()
        baseline_hash.update(str(baseline.dtype).encode("ascii"))
        baseline_hash.update(np.asarray(baseline.shape, dtype=np.int64).tobytes())
        baseline_hash.update(baseline.tobytes())
        baseline_sha256 = baseline_hash.hexdigest()
        identity = hashlib.sha256()
        identity.update(b"demo2gripper.gf220_calibration.v1\0")
        identity.update(serial.encode("utf-8"))
        identity.update(bytes.fromhex(warp_sha256))
        identity.update(bytes.fromhex(baseline_sha256))
        baseline.setflags(write=False)
        return cls(
            serial=serial,
            warp_sha256=warp_sha256,
            baseline_sha256=baseline_sha256,
            calibration_id=f"gf220-v1-{identity.hexdigest()}",
            unloaded_warped=baseline,
        )


@dataclass(frozen=True)
class GF220Frame:
    sdk_time_s: float
    capture_monotonic_s: float
    raw_crc32: int
    raw: np.ndarray
    warped: np.ndarray
    diff: np.ndarray
    force6d: np.ndarray

    def __post_init__(self) -> None:
        for name in ("raw", "warped", "diff"):
            image = np.ascontiguousarray(getattr(self, name)).copy()
            image.setflags(write=False)
            object.__setattr__(self, name, image)
        force6d = np.asarray(self.force6d)
        if force6d.shape != (6,):
            raise ValueError(
                f"FORCE6D_VECTOR must have shape (6,), got {force6d.shape}"
            )
        if not np.issubdtype(force6d.dtype, np.number):
            raise ValueError("FORCE6D_VECTOR must have a numeric dtype")
        force6d = np.ascontiguousarray(force6d, dtype=np.float64).copy()
        if not np.isfinite(force6d).all():
            raise ValueError("FORCE6D_VECTOR must contain only finite values")
        force6d.setflags(write=False)
        object.__setattr__(self, "force6d", force6d)


@dataclass(frozen=True)
class GF220ClassifierWindows:
    """Synchronized classifier input for the two serials in profile order."""

    sensor_names: tuple[str, str]
    serials: tuple[str, str]
    windows: tuple[FingerRimClassifierWindow, FingerRimClassifierWindow]
    start_skew_s: float
    end_skew_s: float
    collected_monotonic_s: float

    @property
    def endpoint_skew_s(self) -> float:
        return max(self.start_skew_s, self.end_skew_s)

    def for_serial(self, serial: str) -> FingerRimClassifierWindow:
        try:
            return self.windows[self.serials.index(serial)]
        except ValueError as exc:
            raise KeyError(serial) from exc


class _SensorWorker:
    def __init__(
        self,
        *,
        spec: GF220SensorSpec,
        profile: GF220ManualWarpProfile,
        handle: Any,
        sdk: GF220SDKTypes,
        config: GF220LiveConfig,
        condition: threading.Condition,
    ) -> None:
        self.spec = spec
        self.profile = profile
        self.handle = handle
        self.sdk = sdk
        self.config = config
        self.condition = condition
        self.frames: deque[GF220Frame] = deque(maxlen=config.buffer_frames)
        self.calibration: GF220Calibration | None = None
        self.ready = False
        self.calibration_requested = False
        self.calibrated = False
        self.stop_requested = False
        self.error: BaseException | None = None
        self.rejected_cached_frames = 0
        self.thread = threading.Thread(
            target=self._run,
            name=f"gf220[{spec.name}]",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def request_calibration(self) -> None:
        with self.condition:
            self.calibration_requested = True
            self.condition.notify_all()

    def request_stop(self) -> None:
        with self.condition:
            self.stop_requested = True
            self.condition.notify_all()

    def _stopping(self) -> bool:
        with self.condition:
            return self.stop_requested

    def _collect_new(
        self,
        device: Any,
        data_types: tuple[Any, ...],
        previous_timestamp: int | None,
        previous_crc32: int | None,
    ) -> tuple[dict[Any, Any], int, int] | None:
        consecutive_sdk_errors = 0
        while not self._stopping():
            try:
                sample = device.collect_sensor_data(*data_types)
            except Exception as error:
                consecutive_sdk_errors += 1
                if consecutive_sdk_errors >= self.config.max_consecutive_sdk_errors:
                    raise RuntimeError(
                        f"{self.spec.serial}: collect_sensor_data failed "
                        f"{consecutive_sdk_errors} consecutive times; last error: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                if self.config.sdk_error_retry_delay_s:
                    with self.condition:
                        self.condition.wait(self.config.sdk_error_retry_delay_s)
                continue
            consecutive_sdk_errors = 0
            timestamp = int(np.asarray(sample[self.sdk.time_stamp]).reshape(()))
            raw = np.ascontiguousarray(sample[self.sdk.raw_img])
            raw_crc32 = zlib.crc32(raw)
            timestamp_advanced = previous_timestamp is None or timestamp > previous_timestamp
            image_advanced = previous_crc32 is None or raw_crc32 != previous_crc32
            if timestamp_advanced and image_advanced:
                return sample, timestamp, raw_crc32
            with self.condition:
                self.rejected_cached_frames += 1
            if self.config.cached_poll_delay_s:
                time.sleep(self.config.cached_poll_delay_s)
        return None

    def _run(self) -> None:
        device = None
        try:
            stream_format = self.sdk.stream_format_type(
                width=self.profile.stream_width,
                height=self.profile.stream_height,
                fps=self.profile.stream_fps,
                fourcc=self.profile.stream_fourcc,
            )
            device = self.sdk.device_type(
                config=self.handle,
                stream_format=stream_format,
                output_format=self.sdk.output_profile,
            )
            previous_timestamp: int | None = None
            previous_crc32: int | None = None
            raw_types = (self.sdk.time_stamp, self.sdk.raw_img)
            for _ in range(self.config.warmup_frames):
                result = self._collect_new(device, raw_types, previous_timestamp, previous_crc32)
                if result is None:
                    return
                _, previous_timestamp, previous_crc32 = result

            processor = getattr(device, "_image_processor", None)
            if processor is None or not hasattr(processor, "set_warp_params"):
                raise RuntimeError(
                    f"{self.spec.serial}: SDK does not expose manual warp parameters"
                )
            processor.set_warp_params(
                corner_points=[list(row) for row in self.spec.corners_xy],
                mode="manual",
            )

            warped_types = (*raw_types, self.sdk.warped_img)
            expected_shape = (
                self.profile.output_height,
                self.profile.output_width,
                3,
            )
            for _ in range(self.config.warp_validation_frames):
                result = self._collect_new(device, warped_types, previous_timestamp, previous_crc32)
                if result is None:
                    return
                sample, previous_timestamp, previous_crc32 = result
                warped = np.asarray(sample[self.sdk.warped_img])
                if warped.shape != expected_shape:
                    raise RuntimeError(
                        f"{self.spec.serial}: WARPED_IMG shape {warped.shape} "
                        f"does not match {expected_shape}"
                    )

            with self.condition:
                self.ready = True
                self.condition.notify_all()
                while not self.calibration_requested and not self.stop_requested:
                    self.condition.wait()
                if self.stop_requested:
                    return

            baseline = device.calibrate()
            if baseline is None:
                raise RuntimeError(
                    f"{self.spec.serial}: SDK calibrate() did not return a baseline image"
                )
            baseline = np.asarray(baseline)
            if baseline.shape != expected_shape:
                raise RuntimeError(
                    f"{self.spec.serial}: calibration shape {baseline.shape} "
                    f"does not match {expected_shape}"
                )
            calibration = GF220Calibration.create(
                serial=self.spec.serial,
                warp_sha256=self.spec.warp_sha256,
                unloaded_warped=baseline,
            )
            with self.condition:
                self.calibration = calibration
                self.frames.clear()
                self.calibrated = True
                self.condition.notify_all()

            data_types = (*warped_types, self.sdk.diff_img, self.sdk.force6d)
            while not self._stopping():
                result = self._collect_new(device, data_types, previous_timestamp, previous_crc32)
                if result is None:
                    return
                sample, timestamp, raw_crc32 = result
                previous_timestamp = timestamp
                previous_crc32 = raw_crc32
                capture_time = time.monotonic()
                frame = GF220Frame(
                    sdk_time_s=float(timestamp) * self.config.sdk_timestamp_scale_s,
                    capture_monotonic_s=capture_time,
                    raw_crc32=raw_crc32,
                    raw=np.asarray(sample[self.sdk.raw_img]),
                    warped=np.asarray(sample[self.sdk.warped_img]),
                    diff=np.asarray(sample[self.sdk.diff_img]),
                    force6d=np.asarray(sample[self.sdk.force6d]),
                )
                if frame.warped.shape != expected_shape or frame.diff.shape != expected_shape:
                    raise RuntimeError(f"{self.spec.serial}: classifier image shape changed")
                with self.condition:
                    self.frames.append(frame)
                    self.condition.notify_all()
        except BaseException as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()
        finally:
            if device is not None:
                try:
                    release = getattr(device, "release", None)
                    if release is not None:
                        release()
                except BaseException as exc:
                    with self.condition:
                        if self.error is None and not self.stop_requested:
                            self.error = exc
                        self.condition.notify_all()


class GF220LiveSource:
    """Own two long-lived GF220 acquisition threads and produce fresh windows."""

    def __init__(
        self,
        profile_path: str | Path = DEFAULT_PROFILE,
        *,
        config: GF220LiveConfig = GF220LiveConfig(),
        sdk_types: GF220SDKTypes | None = None,
    ) -> None:
        self.profile = GF220ManualWarpProfile.read(profile_path)
        self.config = config
        self._sdk_types = sdk_types
        self._condition = threading.Condition()
        self._finder: Any = None
        self._device_handles: dict[str, Any] = {}
        self._workers: tuple[_SensorWorker, _SensorWorker] | None = None
        self._started = False
        self._calibrate_called = False
        self._closed = False

    @property
    def calibrations(self) -> tuple[GF220Calibration, GF220Calibration]:
        workers = self._require_workers()
        values = tuple(worker.calibration for worker in workers)
        if any(value is None for value in values):
            raise RuntimeError("GF220 sensors have not been calibrated")
        return values  # type: ignore[return-value]

    def start(self, *, timeout_s: float = 15.0) -> GF220LiveSource:
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive and finite")
        if self._closed:
            raise RuntimeError("GF220 source is closed")
        if self._started:
            return self
        self._started = True
        try:
            sdk = self._sdk_types or GF220SDKTypes.load_vendor()
            self._finder, self._device_handles = self._discover(sdk)
            workers = tuple(
                _SensorWorker(
                    spec=spec,
                    profile=self.profile,
                    handle=self._device_handles[spec.serial],
                    sdk=sdk,
                    config=self.config,
                    condition=self._condition,
                )
                for spec in self.profile.sensors
            )
            self._workers = (workers[0], workers[1])
            for worker in self._workers:
                worker.start()
            self._wait_for(
                lambda: all(worker.ready for worker in self._require_workers()),
                timeout_s,
                "GF220 startup and manual warp",
            )
        except BaseException:
            self.close()
            raise
        return self

    def calibrate_unloaded(
        self, *, timeout_s: float = 15.0
    ) -> tuple[GF220Calibration, GF220Calibration]:
        """Capture each pad's one no-contact baseline for this live session.

        The caller is responsible for ensuring that both pads are unloaded.
        This method is deliberately single-use; a new baseline requires a new
        :class:`GF220LiveSource` session.
        """

        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive and finite")
        workers = self._require_workers()
        if self._calibrate_called:
            raise RuntimeError("calibrate_unloaded() may be called only once per session")
        self._calibrate_called = True
        for worker in workers:
            worker.request_calibration()
        self._wait_for(
            lambda: all(worker.calibrated for worker in self._require_workers()),
            timeout_s,
            "GF220 unloaded calibration",
        )
        return self.calibrations

    def collect_classifier_windows(
        self,
        not_before_s: float,
        samples: int,
        timeout: float,
    ) -> GF220ClassifierWindows:
        """Return synchronized, post-command image windows from both fingers.

        Every selected host timestamp is strictly later than ``not_before_s``.
        Both the first and last sample pairs satisfy ``max_pair_skew_s``;
        accepted frames already have strictly increasing SDK timestamps and a
        changed RAW image CRC.
        """

        not_before_s = float(not_before_s)
        timeout = float(timeout)
        if not math.isfinite(not_before_s):
            raise ValueError("not_before_s must be finite")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
            raise ValueError("samples must be an integer >= 2")
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout must be positive and finite")
        workers = self._require_workers()
        if not self._calibrate_called:
            raise RuntimeError("calibrate_unloaded() must complete before collection")

        deadline = time.monotonic() + timeout
        last_reason = "waiting for post-command frames"
        with self._condition:
            while True:
                self._raise_worker_error_locked()
                selection, last_reason = self._select_windows_locked(workers, not_before_s, samples)
                if selection is not None:
                    left_frames, right_frames = selection
                    now_s = time.monotonic()
                    return GF220ClassifierWindows(
                        sensor_names=(workers[0].spec.name, workers[1].spec.name),
                        serials=(workers[0].spec.serial, workers[1].spec.serial),
                        windows=(
                            _classifier_window(workers[0].calibration, left_frames),
                            _classifier_window(workers[1].calibration, right_frames),
                        ),
                        start_skew_s=abs(
                            left_frames[0].capture_monotonic_s - right_frames[0].capture_monotonic_s
                        ),
                        end_skew_s=abs(
                            left_frames[-1].capture_monotonic_s
                            - right_frames[-1].capture_monotonic_s
                        ),
                        collected_monotonic_s=now_s,
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"could not collect synchronized GF220 windows: {last_reason}"
                    )
                self._condition.wait(min(remaining, 0.05))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        workers = self._workers
        if workers is None:
            return
        for worker in workers:
            worker.request_stop()
        deadline = time.monotonic() + self.config.close_timeout_s
        for worker in workers:
            worker.thread.join(max(0.0, deadline - time.monotonic()))
        alive = [worker.spec.serial for worker in workers if worker.thread.is_alive()]
        if alive:
            raise RuntimeError(f"GF220 acquisition threads did not stop: {alive}")

    def __enter__(self) -> GF220LiveSource:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _discover(self, sdk: GF220SDKTypes) -> tuple[Any, dict[str, Any]]:
        serials = tuple(sensor.serial for sensor in self.profile.sensors)
        last_available: list[str] = []
        for attempt in range(self.config.discovery_retries):
            finder = sdk.finder_type()
            last_available = list(finder.get_sns())
            handles = {serial: finder.get_device_by_sn(serial) for serial in serials}
            if all(handle is not None for handle in handles.values()):
                return finder, handles
            if attempt + 1 < self.config.discovery_retries:
                time.sleep(self.config.discovery_retry_delay_s)
        raise RuntimeError(
            "could not discover both GF220 sensors; "
            f"requested={list(serials)}, available={last_available}"
        )

    def _wait_for(self, predicate: Callable[[], bool], timeout_s: float, description: str) -> None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not predicate():
                self._raise_worker_error_locked()
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"timed out waiting for {description}")
                self._condition.wait(min(remaining, 0.05))

    def _raise_worker_error_locked(self) -> None:
        workers = self._workers
        if workers is None:
            return
        for worker in workers:
            if worker.error is not None:
                error = worker.error
                raise RuntimeError(
                    f"GF220 acquisition failed for {worker.spec.serial}: "
                    f"{type(error).__name__}: {error}"
                ) from error

    def _select_windows_locked(
        self,
        workers: tuple[_SensorWorker, _SensorWorker],
        not_before_s: float,
        samples: int,
    ) -> tuple[tuple[list[GF220Frame], list[GF220Frame]] | None, str]:
        eligible = [
            [frame for frame in worker.frames if frame.capture_monotonic_s > not_before_s]
            for worker in workers
        ]
        if any(len(frames) < samples for frames in eligible):
            counts = tuple(len(frames) for frames in eligible)
            return None, f"only {counts} fresh frames are newer than not_before_s"

        now_s = time.monotonic()
        best: tuple[list[GF220Frame], list[GF220Frame]] | None = None
        best_recency = -math.inf
        for left_end in range(samples - 1, len(eligible[0])):
            left = eligible[0][left_end - samples + 1 : left_end + 1]
            for right_end in range(samples - 1, len(eligible[1])):
                right = eligible[1][right_end - samples + 1 : right_end + 1]
                start_skew = abs(left[0].capture_monotonic_s - right[0].capture_monotonic_s)
                end_skew = abs(left[-1].capture_monotonic_s - right[-1].capture_monotonic_s)
                if max(start_skew, end_skew) > self.config.max_pair_skew_s:
                    continue
                endpoint = min(left[-1].capture_monotonic_s, right[-1].capture_monotonic_s)
                oldest = min(left[0].capture_monotonic_s, right[0].capture_monotonic_s)
                if now_s - oldest > self.config.max_frame_age_s:
                    continue
                if endpoint > best_recency:
                    best = (left, right)
                    best_recency = endpoint
        if best is None:
            return None, (
                "no fresh pair satisfies first/last host skew "
                f"<= {self.config.max_pair_skew_s:.6f} s and age "
                f"<= {self.config.max_frame_age_s:.6f} s"
            )
        return best, "ready"

    def _require_workers(self) -> tuple[_SensorWorker, _SensorWorker]:
        if self._closed:
            raise RuntimeError("GF220 source is closed")
        if not self._started or self._workers is None:
            raise RuntimeError("start() must be called first")
        return self._workers


def _classifier_window(
    calibration: GF220Calibration | None,
    frames: list[GF220Frame],
) -> FingerRimClassifierWindow:
    if calibration is None:
        raise RuntimeError("GF220 calibration is unavailable")
    sdk_times = np.asarray([frame.sdk_time_s for frame in frames], dtype=float)
    if np.unique(sdk_times).size != len(frames) or np.any(np.diff(sdk_times) <= 0.0):
        raise RuntimeError("GF220 window contains repeated or regressed SDK timestamps")
    host_times = np.asarray([frame.capture_monotonic_s for frame in frames], dtype=float)
    valid = np.ones(len(frames), dtype=bool)
    return FingerRimClassifierWindow(
        calibration_id=calibration.calibration_id,
        warped=TimedArrayWindow(
            values=np.stack([frame.warped for frame in frames]),
            sdk_time_s=sdk_times,
            capture_monotonic_s=host_times,
            valid=valid,
        ),
        diff=TimedArrayWindow(
            values=np.stack([frame.diff for frame in frames]),
            sdk_time_s=sdk_times,
            capture_monotonic_s=host_times,
            valid=valid,
        ),
        force6d=TimedArrayWindow(
            values=np.stack([frame.force6d for frame in frames]),
            sdk_time_s=sdk_times,
            capture_monotonic_s=host_times,
            valid=valid,
        ),
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


__all__ = [
    "DEFAULT_PROFILE",
    "GF220Calibration",
    "GF220ClassifierWindows",
    "GF220Frame",
    "GF220LiveConfig",
    "GF220LiveSource",
    "GF220ManualWarpProfile",
    "GF220SDKTypes",
    "GF220SensorSpec",
]
