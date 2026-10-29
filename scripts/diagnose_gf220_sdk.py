#!/usr/bin/env python3
"""Read two GF220 sensors through ViTai SDK without connecting to the robot.

The GF220 cameras on this host produce 640x480 MJPEG.  ``pyvitaisdk`` 1.0.11
defaults its GF225-compatible reader to 640x360 and its automatic warp selects
invalid corners for these images.  This diagnostic applies one reviewed set of
four source corners per SDK serial and verifies the resulting 240x240 image.

By default the tool does not calibrate and therefore does not request ``DIFF_IMG``.
Only use ``--calibrate-unloaded`` while both tactile pads are completely free of
contact.  That flag captures an SDK baseline for this process and adds real SDK
``DIFF_IMG`` frames to the output.  ``FORCE6D_VECTOR`` is deliberately not read:
its axes, signs, units, noise, and latency are not calibrated on this installation.

Example (warped-image diagnostic only)::

    /home/vitai/miniconda3/envs/fhh/bin/python \
      scripts/diagnose_gf220_sdk.py \
      --output-dir /tmp/gf220_warp_check

Example after opening the gripper and confirming both pads are unloaded::

    /home/vitai/miniconda3/envs/fhh/bin/python \
      scripts/diagnose_gf220_sdk.py \
      --output-dir /tmp/gf220_unloaded_check \
      --calibrate-unloaded
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import zlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROFILE_SCHEMA = "demo2gripper.gf220_manual_warp_profile.v1"
OUTPUT_SCHEMA = "demo2gripper.gf220_sdk_diagnostic.v1"
DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1] / "configs/gf220_manual_warp_draft.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--new-frame-timeout-s", type=float, default=2.0)
    parser.add_argument("--discovery-retries", type=int, default=10)
    parser.add_argument("--discovery-retry-delay-s", type=float, default=0.5)
    parser.add_argument(
        "--calibrate-unloaded",
        action="store_true",
        help="Capture a no-contact SDK baseline and request DIFF_IMG",
    )
    return parser.parse_args()


def _read_profile(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"Expected profile schema {PROFILE_SCHEMA!r}")

    stream = payload.get("stream")
    output = payload.get("output")
    if not isinstance(stream, dict) or not isinstance(output, dict):
        raise ValueError("profile stream and output must be objects")
    if stream.get("width") != 640 or stream.get("height") != 480:
        raise ValueError("this diagnostic requires the measured 640x480 GF220 stream")
    if output.get("width") != 240 or output.get("height") != 240:
        raise ValueError("this diagnostic requires the SDK 240x240 output profile")

    sensors = payload.get("sensors")
    if not isinstance(sensors, list) or len(sensors) != 2:
        raise ValueError("profile must contain exactly two sensors")
    serials: set[str] = set()
    names: set[str] = set()
    for sensor in sensors:
        if not isinstance(sensor, dict):
            raise ValueError("each sensor profile must be an object")
        name = sensor.get("name")
        serial = sensor.get("sdk_serial")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("sensor names must be non-empty and unique")
        if not isinstance(serial, str) or not serial or serial in serials:
            raise ValueError("SDK serials must be non-empty and unique")
        names.add(name)
        serials.add(serial)

        corners = np.asarray(sensor.get("corners_xy"), dtype=np.float32)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError(f"{name}: corners_xy must be a finite 4x2 array")
        if (
            np.any(corners[:, 0] < 0)
            or np.any(corners[:, 0] >= stream["width"])
            or np.any(corners[:, 1] < 0)
            or np.any(corners[:, 1] >= stream["height"])
        ):
            raise ValueError(f"{name}: corners_xy lie outside the 640x480 frame")
        if cv2.contourArea(corners) < 10_000:
            raise ValueError(f"{name}: manual warp quadrilateral is implausibly small")
        if sensor.get("corner_order") != "top_left_top_right_bottom_right_bottom_left":
            raise ValueError(f"{name}: unsupported corner order")
    return payload


def _discover_devices(
    serials: list[str], retries: int, retry_delay_s: float, finder_type: type
) -> tuple[Any, dict[str, Any]]:
    last_available: list[str] = []
    for attempt in range(retries):
        finder = finder_type()
        last_available = list(finder.get_sns())
        handles = {serial: finder.get_device_by_sn(serial) for serial in serials}
        if all(handle is not None for handle in handles.values()):
            return finder, handles
        if attempt + 1 < retries:
            time.sleep(retry_delay_s)
    raise RuntimeError(
        f"Could not discover both requested GF220 sensors; last SDK list was {last_available}"
    )


def _collect_new(
    device: Any,
    data_types: tuple[Any, ...],
    timestamp_type: Any,
    raw_type: Any,
    previous_timestamp: int | None,
    previous_raw_crc32: int | None,
    timeout_s: float,
) -> tuple[dict[Any, Any], int, int]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sample = device.collect_sensor_data(*data_types)
        timestamp = int(np.asarray(sample[timestamp_type]).reshape(()))
        raw = np.ascontiguousarray(sample[raw_type])
        raw_crc32 = zlib.crc32(raw)
        timestamp_advanced = previous_timestamp is None or timestamp != previous_timestamp
        image_advanced = previous_raw_crc32 is None or raw_crc32 != previous_raw_crc32
        if timestamp_advanced and image_advanced:
            return sample, timestamp, raw_crc32
        time.sleep(0.002)
    raise TimeoutError(
        "SDK timestamp and raw image did not both advance before the frame timeout"
    )


def _image_stats(image: np.ndarray) -> dict[str, float | int | list[int]]:
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "minimum": int(image.min()),
        "maximum": int(image.max()),
        "mean": float(image.mean()),
        "standard_deviation": float(image.std()),
    }


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


def main() -> int:
    args = parse_args()
    if args.frames < 2 or args.warmup_frames < 1:
        raise ValueError("--frames must be >= 2 and --warmup-frames must be >= 1")
    if args.new_frame_timeout_s <= 0.0:
        raise ValueError("--new-frame-timeout-s must be positive")
    if args.discovery_retries < 1 or args.discovery_retry_delay_s < 0.0:
        raise ValueError("discovery retry settings are invalid")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")

    profile = _read_profile(args.profile)
    sensors_profile = profile["sensors"]
    serials = [sensor["sdk_serial"] for sensor in sensors_profile]

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
            "pyvitaisdk is unavailable; run this with the fhh Python environment"
        ) from exc

    _, handles = _discover_devices(
        serials,
        args.discovery_retries,
        args.discovery_retry_delay_s,
        VTSDeviceFinder,
    )
    stream = profile["stream"]
    devices: list[tuple[dict[str, Any], Any]] = []
    last_timestamps: dict[str, int | None] = {serial: None for serial in serials}
    last_raw_crc32: dict[str, int | None] = {serial: None for serial in serials}
    args.output_dir.mkdir(parents=True)

    try:
        for sensor_profile in sensors_profile:
            serial = sensor_profile["sdk_serial"]
            device = GF225(
                config=handles[serial],
                stream_format=VideoStreamFmt(
                    width=stream["width"],
                    height=stream["height"],
                    fps=stream["fps"],
                    fourcc=stream["fourcc"],
                ),
                output_format=GF225OutputProfile.W240_H240,
            )
            devices.append((sensor_profile, device))

        raw_types = (GFDataType.TIME_STAMP, GFDataType.RAW_IMG)
        for _ in range(args.warmup_frames):
            for sensor_profile, device in devices:
                serial = sensor_profile["sdk_serial"]
                _, timestamp, raw_crc32 = _collect_new(
                    device,
                    raw_types,
                    GFDataType.TIME_STAMP,
                    GFDataType.RAW_IMG,
                    last_timestamps[serial],
                    last_raw_crc32[serial],
                    args.new_frame_timeout_s,
                )
                last_timestamps[serial] = timestamp
                last_raw_crc32[serial] = raw_crc32

        warped_types = (
            GFDataType.TIME_STAMP,
            GFDataType.RAW_IMG,
            GFDataType.WARPED_IMG,
        )
        for sensor_profile, device in devices:
            serial = sensor_profile["sdk_serial"]
            warped = None
            processor = getattr(device, "_image_processor", None)
            if processor is None or not hasattr(processor, "set_warp_params"):
                raise RuntimeError(
                    "pyvitaisdk no longer exposes the 1.0.11 image processor needed "
                    "for per-serial manual warp"
                )
            for _ in range(2):
                processor.set_warp_params(
                    corner_points=sensor_profile["corners_xy"], mode="manual"
                )
                for _ in range(12):
                    sample, timestamp, raw_crc32 = _collect_new(
                        device,
                        warped_types,
                        GFDataType.TIME_STAMP,
                        GFDataType.RAW_IMG,
                        last_timestamps[serial],
                        last_raw_crc32[serial],
                        args.new_frame_timeout_s,
                    )
                    last_timestamps[serial] = timestamp
                    last_raw_crc32[serial] = raw_crc32
                    candidate = np.asarray(sample[GFDataType.WARPED_IMG])
                    if candidate.shape == (240, 240, 3) and float(candidate.std()) >= 5.0:
                        warped = candidate
                        break
                if warped is not None:
                    break
            if warped is None:
                raise RuntimeError(
                    f"{serial}: manual WARPED_IMG is invalid: "
                    f"shape={candidate.shape}, std={float(candidate.std()):.3f}"
                )

        calibration_images: dict[str, np.ndarray] = {}
        if args.calibrate_unloaded:
            print(
                "Capturing SDK baselines because --calibrate-unloaded was supplied; "
                "both tactile pads must be free of contact.",
                flush=True,
            )
            for sensor_profile, device in devices:
                image = device.calibrate()
                if image is not None:
                    calibration_images[sensor_profile["sdk_serial"]] = np.asarray(image)

        requested_types = list(warped_types)
        if args.calibrate_unloaded:
            requested_types.append(GFDataType.DIFF_IMG)
        data_types = tuple(requested_types)
        records: dict[str, dict[str, Any]] = {}
        final_samples: dict[str, dict[Any, Any]] = {}
        for sensor_profile, _ in devices:
            records[sensor_profile["sdk_serial"]] = {
                "sdk_timestamps_raw": [],
                "raw_crc32": [],
            }

        for _ in range(args.frames):
            for sensor_profile, device in devices:
                serial = sensor_profile["sdk_serial"]
                sample, timestamp, raw_crc32 = _collect_new(
                    device,
                    data_types,
                    GFDataType.TIME_STAMP,
                    GFDataType.RAW_IMG,
                    last_timestamps[serial],
                    last_raw_crc32[serial],
                    args.new_frame_timeout_s,
                )
                last_timestamps[serial] = timestamp
                last_raw_crc32[serial] = raw_crc32
                records[serial]["sdk_timestamps_raw"].append(timestamp)
                records[serial]["raw_crc32"].append(f"{raw_crc32:08x}")
                final_samples[serial] = sample

        manifest_sensors: list[dict[str, Any]] = []
        for sensor_profile, _ in devices:
            name = sensor_profile["name"]
            serial = sensor_profile["sdk_serial"]
            sensor_dir = args.output_dir / name
            sensor_dir.mkdir()
            sample = final_samples[serial]
            raw = np.asarray(sample[GFDataType.RAW_IMG])
            warped = np.asarray(sample[GFDataType.WARPED_IMG])
            _write_png(sensor_dir / "raw.png", raw)
            _write_png(sensor_dir / "warped.png", warped)

            sensor_manifest: dict[str, Any] = {
                "name": name,
                "sdk_serial": serial,
                "logical_side": sensor_profile.get("logical_side"),
                "corners_xy": sensor_profile["corners_xy"],
                "raw": _image_stats(raw),
                "warped": _image_stats(warped),
                "sdk_timestamps_raw": records[serial]["sdk_timestamps_raw"],
                "unique_sdk_timestamp_count": len(
                    set(records[serial]["sdk_timestamps_raw"])
                ),
                "raw_crc32": records[serial]["raw_crc32"],
                "unique_raw_frame_count": len(set(records[serial]["raw_crc32"])),
                "force6d_requested": False,
                "force6d_calibrated": False,
            }
            timestamp_deltas = np.diff(records[serial]["sdk_timestamps_raw"])
            if timestamp_deltas.size:
                sensor_manifest["median_sdk_timestamp_delta_raw"] = float(
                    statistics.median(timestamp_deltas.tolist())
                )

            if args.calibrate_unloaded:
                diff = np.asarray(sample[GFDataType.DIFF_IMG])
                _write_png(sensor_dir / "diff.png", diff)
                sensor_manifest["diff"] = _image_stats(diff)
                diff_centered = diff.astype(np.int16) - 127
                np.save(sensor_dir / "diff_centered.npy", diff_centered, allow_pickle=False)
                sensor_manifest["diff_centered"] = _image_stats(diff_centered)
                calibration_image = calibration_images.get(serial)
                if calibration_image is not None:
                    _write_png(sensor_dir / "calibration.png", calibration_image)
                    sensor_manifest["calibration"] = _image_stats(calibration_image)
            manifest_sensors.append(sensor_manifest)

        manifest = {
            "schema": OUTPUT_SCHEMA,
            "profile": str(args.profile.resolve()),
            "profile_status": profile["status"],
            "stream": stream,
            "output": profile["output"],
            "sdk_timestamp_semantics": (
                "Raw values returned by GFDataType.TIME_STAMP; no unit conversion or "
                "host-time substitution was applied."
            ),
            "frame_freshness": (
                "A saved sample requires both the SDK timestamp and the CRC32 of RAW_IMG "
                "to advance; this SDK can change TIME_STAMP while returning a cached image."
            ),
            "diff_encoding": (
                "SDK DIFF_IMG is uint8 with neutral value 127. diff_centered.npy is "
                "int16(DIFF_IMG)-127 and is the zero-centred form for evaluators."
            ),
            "calibration": {
                "requested": bool(args.calibrate_unloaded),
                "operator_asserted_both_pads_unloaded": bool(args.calibrate_unloaded),
                "persistent_profile_confirmation": False,
            },
            "force6d": {
                "requested": False,
                "calibrated": False,
                "reason": "axes, signs, units, noise, and latency are not qualified",
            },
            "sensors": manifest_sensors,
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        if args.output_dir.exists() and not any(args.output_dir.iterdir()):
            args.output_dir.rmdir()
        raise
    finally:
        for _, device in devices:
            release = getattr(device, "release", None)
            if release is not None:
                release()

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
