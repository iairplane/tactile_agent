#!/usr/bin/env python3
"""Capture aligned metric RGB-D frames from one RealSense selected by serial number."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--serial", required=True, help="Exact RealSense serial number")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Discard this many synchronized frames before recording",
    )
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--max-pair-skew-ms",
        type=float,
        default=20.0,
        help="Maximum color/depth device timestamp difference for an accepted pair",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class PairState:
    color_frame_number: int
    depth_frame_number: int
    color_sensor_timestamp_s: float
    depth_sensor_timestamp_s: float
    color_sdk_timestamp_s: float
    depth_sdk_timestamp_s: float
    sdk_timestamp_domain: str

    @property
    def sensor_skew_ms(self) -> float:
        return abs(self.color_sensor_timestamp_s - self.depth_sensor_timestamp_s) * 1e3

    @property
    def sdk_skew_ms(self) -> float:
        return abs(self.color_sdk_timestamp_s - self.depth_sdk_timestamp_s) * 1e3


def device_info(device: Any) -> dict[str, str]:
    fields = {
        "name": rs.camera_info.name,
        "serial": rs.camera_info.serial_number,
        "firmware_version": rs.camera_info.firmware_version,
        "usb_type_descriptor": rs.camera_info.usb_type_descriptor,
        "physical_port": rs.camera_info.physical_port,
        "product_id": rs.camera_info.product_id,
        "product_line": rs.camera_info.product_line,
    }
    result: dict[str, str] = {}
    for name, key in fields.items():
        if device.supports(key):
            result[name] = device.get_info(key)
    return result


def find_device(serial: str) -> Any:
    devices = list(rs.context().query_devices())
    for device in devices:
        if device.get_info(rs.camera_info.serial_number) == serial:
            return device
    available = [device.get_info(rs.camera_info.serial_number) for device in devices]
    raise RuntimeError(f"RealSense serial {serial!r} not found; available serials: {available}")


def timestamp_domain_name(frame: Any) -> str:
    return str(frame.get_frame_timestamp_domain()).removeprefix("timestamp_domain.")


def sensor_timestamp_s(frame: Any) -> float:
    metadata = rs.frame_metadata_value.sensor_timestamp
    if not frame.supports_frame_metadata(metadata):
        raise RuntimeError("Frame does not expose the RealSense sensor timestamp metadata")
    return float(frame.get_frame_metadata(metadata)) * 1e-6


def pair_state(color_frame: Any, depth_frame: Any) -> PairState:
    color_domain = timestamp_domain_name(color_frame)
    depth_domain = timestamp_domain_name(depth_frame)
    if color_domain != depth_domain:
        raise RuntimeError(
            f"Color/depth SDK timestamp domains differ: color={color_domain}, depth={depth_domain}"
        )
    return PairState(
        color_frame_number=int(color_frame.get_frame_number()),
        depth_frame_number=int(depth_frame.get_frame_number()),
        color_sensor_timestamp_s=sensor_timestamp_s(color_frame),
        depth_sensor_timestamp_s=sensor_timestamp_s(depth_frame),
        color_sdk_timestamp_s=float(color_frame.get_timestamp()) * 1e-3,
        depth_sdk_timestamp_s=float(depth_frame.get_timestamp()) * 1e-3,
        sdk_timestamp_domain=color_domain,
    )


def wait_for_fresh_aligned_pair(
    pipeline: Any,
    align_to_color: Any,
    previous: PairState | None,
    timeout_ms: int,
    max_pair_skew_ms: float,
) -> tuple[Any, Any, PairState]:
    deadline = time.monotonic() + timeout_ms * 1e-3
    last_problem = "no RGB-D frames received"
    while time.monotonic() < deadline:
        remaining_ms = max(1, int((deadline - time.monotonic()) * 1e3))
        try:
            frames = pipeline.wait_for_frames(remaining_ms)
        except RuntimeError as exc:
            last_problem = str(exc)
            break
        aligned = align_to_color.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            last_problem = "frameset is missing color or aligned depth"
            continue

        state = pair_state(color_frame, depth_frame)
        problems: list[str] = []
        if state.sensor_skew_ms > max_pair_skew_ms:
            problems.append(
                f"sensor timestamp skew {state.sensor_skew_ms:.3f} ms exceeds "
                f"{max_pair_skew_ms:.3f} ms"
            )
        if previous is not None:
            if state.color_frame_number <= previous.color_frame_number:
                problems.append("color frame number did not increase")
            if state.depth_frame_number <= previous.depth_frame_number:
                problems.append("depth frame number did not increase")
            if state.color_sensor_timestamp_s <= previous.color_sensor_timestamp_s:
                problems.append("color sensor timestamp did not increase")
            if state.depth_sensor_timestamp_s <= previous.depth_sensor_timestamp_s:
                problems.append("depth sensor timestamp did not increase")
            if state.color_sdk_timestamp_s <= previous.color_sdk_timestamp_s:
                problems.append("color SDK timestamp did not increase")
            if state.depth_sdk_timestamp_s <= previous.depth_sdk_timestamp_s:
                problems.append("depth SDK timestamp did not increase")
            if state.sdk_timestamp_domain != previous.sdk_timestamp_domain:
                problems.append("SDK timestamp domain changed")
        if not problems:
            return color_frame, depth_frame, state
        last_problem = (
            f"color frame {state.color_frame_number}, depth frame {state.depth_frame_number}: "
            + "; ".join(problems)
        )

    raise RuntimeError(
        f"Timed out after {timeout_ms} ms waiting for a fresh synchronized RGB-D pair; "
        f"last observation: {last_problem}"
    )


def intrinsics_dict(intrinsics: Any) -> dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "distortion_model": str(intrinsics.model).removeprefix("distortion."),
        "distortion_coefficients": [float(value) for value in intrinsics.coeffs],
    }


def make_intrinsics_matrix(intrinsics: Any) -> np.ndarray:
    return np.asarray(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def prepare_staging_output(output_dir: Path) -> tuple[Path, Path, Path]:
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ValueError(f"Output directory must be empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent)
    )
    rgb_dir = staging_dir / "rgb"
    depth_dir = staging_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    return staging_dir, rgb_dir, depth_dir


def main() -> int:
    args = parse_args()
    for name in ("frames", "width", "height", "fps", "timeout_ms"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative")
    if args.max_pair_skew_ms <= 0:
        raise ValueError("--max-pair-skew-ms must be positive")

    requested_device = find_device(args.serial)
    expected_device_info = device_info(requested_device)
    staging_dir, rgb_dir, depth_dir = prepare_staging_output(args.output_dir)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align_to_color = rs.align(rs.stream.color)

    profile = None
    finalized = False
    frame_records: list[dict[str, Any]] = []
    timestamps_s: list[float] = []
    intrinsics = None
    depth_scale_m = None
    sdk_timestamp_domain = None
    previous_pair: PairState | None = None
    started_monotonic_ns = time.monotonic_ns()
    try:
        profile = pipeline.start(config)
        active_device = profile.get_device()
        active_info = device_info(active_device)
        if active_info.get("serial") != args.serial:
            raise RuntimeError(f"Opened serial {active_info.get('serial')}, expected {args.serial}")
        depth_scale_m = float(active_device.first_depth_sensor().get_depth_scale())

        for _ in range(args.warmup_frames):
            _, _, previous_pair = wait_for_fresh_aligned_pair(
                pipeline,
                align_to_color,
                previous_pair,
                args.timeout_ms,
                args.max_pair_skew_ms,
            )

        for index in range(args.frames):
            color_frame, depth_frame, current_pair = wait_for_fresh_aligned_pair(
                pipeline,
                align_to_color,
                previous_pair,
                args.timeout_ms,
                args.max_pair_skew_ms,
            )
            previous_pair = current_pair

            rgb = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            expected_rgb_shape = (args.height, args.width, 3)
            expected_depth_shape = (args.height, args.width)
            if rgb.shape != expected_rgb_shape or rgb.dtype != np.uint8:
                raise RuntimeError(
                    f"Unexpected RGB shape/type {rgb.shape}/{rgb.dtype}; "
                    f"expected {expected_rgb_shape}/uint8"
                )
            if depth_raw.shape != expected_depth_shape or depth_raw.dtype != np.uint16:
                raise RuntimeError(
                    f"Unexpected depth shape/type {depth_raw.shape}/{depth_raw.dtype}; "
                    f"expected {expected_depth_shape}/uint16"
                )

            current_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            if intrinsics is None:
                intrinsics = current_intrinsics
            elif not np.allclose(
                make_intrinsics_matrix(current_intrinsics),
                make_intrinsics_matrix(intrinsics),
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError("Color intrinsics changed during capture")

            if sdk_timestamp_domain is None:
                sdk_timestamp_domain = current_pair.sdk_timestamp_domain

            name = f"{index:06d}"
            rgb_path = rgb_dir / f"{name}.png"
            depth_path = depth_dir / f"{name}.npy"
            if not cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Could not write {rgb_path}")
            depth_m = depth_raw.astype(np.float32) * depth_scale_m
            np.save(depth_path, depth_m, allow_pickle=False)

            timestamps_s.append(current_pair.color_sensor_timestamp_s)
            frame_records.append(
                {
                    "index": index,
                    "rgb": f"rgb/{name}.png",
                    "depth": f"depth/{name}.npy",
                    "color_sensor_timestamp_s": current_pair.color_sensor_timestamp_s,
                    "depth_sensor_timestamp_s": current_pair.depth_sensor_timestamp_s,
                    "color_sdk_timestamp_s": current_pair.color_sdk_timestamp_s,
                    "depth_sdk_timestamp_s": current_pair.depth_sdk_timestamp_s,
                    "color_frame_number": current_pair.color_frame_number,
                    "depth_frame_number": current_pair.depth_frame_number,
                    "sensor_pair_skew_ms": current_pair.sensor_skew_ms,
                    "sdk_pair_skew_ms": current_pair.sdk_skew_ms,
                    "valid_depth_fraction": float(np.count_nonzero(depth_raw) / depth_raw.size),
                }
            )

        assert intrinsics is not None
        assert depth_scale_m is not None
        assert sdk_timestamp_domain is not None
        timestamps = np.asarray(timestamps_s, dtype=np.float64)
        np.save(staging_dir / "K.npy", make_intrinsics_matrix(intrinsics), allow_pickle=False)
        np.save(staging_dir / "timestamps_s.npy", timestamps, allow_pickle=False)

        manifest = {
            "schema": "demo2gripper.realsense_aligned_rgbd.v1",
            "device": expected_device_info,
            "stream": {
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "rgb_format": "RGB8 saved as PNG",
                "depth_source_format": "Z16",
                "depth_output": "float32 metres",
                "depth_scale_m_per_unit": depth_scale_m,
                "depth_alignment": "RealSense SDK depth-to-color reprojection",
                "coordinate_frame": ("OpenCV color optical frame: +X right, +Y down, +Z forward"),
            },
            "intrinsics": intrinsics_dict(intrinsics),
            "timestamp": {
                "array": "timestamps_s.npy",
                "source": "color RS2_FRAME_METADATA_SENSOR_TIMESTAMP (middle of exposure)",
                "domain": "device clock",
                "unit": "seconds",
                "strictly_increasing": True,
                "sdk_timestamp_domain": sdk_timestamp_domain,
            },
            "synchronization": {
                "max_sensor_pair_skew_ms": args.max_pair_skew_ms,
                "color_frame_numbers_strictly_increasing": True,
                "depth_frame_numbers_strictly_increasing": True,
                "color_sensor_timestamps_strictly_increasing": True,
                "depth_sensor_timestamps_strictly_increasing": True,
            },
            "warmup_frames": args.warmup_frames,
            "host_capture_elapsed_s": (time.monotonic_ns() - started_monotonic_ns) * 1e-9,
            "frames": frame_records,
        }
        (staging_dir / "input_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        pipeline.stop()
        profile = None
        if args.output_dir.exists():
            args.output_dir.rmdir()
        staging_dir.replace(args.output_dir)
        finalized = True
    finally:
        if profile is not None:
            pipeline.stop()
        if not finalized and staging_dir.exists():
            shutil.rmtree(staging_dir)

    print(
        f"Captured {args.frames} aligned RGB-D frames from {args.serial} to "
        f"{args.output_dir}; hardware timestamps {timestamps[0]:.6f}..{timestamps[-1]:.6f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
