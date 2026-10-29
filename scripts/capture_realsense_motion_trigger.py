#!/usr/bin/env python3
"""Record aligned RealSense RGB-D around RGB motion in a selected image ROI."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import capture_realsense_rgbd as capture
import cv2
import numpy as np
import pyrealsense2 as rs


@dataclass(frozen=True)
class MotionStats:
    changed_pixels: int
    changed_fraction: float
    largest_component_area_px: int
    largest_component_bbox_xywh: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class BufferedFrame:
    rgb: np.ndarray
    depth_raw: np.ndarray
    pair: capture.PairState
    motion: MotionStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--serial", required=True, help="Exact RealSense serial number")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        required=True,
        help="Motion-detection ROI in RGB pixels; upper bounds are exclusive",
    )
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument(
        "--background-frames",
        type=int,
        default=30,
        help="Still frames whose per-pixel median becomes the RGB background",
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=float,
        default=60.0,
        help="Maximum armed time; a timeout writes no RGB-D sequence",
    )
    parser.add_argument(
        "--pre-roll-seconds",
        type=float,
        default=2.0,
        help="Amount retained in RAM before the trigger",
    )
    post = parser.add_mutually_exclusive_group()
    post.add_argument(
        "--post-trigger-seconds",
        type=float,
        help="Frames to save after the trigger, converted using --fps",
    )
    post.add_argument(
        "--post-trigger-frames",
        type=int,
        help="Exact number of frames to save after the trigger",
    )
    parser.add_argument(
        "--max-output-frames",
        type=int,
        default=600,
        help="Reject settings that could save more than this many frames",
    )
    parser.add_argument(
        "--motion-threshold",
        type=int,
        default=25,
        help="Minimum blurred max-channel RGB difference on the 0..255 scale",
    )
    parser.add_argument(
        "--min-motion-area-px",
        type=int,
        default=900,
        help="Minimum connected changed area for hand-sized motion",
    )
    parser.add_argument(
        "--trigger-consecutive-frames",
        type=int,
        default=3,
        help="Required consecutive frames above the motion-area threshold",
    )
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--max-pair-skew-ms",
        type=float,
        default=20.0,
        help="Maximum color/depth device timestamp difference for an accepted pair",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[tuple[int, int, int, int], int, int]:
    for name in ("width", "height", "fps", "background_frames", "timeout_ms"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative")
    if args.max_wait_seconds <= 0.0:
        raise ValueError("--max-wait-seconds must be positive")
    if args.pre_roll_seconds < 0.0:
        raise ValueError("--pre-roll-seconds must be non-negative")
    if args.max_pair_skew_ms <= 0.0:
        raise ValueError("--max-pair-skew-ms must be positive")
    if not 1 <= args.motion_threshold <= 255:
        raise ValueError("--motion-threshold must be in [1, 255]")
    if args.min_motion_area_px <= 0:
        raise ValueError("--min-motion-area-px must be positive")
    if args.trigger_consecutive_frames <= 0:
        raise ValueError("--trigger-consecutive-frames must be positive")
    if args.max_output_frames <= 0:
        raise ValueError("--max-output-frames must be positive")

    x0, y0, x1, y1 = args.roi
    if not (0 <= x0 < x1 <= args.width and 0 <= y0 < y1 <= args.height):
        raise ValueError(
            f"--roi must satisfy 0 <= X0 < X1 <= {args.width} and "
            f"0 <= Y0 < Y1 <= {args.height}; got {args.roi}"
        )
    roi_area = (x1 - x0) * (y1 - y0)
    if args.min_motion_area_px > roi_area:
        raise ValueError(
            f"--min-motion-area-px {args.min_motion_area_px} exceeds ROI area {roi_area}"
        )

    pre_roll_frames = int(math.ceil(args.pre_roll_seconds * args.fps))
    if args.post_trigger_frames is not None:
        if args.post_trigger_frames <= 0:
            raise ValueError("--post-trigger-frames must be positive")
        post_trigger_frames = args.post_trigger_frames
    else:
        seconds = 12.0 if args.post_trigger_seconds is None else args.post_trigger_seconds
        if seconds <= 0.0:
            raise ValueError("--post-trigger-seconds must be positive")
        post_trigger_frames = int(math.ceil(seconds * args.fps))
    maximum_frames = pre_roll_frames + 1 + post_trigger_frames
    if maximum_frames > args.max_output_frames:
        raise ValueError(
            f"requested capture can write {maximum_frames} frames, above "
            f"--max-output-frames={args.max_output_frames}"
        )
    return (x0, y0, x1, y1), pre_roll_frames, post_trigger_frames


def validate_output_target(output_dir: Path) -> None:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output directory must be empty: {output_dir}")


def frame_arrays(
    color_frame: Any,
    depth_frame: Any,
    width: int,
    height: int,
    reference_intrinsics: Any | None,
) -> tuple[np.ndarray, np.ndarray, Any]:
    rgb = np.asanyarray(color_frame.get_data()).copy()
    depth_raw = np.asanyarray(depth_frame.get_data()).copy()
    expected_rgb_shape = (height, width, 3)
    expected_depth_shape = (height, width)
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
    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    if reference_intrinsics is not None and not np.allclose(
        capture.make_intrinsics_matrix(intrinsics),
        capture.make_intrinsics_matrix(reference_intrinsics),
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError("Color intrinsics changed during capture")
    return rgb, depth_raw, intrinsics


def motion_stats(
    rgb: np.ndarray,
    background_rgb: np.ndarray,
    roi: tuple[int, int, int, int],
    threshold: int,
) -> tuple[MotionStats, np.ndarray]:
    x0, y0, x1, y1 = roi
    difference = cv2.absdiff(rgb[y0:y1, x0:x1], background_rgb[y0:y1, x0:x1])
    score = np.max(difference, axis=2).astype(np.uint8)
    score = cv2.GaussianBlur(score, (5, 5), 0.0)
    changed = np.where(score >= threshold, 255, 0).astype(np.uint8)
    changed = cv2.morphologyEx(
        changed, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    )
    changed = cv2.morphologyEx(
        changed, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
    )
    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
        changed, connectivity=8
    )
    largest_area = 0
    largest_bbox = None
    if component_count > 1:
        largest_index = 1 + int(np.argmax(component_stats[1:, cv2.CC_STAT_AREA]))
        largest_area = int(component_stats[largest_index, cv2.CC_STAT_AREA])
        largest_bbox = (
            int(component_stats[largest_index, cv2.CC_STAT_LEFT]) + x0,
            int(component_stats[largest_index, cv2.CC_STAT_TOP]) + y0,
            int(component_stats[largest_index, cv2.CC_STAT_WIDTH]),
            int(component_stats[largest_index, cv2.CC_STAT_HEIGHT]),
        )
    changed_pixels = int(np.count_nonzero(changed))
    return (
        MotionStats(
            changed_pixels=changed_pixels,
            changed_fraction=changed_pixels / changed.size,
            largest_component_area_px=largest_area,
            largest_component_bbox_xywh=largest_bbox,
        ),
        changed,
    )


def write_frame(
    frame: BufferedFrame,
    index: int,
    trigger_frame_number: int,
    trigger_timestamp_s: float,
    depth_scale_m: float,
    rgb_dir: Path,
    depth_dir: Path,
) -> dict[str, Any]:
    name = f"{index:06d}"
    rgb_path = rgb_dir / f"{name}.png"
    depth_path = depth_dir / f"{name}.npy"
    if not cv2.imwrite(str(rgb_path), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write {rgb_path}")
    depth_m = frame.depth_raw.astype(np.float32) * depth_scale_m
    np.save(depth_path, depth_m, allow_pickle=False)

    if frame.pair.color_frame_number < trigger_frame_number:
        phase = "pre_trigger"
    elif frame.pair.color_frame_number == trigger_frame_number:
        phase = "trigger"
    else:
        phase = "post_trigger"
    return {
        "index": index,
        "phase": phase,
        "rgb": f"rgb/{name}.png",
        "depth": f"depth/{name}.npy",
        "relative_to_trigger_s": (
            frame.pair.color_sensor_timestamp_s - trigger_timestamp_s
        ),
        "color_sensor_timestamp_s": frame.pair.color_sensor_timestamp_s,
        "depth_sensor_timestamp_s": frame.pair.depth_sensor_timestamp_s,
        "color_sdk_timestamp_s": frame.pair.color_sdk_timestamp_s,
        "depth_sdk_timestamp_s": frame.pair.depth_sdk_timestamp_s,
        "color_frame_number": frame.pair.color_frame_number,
        "depth_frame_number": frame.pair.depth_frame_number,
        "sensor_pair_skew_ms": frame.pair.sensor_skew_ms,
        "sdk_pair_skew_ms": frame.pair.sdk_skew_ms,
        "valid_depth_fraction": float(np.count_nonzero(frame.depth_raw) / frame.depth_raw.size),
        "motion": {
            "changed_pixels": frame.motion.changed_pixels,
            "changed_fraction": frame.motion.changed_fraction,
            "largest_component_area_px": frame.motion.largest_component_area_px,
            "largest_component_bbox_xywh": frame.motion.largest_component_bbox_xywh,
        },
    }


def main() -> int:
    args = parse_args()
    roi, pre_roll_frames, post_trigger_frames = validate_args(args)
    validate_output_target(args.output_dir)

    requested_device = capture.find_device(args.serial)
    expected_device_info = capture.device_info(requested_device)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align_to_color = rs.align(rs.stream.color)

    profile = None
    staging_dir: Path | None = None
    finalized = False
    previous_pair: capture.PairState | None = None
    intrinsics = None
    started_monotonic = time.monotonic()
    try:
        profile = pipeline.start(config)
        active_device = profile.get_device()
        active_info = capture.device_info(active_device)
        if active_info.get("serial") != args.serial:
            raise RuntimeError(f"Opened serial {active_info.get('serial')}, expected {args.serial}")
        depth_scale_m = float(active_device.first_depth_sensor().get_depth_scale())
        if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
            raise RuntimeError(f"Invalid RealSense depth scale {depth_scale_m}")

        for _ in range(args.warmup_frames):
            _, _, previous_pair = capture.wait_for_fresh_aligned_pair(
                pipeline,
                align_to_color,
                previous_pair,
                args.timeout_ms,
                args.max_pair_skew_ms,
            )

        print(
            f"Collecting {args.background_frames} still background frames; keep the ROI clear...",
            flush=True,
        )
        background_frames: list[np.ndarray] = []
        background_first_pair = None
        background_last_pair = None
        for _ in range(args.background_frames):
            color_frame, depth_frame, current_pair = capture.wait_for_fresh_aligned_pair(
                pipeline,
                align_to_color,
                previous_pair,
                args.timeout_ms,
                args.max_pair_skew_ms,
            )
            previous_pair = current_pair
            rgb, _, current_intrinsics = frame_arrays(
                color_frame, depth_frame, args.width, args.height, intrinsics
            )
            if intrinsics is None:
                intrinsics = current_intrinsics
                background_first_pair = current_pair
            background_last_pair = current_pair
            background_frames.append(rgb)
        background_rgb = np.median(np.stack(background_frames), axis=0).astype(np.uint8)
        del background_frames

        x0, y0, x1, y1 = roi
        print(
            f"Armed on RealSense {args.serial}; ROI=({x0},{y0})-({x1},{y1}), "
            f"motion area >= {args.min_motion_area_px}px for "
            f"{args.trigger_consecutive_frames} frames, wait <= {args.max_wait_seconds:.1f}s.",
            flush=True,
        )
        ring: deque[BufferedFrame] = deque(maxlen=pre_roll_frames + 1)
        armed_monotonic = time.monotonic()
        armed_deadline = armed_monotonic + args.max_wait_seconds
        consecutive = 0
        examined_frames = 0
        trigger_frame: BufferedFrame | None = None
        trigger_mask = None
        while True:
            remaining_s = armed_deadline - time.monotonic()
            if remaining_s <= 0.0:
                break
            remaining_ms = max(1, int(math.ceil(remaining_s * 1000.0)))
            try:
                color_frame, depth_frame, current_pair = (
                    capture.wait_for_fresh_aligned_pair(
                        pipeline,
                        align_to_color,
                        previous_pair,
                        min(args.timeout_ms, remaining_ms),
                        args.max_pair_skew_ms,
                    )
                )
            except RuntimeError:
                # A short final wait commonly expires between camera frames. It is the
                # normal no-trigger deadline, while an earlier camera timeout remains an
                # acquisition failure and must stay visible to the caller.
                if time.monotonic() >= armed_deadline:
                    break
                raise
            previous_pair = current_pair
            rgb, depth_raw, _ = frame_arrays(
                color_frame, depth_frame, args.width, args.height, intrinsics
            )
            stats, current_mask = motion_stats(
                rgb, background_rgb, roi, args.motion_threshold
            )
            buffered = BufferedFrame(rgb, depth_raw, current_pair, stats)
            ring.append(buffered)
            examined_frames += 1
            if stats.largest_component_area_px >= args.min_motion_area_px:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= args.trigger_consecutive_frames:
                trigger_frame = buffered
                trigger_mask = current_mask
                break

        if trigger_frame is None:
            print(
                f"No trigger after {time.monotonic() - armed_monotonic:.1f}s "
                f"({examined_frames} frames examined); no RGB-D sequence was written.",
                flush=True,
            )
            return 2

        trigger_wait_s = time.monotonic() - armed_monotonic
        trigger_stats = trigger_frame.motion
        print(
            f"Triggered after {trigger_wait_s:.2f}s at hardware timestamp "
            f"{trigger_frame.pair.color_sensor_timestamp_s:.6f}s: largest motion area "
            f"{trigger_stats.largest_component_area_px}px, bbox "
            f"{trigger_stats.largest_component_bbox_xywh}. Capturing "
            f"{post_trigger_frames} post-trigger frames...",
            flush=True,
        )

        # Keep acquiring at camera cadence before doing any PNG/NPY writes. Flushing the
        # pre-roll first can block long enough to miss the start of a short demonstration.
        post_frames: list[BufferedFrame] = []
        for _ in range(post_trigger_frames):
            color_frame, depth_frame, current_pair = capture.wait_for_fresh_aligned_pair(
                pipeline,
                align_to_color,
                previous_pair,
                args.timeout_ms,
                args.max_pair_skew_ms,
            )
            previous_pair = current_pair
            rgb, depth_raw, _ = frame_arrays(
                color_frame, depth_frame, args.width, args.height, intrinsics
            )
            stats, _ = motion_stats(rgb, background_rgb, roi, args.motion_threshold)
            post_frames.append(BufferedFrame(rgb, depth_raw, current_pair, stats))

        pipeline.stop()
        profile = None
        print(
            f"Capture complete; writing {len(ring)} buffered + {len(post_frames)} "
            "post-trigger frames...",
            flush=True,
        )
        staging_dir, rgb_dir, depth_dir = capture.prepare_staging_output(args.output_dir)
        if not cv2.imwrite(
            str(staging_dir / "background_reference.png"),
            cv2.cvtColor(background_rgb, cv2.COLOR_RGB2BGR),
        ):
            raise RuntimeError("Could not write background_reference.png")
        assert trigger_mask is not None
        full_trigger_mask = np.zeros((args.height, args.width), dtype=np.uint8)
        full_trigger_mask[y0:y1, x0:x1] = trigger_mask
        if not cv2.imwrite(str(staging_dir / "trigger_motion_mask.png"), full_trigger_mask):
            raise RuntimeError("Could not write trigger_motion_mask.png")

        frame_records: list[dict[str, Any]] = []
        timestamps_s: list[float] = []
        trigger_frame_number = trigger_frame.pair.color_frame_number
        trigger_timestamp_s = trigger_frame.pair.color_sensor_timestamp_s
        for buffered in ring:
            record = write_frame(
                buffered,
                len(frame_records),
                trigger_frame_number,
                trigger_timestamp_s,
                depth_scale_m,
                rgb_dir,
                depth_dir,
            )
            frame_records.append(record)
            timestamps_s.append(buffered.pair.color_sensor_timestamp_s)

        for buffered in post_frames:
            record = write_frame(
                buffered,
                len(frame_records),
                trigger_frame_number,
                trigger_timestamp_s,
                depth_scale_m,
                rgb_dir,
                depth_dir,
            )
            frame_records.append(record)
            timestamps_s.append(buffered.pair.color_sensor_timestamp_s)

        assert intrinsics is not None
        assert background_first_pair is not None
        assert background_last_pair is not None
        timestamps = np.asarray(timestamps_s, dtype=np.float64)
        if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0):
            raise RuntimeError("Saved color hardware timestamps are not strictly increasing")
        np.save(
            staging_dir / "K.npy",
            capture.make_intrinsics_matrix(intrinsics),
            allow_pickle=False,
        )
        np.save(staging_dir / "timestamps_s.npy", timestamps, allow_pickle=False)

        trigger_index = next(
            record["index"] for record in frame_records if record["phase"] == "trigger"
        )
        manifest = {
            "schema": "demo2gripper.realsense_motion_triggered_rgbd.v1",
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
                "coordinate_frame": "OpenCV color optical frame: +X right, +Y down, +Z forward",
            },
            "intrinsics": capture.intrinsics_dict(intrinsics),
            "timestamp": {
                "array": "timestamps_s.npy",
                "source": "color RS2_FRAME_METADATA_SENSOR_TIMESTAMP (middle of exposure)",
                "domain": "device clock",
                "unit": "seconds",
                "strictly_increasing": True,
                "sdk_timestamp_domain": trigger_frame.pair.sdk_timestamp_domain,
            },
            "synchronization": {
                "max_sensor_pair_skew_ms": args.max_pair_skew_ms,
                "color_and_depth_counters_and_timestamps_strictly_increasing": True,
            },
            "trigger": {
                "detector": "median-background max-channel RGB difference",
                "roi_xyxy": list(roi),
                "background_reference": "background_reference.png",
                "background_frames": args.background_frames,
                "background_sensor_timestamp_range_s": [
                    background_first_pair.color_sensor_timestamp_s,
                    background_last_pair.color_sensor_timestamp_s,
                ],
                "motion_threshold": args.motion_threshold,
                "min_motion_area_px": args.min_motion_area_px,
                "required_consecutive_frames": args.trigger_consecutive_frames,
                "wait_elapsed_s": trigger_wait_s,
                "examined_frames": examined_frames,
                "output_frame_index": trigger_index,
                "color_sensor_timestamp_s": trigger_timestamp_s,
                "largest_component_area_px": trigger_stats.largest_component_area_px,
                "largest_component_bbox_xywh": trigger_stats.largest_component_bbox_xywh,
                "motion_mask": "trigger_motion_mask.png",
            },
            "capture_window": {
                "requested_pre_roll_frames": pre_roll_frames,
                "saved_frames_through_trigger": len(ring),
                "saved_post_trigger_frames": post_trigger_frames,
                "saved_total_frames": len(frame_records),
                "max_output_frames": args.max_output_frames,
            },
            "warmup_frames": args.warmup_frames,
            "host_capture_elapsed_s": time.monotonic() - started_monotonic,
            "frames": frame_records,
        }
        (staging_dir / "input_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        if args.output_dir.exists():
            args.output_dir.rmdir()
        staging_dir.replace(args.output_dir)
        finalized = True
    finally:
        if profile is not None:
            pipeline.stop()
        if staging_dir is not None and not finalized and staging_dir.exists():
            shutil.rmtree(staging_dir)

    print(
        f"Saved {len(frame_records)} aligned RGB-D frames to {args.output_dir}; "
        f"trigger output index {trigger_index}, hardware timestamps "
        f"{timestamps[0]:.6f}..{timestamps[-1]:.6f}s.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
