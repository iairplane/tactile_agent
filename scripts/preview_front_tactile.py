#!/usr/bin/env python3
"""Preview one fixed D435 color stream and two raw GF220 video streams.

This tool never connects to the robot or gripper and never imports the ViTai SDK.
The tactile difference views are local OpenCV ``absdiff`` images.  They are not the
SDK's ``WARPED_IMG`` or ``DIFF_IMG`` outputs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

FRONT_SERIAL = "254622073804"
FRONT_V4L2 = "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:8.3.3.3:1.3-video-index0"
GF_A = "/dev/v4l/by-id/usb-HBVCAM_Camera_ViTai-GF220_GF22000184603-video-index0"
GF_B = "/dev/v4l/by-id/usb-HBVCAM_Camera_ViTai-GF220_GF22000196FB3-video-index0"
WIDTH = 640
HEIGHT = 480
TACTILE_PANEL_WIDTH = 320
TACTILE_PANEL_HEIGHT = 240
WINDOW = "Front D435 | GF220 A raw/local absdiff | GF220 B raw/local absdiff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--front-serial", default=FRONT_SERIAL)
    parser.add_argument(
        "--front-backend", choices=("v4l2", "realsense"), default="v4l2"
    )
    parser.add_argument("--front-device", type=Path, default=Path(FRONT_V4L2))
    parser.add_argument("--gf-a", type=Path, default=Path(GF_A))
    parser.add_argument("--gf-b", type=Path, default=Path(GF_B))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--baseline-warmup-frames",
        type=int,
        default=30,
        help="Complete frames to discard before initializing unloaded tactile baselines",
    )
    parser.add_argument(
        "--diff-threshold",
        type=int,
        default=15,
        help="Pixel threshold used only for the displayed local changed-pixel percentage",
    )
    parser.add_argument(
        "--smoke-frames",
        type=int,
        default=0,
        help="Exit successfully after this many complete three-camera frames; 0 runs until quit",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not create an OpenCV window; requires --smoke-frames > 0",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.baseline_warmup_frames < 0:
        parser.error("--baseline-warmup-frames must be non-negative")
    if not 0 <= args.diff_threshold <= 255:
        parser.error("--diff-threshold must be in [0, 255]")
    if args.smoke_frames < 0:
        parser.error("--smoke-frames must be non-negative")
    if args.headless and args.smoke_frames == 0:
        parser.error("--headless requires --smoke-frames > 0")
    return args


def find_realsense(context: Any, serial: str, rs: Any) -> Any:
    devices = list(context.query_devices())
    for device in devices:
        if device.get_info(rs.camera_info.serial_number) == serial:
            return device
    available = [device.get_info(rs.camera_info.serial_number) for device in devices]
    raise RuntimeError(f"RealSense {serial!r} not found; available serials: {available}")


def fourcc_name(value: float) -> str:
    integer = int(value)
    return "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4))


def open_front_v4l2(path: Path, fps: int) -> cv2.VideoCapture:
    if not path.exists():
        raise RuntimeError(f"Front D435 V4L2 device path does not exist: {path}")
    capture = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open front D435 V4L2 stream {path}; another process may own it")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fourcc_name(capture.get(cv2.CAP_PROP_FOURCC)),
    )
    if actual != (WIDTH, HEIGHT, "YUYV"):
        capture.release()
        raise RuntimeError(
            f"Front D435 negotiated {actual[0]}x{actual[1]} {actual[2]!r}; "
            f"expected {WIDTH}x{HEIGHT} YUYV"
        )
    return capture


def read_front_v4l2(capture: cv2.VideoCapture) -> np.ndarray:
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError("No frame from front D435 V4L2 stream")
    if frame.shape != (HEIGHT, WIDTH, 3) or frame.dtype != np.uint8:
        raise RuntimeError(
            f"Front D435 returned {frame.shape}/{frame.dtype}; "
            f"expected {(HEIGHT, WIDTH, 3)}/uint8 after YUYV conversion"
        )
    return frame


def open_gf220(path: Path, fps: int) -> cv2.VideoCapture:
    if not path.exists():
        raise RuntimeError(f"GF220 device path does not exist: {path}")
    capture = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open GF220 {path}; another process may own it")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def read_gf220(capture: cv2.VideoCapture, name: str) -> np.ndarray:
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"No frame from GF220 {name}")
    if frame.shape != (HEIGHT, WIDTH, 3) or frame.dtype != np.uint8:
        raise RuntimeError(
            f"GF220 {name} returned {frame.shape}/{frame.dtype}; "
            f"expected {(HEIGHT, WIDTH, 3)}/uint8"
        )
    return frame


def label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    result = image.copy()
    box_height = 10 + 22 * len(lines)
    cv2.rectangle(result, (0, 0), (result.shape[1], box_height), (18, 18, 18), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            result,
            line,
            (8, 22 + 22 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return result


def tactile_panel(
    frame: np.ndarray,
    baseline: np.ndarray,
    name: str,
    threshold: int,
) -> tuple[np.ndarray, str]:
    difference = cv2.absdiff(frame, baseline)
    magnitude = np.max(difference, axis=2).astype(np.uint8)
    changed_fraction = float(np.mean(magnitude > threshold))
    mean_change = float(np.mean(magnitude))
    p95_change = float(np.percentile(magnitude, 95))
    status = f"mean={mean_change:.1f} p95={p95_change:.0f} changed={changed_fraction:.1%}"

    raw = cv2.resize(frame, (TACTILE_PANEL_WIDTH, TACTILE_PANEL_HEIGHT))
    heat = cv2.applyColorMap(
        cv2.resize(magnitude, (TACTILE_PANEL_WIDTH, TACTILE_PANEL_HEIGHT)),
        cv2.COLORMAP_INFERNO,
    )
    raw = label(raw, [f"GF220 {name} RAW 640x480"])
    heat = label(heat, [f"{name} LOCAL absdiff (not SDK)", status])
    return np.vstack((raw, heat)), status


def run(args: argparse.Namespace) -> int:
    if not args.headless and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("No desktop display found; use a desktop terminal or --headless")
    if args.gf_a.resolve() == args.gf_b.resolve():
        raise RuntimeError("--gf-a and --gf-b resolve to the same video device")

    rs: Any | None = None
    pipeline: Any | None = None
    front_capture: cv2.VideoCapture | None = None
    front_name = "Intel RealSense D435 color"
    if args.front_backend == "realsense":
        try:
            import pyrealsense2 as rs_module
        except ImportError as exc:
            raise RuntimeError(
                "--front-backend realsense requires pyrealsense2; use the default V4L2 backend"
            ) from exc
        rs = rs_module
        context = rs.context()
        device = find_realsense(context, args.front_serial, rs)
        front_name = device.get_info(rs.camera_info.name)
        pipeline = rs.pipeline(context)
    capture_a: cv2.VideoCapture | None = None
    capture_b: cv2.VideoCapture | None = None
    readers: ThreadPoolExecutor | None = None
    pipeline_started = False
    window_created = False
    complete_frames = 0
    baseline_a: np.ndarray | None = None
    baseline_b: np.ndarray | None = None
    preview_started_at: float | None = None

    try:
        if args.front_backend == "v4l2":
            front_capture = open_front_v4l2(args.front_device, args.fps)
        capture_a = open_gf220(args.gf_a, args.fps)
        capture_b = open_gf220(args.gf_b, args.fps)

        if args.front_backend == "realsense":
            assert rs is not None and pipeline is not None
            config = rs.config()
            config.enable_device(args.front_serial)
            config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, args.fps)
            pipeline.start(config)
            pipeline_started = True
        readers = ThreadPoolExecutor(max_workers=3, thread_name_prefix="camera")

        if args.front_backend == "v4l2":
            print(
                f"Front: {front_name}, serial {args.front_serial}, "
                f"V4L2 YUYV {args.front_device} -> {args.front_device.resolve()}",
                flush=True,
            )
        else:
            print(f"Front: {front_name}, serial {args.front_serial}, RealSense SDK", flush=True)
        print(f"GF220 A: {args.gf_a} -> {args.gf_a.resolve()}", flush=True)
        print(f"GF220 B: {args.gf_b} -> {args.gf_b.resolve()}", flush=True)
        print(
            "A/B are hardware identities; logical left/right is not calibrated. "
            "Displayed tactile differences are local absdiff, not SDK WARPED_IMG/DIFF_IMG.",
            flush=True,
        )
        print(
            "This process owns all three cameras. Stop it before another recorder opens them.",
            flush=True,
        )

        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 1280, 480)
            window_created = True
            print("Keys: b = reset local unloaded baselines, q/Esc = quit", flush=True)

        preview_started_at = time.monotonic()
        while True:
            future_a = readers.submit(read_gf220, capture_a, "A")
            future_b = readers.submit(read_gf220, capture_b, "B")
            if args.front_backend == "v4l2":
                assert front_capture is not None
                front = readers.submit(read_front_v4l2, front_capture).result()
            else:
                assert pipeline is not None
                frames = pipeline.wait_for_frames(5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    raise RuntimeError("D435 frameset does not contain a color frame")
                front = np.asanyarray(color_frame.get_data()).copy()
            if front.shape != (HEIGHT, WIDTH, 3) or front.dtype != np.uint8:
                raise RuntimeError(
                    f"D435 returned {front.shape}/{front.dtype}; "
                    f"expected {(HEIGHT, WIDTH, 3)}/uint8"
                )

            tactile_a = future_a.result()
            tactile_b = future_b.result()
            if (
                baseline_a is None
                and baseline_b is None
                and complete_frames >= args.baseline_warmup_frames
            ):
                baseline_a = tactile_a.copy()
                baseline_b = tactile_b.copy()
                print(
                    "Initialized local unloaded baselines after camera warm-up.",
                    flush=True,
                )

            display_baseline_a = tactile_a if baseline_a is None else baseline_a
            display_baseline_b = tactile_b if baseline_b is None else baseline_b
            panel_a, status_a = tactile_panel(
                tactile_a, display_baseline_a, "A", args.diff_threshold
            )
            panel_b, status_b = tactile_panel(
                tactile_b, display_baseline_b, "B", args.diff_threshold
            )
            front_panel = label(
                front,
                [f"FRONT D435 {args.front_serial} COLOR ({args.front_backend.upper()})"],
            )
            canvas = np.hstack((front_panel, panel_a, panel_b))
            if canvas.shape != (480, 1280, 3):
                raise RuntimeError(f"Internal preview shape error: {canvas.shape}")

            complete_frames += 1
            if not args.headless:
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("b"), ord("B")):
                    baseline_a = tactile_a.copy()
                    baseline_b = tactile_b.copy()
                    print("Reset both local unloaded baselines.", flush=True)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break

            if args.smoke_frames and complete_frames >= args.smoke_frames:
                print(f"Last local diff: A {status_a}; B {status_b}", flush=True)
                break
    finally:
        if readers is not None:
            readers.shutdown(wait=True, cancel_futures=True)
        if pipeline_started and pipeline is not None:
            pipeline.stop()
        if capture_b is not None:
            capture_b.release()
        if capture_a is not None:
            capture_a.release()
        if front_capture is not None:
            front_capture.release()
        if window_created:
            cv2.destroyAllWindows()

    elapsed_s = 0.0 if preview_started_at is None else time.monotonic() - preview_started_at
    rate = complete_frames / elapsed_s if elapsed_s > 0 else 0.0
    print(f"Preview stopped after {complete_frames} complete frames ({rate:.1f} Hz).", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nPreview stopped.", flush=True)
        return 0
    except (RuntimeError, OSError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
