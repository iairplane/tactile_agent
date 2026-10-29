#!/usr/bin/env python3
"""Preview and record one aligned D435 RGB-D stream plus two raw GF220 streams.

One process owns all three devices and serves the composed view to a local browser,
so the operator watches the same frames that are written.  The tactile views are raw
UVC images and local OpenCV ``absdiff`` diagnostics; they are not ViTai SDK
``WARPED_IMG`` or ``DIFF_IMG`` products.

Each atomically published recording receives a random immutable
``recording_session_id`` used to distinguish independent qualification trials.

Keys
----
``R`` starts/stops one recording, ``B`` resets the two displayed unloaded
baselines, and ``Q`` closes the browser-backed capture.  Quitting while recording
first finishes the output atomically.
"""

from __future__ import annotations

import argparse
import http.server
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import preview_front_tactile as preview

SCHEMA = "demo2gripper.realsense_rgbd_raw_tactile.v1"
PAGE_TITLE = "Front RGB-D + two GF220 tactile streams"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--front-serial", default=preview.FRONT_SERIAL)
    parser.add_argument("--gf-a", type=Path, default=Path(preview.GF_A))
    parser.add_argument("--gf-b", type=Path, default=Path(preview.GF_B))
    parser.add_argument("--width", type=int, default=preview.WIDTH)
    parser.add_argument("--height", type=int, default=preview.HEIGHT)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--max-pair-skew-ms", type=float, default=20.0)
    parser.add_argument("--max-tactile-age-ms", type=float, default=250.0)
    parser.add_argument("--max-d435-frame-gap", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--max-pending-writes", type=int, default=32)
    parser.add_argument("--writer-threads", type=int, default=2)
    parser.add_argument("--diff-threshold", type=int, default=15)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not launch xdg-open; print the local preview URL instead",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "width",
        "height",
        "fps",
        "timeout_ms",
        "max_d435_frame_gap",
        "max_frames",
        "max_pending_writes",
        "writer_threads",
        "port",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (args.width, args.height) != (preview.WIDTH, preview.HEIGHT):
        raise ValueError(
            f"This workcell recorder requires {preview.WIDTH}x{preview.HEIGHT}; "
            f"got {args.width}x{args.height}"
        )
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative")
    if args.max_pair_skew_ms <= 0.0:
        raise ValueError("--max-pair-skew-ms must be positive")
    if args.max_tactile_age_ms <= 0.0:
        raise ValueError("--max-tactile-age-ms must be positive")
    if not 0 <= args.diff_threshold <= 255:
        raise ValueError("--diff-threshold must be in [0, 255]")
    if args.gf_a.resolve() == args.gf_b.resolve():
        raise ValueError("--gf-a and --gf-b resolve to the same video device")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")


def validate_output_target(output_dir: Path) -> None:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output directory must be empty: {output_dir}")


def _require_array(
    value: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type[np.generic],
) -> None:
    if value.shape != shape or value.dtype != np.dtype(dtype):
        raise RuntimeError(
            f"{name} returned {value.shape}/{value.dtype}; expected {shape}/{np.dtype(dtype)}"
        )


def frame_arrays(
    color_frame: Any,
    depth_frame: Any,
    width: int,
    height: int,
    reference_intrinsics: Any | None,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Copy and validate one aligned RGB/Z16 pair from the RealSense SDK."""
    rgb = np.asanyarray(color_frame.get_data()).copy()
    depth_raw = np.asanyarray(depth_frame.get_data()).copy()
    _require_array(
        rgb,
        name="D435 RGB",
        shape=(height, width, 3),
        dtype=np.uint8,
    )
    _require_array(
        depth_raw,
        name="D435 aligned depth",
        shape=(height, width),
        dtype=np.uint16,
    )
    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    current_k = np.asarray(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if reference_intrinsics is not None:
        reference_k = np.asarray(
            [
                [reference_intrinsics.fx, 0.0, reference_intrinsics.ppx],
                [0.0, reference_intrinsics.fy, reference_intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        if not np.allclose(current_k, reference_k, rtol=0.0, atol=1e-9):
            raise RuntimeError("D435 color intrinsics changed during capture")
    return rgb, depth_raw, intrinsics


@dataclass(frozen=True)
class TactileSnapshot:
    frame_bgr: np.ndarray
    host_monotonic_s: float
    sequence: int


class LatestTactile:
    """Continuously acquire one raw GF220 stream and publish its newest frame."""

    def __init__(self, path: Path, fps: int, name: str):
        self.name = name
        self.path = path
        self._capture = preview.open_gf220(path, fps)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._frame: np.ndarray | None = None
        self._host_monotonic_s = 0.0
        self._sequence = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name=f"gf220-{name}", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                frame = preview.read_gf220(self._capture, self.name)
                captured_s = time.monotonic()
                with self._condition:
                    self._frame = frame
                    self._host_monotonic_s = captured_s
                    self._sequence += 1
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def wait_first(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._sequence == 0 and self._error is None:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    break
                self._condition.wait(remaining_s)
            if self._error is not None:
                raise RuntimeError(f"GF220 {self.name} acquisition failed") from self._error
            if self._sequence == 0:
                raise RuntimeError(
                    f"GF220 {self.name} produced no frame within {timeout_s:.1f}s"
                )

    def snapshot(self) -> TactileSnapshot:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"GF220 {self.name} acquisition failed") from self._error
            if self._frame is None or self._sequence == 0:
                raise RuntimeError(f"GF220 {self.name} has no frame")
            return TactileSnapshot(
                frame_bgr=self._frame.copy(),
                host_monotonic_s=self._host_monotonic_s,
                sequence=self._sequence,
            )

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            self._capture.release()
            self._thread.join(timeout=1.0)
        else:
            self._capture.release()


@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray
    depth_raw: np.ndarray
    pair: Any
    d435_host_monotonic_s: float
    tactile_a: TactileSnapshot
    tactile_b: TactileSnapshot


def _write_observation(
    staging_dir: Path,
    name: str,
    observation: Observation,
    depth_scale_m: float,
) -> None:
    rgb_path = staging_dir / "rgb" / f"{name}.png"
    depth_path = staging_dir / "depth" / f"{name}.npy"
    depth_raw_path = staging_dir / "depth_raw" / f"{name}.npy"
    tactile_a_path = staging_dir / "tactile_a" / f"{name}.png"
    tactile_b_path = staging_dir / "tactile_b" / f"{name}.png"

    if not cv2.imwrite(str(rgb_path), cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write {rgb_path}")
    np.save(depth_raw_path, observation.depth_raw, allow_pickle=False)
    depth_m = observation.depth_raw.astype(np.float32) * depth_scale_m
    np.save(depth_path, depth_m, allow_pickle=False)
    if not cv2.imwrite(str(tactile_a_path), observation.tactile_a.frame_bgr):
        raise RuntimeError(f"Could not write {tactile_a_path}")
    if not cv2.imwrite(str(tactile_b_path), observation.tactile_b.frame_bgr):
        raise RuntimeError(f"Could not write {tactile_b_path}")


class RecordingSession:
    """Write one bounded recording and publish it with one final directory rename."""

    def __init__(
        self,
        *,
        output_dir: Path,
        depth_scale_m: float,
        intrinsics: Any,
        d435_device: dict[str, str],
        gf_a_path: Path,
        gf_b_path: Path,
        fps: int,
        max_tactile_age_ms: float,
        max_d435_frame_gap: int,
        max_pending_writes: int,
        writer_threads: int,
    ):
        validate_output_target(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent)
        )
        for directory in ("rgb", "depth", "depth_raw", "tactile_a", "tactile_b"):
            (self.staging_dir / directory).mkdir()

        self.depth_scale_m = float(depth_scale_m)
        self.intrinsics = intrinsics
        self.d435_device = d435_device
        self.gf_a_path = gf_a_path
        self.gf_b_path = gf_b_path
        self.fps = fps
        self.max_tactile_age_ms = max_tactile_age_ms
        self.max_d435_frame_gap = max_d435_frame_gap
        self.max_pending_writes = max_pending_writes
        self._executor = ThreadPoolExecutor(
            max_workers=writer_threads, thread_name_prefix="rgbd-tactile-writer"
        )
        self._pending: deque[Future[None]] = deque()
        self.records: list[dict[str, Any]] = []
        self.recording_session_id = uuid.uuid4().hex
        self.started_host_monotonic_s = time.monotonic()
        self._finished = False

    def _drain_one(self) -> None:
        self._pending.popleft().result()

    def append(self, observation: Observation) -> None:
        if self._finished:
            raise RuntimeError("Recording session is already finished")
        _require_array(
            observation.rgb,
            name="D435 RGB",
            shape=(preview.HEIGHT, preview.WIDTH, 3),
            dtype=np.uint8,
        )
        _require_array(
            observation.depth_raw,
            name="D435 aligned depth",
            shape=(preview.HEIGHT, preview.WIDTH),
            dtype=np.uint16,
        )
        for label, snapshot in (
            ("GF220 A", observation.tactile_a),
            ("GF220 B", observation.tactile_b),
        ):
            _require_array(
                snapshot.frame_bgr,
                name=label,
                shape=(preview.HEIGHT, preview.WIDTH, 3),
                dtype=np.uint8,
            )

        pair = observation.pair
        if self.records:
            previous = self.records[-1]
            if pair.color_frame_number <= previous["color_frame_number"]:
                raise RuntimeError("D435 color frame number did not increase")
            if pair.depth_frame_number <= previous["depth_frame_number"]:
                raise RuntimeError("D435 depth frame number did not increase")
            color_gap = pair.color_frame_number - previous["color_frame_number"]
            depth_gap = pair.depth_frame_number - previous["depth_frame_number"]
            if color_gap > self.max_d435_frame_gap or depth_gap > self.max_d435_frame_gap:
                raise RuntimeError(
                    "D435 frame gap exceeded the recording bound: "
                    f"color={color_gap}, depth={depth_gap}, max={self.max_d435_frame_gap}"
                )
            if observation.d435_host_monotonic_s <= previous["d435_host_monotonic_s"]:
                raise RuntimeError("D435 host monotonic timestamp did not increase")

        tactile_a_delta_ms = (
            observation.tactile_a.host_monotonic_s - observation.d435_host_monotonic_s
        ) * 1e3
        tactile_b_delta_ms = (
            observation.tactile_b.host_monotonic_s - observation.d435_host_monotonic_s
        ) * 1e3
        if abs(tactile_a_delta_ms) > self.max_tactile_age_ms:
            raise RuntimeError(
                f"GF220 A is {tactile_a_delta_ms:+.1f} ms from the D435 host timestamp; "
                f"limit is {self.max_tactile_age_ms:.1f} ms"
            )
        if abs(tactile_b_delta_ms) > self.max_tactile_age_ms:
            raise RuntimeError(
                f"GF220 B is {tactile_b_delta_ms:+.1f} ms from the D435 host timestamp; "
                f"limit is {self.max_tactile_age_ms:.1f} ms"
            )

        index = len(self.records)
        name = f"{index:06d}"
        record = {
            "index": index,
            "rgb": f"rgb/{name}.png",
            "depth": f"depth/{name}.npy",
            "depth_raw": f"depth_raw/{name}.npy",
            "tactile_a": f"tactile_a/{name}.png",
            "tactile_b": f"tactile_b/{name}.png",
            "color_sensor_timestamp_s": float(pair.color_sensor_timestamp_s),
            "depth_sensor_timestamp_s": float(pair.depth_sensor_timestamp_s),
            "color_sdk_timestamp_s": float(pair.color_sdk_timestamp_s),
            "depth_sdk_timestamp_s": float(pair.depth_sdk_timestamp_s),
            "sdk_timestamp_domain": str(pair.sdk_timestamp_domain),
            "color_frame_number": int(pair.color_frame_number),
            "depth_frame_number": int(pair.depth_frame_number),
            "sensor_pair_skew_ms": float(pair.sensor_skew_ms),
            "sdk_pair_skew_ms": float(pair.sdk_skew_ms),
            "d435_host_monotonic_s": observation.d435_host_monotonic_s,
            "tactile_a_host_monotonic_s": observation.tactile_a.host_monotonic_s,
            "tactile_b_host_monotonic_s": observation.tactile_b.host_monotonic_s,
            "tactile_a_minus_d435_host_ms": tactile_a_delta_ms,
            "tactile_b_minus_d435_host_ms": tactile_b_delta_ms,
            "tactile_a_sequence": observation.tactile_a.sequence,
            "tactile_b_sequence": observation.tactile_b.sequence,
            "valid_depth_fraction": float(
                np.count_nonzero(observation.depth_raw) / observation.depth_raw.size
            ),
        }
        self.records.append(record)

        while len(self._pending) >= self.max_pending_writes:
            self._drain_one()
        self._pending.append(
            self._executor.submit(
                _write_observation,
                self.staging_dir,
                name,
                observation,
                self.depth_scale_m,
            )
        )

    def finalize(self, intrinsics_dict: dict[str, Any], intrinsics_matrix: np.ndarray) -> None:
        if self._finished:
            return
        if not self.records:
            raise RuntimeError("Refusing to finalize an empty recording")
        try:
            while self._pending:
                self._drain_one()
            self._executor.shutdown(wait=True, cancel_futures=False)

            np.save(self.staging_dir / "K.npy", intrinsics_matrix, allow_pickle=False)
            arrays = {
                "timestamps_s.npy": [r["color_sensor_timestamp_s"] for r in self.records],
                "d435_host_monotonic_s.npy": [
                    r["d435_host_monotonic_s"] for r in self.records
                ],
                "tactile_a_host_monotonic_s.npy": [
                    r["tactile_a_host_monotonic_s"] for r in self.records
                ],
                "tactile_b_host_monotonic_s.npy": [
                    r["tactile_b_host_monotonic_s"] for r in self.records
                ],
            }
            for filename, values in arrays.items():
                np.save(
                    self.staging_dir / filename,
                    np.asarray(values, dtype=np.float64),
                    allow_pickle=False,
                )

            manifest = {
                "schema": SCHEMA,
                "recording_session_id": self.recording_session_id,
                "device": self.d435_device,
                "stream": {
                    "width": preview.WIDTH,
                    "height": preview.HEIGHT,
                    "requested_fps": self.fps,
                    "rgb_format": "RGB8 saved losslessly as PNG",
                    "depth_source_format": "Z16 saved unchanged in depth_raw/*.npy",
                    "depth_output": "float32 metres in depth/*.npy",
                    "depth_scale_m_per_unit": self.depth_scale_m,
                    "depth_alignment": "RealSense SDK depth-to-color reprojection",
                    "coordinate_frame": "OpenCV color optical frame: +X right, +Y down, +Z forward",
                },
                "intrinsics": intrinsics_dict,
                "timestamp": {
                    "d435_device_array": "timestamps_s.npy",
                    "d435_device_source": (
                        "color RS2_FRAME_METADATA_SENSOR_TIMESTAMP (middle of exposure)"
                    ),
                    "d435_sdk_domain": self.records[0]["sdk_timestamp_domain"],
                    "d435_host_array": "d435_host_monotonic_s.npy",
                    "tactile_a_host_array": "tactile_a_host_monotonic_s.npy",
                    "tactile_b_host_array": "tactile_b_host_monotonic_s.npy",
                    "host_clock": "time.monotonic() sampled after each blocking read",
                    "tactile_limitation": (
                        "Raw V4L2 exposes no trusted sensor timestamp; tactile times are host "
                        "read-completion times and are not exposure timestamps"
                    ),
                },
                "tactile": {
                    "a": {
                        "device": str(self.gf_a_path),
                        "resolved_device": str(self.gf_a_path.resolve()),
                    },
                    "b": {
                        "device": str(self.gf_b_path),
                        "resolved_device": str(self.gf_b_path.resolve()),
                    },
                    "format": "raw UVC BGR8 saved losslessly as PNG",
                    "semantics": (
                        "Hardware identities A/B are not calibrated to logical left/right. "
                        "These are raw GF220 frames, not ViTai SDK WARPED_IMG or DIFF_IMG. "
                        "The displayed absdiff is diagnostic only and is not saved as sensor data."
                    ),
                    "maximum_absolute_host_offset_ms": self.max_tactile_age_ms,
                },
                "synchronization": {
                    "join": (
                        "one latest tactile snapshot associated with each accepted "
                        "aligned RGB-D pair"
                    ),
                    "duplicate_tactile_frames": (
                        "possible when D435 is faster; use tactile_*_sequence to identify reuse"
                    ),
                    "max_d435_frame_gap": self.max_d435_frame_gap,
                },
                "host_recording_elapsed_s": time.monotonic()
                - self.started_host_monotonic_s,
                "frames": self.records,
            }
            (self.staging_dir / "input_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

            validate_output_target(self.output_dir)
            if self.output_dir.exists():
                self.output_dir.rmdir()
            self.staging_dir.replace(self.output_dir)
            self._finished = True
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        if self._finished:
            return
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
        self._finished = True


class BrowserPreview:
    """Serve the latest composed frame and enqueue operator controls on localhost."""

    _HTML = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>{PAGE_TITLE}</title>
<style>
body {{ margin: 0; background: #111; color: #eee; font: 16px sans-serif; }}
main {{ max-width: 1280px; margin: 12px auto; padding: 0 12px; }}
img {{ width: 100%; height: auto; border: 1px solid #444; display: block; }}
button {{ margin: 12px 8px 0 0; padding: 10px 18px; font-size: 16px; }}
.note {{ color: #f0c674; }}
</style></head><body><main>
<h2>{PAGE_TITLE}</h2>
<img src=\"/stream.mjpg\" alt=\"live camera stream\">
<div>
<button onclick=\"send('record')\">R — start/finish recording</button>
<button onclick=\"send('baseline')\">B — reset unloaded baselines</button>
<button onclick=\"send('quit')\">Q — finish and quit</button>
</div>
<p class=\"note\">GF220 panels are raw UVC/local absdiff, not SDK WARPED_IMG/DIFF_IMG.</p>
<script>
function send(action) {{ fetch('/control?action=' + action, {{method: 'POST'}}); }}
document.addEventListener('keydown', (event) => {{
  const key = event.key.toLowerCase();
  if (key === 'r') send('record');
  if (key === 'b') send('baseline');
  if (key === 'q') send('quit');
}});
</script></main></body></html>""".encode()

    def __init__(self, host: str, port: int):
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._stopping = False
        self._controls: deque[str] = deque()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                owner._handle(self)

            def do_POST(self) -> None:  # noqa: N802
                owner._handle(self)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self._server = http.server.ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="rgbd-tactile-http",
            daemon=True,
        )
        self._thread.start()
        bound_host, bound_port = self._server.server_address[:2]
        display_host = "127.0.0.1" if bound_host in ("0.0.0.0", "::") else bound_host
        self.url = f"http://{display_host}:{bound_port}/"

    def _handle(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        parsed = urllib.parse.urlparse(handler.path)
        if parsed.path == "/":
            handler.send_response(200)
            handler.send_header("Content-Type", "text/html; charset=utf-8")
            handler.send_header("Content-Length", str(len(self._HTML)))
            handler.send_header("Cache-Control", "no-store")
            handler.end_headers()
            handler.wfile.write(self._HTML)
            return
        if parsed.path == "/control":
            action = urllib.parse.parse_qs(parsed.query).get("action", [""])[0]
            if action not in {"record", "baseline", "quit"}:
                handler.send_error(400, "unknown control")
                return
            with self._condition:
                self._controls.append(action)
            handler.send_response(204)
            handler.end_headers()
            return
        if parsed.path != "/stream.mjpg":
            handler.send_error(404)
            return

        handler.send_response(200)
        handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        handler.send_header("Pragma", "no-cache")
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.end_headers()
        last_sequence = -1
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stopping or self._sequence != last_sequence,
                        timeout=1.0,
                    )
                    if self._stopping:
                        return
                    jpeg = self._jpeg
                    last_sequence = self._sequence
                if jpeg is None:
                    continue
                handler.wfile.write(b"--frame\r\n")
                handler.wfile.write(b"Content-Type: image/jpeg\r\n")
                handler.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                handler.wfile.write(jpeg)
                handler.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return

    def publish(self, canvas: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("Could not encode the browser preview frame")
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._sequence += 1
            self._condition.notify_all()

    def pop_controls(self) -> list[str]:
        with self._condition:
            controls = list(self._controls)
            self._controls.clear()
        return controls

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    validate_output_target(args.output_dir)

    try:
        import capture_realsense_rgbd as capture
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "This recorder needs pyrealsense2; run it with /home/vitai/miniconda3/bin/python"
        ) from exc

    requested_device = capture.find_device(args.front_serial)
    expected_device_info = capture.device_info(requested_device)
    context = rs.context()
    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(args.front_serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align_to_color = rs.align(rs.stream.color)

    profile = None
    tactile_a: LatestTactile | None = None
    tactile_b: LatestTactile | None = None
    recording: RecordingSession | None = None
    recording_completed = False
    browser: BrowserPreview | None = None
    previous_pair = None
    reference_intrinsics = None
    depth_scale_m = 0.0

    def finalize_recording() -> None:
        nonlocal recording, recording_completed
        if recording is None:
            return
        assert reference_intrinsics is not None
        count = len(recording.records)
        recording.finalize(
            capture.intrinsics_dict(reference_intrinsics),
            capture.make_intrinsics_matrix(reference_intrinsics),
        )
        recording = None
        recording_completed = True
        print(f"Finalized {count} synchronized rows at {args.output_dir}", flush=True)

    try:
        profile = pipeline.start(config)
        active_device = profile.get_device()
        active_info = capture.device_info(active_device)
        if active_info.get("serial") != args.front_serial:
            raise RuntimeError(
                f"Opened RealSense {active_info.get('serial')!r}, expected {args.front_serial!r}"
            )
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

        tactile_a = LatestTactile(args.gf_a, args.fps, "A")
        tactile_b = LatestTactile(args.gf_b, args.fps, "B")
        tactile_a.wait_first(args.timeout_ms * 1e-3)
        tactile_b.wait_first(args.timeout_ms * 1e-3)

        browser = BrowserPreview(args.listen_host, args.port)
        print(f"Front aligned RGB-D: {args.front_serial}", flush=True)
        print(f"GF220 A: {args.gf_a} -> {args.gf_a.resolve()}", flush=True)
        print(f"GF220 B: {args.gf_b} -> {args.gf_b.resolve()}", flush=True)
        print(
            "Raw tactile + local absdiff only; not SDK WARPED_IMG/DIFF_IMG. "
            "Browser keys/buttons: R record/finish, B reset unloaded baselines, "
            "Q finish then quit.",
            flush=True,
        )
        print(f"Live preview: {browser.url}", flush=True)
        if not args.no_browser:
            try:
                subprocess.Popen(  # noqa: S603
                    ["xdg-open", browser.url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError as exc:
                print(f"Could not launch browser ({exc}); open {browser.url}", flush=True)

        baseline_a: np.ndarray | None = None
        baseline_b: np.ndarray | None = None
        while True:
            color_frame, depth_frame, current_pair = capture.wait_for_fresh_aligned_pair(
                pipeline,
                align_to_color,
                previous_pair,
                args.timeout_ms,
                args.max_pair_skew_ms,
            )
            previous_pair = current_pair
            d435_host_monotonic_s = time.monotonic()
            rgb, depth_raw, current_intrinsics = frame_arrays(
                color_frame,
                depth_frame,
                args.width,
                args.height,
                reference_intrinsics,
            )
            if reference_intrinsics is None:
                reference_intrinsics = current_intrinsics
            snapshot_a = tactile_a.snapshot()
            snapshot_b = tactile_b.snapshot()
            if baseline_a is None or baseline_b is None:
                baseline_a = snapshot_a.frame_bgr.copy()
                baseline_b = snapshot_b.frame_bgr.copy()
                print("Initialized both local absdiff baselines.", flush=True)

            observation = Observation(
                rgb=rgb,
                depth_raw=depth_raw,
                pair=current_pair,
                d435_host_monotonic_s=d435_host_monotonic_s,
                tactile_a=snapshot_a,
                tactile_b=snapshot_b,
            )
            if recording is not None:
                recording.append(observation)
                if len(recording.records) >= args.max_frames:
                    print(f"Reached --max-frames={args.max_frames}; finalizing.", flush=True)
                    finalize_recording()

            panel_a, _ = preview.tactile_panel(
                snapshot_a.frame_bgr,
                baseline_a,
                "A",
                args.diff_threshold,
            )
            panel_b, _ = preview.tactile_panel(
                snapshot_b.frame_bgr,
                baseline_b,
                "B",
                args.diff_threshold,
            )
            state = (
                f"REC {len(recording.records):04d}/{args.max_frames}"
                if recording is not None
                else ("SAVED" if recording_completed else "READY - press R")
            )
            front_panel = preview.label(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [f"FRONT D435 {args.front_serial} ALIGNED RGB-D", state],
            )
            canvas = np.hstack((front_panel, panel_a, panel_b))
            if canvas.shape != (480, 1280, 3):
                raise RuntimeError(f"Internal preview shape error: {canvas.shape}")
            browser.publish(canvas)

            should_quit = False
            for control in browser.pop_controls():
                if control == "baseline":
                    baseline_a = snapshot_a.frame_bgr.copy()
                    baseline_b = snapshot_b.frame_bgr.copy()
                    print("Reset both local unloaded baselines.", flush=True)
                elif control == "record":
                    if recording is not None:
                        finalize_recording()
                    elif recording_completed:
                        print("This invocation already completed its one recording.", flush=True)
                    else:
                        assert reference_intrinsics is not None
                        validate_output_target(args.output_dir)
                        recording = RecordingSession(
                            output_dir=args.output_dir,
                            depth_scale_m=depth_scale_m,
                            intrinsics=reference_intrinsics,
                            d435_device=expected_device_info,
                            gf_a_path=args.gf_a,
                            gf_b_path=args.gf_b,
                            fps=args.fps,
                            max_tactile_age_ms=args.max_tactile_age_ms,
                            max_d435_frame_gap=args.max_d435_frame_gap,
                            max_pending_writes=args.max_pending_writes,
                            writer_threads=args.writer_threads,
                        )
                        print("Recording started.", flush=True)
                elif control == "quit":
                    finalize_recording()
                    should_quit = True
            if should_quit:
                break
    except KeyboardInterrupt:
        finalize_recording()
    except BaseException:
        if recording is not None:
            recording.abort()
            recording = None
        raise
    finally:
        if browser is not None:
            browser.close()
        if tactile_b is not None:
            tactile_b.close()
        if tactile_a is not None:
            tactile_a.close()
        if profile is not None:
            pipeline.stop()
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except (RuntimeError, ValueError, OSError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
