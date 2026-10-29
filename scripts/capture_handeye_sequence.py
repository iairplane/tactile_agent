#!/usr/bin/env python3
"""Record RGB video and capture quality-checked hand-eye samples at manual pauses.

This tool never sends a robot or gripper command.  It keeps the RealSense stream
open, uses board-pose stability only to decide when the operator has paused, and
then invokes the repository-reviewed read-only libfranka state reader around one
RGB-D exposure.  A new pause is considered only after a clearly different pose
has been observed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ECHO_ROBOT_STATE = (
    REPOSITORY_ROOT / "third_party/franka_ros2_ws/build_ros/libfranka/examples/echo_robot_state"
)


@dataclass(frozen=True)
class TimedPose:
    host_monotonic_ns: int
    transform: np.ndarray


@dataclass(frozen=True)
class PoseDifference:
    translation_m: float
    rotation_deg: float


@dataclass(frozen=True)
class NoveltyResult:
    novel: bool
    nearest_index: int | None
    nearest_difference: PoseDifference | None


def rotation_difference_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return the geodesic angle between two 3x3 rotations."""
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != (3, 3) or second_array.shape != (3, 3):
        raise ValueError("rotation inputs must both have shape (3, 3)")
    relative = first_array.T @ second_array
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def transform_difference(first: np.ndarray, second: np.ndarray) -> PoseDifference:
    """Measure translation and rotation separation between two rigid poses."""
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != (4, 4) or second_array.shape != (4, 4):
        raise ValueError("transform inputs must both have shape (4, 4)")
    return PoseDifference(
        translation_m=float(np.linalg.norm(first_array[:3, 3] - second_array[:3, 3])),
        rotation_deg=rotation_difference_deg(first_array[:3, :3], second_array[:3, :3]),
    )


def pose_window_spread(observations: list[TimedPose] | deque[TimedPose]) -> PoseDifference:
    """Return maximum pairwise pose separation in a short observation window."""
    if not observations:
        raise ValueError("pose window must not be empty")
    maximum_translation = 0.0
    maximum_rotation = 0.0
    for first_index, first in enumerate(observations):
        for second in list(observations)[first_index + 1 :]:
            difference = transform_difference(first.transform, second.transform)
            maximum_translation = max(maximum_translation, difference.translation_m)
            maximum_rotation = max(maximum_rotation, difference.rotation_deg)
    return PoseDifference(maximum_translation, maximum_rotation)


def pose_window_is_stationary(
    observations: list[TimedPose] | deque[TimedPose],
    *,
    minimum_duration_s: float,
    minimum_observations: int,
    maximum_gap_s: float,
    maximum_translation_m: float,
    maximum_rotation_deg: float,
) -> tuple[bool, PoseDifference | None]:
    """Check duration, continuity, and geometric spread of a visual pose window."""
    if len(observations) < minimum_observations:
        return False, None
    times = [observation.host_monotonic_ns for observation in observations]
    duration_s = (times[-1] - times[0]) * 1e-9
    if duration_s < minimum_duration_s:
        return False, None
    if any((later - earlier) * 1e-9 > maximum_gap_s for earlier, later in zip(times, times[1:])):
        return False, None
    spread = pose_window_spread(observations)
    return (
        spread.translation_m <= maximum_translation_m
        and spread.rotation_deg <= maximum_rotation_deg
    ), spread


def moved_enough(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    minimum_translation_m: float,
    minimum_rotation_deg: float,
) -> tuple[bool, PoseDifference]:
    """Use an OR rule so either a useful translation or rotation rearms capture."""
    difference = transform_difference(reference, current)
    return (
        difference.translation_m >= minimum_translation_m
        or difference.rotation_deg >= minimum_rotation_deg
    ), difference


def pose_novelty(
    candidate: np.ndarray,
    references: list[np.ndarray],
    *,
    minimum_translation_m: float,
    minimum_rotation_deg: float,
) -> NoveltyResult:
    """Reject a candidate if any reference is close in both translation and rotation."""
    if not references:
        return NoveltyResult(True, None, None)
    differences = [transform_difference(reference, candidate) for reference in references]
    duplicate_indices = [
        index
        for index, difference in enumerate(differences)
        if difference.translation_m < minimum_translation_m
        and difference.rotation_deg < minimum_rotation_deg
    ]
    normalized = [
        math.hypot(
            difference.translation_m / minimum_translation_m,
            difference.rotation_deg / minimum_rotation_deg,
        )
        for difference in differences
    ]
    nearest_index = (
        min(duplicate_indices, key=normalized.__getitem__)
        if duplicate_indices
        else int(np.argmin(normalized))
    )
    nearest = differences[nearest_index]
    return NoveltyResult(not duplicate_indices, nearest_index, nearest)


def trim_pose_window(history: deque[TimedPose], now_ns: int, duration_s: float) -> None:
    """Keep one observation just before the window boundary for duration coverage."""
    cutoff_ns = now_ns - int(duration_s * 1e9)
    while len(history) >= 2 and history[1].host_monotonic_ns <= cutoff_ns:
        history.popleft()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--serial", required=True, help="Exact RealSense serial number")
    parser.add_argument("--board-metadata", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New sequence directory; formal candidates are written below candidates/",
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--robot-hostname", default="172.16.0.2")
    parser.add_argument("--echo-robot-state", type=Path, default=DEFAULT_ECHO_ROBOT_STATE)
    parser.add_argument("--target-samples", type=int, default=24)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument(
        "--reference-samples-dir",
        type=Path,
        help=(
            "Existing formal samples used for duplicate rejection; the live pose at recorder "
            "startup is always latched separately before capture can arm"
        ),
    )

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--camera-timeout-ms", type=int, default=5000)
    parser.add_argument("--max-pair-skew-ms", type=float, default=20.0)
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--detection-stride", type=int, default=3)

    parser.add_argument("--stationary-window-s", type=float, default=0.75)
    parser.add_argument("--min-stationary-detections", type=int, default=5)
    parser.add_argument("--max-detection-gap-s", type=float, default=0.5)
    parser.add_argument("--max-visual-translation-m", type=float, default=0.002)
    parser.add_argument("--max-visual-rotation-deg", type=float, default=0.5)
    parser.add_argument("--rearm-translation-m", type=float, default=0.015)
    parser.add_argument("--rearm-rotation-deg", type=float, default=6.0)
    parser.add_argument("--rearm-confirmations", type=int, default=2)
    parser.add_argument("--duplicate-translation-m", type=float, default=0.015)
    parser.add_argument("--duplicate-rotation-deg", type=float, default=6.0)
    parser.add_argument("--retry-cooldown-s", type=float, default=0.75)

    parser.add_argument("--min-charuco-corners", type=int, default=12)
    parser.add_argument("--min-charuco-x-span-fraction", type=float, default=0.5)
    parser.add_argument("--min-charuco-y-span-fraction", type=float, default=0.5)
    parser.add_argument("--min-charuco-id-span-fraction", type=float, default=0.5)
    parser.add_argument("--max-reprojection-rms-px", type=float, default=1.0)
    parser.add_argument("--max-reprojection-point-px", type=float, default=2.5)
    parser.add_argument("--min-checkerboard-area-fraction", type=float, default=0.03)
    parser.add_argument("--min-planar-pose-rms-gap-px", type=float, default=0.05)

    parser.add_argument("--robot-timeout-s", type=float, default=15.0)
    parser.add_argument("--min-robot-states", type=int, default=90)
    parser.add_argument("--min-robot-states-before-exposure", type=int, default=10)
    parser.add_argument("--min-robot-states-after-exposure", type=int, default=10)
    parser.add_argument("--max-abs-dq-rad-s", type=float, default=0.01)
    parser.add_argument("--max-pose-translation-m", type=float, default=0.0005)
    parser.add_argument("--max-pose-rotation-deg", type=float, default=0.10)
    return parser


def _capture_module() -> ModuleType:
    try:
        import capture_handeye_sample as capture
    except ModuleNotFoundError:
        from scripts import capture_handeye_sample as capture
    return capture


def validate_args(args: argparse.Namespace, capture: ModuleType) -> None:
    # These fixed values make the imported single-sample validator and writer take
    # only the reviewed live read-only path.
    args.dry_run_camera_only = False
    args.t_base_flange = None
    args.allow_untrusted_robot_state_reader = False
    capture.validate_args(args)

    positive_integer_names = (
        "target_samples",
        "detection_stride",
        "min_stationary_detections",
        "rearm_confirmations",
    )
    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if len(args.video_codec) != 4 or not args.video_codec.isascii():
        raise ValueError("--video-codec must contain exactly four ASCII characters")
    positive_float_names = (
        "stationary_window_s",
        "max_detection_gap_s",
        "max_visual_translation_m",
        "max_visual_rotation_deg",
        "rearm_translation_m",
        "rearm_rotation_deg",
        "duplicate_translation_m",
        "duplicate_rotation_deg",
        "retry_cooldown_s",
    )
    for name in positive_float_names:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.reference_samples_dir is not None:
        args.reference_samples_dir = args.reference_samples_dir.expanduser().absolute()
        if not args.reference_samples_dir.is_dir():
            raise FileNotFoundError(
                f"reference samples directory not found: {args.reference_samples_dir}"
            )


def _natural_key(path: Path) -> list[tuple[int, int | str]]:
    import re

    parts = re.split(r"(\d+)", path.name.casefold())
    return [(0, int(part)) if part.isdigit() else (1, part) for part in parts]


def load_reference_poses(
    samples_dir: Path | None, capture: ModuleType
) -> tuple[list[np.ndarray], np.ndarray | None, list[str]]:
    if samples_dir is None:
        return [], None, []
    base_poses: list[np.ndarray] = []
    camera_poses: list[np.ndarray] = []
    names: list[str] = []
    directories = sorted(
        (entry for entry in samples_dir.iterdir() if entry.is_dir()), key=_natural_key
    )
    for directory in directories:
        base_path = directory / "T_base_flange.npy"
        camera_path = directory / "T_camera_board.npy"
        if not base_path.is_file() or not camera_path.is_file():
            continue
        base_poses.append(
            capture.validate_rigid_transform(
                np.load(base_path, allow_pickle=False), f"reference {base_path}"
            ).copy()
        )
        camera_poses.append(
            capture.validate_rigid_transform(
                np.load(camera_path, allow_pickle=False), f"reference {camera_path}"
            ).copy()
        )
        names.append(directory.name)
    startup_camera_pose = camera_poses[-1] if camera_poses else None
    return base_poses, startup_camera_pose, names


def estimate_board_pose(
    capture: ModuleType,
    args: argparse.Namespace,
    rgb: np.ndarray,
    intrinsics: Any,
    board: Any,
    board_data: dict[str, Any],
) -> Any:
    if board_data.get("schema") == capture.CHECKERBOARD_SCHEMA:
        return capture.estimate_checkerboard_pose(
            rgb,
            intrinsics,
            board_data,
            args.min_checkerboard_area_fraction,
            args.min_planar_pose_rms_gap_px,
            args.max_reprojection_rms_px,
            args.max_reprojection_point_px,
        )
    return capture.estimate_board_pose(
        rgb,
        intrinsics,
        board,
        board_data,
        args.min_charuco_corners,
        args.min_charuco_x_span_fraction,
        args.min_charuco_y_span_fraction,
        args.min_charuco_id_span_fraction,
        args.max_reprojection_rms_px,
        args.max_reprojection_point_px,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def append_jsonl(stream: TextIO, record: dict[str, Any], *, sync: bool = False) -> None:
    stream.write(json.dumps(record, default=_json_default, separators=(",", ":")) + "\n")
    stream.flush()
    if sync:
        os.fsync(stream.fileno())


def event_record(event: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema": "demo2gripper.handeye_sequence_event.v1",
        "event": event,
        "host_utc": datetime.now(UTC).isoformat(),
        "host_monotonic_ns": time.monotonic_ns(),
        **fields,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class SequenceRecording:
    def __init__(
        self,
        output_dir: Path,
        *,
        width: int,
        height: int,
        fps: int,
        codec: str,
    ) -> None:
        self.video_path = output_dir / "rgb.mp4"
        self.frames_path = output_dir / "frames.jsonl"
        self.events_path = output_dir / "events.jsonl"
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(self.video_path), fourcc, float(fps), (width, height))
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(
                f"could not open compressed video writer {self.video_path} with codec {codec!r}"
            )
        self.frames_stream = self.frames_path.open("x", encoding="utf-8", buffering=1)
        self.events_stream = self.events_path.open("x", encoding="utf-8", buffering=1)
        self.expected_shape = (height, width, 3)
        self.frame_count = 0

    def write_frame(
        self,
        rgb: np.ndarray,
        stamp: Any,
        *,
        host_unix_ns: int,
        host_monotonic_ns: int,
        role: str,
    ) -> int:
        if rgb.shape != self.expected_shape or rgb.dtype != np.uint8:
            raise RuntimeError(
                f"unexpected RGB shape/type {rgb.shape}/{rgb.dtype}; "
                f"expected {self.expected_shape}/uint8"
            )
        video_index = self.frame_count
        self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        append_jsonl(
            self.frames_stream,
            {
                "schema": "demo2gripper.handeye_sequence_frame.v1",
                "video_frame_index": video_index,
                "role": role,
                "host_unix_ns_at_frameset_receipt": host_unix_ns,
                "host_monotonic_ns_at_frameset_receipt": host_monotonic_ns,
                "color_sensor_s": stamp.color_sensor_timestamp_s,
                "depth_sensor_s": stamp.depth_sensor_timestamp_s,
                "sensor_pair_skew_ms": stamp.sensor_skew_ms,
                "color_sdk_s": stamp.color_sdk_timestamp_s,
                "depth_sdk_s": stamp.depth_sdk_timestamp_s,
                "sdk_domain": stamp.sdk_timestamp_domain,
                "color_frame_number": stamp.color_frame_number,
                "depth_frame_number": stamp.depth_frame_number,
            },
        )
        self.frame_count += 1
        return video_index

    def event(self, name: str, *, sync: bool = False, **fields: Any) -> None:
        append_jsonl(self.events_stream, event_record(name, **fields), sync=sync)

    def close(self) -> None:
        self.writer.release()
        for stream in (self.frames_stream, self.events_stream):
            if not stream.closed:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()


def receive_pair(
    capture: ModuleType,
    args: argparse.Namespace,
    pipeline: Any,
    align_to_color: Any,
    previous: Any,
    recording: SequenceRecording,
    *,
    role: str,
) -> tuple[Any, Any, Any, np.ndarray, int, int, int]:
    color_frame, depth_frame, stamp = capture.wait_for_aligned_pair(
        pipeline,
        align_to_color,
        previous,
        args.camera_timeout_ms,
        args.max_pair_skew_ms,
    )
    host_unix_ns = time.time_ns()
    host_monotonic_ns = time.monotonic_ns()
    rgb = np.asanyarray(color_frame.get_data()).copy()
    video_index = recording.write_frame(
        rgb,
        stamp,
        host_unix_ns=host_unix_ns,
        host_monotonic_ns=host_monotonic_ns,
        role=role,
    )
    return (
        color_frame,
        depth_frame,
        stamp,
        rgb,
        host_unix_ns,
        host_monotonic_ns,
        video_index,
    )


def capture_candidate(
    capture: ModuleType,
    args: argparse.Namespace,
    pipeline: Any,
    align_to_color: Any,
    previous: Any,
    recording: SequenceRecording,
    *,
    depth_scale_m: float,
    device: dict[str, str],
    board: Any,
    board_data: dict[str, Any],
) -> tuple[Any, Any, np.ndarray, dict[str, Any], Any, int]:
    robot_session = capture.start_robot_reader(args)
    try:
        if robot_session is None:
            raise RuntimeError("internal error: reviewed robot reader was not started")
        capture.wait_for_robot_states(robot_session, args.min_robot_states_before_exposure)

        flush_pair = receive_pair(
            capture,
            args,
            pipeline,
            align_to_color,
            previous,
            recording,
            role="pre_exposure_flush",
        )
        previous = flush_pair[2]
        (
            color_frame,
            depth_frame,
            stamp,
            rgb,
            host_unix_ns,
            host_monotonic_ns,
            video_index,
        ) = receive_pair(
            capture,
            args,
            pipeline,
            align_to_color,
            previous,
            recording,
            role="candidate_exposure",
        )
        previous = stamp
        depth_raw = np.asanyarray(depth_frame.get_data()).copy()
        expected_depth_shape = (args.height, args.width)
        if depth_raw.shape != expected_depth_shape or depth_raw.dtype != np.uint16:
            raise RuntimeError(
                f"unexpected depth shape/type {depth_raw.shape}/{depth_raw.dtype}; "
                f"expected {expected_depth_shape}/uint16"
            )
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        camera = capture.CameraSample(
            rgb=rgb,
            depth_m=depth_raw.astype(np.float32) * depth_scale_m,
            intrinsics=intrinsics,
            depth_scale_m=depth_scale_m,
            stamp=stamp,
            host_timestamp_ns=host_unix_ns,
            host_monotonic_ns=host_monotonic_ns,
            device=device,
        )

        robot_output = capture.finish_robot_reader(robot_session)
        robot_session = None
        pose = estimate_board_pose(capture, args, rgb, intrinsics, board, board_data)
        base_flange, robot_summary = capture.resolve_robot_pose(
            args, robot_output, host_monotonic_ns
        )
        if base_flange is None:
            raise RuntimeError("internal error: live sample has no T_base_flange")
        return camera, pose, base_flange, robot_summary, previous, video_index
    finally:
        capture.stop_failed_robot_reader(robot_session)


def _args_for_report(args: argparse.Namespace) -> dict[str, Any]:
    omitted = {"robot_reader_reviewed"}
    return {
        key: value
        for key, value in vars(args).items()
        if key not in omitted and not key.startswith("_")
    }


def run(args: argparse.Namespace, capture: ModuleType) -> int:
    validate_args(args, capture)
    board_metadata, board_data = capture.load_board_metadata(args.board_metadata)
    board = capture.make_board(board_data)
    base_references, _, reference_names = load_reference_poses(args.reference_samples_dir, capture)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir()
    recording: SequenceRecording | None = None
    pipeline = capture.rs.pipeline()
    config = capture.rs.config()
    config.enable_device(args.serial)
    config.enable_stream(
        capture.rs.stream.color, args.width, args.height, capture.rs.format.rgb8, args.fps
    )
    config.enable_stream(
        capture.rs.stream.depth, args.width, args.height, capture.rs.format.z16, args.fps
    )
    align_to_color = capture.rs.align(capture.rs.stream.color)
    profile = None
    started_utc = datetime.now(UTC).isoformat()
    started_monotonic_ns = time.monotonic_ns()
    status = "error"
    error_message: str | None = None
    accepted_paths: list[str] = []
    attempted_captures = 0
    duplicate_rejections = 0
    quality_rejections = 0

    try:
        requested = capture.find_device(args.serial)
        requested_info = capture.device_info(requested)
        profile = pipeline.start(config)
        active_device = profile.get_device()
        device = capture.device_info(active_device)
        if device.get("serial") != args.serial:
            raise RuntimeError(
                f"opened RealSense {device.get('serial')!r}, expected {args.serial!r}"
            )
        if device != requested_info:
            requested_info = device
        depth_scale_m = float(active_device.first_depth_sensor().get_depth_scale())
        if not math.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
            raise RuntimeError(f"invalid RealSense depth scale {depth_scale_m}")

        previous = None
        for _ in range(args.warmup_frames):
            _, _, previous = capture.wait_for_aligned_pair(
                pipeline,
                align_to_color,
                previous,
                args.camera_timeout_ms,
                args.max_pair_skew_ms,
            )

        recording = SequenceRecording(
            output_dir,
            width=args.width,
            height=args.height,
            fps=args.fps,
            codec=args.video_codec,
        )
        recording.event(
            "sequence_started",
            sync=True,
            session_id=args.session_id,
            camera_serial=args.serial,
            target_samples=args.target_samples,
            initially_armed=False,
            reference_samples=reference_names,
            safety={
                "robot_access": "repository-reviewed read-only libfranka Robot.read",
                "control_commands_sent": False,
                "gripper_commands_sent": False,
            },
        )
        print(
            "Recording started. Capture is initially disarmed; move clearly away from the "
            "current pose, then hold each useful pose still for about 2 seconds.",
            flush=True,
        )

        armed = False
        # Always latch the actual live startup pose.  A previous formal reference may not match
        # where the operator happens to be when a later recording session starts.
        rearm_reference = None
        rearm_count = 0
        history: deque[TimedPose] = deque()
        monitor_frame_count = 0
        last_attempt_monotonic_ns = 0
        last_detection_valid = False

        while len(accepted_paths) < args.target_samples:
            (
                color_frame,
                _depth_frame,
                stamp,
                rgb,
                _host_unix_ns,
                host_monotonic_ns,
                _video_index,
            ) = receive_pair(
                capture,
                args,
                pipeline,
                align_to_color,
                previous,
                recording,
                role="monitor",
            )
            previous = stamp
            monitor_frame_count += 1
            if monitor_frame_count % args.detection_stride:
                continue

            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            try:
                observed_pose = estimate_board_pose(
                    capture, args, rgb, intrinsics, board, board_data
                )
            except (RuntimeError, ValueError) as exc:
                history.clear()
                rearm_count = 0
                if last_detection_valid:
                    recording.event("board_detection_lost", reason=str(exc)[:500])
                last_detection_valid = False
                continue
            last_detection_valid = True
            observation = TimedPose(host_monotonic_ns, observed_pose.transform.copy())

            if not armed:
                history.clear()
                if rearm_reference is None:
                    rearm_reference = observation.transform.copy()
                    recording.event("startup_pose_latched")
                    print(
                        "Startup pose latched. Move to a different pose to arm capture.",
                        flush=True,
                    )
                    continue
                moved, difference = moved_enough(
                    rearm_reference,
                    observation.transform,
                    minimum_translation_m=args.rearm_translation_m,
                    minimum_rotation_deg=args.rearm_rotation_deg,
                )
                rearm_count = rearm_count + 1 if moved else 0
                if rearm_count >= args.rearm_confirmations:
                    armed = True
                    rearm_count = 0
                    history.clear()
                    recording.event(
                        "capture_armed_after_motion",
                        translation_m=difference.translation_m,
                        rotation_deg=difference.rotation_deg,
                    )
                    print(
                        f"Motion detected ({difference.translation_m * 1e3:.1f} mm, "
                        f"{difference.rotation_deg:.1f} deg); waiting for a stable pause.",
                        flush=True,
                    )
                continue

            history.append(observation)
            trim_pose_window(history, host_monotonic_ns, args.stationary_window_s)
            stationary, visual_spread = pose_window_is_stationary(
                history,
                minimum_duration_s=args.stationary_window_s,
                minimum_observations=args.min_stationary_detections,
                maximum_gap_s=args.max_detection_gap_s,
                maximum_translation_m=args.max_visual_translation_m,
                maximum_rotation_deg=args.max_visual_rotation_deg,
            )
            cooldown_elapsed_s = (host_monotonic_ns - last_attempt_monotonic_ns) * 1e-9
            if not stationary or cooldown_elapsed_s < args.retry_cooldown_s:
                continue

            attempted_captures += 1
            last_attempt_monotonic_ns = time.monotonic_ns()
            history.clear()
            assert visual_spread is not None
            recording.event(
                "stationary_pause_detected",
                attempt=attempted_captures,
                visual_translation_spread_m=visual_spread.translation_m,
                visual_rotation_spread_deg=visual_spread.rotation_deg,
            )
            print(f"Stable pause detected; checking candidate {attempted_captures}...", flush=True)
            try:
                (
                    camera,
                    pose,
                    base_flange,
                    robot_summary,
                    previous,
                    video_index,
                ) = capture_candidate(
                    capture,
                    args,
                    pipeline,
                    align_to_color,
                    previous,
                    recording,
                    depth_scale_m=depth_scale_m,
                    device=requested_info,
                    board=board,
                    board_data=board_data,
                )
            except (RuntimeError, ValueError) as exc:
                quality_rejections += 1
                recording.event(
                    "candidate_rejected_quality",
                    sync=True,
                    attempt=attempted_captures,
                    reason=str(exc)[:1000],
                )
                print(f"Candidate rejected: {exc}", flush=True)
                continue

            novelty = pose_novelty(
                base_flange,
                base_references,
                minimum_translation_m=args.duplicate_translation_m,
                minimum_rotation_deg=args.duplicate_rotation_deg,
            )
            rearm_reference = pose.transform.copy()
            armed = False
            rearm_count = 0
            if not novelty.novel:
                duplicate_rejections += 1
                nearest = novelty.nearest_difference
                assert nearest is not None
                recording.event(
                    "candidate_rejected_duplicate",
                    sync=True,
                    attempt=attempted_captures,
                    nearest_reference_index=novelty.nearest_index,
                    nearest_translation_m=nearest.translation_m,
                    nearest_rotation_deg=nearest.rotation_deg,
                    video_frame_index=video_index,
                )
                print(
                    f"Candidate is too close to an earlier pose ({nearest.translation_m * 1e3:.1f} "
                    f"mm, {nearest.rotation_deg:.1f} deg). Move again before the next pause.",
                    flush=True,
                )
                continue

            sample_index = args.start_index + len(accepted_paths)
            sample_name = f"{sample_index:03d}"
            sample_path = candidates_dir / sample_name
            sample_args = argparse.Namespace(**vars(args))
            sample_args.output_dir = sample_path
            capture.write_sample(
                sample_args,
                camera,
                pose,
                board_metadata,
                board_data,
                base_flange,
                robot_summary,
            )
            base_references.append(base_flange.copy())
            accepted_paths.append(str(sample_path.relative_to(output_dir)))
            recording.event(
                "candidate_accepted",
                sync=True,
                attempt=attempted_captures,
                accepted_count=len(accepted_paths),
                target_samples=args.target_samples,
                sample=accepted_paths[-1],
                video_frame_index=video_index,
                reprojection_rms_px=pose.reprojection_rms_px,
                reprojection_max_px=pose.reprojection_max_px,
                checkerboard_area_fraction=pose.area_fraction,
            )
            print(
                f"Accepted {len(accepted_paths)}/{args.target_samples}: {sample_path}. "
                "Move clearly before holding the next pose.",
                flush=True,
            )

        status = "complete"
        recording.event("target_reached", sync=True, accepted_count=len(accepted_paths))
        return_code = 0
    except KeyboardInterrupt:
        status = "interrupted"
        return_code = 130
        if recording is not None:
            recording.event("operator_interrupted", sync=True, accepted_count=len(accepted_paths))
        print("Capture interrupted; closing video and preserving completed candidates.", flush=True)
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        if recording is not None:
            recording.event("sequence_failed", sync=True, reason=error_message[:1000])
        raise
    finally:
        if profile is not None:
            pipeline.stop()
        if recording is not None:
            recording.close()
        summary = {
            "schema": "demo2gripper.handeye_sequence.v1",
            "created_utc": started_utc,
            "finished_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "error": error_message,
            "session_id": args.session_id,
            "camera_serial": args.serial,
            "elapsed_s": (time.monotonic_ns() - started_monotonic_ns) * 1e-9,
            "target_samples": args.target_samples,
            "accepted_count": len(accepted_paths),
            "attempted_captures": attempted_captures,
            "duplicate_rejections": duplicate_rejections,
            "quality_rejections": quality_rejections,
            "candidates": accepted_paths,
            "files": {
                "compressed_rgb_video": "rgb.mp4" if recording is not None else None,
                "frame_timestamps_jsonl": "frames.jsonl" if recording is not None else None,
                "events_jsonl": "events.jsonl" if recording is not None else None,
                "candidates_directory": "candidates",
            },
            "initially_armed": False,
            "safety": {
                "robot_access": "repository-reviewed read-only libfranka Robot.read",
                "control_commands_sent": False,
                "gripper_commands_sent": False,
            },
            "arguments": _args_for_report(args),
        }
        atomic_write_json(output_dir / "sequence_manifest.json", summary)
    return return_code


def main() -> int:
    args = build_parser().parse_args()
    capture = _capture_module()
    return run(args, capture)


if __name__ == "__main__":
    raise SystemExit(main())
