#!/usr/bin/env python3
"""Capture one quality-checked eye-to-hand calibration sample without robot motion."""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs

try:
    from checkerboard_calibration import (
        CHECKERBOARD_SCHEMA,
        detect_checkerboard,
        normalize_checkerboard_metadata,
        solve_checkerboard_pose,
    )
except ModuleNotFoundError:  # Imported as scripts.capture_handeye_sample in CPU tests.
    from scripts.checkerboard_calibration import (
        CHECKERBOARD_SCHEMA,
        detect_checkerboard,
        normalize_checkerboard_metadata,
        solve_checkerboard_pose,
    )

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ECHO_ROBOT_STATE = (
    REPOSITORY_ROOT / "third_party/franka_ros2_ws/build_ros/libfranka/examples/echo_robot_state"
)
DEFAULT_ROS_STATE_READER = REPOSITORY_ROOT / "scripts/ros_franka_state_reader.py"
DEFAULT_ROS_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"
DEFAULT_ROS_PYTHON = Path("/usr/bin/python3")
AT_FDCWD = -100
RENAME_NOREPLACE = 1
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


@dataclass(frozen=True)
class FrameStamp:
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


@dataclass(frozen=True)
class CameraSample:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: Any
    depth_scale_m: float
    stamp: FrameStamp
    host_timestamp_ns: int
    host_monotonic_ns: int
    device: dict[str, str]


@dataclass(frozen=True)
class BoardPose:
    transform: np.ndarray
    overlay_bgr: np.ndarray
    board_kind: str
    corner_count: int
    marker_count: int
    reprojection_rms_px: float
    reprojection_max_px: float
    detected_ids: list[int] | None
    span_x_fraction: float
    span_y_fraction: float
    id_span_fraction: float | None
    area_fraction: float | None
    planar_pose_second_rms_px: float | None
    planar_pose_rms_gap_px: float | None
    planar_pose_positive_candidates: int | None
    planar_pose_second_rotation_difference_deg: float | None
    planar_pose_second_translation_difference_m: float | None


@dataclass(frozen=True)
class TimedRobotLine:
    host_monotonic_ns: int
    text: str


@dataclass(frozen=True)
class RobotReaderOutput:
    stdout_lines: tuple[TimedRobotLine, ...]
    stderr: str
    return_code: int
    started_host_monotonic_ns: int
    finished_host_monotonic_ns: int

    @property
    def stdout(self) -> str:
        return "".join(line.text for line in self.stdout_lines)


@dataclass
class RobotReaderSession:
    process: subprocess.Popen[str]
    condition: threading.Condition
    stdout_lines: list[TimedRobotLine]
    stderr_parts: list[str]
    started_host_monotonic_ns: int
    deadline_monotonic_s: float
    threads: tuple[threading.Thread, threading.Thread] = ()
    stdout_done: bool = False
    stderr_done: bool = False
    stdout_error: str | None = None
    stderr_error: str | None = None


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
        help="New directory for this single calibration sample",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--camera-timeout-ms", type=int, default=5000)
    parser.add_argument("--max-pair-skew-ms", type=float, default=20.0)
    parser.add_argument("--min-charuco-corners", type=int, default=12)
    parser.add_argument(
        "--min-charuco-x-span-fraction",
        type=float,
        default=0.5,
        help="Minimum fraction of the board's full ChArUco-corner X span",
    )
    parser.add_argument(
        "--min-charuco-y-span-fraction",
        type=float,
        default=0.5,
        help="Minimum fraction of the board's full ChArUco-corner Y span",
    )
    parser.add_argument(
        "--min-charuco-id-span-fraction",
        type=float,
        default=0.5,
        help="Minimum (max ID - min ID) fraction across all ChArUco corner IDs",
    )
    parser.add_argument("--max-reprojection-rms-px", type=float, default=1.0)
    parser.add_argument("--max-reprojection-point-px", type=float, default=2.5)
    parser.add_argument(
        "--min-checkerboard-area-fraction",
        type=float,
        default=0.03,
        help="Minimum image fraction covered by the complete inner-corner convex hull",
    )
    parser.add_argument(
        "--min-planar-pose-rms-gap-px",
        type=float,
        default=0.05,
        help=(
            "Minimum IPPE best-versus-second pose reprojection RMS gap; rejects "
            "near-frontal checkerboard poses whose planar orientation is ambiguous"
        ),
    )
    robot_source = parser.add_mutually_exclusive_group()
    robot_source.add_argument(
        "--dry-run-camera-only",
        action="store_true",
        help="Capture and validate only the camera/board side; omit T_base_flange.npy",
    )
    robot_source.add_argument(
        "--t-base-flange",
        type=Path,
        help="Use a reviewed 4x4 .npy transform instead of connecting to the robot",
    )
    robot_source.add_argument(
        "--ros-robot-state-topic",
        nargs="?",
        const=DEFAULT_ROS_STATE_TOPIC,
        help=(
            "Read the existing franka_robot_state_broadcaster topic instead of opening a "
            "second libfranka FCI connection; optionally provide an absolute topic name"
        ),
    )
    parser.add_argument("--robot-hostname", default="172.16.0.2")
    parser.add_argument("--echo-robot-state", type=Path, default=DEFAULT_ECHO_ROBOT_STATE)
    parser.add_argument("--ros-state-reader", type=Path, default=DEFAULT_ROS_STATE_READER)
    parser.add_argument("--ros-python", type=Path, default=DEFAULT_ROS_PYTHON)
    parser.add_argument(
        "--allow-untrusted-robot-state-reader",
        action="store_true",
        help=(
            "Allow a non-default robot-state executable. Its control-command behavior is "
            "recorded as unknown; this option does not make it safe."
        ),
    )
    parser.add_argument(
        "--session-id",
        help=(
            "Dataset session identifier asserting that the camera mount and board-to-flange "
            "mount remain unchanged (required except for camera-only dry runs)"
        ),
    )
    parser.add_argument("--robot-timeout-s", type=float, default=15.0)
    parser.add_argument("--min-robot-states", type=int, default=90)
    parser.add_argument(
        "--min-robot-states-before-exposure",
        type=int,
        default=10,
        help="Minimum timestamped robot states received before the RGB-D frame",
    )
    parser.add_argument(
        "--min-robot-states-after-exposure",
        type=int,
        default=10,
        help="Minimum timestamped robot states received after the RGB-D frame",
    )
    parser.add_argument("--max-abs-dq-rad-s", type=float, default=0.01)
    parser.add_argument("--max-pose-translation-m", type=float, default=0.0005)
    parser.add_argument("--max-pose-rotation-deg", type=float, default=0.10)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("width", "height", "fps", "camera_timeout_ms", "min_charuco_corners"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative")
    for name in (
        "max_pair_skew_ms",
        "max_reprojection_rms_px",
        "max_reprojection_point_px",
        "robot_timeout_s",
        "max_abs_dq_rad_s",
        "max_pose_translation_m",
        "max_pose_rotation_deg",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.min_planar_pose_rms_gap_px) or args.min_planar_pose_rms_gap_px < 0.0:
        raise ValueError("--min-planar-pose-rms-gap-px must be finite and non-negative")
    for name in (
        "min_charuco_x_span_fraction",
        "min_charuco_y_span_fraction",
        "min_charuco_id_span_fraction",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    if (
        not math.isfinite(args.min_checkerboard_area_fraction)
        or not 0.0 < args.min_checkerboard_area_fraction <= 1.0
    ):
        raise ValueError("--min-checkerboard-area-fraction must be greater than 0 and at most 1")
    if args.min_robot_states < 2:
        raise ValueError("--min-robot-states must be at least 2")
    for name in ("min_robot_states_before_exposure", "min_robot_states_after_exposure"):
        if getattr(args, name) < 2:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 2")

    if args.session_id is not None:
        args.session_id = args.session_id.strip()
        if not SESSION_ID_PATTERN.fullmatch(args.session_id):
            raise ValueError(
                "--session-id must be 1-64 characters: ASCII letters/digits followed by "
                "letters, digits, '.', '_' or '-'"
            )
    if not args.dry_run_camera_only and args.session_id is None:
        raise ValueError("--session-id is required for a calibration sample")

    args.output_dir = args.output_dir.expanduser().absolute()
    args.board_metadata = args.board_metadata.expanduser().absolute()
    args.echo_robot_state = args.echo_robot_state.expanduser().absolute()
    args.ros_state_reader = args.ros_state_reader.expanduser().absolute()
    args.ros_python = args.ros_python.expanduser().absolute()
    args.robot_reader_reviewed = None
    args.robot_reader_kind = None
    if args.output_dir.exists():
        raise FileExistsError(f"sample directory already exists: {args.output_dir}")
    if not args.board_metadata.is_file():
        raise FileNotFoundError(f"board metadata not found: {args.board_metadata}")
    if args.t_base_flange is not None:
        args.t_base_flange = args.t_base_flange.expanduser().absolute()
        if not args.t_base_flange.is_file():
            raise FileNotFoundError(f"transform not found: {args.t_base_flange}")
    elif args.ros_robot_state_topic is not None:
        if not args.ros_robot_state_topic.startswith("/"):
            raise ValueError("--ros-robot-state-topic must be an absolute ROS topic")
        if not args.ros_python.is_file() or not os.access(args.ros_python, os.X_OK):
            raise FileNotFoundError(f"executable ROS system Python not found: {args.ros_python}")
        if not args.ros_state_reader.is_file():
            raise FileNotFoundError(f"ROS state reader not found: {args.ros_state_reader}")
        default_reader = DEFAULT_ROS_STATE_READER.expanduser().absolute()
        args.robot_reader_reviewed = default_reader.is_file() and os.path.samefile(
            args.ros_state_reader, default_reader
        )
        args.robot_reader_kind = "ros_topic"
        if not args.robot_reader_reviewed and not args.allow_untrusted_robot_state_reader:
            raise ValueError(
                "--ros-state-reader differs from the repository-reviewed default; pass "
                "--allow-untrusted-robot-state-reader to proceed with command behavior "
                "recorded as unknown"
            )
    elif not args.dry_run_camera_only:
        if not args.echo_robot_state.is_file():
            raise FileNotFoundError(
                f"read-only robot-state executable not found: {args.echo_robot_state}"
            )
        if not os.access(args.echo_robot_state, os.X_OK):
            raise PermissionError(
                f"read-only robot-state executable is not executable: {args.echo_robot_state}"
            )
        default_reader = DEFAULT_ECHO_ROBOT_STATE.expanduser().absolute()
        args.robot_reader_reviewed = default_reader.is_file() and os.path.samefile(
            args.echo_robot_state, default_reader
        )
        args.robot_reader_kind = "libfranka"
        if not args.robot_reader_reviewed and not args.allow_untrusted_robot_state_reader:
            raise ValueError(
                "--echo-robot-state differs from the repository-reviewed default; pass "
                "--allow-untrusted-robot-state-reader to proceed with command behavior "
                "recorded as unknown"
            )


def integer_field(value: Any, name: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"board metadata {name!r} must be an integer >= {minimum}")
    return value


def finite_positive_field(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"board metadata {name!r} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"board metadata {name!r} must be finite and positive")
    return result


def load_board_metadata(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in board metadata {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("board metadata root must be an object")

    if metadata.get("schema") == CHECKERBOARD_SCHEMA:
        return metadata, normalize_checkerboard_metadata(metadata, source=path)
    if metadata.get("type") == "opencv_charuco_board":
        if metadata.get("schema_version") != 1:
            raise ValueError("only opencv_charuco_board schema_version 1 is supported")
        board_data = metadata.get("board")
        if not isinstance(board_data, dict):
            raise ValueError("board metadata must contain a 'board' object")
    elif metadata.get("schema") == "demo2gripper.charuco_board.v1":
        nested = metadata.get("board")
        board_data = nested if isinstance(nested, dict) else metadata
    else:
        raise ValueError(
            "unsupported board metadata schema; expected opencv_charuco_board version 1 "
            f", demo2gripper.charuco_board.v1 or {CHECKERBOARD_SCHEMA}"
        )

    squares_x = integer_field(board_data.get("squares_x"), "squares_x", 2)
    squares_y = integer_field(board_data.get("squares_y"), "squares_y", 2)
    square_length_m = finite_positive_field(board_data.get("square_length_m"), "square_length_m")
    marker_length_m = finite_positive_field(board_data.get("marker_length_m"), "marker_length_m")
    if marker_length_m >= square_length_m:
        raise ValueError("board marker_length_m must be smaller than square_length_m")
    dictionary_name = board_data.get("dictionary")
    if not isinstance(dictionary_name, str) or not dictionary_name.startswith("DICT_"):
        raise ValueError("board metadata 'dictionary' must be an OpenCV DICT_* name")
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if not isinstance(dictionary_id, int):
        raise ValueError(f"OpenCV does not provide dictionary {dictionary_name!r}")
    legacy_pattern = board_data.get("legacy_pattern", False)
    if not isinstance(legacy_pattern, bool):
        raise ValueError("board metadata 'legacy_pattern' must be boolean")

    normalized = {
        "squares_x": squares_x,
        "squares_y": squares_y,
        "square_length_m": square_length_m,
        "marker_length_m": marker_length_m,
        "dictionary": dictionary_name,
        "dictionary_id": dictionary_id,
        "legacy_pattern": legacy_pattern,
    }
    return metadata, normalized


def make_board(board_data: dict[str, Any]) -> Any | None:
    if board_data.get("schema") == CHECKERBOARD_SCHEMA:
        return None
    dictionary = cv2.aruco.getPredefinedDictionary(board_data["dictionary_id"])
    board = cv2.aruco.CharucoBoard(
        (board_data["squares_x"], board_data["squares_y"]),
        board_data["square_length_m"],
        board_data["marker_length_m"],
        dictionary,
    )
    board.setLegacyPattern(board_data["legacy_pattern"])
    return board


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
    return {name: device.get_info(key) for name, key in fields.items() if device.supports(key)}


def find_device(serial: str) -> Any:
    devices = list(rs.context().query_devices())
    for device in devices:
        if device.get_info(rs.camera_info.serial_number) == serial:
            return device
    available = [device.get_info(rs.camera_info.serial_number) for device in devices]
    raise RuntimeError(f"RealSense {serial!r} not found; available serials: {available}")


def sensor_timestamp_s(frame: Any) -> float:
    metadata = rs.frame_metadata_value.sensor_timestamp
    if not frame.supports_frame_metadata(metadata):
        raise RuntimeError("RealSense frame has no hardware sensor timestamp metadata")
    return float(frame.get_frame_metadata(metadata)) * 1e-6


def frame_stamp(color_frame: Any, depth_frame: Any) -> FrameStamp:
    color_domain = str(color_frame.get_frame_timestamp_domain()).removeprefix("timestamp_domain.")
    depth_domain = str(depth_frame.get_frame_timestamp_domain()).removeprefix("timestamp_domain.")
    if color_domain != depth_domain:
        raise RuntimeError(
            f"color/depth timestamp domains differ: {color_domain} versus {depth_domain}"
        )
    return FrameStamp(
        color_frame_number=int(color_frame.get_frame_number()),
        depth_frame_number=int(depth_frame.get_frame_number()),
        color_sensor_timestamp_s=sensor_timestamp_s(color_frame),
        depth_sensor_timestamp_s=sensor_timestamp_s(depth_frame),
        color_sdk_timestamp_s=float(color_frame.get_timestamp()) * 1e-3,
        depth_sdk_timestamp_s=float(depth_frame.get_timestamp()) * 1e-3,
        sdk_timestamp_domain=color_domain,
    )


def wait_for_aligned_pair(
    pipeline: Any,
    align_to_color: Any,
    previous: FrameStamp | None,
    timeout_ms: int,
    max_pair_skew_ms: float,
) -> tuple[Any, Any, FrameStamp]:
    deadline = time.monotonic() + timeout_ms * 1e-3
    last_problem = "no frames received"
    while time.monotonic() < deadline:
        remaining_ms = max(1, int((deadline - time.monotonic()) * 1e3))
        frames = pipeline.wait_for_frames(remaining_ms)
        aligned = align_to_color.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            last_problem = "frameset has no color or aligned depth frame"
            continue
        current = frame_stamp(color_frame, depth_frame)
        problems: list[str] = []
        if current.sensor_skew_ms > max_pair_skew_ms:
            problems.append(
                f"sensor skew {current.sensor_skew_ms:.3f} ms exceeds {max_pair_skew_ms:.3f} ms"
            )
        if previous is not None:
            if current.color_frame_number <= previous.color_frame_number:
                problems.append("color frame number did not increase")
            if current.depth_frame_number <= previous.depth_frame_number:
                problems.append("depth frame number did not increase")
            if current.color_sensor_timestamp_s <= previous.color_sensor_timestamp_s:
                problems.append("color hardware timestamp did not increase")
            if current.depth_sensor_timestamp_s <= previous.depth_sensor_timestamp_s:
                problems.append("depth hardware timestamp did not increase")
        if not problems:
            return color_frame, depth_frame, current
        last_problem = "; ".join(problems)
    raise RuntimeError(
        f"timed out after {timeout_ms} ms waiting for synchronized RGB-D: {last_problem}"
    )


def _looks_like_robot_state(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("{"):
        return False
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and "O_T_EE" in value


def _read_robot_stdout(session: RobotReaderSession) -> None:
    try:
        assert session.process.stdout is not None
        for line in session.process.stdout:
            timed_line = TimedRobotLine(time.monotonic_ns(), line)
            with session.condition:
                session.stdout_lines.append(timed_line)
                session.condition.notify_all()
    except Exception as exc:  # pragma: no cover - exceptional OS/pipe failure
        with session.condition:
            session.stdout_error = repr(exc)
            session.condition.notify_all()
    finally:
        with session.condition:
            session.stdout_done = True
            session.condition.notify_all()


def _read_robot_stderr(session: RobotReaderSession) -> None:
    try:
        assert session.process.stderr is not None
        while chunk := session.process.stderr.read(4096):
            with session.condition:
                session.stderr_parts.append(chunk)
    except Exception as exc:  # pragma: no cover - exceptional OS/pipe failure
        with session.condition:
            session.stderr_error = repr(exc)
            session.condition.notify_all()
    finally:
        with session.condition:
            session.stderr_done = True
            session.condition.notify_all()


def robot_reader_command(args: argparse.Namespace) -> list[str]:
    if args.robot_reader_kind == "ros_topic":
        count = max(
            args.min_robot_states,
            args.min_robot_states_before_exposure + args.min_robot_states_after_exposure + 100,
        )
        return [
            str(args.ros_python),
            str(args.ros_state_reader),
            "--topic",
            args.ros_robot_state_topic,
            "--count",
            str(count),
            "--timeout-s",
            str(args.robot_timeout_s),
        ]
    return [str(args.echo_robot_state), args.robot_hostname]


def start_robot_reader(args: argparse.Namespace) -> RobotReaderSession | None:
    if args.dry_run_camera_only or args.t_base_flange is not None:
        return None
    started = time.monotonic_ns()
    process = subprocess.Popen(
        robot_reader_command(args),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    session = RobotReaderSession(
        process=process,
        condition=threading.Condition(),
        stdout_lines=[],
        stderr_parts=[],
        started_host_monotonic_ns=started,
        deadline_monotonic_s=time.monotonic() + args.robot_timeout_s,
    )
    stdout_thread = threading.Thread(
        target=_read_robot_stdout,
        args=(session,),
        name="echo_robot_state-stdout",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_read_robot_stderr,
        args=(session,),
        name="echo_robot_state-stderr",
        daemon=True,
    )
    session.threads = (stdout_thread, stderr_thread)
    stdout_thread.start()
    stderr_thread.start()
    return session


def _robot_reader_detail(session: RobotReaderSession) -> str:
    with session.condition:
        stderr = "".join(session.stderr_parts).strip()
        stdout = "".join(line.text for line in session.stdout_lines).strip()
    return (stderr or stdout)[-1000:]


def wait_for_robot_states(session: RobotReaderSession, minimum_states: int) -> None:
    with session.condition:
        while True:
            count = sum(_looks_like_robot_state(line.text) for line in session.stdout_lines)
            if count >= minimum_states:
                return
            if session.stdout_error is not None:
                raise RuntimeError(
                    f"failed reading echo_robot_state stdout: {session.stdout_error}"
                )
            if session.process.poll() is not None and session.stdout_done:
                detail = _robot_reader_detail(session)
                raise RuntimeError(
                    f"echo_robot_state ended after {count} states, before the required "
                    f"{minimum_states}: {detail}"
                )
            remaining = session.deadline_monotonic_s - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"timed out waiting for {minimum_states} robot states before RGB-D "
                    f"capture; received {count}"
                )
            session.condition.wait(timeout=min(remaining, 0.1))


def stop_failed_robot_reader(session: RobotReaderSession | None) -> None:
    if session is None:
        return
    process = session.process
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for thread in session.threads:
        thread.join(timeout=1.0)


def finish_robot_reader(session: RobotReaderSession) -> RobotReaderOutput:
    remaining = session.deadline_monotonic_s - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("echo_robot_state exceeded its total timeout")
    try:
        return_code = session.process.wait(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("echo_robot_state exceeded its total timeout") from exc
    for thread in session.threads:
        thread.join(timeout=1.0)
    if any(thread.is_alive() for thread in session.threads):
        raise RuntimeError("echo_robot_state output reader did not finish")
    finished = time.monotonic_ns()
    if session.stdout_error is not None:
        raise RuntimeError(f"failed reading echo_robot_state stdout: {session.stdout_error}")
    if session.stderr_error is not None:
        raise RuntimeError(f"failed reading echo_robot_state stderr: {session.stderr_error}")
    with session.condition:
        return RobotReaderOutput(
            stdout_lines=tuple(session.stdout_lines),
            stderr="".join(session.stderr_parts),
            return_code=return_code,
            started_host_monotonic_ns=session.started_host_monotonic_ns,
            finished_host_monotonic_ns=finished,
        )


def capture_camera_and_robot(
    args: argparse.Namespace,
) -> tuple[CameraSample, RobotReaderOutput | None]:
    requested = find_device(args.serial)
    requested_info = device_info(requested)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align_to_color = rs.align(rs.stream.color)

    profile = None
    robot_session: RobotReaderSession | None = None
    try:
        profile = pipeline.start(config)
        active_device = profile.get_device()
        active_info = device_info(active_device)
        if active_info.get("serial") != args.serial:
            raise RuntimeError(
                f"opened RealSense {active_info.get('serial')!r}, expected {args.serial!r}"
            )
        if active_info != requested_info:
            requested_info = active_info
        depth_scale_m = float(active_device.first_depth_sensor().get_depth_scale())
        if not math.isfinite(depth_scale_m) or depth_scale_m <= 0:
            raise RuntimeError(f"invalid RealSense depth scale {depth_scale_m}")

        previous = None
        for _ in range(args.warmup_frames):
            _, _, previous = wait_for_aligned_pair(
                pipeline,
                align_to_color,
                previous,
                args.camera_timeout_ms,
                args.max_pair_skew_ms,
            )

        robot_session = start_robot_reader(args)
        if robot_session is not None:
            wait_for_robot_states(robot_session, args.min_robot_states_before_exposure)
            # Consume the newest available pair after the pre-frame robot-state gate.  The
            # following pair must advance both hardware counters/timestamps, so a frame that
            # was already queued before the gate cannot become the calibration exposure.
            _, _, previous = wait_for_aligned_pair(
                pipeline,
                align_to_color,
                previous,
                args.camera_timeout_ms,
                args.max_pair_skew_ms,
            )
        color_frame, depth_frame, stamp = wait_for_aligned_pair(
            pipeline,
            align_to_color,
            previous,
            args.camera_timeout_ms,
            args.max_pair_skew_ms,
        )
        host_timestamp_ns = time.time_ns()
        host_monotonic_ns = time.monotonic_ns()
        rgb = np.asanyarray(color_frame.get_data()).copy()
        depth_raw = np.asanyarray(depth_frame.get_data()).copy()
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        expected_rgb_shape = (args.height, args.width, 3)
        expected_depth_shape = (args.height, args.width)
        if rgb.shape != expected_rgb_shape or rgb.dtype != np.uint8:
            raise RuntimeError(
                f"unexpected RGB shape/type {rgb.shape}/{rgb.dtype}; "
                f"expected {expected_rgb_shape}/uint8"
            )
        if depth_raw.shape != expected_depth_shape or depth_raw.dtype != np.uint16:
            raise RuntimeError(
                f"unexpected depth shape/type {depth_raw.shape}/{depth_raw.dtype}; "
                f"expected {expected_depth_shape}/uint16"
            )
        depth_m = depth_raw.astype(np.float32) * depth_scale_m

        robot_output = None
        if robot_session is not None:
            robot_output = finish_robot_reader(robot_session)
            robot_session = None

        sample = CameraSample(
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=intrinsics,
            depth_scale_m=depth_scale_m,
            stamp=stamp,
            host_timestamp_ns=host_timestamp_ns,
            host_monotonic_ns=host_monotonic_ns,
            device=requested_info,
        )
        return sample, robot_output
    finally:
        stop_failed_robot_reader(robot_session)
        if profile is not None:
            pipeline.stop()


def intrinsics_matrix(intrinsics: Any) -> np.ndarray:
    return np.asarray(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distortion_name(intrinsics: Any) -> str:
    return str(intrinsics.model).removeprefix("distortion.")


def distortion_coefficients(intrinsics: Any) -> np.ndarray:
    return np.asarray(intrinsics.coeffs, dtype=np.float64)


def undistort_pixels(points: np.ndarray, intrinsics: Any) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    model = distortion_name(intrinsics)
    coefficients = distortion_coefficients(intrinsics)
    matrix = intrinsics_matrix(intrinsics)
    if model in {"none", "inverse_brown_conrady"}:
        normalized = []
        for point in points:
            ray = rs.rs2_deproject_pixel_to_point(
                intrinsics, [float(point[0]), float(point[1])], 1.0
            )
            if not np.isfinite(ray).all() or ray[2] <= 0:
                raise RuntimeError(f"could not undistort RealSense pixel {point.tolist()}")
            normalized.append([ray[0] / ray[2], ray[1] / ray[2]])
        normalized_array = np.asarray(normalized, dtype=np.float64)
        return np.column_stack(
            (
                normalized_array[:, 0] * matrix[0, 0] + matrix[0, 2],
                normalized_array[:, 1] * matrix[1, 1] + matrix[1, 2],
            )
        )
    if model in {"brown_conrady", "modified_brown_conrady"}:
        return cv2.undistortPoints(
            points.reshape(-1, 1, 2), matrix, coefficients, P=matrix
        ).reshape(-1, 2)
    raise RuntimeError(
        f"unsupported RealSense color distortion model {model!r}; calibration pose would "
        "not be geometrically trustworthy"
    )


def validate_rigid_transform(matrix: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64)
    if result.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-7):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 1e-5:
        raise ValueError(f"{name} rotation determinant is {determinant:.8f}, expected +1")
    return result


def project_camera_points(points_camera: np.ndarray, intrinsics: Any) -> np.ndarray:
    pixels = []
    for point in np.asarray(points_camera, dtype=np.float64).reshape(-1, 3):
        if not np.isfinite(point).all() or point[2] <= 0:
            raise ValueError("axis endpoint is behind the camera")
        pixel = rs.rs2_project_point_to_pixel(intrinsics, point.tolist())
        pixels.append(pixel)
    return np.asarray(pixels, dtype=np.float64)


def draw_pose_overlay(
    rgb: np.ndarray,
    corners: np.ndarray,
    ids: np.ndarray,
    transform: np.ndarray,
    intrinsics: Any,
    axis_length_m: float,
    rms_error_px: float,
) -> np.ndarray:
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.aruco.drawDetectedCornersCharuco(overlay, corners, ids)
    return draw_axes_overlay(
        overlay,
        transform,
        intrinsics,
        axis_length_m,
        f"ChArUco corners={len(ids)}  reprojection RMS={rms_error_px:.3f}px",
    )


def draw_axes_overlay(
    overlay_bgr: np.ndarray,
    transform: np.ndarray,
    intrinsics: Any,
    axis_length_m: float,
    text_label: str,
) -> np.ndarray:
    overlay = overlay_bgr.copy()
    axes_board = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length_m, 0.0, 0.0],
            [0.0, axis_length_m, 0.0],
            [0.0, 0.0, axis_length_m],
        ],
        dtype=np.float64,
    )
    axes_camera = (transform[:3, :3] @ axes_board.T).T + transform[:3, 3]
    pixels = np.rint(project_camera_points(axes_camera, intrinsics)).astype(int)
    origin = tuple(int(value) for value in pixels[0])
    for endpoint, color in zip(pixels[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0)), strict=True):
        cv2.line(
            overlay,
            origin,
            tuple(int(value) for value in endpoint),
            color,
            3,
            cv2.LINE_AA,
        )
    cv2.circle(overlay, origin, 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(
        overlay,
        text_label,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def estimate_board_pose(
    rgb: np.ndarray,
    intrinsics: Any,
    board: Any,
    board_data: dict[str, Any],
    min_corners: int,
    min_x_span_fraction: float,
    min_y_span_fraction: float,
    min_id_span_fraction: float,
    max_rms_px: float,
    max_point_px: float,
) -> BoardPose:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _marker_corners, marker_ids = detector.detectBoard(gray)
    corner_count = 0 if ids is None else len(ids)
    marker_count = 0 if marker_ids is None else len(marker_ids)
    if corners is None or ids is None or corner_count < min_corners:
        raise RuntimeError(
            f"insufficient ChArUco corners: detected {corner_count}, require at least "
            f"{min_corners}; detected markers={marker_count}"
        )
    flattened_ids = [int(value) for value in ids.reshape(-1)]
    if len(set(flattened_ids)) != len(flattened_ids):
        raise RuntimeError("ChArUco detector returned duplicate corner IDs")
    if board.checkCharucoCornersCollinear(ids):
        raise RuntimeError("detected ChArUco corners are collinear")

    all_charuco_points = np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
    if len(all_charuco_points) < 2:
        raise RuntimeError("ChArUco board has too few defined corners for spatial validation")
    if min(flattened_ids) < 0 or max(flattened_ids) >= len(all_charuco_points):
        raise RuntimeError(f"detected ChArUco ID is outside [0, {len(all_charuco_points) - 1}]")
    full_span = np.ptp(all_charuco_points[:, :2], axis=0)
    if np.any(full_span <= 0):
        raise RuntimeError("ChArUco board has no two-dimensional corner span")
    detected_board_points = all_charuco_points[flattened_ids]
    detected_span = np.ptp(detected_board_points[:, :2], axis=0)
    span_x_fraction = float(detected_span[0] / full_span[0])
    span_y_fraction = float(detected_span[1] / full_span[1])
    id_span_fraction = float(
        (max(flattened_ids) - min(flattened_ids)) / (len(all_charuco_points) - 1)
    )
    coverage_problems = []
    if span_x_fraction < min_x_span_fraction:
        coverage_problems.append(f"X span {span_x_fraction:.3f} is below {min_x_span_fraction:.3f}")
    if span_y_fraction < min_y_span_fraction:
        coverage_problems.append(f"Y span {span_y_fraction:.3f} is below {min_y_span_fraction:.3f}")
    if id_span_fraction < min_id_span_fraction:
        coverage_problems.append(
            f"ID span {id_span_fraction:.3f} is below {min_id_span_fraction:.3f}"
        )
    if coverage_problems:
        raise RuntimeError(
            "ChArUco corners cover too little of the board: " + "; ".join(coverage_problems)
        )

    object_points, raw_image_points = board.matchImagePoints(corners, ids)
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    raw_image_points = np.asarray(raw_image_points, dtype=np.float64).reshape(-1, 2)
    if len(object_points) != corner_count or len(raw_image_points) != corner_count:
        raise RuntimeError("CharucoBoard.matchImagePoints returned inconsistent point counts")
    image_points = undistort_pixels(raw_image_points, intrinsics)
    matrix = intrinsics_matrix(intrinsics)
    zero_distortion = np.zeros(5, dtype=np.float64)
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points,
        matrix,
        zero_distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("cv2.solvePnP failed for the detected ChArUco corners")
    rotation_vector, translation_vector = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        matrix,
        zero_distortion,
        rotation_vector,
        translation_vector,
    )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation_vector).reshape(3)
    validate_rigid_transform(transform, "T_camera_board")

    points_camera = (rotation @ object_points.T).T + transform[:3, 3]
    if np.min(points_camera[:, 2]) <= 0:
        raise RuntimeError("estimated ChArUco board is not fully in front of the camera")
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation_vector,
        matrix,
        zero_distortion,
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    rms_error = float(np.sqrt(np.mean(np.square(errors))))
    max_error = float(np.max(errors))
    if not math.isfinite(rms_error) or rms_error > max_rms_px:
        raise RuntimeError(
            f"ChArUco reprojection RMS {rms_error:.3f} px exceeds {max_rms_px:.3f} px"
        )
    if not math.isfinite(max_error) or max_error > max_point_px:
        raise RuntimeError(
            f"ChArUco maximum point error {max_error:.3f} px exceeds {max_point_px:.3f} px"
        )

    overlay = draw_pose_overlay(
        rgb,
        corners,
        ids,
        transform,
        intrinsics,
        board_data["square_length_m"] * 2.0,
        rms_error,
    )
    return BoardPose(
        transform=transform,
        overlay_bgr=overlay,
        board_kind="charuco",
        corner_count=corner_count,
        marker_count=marker_count,
        reprojection_rms_px=rms_error,
        reprojection_max_px=max_error,
        detected_ids=flattened_ids,
        span_x_fraction=span_x_fraction,
        span_y_fraction=span_y_fraction,
        id_span_fraction=id_span_fraction,
        area_fraction=None,
        planar_pose_second_rms_px=None,
        planar_pose_rms_gap_px=None,
        planar_pose_positive_candidates=None,
        planar_pose_second_rotation_difference_deg=None,
        planar_pose_second_translation_difference_m=None,
    )


def estimate_checkerboard_pose(
    rgb: np.ndarray,
    intrinsics: Any,
    board_data: dict[str, Any],
    min_area_fraction: float,
    min_planar_pose_rms_gap_px: float,
    max_rms_px: float,
    max_point_px: float,
) -> BoardPose:
    detection = detect_checkerboard(
        rgb,
        board_data,
        min_area_fraction=min_area_fraction,
    )
    raw_image_points = detection.image_points
    object_points = detection.object_points
    image_points = undistort_pixels(raw_image_points, intrinsics)
    matrix = intrinsics_matrix(intrinsics)
    estimate = solve_checkerboard_pose(object_points, image_points, matrix)
    transform = estimate.transform
    validate_rigid_transform(transform, "T_camera_board")
    rms_error = estimate.reprojection_rms_px
    max_error = estimate.reprojection_max_px
    second_rms = estimate.second_best_reprojection_rms_px
    pose_rms_gap = (
        None if second_rms is None else second_rms - estimate.best_ippe_reprojection_rms_px
    )
    if pose_rms_gap is not None and pose_rms_gap < min_planar_pose_rms_gap_px:
        raise RuntimeError(
            f"checkerboard planar pose is ambiguous: best/second IPPE RMS gap "
            f"{pose_rms_gap:.4f} px is below {min_planar_pose_rms_gap_px:.4f} px; "
            "tilt the board farther from fronto-parallel"
        )
    if not math.isfinite(rms_error) or rms_error > max_rms_px:
        raise RuntimeError(
            f"checkerboard reprojection RMS {rms_error:.3f} px exceeds {max_rms_px:.3f} px"
        )
    if not math.isfinite(max_error) or max_error > max_point_px:
        raise RuntimeError(
            f"checkerboard maximum point error {max_error:.3f} px exceeds {max_point_px:.3f} px"
        )

    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.drawChessboardCorners(
        overlay,
        (board_data["inner_corners_x"], board_data["inner_corners_y"]),
        raw_image_points.reshape(-1, 1, 2).astype(np.float32),
        True,
    )
    overlay = draw_axes_overlay(
        overlay,
        transform,
        intrinsics,
        board_data["square_length_m"] * 2.0,
        f"Checkerboard corners={len(object_points)}  reprojection RMS={rms_error:.3f}px",
    )
    return BoardPose(
        transform=transform,
        overlay_bgr=overlay,
        board_kind="checkerboard",
        corner_count=len(object_points),
        marker_count=0,
        reprojection_rms_px=rms_error,
        reprojection_max_px=max_error,
        detected_ids=None,
        span_x_fraction=1.0,
        span_y_fraction=1.0,
        id_span_fraction=None,
        area_fraction=detection.area_fraction,
        planar_pose_second_rms_px=second_rms,
        planar_pose_rms_gap_px=pose_rms_gap,
        planar_pose_positive_candidates=estimate.positive_ippe_candidates,
        planar_pose_second_rotation_difference_deg=(estimate.second_best_rotation_difference_deg),
        planar_pose_second_translation_difference_m=(estimate.second_best_translation_difference_m),
    )


def matrix_from_column_major(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (16,):
        raise ValueError(f"robot state {name} must contain 16 values, got shape {array.shape}")
    return validate_rigid_transform(array.reshape((4, 4), order="F"), name)


def mean_transform(transforms: np.ndarray, name: str) -> np.ndarray:
    matrices = np.asarray(transforms, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError(f"cannot average {name} with shape {matrices.shape}")
    rotation_sum = matrices[:, :3, :3].sum(axis=0)
    left, _, right_transpose = np.linalg.svd(rotation_sum)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transpose
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = matrices[:, :3, 3].mean(axis=0)
    return validate_rigid_transform(result, name)


def pose_spread(transforms: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    translation_error = np.linalg.norm(transforms[:, :3, 3] - reference[:3, 3], axis=1)
    relative_rotations = np.einsum("ij,njk->nik", reference[:3, :3].T, transforms[:, :3, :3])
    cosines = np.clip(
        (np.trace(relative_rotations, axis1=1, axis2=2) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    rotation_error_deg = np.rad2deg(np.arccos(cosines))
    return float(np.max(translation_error)), float(np.max(rotation_error_deg))


def parse_robot_states(stdout: str) -> list[dict[str, Any]]:
    states = []
    malformed = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            malformed.append(f"line {line_number}: {exc}")
            continue
        if isinstance(value, dict) and "O_T_EE" in value:
            states.append(value)
    if malformed:
        raise RuntimeError("malformed robot-state JSON: " + "; ".join(malformed[:3]))
    return states


def summarize_robot_states(
    output: RobotReaderOutput | tuple[str, str, int],
    args: argparse.Namespace,
    minimum_states: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(output, RobotReaderOutput):
        stdout = output.stdout
        stderr = output.stderr
        return_code = output.return_code
    else:
        stdout, stderr, return_code = output
    if return_code != 0:
        detail = (stderr.strip() or stdout.strip())[-1000:]
        raise RuntimeError(f"echo_robot_state exited with {return_code}: {detail}")
    states = parse_robot_states(stdout)
    required_states = args.min_robot_states if minimum_states is None else minimum_states
    if len(states) < required_states:
        raise RuntimeError(
            f"echo_robot_state returned {len(states)} valid states; require at least "
            f"{required_states}"
        )

    base_ee = []
    flange_ee = []
    base_flange = []
    velocities = []
    joints = []
    for index, state in enumerate(states):
        current_base_ee = matrix_from_column_major(state.get("O_T_EE"), f"O_T_EE[{index}]")
        current_flange_ee = matrix_from_column_major(state.get("F_T_EE"), f"F_T_EE[{index}]")
        current_base_flange = validate_rigid_transform(
            current_base_ee @ np.linalg.inv(current_flange_ee),
            f"T_base_flange[{index}]",
        )
        dq = np.asarray(state.get("dq"), dtype=np.float64)
        q = np.asarray(state.get("q"), dtype=np.float64)
        valid_dq = dq.shape == (7,) and np.isfinite(dq).all()
        valid_q = q.shape == (7,) and np.isfinite(q).all()
        if not valid_dq or not valid_q:
            raise ValueError(f"robot state {index} has invalid q/dq arrays")
        base_ee.append(current_base_ee)
        flange_ee.append(current_flange_ee)
        base_flange.append(current_base_flange)
        velocities.append(dq)
        joints.append(q)

    base_ee_array = np.stack(base_ee)
    flange_ee_array = np.stack(flange_ee)
    base_flange_array = np.stack(base_flange)
    velocities_array = np.stack(velocities)
    joints_array = np.stack(joints)
    representative_base_ee = mean_transform(base_ee_array, "mean O_T_EE")
    representative_flange_ee = mean_transform(flange_ee_array, "mean F_T_EE")
    representative_base_flange = mean_transform(base_flange_array, "mean T_base_flange")

    max_abs_dq = float(np.max(np.abs(velocities_array)))
    translation_spread, rotation_spread_deg = pose_spread(
        base_flange_array, representative_base_flange
    )
    flange_translation_spread, flange_rotation_spread_deg = pose_spread(
        flange_ee_array, representative_flange_ee
    )
    if max_abs_dq > args.max_abs_dq_rad_s:
        raise RuntimeError(
            f"robot is moving: max |dq| {max_abs_dq:.6f} rad/s exceeds "
            f"{args.max_abs_dq_rad_s:.6f} rad/s"
        )
    if translation_spread > args.max_pose_translation_m:
        raise RuntimeError(
            f"robot pose translation spread {translation_spread:.6f} m exceeds "
            f"{args.max_pose_translation_m:.6f} m"
        )
    if rotation_spread_deg > args.max_pose_rotation_deg:
        raise RuntimeError(
            f"robot pose rotation spread {rotation_spread_deg:.6f} deg exceeds "
            f"{args.max_pose_rotation_deg:.6f} deg"
        )
    if flange_translation_spread > 1e-7 or flange_rotation_spread_deg > 1e-5:
        raise RuntimeError("configured F_T_EE changed while reading robot states")

    numeric_times: list[float] = []
    errors_seen: set[str] = set()
    for index, state in enumerate(states):
        value = state.get("time")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"robot state {index} has an invalid time value")
        numeric_times.append(float(value))
        state_errors = state.get("current_errors")
        if not isinstance(state_errors, list):
            raise ValueError(f"robot state {index} current_errors must be a list")
        errors_seen.update(str(error) for error in state_errors)
    if any(current <= previous for previous, current in zip(numeric_times, numeric_times[1:])):
        raise RuntimeError("robot-state time values are not strictly increasing")
    current_errors = sorted(errors_seen)
    if current_errors:
        raise RuntimeError(
            "robot reports active errors while capturing calibration state: "
            + ", ".join(current_errors)
        )
    reader_reviewed = bool(getattr(args, "robot_reader_reviewed", True))
    reader_kind = getattr(args, "robot_reader_kind", "libfranka")
    ros_reader = reader_kind == "ros_topic"
    summary = {
        "source": (
            "ROS 2 franka_robot_state_broadcaster read-only subscription"
            if ros_reader and reader_reviewed
            else (
                "libfranka echo_robot_state (repository-reviewed read-only Robot.read callback)"
                if reader_reviewed
                else "external robot-state executable (command behavior not audited)"
            )
        ),
        "reader_assurance": (
            "repository-reviewed read-only executable"
            if reader_reviewed
            else "unknown; explicitly allowed by operator"
        ),
        "executable": str(args.ros_state_reader if ros_reader else args.echo_robot_state),
        "ros_python": str(args.ros_python) if ros_reader else None,
        "ros_topic": args.ros_robot_state_topic if ros_reader else None,
        "robot_hostname": None if ros_reader else args.robot_hostname,
        "state_count": len(states),
        "robot_modes": sorted(
            {str(state.get("robot_mode")) for state in states if "robot_mode" in state}
        ),
        "robot_time_ms": {
            "first": numeric_times[0] if numeric_times else None,
            "last": numeric_times[-1] if numeric_times else None,
        },
        "stationarity": {
            "max_abs_dq_rad_s": max_abs_dq,
            "limit_max_abs_dq_rad_s": args.max_abs_dq_rad_s,
            "max_translation_from_mean_m": translation_spread,
            "limit_translation_m": args.max_pose_translation_m,
            "max_rotation_from_mean_deg": rotation_spread_deg,
            "limit_rotation_deg": args.max_pose_rotation_deg,
            "joint_peak_to_peak_rad": np.ptp(joints_array, axis=0).tolist(),
            "passed": True,
        },
        "current_errors_seen": current_errors,
        "T_base_ee_mean": representative_base_ee.tolist(),
        "T_flange_ee_mean": representative_flange_ee.tolist(),
        "T_base_flange_mean": representative_base_flange.tolist(),
        "matrix_storage_in_echo_output": "column-major",
        "composition": "T_base_flange = T_base_ee @ inverse(T_flange_ee)",
    }
    return representative_base_flange, summary


def pose_difference(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    translation_m = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    relative_rotation = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
    return translation_m, float(np.rad2deg(np.arccos(cosine)))


def subset_robot_output(
    output: RobotReaderOutput, lines: list[TimedRobotLine]
) -> RobotReaderOutput:
    return RobotReaderOutput(
        stdout_lines=tuple(lines),
        stderr="",
        return_code=0,
        started_host_monotonic_ns=output.started_host_monotonic_ns,
        finished_host_monotonic_ns=output.finished_host_monotonic_ns,
    )


def resolve_robot_pose(
    args: argparse.Namespace,
    robot_output: RobotReaderOutput | None,
    frame_host_monotonic_ns: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if args.dry_run_camera_only:
        return None, {
            "source": "camera-only dry run",
            "valid_for_handeye_calibration": False,
            "reason": "no synchronized stationary robot flange pose was captured",
        }
    if args.t_base_flange is not None:
        transform = validate_rigid_transform(
            np.load(args.t_base_flange, allow_pickle=False), "provided T_base_flange"
        )
        return transform, {
            "source": "provided transform",
            "source_path": str(args.t_base_flange),
            "valid_for_handeye_calibration": True,
            "T_base_flange": transform.tolist(),
        }
    if robot_output is None:
        raise RuntimeError("internal error: live robot-state output is missing")
    transform, summary = summarize_robot_states(robot_output, args)
    state_lines = [line for line in robot_output.stdout_lines if _looks_like_robot_state(line.text)]
    before_lines = [
        line for line in state_lines if line.host_monotonic_ns < frame_host_monotonic_ns
    ]
    after_lines = [line for line in state_lines if line.host_monotonic_ns > frame_host_monotonic_ns]
    before_transform, before_summary = summarize_robot_states(
        subset_robot_output(robot_output, before_lines),
        args,
        minimum_states=args.min_robot_states_before_exposure,
    )
    after_transform, after_summary = summarize_robot_states(
        subset_robot_output(robot_output, after_lines),
        args,
        minimum_states=args.min_robot_states_after_exposure,
    )
    bracket_translation_m, bracket_rotation_deg = pose_difference(before_transform, after_transform)
    if bracket_translation_m > args.max_pose_translation_m:
        raise RuntimeError(
            f"robot pose changed across RGB-D capture by {bracket_translation_m:.6f} m, "
            f"exceeding {args.max_pose_translation_m:.6f} m"
        )
    if bracket_rotation_deg > args.max_pose_rotation_deg:
        raise RuntimeError(
            f"robot pose changed across RGB-D capture by {bracket_rotation_deg:.6f} deg, "
            f"exceeding {args.max_pose_rotation_deg:.6f} deg"
        )
    summary["exposure_bracket"] = {
        "timing_basis": (
            "host CLOCK_MONOTONIC timestamps at flushed state-line receipt and immediately "
            "after RGB-D frameset receipt"
        ),
        "reader_started_host_monotonic_ns": robot_output.started_host_monotonic_ns,
        "frame_received_host_monotonic_ns": frame_host_monotonic_ns,
        "reader_finished_host_monotonic_ns": robot_output.finished_host_monotonic_ns,
        "states_before_frame_receipt": len(before_lines),
        "minimum_states_before_frame_receipt": args.min_robot_states_before_exposure,
        "states_after_frame_receipt": len(after_lines),
        "minimum_states_after_frame_receipt": args.min_robot_states_after_exposure,
        "before_after_translation_m": bracket_translation_m,
        "limit_translation_m": args.max_pose_translation_m,
        "before_after_rotation_deg": bracket_rotation_deg,
        "limit_rotation_deg": args.max_pose_rotation_deg,
        "passed": True,
    }
    summary["stationarity_windows"] = {
        "before_frame_receipt": before_summary,
        "after_frame_receipt": after_summary,
    }
    summary["valid_for_handeye_calibration"] = True
    return transform, summary


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        if destination.exists():
            raise FileExistsError(f"sample directory already exists: {destination}")
        os.rename(source, destination)
        return
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(f"sample directory already exists: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def write_sample(
    args: argparse.Namespace,
    camera: CameraSample,
    pose: BoardPose,
    board_metadata: dict[str, Any],
    board_data: dict[str, Any],
    base_flange: np.ndarray | None,
    robot_summary: dict[str, Any],
) -> None:
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"sample directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent)
    )
    published = False
    try:
        rgb_path = staging_dir / "rgb.png"
        overlay_path = staging_dir / "overlay.png"
        if not cv2.imwrite(str(rgb_path), cv2.cvtColor(camera.rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"could not write {rgb_path}")
        if not cv2.imwrite(str(overlay_path), pose.overlay_bgr):
            raise RuntimeError(f"could not write {overlay_path}")
        np.save(staging_dir / "depth.npy", camera.depth_m, allow_pickle=False)
        np.save(staging_dir / "K.npy", intrinsics_matrix(camera.intrinsics), allow_pickle=False)
        np.save(
            staging_dir / "distortion.npy",
            distortion_coefficients(camera.intrinsics),
            allow_pickle=False,
        )
        np.save(staging_dir / "T_camera_board.npy", pose.transform, allow_pickle=False)
        if base_flange is not None:
            np.save(staging_dir / "T_base_flange.npy", base_flange, allow_pickle=False)
        (staging_dir / "robot_state_summary.json").write_text(
            json.dumps(robot_summary, indent=2) + "\n", encoding="utf-8"
        )
        (staging_dir / "board_metadata.json").write_text(
            json.dumps(board_metadata, indent=2) + "\n", encoding="utf-8"
        )

        stamp = camera.stamp
        if pose.board_kind == "charuco":
            board_observation = {
                "detected_charuco_corners": pose.corner_count,
                "detected_markers": pose.marker_count,
                "detected_charuco_ids": pose.detected_ids,
                "detected_span_x_fraction": pose.span_x_fraction,
                "detected_span_y_fraction": pose.span_y_fraction,
                "detected_id_span_fraction": pose.id_span_fraction,
                "reprojection_rms_px": pose.reprojection_rms_px,
                "reprojection_max_px": pose.reprojection_max_px,
                "limits": {
                    "minimum_charuco_corners": args.min_charuco_corners,
                    "minimum_charuco_x_span_fraction": args.min_charuco_x_span_fraction,
                    "minimum_charuco_y_span_fraction": args.min_charuco_y_span_fraction,
                    "minimum_charuco_id_span_fraction": args.min_charuco_id_span_fraction,
                    "maximum_reprojection_rms_px": args.max_reprojection_rms_px,
                    "maximum_reprojection_point_px": args.max_reprojection_point_px,
                },
            }
        else:
            board_observation = {
                "detected_checkerboard_inner_corners": pose.corner_count,
                "complete_board_detected": True,
                "detected_inner_corner_area_fraction": pose.area_fraction,
                "corner_detector": "findChessboardCornersSBWithMeta",
                "subpixel_localization": "sector-based detector native subpixel output",
                "orientation_signature_verified": True,
                "planar_pose_second_reprojection_rms_px": pose.planar_pose_second_rms_px,
                "planar_pose_best_second_rms_gap_px": pose.planar_pose_rms_gap_px,
                "planar_pose_positive_ippe_candidates": pose.planar_pose_positive_candidates,
                "planar_pose_second_rotation_difference_deg": (
                    pose.planar_pose_second_rotation_difference_deg
                ),
                "planar_pose_second_translation_difference_m": (
                    pose.planar_pose_second_translation_difference_m
                ),
                "reprojection_rms_px": pose.reprojection_rms_px,
                "reprojection_max_px": pose.reprojection_max_px,
                "limits": {
                    "minimum_checkerboard_area_fraction": args.min_checkerboard_area_fraction,
                    "minimum_planar_pose_best_second_rms_gap_px": (args.min_planar_pose_rms_gap_px),
                    "maximum_reprojection_rms_px": args.max_reprojection_rms_px,
                    "maximum_reprojection_point_px": args.max_reprojection_point_px,
                },
            }
        manifest = {
            "schema": "demo2gripper.eye_to_hand_sample.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "session_id": args.session_id,
            "valid_for_handeye_calibration": base_flange is not None,
            "files": {
                "rgb": "rgb.png",
                "aligned_depth_m": "depth.npy",
                "camera_matrix": "K.npy",
                "distortion_coefficients": "distortion.npy",
                "T_camera_board": "T_camera_board.npy",
                "T_base_flange": "T_base_flange.npy" if base_flange is not None else None,
                "pose_overlay": "overlay.png",
                "robot_state_summary": "robot_state_summary.json",
                "board_metadata_copy": "board_metadata.json",
            },
            "camera": {
                "device": camera.device,
                "requested_profile": {
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                    "color_format": "RGB8 saved as PNG",
                    "depth_source_format": "Z16",
                },
                "depth_alignment": "RealSense SDK depth-to-color reprojection",
                "depth_output": "float32 metres",
                "depth_scale_m_per_unit": camera.depth_scale_m,
                "valid_depth_fraction": float(
                    np.count_nonzero(camera.depth_m) / camera.depth_m.size
                ),
                "distortion_model": distortion_name(camera.intrinsics),
                "coordinate_frame": "OpenCV color optical: +X right, +Y down, +Z forward",
                "timestamp": {
                    "host_unix_ns": camera.host_timestamp_ns,
                    "host_monotonic_ns_at_frameset_receipt": camera.host_monotonic_ns,
                    "color_sensor_s": stamp.color_sensor_timestamp_s,
                    "depth_sensor_s": stamp.depth_sensor_timestamp_s,
                    "sensor_pair_skew_ms": stamp.sensor_skew_ms,
                    "color_sdk_s": stamp.color_sdk_timestamp_s,
                    "depth_sdk_s": stamp.depth_sdk_timestamp_s,
                    "sdk_domain": stamp.sdk_timestamp_domain,
                    "color_frame_number": stamp.color_frame_number,
                    "depth_frame_number": stamp.depth_frame_number,
                },
            },
            "board": {
                **board_data,
                "metadata_source": str(args.board_metadata),
                **board_observation,
            },
            "transforms": {
                "convention": "T_parent_child, metres, column vectors",
                "T_camera_board": "T_camera_board.npy",
                "T_base_flange": "T_base_flange.npy" if base_flange is not None else None,
            },
            "robot_pose": robot_summary,
            "safety": {
                "robot_access": (
                    "none"
                    if args.dry_run_camera_only or args.t_base_flange is not None
                    else (
                        "repository-reviewed read-only ROS 2 state subscription"
                        if args.robot_reader_reviewed and args.robot_reader_kind == "ros_topic"
                        else (
                            "repository-reviewed read-only libfranka Robot.read"
                            if args.robot_reader_reviewed
                            else "external executable; robot-control behavior unknown"
                        )
                    )
                ),
                "control_commands_sent": (
                    False
                    if (
                        args.dry_run_camera_only
                        or args.t_base_flange is not None
                        or args.robot_reader_reviewed
                    )
                    else None
                ),
                "control_command_assurance": (
                    "no robot executable invoked"
                    if args.dry_run_camera_only or args.t_base_flange is not None
                    else (
                        "repository-reviewed reader contains no control command path"
                        if args.robot_reader_reviewed
                        else "unknown; operator explicitly allowed an unreviewed executable"
                    )
                ),
            },
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        rename_noreplace(staging_dir, output_dir)
        published = True
    finally:
        if not published and staging_dir.exists():
            shutil.rmtree(staging_dir)


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    board_metadata, board_data = load_board_metadata(args.board_metadata)
    board = make_board(board_data)
    camera, robot_output = capture_camera_and_robot(args)
    if board_data.get("schema") == CHECKERBOARD_SCHEMA:
        pose = estimate_checkerboard_pose(
            camera.rgb,
            camera.intrinsics,
            board_data,
            args.min_checkerboard_area_fraction,
            args.min_planar_pose_rms_gap_px,
            args.max_reprojection_rms_px,
            args.max_reprojection_point_px,
        )
    else:
        pose = estimate_board_pose(
            camera.rgb,
            camera.intrinsics,
            board,
            board_data,
            args.min_charuco_corners,
            args.min_charuco_x_span_fraction,
            args.min_charuco_y_span_fraction,
            args.min_charuco_id_span_fraction,
            args.max_reprojection_rms_px,
            args.max_reprojection_point_px,
        )
    base_flange, robot_summary = resolve_robot_pose(args, robot_output, camera.host_monotonic_ns)
    write_sample(
        args,
        camera,
        pose,
        board_metadata,
        board_data,
        base_flange,
        robot_summary,
    )
    mode = "camera-only dry run" if base_flange is None else "complete calibration sample"
    board_label = "ChArUco" if pose.board_kind == "charuco" else "checkerboard"
    print(
        f"Captured {mode} at {args.output_dir}: {pose.corner_count} {board_label} corners, "
        f"reprojection RMS {pose.reprojection_rms_px:.3f} px"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
