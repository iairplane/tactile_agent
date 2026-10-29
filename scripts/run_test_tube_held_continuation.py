#!/usr/bin/env python3
"""Continue the tube task from a previously verified bilateral held state.

This entry point deliberately does *not* calibrate either GF220 while the tube
is held.  It restores the unloaded WARPED references written by the fast close
run, localizes the selected rack socket from a fresh RGB-D burst, and then
executes one resident sequence: extract, rigid level, collision-clear transfer,
socket alignment/insertion, release, and retreat.

The demonstrated gravity-pivot primitive is not reconstructed from a loaded
sensor restart.  This continuation therefore uses an auditable agentic direct
alignment of the rigidly held tube to the live socket axis.  It never claims a
tactile angle, force, slip, jam, or seating measurement from WARPED contact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import signal
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_test_tube_continuous_close as contact  # noqa: E402
from diagnose_gf220_sdk import (  # noqa: E402
    _collect_new,
    _discover_devices,
    _read_profile,
)
from monitor_test_tube_held_contact import classify_warped_contact  # noqa: E402

from demo2gripper.d435_live import D435LiveConfig, D435LiveSource  # noqa: E402
from demo2gripper.test_tube_closed_loop import (  # noqa: E402
    held_tube_level_target,
    pose_error,
    tube_horizontal_error_deg,
)
from demo2gripper.test_tube_live_io import (  # noqa: E402
    HandEModbusRuntime,
    LiveIOError,
    RosMoveItServoRuntime,
    _load_artifact_transform,
)
from demo2gripper.test_tube_online_socket import (  # noqa: E402
    HoughSocketConfig,
    OnlineSocketEntryTracker,
    OnlineSocketTrackerConfig,
    extract_hough_socket_candidates,
)

LEVEL_ESCAPE_SOCKET_TRACKER_OVERRIDES = {
    "max_translation_jump_m": 0.015,
    "max_orientation_jump_deg": 15.0,
    "max_window_position_std_m": 0.006,
    "max_window_orientation_std_deg": 8.0,
}

COARSE_ENDPOINT_MISMATCH_REASON = (
    "trajectory action reported success, but measured TCP did not reach the requested endpoint"
)
COARSE_SERVO_RECOVERY_MAX_TRANSLATION_M = 0.010
COARSE_SERVO_RECOVERY_MAX_ROTATION_DEG = 3.0


def _resolve(path: Path, *, base: Path = ROOT) -> Path:
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _matrix(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous row")
    if not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if np.linalg.det(result[:3, :3]) < 0.999:
        raise ValueError(f"{name} rotation is not right handed")
    return result


def _source_tube_pose(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload
    for key in ("observation", "tube_pose", "T_base_object"):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"{path}: missing {key} in source-step observation")
        value = value[key]
    return _matrix(value, "source-step T_base_tube")


class LevelEscapeResume:
    """Audited state handed from a held LEVEL stop to this continuation."""

    __slots__ = (
        "held_report",
        "escape_receipt",
        "expected_start_tcp",
        "terminal_tcp",
        "terminal_tube",
        "T_tube_tcp",
        "held_width_m",
        "horizontal_error_deg",
        "artifacts_sha256",
    )

    def __init__(
        self,
        *,
        held_report: Path,
        escape_receipt: Path,
        expected_start_tcp: np.ndarray,
        terminal_tcp: np.ndarray,
        terminal_tube: np.ndarray,
        T_tube_tcp: np.ndarray,
        held_width_m: float,
        horizontal_error_deg: float,
        artifacts_sha256: str,
    ) -> None:
        self.held_report = held_report
        self.escape_receipt = escape_receipt
        self.expected_start_tcp = expected_start_tcp
        self.terminal_tcp = terminal_tcp
        self.terminal_tube = terminal_tube
        self.T_tube_tcp = T_tube_tcp
        self.held_width_m = held_width_m
        self.horizontal_error_deg = horizontal_error_deg
        self.artifacts_sha256 = artifacts_sha256


class FailedContinuationResume:
    """Audited endpoint of a successful coarse move which needs Servo settling."""

    __slots__ = (
        "report_path",
        "coarse_receipt_path",
        "target_tcp",
        "measured_tcp",
        "artifacts_sha256",
    )

    def __init__(
        self,
        *,
        report_path: Path,
        coarse_receipt_path: Path,
        target_tcp: np.ndarray,
        measured_tcp: np.ndarray,
        artifacts_sha256: str,
    ) -> None:
        self.report_path = report_path
        self.coarse_receipt_path = coarse_receipt_path
        self.target_tcp = target_tcp
        self.measured_tcp = measured_tcp
        self.artifacts_sha256 = artifacts_sha256


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _pose_from_fields(payload: dict, position_key: str, quaternion_key: str) -> np.ndarray:
    position = np.asarray(payload.get(position_key), dtype=float)
    quaternion = np.asarray(payload.get(quaternion_key), dtype=float)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError(f"escape receipt lacks {position_key}/{quaternion_key}")
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("escape receipt pose is not finite")
    if abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-3:
        raise ValueError("escape receipt quaternion is not unit length")
    result = np.eye(4, dtype=float)
    result[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    result[:3, 3] = position
    return _matrix(result, f"escape receipt {position_key}")


def _task_directory(path: Path) -> Path:
    root = _resolve(path)
    if (root / "task" / "run_report.json").is_file():
        return root / "task"
    if (root / "run_report.json").is_file():
        return root
    raise ValueError(f"held run has no run_report.json: {root}")


def load_level_escape_resume(
    held_run: Path,
    escape_receipt: Path,
    *,
    horizontal_tolerance_deg: float,
) -> LevelEscapeResume:
    """Prove that an executed escape starts at the recorded held LEVEL stop."""

    task = _task_directory(held_run)
    report_path = task / "run_report.json"
    report = _read_json(report_path)
    if not (
        report.get("schema") == "demo2gripper.test_tube_closed_loop_report.v1"
        and report.get("mode") == "execute"
        and report.get("status") == "held"
        and report.get("final_state") == "HOLD"
        and report.get("hardware_commands_dispatched") is True
    ):
        raise ValueError("held run is not an executed resident HOLD")
    step_files = sorted((task / "steps").glob("[0-9][0-9][0-9][0-9].json"))
    if not step_files or int(report.get("step_count", -1)) != len(step_files):
        raise ValueError("held run step history is incomplete")
    steps = [_read_json(path) for path in step_files]
    if any(step.get("index") != index for index, step in enumerate(steps)):
        raise ValueError("held run step indices are not contiguous")
    terminal = steps[-1]
    decision = terminal.get("decision", {})
    action = decision.get("action", {})
    dispatch = terminal.get("dispatch", {})
    observation = terminal.get("observation", {})
    tactile = observation.get("tactile", {})
    gripper = observation.get("gripper", {})
    if not (
        decision.get("previous_state") == "LEVEL_HELD_TUBE"
        and decision.get("next_state") == "HOLD"
        and action.get("kind") == "stop_and_hold"
        and action.get("reason") == "LEVEL_HELD_TUBE timeout exceeded"
        and dispatch.get("status") == "held"
        and dispatch.get("receipt", {}).get("held") is True
        and tactile.get("valid") is True
        and tactile.get("bilateral_contact") is True
        and tactile.get("bilateral_contact_interior") is True
        and gripper.get("fault_free") is True
        and gripper.get("moving") is False
    ):
        raise ValueError("held run lacks the exact healthy LEVEL timeout contract")

    terminal_tcp = _matrix(
        observation.get("robot", {}).get("T_base_tcp"), "terminal T_base_tcp"
    )
    terminal_tube = _matrix(
        observation.get("tube_pose", {}).get("T_base_object"),
        "terminal T_base_tube",
    )
    held_width_m = float(gripper.get("width_m", math.nan))
    if not math.isfinite(held_width_m) or not 0.005 <= held_width_m <= 0.040:
        raise ValueError("held run has an invalid closed width")
    T_tube_tcp = _matrix(
        np.linalg.inv(terminal_tube) @ terminal_tcp,
        "terminal rigid T_tube_tcp",
    )

    level_step = next(
        (
            step
            for step in reversed(steps[:-1])
            if step.get("decision", {}).get("action", {}).get("kind") == "servo_tcp"
            and step.get("decision", {})
            .get("action", {})
            .get("parameters", {})
            .get("purpose")
            == "rigid_held_tube_level"
        ),
        None,
    )
    if level_step is None:
        raise ValueError("held run has no rigid level target")
    level_target = _matrix(
        level_step["decision"]["action"]["parameters"]["T_base_tcp_live_target"],
        "recorded level target",
    )

    receipt_path = _resolve(escape_receipt)
    receipt = _read_json(receipt_path)
    audit = receipt.get("trajectory_audit", {})
    if not (
        receipt.get("schema") == "demo2gripper.moveit_pose_goal.v2"
        and receipt.get("mode") == "execute"
        and receipt.get("input_target_frame") == "configured-ee"
        and receipt.get("planning_succeeded") is True
        and receipt.get("execution_requested") is True
        and receipt.get("execution_dispatched") is True
        and receipt.get("execution_succeeded") is True
        and receipt.get("execution_outcome") == "measured_target_reached"
        and audit.get("accepted") is True
        and not audit.get("violations")
    ):
        raise ValueError("level escape receipt is not a successful audited execution")
    receipt_start = _pose_from_fields(
        receipt,
        "start_configured_ee_position_base_m",
        "start_configured_ee_orientation_xyzw",
    )
    start_translation_m, start_rotation_deg = pose_error(terminal_tcp, receipt_start)
    if start_translation_m > 0.003 or start_rotation_deg > 2.0:
        raise ValueError("level escape did not start at the recorded terminal hold")
    requested = _pose_from_fields(
        receipt,
        "input_target_position_base_m",
        "input_target_orientation_xyzw",
    )
    target_translation_m, target_rotation_deg = pose_error(level_target, requested)
    lift_m = float(requested[2, 3] - level_target[2, 3])
    lateral_m = float(np.linalg.norm(requested[:2, 3] - level_target[:2, 3]))
    if not (
        0.030 <= lift_m <= 0.080
        and lateral_m <= 0.003
        and target_rotation_deg <= 0.5
        and 0.030 <= target_translation_m <= 0.081
    ):
        raise ValueError("escape target is not the bounded lifted level target")
    expected_start_tcp = _pose_from_fields(
        receipt,
        "measured_end_configured_ee_position_base_m",
        "measured_end_configured_ee_orientation_xyzw",
    )
    expected_tube = _matrix(
        expected_start_tcp @ np.linalg.inv(T_tube_tcp),
        "post-escape expected T_base_tube",
    )
    horizontal_error = tube_horizontal_error_deg(expected_tube)
    if horizontal_error > horizontal_tolerance_deg:
        raise ValueError(
            "executed level escape did not leave the held tube horizontal: "
            f"{horizontal_error:.3f} deg"
        )

    digest = hashlib.sha256(report_path.read_bytes())
    for path in step_files:
        digest.update(path.read_bytes())
    digest.update(receipt_path.read_bytes())
    return LevelEscapeResume(
        held_report=report_path,
        escape_receipt=receipt_path,
        expected_start_tcp=expected_start_tcp,
        terminal_tcp=terminal_tcp,
        terminal_tube=terminal_tube,
        T_tube_tcp=T_tube_tcp,
        held_width_m=held_width_m,
        horizontal_error_deg=horizontal_error,
        artifacts_sha256=digest.hexdigest(),
    )


def _recoverable_coarse_endpoint_shortfall(
    receipt: dict,
    *,
    expected_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the audited target/endpoint for the one allowed fine-settle case."""

    audit = receipt.get("trajectory_audit", {})
    position_error_m = float(
        receipt.get("measured_input_target_position_error_m", math.nan)
    )
    rotation_error_rad = float(
        receipt.get("measured_input_target_rotation_error_rad", math.nan)
    )
    if not (
        receipt.get("schema") == "demo2gripper.moveit_pose_goal.v2"
        and receipt.get("mode") == "execute"
        and receipt.get("input_target_frame") == "configured-ee"
        and receipt.get("planning_succeeded") is True
        and receipt.get("execution_requested") is True
        and receipt.get("execution_dispatched") is True
        and receipt.get("execution_action_reported_success") is True
        and receipt.get("execution_succeeded") is False
        and receipt.get("execution_outcome") == "measured_endpoint_mismatch"
        and receipt.get("execution_failure_reason") == COARSE_ENDPOINT_MISMATCH_REASON
        and receipt.get("measurement_error") is None
        and receipt.get("moveit_error_code") == 1
        and receipt.get("execution_error_code") == 1
        and audit.get("accepted") is True
        and not audit.get("violations")
        and math.isfinite(position_error_m)
        and math.isfinite(rotation_error_rad)
        and 0.0 < position_error_m <= COARSE_SERVO_RECOVERY_MAX_TRANSLATION_M
        and 0.0 <= math.degrees(rotation_error_rad)
        <= COARSE_SERVO_RECOVERY_MAX_ROTATION_DEG
    ):
        raise ValueError("coarse endpoint mismatch is not eligible for Servo settling")
    target = _pose_from_fields(
        receipt,
        "input_target_position_base_m",
        "input_target_orientation_xyzw",
    )
    target_translation_m, target_rotation_deg = pose_error(expected_target, target)
    if target_translation_m > 1e-6 or target_rotation_deg > 0.01:
        raise ValueError("coarse mismatch receipt does not name the expected target")
    measured = _pose_from_fields(
        receipt,
        "measured_end_configured_ee_position_base_m",
        "measured_end_configured_ee_orientation_xyzw",
    )
    measured_translation_m, measured_rotation_deg = pose_error(measured, target)
    if (
        abs(measured_translation_m - position_error_m) > 0.001
        or abs(math.radians(measured_rotation_deg) - rotation_error_rad)
        > math.radians(0.25)
    ):
        raise ValueError("coarse mismatch receipt errors disagree with its measured endpoint")
    return target, measured


def load_failed_continuation_resume(
    path: Path,
    *,
    level_resume: LevelEscapeResume,
    config_sha256: str,
    baseline_sha256: str,
    socket_receipt: Path,
) -> FailedContinuationResume:
    """Validate the exact failed-HOLD handoff at the near socket-standoff endpoint."""

    root = _resolve(path)
    report_path = root / "run_report.json"
    report = _read_json(report_path)
    error = report.get("error", {})
    final_arm = report.get("final_arm_stop", {})
    final_gripper = report.get("final_gripper_stop", {})
    if not (
        report.get("schema") == "demo2gripper.test_tube_held_continuation.v1"
        and report.get("mode") == "execute"
        and report.get("status") == "failed_hold"
        and report.get("hardware_commands_dispatched") is True
        and report.get("config_sha256") == config_sha256
        and report.get("baseline_warped_reference_sha256") == baseline_sha256
        and error.get("type") == "LiveIOError"
        and COARSE_ENDPOINT_MISMATCH_REASON in str(error.get("message", ""))
        and final_arm.get("accepted") is True
        and final_arm.get("reason") == "continuation finished"
        and final_gripper.get("accepted") is True
        and final_gripper.get("reason") == "continuation failed"
    ):
        raise ValueError("prior continuation is not the exact recoverable failed HOLD")
    recorded_level = report.get("level_escape_resume", {})
    if (
        Path(str(recorded_level.get("held_report", ""))).resolve()
        != level_resume.held_report.resolve()
        or Path(str(recorded_level.get("escape_receipt", ""))).resolve()
        != level_resume.escape_receipt.resolve()
        or recorded_level.get("artifacts_sha256") != level_resume.artifacts_sha256
    ):
        raise ValueError("prior continuation does not match the audited level escape")
    prior_socket = report.get("prepared_socket_evidence", {})
    if (
        Path(str(prior_socket.get("receipt", ""))).resolve()
        != socket_receipt.resolve()
    ):
        raise ValueError("prior continuation used a different socket receipt")
    stages = report.get("stages")
    if not isinstance(stages, list):
        raise ValueError("prior continuation stages are missing")
    transfer = [
        item
        for item in stages
        if isinstance(item, dict) and item.get("name") == "transfer_clear"
    ]
    if len(transfer) != 1 or transfer[0].get("status") != "complete":
        raise ValueError("prior continuation did not complete transfer_clear exactly once")
    if any(
        isinstance(item, dict)
        and item.get("name") in {"release", "retreat_50mm"}
        for item in stages
    ):
        raise ValueError("prior continuation already reached release or retreat")

    receipt_path = root / "arm" / "socket_standoff.json"
    receipt = _read_json(receipt_path)
    recorded_target = _pose_from_fields(
        receipt,
        "input_target_position_base_m",
        "input_target_orientation_xyzw",
    )
    target, measured = _recoverable_coarse_endpoint_shortfall(
        receipt,
        expected_target=recorded_target,
    )
    digest = hashlib.sha256(report_path.read_bytes())
    digest.update(receipt_path.read_bytes())
    return FailedContinuationResume(
        report_path=report_path,
        coarse_receipt_path=receipt_path,
        target_tcp=target,
        measured_tcp=measured,
        artifacts_sha256=digest.hexdigest(),
    )


class RestoredWarpedContact:
    """Live WARPED contact classifier backed only by saved unloaded images."""

    def __init__(self, *, baseline_dir: Path, profile_path: Path) -> None:
        metadata = json.loads(
            (baseline_dir / "unloaded_contact_profile.json").read_text(encoding="utf-8")
        )
        if metadata.get("schema") != "demo2gripper.test_tube_contact_approach.v1":
            raise ValueError("unrecognized saved GF220 contact profile")
        profile = _read_profile(profile_path)
        fingers = metadata["fingers"]
        sensors = profile["sensors"]
        serials = [item["sdk_serial"] for item in sensors]
        if len(fingers) != 2 or set(serials) != {item["serial"] for item in fingers}:
            raise ValueError("saved and live GF220 serial sets differ")
        self._references: dict[str, tuple[np.ndarray, np.ndarray, float, int]] = {}
        with np.load(
            baseline_dir / "unloaded_contact_profile.npz", allow_pickle=False
        ) as saved:
            for index, finger in enumerate(fingers):
                self._references[finger["serial"]] = (
                    saved[f"finger_{index}_warped_reference"].copy(),
                    saved[f"finger_{index}_roi_mask"].copy(),
                    float(finger["warped_residual_threshold"]),
                    int(finger["min_component_area_px"]),
                )
        self._profile = profile
        self._serials = serials
        self._devices: list[tuple[dict, object]] = []
        self._previous = {serial: (None, None) for serial in serials}
        self._types = None
        self._last_sdk = {serial: -math.inf for serial in serials}

    def start(self) -> None:
        from pyvitaisdk import GF225, GF225OutputProfile, GFDataType, VTSDeviceFinder
        from pyvitaisdk.vts.core.base.profile import VideoStreamFmt

        _, handles = _discover_devices(self._serials, 10, 0.5, VTSDeviceFinder)
        stream = self._profile["stream"]
        for spec in self._profile["sensors"]:
            serial = spec["sdk_serial"]
            device = GF225(
                config=handles[serial],
                stream_format=VideoStreamFmt(**stream),
                output_format=GF225OutputProfile.W240_H240,
            )
            self._devices.append((spec, device))
        raw = (GFDataType.TIME_STAMP, GFDataType.RAW_IMG)
        for _ in range(8):
            for spec, device in self._devices:
                serial = spec["sdk_serial"]
                _, stamp, crc = _collect_new(
                    device, raw, *raw, *self._previous[serial], 2.0
                )
                self._previous[serial] = (stamp, crc)
        for spec, device in self._devices:
            device._image_processor.set_warp_params(
                corner_points=spec["corners_xy"], mode="manual"
            )
        self._types = (*raw, GFDataType.WARPED_IMG)

    def sample(self) -> dict:
        from pyvitaisdk import GFDataType

        if self._types is None:
            raise RuntimeError("GF220 contact source is not started")
        pads, captures = [], []
        for spec, device in self._devices:
            serial = spec["sdk_serial"]
            sample, stamp, crc = _collect_new(
                device,
                self._types,
                GFDataType.TIME_STAMP,
                GFDataType.RAW_IMG,
                *self._previous[serial],
                2.0,
            )
            captured = time.monotonic()
            if stamp <= self._last_sdk[serial]:
                raise RuntimeError(f"{serial}: GF220 timestamp did not advance")
            self._last_sdk[serial] = stamp
            self._previous[serial] = (stamp, crc)
            warped = np.asarray(sample[GFDataType.WARPED_IMG]).copy()
            evidence = classify_warped_contact(
                warped, *self._references[serial]
            )
            pads.append({"serial": serial, "sdk_timestamp_raw": stamp, **evidence})
            captures.append(captured)
        skew = max(captures) - min(captures)
        return {
            "observed_at_monotonic_s": time.monotonic(),
            "valid": skew <= 0.08,
            "pair_skew_s": skew,
            "bilateral_contact": skew <= 0.08 and all(item["contact"] for item in pads),
            "pads": pads,
        }

    def close(self) -> None:
        for _, device in self._devices:
            try:
                device.release()
            except Exception:
                pass
        self._devices.clear()


def _fresh_socket(
    config: dict,
    *,
    tracker_overrides: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict]:
    camera = config["live_adapter"]["camera"]
    downstream = config["live_adapter"]["scene_interpreter"]["config"][
        "downstream_config"
    ]
    tracker_options = dict(downstream["tracker"])
    if tracker_overrides is not None:
        unknown = set(tracker_overrides).difference(
            LEVEL_ESCAPE_SOCKET_TRACKER_OVERRIDES
        )
        if unknown:
            raise ValueError(f"unsupported socket tracker overrides: {sorted(unknown)}")
        tracker_options.update(tracker_overrides)
    if tracker_options.get("target_hint_uv") is not None:
        tracker_options["target_hint_uv"] = tuple(tracker_options["target_hint_uv"])
    hough_options = dict(downstream["hough"])
    hough_options["roi_uvxy"] = tuple(hough_options["roi_uvxy"])
    tracker = OnlineSocketEntryTracker(OnlineSocketTrackerConfig(**tracker_options))
    hough = HoughSocketConfig(**hough_options)
    source = D435LiveSource(
        D435LiveConfig(
            serial=camera["serial"],
            width=int(camera["width"]),
            height=int(camera["height"]),
            fps=int(camera["fps"]),
            warmup_frames=int(camera["warmup_frames"]),
            frame_timeout_ms=int(camera["frame_timeout_ms"]),
            max_pair_skew_ms=float(camera["max_pair_skew_ms"]),
        )
    )
    extrinsic = _resolve(Path(camera["camera_extrinsic"]))
    T_base_camera = _matrix(np.load(extrinsic, allow_pickle=False), "T_base_camera")
    trace: list[dict] = []
    try:
        source.start()
        for index in range(20):
            frame = source.read()
            candidates = extract_hough_socket_candidates(
                frame.rgb, frame.depth_m, frame.intrinsics.matrix, hough
            )
            qualified = tracker.update(
                timestamp_s=frame.host_received_monotonic_s, candidates=candidates
            )
            trace.append(
                {
                    "index": index,
                    "candidate_count": len(candidates),
                    "centres_uv": [list(item.centre_uv) for item in candidates],
                    "hold_reasons": list(tracker.last_reasons),
                }
            )
            if qualified is not None:
                return T_base_camera @ qualified.T_camera_entry, {
                    "position_std_m": qualified.position_std_m,
                    "orientation_std_deg": qualified.orientation_std_deg,
                    "sample_count": qualified.sample_count,
                    "evidence_id": qualified.evidence_id,
                    "tracker_config": tracker_options,
                    "trace": trace,
                }
    finally:
        source.close()
    raise RuntimeError("selected rack socket did not qualify in 20 fresh RGB-D frames")


def _qualified_socket_receipt(path: Path) -> tuple[np.ndarray, dict]:
    receipt_path = _resolve(path)
    receipt = _read_json(receipt_path)
    position_std_m = float(receipt.get("position_std_m", math.inf))
    orientation_std_deg = float(receipt.get("orientation_std_deg", math.inf))
    sample_count = int(receipt.get("sample_count", 0))
    if not (
        receipt.get("qualified") is True
        and position_std_m <= 0.006
        and orientation_std_deg <= 8.0
        and sample_count >= 4
        and isinstance(receipt.get("evidence_id"), str)
    ):
        raise ValueError("socket pose receipt is not a bounded qualified observation")
    return _matrix(receipt.get("T_base_socket"), "receipt T_base_socket"), {
        "source": "separately qualified current-scene RGB-D burst",
        "receipt": str(receipt_path),
        "observed_at_local": receipt.get("observed_at_local"),
        "position_std_m": position_std_m,
        "orientation_std_deg": orientation_std_deg,
        "sample_count": sample_count,
        "evidence_id": receipt.get("evidence_id"),
    }


def _retained_hold(touch: dict, gripper_feedback) -> bool:
    """Require fresh image contact on both saved-reference WARPED pads.

    Hand-E object status remains recorded as corroborating evidence, but it
    cannot substitute for either tactile pad in this task.
    """

    del gripper_feedback
    return bool(touch["valid"] and touch["bilateral_contact"])


def extraction_target(current: np.ndarray, extract_mm: float) -> np.ndarray:
    if not math.isfinite(extract_mm) or not 40.0 <= extract_mm <= 120.0:
        raise ValueError("--extract-mm must lie in [40, 120]")
    target = _matrix(current, "extraction start").copy()
    target[2, 3] += extract_mm / 1000.0
    return target


def _wait_retained(
    tactile: RestoredWarpedContact,
    gripper: HandEModbusRuntime,
    *,
    windows: int = 2,
) -> list[dict]:
    evidence: list[dict] = []
    streak = 0
    for _ in range(8):
        item = tactile.sample()
        evidence.append(item)
        feedback = gripper.feedback()
        item["hande"] = asdict(feedback)
        item["retained_hold"] = _retained_hold(item, feedback)
        streak = streak + 1 if item["retained_hold"] else 0
        if streak >= windows:
            return evidence
    raise RuntimeError("Hand-E plus saved-reference GF220 did not confirm retained contact")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--source-step", type=Path)
    parser.add_argument(
        "--held-run",
        type=Path,
        help="Resident LEVEL hold to resume after an audited escape trajectory.",
    )
    parser.add_argument(
        "--level-escape-receipt",
        type=Path,
        help="Executed MoveIt receipt which lifted and finished rigid leveling.",
    )
    parser.add_argument(
        "--resume-failed-continuation",
        type=Path,
        help=(
            "Prior continuation that completed transfer_clear and stopped after an "
            "audited <=10 mm socket-standoff endpoint shortfall."
        ),
    )
    parser.add_argument(
        "--socket-pose-json",
        type=Path,
        help="Use a separately qualified current-scene T_base_socket receipt.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--extract-mm",
        type=float,
        default=80.0,
        help="Vertical base-Z extraction before leveling.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-arm-only-collision-model", action="store_true")
    parser.add_argument("--accept-provisional-camera-extrinsic", action="store_true")
    args = parser.parse_args(argv)
    resume_values = (args.held_run, args.level_escape_receipt)
    if any(value is not None for value in resume_values):
        if not all(value is not None for value in resume_values):
            parser.error("--held-run and --level-escape-receipt must be supplied together")
        if args.source_step is not None:
            parser.error("--source-step cannot be combined with held level resume")
    elif args.source_step is None:
        parser.error("--source-step is required unless held level resume is selected")
    if args.resume_failed_continuation is not None:
        if not all(value is not None for value in resume_values):
            parser.error(
                "--resume-failed-continuation requires --held-run and "
                "--level-escape-receipt"
            )
        if args.socket_pose_json is None:
            parser.error("--resume-failed-continuation requires --socket-pose-json")
    extraction_target(np.eye(4), args.extract_mm)
    if args.execute and not args.accept_arm_only_collision_model:
        parser.error("--execute requires --accept-arm-only-collision-model")
    if args.execute and not args.accept_provisional_camera_extrinsic:
        parser.error("--execute requires --accept-provisional-camera-extrinsic")
    return args


def run(args) -> int:
    config_path = _resolve(args.config)
    baseline_dir = _resolve(args.baseline_dir)
    source_step = _resolve(args.source_step) if args.source_step is not None else None
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite non-empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    live = config["live_adapter"]
    robot = copy.deepcopy(live["robot"])
    for key, ceiling in (("max_linear_m_s", 0.015), ("max_angular_rad_s", 0.15)):
        value = float(robot["servo"][key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"invalid robot.servo.{key}")
        robot["servo"][key] = min(value, ceiling)
    profile_json = baseline_dir / "unloaded_contact_profile.json"
    profile_npz = baseline_dir / "unloaded_contact_profile.npz"
    if not profile_json.is_file() or not profile_npz.is_file():
        raise ValueError("baseline directory lacks saved unloaded WARPED references")
    baseline_profile = _read_json(profile_json)
    if baseline_profile.get("schema") != "demo2gripper.test_tube_contact_approach.v1":
        raise ValueError("unrecognized unloaded WARPED reference schema")
    baseline_digest = hashlib.sha256(profile_json.read_bytes())
    baseline_digest.update(profile_npz.read_bytes())

    level_resume = None
    if args.held_run is not None:
        level_resume = load_level_escape_resume(
            args.held_run,
            args.level_escape_receipt,
            horizontal_tolerance_deg=float(config["policy"]["level_axis_tolerance_deg"]),
        )
        T_contact_tcp = level_resume.terminal_tcp.copy()
        expected_width_m = level_resume.held_width_m
        T_contact_tube = level_resume.terminal_tube.copy()
        T_tube_tcp = level_resume.T_tube_tcp.copy()
        expected_resume_tcp = level_resume.expected_start_tcp.copy()
    else:
        baseline = _read_json(baseline_dir / "run_report.json")
        if baseline.get("status") != "grasp_ready":
            raise ValueError("baseline run did not finish with strict tactile grasp_ready")
        T_contact_tcp = _matrix(
            baseline["T_base_tcp_final"], "baseline T_base_tcp_final"
        )
        expected_width_m = float(baseline["final_gripper"]["width_m"])
        assert source_step is not None
        T_contact_tube = _source_tube_pose(source_step)
        T_tube_tcp = np.linalg.inv(T_contact_tube) @ T_contact_tcp
        expected_resume_tcp = None
    prepared_socket = (
        _qualified_socket_receipt(args.socket_pose_json)
        if args.socket_pose_json is not None
        else None
    )
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    failed_continuation_resume = None
    if args.resume_failed_continuation is not None:
        assert level_resume is not None and args.socket_pose_json is not None
        failed_continuation_resume = load_failed_continuation_resume(
            args.resume_failed_continuation,
            level_resume=level_resume,
            config_sha256=config_sha256,
            baseline_sha256=baseline_digest.hexdigest(),
            socket_receipt=_resolve(args.socket_pose_json),
        )
        expected_resume_tcp = failed_continuation_resume.measured_tcp.copy()
    report = {
        "schema": "demo2gripper.test_tube_held_continuation.v1",
        "session_id": f"held-continuation-{uuid.uuid4().hex}",
        "mode": "execute" if args.execute else "dry_run",
        "status": "prepared",
        "config": str(config_path),
        "config_sha256": config_sha256,
        "baseline_dir": str(baseline_dir),
        "baseline_warped_reference_sha256": baseline_digest.hexdigest(),
        "source_step": str(source_step) if source_step is not None else None,
        "level_escape_resume": (
            None
            if level_resume is None
            else {
                "held_report": str(level_resume.held_report),
                "escape_receipt": str(level_resume.escape_receipt),
                "artifacts_sha256": level_resume.artifacts_sha256,
                "expected_start_tcp": level_resume.expected_start_tcp.tolist(),
                "post_escape_horizontal_error_deg": (
                    level_resume.horizontal_error_deg
                ),
                "extraction_and_leveling": "already_complete",
            }
        ),
        "failed_continuation_resume": (
            None
            if failed_continuation_resume is None
            else {
                "report": str(failed_continuation_resume.report_path),
                "coarse_receipt": str(
                    failed_continuation_resume.coarse_receipt_path
                ),
                "artifacts_sha256": failed_continuation_resume.artifacts_sha256,
                "expected_start_tcp": (
                    failed_continuation_resume.measured_tcp.tolist()
                ),
                "socket_standoff_target": (
                    failed_continuation_resume.target_tcp.tolist()
                ),
                "resume_action": "bounded_servo_endpoint_settle",
            }
        ),
        "servo_limits": robot["servo"],
        "T_base_tcp_contact": T_contact_tcp.tolist(),
        "T_base_tube_contact": T_contact_tube.tolist(),
        "T_tube_tcp_rigid": T_tube_tcp.tolist(),
        "expected_width_m": expected_width_m,
        "extract_mm": args.extract_mm,
        "prepared_socket_evidence": (
            prepared_socket[1] if prepared_socket is not None else None
        ),
        "tactile_contract": {
            "input": "saved unloaded WARPED reference",
            "recalibrated_while_loaded": False,
            "decisions": [
                "retained_only_if_both_saved_reference_WARPED_pads_report_contact",
                "Hand-E_object_status_is_corroborating_only",
            ],
            "not_claimed": ["force", "slip", "angle", "jam", "seat"],
        },
        "pivot": {
            "status": "skipped_agentic_direct_alignment",
            "reason": "loaded SDK restart cannot recover the same-session tactile angle state",
        },
        "stages": [],
    }
    report_path = output / "run_report.json"
    contact._atomic_json(report_path, report)
    if not args.execute:
        report["status"] = "dry_run_complete"
        contact._atomic_json(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path)}))
        return 0

    arm = gripper = tactile = transport = None
    handlers = {}

    def interrupted(signum, _frame):
        raise InterruptedError(f"received signal {signum}")

    def persist_stage(name: str, **values) -> None:
        report["stages"].append({"name": name, "at_monotonic_s": time.monotonic(), **values})
        contact._atomic_json(report_path, report)

    def healthy_arm():
        state = arm.state()
        if state.fault:
            raise RuntimeError(f"arm fault: {state.fault}")
        return state

    contact_loss_streak = 0

    def guarded_move(name: str, target: np.ndarray, timeout_s: float = 35.0) -> np.ndarray:
        nonlocal contact_loss_streak
        target = _matrix(target, f"{name} target")
        receipt = dict(arm.servo_target(target=target, action_id=name))
        observations: list[dict] = []
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            touch = tactile.sample()
            state = healthy_arm()
            feedback = gripper.feedback()
            if not feedback.fault_free:
                raise RuntimeError(f"Hand-E fault during {name}")
            retained = _retained_hold(touch, feedback)
            touch["retained_hold"] = retained
            touch["hande_object_detected"] = feedback.object_detected
            contact_loss_streak = 0 if retained else contact_loss_streak + 1
            translation_m, rotation_deg = pose_error(state.T_base_tcp, target)
            observations.append(
                {
                    "at_monotonic_s": time.monotonic(),
                    "translation_error_m": translation_m,
                    "rotation_error_deg": rotation_deg,
                    "motion_complete": state.motion_complete,
                    "width_m": feedback.width_m,
                    "tactile": touch,
                }
            )
            if contact_loss_streak >= 2:
                arm.stop(reason=f"{name}: tactile contact lost")
                raise RuntimeError(
                    f"Hand-E object and both GF220 contacts were lost during {name}"
                )
            if translation_m <= 0.0020 and rotation_deg <= 2.0 and state.motion_complete:
                arm.stop(reason=f"{name}: target reached")
                persist_stage(
                    name,
                    status="complete",
                    target=target.tolist(),
                    receipt=receipt,
                    observations=observations,
                    T_base_tcp_final=state.T_base_tcp.tolist(),
                )
                return state.T_base_tcp.copy()
        arm.stop(reason=f"{name}: timeout")
        raise TimeoutError(f"{name} did not reach its target")

    def guarded_coarse_move(name: str, target: np.ndarray) -> np.ndarray:
        """Use collision-planned redundant IK, with tactile checks before and after."""

        target = _matrix(target, f"{name} target")
        before = _wait_retained(tactile, gripper)
        try:
            receipt = dict(arm.move_coarse(target=target, action_id=name))
        except LiveIOError:
            receipt_path = output / "arm" / f"{name}.json"
            if not receipt_path.is_file():
                raise
            failed_receipt = _read_json(receipt_path)
            _, measured = _recoverable_coarse_endpoint_shortfall(
                failed_receipt,
                expected_target=target,
            )
            report["hardware_commands_dispatched"] = True
            contact._atomic_json(report_path, report)
            state = healthy_arm()
            translation_m, rotation_deg = pose_error(state.T_base_tcp, measured)
            if translation_m > 0.003 or rotation_deg > 2.0:
                raise RuntimeError(
                    f"{name} live TCP differs from the audited coarse endpoint"
                )
            after_coarse = _wait_retained(tactile, gripper)
            settled = guarded_move(
                f"{name}_servo_settle",
                target,
                timeout_s=15.0,
            )
            persist_stage(
                name,
                status="coarse_endpoint_settled_with_servo",
                target=target.tolist(),
                coarse_receipt=str(receipt_path),
                coarse_endpoint=measured.tolist(),
                tactile_before=before,
                tactile_after_coarse=after_coarse,
                T_base_tcp_final=settled.tolist(),
            )
            return settled
        report["hardware_commands_dispatched"] = True
        contact._atomic_json(report_path, report)
        state = healthy_arm()
        translation_m, rotation_deg = pose_error(state.T_base_tcp, target)
        if translation_m > 0.006 or rotation_deg > 3.0:
            arm.stop(reason=f"{name}: measured endpoint mismatch")
            raise RuntimeError(f"{name} did not reach its collision-planned endpoint")
        after = _wait_retained(tactile, gripper)
        persist_stage(
            name,
            status="complete",
            target=target.tolist(),
            receipt=receipt,
            tactile_before=before,
            tactile_after=after,
            T_base_tcp_final=state.T_base_tcp.tolist(),
        )
        return state.T_base_tcp.copy()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, interrupted)
        report["status"] = "localizing_socket"
        contact._atomic_json(report_path, report)
        if prepared_socket is None:
            T_base_socket, socket_evidence = _fresh_socket(
                config,
                tracker_overrides=(
                    LEVEL_ESCAPE_SOCKET_TRACKER_OVERRIDES
                    if level_resume is not None
                    else None
                ),
            )
        else:
            T_base_socket, socket_evidence = prepared_socket
        report["T_base_socket"] = T_base_socket.tolist()
        report["socket_evidence"] = socket_evidence

        tactile_config = live["tactile"]
        transport = contact.SerialModbusRTU(
            tactile_config["hande_port"],
            float(tactile_config.get("modbus_timeout_s", 0.12)),
        )
        contact._read_healthy_initial_state(transport)
        gripper = HandEModbusRuntime(transport)
        profile_path = _resolve(Path(live["tactile"]["gf220_profile"]))
        tactile = RestoredWarpedContact(
            baseline_dir=baseline_dir, profile_path=profile_path
        )
        tactile.start()
        report["initial_tactile"] = _wait_retained(tactile, gripper)
        tcp = _load_artifact_transform(robot, "tcp")
        arm = RosMoveItServoRuntime(config=robot, T_link8_tcp=tcp, output_dir=output)
        contact._read_healthy_initial_state(transport)
        start = healthy_arm()
        feedback = gripper.feedback()
        if failed_continuation_resume is not None:
            assert expected_resume_tcp is not None
            start_translation_m, start_rotation_deg = pose_error(
                start.T_base_tcp, expected_resume_tcp
            )
            if start_translation_m > 0.003 or start_rotation_deg > 2.0:
                raise RuntimeError(
                    "current TCP no longer matches the audited coarse endpoint"
                )
            live_horizontal_error_deg = None
            extracted_translation_m = 0.0
            extracted_rotation_deg = 0.0
            resume_after_extract = True
        elif level_resume is not None:
            assert expected_resume_tcp is not None
            start_translation_m, start_rotation_deg = pose_error(
                start.T_base_tcp, expected_resume_tcp
            )
            if start_translation_m > 0.006 or start_rotation_deg > 3.0:
                raise RuntimeError(
                    "current TCP no longer matches the audited level-escape endpoint"
                )
            live_tube = _matrix(
                start.T_base_tcp @ np.linalg.inv(T_tube_tcp),
                "live post-escape T_base_tube",
            )
            live_horizontal_error_deg = tube_horizontal_error_deg(live_tube)
            if live_horizontal_error_deg > float(
                config["policy"]["level_axis_tolerance_deg"]
            ):
                raise RuntimeError(
                    "current held tube is no longer horizontal after level escape"
                )
            extracted_translation_m = 0.0
            extracted_rotation_deg = 0.0
            resume_after_extract = True
        else:
            start_translation_m, start_rotation_deg = pose_error(
                start.T_base_tcp, T_contact_tcp
            )
            extracted_tcp = extraction_target(T_contact_tcp, args.extract_mm)
            extracted_translation_m, extracted_rotation_deg = pose_error(
                start.T_base_tcp, extracted_tcp
            )
            resume_after_extract = bool(
                extracted_translation_m <= 0.006 and extracted_rotation_deg <= 3.0
            )
            if (
                not resume_after_extract
                and (start_translation_m > 0.006 or start_rotation_deg > 3.0)
            ):
                raise RuntimeError(
                    "current TCP matches neither the verified contact pose nor its "
                    f"completed {args.extract_mm:.0f} mm extraction pose"
                )
        if abs(feedback.width_m - expected_width_m) > 0.003 or feedback.moving:
            raise RuntimeError("Hand-E no longer matches the verified held width")
        if not feedback.fault_free:
            raise RuntimeError("Hand-E reports a fault before continuation")
        persist_stage(
            "held_preflight",
            status="complete",
            T_base_tcp=start.T_base_tcp.tolist(),
            gripper=asdict(feedback),
            baseline_pose_error_m=start_translation_m,
            baseline_rotation_error_deg=start_rotation_deg,
            resume_after_extract=resume_after_extract,
            extracted_pose_error_m=extracted_translation_m,
            held_level_escape_resume=level_resume is not None,
            failed_continuation_resume=failed_continuation_resume is not None,
            live_horizontal_error_deg=(
                live_horizontal_error_deg if level_resume is not None else None
            ),
        )

        report["status"] = "executing"
        contact._atomic_json(report_path, report)
        if failed_continuation_resume is not None:
            measured = start.T_base_tcp.copy()
            persist_stage(
                "failed_socket_standoff_shortfall_resumed",
                status="ready_for_servo_settle",
                T_base_tcp_final=measured.tolist(),
                prior_report=str(failed_continuation_resume.report_path),
                prior_coarse_receipt=str(
                    failed_continuation_resume.coarse_receipt_path
                ),
            )
        elif level_resume is not None:
            measured = start.T_base_tcp.copy()
            persist_stage(
                "audited_level_escape_resumed",
                status="already_complete",
                T_base_tcp_final=measured.tolist(),
                horizontal_error_deg=live_horizontal_error_deg,
            )
        elif resume_after_extract:
            measured = start.T_base_tcp.copy()
            persist_stage(
                f"extract_{args.extract_mm:.0f}mm_resumed",
                status="already_complete",
                T_base_tcp_final=measured.tolist(),
            )
        else:
            extract = extraction_target(start.T_base_tcp, args.extract_mm)
            measured = guarded_move(
                f"extract_{args.extract_mm:.0f}mm", extract, timeout_s=20.0
            )

        if level_resume is None:
            level = held_tube_level_target(
                T_base_tube=T_contact_tube,
                T_base_tcp_measured=measured,
                min_horizontal_projection=0.10,
            )
            measured = guarded_move("rigid_level", level, timeout_s=20.0)

        entry = T_base_socket @ T_tube_tcp
        outward = T_base_socket[:3, 2]
        outward /= np.linalg.norm(outward)
        standoff = entry.copy()
        standoff[:3, 3] += outward * 0.050

        transfer_clear = standoff.copy()
        transfer_clear[2, 3] = max(measured[2, 3], standoff[2, 3] + 0.080)
        if failed_continuation_resume is not None:
            translation_m, rotation_deg = pose_error(
                standoff,
                failed_continuation_resume.target_tcp,
            )
            if translation_m > 1e-6 or rotation_deg > 0.01:
                raise RuntimeError(
                    "current socket evidence changed the audited standoff target"
                )
            guarded_move("socket_standoff_servo_settle", standoff, timeout_s=15.0)
        elif level_resume is not None:
            guarded_coarse_move("transfer_clear", transfer_clear)
            guarded_coarse_move("socket_standoff", standoff)
        else:
            guarded_move("transfer_clear", transfer_clear, timeout_s=35.0)
            guarded_move("socket_standoff", standoff, timeout_s=25.0)

        inward = -outward
        insertion_origin = standoff.copy()
        baseline_touch = tactile.sample()
        baseline_areas = {
            item["serial"]: max(1, int(item["component_area_px"]))
            for item in baseline_touch["pads"]
        }
        insertion_evidence = []
        # 50 mm puts the tube centre at the entry frame; 30 mm more leaves
        # roughly the open-end grasp region outside the rack.
        for index, travel_m in enumerate(np.linspace(0.010, 0.080, 8), start=1):
            target = insertion_origin.copy()
            target[:3, 3] += inward * float(travel_m)
            measured = guarded_move(
                f"guarded_insert_{index:02d}", target, timeout_s=12.0
            )
            touch = tactile.sample()
            ratios = {
                item["serial"]: item["component_area_px"] / baseline_areas[item["serial"]]
                for item in touch["pads"]
            }
            insertion_evidence.append(
                {
                    "travel_m": float(travel_m),
                    "T_base_tcp": measured.tolist(),
                    "tactile": touch,
                    "warped_area_ratio": ratios,
                    "interpretation": "contact-shape trend only; not calibrated force or seat",
                }
            )
        report["insertion_evidence"] = insertion_evidence
        report["insertion_travel_m"] = 0.080
        contact._atomic_json(report_path, report)

        open_receipt = dict(
            gripper.move_width(
                width_m=0.0495, speed=45, force=8, action_id="release"
            )
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            healthy_arm()
            feedback = gripper.feedback()
            if not feedback.fault_free:
                raise RuntimeError("Hand-E fault during release")
            if feedback.width_m >= 0.048 and not feedback.moving:
                break
            time.sleep(0.025)
        else:
            raise TimeoutError("Hand-E release was not confirmed")
        persist_stage(
            "release",
            status="complete",
            receipt=open_receipt,
            gripper=asdict(feedback),
        )

        retreat = healthy_arm().T_base_tcp.copy()
        retreat[:3, 3] += outward * 0.050
        # Contact is expected to disappear after release, so retreat is arm
        # supervised but no longer gated by the held-contact predicate.
        receipt = dict(arm.servo_target(target=retreat, action_id="retreat_50mm"))
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            state = healthy_arm()
            translation_m, rotation_deg = pose_error(state.T_base_tcp, retreat)
            if translation_m <= 0.002 and rotation_deg <= 2.0 and state.motion_complete:
                arm.stop(reason="retreat complete")
                persist_stage(
                    "retreat_50mm",
                    status="complete",
                    receipt=receipt,
                    T_base_tcp_final=state.T_base_tcp.tolist(),
                )
                break
            time.sleep(0.025)
        else:
            arm.stop(reason="retreat timeout")
            raise TimeoutError("retreat did not reach its target")
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed_hold"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if arm is not None:
            try:
                report["final_arm_stop"] = dict(arm.stop(reason="continuation finished"))
            except BaseException as error:
                report["arm_stop_error"] = str(error)
        if gripper is not None and report.get("status") != "complete":
            try:
                report["final_gripper_stop"] = dict(
                    gripper.stop(reason="continuation failed")
                )
            except BaseException as error:
                report["gripper_stop_error"] = str(error)
        for resource in (arm, gripper, tactile):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    report.setdefault("cleanup_errors", []).append(str(error))
        if transport is not None and gripper is None:
            transport.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        contact._atomic_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path)}))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
