#!/usr/bin/env python3
"""Continuously close Hand-E and stop on the first tactile/contact signal.

The default mode is hardware-free.  Live mode sends one low-force close goal,
monitors fresh two-frame dual-GF220 windows plus Hand-E feedback, clears rGTO
immediately on contact/fault/timeout, and then uses exactly two stopped tactile
windows to classify ``grasp_ready`` or ``needs_centering``.  It never moves the
arm, opens the gripper, or extracts the tube.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from demo2gripper.gf220_live import (  # noqa: E402
    DEFAULT_PROFILE,
    GF220ClassifierWindows,
    GF220LiveConfig,
    GF220LiveSource,
)
from demo2gripper.hande_modbus_cli import (  # noqa: E402
    MAX_FORCE_REGISTER,
    MAX_SPEED_REGISTER,
    SerialModbusRTU,
    _target_count,
    feedback_payload,
)
from demo2gripper.test_tube_contact_approach import (  # noqa: E402
    CONTACT_APPROACH_SCHEMA,
    ContactApproachProfile,
    fit_unloaded_contact_profile,
)
from demo2gripper.test_tube_tactile_close import (  # noqa: E402
    TACTILE_CLOSE_SCHEMA,
    TactileCloseDecision,
    TactileCloseMemory,
    evaluate_tactile_close_window,
)

REPORT_SCHEMA = "demo2gripper.test_tube_continuous_close_run.v1"
DEFAULT_HAND_E_PORT = Path(
    "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AQ00O97B-if00-port0"
)
RUNTIME_WINDOW_FRAMES = 2
CONFIRMATION_WINDOWS = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--port", type=Path, default=DEFAULT_HAND_E_PORT)
    parser.add_argument("--minimum-opening-mm", type=float, default=10.0)
    parser.add_argument("--speed", type=int, default=45)
    parser.add_argument("--force", type=int, default=8)
    parser.add_argument("--motion-timeout-s", type=float, default=4.0)
    parser.add_argument("--stop-timeout-s", type=float, default=1.0)
    parser.add_argument("--modbus-timeout-s", type=float, default=0.12)
    parser.add_argument("--gf220-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--baseline-windows", type=int, default=3)
    parser.add_argument("--baseline-window-frames", type=int, default=5)
    parser.add_argument("--sensor-timeout-s", type=float, default=1.0)
    parser.add_argument("--calibration-timeout-s", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--assert-pads-unloaded", action="store_true")
    parser.add_argument("--assert-tube-between-open-fingers", action="store_true")
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.minimum_opening_mm) or not 5.0 <= args.minimum_opening_mm <= 25.0:
        raise ValueError("--minimum-opening-mm must lie in [5, 25]")
    if not 1 <= args.speed <= min(MAX_SPEED_REGISTER, 70):
        raise ValueError("--speed must lie in [1, 70]")
    if not 1 <= args.force <= min(MAX_FORCE_REGISTER, 20):
        raise ValueError("--force must lie in [1, 20]")
    if args.baseline_windows < 2 or args.baseline_window_frames < 5:
        raise ValueError("baseline requires at least two windows of at least five frames")
    for name in (
        "motion_timeout_s",
        "stop_timeout_s",
        "modbus_timeout_s",
        "sensor_timeout_s",
        "calibration_timeout_s",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.execute:
        if not args.assert_pads_unloaded or not args.assert_tube_between_open_fingers:
            raise ValueError(
                "--execute requires --assert-pads-unloaded and "
                "--assert-tube-between-open-fingers"
            )
        if not args.port.is_absolute() or not args.port.exists():
            raise ValueError(f"Hand-E port is not an existing absolute path: {args.port}")
        if not args.gf220_profile.is_file():
            raise ValueError(f"GF220 profile does not exist: {args.gf220_profile}")


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = REPOSITORY_ROOT / "runs/live" / f"test_tube_continuous_close_{stamp}"
    else:
        output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output.resolve()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _save_window(path: Path, pair: GF220ClassifierWindows) -> None:
    arrays: dict[str, np.ndarray] = {
        "serials": np.asarray(pair.serials),
        "endpoint_skew_s": np.asarray(pair.endpoint_skew_s),
    }
    for index, window in enumerate(pair.windows):
        arrays[f"finger_{index}_warped"] = np.asarray(window.warped.values)
        arrays[f"finger_{index}_diff"] = np.asarray(window.diff.values)
        arrays[f"finger_{index}_sdk_time_s"] = np.asarray(window.warped.sdk_time_s)
        arrays[f"finger_{index}_capture_monotonic_s"] = np.asarray(
            window.warped.capture_monotonic_s
        )
    _atomic_npz(path, **arrays)


def _save_profile(output_dir: Path, profile: ContactApproachProfile) -> None:
    arrays: dict[str, np.ndarray] = {}
    fingers: list[dict[str, Any]] = []
    for index, finger in enumerate(profile.fingers):
        arrays[f"finger_{index}_warped_reference"] = finger.warped_reference
        arrays[f"finger_{index}_diff_reference"] = finger.diff_reference
        arrays[f"finger_{index}_roi_mask"] = finger.roi_mask
        fingers.append(
            {
                "serial": finger.serial,
                "calibration_id": finger.calibration_id,
                "warped_residual_threshold": finger.warped_residual_threshold,
                "diff_residual_threshold": finger.diff_residual_threshold,
                "min_component_area_px": finger.min_component_area_px,
                "unloaded_frame_count": finger.unloaded_frame_count,
            }
        )
    _atomic_npz(output_dir / "unloaded_contact_profile.npz", **arrays)
    _atomic_json(
        output_dir / "unloaded_contact_profile.json",
        {
            "schema": CONTACT_APPROACH_SCHEMA,
            "session_id": profile.session_id,
            "runtime_window_frames": RUNTIME_WINDOW_FRAMES,
            "force6d_used_for_decision": False,
            "fingers": fingers,
        },
    )


def _collect(
    source: GF220LiveSource,
    *,
    frames: int,
    timeout_s: float,
    not_before_s: float | None = None,
) -> GF220ClassifierWindows:
    threshold = time.monotonic() if not_before_s is None else not_before_s
    return source.collect_classifier_windows(threshold, samples=frames, timeout=timeout_s)


def _read_healthy_initial_state(transport: SerialModbusRTU) -> dict[str, Any]:
    consecutive = 0
    state: dict[str, Any] | None = None
    for _ in range(8):
        state = feedback_payload(transport.read_feedback())
        healthy = state["activated"] is True and state["fault_code"] == 0
        consecutive = consecutive + 1 if healthy else 0
        if consecutive >= 2:
            return state
        time.sleep(0.025)
    fault = None if state is None else state.get("fault_code")
    raise RuntimeError(f"Hand-E lacks two healthy startup reads (last fault={fault})")


def _stop_and_confirm(
    transport: SerialModbusRTU, *, timeout_s: float
) -> tuple[dict[str, Any], int]:
    """Clear rGTO, then require two stationary feedback samples."""

    transport.stop_motion()
    deadline = time.monotonic() + timeout_s
    previous_count: int | None = None
    stationary_reads = 0
    reads = 0
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        feedback = transport.read_feedback()
        reads += 1
        last = feedback_payload(feedback)
        count = int(feedback.position_count)
        goto_cleared = int(feedback.go_to_bit) == 0
        stationary = previous_count is not None and abs(count - previous_count) <= 1
        stationary_reads = stationary_reads + 1 if goto_cleared and stationary else 0
        if stationary_reads >= 2:
            return last, reads
        previous_count = count
        time.sleep(0.025)
    detail = "no feedback" if last is None else (
        f"go_to_bit={last['go_to_bit']}, opening={last['opening_width_mm']:.3f} mm"
    )
    raise TimeoutError(f"Hand-E rGTO-clear stop was not confirmed: {detail}")


def _decision_payload(decision: TactileCloseDecision, *, window: str) -> dict[str, Any]:
    return {
        "schema": TACTILE_CLOSE_SCHEMA,
        "window": window,
        "force6d_used_for_decision": False,
        **asdict(decision),
    }


def _observe(
    *,
    source: GF220LiveSource,
    profile: ContactApproachProfile,
    memory: TactileCloseMemory,
    output_dir: Path,
    stem: str,
    timeout_s: float,
    not_before_s: float | None = None,
    persist: bool = True,
) -> tuple[TactileCloseDecision, TactileCloseMemory, str, GF220ClassifierWindows]:
    window = _collect(
        source,
        frames=RUNTIME_WINDOW_FRAMES,
        timeout_s=timeout_s,
        not_before_s=not_before_s,
    )
    decision = evaluate_tactile_close_window(window, profile, memory)
    tactile_name = f"tactile_{stem}.npz"
    decision_name = f"decision_{stem}.json"
    if persist:
        _save_window(output_dir / tactile_name, window)
        _atomic_json(
            output_dir / decision_name,
            _decision_payload(decision, window=tactile_name),
        )
    return decision, decision.next_memory, decision_name, window


def _persist_observation(
    output_dir: Path,
    *,
    stem: str,
    decision: TactileCloseDecision,
    window: GF220ClassifierWindows,
) -> str:
    tactile_name = f"tactile_{stem}.npz"
    decision_name = f"decision_{stem}.json"
    _save_window(output_dir / tactile_name, window)
    _atomic_json(
        output_dir / decision_name,
        _decision_payload(decision, window=tactile_name),
    )
    return decision_name


def _base_report(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "session_id": f"tube-continuous-close-{uuid.uuid4().hex}",
        "created_utc": datetime.now(UTC).isoformat(),
        "updated_utc": datetime.now(UTC).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "status": "running",
        "output_dir": str(output_dir),
        "target_opening_mm": float(args.minimum_opening_mm),
        "speed_register": int(args.speed),
        "force_register": int(args.force),
        "hardware_command_attempted": False,
        "hardware_commands_dispatched": False,
        "stop_command_dispatched": False,
        "monitor_windows": [],
        "confirmation_windows": [],
        "safety": {
            "single_continuous_close_command": True,
            "stop_action_register": "0x0100 (rACT=1, rGTO=0)",
            "runtime_window_frames": RUNTIME_WINDOW_FRAMES,
            "stop_on_first_pad_contact": True,
            "bilateral_confirmation_windows": CONFIRMATION_WINDOWS,
            "maximum_force_register": 20,
            "arm_motion": False,
            "automatic_opening": False,
            "automatic_extraction": False,
            "force6d_used_for_decision": False,
        },
        "error": None,
    }


def _post_stop_result(decisions: list[TactileCloseDecision]) -> tuple[str, str]:
    if len(decisions) != CONFIRMATION_WINDOWS or not all(item.valid for item in decisions):
        return "invalid_tactile_hold", "hold"
    if decisions[-1].grasp_ready:
        return "grasp_ready", "hold_grasp_no_extraction"
    if decisions[-1].needs_centering:
        return "needs_centering", "hold_for_arm_recentering"
    if any(item.active_serials for item in decisions):
        return "contact_unconfirmed_hold", "hold"
    return "contact_lost_after_stop", "hold"


def run(args: argparse.Namespace) -> int:
    output_dir = _output_dir(args)
    report = _base_report(args, output_dir)
    report_path = output_dir / "run_report.json"
    _atomic_json(report_path, report)
    if not args.execute:
        report["status"] = "dry_run_complete"
        report["final_action"] = "no_hardware_command"
        report["updated_utc"] = datetime.now(UTC).isoformat()
        _atomic_json(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path)}))
        return 0

    source = GF220LiveSource(
        args.gf220_profile,
        config=GF220LiveConfig(
            buffer_frames=40,
            warmup_frames=8,
            warp_validation_frames=2,
            max_frame_age_s=0.4,
            max_pair_skew_s=0.08,
        ),
    )
    transport: SerialModbusRTU | None = None
    motion_latched = False
    stop_confirmed = False
    try:
        source.start(timeout_s=args.calibration_timeout_s)
        calibrations = source.calibrate_unloaded(timeout_s=args.calibration_timeout_s)
        unloaded = tuple(
            _collect(
                source,
                frames=args.baseline_window_frames,
                timeout_s=args.sensor_timeout_s,
            )
            for _ in range(args.baseline_windows)
        )
        for index, window in enumerate(unloaded):
            _save_window(output_dir / f"unloaded_window_{index:02d}.npz", window)
        profile = fit_unloaded_contact_profile(
            session_id=report["session_id"],
            calibrations=calibrations,
            windows=unloaded,
            min_unique_frames=RUNTIME_WINDOW_FRAMES,
            contact_frame_fraction=0.5,
        )
        _save_profile(output_dir, profile)

        transport = SerialModbusRTU(str(args.port), args.modbus_timeout_s)
        initial = _read_healthy_initial_state(transport)
        report["initial_hande_state"] = initial
        if float(initial["opening_width_mm"]) <= args.minimum_opening_mm + 1.0:
            raise RuntimeError("Hand-E is already at or below the continuous-close target")

        memory = TactileCloseMemory()
        preflight, memory, preflight_name, _ = _observe(
            source=source,
            profile=profile,
            memory=memory,
            output_dir=output_dir,
            stem="preflight",
            timeout_s=args.sensor_timeout_s,
        )
        report["preflight_decision"] = preflight_name
        if not preflight.valid or preflight.active_serials:
            report["status"] = (
                "invalid_tactile_hold" if not preflight.valid else "preexisting_contact_hold"
            )
            report["final_action"] = "no_close_command"
        else:
            target_count, _ = _target_count("move", args.minimum_opening_mm)
            report["hardware_command_attempted"] = True
            _atomic_json(report_path, report)
            command_time = time.monotonic()
            # A lost Modbus acknowledgement can still mean the write reached the
            # gripper, so arm exception cleanup before sending the request.
            motion_latched = True
            transport.write_motion(target_count, args.speed, args.force)
            report["hardware_commands_dispatched"] = True
            report["close_command_monotonic_s"] = command_time
            _atomic_json(report_path, report)

            trigger_reason = "motion_timeout"
            trigger_decision: TactileCloseDecision | None = None
            deadline = command_time + args.motion_timeout_s
            index = 0
            deferred_monitor: list[
                tuple[str, TactileCloseDecision, GF220ClassifierWindows]
            ] = []
            while time.monotonic() < deadline:
                feedback = transport.read_feedback()
                if feedback.fault_code:
                    trigger_reason = f"hande_fault_{feedback.fault_code}"
                    break
                if feedback.object_status == 2:
                    trigger_reason = "hande_object_status_2"
                    break
                if feedback.object_status == 3 and abs(feedback.position_count - target_count) <= 2:
                    trigger_reason = "minimum_opening_reached_without_contact"
                    break

                stem = f"monitor_{index:03d}"
                decision, memory, _, window = _observe(
                    source=source,
                    profile=profile,
                    memory=memory,
                    output_dir=output_dir,
                    stem=stem,
                    timeout_s=args.sensor_timeout_s,
                    not_before_s=command_time if index == 0 else None,
                    persist=False,
                )
                deferred_monitor.append((stem, decision, window))
                index += 1
                if not decision.valid:
                    trigger_reason = "invalid_tactile_window"
                    trigger_decision = decision
                    break
                if decision.active_serials:
                    trigger_reason = "first_pad_contact"
                    trigger_decision = decision
                    break

            report["trigger"] = {
                "reason": trigger_reason,
                "decision": None if trigger_decision is None else asdict(trigger_decision),
                "latency_s": time.monotonic() - command_time,
            }
            stopped_state, stop_reads = _stop_and_confirm(
                transport, timeout_s=args.stop_timeout_s
            )
            motion_latched = False
            stop_confirmed = True
            report["stop_command_dispatched"] = True
            report["stop_receipt"] = {
                "feedback_reads": stop_reads,
                "state": stopped_state,
            }
            report["measured_stop_opening_mm"] = stopped_state["opening_width_mm"]
            for stem, decision, window in deferred_monitor:
                report["monitor_windows"].append(
                    _persist_observation(
                        output_dir,
                        stem=stem,
                        decision=decision,
                        window=window,
                    )
                )
            _atomic_json(report_path, report)

            confirmation_memory = TactileCloseMemory()
            confirmation_decisions: list[TactileCloseDecision] = []
            stop_time = time.monotonic()
            for confirm_index in range(CONFIRMATION_WINDOWS):
                decision, confirmation_memory, decision_name, _ = _observe(
                    source=source,
                    profile=profile,
                    memory=confirmation_memory,
                    output_dir=output_dir,
                    stem=f"stopped_confirm_{confirm_index:02d}",
                    timeout_s=args.sensor_timeout_s,
                    not_before_s=stop_time if confirm_index == 0 else None,
                )
                confirmation_decisions.append(decision)
                report["confirmation_windows"].append(decision_name)
            report["status"], report["final_action"] = _post_stop_result(
                confirmation_decisions
            )
            if trigger_reason == "hande_object_status_2" and report["status"] != "grasp_ready":
                report["status"] = "hande_contact_without_bilateral_tactile_confirmation"
                report["final_action"] = "hold"

        if transport is not None:
            report["final_hande_state"] = feedback_payload(transport.read_feedback())
        report["updated_utc"] = datetime.now(UTC).isoformat()
        _atomic_json(report_path, report)
    except BaseException as exc:
        if transport is not None and motion_latched:
            try:
                stopped_state, stop_reads = _stop_and_confirm(
                    transport, timeout_s=args.stop_timeout_s
                )
                stop_confirmed = True
                report["stop_command_dispatched"] = True
                report["exception_stop_receipt"] = {
                    "feedback_reads": stop_reads,
                    "state": stopped_state,
                }
            except BaseException as stop_exc:
                report["stop_error"] = {
                    "type": type(stop_exc).__name__,
                    "message": str(stop_exc),
                }
        report["status"] = "failed_hold"
        report["final_action"] = "hold"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["stop_confirmed"] = stop_confirmed
        report["updated_utc"] = datetime.now(UTC).isoformat()
        _atomic_json(report_path, report)
        return 2
    finally:
        if transport is not None:
            transport.close()
        source.close()

    print(json.dumps({"status": report["status"], "report": str(report_path)}))
    return 0 if report["status"] == "grasp_ready" else 4


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (OSError, TypeError, ValueError) as exc:
        print(f"continuous tactile close rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
