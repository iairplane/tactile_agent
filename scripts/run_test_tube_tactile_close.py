#!/usr/bin/env python3
"""Close Hand-E from 25 mm in 2 mm steps under dual GF220 supervision.

The default is a hardware-free dry run. Live mode requires ``--execute``.
Two consecutive bilateral image-contact windows produce ``grasp_ready``. On
the first single-pad contact the runner may close exactly one additional
millimetre; unless that yields bilateral contact it stops and reports that the
tube needs centering. It never commands the arm, opens the fingers, or
extracts the tube.
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
    _wait_motion,
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
    make_staged_close_schedule,
)

REPORT_SCHEMA = "demo2gripper.test_tube_tactile_close_run.v2"
DEFAULT_HAND_E_PORT = Path(
    "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AQ00O97B-if00-port0"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--port", type=Path, default=DEFAULT_HAND_E_PORT)
    parser.add_argument("--initial-opening-mm", type=float, default=25.0)
    parser.add_argument("--minimum-opening-mm", type=float, default=10.0)
    parser.add_argument(
        "--opening-schedule-mm",
        type=float,
        nargs="+",
        help=(
            "optional strictly descending absolute openings; targets at or above "
            "the measured starting opening are skipped so the gripper never reopens"
        ),
    )
    parser.add_argument("--speed", type=int, default=30)
    parser.add_argument("--force", type=int, default=8)
    parser.add_argument("--motion-timeout-s", type=float, default=4.0)
    parser.add_argument("--modbus-timeout-s", type=float, default=0.25)
    parser.add_argument("--gf220-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--baseline-windows", type=int, default=3)
    parser.add_argument("--window-frames", type=int, default=5)
    parser.add_argument("--sensor-timeout-s", type=float, default=5.0)
    parser.add_argument("--calibration-timeout-s", type=float, default=15.0)
    parser.add_argument("--settle-s", type=float, default=0.08)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--assert-pads-unloaded", action="store_true")
    parser.add_argument(
        "--assert-tube-between-open-fingers",
        "--assert-tube-between-fingers",
        dest="assert_tube_between_fingers",
        action="store_true",
    )
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    make_staged_close_schedule(
        args.initial_opening_mm, minimum_opening_mm=args.minimum_opening_mm
    )
    if args.opening_schedule_mm is not None:
        _validate_opening_schedule(
            tuple(args.opening_schedule_mm),
            minimum_opening_mm=args.minimum_opening_mm,
        )
    if not 0 <= args.speed <= min(MAX_SPEED_REGISTER, 50):
        raise ValueError("--speed must lie in [0, 50]")
    if not 0 <= args.force <= min(MAX_FORCE_REGISTER, 40):
        raise ValueError("--force must lie in [0, 40]")
    if args.baseline_windows < 2:
        raise ValueError("--baseline-windows must be at least two")
    if args.window_frames != 5:
        raise ValueError("--window-frames must be exactly five")
    for name in (
        "motion_timeout_s",
        "modbus_timeout_s",
        "sensor_timeout_s",
        "calibration_timeout_s",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.settle_s) or not 0.0 <= args.settle_s <= 1.0:
        raise ValueError("--settle-s must lie in [0, 1]")
    if args.execute:
        if not args.assert_pads_unloaded or not args.assert_tube_between_fingers:
            raise ValueError(
                "--execute requires --assert-pads-unloaded and "
                "--assert-tube-between-open-fingers"
            )
        if not args.port.is_absolute() or not args.port.exists():
            raise ValueError(f"Hand-E port is not an existing absolute path: {args.port}")
        if not args.gf220_profile.is_file():
            raise ValueError(f"GF220 profile does not exist: {args.gf220_profile}")


def _validate_opening_schedule(
    schedule_mm: tuple[float, ...], *, minimum_opening_mm: float
) -> tuple[float, ...]:
    values = tuple(float(value) for value in schedule_mm)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("--opening-schedule-mm must contain finite values")
    if any(not minimum_opening_mm <= value <= 50.0 for value in values):
        raise ValueError(
            "--opening-schedule-mm targets must lie between the configured minimum "
            "and 50 mm"
        )
    if any(current <= following for current, following in zip(values, values[1:])):
        raise ValueError("--opening-schedule-mm must be strictly descending")
    return values


def _remaining_schedule(
    schedule_mm: tuple[float, ...], current_opening_mm: float
) -> tuple[float, ...]:
    """Drop targets that would reopen an already narrower physical gripper."""

    current = float(current_opening_mm)
    if not math.isfinite(current):
        raise ValueError("current opening must be finite")
    values = tuple(float(value) for value in schedule_mm)
    if any(
        current_target <= next_target
        for current_target, next_target in zip(values, values[1:])
    ):
        raise ValueError("opening schedule must be strictly descending")
    return tuple(target for target in values if target < current - 1e-6)


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = REPOSITORY_ROOT / "runs/live" / f"test_tube_tactile_close_{stamp}"
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
            "force6d_used_for_decision": False,
            "fingers": fingers,
        },
    )


def _collect(
    source: GF220LiveSource, *, frames: int, timeout_s: float
) -> GF220ClassifierWindows:
    return source.collect_classifier_windows(
        time.monotonic(), samples=frames, timeout=timeout_s
    )


def _decision_payload(
    decision: TactileCloseDecision, *, window: str
) -> dict[str, Any]:
    return {
        "schema": TACTILE_CLOSE_SCHEMA,
        "window": window,
        "force6d_used_for_decision": False,
        **asdict(decision),
    }


def _command_opening(
    transport: SerialModbusRTU, args: argparse.Namespace, opening_mm: float
) -> dict[str, Any]:
    target_count, _ = _target_count("move", opening_mm)
    transport.write_motion(target_count, args.speed, args.force)
    feedback = _wait_motion(
        transport,
        command="move",
        target_count=target_count,
        timeout_s=args.motion_timeout_s,
    )
    state = feedback_payload(feedback, command="move")
    if state["fault_code"] != 0:
        raise RuntimeError(f"Hand-E fault {state['fault_code']}")
    if args.settle_s:
        time.sleep(args.settle_s)
    return state


def _read_stable_initial_state(
    transport: SerialModbusRTU,
    *,
    attempts: int = 6,
    interval_s: float = 0.05,
) -> dict[str, Any]:
    """Require two consecutive healthy reads before enabling close motion.

    This Hand-E/USB-RS485 setup has repeatedly emitted one transient gFLT=9
    on the first transaction after both GF220 streams start.  The next read is
    healthy.  We do not ignore the value: motion remains disabled until two
    fresh consecutive reads report an activated, fault-free gripper.
    """

    healthy_reads = 0
    last_state: dict[str, Any] | None = None
    for _ in range(attempts):
        last_state = feedback_payload(transport.read_feedback())
        healthy = last_state["activated"] is True and last_state["fault_code"] == 0
        healthy_reads = healthy_reads + 1 if healthy else 0
        if healthy_reads >= 2:
            return last_state
        time.sleep(interval_s)
    fault = None if last_state is None else last_state.get("fault_code")
    raise RuntimeError(f"Hand-E did not produce two healthy startup reads (last fault={fault})")


def _observe(
    *,
    source: GF220LiveSource,
    profile: ContactApproachProfile,
    memory: TactileCloseMemory,
    args: argparse.Namespace,
    output_dir: Path,
    stem: str,
) -> tuple[TactileCloseDecision, TactileCloseMemory, str]:
    window = _collect(source, frames=args.window_frames, timeout_s=args.sensor_timeout_s)
    tactile_name = f"tactile_{stem}.npz"
    _save_window(output_dir / tactile_name, window)
    decision = evaluate_tactile_close_window(window, profile, memory)
    decision_name = f"decision_{stem}.json"
    _atomic_json(
        output_dir / decision_name,
        _decision_payload(decision, window=tactile_name),
    )
    return decision, decision.next_memory, decision_name


def _base_report(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    dry_schedule = (
        make_staged_close_schedule(
            args.initial_opening_mm, minimum_opening_mm=args.minimum_opening_mm
        )
        if args.opening_schedule_mm is None
        else _remaining_schedule(
            tuple(args.opening_schedule_mm), args.initial_opening_mm
        )
    )
    return {
        "schema": REPORT_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "updated_utc": datetime.now(UTC).isoformat(),
        "session_id": f"tube-close-{uuid.uuid4().hex}",
        "mode": "execute" if args.execute else "dry_run",
        "status": "running",
        "output_dir": str(output_dir),
        "nominal_initial_opening_mm": float(args.initial_opening_mm),
        "minimum_opening_mm": float(args.minimum_opening_mm),
        "configured_opening_schedule_mm": list(dry_schedule),
        "speed_register": args.speed,
        "force_register": args.force,
        "steps": [],
        "hardware_command_attempted": False,
        "hardware_commands_dispatched": False,
        "safety": {
            "normal_close_step_max_mm": 2.0,
            "single_pad_supplement_max_mm": 1.0,
            "single_pad_supplement_count_max": 1,
            "bilateral_confirmation_windows": 2,
            "arm_motion": False,
            "automatic_extraction": False,
            "automatic_opening": False,
            "force6d_used_for_decision": False,
        },
        "error": None,
    }


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
            buffer_frames=32,
            warmup_frames=8,
            warp_validation_frames=2,
            max_frame_age_s=0.75,
            max_pair_skew_s=0.08,
        ),
    )
    transport: SerialModbusRTU | None = None
    success = False
    try:
        source.start(timeout_s=args.calibration_timeout_s)
        calibrations = source.calibrate_unloaded(timeout_s=args.calibration_timeout_s)
        unloaded = tuple(
            _collect(source, frames=args.window_frames, timeout_s=args.sensor_timeout_s)
            for _ in range(args.baseline_windows)
        )
        for index, window in enumerate(unloaded):
            _save_window(output_dir / f"unloaded_window_{index:02d}.npz", window)
        profile = fit_unloaded_contact_profile(
            session_id=report["session_id"],
            calibrations=calibrations,
            windows=unloaded,
            min_unique_frames=args.window_frames,
        )
        _save_profile(output_dir, profile)

        transport = SerialModbusRTU(str(args.port), args.modbus_timeout_s)
        initial_state = _read_stable_initial_state(transport)
        report["initial_hande_state"] = initial_state
        initial_opening = float(initial_state["opening_width_mm"])
        schedule = (
            make_staged_close_schedule(
                initial_opening, minimum_opening_mm=args.minimum_opening_mm
            )
            if args.opening_schedule_mm is None
            else _remaining_schedule(
                tuple(args.opening_schedule_mm), initial_opening
            )
        )
        report["active_opening_schedule_mm"] = list(schedule)

        memory = TactileCloseMemory()
        preflight, memory, preflight_name = _observe(
            source=source,
            profile=profile,
            memory=memory,
            args=args,
            output_dir=output_dir,
            stem="preflight",
        )
        report["preflight_decision"] = preflight_name
        if not preflight.valid:
            report["status"] = "invalid_tactile_hold"
            report["final_action"] = "hold"
        elif preflight.active_serials:
            report["status"] = "preexisting_contact_hold"
            report["final_action"] = "hold"
        else:
            for index, target_opening in enumerate(schedule, start=1):
                report["hardware_command_attempted"] = True
                _atomic_json(report_path, report)
                state = _command_opening(transport, args, target_opening)
                report["hardware_commands_dispatched"] = True
                hande_name = f"hande_close_{index:02d}.json"
                _atomic_json(output_dir / hande_name, state)
                decision, memory, decision_name = _observe(
                    source=source,
                    profile=profile,
                    memory=memory,
                    args=args,
                    output_dir=output_dir,
                    stem=f"close_{index:02d}",
                )
                step: dict[str, Any] = {
                    "step": index,
                    "target_opening_mm": target_opening,
                    "measured_opening_mm": state["opening_width_mm"],
                    "hande_receipt": hande_name,
                    "decision": decision_name,
                    "active_serials": list(decision.active_serials),
                    "supplemental_close": None,
                }

                if decision.valid and len(decision.active_serials) == 1:
                    measured = float(state["opening_width_mm"])
                    supplement_target = max(args.minimum_opening_mm, measured - 1.0)
                    if measured - supplement_target > 1e-6:
                        _atomic_json(report_path, report)
                        supplement_state = _command_opening(
                            transport, args, supplement_target
                        )
                        supplement_hande = f"hande_close_{index:02d}_single_pad_probe.json"
                        _atomic_json(output_dir / supplement_hande, supplement_state)
                        decision, memory, supplement_decision = _observe(
                            source=source,
                            profile=profile,
                            memory=memory,
                            args=args,
                            output_dir=output_dir,
                            stem=f"close_{index:02d}_single_pad_probe",
                        )
                        step["supplemental_close"] = {
                            "maximum_delta_mm": 1.0,
                            "target_opening_mm": supplement_target,
                            "measured_opening_mm": supplement_state["opening_width_mm"],
                            "hande_receipt": supplement_hande,
                            "decision": supplement_decision,
                            "active_serials": list(decision.active_serials),
                        }
                    if not decision.bilateral_contact:
                        report["steps"].append(step)
                        report["status"] = "needs_centering"
                        report["final_action"] = "one_pad_probe_failed_hold"
                        break

                if decision.valid and decision.bilateral_contact and not decision.grasp_ready:
                    decision, memory, confirm_name = _observe(
                        source=source,
                        profile=profile,
                        memory=memory,
                        args=args,
                        output_dir=output_dir,
                        stem=f"close_{index:02d}_bilateral_confirm",
                    )
                    step["bilateral_confirmation"] = confirm_name
                report["steps"].append(step)
                report["updated_utc"] = datetime.now(UTC).isoformat()
                _atomic_json(report_path, report)

                if not decision.valid:
                    report["status"] = "invalid_tactile_hold"
                    report["final_action"] = "hold"
                    break
                if decision.grasp_ready:
                    report["status"] = "grasp_ready"
                    report["final_action"] = "hold_grasp_no_extraction"
                    success = True
                    break
                if decision.active_serials:
                    report["status"] = "contact_unconfirmed_hold"
                    report["final_action"] = "hold"
                    break
                if state["object_status"]["grasped"]:
                    report["status"] = "hande_contact_without_tactile_confirmation"
                    report["final_action"] = "hold"
                    break
            else:
                report["status"] = "minimum_opening_reached_without_bilateral_contact"
                report["final_action"] = "hold"
        report["final_hande_state"] = feedback_payload(transport.read_feedback())
        report["updated_utc"] = datetime.now(UTC).isoformat()
        _atomic_json(report_path, report)
    except BaseException as exc:
        report["status"] = "failed_hold"
        report["final_action"] = "hold"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["updated_utc"] = datetime.now(UTC).isoformat()
        _atomic_json(report_path, report)
        return 2
    finally:
        if transport is not None:
            transport.close()
        source.close()

    print(json.dumps({"status": report["status"], "report": str(report_path)}))
    return 0 if success else 4


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (OSError, TypeError, ValueError) as exc:
        print(f"tactile close rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
