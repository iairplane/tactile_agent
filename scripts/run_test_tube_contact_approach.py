#!/usr/bin/env python3
"""Run a bounded image-tactile micro-approach toward a test-tube grasp.

The default is a hardware-free dry run that only writes the ten-step-or-less
schedule.  ``--execute`` starts both GF220 sensors, calibrates them while the
pads are operator-asserted empty, and dispatches one audited 4 mm (maximum)
absolute TCP target at a time through ``moveit_pose_goal.py``.  The first pad
contact stops all forward motion.  The tool never closes the Hand-E and never
extracts the tube.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from demo2gripper.gf220_live import (  # noqa: E402
    DEFAULT_PROFILE,
    GF220ClassifierWindows,
    GF220LiveConfig,
    GF220LiveSource,
)
from demo2gripper.pose_motion import normalize_quaternion_xyzw  # noqa: E402
from demo2gripper.test_tube_contact_approach import (  # noqa: E402
    CONTACT_APPROACH_SCHEMA,
    MAX_APPROACH_DISTANCE_M,
    MAX_APPROACH_STEP_M,
    ContactApproachMemory,
    ContactApproachProfile,
    ContactWindowDecision,
    evaluate_contact_window,
    fit_unloaded_contact_profile,
    make_approach_schedule,
)

REPORT_SCHEMA = "demo2gripper.test_tube_contact_approach_run.v1"
DEFAULT_TOOL_TRANSFORM = REPOSITORY_ROOT / "calibration/T_link8_canonical_tcp.json"
DEFAULT_MOVEIT_SCRIPT = REPOSITORY_ROOT / "scripts/moveit_pose_goal.py"
DEFAULT_ROS_RUNNER = REPOSITORY_ROOT / "scripts/run_ros_python.sh"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--target-grasp-position-m", type=float, nargs=3, required=True, metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--target-grasp-orientation-xyzw",
        type=float,
        nargs=4,
        required=True,
        metavar=("QX", "QY", "QZ", "QW"),
    )
    parser.add_argument(
        "--approach-axis-base",
        type=float,
        nargs=3,
        required=True,
        metavar=("AX", "AY", "AZ"),
        help="unit direction from the present standoff toward the grasp",
    )
    parser.add_argument("--remaining-distance-m", type=float, required=True)
    parser.add_argument("--step-m", type=float, default=MAX_APPROACH_STEP_M)
    parser.add_argument(
        "--maximum-steps",
        type=int,
        default=10,
        help="bounded forward-step budget; may deliberately stop before the grasp center",
    )
    parser.add_argument(
        "--T-link8-tcp",
        type=Path,
        default=DEFAULT_TOOL_TRANSFORM,
        help="JSON or NPY T_link8_tcp transform",
    )
    parser.add_argument("--base-frame", default="fr3_link0")
    parser.add_argument("--gf220-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--baseline-windows", type=int, default=3)
    parser.add_argument("--window-frames", type=int, default=5)
    parser.add_argument("--sensor-timeout-s", type=float, default=5.0)
    parser.add_argument("--calibration-timeout-s", type=float, default=15.0)
    parser.add_argument("--velocity-scale", type=float, default=0.02)
    parser.add_argument("--acceleration-scale", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--moveit-script", type=Path, default=DEFAULT_MOVEIT_SCRIPT)
    parser.add_argument("--ros-runner", type=Path, default=DEFAULT_ROS_RUNNER)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="start GF220 and dispatch audited MoveIt steps; absent means hardware-free dry run",
    )
    parser.add_argument(
        "--assert-pads-unloaded",
        action="store_true",
        help="assert that neither tactile pad touches anything during startup calibration",
    )
    parser.add_argument("--accept-arm-only-collision-model", action="store_true")
    parser.add_argument("--allow-generic-kernel", action="store_true")
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    position = np.asarray(args.target_grasp_position_m, dtype=float)
    axis = np.asarray(args.approach_axis_base, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("target grasp position must be a finite 3-vector")
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 1e-12:
        raise ValueError("approach axis must be a finite non-zero 3-vector")
    normalize_quaternion_xyzw(args.target_grasp_orientation_xyzw)
    if not 0.0 < args.remaining_distance_m <= MAX_APPROACH_DISTANCE_M:
        raise ValueError("--remaining-distance-m must lie in (0, 0.040]")
    if not 0.0 < args.step_m <= MAX_APPROACH_STEP_M:
        raise ValueError("--step-m must lie in (0, 0.004]")
    if isinstance(args.maximum_steps, bool) or not 1 <= args.maximum_steps <= 10:
        raise ValueError("--maximum-steps must lie in [1, 10]")
    if args.baseline_windows < 2:
        raise ValueError("--baseline-windows must be at least two")
    if args.window_frames != 5:
        raise ValueError("--window-frames must be exactly five")
    for name in ("sensor_timeout_s", "calibration_timeout_s"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("velocity_scale", "acceleration_scale"):
        value = float(getattr(args, name))
        if not 0.0 < value <= 0.03:
            raise ValueError(f"--{name.replace('_', '-')} must lie in (0, 0.03]")
    if args.execute and not args.assert_pads_unloaded:
        raise ValueError("--execute requires --assert-pads-unloaded")
    if args.execute and not args.accept_arm_only_collision_model:
        raise ValueError("--execute requires --accept-arm-only-collision-model")
    if args.execute:
        for path in (args.moveit_script, args.ros_runner, args.gf220_profile):
            if not path.is_file():
                raise ValueError(f"required execution input does not exist: {path}")
    if not args.T_link8_tcp.is_file():
        raise ValueError(f"tool transform does not exist: {args.T_link8_tcp}")


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        result = args.output_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = REPOSITORY_ROOT / "runs/live" / f"test_tube_contact_approach_{stamp}"
    if result.exists() and any(result.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output directory {result}")
    result.mkdir(parents=True, exist_ok=True)
    return result.resolve()


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


def _load_tool_transform(path: Path) -> tuple[list[float], tuple[float, float, float, float]]:
    if path.suffix.lower() == ".npy":
        transform = np.asarray(np.load(path), dtype=float)
        if transform.shape != (4, 4):
            raise ValueError("T_link8_tcp NPY must contain a 4x4 matrix")
        translation = transform[:3, 3]
        rotation = transform[:3, :3]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        translation = np.asarray(payload.get("translation_m"), dtype=float)
        rotation = np.asarray(payload.get("rotation_matrix"), dtype=float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("T_link8_tcp translation must be a finite 3-vector")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("T_link8_tcp rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6
    ):
        raise ValueError("T_link8_tcp rotation must be proper")
    quaternion = normalize_quaternion_xyzw(Rotation.from_matrix(rotation).as_quat().tolist())
    return translation.astype(float).tolist(), quaternion


def _profile_payload(profile: ContactApproachProfile, npz_name: str) -> dict[str, Any]:
    return {
        "schema": CONTACT_APPROACH_SCHEMA,
        "session_id": profile.session_id,
        "min_unique_frames": profile.min_unique_frames,
        "contact_frame_fraction": profile.contact_frame_fraction,
        "max_pair_skew_s": profile.max_pair_skew_s,
        "array_evidence": npz_name,
        "force6d_used": False,
        "fingers": [
            {
                "serial": item.serial,
                "calibration_id": item.calibration_id,
                "warped_residual_threshold": item.warped_residual_threshold,
                "diff_residual_threshold": item.diff_residual_threshold,
                "min_component_area_px": item.min_component_area_px,
                "unloaded_frame_count": item.unloaded_frame_count,
            }
            for item in profile.fingers
        ],
    }


def _save_profile(output_dir: Path, profile: ContactApproachProfile) -> None:
    arrays: dict[str, np.ndarray] = {}
    for index, finger in enumerate(profile.fingers):
        arrays[f"finger_{index}_warped_reference"] = finger.warped_reference
        arrays[f"finger_{index}_diff_reference"] = finger.diff_reference
        arrays[f"finger_{index}_roi_mask"] = finger.roi_mask
    npz = output_dir / "unloaded_contact_profile.npz"
    _atomic_npz(npz, **arrays)
    _atomic_json(
        output_dir / "unloaded_contact_profile.json",
        _profile_payload(profile, npz.name),
    )


def _save_window(
    path: Path,
    windows: GF220ClassifierWindows,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "serials": np.asarray(windows.serials),
        "endpoint_skew_s": np.asarray(windows.endpoint_skew_s),
    }
    for index, source in enumerate(windows.windows):
        arrays[f"finger_{index}_warped"] = np.asarray(source.warped.values)
        arrays[f"finger_{index}_diff"] = np.asarray(source.diff.values)
        arrays[f"finger_{index}_sdk_time_s"] = np.asarray(source.warped.sdk_time_s)
        arrays[f"finger_{index}_capture_monotonic_s"] = np.asarray(
            source.warped.capture_monotonic_s
        )
    _atomic_npz(path, **arrays)


def _collect_window(
    source: GF220LiveSource,
    *,
    frames: int,
    timeout_s: float,
) -> GF220ClassifierWindows:
    cutoff = time.monotonic()
    return source.collect_classifier_windows(cutoff, samples=frames, timeout=timeout_s)


def _move_command(
    args: argparse.Namespace,
    *,
    target_position: np.ndarray,
    target_orientation: tuple[float, float, float, float],
    link8_to_tcp_position: list[float],
    link8_to_tcp_orientation: tuple[float, float, float, float],
    output: Path,
) -> list[str]:
    command = [
        str(args.ros_runner),
        str(args.moveit_script),
        "--target-position-m",
        *(f"{value:.12g}" for value in target_position),
        "--target-orientation-xyzw",
        *(f"{value:.12g}" for value in target_orientation),
        "--target-frame",
        "configured-ee",
        "--base-frame",
        args.base_frame,
        "--link8-to-configured-ee-position-m",
        *(f"{value:.12g}" for value in link8_to_tcp_position),
        "--link8-to-configured-ee-orientation-xyzw",
        *(f"{value:.12g}" for value in link8_to_tcp_orientation),
        "--velocity-scale",
        f"{args.velocity_scale:.12g}",
        "--acceleration-scale",
        f"{args.acceleration_scale:.12g}",
        "--max-translation-m",
        "0.0045",
        "--max-rotation-rad",
        "0.05",
        "--output",
        str(output),
        "--execute",
        "--accept-arm-only-collision-model",
    ]
    if args.allow_generic_kernel:
        command.append("--allow-generic-kernel")
    return command


def _run_move(command: list[str], output: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"MoveIt microstep failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()[-800:]}"
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    if not (
        payload.get("schema") == "demo2gripper.moveit_pose_goal.v2"
        and payload.get("mode") == "execute"
        and payload.get("trajectory_audit", {}).get("accepted") is True
        and payload.get("execution_dispatched") is True
        and payload.get("execution_succeeded") is True
    ):
        raise RuntimeError("MoveIt microstep did not produce an accepted executed receipt")
    return payload


def _decision_payload(
    decision: ContactWindowDecision,
    *,
    sequence: int,
    kind: str,
    tactile_npz: str,
) -> dict[str, Any]:
    payload = asdict(decision)
    payload.update(
        {
            "schema": CONTACT_APPROACH_SCHEMA,
            "sequence": sequence,
            "window_kind": kind,
            "tactile_npz": tactile_npz,
            "force6d_used": False,
        }
    )
    return payload


def _base_report(
    args: argparse.Namespace,
    output_dir: Path,
    schedule: tuple[np.ndarray, ...],
    orientation: tuple[float, float, float, float],
    link8_position: list[float],
    link8_orientation: tuple[float, float, float, float],
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "updated_utc": datetime.now(UTC).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "status": "running",
        "output_dir": str(output_dir),
        "session_id": f"tube-contact-{uuid.uuid4().hex}",
        "target_grasp_position_base_m": list(map(float, args.target_grasp_position_m)),
        "target_grasp_orientation_xyzw": list(orientation),
        "approach_axis_base": (
            np.asarray(args.approach_axis_base, dtype=float)
            / np.linalg.norm(np.asarray(args.approach_axis_base, dtype=float))
        ).tolist(),
        "remaining_distance_m": float(args.remaining_distance_m),
        "step_cap_m": float(args.step_m),
        "scheduled_target_positions_base_m": [item.tolist() for item in schedule],
        "scheduled_step_count": len(schedule),
        "configured_maximum_steps": int(args.maximum_steps),
        "scheduled_terminal_clearance_to_grasp_m": float(
            np.linalg.norm(
                np.asarray(args.target_grasp_position_m, dtype=float) - schedule[-1]
            )
        ),
        "T_link8_tcp": {
            "source": str(args.T_link8_tcp.resolve()),
            "translation_m": link8_position,
            "orientation_xyzw": list(link8_orientation),
        },
        "safety": {
            "maximum_total_approach_m": MAX_APPROACH_DISTANCE_M,
            "maximum_step_m": MAX_APPROACH_STEP_M,
            "maximum_steps": int(args.maximum_steps),
            "stop_on_first_pad_contact": True,
            "grasp_ready_requires_bilateral_windows": 2,
            "automatic_gripper_close": False,
            "automatic_extraction": False,
            "force6d_used": False,
            "operator_asserted_pads_unloaded": bool(args.assert_pads_unloaded),
        },
        "steps": [],
        "hardware_commands_dispatched": False,
        "final_action": None,
        "error": None,
    }


def run(args: argparse.Namespace) -> int:
    output_dir = _output_dir(args)
    orientation = normalize_quaternion_xyzw(args.target_grasp_orientation_xyzw)
    schedule = make_approach_schedule(
        np.asarray(args.target_grasp_position_m, dtype=float),
        np.asarray(args.approach_axis_base, dtype=float),
        args.remaining_distance_m,
        step_m=args.step_m,
        maximum_steps=args.maximum_steps,
    )
    link8_position, link8_orientation = _load_tool_transform(args.T_link8_tcp)
    report = _base_report(
        args,
        output_dir,
        schedule,
        orientation,
        link8_position,
        link8_orientation,
    )
    report_path = output_dir / "run_report.json"
    _atomic_json(report_path, report)
    if not args.execute:
        report["status"] = "dry_run_complete"
        report["final_action"] = "no_hardware_command"
        report["updated_utc"] = datetime.now(UTC).isoformat()
        _atomic_json(report_path, report)
        print(json.dumps({"report": str(report_path), "status": report["status"]}, indent=2))
        return 0

    live_config = GF220LiveConfig(
        buffer_frames=32,
        warmup_frames=8,
        warp_validation_frames=2,
        max_frame_age_s=0.75,
        max_pair_skew_s=0.08,
    )
    try:
        with GF220LiveSource(args.gf220_profile, config=live_config) as source:
            calibrations = source.calibrate_unloaded(timeout_s=args.calibration_timeout_s)
            unloaded_windows = tuple(
                _collect_window(
                    source,
                    frames=args.window_frames,
                    timeout_s=args.sensor_timeout_s,
                )
                for _ in range(args.baseline_windows)
            )
            for index, window in enumerate(unloaded_windows):
                _save_window(output_dir / f"unloaded_window_{index:02d}.npz", window)
            profile = fit_unloaded_contact_profile(
                session_id=report["session_id"],
                calibrations=calibrations,
                windows=unloaded_windows,
                min_unique_frames=args.window_frames,
            )
            _save_profile(output_dir, profile)
            report["unloaded_calibration"] = _profile_payload(
                profile, "unloaded_contact_profile.npz"
            )
            _atomic_json(report_path, report)

            memory = ContactApproachMemory()
            contact_sequence = 0
            preflight = _collect_window(
                source,
                frames=args.window_frames,
                timeout_s=args.sensor_timeout_s,
            )
            preflight_tactile_path = output_dir / "tactile_preflight.npz"
            _save_window(preflight_tactile_path, preflight)
            preflight_decision = evaluate_contact_window(preflight, profile, memory)
            memory = preflight_decision.next_memory
            contact_sequence += 1
            preflight_receipt = output_dir / "contact_preflight.json"
            _atomic_json(
                preflight_receipt,
                _decision_payload(
                    preflight_decision,
                    sequence=contact_sequence,
                    kind="stationary_before_first_motion",
                    tactile_npz=preflight_tactile_path.name,
                ),
            )
            report["preflight_contact_receipt"] = preflight_receipt.name
            proceed = preflight_decision.valid and not preflight_decision.stop_forward
            if not preflight_decision.valid:
                report["status"] = "held_invalid_tactile_evidence"
                report["final_action"] = "hold_invalid"
            elif preflight_decision.stop_forward:
                confirmation = _collect_window(
                    source,
                    frames=args.window_frames,
                    timeout_s=args.sensor_timeout_s,
                )
                confirmation_path = output_dir / "tactile_contact_confirmation.npz"
                _save_window(confirmation_path, confirmation)
                confirmed = evaluate_contact_window(confirmation, profile, memory)
                memory = confirmed.next_memory
                contact_sequence += 1
                confirmed_path = output_dir / "contact_confirmation.json"
                _atomic_json(
                    confirmed_path,
                    _decision_payload(
                        confirmed,
                        sequence=contact_sequence,
                        kind="stationary_confirmation",
                        tactile_npz=confirmation_path.name,
                    ),
                )
                report["confirmation_receipt"] = confirmed_path.name
                if not confirmed.valid:
                    report["status"] = "held_invalid_tactile_evidence"
                    report["final_action"] = "hold_invalid"
                else:
                    report["final_action"] = (
                        "grasp_ready" if confirmed.grasp_ready else "one_pad_contact_hold"
                    )
                    report["status"] = report["final_action"]
            _atomic_json(report_path, report)

            for step_index, target in (
                enumerate(schedule, start=1) if proceed else ()
            ):
                move_path = output_dir / f"moveit_step_{step_index:02d}.json"
                move_staging_path = output_dir / (
                    f".{move_path.name}.{os.getpid()}.staging"
                )
                command = _move_command(
                    args,
                    target_position=target,
                    target_orientation=orientation,
                    link8_to_tcp_position=link8_position,
                    link8_to_tcp_orientation=link8_orientation,
                    output=move_staging_path,
                )
                report["motion_command_attempted"] = True
                _atomic_json(report_path, report)
                move_receipt = _run_move(command, move_staging_path)
                move_staging_path.replace(move_path)
                report["hardware_commands_dispatched"] = True
                tactile = _collect_window(
                    source,
                    frames=args.window_frames,
                    timeout_s=args.sensor_timeout_s,
                )
                tactile_path = output_dir / f"tactile_step_{step_index:02d}.npz"
                _save_window(tactile_path, tactile)
                decision = evaluate_contact_window(tactile, profile, memory)
                memory = decision.next_memory
                contact_sequence += 1
                decision_path = output_dir / f"contact_step_{step_index:02d}.json"
                _atomic_json(
                    decision_path,
                    _decision_payload(
                        decision,
                        sequence=contact_sequence,
                        kind="post_motion",
                        tactile_npz=tactile_path.name,
                    ),
                )
                report["steps"].append(
                    {
                        "step": step_index,
                        "target_position_base_m": target.tolist(),
                        "moveit_receipt": move_path.name,
                        "measured_end_configured_ee_position_base_m": move_receipt.get(
                            "measured_end_configured_ee_position_base_m"
                        ),
                        "contact_receipt": decision_path.name,
                        "contact_action": decision.action,
                    }
                )
                report["updated_utc"] = datetime.now(UTC).isoformat()
                _atomic_json(report_path, report)
                if not decision.valid:
                    report["status"] = "held_invalid_tactile_evidence"
                    report["final_action"] = "hold_invalid"
                    break
                if decision.stop_forward:
                    confirmation = _collect_window(
                        source,
                        frames=args.window_frames,
                        timeout_s=args.sensor_timeout_s,
                    )
                    confirmation_path = output_dir / "tactile_contact_confirmation.npz"
                    _save_window(confirmation_path, confirmation)
                    confirmed = evaluate_contact_window(confirmation, profile, memory)
                    memory = confirmed.next_memory
                    contact_sequence += 1
                    confirmed_path = output_dir / "contact_confirmation.json"
                    _atomic_json(
                        confirmed_path,
                        _decision_payload(
                            confirmed,
                            sequence=contact_sequence,
                            kind="stationary_confirmation",
                            tactile_npz=confirmation_path.name,
                        ),
                    )
                    report["confirmation_receipt"] = confirmed_path.name
                    if not confirmed.valid:
                        report["status"] = "held_invalid_tactile_evidence"
                        report["final_action"] = "hold_invalid"
                    else:
                        report["final_action"] = (
                            "grasp_ready"
                            if confirmed.grasp_ready
                            else "one_pad_contact_hold"
                        )
                        report["status"] = report["final_action"]
                    break
            else:
                if proceed:
                    report["status"] = "approach_budget_exhausted_no_contact"
                    report["final_action"] = "no_contact_limit_hold"
    except Exception as exc:
        report["status"] = "failed_closed"
        report["final_action"] = "hold"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["updated_utc"] = datetime.now(UTC).isoformat()
        _atomic_json(report_path, report)
        print(f"contact approach failed closed: {exc}", file=sys.stderr)
        return 2

    report["updated_utc"] = datetime.now(UTC).isoformat()
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": report["status"],
                "final_action": report["final_action"],
                "steps_executed": len(report["steps"]),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (OSError, TypeError, ValueError) as exc:
        print(f"contact approach rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
