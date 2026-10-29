#!/usr/bin/env python3
"""One continuous base-Z descent followed by one tactile-stopped Hand-E close.

Uses an unloaded baseline from the presently open, untouched pads. No camera,
pose service, extraction, automatic recentering, or additional arm move is used.
Only two consecutive stopped windows with contact on both WARPED pads qualify
the grasp; Hand-E object status or one-pad contact never promotes success.
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

from demo2gripper.test_tube_live_io import (  # noqa: E402
    HandEModbusRuntime,
    RosMoveItServoRuntime,
    _load_artifact_transform,
)


def descent_target(current: np.ndarray, down_mm: float) -> np.ndarray:
    if not math.isfinite(down_mm) or not 0.0 < down_mm <= 25.0:
        raise ValueError("--down-mm must be in (0, 25]")
    target = np.asarray(current, dtype=float).copy()
    if target.shape != (4, 4) or not np.isfinite(target).all():
        raise ValueError("measured TCP must be a finite 4x4 transform")
    target[2, 3] -= down_mm / 1000.0
    return target


def stopped_contact_kind(decisions) -> str:
    if len(decisions) != 2 or not all(item.valid for item in decisions):
        return "invalid_tactile_hold"
    if (
        all(len(item.active_serials) == 2 for item in decisions)
        and decisions[-1].grasp_ready
    ):
        return "grasp_ready"
    if any(item.active_serials for item in decisions):
        return "bilateral_contact_not_confirmed"
    return "empty"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--down-mm", type=float, default=20.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-arm-only-collision-model", action="store_true")
    parser.add_argument("--accept-provisional-camera-extrinsic", action="store_true")
    args = parser.parse_args(argv)
    descent_target(np.eye(4), args.down_mm)
    if args.execute and not args.accept_arm_only_collision_model:
        parser.error("--execute requires --accept-arm-only-collision-model")
    return args


def run(args) -> int:
    config_path = args.config.resolve()
    config_bytes = config_path.read_bytes()
    live = json.loads(config_bytes)["live_adapter"]
    robot = copy.deepcopy(live["robot"])
    tactile = live["tactile"]
    # Never raise a configured limit for this probe.
    for key, ceiling in (("max_linear_m_s", .015), ("max_angular_rad_s", .15)):
        value = float(robot["servo"][key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid robot.servo.{key}")
        robot["servo"][key] = min(value, ceiling)
    tcp = _load_artifact_transform(robot, "tcp")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite a non-empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "demo2gripper.test_tube_fast_probe.v1",
        "session_id": f"fast-probe-{uuid.uuid4().hex}",
        "config": str(config_path), "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "mode": "execute" if args.execute else "dry_run", "status": "prepared",
        "down_mm": args.down_mm, "servo": robot["servo"], "T_link8_tcp": tcp.tolist(),
        "camera_used": False, "force6d_used_for_decision": False,
        "automatic_extraction": False, "close_target_m": .010,
        "grasp_gate": "two_consecutive_stopped_bilateral_WARPED_windows",
        "speed_register": 45, "force_register": 8, "observations": [],
        "arm_only_collision_model_accepted": args.accept_arm_only_collision_model,
    }
    report_path = output / "run_report.json"
    contact._atomic_json(report_path, report)
    if not args.execute:
        report["status"] = "dry_run_complete"
        contact._atomic_json(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path)}))
        return 0

    arm = gripper = source = transport = None
    deferred = []
    handlers = {}

    def interrupted(signum, frame):
        raise InterruptedError(f"received signal {signum}")

    def healthy_arm():
        state = arm.state()
        if state.fault:
            raise RuntimeError(f"arm fault: {state.fault}")
        return state

    def observe(stem, memory):
        before = healthy_arm()
        feedback = gripper.feedback()
        if not feedback.fault_free:
            raise RuntimeError("Hand-E fault during probe")
        decision, memory, _, window = contact._observe(
            source=source, profile=profile, memory=memory, output_dir=output,
            stem=stem, timeout_s=.75, persist=False,
        )
        state = healthy_arm()
        deferred.append((stem, decision, window))
        report["observations"].append({
            "stem": stem, "observed_at_monotonic_s": time.monotonic(),
            "T_base_tcp_before": before.T_base_tcp.tolist(),
            "T_base_tcp": state.T_base_tcp.tolist(), "motion_complete": state.motion_complete,
            "gripper": asdict(feedback), "tactile": asdict(decision),
        })
        if not decision.valid:
            raise RuntimeError("invalid or stale tactile window")
        return decision, memory, state, feedback

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, interrupted)
        transport = contact.SerialModbusRTU(
            tactile["hande_port"], float(tactile.get("modbus_timeout_s", .12))
        )
        initial = contact._read_healthy_initial_state(transport)
        gripper = HandEModbusRuntime(transport)
        initial_feedback = gripper.feedback()
        if (
            initial_feedback.width_m < .048 or initial_feedback.moving
            or not initial_feedback.fault_free or initial_feedback.object_detected
        ):
            raise RuntimeError("unloaded baseline requires stationary open jaws >=48 mm")
        report["initial_hande_state"] = initial
        report["baseline_basis"] = "operator-confirmed untouched pads with open jaws >=48 mm"
        profile_path = Path(tactile["gf220_profile"])
        if not profile_path.is_absolute():
            profile_path = ROOT / profile_path
        source = contact.GF220LiveSource(profile_path, config=contact.GF220LiveConfig(
            buffer_frames=40, warmup_frames=8, warp_validation_frames=2,
            max_frame_age_s=.4, max_pair_skew_s=.08,
        ))
        source.start(timeout_s=15.0)
        calibrations = source.calibrate_unloaded(timeout_s=15.0)
        unloaded = tuple(contact._collect(source, frames=5, timeout_s=1.0) for _ in range(3))
        profile = contact.fit_unloaded_contact_profile(
            session_id=report["session_id"], calibrations=calibrations, windows=unloaded,
            min_unique_frames=2, contact_frame_fraction=.5,
        )
        contact._save_profile(output, profile)
        for index, window in enumerate(unloaded):
            contact._save_window(output / f"unloaded_window_{index:02d}.npz", window)
        arm = RosMoveItServoRuntime(config=robot, T_link8_tcp=tcp, output_dir=output)
        # SDK/ROS startup can exceed the Hand-E communication watchdog. Resume
        # healthy status polling before entering the fast contact loop.
        contact._read_healthy_initial_state(transport)
        memory = contact.TactileCloseMemory()
        decision, memory, start, feedback = observe("preflight", memory)
        if decision.active_serials or feedback.width_m < .048 or feedback.moving:
            raise RuntimeError("probe requires initially untouched pads and open stationary jaws")
        if not start.motion_complete:
            raise RuntimeError("probe requires initially stationary arm")
        target = descent_target(start.T_base_tcp, args.down_mm)
        report["start_monotonic_s"] = time.monotonic()
        report["T_base_tcp_start"] = start.T_base_tcp.tolist()
        report["T_base_tcp_descent_target"] = target.tolist()
        report["status"] = "descending"
        contact._atomic_json(report_path, report)
        report["descent_receipt"] = dict(arm.servo_target(target=target, action_id="descent"))
        deadline = time.monotonic() + 12.0
        trigger = None
        index = 0
        while time.monotonic() < deadline:
            decision, memory, state, feedback = observe(f"descent_{index:03d}", memory)
            index += 1
            if decision.active_serials or feedback.object_detected:
                trigger = "contact_during_descent"
                break
            distance = float(np.linalg.norm(state.T_base_tcp[:3, 3] - target[:3, 3]))
            angle = math.degrees(Rotation.from_matrix(
                state.T_base_tcp[:3, :3] @ target[:3, :3].T
            ).magnitude())
            if distance <= .0025 and angle <= 2.0 and state.motion_complete:
                trigger = "descent_target_reached"
                break
        arm.stop(reason=trigger or "descent timeout")
        if trigger is None:
            raise TimeoutError("continuous descent did not reach target in 12 s")
        report["descent_trigger"] = trigger
        if trigger == "descent_target_reached":
            report["status"] = "closing"
            contact._atomic_json(report_path, report)
            report["close_receipt"] = dict(gripper.move_width(
                width_m=.010, speed=45, force=8, action_id="close_once"
            ))
            deadline = time.monotonic() + 4.0
            index = 0
            trigger = "close_timeout"
            while time.monotonic() < deadline:
                decision, memory, _, feedback = observe(f"close_{index:03d}", memory)
                index += 1
                if decision.active_serials or feedback.object_detected:
                    trigger = "first_pad_contact" if decision.active_serials else "hande_contact"
                    break
                if not feedback.moving and feedback.width_m <= .0105:
                    trigger = "minimum_opening_reached"
                    break
            report["close_trigger"] = trigger
        stopped, reads = contact._stop_and_confirm(transport, timeout_s=1.0)
        report["stopped_hande"] = stopped
        report["stop_feedback_reads"] = reads
        confirmations = []
        memory = contact.TactileCloseMemory()
        for index in range(2):
            decision, memory, _, _ = observe(f"stopped_{index}", memory)
            confirmations.append(decision)
        outcome = stopped_contact_kind(confirmations)
        if outcome == "empty" and gripper.feedback().object_detected:
            outcome = "hande_contact_without_tactile_hold"
        report["contact_outcome"] = outcome
        if outcome == "empty":
            report["open_receipt"] = dict(gripper.move_width(
                width_m=.0495, speed=45, force=8, action_id="reopen_empty"
            ))
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline:
                healthy_arm()
                feedback = gripper.feedback()
                if not feedback.fault_free:
                    raise RuntimeError("Hand-E fault while reopening empty grasp")
                if feedback.width_m >= .048 and not feedback.moving:
                    break
                time.sleep(.025)
            else:
                raise TimeoutError("empty grasp reopening not confirmed")
            report["status"] = "empty_reopened"
        else:
            report["status"] = outcome
        report["T_base_tcp_final"] = healthy_arm().T_base_tcp.tolist()
        report["final_gripper"] = asdict(gripper.feedback())
    except BaseException as error:
        report["status"] = "failed_hold"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        # Stop before file compression or resource shutdown, including signals.
        for label, resource in (("arm", arm), ("gripper", gripper)):
            if resource is not None:
                try:
                    report[f"final_{label}_stop"] = dict(resource.stop(reason="probe finished"))
                except BaseException as error:
                    report[f"{label}_stop_error"] = str(error)
                    report["status"] = "failed_hold"
        for stem, decision, window in deferred:
            contact._persist_observation(output, stem=stem, decision=decision, window=window)
        for resource in (arm, gripper, source):
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
    if report["status"] == "failed_hold":
        return 2
    return 0 if report["status"] == "grasp_ready" else 4


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
