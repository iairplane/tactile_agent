#!/usr/bin/env python3
"""Run the reviewed paper-cup tactile demo as one continuous hard-coded take.

The sequence is deliberately task-specific.  It records an unloaded tactile
baseline, performs one lower no-lift probe, unloads and returns to the reviewed
grasp pose, closes again, then lifts vertically.  The raw RGB-D/tactile recorder
is controlled over localhost and all action boundaries use the same host
monotonic clock saved by that recorder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "runs/live/cup_tactile_demo_20261023_08"
ACTION_LOG = RUN_DIR / "action_log.json"
HAND_E_PORT = (
    "/dev/serial/by-id/"
    "usb-FTDI_FT232R_USB_UART_AQ00O97B-if00-port0"
)
HAND_E_PYTHON = Path("/home/vitai/miniconda3/envs/fhh/bin/python")
LEROBOT_SRC = Path("/home/vitai/fhh/lerobot/src")

# The current stack sits about 4 mm below the earlier Run 02 setup.  Put the
# reviewed close at the current rim height and the scripted wrong probe 5 mm
# deeper so the video still shows a clear retry.
X_M = 0.569822
Y_M = 0.073779
SUCCESS_Z_M = 0.081901
LOW_PROBE_Z_M = 0.075901
ORIENTATION_XYZW = (-0.37494456, 0.92704724, 0.0, 0.0)
REQUESTED_ONE_CUP_REFERENCE_TCP_M = (
    0.5705066346798753,
    0.07309322423228727,
    0.08125139615972402,
)
MAX_ONE_CUP_POSE_ERROR_M = 0.002
ONE_CUP_OPENING_RANGE_MM = (2.65, 2.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-arm-only-collision-model", action="store_true")
    parser.add_argument("--recorder-url", default="http://127.0.0.1:8766")
    return parser.parse_args()


class DemoRun:
    def __init__(self, recorder_url: str) -> None:
        self.recorder_url = recorder_url.rstrip("/")
        self.events: list[dict[str, object]] = []
        self.recording_started = False
        self.sequence_started_monotonic_s = time.monotonic()
        RUN_DIR.mkdir(parents=True, exist_ok=True)

    def event(self, name: str, state: str = "instant", **details: object) -> None:
        row: dict[str, object] = {
            "index": len(self.events),
            "name": name,
            "state": state,
            "host_monotonic_s": time.monotonic(),
            "host_wall_time_s": time.time(),
        }
        row.update(details)
        self.events.append(row)
        payload = {
            "schema": "demo2gripper.cup_tactile_video_action_log.v1",
            "sequence_started_host_monotonic_s": self.sequence_started_monotonic_s,
            "hardcoded_pose_source": "reviewed successful real-robot grasp",
            "tactile_semantics": (
                "Raw UVC frames are recorded. Region-template criteria are computed "
                "offline as a shadow gate and do not control this video take."
            ),
            "events": self.events,
        }
        temporary = ACTION_LOG.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(ACTION_LOG)
        print(f"[{name}] {state}", flush=True)

    def recorder_control(self, action: str) -> None:
        url = self.recorder_url + "/control?" + urllib.parse.urlencode({"action": action})
        request = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(request, timeout=3.0) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"recorder {action} returned HTTP {response.status}")
        if action == "record":
            self.recording_started = not self.recording_started

    def run_command(self, name: str, command: list[str], *, allow_failure: bool = False) -> int:
        log_path = RUN_DIR / f"{len(self.events):02d}_{name}.log"
        self.event(name, "start", command=command, log=str(log_path.relative_to(REPO)))
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command,
                cwd=REPO,
                env=self.command_environment(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        self.event(name, "end", returncode=result.returncode)
        if result.returncode and not allow_failure:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"{name} failed with code {result.returncode}:\n{tail}")
        return result.returncode

    @staticmethod
    def command_environment() -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(LEROBOT_SRC)
        return env

    def gripper(self, stage: str, opening_mm: float) -> None:
        command = [
            str(HAND_E_PYTHON),
            "scripts/hande_cli.py",
            "move",
            "--port",
            HAND_E_PORT,
            "--opening-mm",
            f"{opening_mm:.3f}",
            "--speed",
            "15",
            "--force",
            "10",
            "--timeout-s",
            "8",
        ]
        code = self.run_command(stage, command, allow_failure=True)
        if code == 0:
            return

        # A contact stop may be physically valid even if a USB/Modbus completion
        # transition was missed.  Read back without issuing another motion.
        status_command = [
            str(HAND_E_PYTHON),
            "scripts/hande_cli.py",
            "status",
            "--port",
            HAND_E_PORT,
        ]
        status_log = RUN_DIR / f"{len(self.events):02d}_{stage}_status.json"
        result = subprocess.run(
            status_command,
            cwd=REPO,
            env=self.command_environment(),
            capture_output=True,
            text=True,
            check=False,
        )
        status_log.write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{stage} failed and Hand-E status could not be read")
        payload = json.loads(result.stdout.splitlines()[-1])
        measured_mm = float(payload["state"]["opening_width_mm"])
        self.event(
            f"{stage}_readback",
            measured_opening_mm=measured_mm,
            original_returncode=code,
        )
        if opening_mm <= 2.0 and 2.0 <= measured_mm <= 4.5:
            return
        if opening_mm >= 8.0 and abs(measured_mm - opening_mm) <= 1.0:
            return
        raise RuntimeError(
            f"{stage} failed; readback opening {measured_mm:.3f} mm is outside its bound"
        )

    def gripper_opening_readback(self, stage: str) -> float:
        command = [
            str(HAND_E_PYTHON),
            "scripts/hande_cli.py",
            "status",
            "--port",
            HAND_E_PORT,
        ]
        result = subprocess.run(
            command,
            cwd=REPO,
            env=self.command_environment(),
            capture_output=True,
            text=True,
            check=False,
        )
        log_path = RUN_DIR / f"{len(self.events):02d}_{stage}.json"
        log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{stage}: Hand-E status read failed")
        payload = json.loads(result.stdout.splitlines()[-1])
        opening_mm = float(payload["state"]["opening_width_mm"])
        self.event(stage, measured_opening_mm=opening_mm)
        return opening_mm

    @staticmethod
    def write_pose_file(name: str, z_values_m: list[float]) -> Path:
        path = RUN_DIR / f"{name}.json"
        payload = {
            "schema": "demo2gripper.configured_ee_pose_sequence.v1",
            "base_frame": "fr3_link0",
            "waypoints": [
                {
                    "position_m": [X_M, Y_M, z_m],
                    "orientation_xyzw": list(ORIENTATION_XYZW),
                }
                for z_m in z_values_m
            ],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def move(self, stage: str, z_values_m: list[float]) -> dict[str, object]:
        input_path = self.write_pose_file(stage, z_values_m)
        output_path = RUN_DIR / f"{stage}_execution.json"
        command = [
            "scripts/run_ros_python.sh",
            "scripts/moveit_cartesian_pose_sequence.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--link8-to-configured-ee-position-m",
            "0",
            "0",
            "0.19",
            "--link8-to-configured-ee-orientation-xyzw",
            "0",
            "0",
            "0",
            "1",
            "--max-step-m",
            "0.001",
            "--velocity-scale",
            "0.02",
            "--acceleration-scale",
            "0.02",
            "--max-cartesian-speed-m-s",
            "0.008",
            "--execute",
            "--accept-arm-only-collision-model",
        ]
        self.run_command(stage, command)
        return json.loads(output_path.read_text(encoding="utf-8"))

    def run(self) -> None:
        self.event("sequence", "start")

        # Start from the raised reset pose with the fingers fully open, then
        # capture a local unloaded baseline before the visible take begins.
        self.gripper("ensure_fully_open_at_high_ready_pose", 50.0)
        time.sleep(1.0)
        self.recorder_control("baseline")
        self.event("tactile_baseline_reset")
        time.sleep(0.8)
        self.recorder_control("record")
        self.event("recording", "start")
        time.sleep(1.5)

        # Deliberately probe 5 mm below the requested grasp.  This phase never lifts,
        # irrespective of its shadow-classifier result.
        self.move("low_probe_descend", [LOW_PROBE_Z_M])
        time.sleep(0.5)
        self.gripper("low_probe_close", 2.0)
        self.event(
            "low_probe_shadow_gate",
            decision="RETRY",
            motion_policy="NO_LIFT",
            reason="scripted lower probe for demo; actual regions computed offline",
        )
        time.sleep(2.5)

        # Unload first, return exactly to the reviewed pose, and reacquire.
        self.gripper("low_probe_unload", 50.0)
        time.sleep(0.8)
        pose_result = self.move("return_to_reviewed_success_pose", [SUCCESS_Z_M])
        measured_pose = pose_result.get("measured_end_configured_ee")
        if not isinstance(measured_pose, dict):
            raise RuntimeError("reviewed pose execution has no measured configured-EE pose")
        measured_position = measured_pose.get("position_m")
        if not isinstance(measured_position, list) or len(measured_position) != 3:
            raise RuntimeError("reviewed pose execution has malformed measured position")
        pose_error_m = math.dist(
            [float(value) for value in measured_position],
            REQUESTED_ONE_CUP_REFERENCE_TCP_M,
        )
        self.event(
            "reviewed_pose_live_gate",
            measured_position_m=measured_position,
            reference_one_cup_position_m=list(REQUESTED_ONE_CUP_REFERENCE_TCP_M),
            position_error_m=pose_error_m,
            maximum_error_m=MAX_ONE_CUP_POSE_ERROR_M,
            decision="PASS" if pose_error_m <= MAX_ONE_CUP_POSE_ERROR_M else "REJECT",
        )
        if pose_error_m > MAX_ONE_CUP_POSE_ERROR_M:
            self.gripper("pose_gate_reject_keep_open", 50.0)
            raise RuntimeError(
                f"one-cup pose gate rejected {pose_error_m * 1e3:.2f} mm error"
            )
        time.sleep(0.8)
        self.gripper("reviewed_pose_close", 2.0)
        measured_opening_mm = self.gripper_opening_readback(
            "reviewed_pose_opening_gate_readback"
        )
        opening_min_mm, opening_max_mm = ONE_CUP_OPENING_RANGE_MM
        opening_passed = opening_min_mm <= measured_opening_mm <= opening_max_mm
        self.event(
            "reviewed_pose_opening_gate",
            measured_opening_mm=measured_opening_mm,
            accepted_range_mm=[opening_min_mm, opening_max_mm],
            decision="PASS" if opening_passed else "WARN_CONTINUE",
            blocking=False,
            execution_policy="CONTINUE_OPERATOR_APPROVED",
            evidence=(
                "The 2.539 versus 2.734 mm separation is below a robust "
                "standalone decision margin; log it but do not stop this take."
            ),
        )
        self.event(
            "reviewed_pose_shadow_gate",
            expected_serial_184603_regions=1,
            expected_serial_196FB3_regions=2,
            motion_policy="LIFT_REVIEWED_POSE",
            note="actual template-match metrics computed from the recorded frames",
        )
        time.sleep(2.5)

        # Smooth vertical lift with three visible waypoints; leave the cup held.
        self.move(
            "vertical_lift",
            [SUCCESS_Z_M + 0.018, SUCCESS_Z_M + 0.043, SUCCESS_Z_M + 0.068],
        )
        self.event("cup_held_at_lift", expected_lift_m=0.068)
        time.sleep(3.0)

        self.recorder_control("record")
        self.event("recording", "end")
        self.recording_started = False
        time.sleep(1.0)
        self.recorder_control("quit")
        self.event("sequence", "end", result="completed_cup_left_held")


def main() -> int:
    args = parse_args()
    if not args.execute or not args.accept_arm_only_collision_model:
        raise SystemExit(
            "Physical execution requires --execute and --accept-arm-only-collision-model"
        )
    if "realtime" not in platform.release().lower():
        raise SystemExit(f"Refusing physical execution on {platform.release()!r}")

    demo = DemoRun(args.recorder_url)
    try:
        demo.run()
    except BaseException as exc:
        demo.event("sequence", "error", error=f"{type(exc).__name__}: {exc}")
        if demo.recording_started:
            try:
                demo.recorder_control("record")
                demo.event("recording", "end", reason="error_finalize")
            except BaseException as finalize_exc:
                demo.event(
                    "recording_finalize",
                    "error",
                    error=f"{type(finalize_exc).__name__}: {finalize_exc}",
                )
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
