#!/usr/bin/env python3
"""Collect one bounded GF220 grasp/press/twist task-frame calibration.

This tool is intentionally task-specific.  It starts both GF220s unloaded,
moves from the reviewed pre-contact pose around one cap, records raw FORCE6D
windows after grasp, axial press, and a small positive twist, then always opens
and retracts to the reviewed pre-contact pose.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from demo2gripper.gf220_live import (  # noqa: E402, I001
    GF220LiveConfig,
    GF220LiveSource,
)


PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AQ00O97B-if00-port0"
PROFILE = REPO / "configs/gf220_manual_warp_draft.json"
Q0 = np.array([-0.374944558450502, 0.9270472361688794, 0.0, 0.0])
XY = np.array([0.57084, 0.06114])


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_logged(command: list[str], prefix: Path) -> dict[str, object] | None:
    completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".stdout.log").write_text(completed.stdout, encoding="utf-8")
    prefix.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {command!r}")
    if completed.stdout.strip():
        try:
            return json.loads(completed.stdout.splitlines()[-1])
        except json.JSONDecodeError:
            return None
    return None


def pose(z_m: float, quaternion: np.ndarray = Q0) -> dict[str, object]:
    return {
        "position_m": [float(XY[0]), float(XY[1]), float(z_m)],
        "orientation_xyzw": [float(value) for value in quaternion],
    }


def move_sequence(out: Path, label: str, poses: list[dict[str, object]]) -> None:
    input_path = out / "poses" / f"{label}.json"
    output_path = out / "motion" / f"{label}.json"
    write_json(
        input_path,
        {
            "schema": "demo2gripper.configured_ee_pose_sequence.v1",
            "base_frame": "fr3_link0",
            "waypoints": poses,
        },
    )
    run_logged(
        [
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
            "--base-frame",
            "fr3_link0",
            "--server-timeout-s",
            "15",
            "--max-step-m",
            "0.001",
            "--jump-threshold",
            "2.0",
            "--revolute-jump-threshold-rad",
            "0.1",
            "--velocity-scale",
            "0.02",
            "--acceleration-scale",
            "0.02",
            "--max-cartesian-speed-m-s",
            "0.006",
            "--min-joint-limit-margin-rad",
            "0.05",
            "--max-joint-sweep-rad",
            "2.0",
            "--max-total-joint-sweep-rad",
            "6.0",
            "--execute",
            "--accept-arm-only-collision-model",
        ],
        out / "commands" / label,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if payload.get("execution_succeeded") is not True:
        raise RuntimeError(f"{label} did not report execution_succeeded=true")


def gripper(out: Path, label: str, opening_mm: float) -> None:
    payload = run_logged(
        [
            "/home/vitai/miniconda3/envs/fhh/bin/python",
            "scripts/hande_cli.py",
            "move",
            "--port",
            PORT,
            "--opening-mm",
            str(opening_mm),
            "--speed",
            "15",
            "--force",
            "10",
            "--timeout-s",
            "8",
        ],
        out / "commands" / label,
    )
    if not payload or payload.get("ok") is not True:
        raise RuntimeError(f"{label} did not report ok=true")
    write_json(out / "gripper" / f"{label}.json", payload)


def sample(source: GF220LiveSource, out: Path, label: str) -> dict[str, object]:
    cutoff = time.monotonic()
    pair = source.collect_classifier_windows(cutoff, 4, 4.0)
    result: dict[str, object] = {
        "start_skew_s": pair.start_skew_s,
        "end_skew_s": pair.end_skew_s,
        "sensors": {},
    }
    for serial, window in zip(pair.serials, pair.windows):
        force = np.asarray(window.force6d.values, dtype=float)
        np.save(out / "force6d" / f"{label}_{serial}.npy", force)
        np.save(out / "warped" / f"{label}_{serial}.npy", window.warped.values)
        np.save(out / "diff" / f"{label}_{serial}.npy", window.diff.values)
        result["sensors"][serial] = {
            "samples": int(force.shape[0]),
            "median": np.median(force, axis=0).tolist(),
            "std": np.std(force, axis=0).tolist(),
            "capture_monotonic_s": window.force6d.capture_monotonic_s.tolist(),
        }
    write_json(out / "samples" / f"{label}.json", result)
    return result


def recover_from_cap_region(out: Path) -> list[BaseException]:
    """Best-effort release and axial retreat after entering the cap region."""
    errors: list[BaseException] = []
    try:
        gripper(out, "recovery_open", 50.0)
    except BaseException as exc:
        errors.append(exc)
        write_json(out / "recovery_open_error.json", {"error": str(exc)})
    try:
        move_sequence(out, "recovery_retract", [pose(0.0825), pose(0.1050)])
    except BaseException as exc:
        errors.append(exc)
        write_json(out / "recovery_retract_error.json", {"error": str(exc)})
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: calibrate_gf220_bottle_task.py OUTPUT_DIR")
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("force6d", "warped", "diff"):
        (out / name).mkdir()

    source = GF220LiveSource(
        PROFILE,
        config=GF220LiveConfig(
            buffer_frames=64,
            warmup_frames=8,
            warp_validation_frames=2,
            discovery_retries=10,
            discovery_retry_delay_s=0.5,
            cached_poll_delay_s=0.002,
            sdk_timestamp_scale_s=0.001,
            max_frame_age_s=0.6,
            max_pair_skew_s=0.15,
            close_timeout_s=3.0,
        ),
    )
    cap_region_entry_started = False
    try:
        source.start(timeout_s=15.0)
        calibrations = source.calibrate_unloaded(timeout_s=15.0)
        write_json(
            out / "calibrations.json",
            {
                item.serial: {
                    "calibration_id": item.calibration_id,
                    "warp_sha256": item.warp_sha256,
                    "baseline_sha256": item.baseline_sha256,
                }
                for item in calibrations
            },
        )
        gripper(out, "open_before_calibration", 50.0)
        samples: dict[str, object] = {"unloaded": sample(source, out, "unloaded")}

        # Set this before dispatch.  A reflex or partial execution can leave the
        # fingers around the cap even when move_sequence raises.
        cap_region_entry_started = True
        move_sequence(out, "descend_around_cap", [pose(0.0825), pose(0.0600)])
        gripper(out, "close_on_cap", 18.0)
        samples["grasp"] = sample(source, out, "grasp")

        move_sequence(out, "press_2mm", [pose(0.0580)])
        samples["press"] = sample(source, out, "press")

        rotation = Rotation.from_rotvec(np.array([0.0, 0.0, math.radians(5.0)])).as_matrix()
        base = Rotation.from_quat(Q0).as_matrix()
        twisted_q = Rotation.from_matrix(rotation @ base).as_quat()
        move_sequence(out, "twist_positive_5deg", [pose(0.0580, twisted_q)])
        samples["twist_positive_5deg"] = sample(source, out, "twist_positive_5deg")

        medians = {
            stage: {
                serial: np.asarray(values["median"], dtype=float)
                for serial, values in payload["sensors"].items()
            }
            for stage, payload in samples.items()
        }
        deltas: dict[str, object] = {}
        for serial in medians["unloaded"]:
            deltas[serial] = {
                "grasp_minus_unloaded": (
                    medians["grasp"][serial] - medians["unloaded"][serial]
                ).tolist(),
                "press_minus_grasp": (
                    medians["press"][serial] - medians["grasp"][serial]
                ).tolist(),
                "twist_minus_press": (
                    medians["twist_positive_5deg"][serial] - medians["press"][serial]
                ).tolist(),
            }
        write_json(out / "calibration_deltas.json", deltas)
        print(json.dumps(deltas, indent=2), flush=True)
        return 0
    finally:
        active_exception = sys.exc_info()[1]
        cleanup_errors: list[BaseException] = []
        if cap_region_entry_started:
            cleanup_errors.extend(recover_from_cap_region(out))
        try:
            source.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
            write_json(out / "source_close_error.json", {"error": str(exc)})
        if cleanup_errors and active_exception is None:
            messages = "; ".join(str(exc) for exc in cleanup_errors)
            raise RuntimeError(f"calibration cleanup failed: {messages}")


if __name__ == "__main__":
    raise SystemExit(main())
