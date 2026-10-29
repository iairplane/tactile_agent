#!/usr/bin/env python3
"""Validate and plan the staged FR3/Hand-E/GF220 test-tube task.

Dry-run is the default and cannot start ROS or open a device.  ``--plan-only``
invokes the existing MoveIt pose tool without ``--execute`` to audit every
available absolute configured-EE pose.  ``--execute`` is intentionally gated
by qualified geometry and the live-adapter boundary; it never converts a
provisional camera extrinsic or projective socket observation into motion.

The current transparent-rack setup has no qualified metric socket depth.  Its
truthful result is therefore ``ENTRY_STANDOFF`` with a guarded-localization
blocker, rather than a fabricated insertion target.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo2gripper.test_tube_task_runner import (  # noqa: E402
    RunnerMode,
    TaskContractError,
    TestTubeTaskInput,
    TestTubeTaskStage,
    command_preview,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and write the command preview without starting any component",
    )
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help="invoke existing MoveIt tools in their non-executing default mode",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="request live execution; all qualified geometry/runtime gates are required",
    )
    parser.add_argument(
        "--live-adapter",
        choices=("none",),
        default="none",
        help=(
            "live hardware adapter; this runner currently exposes only the fail-closed "
            "pre-execution boundary while the session contact/RGB adapter is commissioned"
        ),
    )
    parser.add_argument(
        "--link8-to-configured-ee-position-m",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.190),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--link8-to-configured-ee-orientation-xyzw",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 1.0),
        metavar=("QX", "QY", "QZ", "QW"),
    )
    parser.add_argument("--velocity-scale", type=float, default=0.03)
    parser.add_argument("--acceleration-scale", type=float, default=0.03)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _mode(args: argparse.Namespace) -> RunnerMode:
    if args.execute:
        return RunnerMode.EXECUTE
    if args.plan_only:
        return RunnerMode.PLAN_ONLY
    return RunnerMode.DRY_RUN


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TaskContractError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise TaskContractError(f"JSON root must be an object: {path}")
    return value


def _prepare_output(path: Path, *, overwrite: bool) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise TaskContractError(
                f"refusing to overwrite non-empty output directory {output}"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _pose_command(
    *,
    task: TestTubeTaskInput,
    pose_name: str,
    output: Path,
    args: argparse.Namespace,
) -> list[str]:
    pose = task.poses[pose_name]
    command = [
        str(REPO_ROOT / "scripts/run_ros_python.sh"),
        str(REPO_ROOT / "scripts/moveit_pose_goal.py"),
        "--target-position-m",
        *(f"{value:.12g}" for value in pose.position_m),
        "--target-orientation-xyzw",
        *(f"{value:.12g}" for value in pose.orientation_xyzw),
        "--target-frame",
        "configured-ee",
        "--base-frame",
        task.base_frame,
        "--link8-to-configured-ee-position-m",
        *(f"{value:.12g}" for value in args.link8_to_configured_ee_position_m),
        "--link8-to-configured-ee-orientation-xyzw",
        *(f"{value:.12g}" for value in args.link8_to_configured_ee_orientation_xyzw),
        "--velocity-scale",
        f"{args.velocity_scale:.12g}",
        "--acceleration-scale",
        f"{args.acceleration_scale:.12g}",
        "--output",
        str(output),
    ]
    # This function is used only for plan-only.  Keeping that invariant local
    # makes it impossible for argument construction to append --execute.
    if "--execute" in command:
        raise AssertionError("plan-only command unexpectedly contains --execute")
    return command


def _run_plan_only(
    task: TestTubeTaskInput,
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for pose_name in (
        "approach",
        "grasp",
        "extract",
        "pivot",
        "reclamp",
        "entry_standoff",
    ):
        if pose_name not in task.poses:
            continue
        result_path = output_dir / "moveit" / f"{pose_name}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        command = _pose_command(
            task=task, pose_name=pose_name, output=result_path, args=args
        )
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        result: dict[str, Any] = {
            "pose_name": pose_name,
            "command": command,
            "execute_flag_present": False,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
            "result_path": str(result_path),
        }
        if result_path.is_file():
            try:
                result["moveit_result"] = _read_json(result_path)
            except TaskContractError as error:
                result["result_read_error"] = str(error)
        results.append(result)
    return results


def run(args: argparse.Namespace) -> int:
    mode = _mode(args)
    task = TestTubeTaskInput.parse(_read_json(args.input.expanduser().resolve()))
    output_dir = _prepare_output(args.output_dir, overwrite=args.overwrite)
    start_blockers = list(task.execution_start_blockers())
    insertion_blockers = list(task.guarded_insertion_blockers())
    report: dict[str, Any] = {
        "schema": "demo2gripper.test_tube_task_run.v1",
        "task_id": task.task_id,
        "mode": mode.value,
        "hardware_commands_dispatched": False,
        "initial_stage": TestTubeTaskStage.APPROACH.value,
        "maximum_authorized_stage": (
            TestTubeTaskStage.ENTRY_STANDOFF.value
            if insertion_blockers
            else TestTubeTaskStage.RELEASE_RETREAT.value
        ),
        "execution_start_blockers": start_blockers,
        "guarded_insertion_blockers": insertion_blockers,
        "command_preview": command_preview(task, mode),
        "moveit_plan_results": [],
        "status": "validated",
    }
    _atomic_json(output_dir / "task_report.json", report)

    if mode is RunnerMode.DRY_RUN:
        report["status"] = "dry_run_complete"
        _atomic_json(output_dir / "task_report.json", report)
        print(f"wrote {output_dir / 'task_report.json'}")
        print("mode: dry_run; no ROS or hardware component was started")
        return 0

    if mode is RunnerMode.PLAN_ONLY:
        results = _run_plan_only(task, output_dir=output_dir, args=args)
        report["moveit_plan_results"] = results
        report["status"] = (
            "plan_only_complete"
            if results and all(item["returncode"] == 0 for item in results)
            else "plan_only_incomplete"
        )
        _atomic_json(output_dir / "task_report.json", report)
        print(f"wrote {output_dir / 'task_report.json'}")
        print("mode: plan_only; every MoveIt invocation omitted --execute")
        return 0 if report["status"] == "plan_only_complete" else 2

    # Fail before importing a live SDK, opening the gripper, or calling ROS.
    if start_blockers:
        report["status"] = "execution_blocked_before_dispatch"
        _atomic_json(output_dir / "task_report.json", report)
        print("execution blocked before dispatch:", file=sys.stderr)
        for blocker in start_blockers:
            print(f"- {blocker}", file=sys.stderr)
        return 2

    # The pure outer state machine and canonical tactile-receipt adapter are
    # available, but a live source/command adapter must be commissioned with
    # robot-held calibration evidence.  Never substitute pre-recorded receipts.
    report["status"] = "execution_blocked_live_adapter_unavailable"
    report["execution_start_blockers"] = [
        "live_gf220_rgb_moveit_hande_adapter_not_commissioned"
    ]
    _atomic_json(output_dir / "task_report.json", report)
    print(
        "execution blocked before dispatch: live GF220/RGB/MoveIt/Hand-E adapter "
        "is not commissioned; no command was sent",
        file=sys.stderr,
    )
    return 2


def main() -> None:
    try:
        code = run(parse_args())
    except (TaskContractError, ValueError) as error:
        print(f"input error: {error}", file=sys.stderr)
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
