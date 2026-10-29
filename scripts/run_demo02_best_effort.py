#!/usr/bin/env python3
"""Attempt the complete fixed demo-02 motion chain without semantic early exits.

This runner is intentionally narrow.  It consumes reviewed, absolute configured-EE
poses and tries every stage in order.  A failed motion or gripper command is recorded
atomically and does not suppress later independent absolute targets.  Motion is still
delegated to the repository's plan-audit-execute tools; this file does not bypass
trajectory limits or send an unaudited trajectory.
"""

from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import math
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo2gripper.autonomous_pick import atomic_write_json, pose_result_succeeded
from demo2gripper.cartesian_pose_sequence import (
    CartesianPose,
    load_pose_sequence,
    parse_pose_sequence_payload,
)
from scripts.run_autonomous_pick import (
    _artifact_summary,
    _hande_command,
    _read_json,
)
from scripts.run_autonomous_pick import _validate_config as _validate_pick_config
from scripts.run_demo02_full_task import (
    BASE_FRAME,
    _pose_command,
    _pose_sequence_command,
    pose_sequence_result_succeeded,
)

SCHEMA = "demo2gripper.demo02_best_effort_config.v1"
REPORT_SCHEMA = "demo2gripper.demo02_best_effort_report.v1"

STAGE_ORDER = (
    "INITIAL_OPEN",
    "PREGRASP",
    "GRASP",
    "OBJECT_CLOSE",
    "LIFT",
    "TRANSPORT",
    "PLACE",
    "OBJECT_RELEASE",
    "PLACE_RETREAT",
    "LID_PRECONTACT",
    "LID_CONTACT",
    "LID_GRIPPER_CLOSE",
    "LID_OPEN",
    "HOLD",
    "LID_CLOSE",
    "LID_RELEASE",
    "FINAL_RETREAT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-arm-only-collision-model", action="store_true")
    return parser.parse_args()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_path(value: Any, *, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _command(config: Mapping[str, Any], name: str) -> list[str]:
    commands = config.get("commands")
    value = commands.get(name) if isinstance(commands, Mapping) else None
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(token, str) or not token for token in value)
    ):
        raise ValueError(f"config.commands.{name} must be a non-empty string array")
    return list(value)


def _load_pose(value: Any, *, base: Path, name: str) -> tuple[CartesianPose, str]:
    if isinstance(value, str):
        path = _resolve_path(value, base=base, name=name)
        poses = load_pose_sequence(path, expected_base_frame=BASE_FRAME)
        if len(poses) != 1:
            raise ValueError(f"{name} must reference exactly one waypoint")
        return poses[0], str(path)
    poses = parse_pose_sequence_payload(
        {
            "schema": "demo2gripper.configured_ee_pose_sequence.v1",
            "base_frame": BASE_FRAME,
            "waypoints": [value],
        },
        expected_base_frame=BASE_FRAME,
    )
    return poses[0], "inline_config"


def _validate_config(config: Mapping[str, Any], *, config_path: Path) -> dict[str, Any]:
    if config.get("schema") != SCHEMA:
        raise ValueError(f"config schema must be {SCHEMA!r}")
    base = config_path.parent.resolve()
    pick_config_path = _resolve_path(
        config.get("pick_config"), base=base, name="pick_config"
    )
    pick = _validate_pick_config(
        _read_json(pick_config_path), config_path=pick_config_path, execute=True
    )
    raw_poses = config.get("absolute_poses")
    if not isinstance(raw_poses, Mapping):
        raise ValueError("config.absolute_poses must be an object")
    pose_names = (
        "pregrasp_tcp",
        "grasp_tcp",
        "lift_tcp",
        "transport_tcp",
        "placement_tcp",
        "placement_retreat_tcp",
        "lid_precontact_tcp",
        "lid_contact_tcp",
        "lid_final_retreat_tcp",
    )
    poses: dict[str, CartesianPose] = {}
    pose_sources: dict[str, str] = {}
    for name in pose_names:
        pose, source = _load_pose(raw_poses.get(name), base=base, name=f"absolute_poses.{name}")
        poses[name] = pose
        pose_sources[name] = source

    if poses["pregrasp_tcp"].position_m[2] <= poses["grasp_tcp"].position_m[2]:
        raise ValueError("pregrasp_tcp must be above grasp_tcp")
    if poses["lift_tcp"].position_m[2] <= poses["grasp_tcp"].position_m[2]:
        raise ValueError("lift_tcp must be above grasp_tcp")
    if poses["placement_retreat_tcp"].position_m[2] <= poses["placement_tcp"].position_m[2]:
        raise ValueError("placement_retreat_tcp must be above placement_tcp")
    if poses["lid_precontact_tcp"].position_m[2] <= poses["lid_contact_tcp"].position_m[2]:
        raise ValueError("lid_precontact_tcp must be above lid_contact_tcp")
    if poses["lid_final_retreat_tcp"].position_m[2] <= poses["lid_contact_tcp"].position_m[2]:
        raise ValueError("lid_final_retreat_tcp must be above lid_contact_tcp")

    raw_sequences = config.get("pose_sequences")
    if not isinstance(raw_sequences, Mapping):
        raise ValueError("config.pose_sequences must be an object")
    sequence_paths = {
        name: _resolve_path(raw_sequences.get(name), base=base, name=f"pose_sequences.{name}")
        for name in ("lid_open", "lid_close")
    }
    sequences = {
        name: load_pose_sequence(path, expected_base_frame=BASE_FRAME)
        for name, path in sequence_paths.items()
    }
    hold_duration_s = float(config.get("hold_duration_s", 1.0))
    if not math.isfinite(hold_duration_s) or not 0.0 <= hold_duration_s <= 10.0:
        raise ValueError("hold_duration_s must lie in [0, 10]")
    return {
        "repo_root": pick["repo_root"],
        "pick_config": pick_config_path,
        "pick": pick,
        "commands": {"pose_sequence": _command(config, "pose_sequence")},
        "poses": poses,
        "pose_sources": pose_sources,
        "sequence_paths": sequence_paths,
        "sequences": sequences,
        "hold_duration_s": hold_duration_s,
        "pose_sequence_max_step_m": float(config.get("pose_sequence_max_step_m", 0.005)),
        "pose_sequence_speed_m_s": float(config.get("pose_sequence_speed_m_s", 0.015)),
    }


def _run_process(
    command: Sequence[str], *, cwd: Path, log_prefix: Path
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command), cwd=cwd, text=True, capture_output=True, check=False
    )
    log_prefix.with_suffix(".stdout.log").write_text(completed.stdout, encoding="utf-8")
    log_prefix.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
    return completed


def _attempt_pose(
    normalized: Mapping[str, Any], name: str, *, output_dir: Path
) -> tuple[bool, dict[str, Any]]:
    target: CartesianPose = normalized["poses"][name]
    artifact = output_dir / f"{name}.json"
    command = [
        *_pose_command(normalized, target, output=artifact, execute=True),
    ]
    completed = _run_process(
        command, cwd=normalized["repo_root"], log_prefix=output_dir / name
    )
    details: dict[str, Any] = {
        "command_returncode": completed.returncode,
        "target": target.as_dict(),
        "target_source": normalized["pose_sources"][name],
        "stdout_log": str((output_dir / name).with_suffix(".stdout.log")),
        "stderr_log": str((output_dir / name).with_suffix(".stderr.log")),
    }
    if not artifact.is_file():
        details["error"] = "motion tool produced no JSON artifact"
        return False, details
    try:
        payload = _read_json(artifact)
    except Exception as exc:
        details.update({"artifact": str(artifact), "error": f"invalid artifact: {exc}"})
        return False, details
    details.update(_artifact_summary(artifact, payload))
    succeeded = completed.returncode == 0 and pose_result_succeeded(payload, executed=True)
    if not succeeded:
        details["error"] = "absolute pose plan/audit/execute did not succeed"
    return succeeded, details


def _attempt_sequence(
    normalized: Mapping[str, Any], name: str, *, output_dir: Path
) -> tuple[bool, dict[str, Any]]:
    input_path: Path = normalized["sequence_paths"][name]
    artifact = output_dir / f"{name}.json"
    command = _pose_sequence_command(
        normalized, input_path, output=artifact, execute=True
    )
    completed = _run_process(
        command, cwd=normalized["repo_root"], log_prefix=output_dir / name
    )
    details: dict[str, Any] = {
        "command_returncode": completed.returncode,
        "input": str(input_path),
        "waypoint_count": len(normalized["sequences"][name]),
        "stdout_log": str((output_dir / name).with_suffix(".stdout.log")),
        "stderr_log": str((output_dir / name).with_suffix(".stderr.log")),
    }
    if not artifact.is_file():
        details["error"] = "pose-sequence tool produced no JSON artifact"
        return False, details
    try:
        payload = _read_json(artifact)
    except Exception as exc:
        details.update({"artifact": str(artifact), "error": f"invalid artifact: {exc}"})
        return False, details
    details.update(_artifact_summary(artifact, payload))
    succeeded = completed.returncode == 0 and pose_sequence_result_succeeded(
        payload, executed=True
    )
    if not succeeded:
        details["error"] = "pose-sequence plan/audit/execute did not succeed"
    return succeeded, details


def _attempt_gripper(
    normalized: Mapping[str, Any], action: str, artifact_name: str, *, output_dir: Path
) -> tuple[bool, dict[str, Any]]:
    completed = _run_process(
        _hande_command(normalized["pick"], action),
        cwd=normalized["repo_root"],
        log_prefix=output_dir / artifact_name,
    )
    artifact = output_dir / f"{artifact_name}.json"
    details: dict[str, Any] = {
        "command_returncode": completed.returncode,
        "action": action,
        "stdout_log": str((output_dir / artifact_name).with_suffix(".stdout.log")),
        "stderr_log": str((output_dir / artifact_name).with_suffix(".stderr.log")),
    }
    try:
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise ValueError("Hand-E JSON is not an object")
        atomic_write_json(artifact, payload)
        details.update(_artifact_summary(artifact, payload))
        details["state"] = payload.get("state")
        succeeded = completed.returncode == 0 and payload.get("ok") is True
    except Exception as exc:
        details["error"] = f"invalid Hand-E output: {exc}"
        succeeded = False
    if not succeeded and "error" not in details:
        details["error"] = "Hand-E command did not report ok"
    return succeeded, details


def _attempt_hold(
    normalized: Mapping[str, Any], *, output_dir: Path
) -> tuple[bool, dict[str, Any]]:
    del output_dir
    duration = normalized["hold_duration_s"]
    time.sleep(duration)
    return True, {"duration_s": duration}


def _append_stage(
    report: dict[str, Any],
    report_path: Path,
    name: str,
    *,
    succeeded: bool,
    details: Mapping[str, Any],
    started_at: str,
) -> None:
    expected = STAGE_ORDER[len(report["stages"])]
    if name != expected:
        raise ValueError(f"cannot append {name}; expected {expected}")
    report["stages"].append(
        {
            "name": name,
            "status": "passed" if succeeded else "failed_continued",
            "started_at": started_at,
            "finished_at": _now(),
            "details": dict(details),
        }
    )
    if not succeeded:
        report["failed_stages"].append(name)
    report["updated_at"] = _now()
    atomic_write_json(report_path, report)


def run(args: argparse.Namespace) -> int:
    if not args.execute or not args.accept_arm_only_collision_model:
        raise ValueError(
            "physical showcase requires both --execute and "
            "--accept-arm-only-collision-model"
        )
    config_path = args.config.expanduser().resolve()
    normalized = _validate_config(_read_json(config_path), config_path=config_path)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    report_path = output_dir / "stage_report.json"
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "running",
        "mode": "execute_best_effort",
        "execution_policy": (
            "attempt_every_stage; each motion remains plan-audit-execute; "
            "semantic or command failures are recorded and do not terminate the chain"
        ),
        "authorized_for_robot_execution": True,
        "arm_only_collision_model_acknowledged": True,
        "config": str(config_path),
        "output_dir": str(output_dir),
        "started_at": _now(),
        "updated_at": _now(),
        "stage_order": list(STAGE_ORDER),
        "stages": [],
        "failed_stages": [],
        "absolute_targets": {
            name: pose.as_dict() for name, pose in normalized["poses"].items()
        },
    }
    atomic_write_json(report_path, report)

    operations: list[tuple[str, Callable[..., tuple[bool, dict[str, Any]]]]] = [
        ("INITIAL_OPEN", lambda **kw: _attempt_gripper(normalized, "open", "initial_open", **kw)),
        ("PREGRASP", lambda **kw: _attempt_pose(normalized, "pregrasp_tcp", **kw)),
        ("GRASP", lambda **kw: _attempt_pose(normalized, "grasp_tcp", **kw)),
        ("OBJECT_CLOSE", lambda **kw: _attempt_gripper(normalized, "close", "object_close", **kw)),
        ("LIFT", lambda **kw: _attempt_pose(normalized, "lift_tcp", **kw)),
        ("TRANSPORT", lambda **kw: _attempt_pose(normalized, "transport_tcp", **kw)),
        ("PLACE", lambda **kw: _attempt_pose(normalized, "placement_tcp", **kw)),
        (
            "OBJECT_RELEASE",
            lambda **kw: _attempt_gripper(normalized, "open", "object_release", **kw),
        ),
        ("PLACE_RETREAT", lambda **kw: _attempt_pose(normalized, "placement_retreat_tcp", **kw)),
        ("LID_PRECONTACT", lambda **kw: _attempt_pose(normalized, "lid_precontact_tcp", **kw)),
        ("LID_CONTACT", lambda **kw: _attempt_pose(normalized, "lid_contact_tcp", **kw)),
        (
            "LID_GRIPPER_CLOSE",
            lambda **kw: _attempt_gripper(
                normalized, "close", "lid_gripper_close", **kw
            ),
        ),
        ("LID_OPEN", lambda **kw: _attempt_sequence(normalized, "lid_open", **kw)),
        ("HOLD", lambda **kw: _attempt_hold(normalized, **kw)),
        ("LID_CLOSE", lambda **kw: _attempt_sequence(normalized, "lid_close", **kw)),
        ("LID_RELEASE", lambda **kw: _attempt_gripper(normalized, "open", "lid_release", **kw)),
        ("FINAL_RETREAT", lambda **kw: _attempt_pose(normalized, "lid_final_retreat_tcp", **kw)),
    ]

    for stage_name, operation in operations:
        started_at = _now()
        try:
            succeeded, details = operation(output_dir=output_dir)
        except Exception as exc:  # continue is the defining contract of this runner
            succeeded = False
            details = {"error_type": type(exc).__name__, "error": str(exc)}
        _append_stage(
            report,
            report_path,
            stage_name,
            succeeded=succeeded,
            details=details,
            started_at=started_at,
        )

    report["status"] = (
        "completed" if not report["failed_stages"] else "completed_with_failures"
    )
    report["finished_at"] = _now()
    atomic_write_json(report_path, report)
    print(f"demo-02 best-effort chain complete: {report_path}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception:  # pragma: no cover
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
