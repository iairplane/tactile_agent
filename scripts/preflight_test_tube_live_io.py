#!/usr/bin/env python3
"""Inspect the test-tube live adapter without opening a camera or robot device."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from demo2gripper.test_tube_live_io import assess_live_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--check-servo-runtime",
        action="store_true",
        help="also inspect the installed ROS graph; still sends no hardware command",
    )
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config root must be an object")
    return payload


def _command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )


def _servo_blockers(config: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    package = _command(["ros2", "pkg", "prefix", "moveit_servo"])
    if package.returncode != 0:
        blockers.append("moveit_servo_package_missing")
        return blockers
    live = config.get("live_adapter", {})
    robot = live.get("robot", {}) if isinstance(live, dict) else {}
    servo = robot.get("servo", {}) if isinstance(robot, dict) else {}
    node_name = servo.get("node_name", "/servo_node")
    topic = servo.get("twist_topic", "/servo_node/delta_twist_cmds")
    nodes = _command(["ros2", "node", "list"])
    if nodes.returncode != 0 or node_name not in nodes.stdout.splitlines():
        blockers.append("moveit_servo_node_not_active")
    topic_info = _command(["ros2", "topic", "info", str(topic)])
    if topic_info.returncode != 0 or "Subscription count: 0" in topic_info.stdout:
        blockers.append("moveit_servo_twist_subscription_missing")
    return blockers


def main() -> None:
    args = parse_args()
    config = _read(args.config.expanduser().resolve())
    assessment = assess_live_config(config)
    blockers = list(assessment.static_blockers)
    if args.check_servo_runtime:
        blockers.extend(_servo_blockers(config))
    blockers = sorted(set(blockers))
    result = {
        "config": str(args.config.expanduser().resolve()),
        "ready": not blockers,
        "hardware_opened": False,
        "hardware_command_dispatched": False,
        "blockers": blockers,
        "runtime_requirements": list(assessment.runtime_requirements),
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not blockers else 2)


if __name__ == "__main__":
    main()
