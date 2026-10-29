#!/usr/bin/env python3
"""Replay the test-tube semantic runtime from JSON; no hardware mode exists.

This command only connects ``JsonReplay`` to the hardware-independent runtime
orchestrator.  It cannot import or start ROS, MoveIt, libfranka, Hand-E, a
camera, or a tactile device.  A compiled plan with unresolved runtime or
metric-geometry blockers produces a durable ``HOLD`` report and exit status 2.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo2gripper.test_tube_runtime import (  # noqa: E402
    JsonReplay,
    RuntimeContractError,
    run_test_tube_runtime,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeContractError(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeContractError(f"JSON root must be an object: {path}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing replay output directory",
    )
    return parser.parse_args(argv)


def _prepare_output(path: Path, *, overwrite: bool) -> Path:
    result = path.expanduser().resolve()
    if result.exists() and any(result.iterdir()):
        if not overwrite:
            raise RuntimeContractError(
                f"refusing to overwrite non-empty output directory {result}; pass --overwrite"
            )
        shutil.rmtree(result)
    result.mkdir(parents=True, exist_ok=True)
    return result


def _task_id(value: Mapping[str, Any], *, name: str) -> str:
    task_id = value.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise RuntimeContractError(f"{name}.task_id must be a non-empty string")
    return task_id


def run(args: argparse.Namespace) -> int:
    plan = _read_json(args.plan.expanduser().resolve())
    replay = JsonReplay(_read_json(args.replay.expanduser().resolve()))
    if replay.task_id != _task_id(plan, name="compiled plan"):
        raise RuntimeContractError("replay.task_id does not match the compiled plan")
    output_dir = _prepare_output(args.output_dir, overwrite=args.overwrite)
    report = run_test_tube_runtime(
        compiled_plan=plan,
        live_geometry=replay.live_geometry,
        predicate_provider=replay,
        action_executor=replay,
        output_dir=output_dir,
        max_steps=replay.max_steps,
        max_grasp_close_probes=replay.max_grasp_close_probes,
    )
    if report["status"] in {"complete", "hold"}:
        replay.assert_fully_consumed()
    print(f"wrote {output_dir / 'runtime_report.json'}")
    print(f"status: {report['status']}")
    print(f"final state: {report['final_state']}")
    print("hardware dispatch: unavailable by construction")
    if report["blocking_conditions"]:
        print("blocking conditions:")
        for blocker in report["blocking_conditions"]:
            print(f"- {blocker}")
    return 0 if report["status"] == "complete" else 2


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
