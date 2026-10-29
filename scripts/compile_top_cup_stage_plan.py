#!/usr/bin/env python3
"""Compile the non-executable top-cup stage contract from reviewed JSON inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from demo2gripper.cup_stage_plan import compile_top_cup_stage_plan


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cup-rim-relation", type=Path, required=True)
    parser.add_argument("--semantic-stages", type=Path, required=True)
    parser.add_argument("--tactile-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise ValueError(f"refusing to overwrite {args.output}; pass --overwrite")
    plan = compile_top_cup_stage_plan(
        cup_rim_relation=_read_object(args.cup_rim_relation),
        semantic_stages=_read_object(args.semantic_stages),
        tactile_policy=_read_object(args.tactile_policy),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print("status: review_only; execution_authorized=false; hardware_execution_enabled=false")


if __name__ == "__main__":
    main()
