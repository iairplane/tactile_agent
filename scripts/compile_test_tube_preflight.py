#!/usr/bin/env python3
"""Compile the test-tube task through the last offline preflight boundary.

The program only reads JSON and writes a review contract/report.  It has no
execution mode and never starts ROS, MoveIt, a gripper process, or a tactile
device.  Missing calibrations are emitted as explicit blockers while the
complete state machine remains available for review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo2gripper.test_tube_predicates import predicate_producer_manifest  # noqa: E402
from demo2gripper.test_tube_preflight import (  # noqa: E402
    TEST_TUBE_FRONT_RGB_REFERENCE_SCHEMA,
    TEST_TUBE_GEOMETRY_SCHEMAS,
    TEST_TUBE_RGB_TRACKER_VALIDATION_SCHEMA,
    TEST_TUBE_TACTILE_OBSERVABILITY_SCHEMA,
    compile_test_tube_preflight,
    validate_test_tube_geometry_artifact,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve_optional_path(value: object, *, config_path: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("evidence path must be null or a non-empty string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--stage-review", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-plan-only-ready",
        action="store_true",
        help="return status 2 unless fresh metric MoveIt targets are available",
    )
    parser.add_argument(
        "--require-offline-contract-ready",
        action="store_true",
        help="return status 2 unless the reviewable task contract is complete",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    stage_path = args.stage_review.resolve()
    config_path = args.config.resolve()
    output_path = args.output.resolve()
    if output_path.exists() and not args.overwrite:
        raise ValueError(f"refusing to overwrite {output_path}; pass --overwrite")

    stage_review = _read_json(stage_path)
    config = _read_json(config_path)
    evidence_config = config.get("evidence", {})
    tactile_ref = evidence_config.get("tactile_observability", {})
    tactile_path = _resolve_optional_path(tactile_ref.get("path"), config_path=config_path)
    tactile: dict[str, Any] | None = None
    tactile_load_status: dict[str, Any]
    if tactile_path is None:
        tactile_load_status = {"path": None, "loaded": False, "reason": "not_configured"}
    elif not tactile_path.exists():
        tactile_load_status = {
            "path": str(tactile_path),
            "loaded": False,
            "reason": "not_found",
        }
    else:
        tactile = _read_json(tactile_path)
        if tactile.get("schema") != TEST_TUBE_TACTILE_OBSERVABILITY_SCHEMA:
            raise ValueError(f"unexpected tactile observability schema in {tactile_path}")
        tactile_load_status = {
            "path": str(tactile_path),
            "loaded": True,
            "reason": None,
        }

    front_ref = evidence_config.get("front_rgb_pivot_reference", {})
    front_path = _resolve_optional_path(front_ref.get("path"), config_path=config_path)
    front_rgb_reference: dict[str, Any] | None = None
    front_rgb_load_status: dict[str, Any]
    if front_path is None:
        front_rgb_load_status = {
            "path": None,
            "loaded": False,
            "reason": "not_configured",
        }
    elif not front_path.exists():
        front_rgb_load_status = {
            "path": str(front_path),
            "loaded": False,
            "reason": "not_found",
        }
    else:
        front_rgb_reference = _read_json(front_path)
        if front_rgb_reference.get("schema") != TEST_TUBE_FRONT_RGB_REFERENCE_SCHEMA:
            raise ValueError(f"unexpected front RGB reference schema in {front_path}")
        front_rgb_load_status = {
            "path": str(front_path),
            "loaded": True,
            "reason": None,
        }

    tracker_ref = evidence_config.get("front_rgb_tracker_validation", {})
    tracker_path = _resolve_optional_path(tracker_ref.get("path"), config_path=config_path)
    front_rgb_tracker_validation: dict[str, Any] | None = None
    tracker_load_status: dict[str, Any]
    if tracker_path is None:
        tracker_load_status = {"path": None, "loaded": False, "reason": "not_configured"}
    elif not tracker_path.exists():
        tracker_load_status = {
            "path": str(tracker_path),
            "loaded": False,
            "reason": "not_found",
        }
    else:
        front_rgb_tracker_validation = _read_json(tracker_path)
        if front_rgb_tracker_validation.get("schema") != TEST_TUBE_RGB_TRACKER_VALIDATION_SCHEMA:
            raise ValueError(f"unexpected front RGB tracker schema in {tracker_path}")
        tracker_load_status = {
            "path": str(tracker_path),
            "loaded": True,
            "reason": None,
        }

    plan = compile_test_tube_preflight(
        stage_review=stage_review,
        config=config,
        tactile_observability=tactile,
        front_rgb_reference=front_rgb_reference,
        front_rgb_tracker_validation=front_rgb_tracker_validation,
    )
    artifact_files: dict[str, Any] = {}
    geometry_summaries: dict[str, Any] = {}
    missing_geometry_files: list[str] = []
    missing_offline_component_files: list[str] = []
    for name, reference in evidence_config.items():
        resolved = _resolve_optional_path(reference.get("path"), config_path=config_path)
        exists = resolved is not None and resolved.is_file()
        artifact_files[name] = {
            "configured_path": reference.get("path"),
            "resolved_path": None if resolved is None else str(resolved),
            "exists": exists,
            "sha256": _sha256(resolved) if exists else None,
        }
        if name in TEST_TUBE_GEOMETRY_SCHEMAS:
            if resolved is not None and not exists:
                missing_geometry_files.append(f"{name}_path_not_found")
            elif exists:
                geometry_summaries[name] = validate_test_tube_geometry_artifact(
                    name,
                    _read_json(resolved),
                    task_id=config["task_id"],
                )
        if name in {
            "d435_live_source",
            "offline_runtime_orchestrator",
            "offline_predicate_builder",
            "rgb_tracker_qualification_tool",
            "rgb_tracker_qualification_evaluator",
        } and not exists:
            missing_offline_component_files.append(f"{name}_path_not_found")
    missing_offline_files = missing_geometry_files + missing_offline_component_files
    if missing_offline_files:
        blockers = sorted(set(plan["preflight"]["blocking_conditions"] + missing_offline_files))
        plan["preflight"]["blocking_conditions"] = blockers
        plan["preflight"]["offline_task_contract_ready"] = False
        if missing_geometry_files:
            plan["preflight"]["geometry_contract_ready"] = False
        plan["preflight"]["moveit_plan_only_ready"] = False
        plan["preflight"]["hardware_execution_ready"] = False
        plan["status"] = "blocked_preflight"
    plan["input_files"] = {
        "stage_review": str(stage_path),
        "config": str(config_path),
        "tactile_observability": tactile_load_status,
        "front_rgb_pivot_reference": front_rgb_load_status,
        "front_rgb_tracker_validation": tracker_load_status,
        "artifacts": artifact_files,
    }
    plan["geometry_artifacts"] = {
        "all_plan_only_artifacts_valid": set(geometry_summaries) == set(TEST_TUBE_GEOMETRY_SCHEMAS),
        "metric_execution_geometry_ready": False,
        "artifacts": geometry_summaries,
    }
    plan["predicate_producer_manifest"] = predicate_producer_manifest()
    _atomic_json(output_path, plan)
    preflight = plan["preflight"]
    print(f"wrote {output_path}")
    print(f"status: {plan['status']}")
    print("hardware dispatch: disabled by construction")
    if preflight["blocking_conditions"]:
        print("blocking conditions:")
        for blocker in preflight["blocking_conditions"]:
            print(f"- {blocker}")
    if args.require_plan_only_ready and not preflight["moveit_plan_only_ready"]:
        return 2
    if args.require_offline_contract_ready and not preflight["offline_task_contract_ready"]:
        return 2
    return 0


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
