#!/usr/bin/env python3
"""Generate and optionally rank plan-only FR3 test-tube pregrasp candidates.

The script cannot execute a trajectory: it exposes no ``--execute`` option and
the generated MoveIt argv never contains that flag.  With
``--invoke-plan-only`` it calls the existing audited pose planner once per roll
candidate, collects every result, and ranks accepted plans by joint sweep.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from demo2gripper.test_tube_pregrasp import (  # noqa: E402
    assess_moveit_plan,
    build_moveit_plan_only_command,
    generate_pregrasp_candidates,
    rank_plan_assessments,
    sha256_file,
    verify_relation_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--current-relation", type=Path, required=True)
    parser.add_argument(
        "--t-base-camera",
        type=Path,
        required=True,
        help="explicit T_base_camera .npy used only to generate plan-only poses",
    )
    parser.add_argument(
        "--t-base-camera-provenance",
        type=Path,
        required=True,
        help="report/manifest explaining where this exact extrinsic came from",
    )
    parser.add_argument(
        "--t-link8-tcp",
        type=Path,
        default=REPOSITORY_ROOT / "calibration/T_link8_canonical_tcp.npy",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planner-output-dir", type=Path)
    parser.add_argument("--approach-distance-m", type=float, default=0.060)
    parser.add_argument("--roll-step-deg", type=float, default=30.0)
    parser.add_argument(
        "--no-axis-flip",
        action="store_true",
        help="sample only canonical TCP +Y aligned with closed-to-open tube axis",
    )
    parser.add_argument("--workspace-min-m", type=float, nargs=3, default=(0.20, -0.50, 0.04))
    parser.add_argument("--workspace-max-m", type=float, nargs=3, default=(0.90, 0.60, 0.80))
    parser.add_argument(
        "--invoke-plan-only",
        action="store_true",
        help="call MoveIt for planning only; no trajectory can be executed",
    )
    parser.add_argument("--planning-time-s", type=float, default=4.0)
    parser.add_argument("--planning-attempts", type=int, default=3)
    parser.add_argument(
        "--ros-wrapper", type=Path, default=REPOSITORY_ROOT / "scripts/run_ros_python.sh"
    )
    parser.add_argument(
        "--planner-script", type=Path, default=REPOSITORY_ROOT / "scripts/moveit_pose_goal.py"
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"required artifact does not exist: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def main() -> int:
    args = parse_args()
    try:
        relation = read_json(args.current_relation)
        verified_relation_inputs = verify_relation_provenance(
            relation, repository_root=REPOSITORY_ROOT
        )
        T_base_camera = np.load(args.t_base_camera, allow_pickle=False)
        T_link8_tcp = np.load(args.t_link8_tcp, allow_pickle=False)
        manifest = generate_pregrasp_candidates(
            relation,
            T_base_camera,
            approach_distance_m=args.approach_distance_m,
            roll_step_deg=args.roll_step_deg,
            include_tube_axis_flip=not args.no_axis_flip,
            workspace_min_m=args.workspace_min_m,
            workspace_max_m=args.workspace_max_m,
        )
        manifest["provenance"] = {
            "current_relation": artifact(args.current_relation),
            "verified_relation_inputs": verified_relation_inputs,
            "T_base_camera": artifact(args.t_base_camera),
            "T_base_camera_provenance": artifact(args.t_base_camera_provenance),
            "T_link8_tcp": artifact(args.t_link8_tcp),
        }
        manifest["camera_extrinsic_policy"] = {
            "explicit_path_required": True,
            "used_for": "plan-only candidate generation",
            "execution_qualification_inferred": False,
            "note": (
                "Hashing the supplied transform and report establishes provenance; "
                "it does not by itself certify the extrinsic for physical execution."
            ),
        }

        plan_dir = (
            args.planner_output_dir.resolve()
            if args.planner_output_dir is not None
            else args.output.resolve().parent / f"{args.output.stem}_plans"
        )
        plan_dir.mkdir(parents=True, exist_ok=True)
        for candidate in manifest["candidates"]:
            plan_path = plan_dir / f"{candidate['id']}.json"
            command = build_moveit_plan_only_command(
                candidate,
                planner_script=args.planner_script.resolve(),
                ros_wrapper=args.ros_wrapper.resolve(),
                output_path=plan_path,
                T_link8_tcp=T_link8_tcp,
                planning_time_s=args.planning_time_s,
                planning_attempts=args.planning_attempts,
            )
            candidate["moveit_plan_only"] = {
                "command_argv": command,
                "output_path": str(plan_path),
                "invoked": False,
                "returncode": None,
                "result": None,
            }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        assessments: list[dict[str, Any]] = []
        if args.invoke_plan_only:
            for candidate in manifest["candidates"]:
                invocation = candidate["moveit_plan_only"]
                command = invocation["command_argv"]
                if "--execute" in command:
                    raise RuntimeError("refusing a planner command containing --execute")
                completed = subprocess.run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                invocation["invoked"] = True
                invocation["returncode"] = completed.returncode
                invocation["stdout_tail"] = completed.stdout[-4000:]
                invocation["stderr_tail"] = completed.stderr[-4000:]
                plan_path = Path(invocation["output_path"])
                if plan_path.is_file():
                    plan = read_json(plan_path)
                    assessment = assess_moveit_plan(candidate, plan)
                    assessment["plan_artifact"] = artifact(plan_path)
                else:
                    assessment = {
                        "candidate_id": candidate["id"],
                        "result_valid": False,
                        "accepted": False,
                        "moveit_ik_collision_plan_found": False,
                        "trajectory_audit_accepted": False,
                        "rejection_reasons": ["planner produced no JSON result"],
                    }
                invocation["result"] = assessment
                assessments.append(assessment)

            ranked = rank_plan_assessments(assessments)
            manifest["ranked_plan_only_results"] = ranked
            accepted = [item for item in ranked if item.get("accepted") is True]
            manifest["selected_plan_only_candidate_id"] = (
                accepted[0]["candidate_id"] if accepted else None
            )
            manifest["status"] = (
                "moveit_plan_only_candidate_ranked"
                if accepted
                else "moveit_plan_only_no_accepted_candidate"
            )
            manifest["hardware_commands_dispatched"] = False
            manifest["execution_authorized"] = False
            args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "candidate_count": len(manifest["candidates"]),
                    "plan_only_invoked": args.invoke_plan_only,
                    "accepted_plan_count": sum(
                        item.get("accepted") is True for item in assessments
                    ),
                    "selected_plan_only_candidate_id": manifest["selected_plan_only_candidate_id"],
                    "execution_authorized": False,
                    "hardware_commands_dispatched": False,
                },
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"pregrasp preparation failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
