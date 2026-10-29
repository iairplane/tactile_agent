#!/usr/bin/env python3
"""Ground a selected object-frame Hand-E grasp with a full live object pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from demo2gripper.franka import validate_rigid_transform
from demo2gripper.runtime_grasp import (
    make_agent_planned_pick_phases,
    make_object_relative_se3_grasp_candidate,
)
from demo2gripper.task_skill import GraspRelationMode, TaskSkill

LIVE_OBJECT_POSE_SCHEMA = "demo2gripper.live_object_pose.v1"
LIVE_OBJECT_ANCHOR_SCHEMA = "demo2gripper.live_supported_object_anchor.v1"
LIVE_POSE_FRAME_CONVENTION = (
    "T_parent_child, right-handed, column vectors, metres"
)
GROUNDED_CANDIDATE_SCHEMA = "demo2gripper.grounded_live_grasp_candidate.v2"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _diagnostic_trajectory(
    report: dict[str, Any], explicit_path: Path | None, report_path: Path
) -> Path | None:
    """Return a trajectory reference for provenance without opening the artifact."""

    if explicit_path is not None:
        return explicit_path
    trajectory_name = report.get("artifacts", {}).get("trajectory")
    if isinstance(trajectory_name, str) and trajectory_name:
        return report_path.parent / trajectory_name
    return None


def _load_live_object_pose(
    path: Path,
    *,
    npy_object_frame_id: str | None,
    live_object_anchor: Path | None = None,
) -> tuple[np.ndarray, str, np.ndarray, dict[str, Any]]:
    """Load an accepted JSON pose or an explicitly identified raw NPY matrix."""

    if not path.is_file():
        raise ValueError(f"live object pose does not exist: {path}")
    if path.suffix.lower() == ".npy":
        if (
            not isinstance(npy_object_frame_id, str)
            or not npy_object_frame_id.strip()
        ):
            raise ValueError(
                "--object-frame-id is required when --live-object-pose is an NPY"
            )
        transform = validate_rigid_transform(
            np.load(path, allow_pickle=False), name="T_base_object"
        )
        return transform, npy_object_frame_id, transform[:3, 3].copy(), {
            "format": "npy",
            "quality_contract": "explicitly_reviewed_matrix_without_embedded_quality",
        }

    if path.suffix.lower() != ".json":
        raise ValueError("live object pose must be a .json or .npy file")
    payload = _read_json(path)
    if payload.get("schema") != LIVE_OBJECT_POSE_SCHEMA:
        raise ValueError(f"live pose JSON schema must be {LIVE_OBJECT_POSE_SCHEMA!r}")
    if payload.get("frame_convention") != LIVE_POSE_FRAME_CONVENTION:
        raise ValueError(
            "live pose frame_convention must be "
            f"{LIVE_POSE_FRAME_CONVENTION!r}"
        )
    if payload.get("base_frame_id") != "base":
        raise ValueError("live pose base_frame_id must be 'base'")
    object_frame_id = payload.get("object_frame_id")
    if not isinstance(object_frame_id, str) or not object_frame_id.strip():
        raise ValueError("live pose object_frame_id must be a non-empty string")
    if npy_object_frame_id is not None and npy_object_frame_id != object_frame_id:
        raise ValueError(
            "--object-frame-id conflicts with live pose JSON object_frame_id"
        )
    if payload.get("status") != "accepted":
        raise ValueError("live object pose status is not accepted")
    quality = payload.get("quality")
    if not isinstance(quality, dict) or quality.get("accepted") is not True:
        raise ValueError("live object pose did not pass its quality gates")
    if live_object_anchor is None:
        raise ValueError("--live-object-anchor is required for a JSON live object pose")
    anchor_path = live_object_anchor.expanduser().resolve()
    if not anchor_path.is_file():
        raise ValueError(f"live object anchor does not exist: {anchor_path}")
    anchor = _read_json(anchor_path)
    if anchor.get("schema") != LIVE_OBJECT_ANCHOR_SCHEMA:
        raise ValueError(
            f"live object anchor schema must be {LIVE_OBJECT_ANCHOR_SCHEMA!r}"
        )
    if anchor.get("status") != "accepted":
        raise ValueError("live object anchor status is not accepted")
    anchor_quality = anchor.get("quality")
    if not isinstance(anchor_quality, dict) or anchor_quality.get("accepted") is not True:
        raise ValueError("live object anchor did not pass its quality gates")
    anchor_center = np.asarray(anchor.get("center_base_m"), dtype=float)
    if anchor_center.shape != (3,) or not np.isfinite(anchor_center).all():
        raise ValueError("live object anchor center_base_m must be a finite three-vector")

    initialization = payload.get("initialization")
    if not isinstance(initialization, dict):
        raise ValueError("JSON live object pose has no anchor initialization evidence")
    initialization_path = initialization.get("path")
    if not isinstance(initialization_path, str) or (
        Path(initialization_path).expanduser().resolve() != anchor_path
    ):
        raise ValueError("JSON live object pose was not initialized from this live object anchor")
    initialization_center = np.asarray(
        initialization.get("anchor_center_base_m"), dtype=float
    )
    if (
        initialization_center.shape != (3,)
        or not np.isfinite(initialization_center).all()
        or not np.allclose(
            initialization_center, anchor_center, atol=1e-9, rtol=0.0
        )
    ):
        raise ValueError(
            "JSON live object pose anchor center does not match the live object anchor"
        )
    transform = validate_rigid_transform(
        payload.get("T_base_object"), name="T_base_object"
    )
    return transform, object_frame_id, anchor_center, {
        "format": "json",
        "schema": payload["schema"],
        "quality": quality,
        "initialization": initialization,
    }


def _require_matching_object_frame(
    live_object_frame_id: str, selected_object_frame_id: str | None
) -> None:
    if live_object_frame_id != selected_object_frame_id:
        raise ValueError(
            "live pose object_frame_id does not match the selected grasp relation"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--compiler-report", type=Path, required=True)
    parser.add_argument(
        "--live-object-pose",
        type=Path,
        required=True,
        help=(
            "JSON live-pose artifact or NPY rigid T_base_object. JSON must use "
            f"{LIVE_OBJECT_POSE_SCHEMA}; NPY also requires --object-frame-id."
        ),
    )
    parser.add_argument(
        "--object-frame-id",
        help="Required identity for an NPY pose; if supplied for JSON it must match",
    )
    parser.add_argument(
        "--live-object-anchor",
        type=Path,
        help=(
            "Accepted live anchor used to initialize a JSON live object pose. "
            "Required for JSON poses and checked by resolved path and center."
        ),
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        help=(
            "Deprecated demonstration trajectory reference. It is recorded as "
            "diagnostic provenance and is never loaded or used for a motion target."
        ),
    )
    parser.add_argument(
        "--agent-lift-delta-base-m",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.04),
        metavar=("DX", "DY", "DZ"),
        help="Current-scene lift goal relative to the grounded grasp position",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = _read_json(args.compiler_report)
    skill = TaskSkill.from_dict(report["task_skill"])
    relation = skill.grasp_relation
    if relation is None:
        raise ValueError("task skill has no grasp relation")
    if relation.mode is not GraspRelationMode.SE3:
        raise ValueError("live pose grounding only accepts an se3 grasp relation")
    if relation.gripper_name != "robotiq_hande":
        raise ValueError("se3 grasp relation gripper_name must be 'robotiq_hande'")

    (
        T_base_object,
        live_object_frame_id,
        live_object_anchor_base_m,
        pose_evidence,
    ) = _load_live_object_pose(
        args.live_object_pose,
        npy_object_frame_id=args.object_frame_id,
        live_object_anchor=args.live_object_anchor,
    )
    _require_matching_object_frame(live_object_frame_id, relation.object_frame_id)

    trajectory_path = _diagnostic_trajectory(
        report, args.trajectory, args.compiler_report
    )
    candidate = make_object_relative_se3_grasp_candidate(
        T_base_object=T_base_object,
        grasp_relation=relation,
    )
    phases = make_agent_planned_pick_phases(
        candidate,
        agent_lift_delta_base_m=np.asarray(
            args.agent_lift_delta_base_m, dtype=float
        ),
    )

    payload = {
        "schema": GROUNDED_CANDIDATE_SCHEMA,
        "status": "review_only",
        "authorized_for_robot_execution": False,
        "frame_convention": LIVE_POSE_FRAME_CONVENTION,
        "base_frame_id": "base",
        "object_frame_id": live_object_frame_id,
        "gripper_name": relation.gripper_name,
        "T_base_object": T_base_object.tolist(),
        "live_object_anchor_base_m": live_object_anchor_base_m.tolist(),
        "candidate": candidate.to_dict(),
        "phase_targets": {
            "APPROACH": {
                "planner_mode": "motion_planning",
                "poses_base": phases.approach_poses_base.tolist(),
                "source": candidate.approach_source.value,
                "target_policy": "selected_object_relative_se3",
            },
            "GRASP": {
                "planner_mode": "gripper_action",
                "T_base_gripper_tcp": candidate.T_base_grasp_tcp.tolist(),
                "source": "selected_object_relative_se3",
                "target_policy": "object_relative_grasp",
                "endpoint_immutable": True,
            },
            "LIFT": {
                "planner_mode": "motion_planning",
                "positions_base_m": phases.lift_positions_base_m.tolist(),
                "orientation_policy": "hold_grasp_orientation",
                "source": "agent_selected",
                "target_policy": "online_agent_plan",
            },
        },
        "evidence": {
            "grasp_relation_mode": relation.mode.value,
            "selected_grasp_frame_validated": True,
            "object_frame_id_matched": True,
            "selected_approach_consumed": True,
            "agent_approach_override_allowed": False,
            "pregrasp_direction": "selected_gripper_tcp_local_minus_z",
            "agent_lift_delta_base_m": list(args.agent_lift_delta_base_m),
            "live_pose": pose_evidence,
        },
        "limitations": [
            "The poses use the canonical Hand-E TCP; a measured command-frame mapping is required.",
            (
                "The selected endpoint still requires IK, collision, support-clearance, "
                "and trajectory checks downstream."
            ),
        ],
        "provenance": {
            "compiler_report": str(args.compiler_report.resolve()),
            "live_object_pose": str(args.live_object_pose.resolve()),
            "live_object_anchor": (
                None
                if args.live_object_pose.suffix.lower() == ".npy"
                else str(args.live_object_anchor.expanduser().resolve())
            ),
            "demonstration_trajectory": {
                "path": (
                    None if trajectory_path is None else str(trajectory_path.resolve())
                ),
                "role": "diagnostic_only",
                "consumed": False,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    xyz = candidate.T_base_grasp_tcp[:3, 3]
    print(f"approach source: {candidate.approach_source.value}")
    print(f"grasp TCP xyz: {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
