#!/usr/bin/env python3
"""Build the review-only open-lid/pick/place stage graph for demo 03.

This script only writes JSON.  It never imports ROS, opens the gripper, or sends a
robot command.  The graph binds each demonstrated intent to an existing planning
primitive where one exists, and leaves missing runtime geometry fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "demo2gripper.demo03_multicontact_stage_graph.v1"
STAGE_SPLIT_SCHEMA = "demo2gripper.demonstration_stage_split.v1"
EXPECTED_DEMO_STAGES = (
    "IDLE_INITIAL",
    "OBSERVE_INTENT",
    "APPROACH_LID",
    "LID_CONTACT",
    "LID_OPEN",
    "LID_RELEASE_TRANSITION",
    "APPROACH_AND_GRASP_DIE",
    "TRANSPORT_DIE",
    "PLACE_IN_BOX",
    "RELEASE_DIE",
    "RETREAT",
    "VERIFY_DONE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--demo-dir", required=True, type=Path)
    parser.add_argument(
        "--stage-split",
        type=Path,
        help="Reviewed stage split (defaults to <demo-dir>/stage_split.json)",
    )
    parser.add_argument(
        "--t-base-camera",
        type=Path,
        default=Path("calibration/T_base_camera.npy"),
    )
    parser.add_argument(
        "--lid-waypoint-dir",
        type=Path,
        help=(
            "Directory produced by build_demo02_lid_relative_waypoints.py "
            "(defaults to <demo-dir>/lid_relative_current_geometry_v1)"
        ),
    )
    parser.add_argument(
        "--reverse-pick-dir",
        type=Path,
        help=(
            "Reverse-placement pickup proxy directory (defaults to the reviewed "
            "demo-03 derived slice)"
        ),
    )
    parser.add_argument(
        "--die-pick-config",
        type=Path,
        help="Resolved run_autonomous_pick.py config for the live die",
    )
    parser.add_argument(
        "--die-pick-contract",
        type=Path,
        help="Minimal TaskSkill v3/compiler report carrying the selected SE(3) relation",
    )
    parser.add_argument(
        "--die-pick-config-template",
        type=Path,
        help="Non-runnable config template awaiting fresh live workspace bounds",
    )
    parser.add_argument(
        "--selected-die-grasp",
        type=Path,
        help="Reviewed Hand-E grasp selected after GraspGenX ranking",
    )
    parser.add_argument(
        "--box-placement-relation",
        type=Path,
        help="Reviewed die-to-box placement relation used for live grounding",
    )
    parser.add_argument(
        "--placement-waypoint-dir",
        type=Path,
        help="Directory containing transport_tcp.json, placement_tcp.json, retreat_tcp.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON (defaults to <demo-dir>/multicontact_plan/stage_graph.json)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: Path, *, repo_root: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (repo_root / expanded).resolve()


def _display(path: Path, *, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def _validate_stage_split(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if payload.get("schema") != STAGE_SPLIT_SCHEMA:
        raise ValueError(f"stage split schema must be {STAGE_SPLIT_SCHEMA!r}")
    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise ValueError("stage split stages must be an array")

    indexed: dict[str, dict[str, Any]] = {}
    previous_end = 0
    for item in stages:
        if not isinstance(item, Mapping):
            raise ValueError("each stage split entry must be an object")
        name = item.get("name")
        frames = item.get("frames")
        if not isinstance(name, str) or not name:
            raise ValueError("each stage split entry needs a non-empty name")
        if name in indexed:
            raise ValueError(f"duplicate demonstrated stage {name!r}")
        if (
            not isinstance(frames, list)
            or len(frames) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in frames)
            or frames[0] < 0
            or frames[1] <= frames[0]
        ):
            raise ValueError(f"stage {name!r} frames must be a non-empty half-open interval")
        if frames[0] < previous_end:
            raise ValueError("demonstrated stage intervals must not overlap or run backward")
        previous_end = frames[1]
        indexed[name] = dict(item)

    if tuple(indexed) != EXPECTED_DEMO_STAGES:
        raise ValueError(
            f"demo-03 stage split does not match the reviewed stage sequence: {tuple(indexed)!r}"
        )
    return indexed


def _evidence(stages: Mapping[str, Mapping[str, Any]], names: Sequence[str]) -> dict[str, Any]:
    intervals = [stages[name]["frames"] for name in names]
    return {
        "demonstrated_stages": list(names),
        "frame_intervals": intervals,
        "source_policies": [stages[name].get("policy") for name in names],
    }


def _artifact(
    path: Path,
    *,
    repo_root: Path,
    role: str,
    producer: str,
) -> dict[str, Any]:
    return {
        "path": _display(path, repo_root=repo_root),
        "role": role,
        "producer": producer,
        "status": "ready" if path.is_file() else "missing",
    }


def _stage(
    name: str,
    *,
    group: str,
    evidence: dict[str, Any],
    target_policy: str,
    planner: str,
    inputs: Sequence[str],
    preconditions: Sequence[str],
    completion: Sequence[str],
    failure_policy: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "group": group,
        "demo_evidence": evidence,
        "target_policy": target_policy,
        "planner_or_action": planner,
        "artifact_inputs": list(inputs),
        "preconditions": list(preconditions),
        "completion": list(completion),
        "failure_policy": failure_policy,
    }


def build_graph(
    *,
    repo_root: Path,
    demo_dir: Path,
    stage_split_path: Path,
    t_base_camera: Path,
    lid_waypoint_dir: Path,
    reverse_pick_dir: Path,
    selected_die_grasp: Path,
    die_pick_contract: Path,
    die_pick_config_template: Path,
    die_pick_config: Path,
    box_placement_relation: Path,
    placement_waypoint_dir: Path,
) -> dict[str, Any]:
    split = _read_json(stage_split_path)
    demonstrated = _validate_stage_split(split)

    artifacts = {
        "T_base_camera": _artifact(
            t_base_camera,
            repo_root=repo_root,
            role="current fixed-camera eye-to-hand calibration",
            producer="scripts/solve_eye_to_hand.py",
        ),
        "lid_relation": _artifact(
            lid_waypoint_dir / "lid_interaction_relation.json",
            repo_root=repo_root,
            role="demonstrated TCP pose relative to the closed lid contact frame",
            producer="scripts/build_demo02_lid_relative_waypoints.py",
        ),
        "lid_precontact_tcp": _artifact(
            lid_waypoint_dir / "lid_precontact_tcp.json",
            repo_root=repo_root,
            role="absolute current-scene lid precontact TCP",
            producer="scripts/build_demo02_lid_relative_waypoints.py",
        ),
        "lid_contact_tcp": _artifact(
            lid_waypoint_dir / "lid_contact_tcp.json",
            repo_root=repo_root,
            role="absolute current-scene lid contact TCP",
            producer="scripts/build_demo02_lid_relative_waypoints.py",
        ),
        "lid_open_waypoints": _artifact(
            lid_waypoint_dir / "lid_open_waypoints.json",
            repo_root=repo_root,
            role="hinge-constrained configured-EE TCP sequence",
            producer="scripts/build_demo02_lid_relative_waypoints.py",
        ),
        "lid_retreat_tcp": _artifact(
            lid_waypoint_dir / "lid_open_release_retreat_tcp.json",
            repo_root=repo_root,
            role="post-release retreat TCP grounded at the open-lid endpoint",
            producer="scripts/build_demo02_lid_relative_waypoints.py plus demo-03 open-lid policy",
        ),
        "lid_retreat_waypoints": _artifact(
            lid_waypoint_dir / "lid_open_release_retreat_waypoints.json",
            repo_root=repo_root,
            role="bounded +Z retreat sequence that leaves the lid open",
            producer="scripts/build_demo02_lid_relative_waypoints.py",
        ),
        "reverse_pickup_result": _artifact(
            reverse_pick_dir / "pickup_result" / "pickup_grasp.json",
            repo_root=repo_root,
            role=(
                "labelled reverse-placement proxy for die/hand relation; it is not "
                "evidence that the original pickup contact was visible"
            ),
            producer="scripts/infer_pickup_demo.py",
        ),
        "robot_adapted_die_intent": _artifact(
            reverse_pick_dir / "hande_intent.json",
            repo_root=repo_root,
            role=(
                "ranking-only Hand-E intent prior derived from the reverse proxy; "
                "execution_consumable=false"
            ),
            producer="scripts/adapt_human_intent_to_hande.py",
        ),
        "selected_die_grasp": _artifact(
            selected_die_grasp,
            repo_root=repo_root,
            role=(
                "review-only robot grasp ranked from the labelled reverse-placement "
                "proxy; not an observed original pickup"
            ),
            producer="GraspGenX candidate generation plus demo2gripper selection",
        ),
        "die_pick_contract": _artifact(
            die_pick_contract,
            repo_root=repo_root,
            role=(
                "TaskSkill v3/compiler report carrying only the selected object-relative "
                "SE(3) relation; no demonstration path is executable"
            ),
            producer="scripts/build_demo03_die_pick_contract.py",
        ),
        "die_pick_config_template": _artifact(
            die_pick_config_template,
            repo_root=repo_root,
            role="non-runnable narrow-pick config awaiting fresh workspace/support bounds",
            producer="scripts/build_demo03_die_pick_contract.py",
        ),
        "die_pick_config": _artifact(
            die_pick_config,
            repo_root=repo_root,
            role="live-scene grounding and plan-only config for the narrow pick runner",
            producer="manual review plus scripts/run_autonomous_pick.py contract",
        ),
        "box_placement_relation": _artifact(
            box_placement_relation,
            repo_root=repo_root,
            role="fixed-scene final die translation anchor inside/on the box bottom",
            producer="scripts/build_demo03_fixed_scene_placement.py",
        ),
        "transport_tcp": _artifact(
            placement_waypoint_dir / "transport_tcp.json",
            repo_root=repo_root,
            role="agent-planned carry pose above the grounded box interior",
            producer="runtime placement grounding (not implemented yet)",
        ),
        "placement_tcp": _artifact(
            placement_waypoint_dir / "placement_tcp.json",
            repo_root=repo_root,
            role="box-relative on-bottom placement TCP",
            producer="runtime placement grounding (not implemented yet)",
        ),
        "retreat_tcp": _artifact(
            placement_waypoint_dir / "retreat_tcp.json",
            repo_root=repo_root,
            role="agent-planned collision-free TCP after release",
            producer="runtime placement grounding (not implemented yet)",
        ),
    }

    # Each row is name, group, evidence stages, target policy, planner/action,
    # artifact inputs, preconditions, completion predicates, and failure policy.
    stage_specs = [
        (
            "OBSERVE_CLOSED_SCENE",
            "OPEN_LID",
            ("OBSERVE_INTENT",),
            "agentic_scene_and_intent",
            "read-only RGB-D plus robot-state observation",
            ("T_base_camera",),
            ("lid_closed", "die_outside_box", "workspace_clear"),
            ("lid_and_die_anchors_stable", "robot_state_fresh"),
            "reobserve_or_stop",
        ),
        (
            "LID_PRECONTACT",
            "OPEN_LID",
            ("APPROACH_LID",),
            "online_agent_plan",
            "scripts/moveit_pose_goal.py (plan-only first)",
            ("lid_precontact_tcp",),
            ("lid_anchor_fresh", "gripper_open", "workspace_clear"),
            ("audited_plan_exists", "measured_tcp_at_precontact_when_executed"),
            "stop_and_reobserve",
        ),
        (
            "LID_CONTACT",
            "OPEN_LID",
            ("LID_CONTACT",),
            "relative_pose_transfer",
            "scripts/moveit_pose_goal.py (plan-only from measured precontact)",
            ("lid_relation", "lid_contact_tcp"),
            ("measured_tcp_at_precontact", "lid_anchor_unchanged"),
            ("measured_tcp_at_contact",),
            "retreat_then_stop",
        ),
        (
            "LID_GRIP",
            "OPEN_LID",
            ("LID_CONTACT",),
            "task_intent",
            "scripts/hande_cli.py close",
            (),
            ("measured_tcp_at_contact",),
            ("hande_gOBJ_2", "hande_fault_0", "lid_contact_width_in_range"),
            "open_retreat_then_stop",
        ),
        (
            "LID_OPEN",
            "OPEN_LID",
            ("LID_OPEN",),
            "relative_constrained_motion",
            "scripts/moveit_cartesian_pose_sequence.py (plan-only first)",
            ("lid_relation", "lid_open_waypoints"),
            ("lid_contact_verified", "measured_tcp_at_contact"),
            ("audited_path_fraction_1", "lid_open", "lid_contact_maintained"),
            "hold_position_and_stop",
        ),
        (
            "LID_RELEASE_AND_RETREAT",
            "OPEN_LID",
            ("LID_RELEASE_TRANSITION",),
            "task_intent_then_online_agent_plan",
            "scripts/hande_cli.py open; scripts/moveit_cartesian_pose_sequence.py",
            ("lid_retreat_tcp", "lid_retreat_waypoints"),
            ("lid_open", "robot_motion_stopped"),
            ("gripper_open", "tcp_clear_of_lid", "lid_remains_open"),
            "stop_without_additional_motion",
        ),
        (
            "REOBSERVE_DIE",
            "PICK_DIE",
            ("APPROACH_AND_GRASP_DIE",),
            "agentic_live_grounding",
            "scripts/run_autonomous_pick.py OBSERVE/GROUND/PLAN prefix",
            (
                "T_base_camera",
                "die_pick_contract",
                "die_pick_config",
                "robot_adapted_die_intent",
                "selected_die_grasp",
            ),
            ("lid_open", "tcp_clear_of_camera_view", "die_visible"),
            ("stable_live_die_anchor", "audited_pregrasp_plan"),
            "reobserve_replan_or_stop",
        ),
        (
            "PICK_DIE",
            "PICK_DIE",
            ("APPROACH_AND_GRASP_DIE",),
            "live_object_relative_grasp_with_proxy_prior",
            "scripts/run_autonomous_pick.py PREGRASP/DESCEND/CLOSE/VERIFY_CONTACT/LIFT",
            (
                "die_pick_config",
                "die_pick_contract",
                "reverse_pickup_result",
                "robot_adapted_die_intent",
                "selected_die_grasp",
            ),
            ("fresh_die_grounding", "audited_pregrasp_plan", "workspace_clear"),
            ("die_contact_verified", "die_held_after_short_lift"),
            "existing_pick_recovery_then_stop",
        ),
        (
            "TRANSPORT_DIE",
            "PLACE_IN_BOX",
            ("TRANSPORT_DIE",),
            "online_agent_plan",
            "scripts/moveit_pose_goal.py (plan-only from measured post-lift state)",
            ("transport_tcp", "box_placement_relation"),
            ("die_held", "fixed_scene_box_unchanged", "lid_open"),
            ("measured_tcp_at_transport", "die_hold_width_stable"),
            "hold_position_and_stop",
        ),
        (
            "PLACE_IN_BOX",
            "PLACE_IN_BOX",
            ("PLACE_IN_BOX",),
            "relative_pose_transfer",
            "scripts/moveit_cartesian_smoke.py segments (plan-only first)",
            ("box_placement_relation", "placement_tcp"),
            ("die_held", "measured_tcp_at_transport", "lid_open"),
            ("tcp_at_placement", "die_inside_box", "die_supported_by_box_bottom"),
            "hold_position_and_stop",
        ),
        (
            "RELEASE_DIE",
            "PLACE_IN_BOX",
            ("RELEASE_DIE",),
            "task_predicate_then_gripper_open",
            "scripts/hande_cli.py open",
            (),
            ("die_inside_box", "die_supported_by_box_bottom", "robot_motion_stopped"),
            ("gripper_open", "die_no_longer_held"),
            "hold_position_and_stop",
        ),
        (
            "RETREAT",
            "RETREAT",
            ("RETREAT",),
            "online_agent_plan",
            "scripts/moveit_pose_goal.py (plan-only from measured release state)",
            ("retreat_tcp",),
            ("die_released", "lid_open"),
            ("tcp_clear_of_box", "hand_clear"),
            "stop_without_additional_motion",
        ),
        (
            "VERIFY_DONE",
            "VERIFY",
            ("VERIFY_DONE",),
            "visual_predicate",
            "read-only RGB-D verification",
            ("T_base_camera",),
            ("robot_motion_stopped",),
            ("lid_open", "die_inside_box", "hand_clear"),
            "report_incomplete_and_stop",
        ),
    ]
    stages = [
        _stage(
            name,
            group=group,
            evidence=_evidence(demonstrated, evidence_names),
            target_policy=target_policy,
            planner=planner,
            inputs=inputs,
            preconditions=preconditions,
            completion=completion,
            failure_policy=failure_policy,
        )
        for (
            name,
            group,
            evidence_names,
            target_policy,
            planner,
            inputs,
            preconditions,
            completion,
            failure_policy,
        ) in stage_specs
    ]

    missing_artifacts = [name for name, item in artifacts.items() if item["status"] == "missing"]
    blockers = [f"missing_artifact:{name}" for name in missing_artifacts]
    blockers.extend(
        (
            "open_lid_before_pick_orchestrator_not_implemented",
            "fixed_scene_box_unchanged_confirmation_pending",
            "hande_box_collision_geometry_not_in_moveit_model",
        )
    )
    return {
        "schema": SCHEMA,
        "status": "review_only",
        "execution_authorized": False,
        "robot_execution_ready": False,
        "source": {
            "demo_dir": _display(demo_dir, repo_root=repo_root),
            "stage_split": _display(stage_split_path, repo_root=repo_root),
            "stage_split_sha256": _sha256(stage_split_path),
            "task_intent": split.get("task_intent"),
        },
        "contracts": {
            "transform": "T_base_tcp = T_base_anchor_live @ T_anchor_tcp_demo",
            "human_path_role": "segmentation_and_intent_evidence_only; never replayed",
            "planning": "incremental_plan_from_measured_state_after_each_verified_stage",
            "agentic_stages": (
                "scene observation, free-space approach, reobservation, transport, and retreat"
            ),
            "relative_stages": (
                "lid contact/opening, die grasp endpoint, and die-to-box placement endpoint"
            ),
            "reverse_proxy_limit": (
                "the time-reversed placement can supply a labelled grasp prior only; the "
                "original pickup contact occurred outside the image"
            ),
        },
        "stage_order": [stage["name"] for stage in stages],
        "stages": stages,
        "artifacts": artifacts,
        "existing_components": {
            "lid_waypoint_compiler": "scripts/build_demo02_lid_relative_waypoints.py",
            "absolute_pose_planner": "scripts/moveit_pose_goal.py",
            "cartesian_sequence_planner": "scripts/moveit_cartesian_pose_sequence.py",
            "bounded_cartesian_delta_planner": "scripts/moveit_cartesian_smoke.py",
            "narrow_pick_runner": "scripts/run_autonomous_pick.py",
            "die_pick_contract_builder": "scripts/build_demo03_die_pick_contract.py",
            "gripper_driver": "scripts/hande_cli.py",
            "predicate_and_report_reference": "scripts/run_demo02_full_task.py",
        },
        "missing_executor_work": [
            (
                "Compose the audited primitives in this graph order. The existing "
                "run_demo02_full_task.py is fixed to pick/place first and lid motion later."
            ),
            (
                "Fill die_pick_config.template.json from a fresh post-lid-opening "
                "workspace/support observation, save die_pick_config.json, and pass a "
                "plan-only narrow-pick run before any execution."
            ),
            (
                "After the fresh live pick is grounded, call "
                "build_demo03_fixed_scene_placement.py with its execution_candidate.json to "
                "emit transport, placement, and retreat TCP artifacts."
            ),
            (
                "Carry object/lid state predicates and atomic stage evidence through one "
                "open-first end-to-end report before enabling physical execution."
            ),
        ],
        "blocking_conditions": blockers,
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    demo_dir = _resolve(args.demo_dir, repo_root=repo_root)
    stage_split = _resolve(args.stage_split or demo_dir / "stage_split.json", repo_root=repo_root)
    t_base_camera = _resolve(args.t_base_camera, repo_root=repo_root)
    lid_waypoint_dir = _resolve(
        args.lid_waypoint_dir or demo_dir / "lid_relative_current_geometry_v1",
        repo_root=repo_root,
    )
    reverse_pick_dir = _resolve(
        args.reverse_pick_dir or demo_dir / "derived" / "reverse_pick_proxy_0650_to_0440",
        repo_root=repo_root,
    )
    plan_input_dir = demo_dir / "multicontact_plan" / "inputs"
    die_pick_config = _resolve(
        args.die_pick_config or plan_input_dir / "die_pick_config.json",
        repo_root=repo_root,
    )
    die_pick_contract = _resolve(
        args.die_pick_contract
        or plan_input_dir / "die_pick_contract" / "compiler_report.json",
        repo_root=repo_root,
    )
    die_pick_config_template = _resolve(
        args.die_pick_config_template or plan_input_dir / "die_pick_config.template.json",
        repo_root=repo_root,
    )
    selected_die_grasp = _resolve(
        args.selected_die_grasp or reverse_pick_dir / "graspgenx_hande" / "selected_grasp.json",
        repo_root=repo_root,
    )
    box_relation = _resolve(
        args.box_placement_relation or plan_input_dir / "box_placement_relation.json",
        repo_root=repo_root,
    )
    placement_dir = _resolve(
        args.placement_waypoint_dir or plan_input_dir / "placement",
        repo_root=repo_root,
    )
    output = _resolve(
        args.output or demo_dir / "multicontact_plan" / "stage_graph.json",
        repo_root=repo_root,
    )
    if not demo_dir.is_dir():
        raise ValueError(f"demo directory does not exist: {demo_dir}")
    if not stage_split.is_file():
        raise ValueError(f"stage split does not exist: {stage_split}")
    if output.exists() and not args.overwrite:
        raise ValueError(f"output already exists; pass --overwrite to replace it: {output}")

    graph = build_graph(
        repo_root=repo_root,
        demo_dir=demo_dir,
        stage_split_path=stage_split,
        t_base_camera=t_base_camera,
        lid_waypoint_dir=lid_waypoint_dir,
        reverse_pick_dir=reverse_pick_dir,
        selected_die_grasp=selected_die_grasp,
        die_pick_contract=die_pick_contract,
        die_pick_config_template=die_pick_config_template,
        die_pick_config=die_pick_config,
        box_placement_relation=box_relation,
        placement_waypoint_dir=placement_dir,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ready = sum(item["status"] == "ready" for item in graph["artifacts"].values())
    total = len(graph["artifacts"])
    print(f"wrote {output}")
    print(f"artifact readiness: {ready}/{total}; status={graph['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
