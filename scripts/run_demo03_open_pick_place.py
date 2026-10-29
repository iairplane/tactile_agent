#!/usr/bin/env python3
"""Run demo-03 in the demonstrated open-lid, pick, and place order.

The lid is opened with the reviewed connected two-stage plan: grasp at the
closed pose, open to 61 degrees, release/retreat, reconfigure through clearance,
regrasp, then open to 100 degrees.  Die reobservation, pick, and fixed-scene
placement follow only after both lid stages pass.

Planning/review is the default.  Physical commands require ``--execute`` plus
the explicit arm-only collision-model and fixed-scene acknowledgements.  Every
arm motion is delegated to the existing audited MoveIt tools; this script does
not emit joint commands or replay the demonstrated human trajectory.
"""

from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import hashlib
import json
import math
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from demo2gripper.autonomous_pick import (
    atomic_write_json,
    hande_open_succeeded,
    hande_width_snapshot_valid,
    pose_result_succeeded,
)
from demo2gripper.cartesian_pose_sequence import (
    MAX_CARTESIAN_SPEED_M_S,
    CartesianPose,
    load_pose_sequence,
    validate_pose_sequence_motion,
)
from scripts.build_demo03_fixed_scene_placement import (
    GROUNDING_SCHEMA,
    RELATION_SCHEMA,
    ground_waypoints,
)
from scripts.run_autonomous_pick import (
    StageFailure,
    _read_json,
    _validate_config as _validate_pick_config,
)
from scripts.run_demo02_full_task import (
    BASE_FRAME,
    _artifact_summary,
    _lid_contact_verified,
    _lid_width_snapshot_valid,
    _pick_command,
    _pick_report_ready,
    _pose_command,
    _poses_match,
    _run_hande,
    _run_pose,
    _run_process,
    _run_sequence,
    _single_pose,
)

SCHEMA = "demo2gripper.demo03_open_pick_place_config.v1"
REPORT_SCHEMA = "demo2gripper.demo03_open_pick_place_report.v1"
LID_MANIFEST_SCHEMA = "demo2gripper.demo03_lid_two_stage_waypoints.v1"
LID_CONNECTED_PROOF_SCHEMA = "demo2gripper.demo03_lid_two_stage_connected_plan_only.v1"

LID_ARTIFACT_NAMES = {
    "connected_plan_only": "connected_plan_only.json",
    "stage1_precontact_tcp": "stage1_precontact_tcp.json",
    "stage1_contact_waypoints": "stage1_contact_approach_waypoints.json",
    "stage1_contact_tcp": "stage1_contact_tcp.json",
    "stage1_open_waypoints": "stage1_open_to_61deg_waypoints.json",
    "stage1_retreat_waypoints": "stage1_release_retreat_waypoints.json",
    "stage2_free_space_waypoints": "stage2_regrasp_free_space_approach_waypoints.json",
    "stage2_precontact_tcp": "stage2_regrasp_precontact_tcp.json",
    "stage2_contact_waypoints": "stage2_regrasp_approach_waypoints.json",
    "stage2_contact_tcp": "stage2_regrasp_contact_tcp.json",
    "stage2_open_waypoints": "stage2_open_61_to_100deg_waypoints.json",
    "stage2_retreat_waypoints": "stage2_release_retreat_waypoints.json",
}


class Demo03Stage(StrEnum):
    LID_STAGE1_PRECONTACT = "LID_STAGE1_PRECONTACT"
    LID_STAGE1_CONTACT = "LID_STAGE1_CONTACT"
    LID_STAGE1_GRIP = "LID_STAGE1_GRIP"
    LID_STAGE1_OPEN_61 = "LID_STAGE1_OPEN_61"
    LID_STAGE1_RELEASE = "LID_STAGE1_RELEASE"
    LID_STAGE1_RETREAT = "LID_STAGE1_RETREAT"
    LID_STAGE1_TRANSITION_GATE = "LID_STAGE1_TRANSITION_GATE"
    LID_STAGE2_CLEARANCE_TRANSITION = "LID_STAGE2_CLEARANCE_TRANSITION"
    LID_STAGE2_PRECONTACT = "LID_STAGE2_PRECONTACT"
    LID_STAGE2_CONTACT = "LID_STAGE2_CONTACT"
    LID_STAGE2_GRIP = "LID_STAGE2_GRIP"
    LID_STAGE2_OPEN_100 = "LID_STAGE2_OPEN_100"
    LID_STAGE2_RELEASE = "LID_STAGE2_RELEASE"
    LID_STAGE2_RETREAT = "LID_STAGE2_RETREAT"
    REOBSERVE_DIE = "REOBSERVE_DIE"
    PICK_AND_LIFT = "PICK_AND_LIFT"
    DERIVE_PLACEMENT = "DERIVE_PLACEMENT"
    TRANSPORT = "TRANSPORT"
    PLACE = "PLACE"
    RELEASE_DIE = "RELEASE_DIE"
    RETREAT = "RETREAT"


STAGE_ORDER = tuple(stage.value for stage in Demo03Stage)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help="run the non-moving review preflight (the default)",
    )
    mode.add_argument("--execute", action="store_true", help="run the physical state machine")
    parser.add_argument(
        "--pick-input-dir",
        type=Path,
        help="existing RGB-D input for plan-only review; execution always reobserves live",
    )
    parser.add_argument(
        "--accept-arm-only-collision-model",
        action="store_true",
        help="acknowledge that the installed Hand-E is absent from the MoveIt model",
    )
    parser.add_argument(
        "--accept-fixed-scene-placement",
        action="store_true",
        help="confirm that the fixed camera and box have not moved since the demonstration",
    )
    parser.add_argument(
        "--accept-unobserved-lid-transition",
        action="store_true",
        help=(
            "continue to the second lid grasp using the measured 61-degree endpoint even "
            "though the released lid angle is not reobserved"
        ),
    )
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
    if not isinstance(commands, Mapping):
        raise ValueError("config.commands must be an object")
    value = commands.get(name)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(token, str) or not token for token in value)
    ):
        raise ValueError(f"config.commands.{name} must be a non-empty string array")
    return list(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_bounded(value: Any, *, name: str, maximum: float, inclusive: bool = True) -> float:
    result = float(value)
    above_maximum = result > maximum if inclusive else result >= maximum
    if not math.isfinite(result) or result <= 0.0 or above_maximum:
        bracket = "]" if inclusive else ")"
        raise ValueError(f"{name} must lie in (0, {maximum}{bracket}")
    return result


def _validate_relation(
    payload: Mapping[str, Any], *, path: Path, t_base_camera: Path
) -> np.ndarray:
    if payload.get("schema") != RELATION_SCHEMA:
        raise ValueError(f"placement relation schema must be {RELATION_SCHEMA!r}")
    if payload.get("status") != "accepted_fixed_scene_translation_only":
        raise ValueError("placement relation was not accepted")
    if payload.get("relation_mode") != "fixed_scene_translation_only":
        raise ValueError("placement relation must use fixed_scene_translation_only")
    if payload.get("base_frame") != BASE_FRAME:
        raise ValueError(f"placement relation base_frame must be {BASE_FRAME!r}")
    quality = payload.get("quality")
    if not isinstance(quality, Mapping) or quality.get("accepted") is not True:
        raise ValueError("placement relation quality gate did not pass")
    anchor = np.asarray(payload.get("final_die_anchor_base_m"), dtype=float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("placement relation final anchor must contain three finite values")
    provenance = payload.get("provenance")
    expected_hash = (
        provenance.get("T_base_camera_sha256") if isinstance(provenance, Mapping) else None
    )
    if not isinstance(expected_hash, str) or expected_hash != _sha256(t_base_camera):
        raise ValueError(
            f"{path} was compiled with a different T_base_camera; fixed-scene placement is invalid"
        )
    return anchor


def _validate_config(
    config: Mapping[str, Any], *, config_path: Path, execute: bool
) -> dict[str, Any]:
    if config.get("schema") != SCHEMA:
        raise ValueError(f"config schema must be {SCHEMA!r}")
    base = config_path.parent.resolve()
    pick_config_path = _resolve_path(config.get("pick_config"), base=base, name="pick_config")
    if not pick_config_path.is_file():
        raise ValueError(f"missing pick_config: {pick_config_path}")
    pick = _validate_pick_config(
        _read_json(pick_config_path), config_path=pick_config_path, execute=execute
    )

    lid = config.get("lid_two_stage")
    placement = config.get("placement")
    motion = config.get("motion", {})
    if not isinstance(lid, Mapping):
        raise ValueError("config.lid_two_stage must be an object")
    if "lid" in config:
        raise ValueError("single-stage config.lid is unsupported; use lid_two_stage")
    if not isinstance(placement, Mapping):
        raise ValueError("config.placement must be an object")
    if not isinstance(motion, Mapping):
        raise ValueError("config.motion must be an object")

    manifest_path = _resolve_path(
        lid.get("manifest"), base=base, name="lid_two_stage.manifest"
    )
    lid_paths = {
        name: _resolve_path(lid.get(name), base=base, name=f"lid_two_stage.{name}")
        for name in LID_ARTIFACT_NAMES
    }
    placement_relation_path = _resolve_path(
        placement.get("relation"), base=base, name="placement.relation"
    )
    for name, artifact_path in {
        "lid_two_stage.manifest": manifest_path,
        **{f"lid_two_stage.{name}": value for name, value in lid_paths.items()},
        "placement.relation": placement_relation_path,
    }.items():
        if not artifact_path.is_file():
            raise ValueError(f"missing {name}: {artifact_path}")

    manifest = _read_json(manifest_path)
    connected_proof = _read_json(lid_paths["connected_plan_only"])
    if manifest.get("schema") != LID_MANIFEST_SCHEMA:
        raise ValueError(f"lid manifest schema must be {LID_MANIFEST_SCHEMA!r}")
    if manifest.get("execution", {}).get("dispatched") is not False:
        raise ValueError("lid manifest must be an offline, non-executed artifact")
    if connected_proof.get("schema") != LID_CONNECTED_PROOF_SCHEMA:
        raise ValueError(f"connected proof schema must be {LID_CONNECTED_PROOF_SCHEMA!r}")
    rejected_direct = connected_proof.get("rejected_direct_transition_diagnostic")
    if (
        connected_proof.get("passed") is not True
        or connected_proof.get("execution_dispatched") is not False
        or connected_proof.get("stage1_end_angle_deg") != 61
        or connected_proof.get("stage2_release_angle_deg") != 100
        or not isinstance(rejected_direct, Mapping)
        or rejected_direct.get("passed") is not False
    ):
        raise ValueError("connected two-stage lid proof did not pass its offline gates")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("lid manifest has no output checksums")
    for config_name, expected_basename in LID_ARTIFACT_NAMES.items():
        artifact_path = lid_paths[config_name]
        if artifact_path.parent != manifest_path.parent:
            raise ValueError("all lid two-stage artifacts must come from the manifest directory")
        if artifact_path.name != expected_basename:
            raise ValueError(
                f"lid_two_stage.{config_name} must name {expected_basename!r}"
            )
        record = outputs.get(expected_basename)
        if (
            not isinstance(record, Mapping)
            or record.get("sha256") != _sha256(artifact_path)
        ):
            raise ValueError(f"lid artifact checksum mismatch: {artifact_path}")

    stage1_precontact = _single_pose(
        lid_paths["stage1_precontact_tcp"], name="lid stage1 precontact"
    )
    stage1_contact = _single_pose(
        lid_paths["stage1_contact_tcp"], name="lid stage1 contact"
    )
    stage1_contact_path = load_pose_sequence(
        lid_paths["stage1_contact_waypoints"], expected_base_frame=BASE_FRAME
    )
    stage1_open = load_pose_sequence(
        lid_paths["stage1_open_waypoints"], expected_base_frame=BASE_FRAME
    )
    stage1_retreat = load_pose_sequence(
        lid_paths["stage1_retreat_waypoints"], expected_base_frame=BASE_FRAME
    )
    stage2_free_space_payload = _read_json(lid_paths["stage2_free_space_waypoints"])
    stage2_free_space = load_pose_sequence(
        lid_paths["stage2_free_space_waypoints"], expected_base_frame=BASE_FRAME
    )
    stage2_precontact = _single_pose(
        lid_paths["stage2_precontact_tcp"], name="lid stage2 precontact"
    )
    stage2_contact = _single_pose(
        lid_paths["stage2_contact_tcp"], name="lid stage2 contact"
    )
    stage2_contact_path = load_pose_sequence(
        lid_paths["stage2_contact_waypoints"], expected_base_frame=BASE_FRAME
    )
    stage2_open = load_pose_sequence(
        lid_paths["stage2_open_waypoints"], expected_base_frame=BASE_FRAME
    )
    stage2_retreat = load_pose_sequence(
        lid_paths["stage2_retreat_waypoints"], expected_base_frame=BASE_FRAME
    )

    if not _poses_match(stage1_contact_path[-1], stage1_contact):
        raise ValueError("stage1 contact path must end at stage1_contact_tcp")
    validate_pose_sequence_motion(stage1_precontact, stage1_contact_path)
    if not _poses_match(stage1_open[0], stage1_contact):
        raise ValueError("stage1 open path must start at stage1_contact_tcp")
    validate_pose_sequence_motion(stage1_contact, stage1_open)
    validate_pose_sequence_motion(stage1_open[-1], stage1_retreat)

    if (
        stage2_free_space_payload.get("not_cartesian") is not True
        or stage2_free_space_payload.get("execution_requested") is not False
        or len(stage2_free_space) != 4
    ):
        raise ValueError("stage2 free-space transition must contain four separate global goals")
    if not _poses_match(stage2_free_space[-1], stage2_precontact):
        raise ValueError("stage2 free-space transition must finish at stage2 precontact")
    if not _poses_match(stage2_contact_path[-1], stage2_contact):
        raise ValueError("stage2 contact path must end at stage2_contact_tcp")
    validate_pose_sequence_motion(stage2_precontact, stage2_contact_path)
    if not _poses_match(stage2_open[0], stage2_contact):
        raise ValueError("stage2 open path must start at stage2_contact_tcp")
    validate_pose_sequence_motion(stage2_contact, stage2_open)
    validate_pose_sequence_motion(stage2_open[-1], stage2_retreat)
    if not np.allclose(stage1_open[-1].position_m, stage2_contact.position_m, atol=1e-6):
        raise ValueError("stage2 regrasp contact must preserve the stage1 61-degree lid point")

    min_lid_opening_m = _positive_bounded(
        lid.get("min_contact_opening_m", 0.002),
        name="lid_two_stage.min_contact_opening_m",
        maximum=0.050,
        inclusive=False,
    )
    max_lid_opening_m = _positive_bounded(
        lid.get("max_contact_opening_m", 0.048),
        name="lid_two_stage.max_contact_opening_m",
        maximum=0.050,
        inclusive=False,
    )
    if min_lid_opening_m >= max_lid_opening_m:
        raise ValueError("lid contact opening min must be below max")

    transport_height_m = _positive_bounded(
        placement.get("transport_height_m", 0.080),
        name="placement.transport_height_m",
        maximum=0.12,
    )
    retreat_height_m = _positive_bounded(
        placement.get("retreat_height_m", 0.080),
        name="placement.retreat_height_m",
        maximum=0.12,
    )
    sequence_speed_m_s = _positive_bounded(
        motion.get("pose_sequence_speed_m_s", MAX_CARTESIAN_SPEED_M_S),
        name="motion.pose_sequence_speed_m_s",
        maximum=MAX_CARTESIAN_SPEED_M_S,
    )
    sequence_step_m = _positive_bounded(
        motion.get("pose_sequence_max_step_m", 0.005),
        name="motion.pose_sequence_max_step_m",
        maximum=0.010,
    )

    relation = _read_json(placement_relation_path)
    final_anchor = _validate_relation(
        relation,
        path=placement_relation_path,
        t_base_camera=pick["t_base_camera"],
    )
    return {
        "repo_root": pick["repo_root"],
        "pick_config": pick_config_path,
        "pick": pick,
        "commands": {
            "pick": _command(config, "pick"),
            "pose_sequence": _command(config, "pose_sequence"),
        },
        "lid_manifest_path": manifest_path,
        "lid_manifest": manifest,
        "lid_connected_proof_path": lid_paths["connected_plan_only"],
        "lid_connected_proof": connected_proof,
        "lid_paths": lid_paths,
        "lid_stage1_precontact": stage1_precontact,
        "lid_stage1_contact": stage1_contact,
        "lid_stage1_open": stage1_open,
        "lid_stage1_retreat": stage1_retreat,
        "lid_stage2_transition_targets": stage2_free_space,
        "lid_stage2_precontact": stage2_precontact,
        "lid_stage2_contact": stage2_contact,
        "lid_stage2_open": stage2_open,
        "lid_stage2_retreat": stage2_retreat,
        "lid_min_opening_m": min_lid_opening_m,
        "lid_max_opening_m": max_lid_opening_m,
        "placement_relation_path": placement_relation_path,
        "placement_relation": relation,
        "final_die_anchor_base_m": final_anchor,
        "transport_height_m": transport_height_m,
        "retreat_height_m": retreat_height_m,
        "pose_sequence_speed_m_s": sequence_speed_m_s,
        "pose_sequence_max_step_m": sequence_step_m,
    }


def _append_stage(
    report: dict[str, Any],
    report_path: Path,
    stage: Demo03Stage,
    *,
    status: str,
    details: Mapping[str, Any],
    started_at: str,
) -> None:
    expected = STAGE_ORDER[len(report["stages"])]
    if stage.value != expected:
        raise ValueError(f"cannot append {stage.value}; expected {expected}")
    if status not in {"passed", "failed"}:
        raise ValueError(f"invalid stage status: {status}")
    report["stages"].append(
        {
            "name": stage.value,
            "status": status,
            "started_at": started_at,
            "finished_at": _now(),
            "details": dict(details),
        }
    )
    if status == "passed":
        report["last_completed_stage"] = stage.value
    report["updated_at"] = _now()
    atomic_write_json(report_path, report)


def _run_pick(
    normalized: Mapping[str, Any],
    *,
    output_dir: Path,
    input_dir: Path | None,
    execute: bool,
) -> tuple[dict[str, Any], Path, subprocess.CompletedProcess[str]]:
    completed = _run_process(
        _pick_command(
            normalized,
            output_dir=output_dir,
            input_dir=input_dir,
            execute=execute,
        ),
        cwd=normalized["repo_root"],
        log_prefix=output_dir.parent / f"{output_dir.name}_subprocess",
    )
    report_path = output_dir / "stage_report.json"
    if not report_path.is_file():
        raise StageFailure("autonomous-pick subprocess produced no stage report")
    payload = _read_json(report_path)
    ready, reason = _pick_report_ready(payload, executed=execute)
    if completed.returncode != 0 or not ready:
        raise StageFailure(
            f"autonomous-pick {'execution' if execute else 'review'} failed: {reason}; "
            f"returncode={completed.returncode}"
        )
    return payload, report_path, completed


def _write_placement(
    normalized: Mapping[str, Any], *, grounded_pick_path: Path, output_dir: Path
) -> dict[str, Any]:
    grounded_pick = _read_json(grounded_pick_path)
    grounding, payloads = ground_waypoints(
        grounded_pick=grounded_pick,
        final_die_anchor_base_m=normalized["final_die_anchor_base_m"],
        transport_height_m=normalized["transport_height_m"],
        retreat_height_m=normalized["retreat_height_m"],
        base_frame=BASE_FRAME,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    relation_path: Path = normalized["placement_relation_path"]
    grounding["provenance"] = {
        "placement_relation": str(relation_path),
        "placement_relation_sha256": _sha256(relation_path),
        "fresh_grounded_pick": str(grounded_pick_path),
        "fresh_grounded_pick_sha256": _sha256(grounded_pick_path),
        "same_orchestration_run": True,
    }
    paths: dict[str, Path] = {}
    for name, payload in payloads.items():
        path = output_dir / f"{name}.json"
        atomic_write_json(path, payload)
        paths[name] = path
    grounding_path = output_dir / "placement_grounding.json"
    atomic_write_json(grounding_path, grounding)

    transport = _single_pose(paths["transport_tcp"], name="runtime transport_tcp")
    placement = _single_pose(paths["placement_tcp"], name="runtime placement_tcp")
    retreat = _single_pose(paths["retreat_tcp"], name="runtime retreat_tcp")
    place_waypoints = load_pose_sequence(paths["place_waypoints"], expected_base_frame=BASE_FRAME)
    retreat_waypoints = load_pose_sequence(
        paths["retreat_waypoints"], expected_base_frame=BASE_FRAME
    )
    if not _poses_match(place_waypoints[0], transport) or not _poses_match(
        place_waypoints[-1], placement
    ):
        raise StageFailure("runtime place waypoints do not connect transport to placement")
    if not _poses_match(retreat_waypoints[0], placement) or not _poses_match(
        retreat_waypoints[-1], retreat
    ):
        raise StageFailure("runtime retreat waypoints do not connect placement to retreat")
    validate_pose_sequence_motion(transport, place_waypoints)
    validate_pose_sequence_motion(placement, retreat_waypoints)
    if grounding.get("schema") != GROUNDING_SCHEMA:
        raise StageFailure("runtime placement grounding has the wrong schema")
    return {
        "grounding": grounding,
        "grounding_path": grounding_path,
        "paths": paths,
        "transport": transport,
        "placement": placement,
        "retreat": retreat,
    }


def _placement_release_guard(
    normalized: Mapping[str, Any],
    placement: Mapping[str, Any],
    *,
    measured_tcp_error_m: float,
) -> dict[str, Any]:
    """Check the implemented fixed-scene release predicate.

    No live box pose sensor exists.  The strongest available pre-release check
    is therefore geometric: the fresh pick anchor must be translated to the
    accepted stationary final-demo die anchor, and the measured TCP must land
    inside an envelope combining demo-anchor scatter and TCP tracking error.
    Object hold is checked separately immediately before this predicate.
    """

    grounding = placement["grounding"]
    relation = normalized["placement_relation"]
    target = np.asarray(grounding.get("fixed_scene_target_die_anchor_base_m"), dtype=float)
    accepted_target = np.asarray(relation.get("final_die_anchor_base_m"), dtype=float)
    pick_anchor = np.asarray(grounding.get("pick_object_anchor_base_m"), dtype=float)
    translation = np.asarray(grounding.get("anchor_translation_base_m"), dtype=float)
    if any(
        value.shape != (3,) or not np.isfinite(value).all()
        for value in (target, accepted_target, pick_anchor, translation)
    ):
        raise StageFailure("fixed-scene placement guard contains an invalid anchor")
    formula_error_m = float(np.linalg.norm((pick_anchor + translation) - target))
    target_error_m = float(np.linalg.norm(target - accepted_target))
    quality = relation.get("quality")
    if not isinstance(quality, Mapping):
        raise StageFailure("fixed-scene relation has no quality evidence")
    demo_scatter_m = float(quality.get("center_max_deviation_m", math.nan))
    tcp_error_m = float(measured_tcp_error_m)
    if not all(math.isfinite(value) and value >= 0.0 for value in (demo_scatter_m, tcp_error_m)):
        raise StageFailure("fixed-scene placement guard has non-finite error evidence")
    envelope_radius_m = (
        demo_scatter_m + normalized["pick"]["thresholds"].max_measured_tcp_position_error_m
    )
    accepted = (
        formula_error_m <= 1e-9
        and target_error_m <= 1e-9
        and tcp_error_m <= normalized["pick"]["thresholds"].max_measured_tcp_position_error_m
    )
    evidence = {
        "accepted": accepted,
        "predicate": "fixed_scene_final_die_anchor_envelope",
        "inference": (
            "the demonstrated final anchor is inside and supported by the unchanged box; "
            "fresh-pick translation and measured TCP endpoint place the held die in that envelope"
        ),
        "fresh_pick_anchor_base_m": pick_anchor.tolist(),
        "anchor_translation_base_m": translation.tolist(),
        "target_die_anchor_base_m": target.tolist(),
        "accepted_final_demo_anchor_base_m": accepted_target.tolist(),
        "formula_error_m": formula_error_m,
        "target_error_m": target_error_m,
        "measured_tcp_error_m": tcp_error_m,
        "demo_anchor_max_deviation_m": demo_scatter_m,
        "combined_envelope_radius_m": envelope_radius_m,
        "live_box_pose_observed": False,
    }
    if not accepted:
        raise StageFailure(f"fixed-scene placement release guard failed: {evidence}")
    return evidence


def _plan_pose(
    normalized: Mapping[str, Any],
    target: CartesianPose,
    *,
    output_dir: Path,
    artifact_name: str,
) -> dict[str, Any]:
    artifact = output_dir / f"{artifact_name}.json"
    from scripts.run_autonomous_pick import _run_command

    _run_command(
        _pose_command(normalized, target, output=artifact, execute=False),
        cwd=normalized["repo_root"],
        log_prefix=output_dir / artifact_name,
    )
    payload = _read_json(artifact)
    if not pose_result_succeeded(payload, executed=False):
        raise StageFailure(f"{artifact_name} did not produce an accepted audited plan")
    return _artifact_summary(artifact, payload)


def _plan_only(
    args: argparse.Namespace,
    normalized: Mapping[str, Any],
    report: dict[str, Any],
    report_path: Path,
    output_dir: Path,
) -> int:
    preflight: dict[str, Any] = {
        "semantics": (
            "reachability prechecks are independently planned from the current measured state; "
            "the physical state machine replans each stage from its then-measured state"
        ),
        "static_chain_validation": {
            "stage1_precontact_contact_open61_release_retreat": "accepted",
            "stage2_four_goal_clearance_transition": "accepted_as_non_cartesian",
            "stage2_precontact_contact_open100_release_retreat": "accepted",
            "offline_connected_proof": _artifact_summary(
                normalized["lid_connected_proof_path"],
                normalized["lid_connected_proof"],
            ),
            "waypoint_manifest": _artifact_summary(
                normalized["lid_manifest_path"], normalized["lid_manifest"]
            ),
            "placement_relation": str(normalized["placement_relation_path"]),
            "T_base_camera_sha256_match": True,
        },
    }
    try:
        preflight["lid_first_approach_plan"] = _plan_pose(
            normalized,
            normalized["lid_stage1_precontact"],
            output_dir=output_dir,
            artifact_name="lid_first_approach_plan",
        )
        preflight["lid_approach_plan_scope"] = {
            "planned_target": "stage1_precontact",
            "two_stage_runtime_paths": (
                "offline connected proof only; every physical substage is freshly planned and "
                "audited from its measured predecessor"
            ),
            "stage1_to_stage2_lid_angle": (
                "not observed by this runner; execution stops after stage1 unless the explicit "
                "unobserved-transition acknowledgement is supplied"
            ),
        }
        pick_input = args.pick_input_dir.expanduser().resolve() if args.pick_input_dir else None
        pick_report, pick_report_path, _ = _run_pick(
            normalized,
            output_dir=output_dir / "pick_review",
            input_dir=pick_input,
            execute=False,
        )
        preflight["die_pick_review"] = _artifact_summary(pick_report_path, pick_report)
        placement = _write_placement(
            normalized,
            grounded_pick_path=output_dir / "pick_review" / "execution_candidate.json",
            output_dir=output_dir / "placement_review",
        )
        preflight["placement_grounding"] = _artifact_summary(
            placement["grounding_path"], placement["grounding"]
        )
        preflight["transport_plan"] = _plan_pose(
            normalized,
            placement["transport"],
            output_dir=output_dir,
            artifact_name="transport_plan",
        )
        report["status"] = "partial_preflight_ready"
        report["preflight"] = preflight
        report["preflight_scope"] = (
            "audited global plans from the current measured state: stage1 lid precontact, die "
            "pregrasp, and transport reachability; both lid-stage chains have an offline "
            "connected proof, while every sequential physical path is replanned/audited from "
            "its measured predecessor at execution"
        )
        report["not_executed"] = list(STAGE_ORDER)
        report["finished_at"] = _now()
        atomic_write_json(report_path, report)
        print(f"demo-03 open-first partial preflight ready: {report_path}")
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"error_type": type(exc).__name__, "error": str(exc)}
        report["preflight"] = preflight
        report["finished_at"] = _now()
        atomic_write_json(report_path, report)
        print(f"demo-03 open-first review failed: {exc}", file=sys.stderr)
        return 2


def _pick_stage_details(
    pick_report: Mapping[str, Any], names: Sequence[str]
) -> list[Mapping[str, Any]]:
    records = pick_report.get("stages")
    if not isinstance(records, list):
        raise StageFailure("autonomous-pick report has no stages")
    selected = [
        record for record in records if isinstance(record, Mapping) and record.get("name") in names
    ]
    if [record.get("name") for record in selected] != list(names):
        raise StageFailure(f"autonomous-pick report is missing expected stages {list(names)}")
    return selected


def _execute(
    normalized: Mapping[str, Any],
    report: dict[str, Any],
    report_path: Path,
    output_dir: Path,
    *,
    accept_unobserved_lid_transition: bool = False,
) -> int:
    active_stage: Demo03Stage | None = None
    active_started = _now()
    try:
        active_stage = Demo03Stage.LID_STAGE1_PRECONTACT
        active_started = _now()
        opened, opened_path = _run_hande(
            normalized,
            "open",
            output_dir=output_dir,
            artifact_name="hande_open_for_lid_stage1",
        )
        if not hande_open_succeeded(opened, normalized["pick"]["thresholds"]):
            raise StageFailure("Hand-E did not reach the safe open width before lid stage1")
        stage1_precontact_motion = _run_pose(
            normalized,
            normalized["lid_stage1_precontact"],
            output_dir=output_dir,
            artifact_name="lid_stage1_precontact_execution",
        )
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "preshape": _artifact_summary(opened_path, opened),
                "motion": stage1_precontact_motion,
            },
            started_at=active_started,
        )

        active_stage = Demo03Stage.LID_STAGE1_CONTACT
        active_started = _now()
        stage1_contact_motion = _run_sequence(
            normalized,
            normalized["lid_paths"]["stage1_contact_waypoints"],
            output_dir=output_dir,
            artifact_name="lid_stage1_contact_execution",
        )
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=stage1_contact_motion,
            started_at=active_started,
        )

        active_stage = Demo03Stage.LID_STAGE1_GRIP
        active_started = _now()
        stage1_close, stage1_close_path = _run_hande(
            normalized,
            "close",
            output_dir=output_dir,
            artifact_name="hande_grip_lid_stage1",
        )
        if not _lid_contact_verified(
            stage1_close,
            min_opening_m=normalized["lid_min_opening_m"],
            max_opening_m=normalized["lid_max_opening_m"],
        ):
            raise StageFailure("Hand-E close did not verify stage1 lid contact")
        stage1_opening_m = float(stage1_close["state"]["opening_width_mm"]) * 1e-3
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=_artifact_summary(stage1_close_path, stage1_close),
            started_at=active_started,
        )

        active_stage = Demo03Stage.LID_STAGE1_OPEN_61
        active_started = _now()
        stage1_open_motion = _run_sequence(
            normalized,
            normalized["lid_paths"]["stage1_open_waypoints"],
            output_dir=output_dir,
            artifact_name="lid_stage1_open_61_execution",
        )
        stage1_hold, stage1_hold_path = _run_hande(
            normalized,
            "status",
            output_dir=output_dir,
            artifact_name="hande_post_lid_stage1_open",
        )
        if not _lid_width_snapshot_valid(
            stage1_hold,
            min_opening_m=normalized["lid_min_opening_m"],
            max_opening_m=normalized["lid_max_opening_m"],
            reference_opening_m=stage1_opening_m,
        ):
            raise StageFailure("lid contact was not maintained to the measured 61-degree endpoint")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "motion": stage1_open_motion,
                "lid_hold": _artifact_summary(stage1_hold_path, stage1_hold),
                "commanded_lid_angle_deg": 61,
            },
            started_at=active_started,
        )

        active_stage = Demo03Stage.LID_STAGE1_RELEASE
        active_started = _now()
        stage1_release, stage1_release_path = _run_hande(
            normalized,
            "open",
            output_dir=output_dir,
            artifact_name="hande_release_lid_stage1",
        )
        if not hande_open_succeeded(stage1_release, normalized["pick"]["thresholds"]):
            raise StageFailure("Hand-E did not fully release the lid after stage1")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=_artifact_summary(stage1_release_path, stage1_release),
            started_at=active_started,
        )

        active_stage = Demo03Stage.LID_STAGE1_RETREAT
        active_started = _now()
        stage1_retreat_motion = _run_sequence(
            normalized,
            normalized["lid_paths"]["stage1_retreat_waypoints"],
            output_dir=output_dir,
            artifact_name="lid_stage1_release_retreat_execution",
        )
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=stage1_retreat_motion,
            started_at=active_started,
        )

        active_stage = Demo03Stage.LID_STAGE1_TRANSITION_GATE
        active_started = _now()
        if not accept_unobserved_lid_transition:
            raise StageFailure(
                "released lid angle was not reobserved; stop after stage1 or rerun with "
                "--accept-unobserved-lid-transition to use the measured 61-degree endpoint"
            )
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "accepted": True,
                "predicate": "explicit_unobserved_lid_transition_acknowledgement",
                "stage1_commanded_endpoint_deg": 61,
                "stage1_endpoint_measured_with_lid_held": True,
                "stage1_release_and_retreat_succeeded": True,
                "released_lid_angle_reobserved": False,
                "limitation": (
                    "the second grasp assumes the released lid remained near 61 degrees; "
                    "the current runner has no live lid-angle estimator"
                ),
            },
            started_at=active_started,
        )

        active_stage = Demo03Stage.LID_STAGE2_CLEARANCE_TRANSITION
        active_started = _now()
        stage2_transition_targets = normalized["lid_stage2_transition_targets"]
        clearance_motions = [
            _run_pose(
                normalized,
                target,
                output_dir=output_dir,
                artifact_name=f"lid_stage2_clearance_{index:02d}_execution",
            )
            for index, target in enumerate(stage2_transition_targets[:-1], start=1)
        ]
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "motions": clearance_motions,
                "semantics": "separate audited MoveIt global goals from measured predecessors",
            },
            started_at=active_started,
        )

        active_stage = Demo03Stage.LID_STAGE2_PRECONTACT
        active_started = _now()
        stage2_precontact_motion = _run_pose(
            normalized,
            stage2_transition_targets[-1],
            output_dir=output_dir,
            artifact_name="lid_stage2_precontact_execution",
        )
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=stage2_precontact_motion,
            started_at=active_started,
        )

        active_stage = Demo03Stage.LID_STAGE2_CONTACT
        active_started = _now()
        stage2_contact_motion = _run_sequence(
            normalized,
            normalized["lid_paths"]["stage2_contact_waypoints"],
            output_dir=output_dir,
            artifact_name="lid_stage2_contact_execution",
        )
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=stage2_contact_motion,
            started_at=active_started,
        )

        active_stage = Demo03Stage.LID_STAGE2_GRIP
        active_started = _now()
        stage2_close, stage2_close_path = _run_hande(
            normalized,
            "close",
            output_dir=output_dir,
            artifact_name="hande_grip_lid_stage2",
        )
        if not _lid_contact_verified(
            stage2_close,
            min_opening_m=normalized["lid_min_opening_m"],
            max_opening_m=normalized["lid_max_opening_m"],
        ):
            raise StageFailure("Hand-E close did not verify stage2 lid contact")
        stage2_opening_m = float(stage2_close["state"]["opening_width_mm"]) * 1e-3
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=_artifact_summary(stage2_close_path, stage2_close),
            started_at=active_started,
        )

        active_stage = Demo03Stage.LID_STAGE2_OPEN_100
        active_started = _now()
        stage2_open_motion = _run_sequence(
            normalized,
            normalized["lid_paths"]["stage2_open_waypoints"],
            output_dir=output_dir,
            artifact_name="lid_stage2_open_100_execution",
        )
        stage2_hold, stage2_hold_path = _run_hande(
            normalized,
            "status",
            output_dir=output_dir,
            artifact_name="hande_post_lid_stage2_open",
        )
        if not _lid_width_snapshot_valid(
            stage2_hold,
            min_opening_m=normalized["lid_min_opening_m"],
            max_opening_m=normalized["lid_max_opening_m"],
            reference_opening_m=stage2_opening_m,
        ):
            raise StageFailure("lid contact was not maintained to the measured 100-degree endpoint")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "motion": stage2_open_motion,
                "lid_hold": _artifact_summary(stage2_hold_path, stage2_hold),
                "commanded_lid_angle_deg": 100,
            },
            started_at=active_started,
        )

        active_stage = Demo03Stage.LID_STAGE2_RELEASE
        active_started = _now()
        stage2_release, stage2_release_path = _run_hande(
            normalized,
            "open",
            output_dir=output_dir,
            artifact_name="hande_release_lid_stage2",
        )
        if not hande_open_succeeded(stage2_release, normalized["pick"]["thresholds"]):
            raise StageFailure("Hand-E did not fully release the lid after stage2")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=_artifact_summary(stage2_release_path, stage2_release),
            started_at=active_started,
        )

        active_stage = Demo03Stage.LID_STAGE2_RETREAT
        active_started = _now()
        stage2_retreat_motion = _run_sequence(
            normalized,
            normalized["lid_paths"]["stage2_retreat_waypoints"],
            output_dir=output_dir,
            artifact_name="lid_stage2_release_retreat_execution",
        )
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "motion": stage2_retreat_motion,
                "final_lid_angle_observed": False,
                "expected_post_release_behavior": (
                    "lid is released past vertical at 100 degrees and gravity is expected to "
                    "continue opening it"
                ),
            },
            started_at=active_started,
        )

        active_stage = Demo03Stage.REOBSERVE_DIE
        active_started = _now()
        review_report, review_path, _ = _run_pick(
            normalized,
            output_dir=output_dir / "die_reobserve",
            input_dir=None,
            execute=False,
        )
        fresh_input = output_dir / "die_reobserve" / "input"
        if not fresh_input.is_dir():
            raise StageFailure("live die reobservation produced no RGB-D input directory")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "pick_review": _artifact_summary(review_path, review_report),
                "atomic_stages": _pick_stage_details(review_report, ("OBSERVE", "GROUND", "PLAN")),
                "fresh_input": str(fresh_input),
            },
            started_at=active_started,
        )

        active_stage = Demo03Stage.PICK_AND_LIFT
        active_started = _now()
        pick_report, pick_path, _ = _run_pick(
            normalized,
            output_dir=output_dir / "die_pick",
            input_dir=fresh_input,
            execute=True,
        )
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "pick_report": _artifact_summary(pick_path, pick_report),
                "atomic_stages": _pick_stage_details(
                    pick_report,
                    (
                        "OBSERVE",
                        "GROUND",
                        "PLAN",
                        "PREGRASP",
                        "DESCEND",
                        "CLOSE",
                        "VERIFY_CONTACT",
                        "LIFT",
                    ),
                ),
                "outcome": "object held after measured short lift",
            },
            started_at=active_started,
        )

        active_stage = Demo03Stage.DERIVE_PLACEMENT
        active_started = _now()
        close_payload = _read_json(output_dir / "die_pick" / "hande_close.json")
        close_opening_m = float(close_payload["state"]["opening_width_mm"]) * 1e-3
        placement = _write_placement(
            normalized,
            grounded_pick_path=output_dir / "die_pick" / "execution_candidate.json",
            output_dir=output_dir / "runtime_placement",
        )
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                **_artifact_summary(placement["grounding_path"], placement["grounding"]),
                "fresh_grounded_pick": str(output_dir / "die_pick" / "execution_candidate.json"),
            },
            started_at=active_started,
        )

        active_stage = Demo03Stage.TRANSPORT
        active_started = _now()
        transport = _run_pose(
            normalized,
            placement["transport"],
            output_dir=output_dir,
            artifact_name="transport_execution",
        )
        held, held_path = _run_hande(
            normalized,
            "status",
            output_dir=output_dir,
            artifact_name="hande_post_transport",
        )
        if not hande_width_snapshot_valid(
            held,
            normalized["pick"]["thresholds"],
            reference_opening_m=close_opening_m,
            max_drift_m=0.002,
        ):
            raise StageFailure("object hold verification failed after TRANSPORT")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "motion": transport,
                "object_hold": _artifact_summary(held_path, held),
            },
            started_at=active_started,
        )

        active_stage = Demo03Stage.PLACE
        active_started = _now()
        place = _run_sequence(
            normalized,
            placement["paths"]["place_waypoints"],
            output_dir=output_dir,
            artifact_name="place_execution",
        )
        held, held_path = _run_hande(
            normalized,
            "status",
            output_dir=output_dir,
            artifact_name="hande_pre_release",
        )
        if not hande_width_snapshot_valid(
            held,
            normalized["pick"]["thresholds"],
            reference_opening_m=close_opening_m,
            max_drift_m=0.002,
        ):
            raise StageFailure("object hold verification failed at the placement endpoint")
        measured_error = place.get("measured_final_position_error_m")
        if not isinstance(measured_error, (int, float)):
            raise StageFailure("place result has no measured TCP endpoint error")
        placement_guard = _placement_release_guard(
            normalized,
            placement,
            measured_tcp_error_m=float(measured_error),
        )
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details={
                "motion": place,
                "object_hold": _artifact_summary(held_path, held),
                "release_guard": placement_guard,
            },
            started_at=active_started,
        )

        active_stage = Demo03Stage.RELEASE_DIE
        active_started = _now()
        if placement_guard.get("accepted") is not True:
            raise StageFailure("fixed-scene placement release guard is not accepted")
        die_release, die_release_path = _run_hande(
            normalized, "open", output_dir=output_dir, artifact_name="hande_release_die"
        )
        if not hande_open_succeeded(die_release, normalized["pick"]["thresholds"]):
            raise StageFailure("Hand-E did not fully release the die")
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=_artifact_summary(die_release_path, die_release),
            started_at=active_started,
        )

        active_stage = Demo03Stage.RETREAT
        active_started = _now()
        final_retreat = _run_sequence(
            normalized,
            placement["paths"]["retreat_waypoints"],
            output_dir=output_dir,
            artifact_name="placement_retreat_execution",
        )
        _append_stage(
            report,
            report_path,
            active_stage,
            status="passed",
            details=final_retreat,
            started_at=active_started,
        )

        report["status"] = "succeeded"
        report["lid_state"] = (
            "released_after_measured_100deg_endpoint; final_post_gravity_angle_unobserved"
        )
        report["object_state"] = "released_at_fixed_scene_placement_target"
        report["finished_at"] = _now()
        atomic_write_json(report_path, report)
        print(f"demo-03 open-first task succeeded: {report_path}")
        return 0
    except Exception as exc:
        if active_stage is not None and len(report["stages"]) < len(STAGE_ORDER):
            if STAGE_ORDER[len(report["stages"])] == active_stage.value:
                _append_stage(
                    report,
                    report_path,
                    active_stage,
                    status="failed",
                    details={"error_type": type(exc).__name__, "error": str(exc)},
                    started_at=active_started,
                )
        report["status"] = "failed"
        report["failed_stage"] = active_stage.value if active_stage else None
        report["failure"] = {"error_type": type(exc).__name__, "error": str(exc)}
        report["recovery"] = {
            "policy": "stop_without_additional_orchestrator_motion",
            "automatic_motion_attempted": False,
            "note": (
                "the nested autonomous-pick runner retains its own pre-lift recovery; "
                "after verified grasp this orchestrator holds position on failure"
            ),
        }
        report["finished_at"] = _now()
        atomic_write_json(report_path, report)
        print(f"demo-03 task failed at {report['failed_stage']}: {exc}", file=sys.stderr)
        return 2


def run(args: argparse.Namespace) -> int:
    accept_unobserved_lid_transition = bool(
        getattr(args, "accept_unobserved_lid_transition", False)
    )
    if args.execute:
        if not args.accept_arm_only_collision_model:
            raise ValueError("--execute requires --accept-arm-only-collision-model")
        if not args.accept_fixed_scene_placement:
            raise ValueError("--execute requires --accept-fixed-scene-placement")
        if args.pick_input_dir is not None:
            raise ValueError("execution always performs a fresh live die reobservation")
    elif (
        args.accept_arm_only_collision_model
        or args.accept_fixed_scene_placement
        or accept_unobserved_lid_transition
    ):
        raise ValueError("execution acknowledgements are only valid with --execute")

    config_path = args.config.expanduser().resolve()
    normalized = _validate_config(
        _read_json(config_path), config_path=config_path, execute=args.execute
    )
    if args.execute and "capture" not in normalized["pick"]["commands"]:
        raise ValueError("execution requires commands.capture for fresh die reobservation")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    report_path = output_dir / "stage_report.json"
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "running",
        "mode": "execute" if args.execute else "plan_only",
        "authorized_for_robot_execution": bool(args.execute),
        "arm_only_collision_model_acknowledged": bool(args.accept_arm_only_collision_model),
        "fixed_scene_placement_acknowledged": bool(args.accept_fixed_scene_placement),
        "unobserved_lid_transition_acknowledged": accept_unobserved_lid_transition,
        "config": str(config_path),
        "output_dir": str(output_dir),
        "started_at": _now(),
        "updated_at": _now(),
        "stage_order": list(STAGE_ORDER),
        "stages": [],
        "last_completed_stage": None,
        "lid_plan_provenance": {
            "manifest": str(normalized["lid_manifest_path"]),
            "manifest_sha256": _sha256(normalized["lid_manifest_path"]),
            "connected_plan_only": str(normalized["lid_connected_proof_path"]),
            "connected_plan_only_sha256": _sha256(
                normalized["lid_connected_proof_path"]
            ),
            "offline_connected_proof_passed": True,
            "stage1_endpoint_deg": 61,
            "stage2_release_endpoint_deg": 100,
        },
        "contracts": {
            "human_trajectory": "phase_and_intent_evidence_only; never replayed",
            "lid": (
                "reviewed connected two-stage hinge path; 61-degree post-release angle is "
                "unobserved when its explicit acknowledgement is used"
            ),
            "die_pick": "fresh live grounding plus reviewed object-relative grasp",
            "placement": "fresh grounded pick translated to fixed demonstrated final anchor",
            "arm_motion": "MoveIt plan, FR3 trajectory audit, then exact-path execution",
        },
    }
    atomic_write_json(report_path, report)
    if not args.execute:
        return _plan_only(args, normalized, report, report_path, output_dir)
    return _execute(
        normalized,
        report,
        report_path,
        output_dir,
        accept_unobserved_lid_transition=accept_unobserved_lid_transition,
    )


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception:  # pragma: no cover - preserve integration diagnostics
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
