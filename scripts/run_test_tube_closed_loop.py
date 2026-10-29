#!/usr/bin/env python3
"""Run or replay the complete seven-stage test-tube closed loop.

The default is a physically inert dry-run.  It validates the demonstration
relation and reports the exact backend capabilities still needed.  ``--replay``
drives the same controller from recorded observations.  ``--execute`` is only
available through an explicitly supplied long-lived adapter implementing
``ClosedLoopIO``; this repository intentionally does not pretend that an
uninstalled or unqualified F2G-Pose checkout is a live 6D tracker.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import platform
import signal
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo2gripper.autonomous_pick import atomic_write_json  # noqa: E402
from demo2gripper.franka import validate_rigid_transform  # noqa: E402
from demo2gripper.test_tube_closed_loop import (  # noqa: E402
    CLOSED_LOOP_CONFIG_SCHEMA,
    CLOSED_LOOP_REPLAY_SCHEMA,
    CLOSED_LOOP_REPORT_SCHEMA,
    HELD_POSE_CAPABILITIES,
    SOCKET_POSE_CAPABILITIES,
    SOURCE_POSE_CAPABILITIES,
    ActionKind,
    ClosedLoopAction,
    ClosedLoopContractError,
    ClosedLoopIO,
    PoseBackendDescriptor,
    PoseCapability,
    TestTubeClosedLoopController,
    parse_backend_descriptor,
    parse_geometry,
    parse_observation,
    parse_policy,
    state_contract,
)
from demo2gripper.test_tube_preflight import TubeState  # noqa: E402

QUALIFICATION_SCHEMA = "demo2gripper.pose_backend_qualification.v1"
ONLINE_SCENE_FACTORY = (
    "demo2gripper.test_tube_online_scene:create_online_test_tube_scene_interpreter"
)
ONLINE_COMPOSED_QUALIFICATION_ID = "online-composed-pose-producers-v1"
ALL_LIVE_CAPABILITIES = frozenset(
    SOURCE_POSE_CAPABILITIES | HELD_POSE_CAPABILITIES | SOCKET_POSE_CAPABILITIES
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate only (default)")
    mode.add_argument("--replay", type=Path, help="replay recorded fused observations")
    mode.add_argument("--execute", action="store_true", help="use an external live adapter")
    parser.add_argument(
        "--live-adapter",
        help="Python factory as module:function; required with --execute",
    )
    parser.add_argument(
        "--accept-arm-only-collision-model",
        action="store_true",
        help="explicitly acknowledge the current incomplete Hand-E collision model",
    )
    parser.add_argument(
        "--accept-provisional-camera-extrinsic",
        action="store_true",
        help=(
            "accept the configured, validated provisional camera recovery for this "
            "process only; does not create a qualification receipt"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument(
        "--recover-from-run", type=Path,
        help="reverse a verified prior open-gripper SIDE_GRASP approach before OBSERVE",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClosedLoopContractError(f"could not read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ClosedLoopContractError(f"JSON root must be an object: {path}")
    return payload


def _now() -> str:
    return datetime.now(UTC).isoformat()


def load_startup_recovery(path: Path, *, task_id: str):
    """Derive a bounded retreat only from a completed, empty-gripper attempt."""
    source = path.expanduser().resolve()
    task_dir = source / "task" if (source / "task" / "run_report.json").is_file() else source
    report_path = task_dir / "run_report.json"
    report = _read_json(report_path)
    if report.get("task_id") != task_id:
        raise ClosedLoopContractError("startup recovery task_id differs from current task")
    if report.get("status") != "held" or report.get("final_state") != "HOLD":
        raise ClosedLoopContractError("startup recovery requires a prior held/HOLD run")
    files = sorted((task_dir / "steps").glob("*.json"))
    if not files or report.get("step_count") != len(files):
        raise ClosedLoopContractError("startup recovery step history is incomplete")
    steps = [_read_json(item) for item in files]
    allowed = {"collect_evidence", "monitor", "none", "move_tcp", "servo_tcp",
               "pause_arm", "stop_and_hold"}
    for index, step in enumerate(steps):
        if step.get("index") != index:
            raise ClosedLoopContractError("startup recovery step indices are not contiguous")
        decision, dispatch = step.get("decision", {}), step.get("dispatch", {})
        action = decision.get("action", {})
        kind = action.get("kind")
        if kind not in allowed or decision.get("previous_state") not in {
            "OBSERVE", "APPROACH_TUBE", "SIDE_GRASP",
        } or decision.get("next_state") not in {"OBSERVE", "APPROACH_TUBE", "SIDE_GRASP", "HOLD"}:
            raise ClosedLoopContractError(
                "startup recovery history contains grasp/release or later task actions"
            )
        if kind in {"move_tcp", "servo_tcp", "pause_arm"}:
            receipt = dispatch.get("receipt", {})
            if dispatch.get("status") != "completed" or receipt.get("accepted") is not True:
                raise ClosedLoopContractError(
                    "startup recovery has an incomplete prior arm dispatch"
                )
        if kind == "stop_and_hold":
            receipt = dispatch.get("receipt", {})
            if (index != len(steps) - 1 or dispatch.get("status") != "held"
                    or receipt.get("held") is not True
                    or receipt.get("arm", {}).get("accepted") is not True):
                raise ClosedLoopContractError(
                    "startup recovery lacks a completed terminal arm hold"
                )
    terminal = steps[-1]
    decision = terminal["decision"]
    if (decision.get("previous_state") != "SIDE_GRASP" or decision.get("next_state") != "HOLD"
            or decision["action"].get("kind") != "stop_and_hold"
            or decision["action"].get("reason") != "SIDE_GRASP timeout exceeded"):
        raise ClosedLoopContractError("startup recovery requires a SIDE_GRASP timeout hold")
    observation = terminal.get("observation", {})
    gripper, tactile, robot = (observation.get(key, {}) for key in ("gripper", "tactile", "robot"))
    terminal_motion_complete = robot.get("motion_complete")
    terminal_hold_acknowledged = bool(
        terminal.get("dispatch", {}).get("status") == "held"
        and terminal.get("dispatch", {}).get("receipt", {}).get("held") is True
        and terminal.get("dispatch", {}).get("receipt", {}).get("arm", {}).get("accepted")
        is True
    )
    width = gripper.get("width_m")
    if (not isinstance(width, (int, float)) or not np.isfinite(width) or width < .048
            or gripper.get("moving") is not False or gripper.get("fault_free") is not True
            or gripper.get("object_detected") is not False
            or tactile.get("valid") is not True or tactile.get("contact_present") is not False
            or tactile.get("bilateral_contact") is not False
            or robot.get("fault") is not None
            or not isinstance(terminal_motion_complete, bool)
            or (terminal_motion_complete is False and not terminal_hold_acknowledged)):
        raise ClosedLoopContractError(
            "startup recovery final feedback is not stationary, empty and healthy"
        )
    measured = []
    indices = []
    for step in steps:
        if step["decision"].get("previous_state") == "SIDE_GRASP":
            contact = step.get("observation", {}).get("tactile", {})
            if (contact.get("contact_present") is not False
                    or contact.get("bilateral_contact") is not False):
                raise ClosedLoopContractError("startup recovery approach history contains contact")
            try:
                pose = validate_rigid_transform(
                    np.asarray(step["observation"]["robot"]["T_base_tcp"], dtype=float),
                    name="recorded recovery TCP",
                ).copy()
            except (KeyError, TypeError, ValueError) as error:
                raise ClosedLoopContractError(
                    "startup recovery measured TCP history is invalid"
                ) from error
            measured.append(pose)
            indices.append(step["index"])
    if len(measured) < 2:
        raise ClosedLoopContractError("startup recovery lacks a measured descent")
    reverse = list(reversed(measured))
    if any(b[2, 3] < a[2, 3] - .001 for a, b in zip(reverse, reverse[1:])):
        raise ClosedLoopContractError("startup recovery measured path reverses downward")
    retained, retained_indices = [reverse[0]], [indices[-1]]
    for pose, index in zip(reverse[1:-1], reversed(indices[1:-1])):
        if np.linalg.norm(pose[:3, 3] - retained[-1][:3, 3]) >= .002:
            retained.append(pose)
            retained_indices.append(index)
    if not np.allclose(retained[-1], reverse[-1], atol=1e-9, rtol=0):
        retained.append(reverse[-1])
        retained_indices.append(indices[0])
    total = sum(float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
                for a, b in zip(retained, retained[1:]))
    rise = max(float(pose[2, 3] - retained[0][2, 3]) for pose in retained)
    if total <= .002 or total > .1 or rise > .08:
        raise ClosedLoopContractError(
            "startup recovery measured path exceeds distance/height bounds"
        )
    digest = hashlib.sha256(report_path.read_bytes())
    for item in files:
        digest.update(item.read_bytes())
    provenance = {
        "schema": "demo2gripper.test_tube_startup_recovery.v1", "source_run": str(source),
        "source_task_report": str(report_path), "source_task_id": task_id,
        "source_artifacts_sha256": digest.hexdigest(), "source_step_count": len(steps),
        "retained_source_step_indices": retained_indices,
        "path_T_base_tcp": [p.tolist() for p in retained],
        "total_distance_m": total, "maximum_rise_m": rise,
        "measured_poses_only": True, "gripper_release_commanded": False,
        "terminal_feedback_motion_complete_before_hold": terminal_motion_complete,
        "terminal_stop_and_hold_acknowledged": terminal_hold_acknowledged,
        "live_stationary_start_required": True,
        "fresh_start_match_required": {"translation_m": .0025, "rotation_deg": 2.0},
    }
    return tuple(retained), provenance


def _prepare_output(path: Path, *, overwrite: bool) -> Path:
    result = path.expanduser().resolve()
    if result.exists() and any(result.iterdir()) and not overwrite:
        raise ClosedLoopContractError(f"output directory is not empty: {result}; pass --overwrite")
    result.mkdir(parents=True, exist_ok=True)
    (result / "steps").mkdir(exist_ok=True)
    return result


def _static_live_adapter_blockers(
    config: Mapping[str, Any],
    *,
    accept_arm_only_collision_model: bool = False,
    accept_provisional_camera_extrinsic: bool = False,
) -> tuple[str, ...]:
    """Run the repository adapter's pure preflight without opening hardware.

    This hook exists primarily for the intentionally unfilled live template:
    its ``null`` metric transforms cannot be passed to ``parse_geometry`` yet,
    but operators still need the complete blocker list rather than the first
    NumPy shape error.  Importing this module is inert; device construction is
    confined to ``create_live_io``.
    """

    live = config.get("live_adapter")
    if not isinstance(live, Mapping):
        return ()
    if live.get("schema") != "demo2gripper.test_tube_live_io.v1":
        return ()
    from demo2gripper.test_tube_live_io import preflight_live_config

    return preflight_live_config(
        config,
        accept_arm_only_collision_model=accept_arm_only_collision_model,
        accept_provisional_camera_extrinsic=accept_provisional_camera_extrinsic,
    )


def _collision_model_acknowledgement(
    config: Mapping[str, Any],
    *,
    accepted: bool,
) -> dict[str, Any]:
    from demo2gripper.test_tube_live_io import reduced_collision_model_acknowledgement

    return reduced_collision_model_acknowledgement(config, accepted=accepted)


def _camera_extrinsic_acknowledgement(
    config: Mapping[str, Any],
    *,
    accepted: bool,
) -> dict[str, Any]:
    from demo2gripper.test_tube_live_io import (
        provisional_camera_extrinsic_acknowledgement,
    )

    return provisional_camera_extrinsic_acknowledgement(config, accepted=accepted)


def _write_unparsed_live_blocker_report(
    *,
    args: argparse.Namespace,
    config_path: Path,
    output_dir: Path,
    task_id: str,
    execution_scope: str,
    blockers: Sequence[str],
    geometry_error: Exception,
    collision_model_acknowledgement: Mapping[str, Any],
    camera_extrinsic_acknowledgement: Mapping[str, Any],
    startup_recovery: Mapping[str, Any] | None = None,
) -> None:
    """Persist a fail-closed report for an unqualified live template."""

    report = {
        "schema": CLOSED_LOOP_REPORT_SCHEMA,
        "created_at": _now(),
        "task_id": task_id,
        "config": str(config_path),
        "mode": "execute" if args.execute else "replay" if args.replay else "dry_run",
        "execution_scope": execution_scope,
        "status": "blocked_live_execution",
        "final_state": TubeState.OBSERVE.value,
        "step_count": 0,
        "hardware_commands_dispatched": False,
        "hardware_opened": False,
        "startup_recovery": startup_recovery,
        "top_approach_hard_constraint": True,
        "demonstrated_relation": {
            "relation_id": None,
            "kind": "full_T_tube_tcp_in_signed_roll_canonical_tube_frame",
            "source": "unparsed_unqualified_template",
            "runtime_qualified": False,
            "human_trajectory_replayed": False,
        },
        "socket_entry_relation": {
            "source": "unparsed_unqualified_template",
            "runtime_qualified": False,
        },
        "pose_backend": {
            "descriptor": None,
            "preflight": {
                "static_live_adapter": True,
                "geometry_parse_error": str(geometry_error),
            },
            "blockers": sorted(set(blockers)),
        },
        "collision_model": dict(collision_model_acknowledgement),
        "camera_extrinsic": dict(camera_extrinsic_acknowledgement),
        "state_contract": state_contract(),
        "fail_closed_boundaries": [
            "unqualified metric geometry is rejected before adapter construction",
            "static preflight never opens the D435, GF220, Hand-E, or FR3",
        ],
    }
    atomic_write_json(output_dir / "run_report.json", report)


def _resolve_path(value: object, *, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ClosedLoopContractError(f"{name} must be a non-empty path")
    result = Path(value).expanduser()
    return (base / result).resolve() if not result.is_absolute() else result.resolve()


def _backend_preflight(
    value: object,
    *,
    config_dir: Path,
    online_composed: bool = False,
) -> tuple[PoseBackendDescriptor, list[str], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ClosedLoopContractError("pose_backend must be an object")
    name = value.get("name")
    if name not in {"f2g_pose", "rgbd_axis_anchor", "external", "synthetic_replay"}:
        raise ClosedLoopContractError(
            "pose_backend.name must be f2g_pose, rgbd_axis_anchor, external, or synthetic_replay"
        )
    detail: dict[str, Any] = {"requested_backend": name}
    blockers: list[str] = []

    if name == "synthetic_replay":
        descriptor = PoseBackendDescriptor(
            name=name,
            available=True,
            runtime_qualified=True,
            qualification_id="synthetic-fixture-only",
            capabilities=ALL_LIVE_CAPABILITIES,
            detail="deterministic replay fixture; never eligible for hardware execution",
        )
        detail["execution_scope"] = "synthetic_replay_only"
    elif name == "rgbd_axis_anchor":
        descriptor = PoseBackendDescriptor(
            name=name,
            available=True,
            runtime_qualified=False,
            qualification_id=None,
            capabilities=frozenset(
                {
                    PoseCapability.METRIC_TRANSLATION,
                    PoseCapability.IMAGE_AXIS,
                    PoseCapability.SOURCE_TUBE,
                }
            ),
            detail=(
                "fallback may seed an initial anchor/image axis; it cannot authorize "
                "SE(3) servo, in-hand tracking, or socket insertion"
            ),
        )
    else:
        root_value = value.get("root")
        root = (
            None
            if root_value is None
            else _resolve_path(root_value, base=config_dir, name="pose_backend.root")
        )
        service_url = value.get("service_url")
        remote_service_configured = isinstance(service_url, str) and bool(service_url.strip())
        adapter_spec = value.get("adapter")
        available = bool((root is not None and root.is_dir()) or remote_service_configured)
        if isinstance(adapter_spec, str) and ":" in adapter_spec:
            module_name = adapter_spec.split(":", 1)[0]
            available = available and importlib.util.find_spec(module_name) is not None
        else:
            available = False
            blockers.append("pose_backend_adapter_not_configured")
        detail.update(
            {
                "root": None if root is None else str(root),
                "root_exists": bool(root is not None and root.is_dir()),
                "service_url": service_url,
                "adapter": adapter_spec,
            }
        )

        if online_composed:
            descriptor = PoseBackendDescriptor(
                name=name,
                available=available,
                runtime_qualified=available,
                qualification_id=(ONLINE_COMPOSED_QUALIFICATION_ID if available else None),
                capabilities=(ALL_LIVE_CAPABILITIES if available else frozenset()),
                detail=("runtime-composed SAM3/F2G source, held-tube, and RGB-D socket producers"),
            )
            detail["qualification_mode"] = "runtime_composed_fail_closed"
            blockers.extend(descriptor.blockers(ALL_LIVE_CAPABILITIES))
            return descriptor, sorted(set(blockers)), detail

        qualification_path_value = value.get("qualification_receipt")
        qualification: dict[str, Any] | None = None
        if qualification_path_value is not None:
            qualification_path = _resolve_path(
                qualification_path_value,
                base=config_dir,
                name="pose_backend.qualification_receipt",
            )
            if qualification_path.is_file():
                qualification = _read_json(qualification_path)
                detail["qualification_receipt"] = str(qualification_path)
        qualified = bool(
            qualification
            and qualification.get("schema") == QUALIFICATION_SCHEMA
            and qualification.get("backend") == name
            and qualification.get("runtime_qualified") is True
        )
        capabilities: frozenset[PoseCapability]
        if qualified:
            raw = qualification.get("capabilities")
            if not isinstance(raw, list):
                raise ClosedLoopContractError(
                    "pose backend qualification capabilities must be an array"
                )
            capabilities = frozenset(PoseCapability(item) for item in raw)
            qualification_id = qualification.get("qualification_id")
        else:
            capabilities = frozenset()
            qualification_id = None
            blockers.append("pose_backend_task_qualification_receipt_missing")
        descriptor = PoseBackendDescriptor(
            name=name,
            available=available,
            runtime_qualified=qualified and available,
            qualification_id=qualification_id if qualified and available else None,
            capabilities=capabilities,
            detail="external 6D pose adapter",
        )

    blockers.extend(descriptor.blockers(ALL_LIVE_CAPABILITIES))
    return descriptor, sorted(set(blockers)), detail


def _uses_online_composed_scene(config: Mapping[str, Any]) -> bool:
    live = config.get("live_adapter")
    scene = live.get("scene_interpreter") if isinstance(live, Mapping) else None
    return bool(isinstance(scene, Mapping) and scene.get("factory") == ONLINE_SCENE_FACTORY)


class ReplayIO:
    """Physically inert adapter that emits a fixed observation trace."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema") != CLOSED_LOOP_REPLAY_SCHEMA:
            raise ClosedLoopContractError(f"replay.schema must be {CLOSED_LOOP_REPLAY_SCHEMA!r}")
        self._descriptor = parse_backend_descriptor(payload.get("pose_backend"))
        scenario = payload.get("synthetic_scenario")
        if scenario is not None:
            if payload.get("synthetic") is not True:
                raise ClosedLoopContractError("synthetic_scenario requires synthetic=true")
            if scenario != "nominal_seven_stage_v1":
                raise ClosedLoopContractError(
                    f"unsupported synthetic replay scenario: {scenario!r}"
                )
            self._observations = _nominal_synthetic_observations(self._descriptor)
        else:
            raw_observations = payload.get("observations")
            if not isinstance(raw_observations, list) or not raw_observations:
                raise ClosedLoopContractError("replay observations must be a non-empty array")
            self._observations = [parse_observation(item) for item in raw_observations]
        self._expected = payload.get("expected_actions", [])
        if not isinstance(self._expected, list):
            raise ClosedLoopContractError("expected_actions must be an array")
        self._observation_index = 0
        self._dispatch_index = 0

    @property
    def pose_backend_descriptor(self) -> PoseBackendDescriptor:
        return self._descriptor

    def observe(self, *, state: TubeState):
        del state
        if self._observation_index >= len(self._observations):
            raise EOFError("replay observation trace is exhausted")
        value = self._observations[self._observation_index]
        self._observation_index += 1
        return value

    def dispatch(self, action: ClosedLoopAction) -> Mapping[str, Any]:
        if self._dispatch_index < len(self._expected):
            expected = self._expected[self._dispatch_index]
            if not isinstance(expected, Mapping):
                raise ClosedLoopContractError("expected action must be an object")
            if expected.get("kind") != action.kind.value:
                raise ClosedLoopContractError(
                    f"replay action {self._dispatch_index} expected "
                    f"{expected.get('kind')!r}, got {action.kind.value!r}"
                )
        self._dispatch_index += 1
        return {
            "replay_only": True,
            "accepted": True,
            "action_id": action.action_id,
            "hardware_commands_dispatched": False,
        }

    def stop_and_hold(self, *, reason: str) -> Mapping[str, Any]:
        return {
            "replay_only": True,
            "held": True,
            "reason": reason,
            "hardware_commands_dispatched": False,
        }

    def close(self) -> None:
        return None


def _nominal_synthetic_observations(
    backend: PoseBackendDescriptor,
) -> list[Any]:
    """Build a deterministic, physically inert trace spanning all seven stages.

    The fixture exists only to exercise the same controller and logging path as
    live mode.  Its identity grasp relation and scene transforms are synthetic
    and must never be promoted to robot calibration.
    """

    backend_payload = {
        "name": backend.name,
        "available": backend.available,
        "runtime_qualified": backend.runtime_qualified,
        "qualification_id": backend.qualification_id,
        "capabilities": sorted(item.value for item in backend.capabilities),
        "detail": backend.detail,
    }

    def transform(x: float, y: float, z: float) -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ]

    tube_source = transform(0.50, 0.00, 0.10)
    top_pregrasp = transform(0.50, 0.00, 0.16)
    grasp = transform(0.50, 0.00, 0.10)
    extracted = transform(0.50, 0.00, 0.14)
    socket = transform(0.70, 0.00, 0.10)
    socket_standoff = transform(0.70, 0.00, 0.15)
    socket_entry = transform(0.70, 0.00, 0.10)
    retreat = transform(0.70, 0.00, 0.15)

    def pose(
        object_id: str,
        now_s: float,
        value: list[list[float]],
    ) -> dict[str, Any]:
        tube = object_id == "tube"
        return {
            "object_id": object_id,
            "T_base_object": value,
            "observed_at_s": now_s - 0.01,
            "position_std_m": 0.0005,
            "orientation_std_deg": 0.5,
            "backend": backend_payload,
            "tracking_id": f"synthetic-{object_id}-track",
            "inference_latency_s": 0.01,
            "mask_id": f"synthetic-{object_id}-mask",
            "mask_iou": 0.95,
            "valid_depth_count": 200,
            "foreground_depth_support_count": 180,
            "foreground_depth_support_fraction": 0.90,
            "background_depth_fraction": 0.05,
            "masked_depth_cluster_qualified": True,
            "axis_sign_source": "open_end_detector" if tube else "not_applicable",
            "cylindrical_roll_canonicalized": tube,
        }

    def observation(
        index: int,
        now_s: float,
        tcp: list[list[float]],
        tube: list[list[float]],
        *,
        width_m: float = 0.040,
        moving: bool = False,
        open_confirmed: bool = False,
        contact: bool = False,
        bilateral: bool = False,
        angle_reached: bool = False,
        rotation_stopped: bool = False,
        **flags: Any,
    ) -> Any:
        payload: dict[str, Any] = {
            "observation_id": f"synthetic-{index:02d}",
            "now_s": now_s,
            "robot": {
                "T_base_tcp": tcp,
                "motion_complete": True,
                "bounded_tcp_servo_supported": True,
            },
            "gripper": {
                "width_m": width_m,
                "moving": moving,
                "fault_free": True,
                "open_confirmed": open_confirmed,
            },
            "tactile": {
                "valid": True,
                "observed_at_s": now_s - 0.01,
                "contact_present": contact,
                "bilateral_contact": bilateral,
                "relative_angle_target_reached": angle_reached,
                "rotation_stopped": rotation_stopped,
            },
            "tube_pose": pose("tube", now_s, tube),
            "socket_pose": pose("socket", now_s, socket),
        }
        payload.update(flags)
        return parse_observation(payload)

    return [
        observation(0, 0.1, transform(0.30, 0.00, 0.30), tube_source),
        observation(1, 0.2, top_pregrasp, tube_source),
        observation(2, 0.3, grasp, tube_source),
        observation(3, 0.4, grasp, tube_source, width_m=0.030, moving=True),
        observation(
            4,
            0.5,
            grasp,
            tube_source,
            width_m=0.015,
            moving=True,
            contact=True,
            bilateral=True,
        ),
        observation(
            5,
            0.6,
            grasp,
            tube_source,
            width_m=0.015,
            contact=True,
            bilateral=True,
        ),
        observation(
            6,
            0.7,
            extracted,
            extracted,
            width_m=0.015,
            contact=True,
            bilateral=True,
            tube_clear_of_source=True,
        ),
        observation(
            7,
            0.8,
            extracted,
            extracted,
            width_m=0.016,
            contact=True,
            bilateral=True,
        ),
        observation(
            8,
            0.9,
            extracted,
            extracted,
            width_m=0.016,
            contact=True,
            bilateral=True,
            angle_reached=True,
        ),
        observation(
            9,
            1.0,
            extracted,
            extracted,
            width_m=0.015,
            contact=True,
            bilateral=True,
            angle_reached=True,
            rotation_stopped=True,
        ),
        observation(
            10,
            1.1,
            socket_standoff,
            socket_standoff,
            width_m=0.015,
            contact=True,
            bilateral=True,
        ),
        observation(
            11,
            1.2,
            socket_entry,
            socket_entry,
            width_m=0.015,
            contact=True,
            bilateral=True,
            tube_tip_at_socket_entry=True,
            tube_socket_axes_aligned=True,
        ),
        observation(
            12,
            1.3,
            socket_entry,
            socket_entry,
            width_m=0.015,
            contact=True,
            bilateral=True,
        ),
        observation(
            13,
            1.4,
            socket_entry,
            socket_entry,
            width_m=0.015,
            contact=True,
            bilateral=True,
            insertion_progress_m=0.0015,
            insertion_depth_m=0.0015,
        ),
        observation(
            14,
            1.5,
            socket_entry,
            socket_entry,
            width_m=0.015,
            contact=True,
            bilateral=True,
        ),
        observation(
            15,
            1.6,
            socket_entry,
            socket_entry,
            width_m=0.015,
            contact=True,
            bilateral=True,
            insertion_progress_m=0.0015,
            insertion_depth_m=0.003,
        ),
        observation(
            16,
            1.7,
            socket_entry,
            socket_entry,
            width_m=0.015,
            contact=True,
            bilateral=True,
            tube_stays_supported=True,
        ),
        observation(
            17,
            1.8,
            socket_entry,
            socket_entry,
            width_m=0.050,
            open_confirmed=True,
            tube_stays_supported=True,
        ),
        observation(
            18,
            1.9,
            retreat,
            socket_entry,
            width_m=0.050,
            open_confirmed=True,
            tube_stays_supported=True,
        ),
    ]


def _load_live_adapter(
    spec: str,
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    accept_arm_only_collision_model: bool = False,
    accept_provisional_camera_extrinsic: bool = False,
) -> ClosedLoopIO:
    if ":" not in spec:
        raise ClosedLoopContractError("--live-adapter must be module:function")
    module_name, factory_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ClosedLoopContractError(f"live adapter factory is not callable: {spec}")
    factory_kwargs: dict[str, Any] = {
        "config": config,
        "output_dir": output_dir,
    }
    if accept_arm_only_collision_model:
        factory_kwargs["accept_arm_only_collision_model"] = True
    if accept_provisional_camera_extrinsic:
        factory_kwargs["accept_provisional_camera_extrinsic"] = True
    adapter = factory(**factory_kwargs)
    if not isinstance(adapter, ClosedLoopIO):
        close = getattr(adapter, "close", None)
        if callable(close):
            try:
                close()
            except Exception as error:  # noqa: BLE001 - retain protocol failure context
                raise ClosedLoopContractError(
                    f"live adapter does not implement ClosedLoopIO and cleanup failed: {error}"
                ) from error
        raise ClosedLoopContractError("live adapter does not implement ClosedLoopIO")
    return adapter


def _write_step(
    output_dir: Path,
    index: int,
    payload: Mapping[str, Any],
) -> Path:
    path = output_dir / "steps" / f"{index:04d}.json"
    atomic_write_json(path, payload)
    return path


def _observation_json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"unsupported observation value: {type(value).__name__}")


class _TerminationRequested(Exception):
    """Convert SIGTERM into the same fail-closed path as an operator interrupt."""


class _ObservationStreamEnded(Exception):
    """Distinguish normal observation exhaustion from an EOF inside dispatch."""


def _best_effort_hold(io: ClosedLoopIO, *, reason: str) -> dict[str, Any]:
    """Stop both actuators without allowing a failed stop to hide the run outcome."""

    try:
        return dict(io.stop_and_hold(reason=reason))
    except Exception as error:  # noqa: BLE001 - the failure belongs in the report
        return {"hold_error": str(error)}


def _close_and_record(io: ClosedLoopIO, report: dict[str, Any]) -> None:
    """Close all owned resources while preserving any earlier run outcome."""

    try:
        io.close()
    except Exception as error:  # noqa: BLE001 - the failure belongs in the report
        report["close_error"] = str(error)


def _run_loop(
    *,
    controller: TestTubeClosedLoopController,
    io: ClosedLoopIO,
    output_dir: Path,
    report: dict[str, Any],
    max_steps: int,
    replay_only: bool,
) -> dict[str, Any]:
    report["status"] = "running"
    atomic_write_json(output_dir / "run_report.json", report)
    previous_sigterm_handler: Any | None = None
    sigterm_handler_installed = False
    if threading.current_thread() is threading.main_thread():
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

        def request_termination(_signum: int, _frame: Any) -> None:
            raise _TerminationRequested("received SIGTERM")

        signal.signal(signal.SIGTERM, request_termination)
        sigterm_handler_installed = True
    try:
        for index in range(max_steps):
            try:
                observation = io.observe(state=controller.state)
            except EOFError as error:
                raise _ObservationStreamEnded(str(error)) from error
            decision = controller.step(observation)
            step = {
                "index": index,
                "observed_at_s": observation.now_s,
                "observation_id": observation.observation_id,
                "observation": json.loads(
                    json.dumps(asdict(observation), default=_observation_json_default)
                ),
                "perception_diagnostics": getattr(io, "observation_diagnostics", {}),
                "decision": decision.to_dict(),
                "dispatch": {
                    "status": "pending" if decision.action.dispatches_hardware else "not_required",
                    "receipt": None,
                },
            }
            step_path = _write_step(output_dir, index, step)
            try:
                if decision.action.kind is ActionKind.STOP_AND_HOLD:
                    receipt = io.stop_and_hold(reason=decision.action.reason)
                    step["dispatch"] = {"status": "held", "receipt": dict(receipt)}
                    _write_step(output_dir, index, step)
                elif decision.action.dispatches_hardware:
                    if not replay_only:
                        report["hardware_commands_dispatched"] = True
                    receipt = io.dispatch(decision.action)
                    step["dispatch"] = {
                        "status": "completed",
                        "receipt": dict(receipt),
                    }
                    _write_step(output_dir, index, step)
            except Exception as error:
                step["dispatch"] = {
                    "status": "failed",
                    "receipt": None,
                    "error": str(error),
                }
                _write_step(output_dir, index, step)
                report["step_count"] = index + 1
                report["last_step"] = str(step_path)
                raise

            report["step_count"] = index + 1
            report["last_step"] = str(step_path)
            report["final_state"] = controller.state.value
            report["status"] = (
                "complete"
                if controller.state is TubeState.COMPLETE
                else "held"
                if controller.state is TubeState.HOLD
                else "running"
            )
            atomic_write_json(output_dir / "run_report.json", report)
            if decision.terminal:
                break
    except _ObservationStreamEnded as error:
        report["status"] = "replay_exhausted" if replay_only else "observation_stream_ended"
        report["final_state"] = TubeState.HOLD.value
        report["error"] = str(error)
        report["hold_receipt"] = _best_effort_hold(io, reason=f"observation stream ended: {error}")
    except (KeyboardInterrupt, _TerminationRequested) as error:
        report["status"] = "interrupted"
        report["final_state"] = TubeState.HOLD.value
        report["error"] = str(error) or type(error).__name__
        report["hold_receipt"] = _best_effort_hold(
            io, reason=f"runner interrupted: {report['error']}"
        )
    except Exception as error:
        report["status"] = "failed_closed"
        report["final_state"] = TubeState.HOLD.value
        report["error"] = str(error)
        report["hold_receipt"] = _best_effort_hold(io, reason=f"runner failure: {error}")
    else:
        if controller.state not in {TubeState.COMPLETE, TubeState.HOLD}:
            report["status"] = "step_budget_exhausted"
            report["final_state"] = TubeState.HOLD.value
            report["error"] = "maximum runtime step budget exhausted"
            report["hold_receipt"] = _best_effort_hold(
                io, reason="maximum runtime step budget exhausted"
            )
    finally:
        try:
            _close_and_record(io, report)
        finally:
            try:
                atomic_write_json(output_dir / "run_report.json", report)
            finally:
                if sigterm_handler_installed:
                    signal.signal(signal.SIGTERM, previous_sigterm_handler)
    return report


def run(args: argparse.Namespace) -> int:
    if args.max_steps < 1:
        raise ClosedLoopContractError("--max-steps must be positive")
    config_path = args.config.expanduser().resolve()
    config = _read_json(config_path)
    if config.get("schema") != CLOSED_LOOP_CONFIG_SCHEMA:
        raise ClosedLoopContractError(f"config.schema must be {CLOSED_LOOP_CONFIG_SCHEMA!r}")
    task_id = config.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ClosedLoopContractError("config.task_id must be a non-empty string")
    execution_scope = config.get("execution_scope", "live_candidate")
    if execution_scope not in {"live_candidate", "synthetic_replay_only"}:
        raise ClosedLoopContractError(
            "config.execution_scope must be live_candidate or synthetic_replay_only"
        )
    recovery_path, recovery_provenance = None, None
    if args.recover_from_run is not None:
        recovery_path, recovery_provenance = load_startup_recovery(
            args.recover_from_run, task_id=task_id
        )
    output_dir = _prepare_output(args.output_dir, overwrite=args.overwrite)
    collision_model_acknowledgement = _collision_model_acknowledgement(
        config,
        accepted=args.accept_arm_only_collision_model,
    )
    camera_extrinsic_acknowledgement = _camera_extrinsic_acknowledgement(
        config,
        accepted=args.accept_provisional_camera_extrinsic,
    )
    static_live_blockers = _static_live_adapter_blockers(
        config,
        accept_arm_only_collision_model=args.accept_arm_only_collision_model,
        accept_provisional_camera_extrinsic=args.accept_provisional_camera_extrinsic,
    )
    try:
        geometry = parse_geometry(config.get("geometry"))
    except (ClosedLoopContractError, TypeError, ValueError) as error:
        if not static_live_blockers:
            raise
        _write_unparsed_live_blocker_report(
            args=args,
            config_path=config_path,
            output_dir=output_dir,
            task_id=task_id,
            execution_scope=execution_scope,
            blockers=static_live_blockers,
            geometry_error=error,
            collision_model_acknowledgement=collision_model_acknowledgement,
            camera_extrinsic_acknowledgement=camera_extrinsic_acknowledgement,
            startup_recovery=recovery_provenance,
        )
        return 2
    policy = parse_policy(config.get("policy"))
    descriptor, backend_blockers, backend_detail = _backend_preflight(
        config.get("pose_backend"),
        config_dir=config_path.parent,
        online_composed=_uses_online_composed_scene(config),
    )
    if (descriptor.name == "synthetic_replay") != (execution_scope == "synthetic_replay_only"):
        raise ClosedLoopContractError(
            "pose_backend synthetic_replay and execution_scope "
            "synthetic_replay_only must be selected together"
        )
    geometry_blockers: list[str] = []
    if not geometry.relation_runtime_qualified:
        geometry_blockers.append("demo_T_tube_tcp_not_runtime_qualified")
    if not geometry.socket_entry_runtime_qualified:
        geometry_blockers.append("T_socket_tube_entry_not_runtime_qualified")
    live_blockers = sorted(set(backend_blockers + geometry_blockers + list(static_live_blockers)))
    report: dict[str, Any] = {
        "schema": CLOSED_LOOP_REPORT_SCHEMA,
        "created_at": _now(),
        "task_id": task_id,
        "config": str(config_path),
        "mode": "execute" if args.execute else "replay" if args.replay else "dry_run",
        "execution_scope": execution_scope,
        "status": "validated",
        "final_state": TubeState.OBSERVE.value,
        "step_count": 0,
        "hardware_commands_dispatched": False,
        "hardware_opened": False,
        "startup_recovery": recovery_provenance,
        "top_approach_hard_constraint": True,
        "demonstrated_relation": {
            "relation_id": geometry.relation_id,
            "kind": (
                "runtime_grounded_normalized_top_down_relation"
                if geometry.normalized_demo_grasp is not None
                else "full_T_tube_tcp_in_signed_roll_canonical_tube_frame"
            ),
            "source": geometry.relation_source,
            "runtime_qualified": geometry.relation_runtime_qualified,
            "human_trajectory_replayed": False,
            "normalized_constraints": (
                None
                if geometry.normalized_demo_grasp is None
                else {
                    "axial_fraction_closed_to_open": (
                        geometry.normalized_demo_grasp.axial_fraction_closed_to_open
                    ),
                    "tube_length_m": geometry.normalized_demo_grasp.tube_length_m,
                    "tube_length_source": (geometry.normalized_demo_grasp.tube_length_source),
                    "tube_pose_origin": (geometry.normalized_demo_grasp.tube_pose_origin),
                    "tube_axis_contract": (geometry.normalized_demo_grasp.tube_axis_contract),
                    "top_approach_direction_base": (
                        geometry.normalized_demo_grasp.top_approach_direction_base.tolist()
                    ),
                    "preferred_jaw_axis_base": (
                        geometry.normalized_demo_grasp.preferred_jaw_axis_base.tolist()
                    ),
                }
            ),
        },
        "socket_entry_relation": {
            "source": geometry.socket_entry_source,
            "runtime_qualified": geometry.socket_entry_runtime_qualified,
        },
        "pose_backend": {
            "descriptor": {
                "name": descriptor.name,
                "available": descriptor.available,
                "runtime_qualified": descriptor.runtime_qualified,
                "qualification_id": descriptor.qualification_id,
                "capabilities": sorted(item.value for item in descriptor.capabilities),
                "detail": descriptor.detail,
            },
            "preflight": backend_detail,
            "blockers": live_blockers,
        },
        "collision_model": collision_model_acknowledgement,
        "camera_extrinsic": camera_extrinsic_acknowledgement,
        "state_contract": state_contract(),
        "fail_closed_boundaries": [
            "unsigned tube axis is rejected until open/closed ends are disambiguated",
            "cylindrical roll is canonicalized and is not treated as observed task state",
            "gripper occlusion freezes the last qualified pose and switches to tactile",
            "RGB-D axis/anchor fallback cannot authorize SE(3) servo or insertion",
            "one-sided contact uses a bounded jaw-axis sign probe; sensor side is not assumed",
            "jam requires low progress plus qualified tactile resistance",
        ],
    }
    atomic_write_json(output_dir / "run_report.json", report)

    if args.replay is None and not args.execute:
        report["status"] = (
            "synthetic_replay_ready"
            if execution_scope == "synthetic_replay_only" and not live_blockers
            else "blocked_live_execution"
            if live_blockers
            else "ready_for_external_live_adapter"
        )
        atomic_write_json(output_dir / "run_report.json", report)
        return 2 if live_blockers else 0

    controller = TestTubeClosedLoopController(
        geometry=geometry, policy=policy, startup_recovery_path=recovery_path
    )
    if args.replay is not None:
        replay = _read_json(args.replay.expanduser().resolve())
        io: ClosedLoopIO = ReplayIO(replay)
        report["pose_backend"]["replay_descriptor"] = {
            "name": io.pose_backend_descriptor.name,
            "runtime_qualified": io.pose_backend_descriptor.runtime_qualified,
            "capabilities": sorted(item.value for item in io.pose_backend_descriptor.capabilities),
        }
        report = _run_loop(
            controller=controller,
            io=io,
            output_dir=output_dir,
            report=report,
            max_steps=args.max_steps,
            replay_only=True,
        )
        return 0 if report["status"] == "complete" else 2

    if execution_scope == "synthetic_replay_only":
        report["status"] = "blocked_live_execution"
        report["pose_backend"]["blockers"].append("synthetic_replay_scope_cannot_execute_hardware")
        atomic_write_json(output_dir / "run_report.json", report)
        return 2
    if (
        collision_model_acknowledgement["receipt_state"]
        == "missing_not_acknowledged"
    ):
        report["status"] = "blocked_live_execution"
        report["pose_backend"]["blockers"].append(
            "arm_only_collision_model_not_explicitly_acknowledged"
        )
        atomic_write_json(output_dir / "run_report.json", report)
        return 2
    if "realtime" not in platform.release().lower():
        report["status"] = "blocked_live_execution"
        report["pose_backend"]["blockers"].append("host_kernel_is_not_realtime")
        atomic_write_json(output_dir / "run_report.json", report)
        return 2
    if live_blockers:
        report["status"] = "blocked_live_execution"
        atomic_write_json(output_dir / "run_report.json", report)
        return 2
    if args.live_adapter is None:
        report["status"] = "blocked_live_execution"
        report["pose_backend"]["blockers"].append("live_adapter_not_supplied")
        atomic_write_json(output_dir / "run_report.json", report)
        return 2

    io = _load_live_adapter(
        args.live_adapter,
        config=config,
        output_dir=output_dir,
        accept_arm_only_collision_model=args.accept_arm_only_collision_model,
        accept_provisional_camera_extrinsic=args.accept_provisional_camera_extrinsic,
    )
    report["hardware_opened"] = True
    try:
        atomic_write_json(output_dir / "run_report.json", report)
        adapter_descriptor = io.pose_backend_descriptor
        adapter_blockers = adapter_descriptor.blockers(ALL_LIVE_CAPABILITIES)
        descriptor_matches = bool(
            adapter_descriptor.name == descriptor.name
            and adapter_descriptor.available == descriptor.available
            and adapter_descriptor.runtime_qualified == descriptor.runtime_qualified
            and adapter_descriptor.qualification_id == descriptor.qualification_id
            and adapter_descriptor.capabilities == descriptor.capabilities
        )
    except Exception as error:
        report["status"] = "failed_closed"
        report["final_state"] = TubeState.HOLD.value
        report["error"] = f"live adapter startup failed: {error}"
        report["hold_receipt"] = _best_effort_hold(io, reason=report["error"])
        _close_and_record(io, report)
        atomic_write_json(output_dir / "run_report.json", report)
        return 2
    if adapter_blockers or not descriptor_matches:
        report["status"] = "blocked_live_execution"
        report["pose_backend"]["blockers"].extend(adapter_blockers)
        if not descriptor_matches:
            report["pose_backend"]["blockers"].append(
                "live_adapter_descriptor_does_not_match_preflight_receipt"
            )
        report["hold_receipt"] = _best_effort_hold(
            io, reason="live adapter descriptor failed runtime preflight"
        )
        _close_and_record(io, report)
        atomic_write_json(output_dir / "run_report.json", report)
        return 2
    report = _run_loop(
        controller=controller,
        io=io,
        output_dir=output_dir,
        report=report,
        max_steps=args.max_steps,
        replay_only=False,
    )
    return 0 if report["status"] == "complete" else 2


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (ClosedLoopContractError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
