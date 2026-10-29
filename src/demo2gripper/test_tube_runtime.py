"""Hardware-independent runtime orchestration for the test-tube task.

The preflight compiler defines the state machine and named predicates.  This
module supplies the missing orchestration boundary: it repeatedly asks an
injected predicate provider for one evidence snapshot, evaluates exactly one
transition with :func:`decide_test_tube_transition`, and asks an injected
semantic-action executor for a receipt.  The core imports no robot, ROS,
MoveIt, gripper, camera, or tactile driver.

The runtime is deliberately fail closed.  It will not start a replay unless
the compiled plan says that the offline contract, metric geometry, plan-only
motion, and runtime gates are all ready, and a fresh qualified live-geometry
bundle is supplied.  Every consumed evidence snapshot, transition decision,
and semantic-action result is written atomically before the next step.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .test_tube_preflight import (
    TEST_TUBE_PLAN_SCHEMA,
    TransitionDecision,
    TubeState,
    decide_test_tube_transition,
)

TEST_TUBE_RUNTIME_GEOMETRY_SCHEMA = "demo2gripper.test_tube_runtime_geometry.v1"
TEST_TUBE_RUNTIME_REPLAY_SCHEMA = "demo2gripper.test_tube_runtime_replay.v1"
TEST_TUBE_RUNTIME_REPORT_SCHEMA = "demo2gripper.test_tube_runtime_report.v1"
TEST_TUBE_RUNTIME_STEP_SCHEMA = "demo2gripper.test_tube_runtime_step.v1"
TEST_TUBE_PREDICATE_RECEIPT_SCHEMA = "demo2gripper.test_tube_predicate_receipt.v2"

_RUNTIME_BUDGET_PREDICATES = frozenset(
    {
        "grasp_close_budget_remaining",
        "grasp_close_budget_exhausted",
        "pivot_budget_remaining",
        "jam_recovery_budget_remaining",
    }
)


class RuntimeContractError(ValueError):
    """A plan, geometry bundle, replay, or provider receipt is malformed."""


@dataclass(frozen=True)
class PredicateSnapshot:
    """One immutable, provenance-carrying predicate snapshot."""

    sequence: int
    state: TubeState
    evidence_id: str
    observed_at_s: float
    predicates: tuple[str, ...]
    evidence_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class SemanticActionResult:
    """Receipt returned after one semantic action request."""

    success: bool
    receipt_id: str
    receipt: Mapping[str, Any]
    error: str | None = None


class PredicateProvider(Protocol):
    """Supplies fresh predicate evidence without owning state transitions."""

    def observe(
        self,
        *,
        state: TubeState,
        step_index: int,
        budgets: Mapping[str, int],
    ) -> PredicateSnapshot: ...


class SemanticActionExecutor(Protocol):
    """Executes one named semantic action and returns its durable receipt."""

    def execute(
        self,
        *,
        decision: TransitionDecision,
        evidence: PredicateSnapshot,
        step_index: int,
    ) -> SemanticActionResult: ...


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeContractError(f"{name} must be an object")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeContractError(f"{name} must be a non-empty string")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeContractError(f"{name} must be boolean")
    return value


def _integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeContractError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise RuntimeContractError(f"{name} is outside the allowed range")
    return value


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    include_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeContractError(f"{name} must be a finite number")
    if minimum is not None:
        invalid = result < minimum if include_minimum else result <= minimum
        if invalid:
            relation = ">=" if include_minimum else ">"
            raise RuntimeContractError(f"{name} must be {relation} {minimum}")
    return result


def _json_safe(value: object, *, name: str) -> Any:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise RuntimeContractError(f"{name} must be finite JSON data: {error}") from error


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _string_array(value: object, *, name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise RuntimeContractError(f"{name} must be an array of non-empty strings")
    if len(value) != len(set(value)):
        raise RuntimeContractError(f"{name} must not contain duplicates")
    return list(value)


def _canonical_predicate_contract() -> tuple[
    dict[str, dict[str, Any]], dict[str, tuple[str, ...]]
]:
    """Load the builder-owned contract after this module has initialized.

    ``test_tube_predicates`` imports :class:`PredicateSnapshot`, so importing it
    at module scope would create a cycle.  Runtime entry points are called only
    after both modules can be initialized; the local import keeps one canonical
    manifest without copying it into the orchestrator.
    """

    from .test_tube_predicates import (  # noqa: PLC0415
        predicate_producer_manifest,
        required_predicate_producers_by_state,
    )

    manifest = predicate_producer_manifest()
    required = required_predicate_producers_by_state()
    return manifest, {
        state: tuple(producers) for state, producers in required.items()
    }


def _normalize_predicate_manifest(value: object) -> dict[str, dict[str, Any]]:
    manifest = _mapping(value, name="compiled_plan.predicate_producer_manifest")
    normalized: dict[str, dict[str, Any]] = {}
    valid_states = {state.value for state in TubeState}
    for predicate, raw_entry in manifest.items():
        predicate_name = _string(predicate, name="predicate manifest key")
        entry = _mapping(
            raw_entry,
            name=f"compiled_plan.predicate_producer_manifest.{predicate_name}",
        )
        producer = _string(
            entry.get("producer"),
            name=f"predicate manifest {predicate_name}.producer",
        )
        raw_states = entry.get("states")
        if raw_states == "all":
            states: str | list[str] = "all"
        else:
            states = _string_array(
                raw_states,
                name=f"predicate manifest {predicate_name}.states",
            )
            if not states or set(states) - valid_states:
                raise RuntimeContractError(
                    f"predicate manifest {predicate_name}.states contains invalid states"
                )
            states = sorted(states)
        if set(entry) != {"producer", "states"}:
            raise RuntimeContractError(
                f"predicate manifest {predicate_name} has unexpected fields"
            )
        normalized[predicate_name] = {"producer": producer, "states": states}
    if not normalized:
        raise RuntimeContractError("compiled predicate producer manifest is empty")
    return dict(sorted(normalized.items()))


def _rigid_transform(value: object, *, name: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise RuntimeContractError(f"{name} must be a 4x4 transform")
    result: list[list[float]] = []
    for row_index, raw_row in enumerate(value):
        if (
            not isinstance(raw_row, Sequence)
            or isinstance(raw_row, (str, bytes))
            or len(raw_row) != 4
        ):
            raise RuntimeContractError(f"{name} must be a 4x4 transform")
        result.append(
            [
                _number(item, name=f"{name}[{row_index}]", minimum=None)
                for item in raw_row
            ]
        )
    if any(abs(value - expected) > 1e-8 for value, expected in zip(result[3], [0, 0, 0, 1])):
        raise RuntimeContractError(f"{name} must have homogeneous final row [0,0,0,1]")

    # Check the 3x3 rotation without importing a numerical or robot stack.
    rotation = [row[:3] for row in result[:3]]
    for left in range(3):
        for right in range(3):
            dot = sum(rotation[row][left] * rotation[row][right] for row in range(3))
            expected = 1.0 if left == right else 0.0
            if abs(dot - expected) > 1e-6:
                raise RuntimeContractError(f"{name} rotation must be orthonormal")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 1e-6:
        raise RuntimeContractError(f"{name} rotation must be right-handed")
    return result


def _unit_vector3(value: object, *, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise RuntimeContractError(f"{name} must contain three finite values")
    result = [_number(item, name=name) for item in value]
    norm = math.sqrt(sum(item * item for item in result))
    if abs(norm - 1.0) > 1e-6:
        raise RuntimeContractError(f"{name} must be unit length")
    return result


def validate_runtime_geometry(
    value: Mapping[str, Any],
    *,
    task_id: str,
    planned_insertion_depth_m: float,
) -> dict[str, Any]:
    """Validate the fresh metric transforms required before any runtime step."""

    geometry = _mapping(value, name="live_geometry")
    if geometry.get("schema") != TEST_TUBE_RUNTIME_GEOMETRY_SCHEMA:
        raise RuntimeContractError(
            f"live_geometry.schema must be {TEST_TUBE_RUNTIME_GEOMETRY_SCHEMA!r}"
        )
    if geometry.get("task_id") != task_id:
        raise RuntimeContractError("live_geometry.task_id does not match the compiled plan")
    if geometry.get("status") != "runtime_qualified":
        raise RuntimeContractError("live_geometry.status must be runtime_qualified")
    if geometry.get("freshness_verified") is not True:
        raise RuntimeContractError("live_geometry must state freshness_verified=true")
    if geometry.get("transform_convention") != "T_parent_child; metres; right-handed":
        raise RuntimeContractError("live_geometry transform convention is invalid")

    depth = _number(
        geometry.get("insertion_target_depth_m"),
        name="live_geometry.insertion_target_depth_m",
        minimum=0.0,
    )
    if not math.isclose(depth, planned_insertion_depth_m, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeContractError(
            "live insertion target depth disagrees with the compiled motion limit"
        )
    normalized = {
        "schema": TEST_TUBE_RUNTIME_GEOMETRY_SCHEMA,
        "task_id": task_id,
        "status": "runtime_qualified",
        "freshness_verified": True,
        "observation_id": _string(
            geometry.get("observation_id"), name="live_geometry.observation_id"
        ),
        "metric_localization_receipt_id": _string(
            geometry.get("metric_localization_receipt_id"),
            name="live_geometry.metric_localization_receipt_id",
        ),
        "collision_scene_id": _string(
            geometry.get("collision_scene_id"), name="live_geometry.collision_scene_id"
        ),
        "contact_profile_id": _string(
            geometry.get("contact_profile_id"), name="live_geometry.contact_profile_id"
        ),
        "front_rgb_tracker_session_id": _string(
            geometry.get("front_rgb_tracker_session_id"),
            name="live_geometry.front_rgb_tracker_session_id",
        ),
        "insertion_jam_profile_id": _string(
            geometry.get("insertion_jam_profile_id"),
            name="live_geometry.insertion_jam_profile_id",
        ),
        "transform_convention": geometry["transform_convention"],
        "T_base_tube": _rigid_transform(
            geometry.get("T_base_tube"), name="live_geometry.T_base_tube"
        ),
        "T_base_socket": _rigid_transform(
            geometry.get("T_base_socket"), name="live_geometry.T_base_socket"
        ),
        "T_tube_tcp_side_grasp": _rigid_transform(
            geometry.get("T_tube_tcp_side_grasp"),
            name="live_geometry.T_tube_tcp_side_grasp",
        ),
        "insertion_axis_base": _unit_vector3(
            geometry.get("insertion_axis_base"),
            name="live_geometry.insertion_axis_base",
        ),
        "insertion_target_depth_m": depth,
    }
    return normalized


def _runtime_plan_contract(plan: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if plan.get("schema") != TEST_TUBE_PLAN_SCHEMA:
        raise RuntimeContractError(f"compiled plan schema must be {TEST_TUBE_PLAN_SCHEMA!r}")
    task_id = _string(plan.get("task_id"), name="compiled_plan.task_id")
    preflight = _mapping(plan.get("preflight"), name="compiled_plan.preflight")
    blockers: list[str] = []
    canonical_manifest_raw, required_producers = _canonical_predicate_contract()
    canonical_manifest = _normalize_predicate_manifest(canonical_manifest_raw)
    try:
        compiled_manifest = _normalize_predicate_manifest(
            plan.get("predicate_producer_manifest")
        )
    except RuntimeContractError as error:
        blockers.append(f"compiled_predicate_producer_manifest_invalid:{error}")
        compiled_manifest = canonical_manifest
    if compiled_manifest != canonical_manifest:
        blockers.append("compiled_predicate_producer_manifest_not_canonical")
    if plan.get("execution_authorized") is not True:
        blockers.append("compiled_execution_authorized_false")
    if plan.get("hardware_execution_enabled") is not True:
        blockers.append("compiled_hardware_execution_enabled_false")
    readiness = (
        "offline_task_contract_ready",
        "metric_execution_geometry_ready",
        "moveit_plan_only_ready",
        "runtime_transition_gates_ready",
        "hardware_execution_ready",
    )
    for field in readiness:
        if preflight.get(field) is not True:
            blockers.append(f"compiled_preflight_{field}_false")
    raw_blockers = preflight.get("blocking_conditions")
    if not isinstance(raw_blockers, list) or any(
        not isinstance(item, str) for item in raw_blockers
    ):
        raise RuntimeContractError("compiled_plan.preflight.blocking_conditions must be strings")
    blockers.extend(f"compiled:{item}" for item in raw_blockers)

    execution_readiness = _mapping(
        plan.get("execution_readiness"), name="compiled_plan.execution_readiness"
    )
    readiness_fields = (
        "exact_tube_tcp_instantiated",
        "gripper_width_and_close_budget_grounded",
        "collision_scene_validated",
        "motion_bounds_validated",
        "offline_runtime_orchestrator_available",
        "offline_predicate_builder_available",
        "rgb_tracker_qualification_tool_available",
        "live_hardware_adapter_available",
        "live_feedback_predicate_producers_available",
        "support_after_release_verifier_available",
        "robot_command_state_path_healthy",
        "long_lived_front_rgb_source_available",
        "front_rgb_stream_healthy",
    )
    for field in readiness_fields:
        if execution_readiness.get(field) is not True:
            blockers.append(f"compiled_execution_readiness_{field}_false")

    motion = _mapping(plan.get("motion_limits"), name="compiled_plan.motion_limits")
    depth = motion.get("insertion_target_depth_m")
    if depth is None:
        blockers.append("compiled_insertion_target_depth_missing")
        normalized_depth = None
    else:
        normalized_depth = _number(
            depth,
            name="compiled_plan.motion_limits.insertion_target_depth_m",
            minimum=0.0,
        )
    insertion_step = _number(
        motion.get("insertion_step_m"),
        name="compiled_plan.motion_limits.insertion_step_m",
        minimum=0.0,
    )
    limits = _mapping(plan.get("recovery_limits"), name="compiled_plan.recovery_limits")
    pivot_max = _integer(
        limits.get("pivot_max_microsteps"),
        name="compiled_plan.recovery_limits.pivot_max_microsteps",
        minimum=1,
    )
    reclamp_max = _integer(
        limits.get("maximum_reclamp_steps"),
        name="compiled_plan.recovery_limits.maximum_reclamp_steps",
        minimum=1,
    )
    jam_max = _integer(
        limits.get("maximum_jam_recoveries"),
        name="compiled_plan.recovery_limits.maximum_jam_recoveries",
        minimum=0,
    )
    if limits.get("on_budget_exhausted") != "HOLD":
        blockers.append("compiled_budget_exhaustion_policy_is_not_hold")

    gates = _mapping(plan.get("gates"), name="compiled_plan.gates")
    pivot_target = _mapping(gates.get("pivot_target"), name="compiled_plan.gates.pivot_target")
    sources = pivot_target.get("allowed_target_sources")
    if (
        not isinstance(sources, list)
        or not sources
        or any(
            source not in {"front_rgb_calibrated", "tactile_relative_angle"}
            for source in sources
        )
        or len(sources) != len(set(sources))
    ):
        raise RuntimeContractError("compiled pivot target sources are invalid")
    if "front_rgb_calibrated" in sources:
        tracker = _mapping(
            plan.get("front_rgb_tracker_validation"),
            name="compiled_plan.front_rgb_tracker_validation",
        )
        reference = _mapping(
            plan.get("front_rgb_pivot_reference"),
            name="compiled_plan.front_rgb_pivot_reference",
        )
        if (
            pivot_target.get("front_rgb_runtime_gate_qualified") is not True
            or tracker.get("runtime_gate_eligible") is not True
            or reference.get("runtime_gate_eligible") is not True
        ):
            blockers.append("compiled_front_rgb_angle_gate_not_runtime_qualified")
    if "tactile_relative_angle" in sources and not pivot_target.get(
        "tactile_angle_calibration_id"
    ):
        blockers.append("compiled_tactile_angle_gate_not_runtime_qualified")
    jam_gate = _mapping(gates.get("insertion_jam"), name="compiled_plan.gates.insertion_jam")
    if jam_gate.get("may_trigger_transition") is not True:
        blockers.append("compiled_insertion_jam_gate_not_runtime_qualified")
    contact_gate = _mapping(gates.get("pivot_contact"), name="compiled_plan.gates.pivot_contact")
    if contact_gate.get("mode") not in {"runtime_qualified", "conditional_runtime_bootstrap"}:
        blockers.append("compiled_contact_gate_not_runtime_qualified")

    return (
        {
            "task_id": task_id,
            "allowed_angle_sources": tuple(sources),
            "planned_insertion_depth_m": normalized_depth,
            "insertion_step_m": insertion_step,
            "pivot_max_microsteps": pivot_max,
            "maximum_reclamp_steps": reclamp_max,
            "maximum_jam_recoveries": jam_max,
            "predicate_producer_manifest": compiled_manifest,
            "predicate_manifest_sha256": _digest(canonical_manifest_raw),
            "required_predicate_producers_by_state": required_producers,
            "slip_runtime_qualified": _mapping(
                gates.get("slip"), name="compiled_plan.gates.slip"
            ).get("may_trigger_transition")
            is True,
        },
        sorted(set(blockers)),
    )


def runtime_start_blockers(
    compiled_plan: Mapping[str, Any],
    live_geometry: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any] | None, dict[str, Any]]:
    """Return fail-closed start blockers plus normalized plan/geometry context."""

    plan_context, blockers = _runtime_plan_contract(compiled_plan)
    geometry: dict[str, Any] | None = None
    if plan_context["planned_insertion_depth_m"] is not None:
        try:
            geometry = validate_runtime_geometry(
                live_geometry,
                task_id=plan_context["task_id"],
                planned_insertion_depth_m=plan_context["planned_insertion_depth_m"],
            )
        except RuntimeContractError as error:
            blockers.append(f"live_geometry:{error}")
    else:
        blockers.append("live_geometry_cannot_be_checked_without_planned_insertion_depth")
    return sorted(set(blockers)), geometry, plan_context


class _Journal:
    def __init__(
        self,
        output_dir: Path,
        *,
        task_id: str,
        plan_digest: str,
        geometry_digest: str | None,
        budget_limits: Mapping[str, int],
    ) -> None:
        self.output_dir = output_dir
        self.steps_dir = output_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = output_dir / "runtime_report.json"
        self.report: dict[str, Any] = {
            "schema": TEST_TUBE_RUNTIME_REPORT_SCHEMA,
            "task_id": task_id,
            "mode": "hardware_independent_replay",
            "hardware_commands_dispatched_by_core": False,
            "status": "running",
            "initial_state": TubeState.OBSERVE.value,
            "final_state": TubeState.OBSERVE.value,
            "plan_digest_sha256": plan_digest,
            "live_geometry_digest_sha256": geometry_digest,
            "budget_limits": dict(budget_limits),
            "budget_usage": {},
            "blocking_conditions": [],
            "step_count": 0,
            "step_receipts": [],
            "pending_dispatch_intents": [],
        }
        self.flush()

    @staticmethod
    def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _step_path(self, payload: Mapping[str, Any]) -> tuple[int, Path]:
        step = _integer(payload.get("step_index"), name="step.step_index", minimum=0)
        return step, self.steps_dir / f"{step:04d}.json"

    def begin_dispatch(self, payload: Mapping[str, Any]) -> None:
        """Durably record evidence, decision, and action intent before dispatch."""

        step, path = self._step_path(payload)
        if path.exists():
            raise RuntimeError(f"refusing to overwrite runtime step receipt {path}")
        normalized = _json_safe(
            {**payload, "receipt_phase": "dispatch_intent_persisted"},
            name="step dispatch intent",
        )
        self._atomic_json(path, normalized)
        self.report["step_receipts"].append(str(path.relative_to(self.output_dir)))
        self.report["step_count"] = len(self.report["step_receipts"])
        self.report["pending_dispatch_intents"].append(step)
        self.flush()

    def complete_dispatch(self, payload: Mapping[str, Any]) -> None:
        """Atomically replace a persisted intent with its action result."""

        step, path = self._step_path(payload)
        if step not in self.report["pending_dispatch_intents"] or not path.is_file():
            raise RuntimeError(f"runtime step {step} has no persisted dispatch intent")
        normalized = _json_safe(
            {**payload, "receipt_phase": "action_result_persisted"},
            name="completed step receipt",
        )
        self._atomic_json(path, normalized)
        self.report["pending_dispatch_intents"].remove(step)
        self.flush()

    def record_without_dispatch(self, payload: Mapping[str, Any]) -> None:
        """Persist a HOLD decision when evidence was rejected before dispatch."""

        _, path = self._step_path(payload)
        if path.exists():
            raise RuntimeError(f"refusing to overwrite runtime step receipt {path}")
        normalized = _json_safe(
            {**payload, "receipt_phase": "not_dispatched"},
            name="non-dispatched step receipt",
        )
        self._atomic_json(path, normalized)
        self.report["step_receipts"].append(str(path.relative_to(self.output_dir)))
        self.report["step_count"] = len(self.report["step_receipts"])
        self.flush()

    def finish(
        self,
        *,
        status: str,
        final_state: TubeState,
        budgets: Mapping[str, int],
        blockers: Sequence[str] = (),
    ) -> dict[str, Any]:
        self.report["status"] = status
        self.report["final_state"] = final_state.value
        self.report["budget_usage"] = dict(budgets)
        self.report["blocking_conditions"] = list(blockers)
        self.flush()
        return dict(self.report)

    def flush(self) -> None:
        self._atomic_json(self.report_path, self.report)


def _validate_builder_receipt(
    receipt: Mapping[str, Any],
    *,
    snapshot: PredicateSnapshot,
    state: TubeState,
    task_id: str,
    predicate_manifest: Mapping[str, Mapping[str, Any]],
    predicate_manifest_sha256: str,
    required_producers_by_state: Mapping[str, Sequence[str]],
    expected_builder_identity: tuple[str, str] | None,
    expected_contact_profile_id: str,
    expected_insertion_profile_id: str,
) -> tuple[str, str]:
    expected_fields = {
        "schema",
        "predicate_manifest_sha256",
        "task_id",
        "session_id",
        "provenance_id",
        "state",
        "evaluated_at_s",
        "producer_receipts",
        "required_producers",
        "promoted_predicates",
        "stale_receipt_ids",
        "rejection_reasons",
        "shadow_observations",
    }
    if set(receipt) != expected_fields:
        raise RuntimeContractError(
            "snapshot evidence receipt fields do not match the predicate-builder schema"
        )
    if receipt.get("schema") != TEST_TUBE_PREDICATE_RECEIPT_SCHEMA:
        raise RuntimeContractError(
            "snapshot evidence receipt was not emitted under the supported "
            "predicate-builder schema"
        )
    if receipt.get("predicate_manifest_sha256") != predicate_manifest_sha256:
        raise RuntimeContractError(
            "snapshot evidence receipt is not bound to the compiled predicate manifest"
        )
    if receipt.get("task_id") != task_id:
        raise RuntimeContractError("snapshot evidence receipt task_id mismatch")
    if receipt.get("state") != state.value:
        raise RuntimeContractError("snapshot evidence receipt state mismatch")
    evaluated_at_s = _number(
        receipt.get("evaluated_at_s"),
        name="snapshot.evidence_receipt.evaluated_at_s",
        minimum=0.0,
        include_minimum=True,
    )
    if not math.isclose(evaluated_at_s, snapshot.observed_at_s, abs_tol=1e-12):
        raise RuntimeContractError(
            "snapshot timestamp is not bound to its predicate-builder receipt"
        )

    session_id = _string(
        receipt.get("session_id"), name="snapshot.evidence_receipt.session_id"
    )
    provenance_id = _string(
        receipt.get("provenance_id"), name="snapshot.evidence_receipt.provenance_id"
    )
    identity = (session_id, provenance_id)
    if expected_builder_identity is not None and identity != expected_builder_identity:
        raise RuntimeContractError(
            "predicate-builder session/provenance changed during the runtime"
        )

    promoted = _string_array(
        receipt.get("promoted_predicates"),
        name="snapshot.evidence_receipt.promoted_predicates",
    )
    if promoted != sorted(promoted) or tuple(promoted) != snapshot.predicates:
        raise RuntimeContractError(
            "snapshot predicates do not exactly match the predicate-builder receipt"
        )
    for predicate in promoted:
        entry = predicate_manifest.get(predicate)
        if entry is None:
            raise RuntimeContractError(
                f"predicate {predicate!r} is absent from the compiled producer manifest"
            )
        allowed_states = entry["states"]
        if allowed_states != "all" and state.value not in allowed_states:
            raise RuntimeContractError(
                f"predicate {predicate!r} is not allowed in state {state.value!r}"
            )

    required = _string_array(
        receipt.get("required_producers"),
        name="snapshot.evidence_receipt.required_producers",
    )
    expected_required = list(required_producers_by_state[state.value])
    if required != sorted(required) or required != expected_required:
        raise RuntimeContractError(
            "snapshot evidence receipt does not declare the builder-owned state requirements"
        )

    producer_receipts = _mapping(
        receipt.get("producer_receipts"),
        name="snapshot.evidence_receipt.producer_receipts",
    )
    allowed_producers = {
        producer
        for producers in required_producers_by_state.values()
        for producer in producers
    }
    if set(producer_receipts) - allowed_producers:
        raise RuntimeContractError("snapshot contains an unknown predicate producer")
    missing = set(required) - set(producer_receipts)
    if missing:
        raise RuntimeContractError(
            f"snapshot is missing required producer receipts: {sorted(missing)}"
        )
    receipt_ids: set[str] = set()
    for producer, raw_producer_receipt in producer_receipts.items():
        producer_receipt = _mapping(
            raw_producer_receipt,
            name=f"snapshot.evidence_receipt.producer_receipts.{producer}",
        )
        if set(producer_receipt) != {
            "receipt_id",
            "producer_id",
            "observed_at_s",
            "qualification_id",
        }:
            raise RuntimeContractError(
                f"snapshot producer receipt {producer!r} has unexpected fields"
            )
        receipt_id = _string(
            producer_receipt.get("receipt_id"),
            name=f"snapshot producer receipt {producer}.receipt_id",
        )
        if receipt_id in receipt_ids:
            raise RuntimeContractError("snapshot producer receipt IDs are not unique")
        receipt_ids.add(receipt_id)
        _string(
            producer_receipt.get("producer_id"),
            name=f"snapshot producer receipt {producer}.producer_id",
        )
        _number(
            producer_receipt.get("observed_at_s"),
            name=f"snapshot producer receipt {producer}.observed_at_s",
            minimum=0.0,
            include_minimum=True,
        )
        qualification_id = producer_receipt.get("qualification_id")
        if qualification_id is not None:
            qualification_id = _string(
                qualification_id,
                name=f"snapshot producer receipt {producer}.qualification_id",
            )
        if producer == "tactile" and qualification_id != expected_contact_profile_id:
            raise RuntimeContractError(
                "tactile producer receipt is not bound to the live contact profile"
            )
        if (
            producer == "insertion"
            and qualification_id != expected_insertion_profile_id
        ):
            raise RuntimeContractError(
                "insertion producer receipt is not bound to the live jam profile"
            )

    stale_ids = _string_array(
        receipt.get("stale_receipt_ids"),
        name="snapshot.evidence_receipt.stale_receipt_ids",
    )
    if not set(stale_ids) <= receipt_ids:
        raise RuntimeContractError("stale receipt IDs do not reference producer receipts")
    rejection_reasons = _string_array(
        receipt.get("rejection_reasons"),
        name="snapshot.evidence_receipt.rejection_reasons",
    )
    if bool(stale_ids) != ("stale_evidence" in promoted):
        raise RuntimeContractError("stale-evidence predicate and receipt list disagree")
    if bool(rejection_reasons) != ("invalid_evidence" in promoted):
        raise RuntimeContractError("invalid-evidence predicate and rejection list disagree")
    if set(promoted) & {"stale_evidence", "invalid_evidence"} and set(promoted) - {
        "stale_evidence",
        "invalid_evidence",
    }:
        raise RuntimeContractError(
            "invalid or stale builder evidence cannot promote control predicates"
        )
    shadow = _mapping(
        receipt.get("shadow_observations"),
        name="snapshot.evidence_receipt.shadow_observations",
    )
    if set(shadow) != {"slip_shadow_candidate", "jam_shadow_candidate"} or any(
        not isinstance(value, bool) for value in shadow.values()
    ):
        raise RuntimeContractError("snapshot shadow observations are malformed")
    if snapshot.evidence_id != _digest(receipt):
        raise RuntimeContractError(
            "snapshot evidence_id does not match the predicate-builder receipt digest"
        )
    return identity


def _normalize_snapshot(
    snapshot: PredicateSnapshot,
    *,
    state: TubeState,
    step_index: int,
    previous_time_s: float | None,
    seen_evidence_ids: set[str],
    task_id: str,
    predicate_manifest: Mapping[str, Mapping[str, Any]],
    predicate_manifest_sha256: str,
    required_producers_by_state: Mapping[str, Sequence[str]],
    expected_builder_identity: tuple[str, str] | None,
    expected_contact_profile_id: str,
    expected_insertion_profile_id: str,
) -> tuple[PredicateSnapshot, tuple[str, str]]:
    if not isinstance(snapshot, PredicateSnapshot):
        raise RuntimeContractError("predicate provider must return PredicateSnapshot")
    if snapshot.sequence != step_index:
        raise RuntimeContractError("predicate snapshot sequence does not match runtime step")
    if snapshot.state is not state:
        raise RuntimeContractError("predicate snapshot state does not match runtime state")
    evidence_id = _string(snapshot.evidence_id, name="snapshot.evidence_id")
    if evidence_id in seen_evidence_ids:
        raise RuntimeContractError("predicate snapshot evidence_id was replayed")
    observed_at_s = _number(
        snapshot.observed_at_s,
        name="snapshot.observed_at_s",
        minimum=0.0,
        include_minimum=True,
    )
    if previous_time_s is not None and observed_at_s <= previous_time_s:
        raise RuntimeContractError("predicate snapshot timestamp must strictly increase")
    if (
        not isinstance(snapshot.predicates, tuple)
        or any(not isinstance(value, str) or not value for value in snapshot.predicates)
        or len(snapshot.predicates) != len(set(snapshot.predicates))
    ):
        raise RuntimeContractError("snapshot predicates must be unique non-empty strings")
    if set(snapshot.predicates) & _RUNTIME_BUDGET_PREDICATES:
        raise RuntimeContractError("budget predicates are owned by the runtime, not the provider")
    receipt = _mapping(snapshot.evidence_receipt, name="snapshot.evidence_receipt")
    _json_safe(receipt, name="snapshot.evidence_receipt")
    normalized = PredicateSnapshot(
        sequence=step_index,
        state=state,
        evidence_id=evidence_id,
        observed_at_s=observed_at_s,
        predicates=tuple(snapshot.predicates),
        evidence_receipt=dict(receipt),
    )
    builder_identity = _validate_builder_receipt(
        receipt,
        snapshot=normalized,
        state=state,
        task_id=task_id,
        predicate_manifest=predicate_manifest,
        predicate_manifest_sha256=predicate_manifest_sha256,
        required_producers_by_state=required_producers_by_state,
        expected_builder_identity=expected_builder_identity,
        expected_contact_profile_id=expected_contact_profile_id,
        expected_insertion_profile_id=expected_insertion_profile_id,
    )
    return normalized, builder_identity


def _validate_qualified_predicates(
    predicates: set[str],
    *,
    plan_context: Mapping[str, Any],
) -> None:
    allowed_sources = set(plan_context["allowed_angle_sources"])
    if (
        "front_rgb_calibrated_angle_reached" in predicates
        and "front_rgb_calibrated" not in allowed_sources
    ):
        raise RuntimeContractError("front RGB angle predicate is not qualified by the plan")
    if (
        "tactile_relative_angle_reached" in predicates
        and "tactile_relative_angle" not in allowed_sources
    ):
        raise RuntimeContractError("tactile angle predicate is not qualified by the plan")
    if "slip_detected" in predicates and not plan_context["slip_runtime_qualified"]:
        raise RuntimeContractError("slip predicate is shadow-only and cannot control transition")


def _budget_predicates(
    state: TubeState,
    budgets: Mapping[str, int],
    limits: Mapping[str, int],
) -> set[str]:
    result: set[str] = set()
    if state is TubeState.SIDE_GRASP:
        result.add(
            "grasp_close_budget_remaining"
            if budgets["grasp_close_probes"] < limits["grasp_close_probes"]
            else "grasp_close_budget_exhausted"
        )
    if budgets["pivot_microsteps"] < limits["pivot_microsteps"]:
        result.add("pivot_budget_remaining")
    if budgets["jam_recoveries"] < limits["jam_recoveries"]:
        result.add("jam_recovery_budget_remaining")
    return result


def _action_budget_key(action: str) -> str | None:
    return {
        "bounded_close_probe": "grasp_close_probes",
        "one_micro_loosen": "pivot_microsteps",
        "immediate_bounded_reclamp": "reclamp_steps",
        "reclamp_then_hold": "reclamp_steps",
        "stop_push_and_retract": "jam_recoveries",
        "one_axial_insert_step": "insertion_microsteps",
    }.get(action)


def _hold_decision(state: TubeState, *, event: str, reason: str) -> TransitionDecision:
    return TransitionDecision(
        state=state,
        next_state=TubeState.HOLD,
        event=event,
        action="stop_and_hold",
        reason=reason,
        recovery=True,
    )


def run_test_tube_runtime(
    *,
    compiled_plan: Mapping[str, Any],
    live_geometry: Mapping[str, Any],
    predicate_provider: PredicateProvider,
    action_executor: SemanticActionExecutor,
    output_dir: Path,
    max_steps: int,
    max_grasp_close_probes: int,
) -> dict[str, Any]:
    """Run one deterministic semantic state-machine replay and persist receipts."""

    max_steps = _integer(max_steps, name="max_steps", minimum=1, maximum=10_000)
    max_grasp_close_probes = _integer(
        max_grasp_close_probes,
        name="max_grasp_close_probes",
        minimum=1,
        maximum=100,
    )
    plan_payload = dict(_mapping(compiled_plan, name="compiled_plan"))
    geometry_payload = dict(_mapping(live_geometry, name="live_geometry"))
    blockers, normalized_geometry, plan_context = runtime_start_blockers(
        plan_payload, geometry_payload
    )
    depth = plan_context["planned_insertion_depth_m"]
    insertion_limit = (
        0
        if depth is None
        else max(1, math.ceil(depth / plan_context["insertion_step_m"]))
    )
    budget_limits = {
        "grasp_close_probes": max_grasp_close_probes,
        "pivot_microsteps": plan_context["pivot_max_microsteps"],
        "reclamp_steps": plan_context["maximum_reclamp_steps"],
        "jam_recoveries": plan_context["maximum_jam_recoveries"],
        "insertion_microsteps": insertion_limit,
        "runtime_steps": max_steps,
    }
    budgets = {name: 0 for name in budget_limits}
    output_dir = output_dir.resolve()
    journal = _Journal(
        output_dir,
        task_id=plan_context["task_id"],
        plan_digest=_digest(plan_payload),
        geometry_digest=None if normalized_geometry is None else _digest(normalized_geometry),
        budget_limits=budget_limits,
    )
    if blockers:
        return journal.finish(
            status="blocked_preflight",
            final_state=TubeState.HOLD,
            budgets=budgets,
            blockers=blockers,
        )

    # Live predicate production is deliberately a concrete boundary rather
    # than an arbitrary duck-typed provider.  JSON replay remains available
    # for deterministic offline verification and crosses the same receipt
    # checks below.
    from .test_tube_predicates import TestTubePredicateProvider  # noqa: PLC0415

    if not isinstance(predicate_provider, (TestTubePredicateProvider, JsonReplay)):
        return journal.finish(
            status="blocked_preflight",
            final_state=TubeState.HOLD,
            budgets=budgets,
            blockers=[
                "predicate_provider_must_be_TestTubePredicateProvider_or_JsonReplay"
            ],
        )
    if isinstance(predicate_provider, JsonReplay) and action_executor is not predicate_provider:
        return journal.finish(
            status="blocked_preflight",
            final_state=TubeState.HOLD,
            budgets=budgets,
            blockers=["JsonReplay_cannot_drive_an_external_action_executor"],
        )

    state = TubeState.OBSERVE
    previous_time_s: float | None = None
    seen_evidence_ids: set[str] = set()
    builder_identity: tuple[str, str] | None = None
    assert normalized_geometry is not None
    for step_index in range(max_steps):
        budgets["runtime_steps"] = step_index + 1
        try:
            raw_snapshot = predicate_provider.observe(
                state=state,
                step_index=step_index,
                budgets=dict(budgets),
            )
            snapshot, observed_builder_identity = _normalize_snapshot(
                raw_snapshot,
                state=state,
                step_index=step_index,
                previous_time_s=previous_time_s,
                seen_evidence_ids=seen_evidence_ids,
                task_id=plan_context["task_id"],
                predicate_manifest=plan_context["predicate_producer_manifest"],
                predicate_manifest_sha256=plan_context[
                    "predicate_manifest_sha256"
                ],
                required_producers_by_state=plan_context[
                    "required_predicate_producers_by_state"
                ],
                expected_builder_identity=builder_identity,
                expected_contact_profile_id=normalized_geometry["contact_profile_id"],
                expected_insertion_profile_id=normalized_geometry[
                    "insertion_jam_profile_id"
                ],
            )
            if builder_identity is None:
                builder_identity = observed_builder_identity
            predicates = set(snapshot.predicates)
            _validate_qualified_predicates(predicates, plan_context=plan_context)
            predicates.update(_budget_predicates(state, budgets, budget_limits))
            decision = decide_test_tube_transition(
                state,
                predicates,
                allowed_angle_sources=plan_context["allowed_angle_sources"],
            )
            budget_key = _action_budget_key(decision.action)
            if budget_key is not None and budgets[budget_key] >= budget_limits[budget_key]:
                decision = _hold_decision(
                    state,
                    event="RUNTIME_BUDGET_EXHAUSTED",
                    reason=f"{budget_key} budget was exhausted before semantic action",
                )
        except Exception as error:  # provider/validation boundary is also fail closed
            decision = _hold_decision(
                state,
                event="EVIDENCE_PROVIDER_FAILURE",
                reason=str(error),
            )
            receipt = {
                "schema": TEST_TUBE_RUNTIME_STEP_SCHEMA,
                "task_id": plan_context["task_id"],
                "step_index": step_index,
                "state": state.value,
                "evidence": {
                    "provider_error": str(error),
                    "exception_type": type(error).__name__,
                },
                "decision": decision.to_dict(),
                "action_result": {
                    "success": False,
                    "receipt_id": f"not-dispatched-{step_index}",
                    "receipt": {"skipped": True},
                    "error": "evidence was rejected before semantic dispatch",
                },
                "applied_next_state": TubeState.HOLD.value,
                "budget_usage_after_step": dict(budgets),
            }
            journal.record_without_dispatch(receipt)
            state = TubeState.HOLD
            return journal.finish(status="hold", final_state=state, budgets=budgets)

        evidence_payload = {
            "sequence": snapshot.sequence,
            "evidence_id": snapshot.evidence_id,
            "observed_at_s": snapshot.observed_at_s,
            "provider_predicates": list(snapshot.predicates),
            "runtime_predicates": sorted(predicates - set(snapshot.predicates)),
            "receipt": dict(snapshot.evidence_receipt),
        }
        dispatch_intent = {
            "schema": TEST_TUBE_RUNTIME_STEP_SCHEMA,
            "task_id": plan_context["task_id"],
            "step_index": step_index,
            "state": state.value,
            "evidence": evidence_payload,
            "decision": decision.to_dict(),
            "action_dispatch_intent": {
                "action": decision.action,
                "intended_next_state": decision.next_state.value,
                "semantic_executor_interface": True,
            },
            "action_result": None,
            "applied_next_state": None,
            "budget_usage_before_dispatch": dict(budgets),
        }
        # The intent must reach stable storage before the injected executor is
        # entered.  An abrupt process failure therefore leaves an auditable
        # pending intent rather than an unrecorded action boundary.
        journal.begin_dispatch(dispatch_intent)
        try:
            action_result = action_executor.execute(
                decision=decision,
                evidence=snapshot,
                step_index=step_index,
            )
            if not isinstance(action_result, SemanticActionResult):
                raise RuntimeContractError(
                    "semantic action executor must return SemanticActionResult"
                )
            _string(action_result.receipt_id, name="action_result.receipt_id")
            _json_safe(action_result.receipt, name="action_result.receipt")
            if action_result.success and action_result.error is not None:
                raise RuntimeContractError("successful action result cannot contain an error")
            if not action_result.success and not action_result.error:
                raise RuntimeContractError("failed action result must contain an error")
        except Exception as error:  # executor boundary is converted to a durable HOLD
            action_result = SemanticActionResult(
                success=False,
                receipt_id=f"executor-error-{step_index}",
                receipt={"exception_type": type(error).__name__},
                error=str(error),
            )

        applied_next_state = decision.next_state if action_result.success else TubeState.HOLD
        if action_result.success and budget_key is not None:
            budgets[budget_key] += 1
        receipt = {
            "schema": TEST_TUBE_RUNTIME_STEP_SCHEMA,
            "task_id": plan_context["task_id"],
            "step_index": step_index,
            "state": state.value,
            "evidence": evidence_payload,
            "decision": decision.to_dict(),
            "action_dispatch_intent": dispatch_intent["action_dispatch_intent"],
            "action_result": {
                "success": action_result.success,
                "receipt_id": action_result.receipt_id,
                "receipt": dict(action_result.receipt),
                "error": action_result.error,
            },
            "applied_next_state": applied_next_state.value,
            "budget_usage_after_step": dict(budgets),
        }
        journal.complete_dispatch(receipt)
        previous_time_s = snapshot.observed_at_s
        seen_evidence_ids.add(snapshot.evidence_id)
        state = applied_next_state

        if state in {TubeState.COMPLETE, TubeState.HOLD}:
            status = "complete" if state is TubeState.COMPLETE else "hold"
            return journal.finish(status=status, final_state=state, budgets=budgets)

    receipt = {
        "schema": TEST_TUBE_RUNTIME_STEP_SCHEMA,
        "task_id": plan_context["task_id"],
        "step_index": max_steps,
        "state": state.value,
        "evidence": {"runtime_step_budget_exhausted": True},
        "decision": _hold_decision(
            state,
            event="RUNTIME_STEP_BUDGET_EXHAUSTED",
            reason="global runtime step budget was exhausted",
        ).to_dict(),
        "action_result": {
            "success": False,
            "receipt_id": f"not-dispatched-{max_steps}",
            "receipt": {"skipped": True},
            "error": "global runtime step budget exhausted",
        },
        "applied_next_state": TubeState.HOLD.value,
        "budget_usage_after_step": dict(budgets),
    }
    journal.record_without_dispatch(receipt)
    return journal.finish(status="hold", final_state=TubeState.HOLD, budgets=budgets)


class JsonReplay:
    """Predicate provider and semantic executor backed by one deterministic JSON trace."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        replay = _mapping(payload, name="replay")
        if replay.get("schema") != TEST_TUBE_RUNTIME_REPLAY_SCHEMA:
            raise RuntimeContractError(
                f"replay.schema must be {TEST_TUBE_RUNTIME_REPLAY_SCHEMA!r}"
            )
        self.task_id = _string(replay.get("task_id"), name="replay.task_id")
        self.max_steps = _integer(
            replay.get("max_steps"), name="replay.max_steps", minimum=1, maximum=10_000
        )
        self.max_grasp_close_probes = _integer(
            replay.get("max_grasp_close_probes"),
            name="replay.max_grasp_close_probes",
            minimum=1,
            maximum=100,
        )
        self.live_geometry = dict(
            _mapping(replay.get("live_geometry"), name="replay.live_geometry")
        )
        steps = replay.get("steps")
        if not isinstance(steps, list) or not steps:
            raise RuntimeContractError("replay.steps must be a non-empty array")
        self.steps = [
            dict(_mapping(step, name=f"replay.steps[{index}]"))
            for index, step in enumerate(steps)
        ]
        self._provider_index = 0
        self._executor_index = 0

    def observe(
        self,
        *,
        state: TubeState,
        step_index: int,
        budgets: Mapping[str, int],
    ) -> PredicateSnapshot:
        del budgets
        if self._provider_index >= len(self.steps):
            raise RuntimeContractError("replay exhausted before a terminal state")
        raw = self.steps[self._provider_index]
        expected_sequence = _integer(
            raw.get("sequence"), name=f"replay.steps[{self._provider_index}].sequence"
        )
        if expected_sequence != step_index:
            raise RuntimeContractError("replay sequence does not match runtime step")
        expected_state = TubeState(raw.get("state"))
        if expected_state is not state:
            raise RuntimeContractError(
                f"replay expected state {expected_state.value}, runtime is {state.value}"
            )
        predicates = raw.get("predicates")
        if not isinstance(predicates, list):
            raise RuntimeContractError("replay predicates must be an array")
        snapshot = PredicateSnapshot(
            sequence=expected_sequence,
            state=expected_state,
            evidence_id=_string(raw.get("evidence_id"), name="replay evidence_id"),
            observed_at_s=_number(
                raw.get("observed_at_s"),
                name="replay observed_at_s",
                minimum=0.0,
                include_minimum=True,
            ),
            predicates=tuple(predicates),
            evidence_receipt=dict(
                _mapping(raw.get("evidence_receipt"), name="replay evidence_receipt")
            ),
        )
        self._provider_index += 1
        return snapshot

    def execute(
        self,
        *,
        decision: TransitionDecision,
        evidence: PredicateSnapshot,
        step_index: int,
    ) -> SemanticActionResult:
        del evidence
        if self._executor_index >= len(self.steps):
            raise RuntimeContractError("replay action results exhausted")
        raw = self.steps[self._executor_index]
        if raw.get("sequence") != step_index:
            raise RuntimeContractError("replay action sequence does not match runtime step")
        expected = _mapping(raw.get("expected"), name="replay expected decision")
        actual = {
            "event": decision.event,
            "action": decision.action,
            "next_state": decision.next_state.value,
        }
        if dict(expected) != actual:
            raise RuntimeContractError(
                f"replay decision mismatch: expected {dict(expected)!r}, got {actual!r}"
            )
        raw_result = _mapping(raw.get("action_result"), name="replay action_result")
        result = SemanticActionResult(
            success=_boolean(raw_result.get("success"), name="action_result.success"),
            receipt_id=_string(
                raw_result.get("receipt_id"), name="action_result.receipt_id"
            ),
            receipt=dict(_mapping(raw_result.get("receipt"), name="action_result.receipt")),
            error=raw_result.get("error"),
        )
        if result.error is not None and not isinstance(result.error, str):
            raise RuntimeContractError("action_result.error must be null or a string")
        self._executor_index += 1
        return result

    def assert_fully_consumed(self) -> None:
        if self._provider_index != len(self.steps) or self._executor_index != len(self.steps):
            raise RuntimeContractError(
                "runtime reached a terminal state before consuming the complete replay"
            )


__all__ = [
    "JsonReplay",
    "PredicateProvider",
    "PredicateSnapshot",
    "RuntimeContractError",
    "SemanticActionExecutor",
    "SemanticActionResult",
    "TEST_TUBE_RUNTIME_GEOMETRY_SCHEMA",
    "TEST_TUBE_PREDICATE_RECEIPT_SCHEMA",
    "TEST_TUBE_RUNTIME_REPLAY_SCHEMA",
    "TEST_TUBE_RUNTIME_REPORT_SCHEMA",
    "TEST_TUBE_RUNTIME_STEP_SCHEMA",
    "run_test_tube_runtime",
    "runtime_start_blockers",
    "validate_runtime_geometry",
]
