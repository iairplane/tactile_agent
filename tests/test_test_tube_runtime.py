from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from demo2gripper.test_tube_predicates import (
    PredicateBuilderConfig,
    PredicateReceiptBundle,
    ReceiptHeader,
    SceneReceipt,
    predicate_producer_manifest,
    required_predicate_producers_by_state,
)
from demo2gripper.test_tube_predicates import (
    TestTubePredicateBuilder as TubePredicateBuilder,
)
from demo2gripper.test_tube_predicates import (
    TestTubePredicateProvider as TubePredicateProvider,
)
from demo2gripper.test_tube_preflight import TEST_TUBE_PLAN_SCHEMA, TubeState
from demo2gripper.test_tube_runtime import (
    TEST_TUBE_PREDICATE_RECEIPT_SCHEMA,
    TEST_TUBE_RUNTIME_GEOMETRY_SCHEMA,
    TEST_TUBE_RUNTIME_REPLAY_SCHEMA,
    TEST_TUBE_RUNTIME_REPORT_SCHEMA,
    JsonReplay,
    PredicateSnapshot,
    SemanticActionResult,
    run_test_tube_runtime,
)
from scripts import replay_test_tube_runtime as cli


def _plan() -> dict:
    return {
        "schema": TEST_TUBE_PLAN_SCHEMA,
        "task_id": "tube-runtime-test",
        "execution_authorized": True,
        "hardware_execution_enabled": True,
        "preflight": {
            "offline_task_contract_ready": True,
            "metric_execution_geometry_ready": True,
            "moveit_plan_only_ready": True,
            "runtime_transition_gates_ready": True,
            "hardware_execution_ready": True,
            "blocking_conditions": [],
        },
        "execution_readiness": {
            "exact_tube_tcp_instantiated": True,
            "gripper_width_and_close_budget_grounded": True,
            "collision_scene_validated": True,
            "motion_bounds_validated": True,
            "offline_runtime_orchestrator_available": True,
            "offline_predicate_builder_available": True,
            "rgb_tracker_qualification_tool_available": True,
            "live_hardware_adapter_available": True,
            "live_feedback_predicate_producers_available": True,
            "support_after_release_verifier_available": True,
            "robot_command_state_path_healthy": True,
            "long_lived_front_rgb_source_available": True,
            "front_rgb_stream_healthy": True,
        },
        "motion_limits": {
            "insertion_step_m": 0.0015,
            "insertion_target_depth_m": 0.003,
        },
        "recovery_limits": {
            "pivot_max_microsteps": 8,
            "maximum_reclamp_steps": 4,
            "maximum_jam_recoveries": 2,
            "on_budget_exhausted": "HOLD",
        },
        "gates": {
            "pivot_contact": {"mode": "runtime_qualified"},
            "slip": {"may_trigger_transition": False},
            "pivot_target": {
                "allowed_target_sources": ["front_rgb_calibrated"],
                "front_rgb_runtime_gate_qualified": True,
                "tactile_angle_calibration_id": None,
            },
            "insertion_jam": {"may_trigger_transition": True},
        },
        "front_rgb_tracker_validation": {"runtime_gate_eligible": True},
        "front_rgb_pivot_reference": {"runtime_gate_eligible": True},
        "predicate_producer_manifest": predicate_producer_manifest(),
    }


def _geometry() -> dict:
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "schema": TEST_TUBE_RUNTIME_GEOMETRY_SCHEMA,
        "task_id": "tube-runtime-test",
        "status": "runtime_qualified",
        "freshness_verified": True,
        "observation_id": "geometry-001",
        "metric_localization_receipt_id": "localization-001",
        "collision_scene_id": "scene-001",
        "contact_profile_id": "contact-001",
        "front_rgb_tracker_session_id": "tracker-001",
        "insertion_jam_profile_id": "jam-001",
        "transform_convention": "T_parent_child; metres; right-handed",
        "T_base_tube": identity,
        "T_base_socket": identity,
        "T_tube_tcp_side_grasp": identity,
        "insertion_axis_base": [0.0, 0.0, 1.0],
        "insertion_target_depth_m": 0.003,
    }


def _step(
    sequence: int,
    state: str,
    predicates: list[str],
    event: str,
    action: str,
    next_state: str,
    *,
    success: bool = True,
) -> dict:
    predicates = sorted(predicates)
    required = required_predicate_producers_by_state()[state]
    producer_receipts = {}
    for producer in required:
        if producer == "tactile":
            qualification_id = "contact-001"
        elif producer == "insertion":
            qualification_id = "jam-001"
        elif producer == "rgb_axis":
            qualification_id = "rgb-qualification-001"
        else:
            qualification_id = None
        producer_receipts[producer] = {
            "receipt_id": f"{producer}-{sequence:03d}",
            "producer_id": f"test-{producer}",
            "observed_at_s": float(sequence),
            "qualification_id": qualification_id,
        }
    manifest = predicate_producer_manifest()
    manifest_digest = hashlib.sha256(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    evidence_receipt = {
        "schema": TEST_TUBE_PREDICATE_RECEIPT_SCHEMA,
        "predicate_manifest_sha256": manifest_digest,
        "task_id": "tube-runtime-test",
        "session_id": "runtime-session-001",
        "provenance_id": "runtime-provenance-001",
        "state": state,
        "evaluated_at_s": float(sequence),
        "producer_receipts": producer_receipts,
        "required_producers": required,
        "promoted_predicates": predicates,
        "stale_receipt_ids": [],
        "rejection_reasons": [],
        "shadow_observations": {
            "slip_shadow_candidate": False,
            "jam_shadow_candidate": False,
        },
    }
    evidence_id = hashlib.sha256(
        json.dumps(
            evidence_receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return {
        "sequence": sequence,
        "state": state,
        "evidence_id": evidence_id,
        "observed_at_s": float(sequence),
        "predicates": predicates,
        "evidence_receipt": evidence_receipt,
        "expected": {"event": event, "action": action, "next_state": next_state},
        "action_result": {
            "success": success,
            "receipt_id": f"action-{sequence:03d}",
            "receipt": {"semantic_only": True},
            "error": None if success else "injected semantic action failure",
        },
    }


def _nominal_steps() -> list[dict]:
    return [
        _step(
            0,
            "OBSERVE",
            ["tube_pose_valid", "socket_frame_valid"],
            "LIVE_FRAMES_READY",
            "request_agentic_approach_plan",
            "APPROACH_TUBE",
        ),
        _step(
            1,
            "APPROACH_TUBE",
            ["approach_pose_reached"],
            "APPROACH_COMPLETE",
            "ground_side_grasp_relation",
            "SIDE_GRASP",
        ),
        _step(
            2,
            "SIDE_GRASP",
            ["grasp_established"],
            "GRASP_ESTABLISHED",
            "request_agentic_extraction",
            "EXTRACT",
        ),
        _step(
            3,
            "EXTRACT",
            ["tube_clear_of_source", "contact_present"],
            "EXTRACTION_COMPLETE",
            "servo_held_tube_to_horizontal",
            "LEVEL_HELD_TUBE",
        ),
        _step(
            4,
            "LEVEL_HELD_TUBE",
            ["tube_horizontal", "level_target_reached", "contact_present"],
            "HELD_TUBE_LEVEL_STABLE",
            "one_micro_loosen",
            "PIVOT_MICRO_LOOSEN",
        ),
        _step(
            5,
            "PIVOT_MICRO_LOOSEN",
            ["one_microstep_complete"],
            "MICROSTEP_COMPLETE",
            "collect_fresh_angle_and_contact_window",
            "PIVOT_EVALUATE",
        ),
        _step(
            6,
            "PIVOT_EVALUATE",
            ["contact_present", "front_rgb_calibrated_angle_reached"],
            "PIVOT_TARGET_REACHED",
            "reclamp_at_target",
            "RECLAMP",
        ),
        _step(
            7,
            "RECLAMP",
            ["contact_present", "reclamp_command_settled", "pivot_target_confirmed"],
            "RECLAMP_STABLE",
            "request_agentic_socket_approach",
            "APPROACH_SOCKET",
        ),
        _step(
            8,
            "APPROACH_SOCKET",
            ["socket_standoff_reached"],
            "SOCKET_STANDOFF_REACHED",
            "ground_tube_socket_axis_relation",
            "ALIGN_SOCKET",
        ),
        _step(
            9,
            "ALIGN_SOCKET",
            ["tube_socket_axes_aligned", "tube_tip_at_socket_entry"],
            "INSERTION_ALIGNED",
            "one_axial_insert_step",
            "INSERT_MICROSTEP",
        ),
        _step(
            10,
            "INSERT_MICROSTEP",
            ["one_axial_step_complete"],
            "AXIAL_STEP_COMPLETE",
            "collect_fresh_progress_and_resistance_window",
            "INSERT_EVALUATE",
        ),
        _step(
            11,
            "INSERT_EVALUATE",
            ["insertion_depth_reached", "no_jam"],
            "INSERTION_COMPLETE",
            "verify_support_then_open",
            "RELEASE",
        ),
        _step(
            12,
            "RELEASE",
            ["gripper_open", "contact_released", "tube_stays_supported"],
            "RELEASE_VERIFIED",
            "request_agentic_retreat",
            "RETREAT",
        ),
        _step(
            13,
            "RETREAT",
            ["retreat_complete"],
            "TASK_COMPLETE",
            "none",
            "COMPLETE",
        ),
    ]


def _replay(steps: list[dict]) -> dict:
    return {
        "schema": TEST_TUBE_RUNTIME_REPLAY_SCHEMA,
        "task_id": "tube-runtime-test",
        "max_steps": 32,
        "max_grasp_close_probes": 4,
        "live_geometry": _geometry(),
        "steps": steps,
    }


def _refresh_step_evidence(step: dict) -> None:
    promoted = sorted(step["predicates"])
    step["predicates"] = promoted
    step["evidence_receipt"]["promoted_predicates"] = promoted
    step["evidence_id"] = hashlib.sha256(
        json.dumps(
            step["evidence_receipt"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _builder_config(*, provenance_id: str = "runtime-provenance-001") -> PredicateBuilderConfig:
    return PredicateBuilderConfig(
        task_id="tube-runtime-test",
        session_id="runtime-session-001",
        provenance_id=provenance_id,
        contact_profile_id="contact-001",
        rgb_qualification_id="rgb-qualification-001",
        insertion_profile_id="jam-001",
        retained_width_min_m=0.005,
        retained_width_max_m=0.04,
        rgb_runtime_qualified=True,
        jam_runtime_qualified=True,
    )


def _header(
    producer: str,
    timestamp: float,
    *,
    provenance_id: str = "runtime-provenance-001",
) -> ReceiptHeader:
    return ReceiptHeader(
        task_id="tube-runtime-test",
        session_id="runtime-session-001",
        provenance_id=provenance_id,
        receipt_id=f"{producer}-{timestamp}",
        producer_id=f"test-{producer}",
        observed_at_s=timestamp,
    )


def test_nominal_replay_reaches_complete_and_persists_atomic_step_receipts(
    tmp_path: Path,
) -> None:
    replay = JsonReplay(_replay(_nominal_steps()))
    report = run_test_tube_runtime(
        compiled_plan=_plan(),
        live_geometry=replay.live_geometry,
        predicate_provider=replay,
        action_executor=replay,
        output_dir=tmp_path / "run",
        max_steps=replay.max_steps,
        max_grasp_close_probes=replay.max_grasp_close_probes,
    )
    replay.assert_fully_consumed()

    assert report["schema"] == TEST_TUBE_RUNTIME_REPORT_SCHEMA
    assert report["status"] == "complete"
    assert report["final_state"] == "COMPLETE"
    assert report["hardware_commands_dispatched_by_core"] is False
    assert report["step_count"] == 14
    assert len(list((tmp_path / "run" / "steps").glob("*.json"))) == 14
    pivot_receipt = json.loads(
        (tmp_path / "run" / "steps" / "0004.json").read_text(encoding="utf-8")
    )
    assert pivot_receipt["decision"]["action"] == "one_micro_loosen"
    assert pivot_receipt["budget_usage_after_step"]["pivot_microsteps"] == 1


def test_unready_compiled_plan_holds_before_consuming_evidence(tmp_path: Path) -> None:
    plan = _plan()
    plan["preflight"]["metric_execution_geometry_ready"] = False
    plan["preflight"]["hardware_execution_ready"] = False
    replay = JsonReplay(_replay(_nominal_steps()))

    report = run_test_tube_runtime(
        compiled_plan=plan,
        live_geometry=replay.live_geometry,
        predicate_provider=replay,
        action_executor=replay,
        output_dir=tmp_path / "blocked",
        max_steps=replay.max_steps,
        max_grasp_close_probes=replay.max_grasp_close_probes,
    )

    assert report["status"] == "blocked_preflight"
    assert report["final_state"] == "HOLD"
    assert report["step_count"] == 0
    assert "compiled_preflight_metric_execution_geometry_ready_false" in report[
        "blocking_conditions"
    ]


def test_top_level_authorization_boundary_cannot_be_bypassed_by_ready_preflight(
    tmp_path: Path,
) -> None:
    plan = _plan()
    plan["execution_authorized"] = False
    plan["hardware_execution_enabled"] = False
    replay = JsonReplay(_replay(_nominal_steps()))

    report = run_test_tube_runtime(
        compiled_plan=plan,
        live_geometry=replay.live_geometry,
        predicate_provider=replay,
        action_executor=replay,
        output_dir=tmp_path / "authorization-blocked",
        max_steps=replay.max_steps,
        max_grasp_close_probes=replay.max_grasp_close_probes,
    )

    assert report["status"] == "blocked_preflight"
    assert "compiled_execution_authorized_false" in report["blocking_conditions"]
    assert "compiled_hardware_execution_enabled_false" in report["blocking_conditions"]


def test_missing_live_metric_transform_holds_before_provider(tmp_path: Path) -> None:
    geometry = _geometry()
    geometry["T_base_socket"] = None
    replay_payload = _replay(_nominal_steps())
    replay_payload["live_geometry"] = geometry
    replay = JsonReplay(replay_payload)

    report = run_test_tube_runtime(
        compiled_plan=_plan(),
        live_geometry=replay.live_geometry,
        predicate_provider=replay,
        action_executor=replay,
        output_dir=tmp_path / "missing-geometry",
        max_steps=replay.max_steps,
        max_grasp_close_probes=replay.max_grasp_close_probes,
    )

    assert report["status"] == "blocked_preflight"
    assert any("T_base_socket" in item for item in report["blocking_conditions"])


def test_action_failure_is_persisted_and_forces_terminal_hold(tmp_path: Path) -> None:
    steps = _nominal_steps()[:1]
    steps[0]["action_result"] = {
        "success": False,
        "receipt_id": "failed-action",
        "receipt": {"semantic_only": True},
        "error": "injected failure",
    }
    replay = JsonReplay(_replay(steps))

    report = run_test_tube_runtime(
        compiled_plan=_plan(),
        live_geometry=replay.live_geometry,
        predicate_provider=replay,
        action_executor=replay,
        output_dir=tmp_path / "failure",
        max_steps=replay.max_steps,
        max_grasp_close_probes=replay.max_grasp_close_probes,
    )
    replay.assert_fully_consumed()

    assert report["status"] == "hold"
    receipt = json.loads(
        (tmp_path / "failure" / "steps" / "0000.json").read_text(encoding="utf-8")
    )
    assert receipt["decision"]["next_state"] == "APPROACH_TUBE"
    assert receipt["applied_next_state"] == "HOLD"
    assert receipt["action_result"]["error"] == "injected failure"


def test_dispatch_intent_is_durable_before_executor_is_called(tmp_path: Path) -> None:
    output_dir = tmp_path / "two-phase"
    steps = _nominal_steps()[:1]
    steps[0]["action_result"] = {
        "success": False,
        "receipt_id": "stop-after-intent",
        "receipt": {"checked_intent": True},
        "error": "stop after checking persisted intent",
    }

    class IntentCheckingReplay(JsonReplay):
        def execute(self, **kwargs):
            intent = json.loads(
                (output_dir / "steps" / "0000.json").read_text(encoding="utf-8")
            )
            running = json.loads(
                (output_dir / "runtime_report.json").read_text(encoding="utf-8")
            )
            assert intent["receipt_phase"] == "dispatch_intent_persisted"
            assert intent["action_result"] is None
            assert intent["action_dispatch_intent"]["action"] == (
                "request_agentic_approach_plan"
            )
            assert running["pending_dispatch_intents"] == [0]
            return super().execute(**kwargs)

    replay = IntentCheckingReplay(_replay(steps))
    report = run_test_tube_runtime(
        compiled_plan=_plan(),
        live_geometry=replay.live_geometry,
        predicate_provider=replay,
        action_executor=replay,
        output_dir=output_dir,
        max_steps=replay.max_steps,
        max_grasp_close_probes=replay.max_grasp_close_probes,
    )

    assert report["status"] == "hold"
    assert report["pending_dispatch_intents"] == []
    final = json.loads((output_dir / "steps" / "0000.json").read_text(encoding="utf-8"))
    assert final["receipt_phase"] == "action_result_persisted"
    assert final["action_result"]["receipt"]["checked_intent"] is True


def test_shadow_slip_cannot_control_runtime_transition(tmp_path: Path) -> None:
    step = _nominal_steps()[0]
    step["predicates"].append("slip_detected")
    _refresh_step_evidence(step)
    replay = JsonReplay(_replay([step]))

    report = run_test_tube_runtime(
        compiled_plan=_plan(),
        live_geometry=replay.live_geometry,
        predicate_provider=replay,
        action_executor=replay,
        output_dir=tmp_path / "shadow-slip",
        max_steps=replay.max_steps,
        max_grasp_close_probes=replay.max_grasp_close_probes,
    )

    assert report["status"] == "hold"
    receipt = json.loads(
        (tmp_path / "shadow-slip" / "steps" / "0000.json").read_text(encoding="utf-8")
    )
    assert receipt["decision"]["event"] == "EVIDENCE_PROVIDER_FAILURE"
    assert "not allowed in state 'OBSERVE'" in receipt["evidence"][
        "provider_error"
    ]


def test_runtime_owns_budget_predicates_and_rejects_provider_claim(tmp_path: Path) -> None:
    step = _nominal_steps()[0]
    step["predicates"].append("pivot_budget_remaining")
    replay = JsonReplay(_replay([step]))

    report = run_test_tube_runtime(
        compiled_plan=_plan(),
        live_geometry=replay.live_geometry,
        predicate_provider=replay,
        action_executor=replay,
        output_dir=tmp_path / "forged-budget",
        max_steps=replay.max_steps,
        max_grasp_close_probes=replay.max_grasp_close_probes,
    )

    assert report["status"] == "hold"
    receipt = json.loads(
        (tmp_path / "forged-budget" / "steps" / "0000.json").read_text(encoding="utf-8")
    )
    assert "owned by the runtime" in receipt["evidence"]["provider_error"]


def test_actual_predicate_builder_snapshot_crosses_runtime_boundary(
    tmp_path: Path,
) -> None:
    class Source:
        def collect(self, **kwargs) -> PredicateReceiptBundle:
            assert kwargs["state"] is TubeState.OBSERVE
            assert kwargs["step_index"] == 0
            return PredicateReceiptBundle(
                evaluated_at_s=1.05,
                scene=SceneReceipt(
                    _header("scene", 1.0),
                    tube_pose_valid=True,
                    socket_frame_valid=True,
                ),
            )

    class FailingAfterReceiptExecutor:
        calls = 0

        def execute(self, **kwargs) -> SemanticActionResult:
            self.calls += 1
            evidence = kwargs["evidence"]
            assert evidence.evidence_id == hashlib.sha256(
                json.dumps(
                    evidence.evidence_receipt,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            return SemanticActionResult(
                success=False,
                receipt_id="intentional-stop",
                receipt={"builder_snapshot_accepted": True},
                error="stop after builder/runtime integration check",
            )

    provider = TubePredicateProvider(
        TubePredicateBuilder(_builder_config()), Source()
    )
    executor = FailingAfterReceiptExecutor()
    report = run_test_tube_runtime(
        compiled_plan=_plan(),
        live_geometry=_geometry(),
        predicate_provider=provider,
        action_executor=executor,
        output_dir=tmp_path / "builder-e2e",
        max_steps=4,
        max_grasp_close_probes=2,
    )

    assert report["status"] == "hold"
    assert executor.calls == 1
    persisted = json.loads(
        (tmp_path / "builder-e2e" / "steps" / "0000.json").read_text()
    )
    assert persisted["evidence"]["receipt"]["schema"] == (
        TEST_TUBE_PREDICATE_RECEIPT_SCHEMA
    )
    assert persisted["action_result"]["receipt"]["builder_snapshot_accepted"] is True


def test_tampered_builder_digest_is_rejected_before_action_dispatch(
    tmp_path: Path,
) -> None:
    steps = _nominal_steps()[:1]
    steps[0]["evidence_id"] = "forged-builder-digest"
    replay = JsonReplay(_replay(steps))

    report = run_test_tube_runtime(
        compiled_plan=_plan(),
        live_geometry=_geometry(),
        predicate_provider=replay,
        action_executor=replay,
        output_dir=tmp_path / "tampered-builder",
        max_steps=4,
        max_grasp_close_probes=2,
    )

    assert report["status"] == "hold"
    persisted = json.loads(
        (tmp_path / "tampered-builder" / "steps" / "0000.json").read_text()
    )
    assert "does not match the predicate-builder receipt digest" in persisted[
        "evidence"
    ]["provider_error"]


def test_builder_provenance_cannot_change_between_runtime_steps(tmp_path: Path) -> None:
    steps = _nominal_steps()[:2]
    steps[1]["evidence_receipt"]["provenance_id"] = "runtime-provenance-002"
    _refresh_step_evidence(steps[1])
    replay = JsonReplay(_replay(steps))
    report = run_test_tube_runtime(
        compiled_plan=_plan(),
        live_geometry=_geometry(),
        predicate_provider=replay,
        action_executor=replay,
        output_dir=tmp_path / "provenance-change",
        max_steps=4,
        max_grasp_close_probes=2,
    )

    assert report["status"] == "hold"
    persisted = json.loads(
        (tmp_path / "provenance-change" / "steps" / "0001.json").read_text()
    )
    assert "session/provenance changed" in persisted["evidence"]["provider_error"]


def test_arbitrary_protocol_provider_is_blocked_before_observation(tmp_path: Path) -> None:
    class ArbitraryProvider:
        called = False

        def observe(self, **kwargs) -> PredicateSnapshot:
            self.called = True
            raise AssertionError("arbitrary provider must not be observed")

    provider = ArbitraryProvider()
    replay = JsonReplay(_replay(_nominal_steps()))
    report = run_test_tube_runtime(
        compiled_plan=_plan(),
        live_geometry=_geometry(),
        predicate_provider=provider,
        action_executor=replay,
        output_dir=tmp_path / "arbitrary-provider",
        max_steps=4,
        max_grasp_close_probes=2,
    )

    assert report["status"] == "blocked_preflight"
    assert report["step_count"] == 0
    assert provider.called is False
    assert report["blocking_conditions"] == [
        "predicate_provider_must_be_TestTubePredicateProvider_or_JsonReplay"
    ]


def test_json_replay_cannot_drive_external_action_executor(tmp_path: Path) -> None:
    class ExternalExecutor:
        called = False

        def execute(self, **kwargs) -> SemanticActionResult:
            self.called = True
            raise AssertionError("replay evidence must not reach an external executor")

    replay = JsonReplay(_replay(_nominal_steps()))
    executor = ExternalExecutor()
    report = run_test_tube_runtime(
        compiled_plan=_plan(),
        live_geometry=_geometry(),
        predicate_provider=replay,
        action_executor=executor,
        output_dir=tmp_path / "replay-external-executor",
        max_steps=4,
        max_grasp_close_probes=2,
    )

    assert report["status"] == "blocked_preflight"
    assert report["step_count"] == 0
    assert executor.called is False
    assert report["blocking_conditions"] == [
        "JsonReplay_cannot_drive_an_external_action_executor"
    ]


def test_noncanonical_compiled_predicate_manifest_blocks_before_provider(
    tmp_path: Path,
) -> None:
    plan = _plan()
    del plan["predicate_producer_manifest"]["tube_pose_valid"]
    replay = JsonReplay(_replay(_nominal_steps()))

    report = run_test_tube_runtime(
        compiled_plan=plan,
        live_geometry=replay.live_geometry,
        predicate_provider=replay,
        action_executor=replay,
        output_dir=tmp_path / "manifest-blocked",
        max_steps=replay.max_steps,
        max_grasp_close_probes=replay.max_grasp_close_probes,
    )

    assert report["status"] == "blocked_preflight"
    assert report["step_count"] == 0
    assert "compiled_predicate_producer_manifest_not_canonical" in report[
        "blocking_conditions"
    ]


def test_pivot_microstep_budget_forces_reclamp_then_hold(tmp_path: Path) -> None:
    plan = _plan()
    plan["recovery_limits"]["pivot_max_microsteps"] = 1
    steps = _nominal_steps()[:6]
    steps.extend(
        [
            _step(
                6,
                "PIVOT_EVALUATE",
                ["contact_present"],
                "PIVOT_BUDGET_EXHAUSTED",
                "reclamp_then_hold",
                "RECLAMP_RECOVERY",
            ),
            _step(
                7,
                "RECLAMP_RECOVERY",
                ["contact_present", "reclamp_command_settled"],
                "PIVOT_RECOVERY_BUDGET_EXHAUSTED",
                "stop_and_hold",
                "HOLD",
            ),
        ]
    )
    replay = JsonReplay(_replay(steps))

    report = run_test_tube_runtime(
        compiled_plan=plan,
        live_geometry=replay.live_geometry,
        predicate_provider=replay,
        action_executor=replay,
        output_dir=tmp_path / "pivot-budget",
        max_steps=replay.max_steps,
        max_grasp_close_probes=replay.max_grasp_close_probes,
    )
    replay.assert_fully_consumed()

    assert report["status"] == "hold"
    assert report["budget_usage"]["pivot_microsteps"] == 1
    assert report["budget_usage"]["reclamp_steps"] == 1


def test_cli_has_only_replay_mode_and_returns_two_for_blocked_plan(tmp_path: Path) -> None:
    plan = _plan()
    plan["preflight"]["runtime_transition_gates_ready"] = False
    plan["preflight"]["hardware_execution_ready"] = False
    plan_path = tmp_path / "plan.json"
    replay_path = tmp_path / "replay.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    replay_path.write_text(json.dumps(_replay(_nominal_steps())), encoding="utf-8")

    result = cli.run(
        argparse.Namespace(
            plan=plan_path,
            replay=replay_path,
            output_dir=tmp_path / "cli-output",
            overwrite=False,
        )
    )

    assert result == 2
    report = json.loads(
        (tmp_path / "cli-output" / "runtime_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "blocked_preflight"


def test_current_style_null_depth_plan_cannot_be_promoted_by_live_geometry(
    tmp_path: Path,
) -> None:
    plan = copy.deepcopy(_plan())
    plan["motion_limits"]["insertion_target_depth_m"] = None
    plan["preflight"]["metric_execution_geometry_ready"] = False
    plan["preflight"]["moveit_plan_only_ready"] = False
    plan["preflight"]["hardware_execution_ready"] = False
    replay = JsonReplay(_replay(_nominal_steps()))

    report = run_test_tube_runtime(
        compiled_plan=plan,
        live_geometry=replay.live_geometry,
        predicate_provider=replay,
        action_executor=replay,
        output_dir=tmp_path / "null-depth",
        max_steps=replay.max_steps,
        max_grasp_close_probes=replay.max_grasp_close_probes,
    )

    assert report["status"] == "blocked_preflight"
    assert "compiled_insertion_target_depth_missing" in report["blocking_conditions"]

