from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_demo03_open_pick_place as runner
from demo2gripper.autonomous_pick import PickThresholds
from demo2gripper.cartesian_pose_sequence import CartesianPose
from scripts.run_demo03_open_pick_place import (
    STAGE_ORDER,
    Demo03Stage,
    _append_stage,
    _execute,
    _plan_only,
    _validate_relation,
    _write_placement,
    run,
)


def test_open_first_stage_order_matches_demonstrated_task() -> None:
    assert STAGE_ORDER == (
        "LID_STAGE1_PRECONTACT",
        "LID_STAGE1_CONTACT",
        "LID_STAGE1_GRIP",
        "LID_STAGE1_OPEN_61",
        "LID_STAGE1_RELEASE",
        "LID_STAGE1_RETREAT",
        "LID_STAGE1_TRANSITION_GATE",
        "LID_STAGE2_CLEARANCE_TRANSITION",
        "LID_STAGE2_PRECONTACT",
        "LID_STAGE2_CONTACT",
        "LID_STAGE2_GRIP",
        "LID_STAGE2_OPEN_100",
        "LID_STAGE2_RELEASE",
        "LID_STAGE2_RETREAT",
        "REOBSERVE_DIE",
        "PICK_AND_LIFT",
        "DERIVE_PLACEMENT",
        "TRANSPORT",
        "PLACE",
        "RELEASE_DIE",
        "RETREAT",
    )


def test_execute_requires_both_physical_acknowledgements(tmp_path: Path) -> None:
    common = {
        "config": tmp_path / "unused.json",
        "output_dir": tmp_path / "unused-output",
        "execute": True,
        "plan_only": False,
        "pick_input_dir": None,
    }
    with pytest.raises(ValueError, match="arm-only"):
        run(
            argparse.Namespace(
                **common,
                accept_arm_only_collision_model=False,
                accept_fixed_scene_placement=False,
            )
        )
    with pytest.raises(ValueError, match="fixed-scene"):
        run(
            argparse.Namespace(
                **common,
                accept_arm_only_collision_model=True,
                accept_fixed_scene_placement=False,
            )
        )


def test_stage_report_cannot_skip_a_critical_stage(tmp_path: Path) -> None:
    report_path = tmp_path / "stage_report.json"
    report = {"stages": [], "updated_at": "test"}

    with pytest.raises(ValueError, match="expected LID_STAGE1_PRECONTACT"):
        _append_stage(
            report,
            report_path,
            Demo03Stage.LID_STAGE1_CONTACT,
            status="passed",
            details={},
            started_at="test",
        )


def test_fixed_scene_relation_rejects_changed_camera_calibration(tmp_path: Path) -> None:
    calibration = tmp_path / "T_base_camera.npy"
    np.save(calibration, np.eye(4))
    relation_path = tmp_path / "relation.json"
    relation = {
        "schema": "demo2gripper.demo03_fixed_scene_placement_relation.v1",
        "status": "accepted_fixed_scene_translation_only",
        "relation_mode": "fixed_scene_translation_only",
        "base_frame": "fr3_link0",
        "final_die_anchor_base_m": [0.6, 0.0, 0.02],
        "quality": {"accepted": True},
        "provenance": {"T_base_camera_sha256": "wrong"},
    }

    with pytest.raises(ValueError, match="different T_base_camera"):
        _validate_relation(relation, path=relation_path, t_base_camera=calibration)


def test_runtime_placement_uses_fresh_grounded_pick_and_preserves_orientation(
    tmp_path: Path,
) -> None:
    relation_path = tmp_path / "relation.json"
    relation_path.write_text("{}\n", encoding="utf-8")
    grounded_path = tmp_path / "fresh-execution-candidate.json"
    T_base_grasp = np.eye(4)
    T_base_grasp[:3, 3] = [0.62, 0.04, 0.03]
    grounded = {
        "schema": "demo2gripper.grounded_live_grasp_candidate.v2",
        "live_object_anchor_base_m": [0.62, 0.04, 0.045],
        "candidate": {"T_base_grasp_tcp": T_base_grasp.tolist()},
    }
    grounded_path.write_text(json.dumps(grounded), encoding="utf-8")
    normalized = {
        "final_die_anchor_base_m": np.array([0.67, 0.00, 0.005]),
        "transport_height_m": 0.08,
        "retreat_height_m": 0.08,
        "placement_relation_path": relation_path,
    }

    result = _write_placement(
        normalized,
        grounded_pick_path=grounded_path,
        output_dir=tmp_path / "placement",
    )

    expected_placement = T_base_grasp[:3, 3] + np.array([0.05, -0.04, -0.04])
    assert np.allclose(result["placement"].position_m, expected_placement)
    assert np.allclose(result["transport"].position_m, expected_placement + [0, 0, 0.08])
    assert result["transport"].orientation_xyzw == result["placement"].orientation_xyzw
    provenance = result["grounding"]["provenance"]
    assert provenance["fresh_grounded_pick"] == str(grounded_path)
    assert provenance["same_orchestration_run"] is True


def test_plan_only_never_calls_moving_helpers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pose = CartesianPose((0.5, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    output_dir = tmp_path / "review"
    output_dir.mkdir()
    report_path = output_dir / "stage_report.json"
    report = {"stages": []}
    pick_path = output_dir / "pick_review" / "stage_report.json"
    grounding_path = output_dir / "placement_review" / "placement_grounding.json"

    monkeypatch.setattr(runner, "_plan_pose", lambda *args, **kwargs: {"planned": True})

    def fake_pick(*args: object, **kwargs: object) -> tuple[dict, Path, object]:
        assert kwargs["execute"] is False
        return {"status": "review_ready"}, pick_path, object()

    monkeypatch.setattr(runner, "_run_pick", fake_pick)
    monkeypatch.setattr(
        runner,
        "_write_placement",
        lambda *args, **kwargs: {
            "grounding_path": grounding_path,
            "grounding": {"schema": "grounding"},
            "transport": pose,
        },
    )

    def physical_call(*args: object, **kwargs: object) -> None:
        raise AssertionError("plan-only called a physical helper")

    monkeypatch.setattr(runner, "_run_hande", physical_call)
    monkeypatch.setattr(runner, "_run_pose", physical_call)
    monkeypatch.setattr(runner, "_run_sequence", physical_call)
    args = argparse.Namespace(pick_input_dir=None)
    normalized = {
        "lid_stage1_precontact": pose,
        "lid_connected_proof_path": tmp_path / "connected.json",
        "lid_connected_proof": {"passed": True},
        "lid_manifest_path": tmp_path / "manifest.json",
        "lid_manifest": {"schema": "manifest"},
        "placement_relation_path": tmp_path / "relation.json",
    }

    assert _plan_only(args, normalized, report, report_path, output_dir) == 0
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["status"] == "partial_preflight_ready"
    assert written["stages"] == []


def test_execute_mock_runs_every_stage_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pose = CartesianPose((0.5, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    output_dir = tmp_path / "execute"
    output_dir.mkdir()
    report_path = output_dir / "stage_report.json"
    report = {"stages": [], "last_completed_stage": None}
    normalized = {
        "pick": {"thresholds": PickThresholds()},
        "lid_stage1_precontact": pose,
        "lid_stage2_transition_targets": [pose, pose, pose, pose],
        "lid_paths": {
            "stage1_contact_waypoints": tmp_path / "stage1-contact.json",
            "stage1_open_waypoints": tmp_path / "stage1-open.json",
            "stage1_retreat_waypoints": tmp_path / "stage1-retreat.json",
            "stage2_contact_waypoints": tmp_path / "stage2-contact.json",
            "stage2_open_waypoints": tmp_path / "stage2-open.json",
            "stage2_retreat_waypoints": tmp_path / "stage2-retreat.json",
        },
        "lid_min_opening_m": 0.002,
        "lid_max_opening_m": 0.048,
    }

    def fake_hande(*args: object, **kwargs: object) -> tuple[dict, Path]:
        name = str(kwargs["artifact_name"])
        path = output_dir / f"{name}.json"
        return {"ok": True, "state": {"opening_width_mm": 20.0}}, path

    monkeypatch.setattr(runner, "_run_hande", fake_hande)
    monkeypatch.setattr(runner, "hande_open_succeeded", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, "hande_width_snapshot_valid", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, "_lid_contact_verified", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, "_lid_width_snapshot_valid", lambda *args, **kwargs: True)
    pose_artifacts: list[str] = []
    sequence_inputs: list[Path] = []

    def fake_pose(*args: object, **kwargs: object) -> dict[str, str]:
        pose_artifacts.append(str(kwargs["artifact_name"]))
        return {"motion": "pose"}

    def fake_sequence(*args: object, **kwargs: object) -> dict[str, object]:
        sequence_inputs.append(Path(args[1]))
        return {
            "motion": "sequence",
            "measured_final_position_error_m": 0.001,
        }

    monkeypatch.setattr(runner, "_run_pose", fake_pose)
    monkeypatch.setattr(
        runner,
        "_run_sequence",
        fake_sequence,
    )

    def fake_pick(*args: object, **kwargs: object) -> tuple[dict, Path, object]:
        execute = bool(kwargs["execute"])
        names = (
            ["OBSERVE", "GROUND", "PLAN"]
            if not execute
            else [
                "OBSERVE",
                "GROUND",
                "PLAN",
                "PREGRASP",
                "DESCEND",
                "CLOSE",
                "VERIFY_CONTACT",
                "LIFT",
            ]
        )
        pick_dir = Path(kwargs["output_dir"])
        if not execute:
            (pick_dir / "input").mkdir(parents=True)
        return (
            {"stages": [{"name": name} for name in names]},
            pick_dir / "stage_report.json",
            object(),
        )

    monkeypatch.setattr(runner, "_run_pick", fake_pick)
    grounding = {
        "fixed_scene_target_die_anchor_base_m": [0.6, 0.0, 0.0],
        "pick_object_anchor_base_m": [0.5, 0.0, 0.05],
        "anchor_translation_base_m": [0.1, 0.0, -0.05],
    }
    monkeypatch.setattr(
        runner,
        "_write_placement",
        lambda *args, **kwargs: {
            "grounding_path": tmp_path / "grounding.json",
            "grounding": grounding,
            "transport": pose,
            "paths": {
                "place_waypoints": tmp_path / "place.json",
                "retreat_waypoints": tmp_path / "retreat.json",
            },
        },
    )
    monkeypatch.setattr(
        runner,
        "_read_json",
        lambda path: {"state": {"opening_width_mm": 20.0}},
    )
    monkeypatch.setattr(
        runner, "_placement_release_guard", lambda *args, **kwargs: {"accepted": True}
    )

    assert (
        _execute(
            normalized,
            report,
            report_path,
            output_dir,
            accept_unobserved_lid_transition=True,
        )
        == 0
    )
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert [stage["name"] for stage in written["stages"]] == list(STAGE_ORDER)
    assert all(stage["status"] == "passed" for stage in written["stages"])
    assert pose_artifacts[:5] == [
        "lid_stage1_precontact_execution",
        "lid_stage2_clearance_01_execution",
        "lid_stage2_clearance_02_execution",
        "lid_stage2_clearance_03_execution",
        "lid_stage2_precontact_execution",
    ]
    assert sequence_inputs[:6] == [
        normalized["lid_paths"]["stage1_contact_waypoints"],
        normalized["lid_paths"]["stage1_open_waypoints"],
        normalized["lid_paths"]["stage1_retreat_waypoints"],
        normalized["lid_paths"]["stage2_contact_waypoints"],
        normalized["lid_paths"]["stage2_open_waypoints"],
        normalized["lid_paths"]["stage2_retreat_waypoints"],
    ]


def test_execute_stops_after_stage1_when_lid_angle_is_unobserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pose = CartesianPose((0.5, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    output_dir = tmp_path / "execute"
    output_dir.mkdir()
    report_path = output_dir / "stage_report.json"
    report = {"stages": [], "last_completed_stage": None}
    normalized = {
        "pick": {"thresholds": PickThresholds()},
        "lid_stage1_precontact": pose,
        "lid_paths": {
            "stage1_contact_waypoints": tmp_path / "stage1-contact.json",
            "stage1_open_waypoints": tmp_path / "stage1-open.json",
            "stage1_retreat_waypoints": tmp_path / "stage1-retreat.json",
        },
        "lid_min_opening_m": 0.002,
        "lid_max_opening_m": 0.048,
    }

    monkeypatch.setattr(
        runner,
        "_run_hande",
        lambda *args, **kwargs: (
            {"ok": True, "state": {"opening_width_mm": 20.0}},
            output_dir / f"{kwargs['artifact_name']}.json",
        ),
    )
    monkeypatch.setattr(runner, "hande_open_succeeded", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, "_lid_contact_verified", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, "_lid_width_snapshot_valid", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, "_run_pose", lambda *args, **kwargs: {"motion": "pose"})
    monkeypatch.setattr(
        runner,
        "_run_sequence",
        lambda *args, **kwargs: {"motion": "sequence"},
    )

    assert _execute(normalized, report, report_path, output_dir) == 2
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["failed_stage"] == "LID_STAGE1_TRANSITION_GATE"
    assert written["stages"][-1]["status"] == "failed"
    assert "not reobserved" in written["stages"][-1]["details"]["error"]
