from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from demo2gripper.test_tube_task_runner import (
    GUARDED_LOCALIZATION_RECEIPT_SCHEMA,
    TEST_TUBE_TASK_INPUT_SCHEMA,
    RunnerMode,
    StageEvidence,
    TaskContractError,
    command_preview,
)
from demo2gripper.test_tube_task_runner import (
    TestTubeTaskInput as TubeTaskInput,
)
from demo2gripper.test_tube_task_runner import (
    TestTubeTaskMachine as TubeTaskMachine,
)
from demo2gripper.test_tube_task_runner import (
    TestTubeTaskStage as TubeTaskStage,
)


def _pose(x: float, y: float, z: float) -> dict[str, object]:
    return {
        "position_m": [x, y, z],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def _evidence(status: str, name: str) -> dict[str, object]:
    if status == "unavailable":
        return {"status": status, "receipt_id": None, "observed_at_s": None}
    return {
        "status": status,
        "receipt_id": f"receipt-{name}",
        "observed_at_s": 100.0,
    }


def _payload(*, qualified: bool = False, guarded: bool = False) -> dict[str, object]:
    status = "qualified" if qualified else "provisional"
    payload: dict[str, object] = {
        "schema": TEST_TUBE_TASK_INPUT_SCHEMA,
        "task_id": "tube-task-1",
        "base_frame": "fr3_link0",
        "geometry": {
            "camera_extrinsic": _evidence(status, "camera"),
            "source": _evidence(status, "source"),
            "socket": _evidence(status, "socket"),
            "collision_scene": _evidence(status, "collision"),
        },
        "poses": {
            "approach": _pose(0.50, 0.10, 0.30),
            "grasp": _pose(0.50, 0.10, 0.20),
            "extract": _pose(0.50, 0.10, 0.27),
            "pivot": _pose(0.50, 0.10, 0.27),
            "reclamp": _pose(0.50, 0.10, 0.27),
            "entry_standoff": _pose(0.35, 0.20, 0.22),
        },
        "pivot": {
            "target_source": "front_rgb",
            "rgb_status": status,
            "rgb_qualification_id": "rgb-q-1",
            "tactile_axis_mode": "shadow",
        },
        "tactile": {
            "bilateral_contact_required": True,
            "jam_profile_status": "qualified" if guarded else "provisional",
        },
        "gripper": {
            "port": "/dev/ttyUSB0",
            "open_width_mm": 49.0,
            "grasp_width_min_mm": 8.0,
            "grasp_width_max_mm": 20.0,
        },
        "policy": {
            "pivot_open_step_mm": 0.25,
            "pivot_max_steps": 8,
            "reclamp_step_mm": 0.50,
            "reclamp_max_steps": 4,
            "insertion_step_m": 0.0015,
            "insertion_max_steps": 20,
            "maximum_jam_recoveries": 2,
        },
        "guarded_localization_receipt": None,
    }
    if guarded:
        payload["guarded_localization_receipt"] = {
            "schema": GUARDED_LOCALIZATION_RECEIPT_SCHEMA,
            "status": "qualified",
            "receipt_id": "guarded-1",
            "task_id": "tube-task-1",
            "observation_id": "obs-1",
            "collision_scene_id": "scene-1",
            "observed_at_s": 101.0,
            "insertion_axis_base": [1.0, 0.0, 0.0],
            "target_depth_m": 0.020,
            "visual_localization_qualified": True,
            "touch_search_qualified": False,
            "runtime_jam_qualified": True,
        }
    return payload


def test_provisional_geometry_is_plan_only_and_never_execute_authority() -> None:
    task = TubeTaskInput.parse(_payload())

    assert set(task.execution_start_blockers()) == {
        "camera_extrinsic_not_qualified",
        "source_geometry_not_qualified",
        "collision_scene_not_qualified",
        "front_rgb_pivot_gate_not_qualified",
    }
    preview = command_preview(task, RunnerMode.PLAN_ONLY)
    assert preview
    assert all(item["execute"] is False for item in preview)

    rejected_execute_preview = command_preview(task, RunnerMode.EXECUTE)
    assert rejected_execute_preview
    assert all(item["execute"] is False for item in rejected_execute_preview)


def test_missing_metric_socket_receipt_holds_at_entry_standoff() -> None:
    task = TubeTaskInput.parse(_payload(qualified=True))
    machine = TubeTaskMachine(task)
    machine.stage = TubeTaskStage.APPROACH_ENTRY

    reached = machine.apply(
        StageEvidence(
            evidence_id="entry-motion",
            motion_complete=True,
            entry_standoff_reached=True,
        )
    )
    assert reached.next_stage is TubeTaskStage.ENTRY_STANDOFF

    blocked = machine.apply(
        StageEvidence(
            evidence_id="entry-check",
            guarded_localization_qualified=False,
        )
    )
    assert blocked.next_stage is TubeTaskStage.ENTRY_STANDOFF
    assert blocked.action == "HOLD_ENTRY_STANDOFF"
    assert "guarded_visual_or_touch_localization_receipt_missing" in blocked.reason


def test_extraction_requires_bilateral_contact() -> None:
    task = TubeTaskInput.parse(_payload(qualified=True))
    machine = TubeTaskMachine(task)
    machine.stage = TubeTaskStage.GRASP_EXTRACT

    transition = machine.apply(
        StageEvidence(
            evidence_id="one-pad-only",
            contact_present=True,
            bilateral_contact=False,
            extraction_complete=True,
        )
    )
    assert transition.next_stage is TubeTaskStage.HOLD
    assert "bilateral" in transition.reason


def test_tactile_axis_shadow_never_replaces_qualified_rgb_pivot() -> None:
    task = TubeTaskInput.parse(_payload(qualified=True))
    machine = TubeTaskMachine(task)
    machine.stage = TubeTaskStage.PIVOT

    shadow = machine.apply(
        StageEvidence(
            evidence_id="shadow-angle",
            contact_present=True,
            tactile_axis_target_shadow=True,
        )
    )
    assert shadow.next_stage is TubeTaskStage.PIVOT
    assert shadow.action == "PIVOT_OPEN_STEP"

    rgb = machine.apply(
        StageEvidence(
            evidence_id="rgb-angle",
            contact_present=True,
            rgb_pivot_target_qualified=True,
        )
    )
    assert rgb.next_stage is TubeTaskStage.RECLAMP


def test_jam_needs_low_progress_resistance_qualification_and_persistence() -> None:
    task = TubeTaskInput.parse(_payload(qualified=True, guarded=True))
    machine = TubeTaskMachine(task)
    machine.stage = TubeTaskStage.GUARDED_INSERT

    low_progress_without_resistance = machine.apply(
        StageEvidence(
            evidence_id="low-progress-only",
            contact_present=True,
            commanded_insertion_step_m=0.0015,
            measured_insertion_progress_m=0.0001,
            jam_runtime_qualified=True,
            tactile_resistance_high=False,
            jam_confirmation_windows=2,
        )
    )
    assert low_progress_without_resistance.next_stage is TubeTaskStage.GUARDED_INSERT
    assert low_progress_without_resistance.action == "INSERT_MICROSTEP"

    qualified_jam = machine.apply(
        StageEvidence(
            evidence_id="qualified-jam",
            contact_present=True,
            commanded_insertion_step_m=0.0015,
            measured_insertion_progress_m=0.0001,
            jam_runtime_qualified=True,
            tactile_resistance_high=True,
            jam_confirmation_windows=2,
        )
    )
    assert qualified_jam.next_stage is TubeTaskStage.APPROACH_ENTRY
    assert qualified_jam.action == "RETRACT_3MM_REOBSERVE_REALIGN"


def test_guarded_receipt_allows_entry_transition() -> None:
    task = TubeTaskInput.parse(_payload(qualified=True, guarded=True))
    assert task.execution_start_blockers() == ()
    assert task.guarded_insertion_blockers() == ()
    machine = TubeTaskMachine(task)
    machine.stage = TubeTaskStage.ENTRY_STANDOFF

    transition = machine.apply(
        StageEvidence(
            evidence_id="guarded-current",
            guarded_localization_qualified=True,
        )
    )
    assert transition.next_stage is TubeTaskStage.GUARDED_INSERT


def test_unavailable_socket_cannot_smuggle_an_entry_pose() -> None:
    payload = _payload()
    payload["geometry"]["socket"] = _evidence("unavailable", "socket")  # type: ignore[index]

    with pytest.raises(TaskContractError, match="entry_standoff"):
        TubeTaskInput.parse(payload)


def test_replayed_evidence_fails_closed() -> None:
    machine = TubeTaskMachine(TubeTaskInput.parse(_payload(qualified=True)))
    evidence = StageEvidence(evidence_id="same", motion_complete=False)
    machine.apply(evidence)

    transition = machine.apply(copy.copy(evidence))
    assert transition.next_stage is TubeTaskStage.HOLD
    assert "replayed" in transition.reason


def _load_runner_script():
    script = Path(__file__).resolve().parents[1] / "scripts/run_test_tube_task.py"
    spec = importlib.util.spec_from_file_location("test_tube_task_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_default_dry_run_never_spawns_a_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_runner_script()
    task_input = tmp_path / "task.json"
    task_input.write_text(json.dumps(_payload()), encoding="utf-8")
    output = tmp_path / "dry-run"
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dry run spawned a component")
        ),
    )

    args = module.parse_args(
        ["--input", str(task_input), "--output-dir", str(output)]
    )
    assert module.run(args) == 0
    report = json.loads((output / "task_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "dry_run"
    assert report["hardware_commands_dispatched"] is False
    assert all(item["execute"] is False for item in report["command_preview"])


def test_cli_rejects_provisional_execute_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_runner_script()
    task_input = tmp_path / "task.json"
    task_input.write_text(json.dumps(_payload()), encoding="utf-8")
    output = tmp_path / "execute-rejected"
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("blocked execute spawned MoveIt")
        ),
    )

    args = module.parse_args(
        [
            "--input",
            str(task_input),
            "--output-dir",
            str(output),
            "--execute",
        ]
    )
    assert module.run(args) == 2
    report = json.loads((output / "task_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "execution_blocked_before_dispatch"
    assert report["hardware_commands_dispatched"] is False
    assert all(item["execute"] is False for item in report["command_preview"])
