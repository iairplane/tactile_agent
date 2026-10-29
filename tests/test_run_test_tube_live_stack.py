from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_test_tube_live_stack.py"
    spec = importlib.util.spec_from_file_location("run_test_tube_live_stack", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifests(tmp_path: Path) -> Path:
    task = {
        "schema": "demo2gripper.test_tube_closed_loop_config.v1",
        "task_id": "blocked-unit-live-task",
        "geometry": {
            "relation_id": "reviewed-normalized-relation",
            "relation_runtime_qualified": False,
            "T_tube_tcp_demo_grasp": None,
            "normalized_demo_grasp": {
                "axial_fraction_closed_to_open": 0.664323374340949,
                "tube_length_m": None,
                "tube_length_source": None,
                "preferred_jaw_axis_base": [1.0, 0.0, 0.0],
                "top_approach_direction_base": [0.0, 0.0, -1.0],
                "tube_pose_origin": "geometric_centre",
                "tube_axis_contract": "closed_to_open",
            },
            "T_socket_tube_entry": None,
            "socket_entry_runtime_qualified": False,
        },
        "pose_backend": {"name": "external", "service_url": "http://127.0.0.1:8765"},
        "live_adapter": {"schema": "demo2gripper.test_tube_live_io.v1"},
    }
    (tmp_path / "task.json").write_text(json.dumps(task), encoding="utf-8")
    stack = {
        "schema": "demo2gripper.test_tube_live_stack.v1",
        "task_config": "task.json",
        "f2g": {"mode": "managed", "start_argv": ["forbidden-f2g"]},
        "servo": {"mode": "managed", "start_argv": ["forbidden-servo"]},
    }
    path = tmp_path / "stack.json"
    path.write_text(json.dumps(stack), encoding="utf-8")
    return path


def test_default_stack_preflight_is_inert_and_lists_physical_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    stack = _write_manifests(tmp_path)

    def forbidden_start(**_: object) -> object:
        raise AssertionError("preflight must not start infrastructure")

    monkeypatch.setattr(module, "_start_managed", forbidden_start)
    monkeypatch.setattr(
        module, "_replace_servo_package_probe", lambda blockers, task: blockers
    )
    output = tmp_path / "output"
    args = module.parse_args(
        ["--stack", str(stack), "--output-dir", str(output)]
    )
    assert module.run(args) == 2
    report = json.loads((output / "stack_report.json").read_text(encoding="utf-8"))
    assert report["hardware_opened"] is False
    assert report["hardware_command_dispatched"] is False
    assert report["managed_processes_started"] == []
    assert "measured_tube_length_missing" in report["static_blockers"]
    assert "T_socket_tube_entry_missing" in report["static_blockers"]


def test_execute_with_static_blockers_never_starts_or_invokes_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    stack = _write_manifests(tmp_path)

    def forbidden(*_: object, **__: object) -> object:
        raise AssertionError("blocked execute must stay inert")

    monkeypatch.setattr(module, "_start_managed", forbidden)
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    monkeypatch.setattr(
        module, "_replace_servo_package_probe", lambda blockers, task: blockers
    )
    output = tmp_path / "output"
    args = module.parse_args(
        ["--stack", str(stack), "--output-dir", str(output), "--execute"]
    )
    assert module.run(args) == 2
    report = json.loads((output / "stack_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["managed_processes_started"] == []
    assert report["hardware_opened"] is False


def test_servo_runtime_preflight_requires_status_publisher_and_pause_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    task = {
        "live_adapter": {
            "robot": {
                "moveit": {"setup_scripts": []},
                "servo": {
                    "node_name": "/servo_node",
                    "twist_topic": "/servo_node/delta_twist_cmds",
                    "status_topic": "/servo_node/status",
                    "switch_command_type_service": "/servo_node/switch_command_type",
                    "pause_service": "/servo_node/pause_servo",
                },
            }
        }
    }

    def fake_ros_lines(_: object, command: str) -> tuple[int, tuple[str, ...]]:
        if command == "ros2 pkg prefix moveit_servo":
            return 0, ("/opt/ros/jazzy",)
        if command == "ros2 node list":
            return 0, ("/servo_node",)
        if command.endswith("/delta_twist_cmds"):
            return 0, ("Publisher count: 0", "Subscription count: 1")
        if command.endswith("/status"):
            return 0, ("Publisher count: 0", "Subscription count: 1")
        if command == "ros2 service list":
            return 0, ("/servo_node/switch_command_type",)
        raise AssertionError(command)

    monkeypatch.setattr(module, "_ros_lines", fake_ros_lines)
    blockers = module._servo_runtime_blockers(task)
    assert "moveit_servo_status_topic_not_ready" in blockers
    assert "moveit_servo_pause_service_missing" in blockers
    assert "moveit_servo_twist_topic_not_ready" not in blockers


def test_local_ros2_control_process_scan_matches_executable_not_arguments(
    tmp_path: Path,
) -> None:
    module = _load_module()
    executable = tmp_path / "bin" / "ros2_control_node"
    executable.parent.mkdir()
    executable.write_bytes(b"")
    proc = tmp_path / "proc"
    controller = proc / "1234"
    unrelated = proc / "5678"
    controller.mkdir(parents=True)
    unrelated.mkdir()
    (controller / "exe").symlink_to(executable)
    (unrelated / "exe").symlink_to(tmp_path / "mentions_ros2_control_node_only")

    assert module._local_ros2_control_processes(proc) == (
        {"pid": 1234, "executable": str(executable)},
    )


def test_managed_servo_never_starts_over_existing_ros2_control_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    task = {
        "schema": "demo2gripper.test_tube_closed_loop_config.v1",
        "task_id": "duplicate-controller-guard",
        "pose_backend": {"name": "external", "service_url": "http://127.0.0.1:8765"},
        "live_adapter": {"schema": "demo2gripper.test_tube_live_io.v1"},
    }
    (tmp_path / "task.json").write_text(json.dumps(task), encoding="utf-8")
    stack = {
        "schema": "demo2gripper.test_tube_live_stack.v1",
        "task_config": "task.json",
        "f2g": {"mode": "existing"},
        "servo": {"mode": "managed", "start_argv": ["forbidden-servo"]},
    }
    stack_path = tmp_path / "stack.json"
    stack_path.write_text(json.dumps(stack), encoding="utf-8")

    monkeypatch.setattr(
        module,
        "assess_live_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            static_blockers=(), runtime_requirements=()
        ),
    )
    acknowledgement = {"active": False, "receipt_state": "not_required"}
    monkeypatch.setattr(
        module,
        "reduced_collision_model_acknowledgement",
        lambda *_args, **_kwargs: acknowledgement,
    )
    monkeypatch.setattr(
        module,
        "provisional_camera_extrinsic_acknowledgement",
        lambda *_args, **_kwargs: acknowledgement,
    )
    monkeypatch.setattr(module.platform, "release", lambda: "6.8.1-realtime")
    monkeypatch.setattr(module, "_f2g_health", lambda *_args, **_kwargs: (True, "ready"))
    monkeypatch.setattr(
        module,
        "_servo_runtime_blockers",
        lambda _task: ["moveit_servo_node_not_active"],
    )
    monkeypatch.setattr(
        module,
        "_local_ros2_control_processes",
        lambda: ({"pid": 42, "executable": "/opt/ros/bin/ros2_control_node"},),
    )

    def forbidden_start(**_: object) -> object:
        raise AssertionError("must not start over an existing ros2_control_node")

    monkeypatch.setattr(module, "_start_managed", forbidden_start)
    output = tmp_path / "output"
    args = module.parse_args(
        ["--stack", str(stack_path), "--output-dir", str(output), "--execute"]
    )

    assert module.run(args) == 2
    report = json.loads((output / "stack_report.json").read_text(encoding="utf-8"))
    assert report["managed_processes_started"] == []
    assert report["status"] == "blocked_runtime"
    assert "existing_ros2_control_process_unqualified" in report["runtime"]["blockers"]
    assert report["ros2_control_process_guard"] == {
        "checked": True,
        "detected_processes": [
            {"pid": 42, "executable": "/opt/ros/bin/ros2_control_node"}
        ],
    }


def test_resident_task_command_sources_ros_and_keeps_current_python(
    tmp_path: Path,
) -> None:
    module = _load_module()
    first = tmp_path / "ros setup.bash"
    second = tmp_path / "workspace setup.bash"
    task = {
        "live_adapter": {
            "robot": {
                "moveit": {"setup_scripts": [str(first), str(second)]}
            }
        }
    }
    command = module._task_command(
        task=task,
        task_config=tmp_path / "task config.json",
        task_output=tmp_path / "task output",
        accept_arm_only_collision_model=True,
        accept_provisional_camera_extrinsic=True,
    )
    assert command[:2] == ["bash", "-lc"]
    shell = command[2]
    assert f"source '{first}'" in shell
    assert f"source '{second}'" in shell
    assert "exec env NO_PROXY=127.0.0.1,localhost,::1" in shell
    assert f"no_proxy=127.0.0.1,localhost,::1 {module.sys.executable}" in shell
    assert "--accept-arm-only-collision-model" in shell
    assert "--accept-provisional-camera-extrinsic" in shell


def test_demo06_inert_preflight_records_process_only_reduced_collision_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    repo = Path(__file__).resolve().parents[1]

    def forbidden_start(**_: object) -> object:
        raise AssertionError("inert preflight must not start infrastructure")

    monkeypatch.setattr(module, "_start_managed", forbidden_start)
    monkeypatch.setattr(
        module,
        "_replace_servo_package_probe",
        lambda blockers, _task: [
            item for item in blockers if item != "moveit_servo_package_missing"
        ],
    )
    output = tmp_path / "preflight"
    args = module.parse_args(
        [
            "--stack",
            str(repo / "configs/test_tube_live_stack.5090.json"),
            "--output-dir",
            str(output),
            "--accept-arm-only-collision-model",
        ]
    )

    assert module.run(args) == 2
    report = json.loads((output / "stack_report.json").read_text(encoding="utf-8"))
    assert report["hardware_opened"] is False
    assert report["hardware_command_dispatched"] is False
    assert report["managed_processes_started"] == []
    assert report["static_blockers"] == [
        "fixed_camera_extrinsic_qualification_receipt_missing",
    ]
    collision = report["collision_model"]
    assert collision["active"] is True
    assert collision["current_process_only"] is True
    assert collision["qualification_receipt_substituted"] is False
    linear = collision["speed_limits"]["servo_max_linear_m_s"]
    assert linear["cap"] == 0.03
    assert linear["effective"] == min(linear["configured"], linear["cap"])
    assert "--accept-arm-only-collision-model" in report["task_command"][2]


def test_demo06_inert_preflight_records_process_only_provisional_camera_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    repo = Path(__file__).resolve().parents[1]

    def forbidden_start(**_: object) -> object:
        raise AssertionError("inert preflight must not start infrastructure")

    monkeypatch.setattr(module, "_start_managed", forbidden_start)
    monkeypatch.setattr(
        module,
        "_replace_servo_package_probe",
        lambda blockers, _task: [
            item for item in blockers if item != "moveit_servo_package_missing"
        ],
    )
    output = tmp_path / "preflight"
    args = module.parse_args(
        [
            "--stack",
            str(repo / "configs/test_tube_live_stack.5090.json"),
            "--output-dir",
            str(output),
            "--accept-arm-only-collision-model",
            "--accept-provisional-camera-extrinsic",
        ]
    )

    assert module.run(args) == 0
    report = json.loads((output / "stack_report.json").read_text(encoding="utf-8"))
    assert report["hardware_opened"] is False
    assert report["hardware_command_dispatched"] is False
    assert report["static_blockers"] == []
    camera = report["camera_extrinsic"]
    assert camera["active"] is True
    assert camera["current_process_only"] is True
    assert camera["qualification_receipt_substituted"] is False
    assert camera["selected_frame_count"] == 8
    assert camera["current_frame_count"] == 10
    assert camera["derived_uncertainty"]["position_std_m"] == pytest.approx(
        0.0060199830007769615
    )
    assert camera["derived_uncertainty"]["orientation_std_deg"] == pytest.approx(
        0.7848673650969283
    )
    assert "--accept-provisional-camera-extrinsic" in report["task_command"][2]
