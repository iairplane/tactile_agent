#!/usr/bin/env python3
"""Preflight or run the complete resident test-tube stack with one command.

The default is inert: it validates the task and stack manifests, inspects only
read-only service/ROS health when requested, writes one report, and never opens
a sensor or sends a robot command. ``--execute`` first requires every static
qualification artifact, then starts only the explicitly listed managed
processes, waits for F2G-Pose and MoveIt Servo, and delegates the entire task to
``run_test_tube_closed_loop.py``. It never invokes individual task stages.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo2gripper.autonomous_pick import atomic_write_json  # noqa: E402
from demo2gripper.test_tube_live_io import (  # noqa: E402
    assess_live_config,
    provisional_camera_extrinsic_acknowledgement,
    reduced_collision_model_acknowledgement,
)

STACK_SCHEMA = "demo2gripper.test_tube_live_stack.v1"
STACK_REPORT_SCHEMA = "demo2gripper.test_tube_live_stack_report.v1"


class StackContractError(ValueError):
    """A stack manifest is ambiguous or could dispatch outside its contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--stack", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--runtime-check",
        action="store_true",
        help="read F2G health and the ROS graph; sends no motion command",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="start qualified infrastructure and run the full resident task",
    )
    parser.add_argument(
        "--accept-arm-only-collision-model",
        action="store_true",
        help="forward the runner's explicit current collision-model acknowledgement",
    )
    parser.add_argument(
        "--accept-provisional-camera-extrinsic",
        action="store_true",
        help=(
            "accept the configured, validated provisional camera recovery for this "
            "process only; does not create a qualification receipt"
        ),
    )
    parser.add_argument(
        "--init-from-template",
        type=Path,
        help=(
            "in inert mode only, create the missing --stack file and a sibling "
            "non-template task config from this stack template before preflight"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--recover-from-run", type=Path,
        help="validate and reverse the recorded open-gripper approach before the task",
    )
    return parser.parse_args(argv)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StackContractError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise StackContractError(f"{label} root must be an object")
    return value


def _template_candidate(path: Path) -> Path:
    """Return the conventional checked-in template beside an actual JSON path."""

    name = path.name
    if name.endswith(".json"):
        name = f"{name[:-5]}.template.json"
    else:
        name = f"{name}.template.json"
    return path.with_name(name)


def _actual_name_from_template(path: Path, *, parent: Path) -> Path:
    marker = ".template.json"
    if not path.name.endswith(marker):
        raise StackContractError(f"nested task template must end in {marker!r}: {path}")
    return parent / f"{path.name[: -len(marker)]}.json"


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a JSON config without ever replacing operator-edited content."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise StackContractError(f"refusing to overwrite existing config: {path}") from error


def _initialize_from_template(*, target_stack: Path, template_stack: Path) -> dict[str, Any]:
    """Materialize inert working copies of both stack and nested task templates."""

    if target_stack.exists():
        raise StackContractError(f"refusing to overwrite existing stack manifest: {target_stack}")
    if ".template." in target_stack.name:
        raise StackContractError("--stack must name a non-template working manifest")
    template_stack = template_stack.expanduser().resolve()
    stack = _read_json(template_stack, label="stack template")
    if stack.get("schema") != STACK_SCHEMA:
        raise StackContractError(f"stack template schema must be {STACK_SCHEMA!r}")
    task_template = _path(
        stack.get("task_config"),
        base=template_stack.parent,
        label="stack template task_config",
    )
    task = _read_json(task_template, label="nested task template")
    target_task = _actual_name_from_template(task_template, parent=target_stack.parent).resolve()

    created_task = False
    if target_task.exists():
        # Reuse an operator-edited working task, but first prove it is readable.
        _read_json(target_task, label="existing working task config")
    else:
        _write_json_exclusive(target_task, task)
        created_task = True

    initialized_stack = dict(stack)
    initialized_stack["task_config"] = os.path.relpath(target_task, start=target_stack.parent)
    try:
        _write_json_exclusive(target_stack, initialized_stack)
    except BaseException:
        if created_task:
            target_task.unlink(missing_ok=True)
        raise
    return {
        "template_stack": str(template_stack),
        "created_stack": str(target_stack),
        "task_config": str(target_task),
        "task_config_created": created_task,
    }


def _missing_stack_error(path: Path) -> StackContractError:
    candidate = _template_candidate(path)
    if candidate.is_file():
        command = shlex.join(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--stack",
                str(path),
                "--output-dir",
                "runs/preflight/test_tube_live_stack",
                "--init-from-template",
                str(candidate),
            ]
        )
        return StackContractError(
            f"stack manifest does not exist: {path}; template found at {candidate}. "
            f"Create non-overwriting inert working copies and preflight with: {command}"
        )
    return StackContractError(f"stack manifest does not exist: {path}")


def _task_template_guidance(path: Path) -> dict[str, str] | None:
    candidate = _template_candidate(path)
    if not candidate.is_file():
        return None
    return {
        "template": str(candidate),
        "target": str(path),
        "safe_copy_command": shlex.join(["cp", "--no-clobber", str(candidate), str(path)]),
    }


def _path(value: object, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise StackContractError(f"{label} must be a non-empty path")
    result = Path(value).expanduser()
    return (base / result).resolve() if not result.is_absolute() else result.resolve()


def _argv(value: object, *, label: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise StackContractError(f"{label} must be a non-empty argv array")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise StackContractError(f"{label}[{index}] must be a non-empty string")
        result.append(item)
    return result


def _prepare_output(path: Path, *, overwrite: bool) -> Path:
    result = path.expanduser().resolve()
    if result.exists() and any(result.iterdir()) and not overwrite:
        raise StackContractError(f"output directory is not empty: {result}; pass --overwrite")
    result.mkdir(parents=True, exist_ok=True)
    (result / "infrastructure").mkdir(exist_ok=True)
    return result


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _endpoint_health_url(endpoint: str) -> str:
    return endpoint.rstrip("/") + "/health"


def _service_health(
    endpoint: object,
    *,
    backend_name: str,
    label: str,
    require_pose_only: bool = False,
    timeout_s: float = 1.0,
) -> tuple[bool, str]:
    if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
        return False, f"{label}_service_url_missing_or_invalid"
    try:
        # Local SSH forwards must never be sent through a workstation's web
        # proxy. Leave proxy behaviour for non-loopback services unchanged.
        open_url = urllib.request.urlopen
        if urllib.parse.urlsplit(endpoint).hostname in {"127.0.0.1", "localhost", "::1"}:
            open_url = urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        with open_url(_endpoint_health_url(endpoint), timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        return False, f"{label}_service_unhealthy:{type(error).__name__}"
    if not isinstance(payload, Mapping) or payload.get("ready") is not True:
        return False, f"{label}_service_not_ready"
    backend = payload.get("backend")
    if not isinstance(backend, Mapping) or backend.get("name") != backend_name:
        return False, f"{label}_service_backend_identity_mismatch"
    if require_pose_only and backend.get("pose_only") is not True:
        return False, f"{label}_service_not_in_pose_only_mode"
    return True, "ready"


def _f2g_health(endpoint: object, *, timeout_s: float = 1.0) -> tuple[bool, str]:
    return _service_health(
        endpoint,
        backend_name="f2g_pose",
        label="pose",
        require_pose_only=True,
        timeout_s=timeout_s,
    )


def _sam3_endpoint(task: Mapping[str, Any]) -> object:
    live = task.get("live_adapter")
    scene = live.get("scene_interpreter") if isinstance(live, Mapping) else None
    config = scene.get("config") if isinstance(scene, Mapping) else None
    return config.get("sam3_service_url") if isinstance(config, Mapping) else None


def _sam3_health(task: Mapping[str, Any], *, timeout_s: float = 1.0) -> tuple[bool, str]:
    endpoint = _sam3_endpoint(task)
    if endpoint is None:
        return True, "not_required"
    return _service_health(
        endpoint,
        backend_name="sam3",
        label="sam3",
        timeout_s=timeout_s,
    )


def _ros_shell_prefix(task: Mapping[str, Any]) -> str:
    live = task.get("live_adapter")
    robot = live.get("robot") if isinstance(live, Mapping) else None
    moveit = robot.get("moveit") if isinstance(robot, Mapping) else None
    setup = moveit.get("setup_scripts") if isinstance(moveit, Mapping) else None
    if isinstance(setup, (str, bytes)) or not isinstance(setup, Sequence):
        return ""
    commands: list[str] = ["set -e"]
    for item in setup:
        if isinstance(item, str) and item:
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = (REPO_ROOT / path).resolve()
            commands.append(f"source {shlex.quote(str(path))}")
    return "; ".join(commands)


def _ros_lines(task: Mapping[str, Any], command: str) -> tuple[int, tuple[str, ...]]:
    prefix = _ros_shell_prefix(task)
    shell_command = f"{prefix}; {command}" if prefix else command
    try:
        result = subprocess.run(
            ["bash", "-c", shell_command],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 127, ()
    return result.returncode, tuple(line.strip() for line in result.stdout.splitlines())


def _servo_runtime_blockers(task: Mapping[str, Any]) -> list[str]:
    live = task.get("live_adapter")
    robot = live.get("robot") if isinstance(live, Mapping) else None
    servo = robot.get("servo") if isinstance(robot, Mapping) else None
    if not isinstance(servo, Mapping):
        return ["moveit_servo_config_missing"]
    node = str(servo.get("node_name", "/servo_node"))
    twist = str(servo.get("twist_topic", "/servo_node/delta_twist_cmds"))
    status = str(servo.get("status_topic", "/servo_node/status"))
    switch = str(servo.get("switch_command_type_service", "/servo_node/switch_command_type"))
    pause = str(servo.get("pause_service", "/servo_node/pause_servo"))
    blockers: list[str] = []
    code, packages = _ros_lines(task, "ros2 pkg prefix moveit_servo")
    if code != 0 or not packages:
        blockers.append("moveit_servo_package_missing")
        return blockers
    code, nodes = _ros_lines(task, "ros2 node list")
    if code != 0 or node not in nodes:
        blockers.append("moveit_servo_node_not_active")
    code, lines = _ros_lines(task, f"ros2 topic info {shlex.quote(twist)}")
    joined = "\n".join(lines)
    if code != 0 or "Subscription count: 0" in joined:
        blockers.append("moveit_servo_twist_topic_not_ready")
    code, lines = _ros_lines(task, f"ros2 topic info {shlex.quote(status)}")
    joined = "\n".join(lines)
    if code != 0 or "Publisher count: 0" in joined:
        blockers.append("moveit_servo_status_topic_not_ready")
    code, services = _ros_lines(task, "ros2 service list")
    if code != 0 or switch not in services:
        blockers.append("moveit_servo_switch_command_type_service_missing")
    if code != 0 or pause not in services:
        blockers.append("moveit_servo_pause_service_missing")
    return blockers


def _moveit_is_active(task: Mapping[str, Any]) -> bool:
    code, nodes = _ros_lines(task, "ros2 node list")
    return code == 0 and "/move_group" in nodes


def _local_ros2_control_processes(
    proc_root: Path = Path("/proc"),
) -> tuple[dict[str, Any], ...]:
    """Return live local ``ros2_control_node`` processes without using ROS discovery.

    A temporarily stale or isolated ROS graph must never be interpreted as permission
    to launch a second hardware controller.  ``/proc/<pid>/exe`` identifies the
    executable independently of DDS discovery and avoids substring matches against
    shell command lines.  Processes which disappear while being inspected are simply
    omitted.
    """

    found: list[dict[str, Any]] = []
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return ()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            executable = os.readlink(entry / "exe")
        except (OSError, ValueError):
            continue
        normalized = executable.removesuffix(" (deleted)")
        if Path(normalized).name != "ros2_control_node":
            continue
        found.append({"pid": int(entry.name), "executable": normalized})
    return tuple(sorted(found, key=lambda item: int(item["pid"])))


def _replace_servo_package_probe(blockers: list[str], task: Mapping[str, Any]) -> list[str]:
    """Avoid a false missing-package result when the venv lacks ROS sys.path."""

    if "moveit_servo_package_missing" not in blockers:
        return blockers
    code, lines = _ros_lines(task, "ros2 pkg prefix moveit_servo")
    if code == 0 and lines:
        return [item for item in blockers if item != "moveit_servo_package_missing"]
    return blockers


def _managed_section(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = manifest.get(name)
    if not isinstance(value, Mapping):
        raise StackContractError(f"stack.{name} must be an object")
    mode = value.get("mode", "existing")
    if mode not in {"existing", "managed"}:
        raise StackContractError(f"stack.{name}.mode must be existing or managed")
    if mode == "managed":
        _argv(value.get("start_argv"), label=f"stack.{name}.start_argv")
    return value


def _start_managed(
    *,
    name: str,
    section: Mapping[str, Any],
    output_dir: Path,
    append_servo_only: bool = False,
) -> tuple[subprocess.Popen[bytes], Any]:
    argv = _argv(section.get("start_argv"), label=f"stack.{name}.start_argv")
    program = Path(argv[0]).expanduser()
    if not program.is_absolute() and "/" in argv[0]:
        argv[0] = str((REPO_ROOT / program).resolve())
    if append_servo_only and "--servo-only" not in argv:
        argv.append("--servo-only")
    log = (output_dir / "infrastructure" / f"{name}.log").open("wb")
    process = subprocess.Popen(
        argv,
        cwd=REPO_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log


def _stop_managed(processes: list[tuple[str, subprocess.Popen[bytes], Any]]) -> None:
    for _, process, _ in reversed(processes):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5.0
    for _, process, log in reversed(processes):
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        log.close()


def _wait_ready(
    *,
    task: Mapping[str, Any],
    processes: list[tuple[str, subprocess.Popen[bytes], Any]],
    timeout_s: float,
) -> list[str]:
    endpoint = task.get("pose_backend", {}).get("service_url")
    deadline = time.monotonic() + timeout_s
    latest = [
        "pose_service_not_ready",
        "sam3_service_not_ready",
        "moveit_servo_node_not_active",
    ]
    while time.monotonic() < deadline:
        exited = [name for name, process, _ in processes if process.poll() is not None]
        if exited:
            return [f"managed_process_exited:{name}" for name in exited]
        health_ok, health_reason = _f2g_health(endpoint, timeout_s=0.5)
        sam3_ok, sam3_reason = _sam3_health(task, timeout_s=0.5)
        servo_blockers = _servo_runtime_blockers(task)
        latest = (
            ([] if health_ok else [health_reason])
            + ([] if sam3_ok else [sam3_reason])
            + servo_blockers
        )
        if not latest:
            return []
        time.sleep(0.2)
    return latest + ["infrastructure_startup_timeout"]


def _task_command(
    *,
    task: Mapping[str, Any],
    task_config: Path,
    task_output: Path,
    accept_arm_only_collision_model: bool,
    accept_provisional_camera_extrinsic: bool = False,
    recover_from_run: Path | None = None,
) -> list[str]:
    resident_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_test_tube_closed_loop.py"),
        "--config",
        str(task_config),
        "--output-dir",
        str(task_output),
        "--execute",
        "--live-adapter",
        "demo2gripper.test_tube_live_io:create_live_io",
    ]
    if accept_arm_only_collision_model:
        resident_command.append("--accept-arm-only-collision-model")
    if accept_provisional_camera_extrinsic:
        resident_command.append("--accept-provisional-camera-extrinsic")
    if recover_from_run is not None:
        resident_command.extend([
            "--recover-from-run", str(recover_from_run.expanduser().resolve()),
        ])
    prefix = _ros_shell_prefix(task)
    # Login/ROS setup scripts can replace proxy variables. Bind these after
    # sourcing them so both local perception clients reach the SSH forwards.
    command_text = (
        "exec env NO_PROXY=127.0.0.1,localhost,::1 no_proxy=127.0.0.1,localhost,::1 "
        + shlex.join(resident_command)
    )
    if prefix:
        command_text = f"{prefix}; {command_text}"
    return ["bash", "-lc", command_text]


def run(args: argparse.Namespace) -> int:
    stack_path = args.stack.expanduser().resolve()
    manifest = _read_json(stack_path, label="stack manifest")
    if manifest.get("schema") != STACK_SCHEMA:
        raise StackContractError(f"stack.schema must be {STACK_SCHEMA!r}")
    task_path = _path(
        manifest.get("task_config"), base=stack_path.parent, label="stack.task_config"
    )
    task = _read_json(task_path, label="task config")
    recovery_provenance = None
    if args.recover_from_run is not None:
        spec = importlib.util.spec_from_file_location(
            "startup_recovery_reader", REPO_ROOT / "scripts/run_test_tube_closed_loop.py"
        )
        assert spec is not None and spec.loader is not None
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        _, recovery_provenance = reader.load_startup_recovery(
            args.recover_from_run, task_id=task.get("task_id")
        )
    f2g = _managed_section(manifest, "f2g")
    sam3_required = _sam3_endpoint(task) is not None
    sam3 = _managed_section(manifest, "sam3") if sam3_required else None
    servo = _managed_section(manifest, "servo")
    timeout_s = float(manifest.get("startup_timeout_s", 45.0))
    if not 1.0 <= timeout_s <= 300.0:
        raise StackContractError("startup_timeout_s must lie in [1,300]")
    output = _prepare_output(args.output_dir, overwrite=args.overwrite)
    collision_model_acknowledgement = reduced_collision_model_acknowledgement(
        task,
        accepted=args.accept_arm_only_collision_model,
    )
    camera_extrinsic_acknowledgement = provisional_camera_extrinsic_acknowledgement(
        task,
        accepted=args.accept_provisional_camera_extrinsic,
    )
    live_preflight = assess_live_config(
        task,
        accept_arm_only_collision_model=args.accept_arm_only_collision_model,
        accept_provisional_camera_extrinsic=args.accept_provisional_camera_extrinsic,
    )
    blockers = _replace_servo_package_probe(list(live_preflight.static_blockers), task)
    if (
        args.execute
        and collision_model_acknowledgement["receipt_state"]
        == "missing_not_acknowledged"
    ):
        blockers.append("arm_only_collision_model_not_explicitly_acknowledged")
    if args.execute and "realtime" not in platform.release().lower():
        blockers.append("host_kernel_is_not_realtime")
    runtime: dict[str, Any] = {"checked": False}
    if args.runtime_check and not args.execute:
        endpoint = task.get("pose_backend", {}).get("service_url")
        f2g_ok, f2g_reason = _f2g_health(endpoint)
        sam3_ok, sam3_reason = _sam3_health(task)
        servo_blockers = _servo_runtime_blockers(task)
        runtime = {
            "checked": True,
            "f2g_ready": f2g_ok,
            "f2g_detail": f2g_reason,
            "sam3_ready": sam3_ok,
            "sam3_detail": sam3_reason,
            "servo_blockers": servo_blockers,
        }
        if not f2g_ok:
            blockers.append(f2g_reason)
        if not sam3_ok:
            blockers.append(sam3_reason)
        blockers.extend(servo_blockers)
    blockers = sorted(set(blockers))
    report: dict[str, Any] = {
        "schema": STACK_REPORT_SCHEMA,
        "created_at": _now(),
        "stack": str(stack_path),
        "task_config": str(task_path),
        "mode": "execute" if args.execute else "preflight",
        "status": "blocked" if blockers else "static_preflight_ready",
        "hardware_opened": False,
        "startup_recovery": recovery_provenance,
        "hardware_command_dispatched": False,
        "managed_processes_started": [],
        "ros2_control_process_guard": {
            "checked": False,
            "detected_processes": [],
        },
        "static_blockers": blockers,
        "runtime_requirements": list(live_preflight.runtime_requirements),
        "runtime": runtime,
        "collision_model": collision_model_acknowledgement,
        "camera_extrinsic": camera_extrinsic_acknowledgement,
        "task_command": _task_command(
            task=task,
            task_config=task_path,
            task_output=output / "task",
            accept_arm_only_collision_model=args.accept_arm_only_collision_model,
            accept_provisional_camera_extrinsic=(
                args.accept_provisional_camera_extrinsic
            ),
            recover_from_run=args.recover_from_run,
        ),
    }
    atomic_write_json(output / "stack_report.json", report)
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 2 if blockers else 0
    if blockers:
        print(json.dumps(report, indent=2))
        return 2

    processes: list[tuple[str, subprocess.Popen[bytes], Any]] = []
    try:
        endpoint = task.get("pose_backend", {}).get("service_url")
        f2g_ready, _ = _f2g_health(endpoint)
        sam3_ready, _ = _sam3_health(task)
        if not f2g_ready and f2g.get("mode") == "managed":
            process, log = _start_managed(name="f2g", section=f2g, output_dir=output)
            processes.append(("f2g", process, log))
        if sam3_required and not sam3_ready and sam3 is not None and sam3.get("mode") == "managed":
            process, log = _start_managed(name="sam3", section=sam3, output_dir=output)
            processes.append(("sam3", process, log))
        servo_initial_blockers = _servo_runtime_blockers(task)
        servo_ready = not servo_initial_blockers
        guarded_ros2_control_processes: tuple[dict[str, Any], ...] = ()
        if not servo_ready and servo.get("mode") == "managed":
            guarded_ros2_control_processes = _local_ros2_control_processes()
            report["ros2_control_process_guard"] = {
                "checked": True,
                "detected_processes": list(guarded_ros2_control_processes),
            }
            if not guarded_ros2_control_processes:
                process, log = _start_managed(
                    name="servo",
                    section=servo,
                    output_dir=output,
                    append_servo_only=_moveit_is_active(task),
                )
                processes.append(("servo", process, log))
        report["managed_processes_started"] = [name for name, _, _ in processes]
        atomic_write_json(output / "stack_report.json", report)
        unavailable_existing: list[str] = []
        if not f2g_ready and f2g.get("mode") == "existing":
            unavailable_existing.append("existing_pose_service_not_ready")
        if sam3_required and not sam3_ready and sam3 is not None and sam3.get("mode") == "existing":
            unavailable_existing.append("existing_sam3_service_not_ready")
        if not servo_ready and servo.get("mode") == "existing":
            unavailable_existing.extend(servo_initial_blockers)
        if guarded_ros2_control_processes:
            unavailable_existing.append("existing_ros2_control_process_unqualified")
            unavailable_existing.extend(servo_initial_blockers)
        if unavailable_existing:
            report["status"] = "blocked_runtime"
            report["runtime"] = {
                "checked": True,
                "blockers": sorted(set(unavailable_existing)),
            }
            atomic_write_json(output / "stack_report.json", report)
            print(json.dumps(report, indent=2))
            return 2
        runtime_blockers = _wait_ready(task=task, processes=processes, timeout_s=timeout_s)
        if runtime_blockers:
            report["status"] = "blocked_runtime"
            report["runtime"] = {
                "checked": True,
                "blockers": sorted(set(runtime_blockers)),
            }
            atomic_write_json(output / "stack_report.json", report)
            print(json.dumps(report, indent=2))
            return 2

        report["status"] = "task_running"
        report["runtime"] = {"checked": True, "blockers": []}
        atomic_write_json(output / "stack_report.json", report)
        task_command = _task_command(
            task=task,
            task_config=task_path,
            task_output=output / "task",
            accept_arm_only_collision_model=args.accept_arm_only_collision_model,
            accept_provisional_camera_extrinsic=(
                args.accept_provisional_camera_extrinsic
            ),
            recover_from_run=args.recover_from_run,
        )
        completed = subprocess.run(task_command, cwd=REPO_ROOT, check=False)
        report["task_returncode"] = completed.returncode
        task_report_path = output / "task" / "run_report.json"
        report["task_report"] = (
            json.loads(task_report_path.read_text(encoding="utf-8"))
            if task_report_path.is_file()
            else None
        )
        report["hardware_opened"] = bool(
            isinstance(report["task_report"], Mapping)
            and report["task_report"].get("hardware_opened", False)
        )
        report["hardware_command_dispatched"] = bool(
            isinstance(report["task_report"], Mapping)
            and report["task_report"].get("hardware_commands_dispatched", False)
        )
        report["status"] = "complete" if completed.returncode == 0 else "task_failed"
        atomic_write_json(output / "stack_report.json", report)
        return completed.returncode
    finally:
        _stop_managed(processes)


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (OSError, StackContractError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
