from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demo2gripper.medicine_bottle_tactile_bootstrap import (
    TactileBootstrapQualityError,
)
from scripts import run_medicine_bottle_task as runner

REPO = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path) -> Path:
    payload = json.loads(
        (REPO / "configs/medicine_bottle_task.example.json").read_text(encoding="utf-8")
    )
    cap_frame = tmp_path / "cap_frame.json"
    cap_frame.write_text(
        json.dumps(
            {
                "schema": "demo2gripper.medicine_bottle_cap_frame.v1",
                "transform_convention": "T_parent_child; right-handed; column vectors",
                "units": "metres",
                "axis_direction": "bottle_base_to_cap",
                "base_frame": "fr3_link0",
                "cap_frame": "medicine_cap_axis",
                "tcp_frame": "hande_tcp",
                "T_base_cap_tcp": [
                    [1.0, 0.0, 0.0, 0.50],
                    [0.0, 1.0, 0.0, 0.00],
                    [0.0, 0.0, 1.0, 0.10],
                    [0.0, 0.0, 0.0, 1.00],
                ],
                "bottle_axis_base": [0.0, 0.0, 1.0],
                "bottle_axis_tcp": [0.0, 0.0, 1.0],
            }
        ),
        encoding="utf-8",
    )
    tool = tmp_path / "T_link8_tcp.npy"
    np.save(tool, np.eye(4))
    payload["repo_root"] = str(REPO)
    payload["inputs"] = {
        "cap_frame": str(cap_frame),
        "T_link8_configured_ee": str(tool),
    }
    payload["gf220"]["manual_warp_profile"] = str(REPO / "configs/gf220_manual_warp_draft.json")
    payload["gf220"]["wrench_transform_profile"] = str(
        REPO / "configs/gf220_wrench_transform_draft.json"
    )
    path = tmp_path / "task.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_default_plan_only_writes_review_artifacts_without_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    output = tmp_path / "plan"

    def forbidden(*args, **kwargs):
        raise AssertionError("plan-only must not start a subprocess")

    monkeypatch.setattr(runner, "_run_process", forbidden)
    args = argparse.Namespace(
        config=config,
        output_dir=output,
        plan_only=False,
        execute=False,
        accept_arm_only_collision_model=False,
    )

    assert runner.run(args) == 0
    report = json.loads((output / "stage_report.json").read_text(encoding="utf-8"))
    plan = json.loads((output / "resolved_plan.json").read_text(encoding="utf-8"))
    assert report["status"] == "review_ready"
    assert plan["execution_authorized"] is False
    assert "--execute" in plan["command_templates"]["approach"]
    assert "--accept-arm-only-collision-model" in plan["command_templates"]["approach"]


def test_contact_approach_uses_bounded_interpolated_waypoints(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output = tmp_path / "plan"
    args = argparse.Namespace(
        config=config,
        output_dir=output,
        plan_only=False,
        execute=False,
        accept_arm_only_collision_model=False,
    )

    assert runner.run(args) == 0
    payload = json.loads(
        (output / "plan_inputs" / "contact_sequence.json").read_text(encoding="utf-8")
    )
    waypoints = payload["waypoints"]
    normalized = runner._validate_config(
        runner._read_json(config), config_path=config, execute=False
    )
    approach = runner._resolved_transforms(normalized)["approach"][:3, 3]
    positions = [approach, *(np.asarray(item["position_m"], dtype=float) for item in waypoints)]
    distances = [
        float(np.linalg.norm(current - previous))
        for previous, current in zip(positions, positions[1:])
    ]
    assert len(waypoints) >= 2
    assert max(distances) <= runner.CONTACT_WAYPOINT_MAX_SEGMENT_M + 1e-12
    np.testing.assert_allclose(
        positions[-1], runner._resolved_transforms(normalized)["contact"][:3, 3]
    )


def test_execute_requires_collision_model_acknowledgement(tmp_path: Path) -> None:
    args = argparse.Namespace(
        config=_config(tmp_path),
        output_dir=tmp_path / "out",
        plan_only=False,
        execute=True,
        accept_arm_only_collision_model=False,
    )
    with pytest.raises(ValueError, match="requires --accept-arm-only-collision-model"):
        runner.run(args)


def test_execute_requires_realtime_kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.platform, "release", lambda: "7.0.0-generic")
    args = argparse.Namespace(
        config=_config(tmp_path),
        output_dir=tmp_path / "out",
        plan_only=False,
        execute=True,
        accept_arm_only_collision_model=True,
    )
    with pytest.raises(RuntimeError, match="realtime kernel"):
        runner.run(args)


def test_draft_wrench_profile_is_geometry_seed_for_runtime_bootstrap(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    payload = runner._read_json(config_path)
    runner._validate_config(payload, config_path=config_path, execute=False)
    payload["grasp"]["fixture_verified"] = True
    payload["grasp"]["orientation_reviewed"] = True
    normalized = runner._validate_config(payload, config_path=config_path, execute=True)
    assert normalized["wrench_meta"]["status"] == "draft"


def test_execute_recovery_failures_do_not_mask_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config(tmp_path)
    config = runner._validate_config(
        runner._read_json(config_path), config_path=config_path, execute=False
    )
    output = tmp_path / "execute"
    output.mkdir()
    log = runner.AtomicActionLog(output / "action_log.json", mode="execute")
    sources: list[object] = []

    class FakeSource:
        def __init__(self, *args, **kwargs) -> None:
            self.closed = False
            sources.append(self)

        def start(self, *, timeout_s: float) -> None:
            return None

        def calibrate_unloaded(self, *, timeout_s: float) -> list[object]:
            return [
                SimpleNamespace(
                    serial="GF220GF22000184603F72A", calibration_id="left-cal"
                ),
                SimpleNamespace(
                    serial="GF220GF22000196FB398F5", calibration_id="right-cal"
                ),
            ]

        def collect_classifier_windows(self, *args, **kwargs) -> object:
            return SimpleNamespace(
                serials=(
                    "GF220GF22000184603F72A",
                    "GF220GF22000196FB398F5",
                )
            )

        def close(self) -> None:
            self.closed = True

    def fake_run_process(
        command: list[str], cwd: Path, prefix: Path
    ) -> subprocess.CompletedProcess[str]:
        if prefix.name == "hande_cap_grasp":
            raise runner.TaskFailure("original grasp failure")
        if prefix.name.startswith("failure_recovery"):
            raise runner.TaskFailure(f"recovery failed: {prefix.name}")
        if "--output" in command:
            output_path = Path(command[command.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps({"execution_succeeded": True}), encoding="utf-8"
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"ok": True, "state": {}}) + "\n",
            stderr="",
        )

    monkeypatch.setattr(runner, "GF220LiveSource", FakeSource)
    monkeypatch.setattr(runner, "_run_process", fake_run_process)

    with pytest.raises(runner.TaskFailure, match="original grasp failure"):
        runner._execute(config, output, log)

    assert sources and sources[0].closed is True
    events = json.loads((output / "action_log.json").read_text(encoding="utf-8"))["events"]
    assert any(
        event["operation"] == "failure_recovery_hande_open"
        and event["state"] == "failed"
        for event in events
    )
    assert any(
        event["operation"] == "failure_recovery_retract"
        and event["state"] == "failed"
        for event in events
    )


def _hande_status(*, opening_mm: float, activated: bool = True, fault: int = 0) -> dict:
    return {
        "ok": True,
        "state": {
            "activated": activated,
            "fault_code": fault,
            "opening_width_mm": opening_mm,
            # The read-only status process may see these transient fields reset.
            "go_to": False,
            "object_status": {"code": 0, "grasped": False},
        },
    }


def test_retry_hold_accepts_retained_opening_when_transient_gobj_is_clear() -> None:
    state = runner._require_hande_hold_retained(
        _hande_status(opening_mm=40.039),
        operation="retry",
        commanded_grasp_opening_mm=18.0,
        full_opening_mm=50.0,
        initial_opening_mm=40.1,
    )

    assert state["opening_width_mm"] == pytest.approx(40.039)


def test_retry_hold_rejects_opening_drift_from_initial_grasp() -> None:
    with pytest.raises(runner.TaskFailure, match="retained opening drift"):
        runner._require_hande_hold_retained(
            _hande_status(opening_mm=36.0),
            operation="retry",
            commanded_grasp_opening_mm=18.0,
            full_opening_mm=50.0,
            initial_opening_mm=40.1,
        )


def test_retry_hold_uses_conservative_band_without_initial_opening() -> None:
    state = runner._require_hande_hold_retained(
        _hande_status(opening_mm=40.039),
        operation="retry",
        commanded_grasp_opening_mm=18.0,
        full_opening_mm=50.0,
    )
    assert state["opening_width_mm"] == pytest.approx(40.039)

    with pytest.raises(runner.TaskFailure, match="does not retain the cap"):
        runner._require_hande_hold_retained(
            _hande_status(opening_mm=49.0),
            operation="retry",
            commanded_grasp_opening_mm=18.0,
            full_opening_mm=50.0,
        )


def _bootstrap_pair(start_s: float) -> SimpleNamespace:
    count = 5
    serials = ("left", "right")
    windows = []
    for index, serial in enumerate(serials):
        host = start_s + index * 0.001 + 0.01 * np.arange(count)
        force = SimpleNamespace(
            values=np.zeros((count, 6), dtype=float),
            sdk_time_s=1000.0 * host,
            capture_monotonic_s=host,
            valid=np.ones(count, dtype=bool),
        )
        windows.append(
            SimpleNamespace(calibration_id=f"{serial}-cal", force6d=force)
        )
    return SimpleNamespace(
        serials=serials,
        windows=tuple(windows),
        endpoint_skew_s=0.001,
        collected_monotonic_s=start_s + 0.05,
    )


class _BootstrapIO:
    def __init__(self, config: dict, *, fail_recheck: bool = False) -> None:
        self.config = config
        self.not_before_s = 0.0
        self.segment_press_depth_m = 0.0
        self.press_attempts: list[int] = []
        self.rechecks: list[int] = []
        self.installed = None
        self.retracted_depth_m: float | None = None
        self.fail_recheck = fail_recheck

    def press_engage(self, *, context) -> None:
        self.press_attempts.append(context.press_retry_count)
        self.segment_press_depth_m += (
            self.config["initial_press_m"]
            if context.press_retry_count == 0
            else self.config["retry_press_m"]
        )
        self.not_before_s += 1.0

    def verify_grasp_before_bootstrap_retry(self, *, attempt: int) -> None:
        self.rechecks.append(attempt)
        if self.fail_recheck:
            raise runner.TaskFailure("Hand-E reported a non-zero fault")

    def install_tactile_bootstrap(self, result) -> None:
        self.installed = result

    def retract_bootstrap_press(self) -> None:
        self.retracted_depth_m = self.segment_press_depth_m
        self.segment_press_depth_m = 0.0


class _BootstrapSource:
    def __init__(self, pairs: list[SimpleNamespace]) -> None:
        self.pairs = list(pairs)

    def collect_classifier_windows(self, *args) -> SimpleNamespace:
        return self.pairs.pop(0)


def _absent_press_response(value: float = 0.0) -> TactileBootstrapQualityError:
    return TactileBootstrapQualityError(
        "axial_press_tactile_response_absent",
        f"any-pad press response {value}",
        details={"observed_any_pad_fz_change": value, "minimum_change": 1e-6},
    )


def _run_bootstrap_helper(
    tmp_path: Path,
    *,
    outcomes: list[object],
    fail_recheck: bool = False,
) -> tuple[object, _BootstrapIO, dict]:
    config_path = _config(tmp_path)
    config = runner._validate_config(
        runner._read_json(config_path), config_path=config_path, execute=False
    )
    output = tmp_path / "bootstrap"
    output.mkdir()
    log = runner.AtomicActionLog(output / "action_log.json", mode="execute")
    io = _BootstrapIO(config, fail_recheck=fail_recheck)
    source = _BootstrapSource([_bootstrap_pair(3.0 + index) for index in range(3)])
    pending = list(outcomes)

    def bootstrap_fn(*args, **kwargs):
        outcome = pending.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    result = runner._run_task_session_tactile_bootstrap(
        config=config,
        output_dir=output,
        log=log,
        source=source,
        io=io,
        seed_adapter=config["adapter_plan_config"],
        unloaded_window=_bootstrap_pair(1.0),
        grasp_window=_bootstrap_pair(2.0),
        bootstrap_fn=bootstrap_fn,
    )
    attempts = json.loads(
        (output / "tactile_bootstrap_attempts.json").read_text(encoding="utf-8")
    )
    return result, io, attempts


def test_bootstrap_retries_2mm_3mm_4mm_only_while_press_response_is_absent(
    tmp_path: Path,
) -> None:
    success = SimpleNamespace(report={"status": "passed"})

    result, io, attempts = _run_bootstrap_helper(
        tmp_path,
        outcomes=[_absent_press_response(), _absent_press_response(), success],
    )

    assert result is success
    assert io.press_attempts == [0, 1, 2]
    assert io.rechecks == [1, 2]
    assert io.retracted_depth_m == pytest.approx(0.004)
    assert attempts["status"] == "passed"
    assert [item["status"] for item in attempts["attempts"]] == [
        "rejected_quality",
        "rejected_quality",
        "passed",
    ]
    with np.load(tmp_path / "bootstrap" / "tactile_bootstrap_windows.npz") as raw:
        assert "axial_press_0_0_wrench" in raw
        assert "axial_press_1_0_wrench" in raw
        assert "axial_press_2_0_wrench" in raw


def test_bootstrap_does_not_retry_nonwhitelisted_quality_failure(
    tmp_path: Path,
) -> None:
    failure = TactileBootstrapQualityError(
        "axial_press_imbalance_exceeded",
        "imbalanced",
        details={"observed_imbalance": 0.9},
    )

    with pytest.raises(runner.TaskFailure, match="quality_reason_not_retryable"):
        _run_bootstrap_helper(tmp_path, outcomes=[failure])

    attempts = json.loads(
        (tmp_path / "bootstrap" / "tactile_bootstrap_attempts.json").read_text()
    )
    assert len(attempts["attempts"]) == 1
    assert attempts["attempts"][0]["retry_authorized"] is False


def test_bootstrap_hande_recheck_failure_blocks_next_press(tmp_path: Path) -> None:
    with pytest.raises(runner.TaskFailure, match="non-zero fault"):
        _run_bootstrap_helper(
            tmp_path,
            outcomes=[_absent_press_response()],
            fail_recheck=True,
        )

    attempts = json.loads(
        (tmp_path / "bootstrap" / "tactile_bootstrap_attempts.json").read_text()
    )
    assert len(attempts["attempts"]) == 1


def test_bootstrap_absent_press_response_exhausts_bounded_retries(
    tmp_path: Path,
) -> None:
    with pytest.raises(runner.TaskFailure, match="bounded_press_retries_exhausted"):
        _run_bootstrap_helper(
            tmp_path,
            outcomes=[
                _absent_press_response(),
                _absent_press_response(),
                _absent_press_response(),
            ],
        )

    attempts = json.loads(
        (tmp_path / "bootstrap" / "tactile_bootstrap_attempts.json").read_text()
    )
    assert len(attempts["attempts"]) == 3
    assert attempts["attempts"][-1]["retry_authorized"] is False


def _capture_only_motion_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[runner.SubprocessMedicineBottleIO, dict, list[tuple[str, list[np.ndarray]]]]:
    config_path = _config(tmp_path)
    config = runner._validate_config(
        runner._read_json(config_path), config_path=config_path, execute=False
    )
    output = tmp_path / "motion"
    output.mkdir()
    io = runner.SubprocessMedicineBottleIO(
        config=config,
        output_dir=output,
        action_log=runner.AtomicActionLog(output / "action_log.json", mode="execute"),
        tactile_source=SimpleNamespace(),
        adapter_config=SimpleNamespace(),
    )
    captured: list[tuple[str, list[np.ndarray]]] = []

    def capture(
        operation: str,
        targets: list[np.ndarray],
        *,
        motion_profile=None,
    ) -> None:
        copied = [np.array(target, copy=True) for target in targets]
        captured.append((operation, copied))
        io.current = copied[-1]

    monkeypatch.setattr(io, "_move_sequence", capture)
    return io, config, captured


def test_twist_uses_fast_motion_profile_without_changing_other_sequences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    io, config, _ = _capture_only_motion_io(tmp_path, monkeypatch)
    profiles: list[tuple[str, object]] = []

    def capture(
        operation: str,
        targets: list[np.ndarray],
        *,
        motion_profile=None,
    ) -> None:
        profiles.append((operation, motion_profile))
        io.current = np.array(targets[-1], copy=True)

    monkeypatch.setattr(io, "_move_sequence", capture)
    context = SimpleNamespace(
        phase=runner.TwistLoopPhase.TWIST,
        twist_step_count=0,
        press_retry_count=0,
    )
    io.execute_twist_step(
        angle_rad=config["loop_config"].twist_step_angle_rad,
        context=context,
    )
    io.press_engage(context=context)

    assert profiles[0] == ("twist_step_1", config["twist_motion"])
    assert profiles[1] == ("press_engage_0", None)
    assert config["twist_motion"] == {
        "velocity_scale": 0.05,
        "acceleration_scale": 0.05,
        "max_cartesian_speed_m_s": 0.015,
    }
    assert config["motion"]["velocity_scale"] == 0.02
    assert config["motion"]["acceleration_scale"] == 0.02
    assert config["motion"]["max_cartesian_speed_m_s"] == 0.008


def test_sequence_command_applies_only_explicit_speed_override(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    config = runner._validate_config(
        runner._read_json(config_path), config_path=config_path, execute=False
    )
    input_path = tmp_path / "poses.json"
    output_path = tmp_path / "motion.json"

    default_command = runner._sequence_command(
        config, input_path, output_path, execute=True
    )
    twist_command = runner._sequence_command(
        config,
        input_path,
        output_path,
        execute=True,
        motion_profile=config["twist_motion"],
    )

    def option(command: list[str], name: str) -> float:
        return float(command[command.index(name) + 1])

    assert option(default_command, "--velocity-scale") == 0.02
    assert option(default_command, "--acceleration-scale") == 0.02
    assert option(default_command, "--max-cartesian-speed-m-s") == 0.008
    assert option(twist_command, "--velocity-scale") == 0.05
    assert option(twist_command, "--acceleration-scale") == 0.05
    assert option(twist_command, "--max-cartesian-speed-m-s") == 0.015


def test_twist_downfeed_accumulates_and_unload_retracts_full_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    io, config, captured = _capture_only_motion_io(tmp_path, monkeypatch)
    axis = np.asarray(config["frame"].bottle_axis_base, dtype=float)
    contact = np.array(config["frame"].T_base_cap_tcp, copy=True)
    initial_press_m = config["initial_press_m"]
    angle_rad = config["loop_config"].twist_step_angle_rad
    downfeed_step_m = (
        config["twist_downfeed_m_per_revolution"]
        * abs(angle_rad)
        / (2.0 * np.pi)
    )

    assert config["twist_downfeed_m_per_revolution"] == pytest.approx(0.002)
    io.current = contact.copy()
    io.current[:3, 3] -= initial_press_m * axis
    io.segment_press_depth_m = initial_press_m
    for step in range(9):
        before = io.current.copy()
        io.execute_twist_step(
            angle_rad=angle_rad,
            context=SimpleNamespace(
                phase=runner.TwistLoopPhase.TWIST,
                twist_step_count=step,
            ),
        )
        axial_delta_m = float(np.dot(io.current[:3, 3] - before[:3, 3], axis))
        assert axial_delta_m == pytest.approx(-downfeed_step_m)

    assert io.segment_press_depth_m == pytest.approx(
        initial_press_m + 9 * downfeed_step_m
    )
    preview = runner._resolved_transforms(config)
    preview_delta_m = float(
        np.dot(
            preview["first_formal_twist"][:3, 3] - preview["first_press"][:3, 3],
            axis,
        )
    )
    assert preview_delta_m == pytest.approx(-downfeed_step_m)

    monkeypatch.setattr(io, "_gripper", lambda *args, **kwargs: {})
    io.unload(context=SimpleNamespace())

    np.testing.assert_allclose(io.current[:3, 3], contact[:3, 3], atol=1e-12)
    assert io.segment_press_depth_m == 0.0
    assert captured[-1][0] == "unload_axial_retract"


def test_second_revolution_switches_from_downfeed_to_helical_upfeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    io, config, _ = _capture_only_motion_io(tmp_path, monkeypatch)
    axis = np.asarray(config["frame"].bottle_axis_base, dtype=float)
    angle_rad = config["loop_config"].twist_step_angle_rad
    before = io.current.copy()

    io.execute_twist_step(
        angle_rad=angle_rad,
        context=SimpleNamespace(
            phase=runner.TwistLoopPhase.TWIST,
            twist_step_count=36,
            total_signed_twist_rad=2.0 * np.pi,
        ),
    )

    axial_delta_m = float(np.dot(io.current[:3, 3] - before[:3, 3], axis))
    assert axial_delta_m == pytest.approx(
        config["thread_pitch_m_per_revolution"]
        * abs(angle_rad)
        / (2.0 * np.pi)
    )
    assert io.total_thread_upfeed_m == pytest.approx(axial_delta_m)
    assert io.segment_press_depth_m == 0.0


def test_twist_batch_is_one_sequence_with_microstep_waypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    io, config, captured = _capture_only_motion_io(tmp_path, monkeypatch)
    angle_rad = config["loop_config"].twist_step_angle_rad

    io.execute_twist_batch(
        angle_rad=angle_rad,
        step_count=6,
        context=SimpleNamespace(
            phase=runner.TwistLoopPhase.TWIST,
            twist_step_count=0,
            total_signed_twist_rad=0.0,
        ),
    )

    assert len(captured) == 1
    operation, targets = captured[0]
    assert operation == "twist_batch_1_6"
    assert len(targets) == 6


def test_release_verify_uses_fresh_stationary_proxy_without_forced_test_lift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    io, config, captured = _capture_only_motion_io(tmp_path, monkeypatch)
    axis = np.asarray(config["frame"].bottle_axis_base, dtype=float)
    contact = np.array(config["frame"].T_base_cap_tcp, copy=True)
    angle_rad = config["loop_config"].twist_step_angle_rad
    downfeed_step_m = (
        config["twist_downfeed_m_per_revolution"]
        * abs(angle_rad)
        / (2.0 * np.pi)
    )
    full_depth_m = config["initial_press_m"] + 9 * downfeed_step_m
    io.current = contact.copy()
    io.current[:3, 3] -= full_depth_m * axis
    io.segment_press_depth_m = full_depth_m
    before = object()
    after = SimpleNamespace(
        left=SimpleNamespace(wrench_task=np.tile([0, 0, 0.10, 0, 0, 0], (5, 1))),
        right=SimpleNamespace(wrench_task=np.tile([0, 0, 0.08, 0, 0, 0], (5, 1))),
    )
    io.last_observation = before
    io.contact_presence_threshold_sensor_units = 0.01
    io.presence_config = SimpleNamespace(
        release_resistance_peak_fraction=0.35,
        release_min_peak_fz=0.002,
        press_retain_fz=0.001,
    )
    io.upfeed_global_peak_resistance_proxy_sensor_units = 0.6397032682700269
    monkeypatch.setattr(io, "collect_force_window", lambda **kwargs: after)
    monkeypatch.setattr(
        io,
        "_window_proxy_metrics",
        lambda observation: {
            "combined_press_aligned_raw_fz_proxy_sensor_units": (
                0.08919650595643053
                if observation is before
                else 0.19887760515649816
            )
        },
    )

    assert io.release_verify(context=SimpleNamespace()) is True

    np.testing.assert_allclose(
        io.current[:3, 3], contact[:3, 3] - full_depth_m * axis, atol=1e-12
    )
    assert io.segment_press_depth_m == 0.0
    assert captured == []
    event = io.action_log.payload["events"][-1]
    assert event["upfeed_global_peak_resistance_proxy_sensor_units"] == pytest.approx(
        0.6397032682700269
    )
    assert event["maximum_release_proxy_sensor_units"] == pytest.approx(
        0.35 * 0.6397032682700269
    )


def test_release_verify_rejects_stationary_proxy_above_global_peak_fraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    io, _, captured = _capture_only_motion_io(tmp_path, monkeypatch)
    before = object()
    after = SimpleNamespace(
        left=SimpleNamespace(wrench_task=np.tile([0, 0, 0.20, 0, 0, 0], (5, 1))),
        right=SimpleNamespace(wrench_task=np.tile([0, 0, 0.10, 0, 0, 0], (5, 1))),
    )
    io.last_observation = before
    io.contact_presence_threshold_sensor_units = 0.01
    io.presence_config = SimpleNamespace(
        release_resistance_peak_fraction=0.35,
        release_min_peak_fz=0.002,
        press_retain_fz=0.001,
    )
    io.upfeed_global_peak_resistance_proxy_sensor_units = 0.6397032682700269
    monkeypatch.setattr(io, "collect_force_window", lambda **kwargs: after)
    monkeypatch.setattr(
        io,
        "_window_proxy_metrics",
        lambda observation: {
            "combined_press_aligned_raw_fz_proxy_sensor_units": (
                0.089 if observation is before else 0.23
            )
        },
    )

    assert io.release_verify(context=SimpleNamespace()) is False
    assert captured == []
