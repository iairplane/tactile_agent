from __future__ import annotations

import importlib.util
import json
import shlex
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from demo2gripper.d435_live import D435Frame, D435Intrinsics
from demo2gripper.hande_modbus_cli import Feedback
from demo2gripper.pose_observation import PoseObservation
from demo2gripper.test_tube_closed_loop import (
    HELD_POSE_CAPABILITIES,
    SOCKET_POSE_CAPABILITIES,
    SOURCE_POSE_CAPABILITIES,
    ActionKind,
    ClosedLoopAction,
    GripperFeedback,
    PoseBackendDescriptor,
    TactileFeedback,
)
from demo2gripper.test_tube_live_io import (
    ONLINE_SCENE_FACTORY,
    SESSION_BOOTSTRAP_TACTILE_FACTORY,
    ArmState,
    HandEModbusRuntime,
    LiveTestTubeIO,
    QualifiedBasePose,
    QualifiedCameraPose,
    RosMoveItServoRuntime,
    SceneEvidence,
    SceneObjectMask,
    _demonstration_contract_blockers,
    _reduced_collision_robot_config,
    _require_pause_response,
    _require_twist_mode_response,
    _servo_status_fault,
    _tcp_calibration_blockers,
    _tcp_calibration_uncertainty,
    _tcp_servo_twist_at_link8,
    assess_live_config,
    preflight_live_config,
    provisional_camera_extrinsic_acknowledgement,
    reduced_collision_model_acknowledgement,
)
from demo2gripper.test_tube_preflight import TubeState


def transform(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    value = np.eye(4)
    value[:3, 3] = [x, y, z]
    return value


def descriptor(name: str, capabilities: frozenset) -> PoseBackendDescriptor:
    return PoseBackendDescriptor(
        name=name,
        available=True,
        runtime_qualified=True,
        qualification_id=f"qualified-{name}",
        capabilities=capabilities,
    )


def hande_feedback(*, fault_code: int, position_count: int = 0) -> Feedback:
    return Feedback(
        activation_bit=1,
        go_to_bit=0,
        activation_status=3,
        object_status=3,
        fault_code=fault_code,
        requested_count=0,
        position_count=position_count,
        current=0,
    )


class SequencedHandETransport:
    def __init__(self, feedback: list[Feedback]) -> None:
        self._feedback = iter(feedback)
        self.read_calls = 0

    def read_feedback(self) -> Feedback:
        self.read_calls += 1
        return next(self._feedback)


def test_hande_feedback_recovers_one_transient_communication_watchdog() -> None:
    transport = SequencedHandETransport(
        [
            hande_feedback(fault_code=0x09, position_count=20),
            hande_feedback(fault_code=0, position_count=40),
        ]
    )
    runtime = HandEModbusRuntime(transport)  # type: ignore[arg-type]

    feedback = runtime.feedback()

    assert transport.read_calls == 2
    assert feedback.fault_free is True
    assert feedback.width_m == pytest.approx((50.0 - 40 * 0.1953125) / 1000.0)


def test_hande_feedback_does_not_retry_other_faults() -> None:
    transport = SequencedHandETransport(
        [
            hande_feedback(fault_code=0x07),
            hande_feedback(fault_code=0),
        ]
    )
    runtime = HandEModbusRuntime(transport)  # type: ignore[arg-type]

    feedback = runtime.feedback()

    assert transport.read_calls == 1
    assert feedback.fault_free is False


def test_hande_feedback_keeps_persistent_watchdog_fail_closed_and_bounded() -> None:
    transport = SequencedHandETransport(
        [
            hande_feedback(fault_code=0x09),
            hande_feedback(fault_code=0x09),
            hande_feedback(fault_code=0),
        ]
    )
    runtime = HandEModbusRuntime(transport)  # type: ignore[arg-type]

    feedback = runtime.feedback()

    assert transport.read_calls == 2
    assert feedback.fault_free is False


class FakeCamera:
    def __init__(self) -> None:
        self.sequence = 0
        self.closed = False

    def read(self) -> D435Frame:
        self.sequence += 1
        intrinsics = D435Intrinsics(4, 3, 100.0, 100.0, 2.0, 1.5, "none", ())
        return D435Frame(
            sequence=self.sequence,
            serial="fake-d435",
            rgb=np.zeros((3, 4, 3), dtype=np.uint8),
            depth_raw=np.ones((3, 4), dtype=np.uint16),
            depth_m=np.ones((3, 4), dtype=float),
            intrinsics=intrinsics,
            color_frame_number=self.sequence,
            depth_frame_number=self.sequence,
            color_sensor_timestamp_s=float(self.sequence),
            depth_sensor_timestamp_s=float(self.sequence),
            color_sdk_timestamp_s=float(self.sequence),
            depth_sdk_timestamp_s=float(self.sequence),
            sdk_timestamp_domain="fake",
            host_received_monotonic_s=float(self.sequence),
        )

    def close(self) -> None:
        self.closed = True


class FakeTactileSource:
    def __init__(self) -> None:
        self.sequence = 0
        self.closed = False

    def collect_classifier_windows(self, *_: object, **__: object) -> object:
        self.sequence += 1
        return SimpleNamespace(collected_monotonic_s=float(self.sequence))

    def close(self) -> None:
        self.closed = True


class FakeScene:
    def __init__(
        self,
        *,
        source_mask: bool = False,
        source_position_std_m: float = 0.001,
    ) -> None:
        self.source_mask = source_mask
        self.source_position_std_m = source_position_std_m
        self.held_calls = 0
        self.grasp_relations: list[tuple[np.ndarray, np.ndarray]] = []
        self.support_history_resets = 0
        self.closed = False

    def observe(
        self,
        *,
        frame: D435Frame,
        state: TubeState,
        T_base_tcp: np.ndarray,
    ) -> SceneEvidence:
        del T_base_tcp
        del state
        tube = None
        if self.source_mask:
            tube = SceneObjectMask(
                object_id="tube",
                mask=np.ones(frame.depth_m.shape, dtype=bool),
                mask_id="tube-mask",
                mask_iou=0.9,
                tracking_id="tube-track",
            )
        return SceneEvidence(tube=tube, socket=None)

    def qualify_pose(self, **_: object) -> QualifiedCameraPose:
        return QualifiedCameraPose(
            T_camera_object=transform(0.2, 0.0, 0.4),
            position_std_m=self.source_position_std_m,
            orientation_std_deg=1.0,
            axis_sign_source="open_end_detector",
            cylindrical_roll_canonicalized=True,
        )

    def held_tube_pose(self, **_: object) -> QualifiedBasePose:
        self.held_calls += 1
        return QualifiedBasePose(
            T_base_object=transform(0.5, 0.0, 0.2),
            position_std_m=0.002,
            orientation_std_deg=2.0,
            tracking_id="tube-track",
            evidence_id=f"held-{self.held_calls}",
            axis_sign_source="temporal_continuity",
        )

    def set_grasp_relation(
        self,
        T_tube_tcp_actual: np.ndarray,
        *,
        T_base_tube_at_contact: np.ndarray,
    ) -> None:
        self.grasp_relations.append((T_tube_tcp_actual.copy(), T_base_tube_at_contact.copy()))

    def reset_support_history_for_release(self) -> None:
        self.support_history_resets += 1

    def close(self) -> None:
        self.closed = True


class FakeTactile:
    def __init__(
        self,
        *,
        stable: bool = False,
        valid: bool = True,
        observed_at_s: float | None = None,
    ) -> None:
        self.stable = stable
        self.valid = valid
        self.observed_at_s = observed_at_s
        self.closed = False

    def evaluate(self, *, now_s: float, **_: object) -> TactileFeedback:
        return TactileFeedback(
            valid=self.valid,
            observed_at_s=(now_s if self.observed_at_s is None else self.observed_at_s),
            contact_present=self.stable,
            bilateral_contact=self.stable,
            rotation_stopped=self.stable,
        )

    def close(self) -> None:
        self.closed = True


class ProgressRecordingTactile(FakeTactile):
    def __init__(self) -> None:
        super().__init__()
        self.measured_progress_m: float | None = None

    def evaluate(
        self,
        *,
        measured_insertion_progress_m: float | None,
        now_s: float,
        **kwargs: object,
    ) -> TactileFeedback:
        self.measured_progress_m = measured_insertion_progress_m
        return super().evaluate(now_s=now_s, **kwargs)


class FakeArm:
    def __init__(self, *, reject_servo: bool = False) -> None:
        self.T = transform(0.5, 0.0, 0.3)
        self.reject_servo = reject_servo
        self.coarse_calls = 0
        self.servo_calls = 0
        self.servo_arguments: list[dict[str, object]] = []
        self.closed = False

    def state(self) -> ArmState:
        return ArmState(self.T, True, True)

    def move_coarse(self, **_: object) -> dict[str, object]:
        self.coarse_calls += 1
        return {"accepted": True}

    def servo_target(self, **arguments: object) -> dict[str, object]:
        self.servo_calls += 1
        self.servo_arguments.append(arguments)
        if self.reject_servo:
            raise RuntimeError("moveit_servo inactive")
        return {"accepted": True}

    def stop(self, **_: object) -> dict[str, object]:
        return {"held": True}

    def close(self) -> None:
        self.closed = True


class FakeGripper:
    def __init__(self) -> None:
        self.width = 0.04
        self.moves: list[float] = []
        self.closed = False

    def feedback(self) -> GripperFeedback:
        return GripperFeedback(self.width, False, True)

    def move_width(self, *, width_m: float, **_: object) -> dict[str, object]:
        self.moves.append(width_m)
        self.width = width_m
        return {"accepted": True}

    def stop(self, **_: object) -> dict[str, object]:
        return {"held": True}

    def close(self) -> None:
        self.closed = True


class LaggingFakeGripper(FakeGripper):
    def move_width(self, *, width_m: float, **_: object) -> dict[str, object]:
        self.moves.append(width_m)
        return {"accepted": True}


class MovingFakeGripper(FakeGripper):
    def __init__(self) -> None:
        super().__init__()
        self.moving = True

    def feedback(self) -> GripperFeedback:
        return GripperFeedback(self.width, self.moving, True)


class FakePoseProvider:
    def __init__(self) -> None:
        self.calls = 0

    def estimate(self, **_: object) -> PoseObservation:
        self.calls += 1
        return PoseObservation.from_dict(
            {
                "schema": "demo2gripper.pose_observation.v1",
                "request_id": "fake-request",
                "capture": {
                    "sequence": self.calls,
                    "frame_id": "fake",
                    "timestamp_s": float(self.calls),
                    "clock_domain": "client_monotonic",
                },
                "accepted": True,
                "pose": {
                    "T_camera_object": transform(0.2, 0.0, 0.4).tolist(),
                    "size_m": [0.02, 0.02, 0.15],
                },
                "coordinate_frame": "opencv_color_optical",
                "produced_at_unix_s": 1.0,
                "mask_quality": {
                    "source": "fake",
                    "source_score": 0.9,
                    "pixel_count": 12,
                    "area_fraction": 1.0,
                    "bbox_xyxy": [0, 0, 4, 3],
                    "gate_passed": True,
                    "reject_reasons": [],
                },
                "foreground_depth": {
                    "policy": "fake",
                    "valid_mask_pixels": 12,
                    "valid_mask_fraction": 1.0,
                    "cluster_count": 1,
                    "selected_cluster_index": 0,
                    "selected_cluster_pixels": 12,
                    "selected_cluster_fraction": 1.0,
                    "selected_median_m": 0.4,
                    "selected_span_m": 0.01,
                    "outside_ring_valid_pixels": 1,
                    "outside_ring_median_m": 0.8,
                    "mask_near_background_fraction": 0.0,
                    "clusters": [
                        {
                            "index": 0,
                            "pixels": 12,
                            "fraction": 1.0,
                            "min_m": 0.39,
                            "median_m": 0.4,
                            "max_m": 0.41,
                            "span_m": 0.02,
                        }
                    ],
                    "gate_passed": True,
                    "reject_reasons": [],
                },
                "pose_quality": {
                    "model_confidence": None,
                    "temporal_track_confirmed": False,
                    "axis_sign_status": "unresolved",
                    "axis_sign_source": None,
                    "cylindrical_roll_status": "unresolved",
                    "cylindrical_roll_source": None,
                    "task_qualified": False,
                },
                "latency": {"preprocess_s": 0.0, "inference_s": 0.01, "total_server_s": 0.01},
                "backend": {"name": "f2g_pose", "pose_only": True},
                "reject_reasons": [],
            }
        )


def make_io(
    *,
    scene: FakeScene,
    tactile: FakeTactile,
    arm: FakeArm | None = None,
    gripper: FakeGripper | None = None,
    T_tube_tcp_demo: np.ndarray | None = None,
    monotonic=lambda: 1.0,
):
    aggregate = descriptor(
        "external",
        frozenset(SOURCE_POSE_CAPABILITIES | HELD_POSE_CAPABILITIES | SOCKET_POSE_CAPABILITIES),
    )
    return LiveTestTubeIO(
        descriptor=aggregate,
        source_descriptor=descriptor("f2g_pose", SOURCE_POSE_CAPABILITIES),
        held_descriptor=descriptor("held", HELD_POSE_CAPABILITIES),
        socket_descriptor=descriptor("socket", SOCKET_POSE_CAPABILITIES),
        camera=FakeCamera(),
        tactile_source=FakeTactileSource(),
        scene=scene,
        tactile=tactile,
        arm=arm or FakeArm(),
        gripper=gripper or FakeGripper(),
        tube_pose_provider=FakePoseProvider(),
        T_base_camera=np.eye(4),
        T_tube_tcp_demo=(transform(0.0, 0.0, 0.1) if T_tube_tcp_demo is None else T_tube_tcp_demo),
        T_base_socket=transform(0.7, 0.0, 0.1),
        camera_position_std_m=0.001,
        camera_orientation_std_deg=1.0,
        tcp_position_std_m=0.001,
        tcp_orientation_std_deg=1.0,
        demo_relation_position_std_m=0.001,
        demo_relation_orientation_std_deg=1.0,
        held_position_std_limit_m=0.005,
        held_orientation_std_limit_deg=5.0,
        socket_position_std_m=0.001,
        socket_orientation_std_deg=1.0,
        monotonic=monotonic,
    )


def action(
    kind: ActionKind,
    parameters: dict[str, object],
    *,
    state: TubeState = TubeState.APPROACH_TUBE,
) -> ClosedLoopAction:
    return ClosedLoopAction("test-action", kind, state, parameters, "test")


def test_source_f2g_and_static_socket_have_distinct_producers() -> None:
    io = make_io(scene=FakeScene(source_mask=True), tactile=FakeTactile())
    observation = io.observe(state=TubeState.OBSERVE)
    assert observation.tube_pose is not None
    assert observation.tube_pose.backend.name == "f2g_pose"
    assert observation.tube_pose.requires_rgbd_quality_gates is True
    assert observation.socket_pose is not None
    assert observation.socket_pose.backend.name == "socket"
    assert observation.socket_pose.requires_rgbd_quality_gates is False


def test_observe_uses_socket_camera_burst_only_during_initial_observe() -> None:
    class BurstScene(FakeScene):
        def __init__(self) -> None:
            super().__init__()
            self.burst_calls = 0
            self.last_frame_sequence = 0

        def observe_burst(self, *, first_frame, read_frame, state, T_base_tcp):
            self.burst_calls += 1
            frame = first_frame
            for _ in range(3):
                frame = read_frame()
            self.last_frame_sequence = frame.sequence
            return frame, self.observe(frame=frame, state=state, T_base_tcp=T_base_tcp)

    scene = BurstScene()
    io = make_io(scene=scene, tactile=FakeTactile())

    first = io.observe(state=TubeState.OBSERVE)
    assert first.observation_id.endswith("-00000004")
    assert scene.burst_calls == 1
    assert scene.last_frame_sequence == 4

    second = io.observe(state=TubeState.APPROACH_TUBE)
    assert second.observation_id.endswith("-00000005")
    assert scene.burst_calls == 1


def test_post_pivot_pose_freezes_then_uses_robot_kinematics() -> None:
    scene = FakeScene()
    io = make_io(scene=scene, tactile=FakeTactile(stable=True))
    reclamp = io.observe(state=TubeState.RECLAMP)
    assert reclamp.tube_pose is not None
    assert scene.held_calls == 1
    approach = io.observe(state=TubeState.APPROACH_SOCKET)
    assert approach.tube_pose is not None
    assert approach.tube_pose.requires_rgbd_quality_gates is False
    assert scene.held_calls == 1


def test_level_state_uses_realized_rigid_grasp_relation_and_live_fk() -> None:
    scene = FakeScene()
    arm = FakeArm()
    io = make_io(scene=scene, tactile=FakeTactile(stable=True), arm=arm)
    relation = transform(0.01, -0.02, 0.08)
    io._T_tube_tcp_grasp_actual = relation  # type: ignore[attr-defined]
    io._grasp_relation_position_std_m = 0.001  # type: ignore[attr-defined]
    io._grasp_relation_orientation_std_deg = 1.0  # type: ignore[attr-defined]

    observation = io.observe(state=TubeState.LEVEL_HELD_TUBE)

    assert observation.tube_pose is not None
    np.testing.assert_allclose(
        observation.tube_pose.T_base_object,
        arm.T @ np.linalg.inv(relation),
    )
    assert observation.tube_pose.requires_rgbd_quality_gates is False
    assert scene.held_calls == 0


def test_servo_failure_never_falls_back_to_blocking_moveit() -> None:
    arm = FakeArm(reject_servo=True)
    io = make_io(scene=FakeScene(), tactile=FakeTactile(), arm=arm)
    with pytest.raises(RuntimeError, match="moveit_servo inactive"):
        io.dispatch(action(ActionKind.SERVO_TCP, {"T_base_tcp_command": np.eye(4)}))
    assert arm.servo_calls == 1
    assert arm.coarse_calls == 0


def test_pause_arm_zeros_servo_without_commanding_gripper() -> None:
    arm = FakeArm()
    io = make_io(scene=FakeScene(), tactile=FakeTactile(), arm=arm)
    receipt = io.dispatch(action(ActionKind.PAUSE_ARM, {"reacquire": True}))
    assert receipt["held"] is True
    assert arm.coarse_calls == 0
    assert arm.servo_calls == 0
    assert io._gripper.moves == []  # type: ignore[attr-defined]


def test_servo_status_other_than_no_warning_fails_closed() -> None:
    assert _servo_status_fault(0, "No warning") is None
    assert _servo_status_fault(4, "decelerating for collision") == (
        "moveit_servo_status:4:decelerating for collision"
    )
    assert _servo_status_fault(5, "halt for collision") == (
        "moveit_servo_status:5:halt for collision"
    )


def test_servo_twist_mode_must_be_affirmatively_accepted() -> None:
    _require_twist_mode_response(SimpleNamespace(success=True))
    with pytest.raises(RuntimeError, match="rejected TWIST"):
        _require_twist_mode_response(SimpleNamespace(success=False))
    with pytest.raises(RuntimeError, match="rejected TWIST"):
        _require_twist_mode_response(None)


def test_servo_pause_state_must_be_affirmatively_accepted() -> None:
    _require_pause_response(SimpleNamespace(success=True), paused=True)
    _require_pause_response(SimpleNamespace(success=True), paused=False)
    with pytest.raises(RuntimeError, match="rejected pause"):
        _require_pause_response(SimpleNamespace(success=False), paused=True)
    with pytest.raises(RuntimeError, match="rejected resume"):
        _require_pause_response(None, paused=False)


def _servo_twist(current: np.ndarray, target: np.ndarray, **overrides: object) -> np.ndarray:
    values = {
        "current": current,
        "target": target,
        "T_link8_tcp": transform(z=0.19),
        "horizon_s": 0.1,
        "max_linear_m_s": 0.015,
        "max_angular_rad_s": 0.15,
        "previous_twist": np.zeros(6),
        "dt_s": 0.02,
        "max_linear_acceleration_m_s2": 0.05,
        "max_angular_acceleration_rad_s2": 0.3,
    }
    values.update(overrides)
    return _tcp_servo_twist_at_link8(**values)


def test_tcp_rotation_keeps_tcp_fixed_by_compensating_link8_velocity() -> None:
    current = transform(0.5, 0.2, 0.3)
    target = current.copy()
    target[:3, :3] = Rotation.from_euler("y", 10.0, degrees=True).as_matrix()
    twist = _servo_twist(current, target, dt_s=10.0)

    assert twist[0] < 0.0
    assert twist[4] > 0.0
    np.testing.assert_allclose(
        twist[:3] + np.cross(twist[3:], [0.0, 0.0, 0.19]), np.zeros(3), atol=1e-12
    )
    assert np.linalg.norm(twist[:3]) <= 0.015 + 1e-12
    assert np.linalg.norm(twist[3:]) <= 0.15 + 1e-12


def test_tcp_origin_conversion_handles_rotated_tool_mount() -> None:
    mount = transform(z=0.19)
    mount[:3, :3] = Rotation.from_euler("z", 45.0, degrees=True).as_matrix()
    current = transform()
    current[:3, :3] = Rotation.from_euler("x", 90.0, degrees=True).as_matrix() @ mount[:3, :3]
    target = current.copy()
    target[:3, :3] = Rotation.from_rotvec([0.1, 0.0, 0.0]).as_matrix() @ current[:3, :3]

    twist = _servo_twist(current, target, T_link8_tcp=mount)

    np.testing.assert_allclose(
        twist[:3] + np.cross(twist[3:], [0.0, -0.19, 0.0]), np.zeros(3), atol=1e-12
    )


def test_tcp_servo_ramps_both_origins_and_can_slow_to_zero() -> None:
    current = transform()
    target = transform(0.1, 0.1, 0.0)
    target[:3, :3] = Rotation.from_euler("y", 30.0, degrees=True).as_matrix()
    previous = np.zeros(6)
    for _ in range(100):
        twist = _servo_twist(current, target, previous_twist=previous)
        delta = twist - previous
        tcp = twist[:3] + np.cross(twist[3:], [0.0, 0.0, 0.19])
        delta_tcp = delta[:3] + np.cross(delta[3:], [0.0, 0.0, 0.19])
        assert np.linalg.norm(delta[:3]) <= 0.05 * 0.02 + 1e-12
        assert np.linalg.norm(delta_tcp) <= 0.05 * 0.02 + 1e-12
        assert np.linalg.norm(delta[3:]) <= 0.3 * 0.02 + 1e-12
        assert np.linalg.norm(twist[:3]) <= 0.015 + 1e-12
        assert np.linalg.norm(tcp) <= 0.015 + 1e-12
        assert np.linalg.norm(twist[3:]) <= 0.15 + 1e-12
        previous = twist
    np.testing.assert_array_equal(_servo_twist(current, current), np.zeros(6))
    tiny = _servo_twist(current, transform(x=1e-8))
    assert 0.0 < np.linalg.norm(tiny[:3]) < 1e-6


def test_zero_twist_immediately_resets_velocity_ramp() -> None:
    runtime = object.__new__(RosMoveItServoRuntime)
    runtime._twist_lock = threading.Lock()
    runtime._last_twist = np.ones(6)
    runtime._last_twist_time_s = 0.0
    runtime._base_frame = "fr3_link0"
    runtime._node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: 1))
    )
    runtime._TwistStamped = lambda: SimpleNamespace(
        header=SimpleNamespace(),
        twist=SimpleNamespace(
            linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            angular=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        ),
    )
    messages = []
    runtime._publisher = SimpleNamespace(publish=messages.append)
    runtime._publish_zero()

    np.testing.assert_array_equal(runtime._last_twist, np.zeros(6))
    assert messages[0].twist.linear.x == 0.0
    assert messages[0].twist.angular.x == 0.0


def test_servo_startup_waits_for_fresh_joint_state_and_tcp_tf() -> None:
    runtime = object.__new__(RosMoveItServoRuntime)
    runtime._startup_state_timeout_s = 0.1  # type: ignore[attr-defined]
    runtime._joint_state_max_age_s = 0.25  # type: ignore[attr-defined]
    runtime._last_joint_state_s = time.monotonic()  # type: ignore[attr-defined]
    lookups: list[bool] = []
    runtime._lookup_tcp = lambda: lookups.append(True) or np.eye(4)  # type: ignore[method-assign]

    runtime._wait_for_initial_state()

    assert lookups == [True]


def _joint_refresh_runtime() -> RosMoveItServoRuntime:
    runtime = object.__new__(RosMoveItServoRuntime)
    runtime._joint_state_condition = threading.Condition()
    runtime._last_joint_state_s = time.monotonic() - 1.0
    runtime._last_joint_stamp_ns = None
    runtime._joint_state_max_age_s = 0.25
    runtime._joint_state_refresh_timeout_s = 0.1
    runtime._target_lock = threading.Lock()
    runtime._servo_target = np.eye(4)
    runtime._position_tolerance_m = 0.0015
    runtime._active_position_tolerance_m = 0.0015
    runtime._node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000))
    )
    runtime._servo_status_lock = threading.Lock()
    runtime._servo_status_latched_fault = None
    runtime._last_servo_error = None
    runtime._lookup_tcp = lambda: np.eye(4)
    runtime._publish_zero = lambda: None
    return runtime


def test_insertion_goal_tolerance_drives_servo_and_completion_then_restores_default() -> None:
    runtime = _joint_refresh_runtime()
    runtime._last_joint_state_s = time.monotonic()
    runtime._last_joint_velocity_norm = 0.0
    runtime._rotation_tolerance_rad = np.deg2rad(1.5)
    runtime._servo_paused = False
    published: list[str] = []
    runtime._publish_zero = lambda: published.append("zero")
    runtime._publish_twist = lambda current, target: published.append("move")

    normal = runtime.servo_target(target=transform(x=0.0015), action_id="normal")
    assert normal["position_tolerance_m"] == pytest.approx(0.0015)
    runtime._servo_cycle()
    assert published.pop() == "zero"
    assert runtime.state().motion_complete

    inserted = runtime.servo_target(
        target=transform(x=0.0015), action_id="insert", position_tolerance_m=0.0005
    )
    assert inserted["position_tolerance_m"] == pytest.approx(0.0005)
    runtime._servo_cycle()
    assert published.pop() == "move"
    assert not runtime.state().motion_complete
    runtime._lookup_tcp = lambda: transform(x=0.0011)
    runtime._servo_cycle()
    assert published.pop() == "zero"
    assert runtime.state().motion_complete

    runtime.servo_target(target=transform(x=0.0025), action_id="next-normal")
    runtime._servo_cycle()
    assert published.pop() == "zero"
    assert runtime.state().motion_complete
    assert runtime._active_position_tolerance_m == pytest.approx(0.0015)


@pytest.mark.parametrize("tolerance", [0.0, -0.001, float("nan"), float("inf")])
def test_invalid_goal_tolerance_does_not_replace_active_target(tolerance: float) -> None:
    runtime = _joint_refresh_runtime()
    runtime._servo_paused = False
    runtime.servo_target(target=transform(x=0.01), action_id="old")
    with pytest.raises(ValueError, match="position_tolerance_m"):
        runtime.servo_target(
            target=transform(x=0.02), action_id="bad", position_tolerance_m=tolerance
        )
    np.testing.assert_allclose(runtime._servo_target, transform(x=0.01))
    assert runtime._active_position_tolerance_m == pytest.approx(0.0015)


def _joint_message(stamp_s: float) -> SimpleNamespace:
    stamp_ns = int(stamp_s * 1e9)
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(
            sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000
        )),
        position=np.zeros(7), velocity=np.zeros(7),
    )


def test_stale_joint_callback_waits_without_motion_until_fresh_source_arrives() -> None:
    runtime = _joint_refresh_runtime()
    zeros = []
    runtime._publish_zero = lambda: zeros.append(True)

    def deliver() -> None:
        time.sleep(0.01)
        runtime._receive_joint_state(_joint_message(10.0))

    thread = threading.Thread(target=deliver)
    thread.start()
    state = runtime.state()
    thread.join()
    assert state.fault is None
    assert runtime._servo_target is None
    assert zeros == [True]


def test_joint_refresh_keeps_fault_if_no_new_live_state_arrives() -> None:
    runtime = _joint_refresh_runtime()
    runtime._joint_state_refresh_timeout_s = 0.001
    state = runtime.state()
    assert state.fault == "joint_state_stale"
    assert runtime._servo_target is None


def test_joint_callback_rejects_delayed_and_replayed_source_stamps() -> None:
    runtime = _joint_refresh_runtime()
    old_receipt = runtime._last_joint_state_s
    assert runtime._receive_joint_state(_joint_message(9.0)) is False
    assert runtime._last_joint_state_s == old_receipt
    assert runtime._receive_joint_state(_joint_message(9.9)) is True
    received = runtime._last_joint_state_s
    assert time.monotonic() - received >= 0.1
    assert runtime._receive_joint_state(_joint_message(9.9)) is False
    assert runtime._receive_joint_state(_joint_message(9.85)) is False
    assert runtime._last_joint_state_s == received


def test_servo_startup_fails_before_observation_when_joint_state_is_missing() -> None:
    runtime = object.__new__(RosMoveItServoRuntime)
    runtime._startup_state_timeout_s = 0.001  # type: ignore[attr-defined]
    runtime._joint_state_max_age_s = 0.25  # type: ignore[attr-defined]
    runtime._last_joint_state_s = None  # type: ignore[attr-defined]
    runtime._lookup_tcp = lambda: np.eye(4)  # type: ignore[method-assign]

    with pytest.raises(
        RuntimeError,
        match="ROS arm state did not become ready before live observation: joint_state_unavailable",
    ):
        runtime._wait_for_initial_state()


def test_coarse_setup_failure_always_resumes_servo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = object.__new__(RosMoveItServoRuntime)
    runtime._target_lock = threading.Lock()  # type: ignore[attr-defined]
    runtime._servo_target = np.eye(4)  # type: ignore[attr-defined]
    runtime._servo_horizon_s = 0.1  # type: ignore[attr-defined]
    runtime._servo_period_s = 0.02  # type: ignore[attr-defined]
    runtime._T_link8_tcp = np.eye(4)  # type: ignore[attr-defined]
    runtime._output_dir = tmp_path  # type: ignore[attr-defined]
    runtime._moveit = {"pose_goal_script": None}  # type: ignore[attr-defined]
    pause_states: list[bool] = []
    selected: list[bool] = []
    runtime._publish_zero = lambda: None  # type: ignore[method-assign]
    runtime._set_servo_paused = pause_states.append  # type: ignore[method-assign]
    runtime._select_twist_command_type = lambda: selected.append(True)  # type: ignore[method-assign]
    monkeypatch.setattr("demo2gripper.test_tube_live_io.time.sleep", lambda _: None)

    with pytest.raises(ValueError, match="pose_goal_script"):
        runtime.move_coarse(target=np.eye(4), action_id="coarse-setup-failure")

    assert pause_states == [True, False]
    assert selected == [True]


@pytest.mark.parametrize("disable_guard", (False, True))
def test_coarse_moveit_forwards_task_scoped_pose_delta_guard_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disable_guard: bool,
) -> None:
    runtime = object.__new__(RosMoveItServoRuntime)
    runtime._target_lock = threading.Lock()  # type: ignore[attr-defined]
    runtime._servo_target = np.eye(4)  # type: ignore[attr-defined]
    runtime._servo_horizon_s = 0.1  # type: ignore[attr-defined]
    runtime._servo_period_s = 0.02  # type: ignore[attr-defined]
    runtime._T_link8_tcp = np.eye(4)  # type: ignore[attr-defined]
    runtime._output_dir = tmp_path  # type: ignore[attr-defined]
    runtime._moveit = {  # type: ignore[attr-defined]
        "pose_goal_script": str(tmp_path / "moveit_pose_goal.py"),
        "setup_scripts": [],
        "disable_pose_delta_guard": disable_guard,
        "disable_joint_sweep_guards": disable_guard,
    }
    pause_states: list[bool] = []
    selected: list[bool] = []
    commands: list[list[str]] = []
    runtime._publish_zero = lambda: None  # type: ignore[method-assign]
    runtime._set_servo_paused = pause_states.append  # type: ignore[method-assign]
    runtime._select_twist_command_type = lambda: selected.append(True)  # type: ignore[method-assign]
    monkeypatch.setattr("demo2gripper.test_tube_live_io.time.sleep", lambda _: None)

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        tokens = shlex.split(command[-1])
        commands.append(tokens)
        output = Path(tokens[tokens.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "execution_succeeded": True,
                    "measured_target_position_error_m": 0.001,
                    "measured_target_rotation_error_rad": 0.01,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("demo2gripper.test_tube_live_io.subprocess.run", fake_run)
    receipt = runtime.move_coarse(
        target=transform(x=0.25), action_id="coarse-large-delta"
    )

    assert len(commands) == 1
    assert ("--disable-pose-delta-guard" in commands[0]) is disable_guard
    assert ("--disable-joint-sweep-guards" in commands[0]) is disable_guard
    assert pause_states == [True, False]
    assert selected == [True]
    assert receipt["accepted"] is True


@pytest.mark.parametrize(
    "field", ("disable_pose_delta_guard", "disable_joint_sweep_guards")
)
def test_coarse_moveit_rejects_non_boolean_guard_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    runtime = object.__new__(RosMoveItServoRuntime)
    runtime._target_lock = threading.Lock()  # type: ignore[attr-defined]
    runtime._servo_target = np.eye(4)  # type: ignore[attr-defined]
    runtime._servo_horizon_s = 0.1  # type: ignore[attr-defined]
    runtime._servo_period_s = 0.02  # type: ignore[attr-defined]
    runtime._T_link8_tcp = np.eye(4)  # type: ignore[attr-defined]
    runtime._output_dir = tmp_path  # type: ignore[attr-defined]
    runtime._moveit = {  # type: ignore[attr-defined]
        "pose_goal_script": str(tmp_path / "moveit_pose_goal.py"),
        "setup_scripts": [],
        field: "true",
    }
    pause_states: list[bool] = []
    selected: list[bool] = []
    runtime._publish_zero = lambda: None  # type: ignore[method-assign]
    runtime._set_servo_paused = pause_states.append  # type: ignore[method-assign]
    runtime._select_twist_command_type = lambda: selected.append(True)  # type: ignore[method-assign]
    monkeypatch.setattr("demo2gripper.test_tube_live_io.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="must be a boolean"):
        runtime.move_coarse(target=transform(x=0.25), action_id="bad-guard")

    assert pause_states == [True, False]
    assert selected == [True]


def test_observation_time_is_sampled_after_tactile_and_pose_producers() -> None:
    values = iter((2.0, 2.2))
    io = make_io(
        scene=FakeScene(source_mask=True),
        tactile=FakeTactile(),
        monotonic=lambda: next(values),
    )

    observation = io.observe(state=TubeState.OBSERVE)

    assert observation.tactile.observed_at_s == pytest.approx(2.0)
    assert observation.now_s == pytest.approx(2.2)
    assert observation.now_s >= observation.tactile.observed_at_s


def test_source_pose_above_receipt_uncertainty_reacquires_without_raising() -> None:
    scene = FakeScene(source_mask=True, source_position_std_m=0.006)
    applied_limits = []
    scene.limit_source_pose_uncertainty = lambda **limits: applied_limits.append(limits)
    io = make_io(
        scene=scene,
        tactile=FakeTactile(),
    )
    assert io.observe(state=TubeState.OBSERVE).tube_pose is None
    assert io._tube_tracking_id is None
    assert applied_limits == [{"position_std_m": .005, "orientation_std_deg": 5.0}]
    scene.source_position_std_m = .001
    assert io.observe(state=TubeState.OBSERVE).tube_pose is not None
    assert io._held_position_std_limit_m == .005


def test_online_pose_warmup_reobserves_without_promoting_a_transform() -> None:
    class BootstrappingScene(FakeScene):
        def qualify_pose(self, **_: object) -> None:
            return None

    io = make_io(
        scene=BootstrappingScene(source_mask=True),
        tactile=FakeTactile(),
    )
    observation = io.observe(state=TubeState.OBSERVE)
    assert observation.tube_pose is None


def test_continuous_close_is_one_absolute_width_command(monkeypatch) -> None:
    io = make_io(scene=FakeScene(), tactile=FakeTactile())
    events = []
    move_width = io._gripper.move_width

    def held(**kwargs):
        events.append("arm_held")
        return {"held": True}

    def close_width(**kwargs):
        events.append("gripper_close")
        return move_width(**kwargs)

    monkeypatch.setattr(io._arm, "stop", held)
    monkeypatch.setattr(io._gripper, "move_width", close_width)
    receipt = io.dispatch(
        action(
            ActionKind.START_CONTINUOUS_CLOSE,
            {"target_width_m": 0.005, "speed": 40, "force": 8},
        )
    )
    assert receipt["accepted"] is True
    assert events == ["arm_held", "gripper_close"]
    assert io._gripper.moves == [0.005]  # type: ignore[attr-defined]


def test_open_release_resets_support_history_after_gripper_accepts() -> None:
    scene = FakeScene()
    gripper = FakeGripper()
    io = make_io(scene=scene, tactile=FakeTactile(), gripper=gripper)

    receipt = io.dispatch(action(ActionKind.OPEN_RELEASE, {}, state=TubeState.RELEASE))

    assert receipt["accepted"] is True
    assert gripper.moves == [0.050]
    assert scene.support_history_resets == 1


def test_normalized_demo_relation_is_frozen_from_last_source_pose_at_extract() -> None:
    scene = FakeScene(source_mask=True)
    arm = FakeArm()
    io = make_io(
        scene=scene,
        tactile=FakeTactile(stable=True),
        arm=arm,
        T_tube_tcp_demo=None,
    )
    # Explicitly select normalized mode; ``make_io`` uses the legacy fixture
    # whenever its optional argument is omitted.
    io._T_tube_tcp_demo = None  # type: ignore[attr-defined]
    source = io.observe(state=TubeState.SIDE_GRASP)
    assert source.tube_pose is not None
    lift = action(
        ActionKind.SERVO_TCP,
        {"T_base_tcp_command": transform(z=0.1)},
        state=TubeState.EXTRACT,
    )
    io.dispatch(lift)
    assert len(scene.grasp_relations) == 1
    expected_relation = np.linalg.inv(source.tube_pose.T_base_object) @ source.robot.T_base_tcp
    np.testing.assert_allclose(scene.grasp_relations[0][0], expected_relation)
    np.testing.assert_allclose(scene.grasp_relations[0][1], source.tube_pose.T_base_object)

    # A later EXTRACT observation must use the pre-motion relation even if the
    # blocking coarse lift has already changed TCP by centimetres.
    arm.T[2, 3] += 0.04
    extracted = io.observe(state=TubeState.EXTRACT)
    assert extracted.tube_pose is not None
    np.testing.assert_allclose(
        extracted.tube_pose.T_base_object,
        arm.T @ np.linalg.inv(expected_relation),
    )
    assert io._T_tube_tcp_grasp_actual is not None  # type: ignore[attr-defined]


def test_extract_motion_fails_before_dispatch_without_contact_relation() -> None:
    arm = FakeArm()
    io = make_io(scene=FakeScene(), tactile=FakeTactile(), arm=arm)
    lift = action(
        ActionKind.SERVO_TCP,
        {"T_base_tcp_command": transform(z=0.1)},
        state=TubeState.EXTRACT,
    )

    with pytest.raises(RuntimeError, match="stationary bilateral-contact"):
        io.dispatch(lift)

    assert arm.servo_calls == 0


def test_contact_relation_uses_fresh_last_source_pose_after_visual_loss() -> None:
    scene = FakeScene(source_mask=True)
    arm = FakeArm()
    times = iter((1.05, 1.10, 1.15, 1.18))
    io = make_io(
        scene=scene,
        tactile=FakeTactile(stable=True),
        arm=arm,
        monotonic=lambda: next(times),
    )
    visible = io.observe(state=TubeState.SIDE_GRASP)
    assert visible.tube_pose is not None

    scene.source_mask = False
    arm.T[0, 3] += 0.001
    occluded = io.observe(state=TubeState.SIDE_GRASP)
    assert occluded.tube_pose is None
    lift = action(
        ActionKind.SERVO_TCP,
        {"T_base_tcp_command": transform(z=0.1)},
        state=TubeState.EXTRACT,
    )
    io.dispatch(lift)

    assert len(scene.grasp_relations) == 1
    expected = np.linalg.inv(visible.tube_pose.T_base_object) @ occluded.robot.T_base_tcp
    np.testing.assert_allclose(scene.grasp_relations[0][0], expected)


def test_stale_last_source_pose_cannot_authorize_extract_dispatch() -> None:
    scene = FakeScene(source_mask=True)
    times = iter((1.05, 1.10, 1.35, 1.40))
    io = make_io(
        scene=scene,
        tactile=FakeTactile(stable=True),
        monotonic=lambda: next(times),
    )
    io.observe(state=TubeState.SIDE_GRASP)
    scene.source_mask = False
    io.observe(state=TubeState.SIDE_GRASP)
    lift = action(
        ActionKind.SERVO_TCP,
        {"T_base_tcp_command": transform(z=0.1)},
        state=TubeState.EXTRACT,
    )

    with pytest.raises(RuntimeError, match="stationary bilateral-contact"):
        io.dispatch(lift)


def test_preclose_latch_survives_source_expiry_without_refreshing_timestamp() -> None:
    now = [1.0]
    scene = FakeScene(source_mask=True)
    arm = FakeArm()
    tactile = FakeTactile(stable=False)
    io = make_io(scene=scene, arm=arm, tactile=tactile, monotonic=lambda: now[0])
    visible = io.observe(state=TubeState.SIDE_GRASP)
    io.dispatch(action(ActionKind.START_CONTINUOUS_CLOSE,
                       {"target_width_m": .005, "speed": 40, "force": 8}))
    latch = io._preclose_source_pose
    assert latch is not None
    assert latch.observed_at_s == 1.0
    scene.source_mask = False
    now[0] = 3.0
    # No tactile contact: the latch alone cannot authorize extraction.
    io.observe(state=TubeState.SIDE_GRASP)
    assert io._pending_contact_T_base_tube is None
    tactile.stable = True
    arm.T[0, 3] += .0005
    occluded = io.observe(state=TubeState.SIDE_GRASP)
    assert occluded.tube_pose is None
    assert io._last_source_observed_at_s == 1.0
    assert io._pending_contact_from_preclose
    assert io._pending_contact_position_std_m > latch.position_std_m
    io.dispatch(action(ActionKind.SERVO_TCP, {"T_base_tcp_command": transform(z=.1)},
                       state=TubeState.EXTRACT))
    expected = np.linalg.inv(visible.tube_pose.T_base_object) @ arm.T
    np.testing.assert_allclose(scene.grasp_relations[0][0], expected)
    assert io._preclose_source_pose is None


def test_shallow_contact_retry_preserves_and_bounds_committed_source_latch() -> None:
    now = [1.0]
    scene = FakeScene(source_mask=True)
    arm = FakeArm()
    tactile = FakeTactile(stable=False)
    io = make_io(scene=scene, arm=arm, tactile=tactile, monotonic=lambda: now[0])
    visible = io.observe(state=TubeState.SIDE_GRASP)
    assert visible.tube_pose is not None
    io.dispatch(
        action(
            ActionKind.START_CONTINUOUS_CLOSE,
            {"target_width_m": 0.005, "speed": 40, "force": 8},
            state=TubeState.SIDE_GRASP,
        )
    )
    assert io._preclose_source_pose is not None

    retry = {
        "shallow_contact_retry": True,
        "retry_index": 1,
        "retry_total_travel_m": 0.005,
        "source_latch_max_displacement_m": 0.0075,
    }
    io.dispatch(
        action(
            ActionKind.OPEN_GRIPPER_STEP,
            {**retry, "delta_m": 0.0445},
            state=TubeState.SIDE_GRASP,
        )
    )
    assert io._preclose_source_pose is not None
    assert io._preclose_source_pose.max_displacement_m == pytest.approx(0.0075)

    target = arm.T.copy()
    target[2, 3] -= 0.005
    io.dispatch(
        action(
            ActionKind.SERVO_TCP,
            {**retry, "T_base_tcp_command": target},
            state=TubeState.SIDE_GRASP,
        )
    )
    assert io._preclose_source_pose is not None
    arm.T = target
    io.dispatch(
        action(
            ActionKind.START_CONTINUOUS_CLOSE,
            {
                **retry,
                "target_width_m": 0.005,
                "speed": 40,
                "force": 8,
            },
            state=TubeState.SIDE_GRASP,
        )
    )
    assert io._preclose_source_pose is not None

    scene.source_mask = False
    tactile.stable = True
    now[0] = 1.4
    io.observe(state=TubeState.SIDE_GRASP)
    assert io._pending_contact_from_preclose is True
    assert io._pending_contact_position_std_m > visible.tube_pose.position_std_m


@pytest.fixture
def final_contact_io():
    now = [1.0]
    scene, arm = FakeScene(source_mask=True), FakeArm()
    tactile, gripper = FakeTactile(), FakeGripper()
    gripper.width = .0494
    io = make_io(scene=scene, arm=arm, tactile=tactile, gripper=gripper, monotonic=lambda: now[0])
    io._final_approach_commit_distance_m = .020
    original_read = io._camera.read
    io._camera.read = lambda: replace(original_read(), host_received_monotonic_s=now[0])
    source = io.observe(state=TubeState.SIDE_GRASP).tube_pose
    target = arm.T.copy()
    target[2, 3] -= .016
    command = arm.T.copy()
    command[2, 3] -= .008
    parameters = {
        "final_contact_approach": True, "final_contact_target": target.copy(),
        "committed_source_observed_at_s": source.observed_at_s,
        "T_base_tcp_command": command.copy(),
    }
    return SimpleNamespace(io=io, scene=scene, arm=arm, tactile=tactile, gripper=gripper,
                           now=now, source=source, target=target, parameters=parameters)


def test_final_contact_approach_retains_source_through_depth_loss_and_close(final_contact_io):
    f = final_contact_io
    f.io.dispatch(action(ActionKind.SERVO_TCP, f.parameters, state=TubeState.SIDE_GRASP))
    f.arm.T = f.parameters["T_base_tcp_command"].copy()
    f.scene.source_mask = False
    f.scene.source_depth_only_loss = True
    f.now[0] = 2.0  # Original metric pose is stale, but this RGB certificate is current.
    observation = f.io.observe(state=TubeState.SIDE_GRASP)
    assert observation.tube_pose is None and observation.source_depth_only_loss
    f.parameters["T_base_tcp_command"] = f.target.copy()
    f.io.dispatch(action(ActionKind.SERVO_TCP, f.parameters, state=TubeState.SIDE_GRASP))
    assert f.io._final_contact_approach.source.observed_at_s == 1.0
    f.arm.T = f.target.copy()
    f.now[0] = 2.1
    f.io.observe(state=TubeState.SIDE_GRASP)
    f.io.dispatch(action(ActionKind.START_CONTINUOUS_CLOSE, {
        **f.parameters, "committed_final_approach": True,
        "target_width_m": .018, "speed": 40, "force": 8,
    }, state=TubeState.SIDE_GRASP))
    assert f.io._preclose_source_pose.observed_at_s == 1.0
    assert f.io._preclose_source_pose.position_std_m == f.source.position_std_m
    assert f.io._final_contact_approach is None
    f.now[0] = 3.0
    f.tactile.stable = True
    f.io.observe(state=TubeState.SIDE_GRASP)
    f.io.dispatch(action(ActionKind.SERVO_TCP, {"T_base_tcp_command": transform(z=.4)},
                         state=TubeState.EXTRACT))
    np.testing.assert_allclose(
        f.scene.grasp_relations[0][0], np.linalg.inv(f.source.T_base_object) @ f.arm.T
    )


def test_final_contact_approach_retains_source_when_only_axis_sign_is_unavailable(
    final_contact_io,
):
    f = final_contact_io
    f.io.dispatch(action(ActionKind.SERVO_TCP, f.parameters, state=TubeState.SIDE_GRASP))
    f.arm.T = f.parameters["T_base_tcp_command"].copy()
    f.scene.source_mask = False
    f.scene.source_depth_only_loss = False
    f.scene.source_rgb_identity_retained = True
    f.now[0] = 2.0
    observation = f.io.observe(state=TubeState.SIDE_GRASP)
    assert observation.tube_pose is None
    assert not observation.source_depth_only_loss
    assert observation.source_rgb_identity_retained
    f.parameters["T_base_tcp_command"] = f.target.copy()
    f.io.dispatch(action(ActionKind.SERVO_TCP, f.parameters, state=TubeState.SIDE_GRASP))
    assert f.io._final_contact_approach.source.observed_at_s == pytest.approx(1.0)


def test_final_contact_can_begin_from_current_rgb_depth_loss_certificate(final_contact_io):
    f = final_contact_io
    f.scene.source_mask = False
    f.scene.source_depth_only_loss = True
    f.now[0] = 3.0
    observation = f.io.observe(state=TubeState.APPROACH_TUBE)
    assert observation.tube_pose is None and observation.source_depth_only_loss
    f.io.dispatch(action(ActionKind.SERVO_TCP, {
        **f.parameters,
        "depth_loss_handoff": True,
    }, state=TubeState.SIDE_GRASP))
    commit = f.io._final_contact_approach
    assert commit is not None
    assert commit.depth_loss_handoff
    assert commit.source.observed_at_s == pytest.approx(1.0)
    np.testing.assert_allclose(commit.source.T_base_tube, f.source.T_base_object)


@pytest.mark.parametrize("failure", [
    "timeout", "mask_loss", "contact", "identity", "sideways", "fault",
    "source_moved", "target_changed",
])
def test_final_contact_approach_rejects_invalid_continuation(final_contact_io, failure):
    f = final_contact_io
    f.io.dispatch(action(ActionKind.SERVO_TCP, f.parameters, state=TubeState.SIDE_GRASP))
    f.scene.source_mask = False
    f.scene.source_depth_only_loss = True
    f.now[0] = 2.0
    if failure == "timeout":
        f.now[0] = 10.0
    elif failure == "mask_loss":
        f.scene.source_depth_only_loss = False
    elif failure == "contact":
        f.tactile.stable = True
    elif failure == "identity":
        f.io._tube_tracking_id = "different-tube"
    elif failure == "sideways":
        f.arm.T[0, 3] += .005
    elif failure == "fault":
        f.arm.state = lambda: ArmState(f.arm.T, True, True, "hardware_fault")
    elif failure == "target_changed":
        f.parameters["final_contact_target"][0, 3] += .001
    f.io.observe(state=TubeState.SIDE_GRASP)
    if failure == "source_moved":
        moved = f.source.T_base_object.copy()
        moved[0, 3] += .004
        f.io._current_source_pose = replace(f.source, T_base_object=moved, observed_at_s=f.now[0])
    with pytest.raises(RuntimeError, match="commitment expired or current evidence changed"):
        f.io.dispatch(action(ActionKind.SERVO_TCP, f.parameters, state=TubeState.SIDE_GRASP))
    assert f.arm.servo_calls == 1
    assert f.io._final_contact_approach is None


def test_final_contact_close_requires_arrival_and_fresh_rgb_certificate(final_contact_io):
    f = final_contact_io
    f.io.dispatch(action(ActionKind.SERVO_TCP, f.parameters, state=TubeState.SIDE_GRASP))
    with pytest.raises(RuntimeError, match="stopped TCP at committed endpoint"):
        f.io.dispatch(action(ActionKind.START_CONTINUOUS_CLOSE, {
            **f.parameters, "committed_final_approach": True,
            "target_width_m": .018, "speed": 40, "force": 8,
        }, state=TubeState.SIDE_GRASP))
    assert f.gripper.moves == []
    f.scene.source_mask = False
    f.scene.source_depth_only_loss = True
    f.now[0] = 3.0
    fresh_read = f.io._camera.read
    f.io._camera.read = lambda: replace(fresh_read(), host_received_monotonic_s=1.0)
    assert not f.io.observe(state=TubeState.SIDE_GRASP).source_depth_only_loss


def _pause_at_final_contact_arrival(f):
    f.io.dispatch(action(ActionKind.SERVO_TCP, f.parameters, state=TubeState.SIDE_GRASP))
    f.arm.T = f.target.copy()
    f.scene.source_mask = False
    f.scene.source_depth_only_loss = True
    f.now[0] = 2.0
    f.io.observe(state=TubeState.SIDE_GRASP)
    f.io.dispatch(action(ActionKind.PAUSE_ARM, {
        **f.parameters, "final_contact_arrived": True,
    }, state=TubeState.SIDE_GRASP))


def test_arrival_pause_hands_off_to_stationary_tactile_without_extending_deadline(final_contact_io):
    f = final_contact_io
    _pause_at_final_contact_arrival(f)
    commit = f.io._final_contact_approach
    assert commit.source.observed_at_s == 1.0
    assert commit.source.deadline_s == 9.0
    np.testing.assert_array_equal(commit.T_base_arrival, f.target)
    f.scene.source_depth_only_loss = False
    f.now[0] = 3.0
    observation = f.io.observe(state=TubeState.SIDE_GRASP)
    assert observation.tube_pose is None and not observation.source_depth_only_loss
    # A repeated settled-window pause must retain the first pose and deadline.
    f.arm.T[0, 3] += .0005
    f.io.dispatch(action(ActionKind.PAUSE_ARM, {
        **f.parameters, "final_contact_arrived": True,
    }, state=TubeState.SIDE_GRASP))
    assert f.io._final_contact_approach is commit
    f.io.dispatch(action(ActionKind.START_CONTINUOUS_CLOSE, {
        **f.parameters, "committed_final_approach": True,
        "target_width_m": .018, "speed": 40, "force": 8,
    }, state=TubeState.SIDE_GRASP))
    assert f.gripper.moves == [.018]
    assert f.io._preclose_source_pose.observed_at_s == 1.0
    assert f.io._preclose_source_pose.position_std_m == f.source.position_std_m
    np.testing.assert_array_equal(f.io._preclose_source_pose.T_base_tcp, f.arm.T)


@pytest.mark.parametrize("failure", [
    "first_frame_occluded", "not_at_target", "still_moving",
])
def test_arrival_must_be_visually_qualified_and_stopped_before_pause(final_contact_io, failure):
    f = final_contact_io
    f.io.dispatch(action(ActionKind.SERVO_TCP, f.parameters, state=TubeState.SIDE_GRASP))
    if failure != "not_at_target":
        f.arm.T = f.target.copy()
    if failure == "first_frame_occluded":
        f.scene.source_mask = False
        f.scene.source_depth_only_loss = False
    elif failure == "still_moving":
        f.arm.state = lambda: ArmState(f.arm.T, False, True)
    f.now[0] = 1.1
    f.io.observe(state=TubeState.SIDE_GRASP)
    with pytest.raises(RuntimeError, match="commitment expired or current evidence changed"):
        f.io.dispatch(action(ActionKind.PAUSE_ARM, {
            **f.parameters, "final_contact_arrived": True,
        }, state=TubeState.SIDE_GRASP))
    assert f.io._final_contact_approach is None
    assert f.gripper.moves == []


@pytest.mark.parametrize("failure", [
    "timeout", "tcp_moved", "tcp_rotated", "still_moving", "source_moved",
    "identity", "robot_fault", "gripper_closed", "tactile_stale", "contact",
])
def test_arrived_handoff_rejects_changed_conditions(final_contact_io, failure):
    f = final_contact_io
    _pause_at_final_contact_arrival(f)
    f.scene.source_depth_only_loss = False
    f.now[0] = 3.0
    f.io.observe(state=TubeState.SIDE_GRASP)
    if failure == "timeout":
        f.now[0] = 9.001
        f.io.observe(state=TubeState.SIDE_GRASP)
    elif failure == "tcp_moved":
        f.arm.T[2, 3] += .003
    elif failure == "tcp_rotated":
        f.arm.T[:3, :3] = Rotation.from_euler("z", 3, degrees=True).as_matrix()
    elif failure == "still_moving":
        f.arm.state = lambda: ArmState(f.arm.T, False, True)
    elif failure == "source_moved":
        moved = f.source.T_base_object.copy()
        moved[0, 3] += .004
        f.io._current_source_pose = replace(f.source, T_base_object=moved, observed_at_s=3.0)
    elif failure == "identity":
        f.io._current_source_pose = replace(f.source, tracking_id="other", observed_at_s=3.0)
    elif failure == "robot_fault":
        f.arm.state = lambda: ArmState(f.arm.T, True, True, "fault")
    elif failure == "gripper_closed":
        f.gripper.width = .04
    elif failure == "tactile_stale":
        f.io._last_tactile_feedback = replace(f.io._last_tactile_feedback, observed_at_s=1.0)
    elif failure == "contact":
        f.io._last_tactile_feedback = replace(f.io._last_tactile_feedback, contact_present=True)
    with pytest.raises(RuntimeError, match="commitment expired or current evidence changed"):
        f.io.dispatch(action(ActionKind.START_CONTINUOUS_CLOSE, {
            **f.parameters, "committed_final_approach": True,
            "target_width_m": .018, "speed": 40, "force": 8,
        }, state=TubeState.SIDE_GRASP))
    assert f.gripper.moves == []


def test_final_contact_servo_is_rejected_after_arrival(final_contact_io):
    f = final_contact_io
    _pause_at_final_contact_arrival(f)
    before = f.arm.servo_calls
    with pytest.raises(RuntimeError, match="cannot move after arrival"):
        f.io.dispatch(action(ActionKind.SERVO_TCP, f.parameters, state=TubeState.SIDE_GRASP))
    assert f.arm.servo_calls == before
    assert f.io._final_contact_approach is None


def test_nonfinal_arm_command_invalidates_final_contact_commitment(final_contact_io):
    f = final_contact_io
    f.io.dispatch(action(ActionKind.SERVO_TCP, f.parameters, state=TubeState.SIDE_GRASP))
    assert f.io._final_contact_approach is not None
    f.io.dispatch(action(ActionKind.SERVO_TCP, {"T_base_tcp_command": f.arm.T.copy()},
                         state=TubeState.SIDE_GRASP))
    assert f.io._final_contact_approach is None


def test_final_contact_rejects_stale_initial_source_and_outside_segment(final_contact_io):
    f = final_contact_io
    f.now[0] = 2.0
    with pytest.raises(RuntimeError, match="fresh qualified bounded source"):
        f.io.dispatch(action(ActionKind.SERVO_TCP, f.parameters, state=TubeState.SIDE_GRASP))
    assert f.arm.servo_calls == 0
    f.now[0] = 1.0
    f.parameters["T_base_tcp_command"][0, 3] += .01
    with pytest.raises(RuntimeError, match="leaves its committed segment"):
        f.io.dispatch(action(ActionKind.SERVO_TCP, f.parameters, state=TubeState.SIDE_GRASP))
    assert f.arm.servo_calls == 0


@pytest.mark.parametrize("tcp_z_m", [.15, .21])
def test_contact_propagation_cancels_fixed_tcp_calibration(tcp_z_m) -> None:
    scene = FakeScene(source_mask=True)
    arm = FakeArm()
    calibration = transform(x=.01, z=tcp_z_m)
    calibration[:3, :3] = Rotation.from_euler("xyz", [.1, -.2, .3]).as_matrix()
    flange_start = transform(.5, 0.0, .3)
    arm.T = flange_start @ calibration
    io = make_io(scene=scene, arm=arm, tactile=FakeTactile(stable=True))
    io._tcp_position_std_m = .005
    io._tcp_orientation_std_deg = 3.0
    source = io.observe(state=TubeState.SIDE_GRASP).tube_pose
    io.dispatch(action(ActionKind.SERVO_TCP, {"T_base_tcp_command": transform(z=.1)},
                       state=TubeState.EXTRACT))
    flange_lifted = transform(.51, 0.0, .34)
    flange_lifted[:3, :3] = Rotation.from_euler("y", .2).as_matrix()
    arm.T = flange_lifted @ calibration
    held = io.observe(state=TubeState.EXTRACT).tube_pose
    np.testing.assert_allclose(
        held.T_base_object, flange_lifted @ np.linalg.inv(flange_start) @ source.T_base_object,
        atol=1e-12,
    )
    assert held.position_std_m == source.position_std_m
    assert held.orientation_std_deg == source.orientation_std_deg


@pytest.mark.parametrize("invalidation", ["expired", "motion", "fault", "identity", "recenter"])
def test_preclose_latch_invalidates_before_extract(invalidation) -> None:
    now = [1.0]
    scene = FakeScene(source_mask=True)
    arm = FakeArm()
    io = make_io(scene=scene, arm=arm, tactile=FakeTactile(stable=True),
                 monotonic=lambda: now[0])
    io.observe(state=TubeState.SIDE_GRASP)
    io.dispatch(action(ActionKind.START_CONTINUOUS_CLOSE,
                       {"target_width_m": .005, "speed": 40, "force": 8}))
    assert io._preclose_source_pose is not None
    scene.source_mask = False
    now[0] = 3.0
    if invalidation == "expired":
        now[0] = 8.0
    elif invalidation == "motion":
        arm.T[0, 3] += .005
    elif invalidation == "fault":
        arm.state = lambda: ArmState(arm.T, True, True, "hardware fault")
    elif invalidation == "identity":
        io._tube_tracking_id = "different-tube"
    elif invalidation == "recenter":
        io.dispatch(action(ActionKind.OPEN_AND_RECENTER,
                           {"open_delta_m": .001, "tcp_local_x_offset_m": .001}))
    io.observe(state=TubeState.SIDE_GRASP)
    assert io._preclose_source_pose is None
    assert io._pending_contact_T_base_tube is None
    with pytest.raises(RuntimeError, match="stationary bilateral-contact"):
        io.dispatch(action(ActionKind.SERVO_TCP, {"T_base_tcp_command": transform(z=.1)},
                           state=TubeState.EXTRACT))


@pytest.mark.parametrize(
    "tactile",
    (
        FakeTactile(stable=True, valid=False),
        FakeTactile(stable=True, observed_at_s=0.0),
        FakeTactile(stable=True, observed_at_s=1.1),
    ),
    ids=("invalid", "stale", "future"),
)
def test_unqualified_tactile_cannot_cache_extract_relation(
    tactile: FakeTactile,
) -> None:
    io = make_io(
        scene=FakeScene(source_mask=True),
        tactile=tactile,
        monotonic=lambda: 1.0,
    )
    io.observe(state=TubeState.SIDE_GRASP)
    lift = action(
        ActionKind.SERVO_TCP,
        {"T_base_tcp_command": transform(z=0.1)},
        state=TubeState.EXTRACT,
    )

    with pytest.raises(RuntimeError, match="stationary bilateral-contact"):
        io.dispatch(lift)


def test_post_pivot_relation_is_not_frozen_while_reclamp_is_moving() -> None:
    scene = FakeScene()
    gripper = MovingFakeGripper()
    times = iter((1.0, 1.01, 1.1, 1.11))
    io = make_io(
        scene=scene,
        tactile=FakeTactile(stable=True),
        gripper=gripper,
        monotonic=lambda: next(times),
    )

    io.observe(state=TubeState.RECLAMP)
    assert io._T_tube_tcp_held is None  # type: ignore[attr-defined]
    assert scene.held_calls == 1

    gripper.moving = False
    io.observe(state=TubeState.RECLAMP)
    assert io._T_tube_tcp_held is not None  # type: ignore[attr-defined]
    assert scene.held_calls == 2


def test_recenter_waits_for_measured_open_width_before_starting_servo() -> None:
    arm = FakeArm()
    gripper = LaggingFakeGripper()
    io = make_io(
        scene=FakeScene(),
        tactile=FakeTactile(),
        arm=arm,
        gripper=gripper,
    )
    io.dispatch(
        action(
            ActionKind.OPEN_AND_RECENTER,
            {"open_delta_m": 0.004, "tcp_local_x_offset_m": 0.002},
        )
    )

    waiting = io.observe(state=TubeState.SIDE_GRASP)
    assert waiting.gripper.moving is True
    assert arm.servo_calls == 0

    gripper.width = 0.044
    io.observe(state=TubeState.SIDE_GRASP)
    assert arm.servo_calls == 1


@pytest.mark.parametrize("distance_m", [0.0015, -0.0015, 0.0006])
def test_only_insertion_commands_request_smaller_position_tolerance(distance_m: float) -> None:
    arm = FakeArm()
    io = make_io(scene=FakeScene(), tactile=FakeTactile(), arm=arm)
    io.dispatch(action(
        ActionKind.INSERT_STEP,
        {"axis_base": [1.0, 0.0, 0.0], "distance_m": distance_m},
    ))
    assert arm.servo_arguments[-1]["position_tolerance_m"] == pytest.approx(
        min(0.0005, abs(distance_m) / 3.0)
    )
    io.dispatch(action(ActionKind.SERVO_TCP, {"T_base_tcp_command": arm.T.tolist()}))
    assert "position_tolerance_m" not in arm.servo_arguments[-1]


def test_jam_retract_reduces_measured_total_insertion_depth() -> None:
    arm = FakeArm()
    io = make_io(scene=FakeScene(), tactile=FakeTactile(), arm=arm)
    insertion = action(
        ActionKind.INSERT_STEP,
        {"axis_base": [1.0, 0.0, 0.0], "distance_m": 0.004},
    )
    io.dispatch(insertion)
    arm.T[0, 3] += 0.004
    inserted = io.observe(state=TubeState.INSERT_EVALUATE)
    assert inserted.insertion_depth_m == pytest.approx(0.004)

    retract = action(
        ActionKind.RETRACT_INSERTION,
        {"axis_base": [1.0, 0.0, 0.0], "distance_m": -0.003},
    )
    io.dispatch(retract)
    arm.T[0, 3] -= 0.003
    retracted = io.observe(state=TubeState.JAM_RECOVERY_RETRACT)
    assert retracted.insertion_depth_m == pytest.approx(0.001)


def test_tactile_uses_arm_progress_sampled_after_window_collection() -> None:
    arm = FakeArm()
    tactile = ProgressRecordingTactile()
    io = make_io(scene=FakeScene(), tactile=tactile, arm=arm)
    io.dispatch(
        action(
            ActionKind.INSERT_STEP,
            {"axis_base": [1.0, 0.0, 0.0], "distance_m": 0.004},
        )
    )
    original_collect = io._tactile_source.collect_classifier_windows  # type: ignore[attr-defined]

    def collect_after_motion(*args: object, **kwargs: object) -> object:
        arm.T[0, 3] += 0.003
        return original_collect(*args, **kwargs)

    io._tactile_source.collect_classifier_windows = collect_after_motion  # type: ignore[attr-defined]
    observation = io.observe(state=TubeState.INSERT_EVALUATE)

    assert tactile.measured_progress_m == pytest.approx(0.003)
    assert observation.insertion_progress_m == pytest.approx(0.003)


def test_unfilled_template_reports_physical_qualification_blockers() -> None:
    blockers = preflight_live_config(
        {
            "geometry": {
                "T_tube_tcp_demo_grasp": None,
                "relation_runtime_qualified": False,
                "T_socket_tube_entry": None,
                "socket_entry_runtime_qualified": False,
            },
            "pose_backend": {"name": "external", "service_url": None},
            "live_adapter": {"schema": "demo2gripper.test_tube_live_io.v1"},
        }
    )
    assert "demo_T_tube_tcp_missing" in blockers
    assert "T_socket_tube_entry_missing" in blockers
    assert "pose_service_url_missing" in blockers
    assert "camera_config_missing" in blockers


def test_static_tcp_measurement_is_reusable_without_a_per_run_receipt(
    tmp_path: Path,
) -> None:
    transform_path = tmp_path / "T_link8_tcp.npy"
    matrix = transform(z=0.190)
    np.save(transform_path, matrix, allow_pickle=False)
    metadata_path = tmp_path / "T_link8_tcp.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "demo2gripper.tool_transform_measurement.v1",
                "transform": transform_path.name,
                "translation_m": matrix[:3, 3].tolist(),
                "rotation_matrix": matrix[:3, :3].tolist(),
                "evidence": {"method": "fixture measurement plus URDF cross-check"},
            }
        ),
        encoding="utf-8",
    )
    robot = {
        "tcp": str(transform_path),
        "tcp_static_calibration": str(metadata_path),
        "tcp_installation_id": "fr3-hande-installation-20261021",
        "tcp_position_bound_m": 0.005,
        "tcp_orientation_bound_deg": 3.0,
    }

    assert _tcp_calibration_blockers(robot) == []
    assert _tcp_calibration_uncertainty(robot) == (0.005, 3.0)

    bad = dict(robot)
    bad["tcp_installation_id"] = None
    assert "hande_tcp_installation_id_missing" in _tcp_calibration_blockers(bad)

    exact = dict(robot)
    exact["tcp_position_bound_m"] = 0.0
    assert "hande_tcp_position_bound_missing" in _tcp_calibration_blockers(exact)


def test_live_template_wires_resident_source_and_downstream_factories_fail_closed() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo / "configs/test_tube_closed_loop.live.template.json").read_text(encoding="utf-8")
    )
    scene = config["live_adapter"]["scene_interpreter"]
    assert scene["factory"] == (
        "demo2gripper.test_tube_pose_tracker:create_static_source_tube_scene_interpreter"
    )
    assert scene["config"]["downstream_factory"] == (
        "demo2gripper.test_tube_downstream_scene:create_qualified_held_tube_scene_interpreter"
    )
    assert scene["config"]["downstream_config"]["T_tube_tcp_demo_grasp"] is None

    blockers = preflight_live_config(config)
    assert "scene_interpreter_factory_missing" not in blockers
    assert "source_reviewed_seed_rgb_missing" in blockers
    assert "source_reviewed_initial_mask_missing" in blockers
    assert "source_reviewed_seed_qualification_receipt_missing" in blockers
    assert "source_reviewed_endpoints_missing_or_invalid" in blockers
    assert "held_rgb_tracker_qualification_report_missing" in blockers
    assert "held_tube_reviewed_seed_line_missing_or_invalid" in blockers
    assert "downstream_static_socket_pose_artifact_missing" in blockers
    assert "downstream_measured_tube_length_missing" in blockers
    assert "test_tube_tactile_bundle_not_runtime_qualified" in blockers


def test_online_factories_defer_scene_and_tactile_evidence_but_not_calibration() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo / "configs/test_tube_closed_loop.live.template.json").read_text(encoding="utf-8")
    )
    scene = config["live_adapter"]["scene_interpreter"]
    scene["factory"] = ONLINE_SCENE_FACTORY
    tactile = config["live_adapter"]["tactile"]
    tactile["factory"] = SESSION_BOOTSTRAP_TACTILE_FACTORY

    result = assess_live_config(config)

    # These immutable frame/model inputs cannot be recovered from an image.
    assert "camera_extrinsic_artifact_missing" in result.static_blockers
    assert "hande_tcp_qualification_receipt_missing" in result.static_blockers
    assert "fr3_hande_collision_payload_qualification_receipt_missing" in result.static_blockers

    # Scene state, object size, tracker seeds and task-specific tactile gates
    # are acquired and validated by the resident run.
    assert "measured_tube_length_missing" not in result.static_blockers
    assert "T_socket_tube_entry_missing" not in result.static_blockers
    assert "static_socket_pose_qualification_receipt_missing" not in result.static_blockers
    assert (
        "test_tube_tactile_interpreter_qualification_receipt_missing" not in result.static_blockers
    )
    assert "runtime_tube_length_required" in result.runtime_requirements
    assert "runtime_socket_pose_and_entry_required" in result.runtime_requirements
    assert "runtime_tactile_contact_thresholds_required" in result.runtime_requirements


def test_process_flag_acknowledges_only_missing_collision_receipt_and_caps_speed() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo / "configs/test_tube_closed_loop.demo06.json").read_text(encoding="utf-8")
    )

    baseline = assess_live_config(config)
    accepted = assess_live_config(config, accept_arm_only_collision_model=True)
    blocker = "fr3_hande_collision_payload_qualification_receipt_missing"
    assert blocker in baseline.static_blockers
    assert blocker not in accepted.static_blockers
    assert "fixed_camera_extrinsic_qualification_receipt_missing" in (
        accepted.static_blockers
    )

    acknowledgement = reduced_collision_model_acknowledgement(config, accepted=True)
    assert acknowledgement["active"] is True
    assert acknowledgement["current_process_only"] is True
    assert acknowledgement["qualification_receipt_substituted"] is False
    assert acknowledgement["source"] == "--accept-arm-only-collision-model"
    assert acknowledgement["speed_limits"]["servo_max_linear_m_s"] == {
        "configured": 0.08,
        "effective": 0.03,
        "cap": 0.03,
    }
    assert acknowledgement["speed_limits"]["servo_max_angular_rad_s"] == {
        "configured": 0.8,
        "effective": 0.3,
        "cap": 0.3,
    }

    original_robot = config["live_adapter"]["robot"]
    effective_robot = _reduced_collision_robot_config(
        original_robot,
        acknowledgement=acknowledgement,
    )
    assert effective_robot["servo"]["max_linear_m_s"] == 0.03
    assert effective_robot["servo"]["max_angular_rad_s"] == 0.3
    assert effective_robot["moveit"]["velocity_scale"] == 0.03
    assert effective_robot["moveit"]["acceleration_scale"] == 0.03
    assert original_robot["servo"]["max_linear_m_s"] == 0.08


def test_process_flag_cannot_waive_invalid_collision_receipt(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo / "configs/test_tube_closed_loop.demo06.json").read_text(encoding="utf-8")
    )
    invalid = tmp_path / "collision-receipt.json"
    invalid.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
    config["live_adapter"]["robot"]["collision_payload_qualification_receipt"] = str(
        invalid
    )

    result = assess_live_config(config, accept_arm_only_collision_model=True)
    acknowledgement = reduced_collision_model_acknowledgement(config, accepted=True)

    assert "fr3_hande_collision_payload_not_runtime_qualified" in result.static_blockers
    assert acknowledgement["active"] is False
    assert acknowledgement["receipt_state"] == "invalid_or_unqualified_receipt"


def test_process_flag_accepts_only_bound_consensus_camera_recovery(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    formal_path = repo / "calibration/T_base_camera.npy"
    formal = np.load(formal_path, allow_pickle=False)
    recovered_path = tmp_path / "T_base_camera_recovered.npy"
    np.save(recovered_path, formal, allow_pickle=False)
    report = {
        "schema": "demo2gripper.provisional_static_rgbd_camera_recovery.v1",
        "source_formal_calibration": str(formal_path),
        "current_frame_count": 10,
        "selected_current_frames": [0, 1, 2, 3, 4, 5, 6, 7],
        "decision": {"plan_only_usable": True, "may_execute": False},
        "T_oldcam_newcam": np.eye(4).tolist(),
        "T_base_newcam_provisional": formal.tolist(),
        "final_rotation_deg": 0.79,
        "final_translation_mm": [2.0, 2.0, 1.0],
        "consensus_median_residual_mm": 5.8,
        "consensus_p90_residual_mm": 11.1,
        "cross_frame_rotation_deviation_deg": {"median": 0.19, "max": 0.78},
        "cross_frame_translation_deviation_mm": {"median": 5.2, "max": 21.5},
        "cross_reference_rotation_deviation_deg": {
            "median": 0.23,
            "p90": 1.04,
            "max": 2.11,
        },
        "cross_reference_translation_deviation_mm": {
            "median": 6.02,
            "p90": 23.9,
            "max": 55.5,
        },
    }
    report_path = tmp_path / "provisional-camera.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    config = {
        "live_adapter": {
            "camera": {
                "camera_extrinsic": str(recovered_path),
                "camera_extrinsic_qualification_receipt": None,
                "provisional_camera_recovery_report": str(report_path),
            }
        }
    }

    acknowledgement = provisional_camera_extrinsic_acknowledgement(
        config,
        accepted=True,
    )

    assert acknowledgement["active"] is True
    assert acknowledgement["current_process_only"] is True
    assert acknowledgement["qualification_receipt_substituted"] is False
    assert acknowledgement["derived_uncertainty"]["position_std_m"] == pytest.approx(
        0.00602
    )
    assert acknowledgement["derived_uncertainty"]["orientation_std_deg"] == pytest.approx(
        0.78
    )
    assert acknowledgement["outlier_statistics_retained_for_review"][
        "cross_reference_translation_deviation_mm"
    ]["max"] == 55.5

    rejected_report = dict(report)
    rejected_report["selected_current_frames"] = list(range(7))
    report_path.write_text(json.dumps(rejected_report), encoding="utf-8")
    rejected = provisional_camera_extrinsic_acknowledgement(config, accepted=True)
    assert rejected["active"] is False
    assert rejected["receipt_state"] == "provisional_evidence_invalid"
    assert (
        "provisional_camera_selected_frame_consensus_insufficient"
        in rejected["validation_reasons"]
    )


def test_provisional_camera_flag_removes_only_missing_camera_receipt() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo / "configs/test_tube_closed_loop.demo06.json").read_text(encoding="utf-8")
    )

    baseline = assess_live_config(config, accept_arm_only_collision_model=True)
    accepted = assess_live_config(
        config,
        accept_arm_only_collision_model=True,
        accept_provisional_camera_extrinsic=True,
    )

    camera_blocker = "fixed_camera_extrinsic_qualification_receipt_missing"
    assert camera_blocker in baseline.static_blockers
    assert camera_blocker not in accepted.static_blockers
    assert accepted.static_blockers == ()


def test_resident_scene_preflight_rejects_duplicate_geometry_and_calibration_drift(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo / "configs/test_tube_closed_loop.live.template.json").read_text(encoding="utf-8")
    )
    scene = config["live_adapter"]["scene_interpreter"]
    source = scene["config"]
    downstream = source["downstream_config"]
    different_camera_receipt = tmp_path / "different-camera-receipt.json"
    different_camera_receipt.write_text("{}", encoding="utf-8")
    different_socket = tmp_path / "different-socket.npy"
    np.save(different_socket, np.eye(4))
    source["camera_extrinsic_qualification_receipt"] = str(different_camera_receipt)
    downstream["static_socket_pose"] = str(different_socket)
    config["geometry"]["T_socket_tube_entry"] = np.eye(4).tolist()
    shifted_entry = np.eye(4)
    shifted_entry[2, 3] = 0.01
    downstream["T_socket_tube_entry"] = shifted_entry.tolist()
    config["geometry"]["normalized_demo_grasp"]["tube_length_m"] = 0.100
    downstream["policy"]["tube_length_m"] = 0.110
    downstream["policy"]["insertion_target_depth_m"] = 0.030

    blockers = preflight_live_config(config)
    assert "source_live_camera_receipt_mismatch" in blockers
    assert "downstream_live_socket_pose_mismatch" in blockers
    assert "downstream_controller_socket_entry_mismatch" in blockers
    assert "downstream_controller_tube_length_mismatch" in blockers
    assert "downstream_controller_insertion_depth_mismatch" in blockers


@pytest.mark.parametrize(
    "mode_args",
    [
        [],
        [
            "--execute",
            "--live-adapter",
            "demo2gripper.test_tube_live_io:create_live_io",
        ],
        ["--replay", "unused-replay.json"],
    ],
    ids=["dry-run", "execute", "replay"],
)
def test_unfilled_live_template_is_reported_before_geometry_or_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode_args: list[str],
) -> None:
    repo = Path(__file__).resolve().parents[1]
    runner_path = repo / "scripts/run_test_tube_closed_loop.py"
    spec = importlib.util.spec_from_file_location(
        f"test_tube_live_template_runner_{len(mode_args)}", runner_path
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def hardware_factory_forbidden(**_: object) -> object:
        raise AssertionError("live hardware factory must not run for the null template")

    monkeypatch.setattr(
        "demo2gripper.test_tube_live_io.create_live_io",
        hardware_factory_forbidden,
    )
    output = tmp_path / ("output-" + (mode_args[0].lstrip("-") if mode_args else "dry"))
    args = runner.parse_args(
        [
            "--config",
            str(repo / "configs/test_tube_closed_loop.live.template.json"),
            "--output-dir",
            str(output),
            *mode_args,
        ]
    )

    assert runner.run(args) == 2
    report = json.loads((output / "run_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked_live_execution"
    assert report["hardware_opened"] is False
    assert report["hardware_commands_dispatched"] is False
    assert "measured_tube_length_missing" in report["pose_backend"]["blockers"]
    assert "tube_length_provenance_missing" in report["pose_backend"]["blockers"]
    assert "T_socket_tube_entry_missing" in report["pose_backend"]["blockers"]
    assert report["pose_backend"]["preflight"]["static_live_adapter"] is True


def test_concrete_demo_contract_binds_one_contiguous_seven_stage_clip(
    tmp_path: Path,
) -> None:
    interval = {
        "schema": "demo2gripper.effective_demo_interval.v1",
        "status": "accepted_unique_complete_interval",
        "start_frame": 10,
        "end_frame_exclusive": 24,
    }
    index = {
        "schema": "demo2gripper.lossless_multimodal_index_view.v1",
        "frame_interval": [10, 24],
        "frame_count": 14,
    }
    names = (
        "APPROACH",
        "GRASP_AND_EXTRACT",
        "LOOSEN_AND_PIVOT",
        "RECLAMP",
        "APPROACH_RACK_ENTRY",
        "ALIGN_AND_INSERT",
        "RELEASE_AND_RETREAT",
    )
    review = {
        "schema": "demo2gripper.test_tube_stage_review.v2",
        "status": "preexecution_analysis_only",
        "hardware_commands_dispatched": False,
        "stages": [
            {"name": name, "frames_inclusive": [10 + 2 * i, 11 + 2 * i]}
            for i, name in enumerate(names)
        ],
    }
    plan = {
        "schema": "demo2gripper.test_tube_online_execution_plan.v1",
        "hardware_commands_dispatched": False,
        "stages": [{"index": i + 1} for i in range(7)],
    }
    paths = {}
    for name, payload in (
        ("effective_interval", interval),
        ("effective_index", index),
        ("stage_review", review),
        ("execution_plan", plan),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = str(path)
    contract = {
        "schema": "demo2gripper.test_tube_execution_demonstration.v1",
        **paths,
        "frame_interval": [10, 24],
        "source_axis_sign_primitive_qualified": False,
        "post_extract_level_controller_available": False,
    }

    assert _demonstration_contract_blockers(contract) == (
        "demo_post_extract_level_controller_missing",
        "demo_source_axis_sign_primitive_not_qualified",
    )

    contract["source_axis_sign_primitive_qualified"] = True
    contract["post_extract_level_controller_available"] = True
    assert _demonstration_contract_blockers(contract) == ()

    review["stages"][-1]["frames_inclusive"] = [23, 24]
    Path(paths["stage_review"]).write_text(json.dumps(review), encoding="utf-8")
    assert "demonstration_seven_stage_contract_mismatch" in (
        _demonstration_contract_blockers(contract)
    )
