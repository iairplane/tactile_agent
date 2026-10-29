"""Long-lived real-hardware adapter for the seven-stage test-tube policy.

The module owns the camera, tactile, Hand-E, and ROS handles for one complete
run.  Task-specific visual and tactile predicate producers are injected as
qualified, pure interpreters: this adapter never invents a tube end, a socket
pose, a tactile angle, or an insertion-jam threshold.

Hardware dependencies are imported only by :func:`create_live_io`.  Unit tests
exercise :class:`LiveTestTubeIO` with fakes and cannot open a device.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from .autonomous_pick import atomic_write_json
from .d435_live import D435Frame, D435LiveConfig, D435LiveSource
from .f2g_pose_service import PoseServiceClient, RemotePoseProvider
from .franka import validate_rigid_transform
from .gf220_live import GF220ClassifierWindows, GF220LiveConfig, GF220LiveSource
from .hande_modbus_cli import (
    MAX_FORCE_REGISTER,
    MAX_SPEED_REGISTER,
    SerialModbusRTU,
    _target_count,
    feedback_payload,
)
from .pose_observation import PoseObservation, PoseObservationGuardConfig
from .test_tube_closed_loop import (
    HELD_POSE_CAPABILITIES,
    SOCKET_POSE_CAPABILITIES,
    SOURCE_POSE_CAPABILITIES,
    ActionKind,
    ClosedLoopAction,
    ClosedLoopContractError,
    ClosedLoopObservation,
    GripperFeedback,
    ObjectPoseEstimate,
    PoseBackendDescriptor,
    PoseCapability,
    RobotFeedback,
    TactileFeedback,
)
from .test_tube_preflight import TubeState

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LIVE_IO_SCHEMA = "demo2gripper.test_tube_live_io.v1"
LIVE_QUALIFICATION_SCHEMA = "demo2gripper.test_tube_live_io_qualification.v1"
POSE_QUALIFICATION_SCHEMA = "demo2gripper.pose_backend_qualification.v1"

ONLINE_SCENE_FACTORY = (
    "demo2gripper.test_tube_online_scene:create_online_test_tube_scene_interpreter"
)
ONLINE_SOCKET_SCENE_FACTORY = (
    "demo2gripper.test_tube_online_socket:create_online_socket_scene_interpreter"
)
ONLINE_HELD_SCENE_FACTORY = (
    "demo2gripper.test_tube_downstream_scene:create_online_held_tube_scene_interpreter"
)
SESSION_BOOTSTRAP_TACTILE_FACTORY = (
    "demo2gripper.test_tube_tactile_interpreter:create_session_bootstrap_tactile_interpreter"
)

REDUCED_COLLISION_ACK_SCHEMA = (
    "demo2gripper.test_tube_reduced_collision_model_acknowledgement.v1"
)
REDUCED_COLLISION_SPEED_CAPS = {
    "moveit_velocity_scale": 0.03,
    "moveit_acceleration_scale": 0.03,
    "servo_max_linear_m_s": 0.03,
    "servo_max_angular_rad_s": 0.30,
}
_COLLISION_RECEIPT_MISSING = (
    "fr3_hande_collision_payload_qualification_receipt_missing"
)
PROVISIONAL_CAMERA_RECOVERY_SCHEMA = (
    "demo2gripper.provisional_static_rgbd_camera_recovery.v1"
)
PROVISIONAL_CAMERA_ACK_SCHEMA = (
    "demo2gripper.test_tube_provisional_camera_extrinsic_acknowledgement.v1"
)
_CAMERA_RECEIPT_MISSING = "fixed_camera_extrinsic_qualification_receipt_missing"
_FORMAL_CAMERA_EXTRINSIC = (REPOSITORY_ROOT / "calibration" / "T_base_camera.npy").resolve()
_PROVISIONAL_CAMERA_MIN_SELECTED_FRAMES = 8
_HANDE_COMMUNICATION_WATCHDOG_FAULT = 0x09
_HANDE_COMMUNICATION_WATCHDOG_RECOVERY_READS = 1
_PROVISIONAL_CAMERA_REQUIRED_FRAME_COUNT = 10
_PROVISIONAL_CAMERA_MAX_FINAL_ROTATION_DEG = 1.0
_PROVISIONAL_CAMERA_MAX_FINAL_TRANSLATION_M = 0.005
_PROVISIONAL_CAMERA_MAX_CROSS_FRAME_ROTATION_DEG = 1.0
_PROVISIONAL_CAMERA_MAX_POSITION_STD_M = 0.008
_PROVISIONAL_CAMERA_MAX_ORIENTATION_STD_DEG = 1.0


@dataclass(frozen=True)
class LivePreflightAssessment:
    """Static deployment blockers separated from online task requirements.

    ``static_blockers`` contains configuration that cannot be recovered from a
    live RGB-D/tactile observation: the fixed camera transform, Hand-E TCP and
    robot/driver model, device identities, and executable backend wiring.
    ``runtime_requirements`` names evidence that the resident controller must
    acquire and validate after its sensors are open and before the first motion
    that depends on it.  A runtime requirement is never treated as permission
    to guess a value; failure to acquire it must end in STOP/HOLD.
    """

    static_blockers: tuple[str, ...]
    runtime_requirements: tuple[str, ...]


def _servo_status_fault(code: int, message: str) -> str | None:
    """Translate hard Servo stops into faults, while allowing safe deceleration.

    MoveIt Servo status 1/3 are active speed reductions on approach to, or
    motion away from, a singularity.  The controller is still commanding a
    bounded velocity in both cases.  Treating either warning as a latched
    hardware fault prevents the slow motion that Servo is explicitly applying.
    """

    numeric = int(code)
    if numeric in {0, 1, 3}:
        return None
    detail = str(message).strip() or "no_detail"
    return f"moveit_servo_status:{numeric}:{detail}"


def _require_twist_mode_response(response: object | None) -> None:
    """Require an affirmative ServoCommandType response."""

    if response is None or getattr(response, "success", False) is not True:
        raise LiveIOError("MoveIt Servo rejected TWIST command mode")


def _require_pause_response(response: object | None, *, paused: bool) -> None:
    """Require an affirmative MoveIt Servo pause-state response."""

    if response is None or getattr(response, "success", False) is not True:
        state = "pause" if paused else "resume"
        raise LiveIOError(f"MoveIt Servo rejected {state} request")


class LiveIOError(RuntimeError):
    """A live device, provider, or command violated the adapter contract."""


@dataclass(frozen=True)
class SceneObjectMask:
    """One temporally tracked mask and the evidence needed around F2G-Pose."""

    object_id: str
    mask: np.ndarray
    mask_id: str
    mask_iou: float
    tracking_id: str
    occluded_by_gripper: bool = False

    def __post_init__(self) -> None:
        if self.object_id not in {"tube", "socket"}:
            raise ValueError("object_id must be tube or socket")
        value = np.asarray(self.mask)
        if value.ndim != 2 or not np.issubdtype(value.dtype, np.bool_):
            raise ValueError("mask must be a boolean HxW array")
        if not value.any():
            raise ValueError("mask must contain at least one foreground pixel")
        for name in ("mask_id", "tracking_id"):
            item = getattr(self, name)
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not math.isfinite(self.mask_iou) or not 0.0 <= self.mask_iou <= 1.0:
            raise ValueError("mask_iou must lie in [0, 1]")


@dataclass(frozen=True)
class QualifiedCameraPose:
    """Task-qualified canonical pose produced from an F2G proposal."""

    T_camera_object: np.ndarray
    position_std_m: float
    orientation_std_deg: float
    axis_sign_source: str = "not_applicable"
    cylindrical_roll_canonicalized: bool = False
    size_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        transform = validate_rigid_transform(
            self.T_camera_object, name="qualified T_camera_object"
        ).copy()
        transform.setflags(write=False)
        object.__setattr__(self, "T_camera_object", transform)
        for name in ("position_std_m", "orientation_std_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.axis_sign_source not in {
            "rim_detector",
            "open_end_detector",
            "source_support_geometry",
            "source_support_anchor",
            "qualified_base_z",
            "temporal_continuity",
            "not_applicable",
        }:
            raise ValueError("unsupported axis_sign_source")
        if self.size_m is not None:
            size = tuple(float(value) for value in self.size_m)
            if len(size) != 3 or not np.isfinite(size).all() or min(size) <= 0.0:
                raise ValueError("size_m must contain three positive finite lengths")
            object.__setattr__(self, "size_m", size)


@dataclass(frozen=True)
class QualifiedBasePose:
    """A held-tube pose recovered from qualified RGB/tactile geometry."""

    T_base_object: np.ndarray
    position_std_m: float
    orientation_std_deg: float
    tracking_id: str
    evidence_id: str
    axis_sign_source: str
    cylindrical_roll_canonicalized: bool = True

    def __post_init__(self) -> None:
        transform = validate_rigid_transform(
            self.T_base_object, name="qualified held T_base_object"
        ).copy()
        transform.setflags(write=False)
        object.__setattr__(self, "T_base_object", transform)
        for name in ("position_std_m", "orientation_std_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("tracking_id", "evidence_id", "axis_sign_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class SceneEvidence:
    """Scene predicates from one fresh RGB-D frame."""

    tube: SceneObjectMask | None
    socket: SceneObjectMask | None
    front_rgb_angle_target_reached: bool = False
    tube_clear_of_source: bool = False
    tube_tip_at_socket_entry: bool = False
    tube_socket_axes_aligned: bool = False
    tube_stays_supported: bool = False


class SceneInterpreter(Protocol):
    """Qualified mask tracker and task-specific RGB pose disambiguator."""

    def observe(
        self,
        *,
        frame: D435Frame,
        state: TubeState,
        T_base_tcp: np.ndarray,
    ) -> SceneEvidence: ...

    def qualify_pose(
        self,
        *,
        object_id: str,
        proposal: PoseObservation,
        mask: SceneObjectMask,
        frame: D435Frame,
    ) -> QualifiedCameraPose | None: ...

    def held_tube_pose(
        self,
        *,
        frame: D435Frame,
        scene: SceneEvidence,
        state: TubeState,
        T_base_tcp: np.ndarray,
        tactile: TactileFeedback,
    ) -> QualifiedBasePose | None: ...

    def set_grasp_relation(
        self,
        T_tube_tcp_actual: np.ndarray,
        *,
        T_base_tube_at_contact: np.ndarray,
    ) -> None:
        """Bind the realized contact relation before any extraction motion."""
        ...

    def reset_support_history_for_release(self) -> None:
        """Discard support evidence collected before the release command."""
        ...

    def close(self) -> None: ...


class TactileInterpreter(Protocol):
    """Session-bound promotion of GF220 windows into qualified predicates."""

    def evaluate(
        self,
        *,
        windows: GF220ClassifierWindows,
        state: TubeState,
        now_s: float,
        T_base_tcp: np.ndarray,
        gripper: GripperFeedback,
        commanded_insertion_step_m: float | None,
        measured_insertion_progress_m: float | None,
    ) -> TactileFeedback: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ArmState:
    T_base_tcp: np.ndarray
    motion_complete: bool
    servo_supported: bool
    fault: str | None = None


@dataclass(frozen=True)
class _PendingRecenter:
    local_x_offset_m: float
    action_id: str
    open_target_width_m: float
    commanded_at_s: float


@dataclass(frozen=True)
class _PreCloseSourcePose:
    T_base_tube: np.ndarray
    T_base_tcp: np.ndarray
    observed_at_s: float
    deadline_s: float
    tracking_id: str
    position_std_m: float
    orientation_std_deg: float
    max_displacement_m: float = 0.0025


@dataclass(frozen=True)
class _FinalContactApproach:
    source: _PreCloseSourcePose
    T_base_target: np.ndarray
    T_base_start: np.ndarray
    T_base_arrival: np.ndarray | None = None
    depth_loss_handoff: bool = False


class ArmRuntime(Protocol):
    """Long-lived FR3 state reader plus coarse MoveIt and Servo command sink."""

    def state(self) -> ArmState: ...

    def move_coarse(self, *, target: np.ndarray, action_id: str) -> Mapping[str, Any]: ...

    def servo_target(
        self, *, target: np.ndarray, action_id: str,
        position_tolerance_m: float | None = None,
    ) -> Mapping[str, Any]: ...

    def stop(self, *, reason: str) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class GripperRuntime(Protocol):
    def feedback(self) -> GripperFeedback: ...

    def move_width(
        self, *, width_m: float, speed: int, force: int, action_id: str
    ) -> Mapping[str, Any]: ...

    def stop(self, *, reason: str) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class HandEModbusRuntime:
    """One long-lived Hand-E Modbus session; it never auto-activates the gripper."""

    def __init__(self, transport: SerialModbusRTU) -> None:
        self._transport = transport
        self._lock = threading.Lock()

    def feedback(self) -> GripperFeedback:
        with self._lock:
            raw = self._transport.read_feedback()
            # A Hand-E left without Modbus traffic can report the communication
            # watchdog (0x09) on the first read even though that read has already
            # restored communication.  Re-read once while holding the session
            # lock.  Every other fault, and a persistent 0x09, remains fail-closed.
            for _ in range(_HANDE_COMMUNICATION_WATCHDOG_RECOVERY_READS):
                if raw.fault_code != _HANDE_COMMUNICATION_WATCHDOG_FAULT:
                    break
                raw = self._transport.read_feedback()
        payload = feedback_payload(raw)
        return GripperFeedback(
            width_m=float(payload["opening_width_mm"]) / 1000.0,
            moving=bool(raw.go_to_bit == 1 and raw.object_status == 0),
            fault_free=bool(
                raw.fault_code == 0 and raw.activation_bit == 1 and raw.activation_status == 3
            ),
            object_detected=bool(raw.object_status == 2),
            open_confirmed=bool(
                raw.object_status == 3 and float(payload["opening_width_mm"]) >= 48.0
            ),
        )

    def move_width(
        self, *, width_m: float, speed: int, force: int, action_id: str
    ) -> Mapping[str, Any]:
        if not math.isfinite(width_m) or not 0.0 <= width_m <= 0.050:
            raise LiveIOError("Hand-E width must lie in [0, 0.050] m")
        if not 1 <= speed <= MAX_SPEED_REGISTER:
            raise LiveIOError("Hand-E speed register is outside its supported range")
        if not 1 <= force <= MAX_FORCE_REGISTER:
            raise LiveIOError("Hand-E force register is outside its supported range")
        target_count, _ = _target_count("move", width_m * 1000.0)
        with self._lock:
            self._transport.write_motion(target_count, speed, force)
        return {
            "accepted": True,
            "hardware_commands_dispatched": True,
            "action_id": action_id,
            "target_width_m": width_m,
            "target_count": target_count,
        }

    def stop(self, *, reason: str) -> Mapping[str, Any]:
        with self._lock:
            self._transport.stop_motion()
        return {
            "accepted": True,
            "hardware_commands_dispatched": True,
            "reason": reason,
            "command": "clear_rGTO_hold_activation",
        }

    def close(self) -> None:
        self._transport.close()


class RosMoveItServoRuntime:
    """Persistent TF/joint-state reader with an actual MoveIt Servo twist loop.

    Coarse collision-aware goals delegate to the audited pose-goal tool.  Servo
    commands never delegate to that blocking tool: they update a target followed
    by this class's fixed-rate TwistStamped loop.  Missing/inactive MoveIt Servo
    therefore fails closed instead of degrading to MoveIt microsteps.
    """

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        T_link8_tcp: np.ndarray,
        output_dir: Path,
    ) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import TwistStamped
            from moveit_msgs.msg import ServoStatus
            from moveit_msgs.srv import ServoCommandType
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.node import Node
            from rclpy.time import Time
            from sensor_msgs.msg import JointState
            from std_srvs.srv import SetBool
            from tf2_ros import Buffer, TransformListener
        except ImportError as error:  # pragma: no cover - requires ROS host
            raise LiveIOError(f"ROS 2 Python dependencies are unavailable: {error}") from error

        self._rclpy = rclpy
        self._Time = Time
        self._TwistStamped = TwistStamped
        self._ServoCommandType = ServoCommandType
        self._SetBool = SetBool
        self._output_dir = output_dir
        self._T_link8_tcp = validate_rigid_transform(T_link8_tcp, name="T_link8_tcp").copy()
        self._base_frame = _string(config.get("base_frame", "fr3_link0"), "base_frame")
        self._link8_frame = _string(config.get("link8_frame", "fr3_link8"), "link8_frame")
        self._joint_topic = _string(config.get("joint_topic", "/joint_states"), "joint_topic")
        servo = _mapping(config.get("servo"), "robot.servo")
        self._servo_node_name = _string(
            servo.get("node_name", "/servo_node"), "robot.servo.node_name"
        )
        self._twist_topic = _string(
            servo.get("twist_topic", "/servo_node/delta_twist_cmds"),
            "robot.servo.twist_topic",
        )
        self._status_topic = _string(
            servo.get("status_topic", "/servo_node/status"),
            "robot.servo.status_topic",
        )
        self._switch_command_type_service = _string(
            servo.get(
                "switch_command_type_service",
                "/servo_node/switch_command_type",
            ),
            "robot.servo.switch_command_type_service",
        )
        self._pause_service = _string(
            servo.get("pause_service", "/servo_node/pause_servo"),
            "robot.servo.pause_service",
        )
        self._servo_period_s = _positive(
            servo.get("command_period_s", 0.02), "robot.servo.command_period_s"
        )
        self._servo_horizon_s = _positive(
            servo.get("command_horizon_s", 0.10), "robot.servo.command_horizon_s"
        )
        self._max_linear_m_s = _positive(
            servo.get("max_linear_m_s", 0.08), "robot.servo.max_linear_m_s"
        )
        self._max_angular_rad_s = _positive(
            servo.get("max_angular_rad_s", 0.8), "robot.servo.max_angular_rad_s"
        )
        self._max_linear_acceleration_m_s2 = _positive(
            servo.get("max_linear_acceleration_m_s2", 0.05),
            "robot.servo.max_linear_acceleration_m_s2",
        )
        self._max_angular_acceleration_rad_s2 = _positive(
            servo.get("max_angular_acceleration_rad_s2", 0.3),
            "robot.servo.max_angular_acceleration_rad_s2",
        )
        self._twist_lock = threading.Lock()
        self._last_twist = np.zeros(6, dtype=float)
        self._last_twist_time_s: float | None = None
        self._position_tolerance_m = _positive(
            servo.get("position_tolerance_m", 0.0015),
            "robot.servo.position_tolerance_m",
        )
        self._rotation_tolerance_rad = _positive(
            servo.get("rotation_tolerance_rad", math.radians(1.5)),
            "robot.servo.rotation_tolerance_rad",
        )
        self._startup_state_timeout_s = _positive(
            servo.get("startup_state_timeout_s", 5.0),
            "robot.servo.startup_state_timeout_s",
        )
        self._joint_state_max_age_s = _positive(
            config.get("joint_state_max_age_s", 0.25), "robot.joint_state_max_age_s"
        )
        self._tf_timeout_s = _positive(config.get("tf_timeout_s", 0.20), "robot.tf_timeout_s")
        self._target_lock = threading.Lock()
        self._servo_target: np.ndarray | None = None
        self._active_position_tolerance_m = self._position_tolerance_m
        self._joint_state_condition = threading.Condition()
        self._last_joint_state_s: float | None = None
        self._last_joint_stamp_ns: int | None = None
        self._joint_state_refresh_timeout_s = 0.5
        self._last_joint_velocity_norm = math.inf
        self._last_servo_error: str | None = None
        self._servo_status_lock = threading.Lock()
        self._servo_status_code: int | None = None
        self._servo_status_message: str | None = None
        self._servo_status_latched_fault: str | None = None
        self._closed = False

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False
        self._node = Node(f"test_tube_live_io_{os.getpid()}")
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self._node, spin_thread=False)
        self._publisher = self._node.create_publisher(TwistStamped, self._twist_topic, 10)

        def servo_status_callback(message: Any) -> None:
            code = int(message.code)
            detail = str(message.message)
            fault = _servo_status_fault(code, detail)
            with self._servo_status_lock:
                self._servo_status_code = code
                self._servo_status_message = detail
                if fault is not None:
                    self._servo_status_latched_fault = fault

        self._servo_status_subscription = self._node.create_subscription(
            ServoStatus,
            self._status_topic,
            servo_status_callback,
            10,
        )
        self._switch_command_type_client = self._node.create_client(
            ServoCommandType,
            self._switch_command_type_service,
        )
        self._pause_client = self._node.create_client(SetBool, self._pause_service)
        self._servo_paused = False

        self._subscription = self._node.create_subscription(
            JointState, self._joint_topic, self._receive_joint_state, 1
        )
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="test-tube-ros-executor",
            daemon=True,
        )
        self._spin_thread.start()
        self._servo_thread = threading.Thread(
            target=self._servo_loop,
            name="test-tube-moveit-servo",
            daemon=True,
        )
        self._servo_thread.start()
        self._moveit = _mapping(config.get("moveit"), "robot.moveit")
        try:
            self._assert_servo_runtime()
            self._select_twist_command_type()
            self._set_servo_paused(False)
            self._wait_for_initial_state()
        except BaseException:
            self.close()
            raise

    def _assert_servo_runtime(self) -> None:
        try:
            from ament_index_python.packages import get_package_prefix

            get_package_prefix("moveit_servo")
        except Exception as error:  # pragma: no cover - host state
            raise LiveIOError("moveit_servo package is not installed") from error
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            names = {
                f"{namespace.rstrip('/')}/{name}".replace("//", "/")
                for name, namespace in self._node.get_node_names_and_namespaces()
            }
            if (
                self._servo_node_name in names
                and self._node.count_subscribers(self._twist_topic) > 0
                and self._node.count_publishers(self._status_topic) > 0
            ):
                return
            time.sleep(0.05)
        raise LiveIOError(
            "MoveIt Servo is not active, does not subscribe to "
            + self._twist_topic
            + ", or does not publish "
            + self._status_topic
        )

    def _select_twist_command_type(self) -> None:
        timeout_s = 3.0
        if not self._switch_command_type_client.wait_for_service(timeout_sec=timeout_s):
            raise LiveIOError(
                "MoveIt Servo command-type service is unavailable: "
                + self._switch_command_type_service
            )
        request = self._ServoCommandType.Request()
        request.command_type = self._ServoCommandType.Request.TWIST
        future = self._switch_command_type_client.call_async(request)
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise LiveIOError("MoveIt Servo TWIST command-mode request timed out")
        _require_twist_mode_response(future.result())

    def _set_servo_paused(self, paused: bool) -> None:
        """Synchronously change Servo's publisher state before controller handoff."""

        timeout_s = 3.0
        if not self._pause_client.wait_for_service(timeout_sec=timeout_s):
            raise LiveIOError("MoveIt Servo pause service is unavailable: " + self._pause_service)
        request = self._SetBool.Request()
        request.data = bool(paused)
        future = self._pause_client.call_async(request)
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            state = "pause" if paused else "resume"
            raise LiveIOError(f"MoveIt Servo {state} request timed out")
        _require_pause_response(future.result(), paused=paused)
        self._servo_paused = bool(paused)

    def _wait_for_initial_state(self) -> None:
        """Do not expose the runtime until joint state and TCP TF are usable."""

        deadline = time.monotonic() + self._startup_state_timeout_s
        fault = "joint_state_unavailable"
        while time.monotonic() < deadline:
            now = time.monotonic()
            if self._last_joint_state_s is None:
                fault = "joint_state_unavailable"
            elif now - self._last_joint_state_s > self._joint_state_max_age_s:
                fault = "joint_state_stale"
            else:
                try:
                    self._lookup_tcp()
                except Exception as error:  # pragma: no cover - host-specific TF detail
                    fault = f"tcp_tf_unavailable:{error}"
                else:
                    return
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        raise LiveIOError(
            "ROS arm state did not become ready before live observation: " + fault
        )

    def _lookup_tcp(self) -> np.ndarray:
        from rclpy.duration import Duration

        transform = self._buffer.lookup_transform(
            self._base_frame,
            self._link8_frame,
            self._Time(),
            timeout=Duration(seconds=self._tf_timeout_s),
        )
        translation = transform.transform.translation
        quaternion = transform.transform.rotation
        T_base_link8 = np.eye(4, dtype=float)
        T_base_link8[:3, :3] = Rotation.from_quat(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        ).as_matrix()
        T_base_link8[:3, 3] = [translation.x, translation.y, translation.z]
        return validate_rigid_transform(
            T_base_link8 @ self._T_link8_tcp, name="measured T_base_tcp"
        )

    def _receive_joint_state(self, message: Any) -> bool:
        positions = np.asarray(message.position, dtype=float)
        velocities = np.asarray(message.velocity, dtype=float)
        if positions.size < 7 or not np.isfinite(positions[:7]).all():
            return False
        stamp = message.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        source_age_s = (self._node.get_clock().now().nanoseconds - stamp_ns) * 1e-9
        if source_age_s > self._joint_state_max_age_s or source_age_s < -0.05:
            return False
        with self._joint_state_condition:
            if self._last_joint_stamp_ns is not None and stamp_ns <= self._last_joint_stamp_ns:
                return False
            self._last_joint_stamp_ns = stamp_ns
            # Preserve source age instead of treating delayed queued callbacks
            # as fresh just because Python has only now processed them.
            self._last_joint_state_s = time.monotonic() - max(0.0, source_age_s)
            self._last_joint_velocity_norm = (
                float(np.linalg.norm(velocities[:7]))
                if velocities.size >= 7 and np.isfinite(velocities[:7]).all()
                else math.inf
            )
            self._joint_state_condition.notify_all()
        return True

    def _refresh_joint_state_if_stale(self) -> None:
        if self._last_joint_state_s is not None and (
            time.monotonic() - self._last_joint_state_s <= self._joint_state_max_age_s
        ):
            return
        with self._target_lock:
            self._servo_target = None
        self._publish_zero()
        deadline = time.monotonic() + self._joint_state_refresh_timeout_s
        # Waiting on the condition releases the GIL so the ROS executor can
        # recover from a slow camera/tactile inference callback on this process.
        with self._joint_state_condition:
            while self._last_joint_state_s is None or (
                time.monotonic() - self._last_joint_state_s > self._joint_state_max_age_s
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return
                self._joint_state_condition.wait(timeout=remaining)

    def state(self) -> ArmState:
        self._refresh_joint_state_if_stale()
        now = time.monotonic()
        with self._servo_status_lock:
            status_fault = self._servo_status_latched_fault
        fault = status_fault or self._last_servo_error
        if self._last_joint_state_s is None:
            fault = fault or "joint_state_unavailable"
        elif now - self._last_joint_state_s > self._joint_state_max_age_s:
            fault = fault or "joint_state_stale"
        try:
            tcp = self._lookup_tcp()
        except Exception as error:  # pragma: no cover - host state
            tcp = np.eye(4)
            fault = fault or f"tcp_tf_unavailable:{error}"
        with self._target_lock:
            target = self._servo_target
            position_tolerance_m = self._active_position_tolerance_m
        complete = target is None
        if target is not None and fault is None:
            translation, rotation = _pose_error(tcp, target)
            complete = bool(
                translation <= position_tolerance_m
                and rotation <= self._rotation_tolerance_rad
                and self._last_joint_velocity_norm <= 0.03
            )
            if complete:
                with self._target_lock:
                    if self._servo_target is target:
                        self._servo_target = None
                    else:
                        complete = False
        return ArmState(tcp, complete, True, fault)

    def move_coarse(self, *, target: np.ndarray, action_id: str) -> Mapping[str, Any]:
        target = validate_rigid_transform(target, name="coarse T_base_tcp target")
        with self._target_lock:
            self._servo_target = None
        self._publish_zero()
        # Servo and MoveIt execute through the same trajectory controller.  A
        # confirmed pause prevents Servo's timeout/halt trajectory from racing
        # the blocking MoveIt goal.  Unpausing resets Servo smoothing from the
        # post-MoveIt joint state.
        self._set_servo_paused(True)
        try:
            # Keep every fallible setup step inside the pause guard.  A bad
            # path/configuration must not leave Servo paused for the rest of
            # the resident session.
            time.sleep(max(self._servo_horizon_s, 2.0 * self._servo_period_s))
            script = _path(
                self._moveit.get("pose_goal_script"),
                "robot.moveit.pose_goal_script",
            )
            setup_scripts = _path_list(
                self._moveit.get("setup_scripts", []), "robot.moveit.setup_scripts"
            )
            disable_pose_delta_guard = self._moveit.get(
                "disable_pose_delta_guard", False
            )
            if not isinstance(disable_pose_delta_guard, bool):
                raise LiveIOError(
                    "robot.moveit.disable_pose_delta_guard must be a boolean"
                )
            disable_joint_sweep_guards = self._moveit.get(
                "disable_joint_sweep_guards", False
            )
            if not isinstance(disable_joint_sweep_guards, bool):
                raise LiveIOError(
                    "robot.moveit.disable_joint_sweep_guards must be a boolean"
                )
            output = self._output_dir / "arm" / f"{action_id}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            quaternion = Rotation.from_matrix(target[:3, :3]).as_quat()
            offset_quaternion = Rotation.from_matrix(self._T_link8_tcp[:3, :3]).as_quat()
            arguments = [
                sys.executable,
                str(script),
                "--target-position-m",
                *(f"{value:.12g}" for value in target[:3, 3]),
                "--target-orientation-xyzw",
                *(f"{value:.12g}" for value in quaternion),
                "--target-frame",
                "configured-ee",
                "--link8-to-configured-ee-position-m",
                *(f"{value:.12g}" for value in self._T_link8_tcp[:3, 3]),
                "--link8-to-configured-ee-orientation-xyzw",
                *(f"{value:.12g}" for value in offset_quaternion),
                "--velocity-scale",
                str(self._moveit.get("velocity_scale", 0.03)),
                "--acceleration-scale",
                str(self._moveit.get("acceleration_scale", 0.03)),
                *(["--disable-pose-delta-guard"] if disable_pose_delta_guard else []),
                *(
                    ["--disable-joint-sweep-guards"]
                    if disable_joint_sweep_guards
                    else []
                ),
                "--output",
                str(output),
                "--execute",
                "--accept-arm-only-collision-model",
            ]
            shell = " && ".join(
                [
                    *(f"source {shlex.quote(str(item))}" for item in setup_scripts),
                    shlex.join(arguments),
                ]
            )
            timeout_s = _positive(
                self._moveit.get("command_timeout_s", 60.0),
                "robot.moveit.command_timeout_s",
            )
            completed = subprocess.run(
                ["bash", "-lc", shell],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        finally:
            self._set_servo_paused(False)
            self._select_twist_command_type()
        if completed.returncode != 0 or not output.is_file():
            detail = (completed.stderr or completed.stdout)[-2000:]
            raise LiveIOError(f"coarse MoveIt action failed ({completed.returncode}): {detail}")
        receipt = json.loads(output.read_text(encoding="utf-8"))
        if receipt.get("execution_succeeded") is not True:
            raise LiveIOError("coarse MoveIt receipt did not confirm execution success")
        return {
            "accepted": True,
            "hardware_commands_dispatched": True,
            "action_id": action_id,
            "moveit_receipt": str(output),
            "measured_position_error_m": receipt.get("measured_target_position_error_m"),
            "measured_rotation_error_rad": receipt.get("measured_target_rotation_error_rad"),
        }

    def servo_target(
        self, *, target: np.ndarray, action_id: str,
        position_tolerance_m: float | None = None,
    ) -> Mapping[str, Any]:
        target = validate_rigid_transform(target, name="servo T_base_tcp target").copy()
        tolerance = self._position_tolerance_m if position_tolerance_m is None else _positive(
            position_tolerance_m, "servo target position_tolerance_m"
        )
        if self._servo_paused:
            raise LiveIOError("MoveIt Servo is paused; refusing Cartesian target")
        with self._target_lock:
            self._servo_target = target
            self._active_position_tolerance_m = tolerance
        return {
            "accepted": True,
            "hardware_commands_dispatched": True,
            "action_id": action_id,
            "controller": "moveit_servo_twist_closed_loop",
            "target": target.tolist(),
            "position_tolerance_m": tolerance,
        }

    def _servo_cycle(self) -> None:
        with self._target_lock:
            target = self._servo_target
            position_tolerance_m = self._active_position_tolerance_m
        if target is not None:
            try:
                current = self._lookup_tcp()
                translation, angle = _pose_error(current, target)
                if (
                    translation <= position_tolerance_m
                    and angle <= self._rotation_tolerance_rad
                ):
                    self._publish_zero()
                else:
                    self._publish_twist(current, target)
                self._last_servo_error = None
            except Exception as error:
                self._last_servo_error = f"servo_loop:{error}"
                self._publish_zero()

    def _servo_loop(self) -> None:  # pragma: no cover - requires ROS host
        while not self._closed:
            started = time.monotonic()
            self._servo_cycle()
            delay = self._servo_period_s - (time.monotonic() - started)
            if delay > 0.0:
                time.sleep(delay)

    def _publish_twist(self, current: np.ndarray, target: np.ndarray) -> None:
        with self._servo_status_lock:
            status_fault = self._servo_status_latched_fault
        if status_fault is not None:
            raise LiveIOError(status_fault)
        with self._twist_lock:
            now = time.monotonic()
            # A paused producer must ramp up again, not spend its whole idle
            # interval as an acceleration allowance on the first command.
            dt = self._servo_period_s if self._last_twist_time_s is None else min(
                self._servo_period_s, max(0.0, now - self._last_twist_time_s)
            )
            twist = _tcp_servo_twist_at_link8(
                current=current,
                target=target,
                T_link8_tcp=self._T_link8_tcp,
                horizon_s=self._servo_horizon_s,
                max_linear_m_s=self._max_linear_m_s,
                max_angular_rad_s=self._max_angular_rad_s,
                previous_twist=self._last_twist,
                dt_s=dt,
                max_linear_acceleration_m_s2=self._max_linear_acceleration_m_s2,
                max_angular_acceleration_rad_s2=self._max_angular_acceleration_rad_s2,
            )
            message = self._TwistStamped()
            message.header.stamp = self._node.get_clock().now().to_msg()
            message.header.frame_id = self._base_frame
            message.twist.linear.x, message.twist.linear.y, message.twist.linear.z = twist[:3]
            message.twist.angular.x, message.twist.angular.y, message.twist.angular.z = twist[3:]
            self._publisher.publish(message)
            self._last_twist = twist
            self._last_twist_time_s = now

    def _publish_zero(self) -> None:
        with self._twist_lock:
            message = self._TwistStamped()
            message.header.stamp = self._node.get_clock().now().to_msg()
            message.header.frame_id = self._base_frame
            self._publisher.publish(message)
            self._last_twist = np.zeros(6, dtype=float)
            self._last_twist_time_s = time.monotonic()

    def stop(self, *, reason: str) -> Mapping[str, Any]:
        with self._target_lock:
            self._servo_target = None
        self._publish_zero()
        return {
            "accepted": True,
            "hardware_commands_dispatched": True,
            "reason": reason,
            "controller": "moveit_servo_zero_twist",
        }

    def close(self) -> None:
        if self._closed:
            return
        self.stop(reason="adapter_close")
        self._closed = True
        self._servo_thread.join(timeout=2.0)
        self._executor.shutdown(timeout_sec=2.0)
        self._spin_thread.join(timeout=2.0)
        self._listener.unregister()
        self._node.destroy_node()
        if self._owns_rclpy:
            self._rclpy.shutdown()


class LiveTestTubeIO:
    """Concrete resident adapter implementing ``ClosedLoopIO``."""

    def __init__(
        self,
        *,
        descriptor: PoseBackendDescriptor,
        source_descriptor: PoseBackendDescriptor,
        held_descriptor: PoseBackendDescriptor,
        socket_descriptor: PoseBackendDescriptor,
        camera: Any,
        tactile_source: Any,
        scene: SceneInterpreter,
        tactile: TactileInterpreter,
        arm: ArmRuntime,
        gripper: GripperRuntime,
        tube_pose_provider: Any,
        T_base_camera: np.ndarray,
        T_tube_tcp_demo: np.ndarray | None,
        T_base_socket: np.ndarray | None,
        camera_position_std_m: float,
        camera_orientation_std_deg: float,
        tcp_position_std_m: float,
        tcp_orientation_std_deg: float,
        demo_relation_position_std_m: float,
        demo_relation_orientation_std_deg: float,
        held_position_std_limit_m: float,
        held_orientation_std_limit_deg: float,
        socket_position_std_m: float,
        socket_orientation_std_deg: float,
        tactile_samples: int = 4,
        tactile_timeout_s: float = 0.25,
        tactile_cache_max_age_s: float = 0.20,
        close_speed: int = 80,
        close_force: int = 15,
        open_speed: int = 80,
        open_force: int = 10,
        recenter_open_width_tolerance_m: float = 0.00075,
        recenter_open_timeout_s: float = 3.0,
        source_pose_cache_max_age_s: float = 0.20,
        close_timeout_s: float = 6.0,
        final_approach_commit_distance_m: float | None = None,
        final_approach_timeout_s: float = 8.0,
        servo_translation_deadband_m: float = 0.0025,
        servo_rotation_deadband_deg: float = 2.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._descriptor = descriptor
        self._source_descriptor = source_descriptor
        self._held_descriptor = held_descriptor
        self._socket_descriptor = socket_descriptor
        self._camera = camera
        self._tactile_source = tactile_source
        self._scene = scene
        self._tactile = tactile
        self._arm = arm
        self._gripper = gripper
        self._tube_provider = tube_pose_provider
        self._T_base_camera = validate_rigid_transform(T_base_camera, name="T_base_camera").copy()
        self._T_tube_tcp_demo = (
            None
            if T_tube_tcp_demo is None
            else validate_rigid_transform(T_tube_tcp_demo, name="T_tube_tcp_demo").copy()
        )
        self._T_base_socket = (
            None
            if T_base_socket is None
            else validate_rigid_transform(T_base_socket, name="T_base_socket").copy()
        )
        self._camera_position_std_m = _nonnegative(camera_position_std_m, "camera_position_std_m")
        self._camera_orientation_std_deg = _nonnegative(
            camera_orientation_std_deg, "camera_orientation_std_deg"
        )
        self._tcp_position_std_m = _nonnegative(tcp_position_std_m, "tcp_position_std_m")
        self._tcp_orientation_std_deg = _nonnegative(
            tcp_orientation_std_deg, "tcp_orientation_std_deg"
        )
        self._demo_relation_position_std_m = _nonnegative(
            demo_relation_position_std_m, "demo_relation_position_std_m"
        )
        self._demo_relation_orientation_std_deg = _nonnegative(
            demo_relation_orientation_std_deg, "demo_relation_orientation_std_deg"
        )
        self._held_position_std_limit_m = _nonnegative(
            held_position_std_limit_m, "held_position_std_limit_m"
        )
        self._held_orientation_std_limit_deg = _nonnegative(
            held_orientation_std_limit_deg, "held_orientation_std_limit_deg"
        )
        self._socket_position_std_m = _nonnegative(socket_position_std_m, "socket_position_std_m")
        self._socket_orientation_std_deg = _nonnegative(
            socket_orientation_std_deg, "socket_orientation_std_deg"
        )
        self._tactile_samples = int(tactile_samples)
        self._tactile_timeout_s = float(tactile_timeout_s)
        self._tactile_cache_max_age_s = _positive(
            tactile_cache_max_age_s, "tactile_cache_max_age_s"
        )
        self._close_speed = int(close_speed)
        self._close_force = int(close_force)
        self._open_speed = int(open_speed)
        self._open_force = int(open_force)
        self._recenter_open_width_tolerance_m = _positive(
            recenter_open_width_tolerance_m,
            "recenter_open_width_tolerance_m",
        )
        self._recenter_open_timeout_s = _positive(
            recenter_open_timeout_s, "recenter_open_timeout_s"
        )
        self._source_pose_cache_max_age_s = _positive(
            source_pose_cache_max_age_s, "source_pose_cache_max_age_s"
        )
        self._close_timeout_s = _positive(close_timeout_s, "close_timeout_s")
        self._final_approach_commit_distance_m = (
            None if final_approach_commit_distance_m is None
            else _positive(final_approach_commit_distance_m, "final_approach_commit_distance_m")
        )
        self._final_approach_timeout_s = _positive(
            final_approach_timeout_s, "final_approach_timeout_s"
        )
        self._arrival_position_m = _positive(
            servo_translation_deadband_m, "servo_translation_deadband_m"
        )
        self._arrival_rotation_deg = _positive(
            servo_rotation_deadband_deg, "servo_rotation_deadband_deg"
        )
        self._monotonic = monotonic
        self._last_tactile_not_before_s = -math.inf
        self._observation_sequence = 0
        self._closed = False
        self._last_insertion_start: np.ndarray | None = None
        self._last_insertion_axis: np.ndarray | None = None
        self._last_insertion_command_m: float | None = None
        self._last_insertion_signed_distance_m: float | None = None
        self._committed_insertion_depth_m = 0.0
        self._T_tube_tcp_held: np.ndarray | None = None
        self._held_tracking_id: str | None = None
        self._held_position_std_m: float | None = None
        self._held_orientation_std_deg: float | None = None
        self._tube_tracking_id: str | None = None
        self._last_source_T_base_tube: np.ndarray | None = None
        self._last_source_observed_at_s: float | None = None
        self._last_source_position_std_m: float | None = None
        self._last_source_orientation_std_deg: float | None = None
        self._T_tube_tcp_grasp_actual: np.ndarray | None = None
        self._grasp_relation_position_std_m: float | None = None
        self._grasp_relation_orientation_std_deg: float | None = None
        self._pending_contact_T_base_tube: np.ndarray | None = None
        self._pending_contact_T_base_tcp: np.ndarray | None = None
        self._pending_contact_position_std_m: float | None = None
        self._pending_contact_orientation_std_deg: float | None = None
        self._preclose_source_pose: _PreCloseSourcePose | None = None
        self._pending_contact_from_preclose = False
        self._final_contact_approach: _FinalContactApproach | None = None
        self._last_tactile_feedback: TactileFeedback | None = None
        self._current_source_pose: ObjectPoseEstimate | None = None
        self._source_depth_only_loss = False
        self._source_rgb_identity_retained = False
        self._source_rgb_observed_at_s: float | None = None
        self._pending_recenter: _PendingRecenter | None = None

    @property
    def pose_backend_descriptor(self) -> PoseBackendDescriptor:
        return self._descriptor

    @property
    def observation_diagnostics(self) -> Mapping[str, Any]:
        """Expose producer rejection reasons without changing control predicates."""
        chain: list[dict[str, Any]] = []
        scene = self._scene
        for _ in range(4):
            chain.append({
                "producer": type(scene).__name__,
                "hold_reasons": list(getattr(scene, "_last_hold_reasons", ())),
            })
            downstream = getattr(scene, "_downstream", None)
            if downstream is None or downstream is scene:
                break
            scene = downstream
        return {"scene_producers": chain}

    def observe(self, *, state: TubeState) -> ClosedLoopObservation:
        if self._closed:
            raise LiveIOError("live adapter is closed")
        frame = self._camera.read()
        arm = self._arm.state()
        socket_burst = getattr(self._scene, "observe_burst", None)
        if state is TubeState.OBSERVE and callable(socket_burst):
            frame, scene = socket_burst(
                first_frame=frame,
                read_frame=self._camera.read,
                state=state,
                T_base_tcp=arm.T_base_tcp,
            )
        else:
            scene = self._scene.observe(
                frame=frame,
                state=state,
                T_base_tcp=arm.T_base_tcp,
            )
        gripper = self._gripper.feedback()
        if self._pending_recenter is not None:
            pending = self._pending_recenter
            open_reached = bool(
                gripper.width_m
                >= pending.open_target_width_m - self._recenter_open_width_tolerance_m
            )
            if not gripper.moving and open_reached and gripper.fault_free:
                target = arm.T_base_tcp.copy()
                target[:3, 3] += arm.T_base_tcp[:3, :3] @ np.asarray(
                    [pending.local_x_offset_m, 0.0, 0.0]
                )
                self._arm.servo_target(target=target, action_id=pending.action_id)
                self._pending_recenter = None
                arm = self._arm.state()
            elif (
                self._monotonic() - pending.commanded_at_s > self._recenter_open_timeout_s
                and gripper.fault_free
            ):
                raise LiveIOError("Hand-E did not reach the recenter opening before its deadline")
            elif not gripper.moving and gripper.fault_free:
                # A Modbus status read can briefly return the terminal state of
                # the previous command just after the opening write.  Keep the
                # state machine in MONITOR until the commanded width is
                # actually observed; otherwise it could declare recentering
                # complete before the arm command has even started.
                gripper = GripperFeedback(
                    width_m=gripper.width_m,
                    moving=True,
                    fault_free=True,
                    object_detected=gripper.object_detected,
                    open_confirmed=gripper.open_confirmed,
                )
        windows = self._tactile_source.collect_classifier_windows(
            max(self._last_tactile_not_before_s, -1e30),
            samples=self._tactile_samples,
            timeout=self._tactile_timeout_s,
        )
        self._last_tactile_not_before_s = windows.collected_monotonic_s
        # Timestamp evaluation only after the window exists.  Sampling before
        # collection made freshly captured GF220 frames appear to come from the
        # future and also understated the age of every later producer.
        tactile_now_s = self._monotonic()
        arm = self._arm.state()
        insertion_progress = self._insertion_progress(arm.T_base_tcp)
        tactile = self._tactile.evaluate(
            windows=windows,
            state=state,
            now_s=tactile_now_s,
            T_base_tcp=arm.T_base_tcp,
            gripper=gripper,
            commanded_insertion_step_m=self._last_insertion_command_m,
            measured_insertion_progress_m=insertion_progress,
        )
        tube_pose = self._tube_pose(
            frame=frame,
            scene=scene,
            state=state,
            arm=arm,
            tactile=tactile,
        )
        scene_tactile_observer = getattr(self._scene, "observe_tactile", None)
        if callable(scene_tactile_observer):
            scene_tactile_observer(
                state=state,
                tactile=tactile,
                T_base_tcp=arm.T_base_tcp,
            )
        # Source-tube inference can consume a material part of the Servo cycle.
        # Refresh FK after that remote call so the correction is based on the
        # current TCP rather than the pre-inference sample.  Held-tube poses are
        # composed from the arm sample above and must stay paired with it.
        if state in {
            TubeState.OBSERVE,
            TubeState.APPROACH_TUBE,
            TubeState.SIDE_GRASP,
        }:
            arm = self._arm.state()
        now_s = self._monotonic()
        self._last_tactile_feedback = tactile
        self._source_depth_only_loss = bool(
            getattr(self._scene, "source_depth_only_loss", False)
            and 0 <= now_s - frame.host_received_monotonic_s <= self._source_pose_cache_max_age_s
        )
        self._source_rgb_identity_retained = bool(
            getattr(
                self._scene,
                "source_rgb_identity_retained",
                getattr(self._scene, "source_depth_only_loss", False),
            )
            and 0 <= now_s - frame.host_received_monotonic_s <= self._source_pose_cache_max_age_s
        )
        self._source_rgb_observed_at_s = frame.host_received_monotonic_s
        self._current_source_pose = tube_pose if state in {
            TubeState.OBSERVE, TubeState.APPROACH_TUBE, TubeState.SIDE_GRASP,
        } else None
        if (
            state is TubeState.RECLAMP
            and self._T_tube_tcp_held is None
            and tube_pose is not None
            and tactile.valid
            and 0.0 <= now_s - tactile.observed_at_s <= self._tactile_cache_max_age_s
            and tactile.contact_present
            and tactile.bilateral_contact
            and tactile.rotation_stopped
            and not gripper.moving
            and arm.motion_complete
        ):
            # Freeze the post-pivot relation only from the same fully settled,
            # qualified observation that can authorize RECLAMP ->
            # APPROACH_SOCKET.  Freezing while a reclamp command is still
            # moving makes every later pose kinematic and hides the remaining
            # tube motion from the held-tube interpreter.
            self._T_tube_tcp_held = np.linalg.inv(tube_pose.T_base_object) @ arm.T_base_tcp
            self._held_tracking_id = tube_pose.tracking_id
            self._held_position_std_m = math.hypot(
                tube_pose.position_std_m, self._tcp_position_std_m
            )
            self._held_orientation_std_deg = math.hypot(
                tube_pose.orientation_std_deg, self._tcp_orientation_std_deg
            )
        if state is TubeState.SIDE_GRASP:
            latch_valid = self._preclose_pose_valid(arm=arm, gripper=gripper, now_s=now_s)
            contact_from_preclose = False
            contact_pose = tube_pose
            if (
                contact_pose is None
                and self._last_source_T_base_tube is not None
                and self._last_source_observed_at_s is not None
                and now_s - self._last_source_observed_at_s <= self._source_pose_cache_max_age_s
                and self._last_source_position_std_m is not None
                and self._last_source_orientation_std_deg is not None
            ):
                contact_transform = self._last_source_T_base_tube
                contact_position_std_m = self._last_source_position_std_m
                contact_orientation_std_deg = self._last_source_orientation_std_deg
            elif contact_pose is not None:
                contact_transform = contact_pose.T_base_object
                contact_position_std_m = contact_pose.position_std_m
                contact_orientation_std_deg = contact_pose.orientation_std_deg
            elif latch_valid:
                latch = self._preclose_source_pose
                assert latch is not None
                contact_transform = latch.T_base_tube
                displacement_m, rotation_deg = self._preclose_displacement(arm.T_base_tcp)
                contact_position_std_m = math.hypot(latch.position_std_m, displacement_m)
                contact_orientation_std_deg = math.hypot(latch.orientation_std_deg, rotation_deg)
                contact_from_preclose = True
            else:
                contact_transform = None
                contact_position_std_m = None
                contact_orientation_std_deg = None
            if (
                contact_transform is not None
                and tactile.valid
                and 0.0 <= now_s - tactile.observed_at_s <= self._tactile_cache_max_age_s
                and tactile.bilateral_contact
                and tactile.contact_present
                and not gripper.moving
                and arm.motion_complete
                and arm.fault is None
                and gripper.fault_free
            ):
                # Cache the exact observation that can authorize the
                # SIDE_GRASP -> EXTRACT transition.  Dispatch freezes this pair
                # immediately before the blocking lift, never from a post-lift
                # EXTRACT observation.
                self._pending_contact_T_base_tube = contact_transform.copy()
                self._pending_contact_T_base_tcp = arm.T_base_tcp.copy()
                self._pending_contact_position_std_m = contact_position_std_m
                self._pending_contact_orientation_std_deg = contact_orientation_std_deg
                self._pending_contact_from_preclose = contact_from_preclose
            else:
                self._pending_contact_T_base_tube = None
                self._pending_contact_T_base_tcp = None
                self._pending_contact_position_std_m = None
                self._pending_contact_orientation_std_deg = None
                self._pending_contact_from_preclose = False
        socket_pose = self._socket_pose(
            frame=frame,
            scene=scene,
            state=state,
            arm=arm,
            tactile=tactile,
            now_s=now_s,
        )
        self._observation_sequence += 1
        insertion_depth = (
            None
            if insertion_progress is None
            else max(
                0.0,
                self._committed_insertion_depth_m
                + self._signed_insertion_displacement(arm.T_base_tcp),
            )
        )
        return ClosedLoopObservation(
            observation_id=f"live-{self._observation_sequence:08d}-{frame.sequence:08d}",
            now_s=now_s,
            robot=RobotFeedback(
                T_base_tcp=arm.T_base_tcp,
                motion_complete=arm.motion_complete,
                bounded_tcp_servo_supported=arm.servo_supported,
                fault=arm.fault,
            ),
            gripper=gripper,
            tactile=tactile,
            tube_pose=tube_pose,
            socket_pose=socket_pose,
            front_rgb_angle_target_reached=scene.front_rgb_angle_target_reached,
            tube_clear_of_source=scene.tube_clear_of_source,
            tube_tip_at_socket_entry=scene.tube_tip_at_socket_entry,
            tube_socket_axes_aligned=scene.tube_socket_axes_aligned,
            insertion_progress_m=insertion_progress,
            insertion_depth_m=insertion_depth,
            tube_stays_supported=scene.tube_stays_supported,
            source_depth_only_loss=self._source_depth_only_loss,
            source_rgb_identity_retained=self._source_rgb_identity_retained,
        )

    def _source_pose(
        self,
        frame: D435Frame,
        mask: SceneObjectMask | None,
        provider: Any,
    ) -> ObjectPoseEstimate | None:
        if mask is None:
            return None
        proposal = provider.estimate(
            frame_id=f"d435-{frame.serial}-{frame.sequence}-{mask.object_id}",
            capture_timestamp_s=frame.host_received_monotonic_s,
            rgb=frame.rgb,
            depth_m=frame.depth_m,
            K=frame.intrinsics.matrix,
            mask=mask.mask,
            mask_source=mask.mask_id,
            mask_source_score=mask.mask_iou,
        )
        limit_uncertainty = getattr(self._scene, "limit_source_pose_uncertainty", None)
        if callable(limit_uncertainty):
            limit_uncertainty(
                position_std_m=self._held_position_std_limit_m,
                orientation_std_deg=self._held_orientation_std_limit_deg,
            )
        qualified = self._scene.qualify_pose(
            object_id=mask.object_id,
            proposal=proposal,
            mask=mask,
            frame=frame,
        )
        if qualified is None:
            return None
        if (
            qualified.position_std_m > self._held_position_std_limit_m
            or qualified.orientation_std_deg > self._held_orientation_std_limit_deg
        ):
            # Noisy/occluded live geometry is an unavailable observation.  The
            # resident policy pauses/reacquires; the receipt limits still apply.
            return None
        diagnostics = proposal.foreground_depth
        background_fraction = diagnostics.mask_near_background_fraction
        if mask.object_id == "tube":
            if self._tube_tracking_id is None:
                self._tube_tracking_id = mask.tracking_id
            elif mask.tracking_id != self._tube_tracking_id:
                raise LiveIOError("tube tracking_id changed before grasp")
        return ObjectPoseEstimate(
            object_id=mask.object_id,
            T_base_object=self._T_base_camera @ qualified.T_camera_object,
            observed_at_s=frame.host_received_monotonic_s,
            position_std_m=math.hypot(qualified.position_std_m, self._camera_position_std_m),
            orientation_std_deg=math.hypot(
                qualified.orientation_std_deg, self._camera_orientation_std_deg
            ),
            backend=self._source_descriptor,
            tracking_id=(self._tube_tracking_id if mask.object_id == "tube" else mask.tracking_id),
            inference_latency_s=proposal.total_server_latency_s,
            mask_id=mask.mask_id,
            mask_iou=mask.mask_iou,
            valid_depth_count=diagnostics.valid_mask_pixels,
            foreground_depth_support_count=diagnostics.selected_cluster_pixels,
            foreground_depth_support_fraction=diagnostics.selected_cluster_fraction,
            background_depth_fraction=(1.0 if background_fraction is None else background_fraction),
            masked_depth_cluster_qualified=diagnostics.gate_passed,
            requires_rgbd_quality_gates=True,
            axis_sign_source=qualified.axis_sign_source,
            cylindrical_roll_canonicalized=qualified.cylindrical_roll_canonicalized,
            occluded_by_gripper=mask.occluded_by_gripper,
            size_m=(proposal.size_m if qualified.size_m is None else qualified.size_m),
        )

    def _tube_pose(
        self,
        *,
        frame: D435Frame,
        scene: SceneEvidence,
        state: TubeState,
        arm: ArmState,
        tactile: TactileFeedback,
    ) -> ObjectPoseEstimate | None:
        source_states = {
            TubeState.OBSERVE,
            TubeState.APPROACH_TUBE,
            TubeState.SIDE_GRASP,
        }
        if state in source_states:
            pose = self._source_pose(frame, scene.tube, self._tube_provider)
            if pose is not None:
                self._last_source_T_base_tube = pose.T_base_object.copy()
                self._last_source_observed_at_s = pose.observed_at_s
                self._last_source_position_std_m = pose.position_std_m
                self._last_source_orientation_std_deg = pose.orientation_std_deg
            return pose

        # Once the tube is rigidly held, its pose is robot kinematics composed
        # with a measured relation.  This intentionally avoids asking F2G to
        # hallucinate a transparent, gripper-occluded tube.
        if state in {TubeState.EXTRACT, TubeState.LEVEL_HELD_TUBE}:
            if self._T_tube_tcp_grasp_actual is None:
                raise LiveIOError(
                    "realized grasp relation was not frozen before extract/level motion"
                )
            if (
                self._grasp_relation_position_std_m is None
                or self._grasp_relation_orientation_std_deg is None
            ):
                raise LiveIOError("realized grasp relation lacks contact uncertainty provenance")
            T_base_tube = arm.T_base_tcp @ np.linalg.inv(self._T_tube_tcp_grasp_actual)
            return self._non_rgbd_pose(
                object_id="tube",
                transform=T_base_tube,
                now_s=frame.host_received_monotonic_s,
                descriptor=self._held_descriptor,
                tracking_id=self._tube_tracking_id or "held-demo-relation",
                evidence_id="kinematic-from-runtime-contact-relation",
                position_std_m=self._grasp_relation_position_std_m,
                orientation_std_deg=self._grasp_relation_orientation_std_deg,
                axis_sign_source="open_end_detector",
                roll_canonicalized=True,
            )

        if self._T_tube_tcp_held is not None:
            if self._held_position_std_m is None or self._held_orientation_std_deg is None:
                raise LiveIOError("frozen held relation lacks uncertainty provenance")
            T_base_tube = arm.T_base_tcp @ np.linalg.inv(self._T_tube_tcp_held)
            return self._non_rgbd_pose(
                object_id="tube",
                transform=T_base_tube,
                now_s=frame.host_received_monotonic_s,
                descriptor=self._held_descriptor,
                tracking_id=(
                    self._tube_tracking_id or self._held_tracking_id or "held-kinematic-track"
                ),
                evidence_id="kinematic-from-post-pivot-held-relation",
                position_std_m=self._held_position_std_m,
                orientation_std_deg=self._held_orientation_std_deg,
                axis_sign_source="temporal_continuity",
                roll_canonicalized=True,
            )

        held = self._scene.held_tube_pose(
            frame=frame,
            scene=scene,
            state=state,
            T_base_tcp=arm.T_base_tcp,
            tactile=tactile,
        )
        if held is None:
            return None
        if (
            held.position_std_m > self._held_position_std_limit_m
            or held.orientation_std_deg > self._held_orientation_std_limit_deg
        ):
            raise LiveIOError("held-tube uncertainty exceeds its qualification receipt")
        if self._tube_tracking_id is None:
            self._tube_tracking_id = held.tracking_id
        elif held.tracking_id != self._tube_tracking_id:
            raise LiveIOError("held-tube interpreter changed the tube tracking_id")
        return self._non_rgbd_pose(
            object_id="tube",
            transform=held.T_base_object,
            now_s=frame.host_received_monotonic_s,
            descriptor=self._held_descriptor,
            tracking_id=self._tube_tracking_id,
            evidence_id=held.evidence_id,
            position_std_m=held.position_std_m,
            orientation_std_deg=held.orientation_std_deg,
            axis_sign_source=held.axis_sign_source,
            roll_canonicalized=held.cylindrical_roll_canonicalized,
        )

    def _socket_pose(
        self,
        *,
        frame: D435Frame,
        scene: SceneEvidence,
        state: TubeState,
        arm: ArmState,
        tactile: TactileFeedback,
        now_s: float,
    ) -> ObjectPoseEstimate | None:
        getter = getattr(self._scene, "socket_pose", None)
        if callable(getter):
            qualified = getter(
                frame=frame,
                scene=scene,
                state=state,
                T_base_tcp=arm.T_base_tcp,
                tactile=tactile,
            )
            if qualified is None:
                return None
            if (
                qualified.position_std_m > self._socket_position_std_m
                or qualified.orientation_std_deg > self._socket_orientation_std_deg
            ):
                raise LiveIOError("online socket uncertainty exceeds its runtime limits")
            return self._non_rgbd_pose(
                object_id="socket",
                transform=qualified.T_base_object,
                now_s=now_s,
                descriptor=self._socket_descriptor,
                tracking_id=qualified.tracking_id,
                evidence_id=qualified.evidence_id,
                position_std_m=qualified.position_std_m,
                orientation_std_deg=qualified.orientation_std_deg,
                axis_sign_source=qualified.axis_sign_source,
                roll_canonicalized=qualified.cylindrical_roll_canonicalized,
            )
        if self._T_base_socket is None:
            return None
        return self._static_socket_pose(now_s)

    def _static_socket_pose(self, now_s: float) -> ObjectPoseEstimate:
        if self._T_base_socket is None:
            raise LiveIOError("static socket pose is unavailable")
        return self._non_rgbd_pose(
            object_id="socket",
            transform=self._T_base_socket,
            now_s=now_s,
            descriptor=self._socket_descriptor,
            tracking_id="qualified-static-socket",
            evidence_id="measured-static-socket-pose",
            position_std_m=self._socket_position_std_m,
            orientation_std_deg=self._socket_orientation_std_deg,
            axis_sign_source="not_applicable",
            roll_canonicalized=False,
        )

    @staticmethod
    def _non_rgbd_pose(
        *,
        object_id: str,
        transform: np.ndarray,
        now_s: float,
        descriptor: PoseBackendDescriptor,
        tracking_id: str,
        evidence_id: str,
        position_std_m: float,
        orientation_std_deg: float,
        axis_sign_source: str,
        roll_canonicalized: bool,
    ) -> ObjectPoseEstimate:
        return ObjectPoseEstimate(
            object_id=object_id,
            T_base_object=transform,
            observed_at_s=now_s,
            position_std_m=position_std_m,
            orientation_std_deg=orientation_std_deg,
            backend=descriptor,
            tracking_id=tracking_id,
            inference_latency_s=0.0,
            mask_id=evidence_id,
            mask_iou=0.0,
            valid_depth_count=0,
            foreground_depth_support_count=0,
            foreground_depth_support_fraction=0.0,
            background_depth_fraction=1.0,
            masked_depth_cluster_qualified=False,
            requires_rgbd_quality_gates=False,
            axis_sign_source=axis_sign_source,
            cylindrical_roll_canonicalized=roll_canonicalized,
        )

    def dispatch(self, action: ClosedLoopAction) -> Mapping[str, Any]:
        if self._closed:
            raise LiveIOError("live adapter is closed")
        kind = action.kind
        if kind is ActionKind.MOVE_TCP:
            self._final_contact_approach = None
            if action.state is TubeState.EXTRACT:
                self._freeze_contact_relation_before_extract()
            else:
                self._clear_preclose_source_pose()
            return self._arm.move_coarse(
                target=_transform_parameter(action, "T_base_tcp_target"),
                action_id=action.action_id,
            )
        if kind is ActionKind.SERVO_TCP:
            final_contact = action.parameters.get("final_contact_approach") is True
            shallow_retry = action.parameters.get("shallow_contact_retry") is True
            if final_contact:
                self._prepare_final_contact_step(action)
            elif shallow_retry:
                self._extend_preclose_source_for_shallow_retry(action)
            else:
                self._final_contact_approach = None
            if action.state is TubeState.EXTRACT:
                self._freeze_contact_relation_before_extract()
            elif not shallow_retry:
                self._clear_preclose_source_pose()
            return self._arm.servo_target(
                target=_transform_parameter(action, "T_base_tcp_command"),
                action_id=action.action_id,
            )
        if kind is ActionKind.START_CONTINUOUS_CLOSE:
            committed = action.parameters.get("committed_final_approach") is True
            shallow_retry = action.parameters.get("shallow_contact_retry") is True
            if committed:
                self._check_final_contact(action)
            elif shallow_retry:
                self._extend_preclose_source_for_shallow_retry(action)
            self._arm.stop(reason="hold measured grasp pose before continuous close")
            if committed:
                self._latch_final_contact_close(action)
            elif shallow_retry:
                self._final_contact_approach = None
            else:
                self._final_contact_approach = None
                self._latch_preclose_source_pose()
            return self._gripper.move_width(
                width_m=float(action.parameters["target_width_m"]),
                speed=int(action.parameters["speed"]),
                force=int(action.parameters["force"]),
                action_id=action.action_id,
            )
        if kind is ActionKind.STOP_GRIPPER:
            return self._gripper.stop(reason=action.reason)
        if kind is ActionKind.OPEN_AND_RECENTER:
            self._final_contact_approach = None
            self._clear_preclose_source_pose()
            feedback = self._gripper.feedback()
            target_width_m = min(
                0.050,
                feedback.width_m + float(action.parameters["open_delta_m"]),
            )
            gripper_receipt = self._gripper.move_width(
                width_m=target_width_m,
                speed=self._open_speed,
                force=self._open_force,
                action_id=action.action_id,
            )
            self._pending_recenter = _PendingRecenter(
                local_x_offset_m=float(action.parameters["tcp_local_x_offset_m"]),
                action_id=action.action_id,
                open_target_width_m=target_width_m,
                commanded_at_s=self._monotonic(),
            )
            return {
                "gripper": dict(gripper_receipt),
                "arm": {
                    "accepted": True,
                    "pending_until_gripper_open_step_completes": True,
                },
            }
        if kind in {ActionKind.OPEN_GRIPPER_STEP, ActionKind.RECLAMP}:
            self._final_contact_approach = None
            shallow_retry = (
                kind is ActionKind.OPEN_GRIPPER_STEP
                and action.parameters.get("shallow_contact_retry") is True
            )
            if shallow_retry:
                self._extend_preclose_source_for_shallow_retry(action)
            else:
                self._clear_preclose_source_pose()
            feedback = self._gripper.feedback()
            delta = float(action.parameters["delta_m"])
            target_width = (
                feedback.width_m + delta
                if kind is ActionKind.OPEN_GRIPPER_STEP
                else feedback.width_m - delta
            )
            return self._gripper.move_width(
                width_m=min(0.050, max(0.0, target_width)),
                speed=(
                    self._open_speed if kind is ActionKind.OPEN_GRIPPER_STEP else self._close_speed
                ),
                force=(
                    self._open_force if kind is ActionKind.OPEN_GRIPPER_STEP else self._close_force
                ),
                action_id=action.action_id,
            )
        if kind in {ActionKind.INSERT_STEP, ActionKind.RETRACT_INSERTION}:
            self._final_contact_approach = None
            return self._dispatch_insertion(action)
        if kind is ActionKind.OPEN_RELEASE:
            self._final_contact_approach = None
            self._clear_preclose_source_pose()
            reset_support_history = getattr(self._scene, "reset_support_history_for_release", None)
            if not callable(reset_support_history):
                raise LiveIOError("scene interpreter cannot reset release support history")
            receipt = self._gripper.move_width(
                width_m=0.050,
                speed=self._open_speed,
                force=self._open_force,
                action_id=action.action_id,
            )
            # Dispatch and observation share the resident-loop thread.  Reset
            # only after the gripper accepts the open command, so every frame
            # subsequently counted by the scene is post-release evidence.
            reset_support_history()
            return receipt
        if kind is ActionKind.PAUSE_ARM:
            if action.parameters.get("final_contact_arrived") is True:
                self._check_final_contact(action)
                receipt = self._arm.stop(reason=action.reason)
                self._latch_final_contact_arrival(action)
                return receipt
            return self._arm.stop(reason=action.reason)
        if kind is ActionKind.STOP_AND_HOLD:
            return self.stop_and_hold(reason=action.reason)
        if kind in {ActionKind.COLLECT_EVIDENCE, ActionKind.MONITOR, ActionKind.NONE}:
            return {
                "accepted": True,
                "hardware_commands_dispatched": False,
                "action_id": action.action_id,
            }
        raise LiveIOError(f"unsupported action kind {kind.value}")

    def _freeze_contact_relation_before_extract(self) -> None:
        """Freeze/inject the realized relation before the first lift command."""

        if self._T_tube_tcp_grasp_actual is not None:
            return
        if self._pending_contact_from_preclose:
            gripper = self._gripper.feedback()
            if gripper.moving:
                self._clear_preclose_source_pose()
            self._preclose_pose_valid(
                arm=self._arm.state(), gripper=gripper, now_s=self._monotonic()
            )
        if (
            self._pending_contact_T_base_tube is None
            or self._pending_contact_T_base_tcp is None
            or self._pending_contact_position_std_m is None
            or self._pending_contact_orientation_std_deg is None
        ):
            raise LiveIOError(
                "extraction requested without a fresh stationary bilateral-contact pose"
            )
        contact_tube = self._pending_contact_T_base_tube.copy()
        contact_tcp = self._pending_contact_T_base_tcp.copy()
        relation = validate_rigid_transform(
            np.linalg.inv(contact_tube) @ contact_tcp,
            name="realized T_tube_tcp at contact",
        ).copy()
        setter = getattr(self._scene, "set_grasp_relation", None)
        if not callable(setter):
            raise LiveIOError("scene interpreter cannot accept the realized grasp relation")
        setter(relation, T_base_tube_at_contact=contact_tube)
        self._T_tube_tcp_grasp_actual = relation
        # The same fixed TCP calibration X appears in both measured TCPs:
        # (A(t) X) (O(0)^-1 A(0) X)^-1 = A(t) A(0)^-1 O(0).
        # Its correlated bias cancels; retain source/contact uncertainty,
        # including any measured displacement during the bounded close latch.
        self._grasp_relation_position_std_m = self._pending_contact_position_std_m
        self._grasp_relation_orientation_std_deg = self._pending_contact_orientation_std_deg
        self._pending_contact_T_base_tube = None
        self._pending_contact_T_base_tcp = None
        self._pending_contact_position_std_m = None
        self._pending_contact_orientation_std_deg = None
        self._preclose_source_pose = None
        self._pending_contact_from_preclose = False

    def _clear_preclose_source_pose(self) -> None:
        self._preclose_source_pose = None
        self._pending_contact_T_base_tube = None
        self._pending_contact_T_base_tcp = None
        self._pending_contact_position_std_m = None
        self._pending_contact_orientation_std_deg = None
        self._pending_contact_from_preclose = False

    def _prepare_final_contact_step(self, action: ClosedLoopAction) -> None:
        if action.state is not TubeState.SIDE_GRASP:
            raise LiveIOError("final contact approach requires SIDE_GRASP")
        if (
            self._final_contact_approach is not None
            and self._final_contact_approach.T_base_arrival is not None
        ):
            self._final_contact_approach = None
            raise LiveIOError("final contact approach cannot move after arrival was latched")
        arm = self._arm.state()
        now_s = self._monotonic()
        target = _transform_parameter(action, "final_contact_target")
        source_stamp = float(action.parameters["committed_source_observed_at_s"])
        depth_loss_handoff = action.parameters.get("depth_loss_handoff") is True
        if self._final_contact_approach is None:
            pose = self._current_source_pose
            bounded = bool(
                self._final_approach_commit_distance_m is not None
                and np.linalg.norm(target[:3, 3] - arm.T_base_tcp[:3, 3])
                <= self._final_approach_commit_distance_m
            )
            if depth_loss_handoff:
                rgb_current = bool(
                    pose is None
                    and self._source_rgb_identity_retained
                    and self._source_rgb_observed_at_s is not None
                    and 0 <= now_s - self._source_rgb_observed_at_s
                    <= self._source_pose_cache_max_age_s
                )
                cached_source_valid = bool(
                    self._tube_tracking_id is not None
                    and self._last_source_T_base_tube is not None
                    and self._last_source_observed_at_s is not None
                    and math.isclose(
                        self._last_source_observed_at_s,
                        source_stamp,
                        abs_tol=1e-6,
                        rel_tol=0,
                    )
                    and self._last_source_position_std_m is not None
                    and self._last_source_orientation_std_deg is not None
                )
                if not (bounded and rgb_current and cached_source_valid):
                    raise LiveIOError(
                        "depth-loss handoff lacks current RGB identity and a bounded cached source"
                    )
                assert self._last_source_T_base_tube is not None
                assert self._last_source_observed_at_s is not None
                assert self._tube_tracking_id is not None
                assert self._last_source_position_std_m is not None
                assert self._last_source_orientation_std_deg is not None
                source = _PreCloseSourcePose(
                    T_base_tube=self._last_source_T_base_tube.copy(),
                    T_base_tcp=arm.T_base_tcp.copy(),
                    observed_at_s=self._last_source_observed_at_s,
                    deadline_s=now_s + self._final_approach_timeout_s,
                    tracking_id=self._tube_tracking_id,
                    position_std_m=self._last_source_position_std_m,
                    orientation_std_deg=self._last_source_orientation_std_deg,
                )
            else:
                if (
                    not bounded
                    or pose is None
                    or pose.tracking_id != self._tube_tracking_id
                    or not math.isclose(
                        pose.observed_at_s, source_stamp, abs_tol=1e-6, rel_tol=0
                    )
                    or not 0 <= now_s - pose.observed_at_s
                    <= self._source_pose_cache_max_age_s
                ):
                    raise LiveIOError(
                        "final contact approach lacks a fresh qualified bounded source"
                    )
                source = _PreCloseSourcePose(
                    T_base_tube=pose.T_base_object.copy(),
                    T_base_tcp=arm.T_base_tcp.copy(),
                    observed_at_s=pose.observed_at_s,
                    deadline_s=now_s + self._final_approach_timeout_s,
                    tracking_id=pose.tracking_id,
                    position_std_m=pose.position_std_m,
                    orientation_std_deg=pose.orientation_std_deg,
                )
            self._final_contact_approach = _FinalContactApproach(
                source,
                target.copy(),
                arm.T_base_tcp.copy(),
                depth_loss_handoff=depth_loss_handoff,
            )
        self._check_final_contact(action)
        command = _transform_parameter(action, "T_base_tcp_command")
        if not self._inside_final_contact_segment(command):
            self._final_contact_approach = None
            raise LiveIOError("final contact command leaves its committed segment")

    def _inside_final_contact_segment(self, transform: np.ndarray) -> bool:
        commit = self._final_contact_approach
        assert commit is not None
        travel = commit.T_base_target[:3, 3] - commit.T_base_start[:3, 3]
        length = float(np.linalg.norm(travel))
        delta = transform[:3, 3] - commit.T_base_start[:3, 3]
        if length < 1e-9:
            position_ok = np.linalg.norm(delta) <= self._arrival_position_m
        else:
            progress = float(np.dot(delta, travel / length))
            sideways = np.linalg.norm(delta - progress * travel / length)
            position_ok = (
                -self._arrival_position_m <= progress <= length + self._arrival_position_m
                and sideways <= self._arrival_position_m
            )
        angle = math.degrees(Rotation.from_matrix(
            transform[:3, :3] @ commit.T_base_target[:3, :3].T
        ).magnitude())
        return bool(position_ok and angle <= self._arrival_rotation_deg)

    def _check_final_contact(self, action: ClosedLoopAction) -> None:
        commit = self._final_contact_approach
        if commit is None:
            raise LiveIOError("final contact commitment is unavailable")
        arm, gripper, now_s = self._arm.state(), self._gripper.feedback(), self._monotonic()
        tactile = self._last_tactile_feedback
        pose = self._current_source_pose
        # The first arrived frame was visually qualified.  While the stopped
        # TCP is held there, the original finite commitment can hand off to
        # tactile even if the gripper subsequently hides the complete rim.
        arrived = commit.T_base_arrival is not None
        pose_valid = bool(
            arrived or (
                self._source_rgb_identity_retained
                and self._source_rgb_observed_at_s is not None
                and 0 <= now_s - self._source_rgb_observed_at_s
                <= self._source_pose_cache_max_age_s
            )
        ) if pose is None else bool(
            pose.tracking_id == commit.source.tracking_id
            and 0 <= now_s - pose.observed_at_s <= self._source_pose_cache_max_age_s
            and np.linalg.norm(pose.T_base_object[:3, 3] - commit.source.T_base_tube[:3, 3]) <= .003
            and math.degrees(Rotation.from_matrix(
                pose.T_base_object[:3, :3] @ commit.source.T_base_tube[:3, :3].T
            ).magnitude()) <= 2.0
        )
        stationary = True
        if arrived or action.parameters.get("final_contact_arrived") is True:
            endpoint_distance, endpoint_angle = _pose_error(arm.T_base_tcp, commit.T_base_target)
            stationary = bool(
                arm.motion_complete
                and endpoint_distance <= self._arrival_position_m
                and math.degrees(endpoint_angle) <= self._arrival_rotation_deg
            )
            if arrived:
                arrival_distance, arrival_angle = _pose_error(
                    arm.T_base_tcp, commit.T_base_arrival
                )
                stationary = bool(
                    stationary and arrival_distance <= self._arrival_position_m
                    and math.degrees(arrival_angle) <= self._arrival_rotation_deg
                )
        valid = bool(
            action.state is TubeState.SIDE_GRASP
            and (action.parameters.get("depth_loss_handoff") is True)
            is commit.depth_loss_handoff
            and now_s <= commit.source.deadline_s
            and self._tube_tracking_id == commit.source.tracking_id
            and np.allclose(_transform_parameter(action, "final_contact_target"),
                            commit.T_base_target, atol=1e-9, rtol=0)
            and math.isclose(float(action.parameters["committed_source_observed_at_s"]),
                             commit.source.observed_at_s, abs_tol=1e-6, rel_tol=0)
            and arm.fault is None and gripper.fault_free and not gripper.moving
            and gripper.width_m >= .048 and tactile is not None and tactile.valid
            and 0 <= now_s - tactile.observed_at_s <= self._tactile_cache_max_age_s
            and not tactile.contact_present and not tactile.bilateral_contact
            and pose_valid and stationary
            and self._inside_final_contact_segment(arm.T_base_tcp)
        )
        if not valid:
            self._final_contact_approach = None
            raise LiveIOError("final contact commitment expired or current evidence changed")

    def _latch_final_contact_arrival(self, action: ClosedLoopAction) -> None:
        self._check_final_contact(action)
        commit = self._final_contact_approach
        assert commit is not None
        arm = self._arm.state()
        distance, angle = _pose_error(arm.T_base_tcp, commit.T_base_target)
        if (
            not arm.motion_complete or distance > self._arrival_position_m
            or math.degrees(angle) > self._arrival_rotation_deg
        ):
            self._final_contact_approach = None
            raise LiveIOError("final contact arrival requires stopped TCP at committed endpoint")
        if commit.T_base_arrival is None:
            self._final_contact_approach = _FinalContactApproach(
                commit.source,
                commit.T_base_target,
                commit.T_base_start,
                arm.T_base_tcp.copy(),
                commit.depth_loss_handoff,
            )

    def _latch_final_contact_close(self, action: ClosedLoopAction) -> None:
        self._check_final_contact(action)
        commit = self._final_contact_approach
        assert commit is not None
        arm = self._arm.state()
        error_m = float(np.linalg.norm(arm.T_base_tcp[:3, 3] - commit.T_base_target[:3, 3]))
        if not arm.motion_complete or error_m > self._arrival_position_m:
            self._final_contact_approach = None
            raise LiveIOError("final contact close requires stopped TCP at committed endpoint")
        self._clear_preclose_source_pose()
        self._preclose_source_pose = _PreCloseSourcePose(
            T_base_tube=commit.source.T_base_tube.copy(), T_base_tcp=arm.T_base_tcp.copy(),
            observed_at_s=commit.source.observed_at_s,
            deadline_s=self._monotonic() + self._close_timeout_s,
            tracking_id=commit.source.tracking_id, position_std_m=commit.source.position_std_m,
            orientation_std_deg=commit.source.orientation_std_deg,
        )
        self._final_contact_approach = None

    def _latch_preclose_source_pose(self) -> None:
        self._clear_preclose_source_pose()
        arm = self._arm.state()
        gripper = self._gripper.feedback()
        now_s = self._monotonic()
        if (
            not arm.motion_complete or arm.fault is not None or not gripper.fault_free
            or self._tube_tracking_id is None
            or self._last_source_T_base_tube is None
            or self._last_source_observed_at_s is None
            or not (
                0.0 <= now_s - self._last_source_observed_at_s <= self._source_pose_cache_max_age_s
            )
            or self._last_source_position_std_m is None
            or self._last_source_orientation_std_deg is None
        ):
            return
        self._preclose_source_pose = _PreCloseSourcePose(
            T_base_tube=self._last_source_T_base_tube.copy(),
            T_base_tcp=arm.T_base_tcp.copy(),
            observed_at_s=self._last_source_observed_at_s,
            deadline_s=now_s + self._close_timeout_s,
            tracking_id=self._tube_tracking_id,
            position_std_m=self._last_source_position_std_m,
            orientation_std_deg=self._last_source_orientation_std_deg,
        )

    def _preclose_displacement(self, T_base_tcp: np.ndarray) -> tuple[float, float]:
        latch = self._preclose_source_pose
        assert latch is not None
        distance = float(np.linalg.norm(T_base_tcp[:3, 3] - latch.T_base_tcp[:3, 3]))
        rotation = math.degrees(Rotation.from_matrix(
            T_base_tcp[:3, :3] @ latch.T_base_tcp[:3, :3].T
        ).magnitude())
        return distance, rotation

    def _preclose_pose_valid(
        self, *, arm: ArmState, gripper: GripperFeedback, now_s: float
    ) -> bool:
        latch = self._preclose_source_pose
        if latch is None:
            return False
        displacement_m, rotation_deg = self._preclose_displacement(arm.T_base_tcp)
        if (
            now_s > latch.deadline_s or now_s < latch.observed_at_s
            or self._tube_tracking_id != latch.tracking_id
            or arm.fault is not None or not gripper.fault_free
            or displacement_m > latch.max_displacement_m
            or rotation_deg > 2.0
        ):
            self._clear_preclose_source_pose()
            return False
        # A bounded retry may be in flight.  Keep its source latch, but only
        # expose it as usable contact evidence once the measured TCP stops.
        return bool(arm.motion_complete)

    def _extend_preclose_source_for_shallow_retry(
        self, action: ClosedLoopAction
    ) -> None:
        latch = self._preclose_source_pose
        if latch is None:
            raise LiveIOError("shallow-contact retry lacks the committed source latch")
        maximum = float(action.parameters["source_latch_max_displacement_m"])
        if (
            not math.isfinite(maximum)
            or maximum < latch.max_displacement_m
            or maximum > 0.0125
        ):
            raise LiveIOError("shallow-contact retry source-latch bound is invalid")
        self._preclose_source_pose = _PreCloseSourcePose(
            T_base_tube=latch.T_base_tube,
            T_base_tcp=latch.T_base_tcp,
            observed_at_s=latch.observed_at_s,
            deadline_s=max(latch.deadline_s, self._monotonic() + self._close_timeout_s),
            tracking_id=latch.tracking_id,
            position_std_m=latch.position_std_m,
            orientation_std_deg=latch.orientation_std_deg,
            max_displacement_m=maximum,
        )

    def _dispatch_insertion(self, action: ClosedLoopAction) -> Mapping[str, Any]:
        current = self._arm.state().T_base_tcp
        axis = np.asarray(action.parameters["axis_base"], dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 0.0:
            raise LiveIOError("insertion axis_base must be a finite non-zero vector")
        axis /= np.linalg.norm(axis)
        distance = float(action.parameters["distance_m"])
        if not math.isfinite(distance) or distance == 0.0:
            raise LiveIOError("insertion distance must be finite and non-zero")
        if (
            self._last_insertion_start is not None
            and self._last_insertion_signed_distance_m is not None
        ):
            previous = self._signed_insertion_displacement(current)
            self._committed_insertion_depth_m = max(
                0.0, self._committed_insertion_depth_m + previous
            )
        self._last_insertion_start = current[:3, 3].copy()
        self._last_insertion_axis = axis.copy()
        self._last_insertion_command_m = abs(distance)
        self._last_insertion_signed_distance_m = distance
        target = current.copy()
        target[:3, 3] += axis * distance
        return self._arm.servo_target(
            target=target, action_id=action.action_id,
            position_tolerance_m=min(0.0005, abs(distance) / 3.0),
        )

    def _insertion_progress(self, T_base_tcp: np.ndarray) -> float | None:
        if self._last_insertion_start is None or self._last_insertion_axis is None:
            return None
        delta = T_base_tcp[:3, 3] - self._last_insertion_start
        return max(0.0, float(np.dot(delta, self._last_insertion_axis)))

    def _signed_insertion_displacement(self, T_base_tcp: np.ndarray) -> float:
        if self._last_insertion_start is None or self._last_insertion_axis is None:
            return 0.0
        delta = T_base_tcp[:3, 3] - self._last_insertion_start
        return float(np.dot(delta, self._last_insertion_axis))

    def stop_and_hold(self, *, reason: str) -> Mapping[str, Any]:
        self._pending_recenter = None
        self._final_contact_approach = None
        self._clear_preclose_source_pose()
        arm_error: str | None = None
        gripper_error: str | None = None
        arm_receipt: Mapping[str, Any] | None = None
        gripper_receipt: Mapping[str, Any] | None = None
        try:
            arm_receipt = self._arm.stop(reason=reason)
        except Exception as error:  # noqa: BLE001
            arm_error = str(error)
        try:
            gripper_receipt = self._gripper.stop(reason=reason)
        except Exception as error:  # noqa: BLE001
            gripper_error = str(error)
        if arm_error or gripper_error:
            raise LiveIOError(
                f"hold incomplete: arm={arm_error or 'ok'}, gripper={gripper_error or 'ok'}"
            )
        return {
            "held": True,
            "arm": dict(arm_receipt or {}),
            "gripper": dict(gripper_receipt or {}),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[str] = []
        for name, resource in (
            ("arm", self._arm),
            ("gripper", self._gripper),
            ("tactile_interpreter", self._tactile),
            ("scene_interpreter", self._scene),
            ("tactile_source", self._tactile_source),
            ("camera", self._camera),
        ):
            try:
                resource.close()
            except Exception as error:  # noqa: BLE001
                errors.append(f"{name}: {error}")
        if errors:
            raise LiveIOError("adapter close errors: " + "; ".join(errors))


def reduced_collision_model_acknowledgement(
    config: Mapping[str, Any],
    *,
    accepted: bool,
) -> dict[str, Any]:
    """Describe the process-local arm-only collision-model acknowledgement.

    This is deliberately not a qualification receipt.  It can waive only the
    *missing* FR3+Hand-E collision/payload receipt for the current process.  A
    present but invalid receipt remains a blocker, and camera/TCP calibration
    is outside this acknowledgement's scope.
    """

    if not isinstance(accepted, bool):
        raise TypeError("accepted must be boolean")
    live = config.get("live_adapter")
    robot = live.get("robot") if isinstance(live, Mapping) else None
    receipt_blockers = (
        _receipt_blockers(
            robot.get("collision_payload_qualification_receipt"),
            "fr3_hande_collision_payload",
        )
        if isinstance(robot, Mapping)
        else ["robot_config_missing"]
    )
    receipt_missing = receipt_blockers == [_COLLISION_RECEIPT_MISSING]
    active = bool(accepted and receipt_missing)

    moveit = robot.get("moveit") if isinstance(robot, Mapping) else None
    servo = robot.get("servo") if isinstance(robot, Mapping) else None

    def limit(
        value: object,
        *,
        default: float,
        cap: float,
    ) -> dict[str, float | None]:
        try:
            configured = float(value if value is not None else default)
        except (TypeError, ValueError):
            configured = math.nan
        valid = math.isfinite(configured) and configured > 0.0
        effective = (
            min(configured, cap)
            if active and valid
            else configured
            if valid
            else None
        )
        return {
            "configured": configured if valid else None,
            "effective": effective,
            "cap": cap,
        }

    limits = {
        "moveit_velocity_scale": limit(
            moveit.get("velocity_scale") if isinstance(moveit, Mapping) else None,
            default=0.03,
            cap=REDUCED_COLLISION_SPEED_CAPS["moveit_velocity_scale"],
        ),
        "moveit_acceleration_scale": limit(
            moveit.get("acceleration_scale") if isinstance(moveit, Mapping) else None,
            default=0.03,
            cap=REDUCED_COLLISION_SPEED_CAPS["moveit_acceleration_scale"],
        ),
        "servo_max_linear_m_s": limit(
            servo.get("max_linear_m_s") if isinstance(servo, Mapping) else None,
            default=0.08,
            cap=REDUCED_COLLISION_SPEED_CAPS["servo_max_linear_m_s"],
        ),
        "servo_max_angular_rad_s": limit(
            servo.get("max_angular_rad_s") if isinstance(servo, Mapping) else None,
            default=0.8,
            cap=REDUCED_COLLISION_SPEED_CAPS["servo_max_angular_rad_s"],
        ),
    }
    return {
        "schema": REDUCED_COLLISION_ACK_SCHEMA,
        "accepted_by_process_flag": accepted,
        "active": active,
        "source": "--accept-arm-only-collision-model" if accepted else None,
        "current_process_only": True,
        "qualification_receipt_substituted": False,
        "receipt_state": (
            "missing_acknowledged_for_this_process"
            if active
            else "missing_not_acknowledged"
            if receipt_missing
            else "qualified_receipt_present"
            if not receipt_blockers
            else "invalid_or_unqualified_receipt"
        ),
        "reduced_collision_scope": (
            "MoveIt collision checking covers the configured FR3 arm-only model; "
            "Hand-E, adapter, fingers, tactile attachments, payload mass, and centre "
            "of gravity are not qualified collision/inertial geometry."
        ),
        "independent_gates": [
            "fixed_camera_extrinsic",
            "hande_tcp",
        ],
        "speed_limits": limits,
    }


def provisional_camera_extrinsic_acknowledgement(
    config: Mapping[str, Any],
    *,
    accepted: bool,
) -> dict[str, Any]:
    """Validate one explicitly accepted, process-local recovered camera pose.

    The recovery report remains plan-only evidence and is never rewritten as a
    qualification receipt.  This acknowledgement can waive only a *missing*
    fixed-camera receipt, and only when the task config binds the exact
    recovered transform and report from the current camera-return check.
    """

    if not isinstance(accepted, bool):
        raise TypeError("accepted must be boolean")
    live = config.get("live_adapter")
    camera = live.get("camera") if isinstance(live, Mapping) else None
    camera = camera if isinstance(camera, Mapping) else {}
    receipt_blockers = _receipt_blockers(
        camera.get("camera_extrinsic_qualification_receipt"),
        "fixed_camera_extrinsic",
    )
    receipt_missing = receipt_blockers == [_CAMERA_RECEIPT_MISSING]
    report_path = _path_or_none(camera.get("provisional_camera_recovery_report"))
    transform_path = _path_or_none(camera.get("camera_extrinsic"))
    reasons: list[str] = []
    report: dict[str, Any] | None = None
    position_std_m: float | None = None
    orientation_std_deg: float | None = None
    outlier_statistics: dict[str, Any] = {}

    if report_path is None or not report_path.is_file():
        reasons.append("provisional_camera_recovery_report_missing")
    else:
        try:
            report = _read_json(report_path)
        except (OSError, ValueError, json.JSONDecodeError):
            reasons.append("provisional_camera_recovery_report_invalid")

    if transform_path is None or not transform_path.is_file():
        reasons.append("camera_extrinsic_artifact_missing")

    if report is not None:
        if report.get("schema") != PROVISIONAL_CAMERA_RECOVERY_SCHEMA:
            reasons.append("provisional_camera_recovery_schema_invalid")
        source_formal = _path_or_none(report.get("source_formal_calibration"))
        if source_formal != _FORMAL_CAMERA_EXTRINSIC:
            reasons.append("provisional_camera_source_formal_calibration_mismatch")
        decision = report.get("decision")
        if not isinstance(decision, Mapping) or decision.get("plan_only_usable") is not True:
            reasons.append("provisional_camera_not_plan_only_usable")
        frame_count = report.get("current_frame_count")
        selected = report.get("selected_current_frames")
        selected_valid = bool(
            isinstance(frame_count, int)
            and not isinstance(frame_count, bool)
            and frame_count == _PROVISIONAL_CAMERA_REQUIRED_FRAME_COUNT
            and isinstance(selected, list)
            and len(selected) >= _PROVISIONAL_CAMERA_MIN_SELECTED_FRAMES
            and len(selected) == len(set(selected))
            and all(
                isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index < frame_count
                for index in selected
            )
        )
        if not selected_valid:
            reasons.append("provisional_camera_selected_frame_consensus_insufficient")

        try:
            final_rotation_deg = _nonnegative(
                report.get("final_rotation_deg"), "final_rotation_deg"
            )
        except (TypeError, ValueError):
            final_rotation_deg = math.inf
        final_translation_mm = _finite_array(report.get("final_translation_mm"), (3,))
        final_translation_m = (
            math.inf
            if final_translation_mm is None
            else float(np.linalg.norm(final_translation_mm)) / 1000.0
        )
        if final_rotation_deg > _PROVISIONAL_CAMERA_MAX_FINAL_ROTATION_DEG:
            reasons.append("provisional_camera_final_rotation_exceeds_limit")
        if final_translation_m > _PROVISIONAL_CAMERA_MAX_FINAL_TRANSLATION_M:
            reasons.append("provisional_camera_final_translation_exceeds_limit")

        cross_frame_rotation = report.get("cross_frame_rotation_deviation_deg")
        cross_frame_translation = report.get("cross_frame_translation_deviation_mm")
        cross_reference_rotation = report.get("cross_reference_rotation_deviation_deg")
        cross_reference_translation = report.get(
            "cross_reference_translation_deviation_mm"
        )
        try:
            cross_frame_rotation_max = _nonnegative(
                cross_frame_rotation.get("max")
                if isinstance(cross_frame_rotation, Mapping)
                else None,
                "cross_frame_rotation.max",
            )
            cross_frame_translation_median_mm = _nonnegative(
                cross_frame_translation.get("median")
                if isinstance(cross_frame_translation, Mapping)
                else None,
                "cross_frame_translation.median",
            )
            cross_reference_rotation_median = _nonnegative(
                cross_reference_rotation.get("median")
                if isinstance(cross_reference_rotation, Mapping)
                else None,
                "cross_reference_rotation.median",
            )
            cross_reference_translation_median_mm = _nonnegative(
                cross_reference_translation.get("median")
                if isinstance(cross_reference_translation, Mapping)
                else None,
                "cross_reference_translation.median",
            )
        except (TypeError, ValueError):
            reasons.append("provisional_camera_dispersion_statistics_invalid")
        else:
            position_std_m = max(
                cross_frame_translation_median_mm,
                cross_reference_translation_median_mm,
            ) / 1000.0
            orientation_std_deg = max(
                cross_frame_rotation_max,
                cross_reference_rotation_median,
            )
            if (
                cross_frame_rotation_max
                > _PROVISIONAL_CAMERA_MAX_CROSS_FRAME_ROTATION_DEG
            ):
                reasons.append("provisional_camera_cross_frame_rotation_exceeds_limit")
            if position_std_m > _PROVISIONAL_CAMERA_MAX_POSITION_STD_M:
                reasons.append("provisional_camera_position_uncertainty_exceeds_live_gate")
            if orientation_std_deg > _PROVISIONAL_CAMERA_MAX_ORIENTATION_STD_DEG:
                reasons.append("provisional_camera_orientation_uncertainty_exceeds_live_gate")

        outlier_statistics = {
            "consensus_median_residual_mm": report.get("consensus_median_residual_mm"),
            "consensus_p90_residual_mm": report.get("consensus_p90_residual_mm"),
            "cross_frame_translation_deviation_mm": cross_frame_translation,
            "cross_reference_translation_deviation_mm": cross_reference_translation,
            "cross_reference_rotation_deviation_deg": cross_reference_rotation,
        }

        if transform_path is not None and transform_path.is_file():
            try:
                configured_transform = _load_transform_path(transform_path)
                reported_transform = validate_rigid_transform(
                    report.get("T_base_newcam_provisional"),
                    name="provisional T_base_newcam",
                )
                formal_transform = _load_transform_path(_FORMAL_CAMERA_EXTRINSIC)
                recovery_transform = validate_rigid_transform(
                    report.get("T_oldcam_newcam"),
                    name="provisional T_oldcam_newcam",
                )
            except (OSError, TypeError, ValueError, ClosedLoopContractError):
                reasons.append("provisional_camera_transform_invalid")
            else:
                if not np.allclose(
                    configured_transform, reported_transform, atol=1e-9, rtol=0.0
                ):
                    reasons.append("provisional_camera_transform_report_mismatch")
                if not np.allclose(
                    reported_transform,
                    formal_transform @ recovery_transform,
                    atol=1e-9,
                    rtol=0.0,
                ):
                    reasons.append("provisional_camera_transform_composition_mismatch")

    report_valid = not reasons
    active = bool(accepted and receipt_missing and report_valid)
    return {
        "schema": PROVISIONAL_CAMERA_ACK_SCHEMA,
        "accepted_by_process_flag": accepted,
        "active": active,
        "source": "--accept-provisional-camera-extrinsic" if accepted else None,
        "current_process_only": True,
        "qualification_receipt_substituted": False,
        "receipt_state": (
            "missing_acknowledged_for_this_process"
            if active
            else "missing_not_acknowledged"
            if receipt_missing and report_valid
            else "provisional_evidence_invalid"
            if receipt_missing
            else "qualified_receipt_present"
            if not receipt_blockers
            else "invalid_or_unqualified_receipt"
        ),
        "camera_extrinsic": None if transform_path is None else str(transform_path),
        "provisional_recovery_report": (
            None if report_path is None else str(report_path)
        ),
        "source_formal_calibration": str(_FORMAL_CAMERA_EXTRINSIC),
        "validation_reasons": sorted(set(reasons)),
        "selected_frame_count": (
            None
            if report is None or not isinstance(report.get("selected_current_frames"), list)
            else len(report["selected_current_frames"])
        ),
        "current_frame_count": None if report is None else report.get("current_frame_count"),
        "final_rotation_deg": None if report is None else report.get("final_rotation_deg"),
        "final_translation_mm": (
            None if report is None else report.get("final_translation_mm")
        ),
        "derived_uncertainty": {
            "position_std_m": position_std_m,
            "orientation_std_deg": orientation_std_deg,
            "method": (
                "max(cross-frame translation median, cross-reference translation "
                "median); max(cross-frame rotation max, cross-reference rotation median)"
            ),
        },
        "outlier_statistics_retained_for_review": outlier_statistics,
        "independent_gates": [
            "fr3_hande_collision_payload",
            "hande_tcp",
        ],
    }


def _reduced_collision_robot_config(
    robot: Mapping[str, Any],
    *,
    acknowledgement: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an in-memory robot config with the reduced-model speed caps."""

    result = dict(robot)
    if acknowledgement.get("active") is not True:
        return result
    moveit = dict(_mapping(robot.get("moveit"), "robot.moveit"))
    servo = dict(_mapping(robot.get("servo"), "robot.servo"))
    limits = _mapping(acknowledgement.get("speed_limits"), "speed_limits")
    for key, field in (
        ("velocity_scale", "moveit_velocity_scale"),
        ("acceleration_scale", "moveit_acceleration_scale"),
    ):
        effective = _mapping(limits.get(field), field).get("effective")
        moveit[key] = _positive(effective, f"reduced collision {field}")
    for key, field in (
        ("max_linear_m_s", "servo_max_linear_m_s"),
        ("max_angular_rad_s", "servo_max_angular_rad_s"),
    ):
        effective = _mapping(limits.get(field), field).get("effective")
        servo[key] = _positive(effective, f"reduced collision {field}")
    result["moveit"] = moveit
    result["servo"] = servo
    return result


def assess_live_config(
    config: Mapping[str, Any],
    *,
    accept_arm_only_collision_model: bool = False,
    accept_provisional_camera_extrinsic: bool = False,
) -> LivePreflightAssessment:
    """Classify inert deployment checks and evidence acquired by the live run.

    The legacy scene/tactile factories consume reviewed artifacts and therefore
    retain their original all-static contract.  The online factories instead
    own scene and tactile bootstrap.  For those factories the corresponding
    values are reported as runtime requirements; they are not launcher
    blockers and may never be replaced by defaults.
    """

    issues = set(_preflight_live_config_legacy(config))
    collision_acknowledgement = reduced_collision_model_acknowledgement(
        config,
        accepted=accept_arm_only_collision_model,
    )
    if collision_acknowledgement["active"] is True:
        issues.discard(_COLLISION_RECEIPT_MISSING)
    camera_acknowledgement = provisional_camera_extrinsic_acknowledgement(
        config,
        accepted=accept_provisional_camera_extrinsic,
    )
    if camera_acknowledgement["active"] is True:
        issues.discard(_CAMERA_RECEIPT_MISSING)
    elif (
        accept_provisional_camera_extrinsic
        and camera_acknowledgement["receipt_state"] == "provisional_evidence_invalid"
    ):
        issues.add("provisional_camera_extrinsic_acknowledgement_invalid")
    issues.update(_demonstration_contract_blockers(config.get("demonstration")))
    runtime: set[str] = set()
    live = config.get("live_adapter")
    scene = live.get("scene_interpreter") if isinstance(live, Mapping) else None
    tactile = live.get("tactile") if isinstance(live, Mapping) else None

    if isinstance(scene, Mapping) and scene.get("factory") == ONLINE_SCENE_FACTORY:
        deferred_scene_issues = {
            "demo_tube_tcp_relation_qualification_receipt_missing",
            "demo_tube_tcp_relation_qualification_receipt_invalid",
            "demo_tube_tcp_relation_not_runtime_qualified",
            "tube_scene_interpreter_qualification_receipt_missing",
            "tube_scene_interpreter_qualification_receipt_invalid",
            "tube_scene_interpreter_not_runtime_qualified",
            "socket_config_missing",
            "static_socket_pose_qualification_receipt_missing",
            "static_socket_pose_qualification_receipt_invalid",
            "static_socket_pose_not_runtime_qualified",
            "pose_artifact_missing",
            "aggregate_pose_producers_qualification_receipt_missing",
            "aggregate_pose_producers_qualification_receipt_invalid",
            "aggregate_pose_producers_not_runtime_qualified",
            "source_f2g_pose_qualification_receipt_missing",
            "source_f2g_pose_qualification_receipt_invalid",
            "source_f2g_pose_not_runtime_qualified",
            "measured_tube_length_missing",
            "tube_length_provenance_missing",
            "demo_relation_not_runtime_qualified",
            "T_socket_tube_entry_missing",
            "T_socket_tube_entry_not_runtime_qualified",
        }
        runtime.update(issues & deferred_scene_issues)
        issues.difference_update(deferred_scene_issues)
        runtime.update(
            {
                "runtime_source_tube_pose_size_and_signed_axis_required",
                "runtime_f2g_service_identity_and_crosschecks_required",
                "runtime_source_tracker_seed_required",
                "runtime_tube_length_required",
                "runtime_socket_pose_and_entry_required",
                "runtime_socket_target_identity_required",
                "runtime_held_tube_tracker_seed_required",
                "runtime_insertion_completion_evidence_required",
            }
        )

    if isinstance(tactile, Mapping) and tactile.get("factory") == SESSION_BOOTSTRAP_TACTILE_FACTORY:
        deferred_tactile_issues = {
            "test_tube_tactile_interpreter_qualification_receipt_missing",
            "test_tube_tactile_interpreter_qualification_receipt_invalid",
            "test_tube_tactile_interpreter_not_runtime_qualified",
            "test_tube_tactile_qualification_bundle_missing",
            "test_tube_tactile_qualification_bundle_invalid",
            "test_tube_tactile_bundle_not_runtime_qualified",
            "test_tube_tactile_receipt_bundle_mismatch",
        }
        runtime.update(issues & deferred_tactile_issues)
        issues.difference_update(deferred_tactile_issues)
        runtime.update(
            {
                "runtime_tactile_unloaded_baseline_required",
                "runtime_tactile_contact_thresholds_required",
                "runtime_tactile_pivot_stability_required",
                "runtime_tactile_insertion_jam_baseline_required",
            }
        )

    return LivePreflightAssessment(
        static_blockers=tuple(sorted(issues)),
        runtime_requirements=tuple(sorted(runtime)),
    )


def _demonstration_contract_blockers(value: object) -> tuple[str, ...]:
    """Bind a concrete online task to one reviewed, contiguous demo clip.

    The online factories may infer scene geometry at runtime, but they must not
    silently fall back to an older demonstration or skip a demonstrated
    subphase that the controller cannot execute.  Templates omit this optional
    section; a concrete task that supplies it is checked before any service or
    hardware process is started.
    """

    if value is None:
        return ()
    if not isinstance(value, Mapping):
        return ("demonstration_contract_invalid",)
    blockers: list[str] = []
    if value.get("schema") != "demo2gripper.test_tube_execution_demonstration.v1":
        blockers.append("demonstration_contract_schema_invalid")

    documents: dict[str, dict[str, Any]] = {}
    for key in ("effective_interval", "effective_index", "stage_review", "execution_plan"):
        path = _path_or_none(value.get(key))
        if path is None or not path.is_file():
            blockers.append(f"demonstration_{key}_missing")
            continue
        try:
            documents[key] = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            blockers.append(f"demonstration_{key}_invalid")

    expected_interval = value.get("frame_interval")
    if (
        not isinstance(expected_interval, list)
        or len(expected_interval) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in expected_interval)
        or expected_interval[0] < 0
        or expected_interval[1] <= expected_interval[0]
    ):
        blockers.append("demonstration_frame_interval_invalid")
        expected_interval = None

    interval = documents.get("effective_interval")
    if interval is not None:
        actual = [interval.get("start_frame"), interval.get("end_frame_exclusive")]
        if (
            interval.get("schema") != "demo2gripper.effective_demo_interval.v1"
            or interval.get("status") != "accepted_unique_complete_interval"
            or expected_interval is None
            or actual != expected_interval
        ):
            blockers.append("demonstration_effective_interval_mismatch")

    index = documents.get("effective_index")
    if index is not None and (
        index.get("schema") != "demo2gripper.lossless_multimodal_index_view.v1"
        or expected_interval is None
        or index.get("frame_interval") != expected_interval
        or index.get("frame_count") != expected_interval[1] - expected_interval[0]
    ):
        blockers.append("demonstration_effective_index_mismatch")

    review = documents.get("stage_review")
    expected_names = (
        "APPROACH",
        "GRASP_AND_EXTRACT",
        "LOOSEN_AND_PIVOT",
        "RECLAMP",
        "APPROACH_RACK_ENTRY",
        "ALIGN_AND_INSERT",
        "RELEASE_AND_RETREAT",
    )
    if review is not None:
        stages = review.get("stages")
        names = (
            tuple(stage.get("name") for stage in stages if isinstance(stage, Mapping))
            if isinstance(stages, list)
            else ()
        )
        intervals = (
            [stage.get("frames_inclusive") for stage in stages]
            if isinstance(stages, list) and all(isinstance(stage, Mapping) for stage in stages)
            else []
        )
        intervals_valid = len(intervals) == 7 and all(
            isinstance(item, list)
            and len(item) == 2
            and all(isinstance(frame, int) and not isinstance(frame, bool) for frame in item)
            and item[0] >= 0
            and item[1] >= item[0]
            for item in intervals
        )
        contiguous = intervals_valid
        for previous, current in zip(intervals, intervals[1:], strict=False):
            contiguous = bool(
                contiguous
                and isinstance(previous, list)
                and len(previous) == 2
                and isinstance(current, list)
                and len(current) == 2
                and current[0] == previous[1] + 1
            )
        covered = (
            expected_interval is not None
            and intervals_valid
            and intervals[0][0] == expected_interval[0]
            and intervals[-1][1] + 1 == expected_interval[1]
        )
        if (
            review.get("schema") != "demo2gripper.test_tube_stage_review.v2"
            or review.get("status") != "preexecution_analysis_only"
            or review.get("hardware_commands_dispatched") is not False
            or names != expected_names
            or not contiguous
            or not covered
        ):
            blockers.append("demonstration_seven_stage_contract_mismatch")

    plan = documents.get("execution_plan")
    if plan is not None and (
        plan.get("schema") != "demo2gripper.test_tube_online_execution_plan.v1"
        or plan.get("hardware_commands_dispatched") is not False
        or not isinstance(plan.get("stages"), list)
        or len(plan["stages"]) != 7
    ):
        blockers.append("demonstration_execution_plan_invalid")

    if value.get("source_axis_sign_primitive_qualified") is not True:
        blockers.append("demo_source_axis_sign_primitive_not_qualified")
    if value.get("post_extract_level_controller_available") is not True:
        blockers.append("demo_post_extract_level_controller_missing")
    return tuple(sorted(set(blockers)))


def preflight_live_config(
    config: Mapping[str, Any],
    *,
    accept_arm_only_collision_model: bool = False,
    accept_provisional_camera_extrinsic: bool = False,
) -> tuple[str, ...]:
    """Return only configuration that must exist before hardware is opened."""

    return assess_live_config(
        config,
        accept_arm_only_collision_model=accept_arm_only_collision_model,
        accept_provisional_camera_extrinsic=accept_provisional_camera_extrinsic,
    ).static_blockers


def runtime_live_config_requirements(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return online evidence gates owned by the resident controller."""

    return assess_live_config(config).runtime_requirements


def _preflight_live_config_legacy(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the original artifact-backed preflight issue set."""

    blockers: list[str] = []
    live = config.get("live_adapter")
    if not isinstance(live, Mapping):
        return ("live_adapter_config_missing",)
    if live.get("schema") != LIVE_IO_SCHEMA:
        blockers.append("live_adapter_schema_invalid")
    blockers.extend(
        _receipt_blockers(
            live.get("demo_relation_qualification_receipt"),
            "demo_tube_tcp_relation",
        )
    )
    camera = live.get("camera")
    robot = live.get("robot")
    tactile = live.get("tactile")
    scene = live.get("scene_interpreter")
    socket = live.get("socket")
    for value, label in (
        (camera, "camera"),
        (robot, "robot"),
        (tactile, "tactile"),
        (scene, "scene_interpreter"),
        (socket, "socket"),
    ):
        if not isinstance(value, Mapping):
            blockers.append(f"{label}_config_missing")
    if isinstance(camera, Mapping):
        if not camera.get("serial"):
            blockers.append("d435_serial_missing")
        blockers.extend(_artifact_blockers(camera, "camera_extrinsic", "fixed_camera_extrinsic"))
    if isinstance(scene, Mapping):
        blockers.extend(_factory_blockers(scene, "scene_interpreter"))
        blockers.extend(
            _receipt_blockers(scene.get("qualification_receipt"), "tube_scene_interpreter")
        )
        blockers.extend(_resident_scene_pipeline_blockers(scene, config))
    if isinstance(socket, Mapping):
        blockers.extend(_artifact_blockers(socket, "pose", "static_socket_pose"))
    if isinstance(tactile, Mapping):
        profile = tactile.get("gf220_profile")
        profile_path = _path_or_none(profile)
        if profile_path is None or not profile_path.is_file():
            blockers.append("gf220_profile_missing")
        blockers.extend(_factory_blockers(tactile, "tactile_interpreter"))
        blockers.extend(
            _receipt_blockers(tactile.get("qualification_receipt"), "test_tube_tactile_interpreter")
        )
        blockers.extend(_resident_tactile_pipeline_blockers(tactile))
        port = tactile.get("hande_port")
        if not isinstance(port, str) or not port.startswith("/"):
            blockers.append("hande_absolute_port_missing")
    if isinstance(robot, Mapping):
        blockers.extend(_tcp_calibration_blockers(robot))
        blockers.extend(
            _receipt_blockers(
                robot.get("collision_payload_qualification_receipt"),
                "fr3_hande_collision_payload",
            )
        )
        moveit = robot.get("moveit")
        if not isinstance(moveit, Mapping):
            blockers.append("moveit_config_missing")
        else:
            script = moveit.get("pose_goal_script")
            if script is None or not _path_or_none(script).is_file():
                blockers.append("moveit_pose_goal_script_missing")
            setup_scripts = moveit.get("setup_scripts")
            if isinstance(setup_scripts, (str, bytes)) or not isinstance(setup_scripts, Sequence):
                blockers.append("moveit_setup_scripts_missing")
            else:
                for index, setup in enumerate(setup_scripts):
                    path = _path_or_none(setup)
                    if path is None or not path.is_file():
                        blockers.append(f"moveit_setup_script_{index}_missing")
        servo = robot.get("servo")
        if not isinstance(servo, Mapping):
            blockers.append("moveit_servo_config_missing")
        else:
            try:
                from ament_index_python.packages import get_package_prefix

                get_package_prefix("moveit_servo")
            except Exception:
                blockers.append("moveit_servo_package_missing")
    service = config.get("pose_backend")
    if not isinstance(service, Mapping) or not service.get("service_url"):
        blockers.append("pose_service_url_missing")
    elif isinstance(service, Mapping):
        aggregate_required = frozenset(
            SOURCE_POSE_CAPABILITIES | HELD_POSE_CAPABILITIES | SOCKET_POSE_CAPABILITIES
        )
        blockers.extend(
            _pose_receipt_blockers(
                service.get("qualification_receipt"),
                backend=str(service.get("name", "external")),
                required=aggregate_required,
                label="aggregate_pose_producers",
            )
        )
        blockers.extend(
            _pose_receipt_blockers(
                service.get("source_qualification_receipt"),
                backend="f2g_pose",
                required=SOURCE_POSE_CAPABILITIES,
                label="source_f2g_pose",
            )
        )
    geometry = config.get("geometry")
    if not isinstance(geometry, Mapping):
        blockers.append("geometry_missing")
    else:
        legacy_relation = geometry.get("T_tube_tcp_demo_grasp")
        normalized_relation = geometry.get("normalized_demo_grasp")
        if legacy_relation is None and normalized_relation is None:
            blockers.append("demo_T_tube_tcp_or_normalized_relation_missing")
            # Retain the original blocker for tools written against the v1
            # template while making the accepted alternative explicit.
            blockers.append("demo_T_tube_tcp_missing")
        elif legacy_relation is not None and normalized_relation is not None:
            blockers.append("demo_relation_modes_ambiguous")
        elif normalized_relation is not None:
            blockers.extend(_normalized_relation_blockers(normalized_relation))
        if geometry.get("relation_runtime_qualified") is not True:
            blockers.append("demo_relation_not_runtime_qualified")
        if geometry.get("T_socket_tube_entry") is None:
            blockers.append("T_socket_tube_entry_missing")
        if geometry.get("socket_entry_runtime_qualified") is not True:
            blockers.append("T_socket_tube_entry_not_runtime_qualified")
    return tuple(sorted(set(blockers)))


def _normalized_relation_blockers(value: object) -> list[str]:
    """Validate the non-6D demo relation without inventing physical inputs."""

    if not isinstance(value, Mapping):
        return ["normalized_demo_relation_invalid"]
    blockers: list[str] = []
    try:
        fraction = float(value.get("axial_fraction_closed_to_open"))
    except (TypeError, ValueError):
        fraction = math.nan
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        blockers.append("normalized_demo_axial_fraction_missing_or_invalid")
    try:
        length = float(value.get("tube_length_m"))
    except (TypeError, ValueError):
        length = math.nan
    if not math.isfinite(length) or length <= 0.0:
        blockers.append("measured_tube_length_missing")
    source = value.get("tube_length_source")
    if not isinstance(source, str) or not source.strip():
        blockers.append("tube_length_provenance_missing")
    for field_name, blocker in (
        ("preferred_jaw_axis_base", "preferred_jaw_axis_base_missing_or_invalid"),
        ("top_approach_direction_base", "top_approach_direction_base_missing_or_invalid"),
    ):
        try:
            vector = np.asarray(value.get(field_name), dtype=float)
        except (TypeError, ValueError):
            vector = np.asarray([])
        if (
            vector.shape != (3,)
            or not np.isfinite(vector).all()
            or float(np.linalg.norm(vector)) <= 1e-12
        ):
            blockers.append(blocker)
    try:
        top_axis = np.asarray(value.get("top_approach_direction_base"), dtype=float)
    except (TypeError, ValueError):
        top_axis = np.asarray([])
    if top_axis.shape == (3,) and np.isfinite(top_axis).all():
        top_norm = float(np.linalg.norm(top_axis))
        if top_norm > 1e-12 and not np.allclose(top_axis / top_norm, [0.0, 0.0, -1.0], atol=1e-9):
            blockers.append("top_approach_direction_must_be_exact_base_minus_z")
    if value.get("tube_pose_origin") != "geometric_centre":
        blockers.append("live_tube_pose_origin_not_geometric_centre")
    if value.get("tube_axis_contract") != "closed_to_open":
        blockers.append("live_tube_axis_not_signed_closed_to_open")
    return blockers


def _resident_scene_pipeline_blockers(
    scene: Mapping[str, Any], root: Mapping[str, Any]
) -> list[str]:
    """Validate the concrete source-to-held scene chain without opening devices.

    The factory itself performs the final hash/session checks.  This earlier
    pass exists so null or plainly invalid physical inputs stop the one-command
    launcher before it starts F2G or ROS infrastructure.
    """

    source_factory = (
        "demo2gripper.test_tube_pose_tracker:create_static_source_tube_scene_interpreter"
    )
    downstream_factory = (
        "demo2gripper.test_tube_downstream_scene:create_qualified_held_tube_scene_interpreter"
    )
    if scene.get("factory") != source_factory:
        return []
    blockers: list[str] = []
    source = scene.get("config")
    if not isinstance(source, Mapping):
        return ["source_scene_config_missing"]

    seed_rgb = _path_or_none(source.get("reviewed_seed_rgb"))
    if seed_rgb is None or not seed_rgb.is_file():
        blockers.append("source_reviewed_seed_rgb_missing")
    mask = _path_or_none(source.get("reviewed_initial_mask"))
    if mask is None or not mask.is_file():
        blockers.append("source_reviewed_initial_mask_missing")
    open_uv = _finite_array(source.get("reviewed_open_uv"), (2,))
    closed_uv = _finite_array(source.get("reviewed_closed_uv"), (2,))
    if open_uv is None or closed_uv is None or np.linalg.norm(open_uv - closed_uv) <= 1.0:
        blockers.append("source_reviewed_endpoints_missing_or_invalid")
    evidence_id = source.get("reviewed_endpoints_evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        blockers.append("source_reviewed_endpoints_evidence_missing")
    source_serial = source.get("camera_serial")
    camera = root.get("live_adapter", {}).get("camera")
    expected_serial = camera.get("serial") if isinstance(camera, Mapping) else None
    if not isinstance(source_serial, str) or source_serial != expected_serial:
        blockers.append("source_seed_camera_serial_mismatch")
    source_intrinsics = _path_or_none(source.get("intrinsics"))
    if source_intrinsics is None or not source_intrinsics.is_file():
        blockers.append("source_seed_camera_intrinsics_missing")
    source_camera_receipt = _path_or_none(source.get("camera_extrinsic_qualification_receipt"))
    blockers.extend(
        _receipt_blockers(
            source.get("camera_extrinsic_qualification_receipt"),
            "fixed_camera_extrinsic",
        )
    )
    source_camera_payload: Mapping[str, Any] | None = None
    if source_camera_receipt is not None and source_camera_receipt.is_file():
        try:
            source_camera_payload = _read_json(source_camera_receipt)
        except (OSError, ValueError, json.JSONDecodeError):
            source_camera_payload = None
    seed_receipt = _path_or_none(source.get("reviewed_seed_qualification_receipt"))
    seed_payload: Mapping[str, Any] | None = None
    if seed_receipt is None or not seed_receipt.is_file():
        blockers.append("source_reviewed_seed_qualification_receipt_missing")
    else:
        try:
            seed_payload = _read_json(seed_receipt)
        except (OSError, ValueError, json.JSONDecodeError):
            blockers.append("source_reviewed_seed_qualification_receipt_invalid")
        else:
            receipt_open_uv = _finite_array(seed_payload.get("reviewed_open_uv"), (2,))
            receipt_closed_uv = _finite_array(seed_payload.get("reviewed_closed_uv"), (2,))
            valid_seed = bool(
                seed_payload.get("schema") == "demo2gripper.test_tube_source_seed_qualification.v1"
                and seed_payload.get("status") == "accepted"
                and seed_payload.get("runtime_qualified") is True
                and isinstance(seed_payload.get("qualification_id"), str)
                and seed_payload["qualification_id"]
                and isinstance(seed_payload.get("workspace_calibration_id"), str)
                and seed_payload["workspace_calibration_id"]
                and isinstance(seed_payload.get("camera_extrinsic_qualification_id"), str)
                and seed_payload["camera_extrinsic_qualification_id"]
                and seed_payload.get("camera_serial") == source_serial
                and source_intrinsics is not None
                and source_intrinsics.is_file()
                and seed_payload.get("intrinsics_sha256") == _sha256_file(source_intrinsics)
                and source_camera_payload is not None
                and seed_payload.get("workspace_calibration_id")
                == source_camera_payload.get("workspace_calibration_id")
                and seed_payload.get("camera_extrinsic_qualification_id")
                == source_camera_payload.get("qualification_id")
                and seed_payload.get("reviewed_endpoints_evidence_id") == evidence_id
                and seed_rgb is not None
                and seed_rgb.is_file()
                and seed_payload.get("seed_rgb_sha256") == _sha256_file(seed_rgb)
                and mask is not None
                and mask.is_file()
                and seed_payload.get("reviewed_mask_sha256") == _sha256_file(mask)
                and open_uv is not None
                and closed_uv is not None
                and receipt_open_uv is not None
                and receipt_closed_uv is not None
                and np.allclose(
                    receipt_open_uv,
                    open_uv,
                    atol=1e-6,
                    rtol=0.0,
                )
                and np.allclose(
                    receipt_closed_uv,
                    closed_uv,
                    atol=1e-6,
                    rtol=0.0,
                )
            )
            if not valid_seed:
                blockers.append("source_reviewed_seed_not_runtime_qualified")

    if source.get("downstream_factory") != downstream_factory:
        blockers.append("qualified_held_tube_downstream_factory_missing")
        return blockers
    downstream = source.get("downstream_config")
    if not isinstance(downstream, Mapping):
        blockers.append("qualified_held_tube_downstream_config_missing")
        return blockers

    live = root.get("live_adapter")
    outer_camera = live.get("camera") if isinstance(live, Mapping) else None
    outer_socket = live.get("socket") if isinstance(live, Mapping) else None
    geometry = root.get("geometry")
    controller_policy = root.get("policy", {})

    def same_path(
        left: object,
        right: object,
        blocker: str,
    ) -> None:
        if _path_or_none(left) != _path_or_none(right):
            blockers.append(blocker)

    if isinstance(outer_camera, Mapping):
        same_path(
            downstream.get("camera_extrinsic"),
            outer_camera.get("camera_extrinsic"),
            "source_downstream_camera_extrinsic_mismatch",
        )
        same_path(
            downstream.get("camera_extrinsic_qualification_receipt"),
            outer_camera.get("camera_extrinsic_qualification_receipt"),
            "source_downstream_live_camera_receipt_mismatch",
        )
        same_path(
            source.get("camera_extrinsic_qualification_receipt"),
            outer_camera.get("camera_extrinsic_qualification_receipt"),
            "source_live_camera_receipt_mismatch",
        )
    if isinstance(outer_socket, Mapping):
        same_path(
            downstream.get("static_socket_pose"),
            outer_socket.get("pose"),
            "downstream_live_socket_pose_mismatch",
        )
        same_path(
            downstream.get("static_socket_pose_qualification_receipt"),
            outer_socket.get("pose_qualification_receipt"),
            "downstream_live_socket_receipt_mismatch",
        )
    same_path(
        downstream.get("scene_qualification_receipt"),
        scene.get("qualification_receipt"),
        "downstream_live_scene_receipt_mismatch",
    )

    for key, blocker in (
        ("camera_extrinsic", "downstream_camera_extrinsic_artifact_missing"),
        ("intrinsics", "downstream_camera_intrinsics_missing"),
        ("static_socket_pose", "downstream_static_socket_pose_artifact_missing"),
        (
            "rgb_tracker_qualification_report",
            "held_rgb_tracker_qualification_report_missing",
        ),
        ("pivot_reference", "held_tube_pivot_reference_missing"),
    ):
        path = _path_or_none(downstream.get(key))
        if path is None or not path.is_file():
            blockers.append(blocker)

    blockers.extend(
        _receipt_blockers(
            downstream.get("camera_extrinsic_qualification_receipt"),
            "fixed_camera_extrinsic",
        )
    )
    blockers.extend(
        _receipt_blockers(
            downstream.get("static_socket_pose_qualification_receipt"),
            "static_socket_pose",
        )
    )
    blockers.extend(
        _receipt_blockers(
            downstream.get("scene_qualification_receipt"),
            "tube_scene_interpreter",
        )
    )

    rgb_report = _path_or_none(downstream.get("rgb_tracker_qualification_report"))
    if rgb_report is not None and rgb_report.is_file():
        try:
            rgb_payload = _read_json(rgb_report)
        except (OSError, ValueError, json.JSONDecodeError):
            blockers.append("held_rgb_tracker_qualification_report_invalid")
        else:
            if (
                rgb_payload.get("schema") != "demo2gripper.test_tube_rgb_tracker_qualification.v2"
                or rgb_payload.get("runtime_gate_eligible") is not True
                or not isinstance(rgb_payload.get("qualification_id"), str)
                or not rgb_payload["qualification_id"]
            ):
                blockers.append("held_rgb_tracker_not_runtime_qualified")
    pivot_reference = _path_or_none(downstream.get("pivot_reference"))
    if pivot_reference is not None and pivot_reference.is_file():
        try:
            pivot_payload = _read_json(pivot_reference)
        except (OSError, ValueError, json.JSONDecodeError):
            blockers.append("held_tube_pivot_reference_invalid")
        else:
            if pivot_payload.get("schema") != "demo2gripper.test_tube_front_rgb_pivot_reference.v1":
                blockers.append("held_tube_pivot_reference_invalid")

    intrinsics = _path_or_none(downstream.get("intrinsics"))
    if intrinsics is not None and intrinsics.is_file():
        try:
            K = np.load(intrinsics, allow_pickle=False)
        except (OSError, ValueError):
            blockers.append("downstream_camera_intrinsics_invalid")
        else:
            if K.shape != (3, 3) or not np.isfinite(K).all():
                blockers.append("downstream_camera_intrinsics_invalid")

    downstream_serial = downstream.get("camera_serial")
    if (
        not isinstance(downstream_serial, str)
        or not downstream_serial
        or downstream_serial != expected_serial
    ):
        blockers.append("downstream_camera_serial_mismatch")
    if _path_or_none(downstream.get("intrinsics")) != source_intrinsics:
        blockers.append("source_downstream_intrinsics_mismatch")
    if (
        _path_or_none(downstream.get("camera_extrinsic_qualification_receipt"))
        != source_camera_receipt
    ):
        blockers.append("source_downstream_camera_receipt_mismatch")

    if _finite_array(downstream.get("reviewed_seed_line_uv"), (2, 2)) is None:
        blockers.append("held_tube_reviewed_seed_line_missing_or_invalid")
    roi = downstream.get("axis_roi_uvxy")
    if (
        isinstance(roi, (str, bytes))
        or not isinstance(roi, Sequence)
        or len(roi) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in roi)
        or not (0 <= roi[0] < roi[2] and 0 <= roi[1] < roi[3])
    ):
        blockers.append("held_tube_axis_roi_missing_or_invalid")
    for key in (
        "socket_outward_axis_socket",
        "pivot_plane_normal_tcp",
        "jaw_axis_tcp",
    ):
        vector = _finite_array(downstream.get(key), (3,))
        if vector is None or np.linalg.norm(vector) <= 1e-9:
            blockers.append(f"downstream_{key}_missing_or_invalid")
    if _finite_array(downstream.get("pivot_point_tcp"), (3,)) is None:
        blockers.append("downstream_pivot_point_tcp_missing_or_invalid")

    socket_entry = _finite_array(downstream.get("T_socket_tube_entry"), (4, 4))
    if socket_entry is None:
        blockers.append("downstream_T_socket_tube_entry_missing_or_invalid")
    root_socket_entry = (
        _finite_array(geometry.get("T_socket_tube_entry"), (4, 4))
        if isinstance(geometry, Mapping)
        else None
    )
    if (
        socket_entry is not None
        and root_socket_entry is not None
        and not np.allclose(socket_entry, root_socket_entry, atol=1e-9, rtol=0.0)
    ):
        blockers.append("downstream_controller_socket_entry_mismatch")
    policy = downstream.get("policy")
    if not isinstance(policy, Mapping):
        blockers.append("downstream_scene_policy_missing")
    else:
        for key, blocker in (
            ("tube_length_m", "downstream_measured_tube_length_missing"),
            ("insertion_target_depth_m", "downstream_insertion_depth_missing"),
        ):
            try:
                value = float(policy.get(key))
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value) or value <= 0.0:
                blockers.append(blocker)
        normalized = (
            geometry.get("normalized_demo_grasp") if isinstance(geometry, Mapping) else None
        )
        try:
            downstream_length = float(policy.get("tube_length_m"))
            root_length = float(
                normalized.get("tube_length_m") if isinstance(normalized, Mapping) else math.nan
            )
        except (TypeError, ValueError):
            downstream_length = root_length = math.nan
        if (
            math.isfinite(downstream_length)
            and math.isfinite(root_length)
            and not math.isclose(downstream_length, root_length, abs_tol=1e-9)
        ):
            blockers.append("downstream_controller_tube_length_mismatch")
        try:
            downstream_depth = float(policy.get("insertion_target_depth_m"))
            root_depth = float(
                controller_policy.get("insertion_target_depth_m", 0.020)
                if isinstance(controller_policy, Mapping)
                else 0.020
            )
        except (TypeError, ValueError):
            downstream_depth = root_depth = math.nan
        if (
            math.isfinite(downstream_depth)
            and math.isfinite(root_depth)
            and not math.isclose(downstream_depth, root_depth, abs_tol=1e-9)
        ):
            blockers.append("downstream_controller_insertion_depth_mismatch")
    if seed_payload is not None:
        camera_receipt = _path_or_none(downstream.get("camera_extrinsic_qualification_receipt"))
        if camera_receipt is not None and camera_receipt.is_file():
            try:
                camera_payload = _read_json(camera_receipt)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            else:
                if seed_payload.get("workspace_calibration_id") != camera_payload.get(
                    "workspace_calibration_id"
                ) or seed_payload.get("camera_extrinsic_qualification_id") != camera_payload.get(
                    "qualification_id"
                ):
                    blockers.append("source_seed_camera_calibration_binding_mismatch")
    if not blockers:
        # Both concrete factories are pure constructors: they validate hashes,
        # shared session IDs, rigid transforms, and policy shapes without
        # opening ROS, cameras, serial ports, or tactile devices.  Running that
        # constructor here keeps such failures ahead of managed infrastructure.
        interpreter: Any | None = None
        try:
            factory = _load_factory(source_factory)
            interpreter = factory(config=source, output_dir=REPOSITORY_ROOT)
        except Exception:  # noqa: BLE001 - converted to one stable static blocker
            blockers.append("resident_scene_factory_static_validation_failed")
        finally:
            close = getattr(interpreter, "close", None)
            if callable(close):
                close()
    return blockers


def _finite_array(value: object, shape: tuple[int, ...]) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if result.shape != shape or not np.isfinite(result).all():
        return None
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resident_tactile_pipeline_blockers(tactile: Mapping[str, Any]) -> list[str]:
    """Reject a draft tube tactile bundle during static preflight."""

    expected_factory = (
        "demo2gripper.test_tube_tactile_interpreter:create_test_tube_tactile_interpreter"
    )
    if tactile.get("factory") != expected_factory:
        return []
    config = tactile.get("config")
    if not isinstance(config, Mapping):
        return ["test_tube_tactile_config_missing"]
    bundle_path = _path_or_none(config.get("qualification_bundle"))
    if bundle_path is None or not bundle_path.is_file():
        return ["test_tube_tactile_qualification_bundle_missing"]
    try:
        bundle = _read_json(bundle_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return ["test_tube_tactile_qualification_bundle_invalid"]
    blockers: list[str] = []
    if (
        bundle.get("schema") != "demo2gripper.test_tube_tactile_interpreter_qualification.v1"
        or bundle.get("status") != "accepted"
        or bundle.get("runtime_qualified") is not True
        or not isinstance(bundle.get("qualification_id"), str)
        or not bundle["qualification_id"]
    ):
        blockers.append("test_tube_tactile_bundle_not_runtime_qualified")
    receipt_path = _path_or_none(tactile.get("qualification_receipt"))
    if receipt_path is not None and receipt_path.is_file():
        try:
            receipt = _read_json(receipt_path)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        else:
            if receipt.get("qualification_id") != bundle.get("qualification_id"):
                blockers.append("test_tube_tactile_receipt_bundle_mismatch")
    return blockers


def _bind_online_scene_camera(
    value: object,
    *,
    T_base_camera: np.ndarray,
    camera_serial: str,
) -> dict[str, Any]:
    """Inject one qualified camera contract into the composed online chain.

    The nested factories run before the D435 is opened, so they cannot fetch
    these installation values themselves.  They receive the already loaded,
    receipt-checked outer transform and serial rather than requiring duplicate
    paths in the task file.  Any explicit nested value must agree exactly.
    """

    root = dict(_mapping(value, "online scene config"))

    def bind_camera(layer: dict[str, Any], *, label: str) -> None:
        nested_serial = layer.get("camera_serial")
        if nested_serial is not None and nested_serial != camera_serial:
            raise ClosedLoopContractError(f"{label}.camera_serial differs from live camera")
        layer["camera_serial"] = camera_serial
        nested_transform = layer.get("T_base_camera")
        if nested_transform is not None:
            candidate = validate_rigid_transform(nested_transform, name=f"{label}.T_base_camera")
            if not np.allclose(candidate, T_base_camera, atol=1e-9, rtol=0.0):
                raise ClosedLoopContractError(
                    f"{label}.T_base_camera differs from qualified camera extrinsic"
                )
        layer["T_base_camera"] = np.asarray(T_base_camera, dtype=float).tolist()
        nested_qualification = layer.get("camera_extrinsic_qualified")
        if nested_qualification is not None and nested_qualification is not True:
            raise ClosedLoopContractError(
                f"{label}.camera_extrinsic_qualified contradicts qualified outer camera"
            )
        # This helper is called only after create_live_io has loaded and
        # receipt-checked the fixed-camera extrinsic.  Propagate that fact so
        # task-specific geometry may use base-frame height as a guarded
        # fallback without accepting an arbitrary transform from task config.
        layer["camera_extrinsic_qualified"] = True

    bind_camera(root, label="online source")
    if root.get("downstream_factory") != ONLINE_SOCKET_SCENE_FACTORY:
        raise ClosedLoopContractError("online source must use the online socket factory")
    socket = dict(_mapping(root.get("downstream_config"), "online socket config"))
    bind_camera(socket, label="online socket")
    if socket.get("downstream_factory") != ONLINE_HELD_SCENE_FACTORY:
        raise ClosedLoopContractError("online socket must use the online held-tube factory")
    held = dict(_mapping(socket.get("downstream_config"), "online held-tube config"))
    bind_camera(held, label="online held tube")
    socket["downstream_config"] = held
    root["downstream_config"] = socket
    return root


def create_live_io(
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    accept_arm_only_collision_model: bool = False,
    accept_provisional_camera_extrinsic: bool = False,
) -> LiveTestTubeIO:
    """Factory loaded by ``run_test_tube_closed_loop.py --live-adapter``."""

    collision_acknowledgement = reduced_collision_model_acknowledgement(
        config,
        accepted=accept_arm_only_collision_model,
    )
    camera_acknowledgement = provisional_camera_extrinsic_acknowledgement(
        config,
        accepted=accept_provisional_camera_extrinsic,
    )
    blockers = preflight_live_config(
        config,
        accept_arm_only_collision_model=accept_arm_only_collision_model,
        accept_provisional_camera_extrinsic=accept_provisional_camera_extrinsic,
    )
    if blockers:
        raise ClosedLoopContractError("live adapter preflight blockers: " + ", ".join(blockers))
    if accept_provisional_camera_extrinsic:
        atomic_write_json(
            output_dir / "provisional_camera_extrinsic_acknowledgement.json",
            camera_acknowledgement,
        )
    live = _mapping(config["live_adapter"], "live_adapter")
    camera_config = _mapping(live["camera"], "live_adapter.camera")
    tactile_config = _mapping(live["tactile"], "live_adapter.tactile")
    robot_config = _reduced_collision_robot_config(
        _mapping(live["robot"], "live_adapter.robot"),
        acknowledgement=collision_acknowledgement,
    )
    scene_config = _mapping(live["scene_interpreter"], "live_adapter.scene_interpreter")
    socket_config = _mapping(live["socket"], "live_adapter.socket")
    online_scene = scene_config.get("factory") == ONLINE_SCENE_FACTORY
    descriptor = (
        PoseBackendDescriptor(
            name=_string(config["pose_backend"].get("name", "external"), "pose_backend.name"),
            available=True,
            runtime_qualified=True,
            qualification_id="online-composed-pose-producers-v1",
            capabilities=frozenset(
                SOURCE_POSE_CAPABILITIES | HELD_POSE_CAPABILITIES | SOCKET_POSE_CAPABILITIES
            ),
            detail="resident online SAM3/F2G, held-tube FK/RGB, and RGB-D socket producers",
        )
        if online_scene
        else _pose_descriptor(config)
    )
    source_descriptor = (
        PoseBackendDescriptor(
            name="f2g_pose",
            available=True,
            runtime_qualified=True,
            qualification_id="online-source-f2g-crosschecked-current-session-v1",
            capabilities=SOURCE_POSE_CAPABILITIES,
            detail="current-frame F2G proposal gated by SAM3 identity, RGB-D, and temporal checks",
        )
        if online_scene
        else _source_pose_descriptor(config)
    )
    camera_receipt = (
        None
        if camera_acknowledgement["active"] is True
        else _qualified_live_receipt(
            camera_config["camera_extrinsic_qualification_receipt"],
            "fixed_camera_extrinsic",
        )
    )
    if camera_receipt is None:
        camera_uncertainty = _mapping(
            camera_acknowledgement["derived_uncertainty"],
            "provisional camera derived_uncertainty",
        )
        camera_position_std_m = _nonnegative(
            camera_uncertainty.get("position_std_m"),
            "provisional camera position_std_m",
        )
        camera_orientation_std_deg = _nonnegative(
            camera_uncertainty.get("orientation_std_deg"),
            "provisional camera orientation_std_deg",
        )
    else:
        camera_position_std_m = float(camera_receipt["position_std_m"])
        camera_orientation_std_deg = float(camera_receipt["orientation_std_deg"])
    tcp_uncertainty = _tcp_calibration_uncertainty(robot_config)
    demo_relation_receipt = (
        None
        if online_scene
        else _qualified_live_receipt(
            live["demo_relation_qualification_receipt"], "demo_tube_tcp_relation"
        )
    )
    scene_receipt = (
        None
        if online_scene
        else _qualified_live_receipt(
            scene_config["qualification_receipt"], "tube_scene_interpreter"
        )
    )
    socket_mode = str(socket_config.get("mode", "static"))
    if socket_mode not in {"static", "online_entry_frame"}:
        raise ClosedLoopContractError(
            "live_adapter.socket.mode must be static or online_entry_frame"
        )
    socket_receipt = (
        _qualified_live_receipt(socket_config["pose_qualification_receipt"], "static_socket_pose")
        if socket_mode == "static"
        else None
    )
    held_descriptor = PoseBackendDescriptor(
        name="online_held_tube_pose" if online_scene else "qualified_held_tube_pose",
        available=True,
        runtime_qualified=True,
        qualification_id=(
            "online-held-tube-current-session-v1"
            if scene_receipt is None
            else _string(scene_receipt["qualification_id"], "qualification_id")
        ),
        capabilities=HELD_POSE_CAPABILITIES,
        detail="RGB/tactile pivot pose followed by frozen robot-relative kinematics",
    )
    socket_descriptor = PoseBackendDescriptor(
        name=(
            "qualified_static_socket_pose"
            if socket_mode == "static"
            else "online_rgbd_socket_entry"
        ),
        available=True,
        runtime_qualified=True,
        qualification_id=(
            _string(socket_receipt["qualification_id"], "qualification_id")
            if socket_receipt is not None
            else _string(
                socket_config.get("runtime_qualification_id", "online-rgbd-socket-entry-v1"),
                "socket.runtime_qualification_id",
            )
        ),
        capabilities=SOCKET_POSE_CAPABILITIES,
        detail=(
            "measured fixed rack socket pose"
            if socket_mode == "static"
            else "temporally qualified current-frame RGB-D socket entry"
        ),
    )
    T_base_camera = _load_artifact_transform(camera_config, "camera_extrinsic")
    T_link8_tcp = _load_artifact_transform(robot_config, "tcp")
    T_base_socket = (
        _load_artifact_transform(socket_config, "pose") if socket_mode == "static" else None
    )
    geometry = _mapping(config["geometry"], "geometry")
    raw_demo_transform = geometry.get("T_tube_tcp_demo_grasp")
    T_tube_tcp_demo = (
        None
        if raw_demo_transform is None
        else validate_rigid_transform(raw_demo_transform, name="geometry.T_tube_tcp_demo_grasp")
    )
    scene_factory = _load_factory(scene_config["factory"])
    tactile_factory = _load_factory(tactile_config["factory"])
    servo_graph_blockers = _servo_graph_blockers(robot_config)
    if servo_graph_blockers:
        raise ClosedLoopContractError(
            "live adapter preflight blockers: " + ", ".join(servo_graph_blockers)
        )
    # Verify the remote inference boundary before opening any local device.
    # A missing/stale F2G service must not leave the D435, tactile cameras,
    # Hand-E serial link, or FR3 ROS handles partially constructed.
    backend = _mapping(config["pose_backend"], "pose_backend")
    endpoint = _string(backend["service_url"], "pose_backend.service_url")
    timeout_s = float(backend.get("timeout_s", 0.25))
    token = backend.get("auth_token")
    client_a = PoseServiceClient(endpoint, timeout_s=timeout_s, auth_token=token)
    health = client_a.health()
    if health.get("ready") is not True:
        raise LiveIOError(f"pose service health failed: {health}")
    service_backend_name = _string(
        backend.get("service_backend_name", "f2g_pose"), "service backend"
    )
    health_backend = health.get("backend")
    if (
        not isinstance(health_backend, Mapping)
        or health_backend.get("name") != service_backend_name
        or health_backend.get("pose_only") is not True
    ):
        raise LiveIOError("pose service backend identity does not match configuration")
    guard = PoseObservationGuardConfig(
        max_staleness_s=float(backend.get("max_staleness_s", 0.25)),
        max_server_latency_s=float(backend.get("max_server_latency_s", 0.20)),
        max_roundtrip_latency_s=float(backend.get("max_roundtrip_latency_s", 0.25)),
        allowed_backends=frozenset({service_backend_name}),
    )
    resources: list[Any] = []
    try:
        raw_scene_factory_config = scene_config.get("config", {})
        scene_factory_config = (
            _bind_online_scene_camera(
                raw_scene_factory_config,
                T_base_camera=T_base_camera,
                camera_serial=_string(camera_config["serial"], "camera.serial"),
            )
            if online_scene
            else raw_scene_factory_config
        )
        scene = scene_factory(config=scene_factory_config, output_dir=output_dir)
        resources.append(scene)
        camera = D435LiveSource(
            D435LiveConfig(
                serial=_string(camera_config["serial"], "camera.serial"),
                width=int(camera_config.get("width", 640)),
                height=int(camera_config.get("height", 480)),
                fps=int(camera_config.get("fps", 30)),
                warmup_frames=int(camera_config.get("warmup_frames", 15)),
                frame_timeout_ms=int(camera_config.get("frame_timeout_ms", 1000)),
                max_pair_skew_ms=float(camera_config.get("max_pair_skew_ms", 20.0)),
            )
        ).start()
        resources.append(camera)
        tactile_source = GF220LiveSource(
            _path(tactile_config["gf220_profile"], "tactile.gf220_profile"),
            config=GF220LiveConfig(
                buffer_frames=int(tactile_config.get("buffer_frames", 32)),
                warmup_frames=int(tactile_config.get("warmup_frames", 8)),
                warp_validation_frames=int(tactile_config.get("warp_validation_frames", 2)),
                max_consecutive_sdk_errors=int(
                    tactile_config.get("max_consecutive_sdk_errors", 3)
                ),
                sdk_error_retry_delay_s=float(
                    tactile_config.get("sdk_error_retry_delay_s", 0.01)
                ),
                # A four-frame classifier window spans about 0.23--0.31 s on
                # the two independent USB GF220 streams on this workstation.
                # Keep the host-clock pair-skew gate strict, but do not fall
                # back to GF220LiveConfig's 0.15 s single-frame default: that
                # makes every physically obtainable four-frame window stale.
                max_frame_age_s=float(tactile_config.get("max_frame_age_s", 0.5)),
                max_pair_skew_s=float(tactile_config.get("max_pair_skew_s", 0.08)),
            ),
        ).start(timeout_s=float(tactile_config.get("startup_timeout_s", 15.0)))
        resources.append(tactile_source)
        calibrations = tactile_source.calibrate_unloaded(
            timeout_s=float(tactile_config.get("calibration_timeout_s", 15.0))
        )
        tactile = tactile_factory(
            config=tactile_config.get("config", {}),
            calibrations=calibrations,
            output_dir=output_dir,
        )
        resources.append(tactile)
        transport = SerialModbusRTU(
            _string(tactile_config["hande_port"], "tactile.hande_port"),
            float(tactile_config.get("modbus_timeout_s", 0.12)),
        )
        gripper = HandEModbusRuntime(transport)
        resources.append(gripper)
        initial = gripper.feedback()
        if not initial.fault_free:
            raise LiveIOError("Hand-E is not activated and fault-free")
        arm = RosMoveItServoRuntime(
            config=robot_config, T_link8_tcp=T_link8_tcp, output_dir=output_dir
        )
        resources.append(arm)
        return LiveTestTubeIO(
            descriptor=descriptor,
            source_descriptor=source_descriptor,
            held_descriptor=held_descriptor,
            socket_descriptor=socket_descriptor,
            camera=camera,
            tactile_source=tactile_source,
            scene=scene,
            tactile=tactile,
            arm=arm,
            gripper=gripper,
            tube_pose_provider=RemotePoseProvider(client_a, guard_config=guard),
            T_base_camera=T_base_camera,
            T_tube_tcp_demo=T_tube_tcp_demo,
            T_base_socket=T_base_socket,
            camera_position_std_m=camera_position_std_m,
            camera_orientation_std_deg=camera_orientation_std_deg,
            tcp_position_std_m=tcp_uncertainty[0],
            tcp_orientation_std_deg=tcp_uncertainty[1],
            demo_relation_position_std_m=(
                0.0
                if demo_relation_receipt is None
                else float(demo_relation_receipt["position_std_m"])
            ),
            demo_relation_orientation_std_deg=(
                0.0
                if demo_relation_receipt is None
                else float(demo_relation_receipt["orientation_std_deg"])
            ),
            held_position_std_limit_m=(
                float(
                    _mapping(config.get("policy", {}), "policy").get(
                        "pose_max_position_std_m", 0.008
                    )
                )
                if scene_receipt is None
                else float(scene_receipt["position_std_m"])
            ),
            held_orientation_std_limit_deg=(
                float(
                    _mapping(config.get("policy", {}), "policy").get(
                        "pose_max_orientation_std_deg", 6.0
                    )
                )
                if scene_receipt is None
                else float(scene_receipt["orientation_std_deg"])
            ),
            socket_position_std_m=(
                float(socket_receipt["position_std_m"])
                if socket_receipt is not None
                else float(socket_config.get("max_position_std_m", 0.004))
            ),
            socket_orientation_std_deg=(
                float(socket_receipt["orientation_std_deg"])
                if socket_receipt is not None
                else float(socket_config.get("max_orientation_std_deg", 4.0))
            ),
            tactile_samples=int(tactile_config.get("runtime_samples", 4)),
            tactile_timeout_s=float(tactile_config.get("runtime_timeout_s", 0.25)),
            tactile_cache_max_age_s=float(
                _mapping(config.get("policy", {}), "policy").get("tactile_max_age_s", 0.20)
            ),
            source_pose_cache_max_age_s=float(
                _mapping(config.get("policy", {}), "policy").get("pose_max_age_s", 0.20)
            ),
            close_timeout_s=float(
                _mapping(config.get("policy", {}), "policy").get("close_timeout_s", 6.0)
            ),
            final_approach_commit_distance_m=_mapping(config.get("policy", {}), "policy").get(
                "final_approach_commit_distance_m"
            ),
            final_approach_timeout_s=float(_mapping(config.get("policy", {}), "policy").get(
                "final_approach_timeout_s", 8.0
            )),
            servo_translation_deadband_m=float(_mapping(config.get("policy", {}), "policy").get(
                "servo_translation_deadband_m", .0025
            )),
            servo_rotation_deadband_deg=float(_mapping(config.get("policy", {}), "policy").get(
                "servo_rotation_deadband_deg", 2.0
            )),
        )
    except BaseException:
        for resource in reversed(resources):
            try:
                resource.close()
            except Exception:
                pass
        raise


def _pose_descriptor(config: Mapping[str, Any]) -> PoseBackendDescriptor:
    backend = _mapping(config.get("pose_backend"), "pose_backend")
    receipt_path = _path(backend.get("qualification_receipt"), "pose qualification receipt")
    receipt = _read_json(receipt_path)
    if receipt.get("schema") != POSE_QUALIFICATION_SCHEMA:
        raise ClosedLoopContractError("pose qualification receipt schema is invalid")
    capabilities = frozenset(PoseCapability(item) for item in receipt["capabilities"])
    return PoseBackendDescriptor(
        name=_string(backend.get("name"), "pose_backend.name"),
        available=True,
        runtime_qualified=True,
        qualification_id=_string(receipt.get("qualification_id"), "qualification_id"),
        capabilities=capabilities,
        detail="external 6D pose adapter",
    )


def _source_pose_descriptor(config: Mapping[str, Any]) -> PoseBackendDescriptor:
    backend = _mapping(config.get("pose_backend"), "pose_backend")
    receipt_path = _path(
        backend.get("source_qualification_receipt"),
        "source pose qualification receipt",
    )
    receipt = _read_json(receipt_path)
    capabilities = frozenset(PoseCapability(item) for item in receipt["capabilities"])
    if (
        receipt.get("schema") != POSE_QUALIFICATION_SCHEMA
        or receipt.get("backend") != "f2g_pose"
        or receipt.get("runtime_qualified") is not True
        or not SOURCE_POSE_CAPABILITIES <= capabilities
    ):
        raise ClosedLoopContractError("source F2G-Pose qualification receipt is invalid")
    return PoseBackendDescriptor(
        name="f2g_pose",
        available=True,
        runtime_qualified=True,
        qualification_id=_string(receipt.get("qualification_id"), "qualification_id"),
        capabilities=capabilities,
        detail="source-tube RGB-D proposal plus qualified sign/roll canonicalizer",
    )


def _qualified_live_receipt(value: object, expected_kind: str) -> dict[str, Any]:
    path = _path(value, f"{expected_kind} qualification receipt")
    receipt = _read_json(path)
    blockers = _receipt_blockers(str(path), expected_kind)
    if blockers:
        raise ClosedLoopContractError(
            f"{expected_kind} qualification receipt is invalid: {', '.join(blockers)}"
        )
    return receipt


def _artifact_blockers(config: Mapping[str, Any], prefix: str, receipt_kind: str) -> list[str]:
    blockers: list[str] = []
    value = config.get(prefix)
    if value is None or not _path_or_none(value).is_file():
        blockers.append(f"{prefix}_artifact_missing")
    blockers.extend(_receipt_blockers(config.get(f"{prefix}_qualification_receipt"), receipt_kind))
    return blockers


def _tcp_calibration_blockers(robot: Mapping[str, Any]) -> list[str]:
    """Accept either the legacy receipt or one installation-bound static record.

    A flange-to-TCP transform is fixed by the mounted tool and cannot be
    inferred uniquely from each task image.  The static path therefore checks
    that the companion measurement names the exact NPY artifact and that the
    manifest supplies a stable installation id plus conservative error bounds.
    It is reusable across sessions; moving/remounting the gripper requires a new
    installation id/calibration.
    """

    tcp_path = _path_or_none(robot.get("tcp"))
    if tcp_path is None or not tcp_path.is_file():
        return ["tcp_artifact_missing"]
    static_value = robot.get("tcp_static_calibration")
    if static_value is None:
        return _receipt_blockers(robot.get("tcp_qualification_receipt"), "hande_tcp")
    metadata_path = _path_or_none(static_value)
    if metadata_path is None or not metadata_path.is_file():
        return ["hande_tcp_static_calibration_missing"]
    blockers: list[str] = []
    try:
        metadata = _read_json(metadata_path)
        if metadata.get("schema") != "demo2gripper.tool_transform_measurement.v1":
            blockers.append("hande_tcp_static_calibration_schema_invalid")
        reference = metadata.get("transform")
        if not isinstance(reference, str) or not reference:
            blockers.append("hande_tcp_static_transform_reference_missing")
        else:
            referenced = (metadata_path.parent / reference).resolve()
            if referenced != tcp_path.resolve():
                blockers.append("hande_tcp_static_transform_artifact_mismatch")
        matrix = _load_artifact_transform(robot, "tcp")
        translation = _finite_array(metadata.get("translation_m"), (3,))
        rotation = _finite_array(metadata.get("rotation_matrix"), (3, 3))
        if (
            translation is None
            or rotation is None
            or not np.allclose(matrix[:3, 3], translation, atol=1e-9, rtol=0.0)
            or not np.allclose(matrix[:3, :3], rotation, atol=1e-9, rtol=0.0)
        ):
            blockers.append("hande_tcp_static_metadata_matrix_mismatch")
        evidence = metadata.get("evidence")
        if (
            not isinstance(evidence, Mapping)
            or not isinstance(evidence.get("method"), str)
            or not evidence["method"].strip()
        ):
            blockers.append("hande_tcp_static_measurement_evidence_missing")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        blockers.append("hande_tcp_static_calibration_invalid")
    installation_id = robot.get("tcp_installation_id")
    if not isinstance(installation_id, str) or not installation_id.strip():
        blockers.append("hande_tcp_installation_id_missing")
    for key, blocker in (
        ("tcp_position_bound_m", "hande_tcp_position_bound_missing"),
        ("tcp_orientation_bound_deg", "hande_tcp_orientation_bound_missing"),
    ):
        if not _is_positive_number(robot.get(key)):
            blockers.append(blocker)
    return blockers


def _tcp_calibration_uncertainty(robot: Mapping[str, Any]) -> tuple[float, float]:
    """Return uncertainty from the selected reusable/static or legacy path."""

    if robot.get("tcp_static_calibration") is not None:
        blockers = _tcp_calibration_blockers(robot)
        if blockers:
            raise ClosedLoopContractError(
                "static Hand-E TCP calibration is invalid: " + ", ".join(blockers)
            )
        return (
            float(robot["tcp_position_bound_m"]),
            float(robot["tcp_orientation_bound_deg"]),
        )
    receipt = _qualified_live_receipt(robot.get("tcp_qualification_receipt"), "hande_tcp")
    return float(receipt["position_std_m"]), float(receipt["orientation_std_deg"])


def _receipt_blockers(value: object, expected_kind: str) -> list[str]:
    path = _path_or_none(value)
    if path is None or not path.is_file():
        return [f"{expected_kind}_qualification_receipt_missing"]
    try:
        receipt = _read_json(path)
    except (OSError, ValueError):
        return [f"{expected_kind}_qualification_receipt_invalid"]
    valid = bool(
        receipt.get("schema") == LIVE_QUALIFICATION_SCHEMA
        and receipt.get("kind") == expected_kind
        and receipt.get("runtime_qualified") is True
        and isinstance(receipt.get("qualification_id"), str)
        and receipt["qualification_id"]
    )
    if expected_kind in {
        "fixed_camera_extrinsic",
        "hande_tcp",
        "tube_scene_interpreter",
        "static_socket_pose",
        "demo_tube_tcp_relation",
    }:
        valid = bool(
            valid
            and _is_nonnegative_number(receipt.get("position_std_m"))
            and _is_nonnegative_number(receipt.get("orientation_std_deg"))
        )
    return [] if valid else [f"{expected_kind}_not_runtime_qualified"]


def _pose_receipt_blockers(
    value: object,
    *,
    backend: str,
    required: frozenset[PoseCapability],
    label: str,
) -> list[str]:
    path = _path_or_none(value)
    if path is None or not path.is_file():
        return [f"{label}_qualification_receipt_missing"]
    try:
        receipt = _read_json(path)
        capabilities = frozenset(PoseCapability(item) for item in receipt.get("capabilities", []))
    except (OSError, ValueError, TypeError):
        return [f"{label}_qualification_receipt_invalid"]
    valid = bool(
        receipt.get("schema") == POSE_QUALIFICATION_SCHEMA
        and receipt.get("backend") == backend
        and receipt.get("runtime_qualified") is True
        and isinstance(receipt.get("qualification_id"), str)
        and receipt["qualification_id"]
        and required <= capabilities
    )
    return [] if valid else [f"{label}_not_runtime_qualified"]


def _factory_blockers(config: Mapping[str, Any], label: str) -> list[str]:
    value = config.get("factory")
    if not isinstance(value, str) or ":" not in value:
        return [f"{label}_factory_missing"]
    module_name, _ = value.split(":", 1)
    try:
        available = importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        available = False
    return [] if available else [f"{label}_module_unavailable"]


def _servo_graph_blockers(robot_config: Mapping[str, Any]) -> tuple[str, ...]:
    """Read-only ROS graph check performed before any device is opened."""

    servo = _mapping(robot_config.get("servo"), "robot.servo")
    node_name = _string(servo.get("node_name", "/servo_node"), "servo.node_name")
    topic = _string(
        servo.get("twist_topic", "/servo_node/delta_twist_cmds"),
        "servo.twist_topic",
    )
    status_topic = _string(
        servo.get("status_topic", "/servo_node/status"),
        "servo.status_topic",
    )
    switch_service = _string(
        servo.get(
            "switch_command_type_service",
            "/servo_node/switch_command_type",
        ),
        "servo.switch_command_type_service",
    )
    pause_service = _string(
        servo.get("pause_service", "/servo_node/pause_servo"),
        "servo.pause_service",
    )
    blockers: list[str] = []
    try:
        nodes = subprocess.run(
            ["ros2", "node", "list"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if nodes.returncode != 0 or node_name not in nodes.stdout.splitlines():
            blockers.append("moveit_servo_node_not_active")
        info = subprocess.run(
            ["ros2", "topic", "info", topic],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if info.returncode != 0 or "Subscription count: 0" in info.stdout:
            blockers.append("moveit_servo_twist_subscription_missing")
        status_info = subprocess.run(
            ["ros2", "topic", "info", status_topic],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if status_info.returncode != 0 or "Publisher count: 0" in status_info.stdout:
            blockers.append("moveit_servo_status_publisher_missing")
        services = subprocess.run(
            ["ros2", "service", "list"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if services.returncode != 0 or switch_service not in services.stdout.splitlines():
            blockers.append("moveit_servo_switch_command_type_service_missing")
        if services.returncode != 0 or pause_service not in services.stdout.splitlines():
            blockers.append("moveit_servo_pause_service_missing")
    except (OSError, subprocess.TimeoutExpired):
        blockers.append("ros_graph_query_failed")
    return tuple(blockers)


def _load_factory(spec: object) -> Callable[..., Any]:
    value = _string(spec, "factory")
    if ":" not in value:
        raise ClosedLoopContractError("factory must use module:function syntax")
    module_name, function_name = value.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(factory):
        raise ClosedLoopContractError(f"factory is not callable: {value}")
    return factory


def _load_artifact_transform(config: Mapping[str, Any], prefix: str) -> np.ndarray:
    path = _path(config[prefix], prefix)
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        payload = _read_json(path)
        value = payload.get("matrix", payload.get("T", payload.get("transform")))
        if isinstance(value, str):
            return _load_transform_path((path.parent / value).resolve())
    return validate_rigid_transform(value, name=prefix)


def _load_transform_path(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return validate_rigid_transform(np.load(path, allow_pickle=False), name=str(path))
    payload = _read_json(path)
    return validate_rigid_transform(payload.get("matrix", payload.get("T")), name=str(path))


def _transform_parameter(action: ClosedLoopAction, name: str) -> np.ndarray:
    if name not in action.parameters:
        raise LiveIOError(f"{action.kind.value} action lacks {name}")
    return validate_rigid_transform(action.parameters[name], name=name)


def _pose_error(current: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(target[:3, 3] - current[:3, 3]))
    rotation = float(Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).magnitude())
    return translation, rotation


def _tcp_servo_twist_at_link8(
    *,
    current: np.ndarray,
    target: np.ndarray,
    T_link8_tcp: np.ndarray,
    horizon_s: float,
    max_linear_m_s: float,
    max_angular_rad_s: float,
    previous_twist: np.ndarray,
    dt_s: float,
    max_linear_acceleration_m_s2: float,
    max_angular_acceleration_rad_s2: float,
) -> np.ndarray:
    """Return a base-frame twist at Servo's link8 origin for a TCP target.

    Servo's IK tip is link8, while the policy observes the installed TCP.
    Convert the velocity origin before publishing; a pure TCP rotation must
    translate link8 by ``-omega x r``. Common scaling keeps that relationship
    while bounding speeds at both points and ramping nonzero commands.
    """

    linear_tcp = (target[:3, 3] - current[:3, 3]) / horizon_s
    angular = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
    angular /= horizon_s
    R_base_link8 = current[:3, :3] @ T_link8_tcp[:3, :3].T
    offset_base = R_base_link8 @ T_link8_tcp[:3, 3]
    linear_link8 = linear_tcp - np.cross(angular, offset_base)
    speed_ratio = max(
        1.0,
        float(np.linalg.norm(linear_tcp)) / max_linear_m_s,
        float(np.linalg.norm(linear_link8)) / max_linear_m_s,
        float(np.linalg.norm(angular)) / max_angular_rad_s,
    )
    desired = np.concatenate((linear_link8, angular)) / speed_ratio
    previous = np.asarray(previous_twist, dtype=float)
    delta = desired - previous
    delta_tcp = delta[:3] + np.cross(delta[3:], offset_base)
    fraction = 1.0
    for change, allowance in (
        (delta[:3], max_linear_acceleration_m_s2 * dt_s),
        (delta_tcp, max_linear_acceleration_m_s2 * dt_s),
        (delta[3:], max_angular_acceleration_rad_s2 * dt_s),
    ):
        norm = float(np.linalg.norm(change))
        if norm > 0.0:
            fraction = min(fraction, allowance / norm)
    twist = previous + fraction * delta
    # The lever changes slightly as the arm rotates. Re-apply a common speed
    # scale so both origins remain bounded even for a previous-frame command.
    tcp = twist[:3] + np.cross(twist[3:], offset_base)
    speed_ratio = max(
        1.0,
        float(np.linalg.norm(tcp)) / max_linear_m_s,
        float(np.linalg.norm(twist[:3])) / max_linear_m_s,
        float(np.linalg.norm(twist[3:])) / max_angular_rad_s,
    )
    return twist / speed_ratio


def _clip_norm(value: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value if norm <= maximum or norm == 0.0 else value * (maximum / norm)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ClosedLoopContractError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClosedLoopContractError(f"{name} must be a non-empty string")
    return value


def _positive(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ClosedLoopContractError(f"{name} must be positive and finite")
    return result


def _nonnegative(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ClosedLoopContractError(f"{name} must be non-negative and finite")
    return result


def _is_nonnegative_number(value: object) -> bool:
    try:
        _nonnegative(value, "qualification uncertainty")
    except (TypeError, ValueError):
        return False
    return True


def _is_positive_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        _positive(value, "qualification uncertainty bound")
    except (TypeError, ValueError):
        return False
    return True


def _path_or_none(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    result = Path(value).expanduser()
    return result.resolve() if result.is_absolute() else (REPOSITORY_ROOT / result).resolve()


def _path(value: object, name: str) -> Path:
    result = _path_or_none(value)
    if result is None:
        raise ClosedLoopContractError(f"{name} must be a non-empty path")
    return result


def _path_list(value: object, name: str) -> tuple[Path, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ClosedLoopContractError(f"{name} must be an array of paths")
    return tuple(_path(item, f"{name}[]") for item in value)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ClosedLoopContractError(f"JSON root must be an object: {path}")
    return payload


__all__ = [
    "ArmRuntime",
    "ArmState",
    "GripperRuntime",
    "HandEModbusRuntime",
    "LIVE_IO_SCHEMA",
    "LIVE_QUALIFICATION_SCHEMA",
    "LiveIOError",
    "LiveTestTubeIO",
    "QualifiedBasePose",
    "QualifiedCameraPose",
    "RosMoveItServoRuntime",
    "SceneEvidence",
    "SceneInterpreter",
    "SceneObjectMask",
    "TactileInterpreter",
    "create_live_io",
    "preflight_live_config",
    "provisional_camera_extrinsic_acknowledgement",
]
