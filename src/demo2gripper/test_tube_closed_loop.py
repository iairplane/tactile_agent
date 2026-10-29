"""Pure closed-loop policy for the demonstrated test-tube transfer task.

The policy keeps inference, geometry, and transition predicates in one place,
while hardware access remains behind a small adapter protocol.  It reuses the
reviewed :mod:`demo2gripper.test_tube_preflight` state machine; this module only
grounds its semantic actions into bounded TCP/gripper commands.

Transforms follow ``T_parent_child``.  The demonstration may be represented
either by a fully reconstructed ``T_tube_tcp`` or by the observable normalized
relation: signed axial fraction, top-down tool axis, and a jaw axis
perpendicular to the live tube.  The normalized form is preferable when the
human recording did not contain a qualified metric tube pose.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy.spatial.transform import Rotation

from .franka import validate_rigid_transform
from .geometry import invert_transforms
from .test_tube_preflight import (
    TransitionDecision,
    TubeState,
    decide_test_tube_transition,
)

CLOSED_LOOP_CONFIG_SCHEMA = "demo2gripper.test_tube_closed_loop_config.v1"
CLOSED_LOOP_REPLAY_SCHEMA = "demo2gripper.test_tube_closed_loop_replay.v1"
CLOSED_LOOP_REPORT_SCHEMA = "demo2gripper.test_tube_closed_loop_report.v1"


class ClosedLoopContractError(ValueError):
    """A configuration, observation, or adapter violated the task contract."""


class PoseCapability(StrEnum):
    """Capabilities asserted by a task-qualified pose backend receipt."""

    METRIC_TRANSLATION = "metric_translation"
    FULL_SE3 = "full_se3"
    CONTINUOUS_TRACKING = "continuous_tracking"
    SOURCE_TUBE = "source_tube"
    IN_HAND_TUBE = "in_hand_tube"
    SOCKET_SE3 = "socket_se3"
    IMAGE_AXIS = "image_axis"
    TUBE_ENDS_DISAMBIGUATED = "tube_ends_disambiguated"


SOURCE_POSE_CAPABILITIES = frozenset(
    {
        PoseCapability.METRIC_TRANSLATION,
        PoseCapability.FULL_SE3,
        PoseCapability.CONTINUOUS_TRACKING,
        PoseCapability.SOURCE_TUBE,
        PoseCapability.TUBE_ENDS_DISAMBIGUATED,
    }
)
HELD_POSE_CAPABILITIES = frozenset(
    {
        PoseCapability.METRIC_TRANSLATION,
        PoseCapability.FULL_SE3,
        PoseCapability.CONTINUOUS_TRACKING,
        PoseCapability.IN_HAND_TUBE,
        PoseCapability.TUBE_ENDS_DISAMBIGUATED,
    }
)
SOCKET_POSE_CAPABILITIES = frozenset(
    {
        PoseCapability.METRIC_TRANSLATION,
        PoseCapability.FULL_SE3,
        PoseCapability.CONTINUOUS_TRACKING,
        PoseCapability.SOCKET_SE3,
    }
)


class ActionKind(StrEnum):
    """Atomic commands understood by a live or replay adapter."""

    MOVE_TCP = "move_tcp"
    SERVO_TCP = "servo_tcp"
    START_CONTINUOUS_CLOSE = "start_continuous_close"
    STOP_GRIPPER = "stop_gripper"
    OPEN_AND_RECENTER = "open_and_recenter"
    OPEN_GRIPPER_STEP = "open_gripper_step"
    RECLAMP = "reclamp"
    INSERT_STEP = "insert_step"
    RETRACT_INSERTION = "retract_insertion"
    OPEN_RELEASE = "open_release"
    PAUSE_ARM = "pause_arm"
    COLLECT_EVIDENCE = "collect_evidence"
    MONITOR = "monitor"
    STOP_AND_HOLD = "stop_and_hold"
    NONE = "none"


@dataclass(frozen=True)
class PoseBackendDescriptor:
    """Auditable availability and qualification boundary for live pose inference."""

    name: str
    available: bool
    runtime_qualified: bool
    qualification_id: str | None
    capabilities: frozenset[PoseCapability]
    detail: str = ""

    def __post_init__(self) -> None:
        _identifier(self.name, name="pose backend name")
        if not isinstance(self.available, bool) or not isinstance(
            self.runtime_qualified, bool
        ):
            raise TypeError("backend availability and qualification must be boolean")
        if self.runtime_qualified and (not self.available or not self.qualification_id):
            raise ValueError(
                "a runtime-qualified backend must be available and carry a qualification id"
            )
        if self.qualification_id is not None:
            _identifier(self.qualification_id, name="pose backend qualification_id")
        object.__setattr__(
            self,
            "capabilities",
            frozenset(PoseCapability(value) for value in self.capabilities),
        )

    def blockers(self, required: frozenset[PoseCapability]) -> tuple[str, ...]:
        blockers: list[str] = []
        if not self.available:
            blockers.append(f"pose_backend_{self.name}_unavailable")
        if not self.runtime_qualified:
            blockers.append(f"pose_backend_{self.name}_not_runtime_qualified")
        missing = sorted(value.value for value in required - self.capabilities)
        blockers.extend(f"pose_backend_missing_{value}" for value in missing)
        return tuple(blockers)


@dataclass(frozen=True)
class ObjectPoseEstimate:
    """One live 6D estimate with uncertainty and backend provenance."""

    object_id: str
    T_base_object: np.ndarray
    observed_at_s: float
    position_std_m: float
    orientation_std_deg: float
    backend: PoseBackendDescriptor
    tracking_id: str
    inference_latency_s: float
    mask_id: str
    mask_iou: float
    valid_depth_count: int
    foreground_depth_support_count: int
    foreground_depth_support_fraction: float
    background_depth_fraction: float
    masked_depth_cluster_qualified: bool
    requires_rgbd_quality_gates: bool = True
    axis_sign_source: str = "not_applicable"
    cylindrical_roll_canonicalized: bool = False
    occluded_by_gripper: bool = False
    size_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        _identifier(self.object_id, name="object_id")
        _identifier(self.tracking_id, name="tracking_id")
        transform = validate_rigid_transform(
            self.T_base_object, name=f"T_base_{self.object_id}"
        ).copy()
        transform.setflags(write=False)
        object.__setattr__(self, "T_base_object", transform)
        _finite(self.observed_at_s, name="observed_at_s")
        _nonnegative(self.position_std_m, name="position_std_m")
        _nonnegative(self.orientation_std_deg, name="orientation_std_deg")
        _nonnegative(self.inference_latency_s, name="inference_latency_s")
        _identifier(self.mask_id, name="mask_id")
        if not 0.0 <= self.mask_iou <= 1.0 or not math.isfinite(self.mask_iou):
            raise ValueError("mask_iou must lie in [0,1]")
        _nonnegative_integer(self.valid_depth_count, name="valid_depth_count")
        _nonnegative_integer(
            self.foreground_depth_support_count,
            name="foreground_depth_support_count",
        )
        for name in (
            "foreground_depth_support_fraction",
            "background_depth_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if not isinstance(self.masked_depth_cluster_qualified, bool):
            raise TypeError("masked_depth_cluster_qualified must be boolean")
        if not isinstance(self.requires_rgbd_quality_gates, bool):
            raise TypeError("requires_rgbd_quality_gates must be boolean")
        if self.foreground_depth_support_count > self.valid_depth_count:
            raise ValueError("foreground depth support cannot exceed valid depth count")
        if self.axis_sign_source not in {
            "rim_detector",
            "open_end_detector",
            "source_support_geometry",
            "source_support_anchor",
            "qualified_base_z",
            "temporal_continuity",
            "not_applicable",
        }:
            raise ValueError("axis_sign_source is not a supported sign disambiguator")
        for name in ("cylindrical_roll_canonicalized", "occluded_by_gripper"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if not isinstance(self.backend, PoseBackendDescriptor):
            raise TypeError("backend must be PoseBackendDescriptor")
        if self.size_m is not None:
            size = tuple(float(value) for value in self.size_m)
            if len(size) != 3 or not np.isfinite(size).all() or min(size) <= 0.0:
                raise ValueError("size_m must contain three positive finite lengths")
            object.__setattr__(self, "size_m", size)

    @property
    def long_axis_length_m(self) -> float | None:
        return None if self.size_m is None else max(self.size_m)

    def blockers(
        self,
        *,
        now_s: float,
        required: frozenset[PoseCapability],
        max_age_s: float,
        max_position_std_m: float,
        max_orientation_std_deg: float,
        max_inference_latency_s: float,
        min_mask_iou: float,
        min_valid_depth_count: int,
        min_foreground_depth_support_count: int,
        min_foreground_depth_support_fraction: float,
        max_background_depth_fraction: float,
    ) -> tuple[str, ...]:
        blockers = list(self.backend.blockers(required))
        age_s = float(now_s) - self.observed_at_s
        if age_s < -1e-3:
            blockers.append(f"{self.object_id}_pose_timestamp_is_in_the_future")
        elif age_s > max_age_s:
            blockers.append(f"{self.object_id}_pose_stale")
        if self.position_std_m > max_position_std_m:
            blockers.append(f"{self.object_id}_position_uncertainty_too_large")
        if self.orientation_std_deg > max_orientation_std_deg:
            blockers.append(f"{self.object_id}_orientation_uncertainty_too_large")
        if self.inference_latency_s > max_inference_latency_s:
            blockers.append(f"{self.object_id}_inference_latency_too_large")
        if self.requires_rgbd_quality_gates:
            if self.mask_iou < min_mask_iou:
                blockers.append(f"{self.object_id}_mask_iou_too_low")
            if self.valid_depth_count < min_valid_depth_count:
                blockers.append(f"{self.object_id}_valid_depth_count_too_low")
            if self.foreground_depth_support_count < min_foreground_depth_support_count:
                blockers.append(f"{self.object_id}_foreground_depth_support_too_low")
            if self.foreground_depth_support_fraction < min_foreground_depth_support_fraction:
                blockers.append(f"{self.object_id}_foreground_depth_fraction_too_low")
            if self.background_depth_fraction > max_background_depth_fraction:
                blockers.append(f"{self.object_id}_background_depth_dominates_mask")
            if not self.masked_depth_cluster_qualified:
                blockers.append(f"{self.object_id}_masked_depth_cluster_not_qualified")
        if self.object_id == "tube":
            if self.axis_sign_source == "not_applicable":
                blockers.append("tube_open_closed_end_sign_not_disambiguated")
            if not self.cylindrical_roll_canonicalized:
                blockers.append("tube_cylindrical_roll_not_canonicalized")
        if self.occluded_by_gripper:
            blockers.append(f"{self.object_id}_occluded_by_gripper")
        return tuple(blockers)


@dataclass(frozen=True)
class RobotFeedback:
    T_base_tcp: np.ndarray
    motion_complete: bool = False
    bounded_tcp_servo_supported: bool = False
    fault: str | None = None

    def __post_init__(self) -> None:
        transform = validate_rigid_transform(self.T_base_tcp, name="T_base_tcp").copy()
        transform.setflags(write=False)
        object.__setattr__(self, "T_base_tcp", transform)
        if not isinstance(self.motion_complete, bool):
            raise TypeError("motion_complete must be boolean")
        if not isinstance(self.bounded_tcp_servo_supported, bool):
            raise TypeError("bounded_tcp_servo_supported must be boolean")
        if self.fault is not None:
            _identifier(self.fault, name="robot fault")


@dataclass(frozen=True)
class GripperFeedback:
    width_m: float
    moving: bool
    fault_free: bool
    object_detected: bool = False
    open_confirmed: bool = False

    def __post_init__(self) -> None:
        width = _nonnegative(self.width_m, name="gripper width_m")
        if width > 0.055:
            raise ValueError("gripper width_m exceeds the supported Hand-E range")
        for name in ("moving", "fault_free", "object_detected", "open_confirmed"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")


@dataclass(frozen=True)
class TactileFeedback:
    valid: bool
    observed_at_s: float
    contact_present: bool = False
    bilateral_contact: bool = False
    # Both optical contact patches are retained away from the pad image
    # boundary for the configured consecutive-window debounce.  FORCE6D is
    # intentionally not part of this geometric grasp-depth predicate: force
    # remains corroborating evidence for load, slip, pivot and insertion.
    bilateral_contact_interior: bool = False
    single_contact_side: str | None = None
    slip_detected: bool = False
    relative_angle_target_reached: bool = False
    rotation_stopped: bool = False
    insertion_jam: bool = False
    insertion_resistance_high: bool = False
    insertion_seated: bool = False
    insertion_reference_ready: bool = True

    def __post_init__(self) -> None:
        _finite(self.observed_at_s, name="tactile observed_at_s")
        for name in (
            "valid",
            "contact_present",
            "bilateral_contact",
            "bilateral_contact_interior",
            "slip_detected",
            "relative_angle_target_reached",
            "rotation_stopped",
            "insertion_jam",
            "insertion_resistance_high",
            "insertion_seated",
            "insertion_reference_ready",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.single_contact_side not in {None, "left", "right"}:
            raise ValueError("single_contact_side must be left, right, or null")
        if self.bilateral_contact and self.single_contact_side is not None:
            raise ValueError("bilateral contact cannot also name one contact side")
        if self.bilateral_contact and not self.contact_present:
            raise ValueError("bilateral contact requires contact_present")
        if self.bilateral_contact_interior and not self.bilateral_contact:
            raise ValueError("interior bilateral contact requires bilateral_contact")
        if self.insertion_jam and self.insertion_seated:
            raise ValueError("insertion cannot be jammed and seated simultaneously")


@dataclass(frozen=True)
class ClosedLoopObservation:
    """Fresh fused observation consumed by exactly one controller step."""

    observation_id: str
    now_s: float
    robot: RobotFeedback
    gripper: GripperFeedback
    tactile: TactileFeedback
    tube_pose: ObjectPoseEstimate | None = None
    socket_pose: ObjectPoseEstimate | None = None
    source_depth_only_loss: bool = False
    source_rgb_identity_retained: bool = False
    front_rgb_angle_target_reached: bool = False
    tube_clear_of_source: bool = False
    tube_tip_at_socket_entry: bool = False
    tube_socket_axes_aligned: bool = False
    insertion_progress_m: float | None = None
    insertion_depth_m: float | None = None
    tube_stays_supported: bool = False

    def __post_init__(self) -> None:
        _identifier(self.observation_id, name="observation_id")
        _finite(self.now_s, name="now_s")
        for name in (
            "source_depth_only_loss",
            "source_rgb_identity_retained",
            "front_rgb_angle_target_reached",
            "tube_clear_of_source",
            "tube_tip_at_socket_entry",
            "tube_socket_axes_aligned",
            "tube_stays_supported",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        for name in ("insertion_progress_m", "insertion_depth_m"):
            value = getattr(self, name)
            if value is not None:
                _nonnegative(value, name=name)


@dataclass(frozen=True)
class NormalizedDemoGraspRelation:
    """Runtime-grounded part of the demonstrated human/tube relation.

    A circular tube has no observable roll. A fixed ``T_tube_tcp`` also cannot
    keep TCP ``+Z`` equal to base ``-Z`` when the live tube orientation changes.
    This representation retains exactly what the reviewed demonstration makes
    observable and instantiates the full target from each signed live tube pose.
    """

    axial_fraction_closed_to_open: float
    tube_length_m: float | None
    tube_length_source: str
    preferred_jaw_axis_base: np.ndarray
    top_approach_direction_base: np.ndarray = field(
        default_factory=lambda: np.asarray([0.0, 0.0, -1.0], dtype=float)
    )
    tube_pose_origin: str = "geometric_centre"
    tube_axis_contract: str = "closed_to_open"

    def __post_init__(self) -> None:
        fraction = _finite(
            self.axial_fraction_closed_to_open,
            name="axial_fraction_closed_to_open",
        )
        if not 0.0 < fraction < 1.0:
            raise ClosedLoopContractError(
                "axial_fraction_closed_to_open must lie in (0,1)"
            )
        _identifier(self.tube_length_source, name="tube_length_source")
        if self.tube_length_m is None:
            if self.tube_length_source != "runtime_f2g_size":
                raise ClosedLoopContractError(
                    "null tube_length_m requires tube_length_source=runtime_f2g_size"
                )
        else:
            object.__setattr__(
                self,
                "tube_length_m",
                _positive(self.tube_length_m, name="tube_length_m"),
            )
        if self.tube_pose_origin != "geometric_centre":
            raise ClosedLoopContractError(
                "normalized grasp requires a geometric-centre tube pose"
            )
        if self.tube_axis_contract != "closed_to_open":
            raise ClosedLoopContractError(
                "normalized grasp requires a signed closed_to_open tube axis"
            )
        for name in ("preferred_jaw_axis_base", "top_approach_direction_base"):
            value = _unit(getattr(self, name), name=name)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.allclose(
            self.top_approach_direction_base,
            [0.0, 0.0, -1.0],
            atol=1e-9,
        ):
            raise ClosedLoopContractError(
                "normalized top approach must use exact base -Z"
            )


@dataclass(frozen=True)
class TestTubeClosedLoopGeometry:
    """Object-relative geometry compiled from the human demonstration."""

    relation_id: str
    T_tube_tcp_demo_grasp: np.ndarray | None
    T_socket_tube_entry: np.ndarray
    normalized_demo_grasp: NormalizedDemoGraspRelation | None = None
    relation_source: str = "unspecified"
    relation_runtime_qualified: bool = False
    socket_entry_source: str = "unspecified"
    socket_entry_runtime_qualified: bool = False
    tube_axis_tube: np.ndarray = field(
        default_factory=lambda: np.asarray([0.0, 0.0, 1.0], dtype=float)
    )
    socket_outward_axis_socket: np.ndarray = field(
        default_factory=lambda: np.asarray([0.0, 0.0, 1.0], dtype=float)
    )

    def __post_init__(self) -> None:
        _identifier(self.relation_id, name="relation_id")
        _identifier(self.relation_source, name="relation_source")
        _identifier(self.socket_entry_source, name="socket_entry_source")
        for name in (
            "relation_runtime_qualified",
            "socket_entry_runtime_qualified",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if (self.T_tube_tcp_demo_grasp is None) == (self.normalized_demo_grasp is None):
            raise ClosedLoopContractError(
                "geometry must select exactly one of T_tube_tcp_demo_grasp or "
                "normalized_demo_grasp"
            )
        if self.T_tube_tcp_demo_grasp is not None:
            value = validate_rigid_transform(
                self.T_tube_tcp_demo_grasp, name="T_tube_tcp_demo_grasp"
            ).copy()
            value.setflags(write=False)
            object.__setattr__(self, "T_tube_tcp_demo_grasp", value)
        if self.normalized_demo_grasp is not None and not isinstance(
            self.normalized_demo_grasp, NormalizedDemoGraspRelation
        ):
            raise TypeError("normalized_demo_grasp has the wrong type")
        socket = validate_rigid_transform(
            self.T_socket_tube_entry, name="T_socket_tube_entry"
        ).copy()
        socket.setflags(write=False)
        object.__setattr__(self, "T_socket_tube_entry", socket)
        for name in ("tube_axis_tube", "socket_outward_axis_socket"):
            value = _unit(getattr(self, name), name=name)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ClosedLoopPolicy:
    """Bounds for vision, motion, gripper, pivot, and insertion decisions."""

    top_standoff_m: float = 0.060
    extract_clearance_m: float = 0.040
    rack_standoff_m: float = 0.050
    retreat_m: float = 0.050
    servo_translation_gain: float = 0.7
    servo_rotation_gain: float = 0.7
    servo_max_translation_step_m: float = 0.008
    servo_max_rotation_step_deg: float = 4.0
    servo_translation_deadband_m: float = 0.0025
    servo_rotation_deadband_deg: float = 2.0
    servo_max_total_target_drift_m: float = 0.035
    servo_max_total_target_drift_deg: float = 20.0
    servo_max_target_jump_m: float = 0.015
    servo_max_target_jump_deg: float = 8.0
    servo_settle_windows: int = 2
    pose_max_age_s: float = 0.20
    # OBSERVE is the only state allowed to wait on the slower semantic
    # bootstrap (two SAM prompts plus F2G). No motion has been dispatched yet,
    # so its capture may be older than the continuous-servo pose budget.
    observe_pose_max_age_s: float = 5.0
    pose_max_position_std_m: float = 0.008
    pose_max_orientation_std_deg: float = 6.0
    pose_max_inference_latency_s: float = 0.15
    pose_min_mask_iou: float = 0.55
    pose_min_valid_depth_count: int = 20
    pose_min_foreground_depth_support_count: int = 20
    pose_min_foreground_depth_support_fraction: float = 0.05
    pose_max_background_depth_fraction: float = 0.75
    occlusion_switch_translation_m: float = 0.004
    occlusion_switch_rotation_deg: float = 3.0
    # An explicit task-level contact window may close in place when the
    # gripper hides the last few millimetres of a freshly observed grasp.
    # None retains the standard final visual-servo deadband.
    occluded_grasp_close_distance_m: float | None = None
    final_approach_commit_distance_m: float | None = None
    final_approach_timeout_s: float = 8.0
    tactile_max_age_s: float = 0.20
    continuous_close_min_width_m: float = 0.005
    continuous_close_speed: int = 80
    continuous_close_force: int = 15
    retained_width_min_m: float = 0.005
    retained_width_max_m: float = 0.035
    shallow_retry_step_m: float = 0.005
    max_shallow_contact_retries: int = 2
    shallow_retry_total_travel_budget_m: float = 0.010
    shallow_retry_open_width_m: float = 0.0495
    shallow_contact_confirmation_windows: int = 2
    single_contact_confirmation_windows: int = 2
    max_centering_recoveries: int = 2
    recenter_open_delta_m: float = 0.003
    recenter_probe_step_m: float = 0.001
    recenter_reverse_step_m: float = 0.002
    recenter_total_travel_budget_m: float = 0.004
    recenter_width_improvement_m: float = 0.0005
    level_axis_tolerance_deg: float = 4.0
    level_min_horizontal_projection: float = 0.10
    level_settle_windows: int = 3
    # A fixed-TCP Cartesian leveling arc can consume the arm's redundant
    # posture and stop making progress near a joint bound or singularity even
    # though an equivalent horizontal pose remains reachable.  Detect that
    # measured plateau and make one collision-planned base-Z lift while
    # retaining the same final orientation and closed grasp.
    level_stall_confirmation_windows: int = 12
    level_stall_min_progress_deg: float = 0.5
    level_escape_lift_m: float = 0.050
    # A commanded rigid wrist rotation can move an optical patch while the
    # tube is still securely held.  Once motion stops, require this many
    # consecutive slip windows before promoting slip to terminal HOLD.
    level_stationary_slip_confirmation_windows: int = 2
    pivot_open_step_m: float = 0.00025
    pivot_max_steps: int = 8
    reclamp_step_m: float = 0.0005
    insertion_step_m: float = 0.0015
    # A measured target may be retained for legacy/static fixtures.  Live
    # insertion may leave it unset and finish from a debounced tactile seat
    # event.  ``insertion_max_travel_m`` is always mandatory and bounds both.
    insertion_target_depth_m: float | None = 0.020
    insertion_max_travel_m: float = 0.030
    insertion_progress_ratio_min: float = 0.25
    jam_retract_m: float = 0.003
    max_jam_recoveries: int = 2
    default_stage_timeout_s: float = 12.0
    approach_timeout_s: float = 12.0
    max_approach_timeout_recoveries: int = 0
    observe_timeout_s: float = 30.0
    close_timeout_s: float = 6.0
    insertion_timeout_s: float = 20.0

    def __post_init__(self) -> None:
        positive = (
            "top_standoff_m",
            "extract_clearance_m",
            "rack_standoff_m",
            "retreat_m",
            "servo_translation_gain",
            "servo_rotation_gain",
            "servo_max_translation_step_m",
            "servo_max_rotation_step_deg",
            "servo_translation_deadband_m",
            "servo_rotation_deadband_deg",
            "servo_max_total_target_drift_m",
            "servo_max_total_target_drift_deg",
            "servo_max_target_jump_m",
            "servo_max_target_jump_deg",
            "pose_max_age_s",
            "observe_pose_max_age_s",
            "pose_max_position_std_m",
            "pose_max_orientation_std_deg",
            "pose_max_inference_latency_s",
            "pose_min_mask_iou",
            "pose_min_foreground_depth_support_fraction",
            "pose_max_background_depth_fraction",
            "occlusion_switch_translation_m",
            "occlusion_switch_rotation_deg",
            "tactile_max_age_s",
            "continuous_close_speed",
            "continuous_close_force",
            "retained_width_max_m",
            "shallow_retry_step_m",
            "shallow_retry_total_travel_budget_m",
            "shallow_retry_open_width_m",
            "recenter_open_delta_m",
            "recenter_probe_step_m",
            "recenter_reverse_step_m",
            "recenter_total_travel_budget_m",
            "recenter_width_improvement_m",
            "level_axis_tolerance_deg",
            "level_min_horizontal_projection",
            "level_stall_min_progress_deg",
            "level_escape_lift_m",
            "pivot_open_step_m",
            "reclamp_step_m",
            "insertion_step_m",
            "insertion_max_travel_m",
            "insertion_progress_ratio_min",
            "jam_retract_m",
            "default_stage_timeout_s",
            "approach_timeout_s",
            "observe_timeout_s",
            "close_timeout_s",
            "final_approach_timeout_s",
            "insertion_timeout_s",
        )
        for name in positive:
            _positive(getattr(self, name), name=name)
        if self.occluded_grasp_close_distance_m is not None:
            _positive(
                self.occluded_grasp_close_distance_m,
                name="occluded_grasp_close_distance_m",
            )
        if self.final_approach_commit_distance_m is not None:
            _positive(
                self.final_approach_commit_distance_m, name="final_approach_commit_distance_m"
            )
        if self.insertion_target_depth_m is not None:
            _positive(self.insertion_target_depth_m, name="insertion_target_depth_m")
            if self.insertion_target_depth_m > self.insertion_max_travel_m:
                raise ValueError(
                    "insertion_target_depth_m cannot exceed insertion_max_travel_m"
                )
        if self.insertion_step_m > self.insertion_max_travel_m:
            raise ValueError("insertion_step_m cannot exceed insertion_max_travel_m")
        _nonnegative(self.continuous_close_min_width_m, name="continuous_close_min_width_m")
        _nonnegative(self.retained_width_min_m, name="retained_width_min_m")
        if not (
            self.continuous_close_min_width_m
            <= self.retained_width_min_m
            < self.retained_width_max_m
            <= 0.05
        ):
            raise ValueError("retained Hand-E width bounds are inconsistent")
        if not 0.0 < self.insertion_progress_ratio_min < 1.0:
            raise ValueError("insertion_progress_ratio_min must lie in (0,1)")
        if not 0.0 <= self.pose_min_mask_iou <= 1.0:
            raise ValueError("pose_min_mask_iou must lie in [0,1]")
        for name in (
            "pose_min_foreground_depth_support_fraction",
            "pose_max_background_depth_fraction",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        for name in (
            "servo_settle_windows",
            "shallow_contact_confirmation_windows",
            "single_contact_confirmation_windows",
            "level_settle_windows",
            "level_stationary_slip_confirmation_windows",
            "level_stall_confirmation_windows",
            "pivot_max_steps",
        ):
            _positive_integer(getattr(self, name), name=name)
        for name in (
            "max_centering_recoveries",
            "max_shallow_contact_retries",
            "max_jam_recoveries",
            "max_approach_timeout_recoveries",
        ):
            _nonnegative_integer(getattr(self, name), name=name)
        _nonnegative_integer(self.pose_min_valid_depth_count, name="pose_min_valid_depth_count")
        _nonnegative_integer(
            self.pose_min_foreground_depth_support_count,
            name="pose_min_foreground_depth_support_count",
        )
        if self.level_axis_tolerance_deg >= 90.0:
            raise ValueError("level_axis_tolerance_deg must be smaller than 90 degrees")
        if self.level_min_horizontal_projection >= 1.0:
            raise ValueError("level_min_horizontal_projection must be smaller than 1")
        if self.shallow_retry_open_width_m > 0.05:
            raise ValueError("shallow_retry_open_width_m cannot exceed 0.05 m")
        if self.shallow_retry_open_width_m < 0.048:
            raise ValueError("shallow_retry_open_width_m must reopen to at least 0.048 m")
        if (
            self.shallow_retry_step_m * self.max_shallow_contact_retries
            > self.shallow_retry_total_travel_budget_m + 1e-12
        ):
            raise ValueError("shallow-contact retry count exceeds its travel budget")

    def timeout_for(self, state: TubeState) -> float:
        if state is TubeState.OBSERVE:
            return self.observe_timeout_s
        if state is TubeState.APPROACH_TUBE:
            return self.approach_timeout_s
        if state is TubeState.SIDE_GRASP:
            return self.close_timeout_s
        if state in {
            TubeState.INSERT_MICROSTEP,
            TubeState.INSERT_EVALUATE,
            TubeState.JAM_RECOVERY_RETRACT,
        }:
            return self.insertion_timeout_s
        return self.default_stage_timeout_s


@dataclass(frozen=True)
class ServoResult:
    command_target: np.ndarray
    translation_error_m: float
    rotation_error_deg: float
    within_deadband: bool
    blockers: tuple[str, ...]


def bounded_visual_servo(
    *,
    T_base_tcp_measured: np.ndarray,
    T_base_tcp_target: np.ndarray,
    T_base_tcp_initial_target: np.ndarray,
    previous_target: np.ndarray | None,
    policy: ClosedLoopPolicy,
) -> ServoResult:
    """Return one bounded pose correction or explicit drift blockers."""

    measured = validate_rigid_transform(T_base_tcp_measured, name="measured TCP")
    target = validate_rigid_transform(T_base_tcp_target, name="target TCP")
    initial = validate_rigid_transform(T_base_tcp_initial_target, name="initial target TCP")
    previous = (
        None
        if previous_target is None
        else validate_rigid_transform(previous_target, name="previous target TCP")
    )
    drift_m, drift_deg = pose_error(initial, target)
    blockers: list[str] = []
    if drift_m > policy.servo_max_total_target_drift_m:
        blockers.append("live_target_translation_drift_exceeded")
    if drift_deg > policy.servo_max_total_target_drift_deg:
        blockers.append("live_target_rotation_drift_exceeded")
    if previous is not None:
        jump_m, jump_deg = pose_error(previous, target)
        if jump_m > policy.servo_max_target_jump_m:
            blockers.append("live_target_translation_jump_exceeded")
        if jump_deg > policy.servo_max_target_jump_deg:
            blockers.append("live_target_rotation_jump_exceeded")

    translation_delta = target[:3, 3] - measured[:3, 3]
    translation_error = float(np.linalg.norm(translation_delta))
    relative_rotation = target[:3, :3] @ measured[:3, :3].T
    rotation_vector = Rotation.from_matrix(relative_rotation).as_rotvec()
    rotation_error_rad = float(np.linalg.norm(rotation_vector))
    rotation_error_deg = math.degrees(rotation_error_rad)
    within = bool(
        translation_error <= policy.servo_translation_deadband_m
        and rotation_error_deg <= policy.servo_rotation_deadband_deg
        and not blockers
    )

    translation_step = translation_delta * policy.servo_translation_gain
    step_norm = float(np.linalg.norm(translation_step))
    if step_norm > policy.servo_max_translation_step_m:
        translation_step *= policy.servo_max_translation_step_m / step_norm
    rotation_step = rotation_vector * policy.servo_rotation_gain
    rotation_step_norm = float(np.linalg.norm(rotation_step))
    maximum_rotation_step = math.radians(policy.servo_max_rotation_step_deg)
    if rotation_step_norm > maximum_rotation_step:
        rotation_step *= maximum_rotation_step / rotation_step_norm

    command = measured.copy()
    command[:3, 3] += translation_step
    command[:3, :3] = Rotation.from_rotvec(rotation_step).as_matrix() @ measured[:3, :3]
    validate_rigid_transform(command, name="bounded servo command")
    command.setflags(write=False)
    return ServoResult(
        command_target=command,
        translation_error_m=translation_error,
        rotation_error_deg=rotation_error_deg,
        within_deadband=within,
        blockers=tuple(blockers),
    )


def pose_error(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Return translation and geodesic rotation difference."""

    first_value = validate_rigid_transform(first, name="first pose")
    second_value = validate_rigid_transform(second, name="second pose")
    translation = float(np.linalg.norm(second_value[:3, 3] - first_value[:3, 3]))
    rotation = Rotation.from_matrix(
        second_value[:3, :3] @ first_value[:3, :3].T
    ).magnitude()
    return translation, math.degrees(float(rotation))


def tube_horizontal_error_deg(
    T_base_tube: np.ndarray,
    *,
    tube_axis_tube: Sequence[float] = (0.0, 0.0, 1.0),
) -> float:
    """Return the unsigned elevation of the signed tube axis above base XY."""

    tube = validate_rigid_transform(T_base_tube, name="T_base_tube")
    local_axis = _unit(tube_axis_tube, name="tube_axis_tube")
    axis_base = _unit(tube[:3, :3] @ local_axis, name="live tube axis base")
    return math.degrees(math.asin(float(np.clip(abs(axis_base[2]), 0.0, 1.0))))


def held_tube_level_target(
    *,
    T_base_tube: np.ndarray,
    T_base_tcp_measured: np.ndarray,
    tube_axis_tube: Sequence[float] = (0.0, 0.0, 1.0),
    min_horizontal_projection: float = 0.10,
) -> np.ndarray:
    """Rotate a rigidly held tube to base-horizontal around the measured TCP.

    The target azimuth is the signed live tube axis projected into base XY, so
    no arbitrary left/right direction is introduced.  When that projection is
    too small, the azimuth is unobservable and the function fails closed.  The
    TCP translation is retained and the same minimal base-frame rotation is
    applied to the measured TCP orientation; this models a rigid wrist rotation
    around the grasp point rather than changing the tube/TCP relation.
    """

    tube = validate_rigid_transform(T_base_tube, name="T_base_tube")
    tcp = validate_rigid_transform(T_base_tcp_measured, name="T_base_tcp_measured")
    threshold = _positive(
        min_horizontal_projection,
        name="min_horizontal_projection",
    )
    if threshold >= 1.0:
        raise ClosedLoopContractError(
            "min_horizontal_projection must be smaller than 1"
        )
    local_axis = _unit(tube_axis_tube, name="tube_axis_tube")
    axis_base = _unit(tube[:3, :3] @ local_axis, name="live tube axis base")
    horizontal = axis_base.copy()
    horizontal[2] = 0.0
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm < threshold:
        raise ClosedLoopContractError(
            "held_tube_horizontal_direction_unobservable"
        )
    horizontal /= horizontal_norm

    cross = np.cross(axis_base, horizontal)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(axis_base @ horizontal, -1.0, 1.0))
    if sine <= 1e-12:
        delta_rotation = np.eye(3)
    else:
        rotation_axis = cross / sine
        angle = math.atan2(sine, cosine)
        delta_rotation = Rotation.from_rotvec(rotation_axis * angle).as_matrix()

    target = tcp.copy()
    target[:3, :3] = delta_rotation @ tcp[:3, :3]
    return validate_rigid_transform(target, name="held-tube level T_base_tcp")


def canonicalize_cylindrical_tube_pose(
    T_base_raw: np.ndarray,
    *,
    long_axis_index: int,
    open_end_sign: int,
    transverse_reference_base: Sequence[float] = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """Remove unobservable cylinder roll while retaining a signed tube axis.

    ``open_end_sign`` must come from a rim/open-end detector or a temporally
    continuous signed track.  PCA/F2G's unsigned long axis alone is
    insufficient: swapping the closed and open ends changes grasp and
    insertion semantics.  The returned frame always has local ``+Z`` from the
    closed end toward the open end; local roll is fixed by a base-frame
    transverse reference and carries no task meaning.
    """

    raw = validate_rigid_transform(T_base_raw, name="raw cylindrical tube pose")
    if long_axis_index not in (0, 1, 2):
        raise ValueError("long_axis_index must be 0, 1, or 2")
    if open_end_sign not in (-1, 1):
        raise ValueError("open_end_sign must be -1 or +1")
    axis = raw[:3, long_axis_index] * open_end_sign
    axis /= np.linalg.norm(axis)
    reference = _unit(transverse_reference_base, name="transverse_reference_base")
    transverse = reference - float(reference @ axis) * axis
    if np.linalg.norm(transverse) <= 1e-6:
        fallback = np.asarray([0.0, 1.0, 0.0])
        transverse = fallback - float(fallback @ axis) * axis
    x_axis = transverse / np.linalg.norm(transverse)
    y_axis = np.cross(axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((x_axis, y_axis, axis))
    result[:3, 3] = raw[:3, 3]
    return validate_rigid_transform(result, name="canonical signed tube pose")


def grounded_grasp_target(
    tube_pose: ObjectPoseEstimate,
    geometry: TestTubeClosedLoopGeometry,
) -> np.ndarray:
    if geometry.normalized_demo_grasp is not None:
        return normalized_top_down_grasp_target(
            tube_pose.T_base_object,
            geometry.normalized_demo_grasp,
            runtime_tube_length_m=tube_pose.long_axis_length_m,
        )
    assert geometry.T_tube_tcp_demo_grasp is not None
    result = tube_pose.T_base_object @ geometry.T_tube_tcp_demo_grasp
    return validate_rigid_transform(result, name="grounded T_base_tcp grasp")


def normalized_top_down_grasp_target(
    T_base_tube: np.ndarray,
    relation: NormalizedDemoGraspRelation,
    *,
    runtime_tube_length_m: float | None = None,
) -> np.ndarray:
    """Instantiate the reviewed human relation from a signed live tube pose.

    The tube pose must be centred at the tube's geometric centre and local
    ``+Z`` must point from its closed end to its open end. TCP ``+Z`` stays at
    the requested base-frame top-approach direction for every live tube pose.
    The human thumb/index direction selects only the symmetric jaw-axis sign.
    """

    tube = validate_rigid_transform(T_base_tube, name="T_base_tube")
    if not isinstance(relation, NormalizedDemoGraspRelation):
        raise TypeError("relation must be NormalizedDemoGraspRelation")
    tube_axis = _unit(tube[:3, 2], name="signed live tube axis")
    z_tcp = relation.top_approach_direction_base
    preferred = relation.preferred_jaw_axis_base
    x_tcp = np.cross(z_tcp, tube_axis)
    if np.linalg.norm(x_tcp) <= 1e-6:
        x_tcp = preferred - float(preferred @ z_tcp) * z_tcp
    if np.linalg.norm(x_tcp) <= 1e-6:
        for fallback in (
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
        ):
            x_tcp = fallback - float(fallback @ z_tcp) * z_tcp
            if np.linalg.norm(x_tcp) > 1e-6:
                break
    x_tcp /= np.linalg.norm(x_tcp)
    preferred_projected = preferred - float(preferred @ z_tcp) * z_tcp
    if (
        np.linalg.norm(preferred_projected) > 1e-6
        and float(x_tcp @ preferred_projected) < 0.0
    ):
        x_tcp *= -1.0
    y_tcp = np.cross(z_tcp, x_tcp)
    y_tcp /= np.linalg.norm(y_tcp)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((x_tcp, y_tcp, z_tcp))
    length = relation.tube_length_m
    if length is None:
        if runtime_tube_length_m is None:
            raise ClosedLoopContractError(
                "runtime F2G tube length is unavailable for normalized grasp"
            )
        length = _positive(runtime_tube_length_m, name="runtime_tube_length_m")
    axial_offset_m = (
        relation.axial_fraction_closed_to_open - 0.5
    ) * length
    result[:3, 3] = tube[:3, 3] + axial_offset_m * tube_axis
    return validate_rigid_transform(result, name="normalized top-down grasp target")


def top_pregrasp_target(grasp_target: np.ndarray, standoff_m: float) -> np.ndarray:
    """Move the grounded grasp vertically upward while preserving orientation."""

    result = validate_rigid_transform(grasp_target, name="grasp target").copy()
    result[:3, 3] += [0.0, 0.0, _positive(standoff_m, name="standoff_m")]
    return result


def socket_tcp_target(
    socket_pose: ObjectPoseEstimate,
    geometry: TestTubeClosedLoopGeometry,
    T_tube_tcp_held: np.ndarray,
) -> np.ndarray:
    result = (
        socket_pose.T_base_object
        @ geometry.T_socket_tube_entry
        @ validate_rigid_transform(T_tube_tcp_held, name="held T_tube_tcp")
    )
    return validate_rigid_transform(result, name="socket entry T_base_tcp")


@dataclass(frozen=True)
class ClosedLoopAction:
    action_id: str
    kind: ActionKind
    state: TubeState
    parameters: Mapping[str, Any]
    reason: str

    def __post_init__(self) -> None:
        _identifier(self.action_id, name="action_id")
        if not isinstance(self.kind, ActionKind):
            object.__setattr__(self, "kind", ActionKind(self.kind))
        if not isinstance(self.state, TubeState):
            object.__setattr__(self, "state", TubeState(self.state))
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        _identifier(self.reason, name="action reason")

    @property
    def dispatches_hardware(self) -> bool:
        return self.kind not in {
            ActionKind.COLLECT_EVIDENCE,
            ActionKind.MONITOR,
            ActionKind.NONE,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "state": self.state.value,
            "parameters": _jsonable(self.parameters),
            "reason": self.reason,
            "dispatches_hardware": self.dispatches_hardware,
        }


@dataclass(frozen=True)
class ClosedLoopDecision:
    previous_state: TubeState
    next_state: TubeState
    action: ClosedLoopAction
    predicates: tuple[str, ...]
    semantic_transition: TransitionDecision | None
    terminal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_state": self.previous_state.value,
            "next_state": self.next_state.value,
            "action": self.action.to_dict(),
            "predicates": list(self.predicates),
            "semantic_transition": (
                None
                if self.semantic_transition is None
                else self.semantic_transition.to_dict()
            ),
            "terminal": self.terminal,
        }


@runtime_checkable
class ClosedLoopIO(Protocol):
    """Live adapter boundary; implementations own every hardware handle."""

    @property
    def pose_backend_descriptor(self) -> PoseBackendDescriptor: ...

    def observe(self, *, state: TubeState) -> ClosedLoopObservation: ...

    def dispatch(self, action: ClosedLoopAction) -> Mapping[str, Any]: ...

    def stop_and_hold(self, *, reason: str) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class TestTubeClosedLoopController:
    """One continuous, predicate-driven task policy with bounded recoveries."""

    def __init__(
        self,
        *,
        geometry: TestTubeClosedLoopGeometry,
        policy: ClosedLoopPolicy | None = None,
        startup_recovery_path: Sequence[np.ndarray] | None = None,
    ) -> None:
        self.geometry = geometry
        self.policy = policy or ClosedLoopPolicy()
        self.state = TubeState.OBSERVE
        self._state_started_s: float | None = None
        self._last_now_s: float | None = None
        self._action_sequence = 0
        self._phase = "initial"
        self._initial_target: np.ndarray | None = None
        self._previous_target: np.ndarray | None = None
        self._settled_windows = 0
        self._level_stationary_slip_windows = 0
        self._level_progress_reference_deg: float | None = None
        self._level_stall_windows = 0
        self._level_escape_attempted = False
        self._single_contact_windows = 0
        self._centering_recoveries = 0
        self._approach_timeout_recoveries = 0
        self._recenter_baseline_contact_width_m: float | None = None
        self._recenter_direction_reversed = False
        self._recenter_total_travel_m = 0.0
        self._contact_stop_width_m: float | None = None
        self._shallow_contact_unready_windows = 0
        self._shallow_contact_ready_windows = 0
        self._shallow_contact_retries = 0
        self._shallow_retry_total_travel_m = 0.0
        self._shallow_retry_origin_tcp: np.ndarray | None = None
        self._shallow_retry_target: np.ndarray | None = None
        self._approach_direction_base: np.ndarray | None = None
        self._pivot_steps = 0
        self._jam_recoveries = 0
        self._insertion_depth_commanded_m = 0.0
        self._T_tube_tcp_held: np.ndarray | None = None
        self._extract_target: np.ndarray | None = None
        self._level_target: np.ndarray | None = None
        self._retreat_target: np.ndarray | None = None
        self._last_qualified_grasp_target: np.ndarray | None = None
        self._last_qualified_grasp_observed_at_s: float | None = None
        # Sum of the bounded jaw-axis sign probes that have physically been
        # applied.  Re-grounding the demo relation after a probe must retain
        # this correction; otherwise the next visual-servo command would drive
        # straight back to the original one-pad-contact pose.
        self._grasp_recenter_local_x_m = 0.0
        self._vision_frozen_for_close = False
        self._tracking_ids: dict[str, str] = {}
        self._final_contact_target: np.ndarray | None = None
        self._final_contact_start_tcp: np.ndarray | None = None
        self._final_contact_source_observed_at_s: float | None = None
        self._final_contact_arrival_tcp: np.ndarray | None = None
        self._final_contact_depth_loss_handoff = False
        if startup_recovery_path is not None:
            if (
                isinstance(startup_recovery_path, (str, bytes))
                or not isinstance(startup_recovery_path, Sequence)
                or len(startup_recovery_path) < 2
            ):
                raise ClosedLoopContractError("startup recovery needs an expected pose and a path")
            self._startup_recovery_path = tuple(
                validate_rigid_transform(pose, name=f"startup recovery pose {index}").copy()
                for index, pose in enumerate(startup_recovery_path)
            )
        else:
            self._startup_recovery_path = ()
        self._startup_recovery_armed = False
        self._startup_recovery_complete = startup_recovery_path is None
        self._startup_recovery_index = 1

    def step(self, observation: ClosedLoopObservation) -> ClosedLoopDecision:
        if not isinstance(observation, ClosedLoopObservation):
            raise TypeError("observation must be ClosedLoopObservation")
        previous = self.state
        if self._last_now_s is not None and observation.now_s <= self._last_now_s:
            return self._hold(previous, "observation timestamps must be strictly increasing")
        self._last_now_s = observation.now_s
        if self._state_started_s is None:
            self._state_started_s = observation.now_s
        if observation.robot.fault is not None:
            return self._hold(previous, f"robot fault: {observation.robot.fault}")
        if not observation.gripper.fault_free:
            return self._hold(previous, "Hand-E feedback reports a fault")
        timeout_s = self.policy.timeout_for(self.state)
        if self.state is TubeState.OBSERVE and self._startup_recovery_armed:
            timeout_s = self.policy.approach_timeout_s
        if self.state is TubeState.SIDE_GRASP:
            if self._phase == "final_contact_approach":
                timeout_s = self.policy.final_approach_timeout_s
            elif self._phase in {
                "initial",
                "servo_grasp",
                "recenter_complete",
                "recentering",
                "shallow_retry_opening",
                "shallow_retry_approach",
            }:
                timeout_s = self.policy.approach_timeout_s
        if observation.now_s - self._state_started_s > timeout_s:
            if (
                self.state is TubeState.APPROACH_TUBE
                and self._approach_timeout_recoveries
                < self.policy.max_approach_timeout_recoveries
            ):
                self._approach_timeout_recoveries += 1
                self._state_started_s = observation.now_s
                # Stop the active target for a fresh observation before
                # continuing slow visual correction. Keep the original target
                # and drift limits; a timeout does not requalify a moved object.
                return self._pause_arm_for_reacquisition(
                    "APPROACH_TUBE correction window elapsed; automatic retry "
                    f"{self._approach_timeout_recoveries}/"
                    f"{self.policy.max_approach_timeout_recoveries}"
                )
            return self._hold(previous, f"{self.state.value} timeout exceeded")

        if self.state is TubeState.OBSERVE:
            return self._observe_scene(observation)
        if self.state is TubeState.APPROACH_TUBE:
            return self._approach_tube(observation)
        if self.state is TubeState.SIDE_GRASP:
            return self._side_grasp(observation)
        if self.state is TubeState.EXTRACT:
            return self._extract(observation)
        if self.state is TubeState.LEVEL_HELD_TUBE:
            return self._level_held_tube(observation)
        if self.state in {TubeState.PIVOT_MICRO_LOOSEN, TubeState.PIVOT_EVALUATE}:
            return self._pivot(observation)
        if self.state in {TubeState.RECLAMP, TubeState.RECLAMP_RECOVERY}:
            return self._reclamp(observation)
        if self.state is TubeState.APPROACH_SOCKET:
            return self._approach_socket(observation)
        if self.state is TubeState.ALIGN_SOCKET:
            return self._align_socket(observation)
        if self.state in {TubeState.INSERT_MICROSTEP, TubeState.INSERT_EVALUATE}:
            return self._insert(observation)
        if self.state is TubeState.JAM_RECOVERY_RETRACT:
            return self._jam_recovery(observation)
        if self.state is TubeState.RELEASE:
            return self._release(observation)
        if self.state is TubeState.RETREAT:
            return self._retreat(observation)
        kind = ActionKind.NONE if self.state is TubeState.COMPLETE else ActionKind.STOP_AND_HOLD
        return self._decision(previous, self.state, kind, {}, "terminal state", (), None)

    def _observe_scene(self, obs: ClosedLoopObservation) -> ClosedLoopDecision:
        if not self._startup_recovery_complete:
            return self._startup_recovery(obs)
        tactile_blockers = self._tactile_blockers(obs)
        if tactile_blockers:
            # The session-bootstrap interpreter learns its unloaded envelope in
            # OBSERVE.  Do not move toward the scene until that evidence exists.
            return self._wait_for_evidence("; ".join(tactile_blockers))
        blockers = tuple(
            f"tube_pose:{reason}"
            for reason in self._pose_blockers(
                obs.tube_pose,
                obs.now_s,
                SOURCE_POSE_CAPABILITIES,
                max_age_s=self.policy.observe_pose_max_age_s,
            )
        )
        blockers += tuple(
            f"socket_pose:{reason}"
            for reason in self._pose_blockers(
                obs.socket_pose,
                obs.now_s,
                SOCKET_POSE_CAPABILITIES,
                max_age_s=self.policy.observe_pose_max_age_s,
            )
        )
        if blockers:
            # No motion has been dispatched in OBSERVE. Temporal pose warm-up,
            # a rejected transparent-depth frame, or a temporarily lost mask
            # means keep observing. The state's deadline remains the bounded
            # retry budget.
            return self._wait_for_evidence("; ".join(blockers))
        assert obs.tube_pose is not None
        predicates = ("tube_pose_valid", "socket_frame_valid")
        grasp = grounded_grasp_target(obs.tube_pose, self.geometry)
        # Retain the metric target selected before the coarse top approach.
        # A transparent tube may lose usable depth once the open Hand-E enters
        # the camera ray.  Current RGB identity can then supervise one bounded
        # finite descent; it must never synthesize a renewed metric pose.
        self._last_qualified_grasp_target = grasp.copy()
        self._last_qualified_grasp_observed_at_s = obs.tube_pose.observed_at_s
        if (
            self.policy.occluded_grasp_close_distance_m is not None
            and self._within_occluded_close_window(obs.robot.T_base_tcp, grasp)
        ):
            recent_blockers = self._pose_blockers(
                obs.tube_pose, obs.now_s, SOURCE_POSE_CAPABILITIES
            ) + self._pose_blockers(obs.socket_pose, obs.now_s, SOCKET_POSE_CAPABILITIES)
            if recent_blockers:
                return self._wait_for_evidence(
                    "in-place grasp resume requires fresh task frames: "
                    + "; ".join(recent_blockers)
                )
            self.state = TubeState.SIDE_GRASP
            self._state_started_s = obs.now_s
            self._phase = "initial"
            self._last_qualified_grasp_target = grasp.copy()
            self._last_qualified_grasp_observed_at_s = obs.tube_pose.observed_at_s
            self._reset_servo(grasp)
            return self._decision(
                TubeState.OBSERVE,
                self.state,
                ActionKind.PAUSE_ARM,
                {"reacquire": True, "zero_active_servo": True, "resume_grasp_in_place": True},
                "fresh task frames place the measured TCP inside the grasp window; "
                "stop arm and resume grasp without a coarse retreat",
                (*predicates, "tcp_in_grasp_window"),
                TransitionDecision(
                    TubeState.OBSERVE,
                    self.state,
                    "LIVE_GRASP_REGION_READY",
                    "resume_grasp_in_place",
                    "fresh task frames and measured TCP permit in-place grasp resumption",
                    recovery=True,
                ),
            )
        semantic = self._transition(predicates)
        target = top_pregrasp_target(grasp, self.policy.top_standoff_m)
        self._reset_servo(target)
        self._defer_state_timeout_for_blocking_move()
        return self._decision(
            TubeState.OBSERVE,
            self.state,
            ActionKind.MOVE_TCP,
            {"T_base_tcp_target": target, "path": "top_approach", "plan_with_moveit": True},
            "move to a collision-checked vertical standoff while preserving demo orientation",
            predicates,
            semantic,
        )

    def _startup_recovery(self, obs: ClosedLoopObservation) -> ClosedLoopDecision:
        """Reverse one provenance-checked abandoned approach before reobserving."""

        tactile_blockers = self._tactile_blockers(obs)
        if tactile_blockers:
            if self._startup_recovery_armed:
                return self._hold(self.state, "; ".join(tactile_blockers))
            return self._wait_for_evidence("startup recovery: " + "; ".join(tactile_blockers))
        if (
            obs.tactile.contact_present or obs.tactile.bilateral_contact
            or obs.tactile.single_contact_side is not None or obs.tactile.slip_detected
            or obs.gripper.object_detected
        ):
            return self._hold(self.state, "startup reverse approach requires an empty gripper")
        if obs.gripper.moving or obs.gripper.width_m < 0.048:
            return self._hold(self.state, "startup reverse approach requires stationary open jaws")
        if not self._startup_recovery_armed and not obs.robot.motion_complete:
            return self._wait_for_evidence(
                "startup reverse approach waits for the accepted terminal hold to settle"
            )
        if not obs.robot.bounded_tcp_servo_supported:
            return self._hold(self.state, "startup reverse approach requires bounded TCP Servo")
        path = self._startup_recovery_path
        if not self._startup_recovery_armed:
            error_m, error_deg = pose_error(obs.robot.T_base_tcp, path[0])
            if error_m > 0.0025 or error_deg > 2.0:
                return self._hold(
                    self.state, "current TCP differs from recorded abandoned approach"
                )
            self._startup_recovery_armed = True
            self._state_started_s = obs.now_s
            self._reset_servo(path[self._startup_recovery_index])

        while self._startup_recovery_index < len(path):
            index = self._startup_recovery_index
            start, target = path[index - 1], path[index]
            segment = target[:3, 3] - start[:3, 3]
            squared_length = float(segment @ segment)
            fraction = (
                0.0 if squared_length < 1e-12 else float(np.clip(
                    (obs.robot.T_base_tcp[:3, 3] - start[:3, 3]) @ segment / squared_length,
                    0.0, 1.0,
                ))
            )
            nearest = start[:3, 3] + fraction * segment
            deviation = float(np.linalg.norm(obs.robot.T_base_tcp[:3, 3] - nearest))
            _, segment_rotation = pose_error(start, target)
            _, start_rotation = pose_error(obs.robot.T_base_tcp, start)
            _, end_rotation = pose_error(obs.robot.T_base_tcp, target)
            if (
                deviation > self.policy.servo_max_translation_step_m
                or min(start_rotation, end_rotation) > segment_rotation + 2.0
            ):
                return self._hold(
                    self.state, "measured TCP left the recorded reverse-approach path"
                )
            servo = self._servo(obs.robot.T_base_tcp, target)
            if servo.blockers:
                return self._hold(self.state, "; ".join(servo.blockers))
            final_waypoint = index == len(path) - 1
            if servo.within_deadband and (not final_waypoint or obs.robot.motion_complete):
                self._startup_recovery_index += 1
                if self._startup_recovery_index < len(path):
                    self._reset_servo(path[self._startup_recovery_index])
                continue
            return self._decision(
                self.state, self.state, ActionKind.SERVO_TCP,
                {
                    **self._servo_parameters(servo, target, "startup_reverse_approach"),
                    "startup_recovery": True,
                    "recovery_path_index": index,
                    "recovery_waypoint_count": len(path) - 1,
                },
                "reverse the recorded empty-gripper approach before fresh scene acquisition",
                ("recorded_recovery_path", "empty_open_gripper"), None,
            )
        self._startup_recovery_complete = True
        self._startup_recovery_armed = False
        self._state_started_s = obs.now_s
        self._initial_target = None
        self._previous_target = None
        return self._decision(
            self.state, self.state, ActionKind.PAUSE_ARM,
            {"zero_active_servo": True, "startup_recovery_complete": True},
            "recorded reverse approach completed; stop and acquire fresh task frames",
            ("startup_recovery_complete",), None,
        )

    def _approach_tube(self, obs: ClosedLoopObservation) -> ClosedLoopDecision:
        target_or_blockers = self._live_grasp_target(obs, source=True)
        if isinstance(target_or_blockers, tuple):
            if self._can_handoff_depth_loss_at_standoff(obs, target_or_blockers):
                assert self._last_qualified_grasp_target is not None
                assert self._last_qualified_grasp_observed_at_s is not None
                target = self._last_qualified_grasp_target.copy()
                predicates = ("approach_pose_reached",)
                semantic = self._transition(predicates)
                self._final_contact_target = target
                self._final_contact_start_tcp = obs.robot.T_base_tcp.copy()
                self._final_contact_arrival_tcp = None
                self._final_contact_source_observed_at_s = (
                    self._last_qualified_grasp_observed_at_s
                )
                self._final_contact_depth_loss_handoff = True
                self._phase = "final_contact_approach"
                self._state_started_s = obs.now_s
                self._settled_windows = 0
                self._reset_servo(target)
                decision = self._committed_final_approach(obs)
                return replace(
                    decision,
                    predicates=(*decision.predicates, "current_rgb_identity_retained"),
                    semantic_transition=semantic,
                )
            return self._pause_arm_for_reacquisition("; ".join(target_or_blockers))
        target = top_pregrasp_target(target_or_blockers, self.policy.top_standoff_m)
        servo = self._servo(obs.robot.T_base_tcp, target)
        if servo.blockers:
            return self._pause_arm_for_reacquisition("; ".join(servo.blockers))
        if servo.within_deadband and obs.robot.motion_complete:
            self._settled_windows += 1
        else:
            self._settled_windows = 0
        if self._settled_windows >= self.policy.servo_settle_windows:
            predicates = ("approach_pose_reached",)
            semantic = self._transition(predicates)
            approach_delta = target_or_blockers[:3, 3] - obs.robot.T_base_tcp[:3, 3]
            approach_norm = float(np.linalg.norm(approach_delta))
            if approach_norm > 1e-6:
                self._approach_direction_base = approach_delta / approach_norm
            self._reset_servo(target_or_blockers)
            grasp_servo = self._servo(obs.robot.T_base_tcp, target_or_blockers)
            return self._decision(
                TubeState.APPROACH_TUBE,
                self.state,
                ActionKind.SERVO_TCP,
                self._servo_parameters(grasp_servo, target_or_blockers, "grasp"),
                "descend from above toward the live demo-relative grasp",
                predicates,
                semantic,
            )
        return self._decision(
            self.state,
            self.state,
            ActionKind.SERVO_TCP,
            self._servo_parameters(servo, target, "top_pregrasp"),
            "bounded correction to the moving top-pregrasp target",
            (),
            None,
        )

    def _can_handoff_depth_loss_at_standoff(
        self, obs: ClosedLoopObservation, blockers: tuple[str, ...]
    ) -> bool:
        """Commit the cached metric target only from its reached top standoff.

        The RGB-identity certificate is emitted only when the current SAM
        mask, fixed tracking id, IoU and both image endpoints still match the
        last metric-qualified frame.  It is current visibility evidence, not
        a replacement pose.  The remaining motion stays inside the explicitly
        configured final-approach segment and is supervised by fresh
        no-contact tactile evidence on every resident-loop iteration.
        """

        target = self._last_qualified_grasp_target
        distance_limit = self.policy.final_approach_commit_distance_m
        if (
            blockers != ("required_live_pose_missing",)
            or not self._source_rgb_identity_current(obs)
            or target is None
            or self._last_qualified_grasp_observed_at_s is None
            or distance_limit is None
            or not obs.robot.motion_complete
            or not obs.robot.bounded_tcp_servo_supported
            or obs.gripper.moving
            or obs.gripper.object_detected
            or obs.gripper.width_m < 0.048
            or self._tactile_blockers(obs)
            or obs.tactile.contact_present
            or obs.tactile.bilateral_contact
            or obs.tactile.single_contact_side is not None
            or obs.tactile.slip_detected
        ):
            return False
        standoff = top_pregrasp_target(target, self.policy.top_standoff_m)
        standoff_distance, standoff_angle = pose_error(obs.robot.T_base_tcp, standoff)
        remaining_distance, remaining_angle = pose_error(obs.robot.T_base_tcp, target)
        return bool(
            standoff_distance <= self.policy.servo_translation_deadband_m
            and standoff_angle <= self.policy.servo_rotation_deadband_deg
            and self.policy.servo_translation_deadband_m
            < remaining_distance <= distance_limit
            and remaining_angle <= self.policy.servo_rotation_deadband_deg
        )

    def _side_grasp(self, obs: ClosedLoopObservation) -> ClosedLoopDecision:
        if self._phase == "final_contact_approach":
            return self._committed_final_approach(obs)
        if self._phase in {"initial", "servo_grasp", "recenter_complete"}:
            previous_target = self._last_qualified_grasp_target
            previous_source_time = self._last_qualified_grasp_observed_at_s
            target_or_blockers = self._live_grasp_target(obs, source=True)
            if isinstance(target_or_blockers, tuple):
                return self._pause_arm_for_reacquisition("; ".join(target_or_blockers))
            servo = self._servo(obs.robot.T_base_tcp, target_or_blockers)
            if servo.blockers:
                return self._pause_arm_for_reacquisition("; ".join(servo.blockers))
            if self._can_commit_final_approach(
                obs, target_or_blockers, previous_target, previous_source_time
            ):
                self._final_contact_target = target_or_blockers.copy()
                self._final_contact_start_tcp = obs.robot.T_base_tcp.copy()
                self._final_contact_arrival_tcp = None
                assert obs.tube_pose is not None
                self._final_contact_source_observed_at_s = obs.tube_pose.observed_at_s
                self._final_contact_depth_loss_handoff = False
                self._phase = "final_contact_approach"
                self._state_started_s = obs.now_s
                self._settled_windows = 0
                self._reset_servo(self._final_contact_target)
                return self._committed_final_approach(obs)
            occluded_close_ready = bool(
                self._vision_frozen_for_close
                and self._within_occluded_close_window(obs.robot.T_base_tcp, target_or_blockers)
            )
            if self._vision_frozen_for_close and not occluded_close_ready:
                return self._hold(
                    self.state,
                    "tube became gripper-occluded before the frozen target was reached",
                )
            self._settled_windows = (
                self._settled_windows + 1
                if servo.within_deadband and obs.robot.motion_complete
                else 0
            )
            if (
                occluded_close_ready
                and self.policy.occluded_grasp_close_distance_m is not None
            ):
                # START_CONTINUOUS_CLOSE clears active arm motion before
                # commanding the gripper. Do not issue a blind servo step or
                # wait for tactile contact that closing itself must establish.
                self._settled_windows = self.policy.servo_settle_windows
            self._phase = "servo_grasp"
            if self._settled_windows >= self.policy.servo_settle_windows:
                self._phase = "closing"
                self._state_started_s = obs.now_s
                self._shallow_contact_unready_windows = 0
                self._shallow_contact_ready_windows = 0
                return self._decision(
                    self.state,
                    self.state,
                    ActionKind.START_CONTINUOUS_CLOSE,
                    {
                        "target_width_m": self.policy.continuous_close_min_width_m,
                        "speed": self.policy.continuous_close_speed,
                        "force": self.policy.continuous_close_force,
                        "stop_predicate": "first_fresh_optical_pad_contact",
                        "hande_object_status_role": "corroborating_only",
                        "command_count": 1,
                        "zero_active_servo": True,
                    },
                    "issue one continuous close and supervise feedback without stepwise commands",
                    (),
                    None,
                )
            return self._decision(
                self.state,
                self.state,
                (
                    ActionKind.COLLECT_EVIDENCE
                    if self._vision_frozen_for_close
                    else ActionKind.SERVO_TCP
                ),
                (
                    {"vision_pose_frozen": True, "next_supervisor": "tactile"}
                    if self._vision_frozen_for_close
                    else self._servo_parameters(
                        servo, target_or_blockers, "demo_relative_grasp"
                    )
                ),
                (
                    "gripper occlusion detected inside the switch window; "
                    "freeze pose and switch to tactile"
                    if self._vision_frozen_for_close
                    else "live 6D correction before contact"
                ),
                (),
                None,
            )

        tactile_blockers = self._tactile_blockers(obs)
        if tactile_blockers:
            return self._hold(self.state, "; ".join(tactile_blockers))
        if self._phase == "shallow_retry_opening":
            return self._continue_shallow_retry_opening(obs)
        if self._phase == "shallow_retry_approach":
            return self._continue_shallow_retry_approach(obs)
        if self._phase == "closing":
            if obs.tactile.contact_present:
                self._contact_stop_width_m = obs.gripper.width_m
                self._phase = "close_stopping"
                self._shallow_contact_unready_windows = 0
                self._shallow_contact_ready_windows = 0
                return self._decision(
                    self.state,
                    self.state,
                    ActionKind.STOP_GRIPPER,
                    {},
                    "fresh tactile contact was detected; stop the one continuous close command",
                    ("contact_present",),
                    None,
                )
            if obs.gripper.object_detected or not obs.gripper.moving:
                # gOBJ is corroborating-only.  Stop first, then inspect two
                # fresh stationary optical windows before deciding whether a
                # bounded deeper retry is needed.
                self._contact_stop_width_m = obs.gripper.width_m
                self._phase = "close_stopping"
                self._shallow_contact_unready_windows = 0
                self._shallow_contact_ready_windows = 0
                return self._decision(
                    self.state,
                    self.state,
                    ActionKind.STOP_GRIPPER,
                    {"hande_object_status_role": "corroborating_only"},
                    "close stopped before optical contact; collect stationary pad images",
                    (),
                    None,
                )
            return self._decision(
                self.state,
                self.state,
                ActionKind.MONITOR,
                {},
                "continuous close remains active; no new gripper command",
                (),
                None,
            )

        if self._phase == "close_stopping":
            if obs.gripper.moving:
                return self._decision(
                    self.state,
                    self.state,
                    ActionKind.MONITOR,
                    {},
                    "wait for the Hand-E stop acknowledgement",
                    (),
                    None,
                )
            width_ok = (
                self.policy.retained_width_min_m
                <= obs.gripper.width_m
                <= self.policy.retained_width_max_m
            )
            if obs.tactile.bilateral_contact_interior and width_ok:
                self._shallow_contact_ready_windows += 1
                self._shallow_contact_unready_windows = 0
            else:
                self._shallow_contact_ready_windows = 0
            if (
                self._shallow_contact_ready_windows
                >= self.policy.shallow_contact_confirmation_windows
            ):
                approach_direction = self._committed_approach_direction()
                if approach_direction is None:
                    return self._hold(
                        self.state,
                        "retained contact is valid but committed approach direction is unavailable",
                    )
                predicates = ("grasp_established",)
                semantic = self._transition(predicates)
                self._extract_target = obs.robot.T_base_tcp.copy()
                self._extract_target[:3, 3] -= (
                    approach_direction * self.policy.extract_clearance_m
                )
                self._reset_servo(self._extract_target)
                servo = self._servo(obs.robot.T_base_tcp, self._extract_target)
                return self._decision(
                    TubeState.SIDE_GRASP,
                    self.state,
                    ActionKind.SERVO_TCP,
                    self._servo_parameters(
                        servo, self._extract_target, "vertical_extract"
                    ),
                    "start supervised vertical extraction with one bounded Servo step",
                    predicates,
                    semantic,
                )
            if obs.tactile.single_contact_side is not None:
                self._single_contact_windows += 1
                self._shallow_contact_unready_windows = 0
                self._shallow_contact_ready_windows = 0
            else:
                self._single_contact_windows = 0
            if (
                self._single_contact_windows
                >= self.policy.single_contact_confirmation_windows
            ):
                if not obs.robot.bounded_tcp_servo_supported:
                    return self._hold(
                        self.state,
                        "one-sided contact needs a qualified bounded TCP jaw-axis servo",
                    )
                if self._centering_recoveries >= self.policy.max_centering_recoveries:
                    return self._hold(self.state, "one-sided contact recovery budget exhausted")
                if self._contact_stop_width_m is None:
                    return self._hold(self.state, "first-contact Hand-E width is unavailable")
                if self._recenter_baseline_contact_width_m is None:
                    self._recenter_baseline_contact_width_m = self._contact_stop_width_m
                    offset = self.policy.recenter_probe_step_m
                    probe_reason = "first sign probe along TCP +X"
                else:
                    improvement = (
                        self._recenter_baseline_contact_width_m
                        - self._contact_stop_width_m
                    )
                    if (
                        improvement >= self.policy.recenter_width_improvement_m
                        and not self._recenter_direction_reversed
                    ):
                        offset = self.policy.recenter_probe_step_m
                        probe_reason = "positive jaw-axis probe improved first-contact width"
                    elif not self._recenter_direction_reversed:
                        offset = -self.policy.recenter_reverse_step_m
                        self._recenter_direction_reversed = True
                        probe_reason = (
                            "positive probe did not improve contact; reverse once across baseline"
                        )
                    else:
                        return self._hold(
                            self.state,
                            "one-sided contact persisted after the single sign reversal",
                        )
                if (
                    self._recenter_total_travel_m + abs(offset)
                    > self.policy.recenter_total_travel_budget_m + 1e-12
                ):
                    return self._hold(self.state, "jaw-axis sign-probe travel budget exhausted")
                self._centering_recoveries += 1
                self._recenter_total_travel_m += abs(offset)
                self._grasp_recenter_local_x_m += offset
                self._phase = "recentering"
                self._single_contact_windows = 0
                return self._decision(
                    self.state,
                    self.state,
                    ActionKind.OPEN_AND_RECENTER,
                    {
                        "open_delta_m": self.policy.recenter_open_delta_m,
                        "tcp_local_x_offset_m": offset,
                        "strategy": "bounded_unknown_sensor_sign_probe",
                        "observed_single_contact_sensor": obs.tactile.single_contact_side,
                        "first_contact_width_m": self._contact_stop_width_m,
                        "baseline_first_contact_width_m": (
                            self._recenter_baseline_contact_width_m
                        ),
                        "direction_reversed": self._recenter_direction_reversed,
                        "then_reobserve_6d_pose": True,
                    },
                    f"persistent one-pad contact: {probe_reason}",
                    ("one_sided_contact",),
                    None,
                )
            # A stopped close gets a short stationary confirmation window.
            # Bilateral optical contact at a pad edge is still too shallow;
            # gOBJ alone is never allowed to authorize extraction.
            if (
                obs.tactile.single_contact_side is None
                and not (obs.tactile.bilateral_contact_interior and width_ok)
            ):
                self._shallow_contact_unready_windows += 1
                if (
                    self._shallow_contact_unready_windows
                    >= self.policy.shallow_contact_confirmation_windows
                ):
                    reason = (
                        "bilateral optical patches remain at the pad boundary"
                        if obs.tactile.bilateral_contact
                        else "stationary close retained no bilateral optical contact"
                    )
                    if not width_ok:
                        reason += "; retained Hand-E width is outside the grasp band"
                    return self._start_shallow_contact_retry(obs, reason=reason)
            return self._decision(
                self.state,
                self.state,
                ActionKind.COLLECT_EVIDENCE,
                {},
                "stationary tactile confirmation after stopping",
                (),
                None,
            )

        if self._phase == "recentering":
            if obs.gripper.moving or not obs.robot.motion_complete:
                return self._decision(
                    self.state,
                    self.state,
                    ActionKind.MONITOR,
                    {},
                    "wait for bounded open/recenter completion",
                    (),
                    None,
                )
            self._phase = "recenter_complete"
            self._settled_windows = 0
            return self._decision(
                self.state,
                self.state,
                ActionKind.COLLECT_EVIDENCE,
                {},
                "reobserve the tube before another continuous close",
                (),
                None,
            )
        return self._hold(self.state, f"unknown side-grasp phase {self._phase}")

    def _committed_approach_direction(self) -> np.ndarray | None:
        """Return the measured finite approach direction, never a world-axis guess."""

        if self._final_contact_start_tcp is not None and self._final_contact_target is not None:
            delta = (
                self._final_contact_target[:3, 3]
                - self._final_contact_start_tcp[:3, 3]
            )
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                return delta / norm
        if self._approach_direction_base is not None:
            return self._approach_direction_base.copy()
        if self._last_qualified_grasp_target is not None:
            standoff = top_pregrasp_target(
                self._last_qualified_grasp_target,
                self.policy.top_standoff_m,
            )
            delta = (
                self._last_qualified_grasp_target[:3, 3]
                - standoff[:3, 3]
            )
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                return delta / norm
        return None

    def _start_shallow_contact_retry(
        self,
        obs: ClosedLoopObservation,
        *,
        reason: str,
    ) -> ClosedLoopDecision:
        direction = self._committed_approach_direction()
        if direction is None:
            return self._hold(
                self.state,
                f"{reason}; no committed approach direction is available",
            )
        next_total = self._shallow_retry_total_travel_m + self.policy.shallow_retry_step_m
        if (
            self._shallow_contact_retries >= self.policy.max_shallow_contact_retries
            or next_total > self.policy.shallow_retry_total_travel_budget_m + 1e-12
        ):
            return self._hold(
                self.state,
                f"{reason}; bounded shallow-contact retry budget exhausted",
            )
        if not obs.robot.motion_complete or not obs.robot.bounded_tcp_servo_supported:
            return self._hold(
                self.state,
                f"{reason}; deeper retry requires a stationary bounded TCP Servo",
            )

        self._shallow_contact_retries += 1
        self._shallow_retry_total_travel_m = next_total
        self._shallow_contact_unready_windows = 0
        self._shallow_contact_ready_windows = 0
        self._single_contact_windows = 0
        if self._shallow_retry_origin_tcp is None:
            self._shallow_retry_origin_tcp = obs.robot.T_base_tcp.copy()
        self._shallow_retry_target = self._shallow_retry_origin_tcp.copy()
        self._shallow_retry_target[:3, 3] += direction * next_total
        self._phase = "shallow_retry_opening"
        self._state_started_s = obs.now_s
        delta = max(0.0, self.policy.shallow_retry_open_width_m - obs.gripper.width_m)
        return self._decision(
            self.state,
            self.state,
            ActionKind.OPEN_GRIPPER_STEP,
            {
                "delta_m": delta,
                "shallow_contact_retry": True,
                "retry_index": self._shallow_contact_retries,
                "retry_step_m": self.policy.shallow_retry_step_m,
                "retry_total_travel_m": self._shallow_retry_total_travel_m,
                "source_latch_max_displacement_m": (
                    self._shallow_retry_total_travel_m
                    + self.policy.servo_translation_deadband_m
                ),
                "approach_direction_base": direction.copy(),
            },
            f"{reason}; reopen before the automatic 5 mm deeper retry",
            ("shallow_contact_recovery",),
            None,
        )

    def _continue_shallow_retry_opening(
        self, obs: ClosedLoopObservation
    ) -> ClosedLoopDecision:
        if (
            obs.gripper.moving
            or obs.gripper.width_m
            < self.policy.shallow_retry_open_width_m - 0.00075
        ):
            return self._decision(
                self.state,
                self.state,
                ActionKind.MONITOR,
                {},
                "wait for the bounded shallow-retry reopen",
                (),
                None,
            )
        if obs.tactile.contact_present:
            return self._decision(
                self.state,
                self.state,
                ActionKind.COLLECT_EVIDENCE,
                {},
                "wait for fresh no-contact pad images after reopening",
                (),
                None,
            )
        if not obs.robot.motion_complete:
            return self._decision(
                self.state,
                self.state,
                ActionKind.MONITOR,
                {},
                "wait for the arm to remain stationary before the deeper probe",
                (),
                None,
            )
        self._phase = "shallow_retry_approach"
        return self._continue_shallow_retry_approach(obs)

    def _continue_shallow_retry_approach(
        self, obs: ClosedLoopObservation
    ) -> ClosedLoopDecision:
        target = self._shallow_retry_target
        if target is None:
            return self._hold(self.state, "shallow-contact retry target is unavailable")
        if obs.tactile.contact_present:
            return self._hold(
                self.state,
                "unexpected open-finger contact during the bounded deeper descent",
            )
        if obs.gripper.moving or obs.gripper.width_m < 0.048:
            return self._hold(
                self.state,
                "gripper did not remain open during the bounded deeper descent",
            )
        if not obs.robot.bounded_tcp_servo_supported:
            return self._hold(self.state, "bounded TCP Servo became unavailable")
        distance, angle = pose_error(obs.robot.T_base_tcp, target)
        if angle > self.policy.servo_rotation_deadband_deg:
            return self._hold(self.state, "TCP orientation changed during deeper descent")
        if distance > self.policy.shallow_retry_step_m + self.policy.servo_translation_deadband_m:
            return self._hold(self.state, "TCP left the bounded deeper-descent segment")
        retry_evidence = {
            "shallow_contact_retry": True,
            "retry_index": self._shallow_contact_retries,
            "retry_total_travel_m": self._shallow_retry_total_travel_m,
            "source_latch_max_displacement_m": (
                self._shallow_retry_total_travel_m
                + self.policy.servo_translation_deadband_m
            ),
        }
        if (
            obs.robot.motion_complete
            and distance <= self.policy.servo_translation_deadband_m
        ):
            self._phase = "closing"
            self._state_started_s = obs.now_s
            self._shallow_contact_unready_windows = 0
            self._shallow_contact_ready_windows = 0
            return self._decision(
                self.state,
                self.state,
                ActionKind.START_CONTINUOUS_CLOSE,
                {
                    **retry_evidence,
                    "target_width_m": self.policy.continuous_close_min_width_m,
                    "speed": self.policy.continuous_close_speed,
                    "force": self.policy.continuous_close_force,
                    "stop_predicate": "first_fresh_optical_pad_contact",
                    "hande_object_status_role": "corroborating_only",
                    "command_count": 1,
                    "zero_active_servo": True,
                },
                "deeper endpoint reached under no-contact tactile supervision; close once",
                ("bounded_deeper_retry_arrived",),
                None,
            )
        return self._decision(
            self.state,
            self.state,
            ActionKind.SERVO_TCP,
            {
                **retry_evidence,
                "T_base_tcp_command": target.copy(),
                "T_base_tcp_live_target": target.copy(),
                "translation_error_m": distance,
                "rotation_error_deg": angle,
                "grounding": "committed_approach_direction_shallow_retry",
            },
            "one bounded 5 mm deeper probe with fresh no-contact tactile supervision",
            (),
            None,
        )

    def _can_commit_final_approach(
        self,
        obs: ClosedLoopObservation,
        target: np.ndarray,
        previous_target: np.ndarray | None,
        previous_source_time: float | None,
    ) -> bool:
        distance_limit = self.policy.final_approach_commit_distance_m
        if (
            distance_limit is None
            or previous_target is None
            or previous_source_time is None
            or obs.tube_pose is None
            or self._vision_frozen_for_close
            or not obs.robot.bounded_tcp_servo_supported
            or self._tactile_blockers(obs)
            or obs.tactile.contact_present
            or obs.tactile.slip_detected
            or obs.gripper.object_detected
            or obs.gripper.moving
            or obs.gripper.width_m < 0.048
            or obs.tube_pose.observed_at_s <= previous_source_time
        ):
            return False
        distance, orientation = pose_error(obs.robot.T_base_tcp, target)
        drift, rotation = pose_error(previous_target, target)
        return bool(
            self.policy.servo_translation_deadband_m < distance <= distance_limit
            and orientation <= self.policy.servo_rotation_deadband_deg
            and drift <= 0.003
            and rotation <= 2.0
        )

    def _committed_final_approach(self, obs: ClosedLoopObservation) -> ClosedLoopDecision:
        target = self._final_contact_target
        started_tcp = self._final_contact_start_tcp
        if (
            target is None or started_tcp is None
            or self._final_contact_source_observed_at_s is None
        ):
            return self._hold(self.state, "committed final approach lacks its original evidence")
        blockers = self._tactile_blockers(obs)
        if blockers:
            return self._hold(self.state, "; ".join(blockers))
        if (
            obs.tactile.contact_present or obs.tactile.bilateral_contact
            or obs.tactile.single_contact_side is not None
            or obs.gripper.object_detected or obs.tactile.slip_detected
        ):
            return self._hold(self.state, "unexpected contact during committed final approach")
        if obs.gripper.moving or obs.gripper.width_m < 0.048:
            return self._hold(self.state, "open gripper changed during committed final approach")
        if not obs.robot.bounded_tcp_servo_supported:
            return self._hold(self.state, "committed final approach requires bounded TCP Servo")
        if obs.tube_pose is None:
            if (
                self._final_contact_arrival_tcp is None
                and not self._source_rgb_identity_current(obs)
            ):
                return self._pause_arm_for_reacquisition(
                    "final approach needs current RGB identity when depth is unavailable"
                )
        else:
            pose_blockers = self._pose_blockers(obs.tube_pose, obs.now_s, SOURCE_POSE_CAPABILITIES)
            if pose_blockers:
                return self._hold(self.state, "; ".join(pose_blockers))
            observed_target = grounded_grasp_target(obs.tube_pose, self.geometry)
            observed_target[:3, 3] += observed_target[:3, :3] @ np.asarray(
                [self._grasp_recenter_local_x_m, 0.0, 0.0]
            )
            drift, angle = pose_error(target, observed_target)
            if drift > 0.003 or angle > 2.0:
                return self._hold(self.state, "live target moved during committed final approach")
        travel, _ = pose_error(started_tcp, obs.robot.T_base_tcp)
        committed_distance, _ = pose_error(started_tcp, target)
        _, orientation = pose_error(obs.robot.T_base_tcp, target)
        if (
            travel > committed_distance + self.policy.servo_translation_deadband_m
            or orientation > self.policy.servo_rotation_deadband_deg
        ):
            return self._hold(self.state, "measured TCP left the committed final approach bounds")
        servo = self._servo(obs.robot.T_base_tcp, target)
        if servo.blockers:
            return self._hold(self.state, "; ".join(servo.blockers))
        evidence = {
            "final_contact_target": target.copy(),
            "committed_source_observed_at_s": self._final_contact_source_observed_at_s,
            "depth_loss_handoff": self._final_contact_depth_loss_handoff,
        }
        if self._final_contact_arrival_tcp is not None:
            drift, rotation = pose_error(self._final_contact_arrival_tcp, obs.robot.T_base_tcp)
            if (
                not servo.within_deadband or not obs.robot.motion_complete
                or drift > self.policy.servo_translation_deadband_m
                or rotation > self.policy.servo_rotation_deadband_deg
            ):
                return self._hold(self.state, "measured TCP moved after final contact arrival")
            self._settled_windows += 1
        elif servo.within_deadband and obs.robot.motion_complete:
            # Arrival is evidenced while RGB identity still qualifies. Stop
            # here and keep the measured endpoint; closing can then complete
            # its stationary confirmation even if the finger hides the tube.
            self._final_contact_arrival_tcp = obs.robot.T_base_tcp.copy()
            self._settled_windows = 1
            return self._decision(
                self.state, self.state, ActionKind.PAUSE_ARM,
                {**evidence, "final_contact_arrived": True, "zero_active_servo": True},
                "measured final target reached; stop arm before stationary tactile handoff",
                ("committed_final_target_reached",), None,
            )
        else:
            self._settled_windows = 0
        if self._settled_windows >= self.policy.servo_settle_windows:
            self._phase = "closing"
            self._state_started_s = obs.now_s
            self._shallow_contact_unready_windows = 0
            self._shallow_contact_ready_windows = 0
            return self._decision(
                self.state, self.state, ActionKind.START_CONTINUOUS_CLOSE,
                {
                    **evidence,
                    "committed_final_approach": True,
                    "target_width_m": self.policy.continuous_close_min_width_m,
                    "speed": self.policy.continuous_close_speed,
                    "force": self.policy.continuous_close_force,
                    "stop_predicate": "first_fresh_optical_pad_contact",
                    "hande_object_status_role": "corroborating_only",
                    "command_count": 1,
                    "zero_active_servo": True,
                },
                "committed target reached with measured TCP; stop arm and close continuously",
                ("committed_final_target_reached",), None,
            )
        if self._final_contact_arrival_tcp is not None:
            return self._decision(
                self.state, self.state, ActionKind.PAUSE_ARM,
                {**evidence, "final_contact_arrived": True, "zero_active_servo": True},
                "confirm stationary final arrival before closing; no further arm motion",
                ("committed_final_target_reached",), None,
            )
        return self._decision(
            self.state, self.state, ActionKind.SERVO_TCP,
            {
                **self._servo_parameters(servo, target, "committed_final_contact_approach"),
                **evidence,
                "final_contact_approach": True,
            },
            "finish the committed approach with current RGB identity and tactile supervision",
            (), None,
        )

    def _extract(self, obs: ClosedLoopObservation) -> ClosedLoopDecision:
        tactile_blockers = self._tactile_blockers(obs)
        if tactile_blockers:
            return self._hold(self.state, "; ".join(tactile_blockers))
        if not obs.tactile.contact_present:
            return self._hold(self.state, "contact was lost during extraction")
        if self._extract_target is None:
            return self._hold(self.state, "extract target was not frozen at grasp")
        servo = self._servo(obs.robot.T_base_tcp, self._extract_target)
        if servo.within_deadband and obs.robot.motion_complete and obs.tube_clear_of_source:
            level_target_or_blockers = self._held_level_target(obs)
            if isinstance(level_target_or_blockers, tuple):
                return self._hold(self.state, "; ".join(level_target_or_blockers))
            predicates = ("tube_clear_of_source", "contact_present")
            semantic = self._transition(predicates)
            self._level_target = level_target_or_blockers
            self._level_stationary_slip_windows = 0
            self._level_progress_reference_deg = None
            self._level_stall_windows = 0
            self._level_escape_attempted = False
            self._reset_servo(self._level_target)
            level_servo = self._servo(obs.robot.T_base_tcp, self._level_target)
            return self._decision(
                TubeState.EXTRACT,
                self.state,
                ActionKind.SERVO_TCP,
                self._servo_parameters(
                    level_servo,
                    self._level_target,
                    "rigid_held_tube_level",
                ),
                "keep the grasp closed and rotate the live held-tube axis to horizontal",
                predicates,
                semantic,
            )
        return self._decision(
            self.state,
            self.state,
            ActionKind.SERVO_TCP,
            self._servo_parameters(servo, self._extract_target, "vertical_extract"),
            "vertical extraction while tactile contact remains",
            (),
            None,
        )

    @staticmethod
    def _source_rgb_identity_current(obs: ClosedLoopObservation) -> bool:
        """Accept the explicit certificate; depth-only loss implies it for compatibility."""

        return bool(obs.source_rgb_identity_retained or obs.source_depth_only_loss)

    def _level_held_tube(self, obs: ClosedLoopObservation) -> ClosedLoopDecision:
        tactile_blockers = self._tactile_blockers(obs)
        if tactile_blockers:
            return self._hold(self.state, "; ".join(tactile_blockers))

        contact_predicates: list[str] = []
        if not obs.tactile.contact_present:
            contact_predicates.append("contact_lost")
        else:
            contact_predicates.append("contact_present")
        if "contact_lost" in contact_predicates:
            semantic = self._transition(contact_predicates)
            return self._decision(
                TubeState.LEVEL_HELD_TUBE,
                self.state,
                ActionKind.STOP_AND_HOLD,
                {
                    "cancel_active_arm_goal": True,
                    "stop_active_gripper_command": True,
                    "cease_dispatch": True,
                },
                semantic.reason,
                tuple(contact_predicates),
                semantic,
            )

        # A deliberate rigid wrist-level command changes the tactile image and
        # can move a patch centre even while both finger interiors retain the
        # tube.  Motion-correlated slip therefore supervises the move but does
        # not terminate it.  Once stationary, independently debounce a
        # persistent slip before promoting the existing ``slip_detected``
        # transition predicate.  A slip that no longer has bilateral interior
        # contact remains an immediate failure.
        leveling_slip_during_motion = bool(
            obs.tactile.slip_detected
            and obs.tactile.bilateral_contact_interior
            and not obs.robot.motion_complete
        )
        stationary_slip_pending = False
        if obs.tactile.slip_detected and not leveling_slip_during_motion:
            if not obs.tactile.bilateral_contact_interior:
                self._level_stationary_slip_windows = (
                    self.policy.level_stationary_slip_confirmation_windows
                )
            else:
                self._level_stationary_slip_windows += 1
            if (
                self._level_stationary_slip_windows
                >= self.policy.level_stationary_slip_confirmation_windows
            ):
                contact_predicates.append("slip_detected")
                semantic = self._transition(contact_predicates)
                return self._decision(
                    TubeState.LEVEL_HELD_TUBE,
                    self.state,
                    ActionKind.STOP_AND_HOLD,
                    {
                        "cancel_active_arm_goal": True,
                        "stop_active_gripper_command": True,
                        "cease_dispatch": True,
                        "stationary_slip_confirmation_windows": (
                            self._level_stationary_slip_windows
                        ),
                    },
                    semantic.reason,
                    tuple(contact_predicates),
                    semantic,
                )
            stationary_slip_pending = True
        else:
            self._level_stationary_slip_windows = 0

        if stationary_slip_pending:
            self._settled_windows = 0
            return self._decision(
                self.state,
                self.state,
                ActionKind.COLLECT_EVIDENCE,
                {
                    "stationary_slip_confirmation_windows": (
                        self._level_stationary_slip_windows
                    ),
                    "stationary_slip_confirmation_required": (
                        self.policy.level_stationary_slip_confirmation_windows
                    ),
                    "bilateral_contact_interior": True,
                },
                "wait for stationary slip confirmation while bilateral interior "
                "contact remains",
                ("contact_present",),
                None,
            )

        pose_blockers = self._pose_blockers(
            obs.tube_pose,
            obs.now_s,
            HELD_POSE_CAPABILITIES,
        )
        if pose_blockers:
            return self._hold(self.state, "; ".join(pose_blockers))
        assert obs.tube_pose is not None

        if self._level_target is None:
            level_target_or_blockers = self._held_level_target(obs)
            if isinstance(level_target_or_blockers, tuple):
                return self._hold(self.state, "; ".join(level_target_or_blockers))
            self._level_target = level_target_or_blockers
            self._reset_servo(self._level_target)

        servo = self._servo(obs.robot.T_base_tcp, self._level_target)
        if servo.blockers:
            return self._hold(self.state, "; ".join(servo.blockers))
        horizontal_error_deg = tube_horizontal_error_deg(
            obs.tube_pose.T_base_object,
            tube_axis_tube=self.geometry.tube_axis_tube,
        )
        tube_horizontal = horizontal_error_deg <= self.policy.level_axis_tolerance_deg
        target_reached = bool(servo.within_deadband and obs.robot.motion_complete)
        if tube_horizontal and target_reached:
            self._settled_windows += 1
        else:
            self._settled_windows = 0

        predicates = ["contact_present"]
        if self._settled_windows >= self.policy.level_settle_windows:
            predicates.extend(("tube_horizontal", "level_target_reached"))
            semantic = self._transition(predicates)
            self._pivot_steps = 1
            self._phase = "pivot_commanded"
            return self._decision(
                TubeState.LEVEL_HELD_TUBE,
                self.state,
                ActionKind.OPEN_GRIPPER_STEP,
                {
                    "delta_m": self.policy.pivot_open_step_m,
                    "level_confirmation_windows": self.policy.level_settle_windows,
                    "tube_horizontal_error_deg": horizontal_error_deg,
                },
                "stable horizontal hold confirmed; apply exactly one bounded micro-loosen",
                predicates,
                semantic,
            )

        semantic = self._transition(predicates)
        # Do not spend the entire state timeout repeatedly issuing the same
        # Cartesian rotation after the measured error has stopped decreasing.
        # One MoveIt pose goal lets the redundant arm choose a different joint
        # path.  The tube target orientation is unchanged; only base Z is
        # lifted to create elbow room, and the Hand-E remains closed.
        if not self._level_escape_attempted:
            if self._level_progress_reference_deg is None:
                self._level_progress_reference_deg = servo.rotation_error_deg
                self._level_stall_windows = 0
            elif (
                self._level_progress_reference_deg - servo.rotation_error_deg
                >= self.policy.level_stall_min_progress_deg
            ):
                self._level_progress_reference_deg = servo.rotation_error_deg
                self._level_stall_windows = 0
            else:
                self._level_stall_windows += 1
            if (
                self._level_stall_windows
                >= self.policy.level_stall_confirmation_windows
            ):
                escape_target = self._level_target.copy()
                escape_target[2, 3] += self.policy.level_escape_lift_m
                self._level_target = escape_target
                self._level_escape_attempted = True
                self._level_stall_windows = 0
                self._reset_servo(escape_target)
                self._defer_state_timeout_for_blocking_move()
                return self._decision(
                    self.state,
                    self.state,
                    ActionKind.MOVE_TCP,
                    {
                        "T_base_tcp_target": escape_target,
                        "plan_with_moveit": True,
                        "level_singularity_escape": True,
                        "level_escape_lift_m": self.policy.level_escape_lift_m,
                        "stalled_rotation_error_deg": servo.rotation_error_deg,
                        "stall_confirmation_windows": (
                            self.policy.level_stall_confirmation_windows
                        ),
                        "gripper_remains_closed": True,
                    },
                    "leveling progress stalled; keep the grasp closed and use "
                    "one collision-planned lifted redundancy escape",
                    tuple(predicates),
                    semantic,
                )

        parameters = self._servo_parameters(
            servo,
            self._level_target,
            "rigid_held_tube_level",
        )
        parameters.update(
            {
                "tube_horizontal_error_deg": horizontal_error_deg,
                "tube_horizontal_tolerance_deg": self.policy.level_axis_tolerance_deg,
                "level_confirmation_windows": self._settled_windows,
                "level_confirmation_required": self.policy.level_settle_windows,
                "gripper_remains_closed": True,
                "motion_correlated_slip_ignored": leveling_slip_during_motion,
                "bilateral_contact_interior": (
                    obs.tactile.bilateral_contact_interior
                ),
            }
        )
        return self._decision(
            self.state,
            self.state,
            ActionKind.SERVO_TCP,
            parameters,
            "closed-loop rigid leveling before any gripper opening",
            predicates,
            semantic,
        )

    def _pivot(self, obs: ClosedLoopObservation) -> ClosedLoopDecision:
        tactile_blockers = self._tactile_blockers(obs)
        if tactile_blockers:
            return self._hold(self.state, "; ".join(tactile_blockers))
        if self.state is TubeState.PIVOT_MICRO_LOOSEN:
            if obs.gripper.moving:
                return self._decision(
                    self.state,
                    self.state,
                    ActionKind.MONITOR,
                    {},
                    "pivot opening step active",
                    (),
                    None,
                )
            predicates = ("one_microstep_complete",)
            semantic = self._transition(predicates)
            self._phase = "pivot_evaluate"
            return self._decision(
                TubeState.PIVOT_MICRO_LOOSEN,
                self.state,
                ActionKind.COLLECT_EVIDENCE,
                {},
                "collect fresh tactile and front-RGB angle evidence",
                predicates,
                semantic,
            )

        predicates: list[str] = []
        if obs.tactile.contact_present:
            predicates.append("contact_present")
        else:
            predicates.append("contact_lost")
        if obs.tactile.slip_detected:
            predicates.append("slip_detected")
        if obs.tactile.relative_angle_target_reached:
            predicates.append("tactile_relative_angle_reached")
        if obs.front_rgb_angle_target_reached:
            predicates.append("front_rgb_calibrated_angle_reached")
        if self._pivot_steps < self.policy.pivot_max_steps:
            predicates.append("pivot_budget_remaining")
        semantic = self._transition(predicates)
        if self.state in {TubeState.RECLAMP, TubeState.RECLAMP_RECOVERY}:
            self._phase = "reclamp_commanded"
            return self._decision(
                TubeState.PIVOT_EVALUATE,
                self.state,
                ActionKind.RECLAMP,
                {"delta_m": self.policy.reclamp_step_m},
                semantic.reason,
                tuple(predicates),
                semantic,
            )
        if self.state is TubeState.PIVOT_MICRO_LOOSEN:
            self._pivot_steps += 1
            return self._decision(
                TubeState.PIVOT_EVALUATE,
                self.state,
                ActionKind.OPEN_GRIPPER_STEP,
                {"delta_m": self.policy.pivot_open_step_m},
                "continue bounded gravity pivot",
                tuple(predicates),
                semantic,
            )
        return self._hold(self.state, semantic.reason)

    def _reclamp(self, obs: ClosedLoopObservation) -> ClosedLoopDecision:
        tactile_blockers = self._tactile_blockers(obs)
        if tactile_blockers:
            return self._hold(self.state, "; ".join(tactile_blockers))
        if obs.gripper.moving:
            return self._decision(
                self.state, self.state, ActionKind.MONITOR, {}, "reclamp command active", (), None
            )
        stable = bool(
            obs.tactile.contact_present
            and obs.tactile.bilateral_contact
            and obs.tactile.rotation_stopped
        )
        predicates = ["contact_present", "reclamp_command_settled"] if stable else []
        if self.state is TubeState.RECLAMP:
            if stable:
                predicates.append("pivot_target_confirmed")
                # Freeze the post-pivot held relation before mutating the
                # controller state.  If the held pose is transiently absent,
                # stay in RECLAMP so the next fresh frame can retry; moving to
                # APPROACH_SOCKET with no relation leaves no path that can
                # ever populate it.
                blockers = self._freeze_held_relation(obs)
                if blockers:
                    return self._pause_arm_for_reacquisition("; ".join(blockers))
        elif self._pivot_steps < self.policy.pivot_max_steps:
            predicates.append("pivot_budget_remaining")
        semantic = self._transition(predicates)
        if self.state is TubeState.APPROACH_SOCKET:
            target_or_blockers = self._rack_standoff_target(obs)
            if isinstance(target_or_blockers, tuple):
                return self._pause_arm_for_reacquisition("; ".join(target_or_blockers))
            self._reset_servo(target_or_blockers)
            self._defer_state_timeout_for_blocking_move()
            return self._decision(
                semantic.state,
                self.state,
                ActionKind.MOVE_TCP,
                {"T_base_tcp_target": target_or_blockers, "plan_with_moveit": True},
                "move to rack standoff using the frozen post-pivot tube-to-TCP relation",
                tuple(predicates),
                semantic,
            )
        if self.state is TubeState.PIVOT_EVALUATE:
            return self._decision(
                semantic.state,
                self.state,
                ActionKind.COLLECT_EVIDENCE,
                {},
                semantic.reason,
                tuple(predicates),
                semantic,
            )
        if stable and self.state is TubeState.HOLD:
            return self._hold(self.state, semantic.reason)
        return self._decision(
            self.state,
            self.state,
            ActionKind.COLLECT_EVIDENCE,
            {},
            "wait for bilateral contact and stopped relative rotation",
            tuple(predicates),
            semantic,
        )

    def _approach_socket(self, obs: ClosedLoopObservation) -> ClosedLoopDecision:
        target_or_blockers = self._rack_standoff_target(obs)
        if isinstance(target_or_blockers, tuple):
            return self._pause_arm_for_reacquisition("; ".join(target_or_blockers))
        servo = self._servo(obs.robot.T_base_tcp, target_or_blockers)
        if servo.blockers:
            return self._pause_arm_for_reacquisition("; ".join(servo.blockers))
        if servo.within_deadband and obs.robot.motion_complete:
            self._settled_windows += 1
        else:
            self._settled_windows = 0
        if self._settled_windows >= self.policy.servo_settle_windows:
            predicates = ("socket_standoff_reached",)
            semantic = self._transition(predicates)
            entry = self._rack_entry_target(obs)
            if isinstance(entry, tuple):
                return self._pause_arm_for_reacquisition("; ".join(entry))
            self._reset_servo(entry)
            entry_servo = self._servo(obs.robot.T_base_tcp, entry)
            return self._decision(
                TubeState.APPROACH_SOCKET,
                self.state,
                ActionKind.SERVO_TCP,
                self._servo_parameters(entry_servo, entry, "rack_entry_alignment"),
                "servo from rack standoff to the live tube/socket relative target",
                predicates,
                semantic,
            )
        return self._decision(
            self.state,
            self.state,
            ActionKind.SERVO_TCP,
            self._servo_parameters(servo, target_or_blockers, "rack_standoff"),
            "bounded correction to the live rack standoff",
            (),
            None,
        )

    def _align_socket(self, obs: ClosedLoopObservation) -> ClosedLoopDecision:
        target_or_blockers = self._rack_entry_target(obs)
        if isinstance(target_or_blockers, tuple):
            return self._pause_arm_for_reacquisition("; ".join(target_or_blockers))
        servo = self._servo(obs.robot.T_base_tcp, target_or_blockers)
        if servo.blockers:
            return self._pause_arm_for_reacquisition("; ".join(servo.blockers))
        predicates: list[str] = []
        if obs.tactile.insertion_reference_ready:
            predicates.append("insertion_reference_ready")
        if (
            servo.within_deadband
            and obs.tactile.insertion_reference_ready
            and obs.tube_socket_axes_aligned
        ):
            predicates.append("tube_socket_axes_aligned")
        if (
            servo.within_deadband
            and obs.tactile.insertion_reference_ready
            and obs.tube_tip_at_socket_entry
        ):
            predicates.append("tube_tip_at_socket_entry")
        semantic = self._transition(predicates)
        if self.state is TubeState.INSERT_MICROSTEP:
            if (
                self._insertion_depth_commanded_m + self.policy.insertion_step_m
                > self.policy.insertion_max_travel_m + 1e-12
            ):
                return self._hold(
                    self.state,
                    "remaining bounded insertion travel is smaller than one microstep",
                )
            self._phase = "insert_commanded"
            self._insertion_depth_commanded_m += self.policy.insertion_step_m
            return self._decision(
                TubeState.ALIGN_SOCKET,
                self.state,
                ActionKind.INSERT_STEP,
                self._insertion_parameters(obs, self.policy.insertion_step_m),
                "start one bounded axial insertion step",
                tuple(predicates),
                semantic,
            )
        return self._decision(
            self.state,
            self.state,
            ActionKind.SERVO_TCP,
            self._servo_parameters(servo, target_or_blockers, "rack_entry"),
            "align tube axis and tip to the live socket frame",
            tuple(predicates),
            semantic,
        )

    def _insert(self, obs: ClosedLoopObservation) -> ClosedLoopDecision:
        tactile_blockers = self._tactile_blockers(obs)
        if tactile_blockers:
            return self._hold(self.state, "; ".join(tactile_blockers))
        if not obs.tactile.insertion_reference_ready:
            return self._hold(
                self.state,
                "same-session insertion tactile reference is unavailable or was invalidated",
            )
        if not obs.tactile.bilateral_contact:
            return self._hold(self.state, "bilateral tactile contact was lost during insertion")
        if self.state is TubeState.INSERT_MICROSTEP:
            if not obs.robot.motion_complete:
                return self._decision(
                    self.state,
                    self.state,
                    ActionKind.MONITOR,
                    {},
                    "bounded insertion step active",
                    (),
                    None,
                )
            predicates = ("one_axial_step_complete",)
            semantic = self._transition(predicates)
            return self._decision(
                TubeState.INSERT_MICROSTEP,
                self.state,
                ActionKind.COLLECT_EVIDENCE,
                {},
                "measure progress and tactile resistance after the step",
                predicates,
                semantic,
            )

        progress = obs.insertion_progress_m
        depth = obs.insertion_depth_m
        predicates: list[str] = []
        jam = bool(
            obs.tactile.insertion_jam
            and obs.tactile.insertion_resistance_high
            and progress is not None
            and progress
            < self.policy.insertion_progress_ratio_min * self.policy.insertion_step_m
        )
        if jam:
            predicates.append("insertion_jam")
            if self._jam_recoveries < self.policy.max_jam_recoveries:
                predicates.append("jam_recovery_budget_remaining")
        elif obs.tactile.insertion_seated:
            predicates.extend(("insertion_seated", "no_jam"))
        elif (
            self.policy.insertion_target_depth_m is not None
            and depth is not None
            and depth >= self.policy.insertion_target_depth_m
        ):
            predicates.extend(("insertion_depth_reached", "no_jam"))
        elif depth is not None and depth >= self.policy.insertion_max_travel_m:
            return self._hold(
                self.state,
                "maximum bounded insertion travel reached without a seated event",
            )
        elif progress is not None and progress >= (
            self.policy.insertion_progress_ratio_min * self.policy.insertion_step_m
        ):
            if (
                self._insertion_depth_commanded_m + self.policy.insertion_step_m
                > self.policy.insertion_max_travel_m + 1e-12
            ):
                return self._hold(
                    self.state,
                    "remaining bounded insertion travel is smaller than one microstep",
                )
            predicates.extend(("axial_progress_ok", "insertion_depth_remaining"))
        semantic = self._transition(predicates)
        if self.state is TubeState.JAM_RECOVERY_RETRACT:
            self._jam_recoveries += 1
            self._phase = "jam_retract_commanded"
            return self._decision(
                TubeState.INSERT_EVALUATE,
                self.state,
                ActionKind.RETRACT_INSERTION,
                self._insertion_parameters(obs, -self.policy.jam_retract_m),
                semantic.reason,
                tuple(predicates),
                semantic,
            )
        if self.state is TubeState.INSERT_MICROSTEP:
            self._insertion_depth_commanded_m += self.policy.insertion_step_m
            return self._decision(
                TubeState.INSERT_EVALUATE,
                self.state,
                ActionKind.INSERT_STEP,
                self._insertion_parameters(obs, self.policy.insertion_step_m),
                semantic.reason,
                tuple(predicates),
                semantic,
            )
        if self.state is TubeState.RELEASE:
            self._phase = "verify_support_before_open"
            return self._decision(
                TubeState.INSERT_EVALUATE,
                self.state,
                ActionKind.COLLECT_EVIDENCE,
                {"required": "tube_stays_supported"},
                "verify rack support before opening",
                tuple(predicates),
                semantic,
            )
        if self.state is TubeState.HOLD:
            return self._hold(self.state, semantic.reason)
        return self._decision(
            self.state,
            self.state,
            ActionKind.COLLECT_EVIDENCE,
            {},
            "insertion evidence is insufficient; do not push again",
            tuple(predicates),
            semantic,
        )

    def _jam_recovery(self, obs: ClosedLoopObservation) -> ClosedLoopDecision:
        if not obs.robot.motion_complete:
            return self._decision(
                self.state,
                self.state,
                ActionKind.MONITOR,
                {},
                "bounded jam retract active",
                (),
                None,
            )
        blockers = self._pose_blockers(obs.socket_pose, obs.now_s, SOCKET_POSE_CAPABILITIES)
        if blockers:
            return self._pause_arm_for_reacquisition("; ".join(blockers))
        predicates = ("retract_complete", "scene_reobserved")
        semantic = self._transition(predicates)
        return self._decision(
            TubeState.JAM_RECOVERY_RETRACT,
            self.state,
            ActionKind.COLLECT_EVIDENCE,
            {},
            "re-ground the socket entry after the bounded retract",
            predicates,
            semantic,
        )

    def _release(self, obs: ClosedLoopObservation) -> ClosedLoopDecision:
        tactile_blockers = self._tactile_blockers(obs)
        if tactile_blockers:
            return self._hold(self.state, "; ".join(tactile_blockers))
        if self._phase == "verify_support_before_open":
            if not obs.tube_stays_supported:
                return self._decision(
                    self.state,
                    self.state,
                    ActionKind.COLLECT_EVIDENCE,
                    {"required": "tube_stays_supported"},
                    "rack support is not yet verified",
                    (),
                    None,
                )
            self._phase = "release_opening"
            return self._decision(
                self.state,
                self.state,
                ActionKind.OPEN_RELEASE,
                {},
                "support is verified; open the Hand-E",
                (),
                None,
            )
        predicates: list[str] = []
        if obs.gripper.open_confirmed:
            predicates.append("gripper_open")
        if not obs.tactile.contact_present:
            predicates.append("contact_released")
        if obs.tube_stays_supported:
            predicates.append("tube_stays_supported")
        semantic = self._transition(predicates)
        if self.state is TubeState.RETREAT:
            self._retreat_target = obs.robot.T_base_tcp.copy()
            self._retreat_target[:3, 3] += [0.0, 0.0, self.policy.retreat_m]
            self._reset_servo(self._retreat_target)
            self._defer_state_timeout_for_blocking_move()
            return self._decision(
                TubeState.RELEASE,
                self.state,
                ActionKind.MOVE_TCP,
                {"T_base_tcp_target": self._retreat_target, "plan_with_moveit": True},
                "release is verified; retreat vertically clear",
                tuple(predicates),
                semantic,
            )
        return self._decision(
            self.state,
            self.state,
            ActionKind.COLLECT_EVIDENCE,
            {},
            "verify open, tactile release, and supported tube",
            tuple(predicates),
            semantic,
        )

    def _retreat(self, obs: ClosedLoopObservation) -> ClosedLoopDecision:
        if self._retreat_target is None:
            return self._hold(self.state, "retreat target was not frozen")
        servo = self._servo(obs.robot.T_base_tcp, self._retreat_target)
        predicates = (
            ("retreat_complete",)
            if servo.within_deadband and obs.robot.motion_complete
            else ()
        )
        semantic = self._transition(predicates)
        if self.state is TubeState.COMPLETE:
            return self._decision(
                TubeState.RETREAT,
                self.state,
                ActionKind.NONE,
                {},
                "task completed",
                predicates,
                semantic,
            )
        return self._decision(
            self.state,
            self.state,
            ActionKind.SERVO_TCP,
            self._servo_parameters(servo, self._retreat_target, "retreat"),
            "bounded retreat to clearance",
            predicates,
            semantic,
        )

    def _transition(self, predicates: Sequence[str]) -> TransitionDecision:
        previous = self.state
        decision = decide_test_tube_transition(previous, predicates)
        if decision.next_state is not previous:
            self.state = decision.next_state
            self._state_started_s = self._last_now_s
            self._settled_windows = 0
            self._initial_target = None
            self._previous_target = None
        return decision

    def _pose_blockers(
        self,
        estimate: ObjectPoseEstimate | None,
        now_s: float,
        required: frozenset[PoseCapability],
        *,
        max_age_s: float | None = None,
    ) -> tuple[str, ...]:
        if estimate is None:
            return ("required_live_pose_missing",)
        blockers = estimate.blockers(
            now_s=now_s,
            required=required,
            max_age_s=(self.policy.pose_max_age_s if max_age_s is None else max_age_s),
            max_position_std_m=self.policy.pose_max_position_std_m,
            max_orientation_std_deg=self.policy.pose_max_orientation_std_deg,
            max_inference_latency_s=self.policy.pose_max_inference_latency_s,
            min_mask_iou=self.policy.pose_min_mask_iou,
            min_valid_depth_count=self.policy.pose_min_valid_depth_count,
            min_foreground_depth_support_count=(
                self.policy.pose_min_foreground_depth_support_count
            ),
            min_foreground_depth_support_fraction=(
                self.policy.pose_min_foreground_depth_support_fraction
            ),
            max_background_depth_fraction=self.policy.pose_max_background_depth_fraction,
        )
        normalized = self.geometry.normalized_demo_grasp
        if (
            estimate.object_id == "tube"
            and PoseCapability.SOURCE_TUBE in required
            and normalized is not None
            and normalized.tube_length_m is None
            and estimate.long_axis_length_m is None
        ):
            blockers += ("runtime_f2g_tube_length_missing",)
        expected_tracking_id = self._tracking_ids.get(estimate.object_id)
        if expected_tracking_id is not None and estimate.tracking_id != expected_tracking_id:
            blockers += (f"{estimate.object_id}_tracking_id_changed",)
        elif not blockers:
            self._tracking_ids[estimate.object_id] = estimate.tracking_id
        return blockers

    def _tactile_blockers(self, obs: ClosedLoopObservation) -> tuple[str, ...]:
        blockers: list[str] = []
        if not obs.tactile.valid:
            blockers.append("tactile_evidence_invalid")
        age = obs.now_s - obs.tactile.observed_at_s
        if age < -1e-3 or age > self.policy.tactile_max_age_s:
            blockers.append("tactile_evidence_stale")
        return tuple(blockers)

    def _live_grasp_target(
        self, obs: ClosedLoopObservation, *, source: bool
    ) -> np.ndarray | tuple[str, ...]:
        required = SOURCE_POSE_CAPABILITIES if source else HELD_POSE_CAPABILITIES
        blockers = self._pose_blockers(obs.tube_pose, obs.now_s, required)
        if blockers:
            occlusion_blockers = {
                "tube_occluded_by_gripper",
                "tube_mask_iou_too_low",
                "tube_valid_depth_count_too_low",
                "tube_foreground_depth_support_too_low",
                "tube_foreground_depth_fraction_too_low",
                "tube_background_depth_dominates_mask",
                "tube_masked_depth_cluster_not_qualified",
            }
            explicit_occlusion = bool(
                obs.tube_pose is not None and obs.tube_pose.occluded_by_gripper
            )
            missing_after_last_qualified = obs.tube_pose is None
            # Closing creates the first contact. Require the tactile observer
            # to be ready, not contact to exist before closing is permitted.
            tactile_supervisor_ready = not self._tactile_blockers(obs)
            cached_pose_fresh = bool(
                self._last_qualified_grasp_observed_at_s is not None
                and -1e-3 <= obs.now_s - self._last_qualified_grasp_observed_at_s
                <= self.policy.pose_max_age_s
            )
            if (
                source
                and self.state is TubeState.SIDE_GRASP
                and (explicit_occlusion or missing_after_last_qualified)
                and self._last_qualified_grasp_target is not None
                and tactile_supervisor_ready
                and cached_pose_fresh
                and (
                    obs.robot.motion_complete
                    or self.policy.occluded_grasp_close_distance_m is not None
                )
                and (
                    missing_after_last_qualified
                    or set(blockers) <= occlusion_blockers
                )
            ):
                if self._within_occluded_close_window(
                    obs.robot.T_base_tcp, self._last_qualified_grasp_target
                ):
                    self._vision_frozen_for_close = True
                    return self._last_qualified_grasp_target.copy()
                return (*blockers, "gripper_occlusion_before_tactile_switch_window")
            return blockers
        assert obs.tube_pose is not None
        target = grounded_grasp_target(obs.tube_pose, self.geometry)
        if source:
            target = target.copy()
            target[:3, 3] += target[:3, :3] @ np.asarray(
                [self._grasp_recenter_local_x_m, 0.0, 0.0]
            )
            self._last_qualified_grasp_target = target.copy()
            self._last_qualified_grasp_observed_at_s = (
                obs.tube_pose.observed_at_s
            )
            self._vision_frozen_for_close = False
        return target

    def _within_occluded_close_window(self, measured: np.ndarray, target: np.ndarray) -> bool:
        translation_m, rotation_deg = pose_error(measured, target)
        distance_m = self.policy.occluded_grasp_close_distance_m
        if distance_m is None:
            distance_m = min(
                self.policy.occlusion_switch_translation_m,
                self.policy.servo_translation_deadband_m,
            )
        return bool(
            translation_m <= distance_m
            and rotation_deg <= min(
                self.policy.occlusion_switch_rotation_deg,
                self.policy.servo_rotation_deadband_deg,
            )
        )

    def _freeze_held_relation(self, obs: ClosedLoopObservation) -> tuple[str, ...]:
        blockers = self._pose_blockers(obs.tube_pose, obs.now_s, HELD_POSE_CAPABILITIES)
        if blockers:
            return blockers
        assert obs.tube_pose is not None
        self._T_tube_tcp_held = (
            invert_transforms(obs.tube_pose.T_base_object) @ obs.robot.T_base_tcp
        )
        validate_rigid_transform(self._T_tube_tcp_held, name="frozen held relation")
        return ()

    def _held_level_target(
        self,
        obs: ClosedLoopObservation,
    ) -> np.ndarray | tuple[str, ...]:
        blockers = self._pose_blockers(
            obs.tube_pose,
            obs.now_s,
            HELD_POSE_CAPABILITIES,
        )
        if blockers:
            return blockers
        assert obs.tube_pose is not None
        try:
            return held_tube_level_target(
                T_base_tube=obs.tube_pose.T_base_object,
                T_base_tcp_measured=obs.robot.T_base_tcp,
                tube_axis_tube=self.geometry.tube_axis_tube,
                min_horizontal_projection=(
                    self.policy.level_min_horizontal_projection
                ),
            )
        except ClosedLoopContractError as exc:
            return (str(exc),)

    def _rack_entry_target(
        self, obs: ClosedLoopObservation
    ) -> np.ndarray | tuple[str, ...]:
        blockers = self._pose_blockers(obs.socket_pose, obs.now_s, SOCKET_POSE_CAPABILITIES)
        blockers += self._pose_blockers(obs.tube_pose, obs.now_s, HELD_POSE_CAPABILITIES)
        if blockers:
            return blockers
        if self._T_tube_tcp_held is None:
            return ("post_pivot_T_tube_tcp_not_frozen",)
        assert obs.socket_pose is not None
        return socket_tcp_target(obs.socket_pose, self.geometry, self._T_tube_tcp_held)

    def _rack_standoff_target(
        self, obs: ClosedLoopObservation
    ) -> np.ndarray | tuple[str, ...]:
        entry = self._rack_entry_target(obs)
        if isinstance(entry, tuple):
            return entry
        assert obs.socket_pose is not None
        outward_base = (
            obs.socket_pose.T_base_object[:3, :3]
            @ self.geometry.socket_outward_axis_socket
        )
        target = entry.copy()
        target[:3, 3] += outward_base * self.policy.rack_standoff_m
        return target

    def _servo(self, measured: np.ndarray, target: np.ndarray) -> ServoResult:
        if self._initial_target is None:
            self._initial_target = target.copy()
        result = bounded_visual_servo(
            T_base_tcp_measured=measured,
            T_base_tcp_target=target,
            T_base_tcp_initial_target=self._initial_target,
            previous_target=self._previous_target,
            policy=self.policy,
        )
        self._previous_target = target.copy()
        return result

    def _reset_servo(self, target: np.ndarray) -> None:
        self._initial_target = target.copy()
        self._previous_target = target.copy()
        self._settled_windows = 0

    def _defer_state_timeout_for_blocking_move(self) -> None:
        """Start the entered state's deadline after blocking MoveIt returns.

        ``MOVE_TCP`` is synchronous in the resident adapter and may legitimately
        take longer than the next state's evidence timeout.  The first
        post-dispatch observation initializes the deadline, so successful slow
        free-space motion cannot make that observation immediately time out.
        """

        self._state_started_s = None

    def _servo_parameters(
        self, servo: ServoResult, final_target: np.ndarray, purpose: str
    ) -> dict[str, Any]:
        return {
            "T_base_tcp_command": servo.command_target,
            "T_base_tcp_live_target": final_target,
            "translation_error_m": servo.translation_error_m,
            "rotation_error_deg": servo.rotation_error_deg,
            "purpose": purpose,
            "bounded": True,
        }

    def _insertion_parameters(
        self, obs: ClosedLoopObservation, distance_m: float
    ) -> dict[str, Any]:
        if obs.socket_pose is None:
            return {"invalid": "socket pose missing"}
        inward_base = -(
            obs.socket_pose.T_base_object[:3, :3]
            @ self.geometry.socket_outward_axis_socket
        )
        return {
            "distance_m": float(distance_m),
            "axis_base": inward_base,
            "require_tactile_jam_monitor": True,
        }

    def _decision(
        self,
        previous: TubeState,
        next_state: TubeState,
        kind: ActionKind,
        parameters: Mapping[str, Any],
        reason: str,
        predicates: Sequence[str],
        semantic: TransitionDecision | None,
    ) -> ClosedLoopDecision:
        self._action_sequence += 1
        action = ClosedLoopAction(
            action_id=f"action-{self._action_sequence:04d}",
            kind=kind,
            state=next_state,
            parameters=parameters,
            reason=reason,
        )
        return ClosedLoopDecision(
            previous_state=previous,
            next_state=next_state,
            action=action,
            predicates=tuple(sorted(set(predicates))),
            semantic_transition=semantic,
            terminal=next_state in {TubeState.COMPLETE, TubeState.HOLD},
        )

    def _hold(self, previous: TubeState, reason: str) -> ClosedLoopDecision:
        self.state = TubeState.HOLD
        self._state_started_s = self._last_now_s
        return self._decision(
            previous,
            self.state,
            ActionKind.STOP_AND_HOLD,
            {
                "cancel_active_arm_goal": True,
                "stop_active_gripper_command": True,
                "cease_dispatch": True,
            },
            reason,
            ("invalid_evidence",),
            None,
        )

    def _wait_for_evidence(self, reason: str) -> ClosedLoopDecision:
        """Wait before motion while a transient producer warms up."""

        return self._decision(
            self.state,
            self.state,
            ActionKind.COLLECT_EVIDENCE,
            {"reacquire": True},
            f"wait for fresh qualified evidence: {reason}",
            ("evidence_pending",),
            None,
        )

    def _pause_arm_for_reacquisition(self, reason: str) -> ClosedLoopDecision:
        """Zero active arm motion and retry perception in the same state.

        The state deadline is the finite retry budget. Robot/gripper faults,
        tactile faults, and exhausted task recoveries still enter terminal
        ``STOP_AND_HOLD``.
        """

        self._settled_windows = 0
        return self._decision(
            self.state,
            self.state,
            ActionKind.PAUSE_ARM,
            {"reacquire": True, "zero_active_servo": True},
            f"pause arm and reacquire qualified evidence: {reason}",
            ("evidence_temporarily_invalid",),
            None,
        )


def parse_backend_descriptor(value: object) -> PoseBackendDescriptor:
    payload = _mapping(value, name="pose backend descriptor")
    raw_capabilities = payload.get("capabilities")
    if not isinstance(raw_capabilities, Sequence) or isinstance(
        raw_capabilities, (str, bytes)
    ):
        raise ClosedLoopContractError("pose backend capabilities must be an array")
    return PoseBackendDescriptor(
        name=_identifier(payload.get("name"), name="pose backend name"),
        available=_boolean(payload.get("available"), name="pose backend available"),
        runtime_qualified=_boolean(
            payload.get("runtime_qualified"), name="pose backend runtime_qualified"
        ),
        qualification_id=payload.get("qualification_id"),
        capabilities=frozenset(PoseCapability(value) for value in raw_capabilities),
        detail=str(payload.get("detail", "")),
    )


def parse_pose_estimate(value: object, *, name: str) -> ObjectPoseEstimate:
    payload = _mapping(value, name=name)
    return ObjectPoseEstimate(
        object_id=_identifier(payload.get("object_id"), name=f"{name}.object_id"),
        T_base_object=np.asarray(payload.get("T_base_object"), dtype=float),
        observed_at_s=_finite(payload.get("observed_at_s"), name=f"{name}.observed_at_s"),
        position_std_m=_nonnegative(
            payload.get("position_std_m"), name=f"{name}.position_std_m"
        ),
        orientation_std_deg=_nonnegative(
            payload.get("orientation_std_deg"), name=f"{name}.orientation_std_deg"
        ),
        backend=parse_backend_descriptor(payload.get("backend")),
        tracking_id=_identifier(payload.get("tracking_id"), name=f"{name}.tracking_id"),
        inference_latency_s=_nonnegative(
            payload.get("inference_latency_s"), name=f"{name}.inference_latency_s"
        ),
        mask_id=_identifier(payload.get("mask_id"), name=f"{name}.mask_id"),
        mask_iou=_nonnegative(payload.get("mask_iou"), name=f"{name}.mask_iou"),
        valid_depth_count=_nonnegative_integer(
            payload.get("valid_depth_count"), name=f"{name}.valid_depth_count"
        ),
        foreground_depth_support_count=_nonnegative_integer(
            payload.get("foreground_depth_support_count"),
            name=f"{name}.foreground_depth_support_count",
        ),
        foreground_depth_support_fraction=_nonnegative(
            payload.get("foreground_depth_support_fraction"),
            name=f"{name}.foreground_depth_support_fraction",
        ),
        background_depth_fraction=_nonnegative(
            payload.get("background_depth_fraction"),
            name=f"{name}.background_depth_fraction",
        ),
        masked_depth_cluster_qualified=_boolean(
            payload.get("masked_depth_cluster_qualified"),
            name=f"{name}.masked_depth_cluster_qualified",
        ),
        requires_rgbd_quality_gates=_boolean(
            payload.get("requires_rgbd_quality_gates", True),
            name=f"{name}.requires_rgbd_quality_gates",
        ),
        axis_sign_source=str(payload.get("axis_sign_source", "not_applicable")),
        cylindrical_roll_canonicalized=_boolean(
            payload.get("cylindrical_roll_canonicalized", False),
            name=f"{name}.cylindrical_roll_canonicalized",
        ),
        occluded_by_gripper=_boolean(
            payload.get("occluded_by_gripper", False),
            name=f"{name}.occluded_by_gripper",
        ),
        size_m=(
            None
            if payload.get("size_m") is None
            else tuple(payload.get("size_m"))
        ),
    )


def parse_observation(value: object) -> ClosedLoopObservation:
    payload = _mapping(value, name="observation")
    robot = _mapping(payload.get("robot"), name="observation.robot")
    gripper = _mapping(payload.get("gripper"), name="observation.gripper")
    tactile = _mapping(payload.get("tactile"), name="observation.tactile")
    return ClosedLoopObservation(
        observation_id=_identifier(
            payload.get("observation_id"), name="observation.observation_id"
        ),
        now_s=_finite(payload.get("now_s"), name="observation.now_s"),
        source_depth_only_loss=_boolean(
            payload.get("source_depth_only_loss", False), name="source_depth_only_loss"
        ),
        source_rgb_identity_retained=_boolean(
            payload.get("source_rgb_identity_retained", False),
            name="source_rgb_identity_retained",
        ),
        robot=RobotFeedback(
            T_base_tcp=np.asarray(robot.get("T_base_tcp"), dtype=float),
            motion_complete=_boolean(
                robot.get("motion_complete", False), name="robot.motion_complete"
            ),
            bounded_tcp_servo_supported=_boolean(
                robot.get("bounded_tcp_servo_supported", False),
                name="robot.bounded_tcp_servo_supported",
            ),
            fault=robot.get("fault"),
        ),
        gripper=GripperFeedback(
            width_m=_nonnegative(gripper.get("width_m"), name="gripper.width_m"),
            moving=_boolean(gripper.get("moving"), name="gripper.moving"),
            fault_free=_boolean(gripper.get("fault_free"), name="gripper.fault_free"),
            object_detected=_boolean(
                gripper.get("object_detected", False), name="gripper.object_detected"
            ),
            open_confirmed=_boolean(
                gripper.get("open_confirmed", False), name="gripper.open_confirmed"
            ),
        ),
        tactile=TactileFeedback(
            valid=_boolean(tactile.get("valid"), name="tactile.valid"),
            observed_at_s=_finite(
                tactile.get("observed_at_s"), name="tactile.observed_at_s"
            ),
            contact_present=_boolean(
                tactile.get("contact_present", False), name="tactile.contact_present"
            ),
            bilateral_contact=_boolean(
                tactile.get("bilateral_contact", False), name="tactile.bilateral_contact"
            ),
            bilateral_contact_interior=_boolean(
                tactile.get("bilateral_contact_interior", False),
                name="tactile.bilateral_contact_interior",
            ),
            single_contact_side=tactile.get("single_contact_side"),
            slip_detected=_boolean(
                tactile.get("slip_detected", False), name="tactile.slip_detected"
            ),
            relative_angle_target_reached=_boolean(
                tactile.get("relative_angle_target_reached", False),
                name="tactile.relative_angle_target_reached",
            ),
            rotation_stopped=_boolean(
                tactile.get("rotation_stopped", False), name="tactile.rotation_stopped"
            ),
            insertion_jam=_boolean(
                tactile.get("insertion_jam", False), name="tactile.insertion_jam"
            ),
            insertion_resistance_high=_boolean(
                tactile.get("insertion_resistance_high", False),
                name="tactile.insertion_resistance_high",
            ),
            insertion_seated=_boolean(
                tactile.get("insertion_seated", False),
                name="tactile.insertion_seated",
            ),
            insertion_reference_ready=_boolean(
                tactile.get("insertion_reference_ready", True),
                name="tactile.insertion_reference_ready",
            ),
        ),
        tube_pose=(
            None
            if payload.get("tube_pose") is None
            else parse_pose_estimate(payload.get("tube_pose"), name="tube_pose")
        ),
        socket_pose=(
            None
            if payload.get("socket_pose") is None
            else parse_pose_estimate(payload.get("socket_pose"), name="socket_pose")
        ),
        front_rgb_angle_target_reached=_boolean(
            payload.get("front_rgb_angle_target_reached", False),
            name="front_rgb_angle_target_reached",
        ),
        tube_clear_of_source=_boolean(
            payload.get("tube_clear_of_source", False), name="tube_clear_of_source"
        ),
        tube_tip_at_socket_entry=_boolean(
            payload.get("tube_tip_at_socket_entry", False),
            name="tube_tip_at_socket_entry",
        ),
        tube_socket_axes_aligned=_boolean(
            payload.get("tube_socket_axes_aligned", False),
            name="tube_socket_axes_aligned",
        ),
        insertion_progress_m=_optional_nonnegative(
            payload.get("insertion_progress_m"), name="insertion_progress_m"
        ),
        insertion_depth_m=_optional_nonnegative(
            payload.get("insertion_depth_m"), name="insertion_depth_m"
        ),
        tube_stays_supported=_boolean(
            payload.get("tube_stays_supported", False), name="tube_stays_supported"
        ),
    )


def parse_geometry(value: object) -> TestTubeClosedLoopGeometry:
    payload = _mapping(value, name="geometry")
    normalized_payload = payload.get("normalized_demo_grasp")
    normalized = None
    if normalized_payload is not None:
        normalized_mapping = _mapping(
            normalized_payload, name="geometry.normalized_demo_grasp"
        )
        normalized = NormalizedDemoGraspRelation(
            axial_fraction_closed_to_open=normalized_mapping.get(
                "axial_fraction_closed_to_open"
            ),
            tube_length_m=normalized_mapping.get("tube_length_m"),
            tube_length_source=_identifier(
                normalized_mapping.get("tube_length_source"),
                name="geometry.normalized_demo_grasp.tube_length_source",
            ),
            preferred_jaw_axis_base=np.asarray(
                normalized_mapping.get("preferred_jaw_axis_base"), dtype=float
            ),
            top_approach_direction_base=np.asarray(
                normalized_mapping.get(
                    "top_approach_direction_base", [0.0, 0.0, -1.0]
                ),
                dtype=float,
            ),
            tube_pose_origin=str(
                normalized_mapping.get("tube_pose_origin", "geometric_centre")
            ),
            tube_axis_contract=str(
                normalized_mapping.get("tube_axis_contract", "closed_to_open")
            ),
        )
    legacy_transform = payload.get("T_tube_tcp_demo_grasp")
    return TestTubeClosedLoopGeometry(
        relation_id=_identifier(payload.get("relation_id"), name="geometry.relation_id"),
        T_tube_tcp_demo_grasp=(
            None
            if legacy_transform is None
            else np.asarray(legacy_transform, dtype=float)
        ),
        T_socket_tube_entry=np.asarray(payload.get("T_socket_tube_entry"), dtype=float),
        normalized_demo_grasp=normalized,
        relation_source=_identifier(
            payload.get("relation_source", "unspecified"),
            name="geometry.relation_source",
        ),
        relation_runtime_qualified=_boolean(
            payload.get("relation_runtime_qualified", False),
            name="geometry.relation_runtime_qualified",
        ),
        socket_entry_source=_identifier(
            payload.get("socket_entry_source", "unspecified"),
            name="geometry.socket_entry_source",
        ),
        socket_entry_runtime_qualified=_boolean(
            payload.get("socket_entry_runtime_qualified", False),
            name="geometry.socket_entry_runtime_qualified",
        ),
        tube_axis_tube=np.asarray(payload.get("tube_axis_tube", [0, 0, 1]), dtype=float),
        socket_outward_axis_socket=np.asarray(
            payload.get("socket_outward_axis_socket", [0, 0, 1]), dtype=float
        ),
    )


def parse_policy(value: object | None) -> ClosedLoopPolicy:
    if value is None:
        return ClosedLoopPolicy()
    payload = _mapping(value, name="policy")
    known = set(ClosedLoopPolicy.__dataclass_fields__)
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ClosedLoopContractError(f"unknown policy fields: {unknown}")
    return ClosedLoopPolicy(**payload)


def state_contract() -> list[dict[str, Any]]:
    """Reviewable summary used by the default dry-run report."""

    return [
        {"state": "OBSERVE", "predicate": "fresh qualified tube+socket 6D poses"},
        {"state": "APPROACH_TUBE", "action": "MoveIt top standoff + bounded visual servo"},
        {
            "state": "SIDE_GRASP",
            "action": (
                "live T_base_tube @ demo T_tube_tcp; one continuous close; "
                "two stationary centred optical-contact windows; bounded "
                "5 mm deeper retry (maximum 2/10 mm) when shallow"
            ),
        },
        {
            "state": "EXTRACT",
            "predicate": "fresh retained centred bilateral optical contact while lifting",
        },
        {
            "state": "LEVEL_HELD_TUBE",
            "predicate": (
                "live held-axis horizontal and measured TCP target converged for "
                "debounced windows"
            ),
            "action": (
                "keep grasp rigid and servo wrist before the first micro-loosen; "
                "on a measured progress plateau, apply one configured base-Z lift "
                "through a "
                "collision-planned redundant posture while keeping the grasp closed"
            ),
        },
        {
            "state": "PIVOT_MICRO_LOOSEN/PIVOT_EVALUATE",
            "predicate": "contact+angle; early reclamp on slip",
        },
        {"state": "RECLAMP", "action": "freeze measured post-pivot T_tube_tcp"},
        {"state": "APPROACH_SOCKET", "action": "MoveIt rack standoff from live socket pose"},
        {"state": "ALIGN_SOCKET", "action": "bounded live tube/socket servo"},
        {
            "state": "INSERT_*",
            "predicate": "progress+calibrated tactile jam; bounded retract/re-align",
        },
        {"state": "RELEASE", "predicate": "support before open, then contact released"},
        {"state": "RETREAT", "action": "collision-checked clearance retreat"},
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ClosedLoopContractError(f"{name} must be an object")
    return value


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClosedLoopContractError(f"{name} must be a non-empty string")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ClosedLoopContractError(f"{name} must be boolean")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ClosedLoopContractError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ClosedLoopContractError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ClosedLoopContractError(f"{name} must be finite")
    return result


def _positive(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ClosedLoopContractError(f"{name} must be positive")
    return result


def _nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        raise ClosedLoopContractError(f"{name} must be non-negative")
    return result


def _optional_nonnegative(value: object, *, name: str) -> float | None:
    return None if value is None else _nonnegative(value, name=name)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ClosedLoopContractError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ClosedLoopContractError(f"{name} must be a non-negative integer")
    return value


def _unit(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ClosedLoopContractError(f"{name} must contain three finite values")
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise ClosedLoopContractError(f"{name} must be non-zero")
    return result / norm


__all__ = [
    "ActionKind",
    "CLOSED_LOOP_CONFIG_SCHEMA",
    "CLOSED_LOOP_REPLAY_SCHEMA",
    "CLOSED_LOOP_REPORT_SCHEMA",
    "ClosedLoopAction",
    "ClosedLoopContractError",
    "ClosedLoopDecision",
    "ClosedLoopIO",
    "ClosedLoopObservation",
    "ClosedLoopPolicy",
    "GripperFeedback",
    "HELD_POSE_CAPABILITIES",
    "NormalizedDemoGraspRelation",
    "ObjectPoseEstimate",
    "PoseBackendDescriptor",
    "PoseCapability",
    "RobotFeedback",
    "SOCKET_POSE_CAPABILITIES",
    "SOURCE_POSE_CAPABILITIES",
    "ServoResult",
    "TactileFeedback",
    "TestTubeClosedLoopController",
    "TestTubeClosedLoopGeometry",
    "bounded_visual_servo",
    "canonicalize_cylindrical_tube_pose",
    "grounded_grasp_target",
    "normalized_top_down_grasp_target",
    "parse_backend_descriptor",
    "parse_geometry",
    "parse_observation",
    "parse_policy",
    "pose_error",
    "socket_tcp_target",
    "state_contract",
    "top_pregrasp_target",
]
