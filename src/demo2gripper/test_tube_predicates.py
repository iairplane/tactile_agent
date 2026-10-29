"""Fail-closed predicate construction for the test-tube runtime.

This module is the hardware-independent boundary between typed producer
receipts and :func:`demo2gripper.test_tube_preflight.decide_test_tube_transition`.
It owns no camera, tactile device, gripper, robot, ROS node, or command path.

Raw or shadow observations are deliberately not state-machine predicates.
Every promoted predicate is reconstructed from fresh, session-bound receipts;
the returned audit receipt records exactly which producer receipts were used.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from .test_tube_preflight import TubeState
from .test_tube_runtime import (
    TEST_TUBE_PREDICATE_RECEIPT_SCHEMA,
    PredicateSnapshot,
)

if TYPE_CHECKING:
    from .test_tube_tactile_controller import (
        TestTubeTactileControllerProfile,
        TestTubeTactileEvaluation,
    )


class ReceiptValidationError(ValueError):
    """A producer receipt violates the predicate-builder contract."""


class MotionEvent(StrEnum):
    APPROACH_POSE_REACHED = "approach_pose_reached"
    LEVEL_TARGET_REACHED = "level_target_reached"
    PIVOT_MICROSTEP_COMPLETE = "pivot_microstep_complete"
    RECLAMP_COMMAND_SETTLED = "reclamp_command_settled"
    SOCKET_STANDOFF_REACHED = "socket_standoff_reached"
    AXIAL_STEP_COMPLETE = "axial_step_complete"
    RETRACT_COMPLETE = "retract_complete"
    RETREAT_COMPLETE = "retreat_complete"


class ContactStatus(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    LOST_AFTER_PRESENT = "lost_after_present"
    RELEASED_AFTER_OPEN = "released_after_open"
    INVALID = "invalid"


class TrackerStatus(StrEnum):
    TRACKED = "tracked"
    HOLD = "hold"
    RESEED_REQUIRED = "reseed_required"


class JamAssessment(StrEnum):
    CLEAR = "clear"
    JAM = "jam"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ReceiptHeader:
    """Identity, clock, and common task provenance for one producer receipt."""

    task_id: str
    session_id: str
    provenance_id: str
    receipt_id: str
    producer_id: str
    observed_at_s: float
    valid: bool = True

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "session_id",
            "provenance_id",
            "receipt_id",
            "producer_id",
        ):
            _identifier(getattr(self, name), name=name)
        _finite(self.observed_at_s, name="observed_at_s", minimum=0.0)
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be boolean")


@dataclass(frozen=True)
class SceneReceipt:
    header: ReceiptHeader
    tube_pose_valid: bool = False
    socket_frame_valid: bool = False
    tube_clear_of_source: bool = False
    tube_horizontal: bool = False
    tube_socket_axes_aligned: bool = False
    tube_tip_at_socket_entry: bool = False
    scene_reobserved: bool = False

    def __post_init__(self) -> None:
        _receipt_header(self.header)
        _booleans(
            self,
            "tube_pose_valid",
            "socket_frame_valid",
            "tube_clear_of_source",
            "tube_horizontal",
            "tube_socket_axes_aligned",
            "tube_tip_at_socket_entry",
            "scene_reobserved",
        )


@dataclass(frozen=True)
class MotionReceipt:
    header: ReceiptHeader
    event: MotionEvent
    command_terminal: bool
    measured_target_within_tolerance: bool
    success: bool

    def __post_init__(self) -> None:
        _receipt_header(self.header)
        if not isinstance(self.event, MotionEvent):
            raise TypeError("event must be MotionEvent")
        _booleans(self, "command_terminal", "measured_target_within_tolerance", "success")


@dataclass(frozen=True)
class GripperReceipt:
    header: ReceiptHeader
    measured_width_m: float
    command_settled: bool
    fault_free: bool
    open_confirmed: bool = False

    def __post_init__(self) -> None:
        _receipt_header(self.header)
        _finite(self.measured_width_m, name="measured_width_m", minimum=0.0)
        _booleans(self, "command_settled", "fault_free", "open_confirmed")


@dataclass(frozen=True)
class TactileAxisReceipt:
    """Session-bound result of one pure tactile-controller evaluation.

    This receipt is nested in :class:`TactileContactReceipt`, so existing
    contact producers and the runtime's single ``tactile`` producer boundary
    remain compatible.  Candidate observations are retained for audit, while
    the two ``*_confirmed`` fields can only be true after the controller's
    configured consecutive-window count has been reached.
    """

    profile_id: str
    session_id: str
    provenance_id: str
    source_evidence_id: str
    evaluation_valid: bool
    axis_tcp: tuple[float, float, float] | None
    axis_base: tuple[float, float, float] | None
    target_error_deg: float | None
    target_tolerance_deg: float
    target_candidate: bool
    target_confirmed: bool
    target_confirmation_count: int
    target_confirmation_required: int
    runtime_axis_qualified: bool
    slip_candidate: bool
    slip_confirmed: bool
    slip_confirmation_count: int
    slip_confirmation_required: int
    runtime_slip_qualified: bool

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "session_id",
            "provenance_id",
            "source_evidence_id",
        ):
            _identifier(getattr(self, name), name=name)
        _booleans(
            self,
            "evaluation_valid",
            "target_candidate",
            "target_confirmed",
            "runtime_axis_qualified",
            "slip_candidate",
            "slip_confirmed",
            "runtime_slip_qualified",
        )
        for name in ("axis_tcp", "axis_base"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _unit_axis_tuple(value, name=name))
        if self.target_error_deg is not None:
            _finite(self.target_error_deg, name="target_error_deg", minimum=0.0)
        _finite(self.target_tolerance_deg, name="target_tolerance_deg", minimum=0.0)
        if self.target_tolerance_deg <= 0.0:
            raise ValueError("target_tolerance_deg must be positive")
        for name in (
            "target_confirmation_count",
            "target_confirmation_required",
            "slip_confirmation_count",
            "slip_confirmation_required",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.target_confirmation_required < 1:
            raise ValueError("target_confirmation_required must be positive")
        if self.slip_confirmation_required < 1:
            raise ValueError("slip_confirmation_required must be positive")
        if self.target_candidate and self.target_error_deg is None:
            raise ValueError("target_candidate requires target_error_deg")
        if (
            self.target_candidate
            and self.target_error_deg is not None
            and self.target_error_deg > self.target_tolerance_deg
        ):
            raise ValueError("target_candidate exceeds target_tolerance_deg")
        if self.target_confirmed and not (
            self.evaluation_valid
            and self.runtime_axis_qualified
            and self.target_candidate
            and self.axis_tcp is not None
            and self.axis_base is not None
            and self.target_confirmation_count >= self.target_confirmation_required
        ):
            raise ValueError(
                "target_confirmed requires a valid, qualified, consecutively confirmed axis"
            )
        if self.slip_confirmed and not (
            self.evaluation_valid
            and self.runtime_slip_qualified
            and self.slip_candidate
            and self.slip_confirmation_count >= self.slip_confirmation_required
        ):
            raise ValueError(
                "slip_confirmed requires a valid, qualified, consecutively confirmed slip"
            )


@dataclass(frozen=True)
class TactileContactReceipt:
    header: ReceiptHeader
    contact_profile_id: str
    status: ContactStatus
    runtime_contact_qualified: bool
    bilateral_contact: bool = False
    slip_shadow_candidate: bool = False
    axis_evaluation: TactileAxisReceipt | None = None

    def __post_init__(self) -> None:
        _receipt_header(self.header)
        _identifier(self.contact_profile_id, name="contact_profile_id")
        if not isinstance(self.status, ContactStatus):
            raise TypeError("status must be ContactStatus")
        _booleans(
            self,
            "runtime_contact_qualified",
            "bilateral_contact",
            "slip_shadow_candidate",
        )
        if self.bilateral_contact and self.status is not ContactStatus.PRESENT:
            raise ValueError("bilateral_contact requires PRESENT status")
        if self.axis_evaluation is not None:
            if not isinstance(self.axis_evaluation, TactileAxisReceipt):
                raise TypeError("axis_evaluation must be TactileAxisReceipt or None")
            if self.axis_evaluation.session_id != self.header.session_id:
                raise ValueError("axis_evaluation belongs to a different session")
            if self.axis_evaluation.provenance_id != self.header.provenance_id:
                raise ValueError("axis_evaluation provenance does not match the contact receipt")


@dataclass(frozen=True)
class RgbAxisReceipt:
    header: ReceiptHeader
    qualification_id: str
    tracker_status: TrackerStatus
    runtime_qualified: bool
    target_angle_reached: bool = False
    pivot_target_confirmed: bool = False
    tube_between_fingers: bool = False

    def __post_init__(self) -> None:
        _receipt_header(self.header)
        _identifier(self.qualification_id, name="qualification_id")
        if not isinstance(self.tracker_status, TrackerStatus):
            raise TypeError("tracker_status must be TrackerStatus")
        _booleans(
            self,
            "runtime_qualified",
            "target_angle_reached",
            "pivot_target_confirmed",
            "tube_between_fingers",
        )


@dataclass(frozen=True)
class InsertionProgressReceipt:
    header: ReceiptHeader
    profile_id: str
    measured_depth_m: float
    target_depth_m: float
    depth_tolerance_m: float
    axial_progress_m: float
    minimum_progress_m: float
    jam_assessment: JamAssessment
    runtime_jam_qualified: bool
    jam_shadow_candidate: bool = False

    def __post_init__(self) -> None:
        _receipt_header(self.header)
        _identifier(self.profile_id, name="profile_id")
        for name in (
            "measured_depth_m",
            "target_depth_m",
            "depth_tolerance_m",
            "axial_progress_m",
            "minimum_progress_m",
        ):
            _finite(getattr(self, name), name=name, minimum=0.0)
        if self.target_depth_m <= 0.0:
            raise ValueError("target_depth_m must be positive")
        if self.depth_tolerance_m >= self.target_depth_m:
            raise ValueError("depth_tolerance_m must be smaller than target_depth_m")
        if not isinstance(self.jam_assessment, JamAssessment):
            raise TypeError("jam_assessment must be JamAssessment")
        _booleans(self, "runtime_jam_qualified", "jam_shadow_candidate")


@dataclass(frozen=True)
class SupportAfterReleaseReceipt:
    """Explicit post-release visual/scene support evidence."""

    header: ReceiptHeader
    verification_method: str
    scene_receipt_id: str
    gripper_receipt_id: str
    tactile_receipt_id: str
    support_confirmed: bool

    def __post_init__(self) -> None:
        _receipt_header(self.header)
        if self.verification_method not in {"front_rgb", "aligned_rgbd", "scene_tracker"}:
            raise ValueError("verification_method must be an explicit visual/scene source")
        for name in ("scene_receipt_id", "gripper_receipt_id", "tactile_receipt_id"):
            _identifier(getattr(self, name), name=name)
        if not isinstance(self.support_confirmed, bool):
            raise TypeError("support_confirmed must be boolean")


@dataclass(frozen=True)
class PredicateReceiptBundle:
    """One time-aligned collection from independent live producers."""

    evaluated_at_s: float
    scene: SceneReceipt | None = None
    motion: MotionReceipt | None = None
    gripper: GripperReceipt | None = None
    tactile: TactileContactReceipt | None = None
    rgb_axis: RgbAxisReceipt | None = None
    insertion: InsertionProgressReceipt | None = None
    support: SupportAfterReleaseReceipt | None = None

    def __post_init__(self) -> None:
        _finite(self.evaluated_at_s, name="evaluated_at_s", minimum=0.0)
        expected = {
            "scene": SceneReceipt,
            "motion": MotionReceipt,
            "gripper": GripperReceipt,
            "tactile": TactileContactReceipt,
            "rgb_axis": RgbAxisReceipt,
            "insertion": InsertionProgressReceipt,
            "support": SupportAfterReleaseReceipt,
        }
        for name, receipt_type in expected.items():
            value = getattr(self, name)
            if value is not None and not isinstance(value, receipt_type):
                raise TypeError(f"{name} must be {receipt_type.__name__} or None")


@dataclass(frozen=True)
class PredicateBuilderConfig:
    task_id: str
    session_id: str
    provenance_id: str
    contact_profile_id: str
    rgb_qualification_id: str
    insertion_profile_id: str
    retained_width_min_m: float
    retained_width_max_m: float
    max_age_s: float = 0.25
    max_bundle_skew_s: float = 0.10
    future_tolerance_s: float = 0.01
    rgb_runtime_qualified: bool = False
    jam_runtime_qualified: bool = False
    tactile_profile_id: str | None = None
    tactile_axis_runtime_qualified: bool = False
    slip_runtime_qualified: bool = False

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "session_id",
            "provenance_id",
            "contact_profile_id",
            "rgb_qualification_id",
            "insertion_profile_id",
        ):
            _identifier(getattr(self, name), name=name)
        for name in (
            "retained_width_min_m",
            "retained_width_max_m",
            "max_age_s",
            "max_bundle_skew_s",
            "future_tolerance_s",
        ):
            _finite(getattr(self, name), name=name, minimum=0.0)
        if self.retained_width_min_m >= self.retained_width_max_m:
            raise ValueError("retained width range must be increasing")
        if self.max_age_s <= 0.0 or self.max_bundle_skew_s <= 0.0:
            raise ValueError("freshness limits must be positive")
        if not isinstance(self.rgb_runtime_qualified, bool):
            raise TypeError("rgb_runtime_qualified must be boolean")
        if not isinstance(self.jam_runtime_qualified, bool):
            raise TypeError("jam_runtime_qualified must be boolean")
        for name in ("tactile_axis_runtime_qualified", "slip_runtime_qualified"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.tactile_profile_id is not None:
            _identifier(self.tactile_profile_id, name="tactile_profile_id")
        if (
            self.tactile_axis_runtime_qualified or self.slip_runtime_qualified
        ) and self.tactile_profile_id is None:
            raise ValueError(
                "tactile_profile_id is required for qualified tactile axis or slip"
            )


# Predicates owned by the runtime budgets are intentionally absent.  The
# tactile axis and slip channels are present in the manifest, but their builder
# configuration defaults to fail-closed shadow-only operation.
PREDICATE_TO_PRODUCER_MANIFEST: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "invalid_evidence": {"producer": "builder", "states": "all"},
        "stale_evidence": {"producer": "builder", "states": "all"},
        "tube_pose_valid": {"producer": "scene", "states": ["OBSERVE"]},
        "socket_frame_valid": {"producer": "scene", "states": ["OBSERVE"]},
        "approach_pose_reached": {"producer": "motion", "states": ["APPROACH_TUBE"]},
        "contact_present": {
            "producer": "tactile",
            "states": [
                "SIDE_GRASP",
                "EXTRACT",
                "LEVEL_HELD_TUBE",
                "PIVOT_MICRO_LOOSEN",
                "PIVOT_EVALUATE",
                "RECLAMP",
                "RECLAMP_RECOVERY",
            ],
        },
        "contact_lost": {
            "producer": "tactile",
            "states": [
                "EXTRACT",
                "LEVEL_HELD_TUBE",
                "PIVOT_MICRO_LOOSEN",
                "PIVOT_EVALUATE",
            ],
        },
        "grasp_established": {
            "producer": "tactile+gripper+(bilateral_tactile|qualified_rgb)",
            "states": ["SIDE_GRASP"],
        },
        "tube_clear_of_source": {"producer": "scene", "states": ["EXTRACT"]},
        "tube_horizontal": {
            "producer": "held_pose_geometry",
            "states": ["LEVEL_HELD_TUBE"],
        },
        "level_target_reached": {
            "producer": "motion",
            "states": ["LEVEL_HELD_TUBE"],
        },
        "one_microstep_complete": {
            "producer": "motion",
            "states": ["PIVOT_MICRO_LOOSEN"],
        },
        "front_rgb_calibrated_angle_reached": {
            "producer": "qualified_rgb_axis",
            "states": ["PIVOT_EVALUATE"],
        },
        "tactile_relative_angle_reached": {
            "producer": "qualified_tactile_axis_evaluation",
            "states": ["PIVOT_EVALUATE"],
        },
        "slip_detected": {
            "producer": "qualified_tactile_axis_evaluation",
            "states": [
                "EXTRACT",
                "LEVEL_HELD_TUBE",
                "PIVOT_MICRO_LOOSEN",
                "PIVOT_EVALUATE",
            ],
        },
        "reclamp_command_settled": {
            "producer": "motion",
            "states": ["RECLAMP", "RECLAMP_RECOVERY"],
        },
        "pivot_target_confirmed": {
            "producer": "qualified_rgb_axis|qualified_tactile_axis_evaluation",
            "states": ["RECLAMP"],
        },
        "socket_standoff_reached": {
            "producer": "motion",
            "states": ["APPROACH_SOCKET"],
        },
        "tube_socket_axes_aligned": {"producer": "scene", "states": ["ALIGN_SOCKET"]},
        "tube_tip_at_socket_entry": {"producer": "scene", "states": ["ALIGN_SOCKET"]},
        "one_axial_step_complete": {
            "producer": "motion",
            "states": ["INSERT_MICROSTEP"],
        },
        "insertion_jam": {
            "producer": "qualified_insertion_progress",
            "states": ["INSERT_EVALUATE"],
        },
        "no_jam": {
            "producer": "qualified_insertion_progress",
            "states": ["INSERT_EVALUATE"],
        },
        "insertion_depth_reached": {
            "producer": "insertion_progress",
            "states": ["INSERT_EVALUATE"],
        },
        "insertion_depth_remaining": {
            "producer": "insertion_progress",
            "states": ["INSERT_EVALUATE"],
        },
        "axial_progress_ok": {
            "producer": "insertion_progress",
            "states": ["INSERT_EVALUATE"],
        },
        "retract_complete": {
            "producer": "motion",
            "states": ["JAM_RECOVERY_RETRACT"],
        },
        "scene_reobserved": {
            "producer": "scene",
            "states": ["JAM_RECOVERY_RETRACT"],
        },
        "gripper_open": {"producer": "gripper", "states": ["RELEASE"]},
        "contact_released": {"producer": "tactile", "states": ["RELEASE"]},
        "tube_stays_supported": {
            "producer": "explicit_visual_support",
            "states": ["RELEASE"],
        },
        "retreat_complete": {"producer": "motion", "states": ["RETREAT"]},
    }
)

_REQUIRED_PRODUCERS: Mapping[TubeState, frozenset[str]] = {
    TubeState.OBSERVE: frozenset({"scene"}),
    TubeState.APPROACH_TUBE: frozenset({"motion"}),
    TubeState.SIDE_GRASP: frozenset({"gripper", "tactile"}),
    TubeState.EXTRACT: frozenset({"scene", "tactile"}),
    TubeState.LEVEL_HELD_TUBE: frozenset({"scene", "motion", "tactile"}),
    TubeState.PIVOT_MICRO_LOOSEN: frozenset({"motion", "tactile"}),
    # Angle evidence may come from either the optional RGB receipt or the
    # qualified tactile-axis evaluation nested in the tactile receipt.
    TubeState.PIVOT_EVALUATE: frozenset({"tactile"}),
    TubeState.RECLAMP: frozenset({"motion", "tactile"}),
    TubeState.RECLAMP_RECOVERY: frozenset({"motion", "tactile"}),
    TubeState.APPROACH_SOCKET: frozenset({"motion"}),
    TubeState.ALIGN_SOCKET: frozenset({"scene"}),
    TubeState.INSERT_MICROSTEP: frozenset({"motion"}),
    TubeState.INSERT_EVALUATE: frozenset({"insertion"}),
    TubeState.JAM_RECOVERY_RETRACT: frozenset({"motion", "scene"}),
    TubeState.RELEASE: frozenset({"gripper", "scene", "support", "tactile"}),
    TubeState.RETREAT: frozenset({"motion"}),
    TubeState.COMPLETE: frozenset(),
    TubeState.HOLD: frozenset(),
}

_MOTION_BY_STATE = {
    TubeState.APPROACH_TUBE: (MotionEvent.APPROACH_POSE_REACHED, "approach_pose_reached"),
    TubeState.LEVEL_HELD_TUBE: (
        MotionEvent.LEVEL_TARGET_REACHED,
        "level_target_reached",
    ),
    TubeState.PIVOT_MICRO_LOOSEN: (
        MotionEvent.PIVOT_MICROSTEP_COMPLETE,
        "one_microstep_complete",
    ),
    TubeState.RECLAMP: (MotionEvent.RECLAMP_COMMAND_SETTLED, "reclamp_command_settled"),
    TubeState.RECLAMP_RECOVERY: (
        MotionEvent.RECLAMP_COMMAND_SETTLED,
        "reclamp_command_settled",
    ),
    TubeState.APPROACH_SOCKET: (
        MotionEvent.SOCKET_STANDOFF_REACHED,
        "socket_standoff_reached",
    ),
    TubeState.INSERT_MICROSTEP: (MotionEvent.AXIAL_STEP_COMPLETE, "one_axial_step_complete"),
    TubeState.JAM_RECOVERY_RETRACT: (MotionEvent.RETRACT_COMPLETE, "retract_complete"),
    TubeState.RETREAT: (MotionEvent.RETREAT_COMPLETE, "retreat_complete"),
}

_CONTACT_PRESENT_STATES = frozenset(
    {
        TubeState.SIDE_GRASP,
        TubeState.EXTRACT,
        TubeState.LEVEL_HELD_TUBE,
        TubeState.PIVOT_MICRO_LOOSEN,
        TubeState.PIVOT_EVALUATE,
        TubeState.RECLAMP,
        TubeState.RECLAMP_RECOVERY,
    }
)


class ReceiptSource(Protocol):
    """Injected hardware adapter boundary used by :class:`TestTubePredicateProvider`."""

    def collect(
        self,
        *,
        state: TubeState,
        step_index: int,
        budgets: Mapping[str, int],
    ) -> PredicateReceiptBundle: ...


class TestTubePredicateBuilder:
    """Promote one receipt bundle into an auditable predicate snapshot."""

    def __init__(self, config: PredicateBuilderConfig) -> None:
        if not isinstance(config, PredicateBuilderConfig):
            raise TypeError("config must be PredicateBuilderConfig")
        self.config = config

    def build(
        self,
        *,
        state: TubeState,
        step_index: int,
        bundle: PredicateReceiptBundle,
    ) -> PredicateSnapshot:
        if not isinstance(state, TubeState):
            raise TypeError("state must be TubeState")
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        if not isinstance(bundle, PredicateReceiptBundle):
            raise TypeError("bundle must be PredicateReceiptBundle")

        receipts = _receipts(bundle)
        missing = sorted(_REQUIRED_PRODUCERS[state] - receipts.keys())
        rejection_reasons: list[str] = []
        stale_ids: list[str] = []
        if missing:
            rejection_reasons.append(f"missing_required_producers:{','.join(missing)}")

        for producer, receipt in receipts.items():
            header = receipt.header
            if not header.valid:
                rejection_reasons.append(f"{producer}:producer_marked_invalid")
            for name, actual, expected in (
                ("task_id", header.task_id, self.config.task_id),
                ("session_id", header.session_id, self.config.session_id),
                ("provenance_id", header.provenance_id, self.config.provenance_id),
            ):
                if actual != expected:
                    rejection_reasons.append(f"{producer}:{name}_mismatch")
            age = bundle.evaluated_at_s - header.observed_at_s
            if age < -self.config.future_tolerance_s:
                rejection_reasons.append(f"{producer}:timestamp_in_future")
            elif age > self.config.max_age_s:
                stale_ids.append(header.receipt_id)

        receipt_ids = [receipt.header.receipt_id for receipt in receipts.values()]
        if len(receipt_ids) != len(set(receipt_ids)):
            rejection_reasons.append("duplicate_receipt_id")
        required_times = [
            receipts[name].header.observed_at_s
            for name in _REQUIRED_PRODUCERS[state]
            if name in receipts
        ]
        if (
            required_times
            and max(required_times) - min(required_times) > self.config.max_bundle_skew_s
        ):
            rejection_reasons.append("required_receipt_skew_exceeded")

        predicates: set[str] = set()
        if stale_ids:
            predicates.add("stale_evidence")
        if rejection_reasons:
            predicates.add("invalid_evidence")
        shadow = _shadow_observations(receipts)
        if not predicates:
            self._promote(state, receipts, predicates, rejection_reasons)
            if rejection_reasons:
                predicates = {"invalid_evidence"}

        unexpected = predicates - PREDICATE_TO_PRODUCER_MANIFEST.keys()
        if unexpected:  # internal programming error; never leak an unowned predicate
            raise AssertionError(f"unmanifested predicates: {sorted(unexpected)}")
        for predicate in predicates:
            states = PREDICATE_TO_PRODUCER_MANIFEST[predicate]["states"]
            if states != "all" and state.value not in states:
                raise AssertionError(
                    f"predicate {predicate!r} is not owned by state {state.value!r}"
                )

        receipt_payload = {
            "schema": TEST_TUBE_PREDICATE_RECEIPT_SCHEMA,
            "predicate_manifest_sha256": _digest(predicate_producer_manifest()),
            "task_id": self.config.task_id,
            "session_id": self.config.session_id,
            "provenance_id": self.config.provenance_id,
            "state": state.value,
            "evaluated_at_s": bundle.evaluated_at_s,
            "producer_receipts": {
                name: {
                    "receipt_id": receipt.header.receipt_id,
                    "producer_id": receipt.header.producer_id,
                    "observed_at_s": receipt.header.observed_at_s,
                    "qualification_id": _qualification_id(receipt),
                }
                for name, receipt in sorted(receipts.items())
            },
            "required_producers": sorted(_REQUIRED_PRODUCERS[state]),
            "promoted_predicates": sorted(predicates),
            "stale_receipt_ids": sorted(stale_ids),
            "rejection_reasons": sorted(set(rejection_reasons)),
            "shadow_observations": shadow,
        }
        evidence_id = _digest(receipt_payload)
        return PredicateSnapshot(
            sequence=step_index,
            state=state,
            evidence_id=evidence_id,
            observed_at_s=bundle.evaluated_at_s,
            predicates=tuple(sorted(predicates)),
            evidence_receipt=receipt_payload,
        )

    def _promote(
        self,
        state: TubeState,
        receipts: Mapping[str, Any],
        predicates: set[str],
        rejection_reasons: list[str],
    ) -> None:
        scene = receipts.get("scene")
        motion = receipts.get("motion")
        gripper = receipts.get("gripper")
        tactile = receipts.get("tactile")
        rgb = receipts.get("rgb_axis")
        insertion = receipts.get("insertion")
        support = receipts.get("support")

        if isinstance(tactile, TactileContactReceipt):
            if (
                not tactile.runtime_contact_qualified
                or tactile.contact_profile_id != self.config.contact_profile_id
            ):
                rejection_reasons.append("tactile:contact_profile_not_runtime_qualified")

        rgb_qualified = isinstance(rgb, RgbAxisReceipt) and self._rgb_qualified(rgb)
        tactile_axis_source_qualified = bool(
            isinstance(tactile, TactileContactReceipt)
            and self._tactile_axis_source_qualified(tactile)
        )
        if state in {TubeState.PIVOT_EVALUATE, TubeState.RECLAMP}:
            if not (rgb_qualified or tactile_axis_source_qualified):
                rejection_reasons.append("angle_source:runtime_qualification_missing")
            elif (
                rgb_qualified
                and not tactile_axis_source_qualified
                and rgb.tracker_status is not TrackerStatus.TRACKED
            ):
                rejection_reasons.append(f"rgb_axis:tracker_{rgb.tracker_status.value}")

        if state is TubeState.INSERT_EVALUATE and isinstance(
            insertion, InsertionProgressReceipt
        ):
            if not self._jam_qualified(insertion):
                rejection_reasons.append("insertion:jam_assessment_not_runtime_qualified")

        if rejection_reasons:
            return

        if isinstance(motion, MotionReceipt) and state in _MOTION_BY_STATE:
            expected_event, predicate = _MOTION_BY_STATE[state]
            if (
                motion.event is expected_event
                and motion.success
                and motion.command_terminal
                and motion.measured_target_within_tolerance
            ):
                predicates.add(predicate)

        if state is TubeState.OBSERVE and isinstance(scene, SceneReceipt):
            if scene.tube_pose_valid:
                predicates.add("tube_pose_valid")
            if scene.socket_frame_valid:
                predicates.add("socket_frame_valid")

        if isinstance(tactile, TactileContactReceipt):
            if tactile.status is ContactStatus.PRESENT and state in _CONTACT_PRESENT_STATES:
                predicates.add("contact_present")
            elif tactile.status is ContactStatus.LOST_AFTER_PRESENT and state in {
                TubeState.EXTRACT,
                TubeState.LEVEL_HELD_TUBE,
                TubeState.PIVOT_MICRO_LOOSEN,
                TubeState.PIVOT_EVALUATE,
            }:
                predicates.add("contact_lost")
            axis_evaluation = tactile.axis_evaluation
            if (
                state
                in {
                    TubeState.EXTRACT,
                    TubeState.LEVEL_HELD_TUBE,
                    TubeState.PIVOT_MICRO_LOOSEN,
                    TubeState.PIVOT_EVALUATE,
                }
                and axis_evaluation is not None
                and self._slip_source_qualified(tactile)
                and axis_evaluation.slip_confirmed
            ):
                predicates.add("slip_detected")

        if state is TubeState.SIDE_GRASP:
            assert isinstance(gripper, GripperReceipt)
            assert isinstance(tactile, TactileContactReceipt)
            retained_width = (
                gripper.command_settled
                and gripper.fault_free
                and self.config.retained_width_min_m
                <= gripper.measured_width_m
                <= self.config.retained_width_max_m
            )
            visual_between = bool(
                rgb_qualified
                and rgb is not None
                and rgb.tracker_status is TrackerStatus.TRACKED
                and rgb.tube_between_fingers
            )
            if (
                tactile.status is ContactStatus.PRESENT
                and retained_width
                and (tactile.bilateral_contact or visual_between)
            ):
                predicates.add("grasp_established")

        if state is TubeState.EXTRACT and isinstance(scene, SceneReceipt):
            if scene.tube_clear_of_source:
                predicates.add("tube_clear_of_source")
        if state is TubeState.LEVEL_HELD_TUBE and isinstance(scene, SceneReceipt):
            if scene.tube_horizontal:
                predicates.add("tube_horizontal")

        if state is TubeState.PIVOT_EVALUATE and rgb_qualified:
            assert isinstance(rgb, RgbAxisReceipt)
            if rgb.target_angle_reached:
                predicates.add("front_rgb_calibrated_angle_reached")
        if (
            state is TubeState.PIVOT_EVALUATE
            and isinstance(tactile, TactileContactReceipt)
            and tactile_axis_source_qualified
            and tactile.axis_evaluation is not None
            and tactile.axis_evaluation.target_confirmed
        ):
            predicates.add("tactile_relative_angle_reached")

        if state is TubeState.RECLAMP and rgb_qualified:
            assert isinstance(rgb, RgbAxisReceipt)
            if rgb.pivot_target_confirmed:
                predicates.add("pivot_target_confirmed")
        if (
            state is TubeState.RECLAMP
            and isinstance(tactile, TactileContactReceipt)
            and tactile_axis_source_qualified
            and tactile.axis_evaluation is not None
            and tactile.axis_evaluation.target_confirmed
        ):
            predicates.add("pivot_target_confirmed")

        if state is TubeState.ALIGN_SOCKET and isinstance(scene, SceneReceipt):
            if scene.tube_socket_axes_aligned:
                predicates.add("tube_socket_axes_aligned")
            if scene.tube_tip_at_socket_entry:
                predicates.add("tube_tip_at_socket_entry")

        if state is TubeState.INSERT_EVALUATE and isinstance(
            insertion, InsertionProgressReceipt
        ):
            if insertion.jam_assessment is JamAssessment.JAM:
                predicates.add("insertion_jam")
            elif insertion.jam_assessment is JamAssessment.CLEAR:
                predicates.add("no_jam")
            depth_reached = (
                insertion.measured_depth_m
                >= insertion.target_depth_m - insertion.depth_tolerance_m
            )
            if depth_reached:
                predicates.add("insertion_depth_reached")
            else:
                predicates.add("insertion_depth_remaining")
                if insertion.axial_progress_m >= insertion.minimum_progress_m:
                    predicates.add("axial_progress_ok")

        if state is TubeState.JAM_RECOVERY_RETRACT and isinstance(scene, SceneReceipt):
            if scene.scene_reobserved:
                predicates.add("scene_reobserved")

        if state is TubeState.RELEASE:
            assert isinstance(gripper, GripperReceipt)
            assert isinstance(tactile, TactileContactReceipt)
            assert isinstance(scene, SceneReceipt)
            assert isinstance(support, SupportAfterReleaseReceipt)
            if gripper.open_confirmed and gripper.command_settled and gripper.fault_free:
                predicates.add("gripper_open")
            if tactile.status is ContactStatus.RELEASED_AFTER_OPEN:
                predicates.add("contact_released")
            links_valid = (
                support.scene_receipt_id == scene.header.receipt_id
                and support.gripper_receipt_id == gripper.header.receipt_id
                and support.tactile_receipt_id == tactile.header.receipt_id
                and support.header.observed_at_s
                >= max(
                    scene.header.observed_at_s,
                    gripper.header.observed_at_s,
                    tactile.header.observed_at_s,
                )
            )
            if support.support_confirmed and links_valid:
                predicates.add("tube_stays_supported")
            elif support.support_confirmed:
                rejection_reasons.append("support:receipt_links_or_order_invalid")

    def _rgb_qualified(self, receipt: RgbAxisReceipt) -> bool:
        return (
            self.config.rgb_runtime_qualified
            and receipt.runtime_qualified
            and receipt.qualification_id == self.config.rgb_qualification_id
        )

    def _jam_qualified(self, receipt: InsertionProgressReceipt) -> bool:
        return (
            self.config.jam_runtime_qualified
            and receipt.runtime_jam_qualified
            and receipt.profile_id == self.config.insertion_profile_id
        )

    def _tactile_axis_source_qualified(
        self, receipt: TactileContactReceipt
    ) -> bool:
        axis = receipt.axis_evaluation
        return bool(
            axis is not None
            and self.config.tactile_axis_runtime_qualified
            and self.config.tactile_profile_id is not None
            and axis.profile_id == self.config.tactile_profile_id
            and axis.session_id == self.config.session_id
            and axis.provenance_id == self.config.provenance_id
            and axis.evaluation_valid
            and axis.runtime_axis_qualified
        )

    def _slip_source_qualified(self, receipt: TactileContactReceipt) -> bool:
        axis = receipt.axis_evaluation
        return bool(
            axis is not None
            and self.config.slip_runtime_qualified
            and self.config.tactile_profile_id is not None
            and axis.profile_id == self.config.tactile_profile_id
            and axis.session_id == self.config.session_id
            and axis.provenance_id == self.config.provenance_id
            and axis.evaluation_valid
            and axis.runtime_slip_qualified
        )


class TestTubePredicateProvider:
    """Concrete :class:`PredicateProvider` backed by an injected receipt source."""

    def __init__(
        self,
        builder: TestTubePredicateBuilder,
        source: ReceiptSource | Callable[..., PredicateReceiptBundle],
    ) -> None:
        if not isinstance(builder, TestTubePredicateBuilder):
            raise TypeError("builder must be TestTubePredicateBuilder")
        if not callable(source) and not callable(getattr(source, "collect", None)):
            raise TypeError("source must be callable or implement collect")
        self.builder = builder
        self.source = source
        self._last_evaluated_at_s: float | None = None

    def observe(
        self,
        *,
        state: TubeState,
        step_index: int,
        budgets: Mapping[str, int],
    ) -> PredicateSnapshot:
        collector = getattr(self.source, "collect", self.source)
        bundle = collector(state=state, step_index=step_index, budgets=budgets)
        if not isinstance(bundle, PredicateReceiptBundle):
            raise ReceiptValidationError("receipt source returned the wrong bundle type")
        if (
            self._last_evaluated_at_s is not None
            and bundle.evaluated_at_s <= self._last_evaluated_at_s
        ):
            raise ReceiptValidationError("bundle evaluation timestamps must strictly increase")
        snapshot = self.builder.build(state=state, step_index=step_index, bundle=bundle)
        self._last_evaluated_at_s = bundle.evaluated_at_s
        return snapshot


def predicate_producer_manifest() -> dict[str, dict[str, Any]]:
    """Return a mutable JSON-safe copy for reports and adapter generation."""

    manifest = {name: dict(value) for name, value in PREDICATE_TO_PRODUCER_MANIFEST.items()}
    return json.loads(json.dumps(manifest, sort_keys=True))


def required_predicate_producers_by_state() -> dict[str, list[str]]:
    """Return the builder-owned producer requirements for runtime validation."""

    return {
        state.value: sorted(producers)
        for state, producers in sorted(_REQUIRED_PRODUCERS.items(), key=lambda item: item[0].value)
    }


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")
    if len(value) > 256 or any(ord(character) < 0x20 for character in value):
        raise ValueError(f"{name} is invalid")
    return value


def _finite(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _unit_axis_tuple(value: object, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise TypeError(f"{name} must be a finite 3-vector")
    components = tuple(_finite(item, name=f"{name}[{index}]") for index, item in enumerate(value))
    norm = math.sqrt(sum(item * item for item in components))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(f"{name} must be unit length")
    return tuple(item / norm for item in components)


def _receipt_header(value: object) -> ReceiptHeader:
    if not isinstance(value, ReceiptHeader):
        raise TypeError("header must be ReceiptHeader")
    return value


def _booleans(value: object, *names: str) -> None:
    for name in names:
        if not isinstance(getattr(value, name), bool):
            raise TypeError(f"{name} must be boolean")


def _receipts(bundle: PredicateReceiptBundle) -> dict[str, Any]:
    return {
        field.name: getattr(bundle, field.name)
        for field in fields(bundle)
        if field.name != "evaluated_at_s" and getattr(bundle, field.name) is not None
    }


def _qualification_id(receipt: object) -> str | None:
    if isinstance(receipt, TactileContactReceipt):
        return receipt.contact_profile_id
    if isinstance(receipt, RgbAxisReceipt):
        return receipt.qualification_id
    if isinstance(receipt, InsertionProgressReceipt):
        return receipt.profile_id
    return None


def _shadow_observations(receipts: Mapping[str, Any]) -> dict[str, bool]:
    tactile = receipts.get("tactile")
    insertion = receipts.get("insertion")
    axis = (
        tactile.axis_evaluation
        if isinstance(tactile, TactileContactReceipt)
        else None
    )
    return {
        "slip_shadow_candidate": bool(
            isinstance(tactile, TactileContactReceipt)
            and (
                tactile.slip_shadow_candidate
                or (axis is not None and axis.slip_candidate and not axis.slip_confirmed)
            )
        ),
        "jam_shadow_candidate": bool(
            isinstance(insertion, InsertionProgressReceipt)
            and insertion.jam_shadow_candidate
        ),
    }


def tactile_axis_receipt_from_evaluation(
    *,
    evaluation: TestTubeTactileEvaluation,
    profile: TestTubeTactileControllerProfile,
    provenance_id: str,
) -> TactileAxisReceipt:
    """Convert one pure controller evaluation into the typed builder receipt.

    Confirmation is recomputed from the immutable profile and the controller's
    consecutive-window counters.  Merely placing a predicate name in the
    evaluation cannot bypass either runtime qualification or debouncing.
    """

    from .test_tube_tactile_controller import (
        TestTubeTactileControllerProfile,
        TestTubeTactileEvaluation,
    )

    if not isinstance(evaluation, TestTubeTactileEvaluation):
        raise TypeError("evaluation must be TestTubeTactileEvaluation")
    if not isinstance(profile, TestTubeTactileControllerProfile):
        raise TypeError("profile must be TestTubeTactileControllerProfile")
    provenance = _identifier(provenance_id, name="provenance_id")
    promoted = set(evaluation.promoted_predicates)
    evaluation_valid = evaluation.valid
    target_candidate = bool(
        evaluation_valid and evaluation.primitives.target_axis_candidate
    )
    slip_candidate = bool(evaluation_valid and evaluation.primitives.slip_candidate)
    target_count = (
        evaluation.next_memory.target_axis_windows if evaluation_valid else 0
    )
    slip_count = evaluation.next_memory.slip_windows if evaluation_valid else 0
    runtime_axis_qualified = bool(
        evaluation_valid and profile.axis_runtime_qualified
    )
    runtime_slip_qualified = bool(
        evaluation_valid and profile.slip_runtime_qualified
    )
    target_confirmed = bool(
        runtime_axis_qualified
        and target_candidate
        and evaluation.primitives.axis_tcp is not None
        and evaluation.primitives.axis_base is not None
        and target_count >= profile.axis_confirmation_windows
        and "tactile_relative_angle_reached" in promoted
    )
    slip_confirmed = bool(
        runtime_slip_qualified
        and slip_candidate
        and slip_count >= profile.slip_confirmation_windows
        and "slip_detected" in promoted
    )
    return TactileAxisReceipt(
        profile_id=profile.profile_id,
        session_id=profile.session_id,
        provenance_id=provenance,
        source_evidence_id=evaluation.evidence_id,
        evaluation_valid=evaluation_valid,
        axis_tcp=evaluation.primitives.axis_tcp,
        axis_base=evaluation.primitives.axis_base,
        target_error_deg=evaluation.primitives.target_axis_error_deg,
        target_tolerance_deg=profile.target_axis_tolerance_deg,
        target_candidate=target_candidate,
        target_confirmed=target_confirmed,
        target_confirmation_count=target_count,
        target_confirmation_required=profile.axis_confirmation_windows,
        runtime_axis_qualified=runtime_axis_qualified,
        slip_candidate=slip_candidate,
        slip_confirmed=slip_confirmed,
        slip_confirmation_count=slip_count,
        slip_confirmation_required=profile.slip_confirmation_windows,
        runtime_slip_qualified=runtime_slip_qualified,
    )


def tactile_contact_receipt_from_evaluation(
    *,
    header: ReceiptHeader,
    evaluation: TestTubeTactileEvaluation,
    profile: TestTubeTactileControllerProfile,
    lost_after_present: bool = False,
) -> TactileContactReceipt:
    """Map controller contact/axis output into the existing tactile producer."""

    from .test_tube_tactile_controller import (
        TestTubeTactileAction,
        TestTubeTactileControllerProfile,
        TestTubeTactileEvaluation,
    )

    if not isinstance(header, ReceiptHeader):
        raise TypeError("header must be ReceiptHeader")
    if not isinstance(evaluation, TestTubeTactileEvaluation):
        raise TypeError("evaluation must be TestTubeTactileEvaluation")
    if not isinstance(profile, TestTubeTactileControllerProfile):
        raise TypeError("profile must be TestTubeTactileControllerProfile")
    if not isinstance(lost_after_present, bool):
        raise TypeError("lost_after_present must be boolean")
    if header.session_id != profile.session_id:
        raise ValueError("profile belongs to a different receipt session")

    if not evaluation.valid:
        status = ContactStatus.INVALID
    elif evaluation.primitives.contact_present:
        status = ContactStatus.PRESENT
    elif evaluation.action is TestTubeTactileAction.RELEASE_READY:
        status = ContactStatus.RELEASED_AFTER_OPEN
    elif lost_after_present:
        status = ContactStatus.LOST_AFTER_PRESENT
    else:
        status = ContactStatus.ABSENT
    contact_runtime_qualified = bool(
        evaluation.valid
        and all(
            finger.contact_response_qualified
            for finger in profile.contact_profile.fingers
        )
    )
    axis = tactile_axis_receipt_from_evaluation(
        evaluation=evaluation,
        profile=profile,
        provenance_id=header.provenance_id,
    )
    return TactileContactReceipt(
        header=header,
        contact_profile_id=profile.contact_profile.profile_id,
        status=status,
        runtime_contact_qualified=contact_runtime_qualified,
        bilateral_contact=bool(
            status is ContactStatus.PRESENT
            and evaluation.primitives.bilateral_contact
        ),
        slip_shadow_candidate=bool(
            evaluation.primitives.slip_candidate and not axis.slip_confirmed
        ),
        axis_evaluation=axis,
    )


def insertion_progress_receipt_from_evaluation(
    *,
    header: ReceiptHeader,
    evaluation: TestTubeTactileEvaluation,
    profile: TestTubeTactileControllerProfile,
    measured_depth_m: float,
    target_depth_m: float,
    depth_tolerance_m: float,
    axial_progress_m: float,
    minimum_progress_m: float,
) -> InsertionProgressReceipt:
    """Map qualified controller jam output into the existing insertion receipt."""

    from .test_tube_tactile_controller import (
        TestTubeTactileControllerProfile,
        TestTubeTactileEvaluation,
        TestTubeTactileMode,
    )

    if not isinstance(header, ReceiptHeader):
        raise TypeError("header must be ReceiptHeader")
    if not isinstance(evaluation, TestTubeTactileEvaluation):
        raise TypeError("evaluation must be TestTubeTactileEvaluation")
    if not isinstance(profile, TestTubeTactileControllerProfile):
        raise TypeError("profile must be TestTubeTactileControllerProfile")
    if header.session_id != profile.session_id:
        raise ValueError("profile belongs to a different receipt session")
    if profile.jam is None:
        raise ValueError("controller profile has no insertion-jam profile")

    runtime_qualified = bool(
        evaluation.valid
        and evaluation.mode is TestTubeTactileMode.INSERT
        and profile.jam.runtime_qualified
    )
    promoted = set(evaluation.promoted_predicates)
    if runtime_qualified and "insertion_jam" in promoted:
        assessment = JamAssessment.JAM
    elif runtime_qualified and "no_jam" in promoted:
        assessment = JamAssessment.CLEAR
    else:
        assessment = JamAssessment.UNKNOWN
    return InsertionProgressReceipt(
        header=header,
        profile_id=profile.jam.profile_id,
        measured_depth_m=measured_depth_m,
        target_depth_m=target_depth_m,
        depth_tolerance_m=depth_tolerance_m,
        axial_progress_m=axial_progress_m,
        minimum_progress_m=minimum_progress_m,
        jam_assessment=assessment,
        runtime_jam_qualified=runtime_qualified,
        jam_shadow_candidate=bool(
            evaluation.primitives.insertion_jam_candidate
            and assessment is not JamAssessment.JAM
        ),
    )


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


__all__ = [
    "ContactStatus",
    "InsertionProgressReceipt",
    "JamAssessment",
    "MotionEvent",
    "MotionReceipt",
    "PREDICATE_TO_PRODUCER_MANIFEST",
    "PredicateBuilderConfig",
    "PredicateReceiptBundle",
    "ReceiptHeader",
    "ReceiptSource",
    "ReceiptValidationError",
    "RgbAxisReceipt",
    "SceneReceipt",
    "SupportAfterReleaseReceipt",
    "TEST_TUBE_PREDICATE_RECEIPT_SCHEMA",
    "TactileAxisReceipt",
    "TactileContactReceipt",
    "TestTubePredicateBuilder",
    "TestTubePredicateProvider",
    "TrackerStatus",
    "GripperReceipt",
    "predicate_producer_manifest",
    "required_predicate_producers_by_state",
    "insertion_progress_receipt_from_evaluation",
    "tactile_axis_receipt_from_evaluation",
    "tactile_contact_receipt_from_evaluation",
]
