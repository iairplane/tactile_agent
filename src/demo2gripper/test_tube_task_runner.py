"""Pure contracts for the task-specific test-tube hardware runner.

The module deliberately sits above MoveIt, Hand-E, RealSense, and GF220.  It
validates the metric poses that those components consume and implements the
small outer state machine used by :mod:`scripts.run_test_tube_task`.  It never
opens a device and never dispatches a command, which keeps dry-run and unit
tests physically inert.

The transparent rack entry is a special boundary.  Reaching a reviewed entry
standoff does not authorize insertion.  A fresh, qualified guarded
visual/touch-localization receipt is required before the state machine can
enter ``GUARDED_INSERT``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from .pose_motion import (
    QuaternionXYZW,
    normalize_quaternion_xyzw,
)

TEST_TUBE_TASK_INPUT_SCHEMA = "demo2gripper.test_tube_task_input.v1"
GUARDED_LOCALIZATION_RECEIPT_SCHEMA = (
    "demo2gripper.test_tube_guarded_localization_receipt.v1"
)


class TaskContractError(ValueError):
    """The runner input is malformed or claims unsupported authority."""


class RunnerMode(StrEnum):
    DRY_RUN = "dry_run"
    PLAN_ONLY = "plan_only"
    EXECUTE = "execute"


class GeometryStatus(StrEnum):
    QUALIFIED = "qualified"
    PROVISIONAL = "provisional"
    UNAVAILABLE = "unavailable"


class TestTubeTaskStage(StrEnum):
    APPROACH = "APPROACH"
    GRASP_EXTRACT = "GRASP_EXTRACT"
    PIVOT = "PIVOT"
    RECLAMP = "RECLAMP"
    APPROACH_ENTRY = "APPROACH_ENTRY"
    ENTRY_STANDOFF = "ENTRY_STANDOFF"
    GUARDED_INSERT = "GUARDED_INSERT"
    RELEASE_RETREAT = "RELEASE_RETREAT"
    COMPLETE = "COMPLETE"
    HOLD = "HOLD"


@dataclass(frozen=True)
class ToolPose:
    """One absolute configured-EE pose in the declared base frame."""

    position_m: tuple[float, float, float]
    orientation_xyzw: QuaternionXYZW

    @classmethod
    def parse(cls, value: object, *, name: str) -> ToolPose:
        payload = _mapping(value, name=name)
        position = _vector(
            payload.get("position_m"), length=3, name=f"{name}.position_m"
        )
        orientation = _vector(
            payload.get("orientation_xyzw"),
            length=4,
            name=f"{name}.orientation_xyzw",
        )
        try:
            quaternion = normalize_quaternion_xyzw(orientation)
        except ValueError as error:
            raise TaskContractError(f"{name}.orientation_xyzw: {error}") from error
        return cls(
            position_m=(position[0], position[1], position[2]),
            orientation_xyzw=quaternion,
        )

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "position_m": list(self.position_m),
            "orientation_xyzw": list(self.orientation_xyzw),
        }


@dataclass(frozen=True)
class GeometryEvidence:
    """Qualification state and provenance for one geometry layer."""

    status: GeometryStatus
    receipt_id: str | None
    observed_at_s: float | None

    @classmethod
    def parse(cls, value: object, *, name: str) -> GeometryEvidence:
        payload = _mapping(value, name=name)
        try:
            status = GeometryStatus(payload.get("status"))
        except ValueError as error:
            raise TaskContractError(
                f"{name}.status must be qualified, provisional, or unavailable"
            ) from error
        receipt = payload.get("receipt_id")
        observed = payload.get("observed_at_s")
        if status is GeometryStatus.UNAVAILABLE:
            if receipt is not None or observed is not None:
                raise TaskContractError(
                    f"{name} unavailable evidence cannot carry a receipt or timestamp"
                )
            return cls(status=status, receipt_id=None, observed_at_s=None)
        return cls(
            status=status,
            receipt_id=_identifier(receipt, name=f"{name}.receipt_id"),
            observed_at_s=_number(
                observed, name=f"{name}.observed_at_s", minimum=0.0
            ),
        )


@dataclass(frozen=True)
class GuardedLocalizationReceipt:
    """Fresh socket-entry authority needed to start contact insertion."""

    receipt_id: str
    task_id: str
    observation_id: str
    collision_scene_id: str
    observed_at_s: float
    insertion_axis_base: tuple[float, float, float]
    target_depth_m: float
    visual_localization_qualified: bool
    touch_search_qualified: bool
    runtime_jam_qualified: bool

    @classmethod
    def parse(
        cls,
        value: object,
        *,
        task_id: str,
    ) -> GuardedLocalizationReceipt:
        payload = _mapping(value, name="guarded_localization_receipt")
        if payload.get("schema") != GUARDED_LOCALIZATION_RECEIPT_SCHEMA:
            raise TaskContractError(
                "guarded localization receipt has an unsupported schema"
            )
        if payload.get("status") != "qualified":
            raise TaskContractError(
                "guarded localization receipt status must be qualified"
            )
        if payload.get("task_id") != task_id:
            raise TaskContractError(
                "guarded localization receipt task_id does not match"
            )
        axis = _unit_vector(
            payload.get("insertion_axis_base"),
            name="guarded_localization_receipt.insertion_axis_base",
        )
        booleans: dict[str, bool] = {}
        for field in (
            "visual_localization_qualified",
            "touch_search_qualified",
            "runtime_jam_qualified",
        ):
            raw = payload.get(field)
            if not isinstance(raw, bool):
                raise TaskContractError(
                    f"guarded_localization_receipt.{field} must be boolean"
                )
            booleans[field] = raw
        if not (
            booleans["visual_localization_qualified"]
            or booleans["touch_search_qualified"]
        ):
            raise TaskContractError(
                "guarded localization needs qualified visual localization or touch search"
            )
        return cls(
            receipt_id=_identifier(
                payload.get("receipt_id"),
                name="guarded_localization_receipt.receipt_id",
            ),
            task_id=task_id,
            observation_id=_identifier(
                payload.get("observation_id"),
                name="guarded_localization_receipt.observation_id",
            ),
            collision_scene_id=_identifier(
                payload.get("collision_scene_id"),
                name="guarded_localization_receipt.collision_scene_id",
            ),
            observed_at_s=_number(
                payload.get("observed_at_s"),
                name="guarded_localization_receipt.observed_at_s",
                minimum=0.0,
            ),
            insertion_axis_base=axis,
            target_depth_m=_number(
                payload.get("target_depth_m"),
                name="guarded_localization_receipt.target_depth_m",
                minimum=0.0,
                strictly_positive=True,
            ),
            visual_localization_qualified=booleans[
                "visual_localization_qualified"
            ],
            touch_search_qualified=booleans["touch_search_qualified"],
            runtime_jam_qualified=booleans["runtime_jam_qualified"],
        )


@dataclass(frozen=True)
class TestTubeTaskInput:
    """Normalized input shared by dry-run, plan-only, and execution modes."""

    task_id: str
    base_frame: str
    camera_extrinsic: GeometryEvidence
    source_geometry: GeometryEvidence
    socket_geometry: GeometryEvidence
    collision_scene: GeometryEvidence
    poses: Mapping[str, ToolPose]
    pivot_rgb_status: GeometryStatus
    pivot_rgb_qualification_id: str | None
    tactile_axis_mode: str
    jam_profile_status: GeometryStatus
    gripper_port: str
    gripper_open_width_mm: float
    grasp_width_min_mm: float
    grasp_width_max_mm: float
    pivot_open_step_mm: float
    pivot_max_steps: int
    reclamp_step_mm: float
    reclamp_max_steps: int
    insertion_step_m: float
    insertion_max_steps: int
    maximum_jam_recoveries: int
    guarded_localization: GuardedLocalizationReceipt | None

    @classmethod
    def parse(cls, value: object) -> TestTubeTaskInput:
        root = _mapping(value, name="task input")
        if root.get("schema") != TEST_TUBE_TASK_INPUT_SCHEMA:
            raise TaskContractError(
                f"task input schema must be {TEST_TUBE_TASK_INPUT_SCHEMA!r}"
            )
        task_id = _identifier(root.get("task_id"), name="task_id")
        base_frame = _identifier(root.get("base_frame"), name="base_frame")
        geometry = _mapping(root.get("geometry"), name="geometry")
        camera = GeometryEvidence.parse(
            geometry.get("camera_extrinsic"), name="geometry.camera_extrinsic"
        )
        source = GeometryEvidence.parse(
            geometry.get("source"), name="geometry.source"
        )
        socket = GeometryEvidence.parse(
            geometry.get("socket"), name="geometry.socket"
        )
        collision = GeometryEvidence.parse(
            geometry.get("collision_scene"), name="geometry.collision_scene"
        )

        raw_poses = _mapping(root.get("poses"), name="poses")
        required = ("approach", "grasp", "extract", "pivot", "reclamp")
        poses: dict[str, ToolPose] = {
            name: ToolPose.parse(raw_poses.get(name), name=f"poses.{name}")
            for name in required
        }
        if socket.status is not GeometryStatus.UNAVAILABLE:
            poses["entry_standoff"] = ToolPose.parse(
                raw_poses.get("entry_standoff"), name="poses.entry_standoff"
            )
        elif raw_poses.get("entry_standoff") is not None:
            raise TaskContractError(
                "entry_standoff cannot be supplied when socket geometry is unavailable"
            )

        pivot = _mapping(root.get("pivot"), name="pivot")
        if pivot.get("target_source") != "front_rgb":
            raise TaskContractError(
                "pivot.target_source must be front_rgb until tactile angle is calibrated"
            )
        try:
            rgb_status = GeometryStatus(pivot.get("rgb_status"))
        except ValueError as error:
            raise TaskContractError("pivot.rgb_status is invalid") from error
        rgb_id = pivot.get("rgb_qualification_id")
        if rgb_status is GeometryStatus.UNAVAILABLE:
            if rgb_id is not None:
                raise TaskContractError(
                    "unavailable RGB qualification cannot carry an id"
                )
            normalized_rgb_id = None
        else:
            normalized_rgb_id = _identifier(
                rgb_id, name="pivot.rgb_qualification_id"
            )
        tactile_axis_mode = pivot.get("tactile_axis_mode")
        if tactile_axis_mode not in {"shadow", "qualified"}:
            raise TaskContractError(
                "pivot.tactile_axis_mode must be shadow or qualified"
            )
        # This task currently requires the independently qualified RGB gate
        # even if a future tactile angle profile is also qualified.
        if tactile_axis_mode == "qualified" and rgb_status is not GeometryStatus.QUALIFIED:
            raise TaskContractError(
                "qualified tactile angle does not replace the required RGB pivot gate"
            )

        tactile = _mapping(root.get("tactile"), name="tactile")
        if tactile.get("bilateral_contact_required") is not True:
            raise TaskContractError(
                "tactile.bilateral_contact_required must be true"
            )
        try:
            jam_status = GeometryStatus(tactile.get("jam_profile_status"))
        except ValueError as error:
            raise TaskContractError("tactile.jam_profile_status is invalid") from error

        gripper = _mapping(root.get("gripper"), name="gripper")
        open_width = _number(
            gripper.get("open_width_mm"), name="gripper.open_width_mm", minimum=0.0
        )
        width_min = _number(
            gripper.get("grasp_width_min_mm"),
            name="gripper.grasp_width_min_mm",
            minimum=0.0,
        )
        width_max = _number(
            gripper.get("grasp_width_max_mm"),
            name="gripper.grasp_width_max_mm",
            minimum=0.0,
        )
        if not width_min < width_max < open_width <= 50.0:
            raise TaskContractError(
                "gripper widths must satisfy 0 <= min < max < open <= 50 mm"
            )

        policy = _mapping(root.get("policy"), name="policy")
        guarded_raw = root.get("guarded_localization_receipt")
        guarded = (
            None
            if guarded_raw is None
            else GuardedLocalizationReceipt.parse(guarded_raw, task_id=task_id)
        )
        return cls(
            task_id=task_id,
            base_frame=base_frame,
            camera_extrinsic=camera,
            source_geometry=source,
            socket_geometry=socket,
            collision_scene=collision,
            poses=poses,
            pivot_rgb_status=rgb_status,
            pivot_rgb_qualification_id=normalized_rgb_id,
            tactile_axis_mode=tactile_axis_mode,
            jam_profile_status=jam_status,
            gripper_port=_absolute_device_path(
                gripper.get("port"), name="gripper.port"
            ),
            gripper_open_width_mm=open_width,
            grasp_width_min_mm=width_min,
            grasp_width_max_mm=width_max,
            pivot_open_step_mm=_bounded_positive(
                policy.get("pivot_open_step_mm"),
                name="policy.pivot_open_step_mm",
                maximum=0.50,
            ),
            pivot_max_steps=_integer(
                policy.get("pivot_max_steps"),
                name="policy.pivot_max_steps",
                minimum=1,
                maximum=12,
            ),
            reclamp_step_mm=_bounded_positive(
                policy.get("reclamp_step_mm"),
                name="policy.reclamp_step_mm",
                maximum=1.0,
            ),
            reclamp_max_steps=_integer(
                policy.get("reclamp_max_steps"),
                name="policy.reclamp_max_steps",
                minimum=1,
                maximum=8,
            ),
            insertion_step_m=_bounded_positive(
                policy.get("insertion_step_m"),
                name="policy.insertion_step_m",
                maximum=0.003,
            ),
            insertion_max_steps=_integer(
                policy.get("insertion_max_steps"),
                name="policy.insertion_max_steps",
                minimum=1,
                maximum=50,
            ),
            maximum_jam_recoveries=_integer(
                policy.get("maximum_jam_recoveries"),
                name="policy.maximum_jam_recoveries",
                minimum=0,
                maximum=3,
            ),
            guarded_localization=guarded,
        )

    def execution_start_blockers(self) -> tuple[str, ...]:
        """Blockers that forbid *any* robot or gripper command."""

        blockers: list[str] = []
        for name, evidence in (
            ("camera_extrinsic", self.camera_extrinsic),
            ("source_geometry", self.source_geometry),
            ("collision_scene", self.collision_scene),
        ):
            if evidence.status is not GeometryStatus.QUALIFIED:
                blockers.append(f"{name}_not_qualified")
        if self.pivot_rgb_status is not GeometryStatus.QUALIFIED:
            blockers.append("front_rgb_pivot_gate_not_qualified")
        return tuple(blockers)

    def guarded_insertion_blockers(self) -> tuple[str, ...]:
        """Blockers checked at ENTRY_STANDOFF, after contact-free approach."""

        blockers: list[str] = []
        if self.socket_geometry.status is not GeometryStatus.QUALIFIED:
            blockers.append("metric_socket_geometry_not_qualified")
        if self.guarded_localization is None:
            blockers.append("guarded_visual_or_touch_localization_receipt_missing")
        elif not self.guarded_localization.runtime_jam_qualified:
            blockers.append("guarded_localization_jam_gate_not_qualified")
        if self.jam_profile_status is not GeometryStatus.QUALIFIED:
            blockers.append("insertion_jam_profile_not_qualified")
        return tuple(blockers)


@dataclass(frozen=True)
class StageEvidence:
    """One fresh stage-bound observation consumed by the outer state machine."""

    evidence_id: str
    motion_complete: bool = False
    bilateral_contact: bool = False
    contact_present: bool = False
    extraction_complete: bool = False
    rgb_pivot_target_qualified: bool = False
    tactile_axis_target_shadow: bool = False
    reclamp_stable: bool = False
    entry_standoff_reached: bool = False
    guarded_localization_qualified: bool = False
    commanded_insertion_step_m: float | None = None
    measured_insertion_progress_m: float | None = None
    tactile_resistance_high: bool = False
    jam_runtime_qualified: bool = False
    jam_confirmation_windows: int = 0
    insertion_target_reached: bool = False
    release_confirmed: bool = False
    retreat_complete: bool = False
    invalid_reason: str | None = None

    @classmethod
    def from_typed_receipts(
        cls,
        *,
        evidence_id: str,
        tactile: object | None = None,
        rgb_axis: object | None = None,
        insertion: object | None = None,
        commanded_insertion_step_m: float | None = None,
        motion_complete: bool = False,
        extraction_complete: bool = False,
        reclamp_stable: bool = False,
        entry_standoff_reached: bool = False,
        guarded_localization_qualified: bool = False,
        insertion_target_reached: bool = False,
        release_confirmed: bool = False,
        retreat_complete: bool = False,
    ) -> StageEvidence:
        """Map canonical producer receipts into the outer-stage vocabulary.

        The tactile receipt is expected to be produced by
        ``tactile_contact_receipt_from_evaluation``.  This keeps contact,
        tactile-axis, slip, and jam classification in the canonical tactile
        controller/predicate adapter instead of duplicating those thresholds
        in this runner.
        """

        from .test_tube_predicates import (  # noqa: PLC0415
            ContactStatus,
            InsertionProgressReceipt,
            JamAssessment,
            RgbAxisReceipt,
            TactileContactReceipt,
        )

        contact_present = False
        bilateral = False
        tactile_shadow = False
        tactile_resistance_high = False
        jam_qualified = False
        jam_windows = 0
        measured_progress: float | None = None
        invalid: list[str] = []
        if tactile is not None:
            if not isinstance(tactile, TactileContactReceipt):
                raise TypeError("tactile must be TactileContactReceipt or None")
            contact_present = tactile.status is ContactStatus.PRESENT
            bilateral = bool(
                tactile.runtime_contact_qualified and tactile.bilateral_contact
            )
            if tactile.status is ContactStatus.INVALID:
                invalid.append("tactile receipt is invalid")
            axis = tactile.axis_evaluation
            if axis is not None:
                tactile_shadow = bool(
                    axis.target_candidate and not axis.target_confirmed
                )

        rgb_target = False
        if rgb_axis is not None:
            if not isinstance(rgb_axis, RgbAxisReceipt):
                raise TypeError("rgb_axis must be RgbAxisReceipt or None")
            rgb_target = bool(
                rgb_axis.runtime_qualified
                and rgb_axis.target_angle_reached
                and rgb_axis.pivot_target_confirmed
            )

        if insertion is not None:
            if not isinstance(insertion, InsertionProgressReceipt):
                raise TypeError("insertion must be InsertionProgressReceipt or None")
            measured_progress = insertion.axial_progress_m
            jam_qualified = bool(insertion.runtime_jam_qualified)
            tactile_resistance_high = insertion.jam_assessment is JamAssessment.JAM
            # A JAM assessment emitted by the canonical adapter already means
            # the controller's configured consecutive-window count was met.
            jam_windows = 2 if tactile_resistance_high else 0
            insertion_target_reached = bool(
                insertion_target_reached
                or insertion.measured_depth_m
                >= insertion.target_depth_m - insertion.depth_tolerance_m
            )

        return cls(
            evidence_id=evidence_id,
            motion_complete=motion_complete,
            bilateral_contact=bilateral,
            contact_present=contact_present,
            extraction_complete=extraction_complete,
            rgb_pivot_target_qualified=rgb_target,
            tactile_axis_target_shadow=tactile_shadow,
            reclamp_stable=reclamp_stable,
            entry_standoff_reached=entry_standoff_reached,
            guarded_localization_qualified=guarded_localization_qualified,
            commanded_insertion_step_m=commanded_insertion_step_m,
            measured_insertion_progress_m=measured_progress,
            tactile_resistance_high=tactile_resistance_high,
            jam_runtime_qualified=jam_qualified,
            jam_confirmation_windows=jam_windows,
            insertion_target_reached=insertion_target_reached,
            release_confirmed=release_confirmed,
            retreat_complete=retreat_complete,
            invalid_reason="; ".join(invalid) if invalid else None,
        )

    def __post_init__(self) -> None:
        _identifier(self.evidence_id, name="evidence_id")
        for field in (
            "motion_complete",
            "bilateral_contact",
            "contact_present",
            "extraction_complete",
            "rgb_pivot_target_qualified",
            "tactile_axis_target_shadow",
            "reclamp_stable",
            "entry_standoff_reached",
            "guarded_localization_qualified",
            "tactile_resistance_high",
            "jam_runtime_qualified",
            "insertion_target_reached",
            "release_confirmed",
            "retreat_complete",
        ):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be boolean")
        if self.invalid_reason is not None:
            _identifier(self.invalid_reason, name="invalid_reason")
        _integer(
            self.jam_confirmation_windows,
            name="jam_confirmation_windows",
            minimum=0,
            maximum=1000,
        )
        for field in (
            "commanded_insertion_step_m",
            "measured_insertion_progress_m",
        ):
            value = getattr(self, field)
            if value is not None:
                _number(value, name=field, minimum=0.0)


@dataclass(frozen=True)
class StageTransition:
    previous_stage: TestTubeTaskStage
    next_stage: TestTubeTaskStage
    action: str
    reason: str
    evidence_id: str


class TestTubeTaskMachine:
    """Deterministic outer state machine; command execution stays outside."""

    def __init__(self, task_input: TestTubeTaskInput) -> None:
        if not isinstance(task_input, TestTubeTaskInput):
            raise TypeError("task_input must be TestTubeTaskInput")
        self.task_input = task_input
        self.stage = TestTubeTaskStage.APPROACH
        self.pivot_steps = 0
        self.reclamp_steps = 0
        self.insertion_steps = 0
        self.jam_recoveries = 0
        self._evidence_ids: set[str] = set()

    def apply(self, evidence: StageEvidence) -> StageTransition:
        if not isinstance(evidence, StageEvidence):
            raise TypeError("evidence must be StageEvidence")
        previous = self.stage
        if evidence.evidence_id in self._evidence_ids:
            return self._hold(evidence, "evidence receipt was replayed")
        self._evidence_ids.add(evidence.evidence_id)
        if evidence.invalid_reason is not None:
            return self._hold(evidence, evidence.invalid_reason)

        if previous is TestTubeTaskStage.APPROACH:
            if evidence.motion_complete:
                return self._move(
                    evidence,
                    TestTubeTaskStage.GRASP_EXTRACT,
                    "OPEN_AND_SIDE_GRASP",
                    "approach pose reached",
                )
            return self._stay(
                evidence, "WAIT_APPROACH", "approach motion is incomplete"
            )

        if previous is TestTubeTaskStage.GRASP_EXTRACT:
            if evidence.extraction_complete and evidence.bilateral_contact:
                return self._move(
                    evidence,
                    TestTubeTaskStage.PIVOT,
                    "START_BOUNDED_PIVOT",
                    "extraction completed with bilateral tactile contact",
                )
            if evidence.extraction_complete and not evidence.bilateral_contact:
                return self._hold(
                    evidence, "extraction cannot continue without bilateral contact"
                )
            return self._stay(
                evidence,
                "CLOSE_OR_EXTRACT",
                "waiting for bilateral contact and measured extraction",
            )

        if previous is TestTubeTaskStage.PIVOT:
            if not evidence.contact_present:
                return self._move(
                    evidence,
                    TestTubeTaskStage.RECLAMP,
                    "RECLAMP_STEP",
                    "contact was lost during pivot",
                )
            if evidence.rgb_pivot_target_qualified:
                return self._move(
                    evidence,
                    TestTubeTaskStage.RECLAMP,
                    "RECLAMP",
                    "qualified front-RGB pivot target reached",
                )
            # Tactile axis remains a diagnostic here.  It never authorizes the
            # transition while the task declares it shadow-only.
            if self.pivot_steps >= self.task_input.pivot_max_steps:
                return self._hold(
                    evidence, "pivot micro-open budget exhausted before RGB target"
                )
            self.pivot_steps += 1
            return self._stay(
                evidence,
                "PIVOT_OPEN_STEP",
                "RGB target not reached; tactile angle is shadow evidence",
            )

        if previous is TestTubeTaskStage.RECLAMP:
            if evidence.reclamp_stable and evidence.contact_present:
                return self._move(
                    evidence,
                    TestTubeTaskStage.APPROACH_ENTRY,
                    "PLAN_ENTRY_STANDOFF",
                    "reclamp contact is stable",
                )
            if self.reclamp_steps >= self.task_input.reclamp_max_steps:
                return self._hold(evidence, "reclamp budget exhausted")
            self.reclamp_steps += 1
            return self._stay(evidence, "RECLAMP_STEP", "waiting for stable tactile contact")

        if previous is TestTubeTaskStage.APPROACH_ENTRY:
            if evidence.entry_standoff_reached and evidence.motion_complete:
                return self._move(
                    evidence,
                    TestTubeTaskStage.ENTRY_STANDOFF,
                    "STOP_AT_ENTRY_STANDOFF",
                    "entry standoff reached; insertion authority checked separately",
                )
            return self._stay(
                evidence,
                "APPROACH_ENTRY_STANDOFF",
                "entry standoff motion is incomplete",
            )

        if previous is TestTubeTaskStage.ENTRY_STANDOFF:
            blockers = self.task_input.guarded_insertion_blockers()
            if blockers or not evidence.guarded_localization_qualified:
                reason = (
                    ",".join(blockers)
                    if blockers
                    else "fresh guarded localization evidence is absent"
                )
                return self._stay(evidence, "HOLD_ENTRY_STANDOFF", reason)
            return self._move(
                evidence,
                TestTubeTaskStage.GUARDED_INSERT,
                "START_GUARDED_INSERT",
                "qualified guarded localization authorizes insertion microsteps",
            )

        if previous is TestTubeTaskStage.GUARDED_INSERT:
            if not evidence.contact_present:
                return self._hold(evidence, "tube contact was lost during insertion")
            if evidence.insertion_target_reached:
                return self._move(
                    evidence,
                    TestTubeTaskStage.RELEASE_RETREAT,
                    "RELEASE",
                    "qualified insertion depth reached",
                )
            if self._qualified_jam(evidence):
                if self.jam_recoveries >= self.task_input.maximum_jam_recoveries:
                    return self._hold(evidence, "jam recovery budget exhausted")
                self.jam_recoveries += 1
                return self._move(
                    evidence,
                    TestTubeTaskStage.APPROACH_ENTRY,
                    "RETRACT_3MM_REOBSERVE_REALIGN",
                    "low progress and qualified tactile resistance persisted",
                )
            if self.insertion_steps >= self.task_input.insertion_max_steps:
                return self._hold(evidence, "insertion microstep budget exhausted")
            self.insertion_steps += 1
            return self._stay(
                evidence,
                "INSERT_MICROSTEP",
                "no qualified jam and target depth not reached",
            )

        if previous is TestTubeTaskStage.RELEASE_RETREAT:
            if evidence.release_confirmed and evidence.retreat_complete:
                return self._move(
                    evidence,
                    TestTubeTaskStage.COMPLETE,
                    "COMPLETE",
                    "release and retreat are confirmed",
                )
            return self._stay(
                evidence,
                "RELEASE_OR_RETREAT",
                "waiting for release and retreat confirmation",
            )

        return self._stay(evidence, "NOOP", "state machine is terminal")

    @staticmethod
    def _qualified_jam(evidence: StageEvidence) -> bool:
        command = evidence.commanded_insertion_step_m
        progress = evidence.measured_insertion_progress_m
        return bool(
            evidence.jam_runtime_qualified
            and evidence.tactile_resistance_high
            and evidence.jam_confirmation_windows >= 2
            and command is not None
            and command > 0.0
            and progress is not None
            and progress < 0.25 * command
        )

    def _move(
        self,
        evidence: StageEvidence,
        target: TestTubeTaskStage,
        action: str,
        reason: str,
    ) -> StageTransition:
        previous = self.stage
        self.stage = target
        return StageTransition(previous, target, action, reason, evidence.evidence_id)

    def _stay(self, evidence: StageEvidence, action: str, reason: str) -> StageTransition:
        return StageTransition(self.stage, self.stage, action, reason, evidence.evidence_id)

    def _hold(self, evidence: StageEvidence, reason: str) -> StageTransition:
        previous = self.stage
        self.stage = TestTubeTaskStage.HOLD
        return StageTransition(previous, self.stage, "STOP_AND_HOLD", reason, evidence.evidence_id)


def command_preview(task: TestTubeTaskInput, mode: RunnerMode) -> list[dict[str, Any]]:
    """Describe component calls without importing or starting a device stack."""

    if not isinstance(task, TestTubeTaskInput):
        raise TypeError("task must be TestTubeTaskInput")
    if not isinstance(mode, RunnerMode):
        mode = RunnerMode(mode)
    # A preview must not accidentally describe provisional geometry as live
    # execution authority.  ``run_test_tube_task.py`` also checks the same
    # blockers before importing a live adapter, but keeping the preview
    # truthful matters because it is persisted as a review artifact.
    execute = mode is RunnerMode.EXECUTE and not task.execution_start_blockers()
    preview: list[dict[str, Any]] = []
    pose_stages = (
        (TestTubeTaskStage.APPROACH, "approach", "moveit_pose_goal"),
        (TestTubeTaskStage.GRASP_EXTRACT, "grasp", "moveit_cartesian_pose_sequence"),
        (TestTubeTaskStage.GRASP_EXTRACT, "extract", "moveit_cartesian_pose_sequence"),
        (TestTubeTaskStage.PIVOT, "pivot", "moveit_cartesian_pose_sequence"),
        (TestTubeTaskStage.RECLAMP, "reclamp", "moveit_cartesian_pose_sequence"),
    )
    for stage, pose_name, component in pose_stages:
        preview.append(
            {
                "stage": stage.value,
                "component": component,
                "pose_name": pose_name,
                "pose": task.poses[pose_name].as_dict(),
                "execute": execute,
            }
        )
    if "entry_standoff" in task.poses:
        preview.append(
            {
                "stage": TestTubeTaskStage.APPROACH_ENTRY.value,
                "component": "moveit_pose_goal",
                "pose_name": "entry_standoff",
                "pose": task.poses["entry_standoff"].as_dict(),
                "execute": execute,
            }
        )
    preview.extend(
        [
            {
                "stage": TestTubeTaskStage.GRASP_EXTRACT.value,
                "component": "hande_modbus_cli",
                "operation": "bounded_close_then_verify_retained_width",
                "bilateral_tactile_required_before_extract": True,
                "execute": execute,
            },
            {
                "stage": TestTubeTaskStage.PIVOT.value,
                "component": "test_tube_tactile_controller",
                "operation": "0.25mm_bounded_open_steps_contact_supervision",
                "front_rgb_is_transition_authority": True,
                "tactile_axis_mode": task.tactile_axis_mode,
                "execute": execute,
            },
            {
                "stage": TestTubeTaskStage.GUARDED_INSERT.value,
                "component": "test_tube_tactile_controller",
                "operation": "microstep_progress_and_qualified_resistance_jam_gate",
                "entry_standoff_barrier": True,
                "guarded_insertion_blockers": list(task.guarded_insertion_blockers()),
                "execute": execute and not task.guarded_insertion_blockers(),
            },
        ]
    )
    return preview


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskContractError(f"{name} must be an object")
    return value


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TaskContractError(f"{name} must be a non-empty trimmed string")
    if any(ord(character) < 0x20 for character in value):
        raise TaskContractError(f"{name} contains control characters")
    return value


def _absolute_device_path(value: object, *, name: str) -> str:
    result = _identifier(value, name=name)
    if not result.startswith("/dev/"):
        raise TaskContractError(f"{name} must be an absolute /dev path")
    return result


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TaskContractError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise TaskContractError(f"{name} must be >= {minimum}")
    if strictly_positive and result <= 0.0:
        raise TaskContractError(f"{name} must be positive")
    return result


def _bounded_positive(value: object, *, name: str, maximum: float) -> float:
    result = _number(value, name=name, minimum=0.0, strictly_positive=True)
    if result > maximum:
        raise TaskContractError(f"{name} must be <= {maximum}")
    return result


def _integer(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TaskContractError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise TaskContractError(f"{name} must lie in [{minimum}, {maximum}]")
    return value


def _vector(value: object, *, length: int, name: str) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise TaskContractError(f"{name} must contain {length} finite numbers")
    return tuple(_number(item, name=name) for item in value)


def _unit_vector(value: object, *, name: str) -> tuple[float, float, float]:
    vector = np.asarray(_vector(value, length=3, name=name), dtype=float)
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, abs_tol=1e-6):
        raise TaskContractError(f"{name} must be unit length")
    return float(vector[0]), float(vector[1]), float(vector[2])


__all__ = [
    "GUARDED_LOCALIZATION_RECEIPT_SCHEMA",
    "TEST_TUBE_TASK_INPUT_SCHEMA",
    "GeometryEvidence",
    "GeometryStatus",
    "GuardedLocalizationReceipt",
    "RunnerMode",
    "StageEvidence",
    "StageTransition",
    "TaskContractError",
    "TestTubeTaskInput",
    "TestTubeTaskMachine",
    "TestTubeTaskStage",
    "ToolPose",
    "command_preview",
]
