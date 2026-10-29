"""Qualified downstream scene interpretation for the seven-stage tube task.

The source-tube F2G tracker stops being the right pose producer once the
gripper occludes the transparent tube.  This module supplies the complementary
producer used after contact:

* robot kinematics carry the actual contact tube/TCP relation through extract;
* a reviewed front-RGB line, a calibrated camera, and a qualified pivot plane
  recover the tube axis during the gravity pivot;
* the relation measured at tactile-confirmed reclamp is frozen and propagated
  from the current TCP thereafter;
* a measured static socket pose supplies all metric rack constraints while RGB
  only confirms that the visible tube agrees with those constraints.

An image line never becomes a metric socket pose.  Missing calibration,
tracking loss, an ambiguous line, or missing current TCP state produces false
predicates/``None`` pose evidence and therefore pauses the resident controller.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .franka import validate_rigid_transform
from .test_tube_preflight import TubeState
from .test_tube_rgb_tracker import (
    ImageLineCandidate,
    TubeAxisCandidateTracker,
    TubeAxisTrackerConfig,
    extract_hough_line_candidates,
)
from .test_tube_visual import (
    VisualAngleGate,
    VisualAngleGateConfig,
    axis_angle_error_deg,
)

RGB_TRACKER_QUALIFICATION_SCHEMA = "demo2gripper.test_tube_rgb_tracker_qualification.v2"
LIVE_QUALIFICATION_SCHEMA = "demo2gripper.test_tube_live_io_qualification.v1"
PIVOT_REFERENCE_SCHEMA = "demo2gripper.test_tube_front_rgb_pivot_reference.v1"


class DownstreamSceneError(RuntimeError):
    """Downstream evidence cannot be promoted safely."""


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unit(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(result))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be non-zero")
    return result / norm


def _line(value: object, name: str) -> ImageLineCandidate:
    result = np.asarray(value, dtype=float)
    if result.shape != (2, 2) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite [[u0,v0],[u1,v1]]")
    return ImageLineCandidate(tuple(result[0]), tuple(result[1]))


def _roi(value: object) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError("axis_roi_uvxy must contain four integers")
    u0, v0, u1, v1 = (int(item) for item in value)
    if not 0 <= u0 < u1 or not 0 <= v0 < v1:
        raise ValueError("axis_roi_uvxy must have positive extent")
    return u0, v0, u1, v1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_transform(path: Path, name: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        payload = _json(path)
        value = payload.get("matrix", payload.get("T", payload.get("transform")))
        if isinstance(value, str):
            return _load_transform((path.parent / value).resolve(), name)
    return validate_rigid_transform(np.asarray(value, dtype=float), name=name).copy()


def _require_live_receipt(
    path: Path,
    *,
    kind: str,
    artifact: Path | None = None,
) -> dict[str, Any]:
    receipt = _json(path)
    valid = bool(
        receipt.get("schema") == LIVE_QUALIFICATION_SCHEMA
        and receipt.get("kind") == kind
        and receipt.get("runtime_qualified") is True
        and isinstance(receipt.get("qualification_id"), str)
        and receipt["qualification_id"]
    )
    if kind in {"fixed_camera_extrinsic", "static_socket_pose", "tube_scene_interpreter"}:
        for field in ("position_std_m", "orientation_std_deg"):
            value = receipt.get(field)
            valid = bool(
                valid
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0.0
            )
    if artifact is not None:
        valid = bool(valid and receipt.get("artifact_sha256") == _sha256(artifact))
    if not valid:
        raise DownstreamSceneError(f"{kind} qualification receipt is invalid")
    return receipt


@dataclass(frozen=True)
class DownstreamScenePolicy:
    """Fixed geometric and temporal limits backed by the scene qualification."""

    tube_length_m: float | None
    extract_clearance_m: float = 0.030
    visual_axis_consistency_deg: float = 6.0
    visual_line_distance_px: float = 18.0
    pivot_projection_error_px: float = 12.0
    socket_axis_tolerance_deg: float = 5.0
    socket_tip_radial_tolerance_m: float = 0.006
    socket_tip_axial_tolerance_m: float = 0.008
    insertion_target_depth_m: float | None = 0.020
    support_position_tolerance_m: float = 0.008
    support_axis_tolerance_deg: float = 6.0
    support_min_consecutive_frames: int = 4
    support_max_centroid_jitter_px: float = 5.0
    support_max_axis_jitter_deg: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "extract_clearance_m",
            "visual_axis_consistency_deg",
            "visual_line_distance_px",
            "pivot_projection_error_px",
            "socket_axis_tolerance_deg",
            "socket_tip_radial_tolerance_m",
            "socket_tip_axial_tolerance_m",
            "support_position_tolerance_m",
            "support_axis_tolerance_deg",
            "support_max_centroid_jitter_px",
            "support_max_axis_jitter_deg",
        ):
            _positive(getattr(self, name), name)
        if self.tube_length_m is not None:
            _positive(self.tube_length_m, "tube_length_m")
        if self.insertion_target_depth_m is not None:
            _positive(self.insertion_target_depth_m, "insertion_target_depth_m")
        _positive_int(self.support_min_consecutive_frames, "support_min_consecutive_frames")
        if self.visual_axis_consistency_deg >= 45.0:
            raise ValueError("visual_axis_consistency_deg must be below 45 degrees")
        if self.socket_axis_tolerance_deg >= 45.0:
            raise ValueError("socket_axis_tolerance_deg must be below 45 degrees")


@dataclass(frozen=True)
class HeldTubeSceneDiagnostics:
    """Latest independently inspectable downstream result."""

    frame_sequence: int | None
    state: str | None
    tracker_decision: str
    hold_reasons: tuple[str, ...]
    visual_metric_consistent: bool
    tube_clear_of_source: bool
    angle_target_reached: bool
    tube_socket_axes_aligned: bool
    tube_tip_at_socket_entry: bool
    tube_stays_supported: bool


CandidateExtractor = Callable[[np.ndarray, tuple[int, int, int, int]], Sequence[ImageLineCandidate]]


class QualifiedHeldTubeSceneInterpreter:
    """Downstream ``SceneInterpreter`` for extract through release."""

    _RIGID_HELD_STATES = {
        TubeState.EXTRACT,
        TubeState.LEVEL_HELD_TUBE,
    }
    _PIVOT_STATES = {
        TubeState.PIVOT_MICRO_LOOSEN,
        TubeState.PIVOT_EVALUATE,
        TubeState.RECLAMP,
        TubeState.RECLAMP_RECOVERY,
    }
    _KINEMATIC_STATES = {
        TubeState.APPROACH_SOCKET,
        TubeState.ALIGN_SOCKET,
        TubeState.INSERT_MICROSTEP,
        TubeState.INSERT_EVALUATE,
        TubeState.JAM_RECOVERY_RETRACT,
        TubeState.RELEASE,
    }

    def __init__(
        self,
        *,
        T_base_camera: np.ndarray,
        T_tube_tcp_demo: np.ndarray | None,
        T_base_socket: np.ndarray,
        T_socket_tube_entry: np.ndarray,
        socket_outward_axis_socket: Sequence[float],
        pivot_plane_normal_tcp: Sequence[float],
        pivot_point_tcp: Sequence[float],
        jaw_axis_tcp: Sequence[float],
        reviewed_seed_line: ImageLineCandidate,
        axis_roi_uvxy: tuple[int, int, int, int],
        tracker_config: TubeAxisTrackerConfig,
        angle_config: VisualAngleGateConfig,
        policy: DownstreamScenePolicy,
        tracking_id: str,
        evidence_id: str,
        position_std_m: float,
        orientation_std_deg: float,
        expected_camera_serial: str,
        expected_K: np.ndarray | None,
        candidate_extractor: CandidateExtractor | None = None,
        socket_pose_ready: bool = True,
        defer_visual_seed_until_contact: bool = False,
    ) -> None:
        self._T_base_camera = validate_rigid_transform(T_base_camera, name="T_base_camera").copy()
        self._T_camera_base = np.linalg.inv(self._T_base_camera)
        # A configured demonstration transform is only a planning candidate.
        # In normalized top-down mode there is no full transform here at all;
        # in legacy mode Servo/recentering can still make the contact relation
        # differ from the candidate.  Neither case is allowed to drive held
        # pose propagation until ``set_grasp_relation`` freezes the relation
        # and the tube pose from the same contact observation.
        self._T_tube_tcp_demo_candidate = (
            None
            if T_tube_tcp_demo is None
            else validate_rigid_transform(T_tube_tcp_demo, name="T_tube_tcp_demo candidate").copy()
        )
        self._T_tube_tcp_actual: np.ndarray | None = None
        self._T_tcp_tube_actual: np.ndarray | None = None
        self._T_base_socket = validate_rigid_transform(T_base_socket, name="T_base_socket").copy()
        self._socket_pose_ready = bool(socket_pose_ready)
        self._T_socket_tube_entry = validate_rigid_transform(
            T_socket_tube_entry, name="T_socket_tube_entry"
        ).copy()
        self._socket_outward_axis_socket = _unit(
            socket_outward_axis_socket, "socket_outward_axis_socket"
        )
        self._pivot_plane_normal_tcp = _unit(pivot_plane_normal_tcp, "pivot_plane_normal_tcp")
        self._pivot_point_tcp = np.asarray(pivot_point_tcp, dtype=float)
        if self._pivot_point_tcp.shape != (3,) or not np.isfinite(self._pivot_point_tcp).all():
            raise ValueError("pivot_point_tcp must be a finite 3-vector")
        self._jaw_axis_tcp = _unit(jaw_axis_tcp, "jaw_axis_tcp")
        self._pivot_point_tube: np.ndarray | None = None
        self._reviewed_seed_line = reviewed_seed_line
        self._axis_roi = axis_roi_uvxy
        self._tracker = TubeAxisCandidateTracker(tracker_config)
        self._angle_gate = VisualAngleGate(angle_config)
        self._policy = policy
        self._tube_length_m = policy.tube_length_m
        if not isinstance(tracking_id, str) or not tracking_id:
            raise ValueError("tracking_id must be non-empty")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("evidence_id must be non-empty")
        self._tracking_id = tracking_id
        self._evidence_id = evidence_id
        self._position_std_m = float(position_std_m)
        self._orientation_std_deg = float(orientation_std_deg)
        if (
            not math.isfinite(self._position_std_m)
            or self._position_std_m < 0.0
            or not math.isfinite(self._orientation_std_deg)
            or self._orientation_std_deg < 0.0
        ):
            raise ValueError("pose uncertainty must be finite and non-negative")
        if not isinstance(expected_camera_serial, str) or not expected_camera_serial:
            raise ValueError("expected_camera_serial must be non-empty")
        self._expected_camera_serial = expected_camera_serial
        self._expected_K = None if expected_K is None else np.asarray(expected_K, dtype=float)
        if self._expected_K is not None and (
            self._expected_K.shape != (3, 3) or not np.isfinite(self._expected_K).all()
        ):
            raise ValueError("expected_K must be finite 3x3")
        self._extract_candidates = candidate_extractor or self._default_candidates
        self._defer_visual_seed_until_contact = bool(defer_visual_seed_until_contact)
        self._latest_line: ImageLineCandidate | None = None
        self._latest_pose: np.ndarray | None = None
        self._latest_frame_sequence: int | None = None
        self._latest_timestamp_s: float | None = None
        self._last_axis_base: np.ndarray | None = None
        self._T_base_tube_at_contact: np.ndarray | None = None
        self._source_min_tip_z_m: float | None = None
        self._runtime_grasp_relation_set = False
        self._downstream_observation_started = False
        self._T_tcp_tube_frozen: np.ndarray | None = None
        self._T_base_tube_at_seated: np.ndarray | None = None
        self._support_history: deque[tuple[float, float, float, float]] = deque(
            maxlen=policy.support_min_consecutive_frames
        )
        self._diagnostics = HeldTubeSceneDiagnostics(
            None,
            None,
            "UNINITIALIZED",
            ("no_frame_observed",),
            False,
            False,
            False,
            False,
            False,
            False,
        )
        self._closed = False

    @property
    def diagnostics(self) -> HeldTubeSceneDiagnostics:
        return self._diagnostics

    def set_grasp_relation(
        self,
        T_tube_tcp_actual: np.ndarray,
        *,
        T_base_tube_at_contact: np.ndarray,
    ) -> None:
        """Freeze the actual contact relation and extraction baseline.

        The two transforms must come from the same final, stationary,
        bilateral-contact source observation.  A configured demo transform is
        deliberately not used as a fallback: normalized top-down grasping and
        Servo/recentering both make the runtime relation the authoritative one.
        """

        if self._closed:
            raise DownstreamSceneError("scene interpreter is closed")
        relation = validate_rigid_transform(
            T_tube_tcp_actual, name="runtime T_tube_tcp_actual"
        ).copy()
        contact = validate_rigid_transform(
            T_base_tube_at_contact, name="contact T_base_tube"
        ).copy()
        if self._runtime_grasp_relation_set:
            assert self._T_tube_tcp_actual is not None
            assert self._T_base_tube_at_contact is not None
            if not np.allclose(
                relation, self._T_tube_tcp_actual, atol=1e-9, rtol=0.0
            ) or not np.allclose(contact, self._T_base_tube_at_contact, atol=1e-9, rtol=0.0):
                raise DownstreamSceneError("runtime grasp relation/contact pose cannot be rebound")
            return
        if self._downstream_observation_started:
            raise DownstreamSceneError(
                "runtime grasp relation arrived after downstream motion observation"
            )

        inverse = np.linalg.inv(relation)
        pivot_point_tube = relation[:3, :3] @ self._pivot_point_tcp + relation[:3, 3]
        tips = self._tube_tips(contact)

        # Assign only after every validation/computation succeeds so callers
        # cannot leave a partially armed interpreter.
        self._T_tube_tcp_actual = relation
        self._T_tcp_tube_actual = inverse
        self._pivot_point_tube = pivot_point_tube
        self._T_base_tube_at_contact = contact
        self._source_min_tip_z_m = min(float(tips[0][2]), float(tips[1][2]))
        self._last_axis_base = contact[:3, 2].copy()
        self._runtime_grasp_relation_set = True

    def reset_support_history_for_release(self) -> None:
        """Start release support qualification from post-open RGB frames."""

        if self._closed:
            raise DownstreamSceneError("scene interpreter is closed")
        self._support_history.clear()

    def set_socket_pose(self, T_base_socket_entry: np.ndarray) -> None:
        """Update the temporally qualified online socket-entry frame.

        Runtime socket mode defines the socket frame at the entry itself, so
        ``T_socket_tube_entry`` is identity.  The online producer owns
        identity/temporal gates; this consumer only accepts a rigid transform.
        """

        if self._closed:
            raise DownstreamSceneError("scene interpreter is closed")
        self._T_base_socket = validate_rigid_transform(
            T_base_socket_entry, name="online T_base_socket_entry"
        ).copy()
        self._socket_pose_ready = True

    def set_tube_length_m(self, tube_length_m: float) -> None:
        """Freeze the temporally qualified F2G long-axis length once."""

        length = _positive(tube_length_m, "runtime tube_length_m")
        if self._tube_length_m is not None and not math.isclose(
            self._tube_length_m, length, rel_tol=0.05, abs_tol=0.002
        ):
            raise DownstreamSceneError("runtime tube length changed after qualification")
        self._tube_length_m = length

    def observe_tactile(self, *, state: TubeState, tactile: object, T_base_tcp: np.ndarray) -> None:
        """Freeze the kinematic tube pose at a qualified seated event."""

        if (
            state is TubeState.INSERT_EVALUATE
            and bool(getattr(tactile, "valid", False))
            and bool(getattr(tactile, "insertion_seated", False))
            and not bool(getattr(tactile, "insertion_jam", False))
            and self._T_tcp_tube_frozen is not None
        ):
            tcp = validate_rigid_transform(T_base_tcp, name="seated T_base_tcp")
            self._T_base_tube_at_seated = tcp @ self._T_tcp_tube_frozen

    def observe(
        self,
        *,
        frame: object,
        state: TubeState,
        T_base_tcp: np.ndarray,
    ) -> object:
        """Interpret one frame with the current robot pose.

        ``T_base_tcp`` is required.  Falling back to the previous TCP would
        silently turn a moving-arm alignment predicate into stale evidence.
        """

        from .test_tube_live_io import SceneEvidence

        if self._closed:
            raise DownstreamSceneError("scene interpreter is closed")
        if not isinstance(state, TubeState):
            state = TubeState(state)
        if state in self._RIGID_HELD_STATES or state in self._PIVOT_STATES:
            self._downstream_observation_started = True
        tcp = validate_rigid_transform(T_base_tcp, name="current T_base_tcp")
        self._validate_frame_contract(frame)
        timestamp = float(getattr(frame, "host_received_monotonic_s"))
        sequence = int(getattr(frame, "sequence"))
        image = np.asarray(getattr(frame, "rgb"))
        downstream_state = bool(
            state in self._RIGID_HELD_STATES
            or state in self._PIVOT_STATES
            or state in self._KINEMATIC_STATES
        )
        if self._defer_visual_seed_until_contact and not downstream_state:
            return SceneEvidence(tube=None, socket=None)
        if (downstream_state) and self._tube_length_m is None:
            self._diagnostics = HeldTubeSceneDiagnostics(
                sequence,
                state.value,
                "HOLD_AND_REOBSERVE",
                ("runtime_f2g_tube_length_missing",),
                False,
                False,
                False,
                False,
                False,
                False,
            )
            return SceneEvidence(tube=None, socket=None)
        try:
            candidates = tuple(self._extract_candidates(image, self._axis_roi))
        except Exception as error:  # noqa: BLE001
            candidates = ()
            tracker_decision = "HOLD_AND_REOBSERVE"
            tracker_reason = f"candidate_extraction_failed:{type(error).__name__}"
        else:
            tracker_decision, tracker_reason = self._update_tracker(
                timestamp=timestamp,
                candidates=candidates,
                state=state,
                T_base_tcp=tcp,
                K=self._expected_K,
            )

        track_valid = self._latest_line is not None and tracker_decision in {
            "TRACKED",
            "MODEL_RESEEDED",
        }
        pose = self._pose_for_state(state=state, T_base_tcp=tcp, track_valid=track_valid)
        self._latest_pose = None if pose is None else pose.copy()
        self._latest_frame_sequence = sequence
        self._latest_timestamp_s = timestamp
        visual_consistent = bool(
            track_valid
            and pose is not None
            and self._visual_metric_consistent(pose, self._latest_line, self._expected_K)
        )

        clear = self._clear_of_source(state, pose, visual_consistent)
        angle_ready = self._angle_ready(
            state=state,
            timestamp=timestamp,
            tracker_valid=(track_valid and pose is not None and self._runtime_grasp_relation_set),
        )
        axes_aligned, tip_at_entry = self._socket_entry_predicates(
            state=state,
            pose=pose,
            visual_consistent=visual_consistent,
        )
        supported = self._support_predicate(
            state=state,
            timestamp=timestamp,
            pose=pose,
            visual_consistent=visual_consistent,
        )
        reasons: list[str] = []
        if not track_valid:
            reasons.append(tracker_reason)
        if (
            state in self._RIGID_HELD_STATES or state in self._PIVOT_STATES
        ) and not self._runtime_grasp_relation_set:
            reasons.append("runtime_grasp_relation_not_set")
        if pose is None and state in (
            self._RIGID_HELD_STATES | self._PIVOT_STATES | self._KINEMATIC_STATES
        ):
            reasons.append("held_tube_pose_unavailable")
        if pose is not None and not visual_consistent:
            reasons.append("front_rgb_disagrees_with_metric_tube_pose")
        self._diagnostics = HeldTubeSceneDiagnostics(
            frame_sequence=sequence,
            state=state.value,
            tracker_decision=tracker_decision,
            hold_reasons=tuple(dict.fromkeys(reasons)),
            visual_metric_consistent=visual_consistent,
            tube_clear_of_source=clear,
            angle_target_reached=angle_ready,
            tube_socket_axes_aligned=axes_aligned,
            tube_tip_at_socket_entry=tip_at_entry,
            tube_stays_supported=supported,
        )
        return SceneEvidence(
            tube=None,
            socket=None,
            front_rgb_angle_target_reached=angle_ready,
            tube_clear_of_source=clear,
            tube_tip_at_socket_entry=tip_at_entry,
            tube_socket_axes_aligned=axes_aligned,
            tube_stays_supported=supported,
        )

    def qualify_pose(self, **_: object) -> object:
        raise DownstreamSceneError(
            "the downstream interpreter does not promote F2G/socket image proposals"
        )

    def held_tube_pose(
        self,
        *,
        frame: object,
        scene: object,
        state: TubeState,
        T_base_tcp: np.ndarray,
        tactile: object,
    ) -> object | None:
        del scene
        from .test_tube_live_io import QualifiedBasePose

        if not isinstance(state, TubeState):
            state = TubeState(state)
        if int(getattr(frame, "sequence")) != self._latest_frame_sequence:
            return None
        if self._latest_pose is None:
            return None
        tcp = validate_rigid_transform(T_base_tcp, name="current T_base_tcp")
        pose = self._latest_pose.copy()
        if (
            state in {TubeState.RECLAMP, TubeState.RECLAMP_RECOVERY}
            and bool(getattr(tactile, "bilateral_contact", False))
            and bool(getattr(tactile, "rotation_stopped", False))
        ):
            self._T_tcp_tube_frozen = np.linalg.inv(tcp) @ pose
        return QualifiedBasePose(
            T_base_object=pose,
            position_std_m=self._position_std_m,
            orientation_std_deg=self._orientation_std_deg,
            tracking_id=self._tracking_id,
            evidence_id=(
                f"{self._evidence_id}:frame-{self._latest_frame_sequence}:"
                f"{self._diagnostics.tracker_decision.lower()}"
            ),
            axis_sign_source="temporal_continuity",
            cylindrical_roll_canonicalized=True,
        )

    def close(self) -> None:
        self._closed = True
        self._support_history.clear()

    def _validate_frame_contract(self, frame: object) -> None:
        serial = str(getattr(frame, "serial"))
        if serial != self._expected_camera_serial:
            raise DownstreamSceneError("front camera serial changed from qualification")
        K = np.asarray(getattr(getattr(frame, "intrinsics"), "matrix"), dtype=float)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise DownstreamSceneError("front camera intrinsics are invalid")
        if self._expected_K is None:
            self._expected_K = K.copy()
        elif not np.allclose(K, self._expected_K, atol=1e-9, rtol=0.0):
            raise DownstreamSceneError("front camera intrinsics changed from qualification")
        image = np.asarray(getattr(frame, "rgb"))
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise DownstreamSceneError("front RGB frame contract is invalid")
        u0, v0, u1, v1 = self._axis_roi
        if u1 > image.shape[1] or v1 > image.shape[0]:
            raise DownstreamSceneError("qualified tube ROI lies outside the current image")

    @staticmethod
    def _default_candidates(
        image: np.ndarray, roi: tuple[int, int, int, int]
    ) -> Sequence[ImageLineCandidate]:
        return extract_hough_line_candidates(image, roi_uvxy=roi)

    def _update_tracker(
        self,
        *,
        timestamp: float,
        candidates: Sequence[ImageLineCandidate],
        state: TubeState,
        T_base_tcp: np.ndarray,
        K: np.ndarray,
    ) -> tuple[str, str]:
        if not self._tracker.seeded:
            prediction = self._predicted_line(state, T_base_tcp, K)
            reference = self._reviewed_seed_line if prediction is None else prediction
            selected = self._unique_candidate_near(reference, candidates)
            if selected is None:
                self._latest_line = None
                return "RESEED_REQUIRED", "no_unique_candidate_matches_qualified_seed"
            self._tracker.seed(timestamp_s=timestamp, line=selected)
            self._latest_line = selected
            return "MODEL_RESEEDED", "unique candidate matched qualified seed/model"
        result = self._tracker.update(
            timestamp_s=timestamp,
            candidates=candidates,
            roi_uvxy=tuple(float(value) for value in self._axis_roi),
        )
        if result.decision == "TRACKED" and result.selected_line is not None:
            self._latest_line = result.selected_line
        else:
            self._latest_line = None
        return result.decision, result.reason

    def _predicted_line(
        self, state: TubeState, T_base_tcp: np.ndarray, K: np.ndarray
    ) -> ImageLineCandidate | None:
        if state in self._RIGID_HELD_STATES:
            if self._T_tcp_tube_actual is None:
                return None
            pose = T_base_tcp @ self._T_tcp_tube_actual
        elif self._T_tcp_tube_frozen is not None and state in self._KINEMATIC_STATES:
            pose = T_base_tcp @ self._T_tcp_tube_frozen
        elif self._latest_pose is not None:
            pose = self._latest_pose
        else:
            return None
        return self._project_axis_line(pose, K)

    def _unique_candidate_near(
        self,
        reference: ImageLineCandidate,
        candidates: Sequence[ImageLineCandidate],
    ) -> ImageLineCandidate | None:
        scored: list[tuple[float, ImageLineCandidate]] = []
        for candidate in candidates:
            angle = abs(axis_angle_error_deg(candidate.axis_angle_deg, reference.axis_angle_deg))
            distance = self._point_line_distance(candidate.midpoint_uv, reference)
            if (
                angle <= self._policy.visual_axis_consistency_deg
                and distance <= self._policy.visual_line_distance_px
            ):
                score = angle / self._policy.visual_axis_consistency_deg + distance / (
                    self._policy.visual_line_distance_px
                )
                scored.append((score, candidate))
        scored.sort(key=lambda item: item[0])
        if not scored:
            return None
        if len(scored) > 1 and scored[1][0] - scored[0][0] < 0.15:
            first, second = scored[0][1], scored[1][1]
            mutual = max(
                self._point_line_distance(first.midpoint_uv, second),
                self._point_line_distance(second.midpoint_uv, first),
            )
            if mutual > self._policy.visual_line_distance_px:
                return None
        return scored[0][1]

    def _pose_for_state(
        self, *, state: TubeState, T_base_tcp: np.ndarray, track_valid: bool
    ) -> np.ndarray | None:
        if state in self._RIGID_HELD_STATES:
            if (
                not self._runtime_grasp_relation_set
                or self._T_tcp_tube_actual is None
                or self._T_base_tube_at_contact is None
            ):
                return None
            pose = T_base_tcp @ self._T_tcp_tube_actual
            self._last_axis_base = pose[:3, 2].copy()
            return validate_rigid_transform(pose, name="extracted T_base_tube").copy()
        if state in self._PIVOT_STATES:
            if (
                not self._runtime_grasp_relation_set
                or self._pivot_point_tube is None
                or not track_valid
                or self._latest_line is None
            ):
                return None
            return self._reconstruct_pivot_pose(T_base_tcp, self._latest_line)
        if state in self._KINEMATIC_STATES:
            if self._T_tcp_tube_frozen is None:
                return None
            pose = T_base_tcp @ self._T_tcp_tube_frozen
            self._last_axis_base = pose[:3, 2].copy()
            return validate_rigid_transform(pose, name="post-pivot T_base_tube").copy()
        return None

    def _reconstruct_pivot_pose(
        self, T_base_tcp: np.ndarray, line: ImageLineCandidate
    ) -> np.ndarray | None:
        if self._pivot_point_tube is None:
            return None
        K_inv = np.linalg.inv(self._expected_K)
        ray0 = K_inv @ np.array([*line.start_uv, 1.0])
        ray1 = K_inv @ np.array([*line.end_uv, 1.0])
        image_plane_normal_camera = np.cross(ray0, ray1)
        norm = float(np.linalg.norm(image_plane_normal_camera))
        if norm <= 1e-9:
            return None
        image_plane_normal_base = self._T_base_camera[:3, :3] @ (image_plane_normal_camera / norm)
        pivot_plane_normal_base = T_base_tcp[:3, :3] @ self._pivot_plane_normal_tcp
        axis = np.cross(image_plane_normal_base, pivot_plane_normal_base)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            return None
        axis /= axis_norm
        if self._last_axis_base is not None and float(axis @ self._last_axis_base) < 0.0:
            axis *= -1.0
        jaw_base = T_base_tcp[:3, :3] @ self._jaw_axis_tcp
        x_axis = jaw_base - float(jaw_base @ axis) * axis
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm <= 1e-6:
            return None
        x_axis /= x_norm
        y_axis = np.cross(axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        rotation = np.column_stack((x_axis, y_axis, axis))
        pivot_base = T_base_tcp[:3, :3] @ self._pivot_point_tcp + T_base_tcp[:3, 3]
        centre = pivot_base - rotation @ self._pivot_point_tube
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = centre
        projected = self._project_axis_line(pose, self._expected_K)
        if projected is None:
            return None
        if self._point_line_distance(projected.midpoint_uv, line) > (
            self._policy.pivot_projection_error_px
        ):
            return None
        self._last_axis_base = axis.copy()
        return validate_rigid_transform(pose, name="pivot T_base_tube").copy()

    def _project_axis_line(
        self, T_base_tube: np.ndarray, K: np.ndarray
    ) -> ImageLineCandidate | None:
        assert self._tube_length_m is not None
        half = 0.5 * self._tube_length_m
        points_tube = np.array([[0.0, 0.0, -half, 1.0], [0.0, 0.0, half, 1.0]]).T
        points_camera = self._T_camera_base @ T_base_tube @ points_tube
        if np.any(points_camera[2] <= 1e-5):
            return None
        pixels = K @ points_camera[:3]
        pixels = (pixels[:2] / pixels[2]).T
        if float(np.linalg.norm(pixels[1] - pixels[0])) <= 1e-6:
            return None
        return ImageLineCandidate(tuple(pixels[0]), tuple(pixels[1]))

    def _visual_metric_consistent(
        self,
        pose: np.ndarray,
        line: ImageLineCandidate | None,
        K: np.ndarray,
    ) -> bool:
        if line is None:
            return False
        projected = self._project_axis_line(pose, K)
        if projected is None:
            return False
        return bool(
            abs(axis_angle_error_deg(line.axis_angle_deg, projected.axis_angle_deg))
            <= self._policy.visual_axis_consistency_deg
            and self._point_line_distance(line.midpoint_uv, projected)
            <= self._policy.visual_line_distance_px
        )

    def _clear_of_source(
        self, state: TubeState, pose: np.ndarray | None, visual_consistent: bool
    ) -> bool:
        if (
            state is not TubeState.EXTRACT
            or pose is None
            or not self._runtime_grasp_relation_set
            or self._source_min_tip_z_m is None
        ):
            return False
        tips = self._tube_tips(pose)
        current_min = min(float(tips[0][2]), float(tips[1][2]))
        return bool(
            visual_consistent
            and current_min - self._source_min_tip_z_m >= self._policy.extract_clearance_m
        )

    def _angle_ready(self, *, state: TubeState, timestamp: float, tracker_valid: bool) -> bool:
        if state not in self._PIVOT_STATES:
            return False
        if not tracker_valid or self._latest_line is None:
            from .test_tube_visual import TubeAxisObservation

            invalid = TubeAxisObservation(None, None, 0.0, math.inf, 0.0, 0, False)
            self._angle_gate.update(timestamp_s=timestamp, observation=invalid)
            return False
        from .test_tube_visual import TubeAxisObservation

        line = self._latest_line
        observation = TubeAxisObservation(
            line.axis_angle_deg,
            line.midpoint_uv,
            line.length_px,
            0.0,
            1.0,
            max(2, round(line.length_px)),
            True,
        )
        return self._angle_gate.update(timestamp_s=timestamp, observation=observation).accepted

    def _socket_entry_predicates(
        self,
        *,
        state: TubeState,
        pose: np.ndarray | None,
        visual_consistent: bool,
    ) -> tuple[bool, bool]:
        if state is not TubeState.ALIGN_SOCKET or pose is None or not visual_consistent:
            return False, False
        if not self._socket_pose_ready:
            return False, False
        target = self._T_base_socket @ self._T_socket_tube_entry
        angle = self._axis_angle_deg(pose[:3, 2], target[:3, 2])
        current_tip = self._tube_tips(pose)[0]
        target_tip = self._tube_tips(target)[0]
        delta = current_tip - target_tip
        axis = target[:3, 2]
        axial = abs(float(delta @ axis))
        radial = float(np.linalg.norm(delta - float(delta @ axis) * axis))
        aligned = angle <= self._policy.socket_axis_tolerance_deg
        at_entry = bool(
            aligned
            and radial <= self._policy.socket_tip_radial_tolerance_m
            and axial <= self._policy.socket_tip_axial_tolerance_m
        )
        return aligned, at_entry

    def _support_predicate(
        self,
        *,
        state: TubeState,
        timestamp: float,
        pose: np.ndarray | None,
        visual_consistent: bool,
    ) -> bool:
        if (
            state is not TubeState.RELEASE
            or pose is None
            or not visual_consistent
            or self._latest_line is None
            or not self._socket_pose_ready
        ):
            self._support_history.clear()
            return False
        if self._policy.insertion_target_depth_m is None:
            if self._T_base_tube_at_seated is None:
                self._support_history.clear()
                return False
            target = self._T_base_tube_at_seated
        else:
            target = self._T_base_socket @ self._T_socket_tube_entry
            inward = -(self._T_base_socket[:3, :3] @ self._socket_outward_axis_socket)
            target = target.copy()
            target[:3, 3] += inward * self._policy.insertion_target_depth_m
        position_error = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]))
        angle_error = self._axis_angle_deg(pose[:3, 2], target[:3, 2])
        if (
            position_error > self._policy.support_position_tolerance_m
            or angle_error > self._policy.support_axis_tolerance_deg
        ):
            self._support_history.clear()
            return False
        self._support_history.append(
            (
                timestamp,
                self._latest_line.axis_angle_deg,
                float(self._latest_line.midpoint_uv[0]),
                float(self._latest_line.midpoint_uv[1]),
            )
        )
        if len(self._support_history) < self._policy.support_min_consecutive_frames:
            return False
        angles = [item[1] for item in self._support_history]
        centres = [np.asarray(item[2:4], dtype=float) for item in self._support_history]
        angle_jitter = max(abs(axis_angle_error_deg(value, angles[0])) for value in angles[1:])
        centre_jitter = max(float(np.linalg.norm(value - centres[0])) for value in centres[1:])
        return bool(
            angle_jitter <= self._policy.support_max_axis_jitter_deg
            and centre_jitter <= self._policy.support_max_centroid_jitter_px
        )

    def _tube_tips(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._tube_length_m is not None
        half = 0.5 * self._tube_length_m
        centre = pose[:3, 3]
        axis = pose[:3, 2]
        return centre - half * axis, centre + half * axis

    @staticmethod
    def _axis_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
        cosine = float(np.clip(np.asarray(first) @ np.asarray(second), -1.0, 1.0))
        return math.degrees(math.acos(cosine))

    @staticmethod
    def _point_line_distance(point_uv: Sequence[float], line: ImageLineCandidate) -> float:
        start = np.asarray(line.start_uv, dtype=float)
        delta = np.asarray(line.end_uv, dtype=float) - start
        direction = delta / np.linalg.norm(delta)
        relative = np.asarray(point_uv, dtype=float) - start
        return float(abs(direction[0] * relative[1] - direction[1] * relative[0]))


def create_qualified_held_tube_scene_interpreter(
    *, config: Mapping[str, object], output_dir: object
) -> QualifiedHeldTubeSceneInterpreter:
    """Factory for ``StaticSourceTubeSceneInterpreter.downstream_factory``.

    The factory validates every qualification before returning an interpreter;
    it opens no camera, ROS, gripper, or robot handle.
    """

    del output_dir
    if not isinstance(config, Mapping):
        raise ValueError("downstream scene config must be an object")

    def path(name: str) -> Path:
        value = config.get(name)
        if not isinstance(value, str) or not value:
            raise DownstreamSceneError(f"{name} is required")
        result = Path(value).expanduser().resolve()
        if not result.is_file():
            raise DownstreamSceneError(f"{name} does not exist: {result}")
        return result

    camera_path = path("camera_extrinsic")
    camera_receipt = _require_live_receipt(
        path("camera_extrinsic_qualification_receipt"),
        kind="fixed_camera_extrinsic",
        artifact=camera_path,
    )
    socket_pose_mode = str(config.get("socket_pose_mode", "static"))
    if socket_pose_mode not in {"static", "online_entry_frame"}:
        raise DownstreamSceneError("socket_pose_mode must be static or online_entry_frame")
    socket_path = None
    socket_receipt = None
    if socket_pose_mode == "static":
        socket_path = path("static_socket_pose")
        socket_receipt = _require_live_receipt(
            path("static_socket_pose_qualification_receipt"),
            kind="static_socket_pose",
            artifact=socket_path,
        )
    scene_receipt = _require_live_receipt(
        path("scene_qualification_receipt"), kind="tube_scene_interpreter"
    )
    workspace_ids = {
        receipt.get("workspace_calibration_id")
        for receipt in (camera_receipt, socket_receipt, scene_receipt)
        if receipt is not None
    }
    if len(workspace_ids) != 1 or None in workspace_ids or "" in workspace_ids:
        raise DownstreamSceneError(
            "camera, socket, and scene receipts must share one workspace_calibration_id"
        )

    rgb_report_path = path("rgb_tracker_qualification_report")
    rgb_report = _json(rgb_report_path)
    if (
        rgb_report.get("schema") != RGB_TRACKER_QUALIFICATION_SCHEMA
        or rgb_report.get("runtime_gate_eligible") is not True
        or not isinstance(rgb_report.get("qualification_id"), str)
        or not rgb_report["qualification_id"]
    ):
        raise DownstreamSceneError("front-RGB tracker is not runtime qualified")
    reference_path = path("pivot_reference")
    reference = _json(reference_path)
    if reference.get("schema") != PIVOT_REFERENCE_SCHEMA:
        raise DownstreamSceneError("pivot reference schema is invalid")
    source = rgb_report.get("source")
    if not isinstance(source, Mapping) or source.get("reference_config_sha256") != _sha256(
        reference_path
    ):
        raise DownstreamSceneError("RGB qualification does not bind the pivot reference")

    expected_serial = str(config.get("camera_serial", ""))
    camera_reference = reference.get("camera")
    if (
        not expected_serial
        or not isinstance(camera_reference, Mapping)
        or camera_reference.get("serial") != expected_serial
        or source.get("camera_serial") != expected_serial
    ):
        raise DownstreamSceneError("camera serial is not bound across RGB qualifications")
    intrinsics_path = path("intrinsics")
    expected_K = np.load(intrinsics_path, allow_pickle=False)
    intrinsics_hash = _sha256(intrinsics_path)
    if (
        camera_reference.get("intrinsics_sha256") != intrinsics_hash
        or source.get("intrinsics_sha256") != intrinsics_hash
    ):
        raise DownstreamSceneError("camera intrinsics are not bound across qualifications")

    target = reference.get("target")
    if not isinstance(target, Mapping):
        raise DownstreamSceneError("pivot reference target is missing")
    raw_tracker = config.get("tracker", {})
    raw_policy = config.get("policy", {})
    if not isinstance(raw_tracker, Mapping) or not isinstance(raw_policy, Mapping):
        raise ValueError("tracker and policy must be objects")
    raw_hough = config.get("hough", {})
    if not isinstance(raw_hough, Mapping):
        raise ValueError("hough must be an object")
    hough_options = dict(raw_hough)

    def extractor(
        image: np.ndarray, roi_value: tuple[int, int, int, int]
    ) -> Sequence[ImageLineCandidate]:
        return extract_hough_line_candidates(image, roi_uvxy=roi_value, **hough_options)

    raw_demo_relation = config.get("T_tube_tcp_demo_grasp")
    demo_relation = (
        None if raw_demo_relation is None else np.asarray(raw_demo_relation, dtype=float)
    )
    socket_entry = np.asarray(config.get("T_socket_tube_entry"), dtype=float)
    if socket_pose_mode == "online_entry_frame" and not np.allclose(
        socket_entry, np.eye(4), atol=1e-9, rtol=0.0
    ):
        raise DownstreamSceneError("online_entry_frame requires identity T_socket_tube_entry")

    return QualifiedHeldTubeSceneInterpreter(
        T_base_camera=_load_transform(camera_path, "T_base_camera"),
        T_tube_tcp_demo=demo_relation,
        T_base_socket=(
            np.eye(4) if socket_path is None else _load_transform(socket_path, "T_base_socket")
        ),
        T_socket_tube_entry=socket_entry,
        socket_outward_axis_socket=config.get("socket_outward_axis_socket", [0, 0, 1]),
        pivot_plane_normal_tcp=config.get("pivot_plane_normal_tcp", [1, 0, 0]),
        pivot_point_tcp=config.get("pivot_point_tcp", [0, 0, 0]),
        jaw_axis_tcp=config.get("jaw_axis_tcp", [1, 0, 0]),
        reviewed_seed_line=_line(config.get("reviewed_seed_line_uv"), "reviewed_seed_line_uv"),
        axis_roi_uvxy=_roi(config.get("axis_roi_uvxy")),
        tracker_config=TubeAxisTrackerConfig(**dict(raw_tracker)),
        angle_config=VisualAngleGateConfig(
            target_image_axis_deg=float(target["target_image_axis_deg"]),
            tolerance_deg=float(target["tolerance_deg"]),
            min_consecutive_frames=int(target["min_consecutive_frames"]),
            max_rate_deg_s=float(target["max_rate_deg_s"]),
        ),
        policy=DownstreamScenePolicy(**dict(raw_policy)),
        tracking_id=str(config.get("tracking_id", "qualified-held-tube")),
        evidence_id=(
            f"{scene_receipt['qualification_id']}:rgb-{rgb_report['qualification_id']}:"
            f"workspace-{next(iter(workspace_ids))}"
        ),
        position_std_m=float(scene_receipt["position_std_m"]),
        orientation_std_deg=float(scene_receipt["orientation_std_deg"]),
        expected_camera_serial=expected_serial,
        expected_K=expected_K,
        candidate_extractor=extractor,
        socket_pose_ready=socket_pose_mode == "static",
    )


def create_online_held_tube_scene_interpreter(
    *, config: Mapping[str, object], output_dir: object
) -> QualifiedHeldTubeSceneInterpreter:
    """Build the held-tube interpreter with current-session RGB seeding.

    Unlike the artifact-backed factory, the first line is predicted from the
    realized contact relation and current FK.  Hough candidates must match
    that prediction before the tracker seeds, so no stale reviewed image line
    or pivot-reference file is consumed.
    """

    del output_dir
    if not isinstance(config, Mapping):
        raise ValueError("online held scene config must be an object")

    def required_path(name: str) -> Path:
        value = config.get(name)
        if not isinstance(value, str) or not value:
            raise DownstreamSceneError(f"{name} is required")
        result = Path(value).expanduser().resolve()
        if not result.is_file():
            raise DownstreamSceneError(f"{name} does not exist: {result}")
        return result

    raw_camera_transform = config.get("T_base_camera")
    if raw_camera_transform is None:
        T_base_camera = _load_transform(required_path("camera_extrinsic"), "T_base_camera")
    else:
        T_base_camera = validate_rigid_transform(
            np.asarray(raw_camera_transform, dtype=float),
            name="runtime T_base_camera",
        )
    intrinsics_value = config.get("intrinsics")
    expected_K = (
        None
        if intrinsics_value is None
        else np.load(required_path("intrinsics"), allow_pickle=False)
    )
    serial = str(config.get("camera_serial", ""))
    if not serial:
        raise DownstreamSceneError("camera_serial is required")
    raw_tracker = config.get("tracker", {})
    raw_hough = config.get("hough", {})
    raw_policy = config.get("policy", {})
    raw_angle = config.get("angle_gate", {})
    if not all(
        isinstance(item, Mapping) for item in (raw_tracker, raw_hough, raw_policy, raw_angle)
    ):
        raise ValueError("tracker, hough, policy, and angle_gate must be objects")
    hough_options = dict(raw_hough)

    def extractor(
        image: np.ndarray, roi_value: tuple[int, int, int, int]
    ) -> Sequence[ImageLineCandidate]:
        return extract_hough_line_candidates(image, roi_uvxy=roi_value, **hough_options)

    roi = _roi(config.get("axis_roi_uvxy", [0, 0, 640, 480]))
    return QualifiedHeldTubeSceneInterpreter(
        T_base_camera=T_base_camera,
        T_tube_tcp_demo=None,
        T_base_socket=np.eye(4),
        T_socket_tube_entry=np.eye(4),
        socket_outward_axis_socket=config.get("socket_outward_axis_socket", [0, 0, 1]),
        pivot_plane_normal_tcp=config.get("pivot_plane_normal_tcp", [1, 0, 0]),
        pivot_point_tcp=config.get("pivot_point_tcp", [0, 0, 0]),
        jaw_axis_tcp=config.get("jaw_axis_tcp", [1, 0, 0]),
        # It is never used as an authority: after contact `_predicted_line`
        # supplies the reseed reference.  A degenerate placeholder is rejected
        # by ImageLineCandidate, hence a one-pixel line is used.
        reviewed_seed_line=ImageLineCandidate((0.0, 0.0), (1.0, 0.0)),
        axis_roi_uvxy=roi,
        tracker_config=TubeAxisTrackerConfig(**dict(raw_tracker)),
        angle_config=VisualAngleGateConfig(
            target_image_axis_deg=float(raw_angle.get("target_image_axis_deg", 45.0)),
            tolerance_deg=float(raw_angle.get("tolerance_deg", 4.0)),
            min_consecutive_frames=int(raw_angle.get("min_consecutive_frames", 3)),
            max_rate_deg_s=float(raw_angle.get("max_rate_deg_s", 120.0)),
        ),
        policy=DownstreamScenePolicy(**dict(raw_policy)),
        tracking_id=str(config.get("tracking_id", "online-test-tube")),
        evidence_id="runtime-current-session-held-tube",
        position_std_m=float(config.get("max_position_std_m", 0.006)),
        orientation_std_deg=float(config.get("max_orientation_std_deg", 5.0)),
        expected_camera_serial=serial,
        expected_K=expected_K,
        candidate_extractor=extractor,
        socket_pose_ready=False,
        defer_visual_seed_until_contact=True,
    )


__all__ = [
    "DownstreamSceneError",
    "DownstreamScenePolicy",
    "HeldTubeSceneDiagnostics",
    "QualifiedHeldTubeSceneInterpreter",
    "create_online_held_tube_scene_interpreter",
    "create_qualified_held_tube_scene_interpreter",
]
