"""Versioned, fail-closed contract for remote RGB-D pose observations.

The contract deliberately separates a model pose from task-level orientation
qualification.  In particular, a category-level pose for a cylindrical test
tube does not by itself identify the open end or make roll observable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

POSE_OBSERVATION_SCHEMA = "demo2gripper.pose_observation.v1"
POSE_REQUEST_SCHEMA = "demo2gripper.pose_request.v1"


class PoseObservationError(ValueError):
    """A pose service payload violated the versioned data contract."""


class PoseObservationRejected(RuntimeError):
    """The service produced a valid rejection rather than a usable pose."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("pose observation rejected: " + ", ".join(self.reasons))


class PoseObservationReplay(PoseObservationError):
    """An observation repeated or regressed within one provider session."""


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PoseObservationError(f"{name} must be an object")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PoseObservationError(f"{name} must be a non-empty string")
    return value


def _finite(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise PoseObservationError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PoseObservationError(f"{name} must be numeric") from error
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise PoseObservationError(f"{name} must be {qualifier}")
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PoseObservationError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise PoseObservationError(f"{name} must be boolean")
    return value


def _string_list(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise PoseObservationError(f"{name} must be an array")
    return tuple(_string(item, name=f"{name}[]") for item in value)


def _rigid_transform(value: object, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PoseObservationError(f"{name} must be numeric 4x4") from error
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise PoseObservationError(f"{name} must be finite 4x4")
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise PoseObservationError(f"{name} has an invalid homogeneous row")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise PoseObservationError(f"{name} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-3):
        raise PoseObservationError(f"{name} rotation determinant must be +1")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PoseCapture:
    """Client-clock identity of the aligned RGB-D frame sent for inference."""

    sequence: int
    frame_id: str
    timestamp_s: float
    clock_domain: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sequence", _integer(self.sequence, name="capture.sequence")
        )
        object.__setattr__(
            self, "frame_id", _string(self.frame_id, name="capture.frame_id")
        )
        object.__setattr__(
            self,
            "timestamp_s",
            _finite(self.timestamp_s, name="capture.timestamp_s"),
        )
        object.__setattr__(
            self,
            "clock_domain",
            _string(self.clock_domain, name="capture.clock_domain"),
        )

    @classmethod
    def from_dict(cls, value: object) -> PoseCapture:
        data = _mapping(value, name="capture")
        return cls(
            sequence=_integer(data.get("sequence"), name="capture.sequence"),
            frame_id=_string(data.get("frame_id"), name="capture.frame_id"),
            timestamp_s=_finite(data.get("timestamp_s"), name="capture.timestamp_s"),
            clock_domain=_string(data.get("clock_domain"), name="capture.clock_domain"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "clock_domain": self.clock_domain,
        }


@dataclass(frozen=True)
class MaskQuality:
    """Mask provenance and cheap geometry diagnostics, without invented IoU."""

    source: str
    pixel_count: int
    area_fraction: float
    bbox_xyxy: tuple[int, int, int, int]
    source_score: float | None
    gate_passed: bool
    reject_reasons: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object) -> MaskQuality:
        data = _mapping(value, name="mask_quality")
        bbox = data.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise PoseObservationError("mask_quality.bbox_xyxy must contain four integers")
        bbox_tuple = tuple(
            _integer(item, name="mask_quality.bbox_xyxy[]") for item in bbox
        )
        fraction = _finite(
            data.get("area_fraction"), name="mask_quality.area_fraction", nonnegative=True
        )
        if fraction > 1.0:
            raise PoseObservationError("mask_quality.area_fraction must be <= 1")
        raw_score = data.get("source_score")
        score = (
            None
            if raw_score is None
            else _finite(raw_score, name="mask_quality.source_score", nonnegative=True)
        )
        if score is not None and score > 1.0:
            raise PoseObservationError("mask_quality.source_score must be <= 1")
        result = cls(
            source=_string(data.get("source"), name="mask_quality.source"),
            pixel_count=_integer(data.get("pixel_count"), name="mask_quality.pixel_count"),
            area_fraction=fraction,
            bbox_xyxy=bbox_tuple,
            source_score=score,
            gate_passed=_boolean(data.get("gate_passed"), name="mask_quality.gate_passed"),
            reject_reasons=_string_list(
                data.get("reject_reasons"), name="mask_quality.reject_reasons"
            ),
        )
        if result.gate_passed == bool(result.reject_reasons):
            raise PoseObservationError(
                "mask_quality gate_passed must be true exactly when reject_reasons is empty"
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_score": self.source_score,
            "pixel_count": self.pixel_count,
            "area_fraction": self.area_fraction,
            "bbox_xyxy": list(self.bbox_xyxy),
            "gate_passed": self.gate_passed,
            "reject_reasons": list(self.reject_reasons),
        }


@dataclass(frozen=True)
class ForegroundDepthDiagnostics:
    """Evidence that target-mask depth is object foreground, not the background."""

    policy: str
    valid_mask_pixels: int
    valid_mask_fraction: float
    cluster_count: int
    selected_cluster_index: int | None
    selected_cluster_pixels: int
    selected_cluster_fraction: float
    selected_median_m: float | None
    selected_span_m: float | None
    outside_ring_valid_pixels: int
    outside_ring_median_m: float | None
    mask_near_background_fraction: float | None
    clusters: tuple[Mapping[str, Any], ...]
    gate_passed: bool
    reject_reasons: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object) -> ForegroundDepthDiagnostics:
        data = _mapping(value, name="foreground_depth")
        cluster_values = data.get("clusters")
        if not isinstance(cluster_values, list):
            raise PoseObservationError("foreground_depth.clusters must be an array")
        clusters: list[Mapping[str, Any]] = []
        for index, raw in enumerate(cluster_values):
            cluster = dict(_mapping(raw, name=f"foreground_depth.clusters[{index}]"))
            for field_name in ("index", "pixels"):
                _integer(cluster.get(field_name), name=f"cluster.{field_name}")
            for field_name in ("fraction", "min_m", "median_m", "max_m", "span_m"):
                _finite(cluster.get(field_name), name=f"cluster.{field_name}", nonnegative=True)
            clusters.append(cluster)

        valid_fraction = _finite(
            data.get("valid_mask_fraction"),
            name="foreground_depth.valid_mask_fraction",
            nonnegative=True,
        )
        selected_fraction = _finite(
            data.get("selected_cluster_fraction"),
            name="foreground_depth.selected_cluster_fraction",
            nonnegative=True,
        )
        near_background = data.get("mask_near_background_fraction")
        near_background_value = (
            None
            if near_background is None
            else _finite(
                near_background,
                name="foreground_depth.mask_near_background_fraction",
                nonnegative=True,
            )
        )
        for name, fraction in (
            ("valid_mask_fraction", valid_fraction),
            ("selected_cluster_fraction", selected_fraction),
            ("mask_near_background_fraction", near_background_value),
        ):
            if fraction is not None and fraction > 1.0:
                raise PoseObservationError(f"foreground_depth.{name} must be <= 1")

        raw_index = data.get("selected_cluster_index")
        selected_index = (
            None
            if raw_index is None
            else _integer(raw_index, name="foreground_depth.selected_cluster_index")
        )
        result = cls(
            policy=_string(data.get("policy"), name="foreground_depth.policy"),
            valid_mask_pixels=_integer(
                data.get("valid_mask_pixels"), name="foreground_depth.valid_mask_pixels"
            ),
            valid_mask_fraction=valid_fraction,
            cluster_count=_integer(
                data.get("cluster_count"), name="foreground_depth.cluster_count"
            ),
            selected_cluster_index=selected_index,
            selected_cluster_pixels=_integer(
                data.get("selected_cluster_pixels"),
                name="foreground_depth.selected_cluster_pixels",
            ),
            selected_cluster_fraction=selected_fraction,
            selected_median_m=_optional_finite(
                data.get("selected_median_m"), name="foreground_depth.selected_median_m"
            ),
            selected_span_m=_optional_finite(
                data.get("selected_span_m"),
                name="foreground_depth.selected_span_m",
                nonnegative=True,
            ),
            outside_ring_valid_pixels=_integer(
                data.get("outside_ring_valid_pixels"),
                name="foreground_depth.outside_ring_valid_pixels",
            ),
            outside_ring_median_m=_optional_finite(
                data.get("outside_ring_median_m"),
                name="foreground_depth.outside_ring_median_m",
            ),
            mask_near_background_fraction=near_background_value,
            clusters=tuple(clusters),
            gate_passed=_boolean(
                data.get("gate_passed"), name="foreground_depth.gate_passed"
            ),
            reject_reasons=_string_list(
                data.get("reject_reasons"), name="foreground_depth.reject_reasons"
            ),
        )
        if result.cluster_count != len(result.clusters):
            raise PoseObservationError("foreground_depth.cluster_count does not match clusters")
        if result.gate_passed == bool(result.reject_reasons):
            raise PoseObservationError(
                "foreground_depth gate_passed must be true exactly when reject_reasons is empty"
            )
        if result.gate_passed and result.selected_cluster_index is None:
            raise PoseObservationError("a passed depth gate must select one foreground cluster")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "valid_mask_pixels": self.valid_mask_pixels,
            "valid_mask_fraction": self.valid_mask_fraction,
            "cluster_count": self.cluster_count,
            "selected_cluster_index": self.selected_cluster_index,
            "selected_cluster_pixels": self.selected_cluster_pixels,
            "selected_cluster_fraction": self.selected_cluster_fraction,
            "selected_median_m": self.selected_median_m,
            "selected_span_m": self.selected_span_m,
            "outside_ring_valid_pixels": self.outside_ring_valid_pixels,
            "outside_ring_median_m": self.outside_ring_median_m,
            "mask_near_background_fraction": self.mask_near_background_fraction,
            "clusters": [dict(cluster) for cluster in self.clusters],
            "gate_passed": self.gate_passed,
            "reject_reasons": list(self.reject_reasons),
        }


def _optional_finite(
    value: object, *, name: str, nonnegative: bool = False
) -> float | None:
    return None if value is None else _finite(value, name=name, nonnegative=nonnegative)


@dataclass(frozen=True)
class PoseQuality:
    """Downstream qualification state for symmetric-object ambiguities."""

    model_confidence: float | None
    temporal_track_confirmed: bool
    axis_sign_status: str
    axis_sign_source: str | None
    cylindrical_roll_status: str
    cylindrical_roll_source: str | None
    task_qualified: bool

    @classmethod
    def from_dict(cls, value: object) -> PoseQuality:
        data = _mapping(value, name="pose_quality")
        model_confidence = _optional_finite(
            data.get("model_confidence"), name="pose_quality.model_confidence", nonnegative=True
        )
        if model_confidence is not None and model_confidence > 1.0:
            raise PoseObservationError("pose_quality.model_confidence must be <= 1")
        axis_status = _string(data.get("axis_sign_status"), name="pose_quality.axis_sign_status")
        if axis_status not in {"unresolved", "resolved", "not_applicable"}:
            raise PoseObservationError("pose_quality.axis_sign_status is unsupported")
        roll_status = _string(
            data.get("cylindrical_roll_status"),
            name="pose_quality.cylindrical_roll_status",
        )
        if roll_status not in {"unresolved", "canonicalized", "observable", "not_applicable"}:
            raise PoseObservationError("pose_quality.cylindrical_roll_status is unsupported")
        axis_source = data.get("axis_sign_source")
        if axis_source is not None:
            axis_source = _string(axis_source, name="pose_quality.axis_sign_source")
        roll_source = data.get("cylindrical_roll_source")
        if roll_source is not None:
            roll_source = _string(roll_source, name="pose_quality.cylindrical_roll_source")
        if axis_status == "resolved" and axis_source is None:
            raise PoseObservationError("resolved axis sign requires axis_sign_source")
        if roll_status in {"canonicalized", "observable"} and roll_source is None:
            raise PoseObservationError("resolved cylinder roll requires cylindrical_roll_source")
        return cls(
            model_confidence=model_confidence,
            temporal_track_confirmed=_boolean(
                data.get("temporal_track_confirmed"),
                name="pose_quality.temporal_track_confirmed",
            ),
            axis_sign_status=axis_status,
            axis_sign_source=axis_source,
            cylindrical_roll_status=roll_status,
            cylindrical_roll_source=roll_source,
            task_qualified=_boolean(
                data.get("task_qualified"), name="pose_quality.task_qualified"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_confidence": self.model_confidence,
            "temporal_track_confirmed": self.temporal_track_confirmed,
            "axis_sign_status": self.axis_sign_status,
            "axis_sign_source": self.axis_sign_source,
            "cylindrical_roll_status": self.cylindrical_roll_status,
            "cylindrical_roll_source": self.cylindrical_roll_source,
            "task_qualified": self.task_qualified,
        }


@dataclass(frozen=True)
class PoseObservation:
    """One accepted model pose or one explicit, diagnostic rejection."""

    request_id: str
    capture: PoseCapture
    accepted: bool
    T_camera_object: np.ndarray | None
    size_m: tuple[float, float, float] | None
    coordinate_frame: str
    produced_at_unix_s: float
    mask_quality: MaskQuality
    foreground_depth: ForegroundDepthDiagnostics
    pose_quality: PoseQuality
    preprocess_latency_s: float
    inference_latency_s: float
    total_server_latency_s: float
    backend: Mapping[str, Any]
    reject_reasons: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object) -> PoseObservation:
        data = _mapping(value, name="pose observation")
        if data.get("schema") != POSE_OBSERVATION_SCHEMA:
            raise PoseObservationError(
                f"pose observation schema must be {POSE_OBSERVATION_SCHEMA!r}"
            )
        accepted = _boolean(data.get("accepted"), name="accepted")
        pose = data.get("pose")
        if accepted:
            pose_data = _mapping(pose, name="pose")
            transform = _rigid_transform(
                pose_data.get("T_camera_object"), name="pose.T_camera_object"
            )
            raw_size = pose_data.get("size_m")
            if not isinstance(raw_size, list) or len(raw_size) != 3:
                raise PoseObservationError("pose.size_m must contain three numbers")
            size = tuple(
                _finite(item, name="pose.size_m[]", nonnegative=True) for item in raw_size
            )
            if any(item <= 0.0 for item in size):
                raise PoseObservationError("pose.size_m entries must be positive")
        else:
            if pose is not None:
                raise PoseObservationError("a rejected observation must have pose=null")
            transform = None
            size = None

        latency = _mapping(data.get("latency"), name="latency")
        backend = dict(_mapping(data.get("backend"), name="backend"))
        _string(backend.get("name"), name="backend.name")
        _boolean(backend.get("pose_only"), name="backend.pose_only")
        mask_quality = MaskQuality.from_dict(data.get("mask_quality"))
        foreground_depth = ForegroundDepthDiagnostics.from_dict(
            data.get("foreground_depth")
        )
        reject_reasons = _string_list(data.get("reject_reasons"), name="reject_reasons")
        if accepted and reject_reasons:
            raise PoseObservationError("an accepted observation cannot have reject_reasons")
        if accepted and (not mask_quality.gate_passed or not foreground_depth.gate_passed):
            raise PoseObservationError("an accepted observation requires passed input gates")
        if not accepted and not reject_reasons:
            raise PoseObservationError("a rejected observation requires reject_reasons")

        return cls(
            request_id=_string(data.get("request_id"), name="request_id"),
            capture=PoseCapture.from_dict(data.get("capture")),
            accepted=accepted,
            T_camera_object=transform,
            size_m=size,
            coordinate_frame=_string(
                data.get("coordinate_frame"), name="coordinate_frame"
            ),
            produced_at_unix_s=_finite(
                data.get("produced_at_unix_s"), name="produced_at_unix_s"
            ),
            mask_quality=mask_quality,
            foreground_depth=foreground_depth,
            pose_quality=PoseQuality.from_dict(data.get("pose_quality")),
            preprocess_latency_s=_finite(
                latency.get("preprocess_s"), name="latency.preprocess_s", nonnegative=True
            ),
            inference_latency_s=_finite(
                latency.get("inference_s"), name="latency.inference_s", nonnegative=True
            ),
            total_server_latency_s=_finite(
                latency.get("total_server_s"),
                name="latency.total_server_s",
                nonnegative=True,
            ),
            backend=backend,
            reject_reasons=reject_reasons,
        )

    def require_pose(self) -> tuple[np.ndarray, tuple[float, float, float]]:
        if not self.accepted or self.T_camera_object is None or self.size_m is None:
            raise PoseObservationRejected(self.reject_reasons)
        return self.T_camera_object, self.size_m

    def to_dict(self) -> dict[str, Any]:
        pose = None
        if self.accepted:
            if self.T_camera_object is None or self.size_m is None:
                raise PoseObservationError("accepted observation is missing pose")
            pose = {
                "T_camera_object": self.T_camera_object.tolist(),
                "size_m": list(self.size_m),
            }
        return {
            "schema": POSE_OBSERVATION_SCHEMA,
            "request_id": self.request_id,
            "capture": self.capture.to_dict(),
            "accepted": self.accepted,
            "pose": pose,
            "coordinate_frame": self.coordinate_frame,
            "produced_at_unix_s": self.produced_at_unix_s,
            "mask_quality": self.mask_quality.to_dict(),
            "foreground_depth": self.foreground_depth.to_dict(),
            "pose_quality": self.pose_quality.to_dict(),
            "latency": {
                "preprocess_s": self.preprocess_latency_s,
                "inference_s": self.inference_latency_s,
                "total_server_s": self.total_server_latency_s,
            },
            "backend": dict(self.backend),
            "reject_reasons": list(self.reject_reasons),
        }


@dataclass(frozen=True)
class PoseObservationGuardConfig:
    """Robot-host checks applied after every remote service response."""

    max_staleness_s: float = 0.25
    max_server_latency_s: float = 0.20
    max_roundtrip_latency_s: float = 0.25
    future_tolerance_s: float = 0.02
    expected_clock_domain: str = "client_monotonic"
    allowed_backends: frozenset[str] = frozenset({"f2g_pose"})

    def __post_init__(self) -> None:
        for name in (
            "max_staleness_s",
            "max_server_latency_s",
            "max_roundtrip_latency_s",
            "future_tolerance_s",
        ):
            value = _finite(getattr(self, name), name=name, nonnegative=True)
            object.__setattr__(self, name, value)
        _string(self.expected_clock_domain, name="expected_clock_domain")
        if not self.allowed_backends:
            raise PoseObservationError("allowed_backends must not be empty")
        object.__setattr__(
            self,
            "allowed_backends",
            frozenset(_string(item, name="allowed_backends[]") for item in self.allowed_backends),
        )


class PoseObservationGuard:
    """Reject stale, mismatched, regressed, replayed, or unapproved replies."""

    def __init__(self, config: PoseObservationGuardConfig | None = None) -> None:
        self.config = config or PoseObservationGuardConfig()
        self._last_sequence: int | None = None
        self._last_timestamp_s: float | None = None

    def validate(
        self,
        observation: PoseObservation,
        *,
        expected_request_id: str,
        expected_sequence: int,
        expected_frame_id: str,
        expected_timestamp_s: float,
        received_at_monotonic_s: float,
        roundtrip_latency_s: float,
    ) -> PoseObservation:
        if observation.request_id != expected_request_id:
            raise PoseObservationReplay("response request_id does not match the active request")
        capture = observation.capture
        if capture.sequence != expected_sequence:
            raise PoseObservationReplay("response sequence does not match the active request")
        if capture.frame_id != expected_frame_id:
            raise PoseObservationReplay("response frame_id does not match the active request")
        if not math.isclose(capture.timestamp_s, expected_timestamp_s, abs_tol=1e-9):
            raise PoseObservationReplay("response timestamp does not match the active request")
        if capture.clock_domain != self.config.expected_clock_domain:
            raise PoseObservationError(
                f"capture clock domain {capture.clock_domain!r} is not "
                f"{self.config.expected_clock_domain!r}"
            )
        if self._last_sequence is not None and capture.sequence <= self._last_sequence:
            raise PoseObservationReplay("pose observation sequence repeated or regressed")
        if (
            self._last_timestamp_s is not None
            and capture.timestamp_s <= self._last_timestamp_s
        ):
            raise PoseObservationReplay("pose observation timestamp repeated or regressed")

        now = _finite(
            received_at_monotonic_s, name="received_at_monotonic_s", nonnegative=True
        )
        age = now - capture.timestamp_s
        if age < -self.config.future_tolerance_s:
            raise PoseObservationError("capture timestamp is in the future")
        if age > self.config.max_staleness_s:
            raise PoseObservationError(
                f"pose observation is stale ({age:.3f}s > {self.config.max_staleness_s:.3f}s)"
            )
        roundtrip = _finite(
            roundtrip_latency_s, name="roundtrip_latency_s", nonnegative=True
        )
        if roundtrip > self.config.max_roundtrip_latency_s:
            raise PoseObservationError(
                "pose service round-trip latency exceeds the configured bound"
            )
        if observation.total_server_latency_s > self.config.max_server_latency_s:
            raise PoseObservationError(
                "pose service server latency exceeds the configured bound"
            )
        backend_name = _string(observation.backend.get("name"), name="backend.name")
        if backend_name not in self.config.allowed_backends:
            raise PoseObservationError(f"pose backend {backend_name!r} is not allowed")
        if observation.backend.get("pose_only") is not True:
            raise PoseObservationError("pose service response is not pose-only")
        if not observation.accepted:
            raise PoseObservationRejected(observation.reject_reasons)

        self._last_sequence = capture.sequence
        self._last_timestamp_s = capture.timestamp_s
        return observation
