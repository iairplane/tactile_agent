"""Small HTTP transport for a remote F2G-Pose worker.

The robot host sends one aligned RGB/depth/mask/K observation as a compressed
NumPy archive.  The GPU host returns only JSON following
``demo2gripper.pose_observation.v1``.  Importing this module does not import
F2G-Pose, Torch, OpenCV, or CUDA libraries.
"""

from __future__ import annotations

import hmac
import io
import json
import math
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .pose_observation import (
    POSE_REQUEST_SCHEMA,
    ForegroundDepthDiagnostics,
    MaskQuality,
    PoseCapture,
    PoseObservation,
    PoseObservationError,
    PoseObservationGuard,
    PoseObservationGuardConfig,
    PoseQuality,
)

POSE_SERVICE_ERROR_SCHEMA = "demo2gripper.pose_service_error.v1"
POSE_SERVICE_HEALTH_SCHEMA = "demo2gripper.pose_service_health.v1"
POSE_REQUEST_CONTENT_TYPE = "application/vnd.demo2gripper.pose-request-npz"
POSE_RESPONSE_CONTENT_TYPE = "application/json"
FOREGROUND_DEPTH_POLICY = "nearest_significant_nonbackground_cluster.v1"
COORDINATE_FRAME = "camera_optical: x right, y down, z forward; p_camera=T_camera_object@p_object"


class PoseServiceError(RuntimeError):
    """The remote service or its transport failed."""


class PoseServiceTimeout(PoseServiceError, TimeoutError):
    """The remote service did not finish inside the configured deadline."""


class PoseBackendBusy(PoseServiceError):
    """A previous request still owns the single resident GPU estimator."""


@dataclass(frozen=True)
class PoseRequest:
    """One aligned model input and its client-clock identity."""

    request_id: str
    capture: PoseCapture
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    mask: np.ndarray
    mask_source: str
    mask_source_score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise PoseObservationError("request_id must be a non-empty string")
        if not isinstance(self.capture, PoseCapture):
            raise PoseObservationError("capture must be PoseCapture")
        if not isinstance(self.mask_source, str) or not self.mask_source.strip():
            raise PoseObservationError("mask_source must be a non-empty string")
        score = self.mask_source_score
        if score is not None and (
            isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise PoseObservationError("mask_source_score must lie in [0,1]")
        rgb, depth, K, mask = validate_pose_request_arrays(
            self.rgb, self.depth_m, self.K, self.mask
        )
        for name, value in (("rgb", rgb), ("depth_m", depth), ("K", K), ("mask", mask)):
            object.__setattr__(self, name, value)

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "schema": POSE_REQUEST_SCHEMA,
            "request_id": self.request_id,
            "capture": self.capture.to_dict(),
            "mask_source": self.mask_source,
            "mask_source_score": self.mask_source_score,
            "depth_units": "metres",
            "rgb_channel_order": "RGB",
        }


def validate_pose_request_arrays(
    rgb: object, depth_m: object, K: object, mask: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth_m)
    K_array = np.asarray(K)
    mask_array = np.asarray(mask)
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3 or rgb_array.dtype != np.uint8:
        raise PoseObservationError("rgb must be HxWx3 uint8 in RGB order")
    if depth_array.ndim != 2 or not np.issubdtype(depth_array.dtype, np.number):
        raise PoseObservationError("depth_m must be a numeric HxW array")
    if mask_array.ndim != 2:
        raise PoseObservationError("mask must be a binary HxW array")
    if rgb_array.shape[:2] != depth_array.shape or depth_array.shape != mask_array.shape:
        raise PoseObservationError("rgb, depth_m, and mask must be pixel-aligned")
    if not np.isfinite(mask_array).all() or not np.isin(mask_array, (0, 1, 255)).all():
        raise PoseObservationError("mask must contain only 0/1 or 0/255")
    mask_bool = mask_array > 0
    if not mask_bool.any():
        raise PoseObservationError("mask is empty")
    if K_array.shape == (4,):
        fx, fy, cx, cy = K_array.astype(np.float64)
        K_array = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
    if (
        K_array.shape != (3, 3)
        or not np.isfinite(K_array).all()
        or K_array[0, 0] <= 0.0
        or K_array[1, 1] <= 0.0
        or not np.allclose(K_array[2], [0.0, 0.0, 1.0], atol=1e-9)
        or abs(float(K_array[0, 1])) > 1e-9
        or abs(float(K_array[1, 0])) > 1e-9
    ):
        raise PoseObservationError("K must be finite pinhole intrinsics with zero skew")
    return (
        np.ascontiguousarray(rgb_array),
        np.ascontiguousarray(depth_array.astype(np.float32, copy=False)),
        np.ascontiguousarray(K_array.astype(np.float64, copy=False)),
        np.ascontiguousarray(mask_bool),
    )


def encode_pose_request(request: PoseRequest) -> bytes:
    """Serialize without pickle; metadata is UTF-8 bytes inside the NPZ."""

    metadata = json.dumps(
        request.metadata_dict(), ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        metadata_json=np.frombuffer(metadata, dtype=np.uint8),
        rgb=request.rgb,
        depth_m=request.depth_m,
        K=request.K,
        mask=request.mask.astype(np.uint8),
    )
    return stream.getvalue()


def decode_pose_request(payload: bytes) -> PoseRequest:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            required = {"metadata_json", "rgb", "depth_m", "K", "mask"}
            if set(archive.files) != required:
                raise PoseObservationError(
                    f"pose request NPZ entries must be exactly {sorted(required)}"
                )
            metadata_bytes = np.asarray(archive["metadata_json"], dtype=np.uint8).tobytes()
            metadata = json.loads(metadata_bytes.decode("utf-8"))
            rgb = np.asarray(archive["rgb"])
            depth_m = np.asarray(archive["depth_m"])
            K = np.asarray(archive["K"])
            mask = np.asarray(archive["mask"])
    except PoseObservationError:
        raise
    except (OSError, ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PoseObservationError(f"invalid pose request archive: {error}") from error
    if not isinstance(metadata, Mapping):
        raise PoseObservationError("pose request metadata must be an object")
    if metadata.get("schema") != POSE_REQUEST_SCHEMA:
        raise PoseObservationError(f"pose request schema must be {POSE_REQUEST_SCHEMA!r}")
    if metadata.get("depth_units") != "metres":
        raise PoseObservationError("pose request depth_units must be metres")
    if metadata.get("rgb_channel_order") != "RGB":
        raise PoseObservationError("pose request rgb_channel_order must be RGB")
    return PoseRequest(
        request_id=metadata.get("request_id"),
        capture=PoseCapture.from_dict(metadata.get("capture")),
        rgb=rgb,
        depth_m=depth_m,
        K=K,
        mask=mask,
        mask_source=metadata.get("mask_source"),
        mask_source_score=metadata.get("mask_source_score"),
    )


@dataclass(frozen=True)
class ForegroundDepthGateConfig:
    """Bounds for rejecting transparent-object masks dominated by background."""

    min_depth_m: float = 0.08
    max_depth_m: float = 2.0
    min_valid_mask_pixels: int = 100
    min_valid_mask_fraction: float = 0.20
    cluster_gap_m: float = 0.008
    min_cluster_pixels: int = 80
    min_cluster_fraction: float = 0.25
    max_cluster_span_m: float = 0.050
    ring_radius_px: int = 5
    min_ring_valid_pixels: int = 50
    background_match_tolerance_m: float = 0.010
    min_background_separation_m: float = 0.012
    max_mask_near_background_fraction: float = 0.50

    def __post_init__(self) -> None:
        numeric_positive = (
            "min_depth_m",
            "max_depth_m",
            "cluster_gap_m",
            "max_cluster_span_m",
            "background_match_tolerance_m",
            "min_background_separation_m",
        )
        for name in numeric_positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.min_depth_m >= self.max_depth_m:
            raise ValueError("min_depth_m must be below max_depth_m")
        for name in (
            "min_valid_mask_pixels",
            "min_cluster_pixels",
            "ring_radius_px",
            "min_ring_valid_pixels",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "min_valid_mask_fraction",
            "min_cluster_fraction",
            "max_mask_near_background_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")


@dataclass(frozen=True)
class MaskGateConfig:
    min_pixels: int = 100
    min_area_fraction: float = 0.0002
    max_area_fraction: float = 0.65
    min_source_score: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.min_pixels, bool) or not isinstance(self.min_pixels, int):
            raise ValueError("min_pixels must be an integer")
        if self.min_pixels < 1:
            raise ValueError("min_pixels must be positive")
        if not 0.0 <= self.min_area_fraction < self.max_area_fraction <= 1.0:
            raise ValueError("mask area fraction bounds are invalid")
        if self.min_source_score is not None and not 0.0 <= self.min_source_score <= 1.0:
            raise ValueError("min_source_score must lie in [0,1]")


def evaluate_mask_quality(request: PoseRequest, config: MaskGateConfig) -> MaskQuality:
    mask = request.mask
    count = int(mask.sum())
    area_fraction = count / mask.size
    rows, columns = np.where(mask)
    bbox = (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )
    reasons: list[str] = []
    if count < config.min_pixels:
        reasons.append("mask_too_small")
    if area_fraction < config.min_area_fraction:
        reasons.append("mask_area_fraction_too_small")
    if area_fraction > config.max_area_fraction:
        reasons.append("mask_area_fraction_too_large")
    if config.min_source_score is not None:
        if request.mask_source_score is None:
            reasons.append("mask_source_score_missing")
        elif request.mask_source_score < config.min_source_score:
            reasons.append("mask_source_score_too_low")
    return MaskQuality(
        source=request.mask_source,
        source_score=request.mask_source_score,
        pixel_count=count,
        area_fraction=float(area_fraction),
        bbox_xyxy=bbox,
        gate_passed=not reasons,
        reject_reasons=tuple(reasons),
    )


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    current = mask.copy()
    for _ in range(radius):
        padded = np.pad(current, 1, mode="constant", constant_values=False)
        current = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )
    return current


def _depth_clusters(values: np.ndarray, gap_m: float) -> list[np.ndarray]:
    if values.size == 0:
        return []
    ordered = np.sort(values.astype(np.float64, copy=False))
    split_indices = np.flatnonzero(np.diff(ordered) > gap_m) + 1
    return [part for part in np.split(ordered, split_indices) if part.size]


def _nearest_distances(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return each value's absolute distance to the nearest sorted reference."""

    if values.size == 0 or reference.size == 0:
        return np.full(values.shape, np.inf, dtype=np.float64)
    ordered = np.sort(reference.astype(np.float64, copy=False))
    query = values.astype(np.float64, copy=False)
    indices = np.searchsorted(ordered, query)
    left = ordered[np.clip(indices - 1, 0, ordered.size - 1)]
    right = ordered[np.clip(indices, 0, ordered.size - 1)]
    return np.minimum(np.abs(query - left), np.abs(query - right))


@dataclass(frozen=True)
class DepthGateResult:
    diagnostics: ForegroundDepthDiagnostics
    sanitized_depth_m: np.ndarray


def evaluate_foreground_depth(
    request: PoseRequest, config: ForegroundDepthGateConfig
) -> DepthGateResult:
    """Reject masks whose depth primarily matches the neighbouring background.

    F2G-Pose's upstream minimum of 50 points only proves that some depth exists.
    This gate additionally clusters target-mask depth, compares it with an
    outside-mask ring, and passes only a sizeable, compact, separated nearest
    cluster.  Passed inference uses only that cluster's depth samples.
    """

    depth = request.depth_m
    mask = request.mask
    valid = np.isfinite(depth) & (depth >= config.min_depth_m) & (depth <= config.max_depth_m)
    values = depth[mask & valid].astype(np.float64, copy=False)
    mask_pixels = int(mask.sum())
    valid_fraction = values.size / mask_pixels
    raw_clusters = _depth_clusters(values, config.cluster_gap_m)
    cluster_records: list[dict[str, Any]] = []
    for index, cluster in enumerate(raw_clusters):
        cluster_records.append(
            {
                "index": index,
                "pixels": int(cluster.size),
                "fraction": float(cluster.size / values.size) if values.size else 0.0,
                "min_m": float(cluster[0]),
                "median_m": float(np.median(cluster)),
                "max_m": float(cluster[-1]),
                "span_m": float(cluster[-1] - cluster[0]),
            }
        )

    ring = _dilate(mask, config.ring_radius_px) & ~mask
    ring_values = depth[ring & valid].astype(np.float64, copy=False)
    ring_median = float(np.median(ring_values)) if ring_values.size else None
    ring_clusters = _depth_clusters(ring_values, config.cluster_gap_m)
    # The outside ring can straddle a foreground support and a distant
    # background plane.  Comparing against every ring sample lets even a
    # minority support-plane return veto an otherwise coherent foreground
    # cluster.  Use the best-supported ring cluster as the robust background
    # reference; masks that truly contain the neighbouring background still
    # fail through the same fraction and separation gates below.
    background_reference = max(ring_clusters, key=lambda item: item.size, default=None)
    near_background_fraction: float | None = None
    if background_reference is not None and values.size:
        near_background_fraction = float(
            np.mean(
                _nearest_distances(values, background_reference)
                <= config.background_match_tolerance_m
            )
        )

    compact = [
        (record, cluster)
        for record, cluster in zip(cluster_records, raw_clusters, strict=True)
        if record["pixels"] >= config.min_cluster_pixels
        and record["span_m"] <= config.max_cluster_span_m
    ]
    separated = compact
    if background_reference is not None:
        separated = [
            (record, cluster)
            for record, cluster in compact
            if float(
                np.mean(
                    _nearest_distances(cluster, background_reference)
                    >= config.min_background_separation_m
                )
            )
            >= 0.80
        ]
    selected_pair = min(
        separated, key=lambda item: item[0]["median_m"], default=None
    )
    selected = None if selected_pair is None else selected_pair[0]

    reasons: list[str] = []
    if values.size < config.min_valid_mask_pixels:
        reasons.append("insufficient_valid_mask_depth_pixels")
    if valid_fraction < config.min_valid_mask_fraction:
        reasons.append("valid_mask_depth_fraction_too_low")
    if ring_values.size < config.min_ring_valid_pixels:
        reasons.append("insufficient_outside_ring_depth_for_background_gate")
    if (
        near_background_fraction is not None
        and near_background_fraction > config.max_mask_near_background_fraction
    ):
        reasons.append("mask_depth_dominated_by_neighbouring_background")
    if not compact:
        reasons.append("no_compact_significant_mask_depth_cluster")
    elif selected is None:
        reasons.append("no_significant_depth_cluster_separated_from_background")
    elif selected["fraction"] < config.min_cluster_fraction:
        reasons.append("selected_foreground_cluster_fraction_too_low")

    sanitized = np.zeros_like(depth, dtype=np.float32)
    if selected is not None:
        selected_pixels = (
            mask
            & valid
            & (depth >= selected["min_m"] - 1e-7)
            & (depth <= selected["max_m"] + 1e-7)
        )
        sanitized[selected_pixels] = depth[selected_pixels]
    sanitized.setflags(write=False)

    diagnostics = ForegroundDepthDiagnostics(
        policy=FOREGROUND_DEPTH_POLICY,
        valid_mask_pixels=int(values.size),
        valid_mask_fraction=float(valid_fraction),
        cluster_count=len(cluster_records),
        selected_cluster_index=None if selected is None else int(selected["index"]),
        selected_cluster_pixels=0 if selected is None else int(selected["pixels"]),
        selected_cluster_fraction=0.0 if selected is None else float(selected["fraction"]),
        selected_median_m=None if selected is None else float(selected["median_m"]),
        selected_span_m=None if selected is None else float(selected["span_m"]),
        outside_ring_valid_pixels=int(ring_values.size),
        outside_ring_median_m=ring_median,
        mask_near_background_fraction=near_background_fraction,
        clusters=tuple(cluster_records),
        gate_passed=not reasons,
        reject_reasons=tuple(reasons),
    )
    return DepthGateResult(diagnostics=diagnostics, sanitized_depth_m=sanitized)


@dataclass(frozen=True)
class BackendPrediction:
    T_camera_object: np.ndarray
    size_m: np.ndarray
    preprocess_latency_s: float
    inference_latency_s: float


class PoseInferenceBackend(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def predict_pose_only(
        self, rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray, mask: np.ndarray
    ) -> BackendPrediction: ...


class F2GPoseBackend:
    """Lazy adapter around the official ``f2g_pose.PoseEstimator`` API."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        config: str | Path | None = None,
        radio_repository: str | Path | None = None,
        radio_checkpoint: str | Path | None = None,
        device: str = "cuda:0",
        code_revision: str = "unknown",
        checkpoint_sha256: str = "unknown",
    ) -> None:
        try:
            from f2g_pose import PoseEstimator
        except ImportError as error:
            raise RuntimeError(
                "f2g_pose is not importable in the GPU service environment"
            ) from error
        self._estimator = PoseEstimator(
            checkpoint,
            config=config,
            radio_repository=radio_repository,
            radio_checkpoint=radio_checkpoint,
            device=device,
        )
        self._metadata = {
            "name": "f2g_pose",
            "implementation_version": "official_pose_estimator_api",
            "code_revision": code_revision,
            "checkpoint_sha256": checkpoint_sha256,
            "device": device,
            "pose_only": True,
        }

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    def predict_pose_only(
        self, rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray, mask: np.ndarray
    ) -> BackendPrediction:
        start = time.monotonic()
        observation = self._estimator.preprocess(rgb, depth_m, K, mask, 1.0)
        after_preprocess = time.monotonic()
        result = self._estimator.forward(observation, return_shape=False)
        after_inference = time.monotonic()
        return BackendPrediction(
            T_camera_object=np.asarray(result["T_cam_obj"], dtype=np.float64),
            size_m=np.asarray(result["size_m"], dtype=np.float64),
            preprocess_latency_s=after_preprocess - start,
            inference_latency_s=after_inference - after_preprocess,
        )


class FakePoseBackend:
    """Deterministic CPU backend for transport tests and explicit dry-runs."""

    def __init__(
        self,
        *,
        T_camera_object: np.ndarray | None = None,
        size_m: tuple[float, float, float] = (0.018, 0.018, 0.120),
        delay_s: float = 0.0,
    ) -> None:
        transform = np.eye(4) if T_camera_object is None else np.asarray(T_camera_object)
        self._transform = transform.astype(np.float64, copy=True)
        self._size = np.asarray(size_m, dtype=np.float64)
        self._delay_s = float(delay_s)
        self._metadata = {
            "name": "fake_pose",
            "implementation_version": "cpu_test_only",
            "code_revision": "none",
            "checkpoint_sha256": "none",
            "device": "cpu",
            "pose_only": True,
        }

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    def predict_pose_only(
        self, rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray, mask: np.ndarray
    ) -> BackendPrediction:
        del rgb, K, mask
        start = time.monotonic()
        if self._delay_s > 0.0:
            time.sleep(self._delay_s)
        transform = self._transform.copy()
        if np.allclose(transform[:3, 3], 0.0):
            valid = depth_m[depth_m > 0.0]
            transform[2, 3] = float(np.median(valid)) if valid.size else 0.3
        return BackendPrediction(
            T_camera_object=transform,
            size_m=self._size.copy(),
            preprocess_latency_s=0.0,
            inference_latency_s=time.monotonic() - start,
        )


class PoseInferenceService:
    """Stateless input gates plus one long-lived inference backend."""

    def __init__(
        self,
        backend: PoseInferenceBackend,
        *,
        mask_config: MaskGateConfig | None = None,
        depth_config: ForegroundDepthGateConfig | None = None,
    ) -> None:
        self.backend = backend
        self.mask_config = mask_config or MaskGateConfig()
        self.depth_config = depth_config or ForegroundDepthGateConfig()
        self._backend_lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "schema": POSE_SERVICE_HEALTH_SCHEMA,
            "ready": True,
            "backend": dict(self.backend.metadata),
            "observation_schema": "demo2gripper.pose_observation.v1",
            "foreground_depth_policy": FOREGROUND_DEPTH_POLICY,
        }

    def infer(self, request: PoseRequest) -> PoseObservation:
        start = time.monotonic()
        mask_quality = evaluate_mask_quality(request, self.mask_config)
        depth_result = evaluate_foreground_depth(request, self.depth_config)
        reasons = tuple(mask_quality.reject_reasons + depth_result.diagnostics.reject_reasons)
        prediction: BackendPrediction | None = None
        if not reasons:
            if not self._backend_lock.acquire(blocking=False):
                raise PoseBackendBusy("pose backend is still processing another request")
            try:
                prediction = self.backend.predict_pose_only(
                    request.rgb,
                    depth_result.sanitized_depth_m,
                    request.K,
                    request.mask,
                )
            finally:
                self._backend_lock.release()
        total = time.monotonic() - start
        payload: dict[str, Any] = {
            "schema": "demo2gripper.pose_observation.v1",
            "request_id": request.request_id,
            "capture": request.capture.to_dict(),
            "accepted": prediction is not None,
            "pose": None,
            "coordinate_frame": COORDINATE_FRAME,
            "produced_at_unix_s": time.time(),
            "mask_quality": mask_quality.to_dict(),
            "foreground_depth": depth_result.diagnostics.to_dict(),
            "pose_quality": PoseQuality(
                model_confidence=None,
                temporal_track_confirmed=False,
                axis_sign_status="unresolved",
                axis_sign_source=None,
                cylindrical_roll_status="unresolved",
                cylindrical_roll_source=None,
                task_qualified=False,
            ).to_dict(),
            "latency": {
                "preprocess_s": 0.0 if prediction is None else prediction.preprocess_latency_s,
                "inference_s": 0.0 if prediction is None else prediction.inference_latency_s,
                "total_server_s": total,
            },
            "backend": dict(self.backend.metadata),
            "reject_reasons": list(reasons),
        }
        if prediction is not None:
            payload["pose"] = {
                "T_camera_object": np.asarray(prediction.T_camera_object).tolist(),
                "size_m": np.asarray(prediction.size_m).tolist(),
            }
        return PoseObservation.from_dict(payload)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def create_pose_http_server(
    host: str,
    port: int,
    service: PoseInferenceService,
    *,
    auth_token: str | None = None,
    max_request_bytes: int = 64 * 1024 * 1024,
) -> ThreadingHTTPServer:
    """Create a server; callers own ``serve_forever`` and ``shutdown``."""

    if not host:
        raise ValueError("host must be non-empty")
    if not 0 <= port <= 65535:
        raise ValueError("port is invalid")
    if max_request_bytes < 1024:
        raise ValueError("max_request_bytes is too small")

    class Handler(BaseHTTPRequestHandler):
        server_version = "demo2gripper-pose/1"

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _authorized(self) -> bool:
            if auth_token is None:
                return True
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {auth_token}"
            return hmac.compare_digest(supplied, expected)

        def _write_json(self, status: HTTPStatus, value: Mapping[str, Any]) -> None:
            body = _json_bytes(value)
            self.send_response(int(status))
            self.send_header("Content-Type", POSE_RESPONSE_CONTENT_TYPE)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: HTTPStatus, code: str, detail: str) -> None:
            self._write_json(
                status,
                {"schema": POSE_SERVICE_ERROR_SCHEMA, "error": code, "detail": detail},
            )

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")
                return
            if not self._authorized():
                self._error(HTTPStatus.UNAUTHORIZED, "unauthorized", "invalid bearer token")
                return
            self._write_json(HTTPStatus.OK, service.health())

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/estimate":
                self._error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")
                return
            if not self._authorized():
                self._error(HTTPStatus.UNAUTHORIZED, "unauthorized", "invalid bearer token")
                return
            if self.headers.get_content_type() != POSE_REQUEST_CONTENT_TYPE:
                self._error(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "content_type",
                    f"expected {POSE_REQUEST_CONTENT_TYPE}",
                )
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = -1
            if length < 1 or length > max_request_bytes:
                self._error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "request_size",
                    "request body length is missing or outside the configured bound",
                )
                return
            try:
                request = decode_pose_request(self.rfile.read(length))
                result = service.infer(request)
            except PoseObservationError as error:
                self._error(HTTPStatus.BAD_REQUEST, "invalid_request", str(error))
                return
            except PoseBackendBusy as error:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, "backend_busy", str(error))
                return
            except Exception as error:  # backend faults become explicit 500s
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "backend_failure", str(error))
                return
            self._write_json(HTTPStatus.OK, result.to_dict())

    return ThreadingHTTPServer((host, port), Handler)


@dataclass(frozen=True)
class PoseServiceReply:
    observation: PoseObservation
    received_at_monotonic_s: float
    roundtrip_latency_s: float


class PoseServiceClient:
    """Bounded stdlib HTTP client; it does not retry stale motion inputs."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float = 0.25,
        auth_token: str | None = None,
        max_response_bytes: int = 2 * 1024 * 1024,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("pose endpoint must be an http(s) URL")
        if timeout_s <= 0.0 or not math.isfinite(timeout_s):
            raise ValueError("timeout_s must be positive and finite")
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s
        self.auth_token = auth_token
        self.max_response_bytes = max_response_bytes
        self._monotonic = monotonic

    def health(self) -> Mapping[str, Any]:
        request = urllib.request.Request(self.endpoint + "/health", method="GET")
        if self.auth_token is not None:
            request.add_header("Authorization", f"Bearer {self.auth_token}")
        return self._request_json(request)

    def estimate(self, request_value: PoseRequest) -> PoseServiceReply:
        body = encode_pose_request(request_value)
        request = urllib.request.Request(
            self.endpoint + "/v1/estimate", data=body, method="POST"
        )
        request.add_header("Content-Type", POSE_REQUEST_CONTENT_TYPE)
        if self.auth_token is not None:
            request.add_header("Authorization", f"Bearer {self.auth_token}")
        started = self._monotonic()
        payload = self._request_json(request)
        received = self._monotonic()
        return PoseServiceReply(
            observation=PoseObservation.from_dict(payload),
            received_at_monotonic_s=received,
            roundtrip_latency_s=received - started,
        )

    def _request_json(self, request: urllib.request.Request) -> Mapping[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read(self.max_response_bytes + 1)
        except urllib.error.HTTPError as error:
            detail = error.read(8192).decode("utf-8", errors="replace")
            raise PoseServiceError(f"pose service HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            reason = getattr(error, "reason", error)
            if isinstance(reason, TimeoutError):
                raise PoseServiceTimeout("pose service request timed out") from error
            raise PoseServiceError(f"pose service transport failed: {reason}") from error
        if len(body) > self.max_response_bytes:
            raise PoseServiceError("pose service response exceeds configured size bound")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PoseServiceError("pose service response is not valid JSON") from error
        if not isinstance(payload, Mapping):
            raise PoseServiceError("pose service JSON root must be an object")
        return payload


class RemotePoseProvider:
    """Stateful robot-host provider with timeout, freshness, and replay checks."""

    def __init__(
        self,
        client: PoseServiceClient,
        *,
        guard_config: PoseObservationGuardConfig | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.guard = PoseObservationGuard(guard_config)
        self._monotonic = monotonic
        self._sequence = 0

    def estimate(
        self,
        *,
        frame_id: str,
        capture_timestamp_s: float,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray,
        mask_source: str,
        mask_source_score: float | None = None,
    ) -> PoseObservation:
        self._sequence += 1
        request_id = uuid.uuid4().hex
        request = PoseRequest(
            request_id=request_id,
            capture=PoseCapture(
                sequence=self._sequence,
                frame_id=frame_id,
                timestamp_s=float(capture_timestamp_s),
                clock_domain="client_monotonic",
            ),
            rgb=rgb,
            depth_m=depth_m,
            K=K,
            mask=mask,
            mask_source=mask_source,
            mask_source_score=mask_source_score,
        )
        reply = self.client.estimate(request)
        return self.guard.validate(
            reply.observation,
            expected_request_id=request_id,
            expected_sequence=self._sequence,
            expected_frame_id=frame_id,
            expected_timestamp_s=float(capture_timestamp_s),
            received_at_monotonic_s=reply.received_at_monotonic_s,
            roundtrip_latency_s=reply.roundtrip_latency_s,
        )
