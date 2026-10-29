"""Small HTTP boundary for one-shot, text-prompted SAM 3 masks.

The live test-tube controller uses this service only to bootstrap an object
identity from the current RGB frame.  It does not load SAM 3 on the robot
host and it never treats a text score as a robot-motion confidence score.
After a stable seed has been accepted, the local optical-flow tracker owns
the mask identity.
"""

from __future__ import annotations

import hmac
import io
import json
import math
import threading
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

import numpy as np

MASK_REQUEST_SCHEMA = "demo2gripper.sam3_mask_request.v1"
MASK_RESPONSE_SCHEMA = "demo2gripper.sam3_mask_response.v1"
MASK_HEALTH_SCHEMA = "demo2gripper.sam3_mask_health.v1"
MASK_ERROR_SCHEMA = "demo2gripper.sam3_mask_error.v1"
MASK_CONTENT_TYPE = "application/vnd.demo2gripper.sam3-mask-npz"


class Sam3MaskServiceError(RuntimeError):
    """The remote mask service or its response violated the contract."""


@dataclass(frozen=True)
class Sam3MaskRequest:
    request_id: str
    frame_id: str
    capture_timestamp_s: float
    prompt: str
    rgb: np.ndarray

    def __post_init__(self) -> None:
        for name in ("request_id", "frame_id", "prompt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if len(self.prompt) > 256:
            raise ValueError("prompt exceeds 256 characters")
        if not math.isfinite(float(self.capture_timestamp_s)):
            raise ValueError("capture_timestamp_s must be finite")
        image = np.asarray(self.rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("rgb must be HxWx3 uint8 in RGB order")
        object.__setattr__(self, "rgb", np.ascontiguousarray(image))


@dataclass(frozen=True)
class Sam3MaskObservation:
    request_id: str
    frame_id: str
    capture_timestamp_s: float
    prompt: str
    mask: np.ndarray
    score: float
    box_xyxy: tuple[float, float, float, float]
    backend: Mapping[str, Any]
    inference_latency_s: float

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.any():
            raise ValueError("mask must be a non-empty HxW boolean array")
        if not math.isfinite(float(self.score)) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must lie in [0,1]")
        if len(self.box_xyxy) != 4 or not np.isfinite(self.box_xyxy).all():
            raise ValueError("box_xyxy must contain four finite numbers")
        if not math.isfinite(float(self.inference_latency_s)) or self.inference_latency_s < 0:
            raise ValueError("inference_latency_s must be finite and non-negative")
        object.__setattr__(self, "mask", np.ascontiguousarray(mask))


class Sam3MaskBackend(Protocol):
    metadata: Mapping[str, Any]

    def segment(self, rgb: np.ndarray, prompt: str) -> tuple[np.ndarray, float, np.ndarray]: ...


def encode_mask_request(request: Sam3MaskRequest) -> bytes:
    metadata = {
        "schema": MASK_REQUEST_SCHEMA,
        "request_id": request.request_id,
        "frame_id": request.frame_id,
        "capture_timestamp_s": request.capture_timestamp_s,
        "prompt": request.prompt,
        "rgb_channel_order": "RGB",
    }
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        metadata_json=np.frombuffer(
            json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode(),
            dtype=np.uint8,
        ),
        rgb=request.rgb,
    )
    return stream.getvalue()


def decode_mask_request(payload: bytes) -> Sam3MaskRequest:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {"metadata_json", "rgb"}:
                raise ValueError("mask request entries are invalid")
            metadata = json.loads(
                np.asarray(archive["metadata_json"], dtype=np.uint8).tobytes().decode()
            )
            rgb = np.asarray(archive["rgb"])
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Sam3MaskServiceError(f"invalid mask request: {error}") from error
    if not isinstance(metadata, Mapping) or metadata.get("schema") != MASK_REQUEST_SCHEMA:
        raise Sam3MaskServiceError("mask request schema is invalid")
    if metadata.get("rgb_channel_order") != "RGB":
        raise Sam3MaskServiceError("mask request must use RGB channel order")
    return Sam3MaskRequest(
        request_id=metadata.get("request_id"),
        frame_id=metadata.get("frame_id"),
        capture_timestamp_s=metadata.get("capture_timestamp_s"),
        prompt=metadata.get("prompt"),
        rgb=rgb,
    )


def encode_mask_observation(observation: Sam3MaskObservation) -> bytes:
    metadata = {
        "schema": MASK_RESPONSE_SCHEMA,
        "request_id": observation.request_id,
        "frame_id": observation.frame_id,
        "capture_timestamp_s": observation.capture_timestamp_s,
        "prompt": observation.prompt,
        "score": observation.score,
        "box_xyxy": list(observation.box_xyxy),
        "backend": dict(observation.backend),
        "inference_latency_s": observation.inference_latency_s,
    }
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        metadata_json=np.frombuffer(
            json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode(),
            dtype=np.uint8,
        ),
        mask=observation.mask.astype(np.uint8),
    )
    return stream.getvalue()


def decode_mask_observation(payload: bytes) -> Sam3MaskObservation:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {"metadata_json", "mask"}:
                raise ValueError("mask response entries are invalid")
            metadata = json.loads(
                np.asarray(archive["metadata_json"], dtype=np.uint8).tobytes().decode()
            )
            raw_mask = np.asarray(archive["mask"])
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Sam3MaskServiceError(f"invalid mask response: {error}") from error
    if not isinstance(metadata, Mapping) or metadata.get("schema") != MASK_RESPONSE_SCHEMA:
        raise Sam3MaskServiceError("mask response schema is invalid")
    if raw_mask.ndim != 2 or not np.isin(raw_mask, (0, 1)).all():
        raise Sam3MaskServiceError("mask response is not binary")
    return Sam3MaskObservation(
        request_id=metadata.get("request_id"),
        frame_id=metadata.get("frame_id"),
        capture_timestamp_s=metadata.get("capture_timestamp_s"),
        prompt=metadata.get("prompt"),
        mask=raw_mask > 0,
        score=metadata.get("score"),
        box_xyxy=tuple(metadata.get("box_xyxy", ())),
        backend=dict(metadata.get("backend", {})),
        inference_latency_s=metadata.get("inference_latency_s"),
    )


class Sam3MaskInferenceService:
    """Serialize access to one resident segmentation model."""

    def __init__(self, backend: Sam3MaskBackend) -> None:
        self.backend = backend
        self._lock = threading.Lock()

    def infer(self, request: Sam3MaskRequest) -> Sam3MaskObservation:
        if not self._lock.acquire(blocking=False):
            raise Sam3MaskServiceError("mask_backend_busy")
        try:
            import time

            started = time.perf_counter()
            mask, score, box = self.backend.segment(request.rgb, request.prompt)
            latency = time.perf_counter() - started
        finally:
            self._lock.release()
        mask_array = np.asarray(mask)
        if mask_array.shape != request.rgb.shape[:2]:
            raise Sam3MaskServiceError("backend mask shape differs from RGB")
        return Sam3MaskObservation(
            request_id=request.request_id,
            frame_id=request.frame_id,
            capture_timestamp_s=request.capture_timestamp_s,
            prompt=request.prompt,
            mask=np.asarray(mask_array > 0),
            score=float(score),
            box_xyxy=tuple(np.asarray(box, dtype=float).tolist()),
            backend=dict(self.backend.metadata),
            inference_latency_s=latency,
        )


def create_mask_http_server(
    host: str,
    port: int,
    service: Sam3MaskInferenceService,
    *,
    auth_token: str | None = None,
    max_request_bytes: int = 16 * 1024 * 1024,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def _authorized(self) -> bool:
            if auth_token is None:
                return True
            expected = f"Bearer {auth_token}"
            return hmac.compare_digest(self.headers.get("Authorization", ""), expected)

        def _json(self, status: int, payload: Mapping[str, Any]) -> None:
            encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._json(
                    HTTPStatus.NOT_FOUND, {"schema": MASK_ERROR_SCHEMA, "error": "not_found"}
                )
                return
            if not self._authorized():
                self._json(
                    HTTPStatus.UNAUTHORIZED, {"schema": MASK_ERROR_SCHEMA, "error": "unauthorized"}
                )
                return
            self._json(
                HTTPStatus.OK,
                {
                    "schema": MASK_HEALTH_SCHEMA,
                    "ready": True,
                    "backend": dict(service.backend.metadata),
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/segment":
                self._json(
                    HTTPStatus.NOT_FOUND, {"schema": MASK_ERROR_SCHEMA, "error": "not_found"}
                )
                return
            if not self._authorized():
                self._json(
                    HTTPStatus.UNAUTHORIZED, {"schema": MASK_ERROR_SCHEMA, "error": "unauthorized"}
                )
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0
            if size <= 0 or size > max_request_bytes:
                self._json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"schema": MASK_ERROR_SCHEMA, "error": "request_size"},
                )
                return
            try:
                result = service.infer(decode_mask_request(self.rfile.read(size)))
                encoded = encode_mask_observation(result)
            except Exception as error:  # noqa: BLE001 - boundary maps errors to one contract
                self._json(
                    HTTPStatus.CONFLICT
                    if str(error) == "mask_backend_busy"
                    else HTTPStatus.BAD_REQUEST,
                    {
                        "schema": MASK_ERROR_SCHEMA,
                        "error": type(error).__name__,
                        "detail": str(error),
                    },
                )
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", MASK_CONTENT_TYPE)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return ThreadingHTTPServer((host, port), Handler)


class Sam3MaskClient:
    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float = 3.0,
        auth_token: str | None = None,
        max_response_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.auth_token = auth_token
        self.max_response_bytes = int(max_response_bytes)

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_token is not None:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def health(self) -> Mapping[str, Any]:
        request = urllib.request.Request(f"{self.endpoint}/health", headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read(self.max_response_bytes).decode())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise Sam3MaskServiceError(f"mask health request failed: {error}") from error
        if not isinstance(payload, Mapping) or payload.get("schema") != MASK_HEALTH_SCHEMA:
            raise Sam3MaskServiceError("mask health response schema is invalid")
        return payload

    def segment(
        self,
        *,
        frame_id: str,
        capture_timestamp_s: float,
        rgb: np.ndarray,
        prompt: str,
    ) -> Sam3MaskObservation:
        request_id = str(uuid.uuid4())
        request_value = Sam3MaskRequest(request_id, frame_id, capture_timestamp_s, prompt, rgb)
        http = urllib.request.Request(
            f"{self.endpoint}/segment",
            data=encode_mask_request(request_value),
            headers={"Content-Type": MASK_CONTENT_TYPE, **self._headers()},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http, timeout=self.timeout_s) as response:
                payload = response.read(self.max_response_bytes + 1)
        except (OSError, urllib.error.URLError) as error:
            raise Sam3MaskServiceError(f"mask inference request failed: {error}") from error
        if len(payload) > self.max_response_bytes:
            raise Sam3MaskServiceError("mask inference response exceeds size bound")
        result = decode_mask_observation(payload)
        if (
            result.request_id != request_id
            or result.frame_id != frame_id
            or result.capture_timestamp_s != capture_timestamp_s
            or result.prompt != prompt
            or result.mask.shape != np.asarray(rgb).shape[:2]
        ):
            raise Sam3MaskServiceError("mask response identity does not match request")
        return result


__all__ = [
    "MASK_CONTENT_TYPE",
    "MASK_HEALTH_SCHEMA",
    "MASK_REQUEST_SCHEMA",
    "MASK_RESPONSE_SCHEMA",
    "Sam3MaskBackend",
    "Sam3MaskClient",
    "Sam3MaskInferenceService",
    "Sam3MaskObservation",
    "Sam3MaskRequest",
    "Sam3MaskServiceError",
    "create_mask_http_server",
    "decode_mask_observation",
    "decode_mask_request",
    "encode_mask_observation",
    "encode_mask_request",
]
