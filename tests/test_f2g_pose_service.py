from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from demo2gripper.f2g_pose_service import (
    BackendPrediction,
    FakePoseBackend,
    ForegroundDepthGateConfig,
    PoseBackendBusy,
    PoseInferenceService,
    PoseRequest,
    PoseServiceClient,
    PoseServiceTimeout,
    RemotePoseProvider,
    create_pose_http_server,
    decode_pose_request,
    encode_pose_request,
    evaluate_foreground_depth,
)
from demo2gripper.pose_observation import (
    PoseCapture,
    PoseObservationError,
    PoseObservationGuard,
    PoseObservationGuardConfig,
    PoseObservationRejected,
    PoseObservationReplay,
)


def _arrays(*, background_dominated: bool = False):
    height, width = 64, 64
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 1] = 80
    depth = np.full((height, width), 0.60, dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[16:48, 24:40] = 1
    if background_dominated:
        target_pixels = np.argwhere(mask > 0)
        object_count = int(round(0.20 * len(target_pixels)))
        for row, column in target_pixels[:object_count]:
            depth[row, column] = 0.30
    else:
        depth[mask > 0] = 0.30
    K = np.array([[600.0, 0.0, 32.0], [0.0, 600.0, 32.0], [0.0, 0.0, 1.0]])
    return rgb, depth, K, mask


def _request(
    *,
    sequence: int = 1,
    timestamp_s: float = 10.0,
    background_dominated: bool = False,
) -> PoseRequest:
    rgb, depth, K, mask = _arrays(background_dominated=background_dominated)
    return PoseRequest(
        request_id=f"request-{sequence}",
        capture=PoseCapture(
            sequence=sequence,
            frame_id=f"front:{sequence}",
            timestamp_s=timestamp_s,
            clock_domain="client_monotonic",
        ),
        rgb=rgb,
        depth_m=depth,
        K=K,
        mask=mask,
        mask_source="unit_test_mask",
        mask_source_score=0.9,
    )


def test_request_npz_round_trip_has_no_pickle_and_preserves_identity() -> None:
    request = _request()
    restored = decode_pose_request(encode_pose_request(request))
    assert restored.request_id == request.request_id
    assert restored.capture == request.capture
    assert restored.mask_source == "unit_test_mask"
    assert np.array_equal(restored.rgb, request.rgb)
    assert np.array_equal(restored.depth_m, request.depth_m)
    assert np.array_equal(restored.K, request.K)
    assert np.array_equal(restored.mask, request.mask)


def test_coherent_foreground_cluster_passes_and_sanitizes_depth() -> None:
    result = evaluate_foreground_depth(_request(), ForegroundDepthGateConfig())
    diagnostics = result.diagnostics
    assert diagnostics.gate_passed
    assert diagnostics.selected_median_m == pytest.approx(0.30)
    assert diagnostics.selected_cluster_fraction == pytest.approx(1.0)
    assert diagnostics.outside_ring_median_m == pytest.approx(0.60)
    assert diagnostics.mask_near_background_fraction == pytest.approx(0.0)
    assert np.count_nonzero(result.sanitized_depth_m) == 512
    assert np.max(result.sanitized_depth_m) == pytest.approx(0.30)


def test_background_dominated_transparent_mask_fails_even_with_many_valid_pixels() -> None:
    config = ForegroundDepthGateConfig(
        min_cluster_pixels=50,
        min_cluster_fraction=0.10,
    )
    result = evaluate_foreground_depth(
        _request(background_dominated=True), config
    ).diagnostics
    assert result.valid_mask_pixels == 512  # far above F2G's 50-pixel minimum
    assert result.mask_near_background_fraction == pytest.approx(0.80, abs=0.01)
    assert not result.gate_passed
    assert "mask_depth_dominated_by_neighbouring_background" in result.reject_reasons


def test_minority_support_returns_in_ring_do_not_veto_coherent_foreground() -> None:
    rgb, depth, K, mask = _arrays()
    # A near support occupies part of the ring around the mask, while the
    # dominant neighbouring background remains the 0.60 m plane.  The support
    # samples must not make every 0.30 m foreground pixel look like background.
    depth[14:16, 22:42] = 0.30
    request = PoseRequest(
        request_id="mixed-ring",
        capture=PoseCapture(1, "front:mixed-ring", 10.0, "client_monotonic"),
        rgb=rgb,
        depth_m=depth,
        K=K,
        mask=mask,
        mask_source="unit_test_mask",
        mask_source_score=0.9,
    )

    result = evaluate_foreground_depth(request, ForegroundDepthGateConfig()).diagnostics

    assert result.gate_passed
    assert result.selected_median_m == pytest.approx(0.30)
    assert result.outside_ring_median_m == pytest.approx(0.60)
    assert result.mask_near_background_fraction == pytest.approx(0.0)


def test_rejected_depth_never_calls_backend() -> None:
    class _Backend(FakePoseBackend):
        calls = 0

        def predict_pose_only(self, rgb, depth_m, K, mask):
            self.calls += 1
            return super().predict_pose_only(rgb, depth_m, K, mask)

    backend = _Backend()
    service = PoseInferenceService(backend)
    result = service.infer(_request(background_dominated=True))
    assert not result.accepted
    assert result.T_camera_object is None
    assert backend.calls == 0
    with pytest.raises(PoseObservationRejected):
        result.require_pose()


def test_fake_http_service_and_local_provider_complete_cpu_round_trip() -> None:
    service = PoseInferenceService(FakePoseBackend())
    server = create_pose_http_server("127.0.0.1", 0, service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = PoseServiceClient(
            f"http://127.0.0.1:{server.server_port}", timeout_s=1.0
        )
        assert client.health()["ready"] is True
        provider = RemotePoseProvider(
            client,
            guard_config=PoseObservationGuardConfig(
                max_staleness_s=1.0,
                max_server_latency_s=1.0,
                max_roundtrip_latency_s=1.0,
                allowed_backends=frozenset({"fake_pose"}),
            ),
        )
        rgb, depth, K, mask = _arrays()
        result = provider.estimate(
            frame_id="front:42",
            capture_timestamp_s=time.monotonic(),
            rgb=rgb,
            depth_m=depth,
            K=K,
            mask=mask,
            mask_source="unit_test_mask",
        )
        transform, size = result.require_pose()
        assert transform[2, 3] == pytest.approx(0.30)
        assert size == pytest.approx((0.018, 0.018, 0.120))
        assert result.pose_quality.axis_sign_status == "unresolved"
        assert result.pose_quality.cylindrical_roll_status == "unresolved"
        assert result.pose_quality.task_qualified is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_guard_rejects_replay_staleness_sequence_and_backend() -> None:
    observation = PoseInferenceService(FakePoseBackend()).infer(_request())
    guard = PoseObservationGuard(
        PoseObservationGuardConfig(
            max_staleness_s=0.5,
            max_server_latency_s=1.0,
            max_roundtrip_latency_s=1.0,
            allowed_backends=frozenset({"fake_pose"}),
        )
    )
    guard.validate(
        observation,
        expected_request_id="request-1",
        expected_sequence=1,
        expected_frame_id="front:1",
        expected_timestamp_s=10.0,
        received_at_monotonic_s=10.1,
        roundtrip_latency_s=0.05,
    )
    with pytest.raises(PoseObservationReplay, match="repeated or regressed"):
        guard.validate(
            observation,
            expected_request_id="request-1",
            expected_sequence=1,
            expected_frame_id="front:1",
            expected_timestamp_s=10.0,
            received_at_monotonic_s=10.2,
            roundtrip_latency_s=0.05,
        )
    stale_guard = PoseObservationGuard(
        PoseObservationGuardConfig(
            max_staleness_s=0.05,
            max_server_latency_s=1.0,
            max_roundtrip_latency_s=1.0,
            allowed_backends=frozenset({"fake_pose"}),
        )
    )
    with pytest.raises(PoseObservationError, match="stale"):
        stale_guard.validate(
            observation,
            expected_request_id="request-1",
            expected_sequence=1,
            expected_frame_id="front:1",
            expected_timestamp_s=10.0,
            received_at_monotonic_s=10.2,
            roundtrip_latency_s=0.05,
        )
    with pytest.raises(PoseObservationReplay, match="sequence"):
        PoseObservationGuard(
            PoseObservationGuardConfig(
                max_server_latency_s=1.0,
                max_roundtrip_latency_s=1.0,
                allowed_backends=frozenset({"fake_pose"}),
            )
        ).validate(
            observation,
            expected_request_id="request-1",
            expected_sequence=2,
            expected_frame_id="front:1",
            expected_timestamp_s=10.0,
            received_at_monotonic_s=10.1,
            roundtrip_latency_s=0.05,
        )


def test_http_client_maps_deadline_to_explicit_timeout(monkeypatch) -> None:
    def _timeout(*args, **kwargs):
        del args, kwargs
        raise TimeoutError("deadline")

    monkeypatch.setattr("urllib.request.urlopen", _timeout)
    with pytest.raises(PoseServiceTimeout):
        PoseServiceClient("http://127.0.0.1:9", timeout_s=0.01).health()


def test_resident_backend_rejects_overlap_instead_of_queueing() -> None:
    started = threading.Event()
    release = threading.Event()

    class _BlockingBackend:
        metadata = {
            "name": "fake_pose",
            "implementation_version": "blocking_test",
            "code_revision": "none",
            "checkpoint_sha256": "none",
            "device": "cpu",
            "pose_only": True,
        }

        def predict_pose_only(self, rgb, depth_m, K, mask):
            del rgb, depth_m, K, mask
            started.set()
            assert release.wait(timeout=2.0)
            return BackendPrediction(np.eye(4), np.ones(3) * 0.02, 0.0, 0.0)

    service = PoseInferenceService(_BlockingBackend())
    first_error: list[BaseException] = []

    def _first_call() -> None:
        try:
            service.infer(_request())
        except BaseException as error:  # propagate from the worker after joining
            first_error.append(error)

    thread = threading.Thread(target=_first_call)
    thread.start()
    assert started.wait(timeout=1.0)
    try:
        with pytest.raises(PoseBackendBusy):
            service.infer(_request(sequence=2))
    finally:
        release.set()
        thread.join(timeout=2.0)
    assert not first_error
