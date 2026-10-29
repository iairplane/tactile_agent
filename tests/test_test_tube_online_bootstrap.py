from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demo2gripper.sam3_mask_service import (
    Sam3MaskClient,
    Sam3MaskInferenceService,
    Sam3MaskObservation,
    Sam3MaskRequest,
    create_mask_http_server,
    decode_mask_observation,
    decode_mask_request,
    encode_mask_observation,
    encode_mask_request,
)
from demo2gripper.test_tube_closed_loop import (
    SOURCE_POSE_CAPABILITIES,
    ClosedLoopContractError,
    NormalizedDemoGraspRelation,
    ObjectPoseEstimate,
    PoseBackendDescriptor,
    grounded_grasp_target,
    parse_geometry,
)
from demo2gripper.test_tube_closed_loop import (
    TestTubeClosedLoopGeometry as ClosedLoopGeometry,
)
from demo2gripper.test_tube_live_io import (
    SceneEvidence,
    _bind_online_scene_camera,
    assess_live_config,
)
from demo2gripper.test_tube_online_scene import (
    OnlineSourceTubeSceneInterpreter,
    OnlineTubeBootstrapConfig,
    OnlineTubeSeedBuilder,
    OnlineTubeSizeConfig,
    OnlineTubeSizeFilter,
    create_online_test_tube_scene_interpreter,
)
from demo2gripper.test_tube_online_socket import (
    OnlineSocketEntryTracker,
    OnlineSocketSceneInterpreter,
    OnlineSocketTrackerConfig,
    SocketEntryCandidate,
)
from demo2gripper.test_tube_pose_tracker import TubePoseTrackingError


def _mask_observation(mask: np.ndarray, *, prompt: str, score: float = 0.9) -> Sam3MaskObservation:
    return Sam3MaskObservation(
        request_id="request",
        frame_id="frame",
        capture_timestamp_s=1.0,
        prompt=prompt,
        mask=mask,
        score=score,
        box_xyxy=(0.0, 0.0, float(mask.shape[1]), float(mask.shape[0])),
        backend={"name": "sam3"},
        inference_latency_s=0.01,
    )


def test_sam3_mask_wire_contract_and_http_identity() -> None:
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    request = Sam3MaskRequest("r1", "f1", 2.0, "test tube", rgb)
    decoded = decode_mask_request(encode_mask_request(request))
    assert decoded.prompt == "test tube"
    assert decoded.rgb.shape == rgb.shape

    class Backend:
        metadata = {"name": "sam3", "text_prompted": True}

        def segment(self, image, prompt):
            assert image.shape == rgb.shape
            assert prompt == "test tube"
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[4:20, 12:18] = True
            return mask, 0.92, np.array([12, 4, 18, 20], dtype=float)

    server = create_mask_http_server("127.0.0.1", 0, Sam3MaskInferenceService(Backend()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = Sam3MaskClient(f"http://127.0.0.1:{server.server_port}")
        assert client.health()["ready"] is True
        result = client.segment(frame_id="f1", capture_timestamp_s=2.0, rgb=rgb, prompt="test tube")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
    assert result.score == pytest.approx(0.92)
    assert int(result.mask.sum()) == 96
    assert decode_mask_observation(encode_mask_observation(result)).request_id == result.request_id


def test_online_tube_seed_uses_source_support_to_sign_free_endpoint() -> None:
    tube = np.zeros((100, 120), dtype=bool)
    tube[45:56, 20:101] = True
    support = np.zeros_like(tube)
    support[35:70, 0:25] = True
    builder = OnlineTubeSeedBuilder(
        OnlineTubeBootstrapConfig(
            min_stable_frames=3,
            min_axis_length_px=30.0,
            min_mask_pixels=100,
            min_support_mask_pixels=100,
            min_support_assignment_margin_px=20.0,
            max_near_support_distance_px=30.0,
        )
    )
    observations = [
        builder.update(
            tube=_mask_observation(tube, prompt="tube"),
            support=_mask_observation(support, prompt="white paper cup"),
            image_shape=tube.shape,
        )
        for _ in range(3)
    ]
    assert observations[:2] == [None, None]
    assert observations[2] is not None
    assert observations[2].open_uv[0] > observations[2].closed_uv[0]
    assert observations[2].axis_sign_source == "source_support_geometry"

    ambiguous_support = np.zeros_like(tube)
    ambiguous_support[30:70, 50:71] = True
    assert (
        builder.update(
            tube=_mask_observation(tube, prompt="tube"),
            support=_mask_observation(ambiguous_support, prompt="ambiguous source support"),
            image_shape=tube.shape,
        )
        is None
    )
    assert "source_support_does_not_resolve_tube_axis_sign" in builder.last_reasons


def test_online_tube_seed_rejects_support_prompt_that_contains_tube() -> None:
    tube = np.zeros((100, 120), dtype=bool)
    tube[45:56, 20:101] = True
    builder = OnlineTubeSeedBuilder(
        OnlineTubeBootstrapConfig(
            min_stable_frames=1,
            min_axis_length_px=30.0,
            min_mask_pixels=100,
            min_support_mask_pixels=100,
        )
    )
    assert (
        builder.update(
            tube=_mask_observation(tube, prompt="tube"),
            support=_mask_observation(tube, prompt="bad open-rim-like result"),
            image_shape=tube.shape,
        )
        is None
    )
    assert "source_support_mask_contains_too_much_tube" in builder.last_reasons


def test_online_tube_seed_requires_stable_support_identity() -> None:
    tube = np.zeros((100, 120), dtype=bool)
    tube[45:56, 20:101] = True
    first_support = np.zeros_like(tube)
    first_support[25:65, 0:25] = True
    shifted_support = np.zeros_like(tube)
    shifted_support[45:85, 0:25] = True
    builder = OnlineTubeSeedBuilder(
        OnlineTubeBootstrapConfig(
            min_stable_frames=2,
            min_axis_length_px=30.0,
            min_mask_pixels=100,
            min_support_mask_pixels=100,
            min_support_assignment_margin_px=20.0,
            max_near_support_distance_px=30.0,
        )
    )
    assert (
        builder.update(
            tube=_mask_observation(tube, prompt="tube"),
            support=_mask_observation(first_support, prompt="white paper cup"),
            image_shape=tube.shape,
        )
        is None
    )
    assert (
        builder.update(
            tube=_mask_observation(tube, prompt="tube"),
            support=_mask_observation(shifted_support, prompt="white paper cup"),
            image_shape=tube.shape,
        )
        is None
    )
    assert builder.last_reasons == ("online_source_support_identity_unstable",)


def test_base_z_axis_fallback_requires_qualified_extrinsic_and_height_separation() -> None:
    tube = np.zeros((120, 100), dtype=bool)
    tube[15:106, 45:56] = True
    depth = np.zeros_like(tube, dtype=float)
    depth[tube] = 1.0
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])
    # Camera +v points toward base -Z.  The upper image endpoint is therefore
    # the higher/free endpoint in the qualified base frame.
    T_base_camera = np.eye(4)
    T_base_camera[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    config = OnlineTubeBootstrapConfig(
        min_stable_frames=1,
        min_axis_length_px=30.0,
        min_mask_pixels=100,
        allow_qualified_base_z_fallback=True,
        min_base_z_separation_m=0.05,
    )
    common = dict(
        tube=_mask_observation(tube, prompt="tube"),
        support=None,
        image_shape=tube.shape,
        depth_m=depth,
        K=K,
        T_base_camera=T_base_camera,
    )
    unqualified = OnlineTubeSeedBuilder(config)
    assert unqualified.update(**common, camera_extrinsic_qualified=False) is None
    assert "qualified_base_z_does_not_resolve_tube_axis_sign" in unqualified.last_reasons

    qualified = OnlineTubeSeedBuilder(config)
    result = qualified.update(**common, camera_extrinsic_qualified=True)
    assert result is not None
    assert result.open_uv[1] < result.closed_uv[1]
    assert result.axis_sign_source == "qualified_base_z"

    flat = OnlineTubeSeedBuilder(config)
    assert (
        flat.update(
            **(common | {"T_base_camera": np.eye(4)}),
            camera_extrinsic_qualified=True,
        )
        is None
    )


def test_online_scene_factory_configures_support_prompt_without_open_rim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeClient:
        def __init__(self, endpoint, *, timeout_s, auth_token):
            assert endpoint == "http://sam3"
            assert timeout_s == pytest.approx(2.0)
            assert auth_token is None

        def health(self):
            return {"ready": True, "backend": {"name": "sam3"}}

    monkeypatch.setattr("demo2gripper.test_tube_online_scene.Sam3MaskClient", FakeClient)
    interpreter = create_online_test_tube_scene_interpreter(
        config={
            "sam3_service_url": "http://sam3",
            "sam3_timeout_s": 2.0,
            "camera_serial": "d435",
            "tube_prompt": "test tube",
            "source_support_prompt": "white paper cup",
            "tube_bootstrap": {
                "source_roi_uvxy": [10, 20, 100, 110],
                "support_roi_uvxy": [0, 80, 120, 120],
            },
        },
        output_dir=tmp_path,
    )
    assert isinstance(interpreter, OnlineSourceTubeSceneInterpreter)
    assert interpreter._tube_prompt == "test tube"
    assert interpreter._support_prompt == "white paper cup"


def test_online_scene_resegments_textureless_tube_instead_of_optical_flow() -> None:
    tube = np.zeros((100, 120), dtype=bool)
    tube[45:56, 20:101] = True
    support = np.zeros_like(tube)
    support[35:70, 0:25] = True

    class Provider:
        calls = 0
        support_score = 0.9

        def segment(self, *, frame_id, capture_timestamp_s, rgb, prompt):
            del frame_id, capture_timestamp_s, rgb
            self.calls += 1
            mask = support if "cup" in prompt else tube
            score = self.support_score if "cup" in prompt else 0.9
            return _mask_observation(mask, prompt=prompt, score=score)

    provider = Provider()
    interpreter = OnlineSourceTubeSceneInterpreter(
        mask_provider=provider,
        expected_camera_serial="d435",
        tube_prompt="transparent test tube",
        source_support_prompt="white paper cup",
        bootstrap_config=OnlineTubeBootstrapConfig(
            min_stable_frames=1,
            min_axis_length_px=30.0,
            min_mask_pixels=100,
            min_support_mask_pixels=100,
            min_support_assignment_margin_px=20.0,
            max_near_support_distance_px=30.0,
        ),
    )
    rgb = np.zeros((*tube.shape, 3), dtype=np.uint8)
    depth = np.full(tube.shape, 1.0, dtype=np.float32)
    depth[tube] = 0.5
    K = np.array([[200.0, 0.0, 60.0], [0.0, 200.0, 50.0], [0.0, 0.0, 1.0]])
    state = SimpleNamespace(value="observe")
    observations = []
    for sequence in (1, 2):
        frame = SimpleNamespace(
            serial="d435",
            sequence=sequence,
            host_received_monotonic_s=float(sequence),
            rgb=rgb,
            depth_m=depth,
            intrinsics=SimpleNamespace(matrix=K),
        )
        observations.append(
            interpreter.observe(frame=frame, state=state, T_base_tcp=np.eye(4))
        )
    assert observations[0].tube is not None
    assert observations[1].tube is not None
    assert provider.calls == 4
    assert interpreter._last_track.score_provenance == "consecutive_sam3_mask_iou"
    assert not interpreter.source_depth_only_loss
    assert not interpreter.source_rgb_identity_retained

    # A preceding metric-qualified frame supplies the fixed RGB reference.
    # Losing only transparent depth must preserve visibility evidence without
    # manufacturing a source pose. A model failure must clear that evidence.
    interpreter._qualified_tracking_id = interpreter._last_track.tracking_id
    interpreter._qualified_image_mask = tube.copy()
    interpreter._qualified_image_ends = interpreter._seed_ends
    depth[:] = 1.0
    frame.sequence = 3
    frame.host_received_monotonic_s = 3.0
    missing_depth = interpreter.observe(frame=frame, state=state, T_base_tcp=np.eye(4))
    assert missing_depth.tube is None
    assert interpreter.source_depth_only_loss
    assert interpreter.source_rgb_identity_retained

    # Losing the cup/sign cue does not erase a current, unchanged tube image
    # identity.  It still cannot manufacture a renewed signed/metric pose.
    depth[tube] = 0.5
    provider.support_score = 0.1
    frame.sequence = 4
    frame.host_received_monotonic_s = 4.0
    missing_sign = interpreter.observe(frame=frame, state=state, T_base_tcp=np.eye(4))
    assert missing_sign.tube is None
    assert not interpreter.source_depth_only_loss
    assert interpreter.source_rgb_identity_retained
    provider.support_score = 0.9

    # Gradual endpoint drift is compared with the metric-qualified reference,
    # not just the preceding frame's mask.
    from dataclasses import replace

    reference = interpreter._qualified_image_ends
    interpreter._seed_ends = replace(
        reference, open_uv=tuple(np.asarray(reference.open_uv) + [9.0, 0.0])
    )
    assert not interpreter._image_matches_qualified_source()

    def failed_provider(**kwargs):
        raise ConnectionError("service disconnected")

    provider.segment = failed_provider
    frame.sequence = 5
    frame.host_received_monotonic_s = 5.0
    missing_service = interpreter.observe(frame=frame, state=state, T_base_tcp=np.eye(4))
    assert missing_service.tube is None
    assert not interpreter.source_depth_only_loss
    assert not interpreter.source_rgb_identity_retained


def _socket_candidate(x: float, *, score: float = 0.9) -> SocketEntryCandidate:
    transform = np.eye(4)
    transform[:3, 3] = [x, 0.0, 0.5]
    return SocketEntryCandidate(
        T_camera_entry=transform,
        score=score,
        position_std_m=0.001,
        orientation_std_deg=1.0,
        centre_uv=(320.0 + 1000.0 * x, 240.0),
        evidence_id=f"candidate-{x}",
    )


def test_online_socket_tracker_is_temporal_and_fails_closed_on_ambiguity() -> None:
    tracker = OnlineSocketEntryTracker(OnlineSocketTrackerConfig(min_stable_frames=3))
    assert tracker.update(timestamp_s=1.0, candidates=[_socket_candidate(0.0)]) is None
    assert tracker.update(timestamp_s=1.1, candidates=[_socket_candidate(0.001)]) is None
    result = tracker.update(timestamp_s=1.2, candidates=[_socket_candidate(0.002)])
    assert result is not None
    assert result.T_camera_entry[0, 3] == pytest.approx(0.001)

    assert (
        tracker.update(
            timestamp_s=1.3,
            candidates=[_socket_candidate(0.0, score=0.90), _socket_candidate(0.03, score=0.86)],
        )
        is None
    )
    assert tracker.last_reasons == ("socket_candidate_missing_or_ambiguous",)

    # A large score margin is still not a semantic target selector.  A rack
    # with multiple holes requires a demo-derived hint or a tighter ROI.
    assert (
        tracker.update(
            timestamp_s=1.4,
            candidates=[_socket_candidate(0.0, score=0.99), _socket_candidate(0.03, score=0.60)],
        )
        is None
    )


def test_online_socket_tracker_rebuilds_confidence_after_frame_gap() -> None:
    tracker = OnlineSocketEntryTracker(
        OnlineSocketTrackerConfig(min_stable_frames=3, max_age_s=0.15)
    )
    assert tracker.update(timestamp_s=1.0, candidates=[_socket_candidate(0.0)]) is None
    assert tracker.update(timestamp_s=1.1, candidates=[_socket_candidate(0.0)]) is None
    # The old two-frame history is stale and cannot combine with this sample.
    assert tracker.update(timestamp_s=1.4, candidates=[_socket_candidate(0.0)]) is None
    assert tracker.last_reasons == ("socket_temporal_window_not_full",)


def test_online_socket_observe_burst_qualifies_at_camera_rate_without_repeating_downstream(
) -> None:
    class Downstream:
        def __init__(self) -> None:
            self.observe_calls = 0
            self.socket_poses: list[np.ndarray] = []

        def set_socket_pose(self, value: np.ndarray) -> None:
            self.socket_poses.append(value.copy())

        def observe(self, *, frame, state, T_base_tcp):
            del state
            del T_base_tcp
            self.observe_calls += 1
            return {"sequence": frame.sequence}

    def frame(sequence: int):
        return type(
            "Frame",
            (),
            {
                "sequence": sequence,
                "serial": "d435",
                "host_received_monotonic_s": 1.0 + 0.05 * sequence,
                "rgb": np.zeros((3, 4, 3), dtype=np.uint8),
                "depth_m": np.ones((3, 4), dtype=float),
                "intrinsics": type("Intrinsics", (), {"matrix": np.eye(3)})(),
            },
        )()

    downstream = Downstream()
    tracker = OnlineSocketEntryTracker(
        OnlineSocketTrackerConfig(min_stable_frames=4, max_age_s=0.25)
    )
    interpreter = OnlineSocketSceneInterpreter(
        T_base_camera=np.eye(4),
        expected_camera_serial="d435",
        tracker=tracker,
        candidate_extractor=lambda *_: (_socket_candidate(0.0),),
        downstream=downstream,
    )
    remaining = iter((frame(2), frame(3), frame(4)))

    final_frame, evidence = interpreter.observe_burst(
        first_frame=frame(1),
        read_frame=lambda: next(remaining),
        state="OBSERVE",
        T_base_tcp=np.eye(4),
    )

    assert final_frame.sequence == 4
    assert evidence == {"sequence": 4}
    assert downstream.observe_calls == 1
    assert len(downstream.socket_poses) == 1
    assert interpreter.socket_pose(frame=final_frame) is not None


def test_online_source_forwards_socket_burst_and_processes_only_final_frame() -> None:
    class Provider:
        def __init__(self) -> None:
            self.frame_ids: list[str] = []

        def segment(self, *, frame_id, prompt, **_):
            self.frame_ids.append(frame_id)
            mask = np.zeros((100, 120), dtype=bool)
            if prompt == "test tube":
                mask[45:56, 20:101] = True
            else:
                mask[35:70, 0:25] = True
            return _mask_observation(mask, prompt=prompt)

    class HeldDownstream:
        def __init__(self) -> None:
            self.observe_sequences: list[int] = []
            self.socket_poses: list[np.ndarray] = []

        def set_socket_pose(self, value: np.ndarray) -> None:
            self.socket_poses.append(value.copy())

        def observe(self, *, frame, **_):
            self.observe_sequences.append(frame.sequence)
            return SceneEvidence(tube=None, socket=None)

    K = np.array([[100.0, 0.0, 60.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])

    def frame(sequence: int):
        return SimpleNamespace(
            sequence=sequence,
            serial="d435",
            host_received_monotonic_s=1.0 + 0.05 * sequence,
            rgb=np.zeros((100, 120, 3), dtype=np.uint8),
            depth_m=np.ones((100, 120), dtype=float),
            intrinsics=SimpleNamespace(matrix=K),
        )

    held = HeldDownstream()
    socket = OnlineSocketSceneInterpreter(
        T_base_camera=np.eye(4),
        expected_camera_serial="d435",
        tracker=OnlineSocketEntryTracker(
            OnlineSocketTrackerConfig(min_stable_frames=4, max_age_s=0.25)
        ),
        candidate_extractor=lambda *_: (_socket_candidate(0.0),),
        downstream=held,
    )
    provider = Provider()
    source = OnlineSourceTubeSceneInterpreter(
        mask_provider=provider,
        expected_camera_serial="d435",
        tube_prompt="test tube",
        source_support_prompt="white paper cup",
        bootstrap_config=OnlineTubeBootstrapConfig(
            min_stable_frames=3,
            min_axis_length_px=30.0,
            min_mask_pixels=100,
            min_support_mask_pixels=100,
        ),
        downstream=socket,
    )
    remaining = iter((frame(2), frame(3), frame(4)))

    final_frame, evidence = source.observe_burst(
        first_frame=frame(1),
        read_frame=lambda: next(remaining),
        state=SimpleNamespace(value="OBSERVE"),
        T_base_tcp=np.eye(4),
    )

    assert final_frame.sequence == 4
    assert evidence.tube is None
    assert held.observe_sequences == [4]
    assert len(held.socket_poses) == 1
    assert provider.frame_ids == ["d435-d435-4", "d435-d435-4"]
    assert source.socket_pose(frame=final_frame) is not None


def test_f2g_tube_size_requires_shape_and_temporal_agreement() -> None:
    size_filter = OnlineTubeSizeFilter(
        OnlineTubeSizeConfig(min_stable_frames=3, max_relative_window_deviation=0.10)
    )
    assert size_filter.update((0.020, 0.021, 0.100)) is None
    assert size_filter.update((0.019, 0.020, 0.101)) is None
    qualified = size_filter.update((0.020, 0.020, 0.099))
    assert qualified is not None
    assert max(qualified) == pytest.approx(0.100)

    assert size_filter.update((0.020, 0.020, 0.140)) is None
    assert size_filter.last_reasons == ("runtime_f2g_tube_size_unstable",)
    assert size_filter.update((0.060, 0.061, 0.070)) is None
    assert "runtime_f2g_shape_not_tube_like" in size_filter.last_reasons


@pytest.fixture
def qualified_size_source():
    lengths = []
    source = OnlineSourceTubeSceneInterpreter(
        mask_provider=SimpleNamespace(),
        expected_camera_serial="d435",
        tube_prompt="test tube",
        source_support_prompt="cup",
        downstream=SimpleNamespace(set_tube_length_m=lengths.append),
    )
    result = SimpleNamespace(decision="TRACKED", reasons=(), pose=None)
    frame = SimpleNamespace(intrinsics=SimpleNamespace(matrix=np.eye(3)))
    mask = SimpleNamespace(tracking_id="tube-1")
    proposal = SimpleNamespace(
        accepted=True, T_camera_object=np.eye(4), size_m=(.02, .02, .1), reject_reasons=(),
    )

    def attach_qualifier():
        source._qualifier = SimpleNamespace(update=lambda **kwargs: result, reset=lambda: None)
        source._seed_ends = SimpleNamespace(warped=lambda transform: None)
        source._last_track = SimpleNamespace(
            tracking_id=mask.tracking_id, seed_to_current=np.eye(3)
        )

    def observe(length=.100, *, open_x=0.0, position_std_m=.001, orientation_std_deg=1.0):
        transform = np.eye(4)
        transform[:3, 3] = [open_x, 0.0, 1.0 - length / 2]
        result.pose = SimpleNamespace(
            T_camera_tube=transform, size_m=(.020, .020, length), position_std_m=position_std_m,
            orientation_std_deg=orientation_std_deg, axis_sign_source="source_support_geometry",
        )
        return source.qualify_pose(object_id="tube", proposal=proposal, mask=mask, frame=frame)

    attach_qualifier()
    return source, lengths, proposal, mask, result, observe, attach_qualifier


def test_source_freezes_dimensions_once_and_preserves_open_end(qualified_size_source) -> None:
    source, lengths, _, _, _, observe, _ = qualified_size_source
    assert observe(.100) is None
    assert observe(.102) is None
    first = observe(.101)
    assert first is not None
    assert lengths == pytest.approx([.101])
    for length in (.108, .110, .109):
        pose = observe(length)
        assert pose is not None
        assert pose.size_m == first.size_m
        assert pose.axis_sign_source == "source_support_geometry"
        assert pose.orientation_std_deg == 1.0
        assert pose.position_std_m >= .001
        # Fixed dimensions and the canonical center still describe the same mouth.
        mouth = pose.T_camera_object[:3, 3] + pose.T_camera_object[:3, 2] * max(pose.size_m) / 2
        assert mouth == pytest.approx([0.0, 0.0, 1.0])
    assert lengths == pytest.approx([.101])
    assert source.estimated_tube_length_m == pytest.approx(.101)


def test_source_dropout_requalifies_pose_without_rebinding_size(qualified_size_source) -> None:
    source, lengths, proposal, _, result, observe, attach = qualified_size_source
    for _ in range(3):
        observe()
    proposal.accepted = False
    proposal.reject_reasons = ("bad_current_depth",)
    assert observe() is None
    assert source.last_hold_reasons == ("bad_current_depth",)
    assert source.estimated_tube_length_m == pytest.approx(.1)
    proposal.accepted = True
    assert observe(.108) is None
    assert observe(.108) is None
    assert observe(.108).size_m == pytest.approx((.02, .02, .1))
    source._reset_source_bootstrap("camera_dropout")
    attach()
    result.decision = "HOLD"
    result.reasons = ("scene_cylinder_geometry_unresolved",)
    assert observe() is None
    assert source.last_hold_reasons == result.reasons
    result.decision = "TRACKED"
    for _ in range(2):
        assert observe() is None
    assert observe() is not None
    assert lengths == pytest.approx([.1])


def test_source_uncertainty_hold_preserves_last_qualified_geometry(qualified_size_source) -> None:
    source, lengths, _, _, _, observe, _ = qualified_size_source
    source.limit_source_pose_uncertainty(position_std_m=.004, orientation_std_deg=2.0)
    # A noisy initial observation must not freeze size or call the downstream setter.
    for _ in range(3):
        assert observe(position_std_m=.005) is None
    assert lengths == []
    assert source.estimated_tube_length_m is None
    for _ in range(3):
        pose = observe()
    assert pose is not None
    previous = source._last_qualified_transform.copy()
    # Size correction contributes uncertainty even when the original pose was good.
    assert observe(.108, open_x=.005) is None
    assert source.last_hold_reasons == ("runtime_tube_pose_uncertainty_exceeded",)
    assert source._last_qualified_transform == pytest.approx(previous)
    for _ in range(3):
        assert observe(orientation_std_deg=3.0) is None
    for _ in range(3):
        recovered = observe()
    assert recovered is not None
    assert recovered.size_m == pose.size_m
    assert lengths == pytest.approx([.1])


def test_source_frozen_size_never_promotes_clipped_or_new_object(qualified_size_source) -> None:
    source, lengths, _, mask, _, observe, attach = qualified_size_source
    for _ in range(3):
        observe()
    # Even temporally stable clipped geometry must not inherit the full tube size.
    source._reset_source_bootstrap("occlusion")
    attach()
    for _ in range(3):
        assert observe(.060) is None
    assert source.last_hold_reasons == ("runtime_tube_geometry_changed",)
    # Same dimensions at a different source location are not a trusted reacquisition.
    for _ in range(3):
        assert observe(.100, open_x=.10) is None
    assert source.last_hold_reasons == ("qualified_tube_pose_continuity_lost",)
    mask.tracking_id = "tube-2"
    attach()
    with pytest.raises(TubePoseTrackingError, match="qualified tube tracking identity changed"):
        observe()
    assert lengths == pytest.approx([.1])


def test_online_camera_binding_uses_one_outer_calibration() -> None:
    config = {
        "downstream_factory": (
            "demo2gripper.test_tube_online_socket:create_online_socket_scene_interpreter"
        ),
        "downstream_config": {
            "downstream_factory": (
                "demo2gripper.test_tube_downstream_scene:create_online_held_tube_scene_interpreter"
            ),
            "downstream_config": {},
        },
    }
    bound = _bind_online_scene_camera(
        config,
        T_base_camera=np.eye(4),
        camera_serial="d435-1",
    )
    socket = bound["downstream_config"]
    held = socket["downstream_config"]
    assert bound["camera_serial"] == socket["camera_serial"] == held["camera_serial"]
    assert np.allclose(socket["T_base_camera"], np.eye(4))
    assert np.allclose(held["T_base_camera"], np.eye(4))
    assert bound["camera_extrinsic_qualified"] is True
    assert socket["camera_extrinsic_qualified"] is True
    assert held["camera_extrinsic_qualified"] is True

    config["downstream_config"]["camera_serial"] = "wrong-camera"
    with pytest.raises(ClosedLoopContractError, match="differs from live camera"):
        _bind_online_scene_camera(
            config,
            T_base_camera=np.eye(4),
            camera_serial="d435-1",
        )


def test_runtime_f2g_size_grounds_normalized_grasp_without_caliper_length() -> None:
    relation = NormalizedDemoGraspRelation(
        axial_fraction_closed_to_open=0.65,
        tube_length_m=None,
        tube_length_source="runtime_f2g_size",
        preferred_jaw_axis_base=np.array([1.0, 0.0, 0.0]),
    )
    geometry = ClosedLoopGeometry(
        relation_id="online",
        T_tube_tcp_demo_grasp=None,
        T_socket_tube_entry=np.eye(4),
        normalized_demo_grasp=relation,
        relation_source="runtime_current_scene",
        relation_runtime_qualified=True,
        socket_entry_source="online_entry_frame",
        socket_entry_runtime_qualified=True,
    )
    descriptor = PoseBackendDescriptor(
        name="f2g_pose",
        available=True,
        runtime_qualified=True,
        qualification_id="f2g-runtime",
        capabilities=SOURCE_POSE_CAPABILITIES,
    )
    common = dict(
        object_id="tube",
        T_base_object=np.eye(4),
        observed_at_s=1.0,
        position_std_m=0.001,
        orientation_std_deg=1.0,
        backend=descriptor,
        tracking_id="tube-1",
        inference_latency_s=0.02,
        mask_id="mask-1",
        mask_iou=0.9,
        valid_depth_count=200,
        foreground_depth_support_count=180,
        foreground_depth_support_fraction=0.9,
        background_depth_fraction=0.1,
        masked_depth_cluster_qualified=True,
        axis_sign_source="open_end_detector",
        cylindrical_roll_canonicalized=True,
    )
    pose = ObjectPoseEstimate(**common, size_m=(0.018, 0.020, 0.100))
    target = grounded_grasp_target(pose, geometry)
    assert target[2, 3] == pytest.approx((0.65 - 0.5) * 0.100)

    missing = ObjectPoseEstimate(**common)
    with pytest.raises(ClosedLoopContractError, match="runtime F2G tube length"):
        grounded_grasp_target(missing, geometry)


def test_online_template_defers_scene_values_but_keeps_installation_blockers() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = json.loads((repo / "configs/test_tube_closed_loop.online.template.json").read_text())
    geometry = parse_geometry(config["geometry"])
    assert geometry.normalized_demo_grasp is not None
    assert geometry.normalized_demo_grasp.tube_length_m is None
    assert np.allclose(geometry.T_socket_tube_entry, np.eye(4))

    assessment = assess_live_config(config)
    assert "camera_extrinsic_artifact_missing" in assessment.static_blockers
    assert "hande_tcp_qualification_receipt_missing" in assessment.static_blockers
    assert "fr3_hande_collision_payload_qualification_receipt_missing" in (
        assessment.static_blockers
    )
    assert "measured_tube_length_missing" not in assessment.static_blockers
    assert "T_socket_tube_entry_missing" not in assessment.static_blockers
    assert "source_f2g_pose_qualification_receipt_missing" not in assessment.static_blockers
    assert "runtime_source_tube_pose_size_and_signed_axis_required" in (
        assessment.runtime_requirements
    )
    assert "runtime_socket_target_identity_required" in assessment.runtime_requirements
    assert "runtime_f2g_service_identity_and_crosschecks_required" in (
        assessment.runtime_requirements
    )
