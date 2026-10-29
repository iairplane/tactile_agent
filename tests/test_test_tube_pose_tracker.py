from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demo2gripper.f2g_pose_service import (
    FakePoseBackend,
    ForegroundDepthGateConfig,
    PoseInferenceService,
    PoseRequest,
    evaluate_foreground_depth,
)
from demo2gripper.pose_observation import PoseCapture
from demo2gripper.test_tube_closed_loop import (
    SOURCE_POSE_CAPABILITIES,
    PoseBackendDescriptor,
)
from demo2gripper.test_tube_pose_tracker import (
    SOURCE_SEED_QUALIFICATION_SCHEMA,
    RealTimeTubePoseTracker,
    RealTimeTubePoseTrackerConfig,
    ReviewedTubeEnds,
    StaticSourceMaskTracker,
    StaticSourceTubeSceneInterpreter,
    TubePoseQualifier,
    TubePoseQualifierConfig,
    TubePoseTrackingError,
    create_static_source_tube_scene_interpreter,
)


def _scene_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = 240, 320
    rng = np.random.default_rng(4)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    mask[45:195, 130:190] = True
    rgb[mask] = rng.integers(0, 256, (int(mask.sum()), 3), dtype=np.uint8)
    depth = np.full((height, width), 0.60, dtype=np.float32)
    depth[mask] = 0.30
    K = np.array(
        [[600.0, 0.0, 160.0], [0.0, 600.0, 120.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rgb, depth, K, mask


def _vertical_tube_transform(*, center_x_m: float = 0.0) -> np.ndarray:
    # Canonical local +Z projects down in the optical image.  The columns are
    # +X, +Y, +Z and form a right-handed rotation.
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    )
    transform[:3, 3] = [center_x_m, 0.0, 0.30]
    return transform


def _proposal(
    *, transform: np.ndarray | None = None, sequence: int = 1
):
    rgb, depth, K, mask = _scene_arrays()
    backend = FakePoseBackend(
        T_camera_object=(
            _vertical_tube_transform() if transform is None else transform
        ),
        size_m=(0.018, 0.018, 0.080),
    )
    request = PoseRequest(
        request_id=f"proposal-{sequence}",
        capture=PoseCapture(
            sequence=sequence,
            frame_id=f"front:{sequence}",
            timestamp_s=float(sequence),
            clock_domain="client_monotonic",
        ),
        rgb=rgb,
        depth_m=depth,
        K=K,
        mask=mask,
        mask_source="reviewed_test_mask",
        mask_source_score=1.0,
    )
    return PoseInferenceService(backend).infer(request), K


def _ends() -> ReviewedTubeEnds:
    # A +/-40 mm tube endpoint at z=0.30 m projects +/-80 pixels.
    return ReviewedTubeEnds(
        open_uv=(160.0, 200.0),
        closed_uv=(160.0, 40.0),
        evidence_id="unit-reviewed-ends",
    )


def _descriptor() -> PoseBackendDescriptor:
    return PoseBackendDescriptor(
        name="f2g_pose",
        available=True,
        runtime_qualified=True,
        qualification_id="unit-f2g-temporal",
        capabilities=SOURCE_POSE_CAPABILITIES,
        detail="unit test",
    )


def test_static_mask_tracker_preserves_identity_and_measures_flow_ecc_iou() -> None:
    cv2 = pytest.importorskip("cv2")
    rgb, _, _, mask = _scene_arrays()
    tracker = StaticSourceMaskTracker()
    seeded = tracker.seed(
        rgb=rgb, reviewed_mask=mask, timestamp_s=1.0, tracking_id="tube-reviewed-1"
    )
    affine = np.float32([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]])
    moved_rgb = cv2.warpAffine(rgb, affine, (rgb.shape[1], rgb.shape[0]))
    expected = cv2.warpAffine(
        mask.astype(np.uint8), affine, (rgb.shape[1], rgb.shape[0])
    ) > 0
    tracked = tracker.update(rgb=moved_rgb, timestamp_s=1.1)
    assert seeded.tracking_id == tracked.tracking_id == "tube-reviewed-1"
    assert tracked.decision == "TRACKED"
    assert tracked.score_provenance == "optical_flow_vs_ecc_iou"
    assert tracked.consistency_iou is not None and tracked.consistency_iou > 0.98
    assert tracked.mask is not None
    union = np.count_nonzero(expected | tracked.mask)
    assert np.count_nonzero(expected & tracked.mask) / union > 0.98


def test_static_mask_tracker_holds_on_explicit_occluder_overlap() -> None:
    rgb, _, _, mask = _scene_arrays()
    tracker = StaticSourceMaskTracker()
    tracker.seed(rgb=rgb, reviewed_mask=mask, timestamp_s=1.0)
    result = tracker.update(
        rgb=rgb,
        timestamp_s=1.1,
        occluder_mask=mask,
    )
    assert result.decision == "HOLD"
    assert result.occluded
    assert result.reasons == ("tube_mask_occluded",)
    assert result.mask is None


def test_pose_qualifier_resolves_open_end_canonicalizes_roll_and_measures_window() -> None:
    proposal, K = _proposal()
    qualifier = TubePoseQualifier(
        _ends(), TubePoseQualifierConfig(window_size=5, min_window_frames=3)
    )
    first = qualifier.update(proposal=proposal, K=K)
    second = qualifier.update(proposal=proposal, K=K)
    third = qualifier.update(proposal=proposal, K=K)
    assert first.decision == second.decision == "WARMUP"
    assert first.provisional_T_camera_tube is not None
    assert third.decision == "TRACKED"
    assert third.pose is not None
    assert third.pose.cylindrical_roll_canonicalized
    assert third.pose.axis_sign_source == "temporal_continuity"
    assert np.dot(third.pose.T_camera_tube[:3, 2], [0.0, 1.0, 0.0]) > 0.999
    assert np.linalg.det(third.pose.T_camera_tube[:3, :3]) == pytest.approx(1.0)
    assert third.pose.position_std_m == pytest.approx(0.0)
    assert third.pose.orientation_std_deg == pytest.approx(0.0)


def test_pose_qualifier_rejects_large_single_frame_jump_without_updating_window() -> None:
    proposal, K = _proposal()
    qualifier = TubePoseQualifier(
        _ends(), TubePoseQualifierConfig(min_window_frames=2)
    )
    assert qualifier.update(proposal=proposal, K=K).decision == "WARMUP"
    assert qualifier.update(proposal=proposal, K=K).decision == "TRACKED"
    jumped, _ = _proposal(transform=_vertical_tube_transform(center_x_m=0.20), sequence=3)
    result = qualifier.update(proposal=jumped, K=K)
    assert result.decision == "HOLD"
    assert result.pose is None
    assert result.provisional_T_camera_tube is not None
    assert "pose_translation_jump" in result.reasons


def test_scene_geometry_canonicalizes_transparent_cylinder_pose() -> None:
    proposal, K = _proposal(
        transform=np.array(
            [
                [1.0, 0.0, 0.0, 0.04],
                [0.0, 0.0, 1.0, 0.05],
                [0.0, -1.0, 0.0, 0.28],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    _, depth, _, mask = _scene_arrays()
    foreground = np.where(mask, depth, np.nan)
    qualifier = TubePoseQualifier(
        _ends(),
        TubePoseQualifierConfig(
            min_window_frames=3,
            canonicalize_cylinder_from_scene_geometry=True,
        ),
    )
    result = None
    for _ in range(3):
        result = qualifier.update(
            proposal=proposal,
            K=K,
            foreground_depth_m=foreground,
        )
    assert result is not None and result.pose is not None
    pose = result.pose
    projected_center = K @ pose.T_camera_tube[:3, 3]
    projected_center = projected_center[:2] / projected_center[2]
    assert projected_center == pytest.approx([160.0, 120.0], abs=1e-6)
    assert np.dot(pose.T_camera_tube[:3, 2], [0.0, 1.0, 0.0]) > 0.999
    assert pose.size_m is not None
    assert max(pose.size_m) == pytest.approx(0.080, abs=1e-3)


class _DirectFakeProvider:
    def __init__(self, transform: np.ndarray | None = None) -> None:
        self.sequence = 0
        self.service = PoseInferenceService(
            FakePoseBackend(
                T_camera_object=(
                    _vertical_tube_transform()
                    if transform is None
                    else transform
                ),
                size_m=(0.018, 0.018, 0.080),
            )
        )

    def estimate(self, **kwargs):
        self.sequence += 1
        return self.service.infer(
            PoseRequest(
                request_id=f"direct-{self.sequence}",
                capture=PoseCapture(
                    sequence=self.sequence,
                    frame_id=kwargs["frame_id"],
                    timestamp_s=kwargs["capture_timestamp_s"],
                    clock_domain="client_monotonic",
                ),
                rgb=kwargs["rgb"],
                depth_m=kwargs["depth_m"],
                K=kwargs["K"],
                mask=kwargs["mask"],
                mask_source=kwargs["mask_source"],
                mask_source_score=kwargs["mask_source_score"],
            )
        )


def test_direct_tracker_returns_existing_object_pose_estimate_after_warmup() -> None:
    rgb, depth, K, mask = _scene_arrays()
    now = [1.01]
    tracker = RealTimeTubePoseTracker(
        provider=_DirectFakeProvider(),
        backend=_descriptor(),
        T_base_camera=np.eye(4),
        reviewed_initial_mask=mask,
        reviewed_ends=_ends(),
        qualifier_config=TubePoseQualifierConfig(min_window_frames=3),
        config=RealTimeTubePoseTrackerConfig(max_capture_age_s=0.1),
        tracking_id="tube-one-session",
        monotonic=lambda: now[0],
    )
    results = []
    for index in range(3):
        timestamp = 1.0 + 0.05 * index
        now[0] = timestamp + 0.01
        results.append(
            tracker.update(
                rgb=rgb,
                depth_m=depth,
                K=K,
                timestamp_s=timestamp,
                frame_id=f"front:{index}",
            )
        )
    assert [item.decision for item in results] == ["WARMUP", "WARMUP", "TRACKED"]
    estimate = results[-1].estimate
    assert estimate is not None
    assert estimate.tracking_id == "tube-one-session"
    assert estimate.mask_iou > 0.98
    assert estimate.masked_depth_cluster_qualified
    assert estimate.foreground_depth_support_count > 100
    assert estimate.axis_sign_source == "temporal_continuity"
    assert estimate.cylindrical_roll_canonicalized
    # The service deliberately has no F2G confidence.  Mask IoU remains a
    # separately named, measured tracking-consistency value.
    assert estimate.backend.name == "f2g_pose"


def test_stable_but_wrong_f2g_pose_is_rejected_by_independent_geometry() -> None:
    rgb, depth, K, mask = _scene_arrays()
    wrong = _vertical_tube_transform(center_x_m=0.080)
    tracker = RealTimeTubePoseTracker(
        provider=_DirectFakeProvider(wrong),
        backend=_descriptor(),
        T_base_camera=np.eye(4),
        reviewed_initial_mask=mask,
        reviewed_ends=_ends(),
        qualifier_config=TubePoseQualifierConfig(min_window_frames=2),
        config=RealTimeTubePoseTrackerConfig(max_capture_age_s=0.1),
        monotonic=lambda: 1.01,
    )
    result = tracker.update(
        rgb=rgb,
        depth_m=depth,
        K=K,
        timestamp_s=1.0,
        frame_id="front:wrong-stable-f2g",
    )
    assert result.decision == "HOLD"
    assert result.estimate is None
    assert {
        "f2g_projected_center_disagrees_with_reviewed_tube",
        "f2g_center_disagrees_with_foreground_depth_line",
    } <= set(result.reasons)


def test_scene_interpreter_factory_is_directly_loadable_and_warmup_blocks_motion(
    tmp_path: Path,
) -> None:
    rgb, depth, K, mask = _scene_arrays()
    mask_path = tmp_path / "reviewed_mask.npy"
    seed_rgb_path = tmp_path / "reviewed_seed_rgb.npy"
    intrinsics_path = tmp_path / "K.npy"
    np.save(mask_path, mask)
    np.save(seed_rgb_path, rgb)
    np.save(intrinsics_path, K)
    camera_receipt_path = tmp_path / "camera_receipt.json"
    camera_receipt_path.write_text(
        json.dumps(
            {
                "schema": "demo2gripper.test_tube_live_io_qualification.v1",
                "kind": "fixed_camera_extrinsic",
                "runtime_qualified": True,
                "qualification_id": "unit-camera-extrinsic",
                "workspace_calibration_id": "unit-workspace",
            }
        ),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "reviewed_seed_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": SOURCE_SEED_QUALIFICATION_SCHEMA,
                "status": "accepted",
                "runtime_qualified": True,
                "qualification_id": "unit-source-seed",
                "workspace_calibration_id": "unit-workspace",
                "camera_extrinsic_qualification_id": "unit-camera-extrinsic",
                "camera_serial": "unit-front-camera",
                "intrinsics_sha256": hashlib.sha256(
                    intrinsics_path.read_bytes()
                ).hexdigest(),
                "seed_rgb_sha256": hashlib.sha256(
                    seed_rgb_path.read_bytes()
                ).hexdigest(),
                "reviewed_mask_sha256": hashlib.sha256(
                    mask_path.read_bytes()
                ).hexdigest(),
                "reviewed_open_uv": [160.0, 200.0],
                "reviewed_closed_uv": [160.0, 40.0],
                "reviewed_endpoints_evidence_id": "unit-reviewed-ends",
            }
        ),
        encoding="utf-8",
    )
    scene = create_static_source_tube_scene_interpreter(
        config={
            "reviewed_seed_rgb": str(seed_rgb_path),
            "reviewed_initial_mask": str(mask_path),
            "reviewed_seed_qualification_receipt": str(receipt_path),
            "camera_extrinsic_qualification_receipt": str(camera_receipt_path),
            "intrinsics": str(intrinsics_path),
            "camera_serial": "unit-front-camera",
            "reviewed_open_uv": [160.0, 200.0],
            "reviewed_closed_uv": [160.0, 40.0],
            "reviewed_endpoints_evidence_id": "unit-reviewed-ends",
            "tracking_id": "tube-scene-session",
            "pose_qualifier": {"min_window_frames": 3},
        },
        output_dir=tmp_path,
    )
    assert isinstance(scene, StaticSourceTubeSceneInterpreter)
    frame = SimpleNamespace(
        rgb=rgb,
        depth_m=depth,
        intrinsics=SimpleNamespace(matrix=K),
        serial="unit-front-camera",
        host_received_monotonic_s=1.0,
        sequence=1,
    )
    evidence = scene.observe(
        frame=frame,
        state=SimpleNamespace(value="observe"),
        T_base_tcp=np.eye(4),
    )
    assert evidence.tube is not None
    proposal, _ = _proposal()
    qualified = scene.qualify_pose(
        object_id="tube", proposal=proposal, mask=evidence.tube, frame=frame
    )
    assert qualified.position_std_m == pytest.approx(1.0)
    assert qualified.orientation_std_deg == pytest.approx(180.0)
    assert scene.last_hold_reasons == ("temporal_window_not_full_enough",)

    with pytest.raises(TubePoseTrackingError, match="camera serial"):
        scene.observe(
            frame=SimpleNamespace(
                **{**frame.__dict__, "serial": "different-camera", "sequence": 2}
            ),
            state=SimpleNamespace(value="observe"),
            T_base_tcp=np.eye(4),
        )


def test_source_factory_rejects_seed_receipt_that_does_not_bind_rgb(
    tmp_path: Path,
) -> None:
    rgb, _, _, mask = _scene_arrays()
    rgb_path = tmp_path / "seed.npy"
    mask_path = tmp_path / "mask.npy"
    intrinsics_path = tmp_path / "K.npy"
    receipt_path = tmp_path / "receipt.json"
    camera_receipt_path = tmp_path / "camera_receipt.json"
    np.save(rgb_path, rgb)
    np.save(mask_path, mask)
    _, _, K, _ = _scene_arrays()
    np.save(intrinsics_path, K)
    camera_receipt_path.write_text(
        json.dumps(
            {
                "schema": "demo2gripper.test_tube_live_io_qualification.v1",
                "kind": "fixed_camera_extrinsic",
                "runtime_qualified": True,
                "qualification_id": "unit-camera-extrinsic",
                "workspace_calibration_id": "unit-workspace",
            }
        ),
        encoding="utf-8",
    )
    receipt_path.write_text(
        json.dumps(
            {
                "schema": SOURCE_SEED_QUALIFICATION_SCHEMA,
                "status": "accepted",
                "runtime_qualified": True,
                "qualification_id": "unit-source-seed",
                "workspace_calibration_id": "unit-workspace",
                "camera_extrinsic_qualification_id": "unit-camera-extrinsic",
                "camera_serial": "unit-front-camera",
                "intrinsics_sha256": hashlib.sha256(
                    intrinsics_path.read_bytes()
                ).hexdigest(),
                "seed_rgb_sha256": "0" * 64,
                "reviewed_mask_sha256": hashlib.sha256(
                    mask_path.read_bytes()
                ).hexdigest(),
                "reviewed_open_uv": [160.0, 200.0],
                "reviewed_closed_uv": [160.0, 40.0],
                "reviewed_endpoints_evidence_id": "unit-reviewed-ends",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not runtime qualified"):
        create_static_source_tube_scene_interpreter(
            config={
                "reviewed_seed_rgb": str(rgb_path),
                "reviewed_initial_mask": str(mask_path),
                "reviewed_seed_qualification_receipt": str(receipt_path),
                "camera_extrinsic_qualification_receipt": str(camera_receipt_path),
                "intrinsics": str(intrinsics_path),
                "camera_serial": "unit-front-camera",
                "reviewed_open_uv": [160.0, 200.0],
                "reviewed_closed_uv": [160.0, 40.0],
                "reviewed_endpoints_evidence_id": "unit-reviewed-ends",
            },
            output_dir=tmp_path,
        )

    stale_camera_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    stale_camera_payload["seed_rgb_sha256"] = hashlib.sha256(
        rgb_path.read_bytes()
    ).hexdigest()
    stale_camera_payload["camera_extrinsic_qualification_id"] = "stale-camera"
    receipt_path.write_text(json.dumps(stale_camera_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not runtime qualified"):
        create_static_source_tube_scene_interpreter(
            config={
                "reviewed_seed_rgb": str(rgb_path),
                "reviewed_initial_mask": str(mask_path),
                "reviewed_seed_qualification_receipt": str(receipt_path),
                "camera_extrinsic_qualification_receipt": str(camera_receipt_path),
                "intrinsics": str(intrinsics_path),
                "camera_serial": "unit-front-camera",
                "reviewed_open_uv": [160.0, 200.0],
                "reviewed_closed_uv": [160.0, 40.0],
                "reviewed_endpoints_evidence_id": "unit-reviewed-ends",
            },
            output_dir=tmp_path,
        )


def test_real_frame_000004_background_dominated_mask_is_rejected_regression() -> None:
    cv2 = pytest.importorskip("cv2")
    root = Path(__file__).resolve().parents[1]
    run = root / "runs/live/test_tube_demo_20261023_02"
    mask_path = (
        root
        / "runs/diagnostics/f2g_tube_probe_20261024/sam3/masks/1/000004.png"
    )
    required = [
        run / "rgb/000004.png",
        run / "depth/000004.npy",
        run / "K.npy",
        mask_path,
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("local real-frame diagnostic artifacts are intentionally untracked")
    bgr = cv2.imread(str(required[0]), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    request = PoseRequest(
        request_id="real-frame-000004",
        capture=PoseCapture(1, "front:000004", 1.0, "client_monotonic"),
        rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        depth_m=np.load(required[1], allow_pickle=False),
        K=np.load(required[2], allow_pickle=False),
        mask=mask > 0,
        mask_source="sam3:transparent_test_tube",
        mask_source_score=0.099,
    )
    diagnostics = evaluate_foreground_depth(
        request, ForegroundDepthGateConfig()
    ).diagnostics
    assert not diagnostics.gate_passed
    assert "mask_depth_dominated_by_neighbouring_background" in diagnostics.reject_reasons
    assert diagnostics.mask_near_background_fraction is not None
    assert (
        diagnostics.mask_near_background_fraction
        > ForegroundDepthGateConfig().max_mask_near_background_fraction
    )

