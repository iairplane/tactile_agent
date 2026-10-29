from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demo2gripper.test_tube_downstream_scene import (
    DownstreamSceneError,
    DownstreamScenePolicy,
    QualifiedHeldTubeSceneInterpreter,
    create_qualified_held_tube_scene_interpreter,
)
from demo2gripper.test_tube_preflight import TubeState
from demo2gripper.test_tube_rgb_tracker import ImageLineCandidate, TubeAxisTrackerConfig
from demo2gripper.test_tube_visual import VisualAngleGateConfig

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
SERIAL = "unit-d435"
LENGTH_M = 0.11


def _tube_pose(*, z_m: float = 1.0, y_m: float = 0.0) -> np.ndarray:
    pose = np.eye(4)
    # local +Z (closed -> open) points along base/camera +Y.  Local +X is the
    # jaw axis; the frame remains right handed.
    pose[:3, :3] = np.column_stack(
        (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 1.0, 0.0]),
        )
    )
    pose[:3, 3] = [0.0, y_m, z_m]
    return pose


def _tcp_pose(*, z_m: float = 1.0, y_m: float = 0.0) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = [0.0, y_m, z_m]
    return pose


def _projected_line(pose: np.ndarray) -> ImageLineCandidate:
    half = LENGTH_M / 2.0
    points = pose @ np.array(
        [[0.0, 0.0, -half, 1.0], [0.0, 0.0, half, 1.0]]
    ).T
    pixels = K @ points[:3]
    pixels = (pixels[:2] / pixels[2]).T
    return ImageLineCandidate(tuple(pixels[0]), tuple(pixels[1]))


def _frame(sequence: int, timestamp_s: float) -> SimpleNamespace:
    return SimpleNamespace(
        sequence=sequence,
        serial=SERIAL,
        rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        intrinsics=SimpleNamespace(matrix=K.copy()),
        host_received_monotonic_s=timestamp_s,
    )


def _interpreter(
    candidate_box: list[ImageLineCandidate],
    *,
    activate_runtime_relation: bool = True,
    include_demo_candidate: bool = True,
) -> QualifiedHeldTubeSceneInterpreter:
    initial_tube = _tube_pose()
    initial_tcp = _tcp_pose()
    T_tube_tcp = np.linalg.inv(initial_tube) @ initial_tcp

    def candidates(_image, _roi):
        return tuple(candidate_box)

    scene = QualifiedHeldTubeSceneInterpreter(
        T_base_camera=np.eye(4),
        T_tube_tcp_demo=T_tube_tcp if include_demo_candidate else None,
        T_base_socket=_tube_pose(z_m=1.031),
        T_socket_tube_entry=np.eye(4),
        socket_outward_axis_socket=[0.0, 0.0, 1.0],
        pivot_plane_normal_tcp=[0.0, 0.0, 1.0],
        pivot_point_tcp=[0.0, 0.0, 0.0],
        jaw_axis_tcp=[1.0, 0.0, 0.0],
        reviewed_seed_line=_projected_line(initial_tube),
        axis_roi_uvxy=(250, 150, 390, 330),
        tracker_config=TubeAxisTrackerConfig(),
        angle_config=VisualAngleGateConfig(
            target_image_axis_deg=90.0,
            tolerance_deg=2.0,
            min_consecutive_frames=2,
            max_rate_deg_s=120.0,
        ),
        policy=DownstreamScenePolicy(
            tube_length_m=LENGTH_M,
            extract_clearance_m=0.030,
            insertion_target_depth_m=0.020,
            support_min_consecutive_frames=3,
        ),
        tracking_id="unit-tube-track",
        evidence_id="unit-scene-qualification",
        position_std_m=0.002,
        orientation_std_deg=2.0,
        expected_camera_serial=SERIAL,
        expected_K=K,
        candidate_extractor=candidates,
    )
    if activate_runtime_relation:
        scene.set_grasp_relation(
            T_tube_tcp,
            T_base_tube_at_contact=initial_tube,
        )
    return scene


def test_missing_runtime_grasp_relation_blocks_extract_and_pivot() -> None:
    candidate_box = [_projected_line(_tube_pose())]
    scene = _interpreter(
        candidate_box,
        activate_runtime_relation=False,
        include_demo_candidate=False,
    )

    extract = scene.observe(
        frame=_frame(1, 1.0), state=TubeState.EXTRACT, T_base_tcp=_tcp_pose()
    )
    assert not extract.tube_clear_of_source
    assert "runtime_grasp_relation_not_set" in scene.diagnostics.hold_reasons

    for sequence in (2, 3):
        pivot = scene.observe(
            frame=_frame(sequence, 1.0 + 0.05 * sequence),
            state=TubeState.PIVOT_EVALUATE,
            T_base_tcp=_tcp_pose(z_m=1.031),
        )
        assert not pivot.front_rgb_angle_target_reached
        held = scene.held_tube_pose(
            frame=_frame(sequence, 1.0 + 0.05 * sequence),
            scene=pivot,
            state=TubeState.PIVOT_EVALUATE,
            T_base_tcp=_tcp_pose(z_m=1.031),
            tactile=SimpleNamespace(
                bilateral_contact=False, rotation_stopped=False
            ),
        )
        assert held is None

    relation = np.linalg.inv(_tube_pose()) @ _tcp_pose()
    with pytest.raises(DownstreamSceneError, match="after downstream motion"):
        scene.set_grasp_relation(
            relation,
            T_base_tube_at_contact=_tube_pose(),
        )


def test_contact_baseline_allows_first_post_lift_frame_to_report_clear() -> None:
    lifted_tcp = _tcp_pose(z_m=1.031)
    lifted_tube = _tube_pose(z_m=1.031)
    candidate_box = [_projected_line(lifted_tube)]
    scene = _interpreter(
        candidate_box,
        activate_runtime_relation=True,
        include_demo_candidate=False,
    )

    first_post_lift = scene.observe(
        frame=_frame(1, 1.0),
        state=TubeState.EXTRACT,
        T_base_tcp=lifted_tcp,
    )
    assert first_post_lift.tube_clear_of_source


def test_runtime_relation_replaces_demo_candidate_and_cannot_be_rebound() -> None:
    candidate_box = [_projected_line(_tube_pose())]
    initial_tube = _tube_pose()
    actual_tcp = _tcp_pose()
    actual = np.linalg.inv(initial_tube) @ actual_tcp
    candidate = actual.copy()
    candidate[:3, 3] += [0.020, 0.0, 0.0]

    def candidates(_image, _roi):
        return tuple(candidate_box)

    scene = QualifiedHeldTubeSceneInterpreter(
        T_base_camera=np.eye(4),
        T_tube_tcp_demo=candidate,
        T_base_socket=_tube_pose(z_m=1.031),
        T_socket_tube_entry=np.eye(4),
        socket_outward_axis_socket=[0.0, 0.0, 1.0],
        pivot_plane_normal_tcp=[0.0, 0.0, 1.0],
        pivot_point_tcp=[0.0, 0.0, 0.0],
        jaw_axis_tcp=[1.0, 0.0, 0.0],
        reviewed_seed_line=_projected_line(initial_tube),
        axis_roi_uvxy=(250, 150, 390, 330),
        tracker_config=TubeAxisTrackerConfig(),
        angle_config=VisualAngleGateConfig(
            target_image_axis_deg=90.0,
            tolerance_deg=2.0,
            min_consecutive_frames=2,
            max_rate_deg_s=120.0,
        ),
        policy=DownstreamScenePolicy(tube_length_m=LENGTH_M),
        tracking_id="unit-tube-track",
        evidence_id="unit-scene-qualification",
        position_std_m=0.002,
        orientation_std_deg=2.0,
        expected_camera_serial=SERIAL,
        expected_K=K,
        candidate_extractor=candidates,
    )
    scene.set_grasp_relation(actual, T_base_tube_at_contact=initial_tube)

    # Mutation by the caller cannot alter the frozen runtime relation.
    actual[0, 3] += 0.010
    observed = scene.observe(
        frame=_frame(1, 1.0), state=TubeState.EXTRACT, T_base_tcp=actual_tcp
    )
    held = scene.held_tube_pose(
        frame=_frame(1, 1.0),
        scene=observed,
        state=TubeState.EXTRACT,
        T_base_tcp=actual_tcp,
        tactile=SimpleNamespace(bilateral_contact=True, rotation_stopped=True),
    )
    assert held is not None
    np.testing.assert_allclose(held.T_base_object, initial_tube, atol=1e-9)

    changed = np.linalg.inv(initial_tube) @ actual_tcp
    changed[0, 3] += 0.001
    with pytest.raises(DownstreamSceneError, match="cannot be rebound"):
        scene.set_grasp_relation(changed, T_base_tube_at_contact=initial_tube)


def test_extract_pivot_reclamp_and_socket_predicates_are_resident() -> None:
    candidate_box = [_projected_line(_tube_pose())]
    scene = _interpreter(candidate_box)

    first = scene.observe(
        frame=_frame(1, 1.0), state=TubeState.EXTRACT, T_base_tcp=_tcp_pose()
    )
    assert not first.tube_clear_of_source
    assert scene.diagnostics.tracker_decision == "MODEL_RESEEDED"

    lifted_tcp = _tcp_pose(z_m=1.031)
    lifted_tube = _tube_pose(z_m=1.031)
    candidate_box[0] = _projected_line(lifted_tube)
    lifted = scene.observe(
        frame=_frame(2, 1.05), state=TubeState.EXTRACT, T_base_tcp=lifted_tcp
    )
    assert lifted.tube_clear_of_source

    first_pivot = scene.observe(
        frame=_frame(3, 1.10),
        state=TubeState.PIVOT_EVALUATE,
        T_base_tcp=lifted_tcp,
    )
    second_pivot = scene.observe(
        frame=_frame(4, 1.15),
        state=TubeState.PIVOT_EVALUATE,
        T_base_tcp=lifted_tcp,
    )
    assert not first_pivot.front_rgb_angle_target_reached
    assert second_pivot.front_rgb_angle_target_reached

    tactile = SimpleNamespace(bilateral_contact=True, rotation_stopped=True)
    held = scene.held_tube_pose(
        frame=_frame(4, 1.15),
        scene=second_pivot,
        state=TubeState.RECLAMP,
        T_base_tcp=lifted_tcp,
        tactile=tactile,
    )
    assert held is not None
    assert np.allclose(held.T_base_object[:3, 2], [0.0, 1.0, 0.0], atol=1e-6)

    aligned = scene.observe(
        frame=_frame(5, 1.20), state=TubeState.ALIGN_SOCKET, T_base_tcp=lifted_tcp
    )
    assert aligned.tube_socket_axes_aligned
    assert aligned.tube_tip_at_socket_entry


def test_release_support_requires_inserted_pose_and_stable_fresh_rgb() -> None:
    candidate_box = [_projected_line(_tube_pose())]
    scene = _interpreter(candidate_box)
    lifted_tcp = _tcp_pose(z_m=1.031)
    lifted_tube = _tube_pose(z_m=1.031)
    candidate_box[0] = _projected_line(lifted_tube)

    scene.observe(
        frame=_frame(1, 1.0), state=TubeState.EXTRACT, T_base_tcp=lifted_tcp
    )
    pivot = scene.observe(
        frame=_frame(2, 1.05), state=TubeState.PIVOT_EVALUATE, T_base_tcp=lifted_tcp
    )
    scene.held_tube_pose(
        frame=_frame(2, 1.05),
        scene=pivot,
        state=TubeState.RECLAMP,
        T_base_tcp=lifted_tcp,
        tactile=SimpleNamespace(bilateral_contact=True, rotation_stopped=True),
    )

    inserted_tcp = _tcp_pose(z_m=1.031, y_m=-0.020)
    inserted_tube = _tube_pose(z_m=1.031, y_m=-0.020)
    candidate_box[0] = _projected_line(inserted_tube)
    decisions = []
    for index in range(3, 6):
        evidence = scene.observe(
            frame=_frame(index, 1.0 + index * 0.05),
            state=TubeState.RELEASE,
            T_base_tcp=inserted_tcp,
        )
        decisions.append(evidence.tube_stays_supported)
    assert decisions == [False, False, True]

    # The stable frames above authorize opening, but they were all observed
    # while the tube was still gripped.  Release verification must establish
    # a fresh consecutive window after OPEN_RELEASE.
    scene.reset_support_history_for_release()
    post_open_decisions = []
    for index in range(6, 9):
        evidence = scene.observe(
            frame=_frame(index, 1.0 + index * 0.05),
            state=TubeState.RELEASE,
            T_base_tcp=inserted_tcp,
        )
        post_open_decisions.append(evidence.tube_stays_supported)
    assert post_open_decisions == [False, False, True]

    candidate_box.clear()
    lost = scene.observe(
        frame=_frame(9, 1.45), state=TubeState.RELEASE, T_base_tcp=inserted_tcp
    )
    assert not lost.tube_stays_supported
    assert scene.diagnostics.hold_reasons


def test_missing_or_wrong_current_robot_pose_fails_closed() -> None:
    candidate_box = [_projected_line(_tube_pose())]
    scene = _interpreter(candidate_box)
    with pytest.raises((ValueError, DownstreamSceneError)):
        scene.observe(
            frame=_frame(1, 1.0),
            state=TubeState.EXTRACT,
            T_base_tcp=None,  # type: ignore[arg-type]
        )

    bad_frame = _frame(2, 1.1)
    bad_frame.serial = "different-camera"
    with pytest.raises(DownstreamSceneError, match="serial"):
        scene.observe(
            frame=bad_frame, state=TubeState.EXTRACT, T_base_tcp=_tcp_pose()
        )


def test_ambiguous_or_missing_rgb_never_returns_a_pivot_pose() -> None:
    candidate_box = [_projected_line(_tube_pose())]
    scene = _interpreter(candidate_box)
    tcp = _tcp_pose(z_m=1.031)
    scene.observe(frame=_frame(1, 1.0), state=TubeState.EXTRACT, T_base_tcp=tcp)
    candidate_box.clear()
    evidence = scene.observe(
        frame=_frame(2, 1.05), state=TubeState.PIVOT_EVALUATE, T_base_tcp=tcp
    )
    held = scene.held_tube_pose(
        frame=_frame(2, 1.05),
        scene=evidence,
        state=TubeState.PIVOT_EVALUATE,
        T_base_tcp=tcp,
        tactile=SimpleNamespace(bilateral_contact=False, rotation_stopped=False),
    )
    assert held is None
    assert not evidence.front_rgb_angle_target_reached


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(path: Path, *, kind: str, artifact: Path | None = None) -> None:
    value = {
        "schema": "demo2gripper.test_tube_live_io_qualification.v1",
        "kind": kind,
        "qualification_id": f"unit-{kind}",
        "runtime_qualified": True,
        "position_std_m": 0.002,
        "orientation_std_deg": 2.0,
        "workspace_calibration_id": "unit-workspace",
    }
    if artifact is not None:
        value["artifact_sha256"] = _digest(artifact)
    _write_json(path, value)


def test_factory_refuses_unqualified_rgb_and_cannot_invent_socket_from_2d(
    tmp_path: Path,
) -> None:
    camera = tmp_path / "T_base_camera.npy"
    socket = tmp_path / "T_base_socket.npy"
    intrinsics = tmp_path / "K.npy"
    np.save(camera, np.eye(4))
    np.save(socket, _tube_pose(z_m=1.031))
    np.save(intrinsics, K)
    camera_receipt = tmp_path / "camera_receipt.json"
    socket_receipt = tmp_path / "socket_receipt.json"
    scene_receipt = tmp_path / "scene_receipt.json"
    _receipt(camera_receipt, kind="fixed_camera_extrinsic", artifact=camera)
    _receipt(socket_receipt, kind="static_socket_pose", artifact=socket)
    _receipt(scene_receipt, kind="tube_scene_interpreter")

    reference = tmp_path / "pivot_reference.json"
    _write_json(
        reference,
        {
            "schema": "demo2gripper.test_tube_front_rgb_pivot_reference.v1",
            "camera": {
                "serial": SERIAL,
                "intrinsics_sha256": _digest(intrinsics),
            },
            "target": {
                "target_image_axis_deg": 90.0,
                "tolerance_deg": 3.0,
                "min_consecutive_frames": 2,
                "max_rate_deg_s": 120.0,
            },
        },
    )
    rgb_report = tmp_path / "rgb_report.json"
    _write_json(
        rgb_report,
        {
            "schema": "demo2gripper.test_tube_rgb_tracker_qualification.v2",
            "runtime_gate_eligible": False,
            "qualification_id": "unit-rgb",
            "source": {
                "camera_serial": SERIAL,
                "intrinsics_sha256": _digest(intrinsics),
                "reference_config_sha256": _digest(reference),
            },
        },
    )
    T_tube_tcp = np.linalg.inv(_tube_pose()) @ _tcp_pose()
    config = {
        "camera_extrinsic": str(camera),
        "camera_extrinsic_qualification_receipt": str(camera_receipt),
        "static_socket_pose": str(socket),
        "static_socket_pose_qualification_receipt": str(socket_receipt),
        "scene_qualification_receipt": str(scene_receipt),
        "rgb_tracker_qualification_report": str(rgb_report),
        "pivot_reference": str(reference),
        "camera_serial": SERIAL,
        "intrinsics": str(intrinsics),
        "T_tube_tcp_demo_grasp": T_tube_tcp.tolist(),
        "T_socket_tube_entry": np.eye(4).tolist(),
        "reviewed_seed_line_uv": [
            list(_projected_line(_tube_pose()).start_uv),
            list(_projected_line(_tube_pose()).end_uv),
        ],
        "axis_roi_uvxy": [250, 150, 390, 330],
        "policy": {"tube_length_m": LENGTH_M},
    }
    with pytest.raises(DownstreamSceneError, match="not runtime qualified"):
        create_qualified_held_tube_scene_interpreter(config=config, output_dir=tmp_path)

    qualified_report = json.loads(rgb_report.read_text(encoding="utf-8"))
    qualified_report["runtime_gate_eligible"] = True
    _write_json(rgb_report, qualified_report)
    interpreter = create_qualified_held_tube_scene_interpreter(
        config=config, output_dir=tmp_path
    )
    assert interpreter.diagnostics.tracker_decision == "UNINITIALIZED"

    # Normalized top-down mode has no complete static relation.  Construction
    # succeeds, but downstream predicates remain closed until LiveIO injects
    # the relation and contact baseline from one stationary source frame.
    config["T_tube_tcp_demo_grasp"] = None
    normalized = create_qualified_held_tube_scene_interpreter(
        config=config, output_dir=tmp_path
    )
    blocked = normalized.observe(
        frame=_frame(1, 1.0), state=TubeState.EXTRACT, T_base_tcp=_tcp_pose()
    )
    assert not blocked.tube_clear_of_source
    assert "runtime_grasp_relation_not_set" in normalized.diagnostics.hold_reasons

    del config["static_socket_pose"]
    with pytest.raises(DownstreamSceneError, match="static_socket_pose is required"):
        create_qualified_held_tube_scene_interpreter(config=config, output_dir=tmp_path)

